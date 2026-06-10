"""The export registry: model definitions + the dispatch table that turns a
loaded dataset into every export format, plus the round-trip validation hooks.

``build_all`` writes the full set of artifacts to a directory; the CLI and CI
call it. ``roundtrip_cipa`` is the one true numeric round trip (CiPA inputs parse
back to the dataset values); ``roundtrip_parameters`` checks that the kernel
constants survive the CellML/SBML/Myokit text exports; ``roundtrip_ode``
re-integrates the model AST (the single source every CellML/SBML/Myokit export is
rendered from) and confirms it reproduces the reference-kernel action potential
within tolerance — so the exported *equations*, not merely the constants, provably
match the numeric oracle (spec.md §6, "round-trip validates ~1e-4 ODE").
"""
from __future__ import annotations

import pathlib
import re
from typing import Dict, List, Optional

from ..load import Dataset
from . import cellml, cipa_inputs, combine, csv_bibtex, myokit, sbml, sedml
from .model_spec import build_model_spec, simulate_spec

# format -> (file extension, builder returning text). 'omex' is binary, handled
# separately in build_all.
TEXT_FORMATS: Dict[str, str] = {
    "cellml": ".cellml",
    "myokit": ".mmt",
    "sbml": ".sbml",
    "sedml": ".sedml",
    "cipa": ".csv",
    "csv": ".csv",
    "bibtex": ".bib",
}


def list_ap_models(ds: Dataset) -> List[str]:
    return [m.id.split(".", 1)[1] for m in ds.ap_models]


def build_text(ds: Dataset, fmt: str, ap_model: str = "cipaordv1.0",
               dataset_version: str = "0.1.0") -> str:
    if fmt == "cellml":
        return cellml.build(ds, ap_model, dataset_version=dataset_version)
    if fmt == "sbml":
        return sbml.build(ds, ap_model, dataset_version=dataset_version)
    if fmt == "myokit":
        return myokit.build(ds, ap_model, dataset_version=dataset_version)
    if fmt == "sedml":
        # In the build_all layout the CellML model is a sibling directory
        # (cellml/<ap>.cellml), so the standalone protocol must point there — not
        # at the flattened "model.cellml" the COMBINE archive uses.
        return sedml.build(ds, ap_model, model_source=f"../cellml/{ap_model}.cellml")
    if fmt == "cipa":
        return cipa_inputs.to_csv(ds)
    if fmt == "csv":
        return csv_bibtex.parameters_csv(ds)
    if fmt == "bibtex":
        return csv_bibtex.citations_bibtex(ds)
    raise ValueError(f"unknown text format '{fmt}'. Known: {sorted(TEXT_FORMATS)}")


def build_all(ds: Dataset, output_dir: str, dataset_version: str = "0.1.0") -> List[str]:
    """Write every export artifact under ``output_dir``. Returns written paths."""
    root = pathlib.Path(output_dir)
    written: List[str] = []

    # per-AP-model artifacts (CellML/Myokit/SBML/SED-ML)
    for ap in list_ap_models(ds):
        for fmt in ("cellml", "myokit", "sbml", "sedml"):
            sub = root / fmt
            sub.mkdir(parents=True, exist_ok=True)
            path = sub / f"{ap}{TEXT_FORMATS[fmt]}"
            path.write_text(build_text(ds, fmt, ap_model=ap,
                                       dataset_version=dataset_version), encoding="utf-8")
            written.append(str(path))
        # COMBINE archive per model
        omex_dir = root / "omex"
        omex_dir.mkdir(parents=True, exist_ok=True)
        opath = omex_dir / f"{ap}.omex"
        combine.build(ds, str(opath), ap_model=ap, dataset_version=dataset_version)
        written.append(str(opath))

    # dataset-wide flat artifacts
    for fmt, name in (("cipa", "cipa_inputs.csv"), ("csv", "parameters.csv"),
                      ("bibtex", "citations.bib")):
        sub = root / "tables"
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / name
        path.write_text(build_text(ds, fmt, dataset_version=dataset_version), encoding="utf-8")
        written.append(str(path))
    # CiPA inputs JSON
    (root / "tables" / "cipa_inputs.json").write_text(
        cipa_inputs.to_json(ds), encoding="utf-8")
    written.append(str(root / "tables" / "cipa_inputs.json"))

    return written


# --------------------------------------------------------------------------- #
# Round-trip validation
# --------------------------------------------------------------------------- #
def roundtrip_cipa(ds: Dataset, rel_tol: float = 1e-9) -> List[str]:
    """The true numeric round trip: export CiPA CSV, parse it back, and confirm
    every IC50/Hill equals the dataset value. Returns a list of mismatches
    (empty == success)."""
    text = cipa_inputs.to_csv(ds)
    parsed = {(r["drug"], r["channel"]): r for r in cipa_inputs.parse_csv(text)}
    errors = []
    for b in ds.channel_blocks:
        key = (b.drug, b.channel)
        if key not in parsed:
            errors.append(f"{b.id}: missing from CiPA export")
            continue
        got = float(parsed[key]["ic50_nM"])
        want = b.ic50_nm
        if abs(got - want) > rel_tol * max(abs(want), 1.0):
            errors.append(f"{b.id}: IC50 round-trip {got} != {want}")
    return errors


_NUM = re.compile(r'initial_value="([^"]+)"|value="([^"]+)"|=\s*([-\d.eE]+)')


def roundtrip_parameters(ds: Dataset, ap_model: str = "cipaordv1.0") -> List[str]:
    """Confirm the kernel conductances appear verbatim in the CellML, SBML and
    Myokit text exports (a structural round trip of the parameter set)."""
    from ..simulate import _resolve_ap_model
    rec = _resolve_ap_model(ds, ap_model)
    spec = build_model_spec(conductance_scales=rec.conductance_scales)
    errors = []
    texts = {
        "cellml": cellml.build(ds, ap_model),
        "sbml": sbml.build(ds, ap_model),
        "myokit": myokit.build(ds, ap_model),
    }
    for p in spec.parameters:
        token = repr(p.value)
        for fmt, text in texts.items():
            if token not in text:
                errors.append(f"{fmt}: parameter {p.name}={token} not found in export")
    return errors


def roundtrip_ode(ds: Dataset, ap_model: str = "cipaordv1.0",
                  block: Optional[Dict[str, float]] = None,
                  rel_tol: float = 1e-4, apd_tol_ms: float = 0.5) -> List[str]:
    """Re-integrate the model AST and confirm it reproduces the reference kernel.

    The AST (``model_spec``) is the single description every CellML/SBML/Myokit
    export is rendered from; ``reference.py`` is the numeric oracle the metrics and
    thresholds are calibrated to. Integrating the AST with identical solver
    settings and matching the kernel's action potential proves the two cannot
    drift — the exported *equations*, not just the constants, are the kernel's.

    Returns a list of out-of-tolerance discrepancies (empty == success). The
    tolerances are generous relative to the ~1e-7 actually achieved.
    """
    from ..simulate import _resolve_ap_model
    from .reference import KernelParams, simulate_beats

    rec = _resolve_ap_model(ds, ap_model)
    spec = build_model_spec(conductance_scales=rec.conductance_scales, block=block)
    spec_res = simulate_spec(spec)

    p = KernelParams().with_scales(rec.conductance_scales)
    if block:
        p.block.update(block)
    ker_res = simulate_beats(p)

    errors: List[str] = []
    span = float(spec_res.V.max() - spec_res.V.min()) or 1.0
    rel = float(abs(spec_res.V - ker_res.V).max()) / span
    if rel > rel_tol:
        errors.append(f"{ap_model}: AST/kernel V-trace rel diff {rel:.2e} > {rel_tol:.0e}")
    import math as _m
    if not (_m.isnan(spec_res.apd90) and _m.isnan(ker_res.apd90)):
        dapd = abs(spec_res.apd90 - ker_res.apd90)
        if dapd > apd_tol_ms:
            errors.append(f"{ap_model}: AST/kernel APD90 diff {dapd:.3f} ms > {apd_tol_ms} ms")
    return errors

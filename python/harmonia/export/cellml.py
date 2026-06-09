"""CellML 2.0 export — the native language of cardiac electrophysiology and the
Physiome model repository, and the format Harmonia shares with Nidus.

The document is well-formed and structurally complete: every parameter and state
is a typed <variable>, every current / gate law is MathML, and the model carries
the MIRIAM RDF annotation (clinicalUse, tier, DOIs).

FIDELITY NOTE (Phase F): <cn> elements are tagged ``cellml:units="dimensionless"``
rather than fully dimensioned, so the file parses and is structurally complete
but is not yet libcellml dimension-checked. Full unit conformance and the
Myokit/OpenCOR cross-check against the canonical ORd CellML are the Phase-F
deliverable.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..load import Dataset
from . import annotate
from .model_spec import ModelSpec, build_model_spec

CELLML_NS = "http://www.cellml.org/cellml/2.0#"
MATHML_NS = "http://www.w3.org/1998/Math/MathML"

# CellML 2.0 built-in units used to define our custom units.
_UNIT_DEFS = {
    "ms": '<unit units="second" prefix="milli"/>',
    "mV": '<unit units="volt" prefix="milli"/>',
    "uA_per_uF": '<unit units="ampere" prefix="micro"/><unit units="farad" prefix="micro" exponent="-1"/>',
    "mS_per_uF": '<unit units="siemens" prefix="milli"/><unit units="farad" prefix="micro" exponent="-1"/>',
    "uF_per_cm2": '<unit units="farad" prefix="micro"/><unit units="metre" prefix="centi" exponent="-2"/>',
}


def _ml(expr) -> str:
    """Render an Expr to MathML with cn units tagged for CellML."""
    return expr.mathml().replace("<cn>", '<cn cellml:units="dimensionless">')


def _render(spec: ModelSpec, tier: str, dataset_version: str, dois: List[str]) -> str:
    L: List[str] = ['<?xml version="1.0" encoding="UTF-8"?>']
    L.append("<!--")
    L.append(annotate.provenance_comment(dois))
    L.append("-->")
    L.append(f'<model xmlns="{CELLML_NS}" '
             f'xmlns:cellml="{CELLML_NS}" name="{spec.name}">')

    # unit definitions
    used_units = {p.units for p in spec.parameters}
    used_units |= {s.units for s in spec.states}
    used_units |= {a.units for a in spec.assignments}
    for u in sorted(used_units):
        if u in _UNIT_DEFS:
            L.append(f'  <units name="{u}">{_UNIT_DEFS[u]}</units>')

    L.append(f'  <component name="cell">')
    L.append(f'    <variable name="time" units="ms"/>')
    for p in spec.parameters:
        L.append(f'    <variable name="{p.name}" units="{p.units}" '
                 f'initial_value="{p.value!r}"/>')
    for s in spec.states:
        L.append(f'    <variable name="{s.name}" units="{s.units}" '
                 f'initial_value="{s.init!r}"/>')
    for a in spec.assignments:
        L.append(f'    <variable name="{a.name}" units="{a.units}"/>')

    L.append(f'    <math xmlns="{MATHML_NS}">')
    # algebraic assignments
    for a in spec.assignments:
        L.append("      <apply><eq/>"
                 f"<ci>{a.name}</ci>{_ml(a.expr)}</apply>")
    # ODEs
    for s in spec.states:
        ode = (f"<apply><diff/><bvar><ci>time</ci></bvar>"
               f"<ci>{s.name}</ci></apply>")
        L.append(f"      <apply><eq/>{ode}{_ml(s.rate)}</apply>")
    L.append("    </math>")
    L.append("  </component>")

    # annotation
    meta_id = spec.name
    L.append(annotate.rdf_block(meta_id, tier, dataset_version, dois, indent="  "))
    L.append("</model>")
    return "\n".join(L) + "\n"


def build(ds: Dataset, ap_model: str = "cipaordv1.0",
          drug: Optional[str] = None, block: Optional[Dict[str, float]] = None,
          dataset_version: str = "0.1.0") -> str:
    from ..simulate import _resolve_ap_model
    rec = _resolve_ap_model(ds, ap_model)
    spec = build_model_spec(name=f"harmonia_{rec.id.split('.')[-1]}".replace(".", "_"),
                            conductance_scales=rec.conductance_scales, block=block)
    cit = ds.citation(rec.primary_citation)
    dois = [cit.doi] if cit and cit.doi else []
    return _render(spec, tier=rec.tier, dataset_version=dataset_version, dois=dois)

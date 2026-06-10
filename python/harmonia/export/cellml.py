"""CellML 2.0 export — the native language of cardiac electrophysiology and the
Physiome model repository, and the format Harmonia shares with Nidus.

The document is well-formed and structurally complete: every parameter and state
is a typed <variable>, every current / gate law is MathML, and the model carries
the MIRIAM RDF annotation (clinicalUse, tier, DOIs).

CONFORMANCE (Phase F). ``conformance_violations`` machine-checks the
*declaration-level* CellML-2.0 unit requirements with no external dependency
(every variable and every <cn> literal carries a units name that is either a
CellML-2.0 built-in unit or is defined by a <units> block in the model — no
dangling unit references).

VALIDATION. ``validity_violations`` runs the *canonical* CellML 2.0 library
(libCellML's Parser + Validator) over the exported model, checking the full
ruleset — MathML correctness, variable/units references, interface consistency,
duplicate/cyclic definitions — well beyond the declaration-level check, and the
lightweight (no-simulation-engine) part of the spec §6 "cross-check against the
canonical CellML." It is wired into ``export --all`` and CI alongside the SBML
``consistency_violations`` gate. What remains an optional local step is full
dimensional-consistency checking and the Myokit/OpenCOR cross-check against the
*canonical* ORd CellML — they need a heavy simulation engine and are not run in
CI. (<cn> literals are tagged ``cellml:units="dimensionless"`` rather than
carrying the dimension implied by their additive context — the same
declaration-level limitation the SBML export documents.)
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

from ..load import Dataset
from . import annotate
from .model_spec import ModelSpec, build_model_spec

CELLML_NS = "http://www.cellml.org/cellml/2.0#"
MATHML_NS = "http://www.w3.org/1998/Math/MathML"

# The CellML-2.0 built-in (SI-derived) units; any other unit a model uses must be
# declared in a <units> block. (CellML 2.0 §19.2.)
_CELLML2_BUILTIN_UNITS = frozenset({
    "ampere", "becquerel", "candela", "coulomb", "dimensionless", "farad",
    "gram", "gray", "henry", "hertz", "joule", "katal", "kelvin", "kilogram",
    "litre", "lumen", "lux", "metre", "mole", "newton", "ohm", "pascal",
    "radian", "second", "siemens", "sievert", "steradian", "tesla", "volt",
    "watt", "weber",
})

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

    L.append('  <component name="cell">')
    L.append('    <variable name="time" units="ms"/>')
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


def conformance_violations(text: str) -> List[str]:
    """Declaration-level CellML-2.0 unit-conformance check (no external engine).

    Verifies the unit *declarations* a CellML-2.0 document requires:

      1. every ``<variable>`` declares a ``units`` attribute;
      2. every ``<cn>`` literal carries a ``cellml:units`` attribute;
      3. every units name referenced (by a variable or a ``<cn>``) is either a
         CellML-2.0 built-in unit or is defined by a ``<units name="...">`` block
         in the model.

    Returns the list of violations (empty == conformant). This is NOT full
    dimensional-consistency validation — see the module docstring — but it is the
    part checkable in CI without a heavy engine, and it catches the real export
    bug: a variable or literal referencing a unit the model never defines.
    """
    violations: List[str] = []
    root = ET.fromstring(text)
    defined = {u.get("name") for u in root.iter(f"{{{CELLML_NS}}}units")}
    known = _CELLML2_BUILTIN_UNITS | {d for d in defined if d}
    units_attr = f"{{{CELLML_NS}}}units"
    for var in root.iter(f"{{{CELLML_NS}}}variable"):
        name = var.get("name", "?")
        u = var.get("units")
        if not u:
            violations.append(f"variable '{name}' declares no units")
        elif u not in known:
            violations.append(f"variable '{name}' references undefined units '{u}'")
    for cn in root.iter(f"{{{MATHML_NS}}}cn"):
        u = cn.get(units_attr)
        if not u:
            violations.append(f"<cn>{(cn.text or '').strip()}</cn> carries no cellml:units")
        elif u not in known:
            violations.append(f"<cn> references undefined units '{u}'")
    return violations


def validity_violations(text: str) -> List[str]:
    """Validate the exported CellML *model* with libCellML (the canonical CellML
    2.0 library) — the strong analog of the declaration-level
    ``conformance_violations`` and of the SBML export's ``consistency_violations``.

    libCellML's Parser + Validator check the full CellML 2.0 ruleset (MathML
    correctness, variable/units references, interface consistency, duplicate and
    cyclic definitions), so this catches model errors the hand-rolled
    declaration check cannot (e.g. a MathML ``<ci>`` naming no declared variable).

    The embedded ``<rdf:RDF>`` MIRIAM annotation (clinicalUse / tier / DOIs,
    spec §7) is a foreign-namespace metadata *island*: CellML 2.0 has no blessed
    annotation wrapper (unlike SBML's ``<annotation>``), so libCellML's parser
    reports it as foreign content. It is stripped before validation here — the
    identical RDF also travels in the COMBINE archive's ``metadata.rdf`` and
    inside the SBML ``<annotation>`` — so this gate validates the model proper.

    Returns ERROR-level issues (empty == valid). Skips (returns ``[]``) if
    libCellML is absent; CI installs it via the ``dev`` extra.
    """
    try:
        import libcellml
    except ImportError:
        return []
    model_only = re.sub(r"\s*<rdf:RDF.*?</rdf:RDF>", "", text, flags=re.DOTALL)
    parser = libcellml.Parser()
    model = parser.parseModel(model_only)
    validator = libcellml.Validator()
    validator.validateModel(model)
    err = int(libcellml.Issue.Level.ERROR)
    out: List[str] = []
    for src, label in ((parser, "parser"), (validator, "validator")):
        for i in range(src.issueCount()):
            issue = src.issue(i)
            if int(issue.level()) == err:
                out.append(f"CellML {label}: {issue.description()}")
    return out


def build(ds: Dataset, ap_model: str = "cipaordv1.0",
          drug: Optional[str] = None, block: Optional[Dict[str, float]] = None,
          dataset_version: Optional[str] = None) -> str:
    from ..simulate import _resolve_ap_model
    from . import default_dataset_version
    dataset_version = dataset_version or default_dataset_version()
    rec = _resolve_ap_model(ds, ap_model)
    spec = build_model_spec(name=f"harmonia_{rec.id.split('.')[-1]}".replace(".", "_"),
                            conductance_scales=rec.conductance_scales, block=block)
    cit = ds.citation(rec.primary_citation)
    dois = [cit.doi] if cit and cit.doi else []
    return _render(spec, tier=rec.tier, dataset_version=dataset_version, dois=dois)

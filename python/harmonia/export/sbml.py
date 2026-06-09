"""SBML Level 3 Version 2 export. The AP models are ODE systems, so they map onto
SBML <parameter> + <rateRule> (states) and <assignmentRule> (intermediates),
giving continuity with COPASI / Tellurium / BioModels and the rest of the family.

States and intermediates are global parameters with ``constant="false"``; the
time symbol uses the SBML csymbol. Every parameter declares ``units`` (mirroring
the CellML export's unit metadata, defined in ``<listOfUnitDefinitions>``), and
the model declares ``timeUnits``. The model carries the same clinicalUse / tier /
DOI RDF annotation as the CellML export.

VALIDATION (the SBML analog of CellML's ``conformance_violations``).
``consistency_violations`` runs the *canonical* SBML validator (libSBML's
``checkConsistency``) over the exported document and returns any ERROR/FATAL-
severity problems — so "SBML → COPASI/Tellurium/BioModels" is a *verified* claim,
not merely asserted (spec.md §7). Unit-consistency *warnings* are tolerated:
like the CellML export, this is a declaration-level model (numeric ``<cn>``
literals are dimensionless rather than carrying the dimension implied by their
additive context), not a fully dimensionally-audited one. ``export --all`` and CI
run this gate where libSBML is installed.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..load import Dataset
from . import annotate
from .model_spec import ModelSpec, build_model_spec

SBML_NS = "http://www.sbml.org/sbml/level3/version2/core"
MATHML_NS = "http://www.w3.org/1998/Math/MathML"
_TIME_CSYMBOL = ('<csymbol encoding="text" '
                 'definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>')

# SBML unit definitions mirroring the CellML export's custom units (cellml._UNIT_DEFS),
# so the two formats carry identical unit metadata. "dimensionless" is an SBML
# base unit and needs no definition. Each <unit> is (multiplier * 10^scale *
# kind)^exponent; e.g. ms = (10^-3 second)^1, uA_per_uF = ampere/farad (the µ
# prefixes cancel: (10^-6 A)/(10^-6 F)).
_SBML_UNIT_DEFS: Dict[str, str] = {
    "ms": '<unit kind="second" exponent="1" scale="-3" multiplier="1"/>',
    "mV": '<unit kind="volt" exponent="1" scale="-3" multiplier="1"/>',
    "uA_per_uF": ('<unit kind="ampere" exponent="1" scale="-6" multiplier="1"/>'
                  '<unit kind="farad" exponent="-1" scale="-6" multiplier="1"/>'),
    "mS_per_uF": ('<unit kind="siemens" exponent="1" scale="-3" multiplier="1"/>'
                  '<unit kind="farad" exponent="-1" scale="-6" multiplier="1"/>'),
    "uF_per_cm2": ('<unit kind="farad" exponent="1" scale="-6" multiplier="1"/>'
                   '<unit kind="metre" exponent="-2" scale="-2" multiplier="1"/>'),
}


def _ml(expr) -> str:
    """Render an Expr to SBML MathML; map the time variable to the SBML csymbol."""
    return expr.mathml().replace("<ci>time</ci>", _TIME_CSYMBOL)


def _math(inner: str) -> str:
    return f'<math xmlns="{MATHML_NS}">{inner}</math>'


def _render(spec: ModelSpec, tier: str, dataset_version: str, dois: List[str]) -> str:
    L: List[str] = ['<?xml version="1.0" encoding="UTF-8"?>']
    L.append("<!--")
    L.append(annotate.provenance_comment(dois))
    L.append("-->")
    L.append(f'<sbml xmlns="{SBML_NS}" level="3" version="2">')
    L.append(f'  <model id="{spec.name}" name="{spec.name}" timeUnits="{spec.time_units}">')

    # unit definitions (only the custom units actually used; "dimensionless" is
    # an SBML base unit and is never defined)
    used_units = {p.units for p in spec.parameters}
    used_units |= {s.units for s in spec.states}
    used_units |= {a.units for a in spec.assignments}
    used_units.add(spec.time_units)
    L.append("    <listOfUnitDefinitions>")
    for u in sorted(used_units):
        if u in _SBML_UNIT_DEFS:
            L.append(f'      <unitDefinition id="{u}"><listOfUnits>'
                     f'{_SBML_UNIT_DEFS[u]}</listOfUnits></unitDefinition>')
    L.append("    </listOfUnitDefinitions>")

    # parameters: constants + states + intermediates (each declares units)
    L.append("    <listOfParameters>")
    for p in spec.parameters:
        L.append(f'      <parameter id="{p.name}" value="{p.value!r}" '
                 f'units="{p.units}" constant="true"/>')
    for s in spec.states:
        L.append(f'      <parameter id="{s.name}" value="{s.init!r}" '
                 f'units="{s.units}" constant="false"/>')
    for a in spec.assignments:
        L.append(f'      <parameter id="{a.name}" units="{a.units}" constant="false"/>')
    L.append("    </listOfParameters>")

    # rules
    L.append("    <listOfRules>")
    for a in spec.assignments:
        L.append(f'      <assignmentRule variable="{a.name}">{_math(_ml(a.expr))}</assignmentRule>')
    for s in spec.states:
        L.append(f'      <rateRule variable="{s.name}">{_math(_ml(s.rate))}</rateRule>')
    L.append("    </listOfRules>")

    # annotation
    L.append("    <annotation>")
    L.append(annotate.rdf_block(spec.name, tier, dataset_version, dois, indent="      "))
    L.append("    </annotation>")
    L.append("  </model>")
    L.append("</sbml>")
    return "\n".join(L) + "\n"


def consistency_violations(text: str) -> List[str]:
    """Validate exported SBML with libSBML (the canonical SBML validator).

    Returns the ERROR/FATAL-severity problems libSBML's ``checkConsistency``
    reports (empty == valid). Unit-consistency *warnings* are tolerated — see the
    module docstring: this is a declaration-level model, not a fully
    dimensionally-audited one.

    If libSBML is not installed the check is skipped and returns ``[]`` (CI
    installs ``python-libsbml`` via the ``dev`` extra, so the gate runs there).
    """
    try:
        import libsbml
    except ImportError:
        return []
    doc = libsbml.readSBMLFromString(text)
    doc.checkConsistency()
    out: List[str] = []
    for i in range(doc.getNumErrors()):
        e = doc.getError(i)
        if e.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
            out.append(f"SBML [{e.getErrorId()}] line {e.getLine()}: {e.getShortMessage()}")
    return out


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

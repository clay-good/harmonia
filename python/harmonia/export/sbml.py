"""SBML Level 3 Version 2 export. The AP models are ODE systems, so they map onto
SBML <parameter> + <rateRule> (states) and <assignmentRule> (intermediates),
giving continuity with COPASI / Tellurium / BioModels and the rest of the family.

States and intermediates are global parameters with ``constant="false"``; the
time symbol uses the SBML csymbol. The model carries the same clinicalUse / tier
/ DOI RDF annotation as the CellML export.
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
    L.append(f'  <model id="{spec.name}" name="{spec.name}" timeUnits="dimensionless">')

    # parameters: constants + states + intermediates
    L.append("    <listOfParameters>")
    for p in spec.parameters:
        L.append(f'      <parameter id="{p.name}" value="{p.value!r}" constant="true"/>')
    for s in spec.states:
        L.append(f'      <parameter id="{s.name}" value="{s.init!r}" constant="false"/>')
    for a in spec.assignments:
        L.append(f'      <parameter id="{a.name}" constant="false"/>')
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

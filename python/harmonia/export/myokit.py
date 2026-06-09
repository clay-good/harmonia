"""Myokit ``.mmt`` export — a complete, runnable single-cell model + pacing
protocol, generated from the model spec. Myokit is the dominant open Python tool
for cardiac AP simulation and CiPA-style work, so this is the most directly
runnable artifact Harmonia produces.

Myokit is NOT a load-time dependency: this builder emits text. If Myokit is
installed, the file loads and runs; CI validates structure without it.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..load import Dataset
from . import annotate
from .model_spec import ModelSpec, build_model_spec


def _render(spec: ModelSpec, tier: str, dataset_version: str,
            dois: List[str]) -> str:
    L: List[str] = []
    comment = annotate.provenance_comment(dois)
    for line in comment.splitlines():
        L.append(f"# {line}")
    L.append("")
    L.append("[[model]]")
    L.append(f"name: {spec.name}")
    L.append("# Initial values")
    for s in spec.states:
        L.append(f"cell.{s.name} = {s.init!r}")
    L.append("")
    L.append("[engine]")
    L.append("time = 0 [ms]")
    L.append("    in [ms]")
    L.append("    bind time")
    L.append("pace = 0")
    L.append("    bind pace")
    L.append("")
    L.append("[cell]")
    # parameters (constants)
    for p in spec.parameters:
        L.append(f"{p.name} = {p.value!r}    # {p.label} [{p.units}]")
    L.append("")
    # ordered assignments, overriding i_stim to use the pacing binding
    for a in spec.assignments:
        if a.name == "i_stim":
            L.append("i_stim = stim_amplitude * engine.pace")
            continue
        L.append(f"{a.name} = {a.expr.mmt()}")
    L.append("")
    # state rates
    for s in spec.states:
        L.append(f"dot({s.name}) = {s.rate.mmt()}")
    L.append("")
    # pacing protocol: level start length period multiplier
    L.append("[[protocol]]")
    L.append("# level  start  length  period  multiplier")
    period = spec.parameter("stim_period").value
    dur = spec.parameter("stim_duration").value
    L.append(f"1.0  0  {dur:g}  {period:g}  0")
    L.append("")
    return "\n".join(L)


def build(ds: Dataset, ap_model: str = "cipaordv1.0",
          drug: Optional[str] = None, block: Optional[Dict[str, float]] = None,
          dataset_version: str = "0.1.0") -> str:
    from ..simulate import _resolve_ap_model
    rec = _resolve_ap_model(ds, ap_model)
    spec = build_model_spec(name=f"harmonia_{rec.id.split('.')[-1]}",
                            conductance_scales=rec.conductance_scales, block=block)
    doi = ds.citation(rec.primary_citation).doi if ds.citation(rec.primary_citation) else None
    return _render(spec, tier=rec.tier, dataset_version=dataset_version,
                   dois=[doi] if doi else [])

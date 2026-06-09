"""Population-of-models proarrhythmia assessment (Phase E) — HYPOTHESIS-TIER.

Where ``simulate.assess`` propagates *input* (IC50) variability, this propagates
*physiological* variability: instead of one ventricular myocyte it builds a
population of virtual myocytes by sampling the kernel conductances (lognormal,
per-channel CVs from a ``population`` record), then runs the drug at a fixed
exposure across the population. Individuals with intrinsically reduced
repolarization reserve (e.g. low IKr/IKs) cross into high risk while robust
individuals do not — so a single drug produces a *spread* of risk
classifications across the population.

NON-NEGOTIABLE (spec.md §3, §10). The population CVs are illustrative, NOT
calibrated to human data, so every population assessment is **Tier D** and
labelled **NOT FOR PREDICTION**. It is a hypothesis-generating, methodology view,
never a per-patient or population safety claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .load import Dataset
from .records import ChannelBlock, Population
from .exposure import resolve_free_exposure
from .export.reference import KernelParams, simulate_beats, hill_block_factor, BLOCKABLE
from .simulate import (RISK_LABELS, classify_metric, DEFAULT_METRIC,
                       _channel_draws, _resolve_ap_model, REFERENCE_EXPOSURE_MULTIPLE)

NOT_FOR_PREDICTION = ("HYPOTHESIS-TIER — NOT FOR PREDICTION. Illustrative "
                      "inter-individual variability (uncalibrated); Tier D.")


@dataclass
class PopulationAssessment:
    drug: str
    ap_model: str
    population: str
    reference_exposure_nM: float
    metric: str
    n_models: int
    qnet_distribution: np.ndarray
    dapd90_distribution: np.ndarray
    classification_distribution: Dict[str, float]   # fraction of the population in each class
    susceptible_fraction: float                      # fraction classified "high"
    repolarization_failures: int
    tier: str = "D"                                  # always capped at D (non-predictive)
    warnings: List[str] = field(default_factory=list)
    excluded_channels: List[str] = field(default_factory=list)
    clinical_use: str = ("PROHIBITED — research / safety-methodology / education only; "
                         "not a regulatory determination")

    def summary(self) -> str:
        dist = ", ".join(f"{k} {self.classification_distribution.get(k, 0):.0%}"
                         for k in RISK_LABELS)
        lines = [
            f"population assessment  drug={self.drug}  ap_model={self.ap_model}  "
            f"population={self.population}  tier={self.tier}",
            f"reference exposure = {self.reference_exposure_nM:g} nM",
            f"{self.n_models} virtual myocytes  [{self.metric}] class mix: {dist}",
            f"SUSCEPTIBLE fraction (classified high): {self.susceptible_fraction:.0%}",
        ]
        if self.repolarization_failures:
            lines.append(f"repolarization failures (EAD/non-repolarizing): "
                         f"{self.repolarization_failures}/{self.n_models}")
        if self.excluded_channels:
            lines.append(f"EXCLUDED (unidentifiable IC50): {', '.join(self.excluded_channels)}")
        lines.append("*** " + NOT_FOR_PREDICTION + " ***")
        for w in self.warnings:
            lines.append(f"  warn: {w}")
        lines.append("NOTE: a population spread, not a verdict. " + self.clinical_use)
        return "\n".join(lines)


def _population_record(ds: Dataset, population: str) -> Population:
    rec = ds.population(population)
    if rec is None:
        raise KeyError(f"unknown population '{population}'. Known: "
                       f"{[p.id for p in ds.populations]}")
    if not isinstance(rec, Population):
        raise TypeError(f"{rec.id} is not a population record")
    return rec


def assess_population(ds: Dataset, drug: str, population: str = "illustrative_v0",
                      ap_model: str = "cipaordv1.0", n_models: Optional[int] = None,
                      exposure_nM: Optional[float] = None, exposure_kind: str = "free",
                      exposure_multiple: float = REFERENCE_EXPOSURE_MULTIPLE,
                      metric: str = DEFAULT_METRIC, seed: int = 0,
                      n_beats: int = 3) -> PopulationAssessment:
    """Run ``drug`` at a fixed exposure across a population of virtual myocytes
    and report the spread of risk classifications. Always Tier D / non-predictive.
    """
    if metric not in ("qnet", "apd90"):
        raise ValueError(f"metric must be 'qnet' or 'apd90', got {metric!r}")
    drug_l = drug.lower()
    blocks = [b for b in ds.blocks_for(drug_l) if isinstance(b, ChannelBlock)]
    if not blocks:
        raise KeyError(f"no channel-block records for drug '{drug}'")

    pop = _population_record(ds, population)
    n = n_models or pop.n_default
    cv = pop.conductance_cv

    ref = ds.drug_reference(drug_l)
    reference_exposure = resolve_free_exposure(
        ref, exposure_nM=exposure_nM, exposure_kind=exposure_kind,
        exposure_multiple=exposure_multiple)

    ap_rec = _resolve_ap_model(ds, ap_model)
    scales = ap_rec.conductance_scales

    # point IC50 per identifiable channel (population varies physiology, not inputs)
    draws = _channel_draws(blocks)
    ic50 = {d.channel: 10 ** d.mu_log10 for d in draws}
    hill_by = {d.channel: d.hill for d in draws}
    excluded = [f"{b.channel}" for b in blocks if not b.identifiable]

    bf = {c: 1.0 for c in BLOCKABLE}
    for ch, c50 in ic50.items():
        if ch in bf:
            bf[ch] = hill_block_factor(reference_exposure, c50, hill_by[ch])

    rng = np.random.default_rng(seed)
    channels = list(cv.keys())
    qn = np.empty(n)
    apd = np.empty(n)
    dapd = np.empty(n)
    classes: List[str] = []
    failures = 0
    for i in range(n):
        # lognormal multiplier per conductance: exp(N(0, ln(1+cv^2))) keeps mean ~1
        mult = {}
        for ch in channels:
            sigma = np.sqrt(np.log(1.0 + cv[ch] ** 2))
            mult[ch] = float(np.exp(rng.normal(0.0, sigma)))
        base = KernelParams().with_scales(scales).with_conductance_multipliers(mult)
        p = KernelParams(base.gNa, base.gNaL, base.gto, base.gCaL, base.gKr,
                         base.gKs, base.gK1, base.gNaCa, dict(base.block))
        p.block.update(bf)
        r = simulate_beats(p, n_beats=n_beats)
        qn[i] = r.qnet
        apd[i] = r.apd90
        # the per-model drug-free baseline is only needed for the APD90 metric
        if metric == "apd90":
            drug_free = simulate_beats(base, n_beats=n_beats)
            dapd[i] = (100.0 * (r.apd90 - drug_free.apd90) / drug_free.apd90
                       if drug_free.apd90 else float("nan"))
        else:
            dapd[i] = float("nan")
        if r.repolarization_failed or r.ead:
            failures += 1
        classes.append(classify_metric(metric, dapd[i], qn[i]))

    counts = {lab: classes.count(lab) / n for lab in RISK_LABELS}
    warnings = [f"population CVs are illustrative (uncalibrated): {cv}"]
    if excluded:
        warnings.append(f"unidentifiable channels excluded: {excluded}")

    return PopulationAssessment(
        drug=drug_l, ap_model=ap_rec.id, population=pop.id,
        reference_exposure_nM=reference_exposure, metric=metric, n_models=n,
        qnet_distribution=qn, dapd90_distribution=dapd,
        classification_distribution=counts, susceptible_fraction=counts.get("high", 0.0),
        repolarization_failures=failures, tier="D", warnings=warnings,
        excluded_channels=excluded,
    )

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

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .load import Dataset
from .records import ChannelBlock, Population
from .exposure import resolve_free_exposure
from .export.reference import KernelParams, simulate_beats, hill_block_factor, BLOCKABLE
from .simulate import (RISK_LABELS, classify_metric, DEFAULT_METRIC, flip_ci,
                       _channel_draws, _resolve_ap_model, REFERENCE_EXPOSURE_MULTIPLE)

NOT_FOR_PREDICTION = ("HYPOTHESIS-TIER — NOT FOR PREDICTION. Illustrative "
                      "inter-individual variability (uncalibrated); Tier D.")
NOT_FOR_PREDICTION_CALIBRATED = (
    "HYPOTHESIS-TIER — NOT FOR PREDICTION. Drug-free-plausibility-calibrated "
    "population (Britton 2013 method) with KERNEL-plausibility acceptance ranges, "
    "NOT a fit to patient data; Tier D.")


def _fmt_susceptible(frac: float, ci: Tuple[float, float], n: int) -> str:
    """Render the susceptible-fraction line with its Wilson CI over n myocytes (v0.7)."""
    lo, hi = ci
    if n <= 0 or math.isnan(lo):
        return f"{frac:.0%}"
    return f"{frac:.0%} (95% CI {lo:.0%}–{hi:.0%}, {n} myocytes)"


# v0.5 — biomarker accessors for the calibration acceptance test, in the kernel's
# own units (the calibration ranges are kernel-plausibility bounds, not patient-fit).
def _biomarker_values(r) -> Dict[str, float]:
    return {"apd90_ms": r.apd90, "vrest_mv": r.vrest, "vpeak_mv": r.vpeak,
            "triangulation_ms": r.triangulation}


@dataclass
class CalibrationResult:
    """The accepted sub-population from a Britton-2013 calibration-by-rejection,
    plus the acceptance bookkeeping (spec v0.5). ``multipliers`` are the per-channel
    conductance draws of the accepted virtual myocytes; ``baseline_beats`` are their
    cached drug-free beats (reused as the APD90-metric baseline)."""
    multipliers: List[Dict[str, float]]
    baseline_beats: list
    n_candidates: int
    acceptance_rate: float
    rejection_reasons: Dict[str, int]
    repolarization_failures: int


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
    # v0.7 — Wilson 95% CI of the susceptible fraction over the n_models myocytes.
    susceptible_fraction_ci: Tuple[float, float]
    repolarization_failures: int
    tier: str = "D"                                  # always capped at D (non-predictive)
    conductance_scale: Dict[str, float] = field(default_factory=dict)  # v0.3 disease mean shift
    warnings: List[str] = field(default_factory=list)
    excluded_channels: List[str] = field(default_factory=list)
    # v0.5 experimentally-calibrated population (Britton 2013) bookkeeping
    calibrated: bool = False
    acceptance_rate: float = float("nan")
    n_candidates: int = 0
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    clinical_use: str = ("PROHIBITED — research / safety-methodology / education only; "
                         "not a regulatory determination")

    def summary(self) -> str:
        dist = ", ".join(f"{k} {self.classification_distribution.get(k, 0):.0%}"
                         for k in RISK_LABELS)
        header = (f"population assessment  drug={self.drug}  ap_model={self.ap_model}  "
                  f"population={self.population}  tier={self.tier}")
        if self.conductance_scale:
            header += f"  [DISEASE background: mean shift {self.conductance_scale}]"
        if self.calibrated:
            header += "  [CALIBRATED: drug-free biomarker acceptance]"
        lines = [
            header,
            f"reference exposure = {self.reference_exposure_nM:g} nM",
            f"{self.n_models} virtual myocytes  [{self.metric}] class mix: {dist}",
            f"SUSCEPTIBLE fraction (classified high): "
            f"{_fmt_susceptible(self.susceptible_fraction, self.susceptible_fraction_ci, self.n_models)}",
        ]
        if self.calibrated:
            rej = ", ".join(f"{k} {v}" for k, v in self.rejection_reasons.items() if v) or "none"
            lines.append(f"CALIBRATION: {self.n_models}/{self.n_candidates} candidates accepted "
                         f"({self.acceptance_rate:.0%}); rejected by: {rej}")
        if self.repolarization_failures:
            lines.append(f"repolarization failures (EAD/non-repolarizing): "
                         f"{self.repolarization_failures}/{self.n_models}")
        if self.excluded_channels:
            lines.append(f"EXCLUDED (unidentifiable IC50): {', '.join(self.excluded_channels)}")
        not_for_pred = (NOT_FOR_PREDICTION_CALIBRATED if self.calibrated
                        else NOT_FOR_PREDICTION)
        lines.append("*** " + not_for_pred + " ***")
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


def _draw_multiplier(rng, channels, cv, scale) -> Dict[str, float]:
    """One virtual myocyte's per-channel conductance multiplier: the disease MEAN
    shift ``s_c`` (v0.3, default 1) times the lognormal inter-individual draw
    ``exp(N(0, ln(1+cv^2)))`` (mean ~1) — ``g = s_c·λ·g_healthy`` (spec v0.3 §2).
    A zero-CV channel consumes no RNG, so the draw sequence is identical to v0.1."""
    mult = {}
    for ch in channels:
        cvv = cv.get(ch, 0.0)
        draw = float(np.exp(rng.normal(0.0, np.sqrt(np.log(1.0 + cvv ** 2))))) if cvv > 0 else 1.0
        mult[ch] = scale.get(ch, 1.0) * draw
    return mult


def calibrate_population(ds: Dataset, population: str = "calibrated_v0",
                         ap_model: str = "cipaordv1.0", n_models: Optional[int] = None,
                         seed: int = 0, n_beats: int = 3) -> CalibrationResult:
    """Experimentally-calibrated population of models (Britton et al. 2013, spec v0.5):
    draw candidate myocytes from the prior conductance cloud and admit one only if its
    DRUG-FREE action-potential biomarkers all fall within the population record's
    accepted ranges (and it repolarizes). The accepted set excludes the abnormal
    repolarization tail of the raw prior, leaving a physiologically plausible
    population. The ranges are kernel-plausibility bounds, NOT a fit to patient data —
    this buys plausibility, not predictiveness (the assessment stays Tier D)."""
    pop = _population_record(ds, population)
    cal = pop.calibration
    if cal is None:
        raise ValueError(f"population '{population}' has no calibration block; use "
                         f"assess_population for an uncalibrated prior-cloud population")
    n = n_models or pop.n_default
    cv = pop.conductance_cv
    scale = pop.conductance_scale
    ranges = cal["biomarkers"]
    max_oversample = int(cal.get("max_oversample", 40))
    ap_rec = _resolve_ap_model(ds, ap_model)
    scales = ap_rec.conductance_scales

    rng = np.random.default_rng(seed)
    channels = list(cv.keys()) + [c for c in scale if c not in cv]
    accepted_mult: List[Dict[str, float]] = []
    accepted_beats: list = []
    reasons = {k: 0 for k in ranges}
    repol_failures = 0
    n_candidates = 0
    cap = n * max_oversample
    while len(accepted_mult) < n and n_candidates < cap:
        n_candidates += 1
        mult = _draw_multiplier(rng, channels, cv, scale)
        base = KernelParams().with_scales(scales).with_conductance_multipliers(mult)
        r = simulate_beats(base, n_beats=n_beats)
        if r.repolarization_failed or r.ead:
            repol_failures += 1
            continue
        bm = _biomarker_values(r)
        ok = True
        for k, lim in ranges.items():
            v = bm.get(k)
            if v is None or math.isnan(v) or not (lim["min"] <= v <= lim["max"]):
                reasons[k] += 1
                ok = False
        if ok:
            accepted_mult.append(mult)
            accepted_beats.append(r)
    acceptance_rate = len(accepted_mult) / n_candidates if n_candidates else float("nan")
    return CalibrationResult(accepted_mult, accepted_beats, n_candidates,
                             acceptance_rate, reasons, repol_failures)


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
    scale = pop.conductance_scale     # v0.3 disease MEAN shift (default 1 per channel)

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

    # Build the population's virtual myocytes (per-channel conductance multipliers).
    # Uncalibrated: draw n from the prior cloud. Calibrated (v0.5): accept only
    # drug-free-plausible myocytes via Britton-2013 calibration-by-rejection, which
    # also caches each accepted myocyte's drug-free beat (reused as the APD90 baseline).
    # iterate cv channels first (RNG order preserved for backward compat), then any
    # disease-only channels that carry a mean shift but no variability
    channels = list(cv.keys()) + [c for c in scale if c not in cv]
    cal_result: Optional[CalibrationResult] = None
    if pop.is_calibrated:
        cal_result = calibrate_population(ds, population=pop.id, ap_model=ap_rec.id,
                                          n_models=n, seed=seed, n_beats=n_beats)
        myo_mult = cal_result.multipliers
        myo_base = cal_result.baseline_beats
        n = len(myo_mult)
        if n == 0:
            raise RuntimeError(
                f"calibration of '{pop.id}' accepted no myocytes in "
                f"{cal_result.n_candidates} candidates — check the acceptance ranges")
    else:
        rng = np.random.default_rng(seed)
        myo_mult = [_draw_multiplier(rng, channels, cv, scale) for _ in range(n)]
        myo_base = [None] * n

    qn = np.empty(n)
    apd = np.empty(n)
    dapd = np.empty(n)
    classes: List[str] = []
    failures = 0
    for i in range(n):
        mult = myo_mult[i]
        base = KernelParams().with_scales(scales).with_conductance_multipliers(mult)
        p = KernelParams(base.gNa, base.gNaL, base.gto, base.gCaL, base.gKr,
                         base.gKs, base.gK1, base.gNaCa, dict(base.block))
        p.block.update(bf)
        r = simulate_beats(p, n_beats=n_beats)
        qn[i] = r.qnet
        apd[i] = r.apd90
        # the per-model drug-free baseline is only needed for the APD90 metric; the
        # calibrated path already computed it during acceptance, so reuse the cache.
        if metric == "apd90":
            drug_free = myo_base[i] if myo_base[i] is not None else simulate_beats(base, n_beats=n_beats)
            dapd[i] = (100.0 * (r.apd90 - drug_free.apd90) / drug_free.apd90
                       if drug_free.apd90 else float("nan"))
        else:
            dapd[i] = float("nan")
        if r.repolarization_failed or r.ead:
            failures += 1
        classes.append(classify_metric(metric, dapd[i], qn[i]))

    counts = {lab: classes.count(lab) / n for lab in RISK_LABELS}
    if cal_result is not None:
        warnings = [
            f"EXPERIMENTALLY-CALIBRATED population (Britton et al. 2013 method): "
            f"{n}/{cal_result.n_candidates} drug-free-plausible myocytes accepted "
            f"({cal_result.acceptance_rate:.0%}). Acceptance ranges are KERNEL-plausibility "
            f"bounds, NOT a fit to patient electrophysiology — calibration buys physiological "
            f"plausibility, not prediction; the assessment is still Tier D."]
    else:
        warnings = [f"population CVs are illustrative (uncalibrated): {cv}"]
    if scale:
        warnings.append(
            f"DISEASE/GENETIC background — illustrative mean conductance shift {scale} "
            f"(heterozygous-scale, NOT genotype-calibrated). qNet/APD thresholds are the "
            f"HEALTHY reference. Mechanism demonstration, never a per-patient/genotype claim.")
    if excluded:
        warnings.append(f"unidentifiable channels excluded: {excluded}")

    return PopulationAssessment(
        drug=drug_l, ap_model=ap_rec.id, population=pop.id,
        reference_exposure_nM=reference_exposure, metric=metric, n_models=n,
        qnet_distribution=qn, dapd90_distribution=dapd,
        classification_distribution=counts, susceptible_fraction=counts.get("high", 0.0),
        susceptible_fraction_ci=flip_ci(counts.get("high", 0.0), n),
        repolarization_failures=failures, tier="D", warnings=warnings,
        excluded_channels=excluded, conductance_scale=dict(scale),
        calibrated=cal_result is not None,
        acceptance_rate=cal_result.acceptance_rate if cal_result else float("nan"),
        n_candidates=cal_result.n_candidates if cal_result else 0,
        rejection_reasons=dict(cal_result.rejection_reasons) if cal_result else {},
    )

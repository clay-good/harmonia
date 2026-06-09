"""Action-potential simulation and the **risk-metric distribution** — never a
bare safety verdict (spec.md §6, §10).

The headline feature: pull the *spread* of published IC50s per channel (across
labs / platforms / temperatures), propagate that spread by Monte-Carlo through
the chosen AP model, and report (1) the distribution of the proarrhythmia metric
and (2) how often the high / intermediate / low classification *flips* depending
on which sources you believe. Channels whose IC50 is unidentifiable (max block
< 60%) are flagged and excluded, never silently point-estimated.

Risk metric (Phase C). The DEFAULT classification metric is **qNet** — the CiPA
net-charge biomarker (integral of INaL + ICaL + IKr + IKs + IK1 + Ito over the
beat), where LOWER qNet = HIGHER TdP risk. Once the reduced kernel gained a
shape-dependent Na-Ca exchanger excluded from that sum (see reference.py), qNet
stopped being charge-conserved and became genuinely discriminating. **ΔAPD90% at a
reference exposure** (default 4x free Cmax, EFTPC) remains available via
``metric="apd90"`` as the classic QT surrogate.

Both metrics' thresholds were calibrated on the 12 CiPA training drugs under the
default AP model (cipaordv1.0): qNet recovers 10/12 training labels (vs 8/12 for
APD90 under this kernel), missing cisapride and sotalol. On the 16-compound CiPA
*validation* set the reduced kernel is honestly weaker (many validation drugs have
very low free Cmax, so block at 4x EFTPC is sub-IC50). ``harmonia performance``
reports the live confusion matrix for either metric. It is a *methodology
demonstrator*, not a qualified classifier; the durable contribution is the
flip-frequency-under-variability machinery, correct regardless of absolute
accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np

from .load import Dataset
from .records import APModel, ChannelBlock
from .exposure import resolve_free_exposure
from .export.reference import (BLOCKABLE, KernelParams, simulate_beats,
                               hill_block_factor, HERGDynamic)

# --- classifier calibration (reduced kernel; see module docstring) ---------- #
REFERENCE_EXPOSURE_MULTIPLE = 4.0     # x EFTPC (free Cmax)
DEFAULT_METRIC = "qnet"               # qNet is the CiPA-canonical metric (Phase C)

# Thresholds calibrated on the 12 CiPA training drugs under the DEFAULT AP model
# (cipaordv1.0). They are model-specific: the flip view applies the same rule to
# the ord / tor_ord variants to expose how the structural choice moves the call.
#
# qNet (default): LOWER qNet = HIGHER TdP risk (the CiPA convention). With the
# Na-Ca exchanger excluded from the six-current sum, qNet now discriminates;
# the reduced-kernel qNet classifier recovers 10/12 training labels.
QNET_THRESH_HIGH = 0.220              # qNet below -> "high" risk
QNET_THRESH_LOW = 0.285               # qNet above -> "low" risk
# APD90 (secondary): higher dAPD90% = higher risk.
THRESH_LOW_PCT = 39.0                 # dAPD90% below -> "low"
THRESH_HIGH_PCT = 51.0                # dAPD90% at/above -> "high"

DEFAULT_SINGLE_SOURCE_SIGMA = 0.25    # log10 SD assumed for a single-source IC50
RISK_LABELS = ("low", "intermediate", "high")

# Order of model tiers (worse = larger index) for worst-wins propagation.
_TIER_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}


def classify_apd90(dapd90_pct: float) -> str:
    if np.isnan(dapd90_pct):
        return "intermediate"
    if dapd90_pct >= THRESH_HIGH_PCT:
        return "high"
    if dapd90_pct < THRESH_LOW_PCT:
        return "low"
    return "intermediate"


def classify_qnet(qnet: float) -> str:
    if np.isnan(qnet):
        return "intermediate"
    if qnet < QNET_THRESH_HIGH:
        return "high"
    if qnet > QNET_THRESH_LOW:
        return "low"
    return "intermediate"


def classify_metric(metric: str, dapd90_pct: float, qnet: float) -> str:
    return classify_qnet(qnet) if metric == "qnet" else classify_apd90(dapd90_pct)


# Backwards-compatible alias (APD90 classifier).
def classify(dapd90_pct: float) -> str:
    return classify_apd90(dapd90_pct)


@lru_cache(maxsize=128)
def _cached_baseline_apd90(scales_items: tuple, n_beats: int) -> float:
    """Drug-free baseline APD90 for an AP-model variant. Deterministic and
    independent of the drug, so it is memoised across the many assess() calls a
    performance sweep or flip view makes."""
    scales = dict(scales_items)
    return simulate_beats(KernelParams().with_scales(scales), n_beats=n_beats).apd90


def _worst_tier(tiers: Sequence[str]) -> str:
    worst = "A"
    for t in tiers:
        if _TIER_ORDER.get(t, 0) > _TIER_ORDER.get(worst, 0):
            worst = t
    return worst


def _resolve_ap_model(ds: Dataset, ap_model: str) -> APModel:
    if ap_model in ds:
        rec = ds[ap_model]
    elif f"ap_model.{ap_model}" in ds:
        rec = ds[f"ap_model.{ap_model}"]
    else:
        raise KeyError(f"unknown AP model '{ap_model}'. Known: "
                       f"{[m.id for m in ds.ap_models]}")
    if not isinstance(rec, APModel):
        raise TypeError(f"{rec.id} is not an ap_model record")
    return rec


def kernel_for_model(ds: Dataset, ap_model: str) -> KernelParams:
    """Build the kernel conductances for an AP-model record id (or short name)."""
    rec = _resolve_ap_model(ds, ap_model)
    return KernelParams().with_scales(rec.conductance_scales)


@dataclass
class ChannelDraw:
    """How a single channel's IC50 is sampled in the Monte-Carlo."""
    channel: str
    mu_log10: float
    sigma_log10: float
    hill: float
    n_sources: int
    single_source: bool


def _channel_draws(blocks: List[ChannelBlock]) -> List[ChannelDraw]:
    draws = []
    for b in blocks:
        if not b.identifiable:
            continue
        ic50s = np.array(b.source_ic50s_nm, dtype=float)
        mu = float(np.log10(np.exp(np.mean(np.log(ic50s)))))  # log10 of geomean
        if ic50s.size >= 2:
            sigma = max(float(np.std(np.log10(ic50s), ddof=1)), 0.03)
            single = False
        else:
            sigma = DEFAULT_SINGLE_SOURCE_SIGMA
            single = True
        draws.append(ChannelDraw(b.channel, mu, sigma, b.hill, ic50s.size, single))
    return draws


def _block_factors(ic50_by_channel: Dict[str, float], hill_by_channel: Dict[str, float],
                   conc_nm: float) -> Dict[str, float]:
    bf = {c: 1.0 for c in BLOCKABLE}
    for ch, ic50 in ic50_by_channel.items():
        if ch in bf:
            bf[ch] = hill_block_factor(conc_nm, ic50, hill_by_channel.get(ch, 1.0))
    return bf


@dataclass
class RiskAssessment:
    drug: str
    ap_model: str
    reference_exposure_nM: float
    metric: str

    # point estimate (geomean IC50 per channel)
    apd90: float
    baseline_apd90: float
    dapd90_pct: float
    qnet: float
    ead: bool
    classification: str

    # variability propagation
    n_mc: int
    dapd90_distribution: np.ndarray
    apd90_distribution: np.ndarray
    qnet_distribution: np.ndarray
    classification_distribution: Dict[str, float]
    classification_flip_frequency: float

    # provenance / honesty
    tier: str
    warnings: List[str] = field(default_factory=list)
    excluded_channels: List[str] = field(default_factory=list)
    channels_used: List[str] = field(default_factory=list)
    herg_dynamic: bool = False

    clinical_use: str = ("PROHIBITED — research / safety-methodology / education only; "
                         "not a regulatory determination")

    def summary(self) -> str:
        dist = ", ".join(f"{k} {self.classification_distribution.get(k, 0):.0%}"
                         for k in RISK_LABELS)
        lines = [
            f"drug={self.drug}  ap_model={self.ap_model}  tier={self.tier}",
            f"reference exposure = {self.reference_exposure_nM:g} nM "
            f"({REFERENCE_EXPOSURE_MULTIPLE:g}x EFTPC)",
            f"point: APD90 {self.apd90:.0f} ms (dAPD90 {self.dapd90_pct:+.1f}%), "
            f"qNet {self.qnet:.4f}  -> [{self.metric}] point class: {self.classification.upper()}",
            f"variability ({self.n_mc} MC draws): {dist}",
            f"classification-flip frequency: {self.classification_flip_frequency:.0%}",
        ]
        if self.excluded_channels:
            lines.append(f"EXCLUDED (unidentifiable IC50, max block < 60%): "
                         f"{', '.join(self.excluded_channels)}")
        for w in self.warnings:
            lines.append(f"  warn: {w}")
        lines.append("NOTE: a risk distribution, not a safety verdict. " + self.clinical_use)
        return "\n".join(lines)


def assess(ds: Dataset, drug: str, ap_model: str = "cipaordv1.0",
           exposure_nM: Optional[float] = None, exposure_kind: str = "free",
           exposure_multiple: float = REFERENCE_EXPOSURE_MULTIPLE,
           metric: str = DEFAULT_METRIC, n_mc: int = 200, seed: int = 0,
           n_beats: int = 3, herg_dynamic: bool = False,
           n_beats_dynamic: int = 10) -> RiskAssessment:
    """Assess a drug's proarrhythmia-metric *distribution* under input variability.

    Returns a :class:`RiskAssessment` — a distribution and a classification-flip
    frequency, with the propagated tier and unidentifiable-channel flags. It is
    not, and must not be presented as, a safety determination.

    If ``herg_dynamic`` is True and the drug's hERG record carries
    ``dynamic_binding`` kinetics, hERG block is simulated with the time-dependent
    (CiPA-style) binding ODE instead of a static Hill factor — capturing
    use-dependent trapping. The binding equilibrates slowly, so the dynamic path
    paces ``n_beats_dynamic`` beats.
    """
    drug_l = drug.lower()
    blocks = [b for b in ds.blocks_for(drug_l) if isinstance(b, ChannelBlock)]
    if not blocks:
        raise KeyError(f"no channel-block records for drug '{drug}'")

    # dynamic-hERG kinetics, if requested and available
    herg_rec = next((b for b in blocks if b.channel == "IKr" and b.dynamic_binding), None)
    use_dynamic = herg_dynamic and herg_rec is not None
    if use_dynamic:
        n_beats = n_beats_dynamic

    ref = ds.drug_reference(drug_l)
    if exposure_nM is None and ref is None:
        raise ValueError(f"no EFTPC for '{drug}'; pass exposure_nM explicitly")
    reference_exposure = resolve_free_exposure(
        ref, exposure_nM=exposure_nM, exposure_kind=exposure_kind,
        exposure_multiple=exposure_multiple)

    ap_rec = _resolve_ap_model(ds, ap_model)
    scales = ap_rec.conductance_scales

    # drug-free baseline for this model (memoised; deterministic & drug-free)
    baseline = _cached_baseline_apd90(tuple(sorted(scales.items())), n_beats)

    if metric not in ("qnet", "apd90"):
        raise ValueError(f"metric must be 'qnet' or 'apd90', got {metric!r}")

    # ---- honesty: tier propagation + flags (over ALL channel records) ------- #
    excluded, warnings = [], []
    tiers = [ap_rec.tier]
    for b in blocks:
        tiers.append(b.tier)
        if not b.identifiable:
            mb = b.assay_context.max_block_observed_percent
            excluded.append(f"{b.channel} (max block {mb:g}%)")
            warnings.append(
                f"{b.channel}: IC50 unidentifiable (max block "
                f"{mb:g}% < 60%) — excluded from simulation; caps assessment at Tier D")
        elif b.variability.fold_range and b.variability.fold_range > 5:
            warnings.append(f"{b.channel}: inter-source IC50 fold-range "
                            f"{b.variability.fold_range:g} — classification may flip with source")
    tier = _worst_tier(tiers)

    draws = _channel_draws(blocks)
    channels_used = [d.channel for d in draws]
    singles = [d.channel for d in draws if d.single_source]
    if singles:
        warnings.append(
            f"single-source channels {singles} sampled with a default log10 SD of "
            f"{DEFAULT_SINGLE_SOURCE_SIGMA} (assumed inter-lab prior; flagged, not measured)")

    hill_by = {d.channel: d.hill for d in draws}

    def _herg_for(ic50_nm: float):
        """Build a HERGDynamic at the reference exposure with kon set so the
        binding IC50 (koff/kon) tracks the sampled hERG IC50."""
        if not use_dynamic:
            return None
        koff = herg_rec.dynamic_binding["koff"]
        trapping = bool(herg_rec.dynamic_binding["trapping"])
        return HERGDynamic(conc_nm=reference_exposure, kon=koff / ic50_nm,
                           koff=koff, trapping=trapping)

    # ---- point estimate (geomean IC50 per channel) -------------------------- #
    ic50_point = {d.channel: 10 ** d.mu_log10 for d in draws}
    bf = _block_factors(ic50_point, hill_by, reference_exposure)
    p = KernelParams().with_scales(scales)
    p.block.update(bf)
    pt = simulate_beats(p, n_beats=n_beats, herg=_herg_for(ic50_point.get("IKr", 1e9)))
    dapd_point = 100.0 * (pt.apd90 - baseline) / baseline if baseline else float("nan")
    point_class = classify_metric(metric, dapd_point, pt.qnet)

    # ---- Monte-Carlo over source variability -------------------------------- #
    rng = np.random.default_rng(seed)
    dapd = np.empty(n_mc)
    apd = np.empty(n_mc)
    qn = np.empty(n_mc)
    classes: List[str] = []
    for i in range(n_mc):
        ic50_s = {d.channel: 10 ** rng.normal(d.mu_log10, d.sigma_log10) for d in draws}
        bf = _block_factors(ic50_s, hill_by, reference_exposure)
        pp = KernelParams().with_scales(scales)
        pp.block.update(bf)
        r = simulate_beats(pp, n_beats=n_beats, herg=_herg_for(ic50_s.get("IKr", 1e9)))
        apd[i] = r.apd90
        dapd[i] = 100.0 * (r.apd90 - baseline) / baseline if baseline else float("nan")
        qn[i] = r.qnet
        classes.append(classify_metric(metric, dapd[i], qn[i]))

    # n_mc == 0 means "point estimate only" (used by performance scoring); the
    # distribution collapses to the point classification.
    if n_mc:
        counts = {lab: classes.count(lab) / n_mc for lab in RISK_LABELS}
        flip_freq = float(np.mean([c != point_class for c in classes]))
    else:
        counts = {lab: (1.0 if lab == point_class else 0.0) for lab in RISK_LABELS}
        flip_freq = 0.0

    return RiskAssessment(
        drug=drug_l, ap_model=ap_rec.id, reference_exposure_nM=reference_exposure,
        metric=metric, apd90=pt.apd90, baseline_apd90=baseline, dapd90_pct=dapd_point,
        qnet=pt.qnet, ead=pt.ead, classification=point_class, n_mc=n_mc,
        dapd90_distribution=dapd, apd90_distribution=apd, qnet_distribution=qn,
        classification_distribution=counts, classification_flip_frequency=flip_freq,
        tier=tier, warnings=warnings, excluded_channels=excluded,
        channels_used=channels_used, herg_dynamic=use_dynamic,
    )


@dataclass
class CombinationAssessment:
    """Proarrhythmia assessment of a DRUG COMBINATION (polypharmacy). Same honest
    posture as a single-drug assessment: a metric distribution and a
    classification-flip frequency, never a verdict."""
    drugs: List[str]
    ap_model: str
    exposures_nM: Dict[str, float]
    metric: str
    qnet: float
    apd90: float
    baseline_apd90: float
    dapd90_pct: float
    classification: str
    n_mc: int
    qnet_distribution: np.ndarray
    dapd90_distribution: np.ndarray
    classification_distribution: Dict[str, float]
    classification_flip_frequency: float
    tier: str
    # the extra prolongation of the combination beyond the worst single agent
    interaction_dapd90_pct: float = float("nan")
    warnings: List[str] = field(default_factory=list)
    excluded_channels: List[str] = field(default_factory=list)
    clinical_use: str = ("PROHIBITED — research / safety-methodology / education only; "
                         "not a regulatory determination")

    def summary(self) -> str:
        dist = ", ".join(f"{k} {self.classification_distribution.get(k, 0):.0%}"
                         for k in RISK_LABELS)
        expo = ", ".join(f"{d}={self.exposures_nM[d]:g}nM" for d in self.drugs)
        lines = [
            f"combination = {' + '.join(self.drugs)}   ap_model={self.ap_model}  tier={self.tier}",
            f"free exposures: {expo}",
            f"point: APD90 {self.apd90:.0f} ms (dAPD90 {self.dapd90_pct:+.1f}%), "
            f"qNet {self.qnet:.4f}  -> [{self.metric}] point class: {self.classification.upper()}",
            f"interaction (extra dAPD90 beyond the worst single agent): "
            f"{self.interaction_dapd90_pct:+.1f}%",
            f"variability ({self.n_mc} MC draws): {dist}",
            f"classification-flip frequency: {self.classification_flip_frequency:.0%}",
        ]
        if self.excluded_channels:
            lines.append(f"EXCLUDED (unidentifiable IC50): {', '.join(self.excluded_channels)}")
        for w in self.warnings:
            lines.append(f"  warn: {w}")
        lines.append("NOTE: a risk distribution, not a safety verdict. " + self.clinical_use)
        return "\n".join(lines)


def assess_combination(ds: Dataset, drugs: Sequence[str], ap_model: str = "cipaordv1.0",
                       exposures_nM: Optional[Sequence[float]] = None,
                       exposure_multiple: float = REFERENCE_EXPOSURE_MULTIPLE,
                       exposure_kind: str = "free", metric: str = DEFAULT_METRIC,
                       n_mc: int = 200, seed: int = 0,
                       n_beats: int = 3) -> CombinationAssessment:
    """Assess a COMBINATION of drugs given together (polypharmacy proarrhythmia).

    Block from independent agents combines multiplicatively per channel: the
    fraction of a current REMAINING is the product of each drug's remaining
    fraction (the standard non-interacting / additive-occupancy assumption). IC50
    variability is propagated for every drug independently by Monte-Carlo, so the
    combination's classification-flip frequency reflects the *joint* input
    uncertainty. This is the polypharmacy analog of Harmonia's single-drug thesis:
    a combined safety call is only as trustworthy as its least-identifiable input.
    """
    if metric not in ("qnet", "apd90"):
        raise ValueError(f"metric must be 'qnet' or 'apd90', got {metric!r}")
    drugs = [d.lower() for d in drugs]
    if len(drugs) < 2:
        raise ValueError("assess_combination needs at least two drugs")
    if exposures_nM is not None and len(exposures_nM) != len(drugs):
        raise ValueError("exposures_nM must align with drugs")

    ap_rec = _resolve_ap_model(ds, ap_model)
    scales = ap_rec.conductance_scales
    baseline = _cached_baseline_apd90(tuple(sorted(scales.items())), n_beats)

    tiers = [ap_rec.tier]
    excluded, warnings = [], []
    per_drug = []   # (drug, draws, exposure, hill_by)
    exposures_map: Dict[str, float] = {}
    for i, drug in enumerate(drugs):
        blocks = [b for b in ds.blocks_for(drug) if isinstance(b, ChannelBlock)]
        if not blocks:
            raise KeyError(f"no channel-block records for drug '{drug}'")
        ref = ds.drug_reference(drug)
        exp_nm = exposures_nM[i] if exposures_nM is not None else None
        exposure = resolve_free_exposure(ref, exposure_nM=exp_nm,
                                         exposure_kind=exposure_kind,
                                         exposure_multiple=exposure_multiple)
        exposures_map[drug] = exposure
        tiers.append(_worst_tier([b.tier for b in blocks]))
        for b in blocks:
            if not b.identifiable:
                excluded.append(f"{drug}:{b.channel}")
                warnings.append(f"{drug} {b.channel}: IC50 unidentifiable — excluded; caps at Tier D")
        draws = _channel_draws(blocks)
        per_drug.append((drug, draws, exposure, {d.channel: d.hill for d in draws}))
    tier = _worst_tier(tiers)

    def combined_block(rng=None):
        bf = {c: 1.0 for c in BLOCKABLE}
        for drug, draws, exposure, hill_by in per_drug:
            for d in draws:
                ic50 = 10 ** (rng.normal(d.mu_log10, d.sigma_log10) if rng is not None
                              else d.mu_log10)
                if d.channel in bf:
                    bf[d.channel] *= hill_block_factor(exposure, ic50, hill_by[d.channel])
        return bf

    def run(bf):
        p = KernelParams().with_scales(scales)
        p.block.update(bf)
        return simulate_beats(p, n_beats=n_beats)

    # point estimate (geomean IC50 per drug/channel)
    pt = run(combined_block(None))
    dapd_point = 100.0 * (pt.apd90 - baseline) / baseline if baseline else float("nan")
    point_class = classify_metric(metric, dapd_point, pt.qnet)

    # interaction: how much the combination prolongs beyond the worst single agent
    worst_single = 0.0
    for drug in drugs:
        a = assess(ds, drug, ap_model=ap_model, n_mc=0, metric=metric, n_beats=n_beats,
                   exposure_nM=exposures_map[drug], exposure_kind="free")
        worst_single = max(worst_single, a.dapd90_pct)
    interaction = dapd_point - worst_single

    # Monte-Carlo over every drug's IC50 variability jointly
    rng = np.random.default_rng(seed)
    dapd = np.empty(n_mc)
    qn = np.empty(n_mc)
    classes: List[str] = []
    for k in range(n_mc):
        r = run(combined_block(rng))
        dapd[k] = 100.0 * (r.apd90 - baseline) / baseline if baseline else float("nan")
        qn[k] = r.qnet
        classes.append(classify_metric(metric, dapd[k], qn[k]))
    if n_mc:
        counts = {lab: classes.count(lab) / n_mc for lab in RISK_LABELS}
        flip = float(np.mean([c != point_class for c in classes]))
    else:
        counts = {lab: (1.0 if lab == point_class else 0.0) for lab in RISK_LABELS}
        flip = 0.0

    return CombinationAssessment(
        drugs=drugs, ap_model=ap_rec.id, exposures_nM=exposures_map, metric=metric,
        qnet=pt.qnet, apd90=pt.apd90, baseline_apd90=baseline, dapd90_pct=dapd_point,
        classification=point_class, n_mc=n_mc, qnet_distribution=qn,
        dapd90_distribution=dapd, classification_distribution=counts,
        classification_flip_frequency=flip, tier=tier,
        interaction_dapd90_pct=interaction, warnings=warnings, excluded_channels=excluded,
    )


@dataclass
class FlipView:
    drug: str
    ap_models: List[str]
    per_model: Dict[str, RiskAssessment]
    excluded: List[str]

    @property
    def flip_by_model(self) -> Dict[str, str]:
        """Point classification under each AP-model variant."""
        return {m: a.classification for m, a in self.per_model.items()}

    @property
    def stable_across_models(self) -> bool:
        return len(set(self.flip_by_model.values())) == 1

    @property
    def flip_frequency_by_model(self) -> Dict[str, float]:
        return {m: a.classification_flip_frequency for m, a in self.per_model.items()}

    def summary(self) -> str:
        lines = [f"flip-view  drug={self.drug}"]
        for m, a in self.per_model.items():
            dist = ", ".join(f"{k} {a.classification_distribution.get(k, 0):.0%}"
                             for k in RISK_LABELS)
            lines.append(f"  {m:22s} point={a.classification.upper():12s} "
                         f"flip={a.classification_flip_frequency:.0%}  [{dist}]")
        lines.append(f"  stable across model variants: {self.stable_across_models}")
        if self.excluded:
            lines.append(f"  excluded channels: {', '.join(self.excluded)}")
        return "\n".join(lines)


def flip_view(ds: Dataset, drug: str,
              ap_models: Sequence[str] = ("ord", "cipaordv1.0", "tor_ord"),
              n_mc: int = 200, seed: int = 0, **kw) -> FlipView:
    """The headline comparison: classification stability across AP-model variants
    *and* across input variability, for one drug."""
    per_model: Dict[str, RiskAssessment] = {}
    excluded: List[str] = []
    for m in ap_models:
        a = assess(ds, drug, ap_model=m, n_mc=n_mc, seed=seed, **kw)
        per_model[m] = a
        excluded = a.excluded_channels  # same drug -> same exclusions
    return FlipView(drug=drug.lower(), ap_models=list(ap_models),
                    per_model=per_model, excluded=excluded)


@dataclass
class ChannelSensitivity:
    """How much one channel's IC50 *uncertainty* drives the classification flip."""
    channel: str
    n_sources: int
    single_source: bool
    fold_range: Optional[float]
    sigma_log10: float
    solo_flip_frequency: float    # only THIS channel's IC50 varies (others at geomean)
    frozen_flip_frequency: float  # this channel pinned at geomean (all others vary)


@dataclass
class FlipSensitivity:
    """Attribution of the classification-flip frequency to individual channels.

    The flip view says *whether* the high/intermediate/low call is unstable under
    input variability; this says *which* channel's IC50 spread drives it — i.e.
    which lab measurement, if pinned down, would most stabilise the call. It is an
    analysis of the data's uncertainty, never a safety verdict.

    - ``solo_flip_frequency`` (main effect): how often the call flips when ONLY
      this channel's IC50 varies and every other channel is held at its geomean.
      The largest is the dominant driver.
    - ``frozen_flip_frequency`` (total effect): how often it flips when this
      channel is pinned at its geomean while all others vary. Much lower than
      ``all_vary_flip_frequency`` ⇒ pinning this channel stabilises the call.
    """
    drug: str
    ap_model: str
    metric: str
    classification: str
    all_vary_flip_frequency: float
    n_mc: int
    channels: List[ChannelSensitivity]   # sorted by solo_flip_frequency, descending
    tier: str
    warnings: List[str] = field(default_factory=list)
    excluded_channels: List[str] = field(default_factory=list)
    clinical_use: str = ("PROHIBITED — research / safety-methodology / education only; "
                         "not a regulatory determination")

    @property
    def dominant_channel(self) -> Optional[str]:
        return self.channels[0].channel if self.channels else None

    def summary(self) -> str:
        lines = [
            f"flip-sensitivity  drug={self.drug}  ap_model={self.ap_model}  "
            f"tier={self.tier}",
            f"point class: {self.classification.upper()}   "
            f"all-vary flip frequency: {self.all_vary_flip_frequency:.0%} "
            f"({self.n_mc} MC draws)",
            "  channel   sources  fold   solo-flip  frozen-flip",
        ]
        for c in self.channels:
            fr = f"{c.fold_range:.1f}" if c.fold_range else "  - "
            lines.append(f"  {c.channel:<8} {c.n_sources:>5}{'*' if c.single_source else ' '}  "
                         f"{fr:>5}  {c.solo_flip_frequency:>8.0%}  {c.frozen_flip_frequency:>10.0%}")
        if self.dominant_channel:
            lines.append(f"dominant uncertainty driver: {self.dominant_channel} "
                         "(largest solo-flip) — pin this IC50 down first")
        if self.excluded_channels:
            lines.append(f"EXCLUDED (unidentifiable IC50): {', '.join(self.excluded_channels)}")
        for w in self.warnings:
            lines.append(f"  warn: {w}")
        lines.append("NOTE: an uncertainty attribution, not a safety verdict. "
                     + self.clinical_use)
        return "\n".join(lines)


def flip_sensitivity(ds: Dataset, drug: str, ap_model: str = "cipaordv1.0",
                     metric: str = DEFAULT_METRIC, n_mc: int = 200, seed: int = 0,
                     exposure_nM: Optional[float] = None, exposure_kind: str = "free",
                     exposure_multiple: float = REFERENCE_EXPOSURE_MULTIPLE,
                     n_beats: int = 3) -> FlipSensitivity:
    """Attribute the classification-flip frequency to each channel's IC50 spread.

    For every identifiable channel it runs two Monte-Carlo scenarios — "only this
    channel varies" (main effect) and "this channel frozen, others vary" (total
    effect) — using common random numbers across scenarios so the comparison is
    apples-to-apples. See :class:`FlipSensitivity`.
    """
    drug_l = drug.lower()
    blocks = [b for b in ds.blocks_for(drug_l) if isinstance(b, ChannelBlock)]
    if not blocks:
        raise KeyError(f"no channel-block records for drug '{drug}'")
    if metric not in ("qnet", "apd90"):
        raise ValueError(f"metric must be 'qnet' or 'apd90', got {metric!r}")

    ref = ds.drug_reference(drug_l)
    reference_exposure = resolve_free_exposure(
        ref, exposure_nM=exposure_nM, exposure_kind=exposure_kind,
        exposure_multiple=exposure_multiple)
    ap_rec = _resolve_ap_model(ds, ap_model)
    scales = ap_rec.conductance_scales
    baseline = _cached_baseline_apd90(tuple(sorted(scales.items())), n_beats)

    excluded, warnings, tiers = [], [], [ap_rec.tier]
    for b in blocks:
        tiers.append(b.tier)
        if not b.identifiable:
            mb = b.assay_context.max_block_observed_percent
            excluded.append(f"{b.channel} (max block {mb:g}%)")
    tier = _worst_tier(tiers)

    draws = _channel_draws(blocks)
    hill_by = {d.channel: d.hill for d in draws}
    fold_by = {b.channel: b.variability.fold_range for b in blocks}
    ic50_geomean = {d.channel: 10 ** d.mu_log10 for d in draws}
    all_channels = {d.channel for d in draws}

    def _classify(ic50_by: Dict[str, float]) -> str:
        bf = _block_factors(ic50_by, hill_by, reference_exposure)
        p = KernelParams().with_scales(scales)
        p.block.update(bf)
        r = simulate_beats(p, n_beats=n_beats)
        dapd = 100.0 * (r.apd90 - baseline) / baseline if baseline else float("nan")
        return classify_metric(metric, dapd, r.qnet)

    point_class = _classify(ic50_geomean)

    def _flip_freq(vary: set) -> float:
        if not n_mc:
            return 0.0
        rng = np.random.default_rng(seed)
        flips = 0
        for _ in range(n_mc):
            # draw for ALL channels every iteration (common random numbers), then
            # keep the draw only for varying channels; others stay at geomean.
            sampled = {d.channel: 10 ** rng.normal(d.mu_log10, d.sigma_log10) for d in draws}
            ic50_s = {ch: (sampled[ch] if ch in vary else ic50_geomean[ch])
                      for ch in all_channels}
            if _classify(ic50_s) != point_class:
                flips += 1
        return flips / n_mc

    all_vary = _flip_freq(all_channels)
    chans: List[ChannelSensitivity] = []
    for d in draws:
        chans.append(ChannelSensitivity(
            channel=d.channel, n_sources=d.n_sources, single_source=d.single_source,
            fold_range=fold_by.get(d.channel), sigma_log10=d.sigma_log10,
            solo_flip_frequency=_flip_freq({d.channel}),
            frozen_flip_frequency=_flip_freq(all_channels - {d.channel})))
    chans.sort(key=lambda c: c.solo_flip_frequency, reverse=True)

    if any(c.single_source for c in chans):
        warnings.append("single-source channels use a default log10 SD prior "
                        f"({DEFAULT_SINGLE_SOURCE_SIGMA}); their sensitivity is prior-driven, "
                        "not measured")
    return FlipSensitivity(
        drug=drug_l, ap_model=ap_rec.id, metric=metric, classification=point_class,
        all_vary_flip_frequency=all_vary, n_mc=n_mc, channels=chans, tier=tier,
        warnings=warnings, excluded_channels=excluded)


def dose_response(ds: Dataset, drug: str, concentrations_nM: Sequence[float],
                  ap_model: str = "cipaordv1.0", n_beats: int = 3) -> Dict[str, np.ndarray]:
    """APD90 / qNet vs concentration, using geomean IC50 per identifiable channel.
    For the dashboard's dose-response curve."""
    blocks = [b for b in ds.blocks_for(drug) if isinstance(b, ChannelBlock)]
    ap_rec = _resolve_ap_model(ds, ap_model)
    scales = ap_rec.conductance_scales
    draws = _channel_draws(blocks)
    ic50 = {d.channel: 10 ** d.mu_log10 for d in draws}
    hill_by = {d.channel: d.hill for d in draws}
    concs = np.asarray(concentrations_nM, dtype=float)
    apd = np.empty(concs.size)
    qn = np.empty(concs.size)
    for i, c in enumerate(concs):
        bf = _block_factors(ic50, hill_by, c)
        p = KernelParams().with_scales(scales)
        p.block.update(bf)
        r = simulate_beats(p, n_beats=n_beats)
        apd[i] = r.apd90
        qn[i] = r.qnet
    return {"concentration_nM": concs, "apd90": apd, "qnet": qn}

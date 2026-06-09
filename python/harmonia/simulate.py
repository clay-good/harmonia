"""Action-potential simulation and the **risk-metric distribution** — never a
bare safety verdict (spec.md §6, §10).

The headline feature: pull the *spread* of published IC50s per channel (across
labs / platforms / temperatures), propagate that spread by Monte-Carlo through
the chosen AP model, and report (1) the distribution of the proarrhythmia metric
and (2) how often the high / intermediate / low classification *flips* depending
on which sources you believe. Channels whose IC50 is unidentifiable (max block
< 60%) are flagged and excluded, never silently point-estimated.

Risk metric, v0.1. The classification metric is **ΔAPD90% at a reference
exposure** (default 4x the free therapeutic Cmax, EFTPC). APD/QT prolongation is
the established proarrhythmia surrogate; qNet — the CiPA metric the reduced,
pump-free kernel cannot make sensitive (see reference.py) — is reported for
transparency but is NOT used to classify in v0.1. The two thresholds below were
calibrated on the 12 CiPA training drugs; the resulting reduced-kernel classifier
recovers 10/12 training labels. The two misses are sotalol (very weak hERG block
but high exposure — its risk only emerges well above 4x EFTPC) and chlorpromazine
(borderline) — the kind of cases that motivated CiPA to replace APD with qNet. It
is a *methodology demonstrator*, not a qualified classifier. The durable
contribution is the flip-frequency-under-variability machinery, which is correct
regardless of the absolute classifier accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .load import Dataset
from .records import APModel, ChannelBlock
from .export.reference import (BLOCKABLE, KernelParams, simulate_beats,
                               hill_block_factor)

# --- classifier calibration (reduced kernel; see module docstring) ---------- #
REFERENCE_EXPOSURE_MULTIPLE = 4.0     # x EFTPC (free Cmax)
THRESH_LOW_PCT = 16.0                 # dAPD90% below -> "low"
THRESH_HIGH_PCT = 33.0                # dAPD90% at/above -> "high"
DEFAULT_SINGLE_SOURCE_SIGMA = 0.25    # log10 SD assumed for a single-source IC50
RISK_LABELS = ("low", "intermediate", "high")

# Order of model tiers (worse = larger index) for worst-wins propagation.
_TIER_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}


def classify(dapd90_pct: float) -> str:
    if np.isnan(dapd90_pct):
        return "intermediate"
    if dapd90_pct >= THRESH_HIGH_PCT:
        return "high"
    if dapd90_pct < THRESH_LOW_PCT:
        return "low"
    return "intermediate"


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

    clinical_use: str = ("PROHIBITED — research / safety-methodology / education only; "
                         "not a regulatory determination")

    def summary(self) -> str:
        dist = ", ".join(f"{k} {self.classification_distribution.get(k, 0):.0%}"
                         for k in RISK_LABELS)
        lines = [
            f"drug={self.drug}  ap_model={self.ap_model}  tier={self.tier}",
            f"reference exposure = {self.reference_exposure_nM:g} nM "
            f"({REFERENCE_EXPOSURE_MULTIPLE:g}x EFTPC)",
            f"point: APD90 {self.apd90:.0f} ms  (dAPD90 {self.dapd90_pct:+.1f}% vs "
            f"drug-free {self.baseline_apd90:.0f} ms)  -> point class: {self.classification.upper()}",
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
           exposure_nM: Optional[float] = None,
           exposure_multiple: float = REFERENCE_EXPOSURE_MULTIPLE,
           metric: str = "apd90", n_mc: int = 200, seed: int = 0,
           n_beats: int = 3) -> RiskAssessment:
    """Assess a drug's proarrhythmia-metric *distribution* under input variability.

    Returns a :class:`RiskAssessment` — a distribution and a classification-flip
    frequency, with the propagated tier and unidentifiable-channel flags. It is
    not, and must not be presented as, a safety determination.
    """
    drug_l = drug.lower()
    blocks = [b for b in ds.blocks_for(drug_l) if isinstance(b, ChannelBlock)]
    if not blocks:
        raise KeyError(f"no channel-block records for drug '{drug}'")

    ref = ds.drug_reference(drug_l)
    if exposure_nM is not None:
        reference_exposure = float(exposure_nM)
    elif ref is not None:
        reference_exposure = ref.eftpc_nm * exposure_multiple
    else:
        raise ValueError(f"no EFTPC for '{drug}'; pass exposure_nM explicitly")

    ap_rec = _resolve_ap_model(ds, ap_model)
    scales = ap_rec.conductance_scales

    # drug-free baseline for this model
    baseline = simulate_beats(KernelParams().with_scales(scales), n_beats=n_beats).apd90

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

    # ---- point estimate (geomean IC50 per channel) -------------------------- #
    ic50_point = {d.channel: 10 ** d.mu_log10 for d in draws}
    bf = _block_factors(ic50_point, hill_by, reference_exposure)
    p = KernelParams().with_scales(scales)
    p.block.update(bf)
    pt = simulate_beats(p, n_beats=n_beats)
    dapd_point = 100.0 * (pt.apd90 - baseline) / baseline if baseline else float("nan")
    point_class = classify(dapd_point)

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
        r = simulate_beats(pp, n_beats=n_beats)
        apd[i] = r.apd90
        dapd[i] = 100.0 * (r.apd90 - baseline) / baseline if baseline else float("nan")
        qn[i] = r.qnet
        classes.append(classify(dapd[i]))

    counts = {lab: classes.count(lab) / n_mc for lab in RISK_LABELS}
    flip_freq = float(np.mean([c != point_class for c in classes])) if n_mc else 0.0

    return RiskAssessment(
        drug=drug_l, ap_model=ap_rec.id, reference_exposure_nM=reference_exposure,
        metric=metric, apd90=pt.apd90, baseline_apd90=baseline, dapd90_pct=dapd_point,
        qnet=pt.qnet, ead=pt.ead, classification=point_class, n_mc=n_mc,
        dapd90_distribution=dapd, apd90_distribution=apd, qnet_distribution=qn,
        classification_distribution=counts, classification_flip_frequency=flip_freq,
        tier=tier, warnings=warnings, excluded_channels=excluded,
        channels_used=channels_used,
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

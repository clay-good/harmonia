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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .load import Dataset
from .records import APModel, ChannelBlock
from .exposure import resolve_free_exposure
from .export.reference import (BLOCKABLE, KernelParams, simulate_beats,
                               hill_block_factor, HERGDynamic, CiPABinding)

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
def _cached_baseline(scales_items: tuple, n_beats: int) -> tuple:
    """Drug-free baseline for an AP-model variant: (APD90 ms, triangulation ms,
    control INaL charge, control ICaL charge). Deterministic and drug-free, so it is
    memoised across the many assess() calls a performance sweep or flip view makes.
    The control inward charges are the denominators of the cqInward biomarker (v0.4);
    callers that only need APD90 still read element [0]."""
    scales = dict(scales_items)
    r = simulate_beats(KernelParams().with_scales(scales), n_beats=n_beats)
    return r.apd90, r.triangulation, r.q_nal, r.q_cal


def _cqinward(q_nal: float, q_cal: float, q_nal_ctrl: float, q_cal_ctrl: float) -> float:
    """The cqInward biomarker (spec v0.4): the control-normalized average of the two
    inward-current (INaL, ICaL) charge ratios. 1 at no drug; <1 = inward charge reduced
    (ICaL/INaL block, protective); >1 = inward charge increased (AP prolongation)."""
    a = q_nal / q_nal_ctrl if q_nal_ctrl else float("nan")
    b = q_cal / q_cal_ctrl if q_cal_ctrl else float("nan")
    return 0.5 * (a + b)


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
    # triangulation (APD90 - APD50, ms): a TRIaD proarrhythmia diagnostic that
    # hERG block widens (spec §3). Reported as a readout, never the classifier.
    triangulation_ms: float
    baseline_triangulation_ms: float

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

    # v0.2 Bayesian uncertainty quantification (spec v0.2). Defaults keep the v0.1
    # moments path unchanged: uq="moments" reports no reproducibility flip and no
    # censored/prior-dominated channels.
    uq: str = "moments"
    reproducibility_flip_frequency: float = float("nan")   # new-lab predictive (sec 2.4)
    censored_channels: List[str] = field(default_factory=list)        # one-sided posteriors (sec 2.3)
    prior_dominated_channels: List[str] = field(default_factory=list)  # sec 6/7 flags

    # v0.4 cqInward: control-normalized inward-charge biomarker (a diagnostic, not the
    # classifier). 1 at no drug; <1 = inward charge reduced (ICaL/INaL block, protective);
    # >1 = inward charge increased (AP prolongation). Propagated like qNet.
    cqinward: float = float("nan")
    cqinward_distribution: np.ndarray = field(default_factory=lambda: np.empty(0))

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
            f"triangulation (APD90-APD50): {self.triangulation_ms:.0f} ms "
            f"(drug-free {self.baseline_triangulation_ms:.0f} ms) — diagnostic, not the classifier",
            f"cqInward (inward-charge vs drug-free): {self.cqinward:.3f} "
            f"(<1 inward charge reduced / protective, >1 increased) — diagnostic, not the classifier",
            f"variability ({self.n_mc} MC draws, uq={self.uq}): {dist}",
            f"classification-flip frequency: {self.classification_flip_frequency:.0%}",
        ]
        if self.uq == "bayes" and not np.isnan(self.reproducibility_flip_frequency):
            lines.append(f"reproducibility-flip frequency (new-lab predictive): "
                         f"{self.reproducibility_flip_frequency:.0%}")
        if self.excluded_channels:
            lines.append(f"EXCLUDED (unidentifiable IC50, max block < 60%): "
                         f"{', '.join(self.excluded_channels)}")
        if self.censored_channels:
            lines.append(f"CENSORED (one-sided posterior, max block < 60%, prior-dominated; "
                         f"still Tier-D-capped): {', '.join(self.censored_channels)}")
        if self.prior_dominated_channels:
            lines.append(f"PRIOR-DOMINATED channels (posterior is prior-shaped, not "
                         f"data-shaped — measure at higher doses): "
                         f"{', '.join(self.prior_dominated_channels)}")
        for w in self.warnings:
            lines.append(f"  warn: {w}")
        lines.append("NOTE: a risk distribution, not a safety verdict. " + self.clinical_use)
        return "\n".join(lines)


def assess(ds: Dataset, drug: str, ap_model: str = "cipaordv1.0",
           exposure_nM: Optional[float] = None, exposure_kind: str = "free",
           exposure_multiple: float = REFERENCE_EXPOSURE_MULTIPLE,
           metric: str = DEFAULT_METRIC, n_mc: int = 200, seed: int = 0,
           n_beats: int = 3, herg_dynamic: object = False,
           n_beats_dynamic: int = 10, uq: str = "moments",
           prior: Optional[object] = None) -> RiskAssessment:
    """Assess a drug's proarrhythmia-metric *distribution* under input variability.

    Returns a :class:`RiskAssessment` — a distribution and a classification-flip
    frequency, with the propagated tier and unidentifiable-channel flags. It is
    not, and must not be presented as, a safety determination.

    ``herg_dynamic`` selects the hERG block model (opt-in; neither dynamic path
    changes any default/recorded number):

    - ``False`` (default) — static Hill block.
    - ``True`` — the v0.1 Langmuir ``dynamic_binding`` (kon/koff/trapping) ODE,
      where present (dofetilide, verapamil), capturing use-dependent trapping.
    - ``"cipa"`` — the published **CiPA Li-2017 binding kinetics** (``cipa_binding``:
      Kmax/Ku/halfmax/n/Vhalf) for the 12 CiPA dynamic-fit compounds (spec v0.6),
      coupled to the reduced IKr gate. The CiPA kinetics equilibrate slowly, so this
      research path benefits from a larger ``n_beats_dynamic``.

    Either dynamic path paces ``n_beats_dynamic`` beats (the binding equilibrates
    slowly).

    ``uq`` selects the uncertainty-propagation engine (spec v0.2):

    - ``"moments"`` (default) — the v0.1 method-of-moments lognormal sampler. Exact
      backward compatibility; every v0.1 number reproduces.
    - ``"bayes"`` — the hierarchical Bayesian posterior-predictive sampler
      (:mod:`harmonia.infer`). The per-channel ``(IC50, Hill)`` posterior is inferred
      under a declared ``prior`` and sampled in place of the lognormal draw, so Hill
      uncertainty propagates, single-source channels borrow the dataset-learned
      between-lab spread instead of a magic constant, and a sub-60%-block channel
      contributes a one-sided **censored** posterior rather than being excluded (it
      stays Tier-D-capped). The headline flip frequency samples the *true-value*
      posterior; a separate ``reproducibility_flip_frequency`` samples the *new-lab
      predictive* (sec 2.4). Neither is a verdict.
    """
    drug_l = drug.lower()
    blocks = [b for b in ds.blocks_for(drug_l) if isinstance(b, ChannelBlock)]
    if not blocks:
        raise KeyError(f"no channel-block records for drug '{drug}'")

    # dynamic-hERG kinetics, if requested and available. ``herg_dynamic`` is False
    # (static Hill, default), True (the v0.1 Langmuir kon/koff path), or "cipa" (the
    # published CiPA Li-2017 binding kinetics, spec v0.6 — opt-in, does not change any
    # default/recorded number).
    cipa_mode = herg_dynamic == "cipa"
    if cipa_mode:
        herg_rec = next((b for b in blocks if b.channel == "IKr" and b.cipa_binding), None)
    else:
        herg_rec = next((b for b in blocks if b.channel == "IKr" and b.dynamic_binding), None)
    use_dynamic = bool(herg_dynamic) and herg_rec is not None
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
    baseline, baseline_tri, q_nal_ctrl, q_cal_ctrl = _cached_baseline(
        tuple(sorted(scales.items())), n_beats)

    if metric not in ("qnet", "apd90"):
        raise ValueError(f"metric must be 'qnet' or 'apd90', got {metric!r}")
    if uq not in ("moments", "bayes"):
        raise ValueError(f"uq must be 'moments' or 'bayes', got {uq!r}")

    # tier propagation is identical under either uq engine: worst-wins over every
    # channel record + the AP model; a censored (sub-60%-block) channel keeps Tier D.
    tier = _worst_tier([ap_rec.tier] + [b.tier for b in blocks])

    def _herg_for(ic50_nm: float):
        """Build the dynamic-hERG binding model at the reference exposure. In the
        legacy Langmuir path the binding IC50 (koff/kon) tracks the sampled hERG
        IC50; in the CiPA path the published Li-2017 kinetics are fixed (the binding
        kinetics carry no multi-source spread), so the sampled IC50 is not used."""
        if not use_dynamic or herg_rec is None:
            return None
        if cipa_mode:
            cb = herg_rec.cipa_binding
            assert cb is not None  # cipa_mode + use_dynamic implies the record carries it
            return CiPABinding(conc_nm=reference_exposure, kmax=cb["Kmax"], ku=cb["Ku"],
                               n=cb["n"], halfmax=cb["halfmax"], vhalf=cb["Vhalf"],
                               kt=cb.get("Kt", 3.5e-05))
        kinetics = herg_rec.dynamic_binding
        assert kinetics is not None  # use_dynamic implies the hERG record carries kinetics
        koff = kinetics["koff"]
        trapping = bool(kinetics["trapping"])
        return HERGDynamic(conc_nm=reference_exposure, kon=koff / ic50_nm,
                           koff=koff, trapping=trapping)

    if uq == "bayes":
        return _assess_bayes(
            ds, drug_l, blocks, ap_rec, scales, baseline, baseline_tri, metric,
            reference_exposure, n_mc, seed, n_beats, _herg_for, use_dynamic, tier,
            prior, q_nal_ctrl, q_cal_ctrl)

    # ====================================================================== #
    # v0.1 method-of-moments path (default) — kept byte-identical for non-drift
    # ====================================================================== #
    excluded, warnings = [], []
    for b in blocks:
        if not b.identifiable:
            mb = b.assay_context.max_block_observed_percent
            excluded.append(f"{b.channel} (max block {mb:g}%)")
            warnings.append(
                f"{b.channel}: IC50 unidentifiable (max block "
                f"{mb:g}% < 60%) — excluded from simulation; caps assessment at Tier D")
        elif b.variability.fold_range and b.variability.fold_range > 5:
            warnings.append(f"{b.channel}: inter-source IC50 fold-range "
                            f"{b.variability.fold_range:g} — classification may flip with source")

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
    pt = simulate_beats(p, n_beats=n_beats, herg=_herg_for(ic50_point.get("IKr", 1e9)))
    dapd_point = 100.0 * (pt.apd90 - baseline) / baseline if baseline else float("nan")
    point_class = classify_metric(metric, dapd_point, pt.qnet)
    cqinward_point = _cqinward(pt.q_nal, pt.q_cal, q_nal_ctrl, q_cal_ctrl)

    # ---- Monte-Carlo over source variability -------------------------------- #
    rng = np.random.default_rng(seed)
    dapd = np.empty(n_mc)
    apd = np.empty(n_mc)
    qn = np.empty(n_mc)
    cqin = np.empty(n_mc)
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
        cqin[i] = _cqinward(r.q_nal, r.q_cal, q_nal_ctrl, q_cal_ctrl)
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
        qnet=pt.qnet, ead=pt.ead, classification=point_class,
        triangulation_ms=pt.triangulation, baseline_triangulation_ms=baseline_tri,
        n_mc=n_mc,
        dapd90_distribution=dapd, apd90_distribution=apd, qnet_distribution=qn,
        classification_distribution=counts, classification_flip_frequency=flip_freq,
        tier=tier, warnings=warnings, excluded_channels=excluded,
        channels_used=channels_used, herg_dynamic=use_dynamic, uq="moments",
        cqinward=cqinward_point, cqinward_distribution=cqin,
    )


def _assess_bayes(ds, drug_l, blocks, ap_rec, scales, baseline, baseline_tri, metric,
                  reference_exposure, n_mc, seed, n_beats, _herg_for, use_dynamic, tier,
                  prior, q_nal_ctrl, q_cal_ctrl):
    """The v0.2 Bayesian posterior-predictive path of :func:`assess`.

    Per channel (identifiable *and* censored), infer the ``(IC50, Hill)`` posterior
    under the declared prior, then sample it in place of the v0.1 lognormal draw.
    Channels are inferred with independent RNG streams so their IC50 uncertainties
    are propagated independently; each channel's draws are a fixed, indexed sample
    set, so the i-th Monte-Carlo iteration uses the i-th posterior sample (common
    random numbers, exactly as the moments path and flip_sensitivity require).
    """
    from .infer import infer_channel, learn_tau_pop, resolve_prior

    prior_obj = resolve_prior(ds, prior)
    tau_pop = learn_tau_pop(ds.channel_blocks, prior_obj)
    n_draws = max(n_mc, 500)   # floor so point/summary stats stay stable when n_mc is tiny

    posts, warnings, censored_channels, prior_dominated = {}, [], [], []
    for j, b in enumerate(blocks):
        post = infer_channel(b, prior_obj, tau_pop, n_draws=n_draws,
                             seed=seed + 1009 * (j + 1))
        posts[b.channel] = post
        if post.censored:
            censored_channels.append(f"{b.channel} (max block "
                                     f"{b.assay_context.max_block_observed_percent:g}%)")
            warnings.append(
                f"{b.channel}: IC50 unidentifiable (max block "
                f"{b.assay_context.max_block_observed_percent:g}% < 60%) — contributes a "
                f"one-sided CENSORED posterior (bounded below near the top tested dose, "
                f"prior-dominated); still caps assessment at Tier D. Measure at higher doses.")
        if post.prior_dominated:
            prior_dominated.append(b.channel)
        if b.variability.fold_range and b.variability.fold_range > 5:
            warnings.append(f"{b.channel}: inter-source IC50 fold-range "
                            f"{b.variability.fold_range:g} — classification may flip with source")
    if prior_dominated:
        warnings.append(
            f"prior-dominated channels {prior_dominated}: the posterior is prior-shaped, "
            f"not data-shaped (prior_sensitivity exceeds threshold) — the honest fix is "
            f"more / higher-dose data, not a tighter prior.")

    channels_used = list(posts.keys())

    def _run(ic50_by, hill_by):
        bf = _block_factors(ic50_by, hill_by, reference_exposure)
        p = KernelParams().with_scales(scales)
        p.block.update(bf)
        return simulate_beats(p, n_beats=n_beats, herg=_herg_for(ic50_by.get("IKr", 1e9)))

    # ---- point estimate (posterior MEDIAN per channel — a readout, not a verdict) -- #
    ic50_point = {ch: float(np.median(p.ic50_samples())) for ch, p in posts.items()}
    hill_point = {ch: float(np.median(p.hill)) for ch, p in posts.items()}
    pt = _run(ic50_point, hill_point)
    dapd_point = 100.0 * (pt.apd90 - baseline) / baseline if baseline else float("nan")
    point_class = classify_metric(metric, dapd_point, pt.qnet)
    cqinward_point = _cqinward(pt.q_nal, pt.q_cal, q_nal_ctrl, q_cal_ctrl)

    def _mc(predictive: bool):
        dapd = np.empty(n_mc); apd = np.empty(n_mc); qn = np.empty(n_mc); cqin = np.empty(n_mc)
        classes = []
        for i in range(n_mc):
            ic50_s = {ch: (p.predictive_ic50_samples()[i] if predictive
                           else p.ic50_samples()[i]) for ch, p in posts.items()}
            hill_s = {ch: p.hill[i] for ch, p in posts.items()}
            r = _run(ic50_s, hill_s)
            apd[i] = r.apd90
            dapd[i] = 100.0 * (r.apd90 - baseline) / baseline if baseline else float("nan")
            qn[i] = r.qnet
            cqin[i] = _cqinward(r.q_nal, r.q_cal, q_nal_ctrl, q_cal_ctrl)
            classes.append(classify_metric(metric, dapd[i], qn[i]))
        return dapd, apd, qn, cqin, classes

    if n_mc:
        dapd, apd, qn, cqin, classes = _mc(predictive=False)
        counts = {lab: classes.count(lab) / n_mc for lab in RISK_LABELS}
        flip_freq = float(np.mean([c != point_class for c in classes]))
        _, _, _, _, repro_classes = _mc(predictive=True)
        repro_flip = float(np.mean([c != point_class for c in repro_classes]))
    else:
        dapd = np.empty(0); apd = np.empty(0); qn = np.empty(0); cqin = np.empty(0)
        counts = {lab: (1.0 if lab == point_class else 0.0) for lab in RISK_LABELS}
        flip_freq = 0.0
        repro_flip = float("nan")

    return RiskAssessment(
        drug=drug_l, ap_model=ap_rec.id, reference_exposure_nM=reference_exposure,
        metric=metric, apd90=pt.apd90, baseline_apd90=baseline, dapd90_pct=dapd_point,
        qnet=pt.qnet, ead=pt.ead, classification=point_class,
        triangulation_ms=pt.triangulation, baseline_triangulation_ms=baseline_tri,
        n_mc=n_mc, dapd90_distribution=dapd, apd90_distribution=apd, qnet_distribution=qn,
        classification_distribution=counts, classification_flip_frequency=flip_freq,
        tier=tier, warnings=warnings, excluded_channels=[], channels_used=channels_used,
        herg_dynamic=use_dynamic, uq="bayes", reproducibility_flip_frequency=repro_flip,
        censored_channels=censored_channels, prior_dominated_channels=prior_dominated,
        cqinward=cqinward_point, cqinward_distribution=cqin,
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
    baseline = _cached_baseline(tuple(sorted(scales.items())), n_beats)[0]

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


@dataclass
class SobolChannel:
    """Variance-based sensitivity indices for one channel's IC50 (spec v0.2 sec 5)."""
    channel: str
    n_sources: int
    first_order: float       # S_i  — variance explained by this channel alone
    first_order_se: float
    total_effect: float      # S_Ti — variance explained including all interactions
    total_effect_se: float

    @property
    def interaction_load(self) -> float:
        """S_Ti - S_i — the interaction contribution, invisible to a one-at-a-time
        design. A channel with a large interaction load matters only jointly."""
        return self.total_effect - self.first_order


@dataclass
class SobolSensitivity:
    """Global, variance-based attribution of the metric's uncertainty to each channel.

    Generalizes the OAT :class:`FlipSensitivity`: first-order ``S_i`` approximates the
    OAT solo effect, total-effect ``S_Ti`` approximates the OAT frozen-complement, and
    ``interaction_load = S_Ti - S_i`` is the part a one-at-a-time design cannot see. The
    decomposition is of the *continuous* metric variance (the binary flip is reported
    alongside as ``all_vary_flip_frequency``). An analysis of uncertainty, not a verdict.
    """
    drug: str
    ap_model: str
    metric: str
    uq: str
    classification: str
    all_vary_flip_frequency: float
    metric_variance: float
    n_base: int                       # Saltelli base sample size N (cost ~ N*(d+2))
    channels: List[SobolChannel]      # sorted by total_effect, descending
    tier: str
    warnings: List[str] = field(default_factory=list)
    excluded_channels: List[str] = field(default_factory=list)
    censored_channels: List[str] = field(default_factory=list)
    clinical_use: str = ("PROHIBITED — research / safety-methodology / education only; "
                         "not a regulatory determination")

    @property
    def dominant_channel(self) -> Optional[str]:
        """Interaction-aware dominant driver: the largest total-effect channel."""
        return self.channels[0].channel if self.channels else None

    def summary(self) -> str:
        lines = [
            f"sobol-sensitivity  drug={self.drug}  ap_model={self.ap_model}  "
            f"tier={self.tier}  uq={self.uq}",
            f"point class: {self.classification.upper()}   metric={self.metric}   "
            f"all-vary flip frequency: {self.all_vary_flip_frequency:.0%}   "
            f"Var(metric)={self.metric_variance:.3g}  (N={self.n_base})",
            "  channel   sources   S_i (1st)        S_Ti (total)     interaction",
        ]
        for c in self.channels:
            lines.append(f"  {c.channel:<8} {c.n_sources:>5}    "
                         f"{c.first_order:>5.2f} +/- {c.first_order_se:<4.2f}   "
                         f"{c.total_effect:>5.2f} +/- {c.total_effect_se:<4.2f}   "
                         f"{c.interaction_load:>+5.2f}")
        if self.dominant_channel:
            lines.append(f"dominant uncertainty driver (total-effect, interaction-aware): "
                         f"{self.dominant_channel} — pin this IC50 down first")
        if self.censored_channels:
            lines.append(f"CENSORED (one-sided posterior; included under uq=bayes): "
                         f"{', '.join(self.censored_channels)}")
        if self.excluded_channels:
            lines.append(f"EXCLUDED (unidentifiable IC50): {', '.join(self.excluded_channels)}")
        for w in self.warnings:
            lines.append(f"  warn: {w}")
        lines.append("NOTE: an uncertainty attribution, not a safety verdict. "
                     + self.clinical_use)
        return "\n".join(lines)


def _sobol_sensitivity(ds: Dataset, drug: str, ap_model: str, metric: str, n_mc: int,
                       seed: int, exposure_nM, exposure_kind, exposure_multiple,
                       n_beats, uq: str, prior) -> SobolSensitivity:
    """Saltelli/Jansen first-order and total-effect Sobol indices over the continuous
    metric. Each channel's IC50 is sampled from its marginal (moments lognormal, or
    bootstrapped from the v0.2 Bayesian posterior when ``uq="bayes"``)."""
    drug_l = drug.lower()
    blocks = [b for b in ds.blocks_for(drug_l) if isinstance(b, ChannelBlock)]
    if not blocks:
        raise KeyError(f"no channel-block records for drug '{drug}'")
    if metric not in ("qnet", "apd90"):
        raise ValueError(f"metric must be 'qnet' or 'apd90', got {metric!r}")
    if uq not in ("moments", "bayes"):
        raise ValueError(f"uq must be 'moments' or 'bayes', got {uq!r}")

    ref = ds.drug_reference(drug_l)
    reference_exposure = resolve_free_exposure(
        ref, exposure_nM=exposure_nM, exposure_kind=exposure_kind,
        exposure_multiple=exposure_multiple)
    ap_rec = _resolve_ap_model(ds, ap_model)
    scales = ap_rec.conductance_scales
    baseline = _cached_baseline(tuple(sorted(scales.items())), n_beats)[0]

    excluded, censored, warnings = [], [], []
    tier = _worst_tier([ap_rec.tier] + [b.tier for b in blocks])

    # build per-channel log10-IC50 samplers + fixed Hill (IC50-spread attribution)
    rng = np.random.default_rng(seed)
    samplers: Dict[str, Callable[[int], Any]] = {}
    hill_by: Dict[str, float] = {}
    chan_meta: List[Tuple[str, int]] = []
    if uq == "bayes":
        from .infer import infer_channel, learn_tau_pop, resolve_prior
        prior_obj = resolve_prior(ds, prior)
        tau_pop = learn_tau_pop(ds.channel_blocks, prior_obj)
        for j, b in enumerate(blocks):
            post = infer_channel(b, prior_obj, tau_pop, n_draws=max(4 * n_mc, 2000),
                                 seed=seed + 1009 * (j + 1))
            pool = post.log10_ic50

            def _posterior_sampler(n: int, _p: Any = pool) -> Any:
                return rng.choice(_p, size=n)
            samplers[b.channel] = _posterior_sampler
            hill_by[b.channel] = float(np.median(post.hill))
            chan_meta.append((b.channel, len(b.source_ic50s_nm)))
            if post.censored:
                censored.append(f"{b.channel} (max block {b.assay_context.max_block_observed_percent:g}%)")
    else:
        for d in _channel_draws(blocks):
            mu, sigma = d.mu_log10, d.sigma_log10

            def _lognormal_sampler(n: int, _m: float = mu, _s: float = sigma) -> Any:
                return rng.normal(_m, _s, n)
            samplers[d.channel] = _lognormal_sampler
            hill_by[d.channel] = d.hill
            chan_meta.append((d.channel, d.n_sources))
        for b in blocks:
            if not b.identifiable:
                excluded.append(f"{b.channel} (max block {b.assay_context.max_block_observed_percent:g}%)")

    channels = [c for c, _ in chan_meta]
    n_dims = len(channels)
    if n_dims == 0:
        return SobolSensitivity(drug=drug_l, ap_model=ap_rec.id, metric=metric, uq=uq,
                                classification="intermediate", all_vary_flip_frequency=0.0,
                                metric_variance=0.0, n_base=0, channels=[], tier=tier,
                                warnings=["no identifiable channels to attribute"],
                                excluded_channels=excluded, censored_channels=censored)

    N = max(n_mc, 8)

    def _metric(row: Dict[str, float]) -> float:
        ic50_by = {ch: 10.0 ** row[ch] for ch in channels}
        bf = _block_factors(ic50_by, hill_by, reference_exposure)
        p = KernelParams().with_scales(scales)
        p.block.update(bf)
        r = simulate_beats(p, n_beats=n_beats)
        if metric == "qnet":
            return r.qnet
        return 100.0 * (r.apd90 - baseline) / baseline if baseline else float("nan")

    # Saltelli A/B design (each column is a channel)
    A = {ch: samplers[ch](N) for ch in channels}
    B = {ch: samplers[ch](N) for ch in channels}

    def _eval(mat):
        return np.array([_metric({ch: mat[ch][k] for ch in channels}) for k in range(N)])

    fA = _eval(A)
    fB = _eval(B)
    fAB = {}
    for ch in channels:
        ABi = dict(A)
        ABi[ch] = B[ch]
        fAB[ch] = _eval(ABi)

    allf = np.concatenate([fA, fB])
    var = float(np.var(allf, ddof=1))

    def _class(v: float) -> str:
        return (classify_metric(metric, float("nan"), v) if metric == "qnet"
                else classify_metric(metric, v, float("nan")))

    # flip frequency from the full-variation A samples (all channels vary)
    point_class = _class(float(np.median(allf)))
    flip = float(np.mean([_class(v) != point_class for v in fA])) if N else 0.0

    def _boot_se(estimator) -> float:
        if N < 4:
            return float("nan")
        rb = np.random.default_rng(seed + 31)
        vals = []
        for _ in range(40):
            idx = rb.integers(0, N, N)
            vals.append(estimator(idx))
        return float(np.std(vals, ddof=1))

    def _first_order(fb, fc):
        """Janon (2014) first-order estimator on the pair (f_B, f_AB^i), which share
        factor i. Lower-variance and better-bounded than the Saltelli cross-product."""
        mid = 0.5 * (fb + fc)
        m = float(np.mean(mid))
        num = float(np.mean(fb * fc)) - m * m
        den = float(np.mean(0.5 * (fb * fb + fc * fc))) - m * m
        return num / den if den > 0 else 0.0

    sob_channels: List[SobolChannel] = []
    for ch, nsrc in chan_meta:
        fABi = fAB[ch]
        if var > 0:
            Si = _first_order(fB, fABi)
            STi = float(np.mean((fA - fABi) ** 2) / (2.0 * var))        # Jansen 2010
        else:
            Si = STi = 0.0
        Si = float(np.clip(Si, 0.0, 1.0))
        STi = float(np.clip(STi, 0.0, 1.0))
        se_i = _boot_se(lambda idx, fABi=fABi: _first_order(fB[idx], fABi[idx]))
        se_t = _boot_se(lambda idx, fABi=fABi: np.mean((fA[idx] - fABi[idx]) ** 2) / (2.0 * var) if var > 0 else 0.0)
        sob_channels.append(SobolChannel(channel=ch, n_sources=nsrc, first_order=Si,
                                         first_order_se=se_i, total_effect=STi,
                                         total_effect_se=se_t))
    sob_channels.sort(key=lambda c: c.total_effect, reverse=True)

    if censored:
        warnings.append("censored channels contribute prior-shaped (one-sided) "
                        "uncertainty; their indices are prior-influenced")
    return SobolSensitivity(
        drug=drug_l, ap_model=ap_rec.id, metric=metric, uq=uq, classification=point_class,
        all_vary_flip_frequency=flip, metric_variance=var, n_base=N, channels=sob_channels,
        tier=tier, warnings=warnings, excluded_channels=excluded, censored_channels=censored)


def flip_sensitivity(ds: Dataset, drug: str, ap_model: str = "cipaordv1.0",
                     metric: str = DEFAULT_METRIC, n_mc: int = 200, seed: int = 0,
                     exposure_nM: Optional[float] = None, exposure_kind: str = "free",
                     exposure_multiple: float = REFERENCE_EXPOSURE_MULTIPLE,
                     n_beats: int = 3, method: str = "oat", uq: str = "moments",
                     prior: Optional[object] = None):
    """Attribute the classification-flip frequency to each channel's IC50 spread.

    ``method="oat"`` (default) runs the v0.1 one-at-a-time design: for every
    identifiable channel, "only this channel varies" (main effect) and "this channel
    frozen, others vary" (total effect), with common random numbers across scenarios.
    Returns a :class:`FlipSensitivity`.

    ``method="sobol"`` runs the v0.2 variance-based (Sobol) design over the continuous
    metric, which — unlike OAT — *sees channel interactions*: it reports first-order
    ``S_i``, total-effect ``S_Ti``, and the **interaction load** ``S_Ti - S_i`` per
    channel, each with a bootstrap Monte-Carlo standard error. Returns a
    :class:`SobolSensitivity`. The dominant-driver recommendation becomes
    interaction-aware (a channel can have a small solo effect but a large total
    effect). See spec v0.2 sec 5.
    """
    if method == "sobol":
        return _sobol_sensitivity(ds, drug, ap_model=ap_model, metric=metric, n_mc=n_mc,
                                  seed=seed, exposure_nM=exposure_nM,
                                  exposure_kind=exposure_kind,
                                  exposure_multiple=exposure_multiple, n_beats=n_beats,
                                  uq=uq, prior=prior)
    if method != "oat":
        raise ValueError(f"method must be 'oat' or 'sobol', got {method!r}")
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
    baseline = _cached_baseline(tuple(sorted(scales.items())), n_beats)[0]

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

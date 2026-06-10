"""Hierarchical Bayesian dose-response inference (spec v0.2).

Where v0.1 transcribed an IC50 spread as a *number* (the method-of-moments
``std(log10(source_ic50s))`` in :mod:`simulate`), v0.2 *infers* a posterior over
the per-channel ``(IC50, Hill)`` from the source data under a declared prior
(:mod:`dataset/priors`), and carries three things the moments sampler could not:

1.  **Partial pooling.** A single-source channel no longer gets a hard-coded
    ``DEFAULT_SINGLE_SOURCE_SIGMA``; its between-lab spread is the dataset-learned
    typical spread ``tau_pop`` (``learn_tau_pop``) — a magic constant becomes an
    inferred, citable quantity.
2.  **Hill uncertainty.** The Hill coefficient is given a posterior and propagated,
    instead of being fixed at a point value.
3.  **Censoring.** A sub-saturating max-block observation (``max_block < 60%``) is
    no longer discarded; it becomes a one-sided censored likelihood that bounds the
    IC50 from below (``_infer_censored``), yielding a proper but wide posterior with
    a heavy, prior-shaped right tail. The Tier-D gate is *preserved* (spec v0.2 sec 6);
    v0.2 only stops throwing the information away.

The sampler is a **direct (grid / conjugate) sampler**, not MCMC: the between-lab
SD ``tau`` is drawn from its collapsed 1-D marginal (mu integrated out in closed
form — the standard hierarchical-normal marginal, so there is no Neal funnel), the
true log-IC50 ``mu`` is then a conjugate Normal draw given ``tau``, and the censored
case grids the 1-D nonlinear posterior directly. Draws are therefore i.i.d. and
exactly from the posterior; the reported ``rhat``/``ess`` diagnostics are computed
the usual way and are trivially satisfied because a direct sampler is perfectly
mixed. Everything is seeded with NumPy's platform-independent PCG64, so a posterior
is a deterministic projection of ``(source data + prior)`` — the v0.2 analog of the
v0.1 ``build_records`` reproducibility guard.

NOT a verdict. A posterior is never a permission to point-estimate (spec v0.2 sec 10):
the outputs that leave this module are distributions, flip frequencies, and
explicitly-labeled diagnostics.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .records import ChannelBlock

DEFAULT_PRIOR_ID = "harmonia-ic50-prior-v1"
PRIOR_DOMINANCE_THRESHOLD = 0.5   # prior_sensitivity above this flags a channel (sec 6/7)
_WIDEN = 3.0                      # prior-widening factor for the prior_sensitivity probe
_SIGMA_HILL_OBS = 0.10            # within-source log-Hill observation SD
_TAU_FLOOR = 0.03                 # log10 SD floor (matches v0.1 moments floor)


# --------------------------------------------------------------------------- #
# The prior registry (priors are declared inputs, not hidden choices — sec 7)
# --------------------------------------------------------------------------- #
@dataclass
class Prior:
    id: str
    m0: float            # channel-level prior mean of true log10 IC50 (nM)
    s0: float            # channel-level prior SD
    tau_scale: float     # HalfNormal scale of the between-lab SD (log10)
    hill_mu: float       # lognormal mu of the Hill coefficient
    hill_sigma: float
    block_sigma: float   # fractional-block obs noise for the censored likelihood
    raw: Dict = None     # type: ignore[assignment]

    def widened(self, factor: float) -> "Prior":
        """A deliberately widened copy, used to probe prior-sensitivity (sec 7).

        Only the *subjective* priors widen — the channel-level location prior ``s0``
        and the Hill prior. The between-lab scale is held fixed because ``tau_pop`` is
        an empirical-Bayes quantity *learned from the dataset* (``learn_tau_pop``),
        not a subjective choice; widening it would mislabel data-driven between-lab
        uncertainty as prior dominance.
        """
        return Prior(id=f"{self.id}+wide{factor:g}", m0=self.m0,
                     s0=self.s0 * factor, tau_scale=self.tau_scale,
                     hill_mu=self.hill_mu, hill_sigma=self.hill_sigma * factor,
                     block_sigma=self.block_sigma, raw=self.raw)

    @classmethod
    def from_dict(cls, d: Dict) -> "Prior":
        return cls(
            id=d["id"],
            m0=d["channel_level"]["m_0_log10nm"],
            s0=d["channel_level"]["s_0_log10nm"],
            tau_scale=d["between_lab_tau"]["scale_log10"],
            hill_mu=d["hill"]["mu"],
            hill_sigma=d["hill"]["sigma"],
            block_sigma=d.get("censored", {}).get("block_sigma", 0.05),
            raw=d,
        )


# A code-level fallback identical to harmonia-ic50-prior-v1, so inference works
# even if the dataset/priors directory is unavailable (e.g. a partial checkout).
_FALLBACK_PRIOR = Prior(id=DEFAULT_PRIOR_ID, m0=3.0, s0=2.0, tau_scale=0.5,
                        hill_mu=0.0, hill_sigma=0.3, block_sigma=0.05, raw={})


def load_priors(root: pathlib.Path) -> Dict[str, Prior]:
    out: Dict[str, Prior] = {}
    pdir = pathlib.Path(root) / "priors"
    if pdir.is_dir():
        for p in sorted(pdir.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            out[d["id"]] = Prior.from_dict(d)
    return out


def resolve_prior(ds, prior: Optional[object] = None) -> Prior:
    """Coerce ``prior`` (a :class:`Prior`, an id string, or None) to a Prior,
    looking it up in the dataset's registry and falling back to the pinned
    default."""
    if isinstance(prior, Prior):
        return prior
    key = prior or DEFAULT_PRIOR_ID
    reg = getattr(ds, "priors", {}) or {}
    if key in reg:
        return reg[key]
    return _FALLBACK_PRIOR


# --------------------------------------------------------------------------- #
# Dataset-learned between-lab spread (the partial-pooling hyperparameter)
# --------------------------------------------------------------------------- #
def learn_tau_pop(blocks: Sequence[ChannelBlock], prior: Prior) -> float:
    """The dataset's *learned* typical between-lab log10-IC50 SD.

    Pooled within-channel SD over every identifiable, multi-source channel-block.
    This is what a single-source channel borrows instead of a hard-coded constant:
    "channels like this one vary lab-to-lab by *this* much," and it sharpens as the
    dataset accrues more multi-source channels (sec 2.2). Falls back to the prior
    scale when the dataset has no multi-source channel yet.
    """
    variances = []
    for b in blocks:
        if not isinstance(b, ChannelBlock) or not b.identifiable:
            continue
        ic50s = b.source_ic50s_nm
        if len(ic50s) >= 2:
            variances.append(float(np.var(np.log10(ic50s), ddof=1)))
    if not variances:
        return prior.tau_scale
    return max(float(np.sqrt(np.mean(variances))), _TAU_FLOOR)


# --------------------------------------------------------------------------- #
# The posterior object
# --------------------------------------------------------------------------- #
@dataclass
class Posterior:
    channel: str
    log10_ic50: np.ndarray        # true-value posterior (mu_c) — drives the flip (sec 2.4)
    log10_ic50_pred: np.ndarray   # new-lab predictive (mu_c + tau_c z) — reproducibility readout
    hill: np.ndarray
    tau: np.ndarray               # between-lab SD posterior
    censored: bool
    n_sources: int
    prior_id: str
    rhat_max: float
    ess_min: float
    prior_sensitivity: float      # fraction of posterior variance from the prior (sec 7)
    identifiability_score: float  # continuous identifiability readout (sec 6); 1 = sharp
    x_max_nm: Optional[float] = None

    @property
    def mean_log10(self) -> float:
        return float(np.mean(self.log10_ic50))

    @property
    def sd_log10(self) -> float:
        return float(np.std(self.log10_ic50, ddof=1))

    @property
    def hill_mean(self) -> float:
        return float(np.mean(self.hill))

    @property
    def hill_sd(self) -> float:
        return float(np.std(self.hill, ddof=1))

    @property
    def prior_dominated(self) -> bool:
        return self.prior_sensitivity >= PRIOR_DOMINANCE_THRESHOLD

    def ic50_samples(self) -> np.ndarray:
        return 10.0 ** self.log10_ic50

    def predictive_ic50_samples(self) -> np.ndarray:
        return 10.0 ** self.log10_ic50_pred

    def summary_dict(self) -> Dict:
        """The cached ``posterior_summary`` projection (spec v0.2 sec 3), rounded so
        it is stable across platforms."""
        q = np.quantile(self.log10_ic50, [0.05, 0.95])
        return {
            "log10_ic50_nm": {"mean": round(self.mean_log10, 4),
                              "sd": round(self.sd_log10, 4),
                              "q05": round(float(q[0]), 4),
                              "q95": round(float(q[1]), 4)},
            "hill": {"mean": round(self.hill_mean, 4), "sd": round(self.hill_sd, 4)},
            "rhat_max": round(self.rhat_max, 4),
            "ess_min": int(self.ess_min),
            "identifiability_score": round(self.identifiability_score, 4),
            "prior_sensitivity": round(self.prior_sensitivity, 4),
            "censored": self.censored,
        }


# --------------------------------------------------------------------------- #
# Direct samplers
# --------------------------------------------------------------------------- #
def _grid_sample(rng: np.random.Generator, grid: np.ndarray, logp: np.ndarray,
                 n: int) -> np.ndarray:
    """Smooth inverse-CDF draws from an unnormalized 1-D log-density on ``grid``."""
    logp = logp - np.max(logp)
    w = np.exp(logp)
    cdf = np.cumsum(w)
    total = cdf[-1]
    if not np.isfinite(total) or total <= 0:
        return np.full(n, grid[len(grid) // 2])
    cdf = cdf / total
    return np.interp(rng.random(n), cdf, grid)


def _tau_log_marginal(tau: np.ndarray, d: np.ndarray, s0: float,
                      tau_scale: float) -> np.ndarray:
    """log p(tau | data) with the channel mean mu integrated out analytically.

    Hierarchical normal: y ~ Normal(m0 1, tau^2 I + s0^2 J). By the matrix-
    determinant lemma the log-marginal of n observations d = y - m0 is, up to a
    constant, the expression below; multiplied by a HalfNormal(tau_scale) prior.
    For n = 1 the leading ``(n-1) log tau^2`` term vanishes, so there is no
    tau -> 0 degeneracy and a single source is correctly prior-driven.
    """
    n = d.size
    SSd = float(np.sum(d * d))
    S1 = float(np.sum(d))
    t2 = tau * tau
    denom = t2 + n * s0 * s0
    quad = (SSd - (s0 * s0 / denom) * S1 * S1) / t2
    log_lik = -0.5 * ((n - 1) * np.log(t2) + np.log(denom) + quad)
    log_prior = -0.5 * (tau / tau_scale) ** 2          # HalfNormal (tau > 0)
    return log_lik + log_prior


def _tau_log_marginal_het(tau: np.ndarray, y: np.ndarray, v: np.ndarray, m0: float,
                          s0: float, tau_scale: float) -> np.ndarray:
    """Heteroscedastic generalization of :func:`_tau_log_marginal`: each observation
    ``y_s`` has its *own* extra variance ``v_s`` (the raw-regime per-source fit
    variance, sec 2.1), so the noise is ``Normal(mu, tau^2 + v_s)``. By Sherman-Morrison
    the marginal over ``y ~ Normal(m0 1, D + s0^2 J)`` with ``D = diag(tau^2 + v_s)`` is
    closed-form. It reduces exactly to the homoscedastic form when every ``v_s = 0``
    (the existing summary regime), which is why a record carrying no raw data is left on
    the byte-identical path."""
    r = y - m0
    out = np.empty_like(tau)
    s02 = s0 * s0
    for i, t in enumerate(tau):
        d = t * t + v                       # per-source total variance
        a = 1.0 / d
        sum_a = float(np.sum(a))
        sum_ra = float(np.sum(r * a))
        sum_r2a = float(np.sum(r * r * a))
        logdet = float(np.sum(np.log(d))) + np.log(1.0 + s02 * sum_a)
        quad = sum_r2a - s02 * sum_ra * sum_ra / (1.0 + s02 * sum_a)
        out[i] = -0.5 * (logdet + quad) - 0.5 * (t / tau_scale) ** 2
    return out


def _infer_summary_core(y: np.ndarray, v: np.ndarray, prior: Prior, tau_pop: float,
                        n_draws: int, rng: np.random.Generator):
    """The hierarchical summary-regime draw of (mu, tau): tau from its collapsed
    marginal, then mu | tau conjugate Normal. ``v`` are per-source extra variances
    (zero in the pure-summary regime). Returns (mu, pred, tau)."""
    homoscedastic = not np.any(v > 0)
    tau_hi = max(4.0 * tau_pop, 3.0 * (np.std(y, ddof=1) if y.size >= 2 else tau_pop), 0.5)
    grid = np.linspace(_TAU_FLOOR, tau_hi, 400)
    if homoscedastic:
        # byte-identical to the v0.2 path for every record carrying no raw data
        logp = _tau_log_marginal(grid, y - prior.m0, prior.s0, tau_pop)
    else:
        logp = _tau_log_marginal_het(grid, y, v, prior.m0, prior.s0, tau_pop)
    tau = np.maximum(_grid_sample(rng, grid, logp, n_draws), _TAU_FLOOR)

    # mu | tau, y  ~  Normal (conjugate), heteroscedastic precision
    a = 1.0 / (tau[:, None] ** 2 + v[None, :])      # (n_draws, n_sources)
    prec = 1.0 / prior.s0 ** 2 + np.sum(a, axis=1)
    mu_mean = (prior.m0 / prior.s0 ** 2 + np.sum(a * y[None, :], axis=1)) / prec
    mu_sd = 1.0 / np.sqrt(prec)
    mu = mu_mean + mu_sd * rng.standard_normal(n_draws)
    pred = mu + tau * rng.standard_normal(n_draws)          # new-lab predictive (sec 2.4)
    return mu, pred, tau


def _gather_sources(block: ChannelBlock, prior: Prior):
    """Per-source observations for the hierarchical fit. A source carrying raw
    ``dose_response`` points is *fitted* (raw regime, sec 2.1): its log-IC50 estimate,
    that estimate's genuine fit variance, and a fitted Hill come from the points. A
    source without raw points contributes its transcribed log-IC50 with an optional
    ``fit_sd_log10`` (summary regime). Returns (y, v, hills)."""
    y, v, hills = [], [], []
    for s in block.source_values:
        dr = s.get("dose_response")
        if dr:
            mu_m, mu_sd, h_m, _ = fit_dose_response(
                dr.get("concentration_nm", []), dr.get("fractional_block", []),
                dr.get("sem"), prior, likelihood=dr.get("likelihood", "truncated_normal"))
            y.append(mu_m); v.append(mu_sd ** 2); hills.append(h_m)
        else:
            y.append(float(np.log10(s["ic50_nm"])))
            fs = s.get("fit_sd_log10")
            v.append(float(fs) ** 2 if fs else 0.0)
            if s.get("hill"):
                hills.append(float(s["hill"]))
    return np.asarray(y, dtype=float), np.asarray(v, dtype=float), hills


def _infer_identifiable(block: ChannelBlock, prior: Prior, tau_pop: float,
                        n_draws: int, rng: np.random.Generator):
    """Summary- or raw-regime posterior for an identifiable channel (sec 2.1-2.2)."""
    y, v, hills = _gather_sources(block, prior)
    mu, pred, tau = _infer_summary_core(y, v, prior, tau_pop, n_draws, rng)
    hill = _hill_posterior(block, prior, n_draws, rng, hills=hills or None)
    return mu, pred, tau, hill


def _infer_censored(block: ChannelBlock, prior: Prior, tau_pop: float,
                    n_draws: int, rng: np.random.Generator):
    """One-sided posterior for a sub-60%-block channel (sec 2.3).

    A sub-saturating max-block measurement does not *localize* the IC50 — it *bounds*
    it. We recover the top tested dose ``x_max`` from the stored extrapolated point +
    max block, then encode a genuinely **one-sided (probit-censored) likelihood**: a
    candidate ``(IC50, h)`` is penalized only insofar as it would have produced *more*
    block at ``x_max`` than was observed. Curves with IC50 well above ``x_max`` are all
    consistent, so above the lower edge the posterior is shaped by the prior, not the
    datum. The Hill coefficient is **marginalized over its prior** on the joint grid,
    which is the dominant source of the heavy right tail (a shallow Hill reaching the
    observed block at ``x_max`` implies a far higher IC50). The result is a proper but
    wide posterior with a hard-ish lower edge near ``x_max`` — prior-dominated by
    construction, and still Tier-D-capped downstream (sec 6).
    """
    from scipy.special import ndtr, expit

    m_obs = float(np.clip(block.assay_context.max_block_observed_percent / 100.0, 1e-3, 0.95))
    ic50_point = block.ic50_nm
    h_point = block.hill
    x_max = ic50_point * (m_obs / (1.0 - m_obs)) ** (1.0 / h_point)

    mu_grid = np.linspace(prior.m0 - 4.0 * prior.s0, prior.m0 + 5.0 * prior.s0, 600)
    h_grid = np.exp(prior.hill_mu + prior.hill_sigma * np.linspace(-3.0, 3.0, 25))
    MU, H = np.meshgrid(mu_grid, h_grid, indexing="ij")
    # predicted block at x_max, in the numerically-stable logistic form
    # f = x_max^h / (x_max^h + IC50^h) = expit(h * (log x_max - log IC50))
    f = expit(H * (np.log(x_max) - MU * np.log(10.0)))
    # one-sided: allow predicted block <= observed (+noise); forbid IC50 below x_max
    log_lik = np.log(ndtr((m_obs - f) / prior.block_sigma) + 1e-300)
    log_prior_mu = -0.5 * ((MU - prior.m0) / prior.s0) ** 2
    log_prior_h = -0.5 * ((np.log(H) - prior.hill_mu) / prior.hill_sigma) ** 2
    logpost = log_lik + log_prior_mu + log_prior_h

    mu, hill = _grid_sample_2d(rng, mu_grid, h_grid, logpost, n_draws)
    tau = np.abs(rng.standard_normal(n_draws)) * tau_pop     # HalfNormal(tau_pop)
    tau = np.maximum(tau, _TAU_FLOOR)
    pred = mu + tau * rng.standard_normal(n_draws)
    return mu, pred, tau, hill, x_max


def _grid_sample_2d(rng: np.random.Generator, mu_grid: np.ndarray, h_grid: np.ndarray,
                    logpost: np.ndarray, n: int):
    """Smooth joint draws of (mu, hill) from an unnormalized 2-D log-density."""
    logpost = logpost - np.max(logpost)
    w = np.exp(logpost).ravel()
    cdf = np.cumsum(w)
    total = cdf[-1]
    if not np.isfinite(total) or total <= 0:
        return (np.full(n, mu_grid[len(mu_grid) // 2]),
                np.full(n, h_grid[len(h_grid) // 2]))
    cdf = cdf / total
    idx = np.searchsorted(cdf, rng.random(n))
    idx = np.clip(idx, 0, w.size - 1)
    i_mu, i_h = np.unravel_index(idx, logpost.shape)
    # jitter within each cell so the marginals are continuous, not discretized
    d_mu = mu_grid[1] - mu_grid[0]
    mu = mu_grid[i_mu] + (rng.random(n) - 0.5) * d_mu
    log_h = np.log(h_grid)
    d_h = log_h[1] - log_h[0]
    hill = np.exp(log_h[i_h] + (rng.random(n) - 0.5) * d_h)
    return mu, hill


def _hill_posterior(block: ChannelBlock, prior: Prior, n_draws: int,
                    rng: np.random.Generator, hills=None) -> np.ndarray:
    """Conjugate log-Normal posterior for the Hill coefficient (gap #2): combine the
    lognormal prior with the source Hill values, so Hill uncertainty propagates.
    ``hills`` overrides the transcribed source Hills with raw-regime fitted Hills."""
    if not hills:
        hills = [s.get("hill") for s in block.source_values if s.get("hill")]
    if not hills:
        hills = [block.hill]
    logh = np.log(np.asarray(hills, dtype=float))
    n = logh.size
    prec = 1.0 / prior.hill_sigma ** 2 + n / _SIGMA_HILL_OBS ** 2
    mean = (prior.hill_mu / prior.hill_sigma ** 2 + n * float(np.mean(logh)) / _SIGMA_HILL_OBS ** 2) / prec
    sd = 1.0 / np.sqrt(prec)
    return np.exp(mean + sd * rng.standard_normal(n_draws))


# --------------------------------------------------------------------------- #
# Raw regime (sec 2.1): fit (IC50, Hill) from raw concentration-block points
# --------------------------------------------------------------------------- #
_RAW_DEFAULT_SEM = 0.06        # fractional-block obs SD when a source reports no SEM


def fit_dose_response(concentration_nm, fractional_block, sem, prior: Prior,
                      likelihood: str = "truncated_normal"):
    """Bayesian fit of ``(log10 IC50, Hill)`` to one source's raw dose-response points.

    This is the v0.2 **raw regime** (spec sec 2.1): instead of transcribing a fitted
    IC50 as a point with an *assumed* spread, the IC50 and Hill — and their genuine
    fit uncertainty — are inferred from the actual ``(concentration, fractional_block)``
    curve under the declared prior. A 2-D grid over ``(mu = log10 IC50, h)`` evaluates a
    truncated-normal (or beta) likelihood at each tested concentration; the function
    returns the marginal posterior mean/SD of ``mu`` and of the Hill coefficient.

    The Hill curve is the stable logistic form ``f = expit(h (log10 c - mu) ln10)``.
    A source whose top dose does not saturate yields a wide, right-skewed ``mu``
    marginal automatically — the raw analog of the channel-level censoring in sec 2.3.
    """
    from scipy.special import expit

    x = np.log10(np.asarray(concentration_nm, dtype=float))     # log10 concentration
    y = np.clip(np.asarray(fractional_block, dtype=float), 1e-4, 1 - 1e-4)
    if x.size == 0:
        return prior.m0, prior.s0, np.exp(prior.hill_mu), 0.3
    if sem is not None and len(sem) == len(y):
        sd = np.maximum(np.asarray(sem, dtype=float), 1e-3)
    else:
        sd = np.full(y.size, _RAW_DEFAULT_SEM)

    mu_grid = np.linspace(prior.m0 - 4.0 * prior.s0, prior.m0 + 4.0 * prior.s0, 400)
    h_grid = np.exp(prior.hill_mu + prior.hill_sigma * np.linspace(-3.0, 3.0, 41))
    MU, H = np.meshgrid(mu_grid, h_grid, indexing="ij")          # (nmu, nh)
    # predicted block at every tested concentration: f[k] over the (mu, h) grid
    log_lik = np.zeros_like(MU)
    for k in range(x.size):
        f = expit(H * (x[k] - MU) * np.log(10.0))
        if likelihood == "beta":
            # method-of-moments Beta with the same per-point SD
            var = sd[k] ** 2
            fc = np.clip(f, 1e-4, 1 - 1e-4)
            kappa = np.clip(fc * (1 - fc) / var - 1.0, 0.1, 1e4)
            a = fc * kappa
            b = (1 - fc) * kappa
            from scipy.special import gammaln
            log_lik += ((a - 1) * np.log(y[k]) + (b - 1) * np.log1p(-y[k])
                        - (gammaln(a) + gammaln(b) - gammaln(a + b)))
        else:   # truncated_normal (clipped Hill keeps the mean in [0,1])
            log_lik += -0.5 * ((y[k] - f) / sd[k]) ** 2 - np.log(sd[k])
    log_prior = (-0.5 * ((MU - prior.m0) / prior.s0) ** 2
                 - 0.5 * ((np.log(H) - prior.hill_mu) / prior.hill_sigma) ** 2)
    logpost = log_lik + log_prior

    w = np.exp(logpost - logpost.max())
    w = w / w.sum()
    mu_marg = w.sum(axis=1)                                       # over h
    h_marg = w.sum(axis=0)                                        # over mu
    mu_mean = float(np.sum(mu_grid * mu_marg))
    mu_sd = float(np.sqrt(max(np.sum((mu_grid - mu_mean) ** 2 * mu_marg), 1e-6)))
    h_mean = float(np.sum(h_grid * h_marg))
    h_sd = float(np.sqrt(max(np.sum((h_grid - h_mean) ** 2 * h_marg), 1e-6)))
    return mu_mean, mu_sd, h_mean, h_sd


# --------------------------------------------------------------------------- #
# Diagnostics (split-Rhat and bulk-ESS, the usual estimators)
# --------------------------------------------------------------------------- #
def _split_rhat(x: np.ndarray, n_chains: int = 4) -> float:
    m = x.size // n_chains
    if m < 2:
        return 1.0
    chains = x[: m * n_chains].reshape(n_chains, m)
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)
    W = float(np.mean(chain_vars))
    B = float(m * np.var(chain_means, ddof=1))
    if W <= 0:
        return 1.0
    var_hat = (m - 1) / m * W + B / m
    return float(np.sqrt(var_hat / W))


def _ess(x: np.ndarray) -> float:
    n = x.size
    xc = x - x.mean()
    var = float(np.dot(xc, xc) / n)
    if var <= 0:
        return float(n)
    # autocorrelation via FFT
    f = np.fft.rfft(xc, n=2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n].real
    acf = acf / acf[0]
    rho_sum = 0.0
    for k in range(1, n):
        if acf[k] + acf[k - 1] < 0:
            break
        rho_sum += acf[k]
    tau_int = 1.0 + 2.0 * rho_sum
    return float(n / max(tau_int, 1.0))


def _diagnostics(mu: np.ndarray):
    return _split_rhat(mu), _ess(mu)


# --------------------------------------------------------------------------- #
# The public inference entry point
# --------------------------------------------------------------------------- #
def infer_channel(block: ChannelBlock, prior: Prior, tau_pop: float,
                  n_draws: int = 4000, seed: int = 0,
                  _probe: bool = False) -> Posterior:
    """Infer the per-channel ``(IC50, Hill)`` posterior under ``prior``.

    ``_probe`` runs the widened-prior re-inference used to estimate
    ``prior_sensitivity`` and skips the recursive probe (sec 7).
    """
    rng = np.random.default_rng(seed)
    censored = not block.identifiable
    x_max = None
    if censored:
        mu, pred, tau, hill, x_max = _infer_censored(block, prior, tau_pop, n_draws, rng)
    else:
        mu, pred, tau, hill = _infer_identifiable(block, prior, tau_pop, n_draws, rng)

    rhat, ess = _diagnostics(mu)

    if _probe:
        prior_sens = 0.0
    else:
        wide = prior.widened(_WIDEN)
        n_probe = min(n_draws, 1500)
        wp = infer_channel(block, wide, tau_pop, n_draws=n_probe,
                           seed=seed + 7919, _probe=True)
        var_def = float(np.var(mu, ddof=1))
        var_wide = float(np.var(wp.log10_ic50, ddof=1))
        prior_sens = float(np.clip(1.0 - var_def / var_wide, 0.0, 1.0)) if var_wide > 0 else 0.0

    cv = float(np.std(10.0 ** mu) / np.mean(10.0 ** mu)) if np.mean(10.0 ** mu) > 0 else np.inf
    ident = float(1.0 / (1.0 + cv))

    return Posterior(
        channel=block.channel, log10_ic50=mu, log10_ic50_pred=pred, hill=hill, tau=tau,
        censored=censored, n_sources=len(block.source_ic50s_nm), prior_id=prior.id,
        rhat_max=rhat, ess_min=ess, prior_sensitivity=prior_sens,
        identifiability_score=ident, x_max_nm=x_max)


def posterior(ds, drug: str, channel: str, prior: Optional[object] = None,
              n_draws: int = 4000, seed: int = 0) -> Posterior:
    """Infer one drug x channel posterior (the spec v0.2 sec 8 ``harmonia.posterior``)."""
    from .records import ChannelBlock as _CB
    blocks = [b for b in ds.blocks_for(drug) if isinstance(b, _CB)]
    block = next((b for b in blocks if b.channel.lower() == channel.lower()), None)
    if block is None:
        raise KeyError(f"no {channel} channel-block record for drug '{drug}'")
    pr = resolve_prior(ds, prior)
    tau_pop = learn_tau_pop(ds.channel_blocks, pr)
    return infer_channel(block, pr, tau_pop, n_draws=n_draws, seed=seed)


# --------------------------------------------------------------------------- #
# Calibration of the inference itself (spec v0.2 sec 9): SBC + posterior coverage
# --------------------------------------------------------------------------- #
def _infer_mu_from_logic50s(y: np.ndarray, prior: Prior, tau_scale: float,
                            n_draws: int, seed: int) -> np.ndarray:
    """The true-value (mu) posterior given an array of per-source log10 IC50s — the
    summary-regime core, exposed for the calibration harness."""
    rng = np.random.default_rng(seed)
    mu, _, _ = _infer_summary_core(y, np.zeros_like(y), prior, tau_scale, n_draws, rng)
    return mu


def simulation_based_calibration(prior: Prior, n_sims: int = 256, n_obs: int = 3,
                                 tau_scale: Optional[float] = None, n_draws: int = 400,
                                 seed: int = 0):
    """Simulation-based calibration (spec v0.2 sec 9). Data simulated *from the prior*
    and then re-inferred must yield **rank-uniform** posteriors — the standard proof
    that an inference is correctly *implemented*, not merely plausible.

    For each replicate it draws a true ``mu ~ Normal(m0, s0)``, a between-lab
    ``tau ~ HalfNormal(tau_scale)``, then ``n_obs`` source log10-IC50s
    ``~ Normal(mu, tau)``; it re-infers the ``mu`` posterior and records the rank of
    the truth among the posterior draws. Returns ``(ranks, n_draws)``; under correct
    inference ``ranks`` is uniform on ``0..n_draws``.
    """
    ts = prior.tau_scale if tau_scale is None else tau_scale
    rng = np.random.default_rng(seed)
    ranks = np.empty(n_sims, dtype=int)
    for i in range(n_sims):
        mu_true = prior.m0 + prior.s0 * rng.standard_normal()
        tau_true = max(abs(rng.standard_normal()) * ts, _TAU_FLOOR)
        y = mu_true + tau_true * rng.standard_normal(n_obs)
        draws = _infer_mu_from_logic50s(y, prior, ts, n_draws, seed=seed + 1 + i)
        ranks[i] = int(np.sum(draws < mu_true))
    return ranks, n_draws


def sbc_uniformity_pvalue(ranks: np.ndarray, n_draws: int, n_bins: int = 16) -> float:
    """Chi-square goodness-of-fit p-value that the SBC ranks are uniform (high = good).
    Pure-NumPy survival function of the chi-square via the regularized upper incomplete
    gamma, so there is no SciPy dependency in the test path."""
    from math import gamma, exp, log
    edges = np.linspace(0, n_draws + 1, n_bins + 1)
    counts, _ = np.histogram(ranks, bins=edges)
    expected = ranks.size / n_bins
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    df = n_bins - 1
    # regularized upper incomplete gamma Q(df/2, chi2/2) via a series/continued form
    a, x = df / 2.0, chi2 / 2.0
    if x <= 0:
        return 1.0
    if x < a + 1.0:                       # lower-gamma series
        term = 1.0 / a
        s = term
        for k in range(1, 200):
            term *= x / (a + k)
            s += term
            if abs(term) < abs(s) * 1e-12:
                break
        p_lower = s * exp(-x + a * log(x) - log(gamma(a)))
        return float(max(0.0, min(1.0, 1.0 - p_lower)))
    # upper-gamma continued fraction
    b = x + 1.0 - a
    c = 1e300
    dd = 1.0 / b
    h = dd
    for k in range(1, 200):
        an = -k * (k - a)
        b += 2.0
        dd = an * dd + b
        if abs(dd) < 1e-300:
            dd = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        dd = 1.0 / dd
        delta = dd * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    q = exp(-x + a * log(x) - log(gamma(a))) * h
    return float(max(0.0, min(1.0, q)))


def posterior_coverage(prior: Prior, n_sims: int = 256, n_obs: int = 3,
                       tau_scale: Optional[float] = None, n_draws: int = 1500,
                       seed: int = 0, level: float = 0.90) -> float:
    """Posterior coverage (spec v0.2 sec 9): on synthetic data drawn from the model the
    central ``level`` credible interval should cover the true ``mu`` about ``level`` of
    the time. Returns the empirical coverage fraction."""
    ts = prior.tau_scale if tau_scale is None else tau_scale
    rng = np.random.default_rng(seed)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    hits = 0
    for i in range(n_sims):
        mu_true = prior.m0 + prior.s0 * rng.standard_normal()
        tau_true = max(abs(rng.standard_normal()) * ts, _TAU_FLOOR)
        y = mu_true + tau_true * rng.standard_normal(n_obs)
        draws = _infer_mu_from_logic50s(y, prior, ts, n_draws, seed=seed + 1 + i)
        lo, hi = np.quantile(draws, [lo_q, hi_q])
        hits += int(lo <= mu_true <= hi)
    return hits / n_sims

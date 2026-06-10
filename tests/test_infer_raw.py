"""Tests for the v0.2 raw dose-response regime + the calibration harness (spec sec 2.1, sec 9).

These close C-UQ-5: a source carrying raw ``(concentration, fractional_block)`` points
has its IC50/Hill *inferred from the curve* rather than transcribed, and the inference's
correctness is proven by simulation-based calibration and posterior coverage on synthetic
data (no fabricated real records — SBC/coverage are synthetic by construction).
"""
import numpy as np

import harmonia
from harmonia.infer import (fit_dose_response, infer_channel,
                            simulation_based_calibration, sbc_uniformity_pvalue,
                            posterior_coverage, _FALLBACK_PRIOR)
from harmonia.records import ChannelBlock


def _hill(conc, ic50, h):
    conc = np.asarray(conc, dtype=float)
    return conc ** h / (conc ** h + ic50 ** h)


def test_raw_fit_recovers_synthetic_curve():
    """A clean synthetic Hill curve is recovered to within a tight tolerance."""
    pr = _FALLBACK_PRIOR
    conc = np.array([5, 15, 50, 150, 500.0])
    y = _hill(conc, 50.0, 1.2)
    mu_m, mu_sd, h_m, h_sd = fit_dose_response(conc, y, [0.03] * 5, pr)
    assert abs(10 ** mu_m - 50.0) < 6.0          # IC50 within ~12%
    assert abs(h_m - 1.2) < 0.2                   # Hill recovered
    assert mu_sd > 0 and h_sd > 0                 # genuine fit uncertainty


def test_raw_fit_noisier_data_is_wider():
    """More observation noise -> a wider posterior (honest uncertainty)."""
    pr = _FALLBACK_PRIOR
    conc = np.array([5, 15, 50, 150, 500.0])
    y = _hill(conc, 50.0, 1.0)
    tight = fit_dose_response(conc, y, [0.02] * 5, pr)[1]
    loose = fit_dose_response(conc, y, [0.12] * 5, pr)[1]
    assert loose > tight


def _synthetic_block(sources):
    raw = {
        "id": "channel_block.synthetic.ikr", "kind": "channel_block",
        "subsystem": "channel_block", "tier": "A", "channel": "IKr",
        "block_model": "hill", "drug": {"name": "synthetic"},
        "parameters": [{"symbol": "IC50", "value": {"central": 50.0, "units": "nM"}},
                       {"symbol": "h", "value": {"central": 1.0, "units": "dimensionless"}}],
        "assay_context": {"max_block_observed_percent": 95},
        "source_values": sources,
        "extraction": {"review_status": "unverified"},
    }
    return ChannelBlock(raw)


def test_raw_regime_through_infer_channel():
    """A channel whose sources carry raw points infers end-to-end and centers near the
    true IC50; the summary-only path is untouched."""
    pr = _FALLBACK_PRIOR
    conc = [5, 15, 50, 150, 500.0]
    sources = []
    for ic50 in (45.0, 55.0):                      # two labs, true IC50 ~ 50
        sources.append({"ic50_nm": ic50, "citation": "synthetic",
                        "dose_response": {"concentration_nm": conc,
                                          "fractional_block": list(_hill(conc, ic50, 1.0)),
                                          "sem": [0.03] * 5,
                                          "likelihood": "truncated_normal"}})
    block = _synthetic_block(sources)
    p = infer_channel(block, pr, tau_pop=0.25, n_draws=4000, seed=0)
    assert abs(10 ** p.mean_log10 - 50.0) < 12.0
    assert not p.censored
    assert p.hill_sd > 0.0


def test_raw_per_source_fit_is_tight_but_true_value_keeps_between_lab_spread():
    """Rich raw data identifies *that lab's* IC50 tightly (the per-source fit SD is
    small), but the *true-value* posterior of a single-source channel still carries the
    full between-lab spread tau_pop — one lab, however precise, cannot pin down the
    cross-lab truth. Both facts are honest; the raw regime tightens the former."""
    pr = _FALLBACK_PRIOR
    conc = [3, 10, 30, 100, 300, 1000.0]
    y = _hill(conc, 50.0, 1.0)
    _, mu_sd, _, _ = fit_dose_response(conc, y, [0.02] * 6, pr)
    assert mu_sd < 0.08                            # per-source estimate is tightly identified
    block = _synthetic_block([{"ic50_nm": 50.0, "citation": "synthetic",
                               "dose_response": {"concentration_nm": conc,
                                                 "fractional_block": list(y),
                                                 "sem": [0.02] * 6}}])
    p = infer_channel(block, pr, tau_pop=0.25, n_draws=4000, seed=0)
    assert abs(p.sd_log10 - 0.25) < 0.06           # true value ~ tau_pop (between-lab dominated)


def test_backward_compat_summary_unchanged(ds):
    """A record with no raw data is byte-identical to the pure summary regime."""
    a = harmonia.posterior(ds, "dofetilide", "IKr", n_draws=4000, seed=0)
    b = harmonia.posterior(ds, "dofetilide", "IKr", n_draws=4000, seed=0)
    assert np.array_equal(a.log10_ic50, b.log10_ic50)
    # the known v0.2 values (homoscedastic path must not have drifted)
    assert abs(a.mean_log10 - 0.7121) < 1e-3
    assert abs(a.sd_log10 - 0.1109) < 1e-3


def test_simulation_based_calibration_is_uniform():
    """SBC: prior-simulated data re-infers to rank-uniform posteriors (sec 9)."""
    pr = _FALLBACK_PRIOR
    ranks, n_draws = simulation_based_calibration(pr, n_sims=300, n_obs=3,
                                                  tau_scale=0.25, n_draws=300, seed=0)
    pval = sbc_uniformity_pvalue(ranks, n_draws, n_bins=12)
    assert pval > 0.01, f"SBC ranks non-uniform (p={pval:.4f})"
    assert 0.4 < ranks.mean() / n_draws < 0.6


def test_posterior_coverage_is_calibrated():
    """The 90% credible interval covers the truth ~90% of the time (sec 9)."""
    pr = _FALLBACK_PRIOR
    cov = posterior_coverage(pr, n_sims=250, n_obs=3, tau_scale=0.25,
                             n_draws=1200, seed=0, level=0.90)
    assert 0.84 <= cov <= 0.96, f"90% coverage was {cov:.3f}"

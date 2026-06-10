"""Tests for the v0.2 hierarchical Bayesian dose-response inference (spec v0.2 sec 9).

The point of every guard here, as in v0.1, is *provable non-drift* and honest
behavior, not fixtures: the Bayesian posterior reduces to the v0.1 moments answer
where v0.1 was right, a censored channel produces a proper one-sided posterior and
stays Tier-D-capped, and the inference is a deterministic projection of (data + prior).
"""
import numpy as np
import pytest

import harmonia
from harmonia.infer import (infer_channel, learn_tau_pop, posterior,
                            resolve_prior, PRIOR_DOMINANCE_THRESHOLD)
from harmonia.records import ChannelBlock


def _block(ds, drug, channel):
    for b in ds.blocks_for(drug):
        if isinstance(b, ChannelBlock) and b.channel.lower() == channel.lower():
            return b
    raise KeyError(f"{drug} {channel}")


def test_prior_registry_loaded(ds):
    assert "harmonia-ic50-prior-v1" in ds.priors
    pr = resolve_prior(ds)
    assert pr.id == "harmonia-ic50-prior-v1"
    assert pr.raw["predictive"] is False


def test_reduction_to_moments_for_well_identified_channel(ds):
    """sec 0.2 non-drift: in the multi-source agreed regime the posterior mean of
    log10(IC50) converges to the log-geomean the v0.1 sampler centers on."""
    b = _block(ds, "dofetilide", "IKr")
    log_geomean = float(np.log10(np.exp(np.mean(np.log(b.source_ic50s_nm)))))
    p = posterior(ds, "dofetilide", "IKr", n_draws=8000, seed=0)
    assert abs(p.mean_log10 - log_geomean) < 0.05
    assert not p.censored


def test_inference_is_deterministic(ds):
    """A posterior is a deterministic projection of (data + prior)."""
    a = posterior(ds, "verapamil", "IKr", n_draws=3000, seed=4)
    b = posterior(ds, "verapamil", "IKr", n_draws=3000, seed=4)
    assert np.array_equal(a.log10_ic50, b.log10_ic50)
    assert a.summary_dict() == b.summary_dict()


def test_sampler_converges(ds):
    """Every channel posterior meets the rhat / ess gates (sec 9)."""
    pr = resolve_prior(ds)
    tau_pop = learn_tau_pop(ds.channel_blocks, pr)
    for b in ds.channel_blocks:
        if not isinstance(b, ChannelBlock):
            continue
        p = infer_channel(b, pr, tau_pop, n_draws=2000, seed=0)
        assert p.rhat_max < 1.01, f"{b.id} rhat={p.rhat_max}"
        assert p.ess_min > 400, f"{b.id} ess={p.ess_min}"


def test_single_source_borrows_learned_spread(ds):
    """A single-source channel's spread is the dataset-learned tau_pop, not a magic
    constant (sec 2.2). Its posterior SD should be near tau_pop."""
    pr = resolve_prior(ds)
    tau_pop = learn_tau_pop(ds.channel_blocks, pr)
    single = next(b for b in ds.channel_blocks
                  if isinstance(b, ChannelBlock) and b.identifiable
                  and len(b.source_ic50s_nm) == 1)
    p = infer_channel(single, pr, tau_pop, n_draws=4000, seed=0)
    assert abs(p.sd_log10 - tau_pop) < 0.06
    assert p.n_sources == 1


def test_censored_channel_is_one_sided_wide_and_tier_capped(ds):
    """sec 2.3: a sub-60%-block channel yields a proper, wide, one-sided posterior
    with a lower edge near the top tested dose and a heavy right tail — and it is
    prior-dominated and still produces a Tier-D assessment."""
    censored = next(b for b in ds.channel_blocks
                    if isinstance(b, ChannelBlock) and not b.identifiable)
    pr = resolve_prior(ds)
    tau_pop = learn_tau_pop(ds.channel_blocks, pr)
    p = infer_channel(censored, pr, tau_pop, n_draws=6000, seed=0)
    assert p.censored
    # wide and prior-dominated by construction
    assert p.sd_log10 > 0.4
    assert p.prior_sensitivity >= PRIOR_DOMINANCE_THRESHOLD
    assert p.identifiability_score < 0.2
    # one-sided: lower edge near x_max, heavy right tail above it
    q05, q95 = np.quantile(p.log10_ic50, [0.05, 0.95])
    assert 10 ** q05 > 0.5 * p.x_max_nm          # lower edge bounded near the top dose
    assert (q95 - q05) > 0.8                       # heavy spread
    # still caps the assessment at Tier D
    drug = censored.id.split(".")[1]
    a = harmonia.assess(ds, drug, n_mc=8, uq="bayes")
    assert a.tier == "D"


def test_prior_sensitivity_low_for_multisource(ds):
    """A multi-source channel is data-, not prior-, dominated."""
    p = posterior(ds, "dofetilide", "IKr", n_draws=4000, seed=0)
    assert p.prior_sensitivity < 0.3
    assert not p.prior_dominated


def test_hill_uncertainty_propagates(ds):
    """Gap #2: the Hill coefficient carries a posterior SD > 0 (it is no longer fixed)."""
    p = posterior(ds, "dofetilide", "IKr", n_draws=4000, seed=0)
    assert p.hill_sd > 0.0
    assert 0.4 < p.hill_mean < 2.0


def test_widened_prior_does_not_touch_learned_tau(ds):
    pr = resolve_prior(ds)
    wide = pr.widened(3.0)
    assert wide.s0 == pytest.approx(pr.s0 * 3.0)
    assert wide.tau_scale == pr.tau_scale     # tau is empirical-Bayes, not widened

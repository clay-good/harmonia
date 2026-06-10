"""Tests for the v0.2 variance-based (Sobol) sensitivity (spec v0.2 sec 5, sec 9)."""
import numpy as np

from harmonia.simulate import flip_sensitivity, SobolSensitivity, FlipSensitivity


def test_oat_is_still_the_default(ds):
    s = flip_sensitivity(ds, "verapamil", n_mc=20)
    assert isinstance(s, FlipSensitivity)


def test_sobol_returns_indices(ds):
    s = flip_sensitivity(ds, "verapamil", method="sobol", n_mc=32, seed=0)
    assert isinstance(s, SobolSensitivity)
    assert s.channels
    for c in s.channels:
        assert 0.0 <= c.first_order <= 1.0
        assert 0.0 <= c.total_effect <= 1.0
        assert c.interaction_load == c.total_effect - c.first_order


def test_sobol_total_effect_dominates_per_channel(ds):
    """sec 9 Sobol consistency: total-effect >= first-order for every channel (the
    total effect includes the channel's interactions). With the Janon first-order
    estimator this holds at finite N up to estimator noise; a small tolerance absorbs
    the residual Monte-Carlo error."""
    s = flip_sensitivity(ds, "verapamil", method="sobol", n_mc=96, seed=0)
    for c in s.channels:
        assert c.total_effect >= c.first_order - 0.15, (
            f"{c.channel}: S_Ti={c.total_effect:.2f} < S_i={c.first_order:.2f}")


def test_sobol_reports_standard_errors(ds):
    s = flip_sensitivity(ds, "verapamil", method="sobol", n_mc=32, seed=0)
    for c in s.channels:
        assert c.total_effect_se >= 0.0
        assert not np.isnan(c.total_effect_se)


def test_sobol_dominant_is_total_effect(ds):
    s = flip_sensitivity(ds, "verapamil", method="sobol", n_mc=48, seed=2)
    assert s.dominant_channel == s.channels[0].channel
    # channels are sorted by total-effect, descending
    totals = [c.total_effect for c in s.channels]
    assert totals == sorted(totals, reverse=True)


def test_sobol_deterministic(ds):
    a = flip_sensitivity(ds, "verapamil", method="sobol", n_mc=32, seed=5)
    b = flip_sensitivity(ds, "verapamil", method="sobol", n_mc=32, seed=5)
    assert [c.total_effect for c in a.channels] == [c.total_effect for c in b.channels]


def test_sobol_bayes_includes_censored(ds):
    """Under uq=bayes a censored channel is included (one-sided), and reported."""
    s = flip_sensitivity(ds, "ranolazine", method="sobol", uq="bayes", n_mc=48, seed=0)
    assert s.uq == "bayes"
    assert s.censored_channels  # ranolazine ICaL is sub-60% block

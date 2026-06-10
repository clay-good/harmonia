"""Tests for the uq="bayes" drop-in to assess() (spec v0.2 sec 4, sec 9, sec 10)."""
import numpy as np

from harmonia.simulate import assess


def test_moments_is_the_default(ds):
    a = assess(ds, "dofetilide", n_mc=16)
    assert a.uq == "moments"
    assert np.isnan(a.reproducibility_flip_frequency)
    assert a.censored_channels == []


def test_bayes_path_runs_and_is_labeled(ds):
    a = assess(ds, "dofetilide", n_mc=24, uq="bayes")
    assert a.uq == "bayes"
    assert a.qnet_distribution.shape == (24,)
    assert 0.0 <= a.classification_flip_frequency <= 1.0
    assert 0.0 <= a.reproducibility_flip_frequency <= 1.0
    assert abs(sum(a.classification_distribution.values()) - 1.0) < 1e-9
    assert "PROHIBITED" in a.clinical_use


def test_bayes_is_deterministic(ds):
    a = assess(ds, "verapamil", n_mc=20, uq="bayes", seed=7)
    b = assess(ds, "verapamil", n_mc=20, uq="bayes", seed=7)
    assert np.array_equal(a.qnet_distribution, b.qnet_distribution)
    assert a.reproducibility_flip_frequency == b.reproducibility_flip_frequency


def test_bayes_includes_censored_channel_not_excluded(ds):
    """Under bayes, a sub-60% channel CONTRIBUTES a one-sided posterior (it is no
    longer dropped from the simulation) but still caps the tier at D (sec 2.3, sec 6)."""
    a = assess(ds, "ranolazine", n_mc=16, uq="bayes")
    assert a.tier == "D"
    assert a.excluded_channels == []                  # not excluded under bayes
    assert any("ICaL" in c for c in a.censored_channels)
    assert "ICaL" in a.channels_used                  # it now participates
    assert "ICaL" in a.prior_dominated_channels


def test_reproducibility_flip_at_least_headline(ds):
    """The new-lab predictive adds between-lab spread, so it should flip at least as
    often as the true-value posterior in aggregate (sec 2.4). Checked loosely."""
    a = assess(ds, "dofetilide", n_mc=120, uq="bayes", seed=3)
    assert a.reproducibility_flip_frequency >= a.classification_flip_frequency - 0.05


def test_invalid_uq_raises(ds):
    import pytest
    with pytest.raises(ValueError):
        assess(ds, "dofetilide", n_mc=4, uq="nonsense")

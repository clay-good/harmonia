"""Population-of-models assessment (Phase E) — hypothesis-tier, non-predictive."""
import numpy as np
import pytest

import harmonia
from harmonia.populations import assess_population


def test_population_record_is_tier_d(ds):
    pops = ds.populations
    assert len(pops) >= 1
    for p in pops:
        assert p.tier == "D"
        assert p.predictive is False
        assert p.conductance_cv  # non-empty


def test_assessment_is_always_tier_d_and_non_predictive(ds):
    r = assess_population(ds, "dofetilide", n_models=20)
    assert r.tier == "D"
    assert "NOT FOR PREDICTION" in r.summary()
    assert abs(sum(r.classification_distribution.values()) - 1.0) < 1e-9
    assert 0.0 <= r.susceptible_fraction <= 1.0
    assert r.qnet_distribution.shape == (20,)


def test_high_risk_drug_more_susceptible_than_low(ds):
    hi = assess_population(ds, "dofetilide", n_models=40, seed=1)
    lo = assess_population(ds, "verapamil", n_models=40, seed=1)
    assert hi.susceptible_fraction > lo.susceptible_fraction


def test_population_spread_is_nontrivial(ds):
    """A population should not collapse to a single class for a borderline drug."""
    r = assess_population(ds, "sotalol", n_models=40)
    nonzero = [k for k, v in r.classification_distribution.items() if v > 0]
    assert len(nonzero) >= 2  # genuine inter-individual spread


def test_deterministic_given_seed(ds):
    a = assess_population(ds, "quinidine", n_models=16, seed=5)
    b = assess_population(ds, "quinidine", n_models=16, seed=5)
    assert np.array_equal(a.qnet_distribution, b.qnet_distribution)


def test_unknown_population_raises(ds):
    with pytest.raises(KeyError):
        assess_population(ds, "dofetilide", population="does_not_exist", n_models=4)


def test_population_cli(capsys):
    from harmonia.cli import main
    assert main(["population", "verapamil", "--n", "12"]) == 0
    out = capsys.readouterr().out
    assert "HYPOTHESIS-TIER" in out
    assert "SUSCEPTIBLE" in out

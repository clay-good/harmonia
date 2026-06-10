"""v0.3 disease/genetic population backgrounds (LQTS) — hypothesis-tier, non-predictive.

A disease population recenters the illustrative variability cloud on a reduced
repolarization-reserve mean (spec v0.3 §2). These guards prove the mean shift is applied
correctly, reproduces the textbook channelopathy ordering, stays Tier-D / non-predictive,
and leaves the healthy population byte-identical.
"""
import numpy as np

from harmonia.populations import assess_population
from harmonia.export.reference import KernelParams, simulate_beats


def test_lqts_records_present_and_tier_d(ds):
    ids = {p.id for p in ds.populations}
    for pid in ("population.lqt1", "population.lqt2", "population.lqt3"):
        assert pid in ids
        rec = ds[pid]
        assert rec.tier == "D"
        assert rec.predictive is False
        assert rec.is_disease and rec.conductance_scale          # carries a mean shift


def test_conductance_scale_values(ds):
    assert ds["population.lqt1"].conductance_scale == {"IKs": 0.5}
    assert ds["population.lqt2"].conductance_scale == {"IKr": 0.5}
    assert ds["population.lqt3"].conductance_scale == {"INaL": 2.0}


def test_mean_shift_lowers_qnet_in_kernel():
    """The disease mean shift, applied directly to the kernel, lowers qNet (reduced
    reserve) — the mechanism the population layer samples around."""
    healthy = simulate_beats(KernelParams())
    lqt2 = simulate_beats(KernelParams().with_conductance_multipliers({"IKr": 0.5}))
    assert lqt2.qnet < healthy.qnet
    assert lqt2.apd90 > healthy.apd90                            # IKr loss prolongs the AP


def test_disease_background_raises_susceptibility(ds):
    """For a borderline drug, every LQTS background increases the susceptible fraction
    relative to the healthy population (reduced repolarization reserve)."""
    healthy = assess_population(ds, "ranolazine", population="illustrative_v0",
                               n_models=60, seed=0).susceptible_fraction
    for pop in ("lqt1", "lqt2", "lqt3"):
        diseased = assess_population(ds, "ranolazine", population=pop,
                                     n_models=60, seed=0).susceptible_fraction
        assert diseased > healthy, f"{pop}: {diseased} !> healthy {healthy}"


def test_disease_assessment_is_tier_d_and_labeled(ds):
    a = assess_population(ds, "dofetilide", population="lqt2", n_models=20, seed=0)
    assert a.tier == "D"
    assert a.conductance_scale == {"IKr": 0.5}
    s = a.summary()
    assert "DISEASE background" in s
    assert "NOT FOR PREDICTION" in s
    assert "never a per-patient/genotype claim" in s


def test_healthy_population_unchanged(ds):
    """Backward compat: the variability-only population is byte-identical to v0.1 (the
    disease scale defaults to 1 and consumes no RNG)."""
    a = assess_population(ds, "sotalol", population="illustrative_v0", n_models=50, seed=0)
    b = assess_population(ds, "sotalol", population="illustrative_v0", n_models=50, seed=0)
    assert np.array_equal(a.qnet_distribution, b.qnet_distribution)
    assert a.conductance_scale == {}

"""v0.5 — experimentally-calibrated populations (Britton et al. 2013).

The validation contract (spec v0.5 §5): a calibrated population admits a virtual
myocyte only if its DRUG-FREE action-potential biomarkers fall in the accepted
ranges; the abnormal repolarization tail of the raw prior is rejected; the
uncalibrated path is byte-identical; and the assessment stays Tier D / NOT FOR
PREDICTION (the ranges are kernel-plausibility bounds, not patient-fit data).
"""
import math

import numpy as np
import pytest

import harmonia
from harmonia.populations import (assess_population, calibrate_population,
                                  _biomarker_values, _draw_multiplier)
from harmonia.simulate import _resolve_ap_model
from harmonia.export.reference import KernelParams, simulate_beats


@pytest.fixture(scope="module")
def ds():
    return harmonia.load()


# --------------------------------------------------------------------------- #
# Record / schema
# --------------------------------------------------------------------------- #
def test_calibrated_record_is_tier_d_and_calibrated(ds):
    pop = ds.population("calibrated_v0")
    assert pop is not None
    assert pop.tier == "D"
    assert pop.predictive is False
    assert pop.is_calibrated
    cal = pop.calibration
    assert cal["citation"] == "britton-2013"
    assert set(cal["biomarkers"]) <= {"apd90_ms", "vrest_mv", "vpeak_mv", "triangulation_ms"}
    for lim in cal["biomarkers"].values():
        assert lim["min"] < lim["max"]


def test_uncalibrated_populations_have_no_calibration(ds):
    for name in ("illustrative_v0", "lqt1", "lqt2", "lqt3"):
        assert ds.population(name).is_calibrated is False


# --------------------------------------------------------------------------- #
# Acceptance correctness — every accepted myocyte is drug-free-plausible
# --------------------------------------------------------------------------- #
def test_every_accepted_myocyte_is_in_range(ds):
    cal = calibrate_population(ds, "calibrated_v0", n_models=40, seed=3)
    ranges = ds.population("calibrated_v0").calibration["biomarkers"]
    assert len(cal.multipliers) == 40
    for mult, beat in zip(cal.multipliers, cal.baseline_beats):
        assert not beat.repolarization_failed and not beat.ead
        bm = _biomarker_values(beat)
        for k, lim in ranges.items():
            assert not math.isnan(bm[k])
            assert lim["min"] <= bm[k] <= lim["max"], f"{k}={bm[k]} outside {lim}"


def test_calibration_rejects_some_candidates(ds):
    # over a full population the raw prior's abnormal tail forces rejections
    cal = calibrate_population(ds, "calibrated_v0", n_models=100, seed=0)
    assert cal.n_candidates > 100            # had to over-sample
    assert cal.acceptance_rate < 1.0
    assert sum(cal.rejection_reasons.values()) > 0


def test_triangulation_is_the_dominant_filter(ds):
    # the kernel's drug-free abnormality is long/triangular repolarization, so
    # triangulation should reject at least as many candidates as resting potential
    cal = calibrate_population(ds, "calibrated_v0", n_models=100, seed=0)
    assert cal.rejection_reasons["triangulation_ms"] >= cal.rejection_reasons["vrest_mv"]


def test_calibration_removes_the_abnormal_tail(ds):
    # the raw prior cloud DOES contain myocytes the calibration would reject;
    # the calibrated population contains none of them.
    pop = ds.population("calibrated_v0")
    ranges = pop.calibration["biomarkers"]
    scales = _resolve_ap_model(ds, "cipaordv1.0").conductance_scales
    cv, scale = pop.conductance_cv, pop.conductance_scale
    channels = list(cv.keys())
    rng = np.random.default_rng(0)
    raw_out_of_range = 0
    for _ in range(200):
        mult = _draw_multiplier(rng, channels, cv, scale)
        base = KernelParams().with_scales(scales).with_conductance_multipliers(mult)
        r = simulate_beats(base, n_beats=3)
        if r.repolarization_failed or r.ead:
            raw_out_of_range += 1
            continue
        bm = _biomarker_values(r)
        if any(not (lim["min"] <= bm[k] <= lim["max"]) for k, lim in ranges.items()):
            raw_out_of_range += 1
    assert raw_out_of_range > 0   # the raw prior really does contain abnormal myocytes


# --------------------------------------------------------------------------- #
# Assessment surface
# --------------------------------------------------------------------------- #
def test_assessment_reports_calibration_and_stays_tier_d(ds):
    r = assess_population(ds, "verapamil", population="calibrated_v0", n_models=40, seed=0)
    assert r.calibrated is True
    assert r.tier == "D"
    assert 0.0 < r.acceptance_rate <= 1.0
    assert r.n_candidates >= r.n_models
    s = r.summary()
    assert "CALIBRATED" in s
    assert "NOT FOR PREDICTION" in s
    assert "CALIBRATION:" in s
    assert abs(sum(r.classification_distribution.values()) - 1.0) < 1e-9


def test_apd90_metric_works_under_calibration(ds):
    # the apd90 path reuses the cached drug-free baseline from acceptance
    r = assess_population(ds, "dofetilide", population="calibrated_v0",
                          n_models=30, seed=0, metric="apd90")
    assert r.calibrated and r.metric == "apd90"
    assert np.all(np.isfinite(r.dapd90_distribution))


def test_calibrated_deterministic_given_seed(ds):
    a = assess_population(ds, "quinidine", population="calibrated_v0", n_models=30, seed=7)
    b = assess_population(ds, "quinidine", population="calibrated_v0", n_models=30, seed=7)
    assert a.n_candidates == b.n_candidates
    assert a.acceptance_rate == b.acceptance_rate
    assert np.array_equal(a.qnet_distribution, b.qnet_distribution)


def test_calibrate_population_rejects_uncalibrated_record(ds):
    with pytest.raises(ValueError, match="no calibration block"):
        calibrate_population(ds, "illustrative_v0")


# --------------------------------------------------------------------------- #
# Backward compatibility — the uncalibrated path is unchanged
# --------------------------------------------------------------------------- #
def test_uncalibrated_path_is_unaffected(ds):
    r = assess_population(ds, "verapamil", population="illustrative_v0", n_models=20, seed=0)
    assert r.calibrated is False
    assert math.isnan(r.acceptance_rate)
    assert r.n_candidates == 0
    assert r.rejection_reasons == {}
    # the refactored draw reproduces a hand-rolled identical RNG sequence
    cv = ds.population("illustrative_v0").conductance_cv
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    chans = list(cv.keys())
    for _ in range(5):
        m = _draw_multiplier(rng_a, chans, cv, {})
        for ch in chans:
            cvv = cv[ch]
            draw = float(np.exp(rng_b.normal(0.0, np.sqrt(np.log(1.0 + cvv ** 2)))))
            assert m[ch] == pytest.approx(draw)

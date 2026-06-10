"""v0.7 — Monte-Carlo confidence intervals on the flip frequency.

The flip frequency is a binomial proportion estimated over n_mc draws, so it has
its own sampling error. v0.7 reports a Wilson 95% CI for every flip frequency (and
the population susceptible fraction). These tests pin the interval math, its
behaviour at the extremes and as n grows, that it brackets the reported point, and
that adding it changed no previously-reported number (non-drift).
"""
import math

import pytest

import harmonia
from harmonia.simulate import wilson_interval, flip_ci, Z95


# --------------------------------------------------------------------------- #
# the interval math
# --------------------------------------------------------------------------- #
def _wilson_textbook(k, n, z=Z95):
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


@pytest.mark.parametrize("k,n", [(0, 200), (1, 200), (73, 200), (200, 200), (5, 50), (25, 50)])
def test_wilson_matches_textbook(k, n):
    lo, hi = wilson_interval(k, n)
    elo, ehi = _wilson_textbook(k, n)
    assert lo == pytest.approx(elo, abs=1e-12)
    assert hi == pytest.approx(ehi, abs=1e-12)


def test_wilson_known_value():
    # Wilson 95% CI for 36/100 is approximately [0.272, 0.459] (standard reference).
    lo, hi = wilson_interval(36, 100)
    assert lo == pytest.approx(0.272, abs=2e-3)
    assert hi == pytest.approx(0.459, abs=2e-3)


@pytest.mark.parametrize("k,n", [(0, 200), (1, 200), (100, 200), (199, 200), (200, 200)])
def test_wilson_bounded_and_brackets_point(k, n):
    lo, hi = wilson_interval(k, n)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo <= k / n <= hi


def test_wilson_nondegenerate_at_extremes():
    # Unlike the Wald approximation (which gives a zero-width interval at k=0/k=n),
    # Wilson reports an honest one-sided bound where flip frequencies actually live.
    lo0, hi0 = wilson_interval(0, 200)
    assert lo0 == 0.0 and hi0 > 0.0
    lon, hin = wilson_interval(200, 200)
    assert hin == 1.0 and lon < 1.0


def test_wilson_no_sample_is_nan():
    lo, hi = wilson_interval(0, 0)
    assert math.isnan(lo) and math.isnan(hi)
    lo, hi = flip_ci(0.0, 0)
    assert math.isnan(lo) and math.isnan(hi)
    lo, hi = flip_ci(float("nan"), 200)
    assert math.isnan(lo) and math.isnan(hi)


def test_interval_narrows_as_n_grows():
    # half-width shrinks ~ 1/sqrt(n): much narrower at 2000 than at 50 for the same p.
    def hw(n):
        lo, hi = wilson_interval(round(0.4 * n), n)
        return (hi - lo) / 2
    assert hw(2000) < hw(200) < hw(50)
    # ~1/sqrt(n) scaling: 10x the draws => ~sqrt(10)~3.16x narrower (loose bounds)
    ratio = hw(50) / hw(5000)
    assert 7.0 < ratio < 14.0


def test_flip_ci_recovers_count():
    # freq = k/n exactly for integer k, so flip_ci must equal wilson_interval(k, n).
    assert flip_ci(73 / 200, 200) == wilson_interval(73, 200)
    assert flip_ci(0.5, 50) == wilson_interval(25, 50)


# --------------------------------------------------------------------------- #
# integration: the CI is wired into every reported surface and brackets the point
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ds():
    return harmonia.load()


def test_assess_reports_ci_bracketing_point(ds):
    r = harmonia.assess(ds, "verapamil", n_mc=200, seed=0)
    lo, hi = r.flip_ci
    assert 0.0 <= lo <= r.classification_flip_frequency <= hi <= 1.0
    assert "95% CI" in r.summary()


def test_combination_reports_ci(ds):
    c = harmonia.assess_combination(ds, ["terfenadine", "ondansetron"], n_mc=100, seed=0)
    lo, hi = c.flip_ci
    assert lo <= c.classification_flip_frequency <= hi
    assert "95% CI" in c.summary()


def test_flip_sensitivity_reports_ci(ds):
    fs = harmonia.flip_sensitivity(ds, "verapamil", n_mc=100, seed=0)
    lo, hi = fs.all_vary_flip_ci
    assert lo <= fs.all_vary_flip_frequency <= hi
    assert "95% CI" in fs.summary()


def test_population_reports_susceptible_ci(ds):
    p = harmonia.assess_population(ds, "sotalol", n_models=60, seed=0)
    lo, hi = p.susceptible_fraction_ci
    assert lo <= p.susceptible_fraction <= hi
    assert "95% CI" in p.summary()


def test_bayes_reports_both_cis(ds):
    r = harmonia.assess(ds, "dofetilide", uq="bayes", n_mc=100, seed=0)
    lo, hi = r.flip_ci
    assert lo <= r.classification_flip_frequency <= hi
    rlo, rhi = r.reproducibility_flip_ci
    assert rlo <= r.reproducibility_flip_frequency <= rhi


# --------------------------------------------------------------------------- #
# non-drift: the CI is purely additive — no previously-reported number moved
# --------------------------------------------------------------------------- #
def test_n_mc_zero_has_no_interval(ds):
    # The point-estimate-only path (used by performance scoring) does no sampling,
    # so it reports (nan, nan) and the summary falls back to the bare percentage.
    r = harmonia.assess(ds, "dofetilide", n_mc=0)
    assert math.isnan(r.flip_ci[0]) and math.isnan(r.flip_ci[1])


def test_flip_frequency_value_unchanged(ds):
    # Adding the CI must not perturb the flip frequency itself (common random numbers).
    r = harmonia.assess(ds, "verapamil", n_mc=200, seed=0)
    # 36.5% over 200 draws == 73 flips; the recovered count must be exact.
    assert round(r.classification_flip_frequency * 200) == 73
    assert r.flip_ci == wilson_interval(73, 200)

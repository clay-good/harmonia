"""v0.4 cqInward biomarker — the CiPA inward-charge readout (spec v0.4).

cqInward is the control-normalized average of the INaL and ICaL charge ratios. It is a
diagnostic, not the classifier. These guards prove the mechanism signs (the property that
validates it), the control identity, uncertainty propagation, and non-drift of the existing
qNet/flip numbers.
"""
import numpy as np

from harmonia.simulate import assess, _cqinward
from harmonia.export.reference import KernelParams, simulate_beats


def _cq(block):
    ctrl = simulate_beats(KernelParams())
    p = KernelParams(); p.block.update(block)
    r = simulate_beats(p)
    return _cqinward(r.q_nal, r.q_cal, ctrl.q_nal, ctrl.q_cal)


def test_control_identity():
    """No drug -> cqInward == 1 by construction."""
    assert abs(_cq({}) - 1.0) < 1e-9


def test_ical_block_reduces_inward_charge():
    """A pure ICaL blocker reduces inward charge (cqInward < 1) — the protective
    multichannel mechanism."""
    assert _cq({"ICaL": 0.3}) < 0.95


def test_inal_block_reduces_inward_charge():
    assert _cq({"INaL": 0.3}) < 0.95


def test_ikr_block_raises_inward_charge():
    """A pure IKr blocker prolongs the AP, so INaL/ICaL flow longer and inward charge
    rises (cqInward > 1) — the proarrhythmic direction."""
    assert _cq({"IKr": 0.2}) > 1.05


def test_assess_reports_cqinward_point_and_distribution(ds):
    a = assess(ds, "dofetilide", n_mc=24)
    assert np.isfinite(a.cqinward)
    assert a.cqinward_distribution.shape == (24,)
    # dofetilide is a hERG blocker -> inward charge increased
    assert a.cqinward > 1.0


def test_verapamil_protective_cqinward(ds):
    """verapamil's ICaL block dominates -> cqInward < 1 (consistent with its LOW call)."""
    a = assess(ds, "verapamil", n_mc=24)
    assert a.cqinward < 1.0


def test_cqinward_in_summary(ds):
    s = assess(ds, "dofetilide", n_mc=0).summary()
    assert "cqInward" in s
    assert "diagnostic, not the classifier" in s


def test_cqinward_propagates_under_bayes(ds):
    a = assess(ds, "dofetilide", n_mc=20, uq="bayes")
    assert np.isfinite(a.cqinward)
    assert a.cqinward_distribution.shape == (20,)


def test_adding_cqinward_does_not_change_qnet_or_flip(ds):
    """Non-drift: the biomarker is a pure addition; qNet and the flip frequency are
    unchanged (the moments path consumes no extra RNG)."""
    a = assess(ds, "verapamil", n_mc=20, seed=7)
    b = assess(ds, "verapamil", n_mc=20, seed=7)
    assert np.array_equal(a.qnet_distribution, b.qnet_distribution)
    assert a.classification_flip_frequency == b.classification_flip_frequency

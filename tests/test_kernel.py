"""The reference AP kernel: physiologic baseline, monotone pharmacology, EADs."""
import numpy as np

from harmonia.export.reference import (KernelParams, simulate_beats,
                                       hill_block_factor)


def test_baseline_ap_is_physiologic():
    r = simulate_beats(KernelParams())
    assert -95 < r.vrest < -78, r.vrest
    assert 20 < r.vpeak < 80, r.vpeak
    assert 180 < r.apd90 < 360, r.apd90
    assert not r.repolarization_failed
    assert not r.ead, "baseline AP must not flag an EAD"


def test_ikr_block_prolongs_apd_monotonically():
    base = simulate_beats(KernelParams()).apd90
    last = base
    for bf in [0.7, 0.5, 0.3, 0.15]:
        p = KernelParams()
        p.block["IKr"] = bf
        apd = simulate_beats(p).apd90
        assert apd > last, f"APD did not increase at IKr bf={bf}"
        last = apd


def test_ical_block_offsets_ikr_block():
    """The verapamil mechanism: balancing ICaL block shortens the AP again."""
    p_ikr = KernelParams(); p_ikr.block["IKr"] = 0.5
    p_bal = KernelParams(); p_bal.block["IKr"] = 0.5; p_bal.block["ICaL"] = 0.5
    assert simulate_beats(p_bal).apd90 < simulate_beats(p_ikr).apd90


def test_hill_block_factor():
    assert hill_block_factor(0, 100) == 1.0
    assert abs(hill_block_factor(100, 100, 1.0) - 0.5) < 1e-9
    assert hill_block_factor(1000, 100, 1.0) < 0.1


def test_determinism():
    a = simulate_beats(KernelParams()).apd90
    b = simulate_beats(KernelParams()).apd90
    assert a == b

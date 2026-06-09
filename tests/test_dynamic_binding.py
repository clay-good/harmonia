"""Dynamic hERG binding (Phase B): reduces to Hill at steady state; trapping
accumulates block; wired through assess."""
import harmonia
from harmonia.export.reference import (KernelParams, simulate_beats,
                                       HERGDynamic, hill_block_factor)


def test_dynamic_reduces_to_hill_at_steady_state():
    """Non-trapping Langmuir binding at steady state == static Hill (h=1)."""
    ic50, conc = 5.0, 8.0
    koff = 5e-5
    kon = koff / ic50
    rem = hill_block_factor(conc, ic50, 1.0)
    ps = KernelParams(); ps.block["IKr"] = rem
    static = simulate_beats(ps, n_beats=8)
    hd = HERGDynamic(conc_nm=conc, kon=kon, koff=koff, trapping=False)
    dyn = simulate_beats(KernelParams(), n_beats=16, herg=hd)
    # APD90 should agree within a few ms once both equilibrate
    assert abs(static.apd90 - dyn.apd90) < 12.0
    # bound fraction near the analytic steady state b = conc/(conc+ic50)
    assert abs(dyn.herg_bound_mean - conc / (conc + ic50)) < 0.05


def test_trapping_increases_block():
    conc, koff = 8.0, 5e-5
    kon = koff / 5.0
    no_trap = simulate_beats(KernelParams(), n_beats=12,
                             herg=HERGDynamic(conc, kon, koff, trapping=False))
    trap = simulate_beats(KernelParams(), n_beats=12,
                          herg=HERGDynamic(conc, kon, koff, trapping=True))
    assert trap.herg_bound_mean > no_trap.herg_bound_mean
    assert trap.apd90 > no_trap.apd90


def test_dynamic_binding_in_dataset(ds):
    b = ds["channel_block.dofetilide.ikr"]
    assert b.block_model == "dynamic_binding"
    assert b.dynamic_binding["trapping"] is True
    assert b.dynamic_binding["koff"] / b.dynamic_binding["kon"] > 0


def test_assess_herg_dynamic_path(ds):
    """For a trapped blocker, dynamic binding prolongs more than static."""
    static = harmonia.assess(ds, "dofetilide", n_mc=0, herg_dynamic=False)
    dynamic = harmonia.assess(ds, "dofetilide", n_mc=0, herg_dynamic=True)
    assert dynamic.herg_dynamic is True
    assert static.herg_dynamic is False
    assert dynamic.dapd90_pct > static.dapd90_pct

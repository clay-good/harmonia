"""The reference kernel: a SciPy integration of a reduced O'Hara-Rudy-lineage
human ventricular action potential, with Hill / dynamic-binding pharmacology
applied per current, plus the TdP-risk biomarkers (APD90, qNet, triangulation,
EAD detection).

HONESTY NOTE (spec.md §6). This is a *reduced* reference implementation of the
ORd lineage: seven named currents (INa, INaL, Ito, ICaL, IKr, IKs, IK1) with
Hodgkin-Huxley gating and an algebraic inward rectifier, fixed ionic
concentrations, and a single-cell paced protocol. It is structurally faithful
and numerically stable, and it reproduces the *qualitative* pharmacology the
CiPA paradigm rests on (hERG/IKr block prolongs the AP and lowers qNet; balancing
ICaL/INaL block shortens it again). It is NOT bit-exact to the published ORd
CellML and its qNet is in kernel-specific units — so AP-model records ship at
Tier C and qNet thresholds are calibrated to this kernel, not borrowed from the
regulatory pipeline. Cross-checking against the canonical CellML (Myokit/OpenCOR)
and earning Tier A is the Phase-F deliverable.

Everything here is deterministic; variability propagation lives in simulate.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp, trapezoid

# Currents that carry a Hill/dynamic block factor (IK1 is not a drug target here).
BLOCKABLE = ["INa", "INaL", "Ito", "ICaL", "IKr", "IKs"]

# Non-inactivating fraction of the L-type Ca inactivation gate (the window
# current). Sustains plateau inward Ca so that ICaL block shortens APD90 — the
# physiological mechanism behind the low risk of balanced blockers (verapamil,
# diltiazem). Kept in sync with model_spec._GATES.
ICAL_WINDOW_PEDESTAL = 0.18

# Reversal potentials (mV), from fixed physiological concentrations.
ENA = 70.0
EK = -87.0
EKS = -80.0
ECAL = 45.0   # effective L-type Ca driving reversal (phenomenological)

# State vector layout.
_STATE = ["V", "m", "h", "j", "mL", "hL", "a", "iF", "d", "f", "xr", "xs"]
_IDX = {name: i for i, name in enumerate(_STATE)}


@dataclass
class KernelParams:
    """Maximal conductances (µS/µF) and the per-current block factors in [0, 1]
    (1 = no block). ``g_scale`` rescales conductances to express AP-model
    variants (ORd vs CiPAORd vs ToR-ORd)."""
    gNa: float = 11.0
    gNaL: float = 0.060
    gto: float = 0.18
    gCaL: float = 0.140
    gKr: float = 0.300
    gKs: float = 0.045
    gK1: float = 0.34
    block: Dict[str, float] = field(default_factory=lambda: {c: 1.0 for c in BLOCKABLE})

    def with_scales(self, scales: Dict[str, float]) -> "KernelParams":
        p = KernelParams(self.gNa, self.gNaL, self.gto, self.gCaL, self.gKr,
                         self.gKs, self.gK1, dict(self.block))
        p.gKr *= scales.get("IKr", 1.0)
        p.gCaL *= scales.get("ICaL", 1.0)
        p.gNaL *= scales.get("INaL", 1.0)
        return p


def _sig(V, V0, k):
    return 1.0 / (1.0 + np.exp(-(V - V0) / k))


def gating_targets(V):
    """Steady states (x_inf) and time constants (tau, ms) for every gate."""
    m_inf = _sig(V, -39.0, 8.0)
    tau_m = 0.06 + 0.55 * np.exp(-((V + 40.0) / 20.0) ** 2)

    h_inf = _sig(V, -66.0, -7.0)
    tau_h = 1.0 + 18.0 * np.exp(-((V + 50.0) / 15.0) ** 2)

    j_inf = h_inf
    tau_j = 12.0 + 60.0 * np.exp(-((V + 60.0) / 20.0) ** 2)

    mL_inf = _sig(V, -43.0, 8.0)
    tau_mL = 0.1 + 0.5 * np.exp(-((V + 40.0) / 20.0) ** 2)

    hL_inf = _sig(V, -85.0, -7.0)
    tau_hL = 200.0

    a_inf = _sig(V, -2.0, 15.0)
    tau_a = 1.5 + 5.0 * np.exp(-((V + 30.0) / 30.0) ** 2)

    iF_inf = _sig(V, -40.0, -8.0)
    tau_iF = 18.0 + 35.0 * np.exp(-((V + 40.0) / 25.0) ** 2)

    d_inf = _sig(V, -6.0, 6.0)
    tau_d = 1.5 + 8.0 * np.exp(-((V + 10.0) / 25.0) ** 2)

    # ICaL inactivation with a small non-inactivating pedestal (the L-type
    # *window current*): without it ICaL is fully gone by mid-plateau and its
    # block cannot shorten APD90 — the physiological ICaL-block mechanism that
    # makes verapamil/diltiazem low-risk. The 0.08 pedestal sustains a small
    # inward Ca current through the plateau.
    f_inf = ICAL_WINDOW_PEDESTAL + (1.0 - ICAL_WINDOW_PEDESTAL) * _sig(V, -28.0, -7.0)
    tau_f = 25.0 + 120.0 * np.exp(-((V + 25.0) / 25.0) ** 2)

    xr_inf = _sig(V, -20.0, 8.0)
    tau_xr = 40.0 + 180.0 * np.exp(-((V + 10.0) / 30.0) ** 2)

    xs_inf = _sig(V, -18.0, 14.0)
    tau_xs = 80.0 + 350.0 * np.exp(-((V - 10.0) / 40.0) ** 2)

    return (
        np.array([0.0, m_inf, h_inf, j_inf, mL_inf, hL_inf, a_inf, iF_inf,
                  d_inf, f_inf, xr_inf, xs_inf]),
        np.array([1.0, tau_m, tau_h, tau_j, tau_mL, tau_hL, tau_a, tau_iF,
                  tau_d, tau_f, tau_xr, tau_xs]),
    )


def currents(V, y, p: KernelParams, b_ikr=None):
    """Return the seven ionic currents (µA/µF) at state ``y``.

    If ``b_ikr`` (the dynamically-bound hERG fraction) is given, IKr is scaled by
    ``(1 - b_ikr)`` instead of the static Hill block factor ``p.block['IKr']``."""
    m, h, j = y[1], y[2], y[3]
    mL, hL = y[4], y[5]
    a, iF = y[6], y[7]
    d, f = y[8], y[9]
    xr, xs = y[10], y[11]
    b = p.block

    ikr_factor = (1.0 - b_ikr) if b_ikr is not None else b["IKr"]

    INa = p.gNa * m ** 3 * h * j * (V - ENA) * b["INa"]
    INaL = p.gNaL * mL * hL * (V - ENA) * b["INaL"]
    Ito = p.gto * a * iF * (V - EK) * b["Ito"]
    ICaL = p.gCaL * d * f * (V - ECAL) * b["ICaL"]
    Rkr = 1.0 / (1.0 + np.exp((V + 70.0) / 25.0))   # inward rectification
    IKr = p.gKr * xr * Rkr * (V - EK) * ikr_factor
    IKs = p.gKs * xs ** 2 * (V - EKS) * b["IKs"]
    xK1 = 1.0 / (1.0 + np.exp((V + 100.0) / 12.0))
    IK1 = p.gK1 * xK1 * (V - EK)
    return INa, INaL, Ito, ICaL, IKr, IKs, IK1


def _stim(t, cl, duration=1.0, amp=-52.0):
    """Periodic depolarizing stimulus (µA/µF), inward (negative)."""
    phase = t % cl
    return amp if phase < duration else 0.0


@dataclass
class HERGDynamic:
    """Dynamic (time-dependent) hERG drug binding — the CiPA-style upgrade over a
    static Hill block (spec.md §4, Phase B).

    A first-order Langmuir binding ODE for the bound fraction ``b``:

        db/dt = kon * conc * (1 - b)  -  koff * open_factor * b

    where ``open_factor`` is the hERG activation gate ``xr`` when ``trapping`` is
    True (the drug can only unbind from the activated channel, so block
    accumulates use-dependently) and 1 otherwise. At steady state with
    open_factor=1 this reduces to a Hill block with IC50 = koff / kon and h = 1,
    so the dynamic model is a strict generalisation of the static one — verified
    in the tests.

    Units: ``kon`` in 1/(nM·ms), ``koff`` in 1/ms, ``conc`` in nM.
    """
    conc_nm: float
    kon: float
    koff: float
    trapping: bool = False

    @property
    def ic50_nm(self) -> float:
        return self.koff / self.kon

    def db_dt(self, b, xr):
        open_factor = xr if self.trapping else 1.0
        return self.kon * self.conc_nm * (1.0 - b) - self.koff * open_factor * b


def _rhs(t, y, p: KernelParams, cl: float, herg: "HERGDynamic" = None):
    V = y[0]
    if herg is None:
        INa, INaL, Ito, ICaL, IKr, IKs, IK1 = currents(V, y, p)
        Iion = INa + INaL + Ito + ICaL + IKr + IKs + IK1
        inf, tau = gating_targets(V)
        dy = (inf - y) / tau
        dy[0] = -(Iion + _stim(t, cl))
        return dy
    # dynamic-hERG path: y has one extra state, the bound fraction b (index 12)
    b = y[12]
    INa, INaL, Ito, ICaL, IKr, IKs, IK1 = currents(V, y[:12], p, b_ikr=b)
    Iion = INa + INaL + Ito + ICaL + IKr + IKs + IK1
    inf, tau = gating_targets(V)
    dy = np.empty_like(y)
    dy[:12] = (inf - y[:12]) / tau
    dy[0] = -(Iion + _stim(t, cl))
    dy[12] = herg.db_dt(b, y[10])   # y[10] == xr
    return dy


def _initial_state(herg: "HERGDynamic" = None):
    y0 = np.zeros(len(_STATE))
    y0[0] = -85.0
    inf, _ = gating_targets(-85.0)
    y0[1:] = inf[1:]
    if herg is not None:
        y0 = np.append(y0, 0.0)   # bound fraction starts at 0 (drug-naive)
    return y0


@dataclass
class BeatResult:
    t: np.ndarray
    V: np.ndarray
    currents: Dict[str, np.ndarray]
    apd90: float
    apd50: float
    vrest: float
    vpeak: float
    dvdt_max: float
    qnet: float
    triangulation: float
    ead: bool
    cl: float
    herg_bound_mean: float = float("nan")   # dynamic-binding diagnostics
    herg_bound_max: float = float("nan")

    @property
    def repolarization_failed(self) -> bool:
        return math.isnan(self.apd90) or self.vrest > -60.0


def simulate_beats(p: KernelParams, cl: float = 2000.0, n_beats: int = 3,
                   max_step: float = 2.0, dt_record: float = 1.0,
                   rtol: float = 1e-6, herg: "HERGDynamic" = None) -> BeatResult:
    """Pace to (approximate) steady state; analyse the final beat.

    With fixed ionic concentrations the AP reaches steady state within ~2-3
    beats, so ``n_beats=3`` is sufficient (verified: APD90 is identical from
    beat 3 onward). When ``herg`` is given, hERG block is computed by the dynamic
    binding ODE (one extra state) instead of the static Hill factor; with a slow
    off-rate the bound fraction needs more beats to equilibrate, so callers
    should raise ``n_beats``."""
    y = _initial_state(herg)
    t0 = 0.0
    # pre-pace all but the last beat
    for _ in range(max(n_beats - 1, 0)):
        sol = solve_ivp(_rhs, (t0, t0 + cl), y, args=(p, cl, herg), method="LSODA",
                        rtol=rtol, atol=1e-8, max_step=max_step)
        y = sol.y[:, -1]
        t0 += cl
    # record the final beat
    n = int(cl / dt_record) + 1
    t_eval = np.linspace(t0, t0 + cl, n)
    sol = solve_ivp(_rhs, (t0, t0 + cl), y, args=(p, cl, herg), method="LSODA",
                    rtol=rtol, atol=1e-8, max_step=max_step, t_eval=t_eval)
    t = sol.t - t0
    Y = sol.y
    V = Y[0]

    # vectorised current evaluation over the whole trace
    names = ["INa", "INaL", "Ito", "ICaL", "IKr", "IKs", "IK1"]
    b_ikr = Y[12] if herg is not None else None
    vals = currents(V, Y[:12] if herg is not None else Y, p, b_ikr=b_ikr)
    cur_arrays = {name: np.asarray(val) for name, val in zip(names, vals)}

    res = _analyse(t, V, cur_arrays, cl)
    if herg is not None:
        res.herg_bound_mean = float(np.mean(Y[12]))
        res.herg_bound_max = float(np.max(Y[12]))
    return res


def _analyse(t, V, cur, cl) -> BeatResult:
    vrest = float(V[0])
    vpeak = float(np.max(V))
    amp = vpeak - vrest
    dvdt_max = float(np.max(np.diff(V) / np.diff(t))) if len(t) > 1 else float("nan")

    def apd_at(frac):
        thresh = vpeak - frac * amp
        upi = int(np.argmax(V >= thresh)) if np.any(V >= thresh) else 0
        t_up = t[upi]
        below = np.where(V[upi:] <= thresh)[0]
        if amp < 20.0 or below.size == 0:
            return float("nan")
        return float(t[upi + below[0]] - t_up)

    apd90 = apd_at(0.90)
    apd50 = apd_at(0.50)
    triangulation = apd90 - apd50 if not math.isnan(apd90) and not math.isnan(apd50) else float("nan")

    # qNet = integral over the beat of the net of the six "CiPA" currents
    # (INaL + ICaL + IKr + IKs + IK1 + Ito), in (µA/µF)·s = µC/µF.
    #
    # CAVEAT (documented in this module's header and in the ap_model record):
    # this pump-free reduced kernel conserves charge over the paced cycle, so
    # this qNet is dominated by the fast-INa upstroke charge and is only weakly
    # sensitive to repolarization-current block. It is reported for transparency
    # but is NOT the classification metric in v0.1 — APD90 is. A qNet that
    # discriminates risk needs the pump/exchanger currents of the full ORd
    # (Phase B/C). See simulate.RiskMetric.
    inet = (cur["INaL"] + cur["ICaL"] + cur["IKr"] + cur["IKs"] + cur["IK1"] + cur["Ito"])
    qnet = float(trapezoid(inet, t) / 1000.0)

    ead = _detect_ead(t, V)

    return BeatResult(t=t, V=V, currents=cur, apd90=apd90, apd50=apd50,
                      vrest=vrest, vpeak=vpeak, dvdt_max=dvdt_max, qnet=qnet,
                      triangulation=triangulation, ead=ead, cl=cl)


def _detect_ead(t, V, rise_mv: float = 4.0, t_start: float = 150.0) -> bool:
    """An EAD is a secondary depolarization during *late* repolarization (phase
    2/3): after the early Ito notch / ICaL dome is long over, V reaches a local
    minimum and then rises again by at least ``rise_mv`` while still depolarized.

    The early spike-and-dome (phase 1->2, ~30-60 ms) is a normal AP feature, not
    an EAD, so the scan only starts at ``t_start`` ms — well past the dome. A
    healthy AP repolarizes monotonically thereafter; a genuine reversal in that
    window is the EAD signature."""
    mask = (t >= t_start) & (V > -60.0) & (V < 10.0)
    if np.count_nonzero(mask) < 3:
        return False
    vv = V[mask]
    running_min = vv[0]
    for v in vv:
        if v < running_min:
            running_min = v
        if (v - running_min) >= rise_mv:
            return True
    return False


# --------------------------------------------------------------------------- #
# Pharmacology helpers
# --------------------------------------------------------------------------- #
def hill_block_factor(conc_nm: float, ic50_nm: float, hill: float = 1.0) -> float:
    """Fraction of current REMAINING (1 = no block, 0 = full block)."""
    if conc_nm <= 0 or ic50_nm <= 0:
        return 1.0
    return 1.0 / (1.0 + (conc_nm / ic50_nm) ** hill)

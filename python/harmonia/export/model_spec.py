"""A single, renderer-agnostic description of the reduced ORd-lineage kernel,
plus a tiny expression AST that renders to Myokit infix and to content MathML.

Every model export (Myokit .mmt, CellML, SBML) is generated from THIS spec, so
the three artifacts cannot drift from each other, and the constants here are kept
identical to ``reference.py`` (the numeric oracle). A change to the kernel
equations is a change in one place.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import numpy as np

from . import reference as ref


# --------------------------------------------------------------------------- #
# Expression AST
# --------------------------------------------------------------------------- #
class Expr:
    def mmt(self) -> str:                      # Myokit infix
        raise NotImplementedError

    def mathml(self) -> str:                   # content MathML <apply>...</apply>
        raise NotImplementedError

    def eval(self, env: Dict[str, float]):     # numeric evaluation (round-trip)
        raise NotImplementedError

    # operator overloads for ergonomic construction
    def __add__(self, o): return Add(self, _e(o))
    def __radd__(self, o): return Add(_e(o), self)
    def __sub__(self, o): return Sub(self, _e(o))
    def __rsub__(self, o): return Sub(_e(o), self)
    def __mul__(self, o): return Mul(self, _e(o))
    def __rmul__(self, o): return Mul(_e(o), self)
    def __truediv__(self, o): return Div(self, _e(o))
    def __rtruediv__(self, o): return Div(_e(o), self)
    def __neg__(self): return Neg(self)
    def __pow__(self, o): return Pow(self, _e(o))


def _e(o: Union[Expr, float, int]) -> Expr:
    return o if isinstance(o, Expr) else Num(float(o))


@dataclass
class Num(Expr):
    v: float

    def mmt(self) -> str:
        if self.v == int(self.v):
            return str(int(self.v))
        return repr(self.v)

    def mathml(self) -> str:
        return f"<cn>{self.mmt()}</cn>"

    def eval(self, env):
        return self.v


@dataclass
class Var(Expr):
    name: str

    def mmt(self) -> str:
        return self.name

    def mathml(self) -> str:
        return f"<ci>{self.name}</ci>"

    def eval(self, env):
        return env[self.name]


def _bin(op_ml, *children):
    inner = "".join(c.mathml() for c in children)
    return f"<apply>{op_ml}{inner}</apply>"


@dataclass
class Add(Expr):
    a: Expr
    b: Expr
    def mmt(self): return f"({self.a.mmt()} + {self.b.mmt()})"
    def mathml(self): return _bin("<plus/>", self.a, self.b)
    def eval(self, env): return self.a.eval(env) + self.b.eval(env)


@dataclass
class Sub(Expr):
    a: Expr
    b: Expr
    def mmt(self): return f"({self.a.mmt()} - {self.b.mmt()})"
    def mathml(self): return _bin("<minus/>", self.a, self.b)
    def eval(self, env): return self.a.eval(env) - self.b.eval(env)


@dataclass
class Mul(Expr):
    a: Expr
    b: Expr
    def mmt(self): return f"({self.a.mmt()} * {self.b.mmt()})"
    def mathml(self): return _bin("<times/>", self.a, self.b)
    def eval(self, env): return self.a.eval(env) * self.b.eval(env)


@dataclass
class Div(Expr):
    a: Expr
    b: Expr
    def mmt(self): return f"({self.a.mmt()} / {self.b.mmt()})"
    def mathml(self): return _bin("<divide/>", self.a, self.b)
    def eval(self, env): return self.a.eval(env) / self.b.eval(env)


@dataclass
class Neg(Expr):
    a: Expr
    def mmt(self): return f"(-{self.a.mmt()})"
    def mathml(self): return f"<apply><minus/>{self.a.mathml()}</apply>"
    def eval(self, env): return -self.a.eval(env)


@dataclass
class Pow(Expr):
    a: Expr
    b: Expr
    def mmt(self): return f"({self.a.mmt()}^{self.b.mmt()})"
    def mathml(self): return _bin("<power/>", self.a, self.b)
    def eval(self, env): return self.a.eval(env) ** self.b.eval(env)


@dataclass
class ExpF(Expr):
    a: Expr
    def mmt(self): return f"exp({self.a.mmt()})"
    def mathml(self): return f"<apply><exp/>{self.a.mathml()}</apply>"
    def eval(self, env): return np.exp(self.a.eval(env))


@dataclass
class Floor(Expr):
    a: Expr
    def mmt(self): return f"floor({self.a.mmt()})"
    def mathml(self): return f"<apply><floor/>{self.a.mathml()}</apply>"
    def eval(self, env): return np.floor(self.a.eval(env))


@dataclass
class Lt(Expr):
    a: Expr
    b: Expr
    def mmt(self): return f"({self.a.mmt()} < {self.b.mmt()})"
    def mathml(self): return _bin("<lt/>", self.a, self.b)
    def eval(self, env): return self.a.eval(env) < self.b.eval(env)


@dataclass
class Piecewise(Expr):
    """value if cond else otherwise."""
    value: Expr
    cond: Expr
    otherwise: Expr
    def mmt(self): return f"piecewise({self.cond.mmt()}, {self.value.mmt()}, {self.otherwise.mmt()})"
    def eval(self, env):
        return np.where(self.cond.eval(env), self.value.eval(env), self.otherwise.eval(env))
    def mathml(self):
        return ("<piecewise>"
                f"<piece>{self.value.mathml()}{self.cond.mathml()}</piece>"
                f"<otherwise>{self.otherwise.mathml()}</otherwise>"
                "</piecewise>")


# convenience
def exp(x): return ExpF(_e(x))
def V(): return Var("V")
def sigmoid(V0, k): return Num(1.0) / (Num(1.0) + exp(-(V() - Num(V0)) / Num(k)))
def gaussian(Vp, s): return exp(-(((V() - Num(Vp)) / Num(s)) ** Num(2)))
def sigmoid_pedestal(V0, k, ped):
    return Num(ped) + Num(round(1.0 - ped, 9)) * sigmoid(V0, k)


# --------------------------------------------------------------------------- #
# Model description
# --------------------------------------------------------------------------- #
@dataclass
class Parameter:
    name: str
    value: float
    units: str
    label: str = ""


@dataclass
class StateVar:
    name: str
    init: float
    rate: Expr
    units: str


@dataclass
class Assignment:
    name: str
    expr: Expr
    units: str


@dataclass
class ModelSpec:
    name: str
    parameters: List[Parameter]
    states: List[StateVar]
    assignments: List[Assignment]   # ordered: each may reference earlier ones
    time_units: str = "ms"

    def parameter(self, name: str) -> Parameter:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(name)


# Gating definitions: (name, inf_expr, tau_expr) — kept identical to reference.py
_GATES: List[Tuple[str, Expr, Expr]] = [
    ("m",  sigmoid(-39, 8),  Num(0.06) + Num(0.55) * gaussian(-40, 20)),
    ("h",  sigmoid(-66, -7), Num(1.0) + Num(18.0) * gaussian(-50, 15)),
    ("j",  sigmoid(-66, -7), Num(12.0) + Num(60.0) * gaussian(-60, 20)),
    ("mL", sigmoid(-43, 8),  Num(0.1) + Num(0.5) * gaussian(-40, 20)),
    ("hL", sigmoid(-85, -7), Num(200.0)),
    ("a",  sigmoid(-2, 15),  Num(1.5) + Num(5.0) * gaussian(-30, 30)),
    ("iF", sigmoid(-40, -8), Num(18.0) + Num(35.0) * gaussian(-40, 25)),
    ("d",  sigmoid(-6, 6),   Num(1.5) + Num(8.0) * gaussian(-10, 25)),
    ("f",  sigmoid_pedestal(-28, -7, ref.ICAL_WINDOW_PEDESTAL),
           Num(25.0) + Num(120.0) * gaussian(-25, 25)),
    ("xr", sigmoid(-20, 8),  Num(40.0) + Num(180.0) * gaussian(-10, 30)),
    ("xs", sigmoid(-18, 14), Num(80.0) + Num(350.0) * gaussian(10, 40)),
]


def build_model_spec(name: str = "harmonia_ord_reduced",
                     conductance_scales: Dict[str, float] = None,
                     block: Dict[str, float] = None,
                     cl: float = 2000.0) -> ModelSpec:
    """Build the spec for one AP-model variant + drug block configuration."""
    scales = conductance_scales or {}
    blk = {c: 1.0 for c in ref.BLOCKABLE}
    if block:
        blk.update(block)

    base = ref.KernelParams().with_scales(scales)

    params: List[Parameter] = [
        Parameter("Cm", 1.0, "uF_per_cm2", "membrane capacitance"),
        Parameter("ENa", ref.ENA, "mV", "Na reversal potential"),
        Parameter("EK", ref.EK, "mV", "K reversal potential"),
        Parameter("EKs", ref.EKS, "mV", "IKs reversal potential"),
        Parameter("ECaL", ref.ECAL, "mV", "effective L-type Ca reversal"),
        Parameter("ENCX", ref.ENCX, "mV", "effective Na-Ca exchanger reversal"),
        Parameter("gNa", base.gNa, "mS_per_uF", "fast Na conductance"),
        Parameter("gNaL", base.gNaL, "mS_per_uF", "late Na conductance"),
        Parameter("gto", base.gto, "mS_per_uF", "transient outward conductance"),
        Parameter("gCaL", base.gCaL, "mS_per_uF", "L-type Ca conductance"),
        Parameter("gKr", base.gKr, "mS_per_uF", "rapid delayed rectifier conductance"),
        Parameter("gKs", base.gKs, "mS_per_uF", "slow delayed rectifier conductance"),
        Parameter("gK1", base.gK1, "mS_per_uF", "inward rectifier conductance"),
        Parameter("gNaCa", base.gNaCa, "mS_per_uF", "Na-Ca exchanger scale (excluded from qNet)"),
        Parameter("stim_amplitude", -52.0, "uA_per_uF", "stimulus amplitude"),
        Parameter("stim_duration", 1.0, "ms", "stimulus duration"),
        Parameter("stim_period", cl, "ms", "pacing cycle length"),
    ]
    for ch in ref.BLOCKABLE:
        params.append(Parameter(f"block_{ch}", blk[ch], "dimensionless",
                                f"fraction of {ch} remaining (1=no block, 0=full)"))

    # assignments (ordered)
    assigns: List[Assignment] = []
    for gname, inf, tau in _GATES:
        assigns.append(Assignment(f"{gname}_inf", inf, "dimensionless"))
        assigns.append(Assignment(f"{gname}_tau", tau, "ms"))

    Rkr = Num(1.0) / (Num(1.0) + exp((V() + Num(70.0)) / Num(25.0)))
    xK1 = Num(1.0) / (Num(1.0) + exp((V() + Num(100.0)) / Num(12.0)))
    wncx = Num(1.0) / (Num(1.0) + exp(-(V() + Num(30.0)) / Num(20.0)))
    assigns.append(Assignment("Rkr", Rkr, "dimensionless"))
    assigns.append(Assignment("xK1_inf", xK1, "dimensionless"))
    assigns.append(Assignment("wncx", wncx, "dimensionless"))

    g = Var
    INa = g("gNa") * g("m") ** Num(3) * g("h") * g("j") * (V() - g("ENa")) * g("block_INa")
    INaL = g("gNaL") * g("mL") * g("hL") * (V() - g("ENa")) * g("block_INaL")
    Ito = g("gto") * g("a") * g("iF") * (V() - g("EK")) * g("block_Ito")
    ICaL = g("gCaL") * g("d") * g("f") * (V() - g("ECaL")) * g("block_ICaL")
    IKr = g("gKr") * g("xr") * g("Rkr") * (V() - g("EK")) * g("block_IKr")
    IKs = g("gKs") * g("xs") ** Num(2) * (V() - g("EKs")) * g("block_IKs")
    IK1 = g("gK1") * g("xK1_inf") * (V() - g("EK"))
    INaCa = g("gNaCa") * g("wncx") * (V() - g("ENCX"))
    for nm, ex in [("INa", INa), ("INaL", INaL), ("Ito", Ito), ("ICaL", ICaL),
                   ("IKr", IKr), ("IKs", IKs), ("IK1", IK1), ("INaCa", INaCa)]:
        assigns.append(Assignment(nm, ex, "uA_per_uF"))

    Iion = (g("INa") + g("INaL") + g("Ito") + g("ICaL") + g("IKr") + g("IKs")
            + g("IK1") + g("INaCa"))
    assigns.append(Assignment("i_ion", Iion, "uA_per_uF"))

    # periodic stimulus via floor(): phase = t - floor(t/period)*period
    t = Var("time")
    phase = t - Floor(t / g("stim_period")) * g("stim_period")
    i_stim = Piecewise(g("stim_amplitude"), Lt(phase, g("stim_duration")), Num(0.0))
    assigns.append(Assignment("i_stim", i_stim, "uA_per_uF"))

    # states
    y0 = ref._initial_state()
    init = {nm: float(y0[i]) for i, nm in enumerate(ref._STATE)}
    states: List[StateVar] = [
        StateVar("V", init["V"], (-(g("i_ion") + g("i_stim")) / g("Cm")), "mV"),
    ]
    for gname, _, _ in _GATES:
        rate = (Var(f"{gname}_inf") - Var(gname)) / Var(f"{gname}_tau")
        states.append(StateVar(gname, init[gname], rate, "dimensionless"))

    return ModelSpec(name=name, parameters=params, states=states, assignments=assigns)


# --------------------------------------------------------------------------- #
# ODE round-trip: integrate the AST and compare to the reference kernel
# --------------------------------------------------------------------------- #
def simulate_spec(spec: ModelSpec, cl: float = 2000.0, n_beats: int = 3,
                  max_step: float = 2.0, dt_record: float = 1.0,
                  rtol: float = 1e-6) -> "ref.BeatResult":
    """Integrate a :class:`ModelSpec` by interpreting its expression AST, with the
    *same* solver settings as ``reference.simulate_beats``, and return a
    ``BeatResult`` analysed identically.

    This is the load-bearing half of the ODE round trip: the AST is what every
    model export (Myokit / CellML / SBML) is rendered from, so re-integrating it
    and matching the reference kernel proves the *exported equations* — not just
    the constants — agree with the numeric oracle. See ``registry.roundtrip_ode``.
    """
    from scipy.integrate import solve_ivp

    pvals = {p.name: p.value for p in spec.parameters}
    state_names = [s.name for s in spec.states]
    y0 = np.array([s.init for s in spec.states], dtype=float)

    def rhs(t, y):
        env = dict(pvals)
        env["time"] = t
        for nm, val in zip(state_names, y):
            env[nm] = val
        for a in spec.assignments:
            env[a.name] = a.expr.eval(env)
        return np.array([float(s.rate.eval(env)) for s in spec.states], dtype=float)

    y = y0
    t0 = 0.0
    for _ in range(max(n_beats - 1, 0)):
        sol = solve_ivp(rhs, (t0, t0 + cl), y, method="LSODA",
                        rtol=rtol, atol=1e-8, max_step=max_step)
        y = sol.y[:, -1]
        t0 += cl
    n = int(cl / dt_record) + 1
    t_eval = np.linspace(t0, t0 + cl, n)
    sol = solve_ivp(rhs, (t0, t0 + cl), y, method="LSODA",
                    rtol=rtol, atol=1e-8, max_step=max_step, t_eval=t_eval)
    t = sol.t - t0
    Y = sol.y

    # vectorised current evaluation over the recorded trace (for qNet etc.)
    env = dict(pvals)
    env["time"] = sol.t
    for i, nm in enumerate(state_names):
        env[nm] = Y[i]
    for a in spec.assignments:
        env[a.name] = a.expr.eval(env)
    cur = {name: np.asarray(env[name], dtype=float) for name in ref.CURRENT_NAMES}
    return ref._analyse(t, Y[0], cur, cl)

#!/usr/bin/env python3
"""Regenerate the README figures from the dataset + reference kernel.

    python docs/make_figures.py

Writes PNGs to docs/img/. Requires matplotlib (pip install harmonia[notebooks]).
Deterministic: the figures are a faithful projection of the dataset, not
decoration.
"""
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import harmonia
from harmonia.export.reference import (KernelParams, simulate_beats, HERGDynamic,
                                       hill_block_factor)
from harmonia.simulate import (assess, THRESH_LOW_PCT, THRESH_HIGH_PCT,
                               REFERENCE_EXPOSURE_MULTIPLE)
from harmonia.performance import score

IMG = pathlib.Path(__file__).resolve().parent / "img"
IMG.mkdir(parents=True, exist_ok=True)
BLUE, RED, GREY, GREEN = "#2c6fbb", "#c0392b", "#7f8c8d", "#27ae60"
ds = harmonia.load()


def fig_ap_traces():
    """Baseline vs progressive hERG (IKr) block — the core pharmacology."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for bf, color, label in [(1.0, GREY, "drug-free"),
                             (0.5, BLUE, "50% IKr block"),
                             (0.2, "#8e44ad", "80% IKr block"),
                             (0.08, RED, "92% IKr block")]:
        p = KernelParams(); p.block["IKr"] = bf
        r = simulate_beats(p)
        ax.plot(r.t, r.V, color=color, lw=1.8,
                label=f"{label}  (APD90={r.apd90:.0f} ms)")
    ax.set_xlim(0, 700)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("membrane potential (mV)")
    ax.set_title("Reduced ORd-lineage kernel: hERG block prolongs the action potential")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(IMG / "ap_traces.png", dpi=130)
    plt.close(fig)


def fig_flip_distribution():
    """The headline: ΔAPD90 distribution under IC50 variability for two drugs."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, drug in zip(axes, ["dofetilide", "verapamil"]):
        a = assess(ds, drug, ap_model="cipaordv1.0", n_mc=400)
        ax.hist(a.dapd90_distribution, bins=28, color=BLUE, alpha=0.85)
        ax.axvline(THRESH_LOW_PCT, color=GREEN, ls="--", lw=1.2)
        ax.axvline(THRESH_HIGH_PCT, color=RED, ls="--", lw=1.2)
        ax.axvline(a.dapd90_pct, color="black", lw=1.6)
        ax.set_title(f"{drug}  (tier {a.tier})\npoint={a.classification.upper()}, "
                     f"flip={a.classification_flip_frequency:.0%}", fontsize=10)
        ax.set_xlabel("ΔAPD90 (%) at 4× EFTPC")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Monte-Carlo draws")
    fig.suptitle("Input-variability → classification-flip: low<16% (green) | high≥33% (red)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(IMG / "flip_distribution.png", dpi=130)
    plt.close(fig)


def fig_training_set():
    """ΔAPD90% at 4× EFTPC for all 12 CiPA training drugs, colored by expert label."""
    order = {"high": 0, "intermediate": 1, "low": 2}
    cmap = {"high": RED, "intermediate": "#e69b00", "low": GREEN}
    refs = sorted(ds.drug_references, key=lambda r: (order[r.expert_risk_label], r.drug))
    names, vals, colors = [], [], []
    for ref in refs:
        a = assess(ds, ref.drug, ap_model="cipaordv1.0", n_mc=1)
        names.append(ref.drug); vals.append(a.dapd90_pct)
        colors.append(cmap[ref.expert_risk_label])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(names, vals, color=colors)
    ax.axvline(THRESH_LOW_PCT, color=GREEN, ls="--", lw=1)
    ax.axvline(THRESH_HIGH_PCT, color=RED, ls="--", lw=1)
    ax.set_xlabel("ΔAPD90 (%) at 4× EFTPC  (point estimate)")
    ax.set_title("CiPA training set — expert label = bar color; thresholds dashed")
    ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[k]) for k in ["high", "intermediate", "low"]]
    ax.legend(handles, ["high", "intermediate", "low"], fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(IMG / "training_set.png", dpi=130)
    plt.close(fig)


def fig_dynamic_binding():
    """Static Hill vs dynamic (trapped) hERG block for dofetilide — Phase B."""
    rec = ds["channel_block.dofetilide.ikr"]
    db = rec.dynamic_binding
    ic50 = db["koff"] / db["kon"]
    conc = 4.0 * ds.drug_reference("dofetilide").eftpc_nm
    rem = hill_block_factor(conc, ic50, 1.0)
    ps = KernelParams().with_scales({"IKr": 1.2, "INaL": 1.3}); ps.block["IKr"] = rem
    static = simulate_beats(ps, n_beats=10)
    hd = HERGDynamic(conc_nm=conc, kon=db["koff"] / ic50, koff=db["koff"], trapping=True)
    dyn = simulate_beats(KernelParams().with_scales({"IKr": 1.2, "INaL": 1.3}),
                         n_beats=12, herg=hd)
    base = simulate_beats(KernelParams().with_scales({"IKr": 1.2, "INaL": 1.3}))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(base.t, base.V, color=GREY, lw=1.6, label=f"drug-free (APD90={base.apd90:.0f})")
    ax.plot(static.t, static.V, color=BLUE, lw=1.8,
            label=f"static Hill block (APD90={static.apd90:.0f})")
    ax.plot(dyn.t, dyn.V, color=RED, lw=1.8,
            label=f"dynamic + trapping (APD90={dyn.apd90:.0f})")
    ax.set_xlim(0, 700); ax.set_xlabel("time (ms)"); ax.set_ylabel("Vm (mV)")
    ax.set_title("Dofetilide at 4× EFTPC: trapped dynamic hERG block prolongs more\n"
                 f"(bound fraction reaches {dyn.herg_bound_mean:.0%})", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(IMG / "dynamic_binding.png", dpi=130)
    plt.close(fig)


def fig_validation_set():
    """ΔAPD90% at 4× EFTPC for the 16 CiPA VALIDATION drugs, colored by expert label."""
    rep = score(ds, ap_model="cipaordv1.0", cipa_set="validation")
    cmap = {"high": RED, "intermediate": "#e69b00", "low": GREEN}
    names = [s.drug for s in rep.scores]
    vals = [s.dapd90_pct for s in rep.scores]
    colors = [cmap[s.expert] for s in rep.scores]
    fig, ax = plt.subplots(figsize=(8, 5.0))
    ax.barh(names, vals, color=colors)
    ax.axvline(THRESH_LOW_PCT, color=GREEN, ls="--", lw=1)
    ax.axvline(THRESH_HIGH_PCT, color=RED, ls="--", lw=1)
    ax.set_xlabel("ΔAPD90 (%) at 4× EFTPC  (point estimate)")
    ax.set_title(f"CiPA VALIDATION set (16 drugs) — accuracy {rep.n_correct}/{rep.n}, "
                 f"within-one {rep.adjacent_accuracy():.0%}\nexpert label = bar color")
    ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[k]) for k in ["high", "intermediate", "low"]]
    ax.legend(handles, ["high", "intermediate", "low"], fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(IMG / "validation_set.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_ap_traces()
    fig_flip_distribution()
    fig_training_set()
    fig_dynamic_binding()
    fig_validation_set()
    print(f"wrote figures to {IMG}")

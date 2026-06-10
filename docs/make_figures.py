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
from harmonia.simulate import (assess, assess_combination, THRESH_LOW_PCT,
                               THRESH_HIGH_PCT, QNET_THRESH_LOW, QNET_THRESH_HIGH,
                               RISK_LABELS)
from harmonia.populations import assess_population
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
    """The headline: qNet distribution under IC50 variability for two drugs."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, drug in zip(axes, ["dofetilide", "verapamil"]):
        a = assess(ds, drug, ap_model="cipaordv1.0", n_mc=400)  # metric=qnet (default)
        ax.hist(a.qnet_distribution, bins=28, color=BLUE, alpha=0.85)
        ax.axvline(QNET_THRESH_HIGH, color=RED, ls="--", lw=1.2)    # below -> high risk
        ax.axvline(QNET_THRESH_LOW, color=GREEN, ls="--", lw=1.2)   # above -> low risk
        ax.axvline(a.qnet, color="black", lw=1.6)
        ax.set_title(f"{drug}  (tier {a.tier})\npoint={a.classification.upper()}, "
                     f"flip={a.classification_flip_frequency:.0%}", fontsize=10)
        ax.set_xlabel("qNet (µC/µF) at 4× EFTPC")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Monte-Carlo draws")
    fig.suptitle("Input-variability → classification-flip (qNet): "
                 f"high<{QNET_THRESH_HIGH:g} (red) | low>{QNET_THRESH_LOW:g} (green); "
                 "lower qNet = higher risk", fontsize=10.5)
    fig.tight_layout(); fig.savefig(IMG / "flip_distribution.png", dpi=130)
    plt.close(fig)


def fig_qnet_cipa():
    """qNet at 4× EFTPC for ALL 28 CiPA compounds, colored by expert label — the
    Phase-C headline: clean separation, zero two-category errors."""
    cmap = {"high": RED, "intermediate": "#e69b00", "low": GREEN}
    rows = []
    for ref in ds.drug_references:
        a = assess(ds, ref.drug, ap_model="cipaordv1.0", n_mc=0)  # qnet
        rows.append((ref.expert_risk_label, ref.cipa_set, ref.drug, a.qnet))
    order = {"high": 0, "intermediate": 1, "low": 2}
    rows.sort(key=lambda r: (order[r[0]], r[3]))
    names = [f"{r[2]}{'*' if r[1]=='validation' else ''}" for r in rows]
    vals = [r[3] for r in rows]
    colors = [cmap[r[0]] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 8.0))
    ax.barh(names, vals, color=colors)
    ax.axvline(QNET_THRESH_HIGH, color=RED, ls="--", lw=1)
    ax.axvline(QNET_THRESH_LOW, color=GREEN, ls="--", lw=1)
    ax.set_xlabel("qNet (µC/µF) at 4× EFTPC   (lower = higher risk; * = validation drug)")
    ax.set_title("qNet across all 28 CiPA compounds — expert label = bar color\n"
                 "high-risk drugs left of the red line, low-risk right of the green line")
    ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[k]) for k in ["high", "intermediate", "low"]]
    ax.legend(handles, ["high", "intermediate", "low"], fontsize=8, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(IMG / "qnet_cipa.png", dpi=130)
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


def fig_combination():
    """Polypharmacy (Phase D): two intermediate drugs combine into high risk.
    qNet AP traces and the qNet bars for each single agent vs the combination."""
    pair = ["terfenadine", "ondansetron"]
    combo = assess_combination(ds, pair, n_mc=300)
    singles = {d: assess(ds, d, n_mc=0, exposure_nM=combo.exposures_nM[d]) for d in pair}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    # left: qNet bars, single agents vs combination, with thresholds
    labels = [pair[0], pair[1], "combination"]
    qnets = [singles[pair[0]].qnet, singles[pair[1]].qnet, combo.qnet]
    classes = [singles[pair[0]].classification, singles[pair[1]].classification, combo.classification]
    cmap = {"high": RED, "intermediate": "#e69b00", "low": GREEN}
    axL.bar(labels, qnets, color=[cmap[c] for c in classes])
    axL.axhline(QNET_THRESH_HIGH, color=RED, ls="--", lw=1)
    axL.axhline(QNET_THRESH_LOW, color=GREEN, ls="--", lw=1)
    axL.set_ylabel("qNet (µC/µF)   (lower = higher risk)")
    axL.set_title("Two intermediate drugs → HIGH combined\n"
                  f"interaction {combo.interaction_dapd90_pct:+.0f}% APD, "
                  f"flip {combo.classification_flip_frequency:.0%}", fontsize=10)
    for x, (q, c) in enumerate(zip(qnets, classes)):
        axL.text(x, q + 0.004, c.upper(), ha="center", fontsize=8)
    axL.spines[["top", "right"]].set_visible(False)

    # right: the combination qNet distribution under joint IC50 variability
    axR.hist(combo.qnet_distribution, bins=26, color=BLUE, alpha=0.85)
    axR.axvline(QNET_THRESH_HIGH, color=RED, ls="--", lw=1.2)
    axR.axvline(QNET_THRESH_LOW, color=GREEN, ls="--", lw=1.2)
    axR.axvline(combo.qnet, color="black", lw=1.6)
    axR.set_xlabel("combination qNet (µC/µF)")
    axR.set_ylabel("Monte-Carlo draws")
    axR.set_title("Joint input variability → classification flips", fontsize=10)
    axR.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Polypharmacy: independent block multiplies, uncertainty compounds",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(IMG / "combination.png", dpi=130)
    plt.close(fig)


def fig_population():
    """Population-of-models (Phase E, HYPOTHESIS-TIER): inter-individual conductance
    variability spreads a drug's risk across a population of virtual myocytes."""
    N = 80
    cmap = {"high": RED, "intermediate": "#e69b00", "low": GREEN}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    # left: population qNet distribution for a high vs a low drug
    for drug, color in [("dofetilide", RED), ("verapamil", GREEN)]:
        r = assess_population(ds, drug, n_models=N)
        axL.hist(r.qnet_distribution, bins=22, alpha=0.6, color=color,
                 label=f"{drug} (susceptible {r.susceptible_fraction:.0%})")
    axL.axvline(QNET_THRESH_HIGH, color=RED, ls="--", lw=1)
    axL.axvline(QNET_THRESH_LOW, color=GREEN, ls="--", lw=1)
    axL.set_xlabel("qNet across the population (µC/µF)")
    axL.set_ylabel("virtual myocytes")
    axL.set_title("Same drug, a population of hearts → a spread of risk", fontsize=10)
    axL.legend(fontsize=8, frameon=False)
    axL.spines[["top", "right"]].set_visible(False)

    # right: stacked class fractions per drug (incl. a single-cell 'miss', sotalol)
    drugs = ["dofetilide", "quinidine", "sotalol", "ondansetron", "verapamil", "diltiazem"]
    fracs = {lab: [] for lab in RISK_LABELS}
    for d in drugs:
        r = assess_population(ds, d, n_models=N)
        for lab in RISK_LABELS:
            fracs[lab].append(r.classification_distribution.get(lab, 0.0))
    left = np.zeros(len(drugs))
    for lab in ["high", "intermediate", "low"]:
        axR.barh(drugs, fracs[lab], left=left, color=cmap[lab], label=lab)
        left += np.array(fracs[lab])
    axR.set_xlabel("fraction of the population")
    axR.set_title("Class mix across the population (left=high risk)", fontsize=10)
    axR.invert_yaxis()
    axR.legend(fontsize=8, frameon=False, loc="lower right")
    axR.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Population-of-models — HYPOTHESIS-TIER, NOT FOR PREDICTION "
                 "(illustrative, uncalibrated)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(IMG / "population.png", dpi=130)
    plt.close(fig)


def fig_bayesian_uq():
    """v0.2: the IC50 posterior — reduction, censoring, and the two flip frequencies."""
    from harmonia.infer import infer_channel, resolve_prior, learn_tau_pop
    from harmonia.records import ChannelBlock
    prior = resolve_prior(ds)
    tau_pop = learn_tau_pop(ds.channel_blocks, prior)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

    # (1) well-identified channel: posterior reduces to the log-geomean
    b = next(x for x in ds.blocks_for("dofetilide") if x.channel == "IKr")
    p = infer_channel(b, prior, tau_pop, n_draws=8000, seed=0)
    gm = float(np.log10(np.exp(np.mean(np.log(b.source_ic50s_nm)))))
    ax = axes[0]
    ax.hist(p.log10_ic50, bins=40, color=BLUE, alpha=0.85)
    ax.axvline(gm, color=RED, lw=1.8, label=f"log-geomean (v0.1) = {gm:.2f}")
    ax.axvline(p.mean_log10, color="black", ls="--", lw=1.4,
               label=f"posterior mean = {p.mean_log10:.2f}")
    for sv in b.source_ic50s_nm:
        ax.axvline(np.log10(sv), color=GREY, lw=0.8, alpha=0.7)
    ax.set_title(f"dofetilide hERG — 3 labs agree\nreduces to v0.1 "
                 f"(prior_sens={p.prior_sensitivity:.2f})", fontsize=10)
    ax.set_xlabel("log10 IC50 (nM)"); ax.set_ylabel("posterior draws")
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # (2) censored channel: one-sided posterior, heavy right tail
    cens = next(x for x in ds.channel_blocks
                if isinstance(x, ChannelBlock) and not x.identifiable)
    pc = infer_channel(cens, prior, tau_pop, n_draws=8000, seed=0)
    ax = axes[1]
    ax.hist(pc.log10_ic50, bins=40, color="#8e44ad", alpha=0.85)
    ax.axvline(np.log10(pc.x_max_nm), color=RED, lw=1.8,
               label=f"top tested dose ≈ {pc.x_max_nm:.0f} nM")
    ax.set_title(f"{cens.drug} ICaL — {cens.assay_context.max_block_observed_percent:g}% block "
                 f"(unidentifiable)\none-sided posterior, prior-dominated "
                 f"(prior_sens={pc.prior_sensitivity:.2f})", fontsize=10)
    ax.set_xlabel("log10 IC50 (nM)")
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # (3) true-value vs new-lab predictive flip frequency
    ax = axes[2]
    drugs = ["dofetilide", "verapamil", "ranolazine"]
    true_f, repro_f = [], []
    for drug in drugs:
        a = assess(ds, drug, n_mc=300, uq="bayes", seed=0)
        true_f.append(a.classification_flip_frequency)
        repro_f.append(a.reproducibility_flip_frequency)
    x = np.arange(len(drugs)); w = 0.38
    ax.bar(x - w / 2, true_f, w, color=BLUE, label="true-value flip")
    ax.bar(x + w / 2, repro_f, w, color=GREEN, label="new-lab (reproducibility) flip")
    ax.set_xticks(x); ax.set_xticklabels(drugs, fontsize=9)
    ax.set_ylabel("classification-flip frequency")
    ax.set_title("Two honest flip frequencies (uq=bayes)\ntrue value vs a fresh replication",
                 fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("v0.2 — the IC50 spread is inferred under a declared prior, not transcribed "
                 "(reduction · censoring · reproducibility)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(IMG / "bayesian_uq.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_ap_traces()
    fig_flip_distribution()
    fig_qnet_cipa()        # qNet across all 28 compounds (supersedes the per-set bar charts)
    fig_dynamic_binding()
    fig_combination()
    fig_population()
    fig_bayesian_uq()      # v0.2 Bayesian dose-response UQ
    # fig_training_set() / fig_validation_set() remain available for the APD90 metric
    print(f"wrote figures to {IMG}")

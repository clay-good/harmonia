"""Harmonia dashboard — browse the dataset and run the risk-uncertainty
(classification-flip) view.

    streamlit run dashboard/app.py

This is a presentation layer over the dataset (the source of truth). It NEVER
shows a bare safety verdict: the headline view is a risk-metric *distribution*
and a classification-flip frequency, with unidentifiable channels flagged.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

import harmonia
from harmonia.simulate import (assess, flip_view, flip_sensitivity, dose_response,
                               THRESH_LOW_PCT, THRESH_HIGH_PCT, RISK_LABELS,
                               QNET_THRESH_LOW, QNET_THRESH_HIGH)
from harmonia.populations import assess_population

st.set_page_config(page_title="Harmonia — proarrhythmia risk uncertainty", layout="wide")


@st.cache_resource
def get_dataset():
    return harmonia.load()


@st.cache_data
def run_assess(drug, ap_model, mc, mult, seed, metric, uq="moments"):
    ds = get_dataset()
    a = assess(ds, drug, ap_model=ap_model, n_mc=mc, exposure_multiple=mult,
               seed=seed, metric=metric, uq=uq)
    return {
        "dapd90": a.dapd90_distribution, "qnet_dist": a.qnet_distribution,
        "point": a.dapd90_pct, "qnet": a.qnet,
        "classification": a.classification, "dist": a.classification_distribution,
        "flip": a.classification_flip_frequency, "flip_ci": a.flip_ci, "tier": a.tier,
        "warnings": a.warnings, "excluded": a.excluded_channels,
        "exposure": a.reference_exposure_nM, "baseline": a.baseline_apd90,
        "apd90": a.apd90, "metric": a.metric,
        "tri": a.triangulation_ms, "tri_base": a.baseline_triangulation_ms,
        "cqinward": a.cqinward, "uq": a.uq,
        "repro_flip": a.reproducibility_flip_frequency,
        "repro_flip_ci": a.reproducibility_flip_ci,
        "censored": a.censored_channels, "prior_dom": a.prior_dominated_channels,
    }


@st.cache_data
def run_population(drug, population, ap_model, n_models, mult, metric, seed):
    ds = get_dataset()
    p = assess_population(ds, drug, population=population, ap_model=ap_model,
                          n_models=n_models, exposure_multiple=mult, metric=metric,
                          seed=seed)
    return {
        "qnet_dist": p.qnet_distribution, "dapd90_dist": p.dapd90_distribution,
        "dist": p.classification_distribution, "susceptible": p.susceptible_fraction,
        "susceptible_ci": p.susceptible_fraction_ci,
        "tier": p.tier, "n_models": p.n_models, "metric": p.metric,
        "exposure": p.reference_exposure_nM, "warnings": p.warnings,
        "excluded": p.excluded_channels, "repol_failures": p.repolarization_failures,
        "conductance_scale": p.conductance_scale, "calibrated": p.calibrated,
        "acceptance_rate": p.acceptance_rate, "n_candidates": p.n_candidates,
        "rejection_reasons": p.rejection_reasons,
    }


ds = get_dataset()

st.title("Harmonia")
st.caption("Cardiac ion-channel drug-block data → in-silico proarrhythmia risk "
           "**distribution**. NOT a clinical tool, NOT a regulatory determination, "
           "NOT a safety verdict — a risk distribution with its full input uncertainty.")

tab_flip, tab_combo, tab_pop, tab_browse, tab_about = st.tabs(
    ["Risk-uncertainty (flip) view", "Drug combinations", "Population-of-models",
     "Browse dataset", "About / safety"])

# --------------------------------------------------------------------------- #
with tab_flip:
    import pandas as pd
    c1, c2, c3, c4, c5 = st.columns(5)
    drug = c1.selectbox("Drug", ds.drugs(), index=ds.drugs().index("dofetilide"))
    ap_models = [m.id.split(".", 1)[1] for m in ds.ap_models]
    ap_model = c2.selectbox("AP model", ap_models, index=ap_models.index("cipaordv1.0"))
    metric = c3.selectbox("Risk metric", ["qnet", "apd90"], index=0)
    mc = c4.slider("Monte-Carlo draws", 20, 400, 120, step=20)
    mult = c5.slider("Exposure (× EFTPC)", 1.0, 25.0, 4.0, step=1.0)
    bayes = st.checkbox(
        "Bayesian dose-response UQ (v0.2) — infer the IC50/Hill posterior under a "
        "declared prior instead of transcribing the spread (single-source channels "
        "borrow the dataset-learned between-lab SD; sub-60%-block channels become a "
        "one-sided censored posterior)", value=False)
    uq = "bayes" if bayes else "moments"

    res = run_assess(drug, ap_model, mc, mult, 0, metric, uq)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Point classification", res["classification"].upper())
    flo, fhi = res["flip_ci"]
    m2.metric("Classification-flip frequency", f"{res['flip']:.0%}",
              help=("Monte-Carlo estimate over {} draws; Wilson 95% CI {:.0%}–{:.0%}. "
                    "The flip frequency is itself a binomial proportion with sampling "
                    "error that shrinks only as you add draws.").format(mc, flo, fhi)
                   if flo == flo else "point estimate only (no Monte-Carlo draws)")
    m3.metric("Propagated tier", res["tier"])
    m4.metric("qNet (point)" if metric == "qnet" else "ΔAPD90 (point)",
              f"{res['qnet']:.4f}" if metric == "qnet" else f"{res['point']:+.1f}%")
    if flo == flo:   # not NaN
        st.caption(f"Flip-frequency **Monte-Carlo 95% CI: {flo:.0%}–{fhi:.0%}** "
                   f"({mc} draws). The headline flip frequency is a binomial proportion "
                   "with its own sampling error — wider here means add draws before "
                   "comparing two drugs' flip rates.")
    st.caption(f"Triangulation (APD90−APD50): **{res['tri']:.0f} ms** "
               f"(drug-free {res['tri_base']:.0f} ms) — a TRIaD proarrhythmia "
               "diagnostic that hERG block widens; a readout, never the classifier.")
    st.caption(f"cqInward (inward-charge vs drug-free): **{res['cqinward']:.3f}** — "
               "the CiPA INaL+ICaL charge biomarker; <1 = inward charge reduced "
               "(ICaL/INaL block, protective), >1 = increased (AP prolongation). "
               "A diagnostic, never the classifier.")

    if res["excluded"]:
        st.error("Excluded channels (unidentifiable IC50, max block < 60% → caps at "
                 f"Tier D): {', '.join(res['excluded'])}")
    for w in res["warnings"]:
        st.warning(w)

    if res["uq"] == "bayes":
        b1, b2 = st.columns(2)
        b1.metric("True-value flip frequency", f"{res['flip']:.0%}",
                  help="Samples the posterior of the drug's IC50 (μ): 'what is the value?'")
        rf = res["repro_flip"]
        rlo, rhi = res["repro_flip_ci"]
        b2.metric("New-lab (reproducibility) flip", "—" if rf != rf else f"{rf:.0%}",
                  help=("Samples the new-lab predictive (adds between-lab spread τ): "
                        "'how much would a fresh replication move the call?'"
                        + ("" if rlo != rlo else f"  Wilson 95% CI {rlo:.0%}–{rhi:.0%}.")))
        if res["censored"]:
            st.warning("Censored (one-sided, sub-60%-block) posteriors — wide and "
                       "prior-shaped, still Tier-D-capped: " + ", ".join(res["censored"]))
        if res["prior_dom"]:
            st.info("Prior-dominated channels (posterior mostly reflects the prior, "
                    "not the data — go measure them): " + ", ".join(res["prior_dom"]))

    if metric == "qnet":
        st.subheader("Distribution of qNet under input (IC50) variability")
        data = res["qnet_dist"]
        cap = ("qNet thresholds: high risk < {:.3f} | low risk > {:.3f} "
               "(lower qNet = higher risk). ".format(QNET_THRESH_HIGH, QNET_THRESH_LOW))
        rnd = 4
    else:
        st.subheader("Distribution of ΔAPD90% under input (IC50) variability")
        data = res["dapd90"]
        cap = ("APD90 thresholds: low < {:g}% ≤ intermediate < {:g}% ≤ high. "
               .format(THRESH_LOW_PCT, THRESH_HIGH_PCT))
        rnd = 1
    hist = np.histogram(data, bins=24)
    centers = 0.5 * (hist[1][:-1] + hist[1][1:])
    st.bar_chart(pd.DataFrame({"count": hist[0]}, index=np.round(centers, rnd)))
    st.caption(cap + "Classification mix: "
               + ", ".join(f"{k} {res['dist'].get(k, 0):.0%}" for k in RISK_LABELS))

    st.subheader("Classification stability across AP-model variants")
    fv = flip_view(ds, drug, n_mc=mc, metric=metric)
    rows = []
    for m, a in fv.per_model.items():
        rows.append({"AP model": m, "point class": a.classification,
                     "flip freq": f"{a.classification_flip_frequency:.0%}",
                     **{k: f"{a.classification_distribution.get(k, 0):.0%}" for k in RISK_LABELS}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.info("Stable across model variants: " + ("YES" if fv.stable_across_models else
            "**NO** — the classification depends on which AP-model variant you trust."))

    st.subheader("Which input drives the flip? (per-channel sensitivity)")
    st.caption("Solo-flip = how often the call flips when ONLY that channel's IC50 "
               "varies (others held at geomean) — the dominant driver is the IC50 "
               "worth pinning down first. `*` marks single-source (prior-driven) channels.")
    sens = flip_sensitivity(ds, drug, ap_model=ap_model, metric=metric,
                            n_mc=min(mc, 80), seed=0)
    if sens.channels:
        srows = [{"channel": c.channel,
                  "sources": f"{c.n_sources}{'*' if c.single_source else ''}",
                  "fold range": c.fold_range,
                  "solo-flip": c.solo_flip_frequency,
                  "frozen-flip": c.frozen_flip_frequency} for c in sens.channels]
        st.dataframe(pd.DataFrame(srows), hide_index=True, use_container_width=True)
        st.bar_chart(pd.DataFrame({"solo-flip frequency": [c.solo_flip_frequency for c in sens.channels]},
                                  index=[c.channel for c in sens.channels]))
        if sens.dominant_channel:
            st.info(f"Dominant uncertainty driver: **{sens.dominant_channel}** — "
                    "pinning this IC50 down would most stabilize the classification.")

    st.subheader("Dose–response (APD90 vs concentration, geomean IC50)")
    eftpc = ds.drug_reference(drug).eftpc_nm
    concs = np.geomspace(eftpc * 0.1, eftpc * 30, 24)
    dr = dose_response(ds, drug, concs, ap_model=ap_model)
    st.line_chart(pd.DataFrame({"APD90 (ms)": dr["apd90"]},
                               index=np.round(dr["concentration_nM"], 1)))

# --------------------------------------------------------------------------- #
with tab_combo:
    import pandas as pd
    from harmonia.simulate import assess_combination
    st.caption("Polypharmacy: independent block multiplies per channel, and every "
               "drug's IC50 variability is propagated jointly. Two 'intermediate' "
               "drugs can combine into 'high'.")
    cc1, cc2, cc3 = st.columns([3, 1, 1])
    sel = cc1.multiselect("Drugs (pick 2+)", ds.drugs(),
                          default=["droperidol", "ondansetron"])
    cmetric = cc2.selectbox("Metric", ["qnet", "apd90"], index=0, key="cmetric")
    cmc = cc3.slider("MC draws", 20, 300, 100, step=20, key="cmc")
    if len(sel) < 2:
        st.info("Select at least two drugs.")
    else:
        combo = assess_combination(ds, sel, n_mc=cmc, metric=cmetric)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Combined classification", combo.classification.upper())
        k2.metric("Flip frequency", f"{combo.classification_flip_frequency:.0%}")
        k3.metric("Interaction (extra ΔAPD90)", f"{combo.interaction_dapd90_pct:+.1f}%")
        k4.metric("Propagated tier", combo.tier)
        # single agents vs combination
        rows = [{"agent": d, "class": assess(ds, d, n_mc=0, metric=cmetric,
                 exposure_nM=combo.exposures_nM[d]).classification,
                 "free nM": round(combo.exposures_nM[d], 1)} for d in sel]
        rows.append({"agent": "COMBINATION", "class": combo.classification.upper(),
                     "free nM": ""})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if combo.excluded_channels:
            st.error("Excluded (unidentifiable IC50): " + ", ".join(combo.excluded_channels))
        data = combo.qnet_distribution if cmetric == "qnet" else combo.dapd90_distribution
        hist = np.histogram(data, bins=24)
        centers = 0.5 * (hist[1][:-1] + hist[1][1:])
        st.bar_chart(pd.DataFrame({"count": hist[0]},
                                  index=np.round(centers, 4 if cmetric == "qnet" else 1)))

# --------------------------------------------------------------------------- #
with tab_pop:
    import pandas as pd
    st.error("**HYPOTHESIS-TIER — Tier D, NOT FOR PREDICTION.** Conductance "
             "variability (and any disease mean-shift) is *illustrative*, not "
             "calibrated to patient data; the qNet/APD thresholds stay the healthy "
             "reference. This is a methodology view of *physiological* (between-heart) "
             "variability — never a per-patient or per-genotype safety claim.")
    st.caption("Where the flip view propagates *input* (IC50) variability, this "
               "propagates *physiological* variability: a population of virtual "
               "myocytes (per-channel conductance draws). Disease backgrounds shift a "
               "current's mean (reduced repolarization reserve); the calibrated "
               "population (Britton 2013) admits only drug-free-plausible myocytes.")

    pops = ds.populations
    pop_ids = [p.id.split(".", 1)[1] for p in pops]
    pop_labels = {p.id.split(".", 1)[1]: p.name for p in pops}
    default_pop = pop_ids.index("illustrative_v0") if "illustrative_v0" in pop_ids else 0

    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
    pdrug = pc1.selectbox("Drug", ds.drugs(), index=ds.drugs().index("dofetilide"),
                          key="pop_drug")
    population = pc2.selectbox("Population", pop_ids, index=default_pop,
                               format_func=lambda k: pop_labels.get(k, k), key="pop_sel")
    pmetric = pc3.selectbox("Metric", ["qnet", "apd90"], index=0, key="pop_metric")
    n_models = pc4.slider("Virtual myocytes", 20, 200, 100, step=20, key="pop_n")
    pmult = pc5.slider("Exposure (× EFTPC)", 1.0, 25.0, 4.0, step=1.0, key="pop_mult")

    pr = run_population(pdrug, population, "cipaordv1.0", n_models, pmult, pmetric, 0)

    p1, p2, p3, p4 = st.columns(4)
    slo, shi = pr["susceptible_ci"]
    p1.metric("Susceptible fraction (classified high)", f"{pr['susceptible']:.0%}",
              help=(f"Wilson 95% CI {slo:.0%}–{shi:.0%} over {pr['n_models']} virtual "
                    "myocytes — the susceptible fraction is a binomial proportion with "
                    "sampling error.") if slo == slo else None)
    p2.metric("Population size", pr["n_models"])
    p3.metric("Propagated tier", pr["tier"])
    p4.metric("Reference exposure", f"{pr['exposure']:.1f} nM")

    if pr["conductance_scale"]:
        shift = ", ".join(f"{k}×{v:g}" for k, v in pr["conductance_scale"].items())
        st.warning(f"**Disease/genetic background** — mean conductance shift: {shift} "
                   "(illustrative heterozygous-scale, NOT genotype-calibrated).")
    if pr["calibrated"]:
        rej = ", ".join(f"{k} {v}" for k, v in pr["rejection_reasons"].items() if v) or "none"
        st.info(f"**Experimentally-calibrated** (Britton 2013): "
                f"{pr['n_models']}/{pr['n_candidates']} candidates accepted "
                f"({pr['acceptance_rate']:.0%}) — drug-free-plausible myocytes only. "
                f"Rejected by biomarker: {rej}. Acceptance ranges are kernel-plausibility "
                "bounds, not a fit to patient data.")
    if pr["excluded"]:
        st.error("Excluded channels (unidentifiable IC50): " + ", ".join(pr["excluded"]))

    st.subheader(f"Distribution of {pmetric} across the population")
    pdata = pr["qnet_dist"] if pmetric == "qnet" else pr["dapd90_dist"]
    phist = np.histogram(pdata, bins=24)
    pcenters = 0.5 * (phist[1][:-1] + phist[1][1:])
    st.bar_chart(pd.DataFrame({"count": phist[0]},
                              index=np.round(pcenters, 4 if pmetric == "qnet" else 1)))
    st.caption("Population class mix: "
               + ", ".join(f"{k} {pr['dist'].get(k, 0):.0%}" for k in RISK_LABELS)
               + (f" · repolarization failures: {pr['repol_failures']}/{pr['n_models']}"
                  if pr["repol_failures"] else ""))
    st.caption("A spread of classifications across virtual hearts — never a verdict, "
               "never a prediction.")


# --------------------------------------------------------------------------- #
with tab_browse:
    st.subheader("Channel-block records")
    import pandas as pd
    rows = []
    for b in ds.channel_blocks:
        rows.append({
            "id": b.id, "drug": b.drug, "channel": b.channel,
            "IC50 (nM)": b.ic50_nm, "hill": b.hill, "tier": b.tier,
            "max block %": b.assay_context.max_block_observed_percent,
            "identifiable": b.identifiable, "n_sources": b.variability.n_sources,
            "fold range": b.variability.fold_range, "review": b.review_status,
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(f"{len(ds.channel_blocks)} channel-block records · "
               f"verified: {sum(1 for r in ds if r.review_status=='verified')}/{len(ds)} "
               "(LLM extraction never auto-promotes).")

# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown(
        "**Harmonia** is curated, citation-backed, variability-aware infrastructure "
        "for in-silico proarrhythmia (CiPA-style) assessment.\n\n"
        "- It reports a **risk-metric distribution** and a **classification-flip "
        "frequency**, never a bare high/intermediate/low verdict.\n"
        "- Channels whose IC50 is **unidentifiable** (max block < 60%) are flagged "
        "and excluded, never silently point-estimated.\n"
        "- The bundled AP kernel is a **reduced** O'Hara-Rudy-lineage reference "
        "implementation (Tier C). The default metric is **qNet** (the CiPA "
        "net-charge biomarker; lower = higher risk) which never makes a "
        "two-category error on the 28 CiPA compounds; **ΔAPD90%** is also "
        "selectable. It is a methodology demonstrator, not a qualified regulatory "
        "classifier.\n\n"
        "**It is NOT a clinical tool and NOT a regulatory safety determination.**")

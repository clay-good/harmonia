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
from harmonia.simulate import (assess, flip_view, dose_response,
                               THRESH_LOW_PCT, THRESH_HIGH_PCT, RISK_LABELS)

st.set_page_config(page_title="Harmonia — proarrhythmia risk uncertainty", layout="wide")


@st.cache_resource
def get_dataset():
    return harmonia.load()


@st.cache_data
def run_assess(drug, ap_model, mc, mult, seed):
    ds = get_dataset()
    a = assess(ds, drug, ap_model=ap_model, n_mc=mc, exposure_multiple=mult, seed=seed)
    return {
        "dapd90": a.dapd90_distribution, "point": a.dapd90_pct,
        "classification": a.classification, "dist": a.classification_distribution,
        "flip": a.classification_flip_frequency, "tier": a.tier,
        "warnings": a.warnings, "excluded": a.excluded_channels,
        "exposure": a.reference_exposure_nM, "baseline": a.baseline_apd90,
        "apd90": a.apd90,
    }


ds = get_dataset()

st.title("Harmonia")
st.caption("Cardiac ion-channel drug-block data → in-silico proarrhythmia risk "
           "**distribution**. NOT a clinical tool, NOT a regulatory determination, "
           "NOT a safety verdict — a risk distribution with its full input uncertainty.")

tab_flip, tab_browse, tab_about = st.tabs(
    ["Risk-uncertainty (flip) view", "Browse dataset", "About / safety"])

# --------------------------------------------------------------------------- #
with tab_flip:
    c1, c2, c3, c4 = st.columns(4)
    drug = c1.selectbox("Drug", ds.drugs(), index=ds.drugs().index("dofetilide"))
    ap_models = [m.id.split(".", 1)[1] for m in ds.ap_models]
    ap_model = c2.selectbox("AP model", ap_models, index=ap_models.index("cipaordv1.0"))
    mc = c3.slider("Monte-Carlo draws", 20, 400, 120, step=20)
    mult = c4.slider("Exposure (× EFTPC)", 1.0, 25.0, 4.0, step=1.0)

    res = run_assess(drug, ap_model, mc, mult, 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Point classification", res["classification"].upper())
    m2.metric("Classification-flip frequency", f"{res['flip']:.0%}")
    m3.metric("Propagated tier", res["tier"])
    m4.metric("ΔAPD90 (point)", f"{res['point']:+.1f}%")

    if res["excluded"]:
        st.error("Excluded channels (unidentifiable IC50, max block < 60% → caps at "
                 f"Tier D): {', '.join(res['excluded'])}")
    for w in res["warnings"]:
        st.warning(w)

    st.subheader("Distribution of ΔAPD90% under input (IC50) variability")
    hist = np.histogram(res["dapd90"], bins=24)
    import pandas as pd
    centers = 0.5 * (hist[1][:-1] + hist[1][1:])
    st.bar_chart(pd.DataFrame({"count": hist[0]}, index=np.round(centers, 1)))
    st.caption(f"Thresholds: low < {THRESH_LOW_PCT:g}% ≤ intermediate < "
               f"{THRESH_HIGH_PCT:g}% ≤ high. Classification mix: "
               + ", ".join(f"{k} {res['dist'].get(k, 0):.0%}" for k in RISK_LABELS))

    st.subheader("Classification stability across AP-model variants")
    fv = flip_view(ds, drug, n_mc=mc)
    rows = []
    for m, a in fv.per_model.items():
        rows.append({"AP model": m, "point class": a.classification,
                     "flip freq": f"{a.classification_flip_frequency:.0%}",
                     **{k: f"{a.classification_distribution.get(k, 0):.0%}" for k in RISK_LABELS}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.info("Stable across model variants: " + ("YES" if fv.stable_across_models else
            "**NO** — the classification depends on which AP-model variant you trust."))

    st.subheader("Dose–response (APD90 vs concentration, geomean IC50)")
    eftpc = ds.drug_reference(drug).eftpc_nm
    concs = np.geomspace(eftpc * 0.1, eftpc * 30, 24)
    dr = dose_response(ds, drug, concs, ap_model=ap_model)
    st.line_chart(pd.DataFrame({"APD90 (ms)": dr["apd90"]},
                               index=np.round(dr["concentration_nM"], 1)))

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
        "implementation (Tier C); the v0.1 classifier is an APD90 surrogate and a "
        "methodology demonstrator, not a qualified regulatory classifier.\n\n"
        "**It is NOT a clinical tool and NOT a regulatory safety determination.**")

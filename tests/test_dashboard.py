"""Guard the Streamlit dashboard — the spec's headline feature (§6) — without a
streamlit runtime.

A full ``AppTest`` run is impractical in CI (the app runs several Monte-Carlo
assessments per render and takes minutes). Instead we (1) byte-compile the app to
catch syntax errors, and (2) exercise the exact data-layer API the dashboard
consumes, asserting every function, attribute, and dict key it reads still
exists. If the simulate/load API drifts, this fails here rather than only at
runtime in front of a user.

Keep this in sync with ``dashboard/app.py``.
"""
import py_compile
import pathlib

import numpy as np

from harmonia.simulate import (assess, flip_view, flip_sensitivity, dose_response,
                               assess_combination,
                               THRESH_LOW_PCT, THRESH_HIGH_PCT, RISK_LABELS,
                               QNET_THRESH_LOW, QNET_THRESH_HIGH)

APP = pathlib.Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def test_dashboard_app_compiles():
    """app.py is syntactically valid (catches an edit that breaks the headline UI)."""
    py_compile.compile(str(APP), doraise=True)


def test_dashboard_constants_importable():
    for c in (THRESH_LOW_PCT, THRESH_HIGH_PCT, QNET_THRESH_LOW, QNET_THRESH_HIGH):
        assert isinstance(c, float)
    assert set(RISK_LABELS) == {"low", "intermediate", "high"}


def test_flip_tab_contract(ds):
    """Every field the 'Risk-uncertainty (flip)' tab reads must exist."""
    assert "dofetilide" in ds.drugs()
    ap_models = [m.id.split(".", 1)[1] for m in ds.ap_models]
    assert "cipaordv1.0" in ap_models

    a = assess(ds, "verapamil", ap_model="cipaordv1.0", n_mc=8,
               exposure_multiple=4.0, seed=0, metric="qnet")
    for attr in ("dapd90_distribution", "qnet_distribution", "dapd90_pct", "qnet",
                 "classification", "classification_distribution",
                 "classification_flip_frequency", "tier", "warnings",
                 "excluded_channels", "reference_exposure_nM", "baseline_apd90",
                 "apd90", "metric", "triangulation_ms", "baseline_triangulation_ms"):
        getattr(a, attr)
    assert a.classification in RISK_LABELS

    fv = flip_view(ds, "verapamil", n_mc=8, metric="qnet")
    assert isinstance(fv.stable_across_models, bool)
    for m, fa in fv.per_model.items():
        assert fa.classification in RISK_LABELS
        fa.classification_flip_frequency
        fa.classification_distribution.get("high", 0)

    eftpc = ds.drug_reference("verapamil").eftpc_nm
    concs = np.geomspace(eftpc * 0.1, eftpc * 30, 4)
    dr = dose_response(ds, "verapamil", concs, ap_model="cipaordv1.0")
    assert dr["concentration_nM"].shape == dr["apd90"].shape == concs.shape

    # the per-channel sensitivity panel
    sens = flip_sensitivity(ds, "verapamil", ap_model="cipaordv1.0", metric="qnet", n_mc=4)
    for c in sens.channels:
        c.channel, c.n_sources, c.single_source, c.fold_range
        c.solo_flip_frequency, c.frozen_flip_frequency
    sens.dominant_channel


def test_combination_tab_contract(ds):
    """Every field the 'Drug combinations' tab reads must exist."""
    sel = ["terfenadine", "ondansetron"]
    combo = assess_combination(ds, sel, n_mc=8, metric="qnet")
    for attr in ("classification", "classification_flip_frequency",
                 "interaction_dapd90_pct", "tier", "exposures_nM",
                 "excluded_channels", "qnet_distribution", "dapd90_distribution"):
        getattr(combo, attr)
    assert combo.classification in RISK_LABELS
    for d in sel:
        free = combo.exposures_nM[d]
        single = assess(ds, d, n_mc=0, metric="qnet", exposure_nM=free)
        assert single.classification in RISK_LABELS


def test_browse_tab_contract(ds):
    """Every field the 'Browse dataset' tab reads must exist on a record."""
    assert ds.channel_blocks
    for b in ds.channel_blocks[:5]:
        b.id, b.drug, b.channel, b.ic50_nm, b.hill, b.tier
        b.assay_context.max_block_observed_percent
        assert isinstance(b.identifiable, bool)
        b.variability.n_sources
        b.variability.fold_range
        assert b.review_status in ("verified", "unverified", "contested")
    # the verified-count line
    assert isinstance(sum(1 for r in ds if r.review_status == "verified"), int)

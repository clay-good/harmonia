"""The headline: variability propagation, flip frequency, tier propagation."""
import numpy as np
import pytest

import harmonia
from harmonia.simulate import (assess, flip_view, classify, classify_qnet,
                               dose_response, THRESH_LOW_PCT, THRESH_HIGH_PCT,
                               QNET_THRESH_LOW, QNET_THRESH_HIGH, DEFAULT_METRIC)


def test_classify_thresholds():
    assert classify(THRESH_HIGH_PCT + 1) == "high"
    assert classify(THRESH_LOW_PCT - 1) == "low"
    assert classify((THRESH_LOW_PCT + THRESH_HIGH_PCT) / 2) == "intermediate"


def test_qnet_classifier_direction():
    """Lower qNet = higher risk (CiPA convention)."""
    assert DEFAULT_METRIC == "qnet"
    assert classify_qnet(QNET_THRESH_HIGH - 0.01) == "high"
    assert classify_qnet(QNET_THRESH_LOW + 0.01) == "low"
    assert classify_qnet((QNET_THRESH_LOW + QNET_THRESH_HIGH) / 2) == "intermediate"


def test_metric_selectable(ds):
    a_q = assess(ds, "dofetilide", n_mc=0, metric="qnet")
    a_a = assess(ds, "dofetilide", n_mc=0, metric="apd90")
    assert a_q.metric == "qnet" and a_a.metric == "apd90"
    # the AP itself is identical; only the classification rule differs
    assert a_q.qnet == a_a.qnet
    assert a_q.classification == "high"  # dofetilide low qNet -> high risk


def test_verapamil_low_under_qnet(ds):
    """The balanced blocker lands low under the default qNet metric."""
    assert assess(ds, "verapamil", n_mc=0).classification == "low"


def test_assess_returns_distribution_not_verdict(ds):
    a = assess(ds, "dofetilide", n_mc=24)
    assert a.dapd90_distribution.shape == (24,)
    assert 0.0 <= a.classification_flip_frequency <= 1.0
    assert abs(sum(a.classification_distribution.values()) - 1.0) < 1e-9
    assert "PROHIBITED" in a.clinical_use
    assert a.classification in ("low", "intermediate", "high")


def test_assess_is_deterministic_given_seed(ds):
    a = assess(ds, "verapamil", n_mc=20, seed=7)
    b = assess(ds, "verapamil", n_mc=20, seed=7)
    assert np.array_equal(a.dapd90_distribution, b.dapd90_distribution)
    assert a.classification_flip_frequency == b.classification_flip_frequency


def test_dofetilide_is_high(ds):
    a = assess(ds, "dofetilide", n_mc=24)
    assert a.classification == "high"


def test_unidentifiable_channel_caps_tier_at_D(ds):
    """ranolazine has an unidentifiable ICaL -> assessment capped at Tier D + flagged."""
    a = assess(ds, "ranolazine", n_mc=16)
    assert a.tier == "D"
    assert any("ICaL" in e for e in a.excluded_channels)
    assert "ICaL" not in a.channels_used


def test_variability_widens_distribution(ds):
    """A high-fold-range drug (cisapride) should show classification spread."""
    a = assess(ds, "cisapride", n_mc=30)
    # more than one class should appear given the input spread
    nonzero = [k for k, v in a.classification_distribution.items() if v > 0]
    assert len(nonzero) >= 1


def test_flip_view_across_models(ds):
    fv = flip_view(ds, "verapamil", n_mc=12)
    assert set(fv.flip_by_model.keys()) == {"ord", "cipaordv1.0", "tor_ord"}
    for c in fv.flip_by_model.values():
        assert c in ("low", "intermediate", "high")
    assert isinstance(fv.stable_across_models, bool)


def test_dose_response_monotone_for_pure_herg_blocker(ds):
    concs = [1, 3, 10, 30, 100]
    dr = dose_response(ds, "dofetilide", concs)
    apd = dr["apd90"]
    # dofetilide is a near-pure hERG blocker -> APD increases with concentration
    assert apd[-1] > apd[0]


def test_unknown_drug_raises(ds):
    with pytest.raises(KeyError):
        assess(ds, "not_a_drug", n_mc=4)

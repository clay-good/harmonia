"""Recorded classification performance (Phase B)."""
from harmonia.performance import score


def test_training_accuracy_reasonable(ds):
    rep = score(ds, ap_model="cipaordv1.0", cipa_set="training", metric="qnet")
    assert rep.n == 12
    # the calibrated default qNet metric should recover most training labels
    assert rep.accuracy >= 0.75
    assert rep.adjacent_accuracy() >= 0.9


def test_qnet_never_makes_two_category_error(ds):
    """The headline Phase-C result: across all 28 CiPA compounds the default qNet
    metric never confuses 'high' with 'low' (perfect within-one-category)."""
    rep = score(ds, ap_model="cipaordv1.0", cipa_set="all", metric="qnet")
    assert rep.adjacent_accuracy() == 1.0


def test_validation_set_scored(ds):
    rep = score(ds, ap_model="cipaordv1.0", cipa_set="validation")
    assert rep.n == 16
    assert 0.0 <= rep.accuracy <= 1.0
    assert rep.adjacent_accuracy() >= 0.9


def test_confusion_matrix_totals(ds):
    rep = score(ds, ap_model="cipaordv1.0", cipa_set="all")
    assert sum(rep.confusion().values()) == rep.n == 28
    assert "performance" in rep.summary()

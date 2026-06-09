"""Recorded classification performance (Phase B)."""
from harmonia.performance import score


def test_training_accuracy_reasonable(ds):
    rep = score(ds, ap_model="cipaordv1.0", cipa_set="training")
    assert rep.n == 12
    # the calibrated default model should recover most training labels
    assert rep.accuracy >= 0.75
    assert rep.adjacent_accuracy() >= 0.85


def test_validation_set_scored(ds):
    rep = score(ds, ap_model="cipaordv1.0", cipa_set="validation")
    assert rep.n == 16
    # validation is harder; we only assert it runs and stays within-one mostly
    assert rep.adjacent_accuracy() >= 0.6
    assert 0.0 <= rep.accuracy <= 1.0


def test_confusion_matrix_totals(ds):
    rep = score(ds, ap_model="cipaordv1.0", cipa_set="all")
    assert sum(rep.confusion().values()) == rep.n == 28
    assert "performance" in rep.summary()

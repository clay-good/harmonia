"""Drug-combination assessment (Phase D, polypharmacy)."""
import numpy as np
import pytest

import harmonia
from harmonia.simulate import assess_combination, assess


def test_combination_more_block_than_either_single(ds):
    """Independent block multiplies: a combo prolongs more (lower qNet) than the
    worst single agent at the same exposures."""
    drugs = ["terfenadine", "ondansetron"]
    combo = assess_combination(ds, drugs, n_mc=0)
    singles = [assess(ds, d, n_mc=0, exposure_nM=combo.exposures_nM[d]) for d in drugs]
    worst_single_dapd = max(s.dapd90_pct for s in singles)
    assert combo.dapd90_pct > worst_single_dapd
    assert combo.qnet < min(s.qnet for s in singles)
    assert combo.interaction_dapd90_pct > 0


def test_combination_can_escalate_class(ds):
    """Two intermediate drugs can combine into high risk."""
    combo = assess_combination(ds, ["terfenadine", "ondansetron"], n_mc=0)
    assert combo.classification == "high"
    assert all(assess(ds, d, n_mc=0).classification != "high"
               for d in ["terfenadine", "ondansetron"])


def test_combination_distribution_and_flip(ds):
    combo = assess_combination(ds, ["dofetilide", "verapamil"], n_mc=24)
    assert combo.qnet_distribution.shape == (24,)
    assert 0.0 <= combo.classification_flip_frequency <= 1.0
    assert abs(sum(combo.classification_distribution.values()) - 1.0) < 1e-9
    assert "PROHIBITED" in combo.clinical_use


def test_combination_deterministic(ds):
    a = assess_combination(ds, ["dofetilide", "verapamil"], n_mc=16, seed=3)
    b = assess_combination(ds, ["dofetilide", "verapamil"], n_mc=16, seed=3)
    assert np.array_equal(a.qnet_distribution, b.qnet_distribution)


def test_combination_tier_propagates_worst(ds):
    """A combo touching ranolazine's unidentifiable ICaL is capped at Tier D."""
    combo = assess_combination(ds, ["ranolazine", "dofetilide"], n_mc=0)
    assert combo.tier == "D"
    assert any("ranolazine:ICaL" in e for e in combo.excluded_channels)


def test_combination_requires_two_drugs(ds):
    with pytest.raises(ValueError):
        assess_combination(ds, ["dofetilide"], n_mc=0)


def test_combo_cli(capsys):
    from harmonia.cli import main
    assert main(["combo", "terfenadine", "ondansetron", "--mc", "12"]) == 0
    out = capsys.readouterr().out
    assert "combination = terfenadine + ondansetron" in out
    assert "interaction" in out

"""flip_sensitivity: attribute the classification-flip frequency to channels."""
import pytest

from harmonia.simulate import flip_sensitivity, assess


def test_single_channel_drug(ds):
    """Dofetilide blocks essentially only IKr, so IKr is the lone driver; with one
    channel, 'solo' equals 'all-vary' and 'frozen' (nothing varies) is zero."""
    fs = flip_sensitivity(ds, "dofetilide", n_mc=12, seed=0)
    assert [c.channel for c in fs.channels] == ["IKr"]
    assert fs.dominant_channel == "IKr"
    assert fs.channels[0].frozen_flip_frequency == 0.0
    assert fs.channels[0].solo_flip_frequency == fs.all_vary_flip_frequency


def test_sorted_and_bounded(ds):
    fs = flip_sensitivity(ds, "loratadine", n_mc=10, seed=0)
    solos = [c.solo_flip_frequency for c in fs.channels]
    assert solos == sorted(solos, reverse=True)               # sorted, dominant first
    for c in fs.channels:
        assert 0.0 <= c.solo_flip_frequency <= 1.0
        assert 0.0 <= c.frozen_flip_frequency <= 1.0


def test_classification_matches_assess(ds):
    """The point classification must agree with assess (same geomean, kernel)."""
    fs = flip_sensitivity(ds, "verapamil", n_mc=0)
    a = assess(ds, "verapamil", n_mc=0)
    assert fs.classification == a.classification
    assert fs.all_vary_flip_frequency == 0.0                  # n_mc=0 -> no draws


def test_unidentifiable_channel_excluded_and_caps_tier(ds):
    """Ranolazine's ICaL is unidentifiable (max block < 60%): it must be excluded
    from the channel list and cap the assessment at Tier D."""
    fs = flip_sensitivity(ds, "ranolazine", n_mc=4, seed=0)
    channels = [c.channel for c in fs.channels]
    assert "ICaL" not in channels
    assert any("ICaL" in e for e in fs.excluded_channels)
    assert fs.tier == "D"


def test_bad_metric_rejected(ds):
    with pytest.raises(ValueError):
        flip_sensitivity(ds, "dofetilide", metric="bogus", n_mc=2)

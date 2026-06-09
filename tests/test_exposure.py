"""Exposure layer (Phase D): protein binding free<->total conversion."""
import pytest

import harmonia
from harmonia.exposure import (free_from_total, total_from_free,
                               resolve_free_exposure)


def test_free_total_roundtrip():
    assert free_from_total(1000.0, 0.1) == 100.0
    assert total_from_free(100.0, 0.1) == 1000.0
    assert abs(total_from_free(free_from_total(250.0, 0.36), 0.36) - 250.0) < 1e-9


def test_fraction_unbound_bounds():
    with pytest.raises(ValueError):
        free_from_total(100.0, 0.0)
    with pytest.raises(ValueError):
        free_from_total(100.0, 1.5)


def test_dataset_has_protein_binding(ds):
    r = ds.drug_reference("verapamil")
    assert r.fraction_unbound == 0.10
    # total = free / fu
    assert abs(r.total_cmax_nm - r.eftpc_nm / 0.10) < 1e-6


def test_resolve_free_exposure_total(ds):
    r = ds.drug_reference("verapamil")
    # total 3200 nM * fu 0.1 = 320 free
    assert resolve_free_exposure(r, exposure_nM=3200.0, exposure_kind="total") == 320.0
    # free passthrough
    assert resolve_free_exposure(r, exposure_nM=320.0, exposure_kind="free") == 320.0
    # default multiple of free EFTPC
    assert resolve_free_exposure(r, exposure_multiple=4.0) == r.eftpc_nm * 4.0


def test_assess_total_equals_equivalent_free(ds):
    free = harmonia.assess(ds, "verapamil", n_mc=0, exposure_nM=320.0, exposure_kind="free")
    total = harmonia.assess(ds, "verapamil", n_mc=0, exposure_nM=3200.0, exposure_kind="total")
    assert abs(free.dapd90_pct - total.dapd90_pct) < 1e-9
    assert free.qnet == total.qnet


def test_total_exposure_without_fu_raises(ds):
    # azimilide (validation) has no protein_binding recorded
    with pytest.raises(ValueError):
        harmonia.assess(ds, "azimilide", n_mc=0, exposure_nM=1000.0, exposure_kind="total")

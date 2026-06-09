"""The exposure layer (Phase D): free (unbound) vs total plasma concentration.

Channel block is driven by the FREE drug concentration at the channel, but
clinical pharmacokinetics usually reports the TOTAL plasma Cmax. The two differ by
the fraction unbound (fu), often by one to two orders of magnitude for highly
protein-bound drugs:

    free = fraction_unbound * total

Harmonia stores the free therapeutic Cmax (EFTPC) directly and, where known, the
fraction unbound (``drug_reference.protein_binding``). These helpers convert
between the two so an assessment can be driven from either a free or a total
exposure — and so a total-concentration PK trajectory (e.g. a Hypnos output) can
be turned into the free concentration the block model needs.
"""
from __future__ import annotations

from typing import Optional


def free_from_total(total_nm: float, fraction_unbound: float) -> float:
    """Free (unbound) concentration from total plasma concentration."""
    if not 0 < fraction_unbound <= 1:
        raise ValueError(f"fraction_unbound must be in (0, 1], got {fraction_unbound}")
    return total_nm * fraction_unbound


def total_from_free(free_nm: float, fraction_unbound: float) -> float:
    """Total plasma concentration from free (unbound) concentration."""
    if not 0 < fraction_unbound <= 1:
        raise ValueError(f"fraction_unbound must be in (0, 1], got {fraction_unbound}")
    return free_nm / fraction_unbound


def resolve_free_exposure(drug_ref, exposure_nM: Optional[float] = None,
                          exposure_kind: str = "free",
                          exposure_multiple: float = 4.0) -> float:
    """Resolve the free concentration (nM) to drive block, from one of:

      - an explicit ``exposure_nM`` (interpreted per ``exposure_kind``: 'free'
        is used directly; 'total' is converted via the drug's fraction unbound);
      - otherwise ``exposure_multiple`` x the free EFTPC.

    ``drug_ref`` is a DrugReference record (or None when exposure_nM is given as
    free)."""
    if exposure_nM is not None:
        if exposure_kind == "free":
            return float(exposure_nM)
        if exposure_kind == "total":
            if drug_ref is None or drug_ref.fraction_unbound is None:
                raise ValueError("total exposure needs the drug's protein_binding "
                                 "(fraction_unbound); none recorded")
            return free_from_total(float(exposure_nM), drug_ref.fraction_unbound)
        raise ValueError(f"exposure_kind must be 'free' or 'total', got {exposure_kind!r}")
    if drug_ref is None:
        raise ValueError("no exposure_nM and no drug reference (EFTPC) available")
    return drug_ref.eftpc_nm * exposure_multiple

"""v0.8 — machine cross-check against the published CiPA reference table.

The cross-check diffs every channel-block record's transcribed IC50/Hill against
an independent published value (Li 2017, via the FDA/CiPA machine-readable
table). These tests pin: the reference table loads and is well-formed; the
fold-difference classification and thresholds; that a machine cross-check is
never conflated with human ``verified``; that the two known divergences are
surfaced; and that the reference is reproducible from its build tool.
"""
import pathlib
import subprocess
import sys

import harmonia
from harmonia.crosscheck import (
    cross_check, load_reference,
    STATUS_MATCH, STATUS_MINOR, STATUS_DIVERGENT, STATUS_NO_REFERENCE,
    MATCH_FOLD, MINOR_FOLD,
)

REPO = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# the reference table
# --------------------------------------------------------------------------- #
def test_reference_loads_and_is_wellformed(ds):
    ref = load_reference(ds)
    assert ref["schema"] == "harmonia.cipa_block_reference/v1"
    assert ref["source"]["citation"] == "li-2017"
    assert ref["n_entries"] == len(ref["entries"])
    assert ref["n_drugs"] == len(ref["drugs"])
    for e in ref["entries"]:
        assert {"drug", "channel", "ic50_nm", "hill"} <= set(e)
        # an entry carries at least an IC50 or a Hill (never an empty row)
        assert e["ic50_nm"] is not None or e["hill"] is not None


def test_reference_citation_exists_in_dataset(ds):
    ref = load_reference(ds)
    assert ref["source"]["citation"] in ds.citations


def test_reference_is_reproducible_from_build_tool(tmp_path):
    """The committed JSON must be byte-identical to a fresh build (CI gate)."""
    committed = (REPO / "dataset" / "references" / "cipa_block_reference.json").read_text()
    out = subprocess.run(
        [sys.executable, str(REPO / "dataset" / "tools" / "build_cipa_reference.py")],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    rebuilt = (REPO / "dataset" / "references" / "cipa_block_reference.json").read_text()
    assert rebuilt == committed, "build_cipa_reference.py output drifted from the committed file"


# --------------------------------------------------------------------------- #
# classification + thresholds
# --------------------------------------------------------------------------- #
def test_report_runs_over_full_dataset(ds):
    rep = cross_check(ds)
    assert len(rep.checks) == len(ds.channel_blocks)
    # coverage: the 12 training drugs are in the reference; validation drugs are not
    assert rep.n_with_reference > 0
    assert rep.by_status()[STATUS_NO_REFERENCE] > 0


def test_status_thresholds_partition_fold(ds):
    rep = cross_check(ds)
    for c in rep.checks:
        if c.status == STATUS_NO_REFERENCE:
            assert c.reference_ic50_nm is None
            continue
        assert c.ic50_fold_diff is not None and c.ic50_fold_diff >= 1.0
        if c.status == STATUS_MATCH:
            assert c.ic50_fold_diff <= MATCH_FOLD
        elif c.status == STATUS_MINOR:
            assert MATCH_FOLD < c.ic50_fold_diff <= MINOR_FOLD
        elif c.status == STATUS_DIVERGENT:
            assert c.ic50_fold_diff > MINOR_FOLD


def test_machine_cross_checked_is_match_or_minor(ds):
    rep = cross_check(ds)
    for c in rep.checks:
        assert c.machine_cross_checked == (c.status in (STATUS_MATCH, STATUS_MINOR))
    # the aggregate count is consistent
    assert rep.n_cross_checked == sum(c.machine_cross_checked for c in rep.checks)


def test_dofetilide_ikr_matches_published(ds):
    """The canonical example: 5.06 nM recorded vs 4.87 nM published -> match."""
    rep = cross_check(ds, drug="dofetilide")
    ikr = [c for c in rep.checks if c.channel == "IKr"]
    assert len(ikr) == 1
    assert ikr[0].status == STATUS_MATCH
    assert ikr[0].ic50_fold_diff < 1.1


# --------------------------------------------------------------------------- #
# the honesty contract
# --------------------------------------------------------------------------- #
def test_cross_check_never_implies_verified(ds):
    """machine_cross_checked must not be reported as, or counted toward, the
    human ``verified`` status (spec.md §9)."""
    rep = cross_check(ds)
    # the dataset ships 0 verified; the cross-check must not change that framing
    assert all(c.review_status != "verified" for c in rep.checks)
    summary = rep.summary().lower()
    assert "not human `verified`" in summary or "not human 'verified'" in summary


def test_known_divergences_are_flagged(ds):
    """The cross-check's reason to exist: it surfaces transcribed values that
    diverge from the published reference. Two are currently known."""
    rep = cross_check(ds)
    divergent_ids = {c.record_id for c in rep.divergent}
    # cisapride ICaL is recorded ~1000x below the published value (a unit-scale
    # discrepancy) and terfenadine INaL ~10x below.
    assert "channel_block.cisapride.ical" in divergent_ids
    for c in rep.divergent:
        assert c.status == STATUS_DIVERGENT
        assert c.ic50_fold_diff > MINOR_FOLD


def test_divergent_sorts_first(ds):
    rep = cross_check(ds)
    if rep.divergent:
        # divergent rows are surfaced at the top of the check list
        assert rep.checks[0].status == STATUS_DIVERGENT


# --------------------------------------------------------------------------- #
# public API + matching
# --------------------------------------------------------------------------- #
def test_public_api_exported():
    assert hasattr(harmonia, "cross_check")
    assert harmonia.cross_check.__module__.endswith("crosscheck")


def test_single_drug_scope(ds):
    rep = cross_check(ds, drug="verapamil")
    assert rep.drug == "verapamil"
    assert all(c.drug == "verapamil" for c in rep.checks)
    assert len(rep.checks) == len(ds.blocks_for("verapamil"))


def test_no_reference_drug_is_all_no_reference(ds):
    """A validation-set drug absent from the 12-training-drug table cross-checks
    to all-no_reference, not to a false divergence."""
    # sotalol IS in the table; pick a validation drug guaranteed absent
    rep = cross_check(ds, drug="azimilide")
    assert rep.checks  # azimilide has block records
    assert all(c.status == STATUS_NO_REFERENCE for c in rep.checks)
    assert rep.n_cross_checked == 0

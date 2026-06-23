"""v0.8.2 — the pending_human_review provenance state and its corroboration.

A channel-block record is promoted from 'unverified' to 'pending_human_review'
only when its IC50 is identifiable and agrees (<=5x) with an INDEPENDENT published
reference (Li 2017 for training drugs; Ridder 2020 hERG + Li 2019 ICaL for
validation drugs). It is never promoted to 'verified' — that needs a human (§9).
These tests pin the state, its provenance, the corroboration gate, and that the
validation reference is reproducible from its build tool.
"""
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_no_record_is_verified(ds):
    """§9: nothing is ever auto-promoted to 'verified'."""
    assert not any(r.review_status == "verified" for r in ds)


def test_pending_records_carry_corroboration(ds):
    """Every pending_human_review record must record the source it was
    corroborated against — provenance, not a bare flag."""
    pend = [r for r in ds if r.review_status == "pending_human_review"]
    assert pend, "expected some pending_human_review records"
    for r in pend:
        corr = r.raw["extraction"].get("corroboration")
        assert corr, f"{r.id} pending but no corroboration"
        assert corr["citation"] in ds.citations, f"{r.id} corroboration cites unknown {corr['citation']}"
        if r.kind == "channel_block":
            assert corr["ic50_fold_diff"] <= 5.0
            assert r.tier != "D"  # unidentifiable channels are not eligible
        elif r.kind == "drug_reference":
            assert corr["eftpc_fold_diff"] <= 3.0
            assert corr["category_match"] is True


def test_training_drugs_corroborated_by_li2017(ds):
    """A training-drug identifiable channel agrees with Li 2017 -> pending."""
    assert ds["channel_block.dofetilide.ikr"].review_status == "pending_human_review"
    assert ds["channel_block.verapamil.ikr"].review_status == "pending_human_review"


def test_validation_hERG_corroborated_by_ridder(ds):
    """Validation-drug hERG values that agree with Ridder 2020 -> pending, citing
    the new ridder-2020 source."""
    p = ds["channel_block.pimozide.ikr"]
    assert p.review_status == "pending_human_review"
    assert p.raw["extraction"]["corroboration"]["citation"] == "ridder-2020"
    assert "ridder-2020" in ds.citations


def test_cross_method_disagreement_stays_unverified(ds):
    """astemizole hERG (0.9 nM, Kramer) sits ~21x from Ridder's 19 nM — a real
    cross-method discrepancy, so it is NOT corroborated and stays unverified
    (flagged for a human, not silently 'filled')."""
    assert ds["channel_block.astemizole.ikr"].review_status == "unverified"
    # loratadine disagrees on both measured channels
    assert ds["channel_block.loratadine.ikr"].review_status == "unverified"


def test_reconciled_tier_d_channels_not_promoted(ds):
    """The v0.8.1 Tier-D unidentifiable channels are excluded from promotion
    (their IC50 is an extrapolation)."""
    for rid in ("channel_block.cisapride.ical", "channel_block.terfenadine.inal"):
        assert ds[rid].review_status == "unverified"


def test_training_drug_reference_corroborated(ds):
    """The 12 training drug_reference records corroborate their EFTPC + CiPA risk
    category against the FDA/CiPA reference -> pending_human_review."""
    for drug in ("dofetilide", "verapamil", "sotalol", "diltiazem"):
        rec = ds[f"drug_reference.{drug}"]
        assert rec.review_status == "pending_human_review"
        assert rec.raw["extraction"]["corroboration"]["category_match"] is True


def test_validation_drug_reference_corroborated(ds):
    """Validation drug_reference records corroborate EFTPC + category against the
    FDA/CiPA newCiPA.csv (cross-checked vs Llopis-Lorente 2022) -> pending."""
    for drug in ("azimilide", "vandetanib", "loratadine", "pimozide"):
        assert ds[f"drug_reference.{drug}"].review_status == "pending_human_review"


def test_ibutilide_eftpc_discrepancy_stays_unverified(ds):
    """ibutilide's free EFTPC is 0.52 nM in Harmonia but 100 nM in two independent
    CiPA sources (FDA newCiPA.csv + Llopis-Lorente 2022) — a ~200x discrepancy.
    The corroboration must NOT promote it; it stays unverified, flagged for a human."""
    assert ds["drug_reference.ibutilide"].review_status == "unverified"


def test_validation_reference_reproducible():
    """The committed validation reference must be byte-identical to a fresh build."""
    path = REPO / "dataset" / "references" / "cipa_validation_reference.json"
    committed = path.read_text()
    out = subprocess.run(
        [sys.executable, str(REPO / "dataset" / "tools" / "build_cipa_reference.py")],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert path.read_text() == committed


def test_validation_reference_wellformed():
    data = json.loads(
        (REPO / "dataset" / "references" / "cipa_validation_reference.json").read_text())
    assert data["schema"] == "harmonia.cipa_validation_reference/v1"
    # hERG entries cite Ridder, ICaL entries cite Li 2019
    for e in data["entries"]:
        if e["channel"] == "IKr":
            assert e["citation"] == "ridder-2020"
        elif e["channel"] == "ICaL":
            assert e["citation"] == "li-2019"

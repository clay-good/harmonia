"""Dataset integrity: schema + semantic validation, citations, tier rules."""
from harmonia.records import ChannelBlock
from harmonia.validate import validate_dataset


def test_loads(ds):
    assert len(ds) > 40
    assert len(ds.citations) >= 10
    assert "channel_block.dofetilide.ikr" in ds


def test_validation_passes():
    report = validate_dataset()
    assert report.ok, "\n".join(report.errors)


def test_cipa_compound_sets(ds):
    assert len(ds.drugs()) == 28  # 12 training + 16 validation
    training = [r for r in ds.drug_references if r.cipa_set == "training"]
    validation = [r for r in ds.drug_references if r.cipa_set == "validation"]
    assert len(training) == 12
    assert len(validation) == 16
    for d in ["dofetilide", "verapamil", "sotalol", "ranolazine"]:
        assert d in ds.drugs()
    for d in ["azimilide", "astemizole", "nifedipine", "ibutilide"]:  # validation
        assert d in ds.drugs()


def test_every_channel_block_has_sources_and_citation(ds):
    for b in ds.channel_blocks:
        assert isinstance(b, ChannelBlock)
        assert b.source_values, f"{b.id} has no source_values"
        assert b.primary_citation in ds.citations, f"{b.id} cites unknown citation"
        assert b.variability.n_sources == len(b.source_values)


def test_tier_d_iff_unidentifiable(ds):
    """The reliability gate: max block < 60% <=> Tier D + failure mode."""
    for b in ds.channel_blocks:
        mb = b.assay_context.max_block_observed_percent
        if mb is not None and mb < 60:
            assert b.tier == "D", f"{b.id} unidentifiable but tier {b.tier}"
            assert not b.identifiable
            conditions = " ".join(f["condition"] for f in b.known_failure_modes)
            assert "60" in conditions, f"{b.id} missing reliability failure mode"
        if b.tier == "D":
            assert mb is not None and mb < 60, f"{b.id} tier D but block {mb}"


def test_unverified_posture(ds):
    """Honesty (spec §9): no record is ever auto-promoted to 'verified'. Records
    are 'unverified' (uncorroborated/illustrative) or 'pending_human_review'
    (machine-corroborated against a published source, awaiting human PDF
    confirmation) — never 'verified' without a human."""
    assert all(r.review_status in ("unverified", "pending_human_review") for r in ds)
    assert not any(r.review_status == "verified" for r in ds)
    # every pending_human_review record must carry its corroboration provenance
    for r in ds:
        if r.review_status == "pending_human_review":
            assert r.raw["extraction"].get("corroboration"), \
                f"{r.id} is pending_human_review but lacks corroboration provenance"


def test_drug_references_have_eftpc_and_label(ds):
    for r in ds.drug_references:
        assert r.eftpc_nm > 0
        assert r.expert_risk_label in ("high", "intermediate", "low")


def test_ranolazine_has_unidentifiable_channel(ds):
    """The deliberate Tier-D example must exist."""
    ical = ds.get("channel_block.ranolazine.ical")
    assert ical is not None
    assert ical.tier == "D"
    assert not ical.identifiable

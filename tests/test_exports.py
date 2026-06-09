"""Exports: well-formedness, the clinicalUse annotation, and round trips."""
import io
import xml.etree.ElementTree as ET
import zipfile

import pytest

from harmonia.export import (registry, cellml, sbml, sedml, myokit, combine,
                             cipa_inputs, csv_bibtex)


@pytest.mark.parametrize("fmt", ["cellml", "sbml", "sedml"])
def test_xml_well_formed(ds, fmt):
    text = registry.build_text(ds, fmt, ap_model="ord")
    ET.fromstring(text)  # raises on malformed XML


def test_clinical_use_annotation_on_every_model_export(ds):
    for fmt in ("cellml", "sbml", "myokit"):
        text = registry.build_text(ds, fmt, ap_model="cipaordv1.0")
        assert "PROHIBITED" in text, f"{fmt} missing clinicalUse"


def test_cellml_has_tier_and_doi(ds):
    text = cellml.build(ds, "ord")
    assert "confidenceTier" in text
    assert "doi.org/10.1371/journal.pcbi.1002061" in text  # ORd DOI


@pytest.mark.parametrize("ap", ["ord", "cipaordv1.0", "tor_ord"])
def test_cellml_declaration_level_unit_conformance(ds, ap):
    """Every exported CellML model is declaration-level unit-conformant: no
    variable or <cn> literal references a unit the model never defines."""
    assert cellml.conformance_violations(cellml.build(ds, ap)) == []


def test_cellml_conformance_check_catches_a_dangling_unit(ds):
    """The checker must actually flag a unit that isn't built-in or defined —
    otherwise it would be a no-op that silently passes everything."""
    broken = cellml.build(ds, "ord").replace('units="mV"', 'units="furlong"', 1)
    violations = cellml.conformance_violations(broken)
    assert any("furlong" in v for v in violations)


def test_myokit_structure(ds):
    text = myokit.build(ds, "ord")
    assert "[[model]]" in text
    assert "[[protocol]]" in text
    assert "dot(V)" in text
    assert "bind pace" in text


def test_cipa_roundtrip(ds):
    assert registry.roundtrip_cipa(ds) == []


def test_parameter_roundtrip(ds):
    for ap in ("ord", "cipaordv1.0", "tor_ord"):
        assert registry.roundtrip_parameters(ds, ap) == []


@pytest.mark.parametrize("ap", ["ord", "cipaordv1.0", "tor_ord"])
@pytest.mark.parametrize("block", [None, {"IKr": 0.4}, {"ICaL": 0.5, "IKr": 0.6}])
def test_ode_roundtrip_ast_matches_kernel(ds, ap, block):
    """The exported equations (the model AST) re-integrate to the reference
    kernel's action potential — not just the constants survive the text. This is
    the ~1e-4 ODE round trip the spec/architecture promises."""
    assert registry.roundtrip_ode(ds, ap, block=block) == []


def test_ode_roundtrip_catches_drift(ds, monkeypatch):
    """If the AST drifted from the kernel, the round trip must fail — otherwise it
    is a no-op. Perturb only the AST-integration side and confirm a discrepancy is
    reported."""
    real = registry.simulate_spec

    def drifted(spec, **kw):
        r = real(spec, **kw)
        r.V = r.V + 5.0        # 5 mV offset, well beyond the V-trace tolerance
        r.apd90 = r.apd90 + 10.0
        return r

    monkeypatch.setattr(registry, "simulate_spec", drifted)
    assert registry.roundtrip_ode(ds, "ord") != []


def test_cipa_csv_reports_identifiability(ds):
    rows = cipa_inputs.parse_csv(cipa_inputs.to_csv(ds))
    by = {(r["drug"], r["channel"]): r for r in rows}
    rano = by[("ranolazine", "ICaL")]
    assert rano["identifiable"] == "False"
    assert rano["tier"] == "D"


def test_omex_is_byte_reproducible(ds):
    """Exports are deterministic projections: regenerating the .omex must produce
    identical bytes (the zip uses a fixed timestamp, not the current time)."""
    a = combine.build_bytes(ds, "ord")
    b = combine.build_bytes(ds, "ord")
    assert a == b


def test_omex_is_valid_zip_with_manifest(ds):
    data = combine.build_bytes(ds, "ord")
    z = zipfile.ZipFile(io.BytesIO(data))
    names = set(z.namelist())
    assert {"manifest.xml", "model.cellml", "model.sbml", "protocol.sedml"} <= names
    ET.fromstring(z.read("manifest.xml"))
    ET.fromstring(z.read("model.cellml"))
    ET.fromstring(z.read("model.sbml"))


@pytest.mark.parametrize("ap", ["ord", "cipaordv1.0", "tor_ord"])
def test_sedml_references_resolve(ds, ap):
    """Every internal SED-ML cross-reference (task→model/sim, variable→task,
    curve→dataGenerator) resolves."""
    assert sedml.reference_violations(registry.build_text(ds, "sedml", ap_model=ap)) == []


def test_sedml_reference_check_catches_a_dangling_reference(ds):
    text = registry.build_text(ds, "sedml", ap_model="ord")
    broken = text.replace('modelReference="apmodel"', 'modelReference="ghost"', 1)
    assert any("ghost" in v for v in sedml.reference_violations(broken))


def test_sedml_model_source_resolves_to_an_exported_file(ds, tmp_path):
    """Regression: the standalone SED-ML protocol must point at the CellML model
    that build_all actually writes (a sibling cellml/ dir), not a flattened
    'model.cellml' that does not exist in the standalone layout."""
    import xml.etree.ElementTree as ET
    registry.build_all(ds, str(tmp_path))
    for sed in (tmp_path / "sedml").glob("*.sedml"):
        root = ET.fromstring(sed.read_text())
        src = next(root.iter(f"{{{sedml.SEDML_NS}}}model")).get("source")
        assert (sed.parent / src).resolve().exists(), f"{sed.name} -> missing model {src}"


@pytest.mark.parametrize("ap", ["ord", "cipaordv1.0", "tor_ord"])
def test_omex_manifest_matches_archive(ds, ap):
    """The COMBINE manifest lists exactly the archive's files, with one master."""
    assert combine.manifest_violations(combine.build_bytes(ds, ap)) == []


def test_omex_manifest_check_catches_a_missing_file(ds):
    """Drop a listed file from the archive and confirm the manifest check flags it."""
    data = combine.build_bytes(ds, "ord")
    z = zipfile.ZipFile(io.BytesIO(data))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as out:
        for n in z.namelist():
            if n != "model.sbml":          # omit a file the manifest still lists
                out.writestr(n, z.read(n))
    assert any("model.sbml" in v for v in combine.manifest_violations(buf.getvalue()))


def test_bibtex_contains_all_citations(ds):
    bib = csv_bibtex.citations_bibtex(ds)
    for key in ds.citations:
        assert f"@article{{{key}" in bib


def test_build_all_writes_artifacts(ds, tmp_path):
    written = registry.build_all(ds, str(tmp_path))
    assert len(written) > 10
    assert all((tmp_path / "cellml").glob("*.cellml") for _ in [0])
    assert (tmp_path / "tables" / "cipa_inputs.csv").exists()

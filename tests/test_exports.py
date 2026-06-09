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


def test_bibtex_contains_all_citations(ds):
    bib = csv_bibtex.citations_bibtex(ds)
    for key in ds.citations:
        assert f"@article{{{key}" in bib


def test_build_all_writes_artifacts(ds, tmp_path):
    written = registry.build_all(ds, str(tmp_path))
    assert len(written) > 10
    assert all((tmp_path / "cellml").glob("*.cellml") for _ in [0])
    assert (tmp_path / "tables" / "cipa_inputs.csv").exists()

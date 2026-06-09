"""COMBINE .omex archive — bundles CellML + SBML + SED-ML + provenance metadata
into one citable, reproducible package.
"""
from __future__ import annotations

import io
import zipfile
from typing import Dict, Optional

from ..load import Dataset
from . import annotate, cellml, sbml, sedml

OMEX_MANIFEST_NS = "http://identifiers.org/combine.specifications/omex-manifest"
_FORMATS = {
    "omex": "http://identifiers.org/combine.specifications/omex",
    "cellml": "http://identifiers.org/combine.specifications/cellml.2.0",
    "sbml": "http://identifiers.org/combine.specifications/sbml.level-3.version-2",
    "sedml": "http://identifiers.org/combine.specifications/sed-ml.level-1.version-3",
    "rdf": "http://identifiers.org/combine.specifications/omex-metadata",
}


def _manifest(entries) -> str:
    L = ['<?xml version="1.0" encoding="UTF-8"?>',
         f'<omexManifest xmlns="{OMEX_MANIFEST_NS}">',
         f'  <content location="." format="{_FORMATS["omex"]}"/>']
    for loc, fmt, master in entries:
        m = ' master="true"' if master else ""
        L.append(f'  <content location="{loc}" format="{fmt}"{m}/>')
    L.append("</omexManifest>")
    return "\n".join(L) + "\n"


def _metadata_rdf(tier: str, dataset_version: str, dois) -> str:
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            + annotate.rdf_block("harmonia_archive", tier, dataset_version, dois, indent=""))


def build_bytes(ds: Dataset, ap_model: str = "cipaordv1.0",
                block: Optional[Dict[str, float]] = None,
                dataset_version: str = "0.1.0") -> bytes:
    from ..simulate import _resolve_ap_model
    rec = _resolve_ap_model(ds, ap_model)
    cit = ds.citation(rec.primary_citation)
    dois = [cit.doi] if cit and cit.doi else []

    files = {
        "model.cellml": cellml.build(ds, ap_model, block=block, dataset_version=dataset_version),
        "model.sbml": sbml.build(ds, ap_model, block=block, dataset_version=dataset_version),
        "protocol.sedml": sedml.build(ds, ap_model, model_source="model.cellml"),
        "metadata.rdf": _metadata_rdf(rec.tier, dataset_version, dois),
    }
    entries = [
        ("./model.cellml", _FORMATS["cellml"], False),
        ("./model.sbml", _FORMATS["sbml"], False),
        ("./protocol.sedml", _FORMATS["sedml"], True),
        ("./metadata.rdf", _FORMATS["rdf"], False),
    ]
    manifest = _manifest(entries)

    # Deterministic archive: a fixed timestamp + fixed ordering so the .omex is
    # byte-reproducible (plain writestr() would stamp the current time, making the
    # archive non-reproducible — exports must be deterministic projections).
    fixed_dt = (1980, 1, 1, 0, 0, 0)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        members = [("manifest.xml", manifest)] + sorted(files.items())
        for name, content in members:
            zi = zipfile.ZipInfo(filename=name, date_time=fixed_dt)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, content)
    return buf.getvalue()


def manifest_violations(data: bytes) -> list:
    """Check a COMBINE ``.omex`` archive's manifest against its actual contents:

      - every manifest ``location`` (other than the archive root ".") names a file
        present in the zip;
      - every file in the zip (other than the manifest itself) is listed in the
        manifest;
      - the manifest declares exactly one ``master`` entry and that file exists.

    Returns the list of inconsistencies (empty == the archive is self-consistent).
    A manifest that points at a missing file — or a file the manifest forgot — is
    the bug this catches.
    """
    import xml.etree.ElementTree as ET

    z = zipfile.ZipFile(io.BytesIO(data))
    members = set(z.namelist())
    if "manifest.xml" not in members:
        return ["archive has no manifest.xml"]
    root = ET.fromstring(z.read("manifest.xml"))
    ns = f"{{{OMEX_MANIFEST_NS}}}"

    listed, masters = set(), []
    for c in root.iter(f"{ns}content"):
        loc = (c.get("location") or "").lstrip("./")
        if c.get("location") == ".":
            continue
        listed.add(loc)
        if c.get("master") == "true":
            masters.append(loc)

    v = []
    for loc in sorted(listed):
        if loc not in members:
            v.append(f"manifest lists '{loc}' but it is not in the archive")
    for m in sorted(members - {"manifest.xml"}):
        if m not in listed:
            v.append(f"archive file '{m}' is not listed in the manifest")
    if len(masters) != 1:
        v.append(f"expected exactly one master entry, found {len(masters)}")
    elif masters[0] not in members:
        v.append(f"master '{masters[0]}' is not in the archive")
    return v


def build(ds: Dataset, output_path: str, ap_model: str = "cipaordv1.0",
          block: Optional[Dict[str, float]] = None, dataset_version: str = "0.1.0") -> str:
    data = build_bytes(ds, ap_model, block=block, dataset_version=dataset_version)
    with open(output_path, "wb") as fh:
        fh.write(data)
    return output_path

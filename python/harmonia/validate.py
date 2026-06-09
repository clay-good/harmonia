"""Validate the dataset against the JSON Schema and Harmonia's semantic rules.

Two layers:
  1. *Schema* validation — every record file satisfies ``record.schema.json``.
  2. *Semantic* validation — the rules a JSON Schema cannot express:
       - every cited citation key resolves to a citation record;
       - the reliability gate: max_block < 60%  =>  tier MUST be D and a
         matching known_failure_mode MUST be present;
       - variability bookkeeping is internally consistent (n_sources matches
         source_values; fold_range matches the source spread);
       - record ids are unique and match their filename.
"""
from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional

from .records import IDENTIFIABILITY_BLOCK_THRESHOLD
from .load import find_dataset_dir


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_records: int = 0
    n_citations: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        lines = [f"validated {self.n_records} records, {self.n_citations} citations"]
        for w in self.warnings:
            lines.append(f"  WARN  {w}")
        for e in self.errors:
            lines.append(f"  ERROR {e}")
        lines.append("OK" if self.ok else f"FAILED ({len(self.errors)} error(s))")
        return "\n".join(lines)


def _load_schema(root: pathlib.Path) -> dict:
    return json.loads((root / "schema" / "record.schema.json").read_text(encoding="utf-8"))


def validate_dataset(path: Optional[str] = None) -> ValidationReport:
    root = find_dataset_dir(path)
    report = ValidationReport()

    # ---- schema layer ----------------------------------------------------- #
    try:
        from jsonschema import Draft202012Validator
        schema = _load_schema(root)
        validator = Draft202012Validator(schema)
    except Exception as exc:  # pragma: no cover - import/setup failure
        report.errors.append(f"could not initialise JSON Schema validator: {exc}")
        return report

    record_files = sorted((root / "records").glob("*.json"))
    citation_files = sorted((root / "citations").glob("*.json"))
    report.n_records = len(record_files)
    report.n_citations = len(citation_files)

    citation_keys = set()
    for cf in citation_files:
        try:
            c = json.loads(cf.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.errors.append(f"{cf.name}: invalid JSON ({exc})")
            continue
        key = c.get("key")
        if key != cf.stem:
            report.errors.append(f"{cf.name}: citation key '{key}' != filename '{cf.stem}'")
        if not c.get("doi") and not c.get("pmid"):
            report.errors.append(f"{cf.name}: citation has neither a DOI nor a PMID")
        citation_keys.add(key)

    seen_ids = set()
    for rf in record_files:
        try:
            rec = json.loads(rf.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.errors.append(f"{rf.name}: invalid JSON ({exc})")
            continue

        for err in sorted(validator.iter_errors(rec), key=lambda e: e.path):
            loc = "/".join(str(p) for p in err.path) or "(root)"
            report.errors.append(f"{rf.name}: schema [{loc}] {err.message}")

        rid = rec.get("id")
        if rid in seen_ids:
            report.errors.append(f"{rf.name}: duplicate record id '{rid}'")
        seen_ids.add(rid)
        if rid and rf.stem != rid:
            report.errors.append(f"{rf.name}: filename does not match id '{rid}'")

        _check_semantics(rf.name, rec, citation_keys, report)

    return report


def _cited_keys(rec: dict) -> List[str]:
    keys = []
    if rec.get("primary_citation"):
        keys.append(rec["primary_citation"])
    for p in rec.get("parameters", []):
        if p.get("primary_citation"):
            keys.append(p["primary_citation"])
    for s in rec.get("source_values", []):
        if s.get("citation"):
            keys.append(s["citation"])
    for f in rec.get("known_failure_modes", []):
        if f.get("citation"):
            keys.append(f["citation"])
    if rec.get("eftpc_nm", {}).get("citation"):
        keys.append(rec["eftpc_nm"]["citation"])
    if rec.get("dynamic_binding", {}).get("citation"):
        keys.append(rec["dynamic_binding"]["citation"])
    if rec.get("protein_binding", {}).get("citation"):
        keys.append(rec["protein_binding"]["citation"])
    return keys


def _check_semantics(name: str, rec: dict, citation_keys: set, report: ValidationReport) -> None:
    # every cited key must resolve
    for key in _cited_keys(rec):
        if key not in citation_keys:
            report.errors.append(f"{name}: cites unknown citation key '{key}'")

    if rec.get("kind") != "channel_block":
        return

    ac = rec.get("assay_context", {})
    max_block = ac.get("max_block_observed_percent")
    tier = rec.get("tier")

    # the reliability gate
    if max_block is not None and max_block < IDENTIFIABILITY_BLOCK_THRESHOLD:
        if tier != "D":
            report.errors.append(
                f"{name}: max_block {max_block}% < {IDENTIFIABILITY_BLOCK_THRESHOLD:g}% "
                f"(IC50 unidentifiable) but tier is '{tier}', must be 'D'")
        fm_conditions = " ".join(f.get("condition", "") for f in rec.get("known_failure_modes", []))
        if "60" not in fm_conditions:
            report.errors.append(
                f"{name}: unidentifiable IC50 (max_block < 60%) but no matching "
                f"known_failure_mode recorded")

    # variability bookkeeping
    sources = rec.get("source_values", [])
    var = rec.get("variability", {})
    if var.get("n_sources") is not None and var["n_sources"] != len(sources):
        report.errors.append(
            f"{name}: variability.n_sources={var['n_sources']} but {len(sources)} source_values")
    if len(sources) >= 2:
        ic50s = [s["ic50_nm"] for s in sources]
        expected_fold = max(ic50s) / min(ic50s)
        got = var.get("fold_range")
        if got is None or not math.isclose(got, expected_fold, rel_tol=1e-2):
            report.warnings.append(
                f"{name}: variability.fold_range={got} but source spread implies "
                f"{expected_fold:.3f}")

    # tier D records should warn that they are not predictive
    if tier == "D" and not rec.get("notes"):
        report.warnings.append(f"{name}: Tier D record has no explanatory note")

"""CiPA-pipeline input export: the IC50 / Hill table the reference CiPA in-silico
tool ingests, annotated with Harmonia's variability and tier fields so curated,
variability-aware block parameters drop straight into the established pipeline.

This export is the one with a true numeric round-trip (see tests): parse the CSV
back and the IC50/Hill values equal the dataset's.
"""
from __future__ import annotations

import csv
import io
import json
from typing import Dict, List, Optional

from ..load import Dataset
from ..records import ChannelBlock

CIPA_CHANNELS = ["IKr", "ICaL", "INaL", "INa", "Ito", "IKs", "IK1"]
COLUMNS = [
    "drug", "channel", "ic50_nM", "hill", "block_model",
    "max_block_percent", "identifiable", "tier", "review_status",
    "n_sources", "fold_range", "primary_citation",
]


def _rows(ds: Dataset, drug: Optional[str] = None) -> List[Dict[str, object]]:
    rows = []
    blocks = ds.channel_blocks
    if drug:
        blocks = [b for b in blocks if b.drug.lower() == drug.lower()]  # type: ignore[attr-defined]
    for b in sorted(blocks, key=lambda r: (r.drug, CIPA_CHANNELS.index(r.channel)
                                           if r.channel in CIPA_CHANNELS else 99)):
        assert isinstance(b, ChannelBlock)
        v = b.variability
        rows.append({
            "drug": b.drug,
            "channel": b.channel,
            "ic50_nM": b.ic50_nm,
            "hill": b.hill,
            "block_model": b.block_model,
            "max_block_percent": b.assay_context.max_block_observed_percent,
            "identifiable": b.identifiable,
            "tier": b.tier,
            "review_status": b.review_status,
            "n_sources": v.n_sources,
            "fold_range": v.fold_range,
            "primary_citation": b.primary_citation,
        })
    return rows


def to_csv(ds: Dataset, drug: Optional[str] = None) -> str:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=COLUMNS)
    w.writeheader()
    for row in _rows(ds, drug):
        w.writerow(row)
    return out.getvalue()


def to_json(ds: Dataset, drug: Optional[str] = None) -> str:
    payload = {
        "format": "harmonia.cipa_inputs",
        "version": "0.1.0",
        "clinical_use": ("PROHIBITED — research / safety-methodology / education only; "
                         "not a regulatory determination"),
        "note": "IC50 in nM; identifiable=false channels (max block < 60%) are reported "
                "but MUST be excluded from point classification.",
        "rows": _rows(ds, drug),
    }
    return json.dumps(payload, indent=2)


def parse_csv(text: str) -> List[Dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))

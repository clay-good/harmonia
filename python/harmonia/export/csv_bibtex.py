"""Flat CSV (all channel-block parameters) and BibTeX (all citations) export."""
from __future__ import annotations

import csv
import io

from ..load import Dataset
from ..records import ChannelBlock

_CSV_COLUMNS = [
    "id", "drug", "channel", "parameter", "central", "low", "high", "units",
    "tier", "review_status", "max_block_percent", "n_sources", "fold_range",
    "primary_citation",
]


def parameters_csv(ds: Dataset) -> str:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=_CSV_COLUMNS)
    w.writeheader()
    for b in ds.channel_blocks:
        assert isinstance(b, ChannelBlock)
        v = b.variability
        for p in b.raw.get("parameters", []):
            val = p["value"]
            w.writerow({
                "id": b.id, "drug": b.drug, "channel": b.channel,
                "parameter": p["symbol"], "central": val.get("central"),
                "low": val.get("low"), "high": val.get("high"),
                "units": val.get("units"), "tier": b.tier,
                "review_status": b.review_status,
                "max_block_percent": b.assay_context.max_block_observed_percent,
                "n_sources": v.n_sources, "fold_range": v.fold_range,
                "primary_citation": b.primary_citation,
            })
    return out.getvalue()


def _bibtex_entry(c) -> str:
    fields = []
    if c.authors:
        fields.append(f"  author = {{{c.authors}}}")
    if c.title:
        fields.append(f"  title = {{{c.title}}}")
    if c.journal:
        fields.append(f"  journal = {{{c.journal}}}")
    if c.year:
        fields.append(f"  year = {{{c.year}}}")
    if c.doi:
        fields.append(f"  doi = {{{c.doi}}}")
    if c.pmid:
        fields.append(f"  note = {{PMID: {c.pmid}}}")
    body = ",\n".join(fields)
    return f"@article{{{c.key},\n{body}\n}}"


def citations_bibtex(ds: Dataset) -> str:
    entries = [_bibtex_entry(c) for c in
               sorted(ds.citations.values(), key=lambda c: c.key)]
    header = ("% Harmonia citations — when you use a record, cite Harmonia AND\n"
              "% the primary source(s) named in that record.\n")
    return header + "\n\n".join(entries) + "\n"

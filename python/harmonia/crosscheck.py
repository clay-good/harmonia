"""Machine cross-check of transcribed channel-block records against an
independent published reference (spec v0.8).

THE PROBLEM THIS ADDRESSES. Every Harmonia record ships ``review_status:
"unverified"`` — the values are literature-derived but nobody has opened the
source PDF *inside Harmonia*, and (spec.md §9) an LLM may never promote a record
to ``verified``. That is honest, but it leaves a reader with no automated signal
of whether a transcribed IC50 is even in the right ballpark.

WHAT THIS LAYER DOES. For each channel-block record it diffs the recorded IC50
and Hill against the published CiPA value for that drug x channel
(``dataset/references/cipa_block_reference.json``, transcribed from Li et al.
2017 via the FDA/CiPA machine-readable table — a *different* transcription pass
than the records, so the comparison is non-circular). It classifies the
agreement and exposes a computed ``machine_cross_checked`` flag.

WHAT THIS LAYER IS NOT. ``machine_cross_checked`` is **not** ``verified``. It
says only "this transcribed number agrees with an independent published number,
within the several-fold spread the literature itself shows." It does not confirm
a human read the primary source, it cannot catch an error shared by both
renderings, and it covers only the 12 CiPA training drugs the reference table
spans. Human verification (a contributor opening the PDF) remains the only path
to ``verified``. This is a weaker, honestly-labeled provenance signal that sits
*between* "unverified" and "verified" — never a substitute for the latter.

Like ``performance``, this is a *derived* quantity computed on demand, never
baked into the dataset records.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from .load import Dataset

# IC50 agreement is scored on the fold-difference max(a/b, b/a). The project's
# own thesis is that IC50s vary several-fold across labs (and a record's central
# value is a geomean that may include other sources than the reference), so a
# 2-3x gap is expected, not an error. The divergence threshold mirrors the
# dataset's own ``fold_range > 5`` failure-mode trigger: only a >5x gap from the
# published value is treated as suspicious (likely a unit/transcription error or
# a mis-mapped drug/channel) and flagged for human review.
MATCH_FOLD = 2.0       # <= : strong agreement
MINOR_FOLD = 5.0       # <= : within documented inter-lab variability (not an error)
# > MINOR_FOLD          : divergent — flag for human review
HILL_NOTE_ABS = 0.3    # |hill difference| above which we annotate (Hill is secondary)

STATUS_MATCH = "match"
STATUS_MINOR = "minor"
STATUS_DIVERGENT = "divergent"
STATUS_NO_REFERENCE = "no_reference"


def _fold(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return float("inf")
    return max(a / b, b / a)


@dataclass
class ChannelCrossCheck:
    """One channel-block record diffed against the published reference."""
    record_id: str
    drug: str
    channel: str
    tier: str
    review_status: str
    recorded_ic50_nm: float
    recorded_hill: float
    reference_ic50_nm: Optional[float]
    reference_hill: Optional[float]
    ic50_fold_diff: Optional[float]
    hill_abs_diff: Optional[float]
    status: str

    @property
    def machine_cross_checked(self) -> bool:
        """True when the recorded IC50 agrees with the independent published
        value within the documented inter-lab spread. NOT the same as a human
        ``verified`` stamp (spec.md §9)."""
        return self.status in (STATUS_MATCH, STATUS_MINOR)

    @property
    def hill_note(self) -> Optional[str]:
        if self.hill_abs_diff is not None and self.hill_abs_diff > HILL_NOTE_ABS:
            return f"Hill differs by {self.hill_abs_diff:.2f}"
        return None


@dataclass
class CrossCheckReport:
    drug: Optional[str]
    checks: List[ChannelCrossCheck]
    reference_source: Dict
    n_records_total: int  # channel-block records considered (for coverage math)

    def by_status(self) -> Dict[str, int]:
        out = {STATUS_MATCH: 0, STATUS_MINOR: 0, STATUS_DIVERGENT: 0, STATUS_NO_REFERENCE: 0}
        for c in self.checks:
            out[c.status] = out.get(c.status, 0) + 1
        return out

    @property
    def n_cross_checked(self) -> int:
        return sum(c.machine_cross_checked for c in self.checks)

    @property
    def n_with_reference(self) -> int:
        return sum(c.status != STATUS_NO_REFERENCE for c in self.checks)

    @property
    def divergent(self) -> List[ChannelCrossCheck]:
        return [c for c in self.checks if c.status == STATUS_DIVERGENT]

    def summary(self) -> str:
        counts = self.by_status()
        scope = f"drug={self.drug}" if self.drug else f"all channel-block records ({len(self.checks)})"
        lines = [
            f"machine cross-check — {scope}",
            f"  reference: {self.reference_source.get('published_table', 'published CiPA table')}",
            f"  conduit:   {self.reference_source.get('machine_readable_conduit', '')} "
            f"@ {self.reference_source.get('conduit_commit', '')[:10]}",
            f"  with a published reference: {self.n_with_reference}/{len(self.checks)}   "
            f"(no_reference={counts[STATUS_NO_REFERENCE]} — drugs/currents outside the "
            f"12-training-drug table)",
            f"  machine_cross_checked: {self.n_cross_checked}/{len(self.checks)}   "
            f"(match={counts[STATUS_MATCH]}, minor={counts[STATUS_MINOR]}, "
            f"DIVERGENT={counts[STATUS_DIVERGENT]})",
        ]
        rows = [c for c in self.checks if c.status != STATUS_NO_REFERENCE]
        if rows:
            lines.append("  recorded vs published (IC50 nM, fold-diff):")
            for c in sorted(rows, key=lambda x: (x.drug, x.channel)):
                tag = {STATUS_MATCH: "ok ", STATUS_MINOR: "~  ", STATUS_DIVERGENT: "!! "}[c.status]
                note = f"  [{c.hill_note}]" if c.hill_note else ""
                lines.append(
                    f"    {tag}{c.drug:14s} {c.channel:5s} "
                    f"rec={c.recorded_ic50_nm:>12.4g}  ref={c.reference_ic50_nm:>12.4g}  "
                    f"x{c.ic50_fold_diff:.2f}{note}")
        if self.divergent:
            lines.append("  DIVERGENT (>5x from the published value — inspect against the "
                         "primary source before relying on these):")
            for c in self.divergent:
                lines.append(f"    {c.record_id}  recorded={c.recorded_ic50_nm:g} nM  "
                             f"published={c.reference_ic50_nm:g} nM  ({c.ic50_fold_diff:.1f}x)")
        lines.append("  NOTE: machine_cross_checked is NOT human `verified` (spec.md §9). It "
                     "confirms agreement with an independent published number, not that anyone "
                     f"read the source PDF. Human-verified records remain "
                     f"{sum(c.review_status == 'verified' for c in self.checks)}/{len(self.checks)}.")
        return "\n".join(lines)


def load_reference(ds: Dataset) -> Dict:
    """Load the published-CiPA block reference table that ships with the dataset."""
    if ds.root is None:
        raise FileNotFoundError("dataset has no root path; cannot locate references/")
    path = ds.root / "references" / "cipa_block_reference.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing reference table: {path} "
                                "(run dataset/tools/build_cipa_reference.py)")
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_index(ref: Dict) -> Dict[tuple, Dict]:
    return {(e["drug"].lower(), e["channel"].lower()): e for e in ref["entries"]}


def _check_one(block, index: Dict[tuple, Dict]) -> ChannelCrossCheck:
    ref_entry = index.get((block.drug.lower(), block.channel.lower()))
    rec_ic50 = block.ic50_nm
    rec_hill = block.hill
    if ref_entry is None or ref_entry.get("ic50_nm") is None:
        return ChannelCrossCheck(
            record_id=block.id, drug=block.drug, channel=block.channel,
            tier=block.tier, review_status=block.review_status,
            recorded_ic50_nm=rec_ic50, recorded_hill=rec_hill,
            reference_ic50_nm=None, reference_hill=None,
            ic50_fold_diff=None, hill_abs_diff=None, status=STATUS_NO_REFERENCE)
    ref_ic50 = float(ref_entry["ic50_nm"])
    ref_hill = ref_entry.get("hill")
    fold = _fold(rec_ic50, ref_ic50)
    hill_diff = abs(rec_hill - ref_hill) if ref_hill is not None else None
    if fold <= MATCH_FOLD:
        status = STATUS_MATCH
    elif fold <= MINOR_FOLD:
        status = STATUS_MINOR
    else:
        status = STATUS_DIVERGENT
    return ChannelCrossCheck(
        record_id=block.id, drug=block.drug, channel=block.channel,
        tier=block.tier, review_status=block.review_status,
        recorded_ic50_nm=rec_ic50, recorded_hill=rec_hill,
        reference_ic50_nm=ref_ic50,
        reference_hill=float(ref_hill) if ref_hill is not None else None,
        ic50_fold_diff=fold,
        hill_abs_diff=hill_diff, status=status)


def cross_check(ds: Dataset, drug: Optional[str] = None) -> CrossCheckReport:
    """Diff every channel-block record (or just one drug's) against the published
    CiPA reference table. Returns a :class:`CrossCheckReport`."""
    ref = load_reference(ds)
    index = _reference_index(ref)
    blocks = ds.blocks_for(drug) if drug else ds.channel_blocks
    checks = [_check_one(b, index) for b in blocks]
    checks.sort(key=lambda c: (c.status != STATUS_DIVERGENT, c.drug, c.channel))
    return CrossCheckReport(drug=drug.lower() if drug else None, checks=checks,
                            reference_source=ref.get("source", {}),
                            n_records_total=len(ds.channel_blocks))

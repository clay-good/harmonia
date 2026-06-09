"""Recorded classification performance of the reference kernel against the CiPA
expert risk labels (spec.md §5, §9 — "did the AP model classify the validation
set correctly?").

This is a *derived* quantity, not curated data, so it lives here and is computed
on demand (``harmonia performance``) rather than baked into the dataset records.
It scores the reduced-kernel point classification against the expert consensus
labels — which are ground truth for scoring ONLY, never a Harmonia output.

The numbers are reported honestly, with the full confusion matrix, so the
reduced kernel's limits (it misses the cases that motivated CiPA to move beyond
APD) are visible rather than hidden behind a single accuracy figure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .load import Dataset
from .simulate import assess, RISK_LABELS, DEFAULT_METRIC


@dataclass
class DrugScore:
    drug: str
    cipa_set: str
    expert: str
    predicted: str
    dapd90_pct: float
    tier: str

    @property
    def correct(self) -> bool:
        return self.expert == self.predicted


@dataclass
class PerformanceReport:
    ap_model: str
    cipa_set: str
    scores: List[DrugScore]
    herg_dynamic: bool = False
    metric: str = DEFAULT_METRIC

    @property
    def n(self) -> int:
        return len(self.scores)

    @property
    def n_correct(self) -> int:
        return sum(s.correct for s in self.scores)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else float("nan")

    def confusion(self) -> Dict[Tuple[str, str], int]:
        """(expert, predicted) -> count."""
        c: Dict[Tuple[str, str], int] = {}
        for s in self.scores:
            key = (s.expert, s.predicted)
            c[key] = c.get(key, 0) + 1
        return c

    def adjacent_accuracy(self) -> float:
        """Fraction within one risk category (low<->intermediate<->high), the
        clinically meaningful "off-by-one is not catastrophic" measure."""
        rank = {"low": 0, "intermediate": 1, "high": 2}
        if not self.scores:
            return float("nan")
        ok = sum(abs(rank[s.expert] - rank[s.predicted]) <= 1 for s in self.scores)
        return ok / self.n

    def summary(self) -> str:
        lines = [
            f"classification performance — ap_model={self.ap_model}  "
            f"set={self.cipa_set}  metric={self.metric}  dynamic_hERG={self.herg_dynamic}",
            f"  accuracy: {self.n_correct}/{self.n} = {self.accuracy:.0%}   "
            f"(within-one-category: {self.adjacent_accuracy():.0%})",
            "  confusion (rows=expert, cols=predicted):",
        ]
        header = "            " + "".join(f"{p:>13}" for p in RISK_LABELS)
        lines.append(header)
        conf = self.confusion()
        for e in RISK_LABELS:
            row = "".join(f"{conf.get((e, p), 0):>13}" for p in RISK_LABELS)
            lines.append(f"    {e:>8}{row}")
        misses = [s for s in self.scores if not s.correct]
        if misses:
            lines.append("  misclassified:")
            for s in misses:
                lines.append(f"    {s.drug:14s} expert={s.expert:12s} "
                             f"predicted={s.predicted:12s} (dAPD90 {s.dapd90_pct:+.1f}%)")
        lines.append("  NOTE: reduced-kernel methodology demonstrator, not a qualified "
                     "classifier; never a clinical determination.")
        return "\n".join(lines)


def score(ds: Dataset, ap_model: str = "cipaordv1.0", cipa_set: str = "all",
          herg_dynamic: bool = False, exposure_multiple: float = 4.0,
          metric: str = DEFAULT_METRIC) -> PerformanceReport:
    """Classify every drug in the requested CiPA set and compare to the expert
    label. ``cipa_set`` is 'training', 'validation', or 'all'."""
    scores: List[DrugScore] = []
    for ref in ds.drug_references:
        if cipa_set != "all" and ref.cipa_set != cipa_set:  # type: ignore[attr-defined]
            continue
        drug = ref.drug  # type: ignore[attr-defined]
        if not ds.blocks_for(drug):
            continue
        a = assess(ds, drug, ap_model=ap_model, n_mc=0, metric=metric,
                   exposure_multiple=exposure_multiple, herg_dynamic=herg_dynamic)
        scores.append(DrugScore(
            drug=drug, cipa_set=ref.cipa_set, expert=ref.expert_risk_label,  # type: ignore[attr-defined]
            predicted=a.classification, dapd90_pct=a.dapd90_pct, tier=a.tier))
    scores.sort(key=lambda s: ({"high": 0, "intermediate": 1, "low": 2}[s.expert], s.drug))
    return PerformanceReport(ap_model=ap_model, cipa_set=cipa_set, scores=scores,
                             herg_dynamic=herg_dynamic, metric=metric)

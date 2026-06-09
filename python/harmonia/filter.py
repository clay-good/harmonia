"""Small, composable filters over a Dataset or any iterable of records."""
from __future__ import annotations

from typing import Callable, Iterable, List

from .records import Record

Predicate = Callable[[Record], bool]


def by_kind(kind: str) -> Predicate:
    return lambda r: r.kind == kind


def by_tier(*tiers: str) -> Predicate:
    s = set(tiers)
    return lambda r: r.tier in s


def by_subsystem(*subsystems: str) -> Predicate:
    s = set(subsystems)
    return lambda r: r.subsystem in s


def by_review_status(*statuses: str) -> Predicate:
    s = set(statuses)
    return lambda r: r.review_status in s


def by_drug(drug: str) -> Predicate:
    d = drug.lower()
    return lambda r: getattr(r, "drug", "").lower() == d


def by_channel(channel: str) -> Predicate:
    return lambda r: getattr(r, "channel", None) == channel


def unidentifiable() -> Predicate:
    """Channel-block records whose IC50 is unidentifiable (max block < 60%)."""
    return lambda r: r.kind == "channel_block" and not getattr(r, "identifiable", True)


def apply(records: Iterable[Record], *predicates: Predicate) -> List[Record]:
    out = list(records)
    for pred in predicates:
        out = [r for r in out if pred(r)]
    return out

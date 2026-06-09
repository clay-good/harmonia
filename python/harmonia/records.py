"""Typed views over Harmonia records.

Records are plain JSON (the source of truth). These dataclasses are a thin,
attribute-access wrapper so callers can write ``b.assay_context.max_block_observed_percent``
instead of indexing dicts. The raw dict is always retained as ``.raw`` so no
field is ever lost in the round trip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# The seven cardiac currents Harmonia models, in the canonical order used by the
# reference kernel and every export.
CHANNELS = ["INa", "INaL", "Ito", "ICaL", "IKr", "IKs", "IK1"]

# Block below this maximum-observed fraction makes the IC50 unidentifiable.
IDENTIFIABILITY_BLOCK_THRESHOLD = 60.0


def _get(d: Dict[str, Any], *path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


@dataclass
class AssayContext:
    platform: Optional[str] = None
    temperature_c: Optional[float] = None
    expression_system: Optional[str] = None
    max_block_observed_percent: Optional[float] = None
    holding_protocol: Optional[str] = None

    @property
    def identifiable(self) -> bool:
        """False when the maximum observed block is below the reliability gate."""
        mb = self.max_block_observed_percent
        return mb is None or mb >= IDENTIFIABILITY_BLOCK_THRESHOLD


@dataclass
class Variability:
    fold_range: Optional[float] = None
    n_sources: int = 0
    iqr_nm: Optional[List[float]] = None
    geomean_nm: Optional[float] = None


@dataclass
class Citation:
    key: str
    title: str = ""
    authors: str = ""
    journal: str = ""
    year: Optional[int] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    url: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Citation":
        return cls(
            key=d["key"], title=d.get("title", ""), authors=d.get("authors", ""),
            journal=d.get("journal", ""), year=d.get("year"), doi=d.get("doi"),
            pmid=d.get("pmid"), url=d.get("url"), raw=d,
        )


@dataclass
class Record:
    """Base record. ``raw`` is the full JSON dict."""
    raw: Dict[str, Any]

    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def kind(self) -> str:
        return self.raw["kind"]

    @property
    def subsystem(self) -> str:
        return self.raw["subsystem"]

    @property
    def tier(self) -> str:
        return self.raw["tier"]

    @property
    def primary_citation(self) -> Optional[str]:
        return self.raw.get("primary_citation")

    @property
    def review_status(self) -> str:
        return _get(self.raw, "extraction", "review_status", default="unverified")

    # alias used in the spec cheat sheet: b.extraction.review_status
    @property
    def extraction(self) -> "Extraction":
        return Extraction(self.raw.get("extraction", {}))

    @property
    def notes(self) -> Optional[str]:
        return self.raw.get("notes")

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.id} tier={self.tier} {self.review_status}>"


@dataclass
class Extraction:
    raw: Dict[str, Any]

    @property
    def review_status(self) -> str:
        return self.raw.get("review_status", "unverified")

    @property
    def verified_by(self) -> List[str]:
        return self.raw.get("verified_by", [])

    @property
    def method(self) -> Optional[str]:
        return self.raw.get("method")

    @property
    def notes(self) -> Optional[str]:
        return self.raw.get("notes")


class ChannelBlock(Record):
    @property
    def drug(self) -> str:
        return self.raw["drug"]["name"]

    @property
    def channel(self) -> str:
        return self.raw["channel"]

    @property
    def block_model(self) -> str:
        return self.raw["block_model"]

    @property
    def ic50_nm(self) -> float:
        for p in self.raw.get("parameters", []):
            if p["symbol"] == "IC50":
                return p["value"]["central"]
        raise KeyError(f"{self.id} has no IC50 parameter")

    @property
    def hill(self) -> float:
        for p in self.raw.get("parameters", []):
            if p["symbol"] in ("h", "hill", "Hill"):
                return p["value"]["central"]
        return 1.0

    @property
    def assay_context(self) -> AssayContext:
        ac = self.raw.get("assay_context", {})
        return AssayContext(
            platform=ac.get("platform"),
            temperature_c=ac.get("temperature_c"),
            expression_system=ac.get("expression_system"),
            max_block_observed_percent=ac.get("max_block_observed_percent"),
            holding_protocol=ac.get("holding_protocol"),
        )

    @property
    def source_values(self) -> List[Dict[str, Any]]:
        return self.raw.get("source_values", [])

    @property
    def source_ic50s_nm(self) -> List[float]:
        return [s["ic50_nm"] for s in self.source_values]

    @property
    def variability(self) -> Variability:
        v = self.raw.get("variability", {})
        return Variability(
            fold_range=v.get("fold_range"), n_sources=v.get("n_sources", 0),
            iqr_nm=v.get("iqr_nm"), geomean_nm=v.get("geomean_nm"),
        )

    @property
    def known_failure_modes(self) -> List[Dict[str, Any]]:
        return self.raw.get("known_failure_modes", [])

    @property
    def dynamic_binding(self) -> Optional[Dict[str, Any]]:
        """hERG dynamic-binding kinetics (kon/koff/trapping), if present."""
        return self.raw.get("dynamic_binding")

    @property
    def identifiable(self) -> bool:
        return self.assay_context.identifiable


class APModel(Record):
    @property
    def name(self) -> str:
        return _get(self.raw, "model", "name", default=self.id)

    @property
    def lineage(self) -> str:
        return _get(self.raw, "model", "lineage", default="")

    @property
    def currents(self) -> List[str]:
        return _get(self.raw, "model", "currents", default=[])

    @property
    def conductance_scales(self) -> Dict[str, float]:
        """g_scale_<CHANNEL> model parameters -> {channel: scale}."""
        out: Dict[str, float] = {}
        for p in self.raw.get("model_parameters", []):
            sym = p["symbol"]
            if sym.startswith("g_scale_"):
                out[sym[len("g_scale_"):]] = p["value"]
        return out


class DrugReference(Record):
    @property
    def drug(self) -> str:
        return self.raw["drug"]["name"]

    @property
    def cipa_set(self) -> str:
        return self.raw["cipa_set"]

    @property
    def expert_risk_label(self) -> str:
        return self.raw["expert_risk_label"]

    @property
    def eftpc_nm(self) -> float:
        """Free (unbound) therapeutic Cmax, nM."""
        return self.raw["eftpc_nm"]["central"]

    @property
    def protein_binding(self) -> Optional[Dict[str, Any]]:
        return self.raw.get("protein_binding")

    @property
    def fraction_unbound(self) -> Optional[float]:
        pb = self.raw.get("protein_binding")
        return pb.get("fraction_unbound") if pb else None

    @property
    def total_cmax_nm(self) -> Optional[float]:
        """Total (bound + free) Cmax, nM, if protein binding is recorded."""
        pb = self.raw.get("protein_binding")
        if not pb:
            return None
        if pb.get("total_cmax_nm") is not None:
            return pb["total_cmax_nm"]
        fu = pb.get("fraction_unbound")
        return self.eftpc_nm / fu if fu else None


def wrap(record_dict: Dict[str, Any]) -> Record:
    kind = record_dict.get("kind")
    if kind == "channel_block":
        return ChannelBlock(record_dict)
    if kind == "ap_model":
        return APModel(record_dict)
    if kind == "drug_reference":
        return DrugReference(record_dict)
    return Record(record_dict)

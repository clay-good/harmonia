"""Load the dataset (the source of truth) into an in-memory ``Dataset``.

``harmonia.load()`` discovers the ``dataset/`` directory from, in order:
  1. the ``HARMONIA_DATASET`` environment variable, if set;
  2. an explicit ``path=`` argument;
  3. a packaged copy shipped inside the wheel (``harmonia/_dataset``), if present;
  4. the ``dataset/`` directory of a source checkout (walking up from this file).
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Dict, Iterable, Iterator, List, Optional

from .records import Citation, Record, wrap


class Dataset:
    """An immutable, indexable collection of records plus their citations."""

    def __init__(self, records: List[Record], citations: Dict[str, Citation],
                 root: Optional[pathlib.Path] = None, priors: Optional[Dict] = None):
        self._records = records
        self._by_id = {r.id: r for r in records}
        self.citations = citations
        self.root = root
        self.priors = priors or {}

    # -- container protocol -------------------------------------------------- #
    def __getitem__(self, record_id: str) -> Record:
        return self._by_id[record_id]

    def __contains__(self, record_id: object) -> bool:
        return record_id in self._by_id

    def __iter__(self) -> Iterator[Record]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def get(self, record_id: str, default=None):
        return self._by_id.get(record_id, default)

    @property
    def records(self) -> List[Record]:
        return list(self._records)

    # -- convenience views --------------------------------------------------- #
    def of_kind(self, kind: str) -> List[Record]:
        return [r for r in self._records if r.kind == kind]

    @property
    def channel_blocks(self) -> List[Record]:
        return self.of_kind("channel_block")

    @property
    def ap_models(self) -> List[Record]:
        return self.of_kind("ap_model")

    @property
    def drug_references(self) -> List[Record]:
        return self.of_kind("drug_reference")

    @property
    def populations(self) -> List[Record]:
        return self.of_kind("population")

    def population(self, name: str) -> Optional[Record]:
        """Lookup a population record by id or short name."""
        if name in self._by_id:
            return self._by_id[name]
        full = f"population.{name}"
        return self._by_id.get(full)

    def drugs(self) -> List[str]:
        names = {r.drug for r in self.channel_blocks}  # type: ignore[attr-defined]
        return sorted(names)

    def blocks_for(self, drug: str) -> List[Record]:
        """All channel-block records for a drug (case-insensitive)."""
        d = drug.lower()
        return [r for r in self.channel_blocks
                if r.drug.lower() == d]  # type: ignore[attr-defined]

    def drug_reference(self, drug: str) -> Optional[Record]:
        d = drug.lower()
        for r in self.drug_references:
            if r.drug.lower() == d:  # type: ignore[attr-defined]
                return r
        return None

    def citation(self, key: str) -> Optional[Citation]:
        return self.citations.get(key)

    def prior(self, key: str):
        """Lookup a v0.2 inference prior by id (the prior registry, spec v0.2 sec 7)."""
        return self.priors.get(key)


# --------------------------------------------------------------------------- #
# Dataset directory discovery
# --------------------------------------------------------------------------- #
def find_dataset_dir(path: Optional[os.PathLike] = None) -> pathlib.Path:
    if path is not None:
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(f"dataset path does not exist: {p}")
        return p

    env = os.environ.get("HARMONIA_DATASET")
    if env:
        p = pathlib.Path(env)
        if not p.exists():
            raise FileNotFoundError(f"HARMONIA_DATASET points to a missing path: {p}")
        return p

    # packaged copy inside the installed wheel
    packaged = pathlib.Path(__file__).resolve().parent / "_dataset"
    if (packaged / "records").is_dir():
        return packaged

    # source checkout: walk up looking for a dataset/ with records/
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "dataset"
        if (candidate / "records").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not locate the Harmonia dataset/ directory. Set HARMONIA_DATASET "
        "or pass load(path=...).")


def _read_json(p: pathlib.Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def load(path: Optional[os.PathLike] = None) -> Dataset:
    """Load every record + citation under the dataset directory."""
    root = find_dataset_dir(path)

    records: List[Record] = []
    for p in sorted((root / "records").glob("*.json")):
        records.append(wrap(_read_json(p)))

    citations: Dict[str, Citation] = {}
    cdir = root / "citations"
    if cdir.is_dir():
        for p in sorted(cdir.glob("*.json")):
            c = Citation.from_dict(_read_json(p))
            citations[c.key] = c

    from .infer import load_priors
    priors = load_priors(root)

    return Dataset(records, citations, root=root, priors=priors)


def iter_record_files(root: Optional[os.PathLike] = None) -> Iterable[pathlib.Path]:
    base = find_dataset_dir(root)
    return sorted((base / "records").glob("*.json"))

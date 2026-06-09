# Changelog

All notable changes to Harmonia are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Standalone SED-ML model source.** `exports/sedml/<model>.sedml` referenced
  `source="model.cellml"`, but in the export layout the CellML model lives in a
  sibling directory — so a SED-ML tool opening the standalone protocol could not
  find its model. It now points at `../cellml/<model>.cellml` (the flattened
  COMBINE archive still uses `model.cellml`, which is correct there). Regression-
  tested by resolving the `source` against the written files.

### Added — SED-ML / COMBINE export integrity
- **`sedml.reference_violations`**: every internal SED-ML cross-reference
  (task→model/simulation, variable→task, curve→dataGenerator) must resolve.
- **`combine.manifest_violations`**: the COMBINE `.omex` manifest must list
  exactly the archive's files, with exactly one master entry that exists.
- `harmonia export --all` now also runs CellML unit conformance, SED-ML reference
  resolution, and OMEX manifest consistency (alongside the three round trips) and
  exits non-zero on any drift.

### Added — test coverage for the previously-unguarded headline surfaces
- **Dashboard data contract** (`tests/test_dashboard.py`): byte-compiles
  `dashboard/app.py` and exercises the exact simulate/load API the dashboard
  consumes (every attribute and dict key it reads, at tiny Monte-Carlo), so
  API drift fails in CI instead of silently breaking the spec's headline feature
  (§6). A full headless `AppTest` run is impractical (minutes per render), so the
  contract test is the robust guard.
- **CLI coverage** for the `combo` (drug-combination) and `population`
  subcommands, which had none.

### Added — export integrity (closes a spec/implementation gap)
- **ODE round-trip validation** (`registry.roundtrip_ode`): the model AST that
  every CellML/SBML/Myokit export is rendered from now carries a pure-Python
  evaluator (`Expr.eval`) and is independently re-integrated by `simulate_spec`
  with the kernel's own solver settings. It reproduces the reference-kernel action
  potential to ≈1e-7 relative on the V trace (well inside the 1e-4 the spec/
  architecture promised but never enforced), so the exported *equations* — not
  just the constants — provably match the numeric oracle. Tested across all three
  AP models, drug-free and under block, with a drift-detection guard.
- `harmonia export --all` now **round-trip-validates** the artifacts it writes
  (CiPA numeric + parameter + ODE round trips) and exits non-zero on any drift,
  making "exports are generated, never hand-edited" enforced rather than asserted.

### Added — Phase F (release hardening)
- **Executable reference notebook** [`notebooks/01_flip_frequency.ipynb`](notebooks/01_flip_frequency.ipynb)
  reproducing the headline input-variability → classification-flip analysis from
  the dataset + reference kernel. It is **executed in CI under `nbmake`** (the
  family convention) and carries inline assertions, so a clean run is a test, not
  just a demo.
- **Declaration-level CellML-2.0 unit-conformance check**
  (`harmonia.export.cellml.conformance_violations`): every exported model is
  verified in CI to declare units on every variable and `<cn>` literal, with no
  dangling unit references. (Full dimensional validation and the Myokit/OpenCOR
  cross-check against the canonical ORd CellML remain an optional local step.)
- **`.zenodo.json`** release metadata for archival + DOI minting on publish.
- **`CHANGELOG.md`** (this file).

### Changed
- CI installs the `notebooks` extra and runs `pytest --nbmake notebooks/`.
- `cellml.py` docstring sharpened to separate the now-machine-checked
  declaration-level conformance from full dimensional/OpenCOR validation.

## [0.1.0] — 2026-06-08

First tagged development release. The CiPA-spine proarrhythmia pipeline, end to
end, covering roadmap phases A–E:

- **A — CiPA spine:** channel-block records + the reduced O'Hara-Rudy-lineage AP
  kernel + the risk metric for the 12 CiPA training drugs, with CellML / Myokit /
  SBML / SED-ML / CiPA-input / COMBINE `.omex` export, round-trip validation, and
  the risk-uncertainty (flip) view.
- **B — Dynamic hERG + validation:** Langmuir `kon`/`koff` hERG binding with
  trapping; the 16 CiPA validation drugs (28 compounds total); recorded
  classification performance with the full confusion matrix.
- **C (start) — Discriminating qNet:** a shape-dependent Na-Ca exchanger
  (excluded from the qNet sum) makes qNet sensitive; qNet is the default metric.
- **D — Exposure layer:** free ↔ total plasma conversion via protein binding
  (composable with a Hypnos PK trajectory); drug-combination (polypharmacy)
  assessment.
- **E — Populations (hypothesis-tier):** population-of-models risk spread,
  shipped non-predictive (Tier D, "NOT FOR PREDICTION").

[Unreleased]: https://github.com/clay-good/harmonia/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/clay-good/harmonia/releases/tag/v0.1.0

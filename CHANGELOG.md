# Changelog

All notable changes to Harmonia are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.1] — 2026-06-10

### Changed — type-safety hardening: the `py.typed` contract is now enforced in CI
The package shipped a `py.typed` marker (advertising itself as fully typed to
downstream tools) but was never type-checked, and `mypy` reported 33 errors. This
release closes that gap with **no runtime behavior change** and adds **mypy** as a
CI gate alongside `ruff`, so the typed surface tools depend on cannot silently
drift.

- **`Dataset` convenience views now return their concrete record subtypes** —
  `channel_blocks → List[ChannelBlock]`, `ap_models → List[APModel]`,
  `drug_references → List[DrugReference]`, `populations → List[Population]`, and
  `population()` / `drug_reference()` their `Optional[...]` counterparts (via
  `isinstance` narrowing). This is a type-level refinement only: every value
  returned is unchanged at runtime, and it removes the scattered
  `# type: ignore[attr-defined]` workarounds in `load`, `cli`, `performance`,
  `registry`, and `cipa_inputs`.
- **Implicit-`Optional` defaults made explicit** (`build_model_spec`'s
  `conductance_scales` / `block`; the reference kernel's `herg` parameter), and
  `Dataset.citation()` / `find_dataset_dir()` / `load()` widened to accept the
  `Optional[str]` / `str`-or-`PathLike` values their callers already pass.
- **Genuine `Optional`-narrowing guards** added where a value could be `None` at
  the type level (the censored-inference max-block read; the dynamic-hERG kinetics
  closure), and the Sobol sampler dictionaries explicitly typed.
- New `[tool.mypy]` config (checks `python/harmonia`, `no_implicit_optional`,
  `warn_unused_ignores`; missing-stub third-party libs — `scipy`, `jsonschema`,
  `libsbml`, `libcellml` — declared ignored), `mypy>=1.8` added to the `dev`
  extra, and a **Type-check (mypy)** step in CI. Runtime support down to Python 3.9
  is still guaranteed by the pytest matrix.

### Fixed
- README: the honest verified-record count was stale (`0/100` → `0/104`).

## [0.5.0] — 2026-06-09

### Added — experimentally-calibrated populations (Britton 2013), completing Phase E (spec v0.5)
The `populations` subsystem could sample a prior conductance cloud (v0.1
`illustrative_v0`) and recenter it on a disease mean (v0.3 LQTS), but it accepted
**every** draw — including the implausible tail where an extreme conductance
combination yields a drug-free action potential no real myocyte would show
(triangulation reaching 250–300 ms against a ~42 ms baseline). v0.5 adds the
landmark **experimentally-calibrated populations-of-models** method
([Britton et al. 2013](https://doi.org/10.1073/pnas.1304382110)): a candidate
myocyte is admitted only if its **drug-free** AP biomarkers are physiologically
plausible. Fully backward-compatible — a population with no `calibration` block is
byte-identical to before (the draw logic was extracted to a shared
`_draw_multiplier` that preserves the exact RNG sequence).

- **`calibration`** (optional, on a `population` record): accepted ranges for
  drug-free biomarkers (`apd90_ms`, `vrest_mv`, `vpeak_mv`, `triangulation_ms`) in
  the kernel's own units, a `max_oversample` termination guard, and the cited method
  (spec v0.5 §2–3). A candidate is accepted iff it repolarizes and every biomarker is
  in range; the accepted myocyte's drug-free beat is cached and reused as the APD90
  baseline, so calibration adds no extra simulation to that path.
- **`calibrate_population`** primitive + **`calibrated_v0`** record: the
  `illustrative_v0` variability cloud admitted through the filter.
  `harmonia population <drug> --population calibrated_v0` runs it. Empirically ≈92%
  of drug-free myocytes are admitted; the **triangulation** bound is the dominant
  filter (high drug-free triangulation is itself a repolarization-instability marker
  — exactly the abnormality calibration should remove). The assessment reports the
  acceptance rate and per-biomarker rejection counts.
- **Strictly hypothesis-tier, never predictive.** The acceptance ranges are
  **kernel-plausibility bounds** (bounds on *this* reduced kernel's biomarkers,
  bracketing its physiological bulk) — the *methodology* is Britton 2013, the
  *numbers* are not a fit to patient data. Every calibrated assessment is **Tier D**
  and stamped NOT FOR PREDICTION; the qNet/APD thresholds stay the healthy reference.
- New `tests/test_calibrated_populations.py` (acceptance correctness, rejection
  happens, triangulation dominance, tail-removal, APD90 path, Tier-D / non-prediction,
  determinism, byte-identical uncalibrated path). Figure
  `docs/img/calibrated_populations.png`; spec
  [`docs/specs/v0.5-calibrated-populations.md`]; README populations section + roadmap.

## [0.4.0] — 2026-06-09

### Added — the cqInward inward-charge biomarker, completing the computable spec §3 metrics (spec v0.4)
spec.md §3 names six TdP-risk biomarkers; Harmonia shipped qNet (the classifier), APD90,
triangulation, and EAD detection, but **cqInward — named in the spec and listed as a kernel
biomarker — was never actually computed.** v0.4 implements it and corrects the docs. It is a
diagnostic, not a classifier, so it adds no threshold and disturbs no calibrated number
(every existing qNet/APD/flip value is byte-identical).

- **cqInward** (`RiskAssessment.cqinward` + `cqinward_distribution`) — the CiPA inward-charge
  biomarker ([Dutta et al. 2017](https://doi.org/10.3389/fphys.2017.00616): "the change in the
  amount of charge carried by INaL and ICaL"): the control-normalized average of the two
  inward-current charge ratios, `½·(qNaL_drug/qNaL_ctrl + qCaL_drug/qCaL_ctrl)`, where
  `qX = ∫ I_X dt` over the beat. It isolates the *inward* side of CiPA's inward/outward
  balance hypothesis. Dimensionless and self-normalizing (1 at no drug; no kernel threshold),
  and **propagated through the same Monte-Carlo as qNet**, so the assessment reports a
  distribution, not a point.
- The mechanism validates it: a pure **ICaL/INaL blocker reduces** inward charge (cqInward < 1
  — the protective multichannel mechanism; verapamil ≈0.81), a pure **IKr blocker prolongs**
  the AP and *raises* it (cqInward > 1; dofetilide ≈1.18). Surfaced in the `RiskAssessment`
  summary, the CLI `simulate` output, and the Streamlit dashboard alongside triangulation —
  always a diagnostic readout, never the verdict.
- The reference kernel now computes the per-beat INaL/ICaL charges (`BeatResult.q_nal`,
  `q_cal`); `_cached_baseline` carries the drug-free control charges (the cqInward
  denominators). EAD occurrence (structurally unreachable in the reduced kernel) and the
  electromechanical window (needs a mechanical model) remain honestly out of reach.
- Spec [`docs/specs/v0.4-cqinward-biomarker.md`]; tests `tests/test_cqinward.py` (control
  identity, ICaL/INaL/IKr block signs, distribution propagation under both uq engines, and a
  non-drift guard); dashboard data-contract extended; README biomarker section + kernel rows.

## [0.3.0] — 2026-06-09

### Added — disease & genetic population backgrounds (LQTS), completing spec §3 (spec v0.3)
v0.1's `populations` subsystem propagated inter-individual **variability** (a cloud of
conductance CVs around the healthy mean) but never the other half spec.md §3 names: a
disease or genetic **background** — a systematic *shift* of a current. v0.3 adds it as the
minimal honest field and ships the three canonical congenital long-QT channelopathies.
Fully backward-compatible: the variability-only `illustrative_v0` population is
byte-identical (the disease shift defaults to 1 and consumes no RNG).

- **`conductance_scale`** (optional, on a `population` record): a per-current **mean**
  multiplier applied *under* the existing `conductance_cv` spread —
  `g = s_c · λ · g_healthy` (spec v0.3 §2). The clinical basis is **repolarization
  reserve** ([Moss & Kass 2005](https://doi.org/10.1172/JCI25537)): a latent channelopathy
  has already spent the redundancy that makes single-current block tolerable, so the same
  drug block can be torsadogenic against the disease background.
- **Three disease population records**: `lqt1` (*KCNQ1* → IKs ×0.5), `lqt2` (*KCNH2*/hERG →
  IKr ×0.5), `lqt3` (*SCN5A* → INaL ×2.0). `harmonia population <drug> --population lqt2`
  re-evaluates a drug's risk distribution against the reduced-reserve background. The
  reduced kernel reproduces the textbook ordering — LQT2 prolongs the AP most (direct IKr
  loss), LQT3 lowers qNet most (the inward INaL shift) — so the *mechanism*, not a tuned
  number, does the work. A low-risk drug like ranolazine goes from ~5% susceptible (healthy)
  to ~24–39% (LQTS); dofetilide on LQT2 reaches ~88%.
- **Strictly hypothesis-tier, never predictive.** The magnitudes are illustrative
  heterozygous-scale shifts (NOT genotype-calibrated); the qNet/APD thresholds stay the
  *healthy* reference; every disease-population assessment is **Tier D** and stamped
  NOT FOR PREDICTION, with an explicit disease-background warning. The schema pins
  `predictive: false`.
- Citation minted: `moss-2005`. New `tests/test_disease_populations.py` (mean-shift
  correctness, channelopathy ordering, susceptibility increase, Tier-D / non-prediction
  guard, healthy-population backward-compat). README + figure
  (`docs/img/disease_populations.png`) + spec [`docs/specs/v0.3-disease-populations.md`].

## [0.2.1] — 2026-06-09

### Added — raw dose-response regime + calibration of the inference (completes spec v0.2 C-UQ-5, §9)
Closes the last open piece of the Bayesian dose-response thread: the raw regime and
the two §9 calibration gates that v0.2.0 deferred. Fully backward-compatible — every
record carrying no raw data is left on the byte-identical v0.2.0 summary path
(confirmed: dofetilide hERG posterior unchanged at 0.7121 ± 0.1109).

- **Raw regime** (`harmonia.fit_dose_response`, spec §2.1): a source carrying raw
  `(concentration, fractional_block, sem)` points now has its `(IC50, Hill)` —and the
  *genuine* fit uncertainty—**inferred from the curve** via a 2-D grid Bayesian fit
  (truncated-normal or beta likelihood, stable logistic Hill form), instead of
  transcribing a fitted IC50 as a point with an assumed spread. Recovers a synthetic
  IC50/Hill to within a few percent.
- **Heteroscedastic hierarchical pooling**: the collapsed between-lab marginal is
  generalized (Sherman-Morrison) to per-source variances `tau^2 + v_s`, so a
  precisely-fit raw source and a loosely-transcribed summary source are weighted by
  their *actual* uncertainty. Reduces exactly to the homoscedastic form when every
  `v_s = 0`, so the existing 68 records do not drift. The optional per-source
  `fit_sd_log10` (summary regime) now feeds this too.
- **Simulation-based calibration** (`harmonia.simulation_based_calibration`, §9): data
  simulated from the prior and re-inferred yields **rank-uniform** posteriors
  (chi-square uniformity p ≈ 0.6) — the standard proof that the inference is correctly
  *implemented*, not merely plausible.
- **Posterior coverage** (`harmonia.posterior_coverage`, §9): the 90% credible interval
  covers the truth ~90% of the time (measured 0.905) on synthetic data; the 50%
  interval covers ~50% (0.512).
- `python dataset/tools/build_posteriors.py --validate` runs both gates as a CLI
  diagnostic. New tests in `tests/test_infer_raw.py` (raw recovery, noise→width
  monotonicity, end-to-end raw inference, SBC uniformity, coverage, backward-compat).

## [0.2.0] — 2026-06-09

### Added — Bayesian dose-response uncertainty quantification (spec v0.2, roadmap Phase C)
Where v0.1 made input variability a first-class **field**, v0.2 makes it a
first-class **inference**: an IC50's spread stops being a number transcribed from a
table and becomes a posterior, derived under a declared prior. The v0.1
method-of-moments path is preserved exactly (`uq="moments"`, the default), so every
v0.1 number reproduces; the new machinery is opt-in via `uq="bayes"`.

- **`harmonia.infer`** — a hierarchical Bayesian inference of the per-channel
  `(IC50, Hill)` posterior, implemented as an exact **direct (grid + conjugate)
  sampler** in pure NumPy (no MCMC funnel, no new dependency): the between-lab SD
  `tau` is drawn from its collapsed 1-D marginal (the channel mean integrated out in
  closed form), the true log-IC50 from a conjugate Normal, and the censored case from
  a 1-D grid. Draws are i.i.d. and exactly from the posterior; `rhat`/`ess` are
  reported and trivially satisfied. Closes the three gaps in the v0.1 sampler:
  - **Partial pooling.** A single-source channel no longer gets the hard-coded
    `DEFAULT_SINGLE_SOURCE_SIGMA`; its spread is `tau_pop`, the between-lab SD
    *learned across every multi-source channel in the dataset* (`learn_tau_pop`) — a
    magic constant becomes an inferred, citable quantity that sharpens as the dataset
    grows.
  - **Hill uncertainty.** The Hill coefficient carries a posterior and propagates,
    instead of being fixed at a point value.
  - **Censoring.** A sub-60%-block channel is no longer discarded; the max-block
    observation becomes a **one-sided (probit-censored) likelihood** that bounds the
    IC50 from below near the recovered top tested dose, with the Hill coefficient
    marginalized over its prior — yielding a proper but wide posterior with a heavy,
    prior-shaped right tail. It is **prior-dominated by construction** and still
    **Tier-D-capped** (the reliability gate is preserved; v0.2 only stops throwing
    the information away).
- **`assess(..., uq="bayes")`** — the posterior-predictive drop-in. The headline
  `classification_flip_frequency` samples the **true-value** posterior; a new
  `reproducibility_flip_frequency` samples the **new-lab predictive** ("how much would
  a fresh replication move the call?"). Censored channels now *contribute*
  (`censored_channels`) instead of being excluded; prior-dominated channels are
  flagged (`prior_dominated_channels`). Exposed on the CLI as `--uq bayes`.
- **The prior registry** (`dataset/priors/harmonia-ic50-prior-v1.json` + its schema):
  every prior is a version-pinned, citable, **non-predictive** object referenced by id;
  each posterior reports its `prior_sensitivity` (fraction of posterior variance from
  the genuinely-subjective priors, probed by re-inference under a widened prior — the
  empirical-Bayes `tau_pop` is held fixed), and a high value drives the prior-dominance
  flag. `harmonia validate` schema-validates every prior and enforces
  `predictive == false` (no prior may carry a risk conclusion). `harmonia priors` lists
  the registry.
- **`harmonia.posterior` / `harmonia infer <drug>`** — inspect the per-channel
  posteriors, their `identifiability_score` (continuous identifiability readout, spec
  v0.2 §6) and sampler diagnostics.
- **Variance-based (Sobol) sensitivity** — `flip_sensitivity(..., method="sobol")`
  generalizes the one-at-a-time attribution to first-order `S_i` (Janon estimator),
  total-effect `S_Ti` (Jansen estimator), and the **interaction load** `S_Ti − S_i`
  that OAT cannot see, each with a bootstrap Monte-Carlo standard error. The
  dominant-driver recommendation is now interaction-aware. `harmonia sensitivity
  --sobol`.
- **Schema delta** (backward-compatible): optional `inference` block + `dose_response`
  / `fit_sd_log10` fields on source values (the raw-regime upgrade path). Every
  existing record validates and simulates unchanged.
- **Citations minted:** `johnstone-2016` (hierarchical Bayesian IC50 inference, the
  methodological precedent) and `elkins-2013` (high-throughput screening variability).
- **`dataset/tools/build_posteriors.py`** regenerates / inspects every channel's
  posterior summary. Design note: posteriors are recomputed on demand (milliseconds)
  rather than cached into the 68 record files — the source of truth stays
  `(source data + prior)`, and `tests/test_infer.py` asserts deterministic
  reproducibility (run twice → byte-identical) without a cross-platform git-diff gate.
- **Notebook** [`notebooks/02_bayesian_uq.ipynb`](notebooks/02_bayesian_uq.ipynb),
  executed in CI under `nbmake`, plus `tests/test_infer.py`, `tests/test_sobol.py`,
  and `tests/test_uq_assess.py` (reduction/non-drift, censoring, single-source pooling,
  prior-sensitivity, Sobol consistency, and the moments-path backward-compatibility
  guard).

### Added — CellML validated against the canonical CellML 2.0 library (libCellML)
- **`cellml.validity_violations`** runs **libCellML's Parser + Validator** (the
  canonical CellML 2.0 library) over every exported model, checking the full
  ruleset — MathML correctness, variable/units references, interface consistency,
  duplicate/cyclic definitions — well beyond the existing declaration-level
  `conformance_violations`. It catches errors the hand-rolled check cannot (e.g. a
  MathML `<ci>` naming an undeclared variable), and is the lightweight,
  no-simulation-engine part of the spec-§6 "cross-check against the canonical
  CellML." Wired into `harmonia export --all` and CI next to the SBML gate;
  `libcellml` added to the `dev` extra (skips gracefully if absent). Tested across
  all three AP models with a negative test (an undeclared math variable is
  caught).
- The exported CellML **model** is valid CellML 2.0 (0 issues). The embedded
  `<rdf:RDF>` MIRIAM annotation (clinicalUse / tier / DOIs, spec §7) is a
  foreign-namespace metadata *island* — CellML 2.0 has no blessed annotation
  wrapper, unlike SBML's `<annotation>` — so it is set aside before validation
  (the identical RDF also travels in the COMBINE archive's `metadata.rdf` and the
  SBML `<annotation>`). Documented in the builder and the README.

### Added — SBML validated against the canonical validator + units declared
- **`sbml.consistency_violations`** runs **libSBML's `checkConsistency`** (the
  canonical SBML validator) over every exported model and returns any
  ERROR/FATAL-severity problem. Wired into `harmonia export --all` and CI, it
  turns "SBML L3v2 → COPASI/Tellurium/BioModels" into a *verified* claim rather
  than an asserted one — the SBML analog of the CellML declaration-level unit
  check. `python-libsbml` is now in the `dev` extra; the check skips gracefully
  if libSBML is absent (CI installs it). Tested across all three AP models, with
  a negative test (a dangling `rateRule` target is caught) so the gate can't be a
  no-op.
- **SBML exports now declare units.** Every `<parameter>` carries a `units`
  attribute and the model a `timeUnits`, backed by a `<listOfUnitDefinitions>`
  that mirrors the CellML export's custom units (ms, mV, µA/µF, mS/µF, µF/cm²).
  This drops libSBML's warnings from **257 → 27** (and eliminates every "no units
  defined" warning); the residual warnings are the same declaration-level
  limitation the CellML export documents (numeric `<cn>` literals are
  dimensionless, not fully dimensionally audited). Exports regenerated.

### Added — triangulation biomarker surfaced (completes a spec §3 readout)
- The reference kernel already computed **triangulation** (APD90 − APD50, the *T*
  in the TRIaD proarrhythmia profile) on every beat but discarded it. `assess`
  now reports it on the `RiskAssessment` (`triangulation_ms` and the drug-free
  `baseline_triangulation_ms`), in the CLI `simulate` summary, and in the
  dashboard headline. hERG block widens triangulation monotonically (the textbook
  signature: late repolarization is prolonged more than early), e.g. dofetilide
  ≈71 ms vs a ≈36 ms drug-free baseline. It is an honest *diagnostic readout*,
  never a second classification — the high/intermediate/low call stays with qNet
  (or ΔAPD90). Tested at the kernel level (monotonicity under IKr block) and the
  assess level; the dashboard data-contract test tracks the two new fields. The
  EAD biomarker is deliberately *not* surfaced: the reduced kernel repolarizes
  monotonically even when massively prolonged and structurally cannot generate an
  EAD, so an always-zero "EAD frequency" would be misleading.

### Added — flip-sensitivity attribution (new analysis)
- **`harmonia.flip_sensitivity`** (+ `harmonia sensitivity <drug>` CLI and a
  dashboard panel) attributes the classification-flip frequency to each channel's
  IC50 spread: a "solo-flip" main effect (only that channel varies) and a
  "frozen-flip" total effect (that channel pinned, others vary), using common
  random numbers across scenarios. It surfaces the **dominant uncertain input** —
  the IC50 to pin down first to stabilize the safety call — and honestly flags
  single-source channels whose sensitivity is prior-driven rather than measured.
  This operationalizes the project's thesis (input variability governs the call)
  one step further: from *whether* the call is unstable to *which input* drives
  it. Still an uncertainty attribution, never a verdict. Tested (incl. a
  dashboard-contract assertion and a CLI smoke test).

### Added — lint gate (housekeeping)
- **Ruff** is now a CI gate (`ruff check .`, config in `pyproject.toml`; added to
  the `dev` extra). Selects Pyflakes + pycodestyle (E/F/W); does not enforce hard
  line length or forbid the deliberate one-line setup style in the figure/test
  scripts. Fixed all findings it surfaced — 12 dead imports across the package,
  2 placeholder-less f-strings, a lambda-assignment, and unused imports in the
  notebook and tests.

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

[Unreleased]: https://github.com/clay-good/harmonia/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/clay-good/harmonia/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/clay-good/harmonia/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/clay-good/harmonia/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/clay-good/harmonia/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/clay-good/harmonia/releases/tag/v0.1.0

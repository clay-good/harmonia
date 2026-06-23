# Changelog

All notable changes to Harmonia are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.1] — 2026-06-23

### Fixed — reconciled the two channels the v0.8 cross-check flagged (data correction)
The v0.8 machine cross-check flagged two channel-block records as diverging >5× from the
published CiPA value. Both were reviewed against the **raw Crumb-2016 dose-response** (the
cited source) and corrected — the cross-check found real errors, and a human confirmed and
reconciled them against the primary source (records stay `unverified`; this is a data fix,
not a `verified` promotion).

- **`channel_block.cisapride.ical`** — the raw Crumb-2016 data shows ≤2.5% calcium block at
  the 125 nM top dose tested, so the IC50 is **unidentifiable**. Corrected to the Li-2017
  extrapolated fit (9,258,075 nM) with `max_block` 62 → 3; now **Tier D** with the
  unidentifiable failure mode (like `ranolazine.ical`). The prior 9,258 nM was a
  dropped-exponent unit error.
- **`channel_block.terfenadine.inal`** — the raw data shows ~15% late-sodium block at the
  800 nM top dose, so the IC50 is **unidentifiable**. Corrected 2,000 → 20,056 nM (Li-2017
  fit), `max_block` 78 → 15; now **Tier D**.
- **Honest performance impact:** correcting `terfenadine.inal` (less *protective* late-Na
  block, faithfully represented) pushed the reduced kernel to over-predict terfenadine to
  `high` — an adjacent (one-category) error. qNet exact accuracy: training 10/12 → **9/12**,
  all-28 17/28 → **16/28**; within-one-category stays **100%** (still zero two-category
  errors). A more faithful dataset, a slightly lower score — the trade the project exists to
  make visible.
- The committed dataset now has **zero** cross-check divergences;
  `tests/test_crosscheck.py` guards that it stays clean and that both channels are Tier-D
  unidentifiable.

## [0.8.0] — 2026-06-23

### Added — machine cross-check against the published CiPA reference (spec v0.8)
Every Harmonia record ships `review_status: "unverified"` (the values are
literature-derived but nobody has opened the source PDF inside Harmonia, and an LLM
never promotes a record to `verified` — spec §9). That is honest, but it left a reader
with no automated signal of whether a transcribed IC50 is even in the right ballpark.
v0.8 adds a second, **independent** rendering of the canonical CiPA block table and diffs
every channel-block record against it — a new, honestly-weaker provenance signal,
`machine_cross_checked`, that sits between "unverified" and "verified".

- **`dataset/references/cipa_block_reference.json`** — IC50 (nM) + Hill per drug × current
  for the 12 CiPA training drugs (65 entries), transcribed from **Li et al. 2017**
  (the `li-2017` citation) via the FDA/CiPA machine-readable table. Built deterministically
  by **`dataset/tools/build_cipa_reference.py`** (byte-reproducible; new CI gate). The
  GPL-3.0 FDA/CiPA *file* is **not** vendored — only the uncopyrightable numeric facts, with
  attribution to Li 2017 under Harmonia's CC-BY-4.0 dataset license.
- **`harmonia.cross_check(ds, drug=None)` → `CrossCheckReport`** — per-record IC50/Hill
  fold-difference with a four-way status: `match` (≤2×), `minor` (≤5×, within documented
  inter-lab spread), `divergent` (>5×, flagged for human review), `no_reference` (outside the
  training-drug table). `machine_cross_checked := status ∈ {match, minor}`. The 5× divergence
  line mirrors the dataset's own `fold_range > 5` failure-mode trigger.
- **CLI `harmonia crosscheck [drug] [--strict]`** and a `MACHINE-CROSS-CHECKED:` line in
  `harmonia info` (printed separately from, and never conflated with, the `VERIFIED:` count).
  `--strict` exits non-zero if any record diverges.
- **It found real defects on first run** (the point of building it): `cisapride.ical` recorded
  9 258 nM vs published 9 258 075 nM (exactly 1000× — a dropped-exponent unit error) and
  `terfenadine.inal` 2 000 vs 20 056 nM (~10×). Both are *flagged for human reconciliation*,
  not auto-corrected — `machine_cross_checked` is **not** `verified`, and no record is mutated.

### Honesty posture
The cross-check changes no risk number, no threshold, and no record; the `verified` count
remains 0/104. A `match` confirms only that two independent renderings of one published number
agree — never that a human read the source PDF, and never a safety claim. Coverage
(`no_reference`) is reported, never hidden. See [spec v0.8](docs/specs/v0.8-machine-crosscheck.md).

## [0.7.0] — 2026-06-10

### Added — Monte-Carlo confidence intervals on the flip frequency (spec v0.7)
Harmonia's thesis is *never report a number without its uncertainty* — yet the headline
**classification-flip frequency**, the number the whole dataset exists to compute, was
itself reported as a bare point estimate. It is a Monte-Carlo **binomial proportion**
`k/n_mc` with sampling error that shrinks only as draws are added, so v0.7 turns the
project's own rule on its own output: every reported flip frequency now carries a
**Wilson score 95% confidence interval**. (The Sobol indices already reported bootstrap
SEs; the headline number now matches that standard.)

- **`wilson_interval(k, n, z=Z95)` and `flip_ci(freq, n)`** — public, pure helpers
  (`harmonia.wilson_interval`, `harmonia.flip_ci`). The **Wilson** interval is used
  deliberately over the normal/Wald approximation: it stays inside `[0, 1]` and is
  non-degenerate at the extremes `k=0` / `k=n`, exactly where flip frequencies live (a
  tight HIGH blocker flips `0/200` ⇒ CI `[0, 1.9%]`, an honest upper bound, not `[0,0]`).
  `n ≤ 0` ⇒ `(nan, nan)`.
- **CI on every reported Monte-Carlo proportion:** `RiskAssessment.flip_ci`,
  `RiskAssessment.reproducibility_flip_ci` (`uq="bayes"`), `CombinationAssessment.flip_ci`,
  `FlipSensitivity.all_vary_flip_ci`, and `PopulationAssessment.susceptible_fraction_ci`.
  Each `summary()` prints it inline (`36% (95% CI 30%–43%, 200 MC draws)`); the CLI,
  dashboard (metric tooltips + caption), and notebook `01_flip_frequency.ipynb` surface it.
- **Purely additive — provable non-drift.** `classification_flip_frequency`,
  `susceptible_fraction`, the class-probability distribution, every qNet/ΔAPD90/cqInward
  value, and all calibrated thresholds are byte-identical with v0.6.x; the `n_mc=0`
  point-estimate path reports `(nan, nan)`. `tests/test_flip_ci.py` (23 tests): Wilson
  math vs textbook, bounds, extremes, `1/√n` convergence, point-bracketing on every
  surface, and the non-drift guarantee. Suite 188 → 211.
- Spec `docs/specs/v0.7-flip-frequency-ci.md`; README section "The flip frequency is
  itself an estimate"; no dataset, model, or threshold change (exports move only the
  `datasetVersion` stamp to 0.7.0).

## [0.6.0] — 2026-06-10

### Added — the published CiPA dynamic-hERG binding kinetics (spec v0.6)
The most prominent remaining roadmap thread was "full CiPA Markov hERG + published
optimized kinetics." v0.6 sources the **real** kinetics — the
[Li et al. 2017](https://doi.org/10.1161/CIRCEP.116.004628) IKr-Markov drug-binding
model, optimized per drug and validated in
[Li et al. 2019](https://doi.org/10.1161/CIRCULATIONAHA.118.035230) — for the **12 CiPA
compounds with published Milnes-protocol fits**, straight from the
[FDA/CiPA repository](https://github.com/FDA/CiPA), and implements the binding kinetics
as an opt-in kernel path. Scrupulously bounded: the data are authoritative; the model is
an honest Tier-C reduction that touches no calibrated number.

- **`cipa_binding` dataset field** (new, optional, IKr records) carrying the published
  `Kmax`/`Ku`/`n`/`halfmax`/`Vhalf` plus the shared fixed `Kt = 3.5×10⁻⁵ ms⁻¹`, for
  bepridil, chlorpromazine, cisapride, diltiazem, dofetilide, mexiletine, ondansetron,
  quinidine, ranolazine, sotalol, terfenadine, verapamil. The other 16 CiPA compounds
  have no published dynamic data and keep static Hill block — **no fabricated kinetics**.
  Values are the FDA/CiPA repository optimal fits; cited to li-2017 (model) + li-2019
  (validation); shipped **`unverified`** (§9). Schema + `build_records.py` updated;
  cross-checked against the published trapping phenotype (dofetilide −1 mV ≫ terfenadine
  −82 > verapamil −97).
- **`CiPABinding` kernel model** implementing the exact CiPA binding kinetics
  (`on = Kmax·Ku·Dⁿ/(Dⁿ+halfmax)`, unbind `Ku`, trap `Kt`, voltage-dependent un-trap
  `Kt/(1+exp(−(V−Vhalf)/6.789))`) via two drug-bound sub-states (open-bound,
  closed-bound) coupled to the reduced IKr gate. Opt in with `assess(..., herg_dynamic="cipa")`.
  The hERG-binding state handling in `reference.py` was generalized to N bound states
  (the v0.1 Langmuir path stays byte-identical at 1 state).
- **Reproduces the trapping phenotype** at matched concentration without a tuned number:
  a near-zero-`Vhalf` blocker (dofetilide) accumulates and retains block beat-over-beat;
  a strongly-negative-`Vhalf` blocker (verapamil) washes out. New figure
  `docs/img/cipa_binding.png`; spec `docs/specs/v0.6-cipa-dynamic-herg.md`; README section.
- `tests/test_cipa_binding.py` (7 tests): data present for exactly the 12 (absent
  otherwise), unverified, zero-drug ⇒ no block, concentration-monotone, trapping
  phenotype, opt-in runs, **default path unaffected**. Suite 181 → 188.

### Honesty boundary (declared)
The **full 9-state CiPA Markov IKr** (the structure the parameters were fit to) is **not**
implemented — v0.6 couples the CiPA *binding* sub-model to the reduced HH IKr gate, a
Tier-C approximation; the full Markov + AP re-validation is future work. Because the
kinetics equilibrate slowly (~1000 beats in the official protocol), the CiPA path is a
research/demonstration surface and is **opt-in**: it changes no default qNet/ΔAPD90
metric, threshold, or recorded performance number.

## [0.5.4] — 2026-06-10

### Added — an executable notebook for the population-of-models subsystem
The populations subsystem spans three releases of work (the v0.1 Phase-E variability
cloud, the v0.3 LQTS disease backgrounds, the v0.5 Britton-2013 calibrated population)
and had README prose + static figures, but — unlike the flip-frequency (nb 01) and
Bayesian-UQ (nb 02) threads — **no executable, CI-asserted notebook**. This adds one,
closing the gap and giving the subsystem the same runnable-documentation guarantee.

- **`notebooks/03_populations.ipynb`** — deterministic (seeded), run in CI under
  `nbmake`, with inline assertions that double as tests: the variability-cloud
  susceptible fraction (a spread, not a point); the LQTS backgrounds each raising the
  susceptible fraction for a borderline drug (reduced repolarization reserve, textbook
  ordering); and the calibrated population's acceptance bookkeeping (rejection happens,
  triangulation is the dominant filter). Every section asserts the **Tier-D /
  NOT-FOR-PREDICTION** guardrail.

### Fixed
- `notebooks/README.md` was stale — it documented only notebook 01, omitting
  `02_bayesian_uq.ipynb` (which has shipped and run in CI since v0.2). Now lists all
  three with accurate summaries.
- README: the validation table, the "what's in the box" table, and the Phase-F
  roadmap row said "an executable notebook" (singular); corrected to three, with a
  pointer to nb 03 from the populations section.
- Committed `exports/` regenerated for the version bump (the v0.5.3 export-freshness
  CI gate requires it).

## [0.5.3] — 2026-06-10

### Fixed — committed exports were stale; the "never hand-edited" guarantee is now enforced
The README promised exports are "generated, never hand-edited (CI regenerates on
every push)", but CI only regenerated them to a throwaway directory and never
checked the committed `exports/` against the dataset. They had drifted: every
model stamped a stale `harmonia:datasetVersion` of `0.1.0` (the package was 0.5.x),
and `exports/tables/citations.bib` was **missing the v0.3/v0.5 citations**
(moss-2005, britton-2013, passini-2017, …) added since they were last generated.

- **`harmonia:datasetVersion` now resolves to the package `__version__`** instead
  of a hardcoded `"0.1.0"`. A new `harmonia.export.default_dataset_version()`
  (lazy, to avoid a circular import) is the single source; every builder
  (`cellml` / `sbml` / `myokit` / `combine` / `registry`) takes
  `dataset_version: Optional[str] = None` and resolves it, so a direct builder call
  stamps the right version, not just the CLI path.
- **Committed `exports/` regenerated** from the current dataset (correct version +
  the full citation set).
- **New CI gate — export reproducibility.** CI regenerates `exports/` in place and
  `git diff --exit-code`s the committed *text* artifacts (CellML, SBML, Myokit,
  SED-ML, tables, BibTeX), mirroring the existing `build_records.py` dataset gate —
  so a hand-edited or stale export now fails the build. The `.omex` zips are
  regenerated and manifest-checked but excluded from the byte diff (zlib's
  compressed output is not guaranteed identical across platforms; their text
  members are covered by the gated directories).

### Changed
- README: the exports section and validation table document the enforced
  reproducibility gate. CONTRIBUTING: lists the actual CI gates (ruff + mypy added
  in 0.5.1) and the regenerate-records-and-exports step; corrects the record-kinds
  line (channel-block / ap-model / drug-reference / population).

## [0.5.2] — 2026-06-10

### Added — the dashboard now surfaces the whole shipped feature set
The Streamlit dashboard (the spec-§6 headline presentation layer) had drifted
behind the library: it exposed the flip view, combinations, and browse, but **not**
the population-of-models subsystem (v0.1 Phase E), the **LQTS disease backgrounds**
(v0.3), the **experimentally-calibrated** population (v0.5), or the **Bayesian
dose-response UQ** engine (v0.2) — all shipped, tested, and figured in the README,
yet unreachable from the headline UI. This release closes that gap (presentation
only — no library, dataset, or kernel change).

- **New "Population-of-models" tab** — pick a drug and a `population` (the
  illustrative variability cloud, the three LQTS backgrounds, or the calibrated
  population) and see the susceptible fraction and class spread across virtual
  myocytes. A disease background shows its mean conductance shift; the calibrated
  population shows its acceptance rate and per-biomarker rejection counts. The tab
  is banner-stamped **Tier D / NOT FOR PREDICTION**, matching the library guardrail.
- **Bayesian dose-response UQ toggle** on the flip tab — switches `assess` to
  `uq="bayes"` and surfaces the true-value vs new-lab (reproducibility) flip split
  and the censored / prior-dominated channel flags.
- `tests/test_dashboard.py` gains a population-tab data contract (uncalibrated,
  disease, and calibrated populations, all asserted Tier D) and the three Bayesian
  fields, so the broadened UI cannot drift from the API silently. Suite 180 → 181.
- README: a new dashboard section documents all five tabs; the architecture
  diagram and repo-layout dashboard references updated to match.

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

# Contributing to Harmonia

Thank you for considering a contribution. Harmonia is **infrastructure, not a
simulator and not a clinical tool**. The most valuable contribution is almost
always **promoting an `unverified` record to `verified` by reading the primary
source** — with the cardiac-specific twist that the *assay context* and the
*multiple source values* are the part most worth scrutiny, because that is where
the hidden variability and the unidentifiable-IC50 traps live.

Please read [`spec.md`](spec.md) first — it defines the scope, the tier system,
and the hard safety line. Contributions that cross that line (anything that
emits an authoritative "this drug is safe/unsafe" verdict without its
uncertainty) will not be merged.

## The dataset is the source of truth

Everything else (the Python package, the CLI, the exports, the dashboard) is a
deterministic projection of `dataset/`. A change to a parameter is a change to a
JSON record under `dataset/records/`, never to a generated artifact.

```
dataset/
├── schema/record.schema.json    # the contract every record must satisfy
├── schema/context.jsonld        # JSON-LD term mapping (RDF export)
├── records/*.json               # one record per file (channel-block / ap-model / drug-reference / population)
└── citations/*.json             # one Crossref/PubMed-checked citation per file
```

Validate before opening a PR — the same gates CI runs:

```bash
ruff check .               # lint (Pyflakes + pycodestyle)
mypy                       # type-check the py.typed package surface
harmonia validate          # JSON-Schema-validate every record + cross-check citations
pytest                     # the full test suite, incl. round-trip export validation
```

If you change a parameter, regenerate the dataset and the committed exports so the
CI reproducibility gates stay green (they `git diff --exit-code` both):

```bash
python dataset/tools/build_records.py      # records are a projection of the curated table
harmonia export --all --output exports     # exports are a projection of the dataset
```

## Confidence tiers (the spine)

| Tier | Channel-block meaning | AP-model meaning |
| --- | --- | --- |
| **A** | Multiple independent labs/assays agree (low fold-range), block ≳60% so IC50 is identifiable, mechanism clear. | Validated on the CiPA validation compound set with good classification performance. |
| **B** | One good-quality measurement with adequate block; a single well-curated source. | Published, internally validated, not yet externally cross-checked here. |
| **C** | Single measurement; low/borderline block fraction; or unresolved manual-vs-automated discrepancy. | Reduced / reference kernel; structurally faithful but not the qualified regulatory model. |
| **D** | **Max block < ~60%** (IC50 unidentifiable / extrapolated), **or** population/disease extrapolation, **or** hypothesis-tier. **Not predictive.** | — |

**Two hard, machine-checked rules:**

1. **The reliability gate.** If `assay_context.max_block_observed_percent < 60`,
   the IC50 is *unidentifiable*. The record must be Tier D and must carry a
   `known_failure_modes` entry. Do not record a point IC50 as if it were
   reliable. `harmonia validate` enforces this.
2. **Worst-input-wins propagation.** A composed TdP-risk assessment inherits the
   *worst* tier among its ~7 channel-block records + AP model + risk metric. One
   unidentifiable IC50 caps the whole assessment at D.

## The multi-source / variability rule

When you have more than one published measurement of the same drug × channel
IC50, record **all** of them in `source_values`, each with its platform,
temperature, and citation. Harmonia computes `variability.fold_range` and the
IQR from these. **Do not silently average them into one number** — the spread is
the load-bearing field.

## PDF-verification checklist (what `verified` means)

Open the source PDF/supplement and confirm, for each record:

- [ ] The IC50 / Hill value **and its units** match the figure/table you cite.
- [ ] The **maximum block actually observed** (the reliability gate) — if it is
      below ~60%, the record is Tier D regardless of any quoted IC50.
- [ ] The **assay context**: platform (manual vs automated patch clamp),
      temperature, expression system, holding/pulse protocol.
- [ ] For hERG dynamic binding: the `kon`, `koff`, and trapping flags.
- [ ] For AP-model records: the equations and parameter set match the paper.
- [ ] For reference-compound records: the expert risk label and the EFTPC.

Only then set `extraction.review_status: "verified"` and add yourself to
`extraction.verified_by`. **LLMs may assist extraction but never promote a
record to `verified`.** The verified count is reported honestly by
`harmonia info`.

## Citations

Every `primary_citation` (and every `source_values[].citation`) must resolve to
a record under `dataset/citations/` with a DOI or PMID. `harmonia validate`
fails the build if a record cites a citation key that does not exist.

## Code style

- Pure-Python reference kernel (SciPy). Myokit/OpenCOR are optional cross-check
  engines, never a load-time dependency.
- Match the surrounding style. Keep changes surgical.
- Add a test for every behavior change.

## Code of conduct & security

Participation is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). To
report a security vulnerability (as opposed to a data or science question, which
belongs in a normal issue), follow the [Security Policy](SECURITY.md).

By contributing you agree that your code is licensed MIT and your data
contributions CC-BY-4.0.

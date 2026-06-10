# Notebooks

Executable analyses, run in CI under [`nbmake`](https://github.com/treebeardtech/nbmake)
(the family convention) so they cannot silently rot. Each is deterministic (seeded)
and carries inline assertions, so a clean run is a test, not just a demo:

- [`01_flip_frequency.ipynb`](01_flip_frequency.ipynb) — the headline
  **input-variability → classification-flip** analysis: the spread of published
  IC50s propagated by Monte-Carlo into a qNet distribution and a flip frequency,
  the flip view across AP-model variants, the recorded classification performance,
  and the unidentifiable-IC50 (Tier-D) handling.
- [`02_bayesian_uq.ipynb`](02_bayesian_uq.ipynb) — the v0.2 **Bayesian dose-response
  UQ**: the declared prior, the non-drift reduction to the v0.1 answer for a
  well-identified channel, a single-source channel borrowing the dataset-learned
  between-lab spread, the one-sided **censored** posterior for a sub-60%-block
  channel, the true-value vs new-lab (reproducibility) flip split, and Sobol
  (interaction-aware) sensitivity.
- [`03_populations.ipynb`](03_populations.ipynb) — the **population-of-models**
  subsystem (HYPOTHESIS-TIER): *physiological* (between-heart) variability as a
  susceptible-fraction spread, the v0.3 **LQTS disease backgrounds** raising
  susceptibility through reduced repolarization reserve, and the v0.5
  **experimentally-calibrated** population (Britton 2013 drug-free-plausibility
  acceptance). Every assertion confirms the Tier-D / NOT-FOR-PREDICTION guardrail.

Run them locally with:

```bash
pip install -e ".[dev,notebooks]"
pytest --nbmake notebooks/
```

The README figures are regenerated (reproducibly, though not executed in CI) by
[`docs/make_figures.py`](../docs/make_figures.py) — a faithful projection of the
dataset + reference kernel.

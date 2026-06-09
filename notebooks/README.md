# Notebooks

Executable analyses, run in CI under [`nbmake`](https://github.com/treebeardtech/nbmake)
(the family convention) so they cannot silently rot:

- [`01_flip_frequency.ipynb`](01_flip_frequency.ipynb) — the headline
  **input-variability → classification-flip** analysis: the spread of published
  IC50s propagated by Monte-Carlo into a qNet distribution and a flip frequency,
  the flip view across AP-model variants, the recorded classification performance,
  and the unidentifiable-IC50 (Tier-D) handling. Deterministic (seeded) and
  carrying inline assertions, so a clean run is a test, not just a demo.

Run them locally with:

```bash
pip install -e ".[dev,notebooks]"
pytest --nbmake notebooks/
```

The README figures are regenerated (reproducibly, though not executed in CI) by
[`docs/make_figures.py`](../docs/make_figures.py) — a faithful projection of the
dataset + reference kernel.

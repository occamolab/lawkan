# Notebooks

Two small, self-contained examples:

- `summary_figure.ipynb` — regenerate the README summary figure from the pinned dumps;
- `synthetic_smoke.ipynb` — run `experiments/synthetic/E19_sparse_edge.py --quick` on CPU.

The notebooks run from the repository root, install only `scienceplots`, and do **not** save
executed outputs back into this repository.

## Local run

```bash
pip install jupyter nbconvert
jupyter nbconvert --to notebook --execute examples/summary_figure.ipynb \
  --output /tmp/summary_executed.ipynb
jupyter nbconvert --to notebook --execute examples/synthetic_smoke.ipynb \
  --output /tmp/synthetic_executed.ipynb
```

You can also run the underlying commands directly from the repository root:

```bash
python tools/figures/fig00_readme_summary.py
python experiments/synthetic/E19_sparse_edge.py --quick
```

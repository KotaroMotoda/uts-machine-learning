# XOR Perceptron

NumPy implementation comparing four feature maps on clean and noisy XOR.

| Map | Features |
|---|---|
| raw | `(1, x1, x2)` |
| square | `(1, x1, x2, x1^2, x2^2)` |
| interaction | `(1, x1, x2, x1*x2)` |
| quadratic | `(1, x1, x2, x1^2, x2^2, x1*x2)` |

Python 3.9+.

```bash
uv sync --dev
MPLBACKEND=Agg uv run pytest -q
MPLBACKEND=Agg uv run python scripts/run_experiments.py
MPLBACKEND=Agg uv run python scripts/build_notebook.py
```

- `src/xor_study/`: datasets, feature maps, Perceptron, experiments, plots.
- `tests/`: unit tests and reproducibility checks.
- `notebooks/`: standalone, code-only notebook.

Results and figures are generated locally and are not tracked by Git.

Random seeds and dependency versions are fixed. No external dataset is needed.

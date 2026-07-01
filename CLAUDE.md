# CLAUDE.md

Realized Volatility Prediction — MTH 5320 Deep Learning, Project 2. Can a small LSTM
beat HAR at forecasting SPY realized volatility? Full scope, invariants, and design
rationale are in `MANIFESTO.md` (v2, canonical) — read it before doing any work here.

## Environment

- Conda env: `dl-p2-env` (Python 3.11), registered as Jupyter kernel `dl-p2-env`.
  Launch Jupyter from `base` (`conda activate base && jupyter notebook`), then select
  this kernel.
- Stack: PyTorch (CPU wheel), Optuna, `statsmodels`, `arch`, `pandas`, `numpy`, `pyarrow`.
  Actual training target is Google Colab per the manifesto's tech-stack section; this
  local env is for development/testing outside Colab.
- `environment.yml` auto-regenerates on commit via `.githooks/pre-commit`
  (`git config core.hooksPath .githooks` already set on this clone — re-run on any
  other machine after cloning, since git doesn't transfer hook activation).

## Status

Manifesto landed; no `SPEC.md`s or `src/` modules written yet.

# CLAUDE.md

Realized Volatility Prediction — MTH 5320 Deep Learning, Project 2. Can a small LSTM
beat HAR at forecasting SPY realized volatility? Full scope, invariants, and design
rationale are in `MANIFESTO.md` (v3, canonical) — read it before doing any work here.

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
- Colab: open via the badge in `README.md` (`notebooks/colab_bootstrap.ipynb`), which
  clones this repo, installs `requirements.txt` (unpinned deltas only — Colab already
  ships numpy/pandas/torch), then shells out to `scripts/run_stage1.py`.

## Status

Stage 0 complete: all 7 `specs/<module>/{SPEC.md,design.md}` pairs landed; `src/`
engine implemented and tested (116 pytests, `python -m pytest`); data snapshot frozen
(`scripts/freeze_snapshot.py`, manifest + SHA-256, floors in config, S8 calibration
passes); classical field scored on the frozen splits, both targets/horizons
(`scripts/run_stage0.py`, artifacts committed under `results/stage0_classical/`).
HAR tops every block; GARCH statistically indistinguishable from HAR on TV.
Manifesto amendments ratified 2026-07-03 (manual Stooq acquisition, Tier 1/2/3 menu,
log-HAR). Next: Stage 1 (leak-safe Optuna tuning on TV, then RV-only net vs HAR).

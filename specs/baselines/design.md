# design — `baselines` (`src/baselines.py`)

## 1. Overview

Implements `specs/baselines/SPEC.md`: the five classical forecasters run through the shared
walk-forward driver over `splits.retrain_folds`. Greenfield; sits on `data`, `splits`,
`metrics`, and `features.make_labels`.

## 2. Approach / architecture

One public entry `forecast_classical(model, ...)` dispatching to per-model fit/predict pairs,
all funneled through a single `_walk_forward(fit_fn, predict_fn, folds, ...)` driver so the
protocol (fit on `fit_idx`, predict each origin in `test_idx`) is written once. Labels come
from `features.make_labels`; predictors (lagged log-target, HAR aggregates) come from small
local helpers mirroring the Tier-1 definitions but built per fold from raw targets — importing
the same lag arithmetic helpers from `features` where shapes allow.

Per-model notes:
- **persistence** — no fit; predict from trailing values at origin.
- **ewma** — `_fit_ewma` grid λ∈[0.80,0.99] step 0.005 minimizing mean `metrics.qlike_series`
  in-sample; causal recursion warm-started on fit window; boundary-argmin → 0.94 fallback, log.
- **ar1 / har** — `statsmodels.OLS` on log-label vs log lag(s); predict
  `exp(Xβ + σ²_resid/2)`.
- **garch** — `arch.arch_model(returns*100, vol="GARCH", p=1, q=1)`; forecast
  `horizon=h`, variance path mean for h=5, `/100²` back; non-convergence retry then
  parameter-carry per SPEC R6.

## 3. File-by-file plan

- `src/baselines.py` — `forecast_classical`, `_walk_forward`, `_fit_ewma`, `_persistence_pred`,
  `_fit_ar1`, `_fit_har`, `_fit_garch`, `_har_design` (log d/w/m aggregates at origins).
- `tests/test_baselines.py` — S1–S7; synthetic generators (lognormal AR(1), HAR-generated,
  arch-simulated GARCH) local to the test module.

## 4. Data models / schemas

Output frame per SPEC §4: index origin dates; columns `y_pred, y_true, model, target, horizon,
fold_id` (+ `attrs["object_mismatch"]` for garch-vs-OC, E4). Per-fold fit diagnostics dict
(HAR betas/R², EWMA λ + fallback flag, GARCH params + carry flag) returned via `attrs["fit_log"]`.

## 5. Key interfaces & signatures

Per SPEC §4. Internal driver:

```python
def _walk_forward(fit, predict, folds, series_bundle, cfg) -> pd.DataFrame
```

## 6. Implementation sequence

1. `_walk_forward` skeleton + persistence (simplest) → S1 fixture green.
2. EWMA (`_fit_ewma`, recursion) → S1 fixture.
3. `_har_design` + `_fit_har`, `_fit_ar1` (shared OLS/back-transform helper) → S3, S5.
4. `_fit_garch` with scaling round-trip → S4.
5. Full-field integration test on synthetic multi-year series through real folds (S2);
   determinism run (S6); edge-case parametrization (S7).

## 7. Integration points

`splits.retrain_folds` (protocol), `features.make_labels` (y — single source of truth),
`metrics.qlike_series` (EWMA λ objective), `data.build_targets` + `Snapshot.spy["close"]`
(GARCH returns, adjusted basis). The Stage-0 notebook calls `forecast_classical` per
(model × target × horizon) and hands frames to `metrics` + `io.save_run`.

## 8. Test plan

SPEC §9 S1–S7. Fixtures favor closed-form checks: persistence/EWMA exact arithmetic; HAR/AR1
parameter recovery within tolerance on synthetic truth; GARCH scaling verified by hand-scaling
one forecast. The S2 integration test is the Stage-0 dress rehearsal.

## 9. Risks & open questions

- Log-HAR vs level-HAR: adopted log-HAR per SPEC Open Q — surfaced to the user before any
  test-region scoring; a one-line change in `_fit_har` if reversed.
- `arch` API drift (result `.forecast(...).variance` shape changed across versions) — pin the
  version; test asserts shape.
- GARCH fit time over ~78 monthly folds × MLE is the slowest classical piece; acceptable
  (seconds/fold, CPU). If it drags, `update_freq=0` and warm-start params from prior fold.

# SPEC — `baselines` (`src/baselines.py`)

> Layer 2. Inherits from `MANIFESTO.md` (v3, canonical). Implements the `baselines` module of
> manifesto §6 (single module — the models+GARCH split was declined; §8 open carve resolved to
> defaults).

## 1. Purpose

The classical field — persistence, EWMA, AR(1) on log-target, HAR, GARCH(1,1) — the models that
give "HAR is hard to beat" empirical content on this target (manifesto §6 `baselines`, research
question (1)). Scored on the frozen splits, both targets, both horizons, through the monthly
walk-forward retrain protocol. Stage 0 ends when this module's outputs exist (§5).

## 2. Scope

Five forecasters, each producing **one variance-space forecast per test date per horizon per
target it is scored on**, re-fit per retrain fold on the fold's fixed-length rolling window:

1. **Persistence** — h=1: today's target value; h=5: today's trailing 5-day mean of the target.
   No parameters.
2. **EWMA** — exponentially weighted moving average of the lagged target; decay λ fit on train
   (grid over [0.80, 0.99] step 0.005, minimizing train QLIKE), RiskMetrics λ=0.94 as the fixed
   fallback if the fit fails or is boundary-stuck.
3. **AR(1) on log-target** — OLS; forecast exponentiated back with the lognormal half-variance
   correction `exp(μ̂ + σ̂²_resid/2)`; h=5 forecasts the 5-day-average label directly (direct, not
   iterated, §8).
4. **HAR** — OLS of the horizon-h log-label on log daily / weekly (5-day mean) / monthly (22-day
   mean) lagged target; same back-transform as AR(1). **The presumptive bar.**
5. **GARCH(1,1)** — fit on close-to-close **returns** (adjusted basis); forecast object is
   *total* return variance → scored against TV; if displayed against OC the object mismatch is
   stated in the table note (§6). h=1: one-step variance forecast; h=5: mean of the 1..5-step
   variance forecasts (matches the h=5 average-target definition).

**Non-goals**
- Not the net (§6). No tuning beyond the pre-specified EWMA grid. No model not in the glossary's
  classical field. No new estimators.
- Does not compute QLIKE/DM (calls `metrics` at the experiment layer) and does not build splits.

## 3. Inherited invariants

- All forecasts delivered in **variance space** before scoring (§7); GARCH's `arch` output is in
  the scaled units the library was fed — this module owns the unit round-trip.
- Chronological walk-forward via `splits.retrain_folds` only; **never** a hand-rolled loop over
  dates with ad-hoc windows (§7, ONE splitter).
- Model log-target internally where a statistical model is estimated (AR(1), HAR); persistence
  and EWMA are level-space constructions with no estimation step; GARCH's likelihood is its own
  level-space model (see Open questions for the recorded interpretation).
- Frozen controls (retrain cadence, window length) come from config; never varied here (§4).
- The test region is touched exactly once per pre-registered experiment (§7) — this module
  produces predictions; the single scoring pass happens at the experiment layer.

## 4. Interfaces / contracts

```python
def forecast_classical(model: str,                       # "persistence"|"ewma"|"ar1"|"har"|"garch"
                       targets: pd.DataFrame,            # from data.build_targets
                       returns_cc: pd.Series | None,     # required for "garch" only
                       folds: list[RetrainFold],
                       *, target: str, horizon: int,
                       cfg: dict) -> pd.DataFrame
    # returns: index = forecast-origin dates (all fold test dates), columns:
    #   y_pred (variance), y_true (variance), model, target, horizon, fold_id
```

Contracts:
- Origin-date convention: a row at origin t is the forecast, made with information ≤ t, of the
  horizon-h label `y_t^{(h)}` (same convention as `features`).
- Fit data for a fold = the fold's `fit_idx` rows only; predictors at test origins may use
  lookback history before the origin (that is information ≤ t, permitted; the leak contract
  binds features to ≤ t, not to the fit window).
- `y_true` is attached for artifact convenience and comes from the same label construction as
  `features` (imported, not re-derived).
- HAR coefficient sanity (done-check): fitted daily/weekly/monthly betas finite; R² in (0, 1);
  persistence-sum `β_d+β_w+β_m` in (0, 1.5) — logged per fold, warned if violated.

## 5. Dependencies

`data`, `splits`, `metrics` (manifesto §6). `features` only for the shared label-construction
helper (single source of truth for y — never re-derive).

## 6. Tech stack (this module)

`statsmodels` (OLS for HAR / AR(1)), `arch` (GARCH(1,1), returns scaled ×100 for optimizer
stability, variance rescaled ÷100² on output), `numpy`/`pandas` one-liners for persistence and
EWMA (§10). Pinned in `requirements.txt`.

## 7. Requirements & behavior

R1. Every model must emit exactly one prediction per fold test date; union over folds covers the
    whole test region (guaranteed by splits R3; asserted here too).
R2. All predictions strictly positive in variance space (exp back-transform for AR(1)/HAR;
    persistence/EWMA of a non-negative series with positive floor guard; GARCH positive by
    construction). Positivity is asserted; a violation is a bug, not something to floor here —
    the QLIKE floor in `metrics` is the only floor.
R3. EWMA recursion must be causal: `ewma_t = λ·ewma_{t-1} + (1−λ)·target_{t}` evaluated on
    lagged values so the forecast at origin t uses targets ≤ t; h=5 uses the same value (EWMA is
    a flat forecaster; the direct-h target is the 5-day average, and the flat forecast is the
    pre-specified behavior for both horizons).
R4. λ fit: grid on the fold's fit window, minimizing in-sample QLIKE via `metrics.qlike_series`;
    if the argmin sits on the grid boundary or the fit errors, fall back to 0.94 and log which
    happened per fold.
R5. AR(1)/HAR regressions must drop no rows silently: the design matrix is built from lagged
    log-target columns provided by the shared feature helpers; row accounting is exposed.
R6. GARCH: fit by MLE on the fold window's close-to-close log returns (×100); non-convergence →
    retry from RiskMetrics-ish start values; still failing → carry the previous fold's fitted
    parameters and log the carry (never a missing prediction; never a peek forward).
R7. Determinism: identical inputs → identical outputs; GARCH optimizer seeded/started
    deterministically.
R8. Each model function is importable and individually testable; the walk-forward driver is
    shared across the five.

## 8. Edge cases & error handling

E1. Fold fit window contains a zero target value → AR(1)/HAR log transform errors loudly (same
    policy as features E3).
E2. EWMA at the first origin of the first fold: recursion warm-started on the fit window's
    history; assert warm-up length ≥ 60 obs.
E3. GARCH variance forecast non-positive or non-finite → hard error (violates `arch` guarantees;
    indicates unit corruption).
E4. Horizon/target combination "garch"+"rv_oc": permitted for the descriptive table but the
    output frame must carry `object_mismatch=True` metadata so the table note (§6) is generated,
    not remembered.
E5. Fold whose test month has fewer origins than h (tail of sample): origins lacking full labels
    were already dropped by the label constructor; a fold may legitimately shrink — never pad.
E6. `returns_cc` passed for a non-GARCH model or missing for GARCH → ValueError.

## 9. Success criteria (executable)

Each maps to a pytest in `tests/test_baselines.py` (manifesto §6 done-check + S1):

S1. Persistence and EWMA match hand-computed fixtures (small synthetic series, both horizons).
S2. Every classical model produces one variance-space forecast per test date per scored target
    on a synthetic multi-year series run through real `splits` folds.
S3. HAR fold coefficients finite and sanity-logged; on a synthetic HAR-generated series the
    fitted betas recover the truth within tolerance.
S4. GARCH forecast positive, finite, and in variance space: on synthetic GARCH(1,1) data with
    known parameters, fitted params within tolerance and the ×100 unit round-trip verified
    against a hand-scaled computation.
S5. AR(1) back-transform: on synthetic lognormal AR(1) data, the half-variance-corrected
    forecast is unbiased in level space within Monte-Carlo tolerance.
S6. Determinism: two runs byte-identical.
S7. E1–E6 behave as specified.

## 10. Open questions

- **Log vs level for classical fits**: the §7 invariant "ALWAYS model log-target internally"
  is read here as binding *estimated* statistical models (AR(1), HAR — both fit in log space),
  while persistence/EWMA (no estimation) and GARCH (own likelihood) stay in level space. The
  manifesto glossary marks AR(1) as log explicitly but is silent for HAR; log-HAR is adopted and
  pre-registered here. Flag if level-HAR (Corsi's original) is wanted instead — decide before
  any test-region contact.
- EWMA λ grid bounds/step are this spec's pre-specified default (manifesto says only "fit on
  train, 0.94 fallback").
- The HAR sanity thresholds (R4 contract) are heuristics for logging, not gates; confirmed as
  non-blocking.

# design — `features` (`src/features.py`)

## 1. Overview

Implements `specs/features/SPEC.md`: leak-safe feature matrix (Tier 1/2/3), horizon labels,
train-only scaler, and the importable future-perturbation leakage probe. Greenfield; sits on
`src/data.py`.

## 2. Approach / architecture

Every feature is a named builder function `(targets, snap) -> pd.Series` registered in an
ordered dict per tier; `build_features` composes requested tiers, joins on the calendar, drops
rows where any feature or the label is NaN (warmup + tail), and returns `FeatureSet`. Causality
by construction: builders may use only `.shift(k≥0)` of day-t-known columns and
`.rolling(...).agg` over past values — code review + the perturbation probe enforce it.

Labels: `y = targets[target].shift(-1)` for h=1; for h=5,
`targets[target].shift(-1).rolling(5).mean().shift(-4)` (mean of t+1..t+5), tail-dropped.

## 3. File-by-file plan

- `src/features.py` — `FeatureSet`, `build_features`, `TrainOnlyScaler`, `leakage_probe`,
  `make_labels`, tier registries `_TIER1/_TIER2/_TIER3` (column names per SPEC §2 menu,
  prefixed `t1_/t2_/t3_`).
- `tests/test_features.py` — S1–S6.

## 4. Data models / schemas

`FeatureSet(X, y, y_log, tiers)` per SPEC. X columns float64 finite; index ⊆ calendar;
`tiers` maps `"t1"→[cols]` etc. Scaler stores `means`, `stds` (pd.Series) + `fit_indices`.

## 5. Key interfaces & signatures

Per SPEC §4, plus `make_labels(targets, target, horizon) -> tuple[pd.Series, pd.Series]`
(level, log) — exported because `baselines` must import the identical label construction
(baselines SPEC §5: never re-derive y).

## 6. Implementation sequence

1. `make_labels` + fixture test (S4).
2. Tier 1 builders (lagged log-target 1..22, HAR aggregates) + S5 fixture.
3. Tier 2 (returns, |r|, overnight, volume ratio) and Tier 3 (log VIX, VIX change, implied var).
4. `build_features` composition + row-accounting; finiteness assertion (R1, E3 zero-target
   error path).
5. `TrainOnlyScaler` (mean/std, fit-index record, zero-std error E4).
6. `leakage_probe`: multiplicative lognormal noise on all raw inputs strictly after t (seeded
   rng), rebuild, `pd.testing.assert_frame_equal(check_exact=True)` on `X.loc[:t]`; S1 test
   sweeps early/middle/late t, both targets, both horizons.

## 7. Integration points

Consumes `data.build_targets` + `Snapshot`; `make_labels` imported by `src/baselines.py`;
`FeatureSet` consumed by `src/train.py` (which does the sequence windowing); scaler fit indices
supplied from `splits.CanonicalSplit`/fold fit windows at the experiment layer.

## 8. Test plan

SPEC §9 S1–S6. The leakage probe is the S3 project gate — parametrize over
`t ∈ {calendar[100], mid, calendar[-300]}`. Scaler test: train mean ≈ 0 / val mean ≠ 0 pattern.
Finiteness on real snapshot (skip-if-absent like data tests).

## 9. Risks & open questions

- Tier menu is spec-proposed, not manifesto-given (flagged there) — if the user amends tiers,
  only the registry dicts change; everything else is invariant to the menu.
- Bitwise identity in the probe assumes deterministic pandas ops — safe on fixed dtypes/versions
  (pinned requirements).
- Label/feature row accounting must match between `features` and `baselines` label usage —
  solved by the shared `make_labels` (single source of truth).

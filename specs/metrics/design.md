# design — `metrics` (`src/metrics.py`)

## 1. Overview

Implements `specs/metrics/SPEC.md`: `qlike` (+floor, bind-rate), `rmse`, `qlike_series`, and
`dm_test` (DM with HLN correction, HAC-Bartlett variance). Greenfield; depends only on
numpy/scipy.

## 2. Approach / architecture

Small pure functions with one shared `_validate_pair(pred, actual)` guard (length, emptiness,
NaN, positivity of actuals) and one `_unit_guard` (median-ratio > 1e3 → UserWarning). DM is
computed from a supplied per-date loss series so the same code serves QLIKE and any future loss:

```
d_t = loss_model_t - loss_bench_t
S = HAC-Bartlett(d, L)          # gamma_0 + 2*sum_{k=1..L} w_k gamma_k, w_k = 1 - k/(L+1)
DM = dbar / sqrt(S / n)
HLN = DM * sqrt((n + 1 - 2h + h(h-1)/n) / n)
p   = 2 * (1 - t.cdf(|HLN|, df=n-1))
```

`L = max(h-1, floor(4*(n/100)**(2/9)))` when `hac_lag=None`.

## 3. File-by-file plan

- `src/metrics.py` — `QlikeResult`, `DMResult`, `qlike`, `qlike_series`, `rmse`, `dm_test`,
  privates `_validate_pair`, `_unit_guard`, `_hac_variance`.
- `tests/test_metrics.py` — S1–S6; reference cross-check uses
  `statsmodels.stats.sandwich_covariance` / `sm.OLS(d, ones).fit(cov_type="HAC")` for the HAC
  variance on the same series.

## 4. Data models / schemas

`QlikeResult(value, bind_rate, n)`, `DMResult(stat, p_value, hac_lag, n, mean_loss_diff)` —
frozen dataclasses per SPEC §4. All array inputs coerced via `np.asarray(..., dtype=float)`.

## 5. Key interfaces & signatures

Per SPEC §4 verbatim. `_hac_variance(d: np.ndarray, lag: int) -> float` is the one internal
worth naming — it is the piece cross-checked against statsmodels in S3.

## 6. Implementation sequence

1. `_validate_pair`, `_unit_guard`.
2. `qlike_series` (elementwise loss with floor), then `qlike` (mean + bind-rate), `rmse`.
3. `_hac_variance`, `dm_test` with HLN + t p-value; degenerate-variance guard (E2).
4. Tests: hand-computed 3-element fixtures (exact fractions in comments), statsmodels
   cross-check, h-lag response, unit-guard fixtures, E1–E6 parametrized.

## 7. Integration points

`src/baselines.py` (EWMA λ fit minimizes `qlike_series` mean), `src/train.py` (torch QLIKE must
equal `metrics.qlike` on fixtures — model-harness S6), the Stage-0 experiment notebook (scoring),
`src/io.py` schema (metrics.json fields mirror `QlikeResult`/`DMResult` fields). Floor value
arrives from `configs/default.yaml` via `io.load_config` — metrics never reads config itself.

## 8. Test plan

SPEC §9 S1–S6, plus: bind-rate exactness on a crafted 20% -binding fixture; p-value sanity on
i.i.d. noise (uniform-ish over replications — smoke, seeded, not a statistical assertion);
`dm_test` symmetry (swap model/bench → stat negates).

## 9. Risks & open questions

- HLN p-value convention: t(n−1) two-sided is pre-specified; some references use normal — the
  statsmodels cross-check covers only the HAC variance, the HLN scaling is hand-verified.
- The automatic NW lag rule is the classic Bartlett rule-of-thumb; recorded in SPEC Open
  questions as the adopted default.
- Zero `actual_var` rejection (R5) interacts with data E5 (zero-target days); on real SPY this
  never fires — if it does, the failure is loud by design.

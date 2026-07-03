# SPEC — `metrics` (`src/metrics.py`)

> Layer 2. Inherits from `MANIFESTO.md` (v3, canonical). Implements the `metrics` module of
> manifesto §6. If anything here contradicts the manifesto, the manifesto wins.

## 1. Purpose

Scoring and significance — the **one** QLIKE/RMSE/DM definition every experiment imports
(manifesto §6 `metrics`, §7 "ONE QLIKE/RMSE/DM definition"). Owns the QLIKE loss with the
pre-specified prediction floor and bind-rate logging, RMSE, the Diebold–Mariano test with the
Harvey–Leybourne–Newbold small-sample correction and HAC (Bartlett) variance, and the
variance-space discipline. S5 ("significance is valid, not just reported") is checked here.

## 2. Scope

Pure scoring functions on aligned numeric sequences. Consumes prediction/actual arrays; produces
losses, test statistics, p-values, and bind-rate diagnostics.

**Non-goals**
- Owns no data, no model, no split. Never loads files, never touches the network.
- Does not decide *which* prediction sequence DM receives — the harness supplies the
  seed-ensemble mean (§7, Stochasticity). This module tests whatever aligned pair it is given.
- Does not choose the floor value — that is fixed at data freeze (1st percentile of the
  train-span target, recorded in `configs/default.yaml`, per §7); this module applies a supplied
  floor.
- No multiple-comparison correction — MCS is Stage 3 (§5); v1 reports all pairwise DM p-values
  together (§7, honesty).

## 3. Inherited invariants

- **ALWAYS score QLIKE in variance space.** Never silently mix volatility and variance. (§7)
- ALWAYS apply the **pre-specified** positive floor to predictions before QLIKE and log the
  bind-rate every run; a materially binding floor (S8 threshold, 0.1% of test predictions)
  invalidates the comparison and must be flagged, not absorbed. (§7)
- For **h=5** the loss differentials are serially correlated by construction: DM variance MUST be
  HAC (Bartlett) with lag ≥ h−1 — this is why DM lives here and is never hand-rolled in a
  notebook. (§7)
- DM uses the **HLN small-sample correction**, two-sided. (§3 S5, §8)
- Report RMSE alongside QLIKE. (§7)

## 4. Interfaces / contracts

```python
@dataclass(frozen=True)
class QlikeResult:
    value: float          # mean QLIKE over the sequence
    bind_rate: float      # fraction of predictions where the floor bound
    n: int

@dataclass(frozen=True)
class DMResult:
    stat: float           # HLN-corrected DM statistic
    p_value: float        # two-sided, Student-t with (n-1) dof per HLN
    hac_lag: int          # Bartlett truncation lag actually used
    n: int
    mean_loss_diff: float # mean d_t = L(model) - L(benchmark); negative favors model

def qlike(pred_var: np.ndarray, actual_var: np.ndarray, floor: float) -> QlikeResult
def rmse(pred_var: np.ndarray, actual_var: np.ndarray) -> float
def dm_test(loss_model: np.ndarray, loss_bench: np.ndarray, h: int,
            hac_lag: int | None = None) -> DMResult
def qlike_series(pred_var: np.ndarray, actual_var: np.ndarray, floor: float) -> np.ndarray
```

Contracts:
- All inputs are **variances**, aligned 1:1 by forecast-origin date (alignment is the caller's
  job; lengths must match).
- `qlike` per-element loss: `actual/pred_f - log(actual/pred_f) - 1` with
  `pred_f = max(pred, floor)`; the `-1` normalization makes a perfect forecast score 0.
- `dm_test` operates on per-date **loss** series (typically from `qlike_series`), so the same DM
  code serves any loss. `hac_lag=None` → `max(h - 1, automatic Newey–West selection)` (§6
  done-check: "HAC lag responds to h; larger if automatic selection says so").
- Bartlett kernel weights `1 - k/(L+1)`; HLN correction factor
  `sqrt((n + 1 - 2h + h(h-1)/n) / n)` applied to the classic DM stat; p-value from Student-t with
  `n-1` degrees of freedom.

## 5. Dependencies

None among project modules (manifesto §6). `numpy`, `scipy.stats` (t distribution) only.

## 6. Tech stack (this module)

`numpy` + `scipy`; pure functions, no state, no I/O. Pinned via project `requirements.txt`.

## 7. Requirements & behavior

R1. `qlike` must floor non-positive and sub-floor predictions at `floor` and report the exact
    bind-rate; it must never return NaN/inf for positive `actual_var` and positive floor.
R2. `rmse` computed in variance space, no floor.
R3. `dm_test` must implement: mean loss differential, HAC (Bartlett) long-run variance with
    truncation lag ≥ h−1, HLN small-sample correction, two-sided Student-t p-value.
R4. A **variance-space guard**: `qlike` and `rmse` must emit a loud `warnings.warn` (category
    `UserWarning`, message naming the suspected mixup) when the median of `pred` and `actual`
    differ by a factor > 1e3 — the signature of volatility-vs-variance mixing on daily SPY
    variance (~1e-4). The guard warns, never auto-corrects.
R5. All functions must reject (ValueError) length mismatch, empty input, NaN in inputs, or
    non-positive `actual_var` entries (targets are non-negative by construction after warmup;
    zero actual is degenerate for QLIKE and must be surfaced, not floored away silently).
R6. `dm_test` must accept the seed-ensemble-mean-derived loss series without knowing its
    provenance; provenance is the harness's contract (§7).
R7. Deterministic, side-effect free (the warning is the only side channel).

## 8. Edge cases & error handling

E1. All predictions above floor → `bind_rate == 0.0` exactly.
E2. Constant loss differential (zero variance) → return `stat = ±inf`-free result: raise
    `ValueError("degenerate loss differential")` rather than emit a fake p-value.
E3. `n <= h` → `ValueError` (HLN correction undefined / meaningless sample).
E4. `hac_lag` supplied smaller than `h-1` → raise `ValueError` (the §7 invariant is a hard floor,
    not a default).
E5. Negative predictions → floored by R1, counted in bind-rate (never an exception; the floor is
    the pre-specified handling).
E6. `floor <= 0` → `ValueError` (the floor must be positive by definition, §7).

## 9. Success criteria (executable)

Each maps to a pytest in `tests/test_metrics.py` (manifesto §6 done-check):

S1. `qlike`/`rmse` match hand-computed fixture values (3-element arrays, exact arithmetic).
S2. Non-positive prediction is floored and bind-rate reflects it exactly.
S3. `dm_test` matches a reference implementation on a fixture: verify against a hand-computed
    HLN-corrected statistic on a small series AND against `statsmodels`' HAC variance on the same
    series (agreement to 1e-8).
S4. HAC lag responds to h: `dm_test(..., h=5).hac_lag >= 4 > dm_test(..., h=1).hac_lag` on the
    same data with automatic selection returning a small lag.
S5. Variance-space guard fires on a vol-vs-variance fixture (pred in vol units) and is silent on
    matched units.
S6. Each edge case E1–E6 behaves as specified.

## 10. Open questions

- The exact automatic HAC lag rule when it exceeds h−1: default Newey–West `floor(4*(n/100)^(2/9))`
  (Bartlett). Adopted as default here; flag if a different rule is preferred.
- The variance-guard threshold (1e3 median ratio) is a heuristic constant not set by the
  manifesto; recorded here as the pre-specified default so it is not tuned after seeing results.
- Whether QLIKE's `-1` normalization is wanted in *reported* tables (rankings and DM are invariant
  to it). Default: yes, report normalized QLIKE.

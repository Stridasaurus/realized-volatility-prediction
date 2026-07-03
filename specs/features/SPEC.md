# SPEC — `features` (`src/features.py`)

> Layer 2. Inherits from `MANIFESTO.md` (v3, canonical). Implements the `features` module of
> manifesto §6. If anything here contradicts the manifesto, the manifesto wins.

## 1. Purpose

Build the leak-safe feature matrix — target history plus tiered aux features — and own the
feature-level leak contract (manifesto §6 `features`). This module is what makes S3 ("no
lookahead, provably") executable: every feature value at day *t* uses only information known at
the close of day *t*.

## 2. Scope

1. **Feature construction** from `data`'s targets and snapshot columns, on the trading-day
   calendar.
2. **Supervised label alignment**: the horizon-h label series — `y_t^{(1)} = target_{t+1}`;
   `y_t^{(5)} = mean(target_{t+1..t+5})` (h=5 is the *average* daily target, forecast directly,
   §8) — labels use future values by definition; the leak contract governs **features only**.
3. **The tiered aux menu** (see Open questions — the manifesto names Tier 1/2/3 but does not
   enumerate them; the menu below is the pre-registered proposal):
   - **Tier 1 (target history):** lagged log-target (lags 0..21 — lag 0 is today's value,
     known at the close of day t); HAR-style aggregates of the target through day t
     (5-day mean, 22-day mean), in log space.
   - **Tier 2 (SPY price/volume):** lagged daily close-to-close log return; lagged |return|;
     lagged overnight log return; volume log-ratio vs its trailing 22-day mean.
   - **Tier 3 (VIX):** lagged VIX close level (log); lagged 1-day VIX log-change; lagged implied
     variance `(VIX/100)²/252` in daily variance units.
   All Tier 2/3 inputs are lagged ≥ 1 day relative to nothing — they are *known at close of day
   t* (VIX close, SPY close/volume at t are known at t's close and are valid at t).
4. **Scaler/transform objects**: fit on train indices only, applied forward; each scaler records
   the exact indices it was fit on.
5. **The future-perturbation leakage test** as an importable function (used by tests and by S3).

**Non-goals**
- Does not own the temporal split (`splits`) or the targets (`data`).
- No feature selection or importance analysis; *which* features enter an experiment is the
  experiment config (RV-only = Tier 1; RV+aux = Tiers 1+2+3, §2 use cases).
- No sequence windowing/tensor shaping for the LSTM (harness concern); this module emits a flat
  per-date feature DataFrame.

## 3. Inherited invariants

- **EVERY feature value at day *t* uses only information known at the close of day *t*, never
  revised afterward.** Test: perturb data after *t*; the feature at *t* must not change. (§7)
- **ALWAYS fit scalers/transforms on training data only, then apply forward.** (§7)
- ALWAYS model **log-target** internally; features that are target transforms live in log space.
  (§7)
- Never copy-paste feature logic into a notebook. (§7)
- The only variable across the two feature experiments is the input feature set. (§7)

## 4. Interfaces / contracts

```python
@dataclass(frozen=True)
class FeatureSet:
    X: pd.DataFrame            # index: trading days; columns: feature names (finite floats)
    y: pd.Series               # horizon-h label, level (variance) space
    y_log: pd.Series           # log of y (the modeled quantity)
    tiers: dict[str, list[str]]  # tier name -> column names

def build_features(targets: pd.DataFrame, snap: Snapshot, *, target: str,
                   horizon: int, tiers: Sequence[str]) -> FeatureSet
class TrainOnlyScaler:          # standardization; stores fit_indices
    def fit(self, X: pd.DataFrame, train_idx: pd.DatetimeIndex) -> "TrainOnlyScaler"
    def transform(self, X: pd.DataFrame) -> pd.DataFrame
    fit_indices: pd.DatetimeIndex
def leakage_probe(build: Callable[[pd.DataFrame, Snapshot], pd.DataFrame],
                  targets, snap, t: pd.Timestamp, rng) -> bool
    # perturbs all raw inputs strictly after t, rebuilds, returns True iff X.loc[:t] identical
```

Contracts:
- `X` rows exist only where **all** requested features and the label are defined (warmup rows and
  the last h rows are dropped; the dropped-row accounting is exposed).
- Feature values at date t must be exactly reproducible from raw data ≤ t (bitwise, not approx).
- `target` ∈ {"rv_tv", "rv_oc"}; `horizon` ∈ {1, 5} (glossary terms, §9).

## 5. Dependencies

`data` only (manifesto §6). Consumed by the model harness; `splits` provides the indices used for
scaler fitting at the experiment layer.

## 6. Tech stack (this module)

`pandas`, `numpy`. No sklearn dependency for the scaler (a 10-line mean/std class keeps the
fit-index contract explicit and inspectable). Pinned in `requirements.txt`.

## 7. Requirements & behavior

R1. Every feature column must be finite (no NaN/inf) over `X`'s index.
R2. Every feature must be computable causally: implemented via shift/rolling-on-past only; no
    centered windows, no full-series statistics (those belong in scalers, which are train-fit).
R3. `TrainOnlyScaler.fit` must record `fit_indices` and refuse to transform if never fit;
    fit-indices ⊆ train indices is asserted at the experiment layer and testable here.
R4. Labels: `y_t^{(h)}` must average exactly the next h daily target values, no partial windows
    (rows without h future days are dropped).
R5. `leakage_probe` must perturb **all** raw inputs after t (prices, volume, VIX) with random
    multiplicative noise and verify `X.loc[:t]` is bitwise identical; it must also verify labels
    at origins whose windows end ≤ t are unchanged.
R6. Tier membership is fixed by this spec (pre-registered); experiments select tiers, never
    individual columns (prevents post-hoc column cherry-picking).
R7. Column names must be stable, snake_case, prefixed by tier (`t1_`, `t2_`, `t3_`).

## 8. Edge cases & error handling

E1. Requested tier with a data column missing from the snapshot (e.g. VIX absent) → ValueError
    naming the tier and column.
E2. Horizon not in {1, 5} → ValueError (frozen horizons, §8; no silent generalization).
E3. Zero target value would make log-target −inf → surfaced as a hard error listing dates
    (consistent with `data` E5; expected never on real SPY).
E4. Zero-variance feature on the fit span (scaler std == 0) → scaler must error, not divide by
    zero; the feature must be fixed or dropped explicitly at the experiment layer.
E5. `X`/`y` index mismatch after assembly → internal assertion error (contract, not user error).
E6. VIX forward-filled dates (data E3): features built from ffilled VIX are still "known at close
    of t" (the fill uses only past values) — permitted, but the ffill count is logged.

## 9. Success criteria (executable)

Each maps to a pytest in `tests/test_features.py` (manifesto §6 done-check + S3):

S1. Future-perturbation leakage test passes at multiple probe dates t (early, middle, late) for
    the full Tier 1+2+3 build, both targets, both horizons.
S2. Every scaler's `fit_indices` ⊆ supplied train indices; transform of val/test uses train
    statistics (fixture: transformed train mean ≈ 0, val mean ≠ 0 in general).
S3. All feature columns finite on the real snapshot.
S4. Label alignment fixture: on a toy series 1..10, `y^{(5)}` at origin t equals the hand-computed
    mean of t+1..t+5 and the last 5 origins are dropped.
S5. HAR-style aggregate columns at t match hand-computed means of lagged values on a fixture.
S6. E1–E5 raise as specified.

## 10. Open questions

- **The manifesto names a "Tier 1/2/3 aux menu" but never enumerates it** (manifesto gap —
  flagged for the workflow audit). The menu in §2 is this spec's pre-registered proposal, chosen
  to need no new data (§4). Confirm or amend *before* Stage 2 runs; after test-region contact it
  is frozen.
- Whether label construction belongs here or in `data` — the manifesto assigns "canonical target
  series" to `data` and feature/label assembly is placed here; boundary adopted, flagged.
- Log-space vs level-space for Tier 2/3 columns is fixed as specified above; scaler
  standardization makes most choices affine-equivalent for the net, but the choice is
  pre-registered to avoid drift.

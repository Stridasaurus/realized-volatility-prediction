# SPEC — `splits` (`src/splits.py`)

> Layer 2. Inherits from `MANIFESTO.md` (v3, canonical). Implements the `splits` module of
> manifesto §6. If anything here contradicts the manifesto, the manifesto wins.

## 1. Purpose

The **one** canonical walk-forward splitter — single source of truth for every temporal split in
the project (manifesto §6 `splits`, §7 "There is ONE walk-forward splitter"). Every experiment and
every model, classical or neural, obtains its train/val/test boundaries and its monthly retrain
folds from this module and nowhere else. If two experiments could disagree on splitting, the
comparison is void (§0, Spine).

## 2. Scope

Produce, from a trading-day date index alone:

1. The **frozen three-way chronological split** — train / val / test — with an embargo gap at each
   boundary.
2. The **monthly walk-forward retrain folds** inside the test region, each with a
   **fixed-length rolling fit window** (length = the initial train span, a frozen control).

**Non-goals**
- Never touches prices, targets, features, or any model object — operates on a date index alone.
- Does not choose the embargo length; it receives a resolved integer (see Interfaces). The
  resolution rule (embargo = longest configured lookback anywhere) lives with config handling.
- Does not implement nested walk-forward (named future work, manifesto §8).
- No cadence or window-length sweep — retrain cadence (monthly) and rolling length are frozen
  controls (§4), not parameters to explore.

## 3. Inherited invariants

- **NEVER shuffle the time series. All splits are chronological.** (§7)
- ALWAYS keep the embargo gap between splits so multi-day lookback windows never straddle a
  boundary. (§7)
- **The test region is touched EXACTLY ONCE per pre-registered experiment** — this module must
  make the test boundary explicit and immutable so that discipline is enforceable. (§7)
- Monthly retrain uses a **fixed-length rolling window** (Giacomini–White finite-memory
  condition), never expanding. (§8)
- Frozen controls live in `configs/default.yaml` and never change between experiments. (§7)
- Deterministic: identical inputs → identical splits, always.

## 4. Interfaces / contracts

Input: a strictly increasing, duplicate-free `pandas.DatetimeIndex` of trading days (the
trading-day calendar owned by `data`), plus a config object.

```python
@dataclass(frozen=True)
class SplitConfig:
    train_start: str      # calendar date, e.g. "2005-01-01"
    val_start: str        # "2018-01-01"
    test_start: str       # "2020-01-01"
    test_end: str         # snapshot end date from the data manifest, e.g. "2026-06-30"
    embargo_days: int     # trading days; resolved upstream as max over all configured lookbacks

@dataclass(frozen=True)
class CanonicalSplit:
    train_idx: pd.DatetimeIndex   # dates in [train_start, val_start) minus trailing embargo
    val_idx: pd.DatetimeIndex     # dates in [val_start, test_start) minus trailing embargo
    test_idx: pd.DatetimeIndex    # dates in [test_start, test_end]

@dataclass(frozen=True)
class RetrainFold:
    fit_idx: pd.DatetimeIndex     # fixed-length rolling window, ends >= embargo before test month
    test_idx: pd.DatetimeIndex    # one calendar month of trading days inside the test region

def canonical_split(index: pd.DatetimeIndex, cfg: SplitConfig) -> CanonicalSplit: ...
def retrain_folds(index: pd.DatetimeIndex, cfg: SplitConfig,
                  window_len: int | None = None) -> list[RetrainFold]: ...
```

Contracts:
- Boundary dates are **calendar dates** interpreted against the trading-day index ("first trading
  day ≥ boundary"). Embargo is counted in **trading days**.
- The embargo is taken out of the *earlier* segment at each boundary (train loses its last
  `embargo_days` rows before val; val loses its last `embargo_days` rows before test), so the test
  region is never shrunk.
- `window_len` defaults to the length in trading days of the initial train segment (after embargo
  removal) — the frozen control. Each fold's `fit_idx` is exactly `window_len` trading days ending
  `embargo_days` before the first date of the fold's test month.
- Fold test months partition the test region: first fold's test month starts at `test_start`; the
  final fold's test month may be a partial month ending at `test_end`.

## 5. Dependencies

None (manifesto §6: "Depends on: nothing"). Uses `pandas`/`numpy` only. Consumed by `baselines`,
`features` (indirectly via experiments), and the model harness.

## 6. Tech stack (this module)

Pure Python + `pandas` + `numpy`, standard-library `dataclasses`. No modeling libraries, no I/O,
no network. Versions pinned in the project-level `requirements.txt` (manifesto §10).

## 7. Requirements & behavior

R1. `canonical_split` must return three chronologically ordered, mutually disjoint index sets with
    train entirely before val entirely before test.
R2. The gap between the last train date and the first val date, and between the last val date and
    the first test date, must be ≥ `embargo_days` trading days.
R3. `retrain_folds` must produce one fold per calendar month of the test region, in order; the
    union of fold `test_idx` must equal `CanonicalSplit.test_idx` exactly (no gaps, no overlap).
R4. Every fold's `fit_idx` must have exactly `window_len` entries and end ≥ `embargo_days` trading
    days before the fold's first test date.
R5. Every fold must be chronological: `fit_idx.max() < test_idx.min()`.
R6. Both functions must be pure and deterministic — no randomness, no clock, no I/O.
R7. Both functions must validate the input index (monotonic increasing, unique, tz-naive) and the
    config (boundary order, positive embargo) and raise `ValueError` with a specific message on
    violation — never silently repair.
R8. The rolling window may reach back into the embargo-trimmed zones and across the val region —
    the embargo protects *evaluation* boundaries; fit windows sliding forward through time is the
    walk-forward design (§8: rolling retrain through the test region). The only hard rule is R4/R5:
    the window ends ≥ embargo before its own test month.

## 8. Edge cases & error handling

E1. Index shorter than `window_len` + embargo + one test month → `ValueError` (never a truncated
    window; the fixed length is a frozen control, R4 is absolute).
E2. Boundary calendar date falls on a non-trading day → resolve to the first trading day ≥ that
    date; document in the docstring.
E3. Embargo so large that train or val becomes empty → `ValueError`.
E4. Final partial month with zero trading days after `test_end` trimming → drop the empty fold,
    never emit an empty `test_idx`.
E5. Duplicate or non-monotonic dates in the index → `ValueError` (upstream data bug; do not sort
    silently).
E6. `test_end` beyond the last index date → `ValueError` (the manifest's pinned end date and the
    calendar must agree; "test = 2020 → snapshot end, never → present", §7).

## 9. Success criteria (executable)

Each maps to a pytest in `tests/test_splits.py` (manifesto §6 done-check):

S1. Zero overlap between any fit window and its test fold; zero overlap among train/val/test.
S2. Measured gap at each split boundary ≥ configured embargo.
S3. Every fold chronological (fit strictly before test).
S4. Every fit window exactly `window_len` long.
S5. Same inputs twice → identical outputs (determinism).
S6. Union of fold `test_idx` == canonical `test_idx`, no gaps, no overlaps (property test over a
    synthetic multi-year index).
S7. Each edge case E1–E6 raises/behaves as specified.

## 10. Open questions

- Exact `embargo_days` value: **resolved by decision** — auto-computed as the longest configured
  lookback anywhere (HAR monthly lag 22, LSTM max sequence length, longest aux window), by the
  config loader, not by this module. This module only asserts it receives a positive int.
- Whether `window_len` should be recorded into run artifacts by the harness (recommended: yes —
  it is a frozen control; belongs in `config.json` per run). Owned by `io`, noted here for trace.

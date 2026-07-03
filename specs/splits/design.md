# design — `splits` (`src/splits.py`)

## 1. Overview

Implements `specs/splits/SPEC.md`: the canonical walk-forward splitter — frozen three-way
chronological split with embargo, plus monthly retrain folds with a fixed-length rolling fit
window. Greenfield: no `src/` exists yet; this file and its test are the first code in the repo.

## 2. Approach / architecture

Pure functions over a `pd.DatetimeIndex` + two frozen dataclasses. All position math is done in
integer index positions (via `index.searchsorted`) and converted back to date slices at the end —
this keeps embargo arithmetic exact in trading days and avoids calendar edge cases. Month folds
are derived from `index.to_period("M")` groupby on the test slice, so partial first/last months
fall out naturally.

## 3. File-by-file plan

- `src/__init__.py` — empty package marker (create once, shared by all modules).
- `src/splits.py` — `SplitConfig`, `CanonicalSplit`, `RetrainFold`, `canonical_split`,
  `retrain_folds`, private `_validate_index`, `_pos` (calendar date → first position ≥ date).
- `tests/test_splits.py` — S1–S7 from the spec.
- `configs/default.yaml` — gains the `splits:` block (boundaries, `embargo_days: auto`); file is
  created by the `io` design; this module never reads it directly (receives ints).

## 4. Data models / schemas

Per the spec's dataclasses verbatim. Internal representation: `(start, stop)` integer positions
into the validated index; dataclass fields hold `pd.DatetimeIndex` slices (views, not copies).

## 5. Key interfaces & signatures

As specified in SPEC §4 (`canonical_split`, `retrain_folds`). One addition (private):

```python
def _month_groups(test_idx: pd.DatetimeIndex) -> list[pd.DatetimeIndex]  # ordered, non-empty
```

## 6. Implementation sequence

1. `src/__init__.py`, `_validate_index` (monotonic ↑, unique, tz-naive, else ValueError).
2. `_pos` + boundary resolution; `canonical_split` with embargo trimmed off the earlier segment;
   emptiness checks (E3).
3. `_month_groups`; `retrain_folds`: for each month, `end = pos(first_test_date) - embargo`,
   `fit = index[end - window_len : end]`; length check (E1).
4. Default `window_len = len(train_idx)` from a fresh `canonical_split` inside `retrain_folds`
   when not supplied.
5. Tests S1–S7 on a synthetic 2004–2026 business-day index; property test loops all folds.

## 7. Integration points

Consumed by `src/baselines.py` (walk-forward driver) and `src/train.py` (fold loop); the
trading-day index argument comes from `src/data.py::trading_calendar`. Config resolution of
`embargo_days: auto` happens in `src/io.py::load_config` (io SPEC R4) — splits stays pure.

## 8. Test plan

Maps 1:1 to SPEC §9: S1 disjointness/overlap; S2 measured embargo gaps; S3 chronology per fold;
S4 exact window length on every fold; S5 determinism (two calls, `.equals()`); S6 union of fold
test indices equals canonical test index (assert concatenation equality); S7 parametrized
E1–E6 raises. Fixture: `pd.bdate_range("2004-01-01","2026-06-30")` (synthetic; no data needed).

## 9. Risks & open questions

- Trading-day vs business-day drift: tests use bdate_range, production uses the real SPY
  calendar — no code path depends on weekday regularity, only on index order. Low risk.
- R8 (fit windows may cross val region) is deliberate; documented in the docstring so a reviewer
  doesn't "fix" it into a leak-paranoid bug. The embargo before each fold's own test month is
  the binding rule.

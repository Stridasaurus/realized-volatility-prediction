# SPEC — `data` (`src/data.py`)

> Layer 2. Inherits from `MANIFESTO.md` (v3, canonical). Implements the `data` module of
> manifesto §6. If anything here contradicts the manifesto, the manifesto wins.

## 1. Purpose

Turn the frozen price/VIX snapshot into the canonical target series (manifesto §6 `data`). Owns
snapshot loading and verification, both v1 target constructions (TV, OC), the S8 calibration
diagnostics, the documented adjustment basis, and the trading-day calendar that indexes
everything downstream.

## 2. Scope

1. **Snapshot contract**: load the committed raw CSVs (Stooq SPY OHLCV, CBOE VIX OHLC), verify
   each file's SHA-256 against the committed manifest, verify the manifest's pinned end date
   (2026-06-30) matches the data.
2. **One-time freeze tooling** (`scripts/freeze_snapshot.py`, invoked once, never at experiment
   time): download the raw files, trim to the pinned end date, write the manifest
   (file hashes, end date, retrieval timestamp, source URLs, adjustment basis), and compute the
   per-target QLIKE floors (1st percentile of the train-span target) into `configs/default.yaml`.
3. **Target construction** on the adjusted basis:
   - **TV** (`rv_tv`, primary): per-day Rogers–Satchell + squared overnight return.
     `RS_t = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)`; `overnight²_t = (ln(O_t / C_{t-1}))²`;
     `rv_tv = RS + overnight²`.
   - **OC** (`rv_oc`, secondary): per-day Garman–Klass.
     `GK_t = 0.5·(ln(H/L))² − (2·ln2 − 1)·(ln(C/O))²`.
4. **Calibration diagnostics (S8)**: `mean(GK)/mean(r²_oc)` and `mean(rv_tv)/mean(r²_cc)` on the
   train span, where `r²_oc = (ln(C/O))²` and `r²_cc = (ln(C_t/C_{t-1}))²`; both must fall in the
   configured tolerance band (default [0.8, 1.25]).
5. **Trading-day calendar**: the DatetimeIndex of the loaded SPY series after warmup — the index
   every downstream module aligns to. VIX is reindexed onto it (see E4).

**Non-goals**
- No features, no splitting, no models (manifesto §6 "Doesn't own").
- No horizon-h label alignment (supervised (X, y) assembly belongs to `features`).
- **No network access at experiment time** — the freeze script is the only code that touches the
  network, and it is never imported by `src/`.

## 3. Inherited invariants

- **Data is a frozen, committed, checksummed snapshot with a pinned end date in its manifest. NO
  module pulls live data at experiment time.** "Test = 2020 → snapshot end," never "→ present." (§7)
- **ONE documented adjustment basis: adjusted prices.** The TV overnight leg crosses the dividend
  boundary and **requires** the adjusted basis (v1 requirement); **NEVER mix an adjusted close
  with raw OHLC inside one range estimator.** (§7)
- Both targets are **variances**, never volatilities. (§9)
- OHLC internal consistency must hold on every row on one basis:
  `low ≤ min(open, close) ≤ max(open, close) ≤ high`. (§6 done-check)
- The proxies are checked, not assumed (S8). (§3)

## 4. Interfaces / contracts

```python
@dataclass(frozen=True)
class Snapshot:
    spy: pd.DataFrame        # index: trading days; columns open, high, low, close, volume (adjusted basis)
    vix: pd.DataFrame        # index: same calendar (reindexed); columns open, high, low, close
    manifest: dict           # parsed data/snapshot_manifest.json

def load_snapshot(data_dir: Path = DATA_DIR, *, verify: bool = True) -> Snapshot
def build_targets(snap: Snapshot) -> pd.DataFrame
    # index: trading-day calendar (post-warmup); columns: rv_tv, rv_oc  (variance space)
def calibration_diagnostics(snap: Snapshot, targets: pd.DataFrame,
                            train_end: str, band: tuple[float, float]) -> dict
    # {"tv_ratio": float, "oc_ratio": float, "band": band, "pass": bool}
def trading_calendar(snap: Snapshot) -> pd.DatetimeIndex
```

Manifest schema (`data/snapshot_manifest.json`): `{"end_date", "retrieved_utc", "sources":
{name: url}, "files": {relpath: sha256}, "adjustment_basis": "adjusted", "notes"}`.

Raw layout: `data/raw/spy_ohlcv.csv`, `data/raw/vix_ohlc.csv`, `data/snapshot_manifest.json` —
all committed (daily CSVs are small; manifesto §10 says commit the snapshot, skip git-LFS).

## 5. Dependencies

The committed snapshot only (manifesto §6). No project modules. Consumed by `features`,
`baselines`, and experiments.

## 6. Tech stack (this module)

`pandas`, `numpy`, `hashlib` (checksums), `json`. Freeze script additionally uses `urllib`/
`requests` against Stooq (`stooq.com` daily CSV endpoint, symbol `spy.us`) and CBOE
(`VIX_History.csv`). Versions pinned in `requirements.txt`.

## 7. Requirements & behavior

R1. `load_snapshot(verify=True)` must recompute each file's SHA-256 and fail loudly on any
    mismatch with the manifest; must fail if the last SPY date ≠ manifest `end_date`.
R2. `build_targets` must produce both targets non-negative with no NaN after warmup (warmup =
    the first row, which lacks a prior close for the overnight leg, plus any leading rows dropped
    for missing data). RS is non-negative by construction for valid OHLC; assert ≥ 0 after the
    OHLC consistency check rather than clipping.
R3. OHLC internal consistency must be validated on every SPY row at load; any violating row is a
    hard error naming the dates (a mixed adjustment basis corrupts both estimators — §8).
R4. The overnight return must use adjusted close_{t-1} and adjusted open_t (same basis, R3).
R5. `calibration_diagnostics` must compute both S8 ratios on the **train span only** (boundary
    passed in; splits module owns the split) and report pass/fail against the band.
R6. The freeze script must: trim strictly to `end_date`; write the manifest; compute each
    target's QLIKE floor as the 1st percentile of the train-span target and write both floors
    into `configs/default.yaml`; be idempotent (re-running against unchanged sources reproduces
    byte-identical committed files or exits telling the user the source drifted).
R7. Stooq data must be spot-checked at freeze time against a second source for level sanity
    (the freeze script prints summary stats + a few known dates for eyeball check; automated
    second-source cross-check is Stage 3, §5).
R8. All prices parsed as float64; dates tz-naive; index strictly increasing, unique.

## 8. Edge cases & error handling

E1. Zero or negative price in any OHLC field → hard error naming rows (log of it is undefined).
E2. `H == L` (flat day): GK/RS terms are 0 via `ln(H/L)=0` — valid, keep; but `H == L != O or C`
    violates R3 and errors.
E3. Missing trading day in SPY vs VIX (different holiday calendars): SPY calendar is canonical;
    VIX reindexed onto it, forward-filled ≤ 2 days, else NaN and reported. VIX gaps never remove
    a SPY day.
E4. Duplicate dates in a raw CSV → hard error (source corruption; the snapshot is frozen, fix at
    freeze time, never auto-dedupe at load).
E5. `rv_tv == 0` exactly (flat day and zero gap): permitted in the series (targets are
    non-negative, §6); its interaction with QLIKE's log(actual) appears downstream — `metrics`
    rejects non-positive actuals, so any zero-target date must be surfaced by `build_targets` as
    a named warning listing dates (expected count on real SPY: zero).
E6. Manifest present but raw file missing / vice versa → hard error; the snapshot is atomic.

## 9. Success criteria (executable)

Each maps to a pytest in `tests/test_data.py` (manifesto §6 done-check):

S1. Snapshot checksum matches committed hash; last date == manifest pinned end date.
S2. Both targets non-negative, no NaN after warmup, on the real snapshot.
S3. GK and RS+overnight² each match a hand-computed value on a small OHLC fixture (closed-form
    arithmetic in the test).
S4. OHLC internal consistency holds on every committed SPY row.
S5. Calibration ratios inside the tolerance band on the train span (band from config).
S6. Loading with a tampered byte in a raw CSV fails checksum verification.
S7. Perturbation of E1–E6 fixtures behaves as specified.

## 10. Open questions

- Stooq's SPY history is dividend/split adjusted across all OHLC columns (their documented
  default for US ETFs). The freeze script must verify internal consistency (R3) and eyeball-check
  levels (R7); if Stooq's basis proves mixed, fallback source must be decided then — flagged as
  the single most likely external failure point of v1.
- Exact warmup row count (default: 1 row for the overnight leg).
- Whether VIX belongs in the committed snapshot from day one even though aux features are a
  Stage-2 experiment (default here: yes — freeze once, §4 "no new data" boundary).
- The S8 band [0.8, 1.25] is the manifesto's stated default; confirmed as adopted.

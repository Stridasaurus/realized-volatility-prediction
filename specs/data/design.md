# design — `data` (`src/data.py` + `scripts/freeze_snapshot.py`)

## 1. Overview

Implements `specs/data/SPEC.md`: snapshot load/verify, TV & OC target construction, S8
calibration diagnostics, trading calendar, plus the one-time freeze script. Greenfield.

## 2. Approach / architecture

Two layers with a hard wall: `scripts/freeze_snapshot.py` (network, run once, writes
`data/raw/*.csv` + `data/snapshot_manifest.json`, updates `configs/default.yaml` floors) and
`src/data.py` (pure load/compute, no network import anywhere). Targets are vectorized
numpy-on-columns; all logs computed once into a small intermediate frame.

Sources: Stooq `https://stooq.com/q/d/l/?s=spy.us&d1=20050101&d2=20260630&i=d`;
CBOE `https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv` (trimmed to the
same end date).

## 3. File-by-file plan

- `src/data.py` — `Snapshot`, `load_snapshot`, `build_targets`, `calibration_diagnostics`,
  `trading_calendar`, privates `_sha256`, `_check_ohlc`, `_read_spy`, `_read_vix`.
- `scripts/freeze_snapshot.py` — `main()`: download → trim → OHLC-consistency + eyeball report
  (R7 summary stats) → write raw CSVs + manifest → compute per-target floors (1st pct of
  train-span target, train boundary from `configs/default.yaml`) → patch config → print diff.
- `data/raw/spy_ohlcv.csv`, `data/raw/vix_ohlc.csv`, `data/snapshot_manifest.json` — committed
  artifacts produced by the script.
- `tests/test_data.py` — S1–S7; synthetic OHLC fixture module-local.

## 4. Data models / schemas

`Snapshot(spy, vix, manifest)` per SPEC. Manifest JSON exactly per SPEC §4. Targets frame:
index = calendar, columns `rv_tv`, `rv_oc`, float64, first row dropped (warmup).

## 5. Key interfaces & signatures

Per SPEC §4. Fixture-facing helpers `garman_klass(o,h,l,c)` and `rogers_satchell(o,h,l,c)` are
exposed as module-level functions so tests hand-compute against the same public formula.

## 6. Implementation sequence

1. `garman_klass`, `rogers_satchell`, `_check_ohlc` + unit fixtures (S3, S4 logic).
2. `_read_spy`/`_read_vix` (CSV dialects), `load_snapshot` with `_sha256` verification (S1, S6).
3. `build_targets` (overnight leg from shifted adjusted close; warmup drop; non-negativity
   asserts; zero-target warning E5).
4. `calibration_diagnostics` (train-span ratios vs band).
5. `freeze_snapshot.py` (download, trim, manifest, floors); run it; commit the snapshot.
6. Real-snapshot tests (S2, S4, S5) marked to skip if `data/raw` absent, so CI-before-freeze
   still passes.

## 7. Integration points

`trading_calendar` feeds `src/splits.py` inputs at the experiment layer; `build_targets` feeds
`src/features.py` and `src/baselines.py`; close-to-close returns for GARCH are derived in
baselines from `Snapshot.spy["close"]` (adjusted basis, SPEC data R4). Floor values written by
the freeze script are read back through `io.load_config` (io R4) — `metrics` receives them as
arguments.

## 8. Test plan

SPEC §9 S1–S7. Hand-computed fixture: 3-day OHLC table with round numbers; GK/RS/overnight²
computed with explicit arithmetic in test comments. Tamper test: copy raw CSV to tmp, flip one
byte, expect checksum failure. Real-snapshot tests parametrized over both targets.

## 9. Risks & open questions

- **Stooq basis risk** (SPEC Open Q1): if Stooq's OHLC turns out internally inconsistent
  (mixed adjustment), the freeze fails loudly at R3 and the fallback source decision goes to the
  user — the single most likely external blocker.
- Stooq may rate-limit or require a session for full history; freeze script retries once and
  otherwise reports — manual download is the documented fallback (paste URL in browser, save to
  data/raw/).
- VIX holiday mismatches (E3): ffill ≤ 2 days is a policy constant; logged count surfaces drift.

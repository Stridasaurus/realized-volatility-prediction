"""One-time data freeze (specs/data/SPEC.md section 2.2). Never imported by src/.

Canonicalizes the raw sources into the committed snapshot:

- SPY: Stooq now login-gates CSV export, so the input is a *manual* export from a
  logged-in session (``data/raw/spy_us_d.csv``, all columns already dividend/split
  adjusted — Stooq's documented default for US ETFs). Canonicalized to
  ``data/raw/spy_ohlcv.csv`` with an explicit ``adj_close`` column (== close, since
  the basis is already adjusted; the loader's adjustment factor is then exactly 1).
- VIX: downloaded from CBOE's official history endpoint (the only network call here).

Then: trim to the pinned end date, write ``data/snapshot_manifest.json`` (SHA-256 per
file), compute each target's QLIKE floor (1st percentile of the train-span target) into
``configs/default.yaml``, and print the R7 eyeball-check summary.

Idempotent: re-running against unchanged sources reproduces byte-identical files;
a drifted source is reported and nothing is overwritten unless --force is given.

Usage: python scripts/freeze_snapshot.py [--offline] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # repo root (NOT src/ — stdlib-shadowing guard)

from src.data import build_targets, calibration_diagnostics, load_snapshot  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
STOOQ_EXPORT = DATA_DIR / "raw" / "spy_us_d.csv"
SPY_OUT = DATA_DIR / "raw" / "spy_ohlcv.csv"
VIX_OUT = DATA_DIR / "raw" / "vix_ohlc.csv"
MANIFEST = DATA_DIR / "snapshot_manifest.json"

END_DATE = "2026-06-30"
VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
STOOQ_NOTE = (
    "SPY source is a manual Stooq export (spy_us_d.csv) from a logged-in browser "
    "session (Stooq login-gates programmatic CSV export as of 2026-07); all OHLC "
    "columns are dividend/split adjusted, so adj_close == close by construction."
)


def _canonical_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True, lineterminator="\n").encode("utf-8")


def canonicalize_spy() -> bytes:
    if not STOOQ_EXPORT.exists():
        sys.exit(f"missing Stooq export: {STOOQ_EXPORT}")
    df = pd.read_csv(STOOQ_EXPORT)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[: pd.Timestamp(END_DATE)]
    df = df[["open", "high", "low", "close", "volume"]].astype("float64")
    df["adj_close"] = df["close"]  # basis already adjusted (see module docstring)
    return _canonical_csv(df)


def fetch_vix(offline: bool) -> bytes:
    if offline:
        if not VIX_OUT.exists():
            sys.exit("--offline given but no existing vix_ohlc.csv to reuse")
        return VIX_OUT.read_bytes()
    print(f"downloading VIX history from {VIX_URL} ...")
    with urllib.request.urlopen(VIX_URL, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(StringIO(raw))
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[pd.Timestamp("2005-01-01") : pd.Timestamp(END_DATE)]
    df = df[["open", "high", "low", "close"]].astype("float64")
    return _canonical_csv(df)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_floors_into_config(floors: dict[str, float]) -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    for target, value in floors.items():
        pattern = rf"(^\s+{target}:).*$"
        replacement = rf"\g<1> {value:.6e}"
        text, n = re.subn(pattern, replacement, text, count=1, flags=re.M)
        if n != 1:
            sys.exit(f"could not find 'floors.{target}' line in {CONFIG_PATH}")
    CONFIG_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--offline",
        action="store_true",
        help="reuse the existing vix_ohlc.csv instead of downloading",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite committed snapshot files that differ from sources",
    )
    args = ap.parse_args()

    new_files = {
        "raw/spy_ohlcv.csv": canonicalize_spy(),
        "raw/vix_ohlc.csv": fetch_vix(args.offline),
    }

    # Idempotency (R6): unchanged sources must reproduce byte-identical files.
    drifted = []
    for rel, blob in new_files.items():
        p = DATA_DIR / rel
        if p.exists() and p.read_bytes() != blob:
            drifted.append(rel)
    if drifted and not args.force:
        sys.exit(
            f"source drift detected for {drifted}; the snapshot is frozen — "
            "re-run with --force only if a re-freeze is intended"
        )
    for rel, blob in new_files.items():
        (DATA_DIR / rel).write_bytes(blob)

    manifest = {
        "end_date": END_DATE,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "spy": "https://stooq.com/q/d/?s=spy.us (manual export)",
            "vix": VIX_URL,
        },
        "files": {rel: _sha256(blob) for rel, blob in new_files.items()},
        "adjustment_basis": "adjusted",
        "notes": STOOQ_NOTE,
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if old.get("files") == manifest["files"]:
            print("snapshot unchanged — keeping the committed manifest timestamp")
            manifest = old
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Verify the frozen snapshot end-to-end through the real loader.
    snap = load_snapshot(DATA_DIR, verify=True)
    targets = build_targets(snap)

    # QLIKE floors: 1st percentile of the train-span target (train = before val_start).
    cfg_text = CONFIG_PATH.read_text(encoding="utf-8")
    val_start = re.search(r'val_start:\s*"([\d-]+)"', cfg_text).group(1)
    train = targets.loc[: pd.Timestamp(val_start) - pd.Timedelta(days=1)]
    floors = {t: float(np.percentile(train[t], 1)) for t in ("rv_tv", "rv_oc")}
    write_floors_into_config(floors)

    band = tuple(
        float(x)
        for x in re.search(
            r"calibration_band:\s*\[([\d.]+),\s*([\d.]+)\]", cfg_text
        ).groups()
    )
    diag = calibration_diagnostics(snap, targets, val_start, band)

    # R7 eyeball check: summary stats + known dates against a second source by hand.
    print(
        f"\nSPY rows {len(snap.spy)}  {snap.spy.index[0].date()} -> "
        f"{snap.spy.index[-1].date()}"
    )
    print(f"VIX rows {snap.vix['close'].notna().sum()} on the SPY calendar")
    ann_vol = np.sqrt(targets["rv_tv"].mean() * 252)
    print(
        f"mean rv_tv {targets['rv_tv'].mean():.3e} (annualized vol "
        f"{ann_vol:.1%}), mean rv_oc {targets['rv_oc'].mean():.3e}"
    )
    for d in ("2008-10-10", "2020-03-23", "2022-01-03", END_DATE):
        ts = pd.Timestamp(d)
        if ts in snap.spy.index:
            print(
                f"  SPY close {d}: {snap.spy.loc[ts, 'close']:.2f}   "
                f"VIX close: {snap.vix.loc[ts, 'close']:.2f}"
            )
    print(f"QLIKE floors written to config: {floors}")
    print(
        f"S8 calibration: tv_ratio={diag['tv_ratio']:.3f} "
        f"oc_ratio={diag['oc_ratio']:.3f} band={band} pass={diag['pass']}"
    )
    if not diag["pass"]:
        sys.exit("S8 calibration diagnostics OUT OF BAND — investigate before scoring")


if __name__ == "__main__":
    main()

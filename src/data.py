"""Snapshot loading/verification and the canonical target series (TV, OC).

Spec: specs/data/SPEC.md. No network access anywhere in this module — the one-time
freeze lives in scripts/freeze_snapshot.py and is never imported from src/.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

_GK_CONST = 2.0 * np.log(2.0) - 1.0


@dataclass(frozen=True)
class Snapshot:
    spy: pd.DataFrame   # adjusted basis; columns: open, high, low, close, volume
    vix: pd.DataFrame   # reindexed to the SPY calendar; columns: open, high, low, close
    manifest: dict


def garman_klass(o, h, l, c):
    """Per-day Garman–Klass variance (the OC target). Adjustment-invariant."""
    return 0.5 * np.log(h / l) ** 2 - _GK_CONST * np.log(c / o) ** 2


def rogers_satchell(o, h, l, c):
    """Per-day Rogers–Satchell variance. Non-negative for internally consistent OHLC."""
    return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_ohlc(df: pd.DataFrame, name: str) -> None:
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        bad = df.index[(df[["open", "high", "low", "close"]] <= 0).any(axis=1)]
        raise ValueError(f"{name}: non-positive price on {list(bad[:10])}")
    lo_ok = df["low"] <= df[["open", "close"]].min(axis=1) + 1e-12
    hi_ok = df[["open", "close"]].max(axis=1) <= df["high"] + 1e-12
    bad = df.index[~(lo_ok & hi_ok)]
    if len(bad):
        raise ValueError(
            f"{name}: OHLC internal consistency violated on {len(bad)} row(s), e.g. "
            f"{list(bad[:10])} — mixed adjustment basis silently corrupts both range "
            "estimators; fix at freeze time"
        )


def _read_raw(path: Path, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df[date_col] = pd.to_datetime(df[date_col], format="mixed")
    df = df.set_index(date_col).sort_index()
    if df.index.has_duplicates:
        dup = df.index[df.index.duplicated()][:5]
        raise ValueError(f"{path.name}: duplicate dates {list(dup)} — source corruption; "
                         "fix at freeze time, never auto-dedupe at load")
    return df.astype("float64")


def load_snapshot(data_dir: Path = DATA_DIR, *, verify: bool = True) -> Snapshot:
    manifest_path = data_dir / "snapshot_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"snapshot manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if verify:
        for rel, expected in manifest["files"].items():
            p = data_dir / rel
            if not p.exists():
                raise FileNotFoundError(f"snapshot file missing: {p} (snapshot is atomic)")
            got = _sha256(p)
            if got != expected:
                raise ValueError(f"checksum mismatch for {rel}: manifest {expected[:12]}…, "
                                 f"file {got[:12]}…")

    spy_raw = _read_raw(data_dir / "raw" / "spy_ohlcv.csv", "date")
    vix_raw = _read_raw(data_dir / "raw" / "vix_ohlc.csv", "date")

    if str(spy_raw.index[-1].date()) != manifest["end_date"]:
        raise ValueError(f"last SPY date {spy_raw.index[-1].date()} != manifest end_date "
                         f"{manifest['end_date']}")

    # One documented adjustment basis: scale O/H/L/C by the per-day back-adjustment
    # factor (adj_close/close). Ratios inside a day are preserved (range estimators
    # invariant); the overnight leg crosses days and REQUIRES this basis (manifesto s7).
    factor = spy_raw["adj_close"] / spy_raw["close"]
    spy = spy_raw[["open", "high", "low", "close"]].mul(factor, axis=0)
    spy["volume"] = spy_raw["volume"]
    _check_ohlc(spy, "SPY(adjusted)")

    calendar = spy.index
    vix = vix_raw[["open", "high", "low", "close"]].reindex(calendar).ffill(limit=2)
    n_missing = int(vix["close"].isna().sum())
    if n_missing:
        warnings.warn(f"VIX has {n_missing} calendar dates unfilled after ffill(limit=2)",
                      UserWarning)

    return Snapshot(spy=spy, vix=vix, manifest=manifest)


def build_targets(snap: Snapshot) -> pd.DataFrame:
    """Both v1 targets, variance space: rv_tv = RS + overnight^2 (primary), rv_oc = GK."""
    s = snap.spy
    o, h, l, c = s["open"], s["high"], s["low"], s["close"]
    rs = rogers_satchell(o, h, l, c)
    gk = garman_klass(o, h, l, c)
    overnight2 = np.log(o / c.shift(1)) ** 2
    out = pd.DataFrame({"rv_tv": rs + overnight2, "rv_oc": gk}).iloc[1:]  # warmup: 1 row

    if out.isna().any().any():
        raise ValueError("NaN in targets after warmup")
    if (out < 0).any().any():
        raise ValueError("negative target value — violates RS/GK non-negativity for valid OHLC")
    zeros = out.index[(out == 0).any(axis=1)]
    if len(zeros):
        warnings.warn(f"exact-zero target value on {len(zeros)} date(s): {list(zeros[:10])} "
                      "(degenerate for QLIKE downstream)", UserWarning)
    return out


def trading_calendar(snap: Snapshot) -> pd.DatetimeIndex:
    """The trading-day index everything downstream aligns to (post-warmup)."""
    return snap.spy.index[1:]


def calibration_diagnostics(snap: Snapshot, targets: pd.DataFrame, train_end: str,
                            band: tuple[float, float]) -> dict:
    """S8: proxy-vs-squared-return calibration ratios on the train span."""
    s = snap.spy.loc[: pd.Timestamp(train_end)]
    t = targets.loc[: pd.Timestamp(train_end)]
    r2_oc = (np.log(s["close"] / s["open"]) ** 2).reindex(t.index)
    r2_cc = (np.log(s["close"] / s["close"].shift(1)) ** 2).reindex(t.index)
    oc_ratio = float(t["rv_oc"].mean() / r2_oc.mean())
    tv_ratio = float(t["rv_tv"].mean() / r2_cc.mean())
    ok = bool(band[0] <= oc_ratio <= band[1] and band[0] <= tv_ratio <= band[1])
    return {"tv_ratio": tv_ratio, "oc_ratio": oc_ratio, "band": list(band), "pass": ok}

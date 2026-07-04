"""Shared synthetic fixtures. No test touches the network; the real snapshot is optional
(tests that need it skip until scripts/freeze_snapshot.py has run)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import Snapshot


def make_ohlc(
    index: pd.DatetimeIndex,
    seed: int = 0,
    vol: float = 0.01,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Internally consistent synthetic OHLCV: low <= min(o,c) <= max(o,c) <= high."""
    rng = np.random.default_rng(seed)
    n = len(index)
    close = start_price * np.exp(np.cumsum(rng.normal(0.0, vol, n)))
    prev_close = np.concatenate([[start_price], close[:-1]])
    open_ = prev_close * np.exp(rng.normal(0.0, vol / 4, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, vol / 2, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, vol / 2, n)))
    volume = rng.integers(10**6, 10**8, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def make_snapshot(index: pd.DatetimeIndex, seed: int = 0) -> Snapshot:
    """In-memory Snapshot (already on the adjusted basis) for feature/baseline tests."""
    spy = make_ohlc(index, seed=seed)
    vix = make_ohlc(index, seed=seed + 1, vol=0.05, start_price=20.0).drop(
        columns="volume"
    )
    manifest = {
        "end_date": str(index[-1].date()),
        "adjustment_basis": "adjusted",
        "files": {},
        "sources": {},
        "retrieved_utc": "synthetic",
    }
    return Snapshot(spy=spy, vix=vix, manifest=manifest)


def write_snapshot_dir(
    root: Path,
    spy: pd.DataFrame,
    vix: pd.DataFrame,
    adj_factor: float = 1.0,
    end_date: str | None = None,
) -> Path:
    """Write a raw snapshot directory (CSVs + manifest with real SHA-256 hashes).

    ``spy`` is the *adjusted-basis* frame the loader should reconstruct; the raw CSV is
    written un-adjusted (divided by ``adj_factor``) with adj_close carrying the factor.
    """
    data_dir = root / "data"
    (data_dir / "raw").mkdir(parents=True)
    spy_raw = (spy[["open", "high", "low", "close"]] / adj_factor).copy()
    spy_raw["volume"] = spy["volume"]
    spy_raw["adj_close"] = spy["close"]
    spy_raw.index.name = "date"
    vix_raw = vix.copy()
    vix_raw.index.name = "date"
    spy_path = data_dir / "raw" / "spy_ohlcv.csv"
    vix_path = data_dir / "raw" / "vix_ohlc.csv"
    spy_raw.to_csv(spy_path)
    vix_raw.to_csv(vix_path)
    manifest = {
        "end_date": end_date or str(spy.index[-1].date()),
        "retrieved_utc": "synthetic",
        "sources": {"spy": "synthetic", "vix": "synthetic"},
        "files": {
            "raw/spy_ohlcv.csv": hashlib.sha256(spy_path.read_bytes()).hexdigest(),
            "raw/vix_ohlc.csv": hashlib.sha256(vix_path.read_bytes()).hexdigest(),
        },
        "adjustment_basis": "adjusted",
        "notes": "synthetic test fixture",
    }
    (data_dir / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return data_dir


@pytest.fixture
def calendar_6y() -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-02", "2020-12-31")


@pytest.fixture
def snapshot_small() -> Snapshot:
    return make_snapshot(pd.bdate_range("2020-01-02", "2022-06-30"), seed=7)

"""Executable success criteria S1-S7 from specs/data/SPEC.md.

Real-snapshot criteria (S1/S2/S4/S5 on committed data) skip until
scripts/freeze_snapshot.py has produced data/snapshot_manifest.json.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.data import (
    DATA_DIR,
    Snapshot,
    build_targets,
    calibration_diagnostics,
    garman_klass,
    load_snapshot,
    rogers_satchell,
    trading_calendar,
)
from tests.conftest import make_ohlc, write_snapshot_dir

SNAPSHOT_FROZEN = (DATA_DIR / "snapshot_manifest.json").exists()
needs_snapshot = pytest.mark.skipif(
    not SNAPSHOT_FROZEN, reason="snapshot not frozen yet (run freeze_snapshot)"
)


@pytest.fixture
def snap_dir(tmp_path):
    idx = pd.bdate_range("2020-01-02", "2021-12-31")
    spy = make_ohlc(idx, seed=11)
    vix = make_ohlc(idx, seed=12, vol=0.05, start_price=20.0).drop(columns="volume")
    return write_snapshot_dir(tmp_path, spy, vix, adj_factor=1.05)


# ------------------------------------------------- S1/S6: checksums + pinned end date


def test_s1_load_verifies_checksums_and_end_date(snap_dir):
    snap = load_snapshot(snap_dir, verify=True)
    assert str(snap.spy.index[-1].date()) == snap.manifest["end_date"]
    assert set(snap.spy.columns) == {"open", "high", "low", "close", "volume"}
    # the adj_close/close factor must reconstruct the adjusted basis exactly
    expected = make_ohlc(pd.bdate_range("2020-01-02", "2021-12-31"), seed=11)
    assert np.allclose(snap.spy["close"], expected["close"])
    assert np.allclose(snap.spy["open"], expected["open"])


def test_s1_end_date_mismatch_fails(snap_dir):
    manifest = json.loads((snap_dir / "snapshot_manifest.json").read_text())
    manifest["end_date"] = "1999-12-31"
    # keep file hashes valid: only the manifest's own end_date is wrong
    (snap_dir / "snapshot_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="end_date"):
        load_snapshot(snap_dir, verify=True)


def test_s6_tampered_byte_fails_checksum(snap_dir):
    p = snap_dir / "raw" / "spy_ohlcv.csv"
    raw = p.read_bytes()
    p.write_bytes(raw[:-2] + b"9\n")
    with pytest.raises(ValueError, match="checksum"):
        load_snapshot(snap_dir, verify=True)


# ------------------------------------------------- S2: targets well-formed


def test_s2_targets_nonnegative_no_nan_after_warmup(snap_dir):
    snap = load_snapshot(snap_dir, verify=True)
    t = build_targets(snap)
    assert list(t.columns) == ["rv_tv", "rv_oc"]
    assert not t.isna().any().any()
    assert (t >= 0).all().all()
    assert len(t) == len(snap.spy) - 1  # warmup = 1 row (overnight leg)
    assert t.index.equals(trading_calendar(snap))


# ------------------------------------------------- S3: hand-computed estimators


def test_s3_garman_klass_hand_computed():
    o, h, lo, c = 100.0, 102.0, 99.0, 101.0
    expected = (
        0.5 * math.log(102 / 99) ** 2 - (2 * math.log(2) - 1) * math.log(101 / 100) ** 2
    )
    assert garman_klass(o, h, lo, c) == pytest.approx(expected, abs=1e-15)


def test_s3_rogers_satchell_plus_overnight_hand_computed():
    o, h, lo, c, prev_c = 100.0, 102.0, 99.0, 101.0, 100.5
    rs = math.log(102 / 101) * math.log(102 / 100) + math.log(99 / 101) * math.log(
        99 / 100
    )
    overnight2 = math.log(100.0 / 100.5) ** 2
    assert rogers_satchell(o, h, lo, c) == pytest.approx(rs, abs=1e-15)

    idx = pd.bdate_range("2021-01-04", periods=2)
    spy = pd.DataFrame(
        {
            "open": [100.0, o],
            "high": [101.0, h],
            "low": [99.5, lo],
            "close": [prev_c, c],
            "volume": [1e6, 1e6],
        },
        index=idx,
    )
    vix = spy.drop(columns="volume")
    snap = Snapshot(spy=spy, vix=vix, manifest={"end_date": "2021-01-05"})
    t = build_targets(snap)
    assert t["rv_tv"].iloc[0] == pytest.approx(rs + overnight2, abs=1e-15)
    assert t["rv_oc"].iloc[0] == pytest.approx(garman_klass(o, h, lo, c), abs=1e-15)


# ------------------------------------------------- S4: OHLC internal consistency


def test_s4_ohlc_consistency_enforced_at_load(snap_dir):
    load_snapshot(snap_dir, verify=True)  # passes on a consistent fixture

    df = pd.read_csv(snap_dir / "raw" / "spy_ohlcv.csv")
    df.loc[10, "low"] = df.loc[10, "high"] * 2  # violate low <= high
    df.to_csv(snap_dir / "raw" / "spy_ohlcv.csv", index=False)
    with pytest.raises(ValueError, match="consistency"):
        load_snapshot(snap_dir, verify=False)


# ------------------------------------------------- S5: calibration diagnostics


def test_s5_calibration_mechanics(snap_dir):
    snap = load_snapshot(snap_dir, verify=True)
    t = build_targets(snap)
    train_end = "2021-06-30"
    d = calibration_diagnostics(snap, t, train_end, band=(0.8, 1.25))
    # ratios recomputed independently
    s = snap.spy.loc[: pd.Timestamp(train_end)]
    tt = t.loc[: pd.Timestamp(train_end)]
    r2_oc = (np.log(s["close"] / s["open"]) ** 2).reindex(tt.index).mean()
    r2_cc = (np.log(s["close"] / s["close"].shift(1)) ** 2).reindex(tt.index).mean()
    assert d["oc_ratio"] == pytest.approx(float(tt["rv_oc"].mean() / r2_oc))
    assert d["tv_ratio"] == pytest.approx(float(tt["rv_tv"].mean() / r2_cc))
    assert d["pass"] == (0.8 <= d["oc_ratio"] <= 1.25 and 0.8 <= d["tv_ratio"] <= 1.25)
    # a band nothing falls in must fail
    assert (
        calibration_diagnostics(snap, t, train_end, band=(100.0, 200.0))["pass"]
        is False
    )


# ------------------------------------------------- S7: edge cases E1-E6


def test_e1_nonpositive_price_rejected(snap_dir):
    df = pd.read_csv(snap_dir / "raw" / "spy_ohlcv.csv")
    df.loc[5, "close"] = -1.0
    df.to_csv(snap_dir / "raw" / "spy_ohlcv.csv", index=False)
    with pytest.raises(ValueError, match="non-positive price"):
        load_snapshot(snap_dir, verify=False)


def test_e2_flat_day_valid_but_inconsistent_flat_day_not():
    idx = pd.bdate_range("2021-01-04", periods=3)
    spy = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 100.0, 101.0],
            "low": [99.0, 100.0, 99.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [1e6] * 3,
        },
        index=idx,
    )
    snap = Snapshot(spy=spy, vix=spy.drop(columns="volume"), manifest={})
    with pytest.warns(UserWarning, match="zero"):
        t = build_targets(snap)
    assert t.loc[idx[1], "rv_oc"] == 0.0  # flat day: GK exactly 0, kept

    from src.data import _check_ohlc

    bad = spy.copy()
    bad.loc[idx[1], ["high", "low", "close"]] = [100.0, 100.0, 102.0]  # H == L != C
    with pytest.raises(ValueError, match="consistency"):
        _check_ohlc(bad, "fixture")


def test_e3_vix_gaps_ffilled_and_reported(snap_dir):
    df = pd.read_csv(snap_dir / "raw" / "vix_ohlc.csv")
    df = df.drop(index=range(50, 54)).reset_index(drop=True)  # 4-day VIX hole
    df.to_csv(snap_dir / "raw" / "vix_ohlc.csv", index=False)
    with pytest.warns(UserWarning, match="VIX"):
        snap = load_snapshot(snap_dir, verify=False)
    assert snap.vix.index.equals(snap.spy.index)  # SPY calendar canonical, never shrunk
    assert snap.vix["close"].isna().sum() == 2  # 4-gap minus ffill(limit=2)


def test_e4_duplicate_dates_rejected(snap_dir):
    df = pd.read_csv(snap_dir / "raw" / "spy_ohlcv.csv")
    df = pd.concat([df, df.iloc[[20]]], ignore_index=True)
    df.to_csv(snap_dir / "raw" / "spy_ohlcv.csv", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_snapshot(snap_dir, verify=False)


def test_e5_zero_tv_target_surfaced_as_warning():
    idx = pd.bdate_range("2021-01-04", periods=3)
    # day 2: flat day AND zero gap (open == prev close) -> rv_tv exactly 0
    spy = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0],
            "high": [101.0, 100.0, 102.0],
            "low": [99.0, 100.0, 100.5],
            "close": [100.0, 100.0, 101.5],
            "volume": [1e6] * 3,
        },
        index=idx,
    )
    snap = Snapshot(spy=spy, vix=spy.drop(columns="volume"), manifest={})
    with pytest.warns(UserWarning, match="zero"):
        t = build_targets(snap)
    assert t.loc[idx[1], "rv_tv"] == 0.0


def test_e6_missing_raw_file_is_atomic_failure(snap_dir):
    (snap_dir / "raw" / "vix_ohlc.csv").unlink()
    with pytest.raises(FileNotFoundError, match="atomic"):
        load_snapshot(snap_dir, verify=True)
    # manifest itself missing
    (snap_dir / "snapshot_manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_snapshot(snap_dir, verify=True)


# ------------------------------------------------- real frozen snapshot (post-freeze)


@needs_snapshot
def test_real_s1_checksums_and_end_date():
    snap = load_snapshot(verify=True)
    assert snap.manifest["adjustment_basis"] == "adjusted"


@needs_snapshot
def test_real_s2_s4_targets_and_consistency():
    snap = load_snapshot(verify=True)
    t = build_targets(snap)
    assert not t.isna().any().any()
    assert (t >= 0).all().all()


@needs_snapshot
def test_real_s5_calibration_in_band():
    from src import io as io_mod

    cfg = io_mod.load_config()
    snap = load_snapshot(verify=True)
    t = build_targets(snap)
    d = calibration_diagnostics(
        snap, t, cfg["splits"]["val_start"], band=tuple(cfg["calibration_band"])
    )
    assert d["pass"], f"S8 calibration ratios out of band: {d}"

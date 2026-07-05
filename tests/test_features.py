"""Executable success criteria S1-S6 from specs/features/SPEC.md."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import build_targets
from src.features import (
    FeatureSet,
    TrainOnlyScaler,
    build_features,
    leakage_probe,
    make_labels,
)
from conftest import make_snapshot


@pytest.fixture(scope="module")
def snap():
    return make_snapshot(pd.bdate_range("2019-01-02", "2021-12-31"), seed=21)


@pytest.fixture(scope="module")
def targets(snap):
    return build_targets(snap)


# ------------------------------------------------- S1: future-perturbation leakage


@pytest.mark.parametrize(
    "target,horizon", [("rv_tv", 1), ("rv_tv", 5), ("rv_oc", 1), ("rv_oc", 5)]
)
def test_s1_leakage_probe_all_tiers(snap, target, horizon):
    idx = snap.spy.index
    probes = [idx[len(idx) // 4], idx[len(idx) // 2], idx[-40]]  # early, middle, late
    for i, t in enumerate(probes):
        rng = np.random.default_rng(100 + i)
        assert leakage_probe(
            snap, t, rng, target=target, horizon=horizon, tiers=("t1", "t2", "t3")
        ), f"leak detected at probe {t}"


def test_s1_probe_detects_a_planted_leak(snap):
    """The probe must actually fail on a leaky builder (centered window ~ future info)."""
    import src.features as feat

    original = feat._tier2

    def leaky_tier2(targets, snap, target):
        cols = original(targets, snap, target)
        c = snap.spy["close"]
        # centered rolling mean uses future closes -> leak by construction
        cols["t2_leaky"] = (
            np.log(c).rolling(5, center=True).mean().reindex(targets.index)
        )
        return cols

    feat._TIERS["t2"] = leaky_tier2
    try:
        t = snap.spy.index[len(snap.spy) // 2]
        clean = leakage_probe(
            snap,
            t,
            np.random.default_rng(0),
            target="rv_tv",
            horizon=1,
            tiers=("t1", "t2"),
        )
    finally:
        feat._TIERS["t2"] = original
    assert clean is False


# ------------------------------------------------- S2: train-only scaler


def test_s2_scaler_fit_indices_and_forward_application(targets, snap):
    fs = build_features(targets, snap, target="rv_tv", horizon=1, tiers=("t1",))
    train_idx = fs.X.index[:300]
    scaler = TrainOnlyScaler().fit(fs.X, train_idx)
    assert scaler.fit_indices.isin(train_idx).all()

    Z = scaler.transform(fs.X)
    train_mean = Z.loc[scaler.fit_indices].mean()
    assert np.allclose(train_mean, 0.0, atol=1e-12)
    assert np.allclose(Z.loc[scaler.fit_indices].std(ddof=0), 1.0, atol=1e-12)
    # later data transformed with train statistics is not centered in general
    assert (Z.loc[fs.X.index[400:]].mean().abs() > 1e-6).any()


def test_s2_scaler_refuses_before_fit(targets, snap):
    fs = build_features(targets, snap, target="rv_tv", horizon=1)
    with pytest.raises(ValueError, match="before fit"):
        TrainOnlyScaler().transform(fs.X)


# ------------------------------------------------- S3: all features finite


def test_s3_all_columns_finite_synthetic(targets, snap):
    fs = build_features(
        targets, snap, target="rv_tv", horizon=5, tiers=("t1", "t2", "t3")
    )
    assert np.isfinite(fs.X.to_numpy()).all()
    assert fs.X.index.equals(fs.y.index) and fs.X.index.equals(fs.y_log.index)
    # tier-prefixed, registered column names (R6/R7)
    for tier, cols in fs.tiers.items():
        assert all(c.startswith(f"{tier}_") for c in cols)
    assert sorted(sum(fs.tiers.values(), [])) == sorted(fs.X.columns)


# ------------------------------------------------- S4: label alignment fixture


def test_s4_label_alignment_toy_series():
    idx = pd.bdate_range("2021-01-04", periods=10)
    targets = pd.DataFrame(
        {"rv_tv": np.arange(1.0, 11.0), "rv_oc": np.arange(1.0, 11.0)}, index=idx
    )
    y5, y5_log = make_labels(targets, "rv_tv", 5)
    # origin t=0 (value 1): mean of values 2..6
    assert y5.iloc[0] == pytest.approx(np.mean([2, 3, 4, 5, 6]))
    assert y5.iloc[4] == pytest.approx(np.mean([6, 7, 8, 9, 10]))
    assert (
        y5.iloc[5:].isna().all() and y5.iloc[:5].notna().all()
    )  # last 5 origins dropped
    assert np.allclose(y5_log.dropna(), np.log(y5.dropna()))

    y1, _ = make_labels(targets, "rv_tv", 1)
    assert y1.iloc[0] == 2.0 and pd.isna(y1.iloc[-1])


def test_s4_build_drops_partial_label_rows(targets, snap):
    fs1 = build_features(targets, snap, target="rv_tv", horizon=1)
    fs5 = build_features(targets, snap, target="rv_tv", horizon=5)
    assert fs1.X.index[-1] == targets.index[-2]  # h=1: last origin dropped
    assert fs5.X.index[-1] == targets.index[-6]  # h=5: last five dropped
    assert fs5.y.loc[fs5.X.index[0]] == pytest.approx(
        targets["rv_tv"].iloc[targets.index.get_loc(fs5.X.index[0]) + 1 :][:5].mean()
    )


# ------------------------------------------------- S5: HAR aggregates hand-computed


def test_s5_har_aggregates_match_hand_computation(targets, snap):
    fs = build_features(targets, snap, target="rv_tv", horizon=1, tiers=("t1",))
    t = fs.X.index[50]
    s = targets["rv_tv"]
    pos = targets.index.get_loc(t)
    assert fs.X.loc[t, "t1_log_rv_w"] == pytest.approx(
        np.log(s.iloc[pos - 4 : pos + 1].mean()), abs=1e-12
    )
    assert fs.X.loc[t, "t1_log_rv_m"] == pytest.approx(
        np.log(s.iloc[pos - 21 : pos + 1].mean()), abs=1e-12
    )
    assert fs.X.loc[t, "t1_log_rv_lag0"] == pytest.approx(np.log(s.loc[t]), abs=1e-12)
    assert fs.X.loc[t, "t1_log_rv_lag3"] == pytest.approx(
        np.log(s.iloc[pos - 3]), abs=1e-12
    )


# ------------------------------------------------- S6: edge cases


def test_e1_unknown_tier_and_missing_column(targets, snap):
    with pytest.raises(ValueError, match="unknown tier"):
        build_features(targets, snap, target="rv_tv", horizon=1, tiers=("t9",))
    from dataclasses import replace

    snap_novix = replace(snap, vix=snap.vix * np.nan)
    with pytest.raises(ValueError, match="t3"):
        build_features(
            targets, snap_novix, target="rv_tv", horizon=1, tiers=("t1", "t3")
        )


def test_e2_frozen_horizons_and_targets(targets, snap):
    with pytest.raises(ValueError, match="horizon"):
        build_features(targets, snap, target="rv_tv", horizon=3)
    with pytest.raises(ValueError, match="target"):
        build_features(targets, snap, target="rv_hf", horizon=1)


def test_e3_zero_target_is_a_hard_error(snap):
    idx = pd.bdate_range("2021-01-04", periods=60)
    vals = np.full(60, 1e-4)
    vals[30] = 0.0
    targets = pd.DataFrame({"rv_tv": vals, "rv_oc": vals}, index=idx)
    with pytest.raises(ValueError, match="non-positive label"):
        make_labels(targets, "rv_tv", 1)


def test_e4_zero_variance_feature_errors_in_scaler():
    idx = pd.bdate_range("2021-01-04", periods=50)
    X = pd.DataFrame(
        {"a": np.random.default_rng(0).normal(size=50), "b": np.ones(50)}, index=idx
    )
    with pytest.raises(ValueError, match="zero-variance"):
        TrainOnlyScaler().fit(X, idx[:30])


def test_e5_feature_set_is_internally_aligned(targets, snap):
    fs = build_features(
        targets, snap, target="rv_oc", horizon=5, tiers=("t1", "t2", "t3")
    )
    assert isinstance(fs, FeatureSet)
    assert fs.X.index.equals(fs.y.index)
    assert not fs.y.isna().any() and not fs.y_log.isna().any()
    assert np.allclose(np.exp(fs.y_log), fs.y)  # log round-trip

"""Leak-safe feature matrix (target history + tiered aux), labels, scaler, leakage probe.

Spec: specs/features/SPEC.md. Every feature value at day t uses only information known
at the close of day t — enforced by construction (shift/rolling over past values only)
and verified by the future-perturbation leakage probe.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd

from src.data import Snapshot, build_targets

VALID_TARGETS = ("rv_tv", "rv_oc")
VALID_HORIZONS = (1, 5)


@dataclass(frozen=True)
class FeatureSet:
    X: pd.DataFrame
    y: pd.Series          # horizon-h label, level (variance) space
    y_log: pd.Series      # the modeled quantity
    tiers: dict           # tier name -> list of column names


def make_labels(targets: pd.DataFrame, target: str, horizon: int) -> tuple[pd.Series, pd.Series]:
    """Horizon-h labels: y_t = target_{t+1} (h=1) or mean(target_{t+1..t+5}) (h=5)."""
    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of {VALID_TARGETS}, got {target!r}")
    if horizon not in VALID_HORIZONS:
        raise ValueError(f"horizon must be one of {VALID_HORIZONS} (frozen, manifesto s8), "
                         f"got {horizon!r}")
    s = targets[target]
    if horizon == 1:
        y = s.shift(-1)
    else:
        y = s.shift(-1).rolling(5).mean().shift(-4)
    y.name = f"y_{target}_h{horizon}"
    nonpos = y.index[(y <= 0) & y.notna()]
    if len(nonpos):
        raise ValueError(f"non-positive label (log-target undefined) on {list(nonpos[:10])}")
    return y, np.log(y)


# --- tier builders: each returns a dict of {column_name: Series} -----------------

def _tier1(targets: pd.DataFrame, snap: Snapshot, target: str) -> dict[str, pd.Series]:
    log_s = np.log(targets[target])
    s = targets[target]
    cols = {f"t1_log_rv_lag{k}": log_s.shift(k) for k in range(22)}
    cols["t1_log_rv_w"] = np.log(s.rolling(5).mean())
    cols["t1_log_rv_m"] = np.log(s.rolling(22).mean())
    return cols


def _tier2(targets: pd.DataFrame, snap: Snapshot, target: str) -> dict[str, pd.Series]:
    spy = snap.spy
    c, o, v = spy["close"], spy["open"], spy["volume"]
    ret_cc = np.log(c / c.shift(1))
    cols = {
        "t2_ret_cc": ret_cc,
        "t2_abs_ret": ret_cc.abs(),
        "t2_ret_overnight": np.log(o / c.shift(1)),
        "t2_vol_ratio": np.log(v / v.rolling(22).mean()),
    }
    return {k: s.reindex(targets.index) for k, s in cols.items()}


def _tier3(targets: pd.DataFrame, snap: Snapshot, target: str) -> dict[str, pd.Series]:
    vc = snap.vix["close"]
    cols = {
        "t3_log_vix": np.log(vc),
        "t3_vix_chg": np.log(vc / vc.shift(1)),
        "t3_impl_var": (vc / 100.0) ** 2 / 252.0,
    }
    return {k: s.reindex(targets.index) for k, s in cols.items()}


_TIERS = {"t1": _tier1, "t2": _tier2, "t3": _tier3}


def build_features(targets: pd.DataFrame, snap: Snapshot, *, target: str, horizon: int,
                   tiers: Sequence[str] = ("t1",)) -> FeatureSet:
    y, y_log = make_labels(targets, target, horizon)

    tier_cols: dict[str, list[str]] = {}
    all_cols: dict[str, pd.Series] = {}
    for tier in tiers:
        if tier not in _TIERS:
            raise ValueError(f"unknown tier {tier!r}; valid: {sorted(_TIERS)}")
        cols = _TIERS[tier](targets, snap, target)
        for name, series in cols.items():
            if series.isna().all():
                raise ValueError(f"tier {tier} column {name} has no data in the snapshot")
        tier_cols[tier] = list(cols)
        all_cols.update(cols)

    X = pd.DataFrame(all_cols, index=targets.index)
    keep = X.notna().all(axis=1) & y.notna()
    X = X.loc[keep]
    if not np.isfinite(X.to_numpy()).all():
        bad = X.columns[~np.isfinite(X).all(axis=0)].tolist()
        raise ValueError(f"non-finite feature values in columns {bad}")
    return FeatureSet(X=X, y=y.loc[X.index], y_log=y_log.loc[X.index], tiers=tier_cols)


class TrainOnlyScaler:
    """Standardization fit on training rows only, applied forward. Records fit indices."""

    def __init__(self) -> None:
        self.means: pd.Series | None = None
        self.stds: pd.Series | None = None
        self.fit_indices: pd.DatetimeIndex | None = None

    def fit(self, X: pd.DataFrame, train_idx: pd.DatetimeIndex) -> "TrainOnlyScaler":
        idx = X.index.intersection(train_idx)
        if len(idx) == 0:
            raise ValueError("no overlap between X and train indices")
        sub = X.loc[idx]
        stds = sub.std(ddof=0)
        zero = stds.index[stds == 0].tolist()
        if zero:
            raise ValueError(f"zero-variance feature(s) on the fit span: {zero} — fix or drop "
                             "explicitly at the experiment layer")
        self.means, self.stds, self.fit_indices = sub.mean(), stds, idx
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.means is None:
            raise ValueError("scaler used before fit")
        return (X - self.means) / self.stds


def _perturb_after(snap: Snapshot, t: pd.Timestamp, rng: np.random.Generator) -> Snapshot:
    spy = snap.spy.copy()
    vix = snap.vix.copy()
    after_spy = spy.index > t
    after_vix = vix.index > t
    f_spy = np.exp(rng.normal(0.0, 0.02, after_spy.sum()))
    f_vix = np.exp(rng.normal(0.0, 0.02, after_vix.sum()))
    for col in ("open", "high", "low", "close"):
        spy.loc[after_spy, col] = spy.loc[after_spy, col] * f_spy
        vix.loc[after_vix, col] = vix.loc[after_vix, col] * f_vix
    spy.loc[after_spy, "volume"] = spy.loc[after_spy, "volume"] * np.exp(
        rng.normal(0.0, 0.1, after_spy.sum()))
    return replace(snap, spy=spy, vix=vix)


def leakage_probe(snap: Snapshot, t: pd.Timestamp, rng: np.random.Generator, *,
                  target: str, horizon: int, tiers: Sequence[str] = ("t1", "t2", "t3")) -> bool:
    """Perturb ALL raw inputs strictly after day t; features at <= t must be bitwise identical.

    Also checks labels at origins whose full h-day window ends <= t. Returns True iff clean.
    """
    base = build_features(build_targets(snap), snap, target=target, horizon=horizon, tiers=tiers)
    snap_p = _perturb_after(snap, t, rng)
    pert = build_features(build_targets(snap_p), snap_p, target=target, horizon=horizon,
                          tiers=tiers)

    try:
        pd.testing.assert_frame_equal(base.X.loc[:t], pert.X.loc[:t], check_exact=True)
    except AssertionError:
        return False

    # Origins whose full h-day label window ends <= t (y.index is a contiguous slice of
    # the trading calendar, so position arithmetic gives the window end).
    pos_t = int(base.y.index.searchsorted(t, side="right"))
    label_safe = base.y.index[: max(pos_t - horizon, 0)]
    try:
        pd.testing.assert_series_equal(base.y.loc[label_safe],
                                       pert.y.reindex(label_safe), check_exact=True)
    except AssertionError:
        return False
    return True

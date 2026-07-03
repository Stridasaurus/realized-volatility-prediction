"""The canonical walk-forward splitter — single source of truth for every temporal split.

Spec: specs/splits/SPEC.md. Operates on a trading-day date index alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    train_start: str
    val_start: str
    test_start: str
    test_end: str
    embargo_days: int


@dataclass(frozen=True)
class CanonicalSplit:
    train_idx: pd.DatetimeIndex
    val_idx: pd.DatetimeIndex
    test_idx: pd.DatetimeIndex


@dataclass(frozen=True)
class RetrainFold:
    fit_idx: pd.DatetimeIndex
    test_idx: pd.DatetimeIndex


def _validate_index(index: pd.DatetimeIndex) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("index must be a pandas.DatetimeIndex")
    if index.tz is not None:
        raise ValueError("index must be tz-naive")
    if len(index) == 0:
        raise ValueError("index is empty")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be strictly increasing (found non-monotonic dates)")
    if index.has_duplicates:
        raise ValueError("index must not contain duplicate dates")


def _validate_config(cfg: SplitConfig) -> None:
    b = [pd.Timestamp(cfg.train_start), pd.Timestamp(cfg.val_start),
         pd.Timestamp(cfg.test_start), pd.Timestamp(cfg.test_end)]
    if not (b[0] < b[1] < b[2] < b[3]):
        raise ValueError(f"split boundaries must be strictly ordered, got {b}")
    if not isinstance(cfg.embargo_days, int) or cfg.embargo_days <= 0:
        raise ValueError(f"embargo_days must be a positive int, got {cfg.embargo_days!r}")


def _pos(index: pd.DatetimeIndex, date: str, side: str = "left") -> int:
    return int(index.searchsorted(pd.Timestamp(date), side=side))


def canonical_split(index: pd.DatetimeIndex, cfg: SplitConfig) -> CanonicalSplit:
    """Frozen three-way chronological split with the embargo trimmed off the earlier segment."""
    _validate_index(index)
    _validate_config(cfg)
    if pd.Timestamp(cfg.test_end) > index[-1]:
        raise ValueError(
            f"test_end {cfg.test_end} lies beyond the last index date {index[-1].date()}; "
            "the pinned snapshot end date and the calendar must agree"
        )
    p_train = _pos(index, cfg.train_start)
    p_val = _pos(index, cfg.val_start)
    p_test = _pos(index, cfg.test_start)
    p_end = _pos(index, cfg.test_end, side="right")

    train = index[p_train : p_val - cfg.embargo_days]
    val = index[p_val : p_test - cfg.embargo_days]
    test = index[p_test:p_end]
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            "a split segment is empty after embargo trimming "
            f"(train={len(train)}, val={len(val)}, test={len(test)})"
        )
    return CanonicalSplit(train, val, test)


def _month_groups(test_idx: pd.DatetimeIndex) -> list[pd.DatetimeIndex]:
    periods = test_idx.to_period("M")
    return [test_idx[periods == p] for p in periods.unique()]


def retrain_folds(index: pd.DatetimeIndex, cfg: SplitConfig,
                  window_len: int | None = None) -> list[RetrainFold]:
    """Monthly walk-forward retrain folds inside the test region.

    Each fold's fit window is exactly ``window_len`` trading days (default: the initial
    train span length — a frozen control) ending ``embargo_days`` before the fold's first
    test date. Fit windows slide forward through time and may legitimately cross the val
    region; the embargo before each fold's own test month is the binding rule (SPEC R8).
    """
    split = canonical_split(index, cfg)
    if window_len is None:
        window_len = len(split.train_idx)
    if window_len <= 0:
        raise ValueError(f"window_len must be positive, got {window_len}")

    folds: list[RetrainFold] = []
    for month in _month_groups(split.test_idx):
        if len(month) == 0:  # pragma: no cover - unique() guarantees non-empty
            continue
        end = _pos(index, month[0]) - cfg.embargo_days
        start = end - window_len
        if start < 0:
            raise ValueError(
                f"index too short for fixed window: fold starting {month[0].date()} needs "
                f"{window_len} fit days + {cfg.embargo_days} embargo days before it"
            )
        folds.append(RetrainFold(fit_idx=index[start:end], test_idx=month))
    return folds

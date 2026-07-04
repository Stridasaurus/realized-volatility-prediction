"""Executable success criteria S1-S7 from specs/splits/SPEC.md."""

from __future__ import annotations

import pandas as pd
import pytest

from src.splits import SplitConfig, canonical_split, retrain_folds

CFG = SplitConfig(
    train_start="2015-01-01",
    val_start="2018-01-01",
    test_start="2019-01-01",
    test_end="2020-12-31",
    embargo_days=22,
)


@pytest.fixture
def index() -> pd.DatetimeIndex:
    return pd.bdate_range("2015-01-02", "2020-12-31")


# ------------------------------------------------------------------ S1: no overlap


def test_s1_train_val_test_disjoint(index):
    s = canonical_split(index, CFG)
    assert len(s.train_idx.intersection(s.val_idx)) == 0
    assert len(s.val_idx.intersection(s.test_idx)) == 0
    assert len(s.train_idx.intersection(s.test_idx)) == 0


def test_s1_fit_windows_never_overlap_their_test_month(index):
    for fold in retrain_folds(index, CFG, window_len=250):
        assert len(fold.fit_idx.intersection(fold.test_idx)) == 0


# ------------------------------------------------------------------ S2: embargo gap


def test_s2_embargo_gap_at_both_boundaries(index):
    s = canonical_split(index, CFG)
    gap_train_val = index.get_loc(s.val_idx[0]) - index.get_loc(s.train_idx[-1]) - 1
    gap_val_test = index.get_loc(s.test_idx[0]) - index.get_loc(s.val_idx[-1]) - 1
    assert gap_train_val >= CFG.embargo_days
    assert gap_val_test >= CFG.embargo_days


def test_s2_embargo_taken_from_earlier_segment(index):
    """Test region must never shrink: its first date is the first trading day >= test_start."""
    s = canonical_split(index, CFG)
    expected_first = index[index.searchsorted(pd.Timestamp(CFG.test_start))]
    assert s.test_idx[0] == expected_first


def test_s2_fit_window_ends_embargo_before_test_month(index):
    for fold in retrain_folds(index, CFG, window_len=250):
        gap = index.get_loc(fold.test_idx[0]) - index.get_loc(fold.fit_idx[-1]) - 1
        assert gap >= CFG.embargo_days


# ------------------------------------------------------------------ S3: chronological


def test_s3_every_fold_chronological(index):
    for fold in retrain_folds(index, CFG, window_len=250):
        assert fold.fit_idx.max() < fold.test_idx.min()
        assert fold.fit_idx.is_monotonic_increasing
        assert fold.test_idx.is_monotonic_increasing


def test_s3_canonical_ordering(index):
    s = canonical_split(index, CFG)
    assert s.train_idx.max() < s.val_idx.min() < s.val_idx.max() < s.test_idx.min()


# ------------------------------------------------------------------ S4: fixed window


def test_s4_every_fit_window_exactly_window_len(index):
    folds = retrain_folds(index, CFG, window_len=250)
    assert all(len(f.fit_idx) == 250 for f in folds)


def test_s4_default_window_is_initial_train_span(index):
    s = canonical_split(index, CFG)
    folds = retrain_folds(index, CFG)
    assert all(len(f.fit_idx) == len(s.train_idx) for f in folds)


# ------------------------------------------------------------------ S5: determinism


def test_s5_deterministic(index):
    a, b = canonical_split(index, CFG), canonical_split(index, CFG)
    assert a.train_idx.equals(b.train_idx)
    assert a.val_idx.equals(b.val_idx)
    assert a.test_idx.equals(b.test_idx)
    fa, fb = retrain_folds(index, CFG, 250), retrain_folds(index, CFG, 250)
    assert len(fa) == len(fb)
    for x, y in zip(fa, fb):
        assert x.fit_idx.equals(y.fit_idx) and x.test_idx.equals(y.test_idx)


# ------------------------------------------------------------------ S6: fold union


def test_s6_fold_test_months_partition_test_region(index):
    s = canonical_split(index, CFG)
    folds = retrain_folds(index, CFG, window_len=250)
    union = folds[0].test_idx
    for f in folds[1:]:
        union = union.append(f.test_idx)
    assert not union.has_duplicates
    assert union.sort_values().equals(s.test_idx)
    # one fold per calendar month, in order
    months = [f.test_idx[0].to_period("M") for f in folds]
    assert months == sorted(months)
    assert len(set(months)) == len(months)


def test_s6_partial_final_month(index):
    cfg = SplitConfig(**{**CFG.__dict__, "test_end": "2020-11-13"})
    folds = retrain_folds(index, cfg, window_len=250)
    s = canonical_split(index, cfg)
    assert folds[-1].test_idx[-1] == s.test_idx[-1] == pd.Timestamp("2020-11-13")
    assert all(len(f.test_idx) > 0 for f in folds)


# ------------------------------------------------------------------ S7: edge cases


def test_e1_index_too_short_for_fixed_window(index):
    with pytest.raises(ValueError, match="fixed window|too short"):
        retrain_folds(index, CFG, window_len=10_000)


def test_e2_boundary_on_non_trading_day_resolves_forward(index):
    # 2018-01-01 is a holiday; first trading day >= it is 2018-01-01's next bday
    s = canonical_split(index, CFG)
    assert s.val_idx[0] == index[index.searchsorted(pd.Timestamp("2018-01-01"))]
    assert s.val_idx[0] >= pd.Timestamp("2018-01-01")


def test_e3_embargo_swallows_a_segment(index):
    cfg = SplitConfig(**{**CFG.__dict__, "embargo_days": 600})  # > val span (~260 days)
    with pytest.raises(ValueError, match="empty"):
        canonical_split(index, cfg)


def test_e3_embargo_larger_than_whole_train_segment():
    """Regression: a negative slice bound must not wrap around silently."""
    short = pd.bdate_range("2019-06-03", "2020-12-31")
    cfg = SplitConfig(
        train_start="2019-06-01",
        val_start="2019-08-01",
        test_start="2020-01-01",
        test_end="2020-12-31",
        embargo_days=80,
    )
    with pytest.raises(ValueError, match="empty"):
        canonical_split(short, cfg)


def test_e5_duplicate_dates_rejected(index):
    dup = index[:100].append(index[99:200])
    with pytest.raises(ValueError, match="duplicate"):
        canonical_split(dup, CFG)


def test_e5_non_monotonic_rejected(index):
    scrambled = index[:200].append(index[300:400]).append(index[200:300])
    with pytest.raises(ValueError, match="increasing|monotonic"):
        canonical_split(scrambled, CFG)


def test_e6_test_end_beyond_last_index_date(index):
    cfg = SplitConfig(**{**CFG.__dict__, "test_end": "2030-01-01"})
    with pytest.raises(ValueError, match="beyond"):
        canonical_split(index, cfg)


def test_validation_rejects_bad_config(index):
    with pytest.raises(ValueError, match="ordered"):
        canonical_split(
            index,
            SplitConfig("2019-01-01", "2018-01-01", "2020-01-01", "2020-12-31", 22),
        )
    with pytest.raises(ValueError, match="embargo"):
        canonical_split(
            index,
            SplitConfig("2015-01-01", "2018-01-01", "2019-01-01", "2020-12-31", 0),
        )
    with pytest.raises(ValueError, match="tz-naive"):
        canonical_split(index.tz_localize("UTC"), CFG)

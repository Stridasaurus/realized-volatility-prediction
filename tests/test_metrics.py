"""Executable success criteria S1-S6 from specs/metrics/SPEC.md."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.metrics import dm_test, qlike, qlike_series, rmse

FLOOR = 0.5


# ------------------------------------------------------- S1: hand-computed fixtures


def test_s1_qlike_matches_hand_computation():
    pred = np.array([1.0, 2.0, 4.0])
    actual = np.array([2.0, 2.0, 2.0])
    # per-element: r - ln r - 1 with r = actual/pred (no flooring: all preds > 0.5)
    expected = np.mean(
        [
            2.0 - math.log(2.0) - 1.0,
            1.0 - math.log(1.0) - 1.0,
            0.5 - math.log(0.5) - 1.0,
        ]
    )
    res = qlike(pred, actual, floor=FLOOR)
    assert res.value == pytest.approx(expected, abs=1e-15)
    assert res.bind_rate == 0.0
    assert res.n == 3


def test_s1_perfect_forecast_scores_zero():
    a = np.array([1.0, 2.0, 3.0])
    assert qlike(a, a, floor=1e-6).value == pytest.approx(0.0, abs=1e-15)


def test_s1_rmse_matches_hand_computation():
    pred = np.array([1.0, 2.0, 4.0])
    actual = np.array([2.0, 2.0, 2.0])
    assert rmse(pred, actual) == pytest.approx(math.sqrt((1 + 0 + 4) / 3), abs=1e-15)


# ------------------------------------------------------- S2: floor + bind rate


def test_s2_flooring_and_exact_bind_rate():
    pred = np.array([-1.0, 0.0, 0.25, 0.75])  # three below floor=0.5
    actual = np.array([1.0, 1.0, 1.0, 1.0])
    res = qlike(pred, actual, floor=FLOOR)
    assert res.bind_rate == pytest.approx(3 / 4)
    # floored predictions score as if pred == floor
    r = np.array([2.0, 2.0, 2.0, 1.0 / 0.75])
    assert res.value == pytest.approx(np.mean(r - np.log(r) - 1.0), abs=1e-12)
    assert np.isfinite(res.value)


# ------------------------------------------------------- S3: DM reference agreement


def _dm_by_hand(d: np.ndarray, h: int, lag: int) -> float:
    """Independent HLN-corrected DM implementation (explicit loops)."""
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    s = dc @ dc / n
    for k in range(1, lag + 1):
        gamma = float(dc[k:] @ dc[:-k]) / n
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    dm = dbar / math.sqrt(s / n)
    hln = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return dm * hln


@pytest.fixture
def loss_pair():
    rng = np.random.default_rng(42)
    n = 120
    lb = np.abs(rng.normal(1.0, 0.3, n)) + 0.1
    lm = lb + rng.normal(-0.05, 0.2, n)  # model slightly better, noisy
    return lm, lb


def test_s3_dm_matches_hand_computed_hln_stat(loss_pair):
    lm, lb = loss_pair
    for h in (1, 5):
        res = dm_test(lm, lb, h=h)
        assert res.stat == pytest.approx(
            _dm_by_hand(lm - lb, h, res.hac_lag), abs=1e-12
        )
        assert res.mean_loss_diff == pytest.approx((lm - lb).mean(), abs=1e-15)


def test_s3_hac_variance_matches_statsmodels(loss_pair):
    import statsmodels.api as sm

    lm, lb = loss_pair
    d = lm - lb
    n = len(d)
    res = dm_test(lm, lb, h=5)
    ols = sm.OLS(d, np.ones(n)).fit(
        cov_type="HAC", cov_kwds={"maxlags": res.hac_lag, "use_correction": False}
    )
    var_mean_sm = float(np.asarray(ols.cov_params())[0, 0])
    hln = math.sqrt((n + 1 - 2 * 5 + 5 * 4 / n) / n)
    stat_sm = d.mean() / math.sqrt(var_mean_sm) * hln
    assert res.stat == pytest.approx(stat_sm, abs=1e-8)


def test_s3_p_value_is_two_sided_student_t(loss_pair):
    from scipy import stats

    lm, lb = loss_pair
    res = dm_test(lm, lb, h=1)
    expected = 2.0 * (1.0 - stats.t.cdf(abs(res.stat), df=res.n - 1))
    assert res.p_value == pytest.approx(expected, abs=1e-15)
    assert 0.0 <= res.p_value <= 1.0


# ------------------------------------------------------- S4: HAC lag responds to h


def test_s4_hac_lag_floor_tracks_horizon():
    rng = np.random.default_rng(0)
    n = 50  # NW auto lag = floor(4*(0.5)**(2/9)) = 3 < 4
    lm, lb = rng.normal(1.0, 0.1, n) + 0.5, rng.normal(1.0, 0.1, n) + 0.5
    r1, r5 = dm_test(lm, lb, h=1), dm_test(lm, lb, h=5)
    assert r5.hac_lag >= 4 > r1.hac_lag


def test_s4_explicit_lag_respected():
    rng = np.random.default_rng(1)
    lm, lb = rng.normal(1.0, 0.1, 80), rng.normal(1.0, 0.1, 80)
    assert dm_test(lm, lb, h=1, hac_lag=12).hac_lag == 12


# ------------------------------------------------------- S5: variance-space guard


def test_s5_guard_fires_on_vol_vs_variance():
    actual = np.full(50, 1e-4)  # daily variance scale
    pred_vol = np.full(50, 0.5)  # volatility-like units, ratio 5e3 > 1e3
    with pytest.warns(UserWarning, match="variance"):
        qlike(pred_vol, actual, floor=1e-8)
    with pytest.warns(UserWarning, match="variance"):
        rmse(pred_vol, actual)


def test_s5_guard_silent_on_matched_units():
    import warnings

    rng = np.random.default_rng(3)
    a = np.abs(rng.normal(1e-4, 2e-5, 50)) + 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        qlike(a * 1.1, a, floor=1e-8)
        rmse(a * 1.1, a)


# ------------------------------------------------------- S6: edge cases E1-E6


def test_e1_bind_rate_exactly_zero_when_all_above_floor():
    a = np.array([1.0, 2.0, 3.0])
    assert qlike(a + 0.1, a, floor=0.5).bind_rate == 0.0


def test_e2_constant_loss_differential_rejected():
    lb = np.abs(np.random.default_rng(4).normal(1.0, 0.2, 40)) + 0.1
    with pytest.raises(ValueError, match="degenerate"):
        dm_test(lb + 0.5, lb, h=1)


def test_e3_n_leq_h_rejected():
    with pytest.raises(ValueError, match="HLN|n="):
        dm_test(np.ones(5) + np.arange(5), np.ones(5), h=5)


def test_e4_hac_lag_below_floor_rejected():
    rng = np.random.default_rng(5)
    lm, lb = rng.normal(1, 0.1, 60), rng.normal(1, 0.1, 60)
    with pytest.raises(ValueError, match="hac_lag"):
        dm_test(lm, lb, h=5, hac_lag=2)


def test_e5_negative_predictions_floored_not_raised():
    with pytest.warns(UserWarning, match="median prediction is non-positive"):
        res = qlike(np.array([-5.0, 1.0]), np.array([1.0, 1.0]), floor=0.5)
    assert res.bind_rate == pytest.approx(0.5)
    assert np.isfinite(res.value)


def test_e6_nonpositive_floor_rejected():
    a = np.array([1.0, 2.0])
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="floor"):
            qlike(a, a, floor=bad)


def test_r5_input_validation():
    a = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="length"):
        qlike(a[:2], a, floor=0.5)
    with pytest.raises(ValueError, match="empty"):
        qlike(np.array([]), np.array([]), floor=0.5)
    with pytest.raises(ValueError, match="NaN"):
        qlike(np.array([1.0, np.nan, 1.0]), a, floor=0.5)
    with pytest.raises(ValueError, match="non-positive"):
        qlike(a, np.array([1.0, 0.0, 2.0]), floor=0.5)
    with pytest.raises(ValueError, match="NaN"):
        dm_test(np.array([1.0, np.nan]), np.array([1.0, 1.0]), h=1)


def test_r7_deterministic(loss_pair):
    lm, lb = loss_pair
    assert dm_test(lm, lb, h=5) == dm_test(lm, lb, h=5)
    assert qlike_series(lm, lb, 0.01).tolist() == qlike_series(lm, lb, 0.01).tolist()

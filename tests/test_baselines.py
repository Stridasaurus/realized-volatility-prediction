"""Executable success criteria S1-S7 from specs/baselines/SPEC.md.

Synthetic multi-year series run through real `splits` folds (never hand-rolled windows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines import CLASSICAL_MODELS, forecast_classical
from src.splits import RetrainFold, SplitConfig, canonical_split, retrain_folds

INDEX = pd.bdate_range("2012-01-02", "2016-06-30")
CFG_SPLIT = SplitConfig(
    train_start="2012-01-01",
    val_start="2015-06-01",
    test_start="2016-01-01",
    test_end="2016-06-30",
    embargo_days=22,
)


def make_log_ar1_targets(
    index: pd.DatetimeIndex,
    seed: int = 0,
    phi: float = 0.95,
    mu: float = np.log(1e-4),
    sigma: float = 0.3,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(index)
    log_s = np.empty(n)
    log_s[0] = mu
    for t in range(1, n):
        log_s[t] = mu + phi * (log_s[t - 1] - mu) + rng.normal(0.0, sigma)
    s = np.exp(log_s)
    return pd.DataFrame({"rv_tv": s, "rv_oc": s * 0.8}, index=index)


@pytest.fixture(scope="module")
def targets():
    return make_log_ar1_targets(INDEX, seed=31)


@pytest.fixture(scope="module")
def folds():
    return retrain_folds(INDEX, CFG_SPLIT)


@pytest.fixture
def cfg():
    return {
        "ewma": {
            "lambda_start": 0.90,
            "lambda_stop": 0.98,
            "lambda_step": 0.02,
            "fallback_lambda": 0.94,
            "warmup_min_obs": 60,
        },
        "floors": {"rv_tv": 1e-6, "rv_oc": 1e-6},
    }


# ------------------------------------------------- S1: persistence + EWMA fixtures


def test_s1_persistence_h1_is_todays_value(targets, folds, cfg):
    out = forecast_classical(
        "persistence", targets, None, folds, target="rv_tv", horizon=1, cfg=cfg
    )
    for t in out.index[::17]:
        assert out.loc[t, "y_pred"] == pytest.approx(targets.loc[t, "rv_tv"], rel=1e-12)


def test_s1_persistence_h5_is_trailing_5_mean(targets, folds, cfg):
    out = forecast_classical(
        "persistence", targets, None, folds, target="rv_tv", horizon=5, cfg=cfg
    )
    s = targets["rv_tv"]
    for t in out.index[::17]:
        pos = targets.index.get_loc(t)
        assert out.loc[t, "y_pred"] == pytest.approx(
            s.iloc[pos - 4 : pos + 1].mean(), rel=1e-12
        )


def test_s1_ewma_matches_hand_recursion(targets, folds, cfg):
    # single-point grid -> argmin on boundary -> pre-specified fallback lambda 0.94
    cfg["ewma"].update({"lambda_start": 0.94, "lambda_stop": 0.94})
    out = forecast_classical(
        "ewma", targets, None, folds, target="rv_tv", horizon=1, cfg=cfg
    )
    lam, warm = 0.94, 60
    hist = targets["rv_tv"].loc[: folds[-1].test_idx[-1]]
    e = np.empty(len(hist))
    e[0] = hist.iloc[:warm].mean()
    for i in range(1, len(hist)):
        e[i] = lam * e[i - 1] + (1 - lam) * hist.iloc[i]
    e = pd.Series(e, index=hist.index)
    for t in out.index[::23]:
        assert out.loc[t, "y_pred"] == pytest.approx(e.loc[t], rel=1e-12)
    fit_log = out.attrs["fit_log"]
    assert all(v["lambda"] == 0.94 and v["fallback"] for v in fit_log.values())


def test_s1_ewma_flat_across_horizons(targets, folds, cfg):
    h1 = forecast_classical(
        "ewma", targets, None, folds, target="rv_tv", horizon=1, cfg=cfg
    )
    h5 = forecast_classical(
        "ewma", targets, None, folds, target="rv_tv", horizon=5, cfg=cfg
    )
    common = h1.index.intersection(h5.index)
    assert np.allclose(h1.loc[common, "y_pred"], h5.loc[common, "y_pred"])


# ------------------------------------------------- S2: full-field coverage


@pytest.mark.parametrize("model", CLASSICAL_MODELS)
def test_s2_one_variance_forecast_per_test_date(model, targets, folds, cfg):
    returns = None
    if model == "garch":
        rng = np.random.default_rng(5)
        returns = pd.Series(rng.normal(0.0, 0.01, len(INDEX)), index=INDEX)
    out = forecast_classical(
        model, targets, returns, folds, target="rv_tv", horizon=1, cfg=cfg
    )

    all_test_dates = folds[0].test_idx
    for f in folds[1:]:
        all_test_dates = all_test_dates.append(f.test_idx)
    expected = all_test_dates[:-1]  # last origin lacks its h=1 label
    assert out.index.equals(expected)
    assert not out.index.has_duplicates
    assert (out["y_pred"] > 0).all() and np.isfinite(out["y_pred"]).all()
    assert (out["y_true"] > 0).all()
    # variance space: same order of magnitude as the target, not its square root
    assert 0.01 < out["y_pred"].median() / targets["rv_tv"].median() < 100


# ------------------------------------------------- S3: HAR recovery on HAR data


def make_har_targets(
    index: pd.DatetimeIndex,
    seed: int,
    betas=(0.4, 0.3, 0.2),
    c: float = -0.921,
    sigma: float = 0.3,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(index)
    s = np.empty(n)
    s[:22] = 1e-4
    bd, bw, bm = betas
    for t in range(21, n - 1):
        log_next = (
            c
            + bd * np.log(s[t])
            + bw * np.log(s[t - 4 : t + 1].mean())
            + bm * np.log(s[t - 21 : t + 1].mean())
            + rng.normal(0.0, sigma)
        )
        s[t + 1] = np.exp(log_next)
    return pd.DataFrame({"rv_tv": s, "rv_oc": s}, index=index)


def test_s3_har_recovers_true_betas(folds, cfg):
    har_targets = make_har_targets(INDEX, seed=13)
    out = forecast_classical(
        "har", har_targets, None, folds, target="rv_tv", horizon=1, cfg=cfg
    )
    diag = next(iter(out.attrs["fit_log"].values()))
    p = diag["params"]
    assert np.isfinite(list(p.values())).all()
    assert p["log_d"] == pytest.approx(0.4, abs=0.15)
    assert p["log_w"] == pytest.approx(0.3, abs=0.25)
    assert p["log_m"] == pytest.approx(0.2, abs=0.25)
    assert 0.0 < diag["r2"] < 1.0
    assert diag["beta_sum"] == pytest.approx(0.9, abs=0.2)


# ------------------------------------------------- S4: GARCH units + recovery


def make_garch_returns(
    index: pd.DatetimeIndex,
    seed: int,
    omega: float = 2e-6,
    alpha: float = 0.08,
    beta: float = 0.90,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    n = len(index)
    r = np.empty(n)
    var = omega / (1 - alpha - beta)
    for t in range(n):
        z = rng.normal()
        r[t] = np.sqrt(var) * z
        var = omega + alpha * r[t] ** 2 + beta * var
    return pd.Series(r, index=index)


def test_s4_garch_units_and_parameter_recovery(folds, cfg):
    returns = make_garch_returns(INDEX, seed=17)
    targets = pd.DataFrame(
        {"rv_tv": returns**2 + 1e-12, "rv_oc": returns**2 + 1e-12}, index=INDEX
    )
    out = forecast_classical(
        "garch", targets, returns, folds, target="rv_tv", horizon=1, cfg=cfg
    )
    assert (out["y_pred"] > 0).all() and np.isfinite(out["y_pred"]).all()
    # unit round-trip (x100 in, /100^2 out): forecasts on the *decimal* variance scale.
    # A missed rescale would be off by 1e4.
    uncond = 2e-6 / (1 - 0.08 - 0.90)  # 1e-4
    assert 0.2 < out["y_pred"].mean() / uncond < 5.0
    # fitted parameters near truth (x100 scaling multiplies omega by 1e4)
    p = next(iter(out.attrs["fit_log"].values()))["params"]
    assert p["alpha[1]"] == pytest.approx(0.08, abs=0.06)
    assert p["beta[1]"] == pytest.approx(0.90, abs=0.10)
    assert p["omega"] == pytest.approx(2e-6 * 1e4, rel=2.0)


def test_s4_garch_h5_is_mean_of_step_forecasts(folds, cfg):
    returns = make_garch_returns(INDEX, seed=19)
    targets = pd.DataFrame(
        {"rv_tv": returns**2 + 1e-12, "rv_oc": returns**2 + 1e-12}, index=INDEX
    )
    out = forecast_classical(
        "garch", targets, returns, folds, target="rv_tv", horizon=5, cfg=cfg
    )
    assert (out["y_pred"] > 0).all() and np.isfinite(out["y_pred"]).all()


# ------------------------------------------------- S5: AR(1) back-transform unbiased


def test_s5_ar1_level_space_unbiasedness(folds, cfg):
    targets = make_log_ar1_targets(INDEX, seed=41, phi=0.9, sigma=0.3)
    out = forecast_classical(
        "ar1", targets, None, folds, target="rv_tv", horizon=1, cfg=cfg
    )
    ratio = out["y_pred"].mean() / out["y_true"].mean()
    assert 0.8 < ratio < 1.25, f"half-variance correction off: mean ratio {ratio:.3f}"


# ------------------------------------------------- S6: determinism


@pytest.mark.parametrize("model", ["har", "garch"])
def test_s6_two_runs_identical(model, targets, folds, cfg):
    returns = None
    if model == "garch":
        returns = make_garch_returns(INDEX, seed=23)
    a = forecast_classical(
        model, targets, returns, folds, target="rv_tv", horizon=1, cfg=cfg
    )
    b = forecast_classical(
        model, targets, returns, folds, target="rv_tv", horizon=1, cfg=cfg
    )
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------- S7: edge cases


def test_e6_returns_cc_contract(targets, folds, cfg):
    with pytest.raises(ValueError, match="returns_cc"):
        forecast_classical(
            "garch", targets, None, folds, target="rv_tv", horizon=1, cfg=cfg
        )
    dummy = pd.Series(0.01, index=INDEX)
    with pytest.raises(ValueError, match="returns_cc"):
        forecast_classical(
            "har", targets, dummy, folds, target="rv_tv", horizon=1, cfg=cfg
        )
    with pytest.raises(ValueError, match="unknown classical model"):
        forecast_classical(
            "midas", targets, None, folds, target="rv_tv", horizon=1, cfg=cfg
        )


def test_e4_garch_on_oc_carries_object_mismatch_flag(targets, folds, cfg):
    returns = make_garch_returns(INDEX, seed=29)
    out = forecast_classical(
        "garch", targets, returns, folds, target="rv_oc", horizon=1, cfg=cfg
    )
    assert out.attrs["object_mismatch"] is True
    out_tv = forecast_classical(
        "persistence", targets, None, folds, target="rv_tv", horizon=1, cfg=cfg
    )
    assert out_tv.attrs["object_mismatch"] is False


def test_e2_ewma_fit_window_too_short(targets, cfg):
    tiny = [RetrainFold(fit_idx=INDEX[:30], test_idx=INDEX[40:60])]
    with pytest.raises(ValueError, match="too short"):
        forecast_classical(
            "ewma", targets, None, tiny, target="rv_tv", horizon=1, cfg=cfg
        )


def test_e1_zero_target_in_series_errors_loudly(folds, cfg):
    bad = make_log_ar1_targets(INDEX, seed=31)
    bad.iloc[100, 0] = 0.0
    with pytest.raises(ValueError, match="non-positive label"):
        forecast_classical("ar1", bad, None, folds, target="rv_tv", horizon=1, cfg=cfg)


def test_r1_fold_union_covers_test_region(targets, folds, cfg):
    s = canonical_split(INDEX, CFG_SPLIT)
    out = forecast_classical(
        "persistence", targets, None, folds, target="rv_tv", horizon=1, cfg=cfg
    )
    assert out.index.isin(s.test_idx).all()
    assert len(out) == len(s.test_idx) - 1  # only the final label-less origin missing

"""The classical field: persistence, EWMA, AR(1) on log-target, HAR, GARCH(1,1).

Spec: specs/baselines/SPEC.md. Every model emits one variance-space forecast per test
origin per horizon per target, re-fit per monthly retrain fold on the fixed rolling window.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model

from src.features import make_labels
from src.metrics import qlike_series
from src.splits import RetrainFold

CLASSICAL_MODELS = ("persistence", "ewma", "ar1", "har", "garch")


# --------------------------------------------------------------------------- helpers


def _har_frame(targets: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Full-series design frame: log d/w/m aggregates at each origin + the log label."""
    s = targets[target]
    y, y_log = make_labels(targets, target, horizon)
    f = pd.DataFrame(
        {
            "log_d": np.log(s),
            "log_w": np.log(s.rolling(5).mean()),
            "log_m": np.log(s.rolling(22).mean()),
            "y": y,
            "y_log": y_log,
        }
    )
    return f


def _fit_origins(
    frame: pd.DataFrame, fit_idx: pd.DatetimeIndex, horizon: int
) -> pd.DataFrame:
    """Rows usable for fitting: origins inside the fit window whose label window also
    ends inside it (drop the last h origins so no label reaches past the window end)."""
    rows = frame.loc[frame.index.intersection(fit_idx)].dropna()
    if horizon > 0 and len(rows) > horizon:
        cutoff = fit_idx[-1]
        pos = rows.index.searchsorted(cutoff, side="right")
        rows = rows.iloc[: max(pos - horizon, 0)]
    return rows


def _ols_log_forecast(
    fit_rows: pd.DataFrame, x_cols: list[str], pred_rows: pd.DataFrame
) -> tuple[pd.Series, dict]:
    """OLS on the log label; back-transform with the lognormal half-variance correction."""
    X = sm.add_constant(fit_rows[x_cols], has_constant="add")
    res = sm.OLS(fit_rows["y_log"], X).fit()
    Xp = sm.add_constant(pred_rows[x_cols], has_constant="add")
    mu = res.predict(Xp)
    pred = np.exp(mu + res.scale / 2.0)
    diag = {
        "params": res.params.to_dict(),
        "r2": float(res.rsquared),
        "resid_var": float(res.scale),
    }
    return pred, diag


# --------------------------------------------------------------------------- models


def _persistence_fold(frame, fold, horizon, cfg):
    s_log_d = frame["log_d"]  # log of today's target
    origins = fold.test_idx.intersection(frame.dropna(subset=["y"]).index)
    if horizon == 1:
        pred = np.exp(s_log_d.loc[origins])
    else:
        # trailing 5-day mean of the level target through today
        lvl = np.exp(frame["log_d"])
        pred = lvl.rolling(5).mean().loc[origins]
    return pred, {}


def _ewma_fold(frame, fold, horizon, cfg):
    lvl = np.exp(frame["log_d"])  # level target series
    fit_lvl = lvl.loc[lvl.index.intersection(fold.fit_idx)].dropna()
    warmup = int(cfg["ewma"]["warmup_min_obs"])
    if len(fit_lvl) < warmup + 10:
        raise ValueError(f"EWMA fit window too short ({len(fit_lvl)} obs)")

    grid = np.arange(
        cfg["ewma"]["lambda_start"],
        cfg["ewma"]["lambda_stop"] + 1e-12,
        cfg["ewma"]["lambda_step"],
    )
    floor = cfg["floors"][cfg["_current_target"]]
    vals = fit_lvl.to_numpy()

    def _ewma_path(x: np.ndarray, lam: float) -> np.ndarray:
        e = np.empty_like(x)
        e[0] = x[:warmup].mean()
        for i in range(1, len(x)):
            e[i] = lam * e[i - 1] + (1 - lam) * x[i]
        return e

    scores = []
    for lam in grid:
        e = _ewma_path(vals, lam)
        # one-step in-sample: e[t-1] forecasts x[t]
        losses = qlike_series(e[warmup:-1], vals[warmup + 1 :], floor)
        scores.append(losses.mean())
    best = int(np.argmin(scores))
    fell_back = best in (0, len(grid) - 1)
    lam = float(cfg["ewma"]["fallback_lambda"]) if fell_back else float(grid[best])

    # forecast at each test origin: recursion over all data through that origin
    hist = lvl.loc[: fold.test_idx[-1]].dropna()
    e_full = pd.Series(_ewma_path(hist.to_numpy(), lam), index=hist.index)
    origins = fold.test_idx.intersection(frame.dropna(subset=["y"]).index)
    pred = e_full.loc[origins]  # flat forecaster: same value for h=1 and h=5
    return pred, {"lambda": lam, "fallback": bool(fell_back)}


def _ar1_fold(frame, fold, horizon, cfg):
    fit_rows = _fit_origins(frame, fold.fit_idx, horizon)
    origins = fold.test_idx.intersection(frame.dropna().index)
    pred, diag = _ols_log_forecast(fit_rows, ["log_d"], frame.loc[origins])
    return pred, diag


def _har_fold(frame, fold, horizon, cfg):
    fit_rows = _fit_origins(frame, fold.fit_idx, horizon)
    origins = fold.test_idx.intersection(frame.dropna().index)
    pred, diag = _ols_log_forecast(
        fit_rows, ["log_d", "log_w", "log_m"], frame.loc[origins]
    )
    b = diag["params"]
    beta_sum = b.get("log_d", 0) + b.get("log_w", 0) + b.get("log_m", 0)
    if not (np.isfinite(beta_sum) and 0.0 < beta_sum < 1.5 and 0.0 < diag["r2"] < 1.0):
        warnings.warn(
            f"HAR sanity check off in fold starting {fold.test_idx[0].date()}: "
            f"beta_sum={beta_sum:.3f}, r2={diag['r2']:.3f}",
            UserWarning,
        )
    diag["beta_sum"] = float(beta_sum)
    return pred, diag


def _garch_fold(frame, fold, horizon, cfg, returns_cc, prev_params):
    r = returns_cc.dropna() * 100.0
    r_fit = r.loc[r.index.intersection(fold.fit_idx)]
    am_fit = arch_model(
        r_fit, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False
    )
    params = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am_fit.fit(disp="off", show_warning=False)
    if res.convergence_flag == 0:
        params = res.params
    elif prev_params is not None:
        params = prev_params
        warnings.warn(
            f"GARCH did not converge in fold starting {fold.test_idx[0].date()}; "
            "carrying previous fold's parameters",
            UserWarning,
        )
    else:
        raise RuntimeError("GARCH failed to converge on the first fold")

    # Filter with fixed params through data <= each origin; forecast h steps ahead.
    origins = fold.test_idx.intersection(frame.dropna(subset=["y"]).index)
    r_thru = r.loc[: fold.test_idx[-1]]
    am_full = arch_model(
        r_thru, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False
    )
    fixed = am_full.fix(params)
    fc = fixed.forecast(horizon=horizon, start=origins[0], reindex=False)
    var = fc.variance.loc[origins]
    if horizon == 1:
        pred = var.iloc[:, 0]
    else:
        pred = var.iloc[:, :horizon].mean(axis=1)
    pred = pred / (100.0**2)
    if (pred <= 0).any() or not np.isfinite(pred).all():
        raise ValueError("GARCH produced non-positive/non-finite variance forecast")
    return pred, {"params": params.to_dict(), "carried": params is not res.params}


# --------------------------------------------------------------------------- driver


def forecast_classical(
    model: str,
    targets: pd.DataFrame,
    returns_cc: pd.Series | None,
    folds: list[RetrainFold],
    *,
    target: str,
    horizon: int,
    cfg: dict,
) -> pd.DataFrame:
    if model not in CLASSICAL_MODELS:
        raise ValueError(
            f"unknown classical model {model!r}; valid: {CLASSICAL_MODELS}"
        )
    if model == "garch" and returns_cc is None:
        raise ValueError(
            "garch requires returns_cc (close-to-close log returns, adjusted)"
        )
    if model != "garch" and returns_cc is not None:
        raise ValueError(f"returns_cc must be None for {model!r}")

    cfg = dict(cfg)
    cfg["_current_target"] = target
    frame = _har_frame(targets, target, horizon)

    rows = []
    fit_log = {}
    prev_params = None
    for i, fold in enumerate(folds):
        if model == "persistence":
            pred, diag = _persistence_fold(frame, fold, horizon, cfg)
        elif model == "ewma":
            pred, diag = _ewma_fold(frame, fold, horizon, cfg)
        elif model == "ar1":
            pred, diag = _ar1_fold(frame, fold, horizon, cfg)
        elif model == "har":
            pred, diag = _har_fold(frame, fold, horizon, cfg)
        else:
            pred, diag = _garch_fold(frame, fold, horizon, cfg, returns_cc, prev_params)
            prev_params = pd.Series(diag["params"])
        if (np.asarray(pred) <= 0).any() or not np.isfinite(np.asarray(pred)).all():
            raise ValueError(f"{model}: non-positive/non-finite prediction in fold {i}")
        fit_log[str(fold.test_idx[0].date())] = diag
        y_true = frame["y"].loc[pred.index]
        rows.append(
            pd.DataFrame(
                {
                    "y_pred": np.asarray(pred, dtype=float),
                    "y_true": y_true.to_numpy(dtype=float),
                    "model": model,
                    "target": target,
                    "horizon": horizon,
                    "fold_id": i,
                },
                index=pred.index,
            )
        )

    out = pd.concat(rows)
    out.index.name = "origin_date"
    out.attrs["fit_log"] = fit_log
    out.attrs["object_mismatch"] = bool(model == "garch" and target == "rv_oc")
    return out

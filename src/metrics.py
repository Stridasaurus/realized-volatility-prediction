"""Scoring and significance — the ONE QLIKE/RMSE/DM definition every experiment imports.

Spec: specs/metrics/SPEC.md. All inputs are variances aligned 1:1 by forecast-origin date.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class QlikeResult:
    value: float
    bind_rate: float
    n: int


@dataclass(frozen=True)
class DMResult:
    stat: float
    p_value: float
    hac_lag: int
    n: int
    mean_loss_diff: float


def _validate_pair(pred: np.ndarray, actual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if pred.ndim != 1 or actual.ndim != 1:
        raise ValueError("pred and actual must be 1-d arrays")
    if len(pred) != len(actual):
        raise ValueError(f"length mismatch: pred={len(pred)}, actual={len(actual)}")
    if len(pred) == 0:
        raise ValueError("empty input")
    if np.isnan(pred).any() or np.isnan(actual).any():
        raise ValueError("NaN in inputs")
    if (actual <= 0).any():
        bad = int((actual <= 0).sum())
        raise ValueError(
            f"{bad} non-positive actual variance value(s); targets are non-negative by "
            "construction after warmup — a zero/negative actual is degenerate for QLIKE "
            "and must be surfaced upstream, not floored away"
        )
    return pred, actual


def _unit_guard(pred: np.ndarray, actual: np.ndarray) -> None:
    med_p, med_a = float(np.median(pred)), float(np.median(actual))
    if med_p <= 0:
        warnings.warn("median prediction is non-positive — check forecast pipeline",
                      UserWarning, stacklevel=3)
        return
    ratio = max(med_p / med_a, med_a / med_p)
    if ratio > 1e3:
        warnings.warn(
            f"median(pred)/median(actual) ratio {ratio:.1e} exceeds 1e3 — suspected "
            "volatility-vs-variance unit mismatch (QLIKE must be scored in variance space)",
            UserWarning, stacklevel=3,
        )


def qlike_series(pred_var: np.ndarray, actual_var: np.ndarray, floor: float) -> np.ndarray:
    """Per-element QLIKE loss with the pre-specified positive floor applied to predictions."""
    pred, actual = _validate_pair(pred_var, actual_var)
    if not (isinstance(floor, (int, float)) and floor > 0):
        raise ValueError(f"floor must be positive, got {floor!r}")
    _unit_guard(pred, actual)
    p = np.maximum(pred, floor)
    r = actual / p
    return r - np.log(r) - 1.0


def qlike(pred_var: np.ndarray, actual_var: np.ndarray, floor: float) -> QlikeResult:
    losses = qlike_series(pred_var, actual_var, floor)
    pred = np.asarray(pred_var, dtype=float)
    bind_rate = float(np.mean(pred < floor))
    return QlikeResult(value=float(np.mean(losses)), bind_rate=bind_rate, n=len(losses))


def rmse(pred_var: np.ndarray, actual_var: np.ndarray) -> float:
    pred, actual = _validate_pair(pred_var, actual_var)
    _unit_guard(pred, actual)
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def _hac_variance(d: np.ndarray, lag: int) -> float:
    """HAC (Bartlett kernel) long-run variance of the mean of d."""
    n = len(d)
    dc = d - d.mean()
    gamma0 = float(np.dot(dc, dc)) / n
    s = gamma0
    for k in range(1, lag + 1):
        gamma_k = float(np.dot(dc[k:], dc[:-k])) / n
        s += 2.0 * (1.0 - k / (lag + 1.0)) * gamma_k
    return s


def dm_test(loss_model: np.ndarray, loss_bench: np.ndarray, h: int,
            hac_lag: int | None = None) -> DMResult:
    """Diebold–Mariano on a loss differential, HLN small-sample corrected, two-sided.

    HAC (Bartlett) variance with truncation lag >= h-1 (hard floor, SPEC E4); when
    ``hac_lag`` is None the lag is max(h-1, floor(4*(n/100)**(2/9))).
    Negative ``mean_loss_diff`` favors the model over the benchmark.
    """
    lm = np.asarray(loss_model, dtype=float)
    lb = np.asarray(loss_bench, dtype=float)
    if lm.shape != lb.shape or lm.ndim != 1:
        raise ValueError("loss series must be 1-d and equal length")
    if np.isnan(lm).any() or np.isnan(lb).any():
        raise ValueError("NaN in loss series")
    n = len(lm)
    if h < 1:
        raise ValueError("h must be >= 1")
    if n <= h:
        raise ValueError(f"n={n} <= h={h}: HLN correction undefined")

    if hac_lag is None:
        lag = max(h - 1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    else:
        if hac_lag < h - 1:
            raise ValueError(f"hac_lag={hac_lag} < h-1={h - 1}: the HAC lag floor is an invariant")
        lag = int(hac_lag)

    d = lm - lb
    dbar = float(d.mean())
    if np.allclose(d, d[0]):
        raise ValueError("degenerate loss differential (constant series)")
    s = _hac_variance(d, lag)
    if s <= 0:
        raise ValueError("degenerate loss differential (non-positive HAC variance)")

    dm = dbar / np.sqrt(s / n)
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln_factor)
    p_value = float(2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1)))
    return DMResult(stat=stat, p_value=p_value, hac_lag=lag, n=n, mean_loss_diff=dbar)

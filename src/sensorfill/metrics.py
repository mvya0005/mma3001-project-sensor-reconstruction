"""Error metrics for reconstructed values (unit topic 5.5).

All metrics are computed **only on the artificially hidden points** (the
``mask``), because observed points are returned unchanged by every method and
would dilute the errors.
"""

from __future__ import annotations

import numpy as np


def _pair(y_true, y_pred, mask=None):
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    m = np.ones(t.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    m = m & np.isfinite(t)
    if np.isnan(p[m]).any():
        raise ValueError("prediction contains NaN at evaluated points (method failed to fill)")
    return t[m], p[m]


def mae(y_true, y_pred, mask=None) -> float:
    """Mean absolute error (same units as the signal)."""
    t, p = _pair(y_true, y_pred, mask)
    return float(np.mean(np.abs(p - t)))


def rmse(y_true, y_pred, mask=None) -> float:
    """Root-mean-square error; penalises large errors more than MAE."""
    t, p = _pair(y_true, y_pred, mask)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def bias(y_true, y_pred, mask=None) -> float:
    """Mean signed error (positive = over-prediction)."""
    t, p = _pair(y_true, y_pred, mask)
    return float(np.mean(p - t))


def max_error(y_true, y_pred, mask=None) -> float:
    """Worst absolute error (robustness indicator)."""
    t, p = _pair(y_true, y_pred, mask)
    return float(np.max(np.abs(p - t)))


def r2(y_true, y_pred, mask=None) -> float:
    """Coefficient of determination on the hidden points (1 is perfect, can be negative)."""
    t, p = _pair(y_true, y_pred, mask)
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def skill_vs(y_true, y_pred, y_ref, mask=None) -> float:
    """RMSE skill score relative to a reference method: ``1 - RMSE/RMSE_ref``.

    Positive means better than the reference (e.g. linear interpolation).
    """
    return 1.0 - rmse(y_true, y_pred, mask) / rmse(y_true, y_ref, mask)


def physical_violations(y_pred, mask, low=None, high=None) -> int:
    """Number of reconstructed points outside physical limits ``[low, high]``."""
    p = np.asarray(y_pred, dtype=float)[np.asarray(mask, dtype=bool)]
    n = 0
    if low is not None:
        n += int(np.sum(p < low))
    if high is not None:
        n += int(np.sum(p > high))
    return n


def evaluate(y_true, y_pred, mask) -> dict:
    """All point metrics in one dictionary (MAE, RMSE, bias, max error, R^2, n)."""
    return {
        "mae": mae(y_true, y_pred, mask),
        "rmse": rmse(y_true, y_pred, mask),
        "bias": bias(y_true, y_pred, mask),
        "max_err": max_error(y_true, y_pred, mask),
        "r2": r2(y_true, y_pred, mask),
        "n": int(np.sum(np.asarray(mask, dtype=bool) & np.isfinite(np.asarray(y_true, dtype=float)))),
    }

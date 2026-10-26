"""Error metrics for reconstructed values (unit topic 5.5).

All metrics are computed **only on the artificially hidden points** (the
``mask``), because observed points are returned unchanged by every method and
would dilute the errors.

Every function accepts the same three array-likes:

* ``y_true`` - the ground truth, i.e. the series *before* points were hidden;
* ``y_pred`` - the reconstructed series returned by a fill method;
* ``mask`` - boolean, ``True`` where a point was hidden and must be scored.

Points where ``y_true`` is itself NaN (a real outage inside a hidden block)
are skipped, because there is nothing to compare against.

Example:
    >>> import numpy as np
    >>> from sensorfill.metrics import evaluate
    >>> truth = np.array([400.0, 420.0, 440.0, 460.0])
    >>> pred = np.array([400.0, 425.0, 435.0, 460.0])
    >>> hidden = np.array([False, True, True, False])
    >>> evaluate(truth, pred, hidden)["rmse"]
    5.0
"""

from __future__ import annotations

import numpy as np


def _pair(y_true, y_pred, mask=None):
    """Return the truth/prediction values at the points that should be scored.

    Raises:
        ValueError: If the arrays differ in shape, or the prediction still
            contains NaN at a scored point (the method failed to fill it).
    """
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    m = np.ones(t.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    # A hidden point with no ground truth (real outage) cannot be scored.
    m = m & np.isfinite(t)
    # A NaN left in the output is a method failure, not a large error, so it is
    # reported loudly instead of being silently dropped from the average.
    if np.isnan(p[m]).any():
        raise ValueError("prediction contains NaN at evaluated points (method failed to fill)")
    return t[m], p[m]


def mae(y_true, y_pred, mask=None) -> float:
    """Mean absolute error, in the same units as the signal.

    Args:
        y_true: Ground-truth values.
        y_pred: Reconstructed values, same shape as ``y_true``.
        mask: Boolean array marking the points to score. ``None`` scores
            every point with finite ground truth.

    Returns:
        The mean of ``|y_pred - y_true|`` over the scored points.

    Raises:
        ValueError: If the shapes differ or ``y_pred`` is NaN at a scored point.

    Example:
        >>> mae([1.0, 2.0, 3.0], [1.0, 2.5, 2.0])
        0.5
    """
    t, p = _pair(y_true, y_pred, mask)
    return float(np.mean(np.abs(p - t)))


def rmse(y_true, y_pred, mask=None) -> float:
    """Root-mean-square error; penalises large errors more than MAE.

    This is the headline metric of the project because a few large misses
    (e.g. a missed occupancy peak) matter more to a ventilation engineer than
    many small ones.

    Args:
        y_true: Ground-truth values.
        y_pred: Reconstructed values, same shape as ``y_true``.
        mask: Boolean array marking the points to score. ``None`` scores
            every point with finite ground truth.

    Returns:
        ``sqrt(mean((y_pred - y_true)**2))`` over the scored points.

    Raises:
        ValueError: If the shapes differ or ``y_pred`` is NaN at a scored point.

    Example:
        >>> rmse([0.0, 0.0], [3.0, 4.0])
        3.5355339059327378
    """
    t, p = _pair(y_true, y_pred, mask)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def bias(y_true, y_pred, mask=None) -> float:
    """Mean signed error (positive means the method over-predicts).

    Args:
        y_true: Ground-truth values.
        y_pred: Reconstructed values, same shape as ``y_true``.
        mask: Boolean array marking the points to score.

    Returns:
        The mean of ``y_pred - y_true`` over the scored points.

    Raises:
        ValueError: If the shapes differ or ``y_pred`` is NaN at a scored point.

    Example:
        >>> bias([10.0, 10.0], [11.0, 12.0])
        1.5
    """
    t, p = _pair(y_true, y_pred, mask)
    return float(np.mean(p - t))


def max_error(y_true, y_pred, mask=None) -> float:
    """Worst absolute error, used as a robustness indicator.

    Args:
        y_true: Ground-truth values.
        y_pred: Reconstructed values, same shape as ``y_true``.
        mask: Boolean array marking the points to score.

    Returns:
        ``max(|y_pred - y_true|)`` over the scored points.

    Raises:
        ValueError: If the shapes differ or ``y_pred`` is NaN at a scored point.

    Example:
        >>> max_error([0.0, 0.0, 0.0], [1.0, -3.0, 2.0])
        3.0
    """
    t, p = _pair(y_true, y_pred, mask)
    return float(np.max(np.abs(p - t)))


def r2(y_true, y_pred, mask=None) -> float:
    """Coefficient of determination on the hidden points.

    Args:
        y_true: Ground-truth values.
        y_pred: Reconstructed values, same shape as ``y_true``.
        mask: Boolean array marking the points to score.

    Returns:
        ``1 - SS_res / SS_tot``. 1 is perfect, 0 is no better than predicting
        the mean of the hidden truth, and negative is worse than that. NaN if
        the hidden truth is constant (``SS_tot = 0``).

    Raises:
        ValueError: If the shapes differ or ``y_pred`` is NaN at a scored point.

    Example:
        >>> r2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        1.0
    """
    t, p = _pair(y_true, y_pred, mask)
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def skill_vs(y_true, y_pred, y_ref, mask=None) -> float:
    """RMSE skill score relative to a reference method: ``1 - RMSE/RMSE_ref``.

    Args:
        y_true: Ground-truth values.
        y_pred: Reconstruction from the method being judged.
        y_ref: Reconstruction from the reference method (normally linear
            interpolation, the project baseline).
        mask: Boolean array marking the points to score.

    Returns:
        Positive if the method beats the reference (1 would be perfect), zero
        if equal, negative if worse.

    Raises:
        ValueError: If shapes differ or either prediction is NaN at a scored point.

    Example:
        >>> skill_vs([0.0, 0.0], [1.0, 1.0], [2.0, 2.0])
        0.5
    """
    return 1.0 - rmse(y_true, y_pred, mask) / rmse(y_true, y_ref, mask)


def physical_violations(y_pred, mask, low=None, high=None) -> int:
    """Count reconstructed points outside physical limits ``[low, high]``.

    Used as a physical-constraint check: e.g. indoor CO2 should not fall far
    below the outdoor level (~400 ppm) and indoor temperature stays in 0-45 degC.

    Args:
        y_pred: Reconstructed values.
        mask: Boolean array marking the reconstructed (hidden) points.
        low: Lower physical limit, or ``None`` for no lower check.
        high: Upper physical limit, or ``None`` for no upper check.

    Returns:
        The number of masked points below ``low`` or above ``high``.

    Example:
        >>> physical_violations([350.0, 500.0, 2000.0], [True, True, True], low=380, high=1500)
        2
    """
    p = np.asarray(y_pred, dtype=float)[np.asarray(mask, dtype=bool)]
    n = 0
    if low is not None:
        n += int(np.sum(p < low))
    if high is not None:
        n += int(np.sum(p > high))
    return n


def evaluate(y_true, y_pred, mask) -> dict:
    """Compute all point metrics in one dictionary.

    Args:
        y_true: Ground-truth values.
        y_pred: Reconstructed values, same shape as ``y_true``.
        mask: Boolean array marking the points to score.

    Returns:
        A dict with keys ``mae``, ``rmse``, ``bias``, ``max_err``, ``r2`` and
        ``n`` (the number of points actually scored).

    Raises:
        ValueError: If the shapes differ or ``y_pred`` is NaN at a scored point.

    Example:
        >>> res = evaluate([1.0, 2.0, 3.0], [1.0, 2.0, 4.0], [False, True, True])
        >>> res["n"], res["mae"]
        (2, 0.5)
    """
    return {
        "mae": mae(y_true, y_pred, mask),
        "rmse": rmse(y_true, y_pred, mask),
        "bias": bias(y_true, y_pred, mask),
        "max_err": max_error(y_true, y_pred, mask),
        "r2": r2(y_true, y_pred, mask),
        "n": int(np.sum(np.asarray(mask, dtype=bool) & np.isfinite(np.asarray(y_true, dtype=float)))),
    }

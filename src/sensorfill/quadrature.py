"""Numerical integration of sensor signals into engineering indicators.

Unit topics 6.1-6.5: Newton-Cotes rules and their errors, Romberg integration
with Richardson extrapolation, Gaussian quadrature and multidimensional
integration.

Why integrate? Building engineers rarely act on single readings; they act on
accumulated quantities such as

* **CO2 excess dose** ``int max(C - C_ref, 0) dt`` [ppm h] - by the mass
  balance this is proportional to occupant-hours divided by ventilation, so it
  is the natural ventilation-adequacy indicator.
* **Cooling degree-hours** ``int max(T - T_set, 0) dt`` [K h] - proportional to
  sensible cooling load.

The project validates reconstruction methods not only point-wise but by how
much they corrupt these integrals (:func:`daily_indicator`).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def trapezoid(y: np.ndarray, h: float) -> float:
    """Composite trapezoidal rule on uniform samples. Error O(h^2)."""
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        raise ValueError("need at least 2 samples")
    return float(h * (y.sum() - 0.5 * (y[0] + y[-1])))


def simpson13(y: np.ndarray, h: float) -> float:
    """Composite Simpson 1/3 rule. Needs an odd number of samples. Error O(h^4).

    Raises:
        ValueError: If the number of samples is even (odd number of intervals).
    """
    y = np.asarray(y, dtype=float)
    if y.size < 3 or y.size % 2 == 0:
        raise ValueError("Simpson 1/3 needs an odd number (>=3) of samples")
    return float(h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum()))


def simpson38(y: np.ndarray, h: float) -> float:
    """Composite Simpson 3/8 rule. Needs ``3k + 1`` samples. Error O(h^4)."""
    y = np.asarray(y, dtype=float)
    if y.size < 4 or (y.size - 1) % 3:
        raise ValueError("Simpson 3/8 needs 3k+1 samples")
    w = np.full(y.size, 3.0)
    w[0] = w[-1] = 1.0
    w[3:-1:3] = 2.0
    return float(3 * h / 8 * (w @ y))


def simpson_any(y: np.ndarray, h: float) -> float:
    """Simpson's rule for any sample count >= 3.

    Uses 1/3 for an even number of intervals, otherwise 3/8 on the last three
    intervals and 1/3 on the rest - the standard textbook combination.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return trapezoid(y, h)
    if n % 2 == 1:
        return simpson13(y, h)
    if n == 4:
        return simpson38(y, h)
    return simpson13(y[: n - 3], h) + simpson38(y[n - 4 :], h)


def romberg(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, max_level: int = 12, tol: float = 1e-10):
    """Romberg integration: trapezoid refinement + Richardson extrapolation.

    ``R[k, 0]`` is the trapezoid rule with ``2^k`` intervals;
    ``R[k, j] = R[k, j-1] + (R[k, j-1] - R[k-1, j-1]) / (4^j - 1)`` removes the
    ``h^{2j}`` error term.

    Args:
        f: Vectorised integrand.
        a, b: Limits.
        max_level: Maximum number of halvings.
        tol: Stop when successive diagonal entries differ by less than ``tol``
            (relative to the value).

    Returns:
        ``(estimate, table)`` with the lower-triangular Romberg table.
    """
    R = np.full((max_level + 1, max_level + 1), np.nan)
    h = b - a
    R[0, 0] = 0.5 * h * (f(np.array([a]))[0] + f(np.array([b]))[0])
    for k in range(1, max_level + 1):
        h /= 2
        new = a + h * np.arange(1, 2**k, 2)  # only the new midpoints are evaluated
        R[k, 0] = 0.5 * R[k - 1, 0] + h * f(new).sum()
        for j in range(1, k + 1):
            R[k, j] = R[k, j - 1] + (R[k, j - 1] - R[k - 1, j - 1]) / (4**j - 1)
        if abs(R[k, k] - R[k - 1, k - 1]) < tol * (1 + abs(R[k, k])):
            return float(R[k, k]), R[: k + 1, : k + 1]
    return float(R[max_level, max_level]), R


def gauss_legendre(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int = 5, panels: int = 1) -> float:
    """Composite Gauss-Legendre quadrature with ``n`` nodes per panel.

    Exact for polynomials of degree ``2n - 1`` on each panel.
    """
    x, w = np.polynomial.legendre.leggauss(n)
    edges = np.linspace(a, b, panels + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        total += half * (w @ f(mid + half * x))
    return float(total)


def gauss_legendre_2d(f: Callable[[np.ndarray, np.ndarray], np.ndarray], ax, bx, ay, by, n: int = 5) -> float:
    """Tensor-product Gauss-Legendre rule over a rectangle (multidimensional integration)."""
    x, w = np.polynomial.legendre.leggauss(n)
    X = (ax + bx) / 2 + (bx - ax) / 2 * x
    Y = (ay + by) / 2 + (by - ay) / 2 * x
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    return float((bx - ax) * (by - ay) / 4 * (w[:, None] * w[None, :] * f(XX, YY)).sum())


def excess(y: np.ndarray, threshold: float) -> np.ndarray:
    """Positive part ``max(y - threshold, 0)`` (NaN preserved)."""
    return np.clip(np.asarray(y, dtype=float) - threshold, 0, None)


def daily_indicator(
    y: np.ndarray,
    ts: pd.DatetimeIndex,
    threshold: float,
    h_hours: float,
    rule: str = "trapezoid",
) -> pd.Series:
    """Integrate ``max(y - threshold, 0)`` over each calendar day.

    Args:
        y: Complete (gap-free) series.
        ts: Timestamps of ``y``.
        threshold: Reference level (e.g. 420 ppm CO2, or 24 degC).
        h_hours: Sample spacing in hours.
        rule: ``"trapezoid"`` or ``"simpson"``.

    Returns:
        Series of daily integrals [units x h], indexed by date. Days that
        contain NaN return NaN rather than a silently biased value.
    """
    df = pd.DataFrame({"e": excess(y, threshold)}, index=pd.DatetimeIndex(ts))
    integ = trapezoid if rule == "trapezoid" else simpson_any

    def one(g):
        v = g["e"].to_numpy()
        return np.nan if np.isnan(v).any() or v.size < 2 else integ(v, h_hours)

    return df.groupby(df.index.date).apply(one)

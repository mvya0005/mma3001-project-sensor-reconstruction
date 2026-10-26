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

Example:
    >>> import numpy as np
    >>> from sensorfill.quadrature import trapezoid, simpson13
    >>> x = np.linspace(0, 1, 5)
    >>> round(trapezoid(x**2, h=0.25), 6), round(simpson13(x**2, h=0.25), 6)
    (0.34375, 0.333333)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def trapezoid(y: np.ndarray, h: float) -> float:
    """Composite trapezoidal rule on uniform samples. Error O(h^2).

    Args:
        y: Samples of the integrand at equally spaced points.
        h: Sample spacing (the integral is in ``units(y) x units(h)``).

    Returns:
        The trapezoid estimate of the integral over the sampled range.

    Raises:
        ValueError: If fewer than 2 samples are given.

    Example:
        >>> trapezoid([0.0, 1.0, 2.0], h=1.0)
        2.0
    """
    y = np.asarray(y, dtype=float)
    if y.size < 2:
        raise ValueError("need at least 2 samples")
    return float(h * (y.sum() - 0.5 * (y[0] + y[-1])))


def simpson13(y: np.ndarray, h: float) -> float:
    """Composite Simpson 1/3 rule. Error O(h^4).

    Args:
        y: Samples at equally spaced points; needs an odd count (>= 3), i.e.
            an even number of intervals.
        h: Sample spacing.

    Returns:
        The Simpson 1/3 estimate of the integral.

    Raises:
        ValueError: If there are fewer than 3 samples or an even number of samples.

    Example:
        >>> simpson13([0.0, 1.0, 4.0], h=1.0)  # x^2 on [0, 2], exact = 8/3
        2.6666666666666665
    """
    y = np.asarray(y, dtype=float)
    if y.size < 3 or y.size % 2 == 0:
        raise ValueError("Simpson 1/3 needs an odd number (>=3) of samples")
    # Weights 1, 4, 2, 4, ..., 2, 4, 1 (odd interior points get 4, even get 2).
    return float(h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum()))


def simpson38(y: np.ndarray, h: float) -> float:
    """Composite Simpson 3/8 rule. Error O(h^4).

    Args:
        y: Samples at equally spaced points; needs ``3k + 1`` samples.
        h: Sample spacing.

    Returns:
        The Simpson 3/8 estimate of the integral.

    Raises:
        ValueError: If the sample count is not ``3k + 1`` with ``k >= 1``.

    Example:
        >>> simpson38([0.0, 1.0, 8.0, 27.0], h=1.0)  # x^3 on [0, 3], exact = 20.25
        20.25
    """
    y = np.asarray(y, dtype=float)
    if y.size < 4 or (y.size - 1) % 3:
        raise ValueError("Simpson 3/8 needs 3k+1 samples")
    # Weights 1, 3, 3, 2, 3, 3, 2, ..., 3, 3, 1 (panel joins counted twice).
    w = np.full(y.size, 3.0)
    w[0] = w[-1] = 1.0
    w[3:-1:3] = 2.0
    return float(3 * h / 8 * (w @ y))


def simpson_any(y: np.ndarray, h: float) -> float:
    """Simpson's rule for any sample count.

    Uses 1/3 for an even number of intervals, otherwise 3/8 on the last three
    intervals and 1/3 on the rest - the standard textbook combination. With
    only 2 samples it falls back to the trapezoid rule.

    Args:
        y: Samples at equally spaced points (at least 2).
        h: Sample spacing.

    Returns:
        The integral estimate.

    Raises:
        ValueError: If fewer than 2 samples are given.

    Example:
        >>> simpson_any([0.0, 1.0, 4.0, 9.0], h=1.0)  # x^2 on [0, 3], exact = 9
        9.0
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return trapezoid(y, h)
    if n % 2 == 1:
        return simpson13(y, h)
    if n == 4:
        return simpson38(y, h)
    # Even sample count: 1/3 rule on the first n-3 samples, 3/8 on the last 4
    # (they share one sample, so nothing is counted twice).
    return simpson13(y[: n - 3], h) + simpson38(y[n - 4 :], h)


def romberg(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, max_level: int = 12, tol: float = 1e-10):
    """Romberg integration: trapezoid refinement + Richardson extrapolation.

    ``R[k, 0]`` is the trapezoid rule with ``2^k`` intervals;
    ``R[k, j] = R[k, j-1] + (R[k, j-1] - R[k-1, j-1]) / (4^j - 1)`` removes the
    ``h^{2j}`` error term.

    Args:
        f: Vectorised integrand, called with a 1-D array of abscissae.
        a: Lower limit.
        b: Upper limit.
        max_level: Maximum number of interval halvings.
        tol: Stop when successive diagonal entries differ by less than
            ``tol * (1 + |R[k, k]|)``.

    Returns:
        ``(estimate, table)``: the best estimate and the lower-triangular
        Romberg table up to the level reached (unused entries are NaN).

    Example:
        >>> import numpy as np
        >>> est, table = romberg(np.sin, 0.0, np.pi)
        >>> round(est, 10)
        2.0
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

    Args:
        f: Vectorised integrand.
        a: Lower limit.
        b: Upper limit.
        n: Gauss points per panel.
        panels: Number of equal sub-intervals of ``[a, b]``.

    Returns:
        The integral estimate.

    Example:
        >>> round(gauss_legendre(lambda x: x**5, 0.0, 1.0, n=3), 12)  # exact for degree <= 5
        0.166666666667
    """
    x, w = np.polynomial.legendre.leggauss(n)
    edges = np.linspace(a, b, panels + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        # Map the reference nodes on [-1, 1] onto this panel.
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        total += half * (w @ f(mid + half * x))
    return float(total)


def gauss_legendre_2d(f: Callable[[np.ndarray, np.ndarray], np.ndarray], ax, bx, ay, by, n: int = 5) -> float:
    """Tensor-product Gauss-Legendre rule over a rectangle (multidimensional integration).

    Args:
        f: Vectorised integrand ``f(X, Y)`` taking two equal-shape arrays.
        ax: Lower x limit.
        bx: Upper x limit.
        ay: Lower y limit.
        by: Upper y limit.
        n: Gauss points per direction (``n**2`` evaluations in total).

    Returns:
        The estimate of ``int_ay^by int_ax^bx f(x, y) dx dy``.

    Example:
        >>> gauss_legendre_2d(lambda x, y: x * y, 0, 1, 0, 2, n=2)
        1.0
    """
    x, w = np.polynomial.legendre.leggauss(n)
    X = (ax + bx) / 2 + (bx - ax) / 2 * x
    Y = (ay + by) / 2 + (by - ay) / 2 * x
    XX, YY = np.meshgrid(X, Y, indexing="ij")
    return float((bx - ax) * (by - ay) / 4 * (w[:, None] * w[None, :] * f(XX, YY)).sum())


def excess(y: np.ndarray, threshold: float) -> np.ndarray:
    """Positive part ``max(y - threshold, 0)``, with NaN preserved.

    Args:
        y: Signal values.
        threshold: Reference level subtracted from ``y``.

    Returns:
        Array of the same shape as ``y``.

    Example:
        >>> excess([400.0, 500.0, float("nan")], 420.0)
        array([ 0., 80., nan])
    """
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

    Example:
        >>> import pandas as pd
        >>> ts = pd.date_range("2024-06-03", periods=3, freq="1h")
        >>> float(daily_indicator([420.0, 520.0, 420.0], ts, threshold=420.0, h_hours=1.0).iloc[0])
        100.0
    """
    df = pd.DataFrame({"e": excess(y, threshold)}, index=pd.DatetimeIndex(ts))
    integ = trapezoid if rule == "trapezoid" else simpson_any

    def one(g):
        """Integrate one day's excess, or NaN if the day is incomplete."""
        v = g["e"].to_numpy()
        return np.nan if np.isnan(v).any() or v.size < 2 else integ(v, h_hours)

    return df.groupby(df.index.date).apply(one)

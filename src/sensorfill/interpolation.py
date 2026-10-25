"""Classical gap-filling by interpolation (unit topics 3.1-3.6).

Every ``fill_*`` function takes a 1-D array ``y`` on a regular grid with NaN
where data is missing, and returns a copy in which the NaNs are replaced.
Observed values are never modified. Gaps that touch either end of the series
cannot be bracketed, so they are filled by holding the nearest observed value
(an explicit, documented extrapolation rule).

Methods
-------
* :func:`fill_locf` - last observation carried forward (naive baseline).
* :func:`fill_linear` - piecewise-linear interpolation (the reference baseline).
* :func:`fill_linear_robust` - linear between noise-averaged gap anchors.
* :func:`fill_local_poly` - local least-squares polynomial regression (smoothing).
* :func:`fill_newton` - local Newton divided-difference polynomial through
  ``k`` points either side of each gap.
* :func:`fill_spline` - natural cubic spline, coefficients from our own
  tridiagonal solve (:func:`sensorfill.linalg.thomas_solve`).
* :func:`fill_pchip` - shape-preserving monotone cubic (SciPy), avoids overshoot.
* :func:`fill_profile` - typical daily profile with endpoint offset correction.
* :func:`fill_grid2d` - 2-D scattered interpolation over the (day, time-of-day) plane.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, griddata

from .data import gap_lengths
from .linalg import fold_daily, thomas_solve, unfold_daily


def _check(y) -> np.ndarray:
    """Validate input and return a float copy.

    Raises:
        ValueError: If ``y`` is not 1-D or has fewer than two observed values.
    """
    y = np.array(y, dtype=float, copy=True)
    if y.ndim != 1:
        raise ValueError("y must be 1-D")
    if np.count_nonzero(~np.isnan(y)) < 2:
        raise ValueError("need at least two observed values to interpolate")
    return y


def gap_spans(isnan: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` index pairs (stop exclusive) of each NaN run."""
    x = np.concatenate([[0], np.asarray(isnan, dtype=np.int8), [0]])
    d = np.diff(x)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def fill_locf(y) -> np.ndarray:
    """Last observation carried forward (leading NaNs take the first observation)."""
    y = _check(y)
    return pd.Series(y).ffill().bfill().to_numpy()


def fill_linear(y) -> np.ndarray:
    """Piecewise-linear interpolation between the observations bracketing each gap.

    Error for a smooth signal is bounded by ``h^2 max|y''| / 8`` where ``h`` is
    the gap duration, so it degrades quadratically with gap length.
    """
    y = _check(y)
    i = np.arange(y.size)
    ok = ~np.isnan(y)
    y[~ok] = np.interp(i[~ok], i[ok], y[ok])
    return y


def gap_anchors(y: np.ndarray, a: int, b: int, k: int = 3) -> tuple[float, float]:
    """Noise-robust boundary values for the gap ``[a, b)``.

    Returns the mean of the ``k`` nearest observations before and after the
    gap. With white sensor noise of standard deviation ``sigma`` this reduces
    the anchor error from ``sigma`` to ``sigma / sqrt(k)`` at the cost of a
    small bias when the signal is curved (an explicit bias-variance trade-off,
    tuned in notebook 01).

    Args:
        y: Series with NaNs.
        a, b: Gap start and stop (exclusive); ``a > 0`` and ``b < len(y)``.
        k: Observations averaged on each side (k=1 reproduces plain anchors).
    """
    left = y[:a][~np.isnan(y[:a])][-k:]
    right = y[b:][~np.isnan(y[b:])][:k]
    return float(left.mean()), float(right.mean())


def fill_linear_robust(y, k: int = 3) -> np.ndarray:
    """Linear interpolation between noise-averaged anchors (see :func:`gap_anchors`).

    Note the fill no longer passes exactly through the neighbouring readings;
    it passes through their local means, which is the better estimate of the
    *true* signal when readings are noisy.
    """
    y = _check(y)
    out = y.copy()
    for a, b in gap_spans(np.isnan(y)):
        if a == 0 or b == y.size:
            continue
        yl, yr = gap_anchors(y, a, b, k)
        pos = np.arange(1, b - a + 1) / (b - a + 1)
        out[a:b] = yl + (yr - yl) * pos
    return fill_locf(out)


def fill_local_poly(y, k: int = 6, degree: int = 1) -> np.ndarray:
    """Local least-squares polynomial *regression* across each gap.

    Unlike interpolation (which passes through every node and so reproduces
    the noise), this fits a degree-``degree`` polynomial to the ``k``
    observations either side by least squares (``numpy.polynomial``, QR-based)
    and evaluates it in the gap. ``2k > degree + 1`` gives an overdetermined,
    smoothing fit.

    Args:
        y: Series with NaNs.
        k: Observations used on each side.
        degree: Polynomial degree (0 = mean, 1 = line, 2 = parabola).
    """
    y = _check(y)
    ok = ~np.isnan(y)
    obs = np.flatnonzero(ok)
    out = y.copy()
    for a, b in gap_spans(~ok):
        left, right = obs[obs < a][-k:], obs[obs >= b][:k]
        if left.size == 0 or right.size == 0:
            continue
        nodes = np.concatenate([left, right])
        deg = min(degree, nodes.size - 1)
        poly = np.polynomial.Polynomial.fit(nodes, y[nodes], deg)  # domain scaling avoids ill-conditioning
        out[a:b] = poly(np.arange(a, b))
    return fill_locf(out)


# ---------------------------------------------------------------- Newton form
def newton_coefficients(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Divided-difference coefficients of the Newton interpolating polynomial.

    Args:
        x: Distinct nodes, shape (m,).
        f: Values at the nodes, shape (m,).

    Returns:
        ``c`` with ``c[k] = f[x_0, ..., x_k]`` so that
        ``p(t) = c0 + c1 (t-x0) + c2 (t-x0)(t-x1) + ...``.

    Raises:
        ValueError: If nodes repeat (division by zero in the table).
    """
    x = np.asarray(x, dtype=float)
    c = np.array(f, dtype=float, copy=True)
    if np.unique(x).size != x.size:
        raise ValueError("Newton interpolation needs distinct nodes")
    m = x.size
    for j in range(1, m):  # column j of the divided-difference table, in place
        c[j:] = (c[j:] - c[j - 1 : -1]) / (x[j:] - x[: m - j])
    return c


def newton_evaluate(c: np.ndarray, x: np.ndarray, t) -> np.ndarray:
    """Evaluate a Newton-form polynomial at ``t`` with nested (Horner) multiplication."""
    t = np.asarray(t, dtype=float)
    p = np.full_like(t, c[-1])
    for k in range(c.size - 2, -1, -1):
        p = p * (t - x[k]) + c[k]
    return p


def fill_newton(y, k: int = 2) -> np.ndarray:
    """Fill each gap with a local Newton polynomial of degree ``2k - 1``.

    Uses the ``k`` nearest observations on each side of the gap. Nodes are
    rescaled to ``[-1, 1]`` before forming the divided-difference table, which
    keeps the products ``(t - x_j)`` O(1) and limits round-off.

    Args:
        y: Series with NaNs.
        k: Points used on each side (k=1 is linear; larger k raises the
            degree and the risk of Runge-type oscillation inside long gaps).

    Returns:
        Filled copy of ``y``.
    """
    y = _check(y)
    if k < 1:
        raise ValueError("k must be >= 1")
    ok = ~np.isnan(y)
    obs = np.flatnonzero(ok)
    out = y.copy()
    for a, b in gap_spans(~ok):
        left = obs[obs < a][-k:]
        right = obs[obs >= b][:k]
        if left.size == 0 or right.size == 0:
            continue  # handled by edge rule below
        nodes = np.concatenate([left, right]).astype(float)
        mid, half = (nodes[0] + nodes[-1]) / 2, (nodes[-1] - nodes[0]) / 2
        xs = (nodes - mid) / half
        c = newton_coefficients(xs, y[nodes.astype(int)])
        out[a:b] = newton_evaluate(c, xs, (np.arange(a, b) - mid) / half)
    # Gaps at either end cannot be bracketed: hold nearest value.
    return fill_locf(out)


# ------------------------------------------------------------- cubic spline
def natural_spline_coefficients(x: np.ndarray, f: np.ndarray):
    """Natural cubic spline through ``(x, f)``.

    Solves the tridiagonal system for the second derivatives ``M_i`` with
    natural end conditions ``M_0 = M_n = 0``.

    Args:
        x: Strictly increasing knots, shape (n+1,).
        f: Knot values, shape (n+1,).

    Returns:
        ``(x, f, M)`` - knots, values and second derivatives, ready for
        :func:`spline_evaluate`.
    """
    x = np.asarray(x, dtype=float)
    f = np.asarray(f, dtype=float)
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("knots must be strictly increasing")
    n = x.size - 1
    M = np.zeros(n + 1)
    if n >= 2:
        # Interior equations: h_{i-1} M_{i-1} + 2(h_{i-1}+h_i) M_i + h_i M_{i+1} = 6(...)
        a = np.concatenate([[0.0], h[1:-1]])
        b = 2.0 * (h[:-1] + h[1:])
        c = np.concatenate([h[1:-1], [0.0]])
        d = 6.0 * (np.diff(f[1:]) / h[1:] - np.diff(f[:-1]) / h[:-1])
        M[1:-1] = thomas_solve(a, b, c, d)
    return x, f, M


def spline_evaluate(coef, t) -> np.ndarray:
    """Evaluate a natural cubic spline returned by :func:`natural_spline_coefficients`."""
    x, f, M = coef
    t = np.asarray(t, dtype=float)
    i = np.clip(np.searchsorted(x, t) - 1, 0, x.size - 2)
    h = x[i + 1] - x[i]
    A = (x[i + 1] - t) / h
    B = (t - x[i]) / h
    return A * f[i] + B * f[i + 1] + ((A**3 - A) * M[i] + (B**3 - B) * M[i + 1]) * h**2 / 6.0


def fill_spline(y, window: int | None = 36) -> np.ndarray:
    """Fill gaps with a natural cubic spline.

    Args:
        y: Series with NaNs.
        window: If given, fit a local spline through the observations within
            ``window`` steps either side of each gap (fast, and stops one bad
            reading influencing distant gaps). ``None`` fits one global spline
            through every observation.

    Returns:
        Filled copy of ``y``.
    """
    y = _check(y)
    ok = ~np.isnan(y)
    idx = np.arange(y.size, dtype=float)
    out = y.copy()
    if window is None:
        coef = natural_spline_coefficients(idx[ok], y[ok])
        out[~ok] = spline_evaluate(coef, idx[~ok])
        return fill_locf(np.where(_inside(ok), out, np.nan))
    for a, b in gap_spans(~ok):
        if a == 0 or b == y.size:
            continue
        lo, hi = max(0, a - window), min(y.size, b + window)
        sel = ok[lo:hi]
        xs = idx[lo:hi][sel]
        if xs.size < 3:
            out[a:b] = np.interp(idx[a:b], xs, y[lo:hi][sel])
            continue
        coef = natural_spline_coefficients(xs, y[lo:hi][sel])
        out[a:b] = spline_evaluate(coef, idx[a:b])
    return fill_locf(out)


def _inside(ok: np.ndarray) -> np.ndarray:
    """Boolean array that is True between the first and last observation."""
    first, last = np.argmax(ok), ok.size - np.argmax(ok[::-1])
    m = np.zeros(ok.size, dtype=bool)
    m[first:last] = True
    return m


def fill_pchip(y) -> np.ndarray:
    """Piecewise cubic Hermite (Fritsch-Carlson) interpolation via SciPy.

    Monotone between knots, so it cannot overshoot the bracketing values -
    useful for CO2, which should not dip below ambient during a gap.
    """
    y = _check(y)
    ok = ~np.isnan(y)
    idx = np.arange(y.size, dtype=float)
    out = y.copy()
    out[~ok] = PchipInterpolator(idx[ok], y[ok], extrapolate=False)(idx[~ok])
    return fill_locf(out)


# ------------------------------------------------------------ daily profile
def daily_profile(y, ts: pd.DatetimeIndex, split_weekend: bool = True) -> pd.Series:
    """Median value for each time-of-day slot (and weekday/weekend if requested).

    Args:
        y: Series values (NaN ignored).
        ts: Matching timestamps.
        split_weekend: Separate profiles for Mon-Fri and Sat-Sun.

    Returns:
        Series indexed by ``(is_weekend, minute_of_day)``.
    """
    ts = pd.DatetimeIndex(ts)
    key_we = (ts.dayofweek >= 5) if split_weekend else np.zeros(len(ts), dtype=bool)
    df = pd.DataFrame({"we": key_we, "mod": ts.hour * 60 + ts.minute, "y": np.asarray(y, dtype=float)})
    return df.groupby(["we", "mod"])["y"].median()


def fill_profile(y, ts, profile: pd.Series | None = None, split_weekend: bool = True) -> np.ndarray:
    """Typical-day profile with a linear offset correction to match gap endpoints.

    Inside a gap ``[a, b)`` the estimate is ``P(t) + delta(t)`` where ``P`` is the
    profile and ``delta`` interpolates linearly between ``y[a-1] - P(a-1)`` and
    ``y[b] - P(b)``. Long gaps therefore keep the daily shape while short gaps
    behave like linear interpolation.

    Args:
        y: Series with NaNs.
        ts: Timestamps of ``y``.
        profile: Pre-computed :func:`daily_profile` (e.g. from training data
            only, to avoid leakage). Computed from ``y`` itself if None.
        split_weekend: Passed to :func:`daily_profile` when computing.

    Returns:
        Filled copy of ``y``.
    """
    y = _check(y)
    ts = pd.DatetimeIndex(ts)
    if profile is None:
        profile = daily_profile(y, ts, split_weekend)
    we = (ts.dayofweek >= 5) if profile.index.get_level_values(0).nunique() > 1 else np.zeros(len(ts), dtype=bool)
    keys = pd.MultiIndex.from_arrays([we, ts.hour * 60 + ts.minute])
    P = profile.reindex(keys).to_numpy()
    P = fill_linear(P) if np.isnan(P).any() else P
    r = y - P
    return P + fill_linear(r)


# --------------------------------------------------------------- 2-D methods
def fill_grid2d(y, steps_per_day: int, offset: int = 0, method: str = "linear") -> np.ndarray:
    """Scattered 2-D interpolation on the (day, time-of-day) plane.

    The series is folded into a ``days x steps_per_day`` image. A missing
    pixel is interpolated from observed pixels at nearby times on the same day
    *and* at the same time on neighbouring days (SciPy ``griddata``, Delaunay
    triangulation for ``"linear"``/``"cubic"``).

    Args:
        y: Series with NaNs.
        steps_per_day: Grid points per day.
        offset: Index of the first midnight in ``y``.
        method: ``"nearest"``, ``"linear"`` or ``"cubic"``.

    Returns:
        Filled copy of ``y``.
    """
    y = _check(y)
    M = fold_daily(y, steps_per_day, offset)
    D, T = np.mgrid[0 : M.shape[0], 0 : M.shape[1]]
    ok = ~np.isnan(M)
    # Scale the day axis so one day ~ one hour of time-of-day distance.
    pts = np.column_stack([D[ok] * (steps_per_day / 24.0), T[ok]])
    q = np.column_stack([D[~ok] * (steps_per_day / 24.0), T[~ok]])
    M[~ok] = griddata(pts, M[ok], q, method=method)
    out = unfold_daily(M, y.size, offset)
    return fill_linear(out) if np.isnan(out).any() else out


def runge_demo(n_nodes: int, n_eval: int = 401):
    """Equispaced polynomial interpolation of Runge's function ``1/(1+25x^2)``.

    Returns ``(t, f(t), p(t))`` so that notebooks can show why high-degree
    global polynomials are a poor choice for long gaps.
    """
    x = np.linspace(-1, 1, n_nodes)
    f = 1.0 / (1 + 25 * x**2)
    t = np.linspace(-1, 1, n_eval)
    return t, 1.0 / (1 + 25 * t**2), newton_evaluate(newton_coefficients(x, f), x, t)


__all__ = [
    "gap_spans",
    "gap_lengths",
    "fill_locf",
    "fill_linear",
    "gap_anchors",
    "fill_linear_robust",
    "fill_local_poly",
    "newton_coefficients",
    "newton_evaluate",
    "fill_newton",
    "natural_spline_coefficients",
    "spline_evaluate",
    "fill_spline",
    "fill_pchip",
    "daily_profile",
    "fill_profile",
    "fill_grid2d",
    "runge_demo",
]

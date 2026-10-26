"""Common benchmark harness: every method, same masks, same metrics.

A *method* is any callable ``f(y_masked, ctx) -> y_filled`` where ``ctx`` is a
:class:`Context` holding the timestamps, occupancy and anything learned from
the training period. :func:`run_benchmark` hides blocks of observed data in
the evaluation series, calls each method, and records accuracy, runtime and
peak memory per gap length and repeat.

Example:
    >>> import numpy as np, pandas as pd
    >>> from sensorfill.benchmark import Context, classical_methods, run_benchmark, summarise
    >>> ts = pd.date_range("2024-06-03", periods=288, freq="10min")
    >>> y = 500 + 100 * np.sin(2 * np.pi * np.arange(288) / 144)
    >>> methods = {k: v for k, v in classical_methods().items() if k in ("LOCF", "Linear")}
    >>> res = run_benchmark(y, Context(ts), methods, gap_lengths=(6,), n_gaps=5, repeats=2)
    >>> summarise(res).index.tolist()  # best (lowest RMSE) first
    ['Linear', 'LOCF']
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np
import pandas as pd

from . import interpolation as itp
from . import metrics
from .linalg import fold_daily, svd_impute, unfold_daily
from .masking import apply_mask, block_mask
from .pde import laplace_fill


@dataclass
class Context:
    """Information available to a method besides the masked series.

    Attributes:
        ts: Timestamps of the evaluation series.
        occ: Occupancy on the same grid (or None).
        steps_per_day: Grid points per day.
        profile: Typical-day profile learned on the *training* period.
        extra: Free-form store for fitted models/parameters.
    """

    ts: pd.DatetimeIndex
    occ: np.ndarray | None = None
    steps_per_day: int = 144
    profile: pd.Series | None = None
    extra: dict = field(default_factory=dict)

    @property
    def midnight_offset(self) -> int:
        """Index of the first point at 00:00 (needed to fold into days).

        Returns:
            The index, or 0 if no timestamp falls exactly at midnight.
        """
        ts = pd.DatetimeIndex(self.ts)
        mins = ts.hour * 60 + ts.minute
        return int(np.argmax(mins == 0)) if (mins == 0).any() else 0


def svd_method(rank: int = 3) -> Callable:
    """Wrap :func:`sensorfill.linalg.svd_impute` as a benchmark method.

    Args:
        rank: Number of singular components kept.

    Returns:
        A method ``f(y, ctx)`` that folds ``y`` into days, completes the
        matrix, unfolds it and keeps every observed value unchanged.
    """

    def f(y, ctx: Context):
        """Fill ``y`` by rank-``rank`` completion of its day x time-of-day matrix."""
        off = ctx.midnight_offset
        M = fold_daily(y, ctx.steps_per_day, off)
        filled = svd_impute(M, rank=rank)
        out = unfold_daily(filled, len(y), off)
        out[~np.isnan(y)] = y[~np.isnan(y)]
        # Safety net: anything SVD could not reach is bridged linearly.
        return itp.fill_linear(out) if np.isnan(out).any() else out

    return f


def hybrid_method(short: Callable, long: Callable, max_short: int) -> Callable:
    """Combine two methods by gap length.

    Both methods see the same input; their outputs are merged gap by gap. This
    is the "policy" form of the final recommendation, where the threshold is
    chosen on validation data.

    Args:
        short: Method used for gaps of at most ``max_short`` steps.
        long: Method used for longer gaps.
        max_short: Threshold gap length in steps.

    Returns:
        A combined method ``f(y, ctx)``.

    Example:
        >>> m = hybrid_method(lambda y, c: itp.fill_locf(y), lambda y, c: itp.fill_linear(y), 1)
        >>> m(np.array([0.0, np.nan, 2.0, np.nan, np.nan, 5.0]), None)
        array([0., 0., 2., 3., 4., 5.])
    """

    def f(y, ctx: Context):
        """Run both methods and take each gap's fill from the one its length selects."""
        a, b = short(y, ctx), long(y, ctx)
        out = a.copy()
        for s0, s1 in itp.gap_spans(np.isnan(y)):
            if s1 - s0 > max_short:
                out[s0:s1] = b[s0:s1]
        return out

    return f


def classical_methods() -> dict[str, Callable]:
    """The training-free methods, keyed by display name.

    Returns:
        ``{name: f(y, ctx)}`` for every classical method compared in notebooks
        01-02 and 06. Settings are the ones selected on validation data.
    """
    return {
        "LOCF": lambda y, c: itp.fill_locf(y),
        "Linear": lambda y, c: itp.fill_linear(y),
        "Newton (k=2, cubic)": lambda y, c: itp.fill_newton(y, k=2),
        "Newton (k=4, deg 7)": lambda y, c: itp.fill_newton(y, k=4),
        "Natural spline": lambda y, c: itp.fill_spline(y, window=36),
        "PCHIP": lambda y, c: itp.fill_pchip(y),
        "Daily profile + offset": lambda y, c: itp.fill_profile(y, c.ts, profile=c.profile),
        "2-D griddata (day x tod)": lambda y, c: itp.fill_grid2d(y, c.steps_per_day, c.midnight_offset, "linear"),
        "Laplace 2-D (sparse direct)": lambda y, c: laplace_fill(y, steps_per_day=c.steps_per_day, w_day=0.5, method="direct"),
        "SVD low-rank (k=3)": svd_method(3),
    }


def run_benchmark(
    y_true: np.ndarray,
    ctx: Context,
    methods: dict[str, Callable],
    gap_lengths=(1, 3, 6, 18, 36, 72, 144),
    n_gaps: int = 30,
    repeats: int = 3,
    seed: int = 0,
    track_memory: bool = False,
    min_observed: float = 0.8,
) -> pd.DataFrame:
    """Score every method on identical artificial gaps.

    Args:
        y_true: Evaluation series (real NaNs allowed; they are left as NaN in
            the input to every method and never scored).
        ctx: Method context for this series.
        methods: ``{name: callable}``.
        gap_lengths: Artificial gap lengths in steps.
        n_gaps: Gaps per length per repeat (halved until they fit).
        repeats: Independent mask draws (for confidence intervals).
        seed: Base random seed; repeat ``r`` uses ``seed + r``.
        track_memory: Record peak Python memory (slows timing; off by default).
        min_observed: Passed to :func:`sensorfill.masking.block_mask`.

    Returns:
        Long table with one row per (method, gap length, repeat) and columns
        ``method, gap_len, repeat, n_gaps, seconds, peak_mb, error`` and the
        metrics of :func:`sensorfill.metrics.evaluate`. A method that raises is
        recorded with its exception in ``error`` and NaN metrics, so one
        failure cannot stop the whole benchmark.

    Raises:
        ValueError: If not even one gap of some length fits in ``y_true``.
    """
    y_true = np.asarray(y_true, dtype=float)
    rows = []
    for r in range(repeats):
        for L in gap_lengths:
            rng = np.random.default_rng(seed + 1000 * r + L)
            k = n_gaps
            # Halve the number of gaps until they fit (long gaps on short series).
            while True:
                try:
                    mask = block_mask(~np.isnan(y_true), L, k, rng, margin=2, min_observed=min_observed)
                    break
                except ValueError:
                    k //= 2
                    if k == 0:
                        raise
            y_in = apply_mask(y_true, mask)
            for name, f in methods.items():
                if track_memory:
                    tracemalloc.start()
                t0 = time.perf_counter()
                try:
                    y_hat = f(y_in, ctx)
                    err = None
                except Exception as exc:  # a failing method is recorded, not fatal
                    y_hat, err = None, repr(exc)
                dt = time.perf_counter() - t0
                peak = tracemalloc.get_traced_memory()[1] / 1e6 if track_memory else np.nan
                if track_memory:
                    tracemalloc.stop()
                row = {"method": name, "gap_len": L, "repeat": r, "n_gaps": k, "seconds": dt, "peak_mb": peak, "error": err}
                if y_hat is not None:
                    row.update(metrics.evaluate(y_true, y_hat, mask))
                rows.append(row)
    return pd.DataFrame(rows)


def confidence_table(results: pd.DataFrame, metric: str = "rmse") -> pd.DataFrame:
    """Mean and 95 % t-interval of ``metric`` across repeats.

    Repeats use independent mask draws, so their spread measures how much the
    score depends on *which* gaps were hidden.

    Args:
        results: Output of :func:`run_benchmark`.
        metric: Column to summarise.

    Returns:
        Table indexed by ``(method, gap_len)`` with ``mean``, ``ci95`` (the
        half-width of the Student-t interval) and ``n_repeats``.

    Example:
        >>> res = pd.DataFrame({"method": "A", "gap_len": 6, "rmse": [1.0, 2.0, 3.0]})
        >>> float(confidence_table(res).loc[("A", 6), "mean"])
        2.0
    """
    from scipy import stats

    g = results.groupby(["method", "gap_len"])[metric]
    n = g.count()
    half = stats.t.ppf(0.975, np.maximum(n - 1, 1)) * g.std() / np.sqrt(n)
    return pd.DataFrame({"mean": g.mean(), "ci95": half, "n_repeats": n})


def summarise(results: pd.DataFrame, metric: str = "rmse") -> pd.DataFrame:
    """Pivot results into a ``method x gap_len`` table of mean ``metric``.

    Args:
        results: Output of :func:`run_benchmark`.
        metric: Column to average (lower is better for all error metrics).

    Returns:
        Table sorted so the best method at the longest gap length comes first.

    Example:
        >>> res = pd.DataFrame({"method": ["A", "B"], "gap_len": [6, 6], "rmse": [2.0, 1.0]})
        >>> summarise(res).index.tolist()
        ['B', 'A']
    """
    return results.pivot_table(index="method", columns="gap_len", values=metric, aggfunc="mean").sort_values(
        results["gap_len"].max()
    )

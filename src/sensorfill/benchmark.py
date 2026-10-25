"""Common benchmark harness: every method, same masks, same metrics.

A *method* is any callable ``f(y_masked, ctx) -> y_filled`` where ``ctx`` is a
:class:`Context` holding the timestamps, occupancy and anything learned from
the training period. :func:`run_benchmark` hides blocks of observed data in
the evaluation series, calls each method, and records accuracy, runtime and
peak memory per gap length and repeat.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Callable

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
        """Index of the first point at 00:00 (needed to fold into days)."""
        ts = pd.DatetimeIndex(self.ts)
        mins = ts.hour * 60 + ts.minute
        return int(np.argmax(mins == 0)) if (mins == 0).any() else 0


def svd_method(rank: int = 3) -> Callable:
    """Wrap :func:`sensorfill.linalg.svd_impute` as a benchmark method."""

    def f(y, ctx: Context):
        off = ctx.midnight_offset
        M = fold_daily(y, ctx.steps_per_day, off)
        # Rows that are entirely missing carry no information; fill them later by linear interpolation.
        filled = svd_impute(M, rank=rank)
        out = unfold_daily(filled, len(y), off)
        out[~np.isnan(y)] = y[~np.isnan(y)]
        return itp.fill_linear(out) if np.isnan(out).any() else out

    return f


def hybrid_method(short: Callable, long: Callable, max_short: int) -> Callable:
    """Combine two methods by gap length: ``short`` for gaps of at most ``max_short`` steps, else ``long``.

    Both methods see the same input; their outputs are merged gap by gap. This
    is the "policy" form of the final recommendation, where the threshold is
    chosen on validation data.
    """

    def f(y, ctx: Context):
        a, b = short(y, ctx), long(y, ctx)
        out = a.copy()
        for s0, s1 in itp.gap_spans(np.isnan(y)):
            if s1 - s0 > max_short:
                out[s0:s1] = b[s0:s1]
        return out

    return f


def classical_methods() -> dict[str, Callable]:
    """The training-free methods, keyed by display name."""
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
        ``method, gap_len, repeat, mae, rmse, bias, max_err, r2, n, seconds, peak_mb``.
    """
    y_true = np.asarray(y_true, dtype=float)
    rows = []
    for r in range(repeats):
        for L in gap_lengths:
            rng = np.random.default_rng(seed + 1000 * r + L)
            k = n_gaps
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
    """Mean and 95 % t-interval half-width of ``metric`` across repeats, per method and gap length.

    Repeats use independent mask draws, so their spread measures how much the
    score depends on *which* gaps were hidden.
    """
    from scipy import stats

    g = results.groupby(["method", "gap_len"])[metric]
    n = g.count()
    half = stats.t.ppf(0.975, np.maximum(n - 1, 1)) * g.std() / np.sqrt(n)
    return pd.DataFrame({"mean": g.mean(), "ci95": half, "n_repeats": n})


def summarise(results: pd.DataFrame, metric: str = "rmse") -> pd.DataFrame:
    """Pivot to ``method x gap_len`` mean of ``metric`` (lower is better)."""
    return results.pivot_table(index="method", columns="gap_len", values=metric, aggfunc="mean").sort_values(
        results["gap_len"].max()
    )

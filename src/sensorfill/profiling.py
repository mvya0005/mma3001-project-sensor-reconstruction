"""Timing, memory, profiling and FLOP/roofline estimates (unit topic 2.1).

The brief asks for profilers, timing routines and estimates of arithmetic
intensity and FLOPs. :func:`measure` gives repeatable wall-clock time and
peak Python memory; :func:`profile_call` wraps :mod:`cProfile`;
:func:`flop_estimates` gives analytical operation counts used in the roofline
analysis (:func:`roofline`).
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Measurement:
    """Result of :func:`measure`.

    Attributes:
        median_s: Median wall-clock time over the repeats [s].
        min_s: Fastest repeat [s] (least affected by background load).
        peak_mb: Peak memory allocated by Python during one call [MB]
            (``tracemalloc``; excludes memory held by native libraries such as
            BLAS work arrays or the GPU).
        result: Return value of the last call.
    """

    median_s: float
    min_s: float
    peak_mb: float
    result: object


def measure(fn: Callable, *args, repeats: int = 3, **kwargs) -> Measurement:
    """Time ``fn(*args, **kwargs)`` over ``repeats`` runs and record peak memory once."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    tracemalloc.start()
    fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return Measurement(float(np.median(times)), float(np.min(times)), peak / 1e6, res)


def profile_call(fn: Callable, *args, top: int = 15, sort: str = "tottime", pattern: str | None = None, **kwargs) -> str:
    """Run ``fn`` under cProfile and return the formatted top-``top`` table.

    Args:
        fn: Function to profile; ``*args``/``**kwargs`` are passed to it.
        top: Number of rows to show.
        sort: pstats sort key (``"tottime"`` = time inside the function itself).
        pattern: Optional regular expression; only matching rows are shown
            (e.g. ``"sensorfill|scipy"`` hides Jupyter's own event-loop frames).
    """
    pr = cProfile.Profile()
    pr.enable()
    fn(*args, **kwargs)
    pr.disable()
    s = io.StringIO()
    st = pstats.Stats(pr, stream=s).sort_stats(sort)
    if pattern:
        st.print_stats(pattern, top)
    else:
        st.print_stats(top)
    return s.getvalue()


def flop_estimates(n_missing: int, n_total: int, gap_len: int, *, k_newton: int = 2, spline_window: int = 36,
                   svd_shape=(100, 144), svd_iters: int = 50, sor_iters: int = 100) -> dict:
    """Analytical FLOP and memory-traffic estimates per method.

    Counts are leading-order (constants matter less than scaling). Bytes
    assume float64 values read/written once per pass (8 B each).

    Args:
        n_missing: Number of missing points.
        n_total: Series length.
        gap_len: Typical gap length.
        k_newton: Points per side for Newton.
        spline_window: Local spline half-window.
        svd_shape: ``(days, steps_per_day)`` of the folded matrix.
        svd_iters: Hard-impute iterations.
        sor_iters: SOR sweeps.

    Returns:
        ``{method: {"flops": F, "bytes": B, "intensity": F/B}}``.
    """
    n_gaps = max(1, n_missing // max(gap_len, 1))
    m = 2 * k_newton
    est = {
        "linear": (3 * n_missing, 8 * 3 * n_missing),
        "newton": (n_gaps * (1.5 * m * m) + n_missing * 2 * m, 8 * (n_gaps * m * 2 + n_missing)),
        "spline": (n_gaps * 8 * (2 * spline_window + gap_len) + n_missing * 15, 8 * n_gaps * 6 * (2 * spline_window + gap_len)),
        "svd": (svd_iters * 4 * svd_shape[0] * svd_shape[1] ** 2, svd_iters * 8 * 3 * svd_shape[0] * svd_shape[1]),
        "sor": (sor_iters * 6 * n_missing, sor_iters * 8 * 4 * n_missing),
    }
    return {k: {"flops": float(f), "bytes": float(b), "intensity": float(f / b)} for k, (f, b) in est.items()}


def roofline(intensity: np.ndarray, peak_gflops: float, bandwidth_gbs: float) -> np.ndarray:
    """Attainable performance ``min(peak, intensity * bandwidth)`` in GFLOP/s.

    Args:
        intensity: Arithmetic intensity [FLOP/byte].
        peak_gflops: Peak compute of the device [GFLOP/s].
        bandwidth_gbs: Memory bandwidth [GB/s].
    """
    return np.minimum(peak_gflops, np.asarray(intensity) * bandwidth_gbs)


#: Nominal specifications used for roofline plots (vendor data sheets).
DEVICES = {
    "Colab T4 GPU (FP32)": {"peak_gflops": 8100.0, "bandwidth_gbs": 320.0},
    "Colab CPU (2 vCPU, AVX2 FP64, approx.)": {"peak_gflops": 70.0, "bandwidth_gbs": 15.0},
}

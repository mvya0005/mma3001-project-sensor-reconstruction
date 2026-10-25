"""Linear-algebra building blocks (unit topic: systems of linear equations, SVD).

* :func:`thomas_solve` - O(n) tridiagonal solver used by the cubic spline.
* :func:`gauss_solve` - Gaussian elimination with partial pivoting, kept as a
  transparent reference implementation for verification tests.
* :func:`fold_daily` / :func:`unfold_daily` - reshape a series into a
  ``days x steps_per_day`` matrix so that daily repetition becomes low rank.
* :func:`svd_impute` - iterative truncated-SVD matrix completion.
"""

from __future__ import annotations

import numpy as np


def thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system with the Thomas algorithm.

    The system is ``a[i] x[i-1] + b[i] x[i] + c[i] x[i+1] = d[i]``. ``a[0]`` and
    ``c[-1]`` are ignored. Cost is ~8n flops and O(n) memory, versus O(n^3)
    flops for dense elimination. Stable without pivoting when the matrix is
    diagonally dominant (true for the natural-spline system).

    Args:
        a: Sub-diagonal, length n.
        b: Main diagonal, length n.
        c: Super-diagonal, length n.
        d: Right-hand side, length n.

    Returns:
        Solution vector ``x`` of length n.

    Raises:
        ZeroDivisionError: If a zero pivot is met (matrix not suitable).
    """
    n = len(d)
    cp = np.empty(n)
    dp = np.empty(n)
    if b[0] == 0:
        raise ZeroDivisionError("zero pivot in Thomas algorithm")
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):  # forward sweep eliminates the sub-diagonal
        m = b[i] - a[i] * cp[i - 1]
        if m == 0:
            raise ZeroDivisionError("zero pivot in Thomas algorithm")
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):  # back substitution
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def gauss_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting.

    Reference implementation (O(n^3)); production code uses
    :func:`thomas_solve` or LAPACK via NumPy.

    Args:
        A: Square matrix (n x n).
        b: Right-hand side (n,).

    Returns:
        Solution vector.

    Raises:
        np.linalg.LinAlgError: If ``A`` is numerically singular.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    for k in range(n - 1):
        p = k + np.argmax(np.abs(A[k:, k]))  # partial pivoting limits error growth
        if abs(A[p, k]) < 1e-14 * np.abs(A).max():
            raise np.linalg.LinAlgError("matrix is singular to working precision")
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
        f = A[k + 1 :, k] / A[k, k]
        A[k + 1 :, k:] -= np.outer(f, A[k, k:])
        b[k + 1 :] -= f * b[k]
    x = np.empty(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1 :] @ x[i + 1 :]) / A[i, i]
    return x


def fold_daily(y: np.ndarray, steps_per_day: int, offset: int = 0) -> np.ndarray:
    """Reshape a regular series into a ``(days, steps_per_day)`` matrix.

    Args:
        y: 1-D series (NaN allowed).
        steps_per_day: Grid points per day (144 for 10-min data).
        offset: Index of the first point that falls at midnight. Points before
            it and any incomplete final day are padded with NaN.

    Returns:
        2-D array; row ``d`` is day ``d``, column ``j`` is time-of-day slot ``j``.
    """
    y = np.asarray(y, dtype=float)
    pad_front = (steps_per_day - offset) % steps_per_day
    z = np.concatenate([np.full(pad_front, np.nan), y])
    n_days = int(np.ceil(z.size / steps_per_day))
    z = np.concatenate([z, np.full(n_days * steps_per_day - z.size, np.nan)])
    return z.reshape(n_days, steps_per_day)


def unfold_daily(M: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
    """Inverse of :func:`fold_daily` for a series of original length ``n``."""
    spd = M.shape[1]
    pad_front = (spd - offset) % spd
    return M.reshape(-1)[pad_front : pad_front + n]


def svd_impute(
    M: np.ndarray,
    rank: int = 3,
    max_iter: int = 200,
    tol: float = 1e-5,
    return_history: bool = False,
):
    """Fill NaNs of a matrix with an iterative rank-``k`` SVD approximation.

    Algorithm (hard-impute): initialise missing entries with column means,
    then repeat: (1) compute the truncated SVD ``U_k S_k V_k^T``; (2) overwrite
    *only the missing entries* with that approximation; stop when the relative
    change of the filled entries falls below ``tol``.

    Assumption: the day-by-time-of-day matrix is approximately low rank
    because building operation repeats daily (a few patterns such as
    "workday" and "weekend" explain most of the variance).

    Args:
        M: 2-D array with NaN for missing entries.
        rank: Number of singular components kept (``k``).
        max_iter: Iteration cap.
        tol: Convergence threshold on the relative update of filled entries.
        return_history: If True, also return the per-iteration relative change
            and the singular values of the final completed matrix.

    Returns:
        Completed matrix, or ``(completed, history, singular_values)``.

    Raises:
        ValueError: If ``rank`` is not in ``[1, min(M.shape)]`` or a column is
            entirely missing.
    """
    M = np.asarray(M, dtype=float)
    if not 1 <= rank <= min(M.shape):
        raise ValueError("rank must be between 1 and min(M.shape)")
    miss = np.isnan(M)
    if miss.all(axis=0).any():
        raise ValueError("a column is entirely missing; SVD completion is undefined")
    col_mean = np.nanmean(M, axis=0)
    X = np.where(miss, col_mean[None, :], M)
    history = []
    for _ in range(max_iter):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        L = (U[:, :rank] * s[:rank]) @ Vt[:rank]
        new = np.where(miss, L, M)
        num = np.linalg.norm((new - X)[miss])
        den = np.linalg.norm(X[miss]) + 1e-12
        X = new
        history.append(num / den)
        if history[-1] < tol:
            break
    if return_history:
        return X, np.array(history), np.linalg.svd(X, compute_uv=False)
    return X


def condition_number(A: np.ndarray) -> float:
    """2-norm condition number ``sigma_max / sigma_min`` (inf if singular)."""
    s = np.linalg.svd(np.asarray(A, dtype=float), compute_uv=False)
    return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")

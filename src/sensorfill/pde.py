"""PDE-based gap filling: Laplace (elliptic) and diffusion (parabolic) inpainting.

Unit topics 10.1 (relaxation), 10.4 (PDE classification), 11.1 (FTCS),
11.2 (boundary conditions), 11.3 (elliptic PDEs), 11.4 (stability) and
11.5 (Crank-Nicolson).

Idea
----
Treat the missing values as the unknown field ``u`` of a PDE whose boundary
conditions are the *observed* readings (Dirichlet conditions). Two couplings
are supported through a graph Laplacian:

* **1-D (time only)**: neighbours ``i-1`` and ``i+1``. The steady (Laplace)
  solution ``u'' = 0`` is exactly linear interpolation, which gives an
  analytical check of the solvers.
* **2-D "day-coupled"**: additionally couples ``i`` with ``i +/- steps_per_day``
  (the same time of day on the previous/next day) with weight ``w_day``.
  Folding the series into a (day x time-of-day) image, this is the 5-point
  Laplacian with a *helical periodic* boundary on the time-of-day axis (the
  last slot of a day connects to the first slot of the next day).

``laplace_fill`` solves the elliptic problem ``L u = 0`` (on unknowns) with
Jacobi, Gauss-Seidel, SOR or a sparse direct solve. ``diffusion_fill``
marches the parabolic problem ``u_t = L u`` with FTCS (explicit, conditionally
stable) or Crank-Nicolson (implicit, unconditionally stable) and shows how the
steady state of diffusion is the Laplace solution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def graph_laplacian(n: int, steps_per_day: int | None = None, w_day: float = 1.0) -> sp.csr_matrix:
    """Build the (negative semi-definite) graph Laplacian ``L = W - D``.

    Args:
        n: Number of grid points.
        steps_per_day: If given, add day-to-day coupling at this lag.
        w_day: Weight of the day-to-day edges relative to the time edges (1.0).

    Returns:
        Sparse ``n x n`` matrix with ``(L y)_i = sum_j w_ij (y_j - y_i)``.
    """
    rows, cols, vals = [], [], []
    i = np.arange(n - 1)
    rows += [i, i + 1]
    cols += [i + 1, i]
    vals += [np.ones(n - 1), np.ones(n - 1)]
    if steps_per_day:
        j = np.arange(n - steps_per_day)
        rows += [j, j + steps_per_day]
        cols += [j + steps_per_day, j]
        vals += [np.full(j.size, w_day)] * 2
    W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    deg = np.asarray(W.sum(axis=1)).ravel()
    return (W - sp.diags(deg)).tocsr()


@dataclass
class SolveInfo:
    """Convergence record of an iterative solve.

    Attributes:
        method: Solver name.
        iterations: Iterations (or time steps) performed.
        converged: Whether the stopping criterion was met.
        residuals: Max-norm update per iteration (for convergence plots).
    """

    method: str
    iterations: int
    converged: bool
    residuals: list = field(default_factory=list)


def _split(y):
    y = np.array(y, dtype=float, copy=True)
    miss = np.isnan(y)
    if miss.all() or not miss.any():
        raise ValueError("need both observed and missing points")
    return y, miss


def laplace_fill(
    y,
    steps_per_day: int | None = None,
    w_day: float = 1.0,
    method: str = "sor",
    omega: float | None = None,
    tol: float = 1e-6,
    max_iter: int = 20_000,
    return_info: bool = False,
):
    """Fill NaNs by solving Laplace's equation with observed values as Dirichlet BCs.

    Args:
        y: Series with NaNs (at least one observed and one missing value).
        steps_per_day: Enables 2-D day coupling (see module docstring).
        w_day: Day-coupling weight.
        method: ``"jacobi"``, ``"gs"`` (Gauss-Seidel), ``"sor"`` or ``"direct"``.
        omega: SOR relaxation factor in ``(0, 2)``. Default 1.9.
        tol: Stop when the largest update is below ``tol * (range of y)``.
            Scaling by the data range makes the criterion unit-independent.
        max_iter: Iteration cap.
        return_info: Also return a :class:`SolveInfo`.

    Returns:
        Filled copy of ``y`` (and the :class:`SolveInfo` if requested).

    Notes:
        Gaps at the ends of the series have a boundary condition on one side
        only; the discrete solution there is constant (zero-flux / Neumann at
        the open end), equivalent to holding the last observed value.
    """
    y, miss = _split(y)
    L = graph_laplacian(y.size, steps_per_day, w_day)
    unk = np.flatnonzero(miss)
    A = L[unk][:, unk].tocsr()
    b = -(L[unk][:, ~miss] @ y[~miss])
    scale = np.nanmax(y) - np.nanmin(y) or 1.0
    # Initial guess: mean of observations.
    u = np.full(unk.size, np.nanmean(y))
    info = SolveInfo(method, 0, False)
    if method == "direct":
        u = spla.spsolve(A.tocsc(), b)
        info.converged, info.iterations = True, 1
    else:
        diag = A.diagonal()
        off = A - sp.diags(diag)
        if method == "jacobi":
            for k in range(max_iter):
                new = (b - off @ u) / diag
                r = np.max(np.abs(new - u))
                u = new
                info.residuals.append(r)
                if r < tol * scale:
                    info.converged = True
                    break
            info.iterations = k + 1
        elif method in ("gs", "sor"):
            w = 1.0 if method == "gs" else (1.9 if omega is None else omega)
            if not 0 < w < 2:
                raise ValueError("SOR requires 0 < omega < 2 for convergence")
            indptr, indices, data = off.indptr, off.indices, off.data
            for k in range(max_iter):
                r = 0.0
                for i in range(unk.size):  # in-place sweep uses freshest values
                    s = b[i]
                    for p in range(indptr[i], indptr[i + 1]):
                        s -= data[p] * u[indices[p]]
                    new = (1 - w) * u[i] + w * s / diag[i]
                    r = max(r, abs(new - u[i]))
                    u[i] = new
                info.residuals.append(r)
                if r < tol * scale:
                    info.converged = True
                    break
            info.iterations = k + 1
        else:
            raise ValueError(f"unknown method {method!r}")
    y[unk] = u
    return (y, info) if return_info else y


def optimal_sor_omega(gap_len: int) -> float:
    """Theoretical optimal SOR factor for a 1-D gap of ``gap_len`` unknowns.

    For the 1-D Dirichlet Laplacian the Jacobi spectral radius is
    ``rho = cos(pi / (m + 1))`` and ``omega* = 2 / (1 + sqrt(1 - rho^2))``.
    """
    rho = np.cos(np.pi / (gap_len + 1))
    return 2.0 / (1.0 + np.sqrt(1.0 - rho**2))


def ftcs_stability_limit(steps_per_day: int | None = None, w_day: float = 1.0) -> float:
    """Largest stable FTCS step ``r = D dt / dx^2`` for the chosen Laplacian.

    Gershgorin bounds the Laplacian eigenvalues by ``2 * max_degree``; FTCS
    (forward Euler in pseudo-time) is stable when ``r * |lambda_max| <= 2``,
    i.e. ``r <= 1 / max_degree``. For the 1-D chain this is the classical 0.5.
    """
    max_deg = 2.0 + (2.0 * w_day if steps_per_day else 0.0)
    return 1.0 / max_deg


def diffusion_fill(
    y,
    r: float = 0.4,
    scheme: str = "ftcs",
    n_steps: int = 5_000,
    steps_per_day: int | None = None,
    w_day: float = 1.0,
    tol: float = 1e-6,
    return_info: bool = False,
):
    """Fill gaps by evolving the diffusion equation ``u_t = L u`` on missing points.

    Observed points are held fixed (Dirichlet). The pseudo-time steady state is
    the Laplace solution, so this is a parabolic route to the same answer;
    stopping early gives a smoother-than-linear "partial" fill.

    Args:
        y: Series with NaNs.
        r: Non-dimensional step ``D dt / dx^2``.
        scheme: ``"ftcs"`` (explicit) or ``"cn"`` (Crank-Nicolson, implicit).
        n_steps: Maximum pseudo-time steps.
        steps_per_day, w_day: Laplacian coupling (see :func:`graph_laplacian`).
        tol: Stop when the max update per step is below ``tol * range(y)``.
        return_info: Also return a :class:`SolveInfo`.

    Returns:
        Filled copy of ``y`` (and info). With FTCS and ``r`` above
        :func:`ftcs_stability_limit` the iteration diverges; the info then has
        ``converged=False`` and growing residuals.
    """
    y, miss = _split(y)
    L = graph_laplacian(y.size, steps_per_day, w_day)
    unk = np.flatnonzero(miss)
    A = L[unk][:, unk].tocsc()
    b = L[unk][:, ~miss] @ y[~miss]  # contribution of fixed (Dirichlet) neighbours
    u = np.full(unk.size, np.nanmean(y))
    scale = np.nanmax(y) - np.nanmin(y) or 1.0
    info = SolveInfo(scheme, 0, False)
    if scheme == "cn":
        I = sp.identity(unk.size, format="csc")
        lu = spla.splu((I - 0.5 * r * A).tocsc())
        B = (I + 0.5 * r * A).tocsr()
    elif scheme != "ftcs":
        raise ValueError("scheme must be 'ftcs' or 'cn'")
    for k in range(n_steps):
        if scheme == "ftcs":
            new = u + r * (A @ u + b)
        else:
            new = lu.solve(B @ u + r * b)
        d = np.max(np.abs(new - u))
        u = new
        info.residuals.append(d)
        if not np.isfinite(d) or d > 1e6 * scale:
            break  # diverged (unstable explicit step)
        if d < tol * scale:
            info.converged = True
            break
    info.iterations = k + 1
    y[unk] = u
    return (y, info) if return_info else y

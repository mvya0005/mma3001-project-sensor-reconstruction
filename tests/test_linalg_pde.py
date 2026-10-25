"""Verification of the linear solvers, SVD completion and PDE inpainting."""

import numpy as np
import pytest

from sensorfill import interpolation as itp
from sensorfill import linalg, pde


def test_thomas_matches_dense_solve(rng):
    n = 50
    a, c = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    b = 4 + rng.uniform(0, 1, n)  # diagonally dominant
    d = rng.normal(size=n)
    A = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    np.testing.assert_allclose(linalg.thomas_solve(a, b, c, d), np.linalg.solve(A, d), atol=1e-12)


def test_gauss_solve_with_pivoting():
    A = np.array([[0.0, 2, 1], [1, 1, 1], [2, 1, 0]])  # zero first pivot needs a row swap
    b = np.array([3.0, 3, 3])
    np.testing.assert_allclose(linalg.gauss_solve(A, b), np.linalg.solve(A, b))
    with pytest.raises(np.linalg.LinAlgError):
        linalg.gauss_solve(np.ones((3, 3)), b)


def test_fold_unfold_roundtrip():
    y = np.arange(500, dtype=float)
    M = linalg.fold_daily(y, 144, offset=30)
    assert M.shape[1] == 144 and M[1, 0] == 30
    np.testing.assert_array_equal(linalg.unfold_daily(M, 500, 30), y)


def test_svd_impute_recovers_low_rank(rng):
    U, V = rng.normal(size=(30, 2)), rng.normal(size=(2, 40))
    M = U @ V
    Mm = M.copy()
    Mm[rng.random(M.shape) < 0.2] = np.nan
    filled, hist, s = linalg.svd_impute(Mm, rank=2, max_iter=2000, tol=1e-10, return_history=True)
    np.testing.assert_allclose(filled, M, atol=1e-4)
    assert hist[-1] < hist[0]
    assert s[2] / s[0] < 1e-4  # numerically rank 2


def test_svd_impute_validation():
    with pytest.raises(ValueError):
        linalg.svd_impute(np.ones((3, 3)), rank=4)
    M = np.ones((3, 3))
    M[:, 1] = np.nan
    with pytest.raises(ValueError):
        linalg.svd_impute(M, rank=1)


def test_condition_number():
    assert linalg.condition_number(np.diag([1.0, 10.0])) == pytest.approx(10.0)


@pytest.mark.parametrize("method", ["jacobi", "gs", "sor", "direct"])
def test_laplace_1d_equals_linear_interpolation(method):
    """Analytical check: the 1-D discrete Laplace solution with Dirichlet ends is the straight line."""
    y = np.r_[np.linspace(0, 1, 5), np.full(15, np.nan), np.linspace(3, 4, 5)]
    out, info = pde.laplace_fill(y, method=method, tol=1e-12, return_info=True)
    np.testing.assert_allclose(out, itp.fill_linear(y), atol=1e-8)
    assert info.converged


def test_sor_converges_faster_than_gs_and_jacobi():
    y = np.r_[[0.0], np.full(40, np.nan), [1.0]]
    its = {m: pde.laplace_fill(y, method=m, tol=1e-8, return_info=True)[1].iterations for m in ("jacobi", "gs")}
    its["sor"] = pde.laplace_fill(y, method="sor", omega=pde.optimal_sor_omega(40), tol=1e-8, return_info=True)[1].iterations
    assert its["sor"] < its["gs"] < its["jacobi"]


def test_sor_rejects_bad_omega():
    with pytest.raises(ValueError):
        pde.laplace_fill(np.array([0, np.nan, 1.0]), method="sor", omega=2.5)


def test_ftcs_stable_below_limit_and_unstable_above():
    y = np.r_[[0.0], np.full(30, np.nan), [1.0]]
    r_max = pde.ftcs_stability_limit()
    assert r_max == pytest.approx(0.5)
    out, info = pde.diffusion_fill(y, r=0.45, scheme="ftcs", n_steps=20000, tol=1e-10, return_info=True)
    assert info.converged
    np.testing.assert_allclose(out, itp.fill_linear(y), atol=1e-6)
    _, bad = pde.diffusion_fill(y, r=0.6, scheme="ftcs", n_steps=3000, return_info=True)
    assert not bad.converged and bad.residuals[-1] > bad.residuals[0]


def test_crank_nicolson_stable_for_large_steps():
    y = np.r_[[0.0], np.full(30, np.nan), [1.0]]
    out, info = pde.diffusion_fill(y, r=5.0, scheme="cn", n_steps=20000, tol=1e-10, return_info=True)
    assert info.converged
    np.testing.assert_allclose(out, itp.fill_linear(y), atol=1e-6)


def test_day_coupled_laplace_uses_previous_day(daily_signal):
    _, clean, _, _ = daily_signal
    y = clean.copy()
    y[144 * 2 + 60 : 144 * 2 + 100] = np.nan
    gap = np.isnan(y)
    e2 = np.abs(pde.laplace_fill(y, steps_per_day=144, method="direct")[gap] - clean[gap]).max()
    e1 = np.abs(itp.fill_linear(y)[gap] - clean[gap]).max()
    assert e2 < e1

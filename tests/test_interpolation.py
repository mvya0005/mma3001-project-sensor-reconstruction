"""Verification of the interpolation fills against exactly reproducible cases."""

import numpy as np
import pytest

from sensorfill import interpolation as itp


def poly_with_gap(deg, n=60, gap=(20, 30)):
    """Polynomial of the given degree with a block of NaN, plus the exact values."""
    x = np.arange(n, dtype=float)
    f = 0.001 * (x - 25) ** deg + 3
    y = f.copy()
    y[gap[0] : gap[1]] = np.nan
    return f, y


def test_linear_exact_on_lines():
    """Linear interpolation is exact for straight-line data."""
    f, y = poly_with_gap(1)
    np.testing.assert_allclose(itp.fill_linear(y), f)


def test_observed_values_are_never_changed(daily_signal):
    """Every fill method returns observed points unchanged and leaves no NaN."""
    _, _, noisy, _ = daily_signal
    y = noisy.copy()
    y[100:140] = np.nan
    for fill in (itp.fill_linear, itp.fill_pchip, itp.fill_newton, itp.fill_spline, itp.fill_locf):
        out = fill(y)
        ok = ~np.isnan(y)
        np.testing.assert_array_equal(out[ok], y[ok])
        assert not np.isnan(out).any()


def test_newton_divided_differences_reproduce_cubic():
    """A 4-node Newton polynomial (degree 3) reproduces cubic data to round-off."""
    f, y = poly_with_gap(3)
    # k=2 uses 4 nodes -> exact for cubics (degree 2k-1).
    np.testing.assert_allclose(itp.fill_newton(y, k=2), f, rtol=1e-10)


def test_newton_coefficients_textbook_example():
    """Divided differences of x^2 match the hand calculation, and repeated nodes are rejected."""
    # f(x) = x^2 at 0,1,2 -> c = [0, 1, 1].
    c = itp.newton_coefficients(np.array([0.0, 1, 2]), np.array([0.0, 1, 4]))
    np.testing.assert_allclose(c, [0, 1, 1])
    assert itp.newton_evaluate(c, np.array([0.0, 1, 2]), 3.0) == pytest.approx(9.0)
    with pytest.raises(ValueError):
        itp.newton_coefficients(np.array([0.0, 0.0]), np.array([1.0, 2.0]))


def test_natural_spline_matches_scipy():
    """Our tridiagonal natural spline agrees with SciPy's CubicSpline to 1e-10."""
    from scipy.interpolate import CubicSpline

    x = np.sort(np.random.default_rng(0).uniform(0, 10, 15))
    f = np.sin(x)
    t = np.linspace(x[0], x[-1], 300)
    ours = itp.spline_evaluate(itp.natural_spline_coefficients(x, f), t)
    ref = CubicSpline(x, f, bc_type="natural")(t)
    np.testing.assert_allclose(ours, ref, atol=1e-10)


def test_spline_error_converges_at_fourth_order():
    """Halving the knot spacing should cut the max error by ~2^4 for a smooth function."""
    errs = []
    for n in (10, 20, 40, 80):
        x = np.linspace(0, np.pi, n + 1)
        coef = itp.natural_spline_coefficients(x, np.sin(x))  # sin'' = 0 at both ends, so natural BC are exact
        t = np.linspace(0, np.pi, 2001)
        errs.append(np.max(np.abs(itp.spline_evaluate(coef, t) - np.sin(t))))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(orders > 3.7)


def test_pchip_does_not_overshoot():
    """PCHIP stays within the bracketing values across a step (no overshoot)."""
    y = np.array([0, 0, 0, np.nan, np.nan, 1, 1, 1], dtype=float)
    out = itp.fill_pchip(y)
    assert out.min() >= 0 and out.max() <= 1


def test_runge_phenomenon_grows_with_degree():
    """Equispaced interpolation error on Runge's function grows with degree."""
    errs = [np.max(np.abs(f - p)) for _, f, p in (itp.runge_demo(n) for n in (5, 11, 21))]
    assert errs[0] < errs[1] < errs[2]


def test_robust_anchors_reduce_noise_error(daily_signal):
    """Averaging k=3 anchors beats single-point anchors on noisy data."""
    _, clean, noisy, _ = daily_signal
    rng = np.random.default_rng(1)
    starts = rng.choice(np.arange(10, clean.size - 10), 300, replace=False)
    y = noisy.copy()
    y[starts] = np.nan
    e1 = np.sqrt(np.mean((itp.fill_linear(y)[starts] - clean[starts]) ** 2))
    e3 = np.sqrt(np.mean((itp.fill_linear_robust(y, 3)[starts] - clean[starts]) ** 2))
    assert e3 < e1


def test_local_poly_exact_on_line():
    """Local least-squares degree-1 fit is exact on a line."""
    f, y = poly_with_gap(1)
    np.testing.assert_allclose(itp.fill_local_poly(y, k=5, degree=1), f)


def test_profile_fill_recovers_periodic_signal(daily_signal):
    """On a purely periodic signal the profile fill is exact where linear is badly wrong."""
    ts, clean, _, _ = daily_signal
    y = clean.copy()
    y[144 * 8 + 50 : 144 * 8 + 120] = np.nan  # 70-step gap on a Tuesday
    out = itp.fill_profile(y, ts)
    gap = np.isnan(y)
    assert np.max(np.abs(out[gap] - clean[gap])) < 1e-9
    assert np.max(np.abs(itp.fill_linear(y)[gap] - clean[gap])) > 10


def test_grid2d_uses_neighbouring_days(daily_signal):
    """2-D interpolation uses the same time on other days and beats linear on a long gap."""
    ts, clean, _, _ = daily_signal
    y = clean.copy()
    y[144 * 2 + 60 : 144 * 2 + 100] = np.nan  # Wednesday afternoon (neighbours are workdays)
    gap = np.isnan(y)
    e2d = np.abs(itp.fill_grid2d(y, 144)[gap] - clean[gap]).max()
    elin = np.abs(itp.fill_linear(y)[gap] - clean[gap]).max()
    assert e2d < elin


def test_input_validation():
    """Too few observations or a 2-D input are rejected with ValueError."""
    with pytest.raises(ValueError):
        itp.fill_linear(np.array([np.nan, 1.0, np.nan]))
    with pytest.raises(ValueError):
        itp.fill_linear(np.ones((2, 2)))

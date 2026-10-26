"""Verification of the Newton-Cotes, Romberg and Gauss rules."""

import numpy as np
import pandas as pd
import pytest

from sensorfill import quadrature as q


def test_rules_exact_on_their_polynomials():
    """Trapezoid is exact for lines and both Simpson rules for cubics."""
    x = np.linspace(0, 2, 7)  # 6 intervals: valid for 1/3 and 3/8
    h = x[1] - x[0]
    assert q.trapezoid(3 * x + 1, h) == pytest.approx(8.0)
    assert q.simpson13(x**3, h) == pytest.approx(4.0)
    assert q.simpson38(x**3, h) == pytest.approx(4.0)
    assert q.simpson_any(x[:6] ** 3, h) == pytest.approx(x[5] ** 4 / 4)


@pytest.mark.parametrize("rule, order", [(q.trapezoid, 2), (q.simpson13, 4)])
def test_newton_cotes_error_orders(rule, order):
    """Trapezoid error falls at order 2 and Simpson at order 4 as h halves."""
    errs = []
    for n in (16, 32, 64):
        x = np.linspace(0, np.pi, n + 1)
        errs.append(abs(rule(np.sin(x), x[1] - x[0]) - 2.0))
    assert np.log2(errs[1] / errs[2]) == pytest.approx(order, abs=0.1)


def test_simpson_input_checks():
    """Simpson rules reject sample counts they cannot use."""
    with pytest.raises(ValueError):
        q.simpson13(np.ones(4), 1.0)
    with pytest.raises(ValueError):
        q.simpson38(np.ones(5), 1.0)


def test_romberg_converges_to_machine_precision():
    """Romberg reaches e - 1 to 1e-12 and extrapolation beats plain trapezoid."""
    val, R = q.romberg(np.exp, 0, 1, tol=1e-13)
    assert val == pytest.approx(np.e - 1, abs=1e-12)
    # Richardson columns improve on the trapezoid column.
    assert abs(R[3, 3] - (np.e - 1)) < abs(R[3, 0] - (np.e - 1))


def test_gauss_legendre_exactness_and_2d():
    """Gauss-Legendre is exact up to degree 2n-1, in 1-D and on a rectangle."""
    assert q.gauss_legendre(lambda x: x**9, -1, 1, n=5) == pytest.approx(0.0, abs=1e-14)
    assert q.gauss_legendre(lambda x: x**8, 0, 1, n=5) == pytest.approx(1 / 9)
    assert q.gauss_legendre_2d(lambda x, y: x * y**2, 0, 1, 0, 2, n=3) == pytest.approx(0.5 * 8 / 3)


def test_daily_indicator_and_nan_handling():
    """Daily excess integral matches the hand value and a day with NaN gives NaN."""
    ts = pd.date_range("2024-01-01", periods=2 * 144, freq="10min")
    y = np.full(ts.size, 520.0)  # 100 ppm above reference all day
    y[200] = np.nan
    d = q.daily_indicator(y, ts, threshold=420, h_hours=1 / 6)
    assert d.iloc[0] == pytest.approx(100 * (143 / 6))  # 143 intervals in a day of 144 samples
    assert np.isnan(d.iloc[1])

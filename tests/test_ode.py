"""Verification of finite differences, ODE integrators, the CO2 model and shooting."""

import numpy as np
import pytest

from sensorfill import ode


def test_finite_difference_orders():
    """Forward differences converge at order 1 and central differences at order 2."""
    errs = {s: [] for s in ("forward", "central")}
    for h in (0.1, 0.05, 0.025):
        x = np.arange(0, 2, h)
        for s in errs:
            d = ode.finite_difference(np.sin(x), h, s)
            errs[s].append(np.max(np.abs(d[2:-2] - np.cos(x[2:-2]))))
    p_fwd = np.log2(errs["forward"][0] / errs["forward"][1])
    p_cen = np.log2(errs["central"][0] / errs["central"][1])
    assert p_fwd == pytest.approx(1, abs=0.15)
    assert p_cen == pytest.approx(2, abs=0.15)


def test_second_difference():
    """The central second difference is exact for x^2."""
    x = np.linspace(0, 1, 101)
    d2 = ode.second_difference(x**2, x[1] - x[0])
    np.testing.assert_allclose(d2[1:-1], 2.0)


@pytest.mark.parametrize("method", list(ode.ORDER))
def test_integrators_converge_at_theoretical_order(method):
    """Empirical order on y' = -2y + sin(t), compared with the exact solution."""
    def f(t, y):
        """Right-hand side of the test ODE."""
        return -2 * y + np.sin(t)

    def exact(t):
        """Closed-form solution of the test ODE."""
        return (2 * np.sin(t) - np.cos(t)) / 5 + (1 + 1 / 5) * np.exp(-2 * t)

    errs = []
    for n in (40, 80, 160):
        t = np.linspace(0, 2, n + 1)
        # Max over the trajectory: the end-point error alone can pass near zero and mislead.
        errs.append(np.max(np.abs(ode.integrate(f, 1.0, t, method)[:, 0] - exact(t))))
    p = np.log2(errs[1] / errs[2])
    assert p == pytest.approx(ode.ORDER[method], abs=0.25)


def test_explicit_euler_unstable_implicit_stable_on_stiff_problem():
    """With h*lambda = 2.5 explicit Euler blows up and implicit Euler stays stable, as |R(z)| predicts."""
    lam = 50.0
    t = np.linspace(0, 1, 21)  # h*lam = 2.5 > 2: outside explicit Euler's stability region
    f = lambda t, y: -lam * y
    assert abs(ode.integrate(f, 1.0, t, "euler")[-1, 0]) > 1.0
    assert abs(ode.integrate(f, 1.0, t, "implicit_euler")[-1, 0]) < 1e-6
    assert abs(ode.amplification_factor("euler", -2.5)) > 1
    assert abs(ode.amplification_factor("implicit_euler", -2.5)) < 1
    assert abs(ode.amplification_factor("trapezoid", -1e6)) <= 1


def _synthetic_co2(p, hours=48, h=1 / 6):
    """Two-day synthetic CO2 series with a 9-17 h occupancy schedule, from the exact solution."""
    t = np.arange(0, hours, h)
    occ = np.where((t % 24 > 9) & (t % 24 < 17), 12.0, 0.0)
    C = ode.simulate_co2(p, t, occ, 420.0, "exact_linear")
    return t, occ, C


def test_rk4_matches_exact_discretisation():
    """RK4 at 10-min steps matches the exact discretisation to well below sensor noise."""
    p = ode.CO2Params(g=5.0, lam=1.5, c_out=420.0)
    t, occ, C = _synthetic_co2(p)
    C_rk4 = ode.simulate_co2(p, t, occ, 420.0, "rk4")
    # 10-min steps, time constant 40 min: RK4 error is ~1e-3 ppm, negligible against 8 ppm sensor noise.
    assert np.max(np.abs(C_rk4 - C)) < 5e-3


def test_steady_state_is_physical():
    """Constant occupancy drives CO2 to the analytical steady state C_out + gN/lambda."""
    p = ode.CO2Params(g=5.0, lam=1.0, c_out=420.0)
    t = np.arange(0, 20, 0.1)
    C = ode.simulate_co2(p, t, np.full(t.size, 10.0), 420.0, "rk4")
    assert C[-1] == pytest.approx(420 + 5 * 10 / 1.0, rel=1e-4)  # C_ss = C_out + gN/lambda


def test_parameter_recovery():
    """Equation-error gives a close start and output-error recovers the true parameters."""
    p = ode.CO2Params(g=4.0, lam=0.8, c_out=410.0)
    t, occ, C = _synthetic_co2(p, hours=96)
    p0 = ode.fit_equation_error(C, occ, t[1] - t[0])
    assert p0.lam == pytest.approx(0.8, rel=0.15)
    p1, _ = ode.fit_output_error([(t, C, occ)], p0)
    assert p1.g == pytest.approx(4.0, rel=1e-3)
    assert p1.lam == pytest.approx(0.8, rel=1e-3)
    assert p1.c_out == pytest.approx(410.0, abs=0.5)


def test_secant_root():
    """The secant method finds sqrt(2) in under 10 iterations."""
    r, k = ode.secant(lambda x: x**2 - 2, 1.0, 2.0)
    assert r == pytest.approx(np.sqrt(2)) and k < 10


def test_shooting_fill_matches_both_ends_and_truth():
    """Shooting and IVP fills match the true trajectory; shooting converges in at most 2 steps."""
    p = ode.CO2Params(g=5.0, lam=1.0, c_out=420.0)
    t, occ, C = _synthetic_co2(p)
    y = C.copy()
    y[50:80] = np.nan
    out, iters = ode.fill_ode(y, occ, p, 1 / 6, mode="shooting", return_iterations=True)
    np.testing.assert_allclose(out[50:80], C[50:80], atol=0.05)
    # Linear ODE -> end mismatch is affine in s -> secant converges in one or two steps.
    assert iters[0] <= 2
    ivp = ode.fill_ode(y, occ, p, 1 / 6, mode="ivp")
    np.testing.assert_allclose(ivp[50:80], C[50:80], atol=0.05)


def test_unknown_method_raises():
    """An unknown integrator name is rejected."""
    with pytest.raises(ValueError):
        ode.integrate(lambda t, y: y, 1.0, np.linspace(0, 1, 3), "magic")

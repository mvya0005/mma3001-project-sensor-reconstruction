"""Physics-based (grey-box) CO2 reconstruction with ordinary differential equations.

Unit topics 2.1 (optimisation), 7.1-7.5 (finite differences, IVPs, Euler,
Runge-Kutta), 9.1-9.4 (stability, stiffness, implicit methods, shooting).

Model
-----
A well-mixed zone of volume ``V`` with ``N(t)`` occupants each emitting CO2 at
``G`` and an outdoor-air supply ``Q`` obeys the mass balance

.. math::  V \\frac{dC}{dt} = G N(t) - Q (C - C_{out})

Dividing by ``V`` gives the linear, first-order ODE used here

.. math::  \\frac{dC}{dt} = g N(t) - \\lambda (C - C_{out}) + s

with lumped parameters ``g = G/V`` [ppm / (person h)], air-change rate
``lambda = Q/V`` [1/h], background ``C_out`` [ppm], and an optional constant
source ``s`` [ppm/h] that absorbs model error. Time is in **hours**.

Assumptions (all stated in the report): perfect mixing; constant ventilation
rate; occupancy measured in *another* zone is a proportional proxy for this
zone (so ``g`` is an effective, not physical, emission rate); outdoor CO2
constant over the study period.

Example:
    >>> import numpy as np
    >>> from sensorfill.ode import CO2Params, simulate_co2
    >>> p = CO2Params(g=0.0, lam=1.0, c_out=420.0)  # empty room, 1 air change per hour
    >>> c = simulate_co2(p, np.array([0.0, 1.0]), np.zeros(2), c0=800.0, method="exact_linear")
    >>> round(float(c[-1]), 1)  # 420 + 380 e^-1
    559.8
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from scipy.optimize import least_squares


# --------------------------------------------------------- finite differences
def finite_difference(y: np.ndarray, h: float, scheme: str = "central") -> np.ndarray:
    """First derivative of uniformly sampled data.

    Args:
        y: Samples (NaN propagates).
        h: Sample spacing.
        scheme: ``"forward"`` (O(h)), ``"backward"`` (O(h)) or ``"central"`` (O(h^2)).
            End points use the one-sided first-order formula.

    Returns:
        Derivative estimate, same length as ``y``.

    Raises:
        ValueError: If ``scheme`` is not one of the three names.

    Example:
        >>> finite_difference([0.0, 1.0, 4.0, 9.0], h=1.0)  # y = x^2
        array([1., 2., 4., 5.])
    """
    y = np.asarray(y, dtype=float)
    d = np.empty_like(y)
    if scheme == "forward":
        d[:-1] = (y[1:] - y[:-1]) / h
        d[-1] = d[-2]
    elif scheme == "backward":
        d[1:] = (y[1:] - y[:-1]) / h
        d[0] = d[1]
    elif scheme == "central":
        d[1:-1] = (y[2:] - y[:-2]) / (2 * h)
        d[0] = (y[1] - y[0]) / h
        d[-1] = (y[-1] - y[-2]) / h
    else:
        raise ValueError("scheme must be forward, backward or central")
    return d


def second_difference(y: np.ndarray, h: float) -> np.ndarray:
    """Central second derivative ``(y[i+1] - 2y[i] + y[i-1]) / h^2``.

    Args:
        y: Uniform samples.
        h: Sample spacing.

    Returns:
        Second-derivative estimate, same length as ``y``; the two end values
        are NaN because the stencil does not fit there.

    Example:
        >>> second_difference([0.0, 1.0, 4.0, 9.0], h=1.0)
        array([nan,  2.,  2., nan])
    """
    y = np.asarray(y, dtype=float)
    d = np.full_like(y, np.nan)
    d[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / h**2
    return d


# ------------------------------------------------------------ IVP integrators
def integrate(
    f: Callable[[float, np.ndarray], np.ndarray],
    y0,
    t: np.ndarray,
    method: str = "rk4",
    newton_tol: float = 1e-10,
    newton_max: int = 50,
) -> np.ndarray:
    """Integrate ``y' = f(t, y)`` on the given time grid with a fixed-step method.

    Args:
        f: Right-hand side; takes a float time and a 1-D state, returns an array.
        y0: Initial state (scalar or 1-D).
        t: Increasing time grid; the step may vary.
        method: ``"euler"`` (explicit, order 1), ``"heun"`` (explicit RK2),
            ``"midpoint"`` (explicit RK2), ``"rk4"`` (classical RK, order 4),
            ``"implicit_euler"`` (backward Euler, order 1, A-stable) or
            ``"trapezoid"`` (implicit, order 2, A-stable).
        newton_tol: Relative step-size tolerance of the Newton iteration used
            by the implicit methods (Jacobian by forward finite differences).
        newton_max: Maximum Newton iterations per implicit step.

    Returns:
        Array of shape ``(len(t), dim)``.

    Raises:
        RuntimeError: If the implicit Newton iteration fails to converge.
        ValueError: If ``method`` is unknown.

    Example:
        >>> t = np.linspace(0.0, 1.0, 11)
        >>> Y = integrate(lambda t, y: -y, 1.0, t, method="rk4")  # exact: e^-1
        >>> round(float(Y[-1, 0]), 6)
        0.36788
    """
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    Y = np.empty((len(t), y0.size))
    Y[0] = y0
    for n in range(len(t) - 1):
        h, tn, yn = t[n + 1] - t[n], t[n], Y[n]
        if method == "euler":
            Y[n + 1] = yn + h * f(tn, yn)
        elif method == "heun":
            k1 = f(tn, yn)
            k2 = f(tn + h, yn + h * k1)
            Y[n + 1] = yn + h / 2 * (k1 + k2)
        elif method == "midpoint":
            k1 = f(tn, yn)
            Y[n + 1] = yn + h * f(tn + h / 2, yn + h / 2 * k1)
        elif method == "rk4":
            k1 = f(tn, yn)
            k2 = f(tn + h / 2, yn + h / 2 * k1)
            k3 = f(tn + h / 2, yn + h / 2 * k2)
            k4 = f(tn + h, yn + h * k3)
            Y[n + 1] = yn + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        elif method in ("implicit_euler", "trapezoid"):
            theta = 1.0 if method == "implicit_euler" else 0.5
            fn = f(tn, yn)

            def G(z, yn=yn, h=h, tn=tn, fn=fn, theta=theta):
                """Residual of the implicit (theta-method) update; zero at the new state."""
                return z - yn - h * (theta * f(tn + h, z) + (1 - theta) * fn)

            z = yn + h * fn  # explicit predictor
            for _ in range(newton_max):
                g = G(z)
                eps = 1e-7 * (1 + np.abs(z))
                J = np.column_stack([(G(z + e) - g) / e[i] for i, e in enumerate(np.diag(eps))])
                dz = np.linalg.solve(J, -g)
                z = z + dz
                if np.max(np.abs(dz)) < newton_tol * (1 + np.max(np.abs(z))):
                    break
            else:
                raise RuntimeError("Newton iteration did not converge in implicit step")
            Y[n + 1] = z
        else:
            raise ValueError(f"unknown method {method!r}")
    return Y


#: Theoretical order of accuracy of each fixed-step method.
ORDER = {"euler": 1, "implicit_euler": 1, "heun": 2, "midpoint": 2, "trapezoid": 2, "rk4": 4}


def amplification_factor(method: str, z: complex) -> complex:
    """Stability function ``R(z)`` of a method for the test equation ``y' = lambda y``.

    One step multiplies the solution by ``R(z)`` with ``z = h lambda``, so a
    step is stable when ``|R(z)| <= 1``. For the CO2 model ``z = -h lambda``.

    Args:
        method: A key of :data:`ORDER`.
        z: ``h * lambda`` (complex allowed).

    Returns:
        ``R(z)``.

    Raises:
        ValueError: If ``method`` is unknown.

    Example:
        >>> amplification_factor("euler", -3.0), amplification_factor("implicit_euler", -3.0)
        (-2.0, 0.25)
    """
    if method == "euler":
        return 1 + z
    if method in ("heun", "midpoint"):
        return 1 + z + z**2 / 2
    if method == "rk4":
        return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
    if method == "implicit_euler":
        return 1 / (1 - z)
    if method == "trapezoid":
        return (1 + z / 2) / (1 - z / 2)
    raise ValueError(method)


# ----------------------------------------------------------- CO2 grey box
@dataclass
class CO2Params:
    """Parameters of the lumped CO2 mass balance (time unit: hours).

    Attributes:
        g: Effective generation per occupant [ppm / (person h)].
        lam: Air-change rate ``Q/V`` [1/h].
        c_out: Background (outdoor/supply) concentration [ppm].
        s: Constant unmodelled source [ppm/h].
    """

    g: float
    lam: float
    c_out: float
    s: float = 0.0

    @property
    def time_constant_h(self) -> float:
        """Room time constant ``1/lambda`` in hours.

        Returns:
            The e-folding time of a CO2 disturbance once the room empties.

        Example:
            >>> CO2Params(g=10.0, lam=2.0, c_out=420.0).time_constant_h
            0.5
        """
        return 1.0 / self.lam


def co2_rhs(params: CO2Params, occ: Callable[[float], float]):
    """Build the right-hand side of the CO2 mass balance.

    Args:
        params: Model parameters.
        occ: Occupancy as a function of time in hours.

    Returns:
        ``f(t, C) = g occ(t) - lambda (C - C_out) + s``, ready for :func:`integrate`.

    Example:
        >>> f = co2_rhs(CO2Params(g=10.0, lam=1.0, c_out=400.0), lambda t: 2.0)
        >>> f(0.0, 420.0)
        0.0
    """

    def f(t, C):
        """dC/dt [ppm/h] at time ``t`` [h] and concentration ``C`` [ppm]."""
        return params.g * occ(t) - params.lam * (C - params.c_out) + params.s

    return f


def simulate_co2(params: CO2Params, t_h: np.ndarray, occ_values: np.ndarray, c0: float, method: str = "rk4") -> np.ndarray:
    """Simulate CO2 given occupancy sampled on the same grid.

    Occupancy is linearly interpolated between samples, so RK stages that fall
    between grid points see a consistent input.

    Args:
        params: Model parameters.
        t_h: Time grid in hours.
        occ_values: Occupancy at ``t_h`` (persons; NaN treated as 0).
        c0: Initial concentration [ppm].
        method: Integrator (see :func:`integrate`), or ``"exact_linear"`` for
            the closed-form step that is exact for piecewise-linear occupancy
            (used for parameter fitting and as the verification reference).

    Returns:
        Simulated concentration on ``t_h``.

    Raises:
        ValueError: If ``method`` is unknown.

    Example:
        >>> p = CO2Params(g=50.0, lam=2.0, c_out=420.0)
        >>> c = simulate_co2(p, np.linspace(0, 6, 61), np.full(61, 4.0), c0=420.0)
        >>> round(float(c[-1]))  # steady state 420 + g N / lambda = 520
        520
    """
    occ_values = np.nan_to_num(np.asarray(occ_values, dtype=float))
    if method == "exact_linear":
        return _simulate_linear_exact(params, t_h, occ_values, c0)

    def occ(t):
        """Occupancy at any time ``t`` by linear interpolation."""
        return np.interp(t, t_h, occ_values)

    return integrate(co2_rhs(params, occ), c0, t_h, method)[:, 0]


def _simulate_linear_exact(p: CO2Params, t_h, occ, c0):
    """Exact discretisation for piecewise-linear occupancy (verification reference).

    For ``C' = -lam C + u(t)`` with ``u`` linear on ``[t_n, t_n+h]`` the step is
    ``C_{n+1} = e^{-lam h} C_n + int_0^h e^{-lam (h-s)} u(t_n + s) ds`` (closed form).
    """
    C = np.empty(len(t_h))
    C[0] = c0
    for n in range(len(t_h) - 1):
        h = t_h[n + 1] - t_h[n]
        u0 = p.g * occ[n] + p.lam * p.c_out + p.s
        u1 = p.g * occ[n + 1] + p.lam * p.c_out + p.s
        a = np.exp(-p.lam * h)
        # int_0^h e^{-lam(h-s)} ds = (1-a)/lam ; int_0^h e^{-lam(h-s)} s ds = h/lam - (1-a)/lam^2
        I0 = (1 - a) / p.lam
        I1 = h / p.lam - (1 - a) / p.lam**2
        C[n + 1] = a * C[n] + u0 * I0 + (u1 - u0) / h * I1
    return C


def fit_equation_error(C: np.ndarray, occ: np.ndarray, h: float) -> CO2Params:
    """Linear least-squares estimate of ``(g, lambda, C_out)`` from finite differences.

    Rewrites the ODE as ``dC/dt = a N + b C + c`` (``a=g``, ``b=-lambda``,
    ``c=lambda C_out``), estimates ``dC/dt`` by central differences and solves
    the overdetermined linear system with ``numpy.linalg.lstsq`` (QR/SVD - more
    stable than forming the normal equations).

    Args:
        C: CO2 series on a uniform grid (NaN rows are dropped).
        occ: Occupancy on the same grid.
        h: Grid spacing in hours.

    Returns:
        :class:`CO2Params` (with ``s = 0``). ``lambda`` is floored at 1e-3 /h
        so the fitted ventilation is always positive.

    Note:
        Differencing noisy data puts noise into the regressor ``dC/dt`` and
        biases ``lambda``; this estimate is therefore used only as the
        starting guess for :func:`fit_output_error`.

    Example:
        >>> p = CO2Params(g=50.0, lam=2.0, c_out=420.0)
        >>> t = np.arange(0, 48, 0.1)
        >>> occ = np.where((t % 24 > 8) & (t % 24 < 17), 5.0, 0.0)
        >>> C = simulate_co2(p, t, occ, 420.0, method="exact_linear")
        >>> est = fit_equation_error(C, occ, h=0.1)
        >>> bool(abs(est.c_out - 420) < 20)
        True
    """
    dC = finite_difference(C, h, "central")
    X = np.column_stack([occ, C, np.ones_like(C)])
    ok = np.isfinite(dC) & np.isfinite(X).all(axis=1)
    (a, b, c), *_ = np.linalg.lstsq(X[ok], dC[ok], rcond=None)
    lam = max(-b, 1e-3)  # physical constraint: positive ventilation
    return CO2Params(g=float(a), lam=float(lam), c_out=float(c / lam))


def fit_output_error(
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    init: CO2Params,
    method: str = "exact_linear",
    fit_s: bool = False,
) -> tuple[CO2Params, object]:
    """Nonlinear least-squares fit by simulating each observed segment.

    Minimises the sum of squared simulation errors over all segments, each
    started from its own first observation. Bounds keep parameters physical
    (``g >= 0``, ``0.01 <= lambda <= 20 /h``, ``300 <= C_out <= 600 ppm``).

    Args:
        segments: List of ``(t_h, C, occ)`` arrays without NaNs in ``C``.
        init: Starting guess (e.g. from :func:`fit_equation_error`).
        method: Integrator used inside the objective.
        fit_s: Also fit the constant source ``s``.

    Returns:
        ``(params, scipy_result)`` where ``scipy_result`` is the
        :class:`scipy.optimize.OptimizeResult` from ``least_squares``.

    Example:
        >>> true = CO2Params(g=50.0, lam=2.0, c_out=420.0)
        >>> t = np.arange(0, 24, 1 / 6)
        >>> occ = np.where((t > 8) & (t < 17), 5.0, 0.0)
        >>> C = simulate_co2(true, t, occ, 420.0, method="exact_linear")
        >>> fit, _ = fit_output_error([(t, C, occ)], CO2Params(30.0, 1.0, 400.0))
        >>> round(fit.g, 3), round(fit.lam, 3), round(fit.c_out, 3)
        (50.0, 2.0, 420.0)
    """

    def unpack(p):
        """Turn the optimiser's parameter vector into :class:`CO2Params`."""
        return CO2Params(float(p[0]), float(p[1]), float(p[2]), float(p[3]) if fit_s else 0.0)

    def resid(p):
        """Simulation-minus-measurement residuals over all segments."""
        prm = unpack(p)
        return np.concatenate([simulate_co2(prm, t, o, C[0], method) - C for t, C, o in segments])

    x0 = [max(init.g, 0.0), np.clip(init.lam, 0.02, 19.0), np.clip(init.c_out, 301, 599)] + ([0.0] if fit_s else [])
    lb = [0.0, 0.01, 300.0] + ([-500.0] if fit_s else [])
    ub = [np.inf, 20.0, 600.0] + ([500.0] if fit_s else [])
    res = least_squares(resid, x0, bounds=(lb, ub), x_scale="jac")
    return unpack(res.x), res


# --------------------------------------------------- gap filling: IVP & BVP
def secant(fun: Callable[[float], float], x0: float, x1: float, tol: float = 1e-8, max_iter: int = 50):
    """Secant root finder (derivative-free Newton).

    Args:
        fun: Scalar function whose root is wanted.
        x0: First starting point.
        x1: Second starting point.
        tol: Stop when ``|fun(x)| < tol`` or the step is below ``tol * (1 + |x|)``.
        max_iter: Iteration cap.

    Returns:
        ``(root, iterations)``.

    Raises:
        RuntimeError: If it fails to converge or the secant slope vanishes.

    Example:
        >>> root, k = secant(lambda x: x**2 - 2, 1.0, 2.0)
        >>> round(root, 8)
        1.41421356
    """
    f0, f1 = fun(x0), fun(x1)
    for k in range(1, max_iter + 1):
        if f1 == f0:
            raise RuntimeError("secant slope is zero")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, f0, x1, f1 = x1, f1, x2, fun(x2)
        if abs(f1) < tol or abs(x1 - x0) < tol * (1 + abs(x1)):
            return x1, k
    raise RuntimeError("secant method did not converge")


def fill_ode(
    y: np.ndarray,
    occ: np.ndarray,
    params: CO2Params,
    h: float,
    mode: str = "shooting",
    method: str = "rk4",
    return_iterations: bool = False,
):
    """Fill CO2 gaps by integrating the mass balance across each gap.

    Args:
        y: CO2 series with NaNs on a uniform grid.
        occ: Occupancy on the same grid.
        params: Fitted :class:`CO2Params`.
        h: Grid spacing in hours.
        mode: ``"ivp"`` integrates forward from the last observation before the
            gap (uses only the left boundary). ``"shooting"`` solves the
            two-point boundary-value problem: it finds the constant source ``s``
            for which the trajectory starting at the left value lands on the
            first observation after the gap (secant iteration), so the fill is
            continuous at both ends.
        method: Integrator.
        return_iterations: Also return the list of secant iteration counts.

    Returns:
        Filled copy of ``y`` (and iteration counts). A gap at the end of the
        series has no right boundary, so it is always integrated as an IVP; a
        gap at the start has no left boundary, so it holds the first
        observation.

    Raises:
        ValueError: If ``mode`` is not ``"ivp"`` or ``"shooting"``.
        RuntimeError: If the shooting iteration fails to converge.

    Example:
        >>> p = CO2Params(g=50.0, lam=2.0, c_out=420.0)
        >>> t = np.arange(0, 12, 1 / 6)
        >>> occ = np.where((t > 3) & (t < 9), 5.0, 0.0)
        >>> truth = simulate_co2(p, t, occ, 420.0, method="exact_linear")
        >>> y = truth.copy()
        >>> y[20:40] = np.nan
        >>> filled = fill_ode(y, occ, p, h=1 / 6)
        >>> bool(np.max(np.abs(filled - truth)) < 0.5)
        True
    """
    from .interpolation import fill_locf, gap_spans

    if mode not in ("ivp", "shooting"):
        raise ValueError("mode must be 'ivp' or 'shooting'")
    y = np.array(y, dtype=float, copy=True)
    miss = np.isnan(y)
    iters = []
    for a, b in gap_spans(miss):
        if a == 0:
            continue
        t = np.arange(a - 1, min(b + 1, y.size)) * h
        o = occ[a - 1 : min(b + 1, y.size)]
        c0 = y[a - 1]
        if mode == "ivp" or b == y.size:
            sim = simulate_co2(params, t, o, c0, method)
        elif mode == "shooting":
            target = y[b]

            def miss_end(s, t=t, o=o, c0=c0, target=target):
                """Miss distance at the right boundary for an extra source ``s``."""
                p = CO2Params(params.g, params.lam, params.c_out, params.s + s)
                return simulate_co2(p, t, o, c0, method)[-1] - target

            s_star, k = secant(miss_end, 0.0, 10.0)
            iters.append(k)
            sim = simulate_co2(CO2Params(params.g, params.lam, params.c_out, params.s + s_star), t, o, c0, method)
        y[a:b] = sim[1 : 1 + (b - a)]
    y = fill_locf(y)
    return (y, iters) if return_iterations else y

"""sensorfill - reconstructing missing building-sensor measurements.

MMA3001 (Numerical Methods and Machine Learning) project on the Monash Smart
Infrastructure occupancy and environmental dataset (Dataset 2).

Modules
-------
- :mod:`sensorfill.data` - load, audit and regularise the raw exports.
- :mod:`sensorfill.masking` - artificial gaps for validation against withheld truth.
- :mod:`sensorfill.interpolation` - linear, Newton, spline, PCHIP, profile and 2-D fills.
- :mod:`sensorfill.linalg` - Thomas/Gauss solvers and SVD low-rank completion.
- :mod:`sensorfill.pde` - Laplace (relaxation) and diffusion (FTCS/Crank-Nicolson) inpainting.
- :mod:`sensorfill.ode` - CO2 mass-balance model, IVP integrators, shooting method.
- :mod:`sensorfill.quadrature` - Newton-Cotes, Romberg, Gauss rules and daily indicators.
- :mod:`sensorfill.features` / :mod:`sensorfill.ml` - regression imputers (scikit-learn).
- :mod:`sensorfill.deep` - BiGRU, physics-informed and neural-ODE imputers (PyTorch).
- :mod:`sensorfill.metrics` - error metrics on hidden points.
- :mod:`sensorfill.profiling` - timing, memory, cProfile and roofline helpers.
- :mod:`sensorfill.benchmark` - one harness that scores every method on identical gaps.
- :mod:`sensorfill.plotting` - colour-blind-checked figure style.
- :mod:`sensorfill.study` - sensors, chronological splits and occupancy proxy used by all notebooks.

``sensorfill.deep`` is imported lazily because PyTorch is optional.
"""

__version__ = "1.0.0"

from . import benchmark, data, features, interpolation, linalg, masking, metrics, ml, ode, pde, profiling, quadrature, study

__all__ = [
    "benchmark", "data", "features", "interpolation", "linalg", "masking",
    "metrics", "ml", "ode", "pde", "profiling", "quadrature", "study", "plotting",
    "deep",  # listed for the documentation; imported on demand because PyTorch is optional
]

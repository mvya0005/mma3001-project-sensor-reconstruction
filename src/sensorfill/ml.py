"""Machine-learning regression imputers (unit topics 5.1-5.6).

Each model learns the residual between the truth and linear interpolation
from gap-aware features (:mod:`sensorfill.features`). Models:

* ``linear``  - ordinary least squares on standardised features (5.1)
* ``ridge``   - L2-regularised least squares (overfitting control, 3.2/5.6)
* ``tree``    - CART decision tree (5.2)
* ``forest``  - random forest (bagged trees; variance reduction)
* ``svr``     - epsilon-SVR with an RBF kernel (5.3)
* ``mlp``     - multilayer perceptron with early stopping (5.4)

Hyperparameters are selected by grouped K-fold cross-validation in which all
rows from one gap stay in the same fold (rows inside a gap are strongly
correlated; splitting them would leak information and overstate skill).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .features import GapFeatureBuilder


def make_model(name: str, random_state: int = 0, **params) -> Pipeline:
    """Create a scikit-learn pipeline (scaler + regressor) by name.

    Args:
        name: One of ``linear, ridge, tree, forest, svr, mlp``.
        random_state: Seed for stochastic models.
        **params: Overrides passed to the regressor.

    Returns:
        Unfitted :class:`~sklearn.pipeline.Pipeline` whose last step is named ``"reg"``.
    """
    if name == "linear":
        reg = LinearRegression(**params)
    elif name == "ridge":
        reg = Ridge(**({"alpha": 1.0} | params))
    elif name == "tree":
        reg = DecisionTreeRegressor(**({"max_depth": 8, "min_samples_leaf": 20, "random_state": random_state} | params))
    elif name == "forest":
        reg = RandomForestRegressor(
            **({"n_estimators": 200, "min_samples_leaf": 5, "n_jobs": -1, "random_state": random_state} | params)
        )
    elif name == "svr":
        reg = SVR(**({"C": 10.0, "epsilon": 1.0, "gamma": "scale"} | params))
    elif name == "mlp":
        reg = MLPRegressor(
            **(
                {
                    "hidden_layer_sizes": (64, 64),
                    "alpha": 1e-3,
                    "learning_rate_init": 1e-3,
                    "early_stopping": True,
                    "max_iter": 500,
                    "random_state": random_state,
                }
                | params
            )
        )
    else:
        raise ValueError(f"unknown model {name!r}")
    # Trees do not need scaling but it is harmless; one pipeline keeps the API uniform.
    return Pipeline([("scale", StandardScaler()), ("reg", reg)])


#: Default hyperparameter grids searched by :func:`tune`.
PARAM_GRIDS = {
    "linear": {},
    "ridge": {"reg__alpha": [0.1, 1, 10, 100]},
    "tree": {"reg__max_depth": [4, 6, 8, 12, None], "reg__min_samples_leaf": [5, 20, 50]},
    "forest": {"reg__max_depth": [8, 16, None], "reg__min_samples_leaf": [2, 5, 20]},
    "svr": {"reg__C": [1, 10, 100], "reg__epsilon": [0.5, 2.0], "reg__gamma": ["scale", 0.05]},
    "mlp": {"reg__hidden_layer_sizes": [(32,), (64, 64), (128, 64, 32)], "reg__alpha": [1e-4, 1e-2]},
}


def subsample(X, y, groups, max_rows: int | None, rng=0):
    """Randomly keep whole gaps until ``max_rows`` rows (SVR training is O(n^2)-O(n^3))."""
    if max_rows is None or len(y) <= max_rows:
        return X, y, groups
    rng = np.random.default_rng(rng)
    ug = rng.permutation(np.unique(groups))
    sizes = pd.Series(groups).value_counts()
    keep, total = [], 0
    for g in ug:
        if total >= max_rows:
            break
        keep.append(g)
        total += sizes[g]
    m = np.isin(groups, keep)
    return X[m], y[m], groups[m]


def tune(name: str, X, y, groups, n_splits: int = 5, max_rows: int | None = None, grid: dict | None = None, rng=0):
    """Grouped K-fold grid search for one model.

    Args:
        name: Model name (see :func:`make_model`).
        X, y, groups: Training set from :func:`sensorfill.features.make_training_set`.
        n_splits: Number of folds.
        max_rows: Optional row cap (whole gaps kept) for expensive models.
        grid: Override of :data:`PARAM_GRIDS`.
        rng: Seed for subsampling.

    Returns:
        Fitted :class:`~sklearn.model_selection.GridSearchCV` (``best_estimator_``
        is refit on the full, possibly subsampled, training set).
    """
    X, y, groups = subsample(X, y, groups, max_rows, rng)
    grid = PARAM_GRIDS[name] if grid is None else grid
    gs = GridSearchCV(
        make_model(name),
        grid or {"reg__fit_intercept": [True]},
        cv=GroupKFold(n_splits=n_splits),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    gs.fit(X, y, groups=groups)
    return gs


@dataclass
class MLImputer:
    """Robust linear baseline plus a learned residual correction.

    Attributes:
        model: Fitted regressor (pipeline) predicting ``y - base``.
        builder: Feature builder for the grid being imputed.
        clip: Optional ``(low, high)`` physical limits applied to filled values.
        fit_seconds: Training time recorded by :meth:`fit`.
    """

    model: object
    builder: GapFeatureBuilder
    clip: tuple | None = None
    fit_seconds: float = field(default=0.0)

    def fit(self, X, target):
        """Fit the residual model and record the wall-clock training time."""
        t0 = time.perf_counter()
        self.model.fit(X, target)
        self.fit_seconds = time.perf_counter() - t0
        return self

    def impute(self, y) -> np.ndarray:
        """Return a filled copy of ``y`` (edge gaps fall back to the baseline)."""
        X, idx, y_lin = self.builder.build(y)
        out = y_lin.copy()
        if len(idx):
            out[idx] = y_lin[idx] + self.model.predict(X)
        if self.clip is not None:
            lo, hi = self.clip
            miss = np.isnan(np.asarray(y, dtype=float))
            out[miss] = np.clip(out[miss], lo, hi)
        return out

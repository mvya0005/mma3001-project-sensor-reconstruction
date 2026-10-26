"""Gap-aware feature engineering for the machine-learning imputers.

A missing point is described by *what is known around it*, not by the
missing sensor's other channels: when a sensor drops out, all of its channels
(CO2, temperature, humidity) vanish together, so using them as features would
be unrealistic (information leakage).

Features for a missing point at index ``i`` inside gap ``[a, b)`` of length ``L``
(``yl``/``yr`` are the noise-averaged anchors from
:func:`sensorfill.interpolation.gap_anchors`):

=================  ===========================================================
``delta``          ``yr - yl``: change across the gap
``level``          ``(yl + yr)/2 - level_ref`` (mean level relative to a
                   training reference, e.g. the training median)
``pos``            relative position in the gap ``(i - a + 1) / (L + 1)``
``gap_len``        ``log(1 + L)``
``dist``           ``log(1 + distance to the nearer gap end)``
``tod_sin/cos``    time of day on the unit circle (continuous at midnight)
``dow_sin/cos``    day of week on the unit circle
``weekend``        1 on Saturday/Sunday
``prof``           training typical-day profile at ``i`` minus the straight
                   line between the profile's values at the gap ends
``yday`` / ``tmrw``  same "shape anomaly" taken from the series itself one day
                   before / after (0 if unavailable), with availability flags
                   ``yday_ok`` / ``tmrw_ok``. Offline reconstruction may use
                   future data; a real-time filler would drop ``tmrw``.
``occ``            occupancy proxy (persons) at ``i`` and its 1-h mean
``occ_delta``      ``occ`` minus the mean occupancy at the two gap ends
=================  ===========================================================

The learning target is the **residual** ``y - base`` where ``base`` is the
straight line between the anchors (:func:`sensorfill.interpolation.fill_linear_robust`).
Models therefore learn a *correction* to the best classical baseline;
"no information" defaults to that baseline, and level shifts between training
and test periods do not push tree models outside their training range.

Example:
    >>> import numpy as np, pandas as pd
    >>> from sensorfill.features import GapFeatureBuilder
    >>> ts = pd.date_range("2024-06-03", periods=8, freq="10min")
    >>> y = np.array([400, 410, np.nan, np.nan, 440, 450, 460, 470], dtype=float)
    >>> X, idx, base = GapFeatureBuilder(ts, use_neighbour_days=False, anchor_k=1).build(y)
    >>> idx.tolist(), X["pos"].round(3).tolist(), base[2:4].tolist()
    ([2, 3], [0.333, 0.667], [420.0, 430.0])
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .interpolation import fill_linear_robust, fill_locf, gap_anchors, gap_spans
from .masking import block_mask


class GapFeatureBuilder:
    """Build feature rows for every interior missing point of a series.

    Args:
        ts: Timestamps of the regular grid.
        occ: Optional occupancy on the same grid (persons; NaN treated as 0).
        profile_values: Optional training typical-day profile evaluated on the grid.
        steps_per_day: Grid points per day (144 for 10-min data).
        anchor_k: Observations averaged for each gap anchor.
        level_ref: Reference level subtracted from the anchors (training median).
        use_neighbour_days: Include the ``yday``/``tmrw`` features.

    Attributes:
        calendar: Calendar features for every grid point (time of day, day of
            week, weekend flag), pre-computed once.
        occ: Occupancy with NaN set to 0, or None.
        occ_1h: Centred 1-hour rolling mean of ``occ`` (only when ``occ`` is given).
        profile: Training profile values on the grid, or None.

    Example:
        >>> ts = pd.date_range("2024-06-03", periods=4, freq="10min")
        >>> GapFeatureBuilder(ts, use_neighbour_days=False).feature_names[:5]
        ['delta', 'level', 'pos', 'gap_len', 'dist']
    """

    def __init__(
        self,
        ts,
        occ=None,
        profile_values=None,
        steps_per_day: int = 144,
        anchor_k: int = 3,
        level_ref: float = 0.0,
        use_neighbour_days: bool = True,
    ):
        """Pre-compute the calendar and occupancy features for the whole grid."""
        self.ts = pd.DatetimeIndex(ts)
        self.n = len(self.ts)
        self.spd = steps_per_day
        self.k = anchor_k
        self.level_ref = level_ref
        self.use_days = use_neighbour_days
        tod = (self.ts.hour + self.ts.minute / 60.0).to_numpy() / 24.0
        dow = self.ts.dayofweek.to_numpy() / 7.0
        self.calendar = pd.DataFrame(
            {
                "tod_sin": np.sin(2 * np.pi * tod),
                "tod_cos": np.cos(2 * np.pi * tod),
                "dow_sin": np.sin(2 * np.pi * dow),
                "dow_cos": np.cos(2 * np.pi * dow),
                "weekend": (self.ts.dayofweek >= 5).astype(float),
            }
        )
        self.occ = None
        if occ is not None:
            o = fill_locf(np.nan_to_num(np.asarray(occ, dtype=float), nan=0.0))
            self.occ = o
            self.occ_1h = pd.Series(o).rolling(steps_per_day // 24, min_periods=1, center=True).mean().to_numpy()
        self.profile = None if profile_values is None else np.asarray(profile_values, dtype=float)

    @property
    def feature_names(self) -> list[str]:
        """Column names produced by :meth:`build`.

        Returns:
            The feature names in their fixed column order; which optional
            groups appear depends on the constructor arguments.
        """
        names = ["delta", "level", "pos", "gap_len", "dist"] + list(self.calendar.columns)
        if self.profile is not None:
            names.append("prof")
        if self.use_days:
            names += ["yday", "yday_ok", "tmrw", "tmrw_ok"]
        if self.occ is not None:
            names += ["occ", "occ_1h", "occ_delta"]
        return names

    def _shape_anomaly(self, y, a, b, shift):
        """Shape of ``y`` over the gap window shifted by ``shift`` steps, minus its own end-to-end line.

        Returns ``(anomaly, ok)``; ``ok`` is 0.0 (and the anomaly zeros) when
        the shifted window leaves the series or is more than 20 % missing.
        """
        lo, hi = a - 1 + shift, b + shift
        L = b - a
        if lo < 0 or hi >= y.size:
            return np.zeros(L), 0.0
        seg = y[lo : hi + 1]
        if np.isnan(seg).mean() > 0.2:
            return np.zeros(L), 0.0
        # Isolated dropouts in the neighbouring day are bridged linearly (ends: nearest value).
        seg = pd.Series(seg).interpolate(limit_direction="both").to_numpy()
        line = seg[0] + (seg[-1] - seg[0]) * np.arange(L + 2) / (L + 1)
        return (seg - line)[1:-1], 1.0

    def build(self, y) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Features for all interior missing points of ``y``.

        Args:
            y: Series with NaNs, same length as the timestamp grid.

        Returns:
            ``(X, idx, base)``: feature table, the indices of its rows, and the
            full robust-linear baseline (length n) that the target is relative to.

        Raises:
            ValueError: If ``y`` has the wrong length.

        Example:
            >>> ts = pd.date_range("2024-06-03", periods=5, freq="10min")
            >>> X, idx, _ = GapFeatureBuilder(ts, use_neighbour_days=False).build(
            ...     [1.0, np.nan, 3.0, np.nan, 5.0])
            >>> X.shape, idx.tolist()
            ((2, 10), [1, 3])
        """
        y = np.asarray(y, dtype=float)
        if y.size != self.n:
            raise ValueError("y length does not match the timestamp grid")
        base = fill_linear_robust(y, self.k)
        rows, idx = [], []
        for a, b in gap_spans(np.isnan(y)):
            if a == 0 or b == y.size:
                continue  # edge gaps have no anchor on one side
            L = b - a
            ii = np.arange(a, b)
            yl, yr = gap_anchors(y, a, b, self.k)
            pos = (ii - a + 1) / (L + 1)
            block = {
                "delta": np.full(L, yr - yl),
                "level": np.full(L, 0.5 * (yl + yr) - self.level_ref),
                "pos": pos,
                "gap_len": np.full(L, np.log1p(L)),
                "dist": np.log1p(np.minimum(ii - a + 1, b - ii)),
            }
            if self.profile is not None:
                p = self.profile
                block["prof"] = p[ii] - (p[a - 1] + (p[b] - p[a - 1]) * pos)
            if self.use_days:
                block["yday"], block["yday_ok"] = self._shape_anomaly(y, a, b, -self.spd)
                block["tmrw"], block["tmrw_ok"] = self._shape_anomaly(y, a, b, +self.spd)
                block["yday_ok"] = np.full(L, block["yday_ok"])
                block["tmrw_ok"] = np.full(L, block["tmrw_ok"])
            if self.occ is not None:
                ends = 0.5 * (self.occ[a - 1] + self.occ[b])
                block.update({"occ": self.occ[ii], "occ_1h": self.occ_1h[ii], "occ_delta": self.occ[ii] - ends})
            rows.append(pd.DataFrame(block))
            idx.append(ii)
        if not rows:
            return pd.DataFrame(columns=self.feature_names), np.array([], dtype=int), base
        idx = np.concatenate(idx)
        X = pd.concat(rows, ignore_index=True)
        X = pd.concat([X, self.calendar.iloc[idx].reset_index(drop=True)], axis=1)
        return X[self.feature_names], idx, base


def make_training_set(
    y_true,
    builder: GapFeatureBuilder,
    gap_lengths=(1, 3, 6, 12, 24, 48, 96, 144),
    hidden_fraction: float = 0.2,
    max_gaps_per_length: int = 60,
    n_draws: int = 3,
    rng=None,
):
    """Create supervised examples by hiding observed values of a training series.

    Each (draw, gap length) pair is a separate masking of the *whole* training
    series, hiding about ``hidden_fraction`` of it. Masking one length at a
    time keeps the surrounding context (neighbouring days, anchors) as complete
    as it is at prediction time; packing many long gaps into one mask would
    make features such as ``yday`` unrealistically unavailable.

    Args:
        y_true: Training series (real NaNs are never targets).
        builder: Feature builder for the same grid.
        gap_lengths: Artificial gap lengths (steps) to sample.
        hidden_fraction: Target fraction of points hidden per mask.
        max_gaps_per_length: Cap on gaps per mask (short gaps would otherwise
            dominate the row count).
        n_draws: Independent mask draws per length (data augmentation).
        rng: Seed or generator.

    Returns:
        ``(X, target, groups)`` where ``target = y_true - base`` at hidden points
        and ``groups`` identifies the gap each row came from, so that
        cross-validation keeps a gap's (highly correlated) rows together.

    Raises:
        ValueError: If no gap of any requested length fits in ``y_true``.

    Example:
        >>> ts = pd.date_range("2024-06-03", periods=600, freq="10min")
        >>> y = 500 + 50 * np.sin(np.arange(600) / 20)
        >>> b = GapFeatureBuilder(ts, use_neighbour_days=False)
        >>> X, target, groups = make_training_set(y, b, gap_lengths=(3, 6), n_draws=1, rng=0)
        >>> len(X) == len(target) == len(groups)
        True
    """
    rng = np.random.default_rng(rng)
    y_true = np.asarray(y_true, dtype=float)
    observed = ~np.isnan(y_true)
    Xs, targets, gs = [], [], []
    gid = 0
    for _ in range(n_draws):
        for L in gap_lengths:
            k = int(min(max_gaps_per_length, max(1, hidden_fraction * y_true.size / (L + 4))))
            mask = None
            # If k gaps do not fit (long gaps on a short series), halve k until they do.
            while k > 0:
                try:
                    mask = block_mask(observed, L, k, rng, margin=2, min_observed=0.8)
                    break
                except ValueError:
                    k //= 2
            if mask is None:
                continue
            y = y_true.copy()
            y[mask] = np.nan
            X, idx, base = builder.build(y)
            keep = mask[idx]  # only artificially hidden points have ground truth
            X, idx = X[keep], idx[keep]
            Xs.append(X)
            targets.append(y_true[idx] - base[idx])
            # Group id = gap id: a jump of more than 2 steps starts a new gap.
            g = gid + np.cumsum(np.concatenate([[True], np.diff(idx) > 2]))
            gs.append(g)
            gid = int(g.max()) + 1 if g.size else gid
    if not Xs:
        raise ValueError("no artificial gap of the requested lengths fits in the training series")
    return pd.concat(Xs, ignore_index=True), np.concatenate(targets), np.concatenate(gs)

"""Artificial gap generation for validation against withheld ground truth.

Real gaps have no ground truth, so every method is scored by *hiding* values
that were actually observed, reconstructing them, and comparing with the hidden
truth. The masks produced here only ever cover observed points, never overlap,
and leave at least one observed neighbour on each side of a gap so that
boundary-value methods (interpolation, relaxation, shooting) are well posed.
"""

from __future__ import annotations

import numpy as np

from .data import gap_lengths


def block_mask(
    observed: np.ndarray,
    gap_len: int,
    n_gaps: int,
    rng: np.random.Generator | int | None = None,
    margin: int = 1,
    max_tries: int = 100_000,
    min_observed: float = 1.0,
) -> np.ndarray:
    """Choose non-overlapping blocks of observed points to hide.

    Args:
        observed: Boolean array, True where a real measurement exists.
        gap_len: Length of each artificial gap in grid steps (>= 1).
        n_gaps: Number of gaps to place.
        rng: Seed or generator for reproducibility.
        margin: Number of observed points required on each side of a block
            (they are kept visible as boundary conditions).
        max_tries: Give up after this many random draws.
        min_observed: Minimum fraction of the block that must be observed.
            1.0 requires a fully observed block. Lower values let a block
            straddle isolated real dropouts (common in this dataset, where a
            reading is skipped every ~10 samples); real NaNs inside the block
            stay NaN and are simply not scored.

    Returns:
        Boolean array, True at points that are hidden. ``mask & ~observed`` is
        always empty, and the ``margin`` points either side of each block are
        observed.

    Raises:
        ValueError: If ``gap_len < 1`` or fewer than ``n_gaps`` blocks fit.
    """
    if gap_len < 1:
        raise ValueError("gap_len must be >= 1")
    rng = np.random.default_rng(rng)
    observed = np.asarray(observed, dtype=bool)
    n = observed.size
    w = gap_len + 2 * margin
    if n < w:
        raise ValueError("Series shorter than one gap plus margins")
    # A window start is valid when its margins are observed and the block is
    # observed enough (prefix sums make each test O(1)).
    c = np.concatenate([[0], np.cumsum(observed)])
    starts = np.arange(n - w + 1)
    inner = c[starts + margin + gap_len] - c[starts + margin]
    left_ok = (c[starts + margin] - c[starts]) == margin
    right_ok = (c[starts + w] - c[starts + margin + gap_len]) == margin
    valid = left_ok & right_ok & (inner >= np.ceil(min_observed * gap_len)) & (inner > 0)
    candidates = np.flatnonzero(valid)
    mask = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    placed, tries = 0, 0
    order = rng.permutation(candidates)
    for s in order:
        tries += 1
        if placed == n_gaps or tries > max_tries:
            break
        if blocked[s : s + w].any():
            continue
        mask[s + margin : s + margin + gap_len] = observed[s + margin : s + margin + gap_len]
        blocked[s : s + w] = True
        placed += 1
    if placed < n_gaps:
        raise ValueError(f"Only {placed} of {n_gaps} gaps of length {gap_len} fit in the series")
    return mask


def random_point_mask(observed: np.ndarray, frac: float, rng=None) -> np.ndarray:
    """Hide a random fraction of isolated observed points (missing-completely-at-random).

    Args:
        observed: Boolean array of real measurements.
        frac: Fraction of observed points to hide, in ``(0, 1)``.
        rng: Seed or generator.

    Returns:
        Boolean hide mask.
    """
    if not 0 < frac < 1:
        raise ValueError("frac must be in (0, 1)")
    rng = np.random.default_rng(rng)
    idx = np.flatnonzero(observed)
    k = int(round(frac * idx.size))
    mask = np.zeros(observed.size, dtype=bool)
    mask[rng.choice(idx, size=k, replace=False)] = True
    return mask


def empirical_mask(observed: np.ndarray, real_gaps: np.ndarray, n_gaps: int, rng=None, cap: int | None = None) -> np.ndarray:
    """Hide blocks whose lengths are resampled from the sensor's real gap lengths.

    This reproduces the *realistic* mix of short dropouts and long outages
    rather than a single fixed length.

    Args:
        observed: Boolean array of real measurements.
        real_gaps: Real gap lengths (e.g. from :func:`sensorfill.data.gap_lengths`).
        n_gaps: Number of gaps to place.
        rng: Seed or generator.
        cap: Optional maximum gap length to draw.

    Returns:
        Boolean hide mask.
    """
    rng = np.random.default_rng(rng)
    real_gaps = np.asarray(real_gaps)
    if cap is not None:
        real_gaps = real_gaps[real_gaps <= cap]
    lengths = rng.choice(real_gaps, size=n_gaps, replace=True)
    mask = np.zeros(observed.size, dtype=bool)
    avail = observed.copy()
    for L in sorted(lengths, reverse=True):  # place long gaps first so they fit
        m = block_mask(avail, int(L), 1, rng)
        mask |= m
        # Keep a one-point buffer so gaps never touch.
        grown = m | np.roll(m, 1) | np.roll(m, -1)
        avail &= ~grown
    return mask


def apply_mask(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a copy of ``y`` with ``mask`` positions set to NaN."""
    out = np.array(y, dtype=float, copy=True)
    out[mask] = np.nan
    return out


def describe_mask(mask: np.ndarray) -> dict:
    """Small summary of a hide mask (number of gaps, total and mean length)."""
    g = gap_lengths(mask)
    return {"n_gaps": int(g.size), "n_hidden": int(mask.sum()), "mean_len": float(g.mean()) if g.size else 0.0}

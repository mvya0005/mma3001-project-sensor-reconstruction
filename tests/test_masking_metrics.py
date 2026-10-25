"""Tests for artificial gap masks and error metrics."""

import numpy as np
import pytest

from sensorfill import masking, metrics
from sensorfill.data import gap_lengths


def test_block_mask_only_hides_observed_with_margins(rng):
    obs = rng.random(2000) > 0.05
    m = masking.block_mask(obs, 12, 20, rng, margin=2)
    assert not (m & ~obs).any()
    assert masking.describe_mask(m)["n_gaps"] == 20
    for i in np.flatnonzero(np.diff(m.astype(int)) == 1):  # block starts at i+1
        assert obs[i] and obs[i - 1]


def test_block_mask_min_observed_allows_real_dropouts():
    obs = np.ones(200, dtype=bool)
    obs[::10] = False  # a reading skipped every 10 samples, as in the real data
    with pytest.raises(ValueError):
        masking.block_mask(obs, 30, 1, 0, min_observed=1.0)
    m = masking.block_mask(obs, 30, 1, 0, min_observed=0.8)
    assert 25 <= m.sum() <= 30 and not (m & ~obs).any()


def test_block_mask_validation():
    with pytest.raises(ValueError):
        masking.block_mask(np.ones(10, bool), 0, 1)
    with pytest.raises(ValueError):
        masking.block_mask(np.ones(10, bool), 5, 5)


def test_random_point_and_empirical_masks(rng):
    obs = np.ones(5000, bool)
    m = masking.random_point_mask(obs, 0.1, rng)
    assert m.sum() == 500
    e = masking.empirical_mask(obs, np.array([1, 1, 2, 10]), 30, rng)
    assert set(gap_lengths(e)) <= {1, 2, 10}


def test_metrics_known_values():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.0, 3.0, 3.0, 2.0])
    m = np.array([False, True, True, True])
    assert metrics.mae(t, p, m) == pytest.approx(1.0)
    assert metrics.rmse(t, p, m) == pytest.approx(np.sqrt(5 / 3))
    assert metrics.bias(t, p, m) == pytest.approx(-1 / 3)
    assert metrics.max_error(t, p, m) == 2.0
    assert metrics.r2(t, t, m) == 1.0
    assert metrics.physical_violations(p, m, low=2.5) == 1


def test_metrics_reject_unfilled_points():
    with pytest.raises(ValueError):
        metrics.rmse([1.0, 2.0], [1.0, np.nan])

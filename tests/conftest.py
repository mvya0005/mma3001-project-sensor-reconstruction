"""Shared fixtures: small synthetic signals with known ground truth."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def daily_signal(rng):
    """Two weeks of a synthetic 10-min CO2-like signal: daily cycle on workdays plus noise.

    Returns ``(ts, clean, noisy, occ)``.
    """
    ts = pd.date_range("2024-03-04", periods=14 * 144, freq="10min")
    hours = np.asarray(ts.hour + ts.minute / 60, dtype=float)
    workday = np.asarray(ts.dayofweek < 5)
    occ = np.where(workday & (hours >= 9) & (hours < 17), 10.0, 0.0)
    clean = 430 + 80 * np.clip(np.sin(np.pi * (hours - 8) / 10), 0, None) * workday
    noisy = clean + rng.normal(0, 3, clean.size)
    return ts, clean, noisy, occ

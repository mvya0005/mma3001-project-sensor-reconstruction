"""Smoke and correctness tests for the PyTorch models (small sizes so they run on CPU)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sensorfill import deep  # noqa: E402


def test_rk4_rollout_matches_exact_decay():
    class Decay(torch.nn.Module):
        def forward(self, c, u):
            return -c

    c0 = torch.ones(1)
    U = torch.zeros(1, 11, 3)
    out = deep.rk4_rollout(Decay(), c0, U, 0.1)[0].numpy()
    np.testing.assert_allclose(out, np.exp(-0.1 * np.arange(11)), rtol=1e-6)


def test_bigru_trains_and_fills(daily_signal):
    ts, clean, noisy, occ = daily_signal
    m = deep.BiGRUImputer(window=144, hidden=8, layers=1, device="cpu", seed=0)
    log = m.fit(noisy[:1440], ts[:1440], occ[:1440], noisy[1440:], ts[1440:], occ[1440:],
                epochs=2, steps_per_epoch=3, batch=4, gap_choices=(6, 12))
    assert len(log.train_loss) == 2 and np.isfinite(log.train_loss).all()
    y = noisy.copy()
    y[1500:1520] = np.nan
    out = m.impute(y, ts, occ)
    assert not np.isnan(out).any()
    np.testing.assert_array_equal(out[~np.isnan(y)], y[~np.isnan(y)])
    assert m.n_parameters() > 0 and m.flops_per_window() > 0


def test_physics_informed_parameters_stay_positive(daily_signal):
    ts, clean, noisy, occ = daily_signal
    m = deep.BiGRUImputer(window=144, hidden=8, layers=1, physics_weight=1.0, device="cpu")
    m.fit(noisy, ts, occ, epochs=1, steps_per_epoch=2, batch=4, gap_choices=(6,))
    p = m.physics_params
    assert p["g"] > 0 and p["lam"] > 0


def test_neural_ode_fill(daily_signal):
    ts, clean, noisy, occ = daily_signal
    m = deep.NeuralODE(horizon=12, hidden=8, device="cpu")
    m.fit(noisy, ts, occ, epochs=1, steps_per_epoch=2, batch=8)
    y = noisy.copy()
    y[500:530] = np.nan
    out = m.impute(y, ts, occ)
    assert not np.isnan(out).any()
    # End correction makes the fill meet the right boundary value.
    assert abs(out[529] - y[530]) < abs(y[530] - y[499]) + 50


def test_impute_before_fit_raises(daily_signal):
    ts, _, noisy, _ = daily_signal
    with pytest.raises(RuntimeError):
        deep.BiGRUImputer(window=144, hidden=4, layers=1).impute(noisy, ts)

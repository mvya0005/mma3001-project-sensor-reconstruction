"""Tests for the feature builder, ML imputers, benchmark harness and profiling helpers."""

import numpy as np
import pandas as pd
import pytest

from sensorfill import benchmark as bm
from sensorfill import features as ft
from sensorfill import interpolation as itp
from sensorfill import ml, profiling


def test_feature_builder_shapes_and_no_edge_rows(daily_signal):
    """Features cover exactly the interior gap points, in the documented column order, with no NaN."""
    ts, clean, noisy, occ = daily_signal
    y = noisy.copy()
    y[:5] = np.nan  # edge gap: excluded
    y[300:320] = np.nan
    b = ft.GapFeatureBuilder(ts, occ)
    X, idx, base = b.build(y)
    assert list(X.columns) == b.feature_names
    assert idx.tolist() == list(range(300, 320))
    assert not X.isna().any().any()
    assert X["yday_ok"].iloc[0] == 1.0  # the previous day is observed


def test_feature_builder_length_check(daily_signal):
    """A series that does not match the timestamp grid is rejected."""
    ts, *_ = daily_signal
    with pytest.raises(ValueError):
        ft.GapFeatureBuilder(ts).build(np.ones(5))


def test_training_set_targets_are_residuals(daily_signal):
    """Training rows, targets and gap groups line up and span many gaps."""
    ts, clean, noisy, occ = daily_signal
    b = ft.GapFeatureBuilder(ts, occ)
    X, r, g = ft.make_training_set(noisy, b, gap_lengths=(3, 24), n_draws=1, rng=0)
    assert len(X) == len(r) == len(g) and len(np.unique(g)) > 5


@pytest.mark.parametrize("name", ["linear", "ridge", "tree", "forest", "svr", "mlp"])
def test_every_model_beats_linear_on_periodic_signal(daily_signal, name):
    """Each regression model learns the daily shape and beats linear interpolation on a long gap."""
    ts, clean, noisy, occ = daily_signal
    n_tr = 10 * 144
    b_tr = ft.GapFeatureBuilder(ts[:n_tr], occ[:n_tr])
    X, r, g = ft.make_training_set(noisy[:n_tr], b_tr, gap_lengths=(36, 72), n_draws=2, rng=0)
    model = ml.make_model(name, max_iter=300) if name == "mlp" else ml.make_model(name)
    imp = ml.MLImputer(model, ft.GapFeatureBuilder(ts[n_tr:], occ[n_tr:])).fit(X, r)
    y = clean[n_tr:].copy()
    y[60:120] = np.nan  # Thursday 10:00-20:00, shape not linear
    gap = np.isnan(y)
    e_ml = np.sqrt(np.mean((imp.impute(y)[gap] - clean[n_tr:][gap]) ** 2))
    e_lin = np.sqrt(np.mean((itp.fill_linear(y)[gap] - clean[n_tr:][gap]) ** 2))
    assert e_ml < e_lin
    assert imp.fit_seconds > 0


def test_unknown_model():
    """An unknown model name is rejected."""
    with pytest.raises(ValueError):
        ml.make_model("xgboost")


def test_tune_grouped_cv(daily_signal):
    """Grouped cross-validation runs and picks a value from the grid."""
    ts, clean, noisy, occ = daily_signal
    b = ft.GapFeatureBuilder(ts, occ)
    X, r, g = ft.make_training_set(noisy, b, gap_lengths=(12, 36), n_draws=1, rng=0)
    gs = ml.tune("tree", X, r, g, n_splits=3, grid={"reg__max_depth": [2, 6]})
    assert gs.best_params_["reg__max_depth"] in (2, 6)


def test_benchmark_runs_all_classical_methods(daily_signal):
    """Every classical method runs without error inside the shared benchmark harness."""
    ts, clean, noisy, occ = daily_signal
    ctx = bm.Context(ts=ts, occ=occ, profile=itp.daily_profile(noisy, ts))
    res = bm.run_benchmark(noisy, ctx, bm.classical_methods(), gap_lengths=(1, 12), n_gaps=5, repeats=1)
    assert res["error"].isna().all(), res.loc[res["error"].notna(), ["method", "error"]]
    assert set(res["method"]) == set(bm.classical_methods())
    s = bm.summarise(res)
    assert s.shape == (len(bm.classical_methods()), 2)


def test_context_midnight_offset():
    """The first midnight index is found from the timestamps."""
    ts = pd.date_range("2024-01-01 22:00", periods=50, freq="10min")
    assert bm.Context(ts=ts).midnight_offset == 12


def test_profiling_helpers():
    """Timing, profiling, FLOP and roofline helpers return sensible values."""
    m = profiling.measure(np.sort, np.random.default_rng(0).random(10_000), repeats=2)
    assert m.median_s > 0 and m.peak_mb > 0
    assert "function calls" in profiling.profile_call(np.sort, np.arange(10))
    est = profiling.flop_estimates(1000, 10_000, 36)
    assert est["svd"]["flops"] > est["linear"]["flops"]
    np.testing.assert_allclose(profiling.roofline([0.1, 1000], 100, 10), [1.0, 100.0])


def test_hybrid_method_dispatches_by_gap_length():
    """Short gaps go to the short-gap method and long gaps to the long-gap method."""
    y = np.r_[np.zeros(5), [np.nan], np.zeros(5), [np.nan] * 4, np.zeros(5)]
    short = lambda yy, c: np.where(np.isnan(yy), 1.0, yy)
    long = lambda yy, c: np.where(np.isnan(yy), 2.0, yy)
    out = bm.hybrid_method(short, long, max_short=2)(y, None)
    assert out[5] == 1.0 and (out[11:15] == 2.0).all()


def test_confidence_table():
    """Mean and t-interval across repeats are computed per method and gap length."""
    res = pd.DataFrame({"method": ["a"] * 3, "gap_len": [1] * 3, "rmse": [1.0, 2.0, 3.0]})
    ci = bm.confidence_table(res)
    assert ci.loc[("a", 1), "mean"] == 2.0 and ci.loc[("a", 1), "ci95"] > 0

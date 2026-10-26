"""Tests for loading, regularising and auditing (sensorfill.data)."""

import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from sensorfill import data


def _env_row(sid, t, co2, temp=21.0):
    """One raw export row with valid CO2/temperature, an error-code record and an empty variable."""
    js = [
        {"values": [{"time": t, "value": co2}], "variable": {"name": "Carbon dioxide", "unit": "ppm"}},
        {"values": [{"time": t, "value": temp}], "variable": {"name": "Temperature", "unit": "°C"}},
        {"values": [{"time": t, "errorCode": 8}], "variable": {"name": "Humidity", "unit": "HR%"}},
        {"values": [], "variable": {"name": "Pressure", "unit": "mb"}},
    ]
    return {"sensorid": sid, "jsondata": json.dumps(js)}


@pytest.fixture
def env_zip(tmp_path):
    """A tiny zipped export with a real gap, a duplicate upload and an implausible reading."""
    t0 = 1_709_200_800  # 2024-02-29 21:00 local (AEDT)
    rows = [_env_row("1", t0 + 600 * i, 400 + i) for i in range(6) if i != 3]
    rows.append(rows[0])  # exact duplicate upload
    rows.append(_env_row("1", t0 + 600 * 4, 9999))  # implausible CO2 at same time as a valid reading
    df = pd.DataFrame(rows)
    csv = tmp_path / "env.csv"
    df.to_csv(csv, index=False)
    z = tmp_path / "env.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.write(csv, "env.csv")
    return z


def test_parse_skips_error_codes_and_empty_lists():
    """Records with an error code or no values carry no measurement and must be skipped."""
    s = _env_row("1", 10, 500)["jsondata"]
    out = data.parse_env_json(s)
    assert [o[0] for o in out] == ["Carbon dioxide", "Temperature"]
    assert out[0][2:] == (10, 500.0)


def test_load_env_long_reads_zip_and_deduplicates(env_zip):
    """Zipped CSV is read, duplicate uploads collapse to one row, and UTC converts to Melbourne time."""
    long = data.load_env_long(env_zip)
    co2 = long[long.variable == "co2"]
    assert co2["time"].is_unique  # duplicate upload collapsed
    assert set(long["variable"]) == {"co2", "temp"}
    # Local time conversion: 2024-02-29 10:00 UTC is 21:00 AEDT (UTC+11).
    assert co2["ts"].iloc[0] == pd.Timestamp("2024-02-29 21:00")


def test_env_to_grid_keeps_missing_as_nan_and_rejects_implausible(env_zip):
    """Real gaps stay NaN on the grid and out-of-range readings are rejected, not clipped."""
    long = data.load_env_long(env_zip)
    grid = data.env_to_grid(long, "1", variables=("co2", "temp"))
    assert len(grid) == 6 and grid.index.freq == "10min"
    assert np.isnan(grid["co2"].iloc[3])  # real gap stays NaN
    # Mean of 404 and 9999 is implausible and must be rejected, not clipped.
    assert np.isnan(grid["co2"].iloc[4])


def test_env_to_grid_unknown_sensor(env_zip):
    """Asking for a sensor that is not in the data is an error, not an empty table."""
    with pytest.raises(ValueError):
        data.env_to_grid(data.load_env_long(env_zip), "nope")


def test_missing_file_raises(tmp_path):
    """A missing raw file gives a clear FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data.load_env_long(tmp_path / "missing.zip")


def test_gap_lengths_and_audit():
    """Gap run lengths and missing fraction match a hand-counted example."""
    s = pd.Series([1, np.nan, np.nan, 2, np.nan, 3, np.nan])
    assert data.gap_lengths(s.isna().to_numpy()).tolist() == [2, 1, 1]
    st = data.audit_series(s)
    assert st.n_missing == 4 and st.frac_missing == pytest.approx(4 / 7)


def test_longest_observed_window_excludes_long_outage():
    """The study window stops at an outage longer than the tolerance and starts/ends on data."""
    y = np.r_[np.ones(10), np.full(50, np.nan), np.ones(30), [np.nan], np.ones(5)]
    w = data.longest_observed_window(pd.Series(y), max_gap=10)
    assert len(w) == 36 and w.notna().iloc[0] and w.notna().iloc[-1]


def test_time_weighted_step_mean_exact():
    """Bin means of the step-held headcount equal the exact integral divided by the bin width."""
    t = np.array([0.0, 300.0, 900.0])
    v = np.array([2.0, 4.0, 0.0])
    m = data.time_weighted_step_mean(t, v, np.array([-600, 0, 600, 1200, 1800.0]), max_hold=1e9)
    assert np.isnan(m[0])  # before the first event
    np.testing.assert_allclose(m[1:], [3.0, 2.0, 0.0])


def test_time_weighted_step_mean_marks_outage():
    """Silence longer than max_hold is treated as an outage (NaN), not as an empty room."""
    m = data.time_weighted_step_mean(np.array([0.0, 10_000.0]), np.array([1.0, 1.0]), np.array([0, 600, 6000, 6600.0]), max_hold=3600)
    assert not np.isnan(m[0]) and np.isnan(m[2])


def test_occupancy_to_grid(tmp_path):
    """Events are converted to time-weighted 10-min headcounts, with columns named by space id."""
    ev = pd.DataFrame(
        {
            "floorspaceid": pd.Categorical(["aaaaaaaa-1"] * 3),
            "status": pd.Categorical(["CurrentlyOccupied"] * 3),
            "headcount": np.array([2, 4, 0], dtype="float32"),
            "ts": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:05", "2024-01-01 00:15"]),
        }
    )
    g = data.occupancy_to_grid(ev)
    assert list(g.columns) == ["aaaaaaaa"]
    np.testing.assert_allclose(g.iloc[:2, 0], [3.0, 2.0])

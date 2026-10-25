"""Loading, auditing and regularising the Monash Smart Infrastructure sensor data.

This module turns the two raw exports supplied with Dataset 2 into tidy,
regularly sampled time series that every reconstruction method in the
package consumes.

Raw inputs
----------
* **Environmental export** (``5EnvSensor_*.csv`` or the ``.zip`` holding it).
  One row per upload. Column ``jsondata`` holds a JSON list of variables,
  each with a ``variable`` block (name, unit) and a ``values`` list of
  ``{"time": <unix seconds>, "value": <float>}`` records. Some records carry
  an ``errorCode`` instead of a value, and many uploads are exact duplicates.
* **Occupancy export** (``5occupancySensor_*.csv`` or ``.zip``). One row per
  occupancy event with ``floorspaceid``, ``occupancystatus``, ``headcount``
  (persons) and ``collecteddate`` (local time with UTC offset).

Outputs
-------
* :func:`load_env_long` -> long table ``[sensorid, time, ts, variable, unit, value]``.
* :func:`env_to_grid` -> wide table on a regular grid (default 10 min) where a
  missing reading is ``NaN``. Nothing is filled here: the gaps are what the
  rest of the project reconstructs.
* :func:`load_occupancy_grid` -> mean headcount per floor space per grid step.
* :func:`audit_series` -> data-quality summary used in the report.

All timestamps are converted to the building's local time zone
(``Australia/Melbourne``) and then made timezone-naive, so that hour-of-day
features follow the building's daily schedule across daylight-saving changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

#: Time zone of the Monash Clayton campus buildings.
LOCAL_TZ = "Australia/Melbourne"

#: Short column names used throughout the package, keyed by the raw variable name.
VARIABLE_ALIASES = {
    "Carbon dioxide": "co2",
    "Temperature": "temp",
    "Humidity": "rh",
    "Pressure": "pressure",
    "Light Volatile Organic Compounds": "voc",
    "Formaldehyde": "hcho",
    "Particulate matter 1": "pm1",
    "Particulate matter 2.5": "pm25",
    "Particulate matter 4": "pm4",
    "Particulate matter 10": "pm10",
    "Battery": "battery",
}

#: Units of each short variable name (taken from the raw export metadata).
UNITS = {
    "co2": "ppm",
    "temp": "degC",
    "rh": "%RH",
    "pressure": "mbar",
    "voc": "ppb",
    "hcho": "ug/m3",
    "pm1": "ug/m3",
    "pm25": "ug/m3",
    "pm4": "ug/m3",
    "pm10": "ug/m3",
    "battery": "V",
}

#: Physically plausible ranges for an indoor sensor. Readings outside these
#: limits are treated as invalid (set to NaN and counted by the audit), not clipped.
PLAUSIBLE_RANGE = {
    "co2": (350.0, 5000.0),  # below ~ambient outdoor CO2 is a sensor fault
    "temp": (0.0, 45.0),
    "rh": (0.0, 100.0),
    "pressure": (900.0, 1100.0),
    "battery": (2.5, 5.0),
}


def _read_csv_any(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a CSV directly or from a single-file ``.zip`` archive.

    Args:
        path: Path to a ``.csv`` file or a ``.zip`` containing exactly one CSV.
        **kwargs: Passed through to :func:`pandas.read_csv`.

    Returns:
        The parsed table.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    # pandas infers zip compression from the extension.
    return pd.read_csv(path, **kwargs)


def parse_env_json(jsondata: str) -> list[tuple[str, str, int, float]]:
    """Parse one ``jsondata`` cell of the environmental export.

    Records without a numeric ``value`` (e.g. ``{"time": ..., "errorCode": 8}``)
    are skipped, because they carry no measurement.

    Args:
        jsondata: JSON string holding a list of variable blocks.

    Returns:
        A list of ``(variable_name, unit, unix_time, value)`` tuples.

    Example:
        >>> s = '[{"values": [{"time": 1, "value": 400}], "variable": {"name": "Carbon dioxide", "unit": "ppm"}}]'
        >>> parse_env_json(s)
        [('Carbon dioxide', 'ppm', 1, 400.0)]
    """
    out = []
    for block in json.loads(jsondata):
        var = block.get("variable", {})
        name, unit = var.get("name"), var.get("unit")
        for rec in block.get("values", []):
            if "value" not in rec or rec["value"] is None:
                continue
            out.append((name, unit, int(rec["time"]), float(rec["value"])))
    return out


def load_env_long(path: str | Path, sensors: Iterable[str] | None = None) -> pd.DataFrame:
    """Load the environmental export into a de-duplicated long table.

    Args:
        path: ``.csv`` or ``.zip`` of the environmental export.
        sensors: Optional sensor ids to keep (as strings). ``None`` keeps all.

    Returns:
        DataFrame with columns ``sensorid`` (str), ``time`` (unix s, int),
        ``ts`` (naive local datetime), ``variable`` (short alias), ``unit`` and
        ``value`` (float). Exact duplicate uploads are collapsed; if the same
        sensor, time and variable appear with different values the mean is kept.
    """
    raw = _read_csv_any(path, usecols=["sensorid", "jsondata"], dtype={"sensorid": str})
    if sensors is not None:
        raw = raw[raw["sensorid"].isin(set(map(str, sensors)))]
    records = []
    for sid, js in zip(raw["sensorid"], raw["jsondata"]):
        for name, unit, t, v in parse_env_json(js):
            records.append((sid, t, VARIABLE_ALIASES.get(name, name), unit, v))
    long = pd.DataFrame(records, columns=["sensorid", "time", "variable", "unit", "value"])
    # The export repeats the same upload many times; keep one value per key.
    long = long.groupby(["sensorid", "time", "variable", "unit"], as_index=False)["value"].mean()
    long["ts"] = (
        pd.to_datetime(long["time"], unit="s", utc=True).dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    )
    return long[["sensorid", "time", "ts", "variable", "unit", "value"]]


def apply_plausibility(df: pd.DataFrame, ranges: dict | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Replace physically implausible readings with NaN.

    Args:
        df: Wide table whose columns include short variable names.
        ranges: ``{variable: (low, high)}``; defaults to :data:`PLAUSIBLE_RANGE`.

    Returns:
        ``(cleaned, n_rejected)`` where ``n_rejected`` counts rejected readings
        per variable.
    """
    ranges = PLAUSIBLE_RANGE if ranges is None else ranges
    out = df.copy()
    rejected = {}
    for col, (lo, hi) in ranges.items():
        if col not in out:
            continue
        bad = (out[col] < lo) | (out[col] > hi)
        rejected[col] = int(bad.sum())
        out.loc[bad, col] = np.nan
    return out, pd.Series(rejected, dtype=int)


def env_to_grid(
    long: pd.DataFrame,
    sensorid: str,
    freq: str = "10min",
    variables: Iterable[str] = ("co2", "temp", "rh", "pressure"),
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    clean: bool = True,
) -> pd.DataFrame:
    """Snap one sensor's readings onto a regular time grid.

    Each reading is assigned to the nearest grid point; if two readings land on
    the same point their mean is used. Grid points with no reading stay NaN.

    Args:
        long: Output of :func:`load_env_long`.
        sensorid: Sensor to extract.
        freq: Grid spacing (pandas offset alias). The sensors report every
            600 s, so ``"10min"`` keeps the native resolution.
        variables: Short variable names to keep.
        start, end: Optional clipping of the grid (inclusive).
        clean: If True, apply :func:`apply_plausibility`.

    Returns:
        DataFrame indexed by a regular ``DatetimeIndex`` named ``ts`` with one
        column per variable.

    Raises:
        ValueError: If the sensor has no readings in ``long``.
    """
    sub = long[(long["sensorid"] == str(sensorid)) & (long["variable"].isin(list(variables)))]
    if sub.empty:
        raise ValueError(f"No readings for sensor {sensorid!r} and variables {list(variables)}")
    sub = sub.assign(ts=sub["ts"].dt.round(freq))
    wide = sub.pivot_table(index="ts", columns="variable", values="value", aggfunc="mean")
    start = wide.index.min() if start is None else pd.Timestamp(start)
    end = wide.index.max() if end is None else pd.Timestamp(end)
    grid = pd.date_range(start.floor(freq), end.ceil(freq), freq=freq, name="ts")
    wide = wide.reindex(grid)[[v for v in variables if v in wide.columns]]
    if clean:
        wide, _ = apply_plausibility(wide)
    return wide


def load_occupancy_events(path: str | Path) -> pd.DataFrame:
    """Load the occupancy event export with memory-efficient dtypes.

    The full export has ~9 million rows; categorical ids keep it under ~1 GB.

    Args:
        path: ``.csv`` or ``.zip`` of the occupancy export.

    Returns:
        DataFrame with ``floorspaceid`` (category), ``status`` (category),
        ``headcount`` (float32, persons) and ``ts`` (naive local datetime).
    """
    occ = _read_csv_any(
        path,
        usecols=["floorspaceid", "occupancystatus", "headcount", "collecteddate"],
        dtype={"floorspaceid": "category", "occupancystatus": "category", "headcount": "float32"},
    )
    # Stamps look like "2023-11-24 00:36:52+11"; the first 19 chars are local wall-clock time.
    occ["ts"] = pd.to_datetime(occ["collecteddate"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S")
    occ = occ.drop(columns="collecteddate").rename(columns={"occupancystatus": "status"})
    return occ


def time_weighted_step_mean(t_events: np.ndarray, values: np.ndarray, edges: np.ndarray, max_hold: float) -> np.ndarray:
    """Exact bin averages of a zero-order-hold (step) signal.

    The occupancy sensor reports *changes*: after an event the headcount stays
    at its value until the next event. The bin mean is therefore the exact
    integral of a step function divided by the bin width. With the running
    integral ``F(t) = int_{t_0}^{t} N(s) ds`` (piecewise linear, exact), the
    mean over ``[e_j, e_{j+1})`` is ``(F(e_{j+1}) - F(e_j)) / (e_{j+1} - e_j)``.

    Args:
        t_events: Event times in seconds, strictly increasing.
        values: Value held from each event until the next.
        edges: Bin edges in seconds.
        max_hold: Longest time a value may be held [s]; silence beyond this is
            treated as a sensor outage and the affected bins become NaN.

    Returns:
        Array of ``len(edges) - 1`` bin means (NaN before the first event or
        inside an outage).
    """
    t = np.asarray(t_events, dtype=float)
    v = np.asarray(values, dtype=float)
    dt = np.diff(t)
    # Running integral at each event time.
    F_ev = np.concatenate([[0.0], np.cumsum(v[:-1] * dt)])
    k = np.searchsorted(t, edges, side="right") - 1  # last event at or before each edge
    kc = np.clip(k, 0, len(t) - 1)
    F = F_ev[kc] + v[kc] * (edges - t[kc])
    means = np.diff(F) / np.diff(edges)
    # Invalidate bins before the first event, or whose edges are further than
    # max_hold from the last event (outage, including after the final event).
    age = edges - t[kc]
    bad = (k[:-1] < 0) | (k[1:] < 0) | (age[:-1] > max_hold) | (age[1:] > max_hold)
    means[bad] = np.nan
    return means


def occupancy_to_grid(
    events: pd.DataFrame, freq: str = "10min", max_headcount: float = 200.0, max_hold_hours: float = 72.0
) -> pd.DataFrame:
    """Time-weighted mean headcount per floor space on a regular grid.

    Args:
        events: Output of :func:`load_occupancy_events`.
        freq: Grid spacing; must match the environmental grid. Grid labels are
            the *start* of each bin.
        max_headcount: Headcounts above this are treated as sensor faults.
        max_hold_hours: Silence longer than this is an outage (NaN), not an
            empty room. The audit showed that >97 % of silences longer than one
            hour start from a ``NotOccupied`` event, so holding the last value
            is the correct physical interpretation of the event stream.

    Returns:
        Wide DataFrame (index ``ts``, one column per floor space, named by the
        first 8 characters of its id) of mean headcount in persons.
    """
    ev = events[(events["headcount"] >= 0) & (events["headcount"] <= max_headcount)]
    start, end = ev["ts"].min().floor(freq), ev["ts"].max().ceil(freq)
    grid = pd.date_range(start, end, freq=freq, name="ts")
    edges_ts = grid.append(pd.DatetimeIndex([grid[-1] + pd.Timedelta(freq)]))
    epoch = pd.Timestamp("1970-01-01")
    edges = ((edges_ts - epoch) / pd.Timedelta("1s")).to_numpy()
    out = {}
    for space, g in ev.groupby("floorspaceid", observed=True):
        g = g.sort_values("ts")
        t = ((g["ts"] - epoch) / pd.Timedelta("1s")).to_numpy()
        # Several events can share a timestamp; keep the last state reported at that instant.
        last = np.append(t[1:] != t[:-1], True)
        out[str(space)[:8]] = time_weighted_step_mean(t[last], g["headcount"].to_numpy()[last], edges, max_hold_hours * 3600)
    return pd.DataFrame(out, index=grid)


def load_occupancy_grid(path: str | Path, freq: str = "10min") -> pd.DataFrame:
    """Convenience wrapper: :func:`load_occupancy_events` then :func:`occupancy_to_grid`."""
    return occupancy_to_grid(load_occupancy_events(path), freq=freq)


@dataclass
class GapStats:
    """Summary of missingness in one regularly sampled series.

    Attributes:
        n: Number of grid points.
        n_missing: Number of NaN grid points.
        frac_missing: ``n_missing / n``.
        gap_lengths: Length (in grid steps) of every contiguous NaN run.
    """

    n: int
    n_missing: int
    frac_missing: float
    gap_lengths: np.ndarray

    def histogram(self, edges=(1, 2, 4, 7, 19, 37, 145, 1009, np.inf)) -> pd.Series:
        """Count gaps by length bin (bins are ``[edges[i], edges[i+1])`` steps)."""
        cats = pd.cut(self.gap_lengths, bins=list(edges), right=False)
        return pd.Series(cats).value_counts().sort_index()


def gap_lengths(isnan: np.ndarray) -> np.ndarray:
    """Return the length of every contiguous run of ``True`` in a boolean array.

    Args:
        isnan: 1-D boolean array (True = missing).

    Returns:
        Integer array of run lengths, in order of occurrence.

    Example:
        >>> gap_lengths(np.array([0, 1, 1, 0, 1], dtype=bool)).tolist()
        [2, 1]
    """
    x = np.concatenate([[0], np.asarray(isnan, dtype=np.int8), [0]])
    d = np.diff(x)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return ends - starts


def audit_series(s: pd.Series) -> GapStats:
    """Compute missingness statistics for one series on a regular grid."""
    isnan = s.isna().to_numpy()
    return GapStats(len(s), int(isnan.sum()), float(isnan.mean()), gap_lengths(isnan))


def longest_observed_window(s: pd.Series, max_gap: int) -> pd.Series:
    """Return the longest stretch of ``s`` that contains no gap longer than ``max_gap``.

    Used to select a study period that excludes multi-month outages, which no
    method in this project can (or should) reconstruct.

    Args:
        s: Regularly sampled series with NaNs.
        max_gap: Largest tolerated NaN run, in steps.

    Returns:
        The slice of ``s`` (trimmed so it starts and ends on observed values).
    """
    isnan = s.isna().to_numpy()
    x = np.concatenate([[0], isnan.astype(np.int8), [0]])
    d = np.diff(x)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    long_gaps = [(a, b) for a, b in zip(starts, ends) if b - a > max_gap]
    # Segment boundaries are the long gaps; choose the longest segment between them.
    cuts = [0] + [v for a, b in long_gaps for v in (a, b)] + [len(s)]
    segs = [(cuts[i], cuts[i + 1]) for i in range(0, len(cuts), 2)]
    a, b = max(segs, key=lambda ab: ab[1] - ab[0])
    seg = s.iloc[a:b]
    valid = seg.notna().to_numpy()
    first, last = np.argmax(valid), len(valid) - np.argmax(valid[::-1])
    return seg.iloc[first:last]

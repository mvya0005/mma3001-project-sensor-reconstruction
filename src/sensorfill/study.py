"""Study configuration shared by all notebooks: sensors, splits and occupancy proxy.

Keeping these choices in one place guarantees that every notebook trains,
tunes and tests on exactly the same periods.

Design decisions (justified in notebook 00):

* **Primary sensor** ``6012002000869`` (Building 60, G.38 Compactus upper):
  15 weeks of 10-min data, only isolated 1-2 sample dropouts, so long
  artificial gaps have clean ground truth.
* **Chronological split** (no shuffling, so no future information leaks into
  training): train 29 Feb - 30 Apr 2024, validation 1 - 21 May, test
  22 May - 14 Jun. Validation is used for every model/parameter choice; the
  test period is touched once, in notebook 06.
* **Occupancy proxy**: the sum of the four floor spaces that report
  throughout the study period. None of them is confirmed to be the same room
  as the CO2 sensor, so the proxy represents building activity, not the
  room's occupants. The fifth space (``3e5a14df``) only starts reporting in
  May, so it would be missing for the whole training period.
* **Secondary sensor** ``6012002000326`` (location not listed): a different
  room and period (Sep 2025 - Apr 2026) used only as an external
  generalisation test, without occupancy (its CO2 does not correlate with
  any occupancy space).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .interpolation import daily_profile

PRIMARY_SENSOR = "6012002000869"
SECONDARY_SENSOR = "6012002000326"
TRAIN_END = "2024-05-01"
VAL_END = "2024-05-22"
OCC_PROXY_SPACES = ["0f3f37f8", "941f0352", "227f1068", "cca1ac3e"]
STEPS_PER_DAY = 144
DT_HOURS = 1 / 6
#: Artificial gap lengths used throughout, in 10-min steps (10 min ... 24 h).
GAP_LENGTHS = (1, 3, 6, 18, 36, 72, 144)
#: Masking protocol of the one-off final test evaluation (identical for every method,
#: including the GPU models scored in notebook 05).
FINAL_PROTOCOL = {"gap_lengths": GAP_LENGTHS, "n_gaps": 30, "repeats": 5, "seed": 2024}


@dataclass
class Study:
    """One target variable of one sensor, with splits and context.

    Attributes:
        y: Target series (NaN = missing) on the 10-min grid.
        ts: Timestamps.
        occ: Occupancy proxy [persons] or None.
        split: Array of ``"train"``, ``"val"`` or ``"test"`` labels.
        variable: Target name (e.g. ``"co2"``).
        unit: Target unit.
    """

    y: np.ndarray
    ts: pd.DatetimeIndex
    occ: np.ndarray | None
    split: np.ndarray
    variable: str
    unit: str

    def part(self, name: str):
        """Return ``(y, ts, occ)`` restricted to one split."""
        m = self.split == name
        return self.y[m], self.ts[m], None if self.occ is None else self.occ[m]

    @property
    def train_profile(self) -> pd.Series:
        """Typical-day profile computed on the training period only (no leakage)."""
        y, ts, _ = self.part("train")
        return daily_profile(y, ts)

    def profile_values(self, ts) -> np.ndarray:
        """Evaluate :attr:`train_profile` on arbitrary timestamps."""
        ts = pd.DatetimeIndex(ts)
        keys = pd.MultiIndex.from_arrays([ts.dayofweek >= 5, ts.hour * 60 + ts.minute])
        return self.train_profile.reindex(keys).to_numpy()

    @property
    def level_ref(self) -> float:
        """Median of the training target (reference level for ML features)."""
        return float(np.nanmedian(self.part("train")[0]))


def load_study(processed: str | Path = "data/processed", variable: str = "co2", sensor: str = PRIMARY_SENSOR) -> Study:
    """Load a processed sensor series with its split labels and occupancy proxy.

    Args:
        processed: Folder written by ``scripts/preprocess.py``.
        variable: ``"co2"``, ``"temp"``, ``"rh"`` or ``"pressure"``.
        sensor: Sensor id.

    Returns:
        A :class:`Study`. For the secondary sensor, ``occ`` is None and the
        whole series is labelled ``"test"``.

    Raises:
        FileNotFoundError: If preprocessing has not been run.
    """
    from .data import UNITS

    processed = Path(processed)
    f = processed / f"env_{sensor}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found - run scripts/preprocess.py (or notebook 00) first")
    env = pd.read_parquet(f)
    ts = pd.DatetimeIndex(env.index)
    y = env[variable].to_numpy(dtype=float)
    if sensor == PRIMARY_SENSOR:
        occ_df = pd.read_parquet(processed / "occupancy.parquet").reindex(ts)
        occ = occ_df[OCC_PROXY_SPACES].sum(axis=1, min_count=len(OCC_PROXY_SPACES)).to_numpy()
        split = np.where(ts < TRAIN_END, "train", np.where(ts < VAL_END, "val", "test"))
    else:
        occ = None
        split = np.full(len(ts), "test")
    return Study(y, ts, occ, split, variable, UNITS.get(variable, ""))

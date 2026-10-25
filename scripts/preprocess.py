"""Convert the raw Dataset 2 exports into compact, regular-grid Parquet files.

Usage (from the repository root)::

    python scripts/preprocess.py --raw data/raw --out data/processed

``--raw`` must contain the two zip files supplied by the unit
(``5EnvSensor_MayToDec2024_180kRows.zip`` and
``5occupancySensor_MayToDec2024_9MRows.zip``); the CSVs are read straight
from the zips. Outputs (10-minute grid, local time, NaN = missing):

* ``env_<sensorid>.parquet`` - co2 [ppm], temp [degC], rh [%RH], pressure [mbar]
* ``occupancy.parquet`` - mean headcount [persons] per floor space
* ``audit.json`` - data-quality counts used in the report
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sensorfill import data

ENV_ZIP = "5EnvSensor_MayToDec2024_180kRows.zip"
OCC_ZIP = "5occupancySensor_MayToDec2024_9MRows.zip"
#: Sensors with enough data to study (the other three report 1-2 readings).
SENSORS = ["6012002000869", "6012002000326"]


def main(raw: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    audit = {}
    t0 = time.perf_counter()
    long = data.load_env_long(raw / ENV_ZIP)
    audit["env_unique_readings"] = int(long.groupby(["sensorid", "time"]).ngroups)
    audit["env_readings_per_sensor"] = long.groupby("sensorid")["time"].nunique().to_dict()
    for sid in SENSORS:
        grid = data.env_to_grid(long, sid, clean=False)
        cleaned, rejected = data.apply_plausibility(grid)
        # Keep the longest stretch without a >7-day outage: multi-month gaps are out of scope.
        keep = data.longest_observed_window(cleaned["co2"], max_gap=7 * 144).index
        cleaned = cleaned.loc[keep]
        cleaned.to_parquet(out / f"env_{sid}.parquet")
        st = data.audit_series(cleaned["co2"])
        audit[sid] = {
            "start": str(keep[0]), "end": str(keep[-1]), "n": st.n, "n_missing": st.n_missing,
            "frac_missing": st.frac_missing, "rejected_implausible": rejected.to_dict(),
            "gap_histogram": {str(k): int(v) for k, v in st.histogram().items()},
            "longest_gap_steps": int(st.gap_lengths.max()) if st.gap_lengths.size else 0,
        }
        print(f"sensor {sid}: {keep[0]} -> {keep[-1]}, {st.frac_missing:.1%} missing")
    print(f"env done in {time.perf_counter() - t0:.0f}s; loading occupancy (9M rows, ~1 min)...")
    ev = data.load_occupancy_events(raw / OCC_ZIP)
    audit["occ_events"] = int(len(ev))
    audit["occ_status_counts"] = ev["status"].value_counts().to_dict()
    audit["occ_negative_or_huge_headcount"] = int(((ev["headcount"] < 0) | (ev["headcount"] > 200)).sum())
    occ = data.occupancy_to_grid(ev)
    occ.to_parquet(out / "occupancy.parquet")
    audit["occ_spaces"] = list(occ.columns)
    (out / "audit.json").write_text(json.dumps(audit, indent=2, default=str))
    print(f"all done in {time.perf_counter() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    a = ap.parse_args()
    main(a.raw, a.out)

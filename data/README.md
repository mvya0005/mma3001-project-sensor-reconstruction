# Data

The sensor data belongs to Monash University (Smart Infrastructure / NEMo cloud
export supplied for MMA3001 Dataset 2) and is **not committed** to this
repository. To reproduce the results:

1. Download the two zip files supplied on Moodle:
   - `5EnvSensor_MayToDec2024_180kRows.zip` (environmental sensors, 7 MB)
   - `5occupancySensor_MayToDec2024_9MRows.zip` (occupancy events, 85 MB)
   - optional: `Sensor ID and Locations.xlsx`
2. **Local:** put them in `data/raw/`.
   **Colab:** put them in `My Drive/MMA3001/raw/` (notebook 00 mounts Drive).
3. Run `python scripts/preprocess.py` (or notebook 00). It writes
   `data/processed/env_<sensor>.parquet`, `occupancy.parquet` and `audit.json`
   (about 2 MB in total, about 1 minute).

## What is in the raw files (from our audit)

| Item | Finding |
|---|---|
| Env rows | 179,447 uploads but only 42,843 unique (sensor, time) readings: uploads are repeated |
| Env sensors | 5 ids; only `6012002000869` (G.38 Compactus upper, Bldg 60) and `6012002000326` (location not in the sheet) have usable series |
| Sampling | Every 600 s; about 7 % of readings skipped (an interval of 1200 s) |
| Time span | Despite the file name, readings run Dec 2023 to Apr 2026 (sensor 869: 29 Feb to 14 Jun 2024; sensor 326: Sep 2025 to Apr 2026) |
| Error records | Values with `errorCode` and empty value lists are present and skipped |
| Occupancy | 9.09 M change events from 5 floor spaces; silence after a `NotOccupied` event means "still empty" |
| Location sheet | The OCC tab is labelled "Incorrect Data"; only 2 of the 5 floor-space ids appear in it |

Variables, units and valid ranges are listed in `sensorfill.data.UNITS` and
`sensorfill.data.PLAUSIBLE_RANGE`.

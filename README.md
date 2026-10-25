# Reconstructing missing building-sensor measurements

**MMA3001 Numerical Methods and Machine Learning, individual project (Dataset 2: Monash Smart Infrastructure occupancy and environmental data).**

Wireless indoor-air sensors drop readings, from single lost packets to multi-day outages. This project builds and validates a library, `sensorfill`, and a set of Colab notebooks that reconstruct missing 10-minute CO₂ and temperature readings. It compares classical numerical methods, a physics-based (mass-balance ODE) model, and machine-learning and deep-learning models on identical, withheld ground truth. The result is an evidence-based policy for which method a building management system should use for each gap length.

<!-- RESULTS -->

---

## 1. Engineering problem

| | |
|---|---|
| **Context** | Building services (HVAC) is a branch of mechanical engineering. Building management systems (BMS) use indoor CO₂ and temperature for demand-controlled ventilation, indoor-air-quality compliance and energy reporting. Gaps break control loops and bias daily indicators. |
| **Problem** | Given a regularly sampled sensor series with gaps (and optionally building occupancy), estimate the missing values and quantify the error as a function of gap length. |
| **Why it matters** | A daily ventilation indicator such as the CO₂ dose ∫(C − 420 ppm)dt is biased whenever a gap falls in working hours. Naive filling (hold-last-value) misstates it by tens of percent. |
| **Intended use** | Offline cleaning of logged data before computing daily indicators. Methods may use data after a gap. A real-time variant (forward IVP) is also provided. |
| **Scope / limits** | Gaps from 10 min to 24 h. Longer outages are flagged, not filled. One building (two sensors, two periods). The occupancy sensors are not co-located with the CO₂ sensor, so occupancy is used as a building-activity proxy. |

## 2. Inputs and outputs

| | Item | Format, units, domain | Invalid / missing / unsupported input |
|---|---|---|---|
| In | `5EnvSensor_*.zip` | CSV with a JSON column; Unix-second times; CO₂ [ppm], T [°C], RH [%], p [mbar] | duplicate uploads collapsed; `errorCode` records skipped; values outside `data.PLAUSIBLE_RANGE` become NaN |
| In | `5occupancySensor_*.zip` | change events; headcount [persons, 0–200]; local time | out-of-range headcounts dropped; silence > 72 h treated as an outage |
| Mid | `data/processed/env_<id>.parquet`, `occupancy.parquet` | 10-min local-time grid, NaN = missing | — |
| Out | reconstructed series | same grid and units; observed values never modified | edge gaps (no reading after the gap) are held at the last value and flagged; gaps beyond a model's window fall back to linear |
| Out | benchmark tables, figures | RMSE / MAE / bias / max error on hidden points per gap length; runtime; peak memory | a method that raises is recorded as failed, not dropped |
| Out | daily indicators | CO₂ dose [ppm·h], degree-hours [K·h] | days with any NaN return NaN |

Function-level inputs and outputs are documented in the docstrings and in the HTML API documentation (`docs/api/index.html`).

## 3. Repository structure

```
├── src/sensorfill/          library (Google-style docstrings throughout)
│   ├── data.py              load, de-duplicate, audit, 10-min grid, occupancy step-function means
│   ├── masking.py           artificial gaps on observed data (the validation ground truth)
│   ├── interpolation.py     linear, Newton, natural spline (own), PCHIP, profile, 2-D griddata
│   ├── linalg.py            Thomas, Gaussian elimination, SVD low-rank completion
│   ├── pde.py               Laplace (Jacobi/GS/SOR/direct) and diffusion (FTCS/Crank–Nicolson) inpainting
│   ├── ode.py               finite differences, Euler/Heun/RK4/implicit integrators, CO₂ model, shooting
│   ├── quadrature.py        trapezoid, Simpson, Romberg, Gauss–Legendre, daily indicators
│   ├── features.py, ml.py   gap-aware features; linear/ridge/tree/forest/SVR/MLP imputers
│   ├── deep.py              BiGRU, physics-informed BiGRU, neural ODE (PyTorch, GPU)
│   ├── metrics.py           error metrics on hidden points
│   ├── profiling.py         timing, memory, cProfile, FLOP and roofline helpers
│   ├── benchmark.py         one harness: every method, identical gaps
│   └── study.py             sensors, chronological splits, protocol shared by all notebooks
├── notebooks/               00 → 06, Colab-ready (open with the badges)
├── tests/                   pytest suite (verification against analytical results)
├── scripts/                 preprocess.py, build_docs.sh, run_tests.sh
├── docs/                    api/ (pdoc HTML), report_outline.md, ai_use_log.md, rubric_check.md
├── reports/                 figures/, result CSVs, tests/test_report.html
└── data/                    README.md only (raw data is Monash's and is not redistributed)
```

## 4. How to reproduce

### On Google Colab (recommended; needed for GPU training)
1. Put the two Moodle zip files in **`My Drive/MMA3001/raw/`**.
2. Open each notebook with its *Open in Colab* badge, in order **00 → 06**. The first cell clones this repository, installs the package, mounts Drive and runs the preprocessing once (cached in `My Drive/MMA3001/processed/`).
3. For **notebook 05** choose *Runtime → Change runtime type → T4 GPU*. It writes `reports/test_deep.csv`. Upload it to the repository (or save the notebook to GitHub) so that notebook 06 includes the deep models.

The repository must be public for the Colab clone to work without a token. Alternatively, upload a zip of the repository to Colab.

### Locally
```bash
git clone https://github.com/mvya0005/mma3001-project-sensor-reconstruction.git
cd mma3001-project-sensor-reconstruction
python -m venv .venv && source .venv/bin/activate      # Python ≥ 3.10
pip install -e ".[dev,deep]"
cp /path/to/*.zip data/raw/
python scripts/preprocess.py                           # ~1 min
jupyter lab notebooks/                                 # run 00 → 06
bash scripts/run_tests.sh                              # writes reports/tests/test_report.html
bash scripts/build_docs.sh                             # writes docs/api/index.html
```

Everything is seeded (mask draws, model initialisation, data augmentation), so re-running gives the same tables, except for GPU non-determinism in notebook 05, which is quantified by training over several seeds.

## 5. Methods and where each unit topic is used

| Unit topic | Implementation | Notebook |
|---|---|---|
| 1.2–1.5 Data management, documentation, AI, testing | git history, pdoc docs, `ai_use_log.md`, pytest suite, CI | all |
| 2.1 Optimisation and profiling | `profiling.py`; solver scaling; cProfile; least-squares fits; roofline | 02, 03, 05, 06 |
| 2.2 Numerical errors | float32 timestamp round-off; Vandermonde conditioning; noise floor | 00, 01 |
| 2.3 Systems of linear equations | Thomas, Gaussian elimination with pivoting, sparse direct solve, `lstsq` | 02, 03 |
| 3.1–3.6 Interpolation | linear, Newton divided differences, natural spline, PCHIP, 2-D griddata, ML as local interpolation | 01, 04 |
| 5.1–5.6 ML regression | linear/ridge, tree, forest, SVR, MLP; grouped CV; learning curve; ablation | 04 |
| 6.1–6.5 Numerical integration | trapezoid, Simpson 1/3 and 3/8, Romberg, Gauss–Legendre (1-D and 2-D); exact step-function integrals | 00, 06 |
| 7.1–7.5 Finite differences and ODEs | FD orders; Euler, Heun, midpoint, RK4; CO₂ mass balance; second-order (sensor-lag) system | 03 |
| 9.1–9.4 Stability, stiffness, implicit methods, shooting | stability functions; stiff sensor-lag model; implicit Euler/trapezoid; secant shooting BVP | 03 |
| 10.1–10.4 Relaxation, neural ODEs, PINNs, PDE classes | Jacobi/GS/SOR Laplace fill; neural ODE; physics-informed loss; elliptic vs parabolic | 02, 05 |
| 11.1–11.5 PDEs | FTCS (with stability limit), Dirichlet/Neumann/periodic BCs, elliptic 2-D Laplace, Crank–Nicolson | 02 |

## 6. Validation design

* **Ground truth by withholding.** Observed readings are hidden in blocks of 10 min to 24 h and reconstructed. Only the hidden points are scored. Masks keep observed margins, so every method has the same boundary information.
* **Chronological split, no leakage.** Train 29 Feb – 30 Apr 2024, validation 1–21 May, test 22 May – 14 Jun. Profiles, scalers, feature references and occupancy-proxy selection use training data only. All model choices are made on validation. The test period is used once, in notebook 06.
* **Repeatability.** 5 independent mask draws give 95 % confidence intervals. The deep models are trained over 3 seeds.
* **Verification of the numerics.** Analytical solutions and convergence orders (`tests/`), physical plausibility of fitted parameters, and stability limits measured against theory.
* **Engineering validation.** The error in the daily CO₂ dose, an integrated indicator, not only point-wise RMSE.
* **External check.** A second sensor in a different room and year, never used for any choice.
* **What it cannot establish.** Real outages may cluster at particular times, unlike random masks. Scores are against noisy readings (an ≈ 8 ppm floor). Only one room had a long, clean record.

## 7. Tests and documentation

* `bash scripts/run_tests.sh` runs the pytest suite and writes `reports/tests/test_report.html`. The tests verify against analytical results: exact polynomial reproduction, spline O(h⁴), FD O(h)/O(h²), integrator orders 1/2/4, FTCS stability limit, SOR speed-up, parameter recovery, Romberg to machine precision and Gauss exactness.
* `bash scripts/build_docs.sh` regenerates the HTML API documentation (`docs/api/index.html`) from the docstrings.
* GitHub Actions runs the tests on every push (`.github/workflows/tests.yml`).

## 8. AI use

This project was developed with substantial help from an AI coding assistant (Claude, Anthropic). What it produced, how each part was verified, and the AI errors that were caught are recorded in [`docs/ai_use_log.md`](docs/ai_use_log.md). The student's own reflection is in the written report.

## 9. Licence and data

Code and documentation: MIT licence (see `LICENSE`). The sensor data belongs to Monash University and is not included. See `data/README.md` for how to obtain it and what the audit found.

# Written report: suggested structure (5–10 pages)

This is a skeleton for the Moodle report. It lists what each section should
contain and which repository evidence (figure, table, notebook) supports it.
**Write the prose yourself.** You will be questioned on it in the presentation,
and the AI reflection must be your own.

Page budget: about 9 pages plus references and appendices (which do not count).

---

## 1. Engineering problem (about 1 page)
*Rubric: "Engineering problem, inputs and outputs".*

- Context: building services (mechanical) engineering. Demand-controlled
  ventilation and indoor-air-quality (IAQ) compliance use CO₂ and temperature
  from wireless sensors. Missing readings break control loops and bias the
  daily indicators used in energy and IAQ reports.
- Problem statement: given a 10-min series with gaps, plus building occupancy,
  estimate the missing readings and say how accurate the estimate is, as a
  function of gap length.
- Why it matters: quote the effect of a gap on the daily CO₂ dose
  (notebook 06, dose table).
- Intended use: offline cleaning before reporting, with a real-time variant.
- Scope and limits: gaps from 10 min to 24 h; one room for training; the
  occupancy sensors are in other spaces, so occupancy is a proxy.
- Figure: `00_series_overview.png`.

## 2. Inputs, outputs and data (about 1 page)
- Copy and condense README section 2 (formats, units, domains, invalid input).
- Data audit: duplicate uploads, dropout rate, the noise floor (≈ 8.5 ppm),
  which sensor was chosen and why (notebook 00, `data/README.md`).
- Figure: `00_noise_day.png` (the noise floor sets the best achievable RMSE).

## 3. Computational approach (about 2.5 pages)
*Rubric: "Justification of computational approach". For each family: how it
works, why it was chosen, its assumptions, settings and limitations.*

| Family | Key point to make | Evidence |
|---|---|---|
| Interpolation (linear, Newton, spline, PCHIP, profile) | short gaps are smooth; long gaps need the daily pattern; Runge and conditioning | nb 01, `01_runge_conditioning.png`, `01_val_summary.png` |
| Linear algebra / PDE inpainting (SVD, Laplace by SOR, diffusion by CN) | the series folded into a day × time-of-day image; elliptic vs parabolic; ω_opt | nb 02, `02_relaxation.png`, `02_ftcs_cn_stability.png` |
| ODE grey-box (CO₂ mass balance) | physical parameters g, λ, C_out; equation-error vs output-error fit | nb 03, `03_free_simulation.png`, `03_stability_functions.png` |
| ML regression (ridge, tree, forest, SVR, MLP) | gap-aware features; grouped CV; leakage avoided | nb 04, `04_learning_curve.png`, `04_permutation_importance.png` |
| Deep learning (BiGRU, PINN, neural ODE) | sequence context; physics loss; GPU training | nb 05 |

Keep the maths short: one equation per family (e.g. the mass balance
dC/dt = g·N − λ(C − C_out), the SOR update, the ridge objective).

## 4. Validation method and results (about 2 pages)
*Rubric: "Validation and engineering credibility".*

- What was validated: point accuracy on hidden readings, and the error in an
  integrated indicator (daily CO₂ dose).
- Why artificial gaps on observed data: the only way to get ground truth for
  missing values.
- Metrics: RMSE (primary), MAE, bias, max error, 95 % CI over 5 mask draws;
  acceptance criterion: an RMSE close to the noise floor for short gaps, and
  better than linear for long gaps.
- Did validation influence selection? Yes: the validation period chose
  hyper-parameters and the hybrid switch point. The test period was used once.
- Verification of the numerics: the pytest suite (convergence orders,
  stability limits, analytic solutions). Cite `reports/tests/test_report.html`.
- External check: sensor 326 (different room and year).
- Failure modes and unsupported inputs: the table in notebook 06.
- What validation cannot establish (README section 6).
- Figures: the main test accuracy-vs-gap-length plot from notebook 06, and the
  dose-error table.

## 5. Comparison and optimisation (about 1.5 pages)
*Rubric: brief section 6, "Optimisation and performance".*

- Baseline: linear interpolation (and hold-last-value).
- Accuracy vs cost: runtime, peak memory, the FLOP estimates and the roofline
  (notebook 06).
- Optimisation examples with before/after numbers:
  - Jacobi → Gauss–Seidel → SOR iteration counts (notebook 02);
  - FTCS step limit vs Crank–Nicolson (notebook 02);
  - ridge α, tree depth and forest size chosen by grouped CV (notebook 04);
  - the hybrid method (linear for short gaps, model for long ones) and its
    switch point chosen on validation (notebook 06);
  - CPU vs T4 GPU training time (notebook 05).
- Final design and the trade-off that justifies it.

## 6. Limitations and lessons learned (about 0.5 page)
- An honest list: one training room; random masks vs real outage patterns;
  scoring against noisy readings; the occupancy proxy; ML not clearly better
  than the profile baseline.
- Two or three lessons in your own words (e.g. the errors-in-variables bias of
  the equation-error fit; a simple physics-aware baseline is hard to beat).
- Improvements you would make next, each justified by a result.

## 7. AI use and reflection (about 0.5–1 page)
*Rubric: "AI use and critical reflection". This is assessed heavily on
ownership. Write it yourself.*

- Use `docs/ai_use_log.md` sections 1–3 as the factual record.
- Answer every prompt in section 4 of that file in your own words. Say which
  decisions you made, which AI errors you caught, and how you checked results.

## References
- Dataset: Monash Smart Infrastructure (as provided on Moodle).
- Textbook(s) used in the unit for the numerical methods.
- Library documentation: NumPy, SciPy, scikit-learn, PyTorch.
- Background: ASHRAE 62.1 (ventilation and CO₂), and Persily & de Jonge (2017)
  on CO₂ generation rates per person, used to sanity-check the fitted g.
- Add anything else you read.

## Appendices (not counted)
- The full method × gap-length result tables (`reports/*.csv`).
- The test report summary.

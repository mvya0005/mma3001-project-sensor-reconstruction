# Rubric and brief self-check

Every requirement in the project brief and every rubric row, mapped to where
the evidence is in this repository. "Student" marks items that only you can
complete. Use this as a checklist before submitting.

## A. Brief: "Each project must include"

| Requirement | Evidence | Status |
|---|---|---|
| Engineering problem relevant to mechanical/aerospace engineering | Building services (HVAC, ventilation control, IAQ compliance): README §1, notebook 00 §1 | done |
| At least one numerical method or ML technique taught in the unit | All content-list topics: README §5 table | done |
| Model development | notebooks 01–05 (each method built, tuned on validation) | done |
| Validation and verification | validation: masks on withheld data, chronological split, CIs (notebook 06); verification: `tests/` + `reports/tests/test_report.html` | done |
| Sensitivity analysis or optimisation | ODE parameter sensitivity (nb 03 §5); SOR ω, Laplace weight w, spline/Newton order (nb 01–02); ML hyper-parameter search (nb 04); hybrid switch point (nb 06) | done |
| Interpretation in an engineering context | daily CO₂ dose error, failure modes, recommended policy (nb 06) | done |
| Reproducible workflow with GitHub | README §4; seeded masks and models; CI workflow | done after push |
| Appropriate computational tools | Python, Jupyter/Colab, NumPy/SciPy, scikit-learn, PyTorch | done |
| Acknowledgement and details of AI use | `docs/ai_use_log.md` §1–3 | record done; **reflection: student** |

## B. Brief: documentation sections 1–7

| Section | Required content | Where |
|---|---|---|
| 1 Engineering problem | context, problem, why it matters, intended use, scope limits | README §1; nb 00 §1 |
| 2 Inputs and outputs | data accepted; format, units, domain; outputs and meaning; invalid/missing/unsupported input; function-level I/O | README §2; nb 00; docstrings and `docs/api/` |
| 3 Computational solution | how each method works, why chosen, assumptions, parameters, implementation, limitations | markdown in nb 01–05; module docstrings |
| 4 Alternative solutions | baseline (linear, LOCF) and credible alternatives compared on accuracy, robustness, time, memory, complexity, interpretability, scalability | nb 06 accuracy, cost and failure-mode tables |
| 5 Validation | what, why, metrics/acceptance criteria, whether validation influenced selection, what it cannot establish | README §6; nb 06 |
| 6 Optimisation and performance | profilers, timing, memory, FLOPs and arithmetic intensity, baseline vs improved, final design justified | `profiling.py`; nb 02 §6 (cProfile); nb 06 (roofline, cost); nb 05 (CPU vs GPU) |
| 7 AI use and reflection | tools, uses, contribution level, why, checking, student decisions, AI errors, effect on learning | factual log done; **reflection: student** |

## C. Brief: GitHub repository contents

| Item | Where | Status |
|---|---|---|
| README | `README.md` | done |
| HTML documentation of source | `docs/api/index.html` (pdoc; rebuild with `scripts/build_docs.sh`) | done |
| Licence | `LICENSE` (MIT; data excluded) | done |
| Source code, functional main branch | `src/sensorfill/`; CI runs the tests on every push | done after push |
| Inline comments and docstrings | Google-style docstrings in every public function, with Args/Returns/Raises | done |
| Test functions and test report | `tests/` (87 tests); `reports/tests/test_report.html` | done |
| Zipped clone submitted to Moodle | — | **student** |
| Accessible repository link | repository must be public, or shared with the marker | **student** |

## D. Rubric rows (aiming at "Excellent")

### Engineering problem, inputs and outputs
- Problem, context, importance, intended use, scope and limits: README §1.
- Inputs and outputs with format, units, domain, meaning and intended use,
  plus invalid-input handling: README §2 and the nb 00 table.
- Report: condense these into section 1–2 of the written report.

### GitHub structure, commits and README
- Logical layout (`src/`, `tests/`, `notebooks/`, `scripts/`, `docs/`,
  `reports/`, `data/`) with consistent naming (notebooks numbered 00–06).
- README covers purpose, usage, environment, dependencies (`pyproject.toml`,
  `requirements.txt`) and the reproducible workflow (Colab and local).
- Data handled appropriately: raw data is not redistributed, the processing
  script is provided, and `data/README.md` records the audit.
- Commits are topic-by-topic with explanatory messages. Feature branches were
  merged with `--no-ff`, so branching is visible in `git log --graph`.
- **Student:** keep committing your own work (report figures, edits,
  reflection) so the history shows your contribution.

### Inline comments, docstrings and HTML documentation
- Comments explain *why* (e.g. pivoting, stability limits, leakage
  avoidance), not what the line does.
- Google-style docstrings with Args, Returns and Raises throughout.
- pdoc HTML is generated from the same docstrings, so the docs stay in sync
  (rebuild before submitting).

### Validation and engineering credibility
- Clear inputs and outputs (above).
- Validation design: ground truth by withholding observed readings; a
  chronological train/validation/test split with no leakage; one-off test
  use; 5 mask draws with 95 % CIs; repeat seeds for the deep models.
- Correct metrics: RMSE, MAE, bias and max error on hidden points only, plus
  the dose error as an engineering metric.
- Comparison with ground truth, a baseline, physical constraints
  (non-negative occupancy, plausible CO₂, physically plausible ODE
  parameters) and an external sensor.
- Unsupported inputs and failure modes: the nb 06 failure-mode table; edge
  gaps flagged; outages above 24 h flagged, not filled.
- Honest limitations: README §6 "What it cannot establish"; negative results
  are kept (SVD, high-order Newton, PINN at short gaps, ML ≈ profile).

### Justification of computational approach
- Clear baseline (linear interpolation; LOCF as the naive one).
- Method selection justified on validation numbers in every notebook.
- Quantitative analysis: accuracy per gap length, runtime, peak memory,
  FLOP estimates, roofline, scaling with n, robustness to the external sensor.
- Evidence-supported improvements: robust anchors, the hybrid method, the
  direct solver over SOR, output-error over equation-error fitting.
- Trade-offs: nb 06 recommendation (accuracy vs cost vs interpretability).

### AI use and critical reflection
- Disclosure and the record of checks and caught errors: `docs/ai_use_log.md`.
- **Student:** the rubric rewards evidence that *you* drove the process.
  Write section 4 of the log, and the report's reflection, in your own words.
  Be ready to explain any function in the presentation without AI.

## E. Presentation (Week 14, 5 min + 3 min questions)

The brief lists likely questions. Where the answers are in the repository:

| Question | Where to find your answer |
|---|---|
| What physical system does the model represent; what assumptions? | nb 03 §1: well-mixed room mass balance, constant generation per person, occupancy proxy |
| Which parameters or boundary conditions matter most; are they realistic? | nb 03 §5 sensitivity; fitted g and 1/λ vs literature; Dirichlet/Neumann/periodic BCs in nb 02 |
| How did you ensure numerical stability? | FTCS limit vs CN (nb 02 §4); stability functions and stiff case (nb 03); pivoting; float64 and scaled time axis (nb 01 §1) |
| Convergence criteria and why appropriate? | range-scaled tolerance (nb 02 §3); SVD relative change; secant tolerance; ML early stopping |
| Main sources of numerical error and how quantified? | noise floor (nb 00 §4); truncation vs round-off (nb 01 §1); integrator orders (nb 03 §1); interpolation error vs gap length |
| How did you verify the implementation? | `tests/`: analytical solutions, convergence orders, comparisons with SciPy |
| Most vulnerable part, and mitigation? | nb 06 failure-mode table (gaps over 24 h, edge gaps, distribution shift, occupancy missing or not co-located) |

## F. Remaining actions (student)

1. Upload the two zips to `My Drive/MMA3001/raw/` and run notebook 05 on a
   Colab T4 GPU (full mode). Save the notebook and `reports/test_deep.csv`
   back to the repository, then re-run notebook 06.
2. Write the AI reflection (log §4 and report section 7).
3. Write the 5–10 page report (`docs/report_outline.md` is a skeleton).
4. Make the repository public (or share it with the marker), then submit the
   link and a zipped clone on Moodle.
5. Rehearse the 5-minute demo: README → notebook 06 summary → test report →
   API docs → AI log.

# AI-use record

The brief requires full disclosure of material AI use, plus a reflection
written by the student. This file has two parts:

1. **The factual record**, kept during development: which tool did what, and
   how the output was checked.
2. **The reflection** (section 4) is left for the student to write in their own
   words. It is the part assessed under "AI use and critical reflection".
   Do not let an AI write it.

## 1. Tools used

| Tool | Used for |
|---|---|
| Claude (Anthropic), via Claude Code in a Claude.ai project | drafting the `sensorfill` package, tests, notebooks, the preprocessing script and the documentation, from the student's brief, unit content list and dataset |
| Google Colab (T4 GPU) | training the deep-learning models (notebook 05) |

## 2. What the AI produced, and how it was checked

| Area | AI contribution | Verification built into the repository |
|---|---|---|
| Data loading and audit | parser for the JSON export, de-duplication, occupancy step-function averaging | unit tests on hand-made files (`tests/test_data.py`); audit counts cross-checked against raw-file row counts in notebook 00 |
| Numerical methods (Newton, spline, Thomas, SOR, FTCS/CN, RK4, implicit Euler, shooting, Romberg, Gauss) | implementations from textbook formulas | analytical checks: exact reproduction of polynomials, comparison with SciPy, measured convergence orders vs theory, stability limits (`tests/`) |
| Feature design and ML training | gap-aware features, grouped CV | learning curves, depth curve, permutation importance; test metrics computed only on hidden points |
| Deep models | BiGRU, physics-informed loss, neural ODE | RK4 roll-out tested against the exact exponential; repeat training over seeds |
| Text | notebook explanations and README | every number quoted was re-read from executed outputs |

## 3. AI errors and unhelpful suggestions found during development

These were caught by the tests or by looking at the results, and were fixed.
They are recorded here as evidence that the output was checked rather than
accepted.

1. **Occupancy aggregation.** The first version averaged the occupancy *events*
   in each 10-min bin and left empty bins as missing. The audit showed that
   silence after a `NotOccupied` event means "still empty", not "missing", so
   this was replaced by an exact time-weighted step-function mean.
2. **Artificial gaps could not be placed.** The first masking routine required
   fully observed blocks. Because this sensor drops a reading about every 10
   samples, no 3-hour block qualified. Masks now straddle isolated real
   dropouts, and those points are never scored.
3. **Machine-learning features leaked level and over-represented long gaps.**
   The first feature set used absolute CO₂ levels and packed hundreds of
   long gaps into one mask. The neural network's validation error was then
   about 3,000 ppm. Features were made relative to the gap anchors, and
   training masks hide one gap length at a time.
4. **Integrator order test.** An RK4 convergence test first showed order 2. The
   cause was the reference solution: linearly interpolating the occupancy
   *input* introduced an O(h²) error. The test now compares against the exact
   solution for the same input.
5. **Equation-error parameter fit.** The finite-difference least-squares fit
   gave a 19-hour room time constant. This is the errors-in-variables bias
   caused by differentiating noisy data, not a coding error. The
   simulation-based (output-error) fit is used instead, and both are shown.
6. **Optimistic claims in draft text.** Several draft conclusions (e.g.
   "averaging anchors clearly helps short gaps", "the profile halves the
   error") were not supported by the executed numbers and were rewritten.
7. **Conclusions written before the test run.** The final recommendation in
   notebook 06 was drafted from validation results ("use the ridge correction
   for long gaps"). The test period contradicted it: the validated hybrid was
   no better than linear interpolation there. The text was rewritten from the
   test numbers, and the distribution shift that caused it is analysed in
   notebook 06 §2. Every other notebook's conclusions were also re-checked
   against the executed outputs, and several overstated claims were corrected
   (e.g. the ODE fill is not better than linear for gaps of 6 h or less).

## 4. Student reflection (to be written by the student)

Prompts taken from the brief. Answer each in your own words:

- Which tools did you use, and what for? Roughly what share of the code and
  text came from AI?
- Why did you use AI for those tasks rather than doing them yourself?
- How did you check the AI's output? Point to specific tests, plots or numbers
  that you inspected.
- Which important decisions did *you* make? For example: the choice of topic,
  sensor and splits, accepting or rejecting a method, the final policy.
- Which AI errors or limitations did you notice (see section 3), and which
  ones did you catch yourself?
- How did using AI change your understanding of the methods, and your
  workflow? What would you do differently?

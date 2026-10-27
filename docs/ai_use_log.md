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

## 4. Student reflection

I used Anthropic’s Claude to aid me in completing this project. It had a high level of contribution as it wrote the code (the sensorfill package, the notebooks, tests and pushed them to the GitHub repository), all under my direction. It ran the analyses, presented their results, drafted this report and collated the references and appendices. Google Colab provided the T4 GPU that was used for training the neural networks. I used it because this project required several tools and skills which I was very unfamiliar with and the project took place over a very busy portion of the semester thus using the AI allowed me to make more use of the limited time I had to complete the project.

I directed by choosing the project dataset, problem and required methods from across the unit’s teachings. I also chose to use Google Colab to test due to my laptop malfunctioning throughout the project. I conversed with the AI to explain my initial ideas and then use its knowledge to bring them to life while also understanding the engineering reasons as to why some of my ideas were correct while others were not. I reviewed and accepted each of Claude’s edits to the code.

The outputs of the AI were checked using 87 automated tests which compared each numerical method against exact answers. I also looked at the figures and the fill plots. I raised any questions or uncertainties I had with its outputs and it often explained its working to me which I would then correct if I deemed it was doing something wrong.Throughout various stages of the project the AI averaged occupancy incorrectly and wrote some code that hid gaps but could now find any valid 3 hour block. It also wrote a recommendation before the test that was then disproved by said test. When it explained the code it claimed to flag long gaps however in reality it only filled them. These errors could be fixed by talking with the AI and making it understand its error or sometimes required manual investigation and mending.

The use of Claude allowed me to produce a project that was significantly more complex than what I could have produced in the same time without it. It allowed me to implement ideas and features very quickly without necessarily needing to learn the coding methods to do so. However, it also supported my understanding as I was able to get Claude to explain each step, each system or reiterate something from the very start of the project upon request and in a form that was formatted for my preferred method of learning. As it could tailor its responses to me and my knowledge I was able to gain a lot from it as it explained the systems or their steps that I did not properly grasp. It improved the workflow significantly as I could direct Claude on what I wanted to implement and what outputs I wanted from it and then it would explain how it would be able to add it to the project and how it worked as well.

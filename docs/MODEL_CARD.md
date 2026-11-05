# Model card and forecast limits

**Intended estimate:** the ordinary grade-11 second-stage score boundary for
qualifying for the 2026/27 DANO final, assuming a 100-point total.

**Recommendation:** 62.5, approximately 63. The prespecified robust ensemble
predicts 62.48; the direct and two-stage ensembles predict 62.05 and 62.88.
The best individual model in supplementary validation, frozen RuBERT plus
LightGBM, predicts 68.46. Every individual prediction is retained.

**Why the estimate is defensible:** all four comparable official seasons are
used; the incompatible legacy cutoff is excluded; the grade-11 historical
scores are 66, 70, 53 and 61; the main ensembles land near their mean; and
whole-year validation, a forward check and year-cluster uncertainty are
reported. The final choice does not optimize blend weights on test errors.

**What is not established:** precise forecasting skill, superiority of any
architecture, or a narrow prediction interval. The data contain only four
independent comparable seasons. Although the best direct text experiment has
lower supplementary error, its descriptive advantage interval spans zero and
the architecture was one of several examined.

**Uncertainty:** [43.2,81.8] is an assumption-dependent 90% Student-t
future-observation range under stationary, approximately Gaussian independent
year effects. It is not a validated coverage statement for DANO's changing
selection process. A nontrivial distribution-free 90% range cannot be
calibrated from four years; [0,100] is the conservative score-scale range.
The spread of fitted models is not used as a confidence interval.

**Distribution shift:** the supplied printout has 30 problems, compared with
15/17/20/20 in the comparable historical papers. No item weights or attached
datasets accompany it. Some formulas and graphs are images omitted from
native text. The semantic estimate therefore sees an incomplete rendering of
mathematical content, and set difficulty uses equal physical-problem weights.

**Confounding:** the cutoff depends on participant preparation, the group
invited to stage two, proctoring/appeals, the number of final places and
organizer decisions. The archive does not provide these as reliable,
forecast-time predictors. Observed difficulty is conditional on each year's
cohort, not intrinsic item complexity. Individual statement variants do not
have separately published outcome labels.

**Compute:** all fits and inference are local, CPU, four threads. Tree budgets
are about 1,000 iterations; the frozen transformer head trains for three
epochs. Full encoder fine-tuning is deliberately not reported as a meaningful
experiment on twelve cutoff labels. Source data, configurations and measured
outputs distinguish observed results from forecasts.

**Next useful evidence:** verified 2026 point weights and task attachments;
additional comparable seasons or external competitions with compatible
outcomes; forecast-time entrant/qualification-capacity information; and a
future untouched season for prospective evaluation. Artificial difficulty
labels or additional training epochs do not replace this missing evidence.

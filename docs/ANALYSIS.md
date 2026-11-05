# Interpretation of the experiments

## Difficulty fits overfit despite having many scoring rows

In the primary diagonal folds, at iteration 1,000, average training RMSE for
the four difficulty predictors is approximately 1.2–1.9 points. Average
validation RMSE remains 27.7–32.3. This is a substantial generalization gap,
not a successful fit with a minor calibration problem. Averaging a cohort's
many participant scores estimates its problem labels precisely, but there
are still only a few distinct annual problem sets, with different cohorts.
Three textual variants of one problem do not supply three independent labels.

Frozen RuBERT improves held-out difficulty MAE relative to the lexical
features: approximately 22.5–23.2 versus 25.6–26.1 points. It does not make the
problem-level estimates accurate enough to identify a stable final cutoff.
Early difficulty curves plateau or worsen while training loss continues to
drop. That is evidence against simply increasing training duration. The
fixed 1,000-iteration benchmark is retained as specified; the outer curves
are not used retrospectively to select an apparently optimal early stop.

## The calibration problem is even smaller

The primary protocol leaves one comparable cutoff season, so the calibrator
has three grade observations and no independent second cutoff year. Grade
and predicted difficulty can vary together without revealing a transferable
between-year relationship. Year-excluded difficulty predictions also come from
a smaller training history than the final first-stage fit; this ordinary
stacking training-size shift becomes pronounced with only one or two years.
Cross-fitting prevents leakage, but it cannot eliminate this variance or
identify missing information.

The two-stage ensemble has primary cutoff MAE 22.81 and supplementary MAE
9.66. Its supplementary custom loss is 35.48 versus 24.74 for the grade mean.
The descriptive paired year-bootstrap difference is positive even at the
95% interval's lower end (about +1.93 to +21.35 custom-loss units). This
supports the limited conclusion that explicit difficulty modeling did not
help this fixed benchmark. It does not establish that a well-supervised
difficulty pipeline with many more seasons could never help.

## Direct text models show a promising but unstable signal

Frozen RuBERT plus LightGBM achieves supplementary MAE 6.65, RMSE 7.17 and
custom loss 18.28. The grade mean has MAE 7.78, RMSE 9.08 and custom loss
24.74. However, the model is one of several examined. Its custom-loss
differences relative to the baseline, averaged across grades within each
season, are approximately +8.83, -23.51, -23.18 and +12.05. It improves two
seasons and worsens two. Resampling the four whole seasons gives a descriptive
95% difference interval of roughly -23.35 to +10.44, which includes no benefit.

The ranking changes under forecast-direction validation: frozen RuBERT plus
ridge has the best forward custom loss (17.87; MAE 6.41), while RuBERT plus
LightGBM has forward MAE 9.47. The forward test has only six cases from two
seasons, so it is also insufficient for a decisive model choice. Together,
these checks show that the supplementary winner is not established as a
reliable future-year winner.

In the severe diagonal folds, CatBoost with lexical and semantic inputs makes
identical cutoff predictions: only one training paper exists, so its text
features are constant across the three grades. LightGBM's minimum leaf size
prevents a useful split with three rows, causing predictions to collapse to
a mean. The frozen residual head starts at the grade prior and cannot learn
a year residual when that one training season has zero residual by definition.
These outcomes are transparent consequences of the sample size, not hidden
evidence that text encodings are equivalent in general.

## Feature importance is diagnostic, not semantic validation

In the final direct RuBERT-LightGBM fit, approximately 64.4% of normalized gain
belongs to embedding dimensions and 35.6% to grade. An embedding can distinguish
the few training papers without learning a causal notion of task difficulty.
The individual embedding coordinates are not named human concepts. Their
grouped importance must not be presented as evidence that the model
understands particular mathematical skills. Other hand-built indicators have
little or no gain in that fit. Importance comparisons across correlated
features and different tree algorithms are descriptive.

## Why the final point stays near 63

The historical grade-11 scores are 66, 70, 53 and 61, with mean 62.5. Fitting
the complete archive gives direct and two-stage ensemble forecasts of 62.05
and 62.88, and the fixed robust blend gives 62.48. The best supplementary
individual forecasts 68.46. A training-only nested choice between the grade
mean and the fixed blend favors the grade mean; the report does not relabel
that baseline as the best individual validation architecture.

Agreement near 63 is a useful sensitivity result. It is not independent
confirmation: all models use the same four cutoff seasons and many share
the same features. The defensible forecast is therefore **about 63**, with a
wide uncertainty range, rather than a claim of half-point accuracy.

The 90% Student-t future-observation reference range, approximately 43–82,
assumes stationary, independent Gaussian year effects. Participant ability,
selection capacity and the new 30-problem format may violate those assumptions.
No narrower empirically calibrated interval can be supported here, and a
distribution-free 90% statement would be uninformative on four years alone.

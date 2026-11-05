# Measured results

Generated from the checked local benchmark outputs. Lower loss is better.

**2026/27 grade 11 recommendation: 62.5/100 (approximately 63).** The fixed ensemble predicts 62.48. The assumption-dependent reference range is 43.2–81.8. A nontrivial distribution-free 90% coverage claim is unsupported by four seasons.

The final selection rule compares the grade mean and the prespecified robust ensemble on training-only inner year folds. The grade mean wins on all four available years. This preserves a conservative recommendation even though an individual RuBERT-LightGBM experiment has the lowest supplementary validation loss.

## Primary: distinct year-and-grade triples

| model                      |   power_1_5 |    mae |   rmse |   bias |   max_abs_error |
|:---------------------------|------------:|-------:|-------:|-------:|----------------:|
| direct_boosting_ensemble   |      32.085 |  8.333 | 11.453 |  0.004 |          23.465 |
| direct_rubert_catboost     |      32.237 |  8.971 | 11.130 |  0.007 |          21.930 |
| direct_tfidf_catboost      |      32.237 |  8.971 | 11.130 |  0.007 |          21.930 |
| direct_ensemble            |      32.336 |  8.336 | 11.545 |  0.002 |          23.759 |
| direct_rubert_ridge        |      33.136 |  8.528 | 11.757 | -0.000 |          24.348 |
| direct_tfidf_ridge         |      33.136 |  8.528 | 11.757 | -0.000 |          24.348 |
| baseline_mean              |      34.320 |  8.778 | 12.030 | -0.000 |          25.000 |
| direct_rubert_lightgbm     |      34.320 |  8.778 | 12.030 | -0.000 |          25.000 |
| direct_tfidf_lightgbm      |      34.320 |  8.778 | 12.030 | -0.000 |          25.000 |
| nested_selected            |      34.360 | 10.000 | 11.113 |  0.000 |          20.000 |
| direct_rubert_head_3ep     |      34.360 | 10.000 | 11.113 |  0.000 |          20.000 |
| baseline_grade_median      |      34.360 | 10.000 | 11.113 |  0.000 |          20.000 |
| baseline_grade_mean        |      34.360 | 10.000 | 11.113 |  0.000 |          20.000 |
| baseline_latest_grade      |      34.360 | 10.000 | 11.113 |  0.000 |          20.000 |
| robust_ensemble            |      38.410 |  9.986 | 12.432 |  5.267 |          24.068 |
| all_text_ensemble          |      59.227 | 14.323 | 15.993 | 10.534 |          28.136 |
| difficulty_rubert_lightgbm |      90.136 | 18.770 | 21.159 | 18.659 |          34.177 |
| difficulty_tfidf_lightgbm  |     116.916 | 22.109 | 25.422 | 17.263 |          47.000 |
| difficulty_ensemble        |     120.262 | 22.806 | 25.565 | 21.066 |          39.257 |
| difficulty_rubert_catboost |     131.868 | 23.391 | 27.981 | 23.391 |          53.000 |
| difficulty_tfidf_catboost  |     178.937 | 30.324 | 32.887 | 24.950 |          47.941 |

## Supplementary: one case, whole-year purge

| model                      |   power_1_5 |    mae |   rmse |   bias |   max_abs_error |
|:---------------------------|------------:|-------:|-------:|-------:|----------------:|
| direct_rubert_lightgbm     |      18.282 |  6.649 |  7.171 |  0.055 |           9.943 |
| baseline_latest_grade      |      21.927 |  7.583 |  8.098 | -4.083 |          15.000 |
| direct_boosting_ensemble   |      22.633 |  7.387 |  8.482 | -0.730 |          13.083 |
| baseline_grade_mean        |      24.736 |  7.778 |  9.080 |  0.000 |          15.000 |
| baseline_grade_median      |      25.764 |  8.333 |  9.101 |  0.167 |          15.000 |
| direct_rubert_head_3ep     |      25.835 |  7.993 |  9.327 | -0.959 |          14.294 |
| direct_rubert_catboost     |      26.360 |  8.137 |  9.376 |  0.113 |          13.912 |
| nested_selected            |      26.400 |  8.086 |  9.510 |  0.308 |          15.000 |
| robust_ensemble            |      26.499 |  7.921 |  9.582 | -0.398 |          14.038 |
| direct_ensemble            |      26.632 |  8.338 |  9.361 | -1.563 |          14.274 |
| direct_tfidf_catboost      |      26.793 |  8.038 |  9.577 | -0.259 |          13.795 |
| all_text_ensemble          |      30.402 |  8.988 | 10.351 | -0.797 |          15.789 |
| direct_rubert_ridge        |      30.882 |  9.224 | 10.396 | -0.792 |          19.148 |
| difficulty_rubert_lightgbm |      31.298 |  9.550 | 10.257 | -4.557 |          14.940 |
| difficulty_rubert_catboost |      31.446 |  9.167 | 10.570 |  0.402 |          16.097 |
| baseline_mean              |      31.729 |  9.333 | 10.644 |  0.000 |          19.111 |
| difficulty_ensemble        |      35.484 |  9.655 | 11.619 | -0.030 |          17.339 |
| direct_tfidf_lightgbm      |      35.928 | 10.487 | 11.239 | -2.831 |          16.072 |
| difficulty_tfidf_lightgbm  |      39.800 | 10.487 | 12.618 |  0.710 |          20.207 |
| difficulty_tfidf_catboost  |      53.085 | 12.518 | 15.437 |  3.324 |          26.971 |
| direct_tfidf_ridge         |      59.042 | 14.055 | 16.050 | -5.665 |          28.765 |

## Forecast-direction sensitivity

| model                      |   power_1_5 |    mae |   rmse |    bias |   max_abs_error |
|:---------------------------|------------:|-------:|-------:|--------:|----------------:|
| direct_rubert_ridge        |      17.869 |  6.411 |  7.254 |   2.932 |          12.924 |
| difficulty_tfidf_lightgbm  |      27.440 |  8.128 |  9.898 |   6.533 |          16.587 |
| difficulty_rubert_catboost |      30.035 |  9.328 |  9.957 |   9.328 |          13.778 |
| robust_ensemble            |      30.320 |  8.432 | 10.647 |   7.737 |          15.929 |
| difficulty_ensemble        |      31.185 |  8.693 | 10.771 |   7.921 |          16.939 |
| direct_rubert_lightgbm     |      31.235 |  9.468 | 10.246 |   3.090 |          13.329 |
| all_text_ensemble          |      31.585 |  9.169 | 10.695 |   6.335 |          16.858 |
| baseline_grade_mean        |      32.464 |  9.139 | 10.990 |   9.139 |          15.500 |
| nested_selected            |      32.464 |  9.139 | 10.990 |   9.139 |          15.500 |
| difficulty_rubert_lightgbm |      32.523 |  9.740 | 10.551 |   3.377 |          14.809 |
| direct_boosting_ensemble   |      33.232 |  9.375 | 11.126 |   7.313 |          16.626 |
| direct_rubert_catboost     |      33.410 |  9.311 | 11.145 |   9.311 |          16.825 |
| direct_ensemble            |      33.747 |  9.678 | 11.015 |   4.749 |          16.777 |
| direct_tfidf_catboost      |      34.887 |  9.319 | 11.625 |   9.187 |          18.008 |
| baseline_grade_median      |      35.126 | 10.083 | 11.293 |  10.083 |          15.500 |
| baseline_mean              |      39.478 | 10.546 | 12.480 |   9.139 |          21.833 |
| direct_tfidf_lightgbm      |      43.503 | 11.445 | 13.142 |   7.663 |          18.340 |
| baseline_latest_grade      |      48.723 | 12.667 | 13.940 |   5.667 |          20.000 |
| difficulty_tfidf_catboost  |      52.761 | 12.582 | 15.099 |  12.446 |          22.582 |
| direct_rubert_head_3ep     |      79.544 | 15.331 | 20.620 | -15.001 |          29.294 |
| direct_tfidf_ridge         |      90.740 | 19.759 | 20.610 |  -3.690 |          28.765 |

## 2026/27 grade 11 forecasts

| model                      |   prediction |
|:---------------------------|-------------:|
| baseline_mean              |       56.000 |
| baseline_grade_mean        |       62.500 |
| baseline_grade_median      |       63.500 |
| baseline_latest_grade      |       61.000 |
| direct_tfidf_catboost      |       60.587 |
| direct_tfidf_lightgbm      |       61.996 |
| direct_tfidf_ridge         |       58.886 |
| direct_rubert_catboost     |       61.152 |
| direct_rubert_lightgbm     |       68.464 |
| direct_rubert_ridge        |       61.239 |
| direct_rubert_head_3ep     |       63.058 |
| direct_boosting_ensemble   |       63.050 |
| direct_ensemble            |       62.054 |
| difficulty_tfidf_catboost  |       61.732 |
| difficulty_tfidf_lightgbm  |       58.984 |
| difficulty_rubert_catboost |       66.871 |
| difficulty_rubert_lightgbm |       63.913 |
| difficulty_ensemble        |       62.875 |
| all_text_ensemble          |       62.464 |
| robust_ensemble            |       62.482 |

## Paired uncertainty relative to the grade mean

These descriptive 95% bootstrap intervals resample entire years, keeping the three grades together. There are only four clusters; the intervals do not account for trying multiple architectures.

| model                    |   mean_loss_difference |   bootstrap_95_low |   bootstrap_95_high |
|:-------------------------|-----------------------:|-------------------:|--------------------:|
| direct_rubert_lightgbm   |                 -6.454 |            -23.345 |              10.437 |
| direct_boosting_ensemble |                 -2.103 |             -5.278 |               1.072 |
| baseline_grade_mean      |                  0.000 |              0.000 |               0.000 |
| robust_ensemble          |                  1.763 |             -1.250 |               5.212 |
| direct_ensemble          |                  1.896 |             -2.303 |               6.295 |
| difficulty_ensemble      |                 10.748 |              1.934 |              21.352 |

## Systematic errors

| model                    |   2022 |   2023 |   2024 |   2025 |
|:-------------------------|-------:|-------:|-------:|-------:|
| baseline_grade_mean      |  -3.56 | -12.00 | +12.44 |  +3.11 |
| difficulty_ensemble      |  -4.09 | -14.30 | +17.15 |  +1.12 |
| direct_boosting_ensemble |  -3.47 | -10.61 | +11.80 |  -0.64 |
| direct_ensemble          |  -2.92 | -11.36 | +12.67 |  -4.65 |
| direct_rubert_lightgbm   |  +5.86 |  -6.81 |  +7.55 |  -6.38 |
| robust_ensemble          |  -3.53 | -12.41 | +13.68 |  +0.67 |

## Observed problem difficulty

|   year |     9 |    10 |    11 |
|-------:|------:|------:|------:|
|   2021 | 81.67 | 79.24 | 75.69 |
|   2022 | 60.02 | 54.36 | 49.12 |
|   2023 | 51.94 | 44.24 | 41.18 |
|   2024 | 67.26 | 61.25 | 57.18 |
|   2025 | 52.46 | 45.49 | 38.87 |

Difficulty is a cohort-dependent partial-credit statistic. It is not an intrinsic psychological scale or an individual solve probability. Variant-specific outcomes are unavailable, so statement variants share a pooled problem-family label.

## Held-out problem-difficulty quality

| model                      |   n_problems |   power_1_5 |    mae |   rmse |   bias |   max_abs_error |
|:---------------------------|-------------:|------------:|-------:|-------:|-------:|----------------:|
| difficulty_rubert_catboost |          216 |     126.331 | 22.481 | 27.528 | 12.387 |          67.048 |
| difficulty_rubert_lightgbm |          216 |     131.949 | 23.232 | 28.286 | 12.417 |          70.547 |
| difficulty_tfidf_catboost  |          216 |     155.884 | 25.588 | 31.857 | 20.179 |          76.290 |
| difficulty_tfidf_lightgbm  |          216 |     160.184 | 26.079 | 32.373 | 20.143 |          73.809 |

## Interpretation

- The strongest supplementary individual model is frozen RuBERT + LightGBM. Its advantage over the grade baseline is uncertain under whole-year resampling.
- Explicit difficulty modeling does not improve this benchmark. Calibrating difficulty into a cutoff is especially unstable when the primary protocol leaves one cutoff year for training.
- The cutoff is determined by participant ability, selection rules and final capacity as well as problem content. Those unobserved factors can produce large common errors across grades.
- A direct model cannot learn a season-level text effect from only one training season. In diagonal folds, lexical and semantic CatBoost forecasts coincide, and LightGBM may find no legal split with three rows.
- The 2026 paper has 30 problems, versus 15/17/20/20 historically. Its point weights and data attachments are missing. Predictions extrapolate beyond the observed problem-count range.
- The three-epoch frozen head does not justify full-encoder fine-tuning on twelve labels. No model architecture is established as a reliable high-precision forecaster.

## Uncertainty

The grade-11 cutoffs are 66, 70, 53 and 61. Their mean is 62.5. Under independent, stationary, approximately Gaussian year effects, a 90% future-observation interval is mean ± t(0.95, 3) × sample_sd × sqrt(1+1/4), yielding approximately [43.2, 81.8]. This is a forecast interval for a new year, not a confidence interval for the historical mean. Stationarity is not established. With four calibration years, the 90% finite-sample conformal order statistic would require rank five; a useful distribution-free guarantee is unavailable. The documented conservative full-score range is [0,100]. Agreement among the fitted models is not evidence of narrow coverage.

## Figures

### Headline grade-11 forecast table

![Headline grade-11 forecast table](../figures/grade11_forecast_table.png)

### Complete training and prediction pipeline

![Complete training and prediction pipeline](../figures/end_to_end_pipeline.png)

### Historical targets and forecast

![Historical targets and forecast](../figures/historical_cutoffs.png)

### All architectures and metrics

![All architectures and metrics](../figures/architecture_comparison.png)

### Predicted versus actual

![Predicted versus actual](../figures/predicted_vs_actual.png)

### Error distributions

![Error distributions](../figures/error_distributions.png)

### Cutoff learning curves

![Cutoff learning curves](../figures/learning_curves_cutoff.png)

### Difficulty learning curves

![Difficulty learning curves](../figures/learning_curves_difficulty.png)

### Frozen transformer head

![Frozen transformer head](../figures/frozen_head_learning.png)

### Feature importance

![Feature importance](../figures/feature_importance.png)

### Problem difficulty

![Problem difficulty](../figures/problem_difficulty_heatmap.png)

### Difficulty by year and grade

![Difficulty by year and grade](../figures/difficulty_by_year_grade.png)

### Ensemble comparison

![Ensemble comparison](../figures/ensemble_comparison.png)

### Paired year uncertainty

![Paired year uncertainty](../figures/paired_year_uncertainty.png)

### Year and grade errors

![Year and grade errors](../figures/systematic_errors.png)

### Validation split design

![Validation split design](../figures/validation_design.png)

### Individual final forecasts

![Individual final forecasts](../figures/forecast_comparison.png)

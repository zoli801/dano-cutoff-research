# Validation and leakage prevention

## Primary protocol: distinct years and grades

Twelve cutoff observations come from four seasons. A year-grade identifier
alone is an insufficient grouping key: grades share the same paper and the
year's selection environment. The primary folds are fixed cyclic triples:

| Fold | Validation cases | Cutoff training season |
|---|---|---|
| 1 | 2022 grade 9; 2023 grade 10; 2024 grade 11 | 2025 |
| 2 | 2023 grade 9; 2024 grade 10; 2025 grade 11 | 2022 |
| 3 | 2024 grade 9; 2025 grade 10; 2022 grade 11 | 2023 |
| 4 | 2025 grade 9; 2022 grade 10; 2023 grade 11 | 2024 |

Years refer to the season's starting year. Every case is validated exactly
once. Each fold has three distinct years and grades in validation, three
cutoff training labels, and six purged companion-grade labels. No training
year occurs in validation. Auxiliary difficulty labels from 2021 are allowed;
they cannot reveal the held-out years' cutoff or item outcomes.

This follows the requested mixed-year/mixed-grade holdouts while additionally
purging shared-year companions. With so little data, it cannot test whether
direct text models have learned a general year effect: one unique training
paper provides no between-year training contrast.

## Supplementary whole-year purge

Each fold contains **one** validation case and purges its entire year. Three
seasons, nine cutoff labels, remain for training. There are twelve cases in
twelve folds, with identical training sets cached across a year's grades.
Cases from the same season remain correlated and do not count as independent
years. This offers more useful training variation while keeping the requested
restriction that no validation fold contains two grades from one year.

## Forward sensitivity

For each grade of 2024/25 and 2025/26, train only on strictly earlier cutoff
seasons, requiring at least two training seasons. Each validation fold again
contains one case. Six cases come from just two future-year clusters. These
results are especially noisy, but they expose retrospective cross-validation
that depends on learning from a later season.

## Isolation of every fitted operation

1. Workbooks supply observed item labels; held-out year labels never enter a
   difficulty or cutoff training fit.
2. TF-IDF vocabularies, inverse document frequencies and SVD fit only the
   relevant training problem texts. Future documents can be transformed but
   cannot update these fitted statistics.
3. Frozen RuBERT weights are pretrained externally, pinned and never
   outcome-adapted. Computing their fixed embeddings globally is allowed;
   supervised heads and downstream transforms still fit inside folds.
4. Numeric feature rules are fixed. Standardization and ridge fit only the
   corresponding training rows.
5. All statement variants and all grades of a validation year are excluded
   from training. Canonical duplicate text families are purged as well.
6. Difficulty-to-cutoff calibration uses year-excluded difficulty predictions
   for every training cutoff row. A year cannot supply its own first-stage
   target when creating those inputs.
7. Boosting budgets are fixed. Held-out learning curves are generated for
   diagnosis **after fitting**, never used for early stopping or parameter
   choice. Current-year outcomes are not available or used.
8. Ensemble weights are prespecified. A nested rule chooses only between a
   baseline and the fixed robust ensemble, using training years alone. With
   only one training cutoff season, it defaults to the baseline.

Assertions and tests check split-year disjointness, one validation year per
case, grade uniqueness inside diagonal folds, scored-column alignment,
variant sample weights, answer removal, absence of a held-out vocabulary
sentinel, exact embedding/text hashes and identical model validation cases.
The saved split manifest makes every selection reviewable.

## What validation cannot establish

Twelve case errors are not twelve independent experiments. Four seasons are
too few for stable model ranking, broad hyperparameter search or reliable
learning-curve-based model-size decisions. Primary folds overlap in training
and validation years across folds. Descriptive bootstrap intervals keep
grades together but still use only four season clusters and do not correct
for inspecting multiple models. The nested selected-model row is a separate
evaluation of that restricted selection rule; it is not the minimum error
picked across all reported rows.

A transformer pretrained on public web text could have encountered older
problem wording. This risk is unmeasured; no claim of controlled pretraining
decontamination is made. It has no access to this project's held-out cutoff
labels through supervised fitting. Shared topic vocabulary itself is not
treated as leakage: the task is to generalize semantic patterns across years.

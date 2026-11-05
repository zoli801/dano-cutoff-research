# Benchmark commitments

Recorded before any cutoff models were fitted.

The comparable target is the published minimum second-stage prizewinner score,
on the 100-point scale, for 2022/23 through 2025/26. The 2021/22 invitation
boundary was 102 out of 225 **across both stages**, independently verified from
the invitation status workbook. It is excluded from cutoff regression. Its
anonymous second-stage item outcomes can train the auxiliary difficulty model.

The primary protocol uses four cyclic folds. A fold validates grade 9, 10 and
11 from three different seasons and purges every grade of those three seasons.
Only one comparable cutoff season remains (three training labels). This is an
intentionally severe test; it cannot identify general text-to-year effects.
The supplementary protocol validates one year-grade at a time and removes the
entire year (nine training labels). An expanding-window sensitivity trains on
strictly earlier seasons. Every architecture uses the same cases and folds.

All boosting fits use 1,000 fixed iterations. No outer-label early stopping,
vocabulary fitting, SVD fitting, scaling, representation tuning, or ensemble
weight optimization is allowed. Frozen pretrained embeddings may be computed
once because the encoder has no DANO supervised fitting. It sees every token
through overlapping windows; it does not truncate a full exam to 512 tokens.

First-stage supervised difficulty fits use training years plus the legacy
2021 season, never any held-out cutoff year. Calibration features for a
training cutoff season are predicted by a difficulty model excluding that
entire season. Variant texts share a pooled year-grade-problem outcome and
receive total weight one per problem; variant copies are not independent
difficulty labels. Canonically duplicate problem texts in a test corpus are
purged from supervised difficulty training as an additional safeguard.

The fixed robust ensemble weights are 50% grade mean, 25% direct-model mean,
and 25% two-stage-model mean. Performance-ranked weights are not fitted on
reported test cases. A nested selection diagnostic chooses a baseline or this
fixed ensemble using only inner, year-purged validation; it must not use outer
labels. The final recommendation will consider this nested diagnostic and
forward sensitivity, and explicitly acknowledge the four independent years.

Full-encoder RuBERT fine-tuning on twelve cutoffs is not statistically
credible. The transformer experiment freezes RuBERT-tiny2 and compares
CatBoost, LightGBM, ridge, and a three-epoch trainable regression head. This
choice follows the user's requested small-data fallback.

Uncertainty must be labeled by its assumptions. Four years cannot support a
useful distribution-free 90% forecast interval: a conformal rank beyond the
available calibration years returns the full score range. Any narrower
Student-t or residual-based interval is an assumption-dependent sensitivity
range, not a validated coverage guarantee. No claim of high precision may
be based on these twelve correlated labels.

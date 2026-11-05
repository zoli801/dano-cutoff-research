# Models and representations

## Research question

Both approaches estimate a season-and-grade cutoff C from the complete
second-stage problem set S and grade g. Approach 1 adds supervised
problem-difficulty predictions as an intermediate representation; Approach 2
maps pooled text features directly to C. Comparing them asks whether the
extra difficulty supervision transfers to the cutoff, not whether an
in-sample task solver can fit its own labels.

## Continuous difficulty definition

Let x_ij be participant i's points on physical problem j and m_j its maximum.
For a year-grade cohort with n_j observed attempts:

$$
d_j=\operatorname{clip}\left(100\left[1-\frac1{n_j}
\sum_i\frac{x_{ij}}{m_j}\right],\;0.001,\;99.999\right).
$$

Higher values mean harder problems for that observed cohort. Partial credit
contributes continuously. The un-clipped sampling standard error is
`100 × sample_sd(x/m) / sqrt(n)`. This is not an intrinsic item-response-theory
parameter: participant preparation, selection and available time affect it.
No subjective LLM-generated scores are substituted for missing outcomes.

## Text features

**Lexical representation.** Concatenate word TF-IDF with 1–2-grams and
character-within-word TF-IDF with 3–5-grams; cap each vocabulary at 3,000.
Use sublinear term frequency, then a training-fold TruncatedSVD projection
with at most 24 components. Word features retain terminology; character
features tolerate Russian morphology and tokenization variation. SVD reduces
dimension before the tree models but can discard useful signal in tiny folds.

**Frozen semantic representation.** Use
[cointegrated/rubert-tiny2](https://huggingface.co/cointegrated/rubert-tiny2),
pinned to revision `e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae`.
It is a Russian-capable BERT encoder with three layers, hidden width 312 and
12 attention heads. The pinned model files and SHA-256 hashes are recorded.
All inference runs locally on CPU with four threads; no hosted embeddings API
or remote training is used.

Each statement is tokenized into windows of maximum length 256 and stride 32.
Masked mean pooling yields each window vector; a token-count-weighted window
average and L2 normalization produce the problem vector. Overlap tokens
contribute to both windows, a fixed approximation rather than exact
deduplicated token pooling. Every statement token is processed. Set pooling
includes every problem and all variants, instead of taking only a paper's
first transformer window. Because the encoder never fits DANO outcomes, its
embeddings can be cached before cross-validation without supervised leakage.

**Interpretable numeric features.** Sixteen per-problem features describe log
character/word count, digit fraction, mathematical-symbol fraction, question
count, mean word length, Latin fraction and nine keyword groups: probability,
statistics, hypothesis testing, regression, ML, visual interpretation, data
files, causality and economics. These fixed indicators are not hand-assigned
difficulty labels. Grade is an explicit input.

## Approach 1: difficulty then cutoff

CatBoost is selected for stable shallow symmetric trees and explicit leaf
regularization. It builds an additive predictor by sequentially improving
residual errors; small learning rates spread that fit across many weak trees.
This project supplies dense numerical representations, so it does not invoke
CatBoost's native string-text processing or invent categorical target encodings.
LightGBM supplies a different tree learner, using histogram-binned features and
leaf-wise growth bounded here by depth and leaf count. Its efficiency makes
the nested local fits inexpensive, while the leaf constraints and L2 penalty
limit variance. Their different split rules can yield complementary errors,
which motivates comparing both individual models and fixed ensembles.

For each of TF-IDF/SVD and frozen RuBERT, fit both:

| Parameter | CatBoost difficulty | LightGBM difficulty |
|---|---:|---:|
| Requested iterations | 1,000 | 1,000 |
| Learning rate | 0.03 | 0.03 |
| Tree depth | 3 | 3 |
| Leaf limit | implicit | 7 |
| L2 regularization | 20 | 20 |
| Minimum leaf samples | library default | 10 |
| Objective | RMSE | squared error |

Supervised training uses only allowed training seasons, plus auxiliary 2021
problem outcomes. Canonically identical test problem families are removed from
the training rows. No cross-year exact family duplicates occur in this
snapshot, but the safeguard is implemented. Three variant texts share one
pooled label's total sample weight.

For each year-grade, average predicted variant scores within the physical
problem. Form eight calibration inputs: mean expected score fraction on the
0–100 scale (`100 − mean difficulty`), difficulty standard deviation,
10th/50th/90th percentiles, minimum, maximum, and grade minus 10. Every physical
problem has equal weight because the current printout provides no point
weights.

Fit StandardScaler + ridge (`alpha=10`) from these inputs to the training
cutoffs. Crucially, each training year's calibration inputs come from a
difficulty model fitted **without that whole year**. For an outer test year,
refit the difficulty predictor on all permitted training years, then use the
trained calibrator. A calibration fit with one cutoff year cannot separately
identify grade effects and year-level problem difficulty; its poor diagonal
performance reflects that limitation.

## Approach 2: cutoff from pooled text

Pool mean semantic vectors, numeric-feature means and standard deviations,
physical-problem count and grade minus 10 over the complete set. This gives 58
TF-IDF/SVD inputs or 346 RuBERT inputs when the full 24-component lexical
projection is available.

For each representation, compare CatBoost, LightGBM and ridge:

Ridge minimizes squared error plus an L2 coefficient penalty. Standardizing
inside each fold gives its feature penalties a consistent scale and keeps
the regression defined when there are more semantic features than cutoff
labels. It is a lower-variance alternative to flexible trees, though excessive
shrinkage can underfit grade differences and cannot manufacture more seasons.

| Parameter | CatBoost direct | LightGBM direct | Ridge direct |
|---|---:|---:|---:|
| Requested iterations | 1,000 | 1,000 | closed-form solver |
| Learning rate | 0.02 | 0.02 | — |
| Depth / leaves | 2 | depth 2 / 3 leaves | — |
| L2 regularization | 20 | 10 | alpha 20 |
| Minimum leaf samples | library default | 2 | — |
| Scaling | none | none | training-only standardization |

When a tiny fold admits no valid split, LightGBM may naturally stop adding
trees despite the 1,000-round budget. This is underidentification rather than
outer-label early stopping. No claim is made that every such fit has 1,000
nontrivial trees.

The frozen-RuBERT regression-head experiment fits a zero-initialized linear
head to grade-prior residuals, scaled by 10. It uses AdamW, learning rate 0.01,
weight decay 0.1, gradient clipping at one, full-batch updates and three epochs.
The grade prior is estimated from the outer training labels. Encoder weights
remain frozen; this is not full RuBERT fine-tuning. Zero-initialization starts
at a meaningful grade baseline instead of fitting an arbitrary random score
with only a handful of updates.

## Baselines and ensembles

Baselines are the global mean, per-grade mean, per-grade median and the latest
available training season's grade cutoff. They expose whether text adds
information beyond persistent grade differences or a recent-season prior.

- Direct boosting ensemble: equal mean of the four direct tree models.
- Direct ensemble: equal mean of four direct tree models plus two ridge models.
- Difficulty ensemble: equal mean of four complete two-stage pipelines.
- Both approaches: equal mean of direct and difficulty ensembles.
- Fixed robust ensemble: 50% per-grade mean + 25% direct + 25% difficulty.

The three-epoch head is reported separately rather than adding an additional
unjustified vote to the direct ensemble. There is no validation-fitted stacking
or optimized blend weight. The final recommendation's restricted nested rule
chooses the grade mean or fixed robust ensemble using only inner year folds.
The best individual model remains visible, with a warning that selecting it
after examining multiple validation rows is optimistic.

## Metrics and losses

$$
P_{1.5}=\frac1N\sum_i|e_i|^{1.5},\quad
MAE=\frac1N\sum_i|e_i|,\quad
RMSE=\sqrt{\frac1N\sum_i e_i^2},\qquad e_i=\hat C_i-C_i.
$$

Power loss increases more steeply than absolute error; a 10-point miss costs
31.623 rather than 10. RMSE remains useful for sensitivity to especially large
misses. Mean signed error detects systematic over/underprediction. Boosters
fit squared-error objectives with fixed iteration counts; power loss is the
comparison and nested-selection criterion. All cutoff predictions are bounded
to [0,100] under the stated score-scale assumption.

# Reproducibility

Run the commands in README.md from the repository root. `uv sync --locked`
recreates the Python 3.12 environment. Poppler is an external system
dependency. Dependencies, model revision, configuration, inputs and outputs
have separate version records:

| Record | Purpose |
|---|---|
| `uv.lock` and `.python-version` | Python package resolution and interpreter |
| `configs/benchmark.json` | Seed, thread budget, model parameters and ensembles |
| `data/processed/source_manifest.json` | Official URLs and raw file checksums |
| `data/processed/pdf_extraction_audit.json` (local, ignored) | PDF hashes, counts and limitations |
| `data/processed/item_alignment.json` | Physical-problem/scoring-column mapping |
| `data/processed/workbook_audit.json` | Anonymous join and exclusion statistics |
| `data/processed/transformer_provenance.json` | Exact frozen encoder revision and hashes |
| `data/interim/embedding_order.json` | Text hashes and cached embedding row order |
| `results/splits.json` | Every training, validation and purged case |
| `results/nested_selections.json` | Training-only model-selection decisions |
| `results/run_metadata.json` | Completed run duration and cutoff input hash |
| `results/replay_check.json` (local, ignored) | Full saved-model replay verification and input hash |
| `results/replay_summary.json` | Published replay counts and numeric agreement |

## Local input

Copy the user-provided 18-page PDF to
`data/raw/current/2026_second_stage_user.pdf`. The benchmark asserts 30 problem
headings. This input is not supplied by a fabricated public link or published
with the GitHub repository. The original filename is
`Второй_отборочный_этап.pdf`; its hash is the durable identifier.

## Local compute and cached replay

The only network steps are public source collection and the first pretrained
weight download. `embeddings` uses local-only model loading thereafter.
The fixed encoder's weights never update. Every booster and regression head
fits locally; no Kaggle, cloud GPU or hosted inference job is launched.

Difficulty model caches depend on training problem identities, all statement
text/label/model-provenance hashes and the fixed configuration. Training
subsets are reused across nested folds when identical. Fitted models and
cached embeddings remain in ignored local directories. Delete `models/cache`
to force a clean fit. Cache reuse is not treated as a new independent run.

After training, `uv run python -m dano.replay` reloads the fitted direct models,
all four difficulty predictors and their calibrators. It recomputes every
grade-11 model forecast and asserts agreement with the saved table to
`1e-8`. It performs no fitting or network call. `--pdf /path/to/file.pdf`
additionally checks that the exact supplied PDF matches the forecast input;
a different PDF requires re-extraction and embedding regeneration.

Figures and documentation tables derive from measured CSV/JSON outputs.
`uv run python -m dano.report` regenerates 17 PNG/SVG figures and the local
HTML report. `uv run pytest -q` checks metric arithmetic, fold construction,
answer removal, score-column alignment, variant weights, fold vocabulary
isolation, prediction cases and available artifacts. Some corpus-dependent
checks skip before data/embeddings are prepared.

`uv run python -m dano.overview` regenerates only the bright grade-11 forecast
table and the overall pipeline diagram from the saved result files, without
retraining or changing any predictions.

Reproducing floating-point predictions on another architecture/library stack
may introduce small differences. The fixed seed, CPU budget and pinned
environment reduce variation but are not a cross-platform bitwise guarantee.
The source archive can change; the manifest snapshot and checksums identify
the files actually used. `collect --refresh-pages` explicitly refreshes page
metadata; cached downloaded file contents are otherwise reused.

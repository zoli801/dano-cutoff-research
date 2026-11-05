# Dataset collection and preprocessing

## Sources and coverage

The entrypoints are the HSE [past-task archive](https://dano.hse.ru/library/past_tasks),
[results page](https://dano.hse.ru/result), [demonstrations](https://dano.hse.ru/demo),
and their linked project-data pages. The official archive currently lists five
completed seasons. All 44 linked archive problem PDFs, including all variants
of both qualifying stages and final individual/project rounds, are downloaded.
Four additional demonstration problem PDFs are retained. A source index and a
JSON manifest preserve every URL, season, stage, file kind, retrieval time,
byte count, local path and SHA-256.

The manifest has 136 entries: 131 downloaded file assets and five official
HTML data-page references. Solutions, rubrics, result workbooks and directly
linked project attachments are also collected. The collector repairs three
official `href` values that accidentally include literal ` target=` text;
original and repaired URLs remain visible. It does not invent an unpublished
season, bypass a login, or treat a broken link as downloaded. Supplementary
assets over 200 MB would be recorded as failed; no such unresolved failure
remains in this snapshot. Downloaded source files total about 528 MiB locally.

Only second-stage statements enter the cutoff benchmark. Earlier-stage and
final-round problems are archived for completeness. Their outcome measures
have different selection mechanisms and are not silently pooled as current
cutoff labels.

## Target definition and the 2021 regime change

For 2022/23–2025/26, the target is the published minimum second-stage
prizewinner score in the grade the participant **entered**, on the 100-point
scale. Winners and prizewinners qualify for the final, as the
[official FAQ](https://dano.hse.ru/faq) describes. Previous final-round diploma
holders can have a separate qualification route; this project predicts the
ordinary second-stage score boundary.

The 2021/22 workbook uses first-stage maximum 98, second-stage maximum 127,
and combined maximum 225. For grades 9, 10 and 11, the lowest observed invited
total is 102 and the highest non-invited total is 101. Invitations depend on
the combined total. Neither 102 nor a rescaled 102/225 is a valid second-stage
cutoff label. The observed boundary is preserved in
`legacy_2021_invitation_bounds.csv`; only twelve later cutoff labels are fitted.

The current 2026/27 second-stage threshold is absent from the collected results
snapshot. Current first-stage thresholds are archived but never used as
second-stage targets or as a disguised 2026 answer.

## Empirical difficulty supervision

For each physical problem, year and entered grade:

1. Read the maximum points from the workbook's scoring header.
2. Join the scoring sheet to the overall results sheet using participant names
   or the legacy participant code, locally and in memory.
3. Require an unambiguous grade. Do not fuzzy-match ambiguous identities.
4. Exclude invalid item scores, mismatched item totals and duplicate records.
5. Treat unanswered item fields as zero **only for a participant with a numeric
   overall attempt score**. Do not turn an absent participant into an attempt.
6. Sum scored subparts back into their physical PDF problem.
7. Calculate mean earned/max points and transform to difficulty as documented
   in METHODS.md, retaining sample size and standard error.

No names, schools, regions, participant codes or individual score records are
published. `workbook_audit.json` records anonymous join and exclusion counts.
There are 291 unique year–grade–physical-problem labels: 25, 15, 17, 20 and 20
problems across the five seasons, each with three grades. Cohort size is
hundreds per grade, but this does not create more independent cutoff years.

The workbook does not identify which statement variant each participant saw.
Consequently, each variant receives its problem family's pooled outcome.
This is meaningful supervision for a pooled family, not verified
variant-specific solve difficulty. Variant rows receive weight `1/V`, where V
is the number of variants for that problem, so they sum to one label's weight.

## Physical problems versus scoring columns

The 2022/23 workbook has 18 scoring columns but 15 physical problems. The
2023/24 workbook has 26 columns but 17 physical problems. Explicit alignments
in `item_alignment.json` map consecutive scoring subparts back to the PDF:
for example, the four 2-point parts of 2023 problem 7 sum to 8, and the seven
2-point parts of problem 16 sum to 14. Coverage assertions require every
scoring column exactly once and preserve the 100-point total. Partial-credit
standard errors are computed **after** summing participant subparts, retaining
their covariance.

The 2021 PDF says two late questions are worth one point while the workbook
headers assign two. Empirical score fractions follow the actual workbook
maximums and total 127; the cutoff regression excludes that assessment regime.

## Statement extraction and solution removal

Poppler `pdftotext -layout` preserves reading order. Explicit question headings
or monotone numbered headings define physical problems. Expected counts are
checked against the verified alignment; extraction stops with an error if a
problem is missing rather than shifting labels silently.

Some official PDFs include answers, `(+ )` correct-choice markers or both
answered and clean copies. The 2022 answered copy has all 15 problems, but one
clean copy omits a bank problem and renumbers the remainder. Clean statements
are matched by title when available; the missing problem comes from its
sanitized full copy. Legacy answers and explanations are cut after the answer
heading. Other answer lines, inline answer annotations, filled answer
expressions and correct-choice markers are removed. Automated checks reject
the retained answer markers. Original PDFs remain available for visual audit.

This is native PDF text extraction, not a claim of full multimodal recovery.
Diagrams and some mathematical glyphs in the user printout are raster images.
Their visual content is not encoded by the text models. The 18-page supplied
PDF was rendered and inspected; its 30 problem headings are extracted. It
contains neither item point weights nor the task data files. Current set
summaries therefore use equal physical-problem weights, consistently with
historical training summaries. A total score of 100 is an explicit forecast
assumption.

## Versioning and access

Anonymous scoring aggregates and public-source provenance are versioned. Raw
source documents, participant workbooks, complete statement text, per-problem
source metadata, supplied-PDF fingerprints and model weights remain local and
are ignored by Git. Source files are downloaded reproducibly. The current user
PDF is copied locally and identified by a locally retained hash, never fetched
from an invented public URL. Published research outputs contain forecasts and
aggregate diagnostics rather than the supplied problem text.

"""Generate publication-quality figures and result tables from measured outputs."""
from __future__ import annotations
import html
import itertools
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

FIG = ROOT / "figures"
BLUE, TEAL, ORANGE, RED, GRAY = "#2457A7", "#008A82", "#EA8C32", "#C64B57", "#6C7A89"
COLORS = {2022: BLUE, 2023: TEAL, 2024: ORANGE, 2025: RED, 2026: "#724CB2"}
FOCUS = ["baseline_grade_mean", "direct_rubert_lightgbm", "direct_boosting_ensemble",
         "direct_ensemble", "difficulty_ensemble", "robust_ensemble"]


def short(name):
    replacements = {"baseline_grade_mean": "Grade mean", "baseline_grade_median": "Grade median",
        "baseline_latest_grade": "Latest season / grade", "baseline_mean": "Global mean",
        "direct_boosting_ensemble": "Direct boosting ensemble", "direct_ensemble": "Direct ensemble",
        "difficulty_ensemble": "Two-stage ensemble", "all_text_ensemble": "Both approaches / 50:50",
        "robust_ensemble": "Fixed robust ensemble", "nested_selected": "Nested baseline / ensemble",
        "direct_rubert_head_3ep": "Frozen RuBERT + 3-epoch head"}
    if name in replacements:
        return replacements[name]
    return name.replace("difficulty_", "Two-stage / ").replace("direct_", "Direct / ").replace("rubert_", "RuBERT / ").replace("tfidf_", "TF-IDF / ").replace("catboost", "CatBoost").replace("lightgbm", "LightGBM").replace("ridge", "Ridge")


def finish(fig, name, title, subtitle=None):
    fig.suptitle(title, x=.02, ha="left", fontsize=17, fontweight="bold", color="#172C45")
    if subtitle:
        fig.text(.02, .925, subtitle, fontsize=10, color=GRAY, ha="left", va="top")
    fig.tight_layout(rect=(0, .01, 1, .90 if subtitle else .95), pad=1.5)
    FIG.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = FIG / f"{name}.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)


def main():
    sns.set_theme(style="whitegrid", context="notebook", font_scale=.95)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
        "axes.spines.right": False, "grid.alpha": .22, "axes.titleweight": "bold"})
    metrics = pd.read_csv(ROOT / "results/metrics.csv")
    pred = pd.read_csv(ROOT / "results/validation_predictions.csv")
    curves = pd.read_csv(ROOT / "results/learning_curves.csv")
    forecast = pd.read_csv(ROOT / "results/forecast_2026.csv")
    cutoffs = pd.read_csv(ROOT / "data/processed/cutoffs.csv")
    labels = pd.read_csv(ROOT / "data/processed/difficulty_labels.csv")
    final = json.loads((ROOT / "results/final_prediction.json").read_text())
    secondary = metrics[metrics.protocol == "single_case_year_purged"].sort_values("power_1_5")

    fig, ax = plt.subplots(figsize=(11, 5.8))
    for grade, color in zip((9, 10, 11), (TEAL, BLUE, ORANGE)):
        d = cutoffs[cutoffs.grade == grade]
        ax.plot(d.year, d.cutoff, "o-", label=f"Grade {grade}", lw=2.4, color=color)
        for r in d.itertuples():
            ax.annotate(f"{r.cutoff:g}", (r.year, r.cutoff), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    lo, hi = final["plausible_interval"]
    ax.errorbar(2026, final["final_prediction"], yerr=[[final["final_prediction"]-lo], [hi-final["final_prediction"]]],
                fmt="D", color=ORANGE, capsize=6, markersize=8, label="2026/27 grade 11 forecast + assumption-based range")
    ax.axvline(2025.5, color=GRAY, ls=":")
    ax.set(xticks=range(2022, 2027), xticklabels=[f"{y}/{str(y+1)[-2:]}" for y in range(2022, 2027)], ylabel="Final-qualification cutoff / 100", ylim=(30, 90))
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    finish(fig, "historical_cutoffs", "Four comparable seasons, twelve cutoff observations",
           "2021/22 used a combined 225-point target and is excluded. The forecast range is not distribution-free coverage.")

    models = secondary.model.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(17, 11.8), sharey=True)
    for ax, measure, title in zip(axes, ("power_1_5", "mae", "rmse"), ("Mean |error|^1.5", "MAE / points", "RMSE / points")):
        for offset, protocol, color in [(-.19, "diagonal", BLUE), (.19, "single_case_year_purged", TEAL)]:
            values = metrics[metrics.protocol == protocol].set_index("model").loc[models, measure]
            ax.barh(np.arange(len(models))+offset, values, height=.35, color=color, alpha=.9,
                    label="Diagonal: 3 training labels" if protocol == "diagonal" else "Year-purged: 9 training labels")
        ax.set(title=title, xlabel="Lower is better", yticks=np.arange(len(models)), yticklabels=[short(m) for m in models])
        ax.invert_yaxis()
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    finish(fig, "architecture_comparison", "Architecture comparison on identical validation cases",
           "Rows are ordered by supplementary year-purged custom loss. Twelve test cases represent only four independent seasons.")

    data = pred[pred.protocol == "single_case_year_purged"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, name in zip(axes.flat, FOCUS):
        d = data[data.model == name]
        for year, group in d.groupby("year"):
            ax.scatter(group.actual, group.prediction, color=COLORS[year], s=65, label=f"{year}/{str(year+1)[-2:]}", edgecolor="white", linewidth=.6)
            for row in group.itertuples():
                ax.annotate(str(row.grade), (row.actual, row.prediction), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.plot([30, 85], [30, 85], ":", color=GRAY)
        ax.set(title=short(name), xlabel="Actual cutoff", ylabel="Predicted cutoff", xlim=(34, 80), ylim=(34, 85))
    axes[0, 0].legend(frameon=False, fontsize=8)
    finish(fig, "predicted_vs_actual", "Predicted versus actual cutoffs",
           "Supplementary whole-year purge; labels next to points show grade. Vertical distance from the dotted line is prediction error.")

    fig, ax = plt.subplots(figsize=(13, 11))
    ordered = secondary.model.tolist()
    plot_data = data.copy()
    plot_data["architecture"] = plot_data.model.map(short)
    sns.boxplot(data=plot_data, x="error", y="architecture", order=[short(m) for m in ordered], color="#DCE9F1", width=.5, showfliers=False, ax=ax)
    sns.stripplot(data=plot_data, x="error", y="architecture", order=[short(m) for m in ordered], hue="year", palette=COLORS, dodge=False, size=5, jitter=.15, ax=ax)
    ax.axvline(0, color=GRAY, ls="--", lw=1)
    ax.set(xlabel="Predicted - actual / points", ylabel="")
    ax.legend(title="Season starts", frameon=False, loc="lower right")
    finish(fig, "error_distributions", "Error distributions expose large year effects",
           "Each architecture has twelve held-out cases; colored dots preserve the season clusters.")

    for stage in ("cutoff", "difficulty"):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, (kind, learner) in zip(axes.flat, itertools.product(("tfidf", "rubert"), ("catboost", "lightgbm"))):
            name = f"{'direct' if stage=='cutoff' else 'difficulty'}_{kind}_{learner}"
            d = curves[(curves.stage == stage) & (curves.model == name)]
            for split, color, ls in [("train", BLUE, "-"), ("validation", ORANGE, "--")]:
                a = d[d.split == split].groupby("iteration").rmse.agg(["mean", "min", "max"])
                ax.plot(a.index, a["mean"], color=color, ls=ls, label=split, lw=2)
                ax.fill_between(a.index, a["min"], a["max"], color=color, alpha=.10)
            ax.set(title=short(name), xlabel="Boosting iteration", ylabel="RMSE / points")
            ax.legend(frameon=False)
        finish(fig, f"learning_curves_{stage}", f"{'Cutoff' if stage=='cutoff' else 'Problem-difficulty'} learning curves",
               "Mean and min-max across the four diagonal folds. Validation is monitoring only; no early stopping or curve-based tuning.")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    d = curves[curves.model == "direct_rubert_head_3ep"]
    for split, color in [("train", BLUE), ("validation", ORANGE)]:
        a = d[d.split == split].groupby("iteration").rmse.agg(["mean", "min", "max"])
        ax.plot(a.index, a["mean"], "o-", label=split, color=color, lw=2)
        ax.fill_between(a.index, a["min"], a["max"], color=color, alpha=.1)
    ax.set(xticks=[1, 2, 3], xlabel="Regression-head epoch", ylabel="Cutoff RMSE / points")
    ax.legend(frameon=False)
    finish(fig, "frozen_head_learning", "Three epochs on a frozen RuBERT regression head",
           "The encoder stays frozen. In diagonal folds, a grade prior exactly fits the three training labels; additional text effects cannot be learned.")

    importance = pd.read_csv(ROOT / "results/feature_importance.csv")
    def feature_group(feature):
        if re.search(r"(?:tfidf|rubert)_\d", feature): return "Semantic representation"
        if "grade" in feature: return "Grade"
        if "problem_count" in feature: return "Problem count"
        if "topic_" in feature: return feature.split("topic_")[-1].replace("_", " ").title()
        return feature.replace("mean_", "").replace("std_", "Variation: ").replace("_", " ").title()
    importance["group"] = importance.feature.map(feature_group)
    grouped = importance.groupby(["model", "group"], as_index=False).importance.sum()
    grouped.to_csv(ROOT / "results/feature_importance_grouped.csv", index=False)
    fig, axes = plt.subplots(4, 2, figsize=(15, 17))
    importance_models = sorted(grouped.model.unique())
    for ax, name in zip(axes.flat, importance_models):
        g = grouped[grouped.model == name].nlargest(8, "importance").sort_values("importance")
        ax.barh(g.group, g.importance, color=TEAL if "lightgbm" in name else BLUE)
        ax.set(title=short(name), xlabel="Relative model importance / %", xlim=(0, 103))
    finish(fig, "feature_importance", "What the boosting models use",
           "Final-fit CatBoost prediction-value importance and LightGBM gain, each normalized to 100%. Correlated features and tiny samples limit interpretation.")

    predicted_diff = pd.read_csv(ROOT / "results/predicted_difficulty.csv")
    actual = labels[labels.grade == 11].pivot(index="year", columns="item", values="difficulty")
    current = predicted_diff[(predicted_diff.year == 2026) & (predicted_diff.grade == 11)].groupby("item").prediction.mean()
    grid = actual.reindex(columns=range(1, 31))
    grid.loc[2026] = current.reindex(range(1, 31))
    fig, ax = plt.subplots(figsize=(16, 5.5))
    sns.heatmap(grid, cmap="YlOrRd", vmin=0, vmax=100, linewidths=.5, linecolor="white",
        cbar_kws={"label": "Difficulty: 100 x (1 - mean score fraction)"}, ax=ax)
    ax.set(xlabel="Physical problem number (numbers are not matched topics across years)", ylabel="Season starts; last row is predicted")
    ax.set_yticklabels([f"{y}/{str(y+1)[-2:]}" + (" predicted" if y==2026 else " observed") for y in grid.index], rotation=0)
    finish(fig, "problem_difficulty_heatmap", "Observed and predicted problem difficulty / grade 11",
           "Gray/empty cells mean no such problem. 2021 is a different assessment regime. The 2026 row averages four first-stage models.")

    means = labels.groupby(["year", "grade"], as_index=False).difficulty.mean()
    predicted_means = predicted_diff[predicted_diff.year == 2026].groupby("grade", as_index=False).prediction.mean()
    fig, ax = plt.subplots(figsize=(10, 5.7))
    for grade, color in zip((9, 10, 11), (TEAL, BLUE, ORANGE)):
        d = means[means.grade == grade]
        ax.plot(d.year, d.difficulty, "o-", lw=2, label=f"Grade {grade}", color=color)
        v = predicted_means[predicted_means.grade == grade].prediction.iloc[0]
        ax.scatter(2026, v, marker="D", color=color, s=70)
        ax.plot([2025, 2026], [d.difficulty.iloc[-1], v], ":", color=color)
    ax.axvline(2025.5, color=GRAY, ls=":")
    ax.set(xticks=range(2021, 2027), xlabel="Season starts", ylabel="Mean problem difficulty / 0-100", ylim=(20, 90))
    ax.legend(frameon=False)
    finish(fig, "difficulty_by_year_grade", "Difficulty changes with both problems and participant cohorts",
           "Observed means use physical problems with equal weights; diamonds are 2026 predictions, not measured solve rates.")

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.6), sharey=True)
    ensembles = ["baseline_grade_mean", "direct_rubert_lightgbm", "direct_boosting_ensemble", "direct_ensemble", "difficulty_ensemble", "all_text_ensemble", "robust_ensemble", "nested_selected"]
    for ax, protocol, title in zip(axes, ("diagonal", "single_case_year_purged", "forward"), ("Diagonal / 12 cases", "Year-purged / 12 cases", "Forward / 6 cases")):
        d = metrics[metrics.protocol == protocol].set_index("model").loc[ensembles]
        ax.barh(np.arange(len(d)), d.power_1_5, color=[TEAL if "ensemble" in n else BLUE for n in ensembles])
        ax.set(yticks=range(len(d)), yticklabels=[short(n) for n in ensembles], xlabel="Mean |error|^1.5", title=title)
        ax.invert_yaxis()
    finish(fig, "ensemble_comparison", "Ensembles do not automatically improve generalization",
           "All weights are fixed before fitting. Nested selection chooses a baseline or the fixed ensemble using inner years only.")

    baseline = data[data.model == "baseline_grade_mean"].set_index(["year", "grade"])
    boot_rows = []
    for name in secondary.model:
        g = data[data.model == name].set_index(["year", "grade"])
        delta = (g.error.abs()**1.5 - baseline.error.abs()**1.5).groupby("year").mean().to_numpy()
        samples = np.array([delta[list(ix)].mean() for ix in itertools.product(range(len(delta)), repeat=len(delta))])
        low, high = np.quantile(samples, [.025, .975])
        boot_rows.append(dict(model=name, mean_loss_difference=float(delta.mean()),
                              bootstrap_95_low=float(low), bootstrap_95_high=float(high), n_years=len(delta)))
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(ROOT / "results/paired_year_bootstrap.csv", index=False)
    show = boot[boot.model.isin(FOCUS) & (boot.model != "baseline_grade_mean")].sort_values("mean_loss_difference")
    fig, ax = plt.subplots(figsize=(12, 5.7))
    ax.errorbar(show.mean_loss_difference, range(len(show)),
        xerr=[show.mean_loss_difference-show.bootstrap_95_low, show.bootstrap_95_high-show.mean_loss_difference],
        fmt="o", capsize=5, color=TEAL, markersize=7)
    ax.axvline(0, color=GRAY, ls="--")
    ax.set(yticks=range(len(show)), yticklabels=[short(n) for n in show.model], xlabel="Custom loss difference relative to grade mean; negative favors text")
    ax.invert_yaxis()
    finish(fig, "paired_year_uncertainty", "The apparent text-model advantage is uncertain",
           "Descriptive paired bootstrap resamples four whole seasons (all 256 ordered resamples). Intervals do not correct model-selection bias.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), sharey=True)
    for ax, name in zip(axes, ("baseline_grade_mean", "direct_rubert_lightgbm", "robust_ensemble")):
        grid = data[data.model == name].pivot(index="year", columns="grade", values="error")
        sns.heatmap(grid, center=0, vmin=-20, vmax=20, cmap="RdBu_r", annot=True, fmt="+.1f", linewidths=1,
                    cbar=(ax is axes[-1]), cbar_kws={"label": "Predicted - actual / points"}, ax=ax)
        ax.set(title=short(name), xlabel="Grade", ylabel="Season starts" if ax is axes[0] else "")
        ax.tick_params(axis="y", rotation=0)
    finish(fig, "systematic_errors", "Systematic errors cluster by season",
           "2023/24 and 2024/25 are difficult to extrapolate from the other three years; grade offsets do not explain the whole shift.")

    folds = [f for f in json.loads((ROOT / "results/splits.json").read_text()) if f["protocol"] == "diagonal"]
    cases = list(zip(cutoffs.year, cutoffs.grade))
    matrix = np.ones((len(folds), len(cases)))
    for i, f in enumerate(folds):
        for j, case in enumerate(cases):
            if case in [(r["year"], r["grade"]) for r in f["train"]]: matrix[i,j] = 0
            if case in [(r["year"], r["grade"]) for r in f["validation"]]: matrix[i,j] = 2
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(14, 4.8))
    sns.heatmap(matrix, cmap=ListedColormap([BLUE, "#E2E7ED", ORANGE]), cbar=False, linewidths=1.5, linecolor="white", ax=ax)
    ax.set(xticks=np.arange(len(cases))+.5, xticklabels=[f"{y}\ngrade {g}" for y,g in cases],
           yticks=np.arange(len(folds))+.5, yticklabels=[f"Fold {i+1}" for i in range(len(folds))])
    ax.tick_params(axis="both", rotation=0)
    ax.legend(handles=[Patch(color=BLUE,label="Train"),Patch(color="#E2E7ED",label="Purged"),Patch(color=ORANGE,label="Validation")],
              loc="upper center", bbox_to_anchor=(.5,-.22), ncol=3, frameon=False)
    finish(fig, "validation_design", "Primary validation prevents shared-year leakage",
           "Each fold tests three different grades in three different years; every other grade from those years is purged.")

    f = forecast[forecast.grade == 11].copy()
    fig, ax = plt.subplots(figsize=(12, 10.5))
    names = f.model.tolist()
    ax.axvspan(*final["plausible_interval"], color="#FBEBD9", label="Assumption-dependent 90% reference range")
    ax.axvline(final["final_prediction"], color=GRAY, ls="--", label="Conservative final recommendation")
    for i, r in enumerate(f.itertuples()):
        color = TEAL if r.model.startswith("difficulty") else BLUE if r.model.startswith("direct") else ORANGE
        ax.scatter(r.prediction, i, color=color, s=65)
        ax.text(r.prediction+.6, i, f"{r.prediction:.2f}", va="center", fontsize=9)
    ax.set(yticks=range(len(f)), yticklabels=[short(n) for n in names], xlabel="Predicted 2026/27 grade-11 cutoff / 100", xlim=(38, 85))
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    finish(fig, "forecast_comparison", "Individual forecasts agree more than their uncertainty justifies",
           "A tight spread among fitted models is not a calibrated prediction interval. Four years require a much wider uncertainty range.")

    from .overview import generate_assets
    generate_assets()
    write_results(metrics, forecast, final, labels, pred, boot)
    print("Generated", len(list(FIG.glob('*.png'))), "figures and measured result tables.")


def write_results(metrics, forecast, final, labels, pred, boot):
    lines = ["# Measured results", "", "Generated from the checked local benchmark outputs. Lower loss is better.", "",
        f"**2026/27 grade 11 recommendation: {final['final_prediction']:.1f}/100 (approximately 63).** "
        f"The fixed ensemble predicts {final['fixed_robust_ensemble']:.2f}. "
        f"The assumption-dependent reference range is {final['plausible_interval'][0]:.1f}–{final['plausible_interval'][1]:.1f}. "
        "A nontrivial distribution-free 90% coverage claim is unsupported by four seasons.", "",
        "The final selection rule compares the grade mean and the prespecified robust ensemble on training-only inner year folds. "
        "The grade mean wins on all four available years. This preserves a conservative recommendation even though an individual "
        "RuBERT-LightGBM experiment has the lowest supplementary validation loss.", ""]
    for protocol, title in [("diagonal", "Primary: distinct year-and-grade triples"),
                             ("single_case_year_purged", "Supplementary: one case, whole-year purge"),
                             ("forward", "Forecast-direction sensitivity")]:
        data = metrics[metrics.protocol == protocol].sort_values("power_1_5")
        lines += ["## " + title, "", data[["model", "power_1_5", "mae", "rmse", "bias", "max_abs_error"]].to_markdown(index=False, floatfmt=".3f"), ""]
    lines += ["## 2026/27 grade 11 forecasts", "", forecast[forecast.grade == 11][["model", "prediction"]].to_markdown(index=False, floatfmt=".3f"), "",
        "## Paired uncertainty relative to the grade mean", "",
        "These descriptive 95% bootstrap intervals resample entire years, keeping the three grades together. "
        "There are only four clusters; the intervals do not account for trying multiple architectures.", "",
        boot[boot.model.isin(FOCUS)][["model", "mean_loss_difference", "bootstrap_95_low", "bootstrap_95_high"]].to_markdown(index=False, floatfmt=".3f"), "",
        "## Systematic errors", ""]
    data = pred[pred.protocol == "single_case_year_purged"]
    bias = data[data.model.isin(FOCUS)].groupby(["model", "year"]).error.mean().unstack("year")
    lines += [bias.to_markdown(floatfmt="+.2f"), "", "## Observed problem difficulty", "",
        labels.groupby(["year", "grade"]).difficulty.mean().unstack("grade").to_markdown(floatfmt=".2f"), "",
        "Difficulty is a cohort-dependent partial-credit statistic. It is not an intrinsic psychological scale or an individual solve probability. "
        "Variant-specific outcomes are unavailable, so statement variants share a pooled problem-family label.", ""]
    dval = pd.read_csv(ROOT / "results/difficulty_validation.csv")
    from .metrics import regression_metrics
    ds = [dict(model=name, n_problems=len(g), **regression_metrics(g.actual, g.prediction)) for name,g in dval.groupby("model")]
    lines += ["## Held-out problem-difficulty quality", "", pd.DataFrame(ds).to_markdown(index=False,floatfmt=".3f"), "",
        "## Interpretation", "",
        "- The strongest supplementary individual model is frozen RuBERT + LightGBM. Its advantage over the grade baseline is uncertain under whole-year resampling.",
        "- Explicit difficulty modeling does not improve this benchmark. Calibrating difficulty into a cutoff is especially unstable when the primary protocol leaves one cutoff year for training.",
        "- The cutoff is determined by participant ability, selection rules and final capacity as well as problem content. Those unobserved factors can produce large common errors across grades.",
        "- A direct model cannot learn a season-level text effect from only one training season. In diagonal folds, lexical and semantic CatBoost forecasts coincide, and LightGBM may find no legal split with three rows.",
        "- The 2026 paper has 30 problems, versus 15/17/20/20 historically. Its point weights and data attachments are missing. Predictions extrapolate beyond the observed problem-count range.",
        "- The three-epoch frozen head does not justify full-encoder fine-tuning on twelve labels. No model architecture is established as a reliable high-precision forecaster.", "",
        "## Uncertainty", "",
        "The grade-11 cutoffs are 66, 70, 53 and 61. Their mean is 62.5. Under independent, stationary, approximately Gaussian year effects, "
        "a 90% future-observation interval is mean ± t(0.95, 3) × sample_sd × sqrt(1+1/4), yielding approximately [43.2, 81.8]. "
        "This is a forecast interval for a new year, not a confidence interval for the historical mean. Stationarity is not established. "
        "With four calibration years, the 90% finite-sample conformal order statistic would require rank five; a useful distribution-free guarantee is unavailable. "
        "The documented conservative full-score range is [0,100]. Agreement among the fitted models is not evidence of narrow coverage.", "",
        "## Figures", ""]
    captions = {"grade11_forecast_table": "Headline grade-11 forecast table",
        "end_to_end_pipeline": "Complete training and prediction pipeline",
        "historical_cutoffs": "Historical targets and forecast", "architecture_comparison": "All architectures and metrics",
        "predicted_vs_actual": "Predicted versus actual", "error_distributions": "Error distributions",
        "learning_curves_cutoff": "Cutoff learning curves", "learning_curves_difficulty": "Difficulty learning curves",
        "frozen_head_learning": "Frozen transformer head", "feature_importance": "Feature importance",
        "problem_difficulty_heatmap": "Problem difficulty", "difficulty_by_year_grade": "Difficulty by year and grade",
        "ensemble_comparison": "Ensemble comparison", "paired_year_uncertainty": "Paired year uncertainty",
        "systematic_errors": "Year and grade errors", "validation_design": "Validation split design",
        "forecast_comparison": "Individual final forecasts"}
    for name, caption in captions.items():
        lines += [f"### {caption}", "", f"![{caption}](../figures/{name}.png)", ""]
    (ROOT / "docs/RESULTS.md").write_text("\n".join(lines))
    # Self-contained structure with local figure assets, usable without a server.
    sections = ''.join(f'<section><h2>{html.escape(caption)}</h2><img src="../figures/{name}.png" alt="{html.escape(caption)}"></section>' for name,caption in captions.items())
    table = metrics[metrics.protocol == "single_case_year_purged"][["model", "power_1_5", "mae", "rmse"]].sort_values("power_1_5").to_html(index=False, float_format=lambda v:f"{v:.3f}", border=0)
    forecasts = forecast[forecast.grade == 11][["model", "prediction"]].to_html(index=False,float_format=lambda v:f"{v:.2f}",border=0)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DANO cutoff research</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#172c45;background:#f3f6fa;margin:0}}main{{max-width:1120px;margin:auto;padding:46px 24px}}h1{{font-size:42px;line-height:1.15}}h2{{font-size:25px}}section,.intro{{background:white;border:1px solid #e2e7ee;border-radius:14px;padding:26px;margin:24px 0}}.eyebrow{{color:#008a82;font-weight:700;letter-spacing:.08em}}.score{{font-size:52px;font-weight:750;color:#2457a7}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px 10px;text-align:left;border-bottom:1px solid #e2e7ee}}th{{background:#eef4fa}}a{{color:#2457a7}}.note{{color:#627487}}@media(max-width:650px){{h1{{font-size:30px}}section{{padding:12px;overflow:auto}}}}</style>
<main><div class="eyebrow">LOCAL, REPRODUCIBLE ML RESEARCH · 2026/27</div><h1>Can problem text predict the DANO final cutoff?</h1>
<div class="intro"><div class="score">62.5 / 100</div><p>Grade 11 conservative recommendation; approximately 63. Fixed ensemble: {final['fixed_robust_ensemble']:.2f}. Assumption-dependent reference interval: 43.2–81.8.</p>
<p class="note">Four comparable seasons, twelve correlated cutoff labels. The data do not establish high predictive precision or a narrow calibrated interval. 2021/22 used combined-stage scoring and is excluded from the cutoff target.</p>
<p><a href="../docs/RESULTS.md">Detailed measured results</a> · <a href="../README.md">README and reproduction</a> · <a href="../docs/METHODS.md">Methods</a></p></div>
<section><h2>Supplementary whole-year-purged benchmark</h2>{table}</section><section><h2>Every grade-11 forecast</h2>{forecasts}</section>{sections}</main></html>'''
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/research_report.html").write_text(page)


if __name__ == "__main__":
    main()

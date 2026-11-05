"""Render the headline forecast table and the complete trained/inference pipeline.

All numerical forecasts are read from measured result files; no model is refit.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as DrawingPath
import pandas as pd

NAVY = "#14243F"
BLUE = "#255FDD"
TEAL = "#007D76"
VIOLET = "#7950D9"
AMBER = "#FFC857"
INK = "#20324B"
MUTED = "#62718A"
OUT = ROOT / "figures"


def save(fig, stem):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, dpi=210, bbox_inches="tight", pad_inches=.15, facecolor=fig.get_facecolor())
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)


def box(ax, x, y, w, h, fill="white", edge="#D9E2F1", radius=.9, linewidth=1.2):
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=fill,
        edgecolor=edge, linewidth=linewidth, zorder=2)
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, size=12, color=INK, weight="normal", **kwargs):
    return ax.text(x, y, text, fontsize=size, color=color, weight=weight,
        fontfamily="DejaVu Sans", va="center", zorder=5, **kwargs)


def arrow(ax, points, color=MUTED, dashed=False, width=1.6, arrowhead=True):
    vertices = points
    codes = [DrawingPath.MOVETO] + [DrawingPath.LINETO] * (len(points)-1)
    path = DrawingPath(vertices, codes)
    if arrowhead:
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=13,
            color=color, linewidth=width, linestyle="--" if dashed else "-", zorder=3))
    else:
        ax.add_patch(PathPatch(path, fill=False, color=color, linewidth=width,
            linestyle="--" if dashed else "-", zorder=3))


def headline_table(final, forecast):
    selected = float(final["final_prediction"])
    rounded = math.floor(selected+.5)
    frame = forecast[forecast.grade == 11].set_index("model")
    value = lambda name: float(frame.loc[name, "prediction"])
    fig, ax = plt.subplots(figsize=(13.8, 11.5))
    fig.patch.set_facecolor("#F2F6FF")
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    box(ax, 1, 72, 98, 27, NAVY, NAVY, radius=2)
    label(ax, 5, 94, "DANO 2026/27  /  GRADE 11", size=15, color="#94E9D2", weight="bold")
    label(ax, 5, 87.5, "PREDICTED FINAL-QUALIFICATION CUTOFF", size=13, color="white", weight="bold")
    label(ax, 5, 78.6, f"≈ {rounded}", size=54, color=AMBER, weight="bold")
    label(ax, 28.3, 77, "/ 100 points", size=17, color="white")
    box(ax, 64, 76, 31, 17, "#24395E", "#24395E", radius=1.1)
    label(ax, 67, 89.2, "SELECTED ESTIMATE", size=10, color="#B8C7E5", weight="bold")
    label(ax, 67, 84.5, f"{selected:.2f}", size=27, color="white", weight="bold")
    label(ax, 67, 79.2, f"Fixed ensemble: {final['fixed_robust_ensemble']:.2f}", size=12, color="#94E9D2")

    label(ax, 4, 67.8, "INDIVIDUAL ML MODELS + APPROACH ENSEMBLES", size=14, color=NAVY, weight="bold")
    box(ax, 2, 59.5, 96, 5.7, BLUE, BLUE, radius=.6)
    label(ax, 5, 62.3, "REPRESENTATION + REGRESSOR", size=11, color="white", weight="bold")
    label(ax, 67, 62.3, "DIRECT", size=11, color="white", weight="bold", ha="center")
    label(ax, 87, 62.3, "DIFFICULTY FIRST", size=11, color="white", weight="bold", ha="center")
    specs = [
        ("TF-IDF + CatBoost", "direct_tfidf_catboost", "difficulty_tfidf_catboost"),
        ("TF-IDF + LightGBM", "direct_tfidf_lightgbm", "difficulty_tfidf_lightgbm"),
        ("Frozen RuBERT + CatBoost", "direct_rubert_catboost", "difficulty_rubert_catboost"),
        ("Frozen RuBERT + LightGBM", "direct_rubert_lightgbm", "difficulty_rubert_lightgbm"),
        ("TF-IDF + ridge", "direct_tfidf_ridge", None),
        ("Frozen RuBERT + ridge", "direct_rubert_ridge", None),
        ("Frozen RuBERT + 3-epoch head", "direct_rubert_head_3ep", None),
        ("APPROACH ENSEMBLES", "direct_ensemble", "difficulty_ensemble"),
    ]
    for i, (name, direct, difficulty) in enumerate(specs):
        top = 59.2-i*5.25
        fill = "#E0F6EF" if i==7 else ("white" if i%2==0 else "#E9F0FF")
        box(ax, 2, top-5, 96, 5, fill, fill, radius=.2, linewidth=0)
        color = TEAL if i==7 else INK
        weight = "bold" if i==7 else "normal"
        label(ax, 5, top-2.5, name, size=12.2, color=color, weight=weight)
        label(ax, 67, top-2.5, f"{value(direct):.2f}", size=15, color=BLUE if i!=7 else TEAL,
              weight="bold", ha="center")
        label(ax, 87, top-2.5, f"{value(difficulty):.2f}" if difficulty else "—",
              size=15, color=TEAL if difficulty else "#A4AEBF", weight="bold", ha="center")
    box(ax, 2, 9.1, 96, 6.6, AMBER, AMBER, radius=.7)
    lo, hi = final["plausible_interval"]
    label(ax, 5, 12.4, f"REFERENCE RANGE  {lo:.1f}–{hi:.1f}", size=13.5, color=NAVY, weight="bold")
    label(ax, 52, 12.4, "Assumption-dependent 90% forecast interval", size=11.2, color=NAVY)
    label(ax, 3, 5.6, "MODEL FORECAST · The official 2026/27 cutoff is not verified here.", size=11, color=MUTED, weight="bold")
    label(ax, 3, 2, "Only 4 comparable seasons / 12 year–grade labels. Assumes a 100-point paper; reliable coverage is unverified.", size=10.2, color=MUTED)
    save(fig, "grade11_forecast_table")


def pipeline_diagram(final, difficulty_count, cutoff_count):
    fig, ax = plt.subplots(figsize=(16.8, 17.8))
    fig.patch.set_facecolor("#F5F8FD")
    ax.set(xlim=(-2, 102), ylim=(-1, 116))
    ax.axis("off")
    label(ax, 4, 112.8, "DANO  /  COMPLETE TRAINING + PREDICTION PIPELINE", size=21, color=NAVY, weight="bold")
    label(ax, 4, 109.4, "Solid arrows: computation and prediction    ·    Dashed arrows: historical training targets", size=11.5, color=MUTED)

    for x, color, title, body in [
        (4, BLUE, "HISTORICAL PROBLEM TEXT", "2021/22–2025/26 · 5 seasons\n2021/22: difficulty branch only\nStatement variants; answers removed"),
        (36, VIOLET, "HISTORICAL TRAINING TARGETS", f"{difficulty_count} year–grade–problem labels\n{cutoff_count} cutoffs · 4 comparable seasons\n2021 cutoff excluded: different scale"),
        (68, TEAL, "2026/27 INFERENCE INPUT", "Supplied PDF · 30 problems\nGrade 11 · assumed maximum 100\nNo 2026 cutoff label is used"),
    ]:
        box(ax, x, 94, 28, 11.2, "white", color, linewidth=1.8)
        label(ax, x+1.5, 102.8, title, size=10.6, color=color, weight="bold")
        label(ax, x+1.5, 98.2, body, size=11.2, linespacing=1.6)

    box(ax, 4, 80, 92, 9, NAVY, NAVY)
    label(ax, 6, 86.5, "STATEMENT PREPROCESSING + FIXED TEXT FEATURES", size=13, color="white", weight="bold")
    label(ax, 6, 82.7, "Extract and align physical problems · remove answer annotations · 16 text/topic features + grade", size=12, color="#D6E3FF")
    arrow(ax, [(18,94),(18,89)], BLUE)
    arrow(ax, [(82,94),(82,89)], TEAL)

    for x, color, title, body in [
        (4, BLUE, "LEXICAL REPRESENTATION", "Word TF-IDF (1–2) + character TF-IDF (3–5)\nSVD: up to 24 components\nVocabulary, IDF and SVD fit on training folds only"),
        (52, VIOLET, "FROZEN SEMANTIC REPRESENTATION", "RuBERT-tiny2 · 3 layers · 312-dimensional vectors\n256-token windows / stride 32 · masked pooling\nAll tokens processed; encoder weights never update"),
    ]:
        box(ax, x, 66, 44, 10, "#EDF2FF" if x==4 else "#F1EBFF", color)
        label(ax, x+1.5, 73.7, title, size=12, color=color, weight="bold")
        label(ax, x+1.5, 69.1, body, size=11.3, linespacing=1.5)
    arrow(ax, [(26,80),(26,76)], BLUE)
    arrow(ax, [(74,80),(74,76)], VIOLET)
    box(ax, 24, 61.5, 52, 3.1, "white", "#CCD7E8", radius=.4)
    label(ax, 50, 63.05, "RUN EACH REPRESENTATION SEPARATELY IN BOTH BRANCHES", size=10.5, color=MUTED, weight="bold", ha="center")
    arrow(ax, [(26,66),(37,64.6)], BLUE)
    arrow(ax, [(74,66),(63,64.6)], VIOLET)
    arrow(ax, [(37,61.5),(26,60.2)], TEAL)
    arrow(ax, [(63,61.5),(74,60.2)], BLUE)

    box(ax, 4, 31, 44, 29, "#E8F7F2", "#8FCDBB", radius=1.2)
    box(ax, 52, 31, 44, 29, "#EAF1FF", "#A1B9EE", radius=1.2)
    label(ax, 6, 58, "APPROACH 1  /  DIFFICULTY FIRST", size=12.5, color=TEAL, weight="bold")
    label(ax, 54, 58, "APPROACH 2  /  DIRECT CUTOFF", size=12.5, color=BLUE, weight="bold")
    box(ax, 7, 48, 38, 8, "white", "#B9DECF")
    label(ax, 8.5, 54.2, "CATBOOST + LIGHTGBM × 2 REPRESENTATIONS", size=10.8, color=TEAL, weight="bold")
    label(ax, 8.5, 50.7, "Fit: d = 100 × (1 − mean earned/max points)\n4 predictors → per-problem difficulty in [0.001, 99.999]", size=10.7, linespacing=1.6)
    box(ax, 7, 36.5, 38, 8.5, "white", "#B9DECF")
    label(ax, 8.5, 43.1, "POOL DIFFICULTY → RIDGE CALIBRATION", size=10.9, color=TEAL, weight="bold")
    label(ax, 8.5, 39.3, "Mean, spread, quantiles, min/max + grade\nFit: historical cutoffs; year-excluded difficulty inputs\nRidge alpha 10 → one cutoff from each of 4 pipelines", size=10.5, linespacing=1.45)
    arrow(ax, [(26,48),(26,45)], TEAL)
    label(ax, 26, 33.8, f"DIFFICULTY ENSEMBLE  {final['difficulty_ensemble']:.2f}", size=14, color=TEAL, weight="bold", ha="center")

    box(ax, 55, 48, 38, 8, "white", "#C0D1F5")
    label(ax, 56.5, 54.2, "POOL THE COMPLETE PROBLEM SET", size=10.9, color=BLUE, weight="bold")
    label(ax, 56.5, 50.7, "Mean vectors; text-feature means + standard deviations\nProblem count + grade → 58 TF-IDF / 346 RuBERT inputs", size=10.6, linespacing=1.6)
    box(ax, 55, 36.5, 38, 8.5, "white", "#C0D1F5")
    label(ax, 56.5, 43.1, "CATBOOST + LIGHTGBM + RIDGE × 2", size=10.9, color=BLUE, weight="bold")
    label(ax, 56.5, 39.3, "Fit: 12 cutoff labels → 6 direct regression models\nTrees: ~1,000 rounds; ridge alpha 20\n3-epoch frozen head is evaluated separately", size=10.6, linespacing=1.45)
    arrow(ax, [(74,48),(74,45)], BLUE)
    label(ax, 74, 33.8, f"DIRECT ENSEMBLE  {final['direct_ensemble']:.2f}", size=14, color=BLUE, weight="bold", ha="center")

    # Explicit target supervision does not pass through the inference text path.
    arrow(ax, [(42,94),(42,91.2),(1,91.2),(1,52),(7,52)], VIOLET, dashed=True, width=1.5)
    label(ax, -.3, 71, "DIFFICULTY LABELS", size=9.3, color=VIOLET, weight="bold", rotation=90, ha="center")
    arrow(ax, [(58,94),(58,92),(99,92),(99,40.7),(93,40.7)], VIOLET, dashed=True, width=1.5)
    label(ax, 100.3, 71, "HISTORICAL CUTOFF LABELS", size=9.3, color=VIOLET, weight="bold", rotation=90, ha="center")
    # The same historical cutoff targets define the grade-only baseline.
    arrow(ax, [(99,40.7),(99,28.4),(16.5,28.4),(16.5,25)], VIOLET, dashed=True, width=1.3)

    box(ax, 4, 14, 25, 11, "#EEE7FF", "#C2ACEF")
    label(ax, 5.5, 22.5, "GRADE-11 PRIOR", size=11.5, color=VIOLET, weight="bold")
    label(ax, 5.5, 19, "Historical cutoffs: 66, 70, 53, 61", size=10.3, color=INK)
    label(ax, 5.5, 15.9, f"MEAN  {final['final_prediction']:.2f}", size=17, color=VIOLET, weight="bold")

    box(ax, 36, 14, 28, 11, NAVY, NAVY)
    label(ax, 37.5, 22.5, "FIXED ROBUST ENSEMBLE", size=11.5, color="#94E9D2", weight="bold")
    label(ax, 37.5, 19.3, "50% prior + 25% direct + 25% difficulty", size=10.2, color="white")
    label(ax, 37.5, 15.9, f"{final['fixed_robust_ensemble']:.2f} / 100", size=18, color=AMBER, weight="bold")
    arrow(ax, [(26,31),(26,29.7),(44,29.7),(44,25)], TEAL)
    arrow(ax, [(74,31),(74,30.4),(56,30.4),(56,25)], BLUE)
    arrow(ax, [(29,19.7),(36,19.7)], VIOLET)

    box(ax, 72, 14, 24, 11, AMBER, AMBER)
    label(ax, 73.5, 22.5, "NESTED FINAL CHOICE", size=10.9, color=NAVY, weight="bold")
    label(ax, 73.5, 18.8, f"≈ {math.floor(final['final_prediction']+.5)} / 100", size=25, color=NAVY, weight="bold")
    label(ax, 73.5, 15.6, f"Selected: grade prior {final['final_prediction']:.2f}", size=10.3, color=NAVY)
    arrow(ax, [(64,20),(72,20)], NAVY, width=1.9)
    arrow(ax, [(16.5,14),(16.5,11.3),(84,11.3),(84,14)], VIOLET, width=1.5)
    label(ax, 49, 12.1, "Inner year-purged CV compares prior vs fixed blend; the prior wins.", size=10.2, color=VIOLET, ha="center",
          bbox=dict(facecolor="#F5F8FD", edgecolor="none", pad=2))

    box(ax, 4, 0, 92, 8.2, "#FFF1CD", "#E9D395")
    label(ax, 6, 6.3, "LEAKAGE CONTROL + RELIABILITY", size=11.5, color=NAVY, weight="bold")
    label(ax, 6, 3.8, "Same year-purged folds for every model · train-only TF-IDF/SVD/scalers · year-cross-fitted calibration · no test-label tuning", size=10.7, color=INK)
    lo, hi = final["plausible_interval"]
    label(ax, 6, 1.4, f"MAE / RMSE / mean |error|^1.5 · only 4 cutoff seasons · reference range {lo:.1f}–{hi:.1f} is assumption-dependent", size=10.7, color=INK)
    save(fig, "end_to_end_pipeline")


def generate_assets():
    final = json.loads((ROOT / "results/final_prediction.json").read_text())
    forecast = pd.read_csv(ROOT / "results/forecast_2026.csv")
    difficulty_count = len(pd.read_csv(ROOT / "data/processed/difficulty_labels.csv"))
    cutoff_count = len(pd.read_csv(ROOT / "data/processed/cutoffs.csv"))
    with plt.rc_context({"font.family": "DejaVu Sans", "svg.fonttype": "path"}):
        headline_table(final, forecast)
        pipeline_diagram(final, difficulty_count, cutoff_count)
    print("Generated the grade-11 forecast table and complete pipeline diagram from measured outputs.")


if __name__ == "__main__":
    generate_assets()

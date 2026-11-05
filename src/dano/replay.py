"""Replay every saved grade-11 forecast without refitting or network access."""
import argparse
import hashlib
import json
import importlib.metadata
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from .benchmark import Experiment, grade_prior, summarize_difficulty

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "data/raw/current/2026_second_stage_user.pdf")
    args = parser.parse_args()
    audit = json.loads((ROOT / "data/processed/pdf_extraction_audit.json").read_text())
    expected_hash = next(r["sha256"] for r in audit if r["year"] == 2026)
    actual_hash = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("PDF differs from the forecast input; rebuild dataset/embeddings and rerun the benchmark")
    exp = Experiment()
    saved = joblib.load(ROOT / "models/final_models.joblib")
    first = joblib.load(ROOT / "models/final_difficulty_models.joblib")
    direct = saved["direct"]
    train = direct["train"]
    values = {"baseline_mean": np.full(len(exp.cases), train.cutoff.mean()),
        "baseline_grade_mean": grade_prior(train, exp.cases),
        "baseline_grade_median": grade_prior(train, exp.cases, True),
        "baseline_latest_grade": grade_prior(train[train.year == train.year.max()], exp.cases)}
    for name, item in direct["objects"].items():
        if not name.startswith("direct_"):
            continue
        model, representation, _ = item
        all_x, _ = representation.set_features(exp.cases)
        if name.endswith("head_3ep"):
            import torch
            torch.set_num_threads(4)
            x = torch.tensor(model["scaler"].transform(all_x), dtype=torch.float32)
            head = torch.nn.Linear(x.shape[1], 1)
            head.load_state_dict(model["state_dict"])
            with torch.inference_mode():
                p = grade_prior(train, exp.cases) + 10 * head(x).flatten().numpy()
        else:
            p = model.predict(all_x)
        values[name] = np.clip(p, 0, 100)
    direct_names = [f"direct_{kind}_{learner}" for kind in ("tfidf", "rubert") for learner in ("catboost", "lightgbm", "ridge")]
    values["direct_boosting_ensemble"] = np.mean([values[n] for n in direct_names if not n.endswith("ridge")], axis=0)
    values["direct_ensemble"] = np.mean([values[n] for n in direct_names], axis=0)
    for name, fit in first.items():
        x = fit["representation"].problem_features(exp.prediction_problems)
        difficulty = np.clip(fit["model"].predict(x), .001, 99.999)
        summary = summarize_difficulty(exp.prediction_problems, difficulty, exp.cases)
        values[name] = np.clip(saved["calibration"][name].predict(summary), 0, 100)
    values["difficulty_ensemble"] = np.mean([values[n] for n in first], axis=0)
    values["all_text_ensemble"] = .5*values["direct_ensemble"] + .5*values["difficulty_ensemble"]
    values["robust_ensemble"] = .5*values["baseline_grade_mean"] + .25*values["direct_ensemble"] + .25*values["difficulty_ensemble"]
    idx = exp.cases.index[(exp.cases.year == 2026) & (exp.cases.grade == 11)][0]
    expected = pd.read_csv(ROOT / "results/forecast_2026.csv")
    expected = expected[expected.grade == 11].set_index("model").prediction.to_dict()
    assert set(values) == set(expected)
    differences = {name: abs(float(v[idx])-expected[name]) for name,v in values.items()}
    maximum = max(differences.values())
    if maximum > 1e-8:
        raise AssertionError(f"Saved forecast replay mismatch: {maximum}")
    final = json.loads((ROOT / "results/final_prediction.json").read_text())
    result = dict(verified_models=len(values), max_absolute_replay_difference=maximum,
        input_pdf_sha256=actual_hash, final_prediction=float(values[final["final_model"]][idx]),
        forecasts={name:float(v[idx]) for name,v in values.items()})
    (ROOT / "results/replay_check.json").write_text(json.dumps(result, indent=2)+"\n")
    (ROOT / "results/replay_summary.json").write_text(json.dumps({k: result[k] for k in
        ("verified_models", "max_absolute_replay_difference", "final_prediction")}, indent=2)+"\n")
    environment = dict(python=sys.version, platform=platform.platform(), machine=platform.machine(),
        packages={name: importlib.metadata.version(name) for name in
            ("numpy", "pandas", "scipy", "scikit-learn", "catboost", "lightgbm", "torch", "transformers", "pymupdf", "matplotlib", "filelock")})
    (ROOT / "results/environment.json").write_text(json.dumps(environment, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

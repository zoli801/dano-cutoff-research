"""Run fixed local experiments, purged validation, nested selection and forecasts."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib"))
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.stats import t as student_t
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from .features import Corpus, Representation
from .metrics import regression_metrics
from .validation import diagonal_folds, single_case_year_folds, expanding_folds

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "configs/benchmark.json").read_text())
SEED = CONFIG["seed"]
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


def new_booster(learner, difficulty=False):
    if learner == "catboost":
        return CatBoostRegressor(iterations=CONFIG["iterations"], loss_function="RMSE",
            random_seed=SEED, thread_count=4, verbose=False, allow_writing_files=False,
            use_best_model=False, **CONFIG["difficulty_catboost" if difficulty else "direct_catboost"])
    return LGBMRegressor(n_estimators=CONFIG["iterations"], objective="regression",
        random_state=SEED, n_jobs=4, verbosity=-1, deterministic=True, force_col_wise=True,
        **CONFIG["difficulty_lightgbm" if difficulty else "direct_lightgbm"])


def grade_prior(train, cases, median=False):
    table = train.groupby("grade").cutoff.median() if median else train.groupby("grade").cutoff.mean()
    return np.array([table.get(int(g), train.cutoff.mean()) for g in cases.grade])


def summarize_difficulty(frame, predictions, cases):
    """Equal physical-problem weights; current paper does not provide point weights."""
    data = frame[["year", "grade", "item"]].copy()
    data["difficulty"] = predictions
    pooled = data.groupby(["year", "grade", "item"], as_index=False).difficulty.mean()
    rows = []
    for case in cases.itertuples():
        values = pooled[(pooled.year == case.year) & (pooled.grade == case.grade)].difficulty.to_numpy()
        if not len(values):
            raise ValueError(f"No difficulty predictions for {case.year}, {case.grade}")
        rows.append([100-values.mean(), values.std(), *np.quantile(values, [.1, .5, .9]),
                     values.min(), values.max(), case.grade-10])
    return np.asarray(rows)


class Experiment:
    def __init__(self):
        self.corpus = Corpus()
        self.input_fingerprint = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in (
            "data/processed/problem_texts.jsonl", "data/processed/difficulty_labels.csv",
            "data/processed/transformer_provenance.json"))).hexdigest()
        self.cutoffs = pd.read_csv(ROOT / "data/processed/cutoffs.csv").reset_index(drop=True)
        assert len(self.cutoffs) == 12 and self.cutoffs.year.nunique() == 4
        current = pd.DataFrame([dict(year=2026, season="2026/2027", grade=g, cutoff=np.nan) for g in (9, 10, 11)])
        self.cases = pd.concat([self.cutoffs, current], ignore_index=True)
        self.prediction_problems = self.corpus.problems.reset_index(names="problem_index").merge(pd.DataFrame({"grade": [9, 10, 11]}), how="cross")
        self.difficulty_cache, self.direct_cache, self.bundle_cache = {}, {}, {}
        self.curves, self.importances, self.timing = [], [], []
        self.difficulty_evaluations = []
        self.final_difficulties = []
        for path in (ROOT / "models/cache", ROOT / "results", ROOT / "figures"):
            path.mkdir(parents=True, exist_ok=True)

    def difficulty_fit(self, years, kind, learner, banned_hashes=()):
        train = self.corpus.supervised(years, banned_hashes)
        id_key = sorted(train.problem_index.unique().tolist())
        key = hashlib.sha256(json.dumps([id_key, kind, learner, CONFIG, self.input_fingerprint], sort_keys=True).encode()).hexdigest()[:20]
        if key in self.difficulty_cache:
            return self.difficulty_cache[key]
        cache_path = ROOT / "models/cache" / f"difficulty_{key}.joblib"
        if cache_path.exists():
            result = joblib.load(cache_path)
            self.difficulty_cache[key] = result
            return result
        start = time.perf_counter()
        representation = Representation(self.corpus, kind, np.array(id_key))
        x = representation.problem_features(train)
        model = new_booster(learner, difficulty=True)
        model.fit(x, train.difficulty.to_numpy(), sample_weight=train.sample_weight.to_numpy())
        test_x = representation.problem_features(self.prediction_problems)
        prediction = np.clip(model.predict(test_x), .001, 99.999)
        result = dict(prediction=prediction, model=model, representation=representation,
                      train=train, train_years=sorted(years), fit_seconds=time.perf_counter()-start)
        joblib.dump(result, cache_path)
        self.difficulty_cache[key] = result
        return result

    def record_curves(self, model, train_x, train_y, test_x, test_y, name, fold, stage):
        iterations = [1] + list(range(50, CONFIG["iterations"]+1, 50))
        for iteration in iterations:
            if isinstance(model, CatBoostRegressor):
                train_pred = model.predict(train_x, ntree_end=iteration)
                test_pred = model.predict(test_x, ntree_end=iteration)
            else:
                train_pred = model.predict(train_x, num_iteration=iteration)
                test_pred = model.predict(test_x, num_iteration=iteration)
            for split, actual, predicted in [("train", train_y, train_pred), ("validation", test_y, test_pred)]:
                metrics = regression_metrics(actual, predicted)
                self.curves.append(dict(model=name, fold=fold, stage=stage,
                    iteration=iteration, split=split, **metrics))

    def frozen_head(self, train_x, train_y, all_x, train_cases):
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(SEED)
        scaler = StandardScaler().fit(train_x)
        x = torch.tensor(scaler.transform(train_x), dtype=torch.float32)
        all_tensor = torch.tensor(scaler.transform(all_x), dtype=torch.float32)
        base_train = grade_prior(train_cases, train_cases)
        base_all = grade_prior(train_cases, self.cases)
        y = torch.tensor((train_y-base_train)/10, dtype=torch.float32)
        head = torch.nn.Linear(x.shape[1], 1)
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.1)
        history = []
        for epoch in range(1, 4):
            head.train()
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(head(x).flatten(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
            optimizer.step()
            head.eval()
            with torch.inference_mode():
                all_prediction = base_all + 10 * head(all_tensor).flatten().numpy()
            history.append(all_prediction.copy())
        return np.clip(history[-1], 0, 100), history, dict(scaler=scaler, state_dict=head.state_dict(), grade_means=train_cases.groupby("grade").cutoff.mean().to_dict())

    def direct_fit(self, years):
        key = tuple(sorted(years))
        if key in self.direct_cache:
            return self.direct_cache[key]
        train = self.cutoffs[self.cutoffs.year.isin(years)]
        indices = train.index.to_numpy()
        outputs = {"baseline_mean": np.full(len(self.cases), train.cutoff.mean()),
                   "baseline_grade_mean": grade_prior(train, self.cases),
                   "baseline_grade_median": grade_prior(train, self.cases, True)}
        recent = train[train.year == train.year.max()]
        outputs["baseline_latest_grade"] = grade_prior(recent, self.cases)
        objects, matrices = {}, {}
        for kind in ("tfidf", "rubert"):
            representation = Representation(self.corpus, kind, self.corpus.years_indices(years))
            all_x, names = representation.set_features(self.cases)
            x, y = all_x[indices], train.cutoff.to_numpy()
            matrices[kind] = (all_x, names)
            for learner in ("catboost", "lightgbm", "ridge"):
                name = f"direct_{kind}_{learner}"
                start = time.perf_counter()
                if learner == "ridge":
                    model = make_pipeline(StandardScaler(), Ridge(alpha=CONFIG["direct_ridge_alpha"]))
                else:
                    model = new_booster(learner)
                model.fit(x, y)
                outputs[name] = np.clip(model.predict(all_x), 0, 100)
                objects[name] = (model, representation, names)
                self.timing.append(dict(stage="direct", years=list(key), model=name, seconds=time.perf_counter()-start))
            if kind == "rubert":
                pred, history, head = self.frozen_head(x, y, all_x, train)
                outputs["direct_rubert_head_3ep"] = pred
                objects["direct_rubert_head_3ep"] = (head, representation, names)
                objects["head_history"] = history
        model_names = [f"direct_{kind}_{learner}" for kind in ("tfidf", "rubert") for learner in ("catboost", "lightgbm", "ridge")]
        outputs["direct_boosting_ensemble"] = np.mean([outputs[n] for n in model_names if not n.endswith("ridge")], axis=0)
        outputs["direct_ensemble"] = np.mean([outputs[n] for n in model_names], axis=0)
        result = dict(outputs=outputs, objects=objects, matrices=matrices, train=train)
        self.direct_cache[key] = result
        return result

    def bundle(self, years, test_years=(), record_fold=None, record_valid_indices=None):
        years = tuple(sorted(years))
        test_years = tuple(sorted(test_years))
        cache_key = (years, test_years)
        if cache_key in self.bundle_cache and record_fold is None:
            return self.bundle_cache[cache_key]
        direct = self.direct_fit(years)
        train = direct["train"]
        outputs = {k: v.copy() for k, v in direct["outputs"].items()}
        banned = set(self.corpus.problems.loc[self.corpus.problems.year.isin(test_years), "family_hash"])
        stage1_years = set(years) | {2021}
        calibration_models = {}
        if record_fold:
            valid_idx = np.asarray(record_valid_indices, int)
            for kind in ("tfidf", "rubert"):
                all_x, names = direct["matrices"][kind]
                for learner in ("catboost", "lightgbm"):
                    name = f"direct_{kind}_{learner}"
                    model = direct["objects"][name][0]
                    self.record_curves(model, all_x[train.index], train.cutoff.to_numpy(),
                        all_x[valid_idx], self.cutoffs.loc[valid_idx, "cutoff"].to_numpy(), name, record_fold, "cutoff")
            for epoch, prediction in enumerate(direct["objects"]["head_history"], 1):
                for split, ids in [("train", train.index), ("validation", valid_idx)]:
                    self.curves.append(dict(model="direct_rubert_head_3ep", fold=record_fold,
                        stage="cutoff", iteration=epoch, split=split,
                        **regression_metrics(self.cutoffs.loc[ids, "cutoff"], prediction[ids])))
        for kind in ("tfidf", "rubert"):
            for learner in ("catboost", "lightgbm"):
                name = f"difficulty_{kind}_{learner}"
                fit = self.difficulty_fit(stage1_years, kind, learner, banned)
                summary_all = summarize_difficulty(self.prediction_problems, fit["prediction"], self.cases)
                oof = np.zeros((len(train), summary_all.shape[1]))
                for year in years:
                    mask = (train.year == year).to_numpy()
                    inner_fit = self.difficulty_fit(stage1_years-{year}, kind, learner, banned)
                    assert year not in inner_fit["train_years"]
                    oof[mask] = summarize_difficulty(self.prediction_problems, inner_fit["prediction"], train[train.year == year])
                calibration = make_pipeline(StandardScaler(), Ridge(alpha=CONFIG["calibration_ridge_alpha"]))
                calibration.fit(oof, train.cutoff.to_numpy())
                outputs[name] = np.clip(calibration.predict(summary_all), 0, 100)
                calibration_models[name] = calibration
                if record_fold:
                    actual_frame = self.corpus.supervised(test_years)
                    valid_pairs = set(map(tuple, self.cutoffs.loc[valid_idx, ["year", "grade"]].to_numpy()))
                    actual_frame = actual_frame[[pair in valid_pairs for pair in zip(actual_frame.year, actual_frame.grade)]]
                    merged = self.prediction_problems[["problem_index", "grade"]].copy()
                    merged["prediction"] = fit["prediction"]
                    evaluation = actual_frame.merge(merged, on=["problem_index", "grade"], validate="one_to_one")
                    pooled = evaluation.groupby(["year", "grade", "item"], as_index=False).agg(
                        actual=("difficulty", "first"), prediction=("prediction", "mean"))
                    pooled["model"], pooled["fold"] = name, record_fold
                    self.difficulty_evaluations.extend(pooled.to_dict("records"))
                    xtrain = fit["representation"].problem_features(fit["train"])
                    xvalid = fit["representation"].problem_features(evaluation)
                    self.record_curves(fit["model"], xtrain, fit["train"].difficulty.to_numpy(),
                        xvalid, evaluation.difficulty.to_numpy(), name, record_fold, "difficulty")
                if len(years) == 4:
                    frame = self.prediction_problems[["year", "grade", "item", "variant"]].copy()
                    frame["prediction"] = fit["prediction"]
                    frame = frame.groupby(["year", "grade", "item"], as_index=False).prediction.mean()
                    frame["model"] = name
                    self.final_difficulties.extend(frame.to_dict("records"))
        difficulty_names = [f"difficulty_{kind}_{learner}" for kind in ("tfidf", "rubert") for learner in ("catboost", "lightgbm")]
        outputs["difficulty_ensemble"] = np.mean([outputs[n] for n in difficulty_names], axis=0)
        outputs["all_text_ensemble"] = .5*outputs["direct_ensemble"] + .5*outputs["difficulty_ensemble"]
        outputs["robust_ensemble"] = .5*outputs["baseline_grade_mean"] + .25*outputs["direct_ensemble"] + .25*outputs["difficulty_ensemble"]
        self.bundle_cache[cache_key] = outputs
        if len(years) == 4:
            joblib.dump(dict(direct=direct, calibration=calibration_models,
                             config=CONFIG), ROOT / "models/final_models.joblib")
        return outputs

    def select_without_outer_labels(self, years):
        """Only choose baseline versus a fixed ensemble; never tune weights."""
        years = sorted(years)
        if len(years) < 2:
            return "baseline_grade_mean", {"reason": "only one cutoff training season"}
        scores = {"baseline_grade_mean": [], "robust_ensemble": []}
        for year in years:
            prediction = self.bundle(set(years)-{year}, {year})
            idx = self.cutoffs.index[self.cutoffs.year == year]
            for name in scores:
                scores[name].extend((np.abs(prediction[name][idx]-self.cutoffs.loc[idx, "cutoff"].to_numpy())**1.5).tolist())
        means = {name: float(np.mean(v)) for name, v in scores.items()}
        return min(means, key=means.get), means

    def evaluate(self):
        records, splits, selections = [], [], []
        protocols = {"diagonal": diagonal_folds(self.cutoffs),
                     "single_case_year_purged": single_case_year_folds(self.cutoffs),
                     "forward": expanding_folds(self.cutoffs)}
        for protocol, folds in protocols.items():
            for fold in folds:
                train_years = set(self.cutoffs.loc[fold.train, "year"])
                test_years = set(self.cutoffs.loc[fold.valid, "year"])
                print(protocol, fold.name, "train years", sorted(train_years), "test", self.cutoffs.loc[fold.valid, ["year", "grade"]].values.tolist(), flush=True)
                prediction = self.bundle(train_years, test_years,
                    record_fold=fold.name if protocol == "diagonal" else None,
                    record_valid_indices=fold.valid if protocol == "diagonal" else None)
                selected, inner = self.select_without_outer_labels(train_years)
                prediction = dict(prediction, nested_selected=prediction[selected])
                selections.append(dict(protocol=protocol, fold=fold.name, selected=selected, inner_scores=inner))
                splits.append(dict(protocol=protocol, fold=fold.name,
                    train=self.cutoffs.loc[fold.train, ["year", "grade"]].to_dict("records"),
                    validation=self.cutoffs.loc[fold.valid, ["year", "grade"]].to_dict("records"),
                    purged=self.cutoffs.loc[fold.purged, ["year", "grade"]].to_dict("records")))
                for name, values in prediction.items():
                    for idx in fold.valid:
                        row = self.cutoffs.loc[idx]
                        records.append(dict(protocol=protocol, fold=fold.name, year=int(row.year),
                            grade=int(row.grade), actual=row.cutoff, model=name, prediction=float(values[idx])))
        predictions = pd.DataFrame(records)
        predictions["error"] = predictions.prediction-predictions.actual
        predictions.to_csv(ROOT / "results/validation_predictions.csv", index=False)
        summary = []
        for (protocol, model), group in predictions.groupby(["protocol", "model"]):
            summary.append(dict(protocol=protocol, model=model, n_cases=len(group), n_years=group.year.nunique(),
                                **regression_metrics(group.actual, group.prediction)))
        pd.DataFrame(summary).sort_values(["protocol", "power_1_5"]).to_csv(ROOT / "results/metrics.csv", index=False)
        (ROOT / "results/splits.json").write_text(json.dumps(splits, indent=2)+"\n")
        (ROOT / "results/nested_selections.json").write_text(json.dumps(selections, indent=2)+"\n")
        pd.DataFrame(self.curves).to_csv(ROOT / "results/learning_curves.csv", index=False)
        pd.DataFrame(self.difficulty_evaluations).to_csv(ROOT / "results/difficulty_validation.csv", index=False)
        self.forecast(predictions)

    def forecast(self, validation):
        years = set(self.cutoffs.year)
        outputs = self.bundle(years, {2026})
        selected, inner = self.select_without_outer_labels(years)
        rows = []
        for name, pred in outputs.items():
            for idx, case in self.cases[self.cases.year == 2026].iterrows():
                rows.append(dict(year=2026, season="2026/2027", grade=int(case.grade),
                    model=name, prediction=float(pred[idx]), selected_final=(name == selected)))
        pd.DataFrame(rows).to_csv(ROOT / "results/forecast_2026.csv", index=False)
        current_idx = self.cases.index[(self.cases.year == 2026) & (self.cases.grade == 11)][0]
        point = float(outputs[selected][current_idx])
        grade11 = self.cutoffs[self.cutoffs.grade == 11].cutoff.to_numpy()
        # Transparent small-sample reference interval; assumes stationary iid
        # approximately Gaussian year effects. It is not conformal coverage.
        halfwidth = float(student_t.ppf(.95, df=len(grade11)-1) * grade11.std(ddof=1) * np.sqrt(1+1/len(grade11)))
        baseline = grade11.mean()
        loyo = validation[(validation.protocol == "single_case_year_purged") & (validation.model == "nested_selected") & (validation.grade == 11)]
        residual_halfwidth = float(np.max(np.abs(loyo.error)))
        interval = [float(max(0, min(point-halfwidth, baseline-halfwidth))),
                    float(min(100, max(point+halfwidth, baseline+halfwidth)))]
        # Exact finite-sample calibration rank demonstrates the resolution limit.
        n_years = len(years)
        required_rank_90 = int(np.ceil((n_years+1)*.9))
        result = dict(target_season="2026/2027", grade=11, scale_assumption=100,
            final_model=selected, final_prediction=point,
            fixed_robust_ensemble=float(outputs["robust_ensemble"][current_idx]),
            direct_ensemble=float(outputs["direct_ensemble"][current_idx]),
            difficulty_ensemble=float(outputs["difficulty_ensemble"][current_idx]),
            final_selection_inner_power_1_5=inner,
            plausible_interval=interval,
            interval_method="Envelope of baseline and selected-point 90% Student-t forecast intervals using four grade-11 historical cutoffs; stationarity/Gaussian-year-effect assumptions; no demonstrated coverage guarantee",
            baseline_stationary_student_t_interval=[float(max(0, baseline-halfwidth)), float(min(100, baseline+halfwidth))],
            loyo_nested_grade11_max_abs_residual=residual_halfwidth,
            distribution_free_90_interval=[0, 100], calibration_years=n_years,
            conformal_90_required_rank=required_rank_90,
            limitations=["Only four comparable cutoff seasons / twelve correlated labels",
                "2026 paper has 30 problems; historical comparable papers have 15, 17, 20, 20",
                "No 2026 item weights or attached task data in provided PDF",
                "Native PDF text does not encode all diagrams and formula images",
                "Difficulty labels are cohort-dependent pooled-variant partial-credit rates",
                "Entrant ability and final capacity/quota are unobserved",
                "Frozen transformer heads, not full encoder fine-tuning",
                "No claim of high predictive precision is supported"])
        (ROOT / "results/final_prediction.json").write_text(json.dumps(result, indent=2)+"\n")
        pd.DataFrame(self.final_difficulties).drop_duplicates().to_csv(ROOT / "results/predicted_difficulty.csv", index=False)
        pd.DataFrame(self.timing).to_csv(ROOT / "results/fit_timings.csv", index=False)
        # Final-fit importance is explanatory, never a feature-selection input.
        direct = self.direct_fit(years)
        importances = []
        for name, item in direct["objects"].items():
            if not name.endswith(("catboost", "lightgbm")):
                continue
            model, _, names = item
            values = model.feature_importances_ if isinstance(model, CatBoostRegressor) else model.booster_.feature_importance(importance_type="gain")
            values = np.asarray(values, float)
            values = values / max(values.sum(), 1e-12) * 100
            importances.extend(dict(model=name, feature=f, importance=float(v)) for f, v in zip(names, values))
        final_difficulty_models = {}
        banned = set(self.corpus.problems.loc[self.corpus.problems.year == 2026, "family_hash"])
        for kind in ("tfidf", "rubert"):
            for learner in ("catboost", "lightgbm"):
                name = f"difficulty_{kind}_{learner}"
                fit = self.difficulty_fit(years | {2021}, kind, learner, banned)
                final_difficulty_models[name] = fit
                model = fit["model"]
                names = fit["representation"].problem_names + ["grade_minus_10"]
                values = model.feature_importances_ if isinstance(model, CatBoostRegressor) else model.booster_.feature_importance(importance_type="gain")
                values = np.asarray(values, float)
                values = values / max(values.sum(), 1e-12) * 100
                importances.extend(dict(model=name, feature=f, importance=float(v)) for f, v in zip(names, values))
        joblib.dump(final_difficulty_models, ROOT / "models/final_difficulty_models.joblib")
        pd.DataFrame(importances).to_csv(ROOT / "results/feature_importance.csv", index=False)
        print("FINAL", json.dumps(result, indent=2), flush=True)


def main():
    start = time.perf_counter()
    exp = Experiment()
    exp.evaluate()
    (ROOT / "results/run_metadata.json").write_text(json.dumps(dict(seed=SEED,
        duration_seconds=time.perf_counter()-start, configuration=CONFIG,
        training_location="local macOS arm64 CPU; four threads; no remote compute",
        cutoff_data_sha256=hashlib.sha256((ROOT / "data/processed/cutoffs.csv").read_bytes()).hexdigest()), indent=2)+"\n")


if __name__ == "__main__":
    main()

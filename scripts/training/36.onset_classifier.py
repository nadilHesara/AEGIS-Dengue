"""
Outbreak-onset classifier: "this district is quiet now -- will it be in outbreak
h weeks from now?"

Motivation, from scripts/34. Every forecaster in the benchmark separates
outbreak weeks from quiet weeks well (ROC-AUC 0.85-0.91 at h=1), but almost all
of that is *continuation*: an outbreak this week predicts one next week. On the
weeks an early-warning system exists for -- onsets, the first outbreak week
after a quiet spell -- the same forecasters are near chance (onset AUC
0.53-0.62). A regression model trained on all weeks spends its capacity on the
bulk of the data, where the answer is "about what it is now".

This script asks the transition question directly. Rows are forecast origins
at which the district is below its outbreak threshold; the label is whether
the target week h ahead is at or above it. The threshold is scripts/15's peak
threshold (90th percentile of the district's pre-test history), fitted per fold
and so never on the test year. Features are scripts/32's origin features plus
the origin level relative to the threshold.

Arms
    onset_lgbm             LightGBM binary classifier, class-balanced,
                           early-stopped on the validation year's log loss.
    onset_lgbm_noclimate   the same without any climate-derived feature --
                           the question of whether weather carries transition
                           information that case history does not.

Scored in scripts/34-style on the same folds: ROC-AUC and PR-AUC over quiet
origins (headline folds), onset AUC (onset targets vs quiet targets), and the
regression forecasters' own score (forecast / threshold) on the identical rows
for comparison.

Output: results/benchmark/onset/onset_predictions.parquet, onset_scores.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.evaluation.long_horizon import (  # noqa: E402
    BENCHMARK_DIR,
    HEADLINE_FOLDS,
    HORIZONS,
    build_benchmark_folds,
    load_pipeline,
    months_for,
)

OUTPUT_DIR = BENCHMARK_DIR / "onset"
ONSET_GAP = 4

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "is_unbalance": True,
    "verbose": -1,
}


def load_tabular():
    path = PROJECT_DIR / "scripts" / "training" / "32.long_horizon_tabular.py"
    spec = importlib.util.spec_from_file_location("tabular", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tabular"] = module
    spec.loader.exec_module(module)
    return module


def roc_auc(score, label):
    from scipy import stats

    positives, negatives = label.sum(), (~label).sum()
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = stats.rankdata(score)
    return float((ranks[label].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def pr_auc(score, label):
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(label, score)) if label.any() else np.nan


def main() -> int:
    import lightgbm as lgb

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    parser.add_argument("--seeds", type=int, default=3)
    arguments = parser.parse_args()

    baseline = load_pipeline()
    tabular = load_tabular()
    tabular.baseline = baseline
    folds = build_benchmark_folds(baseline.folds_module)
    tensors = baseline.folds_module.load_tensors("v2")
    months = months_for(baseline.folds_module)
    names = [str(n) for n in tensors["feature_names"]]
    y, y_mask, period_id = tensors["y"], tensors["y_mask"], tensors["period_id"]

    binary = np.load(PROJECT_DIR / "data" / "processed" / "adjacency.npz")["A_binary"].astype(np.float32)
    np.fill_diagonal(binary, 0.0)
    neighbours = binary / np.clip(binary.sum(axis=1, keepdims=True), 1.0, None)
    start = tensors["start_date"].astype("datetime64[D]")
    doy_angle = 2.0 * np.pi * ((start - start.astype("datetime64[Y]")).astype(int) + 1) / 365.25

    frames = []
    for fold in folds:
        started = time.perf_counter()
        fit = period_id <= fold["fit_end_period"]
        threshold = baseline.naive.peak_thresholds(y, y_mask, fit)
        statistics = baseline.folds_module.fit_fold_statistics(tensors, months, fit)
        scaled = baseline.folds_module.transform(tensors, months, statistics)
        history = np.where(y_mask[fit] == 1, y[fit], np.nan)
        fallback = np.nanmean(history, axis=0)
        cube, labels = tabular.origin_features(scaled, names, neighbours, doy_angle)

        above = (y >= threshold[None, :]) & (y_mask == 1)
        below = (y < threshold[None, :]) & (y_mask == 1)
        onset = np.zeros_like(above)
        for t in range(ONSET_GAP, len(y)):
            onset[t] = above[t] & below[t - ONSET_GAP : t].all(axis=0)

        for horizon in arguments.horizons:
            rows = tabular.build_rows(tensors, cube, labels, fold, horizon, doy_angle, fallback)
            columns = list(rows["labels"])
            origins = np.arange(tabular.MIN_ORIGIN, len(y) - horizon)
            targets = origins + horizon
            origin_counts = np.expm1(rows["anchor"])
            quiet = origin_counts < threshold[None, :]
            relative = np.log1p(origin_counts) - np.log1p(threshold)[None, :]
            X = np.concatenate([rows["X"], relative[..., None],
                                np.broadcast_to(np.log1p(threshold), relative.shape)[..., None]],
                               axis=-1)
            columns = columns + ["level_vs_threshold", "log_threshold"]
            label = above[targets]
            observed = y_mask[targets] == 1
            is_onset = onset[targets]

            for arm in ("onset_lgbm", "onset_lgbm_noclimate"):
                keep_columns = [i for i, c in enumerate(columns)
                                if arm == "onset_lgbm" or not tabular.is_climate(c)]
                district = [keep_columns.index(columns.index("district"))]

                def subset(split):
                    selector = rows["split"][split][:, None] & quiet & observed
                    return X[selector][:, keep_columns], label[selector], selector

                X_train, l_train, _ = subset("train")
                X_val, l_val, _ = subset("val")
                X_test, l_test, selector = subset("test")
                predictions = []
                for seed in range(arguments.seeds):
                    params = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
                    model = lgb.train(
                        params,
                        lgb.Dataset(X_train, l_train.astype(float), categorical_feature=district),
                        num_boost_round=2000,
                        valid_sets=[lgb.Dataset(X_val, l_val.astype(float), categorical_feature=district)],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
                    predictions.append(model.predict(X_test, num_iteration=model.best_iteration))
                index = np.argwhere(selector)
                frames.append(pd.DataFrame({
                    "method": arm, "fold_id": fold["fold_id"], "horizon": horizon,
                    "target_period_id": period_id[targets][index[:, 0]],
                    "node_id": index[:, 1], "probability": np.mean(predictions, axis=0),
                    "event": l_test, "onset": is_onset[selector],
                }))
        print(f"  fold {fold['fold_id']} ({fold['test_year']}) "
              f"{time.perf_counter() - started:5.1f}s", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_parquet(OUTPUT_DIR / "onset_predictions.parquet", index=False)

    # Score, alongside the regression forecasters' own score on the same rows.
    point_frames = []
    for path in sorted((BENCHMARK_DIR / "predictions").glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["method", "fold_id", "split", "horizon", "seed",
                                               "target_period_id", "node_id", "prediction"])
        frame = frame[frame["split"] == "test"]
        frame = frame.groupby(["method", "fold_id", "horizon", "target_period_id", "node_id"],
                              as_index=False)["prediction"].mean()
        point_frames.append(frame)
    point = pd.concat(point_frames, ignore_index=True)

    thresholds = {f["fold_id"]: baseline.naive.peak_thresholds(
        y, y_mask, period_id <= f["fit_end_period"]) for f in folds}
    index_of = {int(p): i for i, p in enumerate(period_id)}
    base = predictions[predictions["method"] == "onset_lgbm"][
        ["fold_id", "horizon", "target_period_id", "node_id", "event", "onset"]]
    scored = [predictions]
    for method, group in point.groupby("method"):
        merged = base.merge(group, on=["fold_id", "horizon", "target_period_id", "node_id"])
        merged["probability"] = merged["prediction"] / [
            thresholds[f][n] for f, n in zip(merged["fold_id"], merged["node_id"])]
        merged["method"] = method
        scored.append(merged.drop(columns="prediction"))
    # Persistence score on the same rows: the origin count over the threshold.
    persistence = base.copy()
    origin_index = persistence["target_period_id"].map(index_of).to_numpy() - persistence["horizon"].to_numpy()
    persistence["probability"] = y[origin_index, persistence["node_id"].to_numpy()] / [
        thresholds[f][n] for f, n in zip(persistence["fold_id"], persistence["node_id"])]
    persistence["method"] = "persistence"
    scored.append(persistence)
    scored = pd.concat(scored, ignore_index=True)
    scored = scored[scored["fold_id"].isin(HEADLINE_FOLDS)]

    records = []
    for (method, horizon, fold_id), group in scored.groupby(["method", "horizon", "fold_id"]):
        event = group["event"].to_numpy().astype(bool)
        onset = group["onset"].to_numpy().astype(bool)
        score = group["probability"].to_numpy()
        non_event = ~event
        records.append({
            "method": method, "horizon": horizon, "fold_id": fold_id, "rows": len(group),
            "events": int(event.sum()), "onsets": int(onset.sum()),
            "transition_auc": roc_auc(score, event),
            "transition_pr_auc": pr_auc(score, event),
            "onset_auc": roc_auc(score[onset | non_event], onset[onset | non_event]),
            "base_rate": float(event.mean()),
        })
    scores = pd.DataFrame(records)
    scores.to_csv(OUTPUT_DIR / "onset_scores.csv", index=False)
    summary = scores.groupby(["method", "horizon"])[
        ["transition_auc", "transition_pr_auc", "onset_auc", "base_rate"]].mean().reset_index()
    summary.to_csv(OUTPUT_DIR / "onset_summary.csv", index=False)
    for column in ("transition_auc", "transition_pr_auc", "onset_auc"):
        print(f"\n{column} (quiet origins, headline folds)")
        print(summary.pivot(index="method", columns="horizon", values=column).round(3).to_string())
    print("\nbase rate")
    print(summary.pivot(index="method", columns="horizon", values="base_rate").round(3).head(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

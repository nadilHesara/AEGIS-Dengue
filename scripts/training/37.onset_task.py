"""
Outbreak onset as a separate prediction task (revision of 29 September 2026).

Replaces the two inconsistent onset definitions of the 24 September benchmark
(scripts/34 scored "onset weeks" against quiet weeks; scripts/36 predicted
"above threshold at t + h" from any below-threshold origin). Here every
comparator answers one question, on one population:

    At an at-risk origin -- the district's last four weeks observed and below
    its outbreak threshold -- will the first threshold crossing come within
    the next K weeks?  (K = 4 and 8; src/evaluation/onset.py.)

Comparators, all scored on the identical eligible origins of each test year:
    persistence          origin count / threshold (a ranking score)
    climatology_rate     the district's historical rate of the event for the
                         same time of year (origin week +/- 2), from pre-test
                         years only, shrunk to the district's overall rate.
                         A probability.
    onset_lgbm           LightGBM classifier trained on this label, at-risk
    onset_lgbm_noclimate origins only, early-stopped on the validation year;
                         without class re-weighting so that its output is a
                         probability. Same features as scripts/36.
    forecasters          for each h <= K the benchmark has (1,2,3,4 and, for
                         K = 8, also 6 and 8), a per-week score; the window
                         score is the maximum over those horizons. Point
                         models: forecast / threshold. Quantile and NB models:
                         P(Y_t+h >= threshold) (NB exact; quantile models by
                         CDF interpolation). The maximum of marginal
                         probabilities is a lower bound on the window
                         probability, never a product of marginals under an
                         independence assumption; its Brier score is reported
                         with that caveat.

Scores: ROC-AUC, PR-AUC with the positive rate, Brier score for probabilities,
and an alarm rule whose cutoff is set on the previous test year to alarm on
ALARM_BUDGET of eligible origins: origin-level sensitivity, realised alarm
rate, false alarms per district-year, and event-level detection with lead time
(each onset counted once).

Output: results/benchmark/onset_task/{origins.parquet, scores_per_fold.csv,
summary.csv, events.csv, counts.csv}
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

from src.evaluation import onset  # noqa: E402
from src.evaluation.long_horizon import (  # noqa: E402
    BENCHMARK_DIR,
    HEADLINE_FOLDS,
    HOLDOUT_FOLD,
    HORIZONS,
    PREDICTIONS_DIR,
    QUANTILES,
    build_benchmark_folds,
    load_pipeline,
    months_for,
    nb_exceedance,
    quantile_column,
    quantile_exceedance,
)
from src.models.climate_ablation import week_of_year  # noqa: E402

OUTPUT_DIR = BENCHMARK_DIR / "onset_task"
FORECASTERS = {"lgbm_v2": "point", "gru_v2": "point", "gru_v2_quantile_lw": "quantile",
               "nb_shared_v2": "nb", "chronos2_joint": "quantile", "chronos2_joint_ft": "quantile"}
QCOLS = [quantile_column(q) for q in QUANTILES]
PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 40,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "verbose": -1}
SHRINK = 10.0  # pseudo-origins pulling the seasonal rate towards the district's overall rate


def load_tabular():
    path = PROJECT_DIR / "scripts" / "training" / "32.long_horizon_tabular.py"
    spec = importlib.util.spec_from_file_location("tabular", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tabular"] = module
    spec.loader.exec_module(module)
    return module


def cell_scores(method: str, kind: str, thresholds: dict[int, np.ndarray]) -> pd.DataFrame:
    """Per (fold, horizon, target, node) test-cell score and probability for one forecaster."""

    frame = pd.read_parquet(PREDICTIONS_DIR / f"{method}.parquet")
    frame = frame[(frame["method"] == method) & (frame["split"] == "test")]
    keys = ["fold_id", "horizon", "target_period_id", "node_id"]
    threshold = np.array([thresholds[f][n] for f, n in zip(frame["fold_id"], frame["node_id"])])
    if kind == "point":
        frame = frame.assign(score=frame["prediction"] / threshold)
        return frame.groupby(keys, as_index=False)["score"].mean().assign(probability=np.nan)
    if kind == "nb":
        probability = nb_exceedance(frame["nb_mu"].to_numpy(float), frame["nb_alpha"].to_numpy(float),
                                    threshold)
    else:
        probability = quantile_exceedance(frame[QCOLS].to_numpy(float), threshold)
    frame = frame.assign(probability=probability)
    out = frame.groupby(keys, as_index=False)["probability"].mean()  # seed mixture
    out["score"] = out["probability"]
    return out


def main() -> int:
    import lightgbm as lgb

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=3)
    arguments = parser.parse_args()

    baseline = load_pipeline()
    tabular = load_tabular()
    tabular.baseline = baseline
    folds = build_benchmark_folds(baseline.folds_module, include_holdout=True)
    tensors = baseline.folds_module.load_tensors("v2")
    months = months_for(baseline.folds_module)
    names = [str(n) for n in tensors["feature_names"]]
    y, y_mask, period_id = tensors["y"], tensors["y_mask"], tensors["period_id"]
    y_filled = np.nan_to_num(y, nan=0.0)
    n_periods, n_nodes = y.shape
    woy = week_of_year(tensors["start_date"])

    binary = np.load(PROJECT_DIR / "data" / "processed" / "adjacency.npz")["A_binary"].astype(np.float32)
    np.fill_diagonal(binary, 0.0)
    neighbours = binary / np.clip(binary.sum(axis=1, keepdims=True), 1.0, None)
    start = tensors["start_date"].astype("datetime64[D]")
    doy_angle = 2.0 * np.pi * ((start - start.astype("datetime64[Y]")).astype(int) + 1) / 365.25

    thresholds = {f["fold_id"]: baseline.naive.peak_thresholds(
        y, y_mask, period_id <= f["fit_end_period"]) for f in folds}
    forecaster_cells = {m: cell_scores(m, k, thresholds) for m, k in FORECASTERS.items()
                        if (PREDICTIONS_DIR / f"{m}.parquet").exists()}

    origin_rows, event_rows, count_rows = [], [], []
    for fold in folds:
        started = time.perf_counter()
        fit = period_id <= fold["fit_end_period"]
        threshold = thresholds[fold["fold_id"]]
        statistics = baseline.folds_module.fit_fold_statistics(tensors, months, fit)
        scaled = baseline.folds_module.transform(tensors, months, statistics)
        cube, cube_labels = tabular.origin_features(scaled, names, neighbours, doy_angle)
        level = np.log1p(y_filled) - np.log1p(threshold)[None, :]
        X_all = np.concatenate([cube, level[..., None],
                                np.broadcast_to(np.log1p(threshold), level.shape)[..., None]], axis=-1)
        columns = cube_labels + ["level_vs_threshold", "log_threshold"]
        events = onset.onset_events(y_filled, y_mask, threshold)

        for window in onset.WINDOWS:
            label, _ = onset.window_label(y_filled, y_mask, threshold, window)
            first = int(period_id.min())
            eligible = {
                "train": onset.eligible_origins(y_filled, y_mask, threshold, window, period_id,
                                                first, fold["train_end_period"]),
                "val": onset.eligible_origins(y_filled, y_mask, threshold, window, period_id,
                                              fold["val_start_period"], fold["val_end_period"]),
                "test": onset.eligible_origins(y_filled, y_mask, threshold, window, period_id,
                                               fold["test_start_period"], fold["test_end_period"]),
                "history": onset.eligible_origins(y_filled, y_mask, threshold, window, period_id,
                                                  first, fold["fit_end_period"]),
            }
            eligible["train"][: tabular.MIN_ORIGIN] = False
            test_t, test_n = np.nonzero(eligible["test"])
            base = pd.DataFrame({"fold_id": fold["fold_id"], "window": window,
                                 "t": test_t, "node_id": test_n,
                                 "label": label[test_t, test_n].astype(int)})
            count_rows.append({"fold_id": fold["fold_id"], "window": window,
                               "eligible_origins": len(base), "positives": int(base["label"].sum()),
                               "train_origins": int(eligible["train"].sum())})

            scored = []
            # Persistence.
            scored.append(base.assign(method="persistence",
                                      score=y_filled[test_t, test_n] / threshold[test_n],
                                      probability=np.nan))

            # Seasonal base rate from pre-test history.
            h_t, h_n = np.nonzero(eligible["history"])
            history = pd.DataFrame({"node_id": h_n, "woy": woy[h_t], "label": label[h_t, h_n]})
            district_rate = history.groupby("node_id")["label"].mean().reindex(range(n_nodes)).fillna(
                history["label"].mean())
            rates = np.empty(len(base))
            for i, (t, node) in enumerate(zip(test_t, test_n)):
                near = np.abs(((history["woy"] - woy[t] + 26) % 52) - 26) <= 2
                local = history[(history["node_id"] == node) & near]["label"]
                rates[i] = (local.sum() + SHRINK * district_rate[node]) / (len(local) + SHRINK)
            scored.append(base.assign(method="climatology_rate", score=rates, probability=rates))

            # Classifiers on this label.
            for arm in ("onset_lgbm", "onset_lgbm_noclimate"):
                keep = [i for i, c in enumerate(columns) if arm == "onset_lgbm" or not tabular.is_climate(c)]
                district = [keep.index(columns.index("district"))]

                def rows_of(split):
                    t_idx, n_idx = np.nonzero(eligible[split])
                    return X_all[t_idx, n_idx][:, keep], label[t_idx, n_idx].astype(float)

                X_train, l_train = rows_of("train")
                X_val, l_val = rows_of("val")
                X_test = X_all[test_t, test_n][:, keep]
                probabilities = []
                for seed in range(arguments.seeds):
                    params = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
                    model = lgb.train(params, lgb.Dataset(X_train, l_train, categorical_feature=district),
                                      num_boost_round=2000,
                                      valid_sets=[lgb.Dataset(X_val, l_val, categorical_feature=district)],
                                      callbacks=[lgb.early_stopping(100, verbose=False)])
                    probabilities.append(model.predict(X_test, num_iteration=model.best_iteration))
                probability = np.mean(probabilities, axis=0) if len(X_test) else np.array([])
                scored.append(base.assign(method=arm, score=probability, probability=probability))

            # Forecasters: max over the horizons inside the window.
            horizons = [h for h in HORIZONS if h <= window]
            for method, cells in forecaster_cells.items():
                cells = cells[cells["fold_id"] == fold["fold_id"]]
                expanded = pd.concat([base.assign(horizon=h, target_period_id=period_id[np.minimum(base["t"] + h, n_periods - 1)])
                                      for h in horizons], ignore_index=True)
                merged = expanded.merge(cells, on=["fold_id", "horizon", "target_period_id", "node_id"], how="left")
                if merged["score"].isna().any():
                    raise ValueError(f"{method}: missing forecasts for eligible origins, fold {fold['fold_id']}")
                window_score = merged.groupby(["t", "node_id"], as_index=False)[["score", "probability"]].max()
                scored.append(base.merge(window_score, on=["t", "node_id"]).assign(method=method))

            frame = pd.concat(scored, ignore_index=True)
            origin_rows.append(frame)

            # Event list for this fold's test year (each onset once).
            test_weeks = (period_id >= fold["test_start_period"]) & (period_id <= fold["test_end_period"])
            event_rows.append((fold["fold_id"], window, events, eligible["test"], test_weeks))
        print(f"  fold {fold['fold_id']} ({fold['test_year']}) {time.perf_counter() - started:5.1f}s", flush=True)

    origins = pd.concat(origin_rows, ignore_index=True)

    # Alarm rule: cutoff from the previous fold's test scores, at ALARM_BUDGET.
    records, event_records = [], []
    event_lookup = {(f, w): (e, el, tw) for f, w, e, el, tw in event_rows}
    fold_ids = sorted(origins["fold_id"].unique())
    for (method, window), group in origins.groupby(["method", "window"]):
        for fold_id in fold_ids:
            if fold_id - 1 not in fold_ids:
                continue
            current = group[group["fold_id"] == fold_id]
            previous = group[group["fold_id"] == fold_id - 1]
            cutoff = onset.alarm_cutoff(previous["score"].to_numpy())
            label = current["label"].to_numpy().astype(bool)
            score = current["score"].to_numpy()
            alarm = score >= cutoff
            probability = current["probability"].to_numpy()
            has_probability = np.isfinite(probability).all() and len(probability)

            events, eligible, test_weeks = event_lookup[(fold_id, window)]
            alarm_grid = np.zeros_like(eligible)
            alarm_grid[current["t"].to_numpy(), current["node_id"].to_numpy()] = alarm
            detected = onset.event_detection(alarm_grid, eligible, events, window, test_weeks)
            detected["method"], detected["window"], detected["fold_id"] = method, window, fold_id
            event_records.append(detected)

            records.append({
                "method": method, "window": window, "fold_id": fold_id,
                "origins": len(current), "positives": int(label.sum()),
                "positive_rate": float(label.mean()) if len(label) else np.nan,
                "roc_auc": onset.roc_auc(score, label), "pr_auc": onset.pr_auc(score, label),
                "brier": float(np.mean((probability - label) ** 2)) if has_probability else np.nan,
                "cutoff": cutoff, "alarm_rate": float(alarm.mean()) if len(alarm) else np.nan,
                "sensitivity": float(alarm[label].mean()) if label.any() else np.nan,
                "false_alarms_per_district_year": float((alarm & ~label).sum() / n_nodes),
                "events": len(detected), "events_detected": int(detected["detected"].sum()),
                "median_lead_weeks": float(detected["lead_weeks"].median()) if detected["detected"].any() else np.nan,
            })
    scores = pd.DataFrame(records)
    events_frame = pd.concat(event_records, ignore_index=True)

    def summarise(subset: pd.DataFrame) -> pd.DataFrame:
        grouped = subset.groupby(["method", "window"])
        out = grouped[["roc_auc", "pr_auc", "brier", "positive_rate", "alarm_rate", "sensitivity",
                       "false_alarms_per_district_year"]].mean()
        out["origins"] = grouped["origins"].sum()
        out["positives"] = grouped["positives"].sum()
        out["events"] = grouped["events"].sum()
        out["events_detected"] = grouped["events_detected"].sum()
        out["event_detection_rate"] = out["events_detected"] / out["events"]
        leads = events_frame[events_frame["fold_id"].isin(subset["fold_id"].unique())]
        out["median_lead_weeks"] = leads[leads["detected"]].groupby(["method", "window"])["lead_weeks"].median()
        return out.reset_index()

    summary = summarise(scores[scores["fold_id"].isin(HEADLINE_FOLDS)])
    holdout = summarise(scores[scores["fold_id"] == HOLDOUT_FOLD])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    origins.to_parquet(OUTPUT_DIR / "origins.parquet", index=False)
    scores.to_csv(OUTPUT_DIR / "scores_per_fold.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    holdout.to_csv(OUTPUT_DIR / "summary_holdout_2026.csv", index=False)
    events_frame.to_csv(OUTPUT_DIR / "events.csv", index=False)
    pd.DataFrame(count_rows).to_csv(OUTPUT_DIR / "counts.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nOnset task, headline folds")
    print(summary.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

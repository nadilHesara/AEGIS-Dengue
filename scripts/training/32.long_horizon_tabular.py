"""
Tabular arms of the long-horizon benchmark: gradient-boosted trees and ridge.

The short paper compared the GRU only against persistence and seasonal naive.
A reviewer's first question is whether a well-built tabular model -- the class
that wins most forecasting competitions on data this size, and the
"Random Forest / XGBoost" rows of the Sri Lanka GNN papers (Weng et al. 2024;
GulMohamed et al. 2026) -- does as well or better. This script answers it on
the identical cells, folds and target as scripts/31.

Target: the same anchored log residual the GRU learns,
    log1p(y[t+h]) - log1p(y[t]),
so a zero prediction is persistence for every method in the benchmark.

Features at forecast origin t, district n, all computed from the fold-scaled v2
tensor (scripts/14's per-fold imputation and scaling, fitted before the test
year), so nothing after t and nothing fitted on the test year is used:
    - every v2 channel at t (cases, climate, 4/8/12-week climate means,
      season, outbreak history);
    - log-case lags t-1..t-12, 4/8/12-week means and 12-week max of log cases,
      1- and 4-week changes;
    - climate further back: 4-week rain/humidity/temperature means at t-4 and
      t-8 and the 12-week rain mean at t-12, which reach the 5-10 week delay
      scripts/17 measured even at h=1;
    - contiguity-neighbour mean of log cases and its 4-week change, and the
      national mean and its 4-week change;
    - season of the *target* week (known in advance), the horizon's calendar;
    - district id (categorical for the trees, one-hot for ridge).

Arms
    lgbm_v2            LightGBM, L2 on the residual, early stopping on the
                       validation year, 3 bagging seeds.
    lgbm_v2_noclimate  the same without any climate-derived feature. For trees,
                       dropping columns does not change the model's capacity the
                       way it does for the GRU's first layer, so this is the
                       clean ablation in this model family.
    ridge_v2           linear ARX on the same features, alpha chosen on the
                       validation year. The linear counterpart; deterministic.

Output: results/benchmark/predictions/<arm>.parquet and, for the trees,
results/benchmark/lgbm_importance.csv (gain by feature group and horizon).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.evaluation.long_horizon import (  # noqa: E402
    BENCHMARK_DIR,
    HORIZONS,
    build_benchmark_folds,
    cell_frame,
    load_pipeline,
    months_for,
    save_predictions,
)
from src.models.climate_ablation import CLIMATE_FEATURES  # noqa: E402

ARMS = ("lgbm_v2", "lgbm_v2_noclimate", "ridge_v2")
MIN_ORIGIN = 24  # enough history for every lag below

LGBM_PARAMS = {
    "objective": "l2",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
}

baseline = None


def is_climate(name: str) -> bool:
    base = name.split("_roll")[0].split("@")[0]
    return base in CLIMATE_FEATURES or name.startswith("rain_") or name.startswith("humid_") \
        or name.startswith("temp_")


def origin_features(scaled: np.ndarray, names: list[str], neighbours: np.ndarray,
                    doy_angle: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Feature cube [periods, nodes, features] indexed by forecast origin."""

    index = {name: i for i, name in enumerate(names)}
    cases = scaled[:, :, index["cases_log1p"]]
    n_periods, n_nodes, _ = scaled.shape

    def shift(a: np.ndarray, k: int) -> np.ndarray:
        out = np.full_like(a, np.nan)
        out[k:] = a[: n_periods - k]
        return out

    def rolling(a: np.ndarray, w: int, fn) -> np.ndarray:
        out = np.full_like(a, np.nan)
        for t in range(w - 1, n_periods):
            out[t] = fn(a[t - w + 1 : t + 1], axis=0)
        return out

    columns, labels = [scaled[:, :, i] for i in range(scaled.shape[2])], list(names)
    for k in range(1, 13):
        columns.append(shift(cases, k)); labels.append(f"cases_lag{k}")
    for w in (4, 8, 12):
        columns.append(rolling(cases, w, np.mean)); labels.append(f"cases_mean{w}")
    columns.append(rolling(cases, 12, np.max)); labels.append("cases_max12")
    columns.append(cases - shift(cases, 1)); labels.append("cases_diff1")
    columns.append(cases - shift(cases, 4)); labels.append("cases_diff4")

    for source, short in (("rainfall_daily_mean_mm", "rain"),
                          ("relative_humidity_mean", "humid"),
                          ("temperature_mean_c", "temp")):
        roll4 = scaled[:, :, index[f"{source}_roll4"]]
        for k in (4, 8):
            columns.append(shift(roll4, k)); labels.append(f"{short}_roll4_lag{k}")
    columns.append(shift(scaled[:, :, index["rainfall_daily_mean_mm_roll12"]], 12))
    labels.append("rain_roll12_lag12")

    neighbour = cases @ neighbours.T
    columns.append(neighbour); labels.append("neighbour_cases")
    columns.append(neighbour - shift(neighbour, 4)); labels.append("neighbour_diff4")
    national = np.repeat(cases.mean(axis=1, keepdims=True), n_nodes, axis=1)
    columns.append(national); labels.append("national_cases")
    columns.append(national - shift(national, 4)); labels.append("national_diff4")

    columns.append(np.repeat(np.arange(n_nodes)[None, :], n_periods, axis=0).astype(np.float32))
    labels.append("district")
    return np.stack(columns, axis=-1).astype(np.float32), labels


def build_rows(tensors, cube, labels, fold, horizon, doy_angle, anchor_fallback):
    """Design matrix for one fold and horizon, rows = (origin, node)."""

    n_periods, n_nodes = tensors["y"].shape
    origins = np.arange(MIN_ORIGIN, n_periods - horizon)
    targets = origins + horizon

    y, y_mask = tensors["y"], tensors["y_mask"]
    anchor_counts = np.where(np.isnan(y[origins]), anchor_fallback[None, :], y[origins])
    anchor = np.log1p(anchor_counts)

    features = cube[origins]  # [origins, nodes, f]
    season = np.stack([np.sin(doy_angle[targets]), np.cos(doy_angle[targets])], axis=-1)
    season = np.repeat(season[:, None, :], n_nodes, axis=1)
    features = np.concatenate([features, season.astype(np.float32)], axis=-1)
    labels = labels + ["target_doy_sin", "target_doy_cos"]

    residual = np.log1p(np.nan_to_num(y[targets], nan=0.0)) - anchor
    split = baseline.folds_module.assign_windows(tensors["period_id"][targets], fold)
    return {
        "X": features, "labels": labels, "residual": residual.astype(np.float32),
        "observed": y_mask[targets] == 1, "anchor": anchor.astype(np.float32),
        "target_period_id": tensors["period_id"][targets], "split": split,
    }


def flatten(rows, split, keep_observed):
    selector = rows["split"][split]
    X = rows["X"][selector].reshape(-1, rows["X"].shape[-1])
    r = rows["residual"][selector].reshape(-1)
    observed = rows["observed"][selector].reshape(-1)
    if keep_observed:
        return X[observed], r[observed]
    return X, r


def to_cases(rows, split, residual_prediction):
    selector = rows["split"][split]
    shape = rows["anchor"][selector].shape
    log_prediction = rows["anchor"][selector] + residual_prediction.reshape(shape)
    return np.clip(np.expm1(log_prediction), 0.0, None), rows["target_period_id"][selector]


def fit_lgbm(rows, columns, seed):
    import lightgbm as lgb

    X_train, r_train = flatten(rows, "train", True)
    X_val, r_val = flatten(rows, "val", True)
    district = columns.index("district")
    train = lgb.Dataset(X_train[:, columns_index(columns)], r_train,
                        categorical_feature=[district], free_raw_data=False)
    valid = lgb.Dataset(X_val[:, columns_index(columns)], r_val, reference=train,
                        categorical_feature=[district])
    params = dict(LGBM_PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    model = lgb.train(params, train, num_boost_round=3000, valid_sets=[valid],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


_COLUMN_CACHE: dict = {}


def columns_index(columns):
    return _COLUMN_CACHE[tuple(columns)]


def fit_ridge(rows, columns):
    from sklearn.linear_model import Ridge

    idx = columns_index(columns)
    district = columns.index("district")

    def design(split, keep_observed):
        X, r = flatten(rows, split, keep_observed)
        X = X[:, idx]
        onehot = np.eye(25, dtype=np.float32)[X[:, district].astype(int)]
        X = np.delete(X, district, axis=1)
        X = np.nan_to_num(X, nan=0.0)  # lags before the series start only
        return np.concatenate([X, onehot], axis=1), r

    X_train, r_train = design("train", True)
    X_val, r_val = design("val", True)
    best = None
    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
        model = Ridge(alpha=alpha).fit(X_train, r_train)
        error = float(np.mean((model.predict(X_val) - r_val) ** 2))
        if best is None or error < best[0]:
            best = (error, alpha, model)
    return best[2], design


def run(arms, folds, horizons, seeds):
    tensors = baseline.folds_module.load_tensors("v2")
    months = months_for(baseline.folds_module)
    names = [str(n) for n in tensors["feature_names"]]

    binary = np.load(PROJECT_DIR / "data" / "processed" / "adjacency.npz")["A_binary"].astype(np.float32)
    np.fill_diagonal(binary, 0.0)
    neighbours = binary / np.clip(binary.sum(axis=1, keepdims=True), 1.0, None)

    start = tensors["start_date"].astype("datetime64[D]")
    doy = (start - start.astype("datetime64[Y]")).astype(int) + 1
    doy_angle = 2.0 * np.pi * doy / 365.25

    frames = {arm: [] for arm in arms}
    importance = []
    for fold in folds:
        started = time.perf_counter()
        statistics = baseline.folds_module.fit_fold_statistics(
            tensors, months, tensors["period_id"] <= fold["fit_end_period"])
        scaled = baseline.folds_module.transform(tensors, months, statistics)
        fit_mask = tensors["period_id"] <= fold["fit_end_period"]
        history = np.where(tensors["y_mask"][fit_mask] == 1, tensors["y"][fit_mask], np.nan)
        fallback = np.nanmean(history, axis=0)

        cube, cube_labels = origin_features(scaled, names, neighbours, doy_angle)
        line = []
        for horizon in horizons:
            rows = build_rows(tensors, cube, cube_labels, fold, horizon, doy_angle, fallback)
            labels = rows["labels"]
            all_columns = list(labels)
            no_climate = [c for c in labels if not is_climate(c)]
            for columns in (all_columns, no_climate):
                _COLUMN_CACHE[tuple(columns)] = [labels.index(c) for c in columns]

            for arm in arms:
                columns = no_climate if arm == "lgbm_v2_noclimate" else all_columns
                if arm.startswith("lgbm"):
                    for seed in range(seeds):
                        model = fit_lgbm(rows, columns, seed)
                        for split in ("val", "test"):
                            X, _ = flatten(rows, split, False)
                            prediction, periods = to_cases(
                                rows, split, model.predict(X[:, columns_index(columns)],
                                                           num_iteration=model.best_iteration))
                            frames[arm].append(cell_frame(arm, fold, split, horizon, periods,
                                                          prediction, seed))
                        if seed == 0:
                            gain = model.feature_importance("gain")
                            importance.append(pd.DataFrame({
                                "arm": arm, "fold_id": fold["fold_id"], "horizon": horizon,
                                "feature": columns, "gain": gain,
                                "best_iteration": model.best_iteration}))
                else:
                    model, design = fit_ridge(rows, columns)
                    for split in ("val", "test"):
                        X, _ = design(split, False)
                        prediction, periods = to_cases(rows, split, model.predict(X))
                        frames[arm].append(cell_frame(arm, fold, split, horizon, periods,
                                                      prediction, -1))
            line.append(f"h{horizon}")
        print(f"  fold {fold['fold_id']} ({fold['test_year']}) {' '.join(line)} "
              f"{time.perf_counter() - started:5.1f}s", flush=True)

    return frames, (pd.concat(importance, ignore_index=True) if importance else None)


def main() -> int:
    global baseline
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    parser.add_argument("--seeds", type=int, default=3)
    arguments = parser.parse_args()

    baseline = load_pipeline()
    folds = build_benchmark_folds(baseline.folds_module)
    if arguments.folds:
        folds = [f for f in folds if f["fold_id"] in arguments.folds]

    started = time.perf_counter()
    frames, importance = run(arguments.arms, folds, tuple(arguments.horizons), arguments.seeds)
    for arm, arm_frames in frames.items():
        path = save_predictions(arm_frames, arm)
        print(f"{arm}: wrote {path.relative_to(PROJECT_DIR)}")
    if importance is not None:
        BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
        importance.to_csv(BENCHMARK_DIR / "lgbm_importance.csv", index=False)
    print(f"done in {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Score the long-horizon benchmark: accuracy, climate attribution, ensembles,
calibrated intervals and outbreak early warning, on one common set of cells.

Reads every results/benchmark/predictions/*.parquet written by scripts 31-33,
adds the naive baselines computed on the same cells, and scores everything on
the intersection of cells all methods cover. Nothing here trains a network.

Naive baselines (all causal: they read only periods at or before the origin)
    persistence            y[origin]                        (the short paper's bar)
    seasonal_naive         y[target - 52]
    climatology            geometric mean of y[target - 52k], k = 1..5
    seasonal_persistence   persistence moved by the typical seasonal change
                           from origin week to target week:
                           log1p y[o] + mean_k log1p y[t-52k] - mean_k log1p y[o-52k]

Ensembles (point; all members' seed-mean predictions, combined in log1p space)
    ens_equal     equal weights over a fixed, pre-specified member list.
    ens_stacked   non-negative least-squares weights over the same members,
                  fitted per horizon on fold f-1's *test* cells (a year no
                  member trained on) and applied to fold f.
    ens_top3      select-then-average: for fold f and horizon h, the three
                  deployable forecasters with the lowest MAE on fold f-1's test
                  year, averaged. Members are chosen on a year no candidate
                  trained on and never on the year being scored.

Probabilistic forecasts
    Every point method is turned into a predictive distribution by split
    conformal calibration in log space: the empirical quantiles of
    log1p(y) - log1p(y_hat) on fold f-1's test cells (same horizon) are added
    to fold f's point forecasts. Methods with native quantiles (the quantile
    GRU and Chronos) are scored raw and after per-level recalibration (the
    shift that makes each level's calibration coverage match its nominal
    level). Scored by WIS (Bracher et al. 2021) and 50%/95% coverage.

Early warning
    An event is a week at or above the district's peak threshold (90th
    percentile of its pre-test history, scripts/15's definition). Scored by
    ROC-AUC of predicted/threshold, the warning rule "forecast >= threshold"
    (sensitivity, precision), and onset detection: the first week of an
    exceedance run (after >= 4 weeks below), warned h weeks ahead.

Statistics: headline folds 1, 2, 3, 6, 7, 8, 9 (fold 0 is calibration only);
paired t-test and fold win counts over those 7 folds, as the README does.

Revision of 29 September 2026 (src/evaluation/revision.py)
    - Two estimands, never mixed in one ranking: `mae` is the mean of the
      three per-seed MAEs; `mae_ens` is the MAE of the seed-averaged forecast.
    - Skill is 1 - (headline mean MAE) / (persistence headline mean MAE);
      `skill_yearly` is the mean of the seven per-year percentage reductions.
    - Holm adjustment over the declared primary family (PRIMARY x horizons vs
      persistence); every other comparison is labelled exploratory.
    - Time-block bootstrap (4-week blocks, all districts jointly) for the
      primary comparisons, the NB ablations and the climate controls.
    - NB arms: head, sharing and feature ablations; mean vs median point
      forecasts; dispersion / zero / peak / 2017 / district calibration.
    - nb_shared_v4_tsel: the thermal-suitability centre chosen per fold on the
      validation year's NB log-likelihood.
    - mix_nb_chronos: an equal-weight *distribution* mixture of nb_shared_v2
      (its three seeds) and chronos2_joint; qavg_nb_chronos averages their
      quantiles instead, for comparison.
    - Intervals: previous-year conformal, rolling conformal (last 52 weeks of
      residuals already observed at the origin) and adaptive conformal
      (Gibbs and Candes 2021), with 50/80/95% coverage and width by year.
    - Fold 10 (2026) is scored separately against the hold-out declaration.
    - A compute table from results/benchmark/compute/*.csv.

Outputs under results/benchmark/: benchmark_report.md, tables/*.csv, figures/*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.evaluation.long_horizon import (  # noqa: E402
    BENCHMARK_DIR,
    HEADLINE_FOLDS,
    HOLDOUT_FOLD,
    PREDICTIONS_DIR,
    QUANTILES,
    build_benchmark_folds,
    load_pipeline,
    quantile_column,
    truth_frame,
    weighted_interval_score,
)
from src.evaluation import revision  # noqa: E402

# The declared primary family (results/benchmark/holdout_declaration.md). Holm
# adjusts these 5 x 8 comparisons with persistence; everything else is exploratory.
PRIMARY = ("gru_v2_quantile_lw", "nb_shared_v2", "chronos2_joint", "lgbm_v2", "ens_equal")
THERMAL_VARIANTS = ("nb_shared_v4_t26", "nb_shared_v4_t27", "nb_shared_v4",
                    "nb_shared_v4_t29", "nb_shared_v4_t30")
ONLINE_CONFORMAL = ("persistence", "lgbm_v2", "gru_v2", "ens_equal")
ACI_GAMMA = 0.05  # fixed in advance; weekly updates of the miscoverage level
COMPUTE_DIR = BENCHMARK_DIR / "compute"

ABLATIONS = (
    # (arm, reference, label, family)
    ("gru_v2_shuffled", "gru_v2", "GRU: climate shuffled", "climate"),
    ("gru_v2_climatology", "gru_v2", "GRU: climate -> training climatology", "climate"),
    ("gru_v2_noclimate", "gru_v2", "GRU: climate removed", "climate"),
    ("lgbm_v2_noclimate", "lgbm_v2", "LightGBM: climate removed", "climate"),
    ("lgbm_v2_climatology", "lgbm_v2", "LightGBM: climate -> training climatology", "climate"),
    ("chronos2_climate", "chronos2", "Chronos-2: climate covariates added", "climate"),
    ("nb_shared_v2_shuffled", "nb_shared_v2", "NB: climate shuffled", "climate"),
    ("nb_shared_v2_climatology", "nb_shared_v2", "NB: climate -> training climatology", "climate"),
    ("nb_shared_v2_noclimate", "nb_shared_v2", "NB: climate removed", "climate"),
    ("nb_shared_v2_wxlag1", "nb_shared_v2", "NB: weather one week late", "availability"),
    ("gru_v2_quantile", "gru_v2", "Head: pinball vs MSE (separate, v2)", "nb"),
    ("gru_v2_nb", "gru_v2", "Head: NB vs MSE (separate, v2)", "nb"),
    ("gru_v2_nb", "gru_v2_quantile", "Head: NB vs pinball (separate, v2)", "nb"),
    ("nb_shared_v2", "gru_v2_nb", "Sharing: shared trunk vs separate (NB, v2)", "nb"),
    ("nb_shared_v3", "nb_shared_v2", "Features: v3 vs v2 (NB shared)", "nb"),
    ("nb_shared_v4", "nb_shared_v2", "Features: v4 vs v2 (NB shared)", "nb"),
    ("nb_shared_v4_tsel", "nb_shared_v4", "v4 thermal centre selected on validation", "nb"),
    ("nb_shared_v2_mean", "nb_shared_v2", "NB point: mean vs median", "nb"),
    ("gru_v2_nb_mean", "gru_v2_nb", "NB point (separate): mean vs median", "nb"),
    ("gru_v1", "gru_v2", "GRU: outbreak-history removed (v1)", "other"),
    ("gcn_v2", "gru_v2", "GRU: contiguity graph added", "spatial"),
    ("adaptive_v2", "gru_v2", "GRU: learned graph added", "spatial"),
    ("gru_v2_lw", "gru_v2", "GRU: level-weighted loss", "other"),
    ("gru_v2_quantile_lw", "gru_v2", "GRU: level-weighted pinball", "other"),
    ("chronos2_joint", "chronos2", "Chronos-2: joint 25-district", "spatial"),
    ("chronos2_ft", "chronos2", "Chronos-2: LoRA fine-tuned", "other"),
    ("chronos2_joint_ft", "chronos2_joint", "Chronos-2 joint: LoRA fine-tuned", "other"),
    ("mix_nb_chronos", "nb_shared_v2", "Mixture NB+Chronos vs NB", "ensemble"),
    ("mix_nb_chronos", "chronos2_joint", "Mixture NB+Chronos vs Chronos-2 joint", "ensemble"),
)

TABLE_DIR = BENCHMARK_DIR / "tables"
FIGURE_DIR = BENCHMARK_DIR / "figures"
REPORT_PATH = BENCHMARK_DIR / "benchmark_report.md"

ENSEMBLE_MEMBERS = ("gru_v2", "lgbm_v2", "chronos2")
# Candidates for the select-then-average ensemble: every deployable forecaster.
# Ablation controls and the naive baselines are excluded, as are other ensembles.
NOT_CANDIDATES = {"gru_v2_shuffled", "lgbm_v2_noclimate", "persistence", "seasonal_naive",
                  "climatology", "seasonal_persistence", "ens_equal", "ens_stacked", "ens_top3",
                  "gru_v2_climatology", "gru_v2_noclimate", "lgbm_v2_climatology",
                  "nb_shared_v2_shuffled", "nb_shared_v2_climatology", "nb_shared_v2_noclimate",
                  "nb_shared_v2_wxlag1", "nb_shared_v4_t26", "nb_shared_v4_t27",
                  "nb_shared_v4_t29", "nb_shared_v4_t30", "mix_nb_chronos", "qavg_nb_chronos"}
TOP_K = 3
QCOLS = [quantile_column(q) for q in QUANTILES]
ONSET_GAP = 4
TARGET_FAR = 0.10


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def load_predictions(exclude: set[str]) -> pd.DataFrame:
    frames = []
    for path in sorted(PREDICTIONS_DIR.glob("*.parquet")):
        if path.stem in exclude:
            continue
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def naive_predictions(cells: pd.DataFrame, tensors, folds) -> pd.DataFrame:
    """Naive baselines on exactly the given (fold, split, horizon, period, node) cells."""

    y = np.where(tensors["y_mask"] == 1, tensors["y"], np.nan).astype(np.float64)
    period_id = tensors["period_id"]
    index_of = {int(p): i for i, p in enumerate(period_id)}
    fold_by_id = {f["fold_id"]: f for f in folds}

    out = []
    for fold_id, group in cells.groupby("fold_id"):
        fold = fold_by_id[fold_id]
        fit = period_id <= fold["fit_end_period"]
        fallback = np.nanmean(y[fit], axis=0)
        filled = np.where(np.isnan(y), fallback[None, :], y)
        log_filled = np.log1p(filled)

        target = np.array([index_of[int(p)] for p in group["target_period_id"]])
        node = group["node_id"].to_numpy().astype(int)
        origin = target - group["horizon"].to_numpy().astype(int)

        def seasonal_mean(anchor_index):
            lags = [anchor_index - 52 * k for k in range(1, 6)]
            values = np.stack([log_filled[np.clip(l, 0, None), node] for l in lags])
            valid = np.stack([l >= 0 for l in lags])
            return np.nanmean(np.where(valid, values, np.nan), axis=0)

        persistence = filled[origin, node]
        seasonal = filled[np.clip(target - 52, 0, None), node]
        clim_target = seasonal_mean(target)
        clim_origin = seasonal_mean(origin)
        predictions = {
            "persistence": persistence,
            "seasonal_naive": seasonal,
            "climatology": np.expm1(clim_target),
            "seasonal_persistence": np.expm1(np.log1p(persistence) + clim_target - clim_origin),
        }
        for method, values in predictions.items():
            frame = group[["fold_id", "split", "horizon", "target_period_id", "node_id"]].copy()
            frame["method"] = method
            frame["seed"] = -1
            frame["prediction"] = np.clip(values, 0.0, None).astype(np.float32)
            out.append(frame)
    return pd.concat(out, ignore_index=True)


def thresholds_by_fold(tensors, folds, naive_module) -> dict[int, np.ndarray]:
    out = {}
    for fold in folds:
        fit = tensors["period_id"] <= fold["fit_end_period"]
        out[fold["fold_id"]] = naive_module.peak_thresholds(tensors["y"], tensors["y_mask"], fit)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_frame(frame: pd.DataFrame) -> pd.Series:
    observed = frame[frame["observed"] == 1]
    error = observed["prediction"] - observed["actual"]
    peak = observed[observed["actual"] >= observed["threshold"]]
    return pd.Series({
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "peak_mae": float(np.abs(peak["prediction"] - peak["actual"]).mean()) if len(peak) else np.nan,
        "cells": len(observed),
    })


def per_fold_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """Metrics per method, seed, fold and horizon on the test split."""

    test = table[table["split"] == "test"]
    return (
        test.groupby(["method", "seed", "fold_id", "horizon"])
        .apply(score_frame, include_groups=False)
        .reset_index()
    )


def headline(metrics: pd.DataFrame) -> pd.DataFrame:
    """Headline-fold means. Multi-seed methods: seed-mean of per-seed metrics
    (`single`) and the seed-ensemble's metric (`ens`, seed == -2)."""

    rows = []
    head = metrics[metrics["fold_id"].isin(HEADLINE_FOLDS)]
    for (method, horizon), group in head.groupby(["method", "horizon"]):
        seeds = sorted(group["seed"].unique())
        single = group[group["seed"] >= 0] if any(s >= 0 for s in seeds) else group[group["seed"] == -1]
        ens = group[group["seed"] == -2] if -2 in seeds else single
        per_fold_single = single.groupby("fold_id")[["mae", "peak_mae", "rmse"]].mean()
        seed_sd = (single.groupby(["fold_id"])["mae"].std().mean()
                   if single["seed"].nunique() > 1 else np.nan)
        rows.append({
            "method": method, "horizon": horizon,
            "mae": per_fold_single["mae"].mean(),
            "peak_mae": per_fold_single["peak_mae"].mean(),
            "rmse": per_fold_single["rmse"].mean(),
            "mae_ens": ens.groupby("fold_id")["mae"].mean().mean(),
            "peak_mae_ens": ens.groupby("fold_id")["peak_mae"].mean().mean(),
            "mae_2017": per_fold_single.loc[1, "mae"] if 1 in per_fold_single.index else np.nan,
            "mae_ex2017": per_fold_single.drop(index=1, errors="ignore")["mae"].mean(),
            "seed_sd": seed_sd,
            "n_folds": int(per_fold_single["mae"].notna().sum()),
        })
    return pd.DataFrame(rows)


def paired(metrics: pd.DataFrame, method: str, reference: str, horizon: int,
           column: str = "mae", ensemble: bool = False) -> dict:
    """Paired comparison over headline folds (method - reference)."""

    def fold_values(name):
        subset = metrics[(metrics["method"] == name) & (metrics["horizon"] == horizon)
                         & metrics["fold_id"].isin(HEADLINE_FOLDS)]
        if ensemble and (subset["seed"] == -2).any():
            subset = subset[subset["seed"] == -2]
        elif (subset["seed"] >= 0).any():
            subset = subset[subset["seed"] >= 0]
        return subset.groupby("fold_id")[column].mean()

    a, b = fold_values(method), fold_values(reference)
    common = a.index.intersection(b.index)
    difference = (a[common] - b[common]).to_numpy()
    if len(difference) < 3:
        return {"delta": np.nan, "t": np.nan, "p": np.nan, "wins": 0, "n": len(difference)}
    t, p = stats.ttest_1samp(difference, 0.0)
    try:
        w_p = stats.wilcoxon(difference).pvalue
    except ValueError:
        w_p = np.nan
    return {"delta": float(difference.mean()), "t": float(t), "p": float(p),
            "wilcoxon_p": float(w_p), "wins": int((difference < 0).sum()), "n": len(difference)}


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

def seed_mean(table: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-seed methods to their seed-mean (seed = -2)."""

    multi = table[table["seed"] >= 0]
    keys = ["method", "fold_id", "split", "horizon", "target_period_id", "node_id"]
    agg = {"prediction": "mean"}
    for column in QCOLS:
        if column in multi.columns:
            agg[column] = "mean"
    collapsed = multi.groupby(keys, as_index=False).agg(agg)
    collapsed["seed"] = -2
    return collapsed


def build_ensembles(point: pd.DataFrame, members, folds) -> pd.DataFrame:
    """Equal-weight and stacked ensembles in log1p space.

    `point` holds one prediction per (method, cell): seed-means for multi-seed
    methods, the single prediction otherwise.
    """

    from scipy.optimize import nnls

    keys = ["fold_id", "split", "horizon", "target_period_id", "node_id"]
    wide = point[point["method"].isin(members)].pivot_table(
        index=keys, columns="method", values="prediction").dropna()
    logs = np.log1p(wide[list(members)])

    equal = wide.index.to_frame(index=False)
    equal["prediction"] = np.expm1(logs.mean(axis=1).to_numpy())
    equal["method"] = "ens_equal"

    truth = point.drop_duplicates(keys).set_index(keys)[["actual", "observed"]]
    stacked_frames, weight_rows = [], []
    fold_ids = sorted(wide.index.get_level_values("fold_id").unique())
    for horizon in sorted(wide.index.get_level_values("horizon").unique()):
        for fold_id in fold_ids:
            if fold_id - 1 not in fold_ids:
                continue
            calibration = logs.xs((fold_id - 1, "test", horizon), level=("fold_id", "split", "horizon"),
                                  drop_level=False)
            observed = truth.loc[calibration.index]
            keep = observed["observed"].to_numpy() == 1
            weights, _ = nnls(calibration.to_numpy()[keep], np.log1p(observed["actual"].to_numpy()[keep]))
            weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(members), 1 / len(members))
            weight_rows.append({"fold_id": fold_id, "horizon": horizon,
                                **{m: w for m, w in zip(members, weights)}})
            target = logs.xs((fold_id, horizon), level=("fold_id", "horizon"), drop_level=False)
            frame = target.index.to_frame(index=False)
            frame["prediction"] = np.expm1(target.to_numpy() @ weights)
            frame["method"] = "ens_stacked"
            stacked_frames.append(frame)

    ensembles = pd.concat([equal] + stacked_frames, ignore_index=True)
    ensembles["seed"] = -1
    return ensembles, pd.DataFrame(weight_rows)


def build_top_k(point: pd.DataFrame, k: int = TOP_K) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select-then-average: for fold f and horizon h, the k candidates with the
    lowest MAE on fold f-1's test year, averaged in log1p space.

    Selection reads only fold f-1, a year every candidate forecast out of sample,
    so the member choice is never informed by the fold being scored.
    """

    keys = ["fold_id", "split", "horizon", "target_period_id", "node_id"]
    candidates = point[~point["method"].isin(NOT_CANDIDATES)
                       & ~point["method"].str.endswith("_mean")
                       & ~point["method"].str.endswith("_tsel")]
    test = candidates[(candidates["split"] == "test") & (candidates["observed"] == 1)]
    error = (test.assign(ae=(test["prediction"] - test["actual"]).abs())
             .groupby(["method", "fold_id", "horizon"])["ae"].mean().reset_index())
    wide = candidates.pivot_table(index=keys, columns="method", values="prediction")
    frames, chosen = [], []
    for (fold_id, horizon), _ in wide.groupby(level=["fold_id", "horizon"]):
        previous = error[(error["fold_id"] == fold_id - 1) & (error["horizon"] == horizon)]
        if previous.empty:
            continue
        members = list(previous.nsmallest(k, "ae")["method"])
        block = wide.xs((fold_id, horizon), level=("fold_id", "horizon"), drop_level=False)
        block = block[members].dropna()
        frame = block.index.to_frame(index=False)
        frame["prediction"] = np.expm1(np.log1p(block).mean(axis=1).to_numpy())
        frames.append(frame)
        chosen.append({"fold_id": fold_id, "horizon": horizon, "members": ", ".join(members)})
    out = pd.concat(frames, ignore_index=True)
    out["method"] = f"ens_top{k}"
    out["seed"] = -1
    return out, pd.DataFrame(chosen)


# ---------------------------------------------------------------------------
# Probabilistic
# ---------------------------------------------------------------------------

def conformal_quantiles(point: pd.DataFrame, folds) -> pd.DataFrame:
    """Split-conformal predictive quantiles for every point method."""

    fold_ids = sorted(point["fold_id"].unique())
    out = []
    for (method, horizon), group in point.groupby(["method", "horizon"]):
        residual = np.log1p(group["actual"]) - np.log1p(group["prediction"])
        for fold_id in fold_ids:
            if fold_id - 1 not in fold_ids:
                continue
            calibration = (group["fold_id"] == fold_id - 1) & (group["split"] == "test") \
                & (group["observed"] == 1)
            if calibration.sum() < 50:
                continue
            shifts = np.quantile(residual[calibration], QUANTILES)
            target = group[(group["fold_id"] == fold_id) & (group["split"] == "test")]
            frame = target[["method", "fold_id", "horizon", "target_period_id", "node_id",
                            "actual", "observed", "threshold"]].copy()
            base = np.log1p(target["prediction"].to_numpy())[:, None] + shifts[None, :]
            frame[QCOLS] = np.clip(np.expm1(base), 0.0, None)
            frame["variant"] = "conformal"
            out.append(frame)
    return pd.concat(out, ignore_index=True)


def native_quantiles(native: pd.DataFrame, folds, recalibrate: bool) -> pd.DataFrame:
    """Raw or per-level recalibrated quantiles for methods that produce them."""

    fold_ids = sorted(native["fold_id"].unique())
    out = []
    for (method, horizon), group in native.groupby(["method", "horizon"]):
        for fold_id in fold_ids:
            target = group[(group["fold_id"] == fold_id) & (group["split"] == "test")]
            if not len(target) or fold_id == min(fold_ids):
                continue
            values = np.log1p(target[QCOLS].to_numpy())
            if recalibrate:
                calibration = group[(group["fold_id"] == fold_id - 1) & (group["split"] == "test")
                                    & (group["observed"] == 1)]
                gap = np.log1p(calibration["actual"].to_numpy())[:, None] - np.log1p(calibration[QCOLS].to_numpy())
                shifts = np.array([np.quantile(gap[:, i], level) for i, level in enumerate(QUANTILES)])
                values = values + shifts[None, :]
            values = np.sort(values, axis=1)
            frame = target[["method", "fold_id", "horizon", "target_period_id", "node_id",
                            "actual", "observed", "threshold"]].copy()
            frame[QCOLS] = np.clip(np.expm1(values), 0.0, None)
            frame["variant"] = "recalibrated" if recalibrate else "native"
            out.append(frame)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def probabilistic_scores(quantiles: pd.DataFrame) -> pd.DataFrame:
    observed = quantiles[(quantiles["observed"] == 1) & quantiles["fold_id"].isin(HEADLINE_FOLDS)]
    rows = []
    for (method, variant, horizon, fold_id), group in observed.groupby(
            ["method", "variant", "horizon", "fold_id"]):
        q = group[QCOLS].to_numpy()
        actual = group["actual"].to_numpy()
        wis = weighted_interval_score(actual, q)
        peak = actual >= group["threshold"].to_numpy()
        rows.append({
            "method": method, "variant": variant, "horizon": horizon, "fold_id": fold_id,
            "wis": wis.mean(), "peak_wis": wis[peak].mean() if peak.any() else np.nan,
            "cov50": np.mean((actual >= q[:, 2]) & (actual <= q[:, 4])),
            "cov80": np.mean((actual >= q[:, 1]) & (actual <= q[:, 5])),
            "cov95": np.mean((actual >= q[:, 0]) & (actual <= q[:, 6])),
            "exceed_prob": np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Early warning
# ---------------------------------------------------------------------------

def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    positives, negatives = label.sum(), (~label).sum()
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = stats.rankdata(score)
    return float((ranks[label].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def early_warning(point: pd.DataFrame, tensors, thresholds) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Event-level scores per method, fold and horizon on the test split.

    Two views. (1) Every outbreak week (actual >= threshold) against every other
    week: ROC-AUC of forecast/threshold and the naive rule "forecast >=
    threshold". (2) Onsets only -- the first outbreak week after >= ONSET_GAP
    observed quiet weeks, the week a warning is actually for -- against quiet
    weeks: onset AUC, and detection at a matched false-alarm rate, where each
    method's alarm cutoff is the one that gave TARGET_FAR false alarms on the
    previous fold's test year (so the cutoff is never fitted on the scored year).
    """

    y, mask = tensors["y"], tensors["y_mask"]
    index_of = {int(p): i for i, p in enumerate(tensors["period_id"])}

    onset_sets = {}
    for fold_id, threshold in thresholds.items():
        above = (y >= threshold[None, :]) & (mask == 1)
        below = (y < threshold[None, :]) & (mask == 1)
        onset = np.zeros_like(above)
        for t in range(ONSET_GAP, len(y)):
            onset[t] = above[t] & below[t - ONSET_GAP : t].all(axis=0)
        onset_sets[fold_id] = onset

    test = point[(point["split"] == "test") & (point["observed"] == 1)].copy()
    test["score"] = test["prediction"] / test["threshold"]
    test["event"] = test["actual"] >= test["threshold"]
    target_index = test["target_period_id"].map(index_of).to_numpy()
    test["onset"] = [onset_sets[f][t, n] for f, t, n in zip(
        test["fold_id"].to_numpy(), target_index, test["node_id"].to_numpy().astype(int))]

    rows, onset_rows = [], []
    for (method, horizon), by_method in test.groupby(["method", "horizon"]):
        for fold_id in sorted(by_method["fold_id"].unique()):
            if fold_id not in HEADLINE_FOLDS:
                continue
            group = by_method[by_method["fold_id"] == fold_id]
            label = group["event"].to_numpy()
            score = group["score"].to_numpy()
            warn = score >= 1.0
            tp = int((warn & label).sum())
            rows.append({
                "method": method, "horizon": horizon, "fold_id": fold_id,
                "auc": roc_auc(score, label), "events": int(label.sum()),
                "sensitivity": tp / label.sum() if label.sum() else np.nan,
                "precision": tp / warn.sum() if warn.sum() else np.nan,
                "false_alarm_rate": float((warn & ~label).sum() / max((~label).sum(), 1)),
            })

            quiet = ~label
            onset = group["onset"].to_numpy()
            if not onset.any():
                continue
            keep = onset | quiet
            previous = by_method[(by_method["fold_id"] == fold_id - 1) & ~by_method["event"]]
            cutoff = np.quantile(previous["score"], 1 - TARGET_FAR) if len(previous) else np.nan
            alarm = score >= cutoff
            onset_rows.append({
                "method": method, "horizon": horizon, "fold_id": fold_id,
                "onsets": int(onset.sum()),
                "detected": int((warn & onset).sum()),
                "onset_auc": roc_auc(score[keep], onset[keep]),
                "detected_at_far": int((alarm & onset).sum()),
                "realised_far": float((alarm & quiet).sum() / max(quiet.sum(), 1)),
            })
    return pd.DataFrame(rows), pd.DataFrame(onset_rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt(value, digits=2):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def cell_abs_error(table: pd.DataFrame, method: str) -> pd.Series:
    """Seed-averaged absolute error per observed test cell (the `mae` estimand)."""

    rows = table[(table["method"] == method) & (table["split"] == "test") & (table["observed"] == 1)]
    rows = rows[rows["seed"] >= 0] if (rows["seed"] >= 0).any() else rows[rows["seed"] == -1]
    error = (rows["prediction"] - rows["actual"]).abs()
    keys = ["fold_id", "horizon", "target_period_id", "node_id"]
    return error.groupby([rows[k] for k in keys]).mean()


def bootstrap_rows(table, pairs, folds_used, label, n_boot) -> list[dict]:
    """Time-block bootstrap for (method, reference, horizon) pairs on `folds_used`."""

    cache: dict[str, pd.Series] = {}
    out = []
    for method, reference, horizon in pairs:
        for name in (method, reference):
            if name not in cache:
                cache[name] = cell_abs_error(table, name)
        a, b = cache[method], cache[reference]
        if a.empty or b.empty:
            continue
        joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").reset_index()
        joined = joined[(joined["horizon"] == horizon) & joined["fold_id"].isin(folds_used)]
        if joined.empty:
            continue
        joined["diff"] = joined["a"] - joined["b"]
        out.append({"set": label, "method": method, "reference": reference, "horizon": horizon,
                    **revision.block_bootstrap(joined[["fold_id", "target_period_id", "diff"]],
                                               n_boot=n_boot)})
    return out


def thermal_selection(predictions: pd.DataFrame, truth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """nb_shared_v4_tsel: per fold, the thermal centre with the lowest validation NLL."""

    present = [m for m in THERMAL_VARIANTS if m in set(predictions["method"])]
    if len(present) < 2:
        return pd.DataFrame(), pd.DataFrame()
    subset = predictions[predictions["method"].isin(present)]
    scored = subset[subset["split"] == "val"].merge(truth, on=["target_period_id", "node_id"])
    nll = revision.nb_validation_nll(scored)
    choice = revision.select_by_validation(nll, present)
    frames = []
    for fold_id, method in choice.items():
        rows = subset[(subset["method"] == method) & (subset["fold_id"] == fold_id)].copy()
        rows["method"] = "nb_shared_v4_tsel"
        frames.append(rows)
    nll["chosen"] = [choice.get(int(f)) == m for f, m in zip(nll["fold_id"], nll["method"])]
    return pd.concat(frames, ignore_index=True), nll


def mixture_ensembles(table: pd.DataFrame, nb: str = "nb_shared_v2",
                      other: str = "chronos2_joint") -> pd.DataFrame:
    """Distribution mixture and quantile average of an NB model and Chronos-2 (test cells)."""

    keys = ["fold_id", "split", "horizon", "target_period_id", "node_id"]
    nb_rows = table[(table["method"] == nb) & (table["seed"] >= 0) & (table["split"] == "test")]
    other_rows = table[(table["method"] == other) & (table["seed"] == -1) & (table["split"] == "test")]
    if nb_rows.empty or other_rows.empty:
        return pd.DataFrame()
    mu = nb_rows.pivot_table(index=keys, columns="seed", values="nb_mu")
    alpha = nb_rows.pivot_table(index=keys, columns="seed", values="nb_alpha")
    nb_q = nb_rows.groupby(keys)[QCOLS].mean()
    other_q = other_rows.set_index(keys)[QCOLS]
    common = mu.dropna().index.intersection(other_q.index)
    mixed = revision.mixture_quantiles(mu.loc[common].to_numpy(), alpha.loc[common].to_numpy(),
                                       other_q.loc[common].to_numpy())
    averaged = np.expm1(0.5 * (np.log1p(nb_q.loc[common].to_numpy())
                               + np.log1p(other_q.loc[common].to_numpy())))
    frames = []
    for name, values in (("mix_nb_chronos", mixed), ("qavg_nb_chronos", averaged)):
        frame = common.to_frame(index=False)
        frame[QCOLS] = np.sort(values, axis=1)
        frame["prediction"] = frame[quantile_column(0.5)]
        frame["method"] = name
        frame["seed"] = -1
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def skill_columns(summary: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Pooled skill (ratio of headline means) and the mean of per-year skills."""

    persistence = summary[summary["method"] == "persistence"].set_index("horizon")
    summary = summary.copy()
    summary["skill"] = [1 - r.mae / persistence.loc[r.horizon, "mae"] for r in summary.itertuples()]
    summary["skill_ens"] = [1 - r.mae_ens / persistence.loc[r.horizon, "mae"] for r in summary.itertuples()]
    head = metrics[metrics["fold_id"].isin(HEADLINE_FOLDS)]
    multi = set(head.loc[head["seed"] >= 0, "method"])
    single = head[((head["seed"] >= 0) & head["method"].isin(multi))
                  | ((head["seed"] == -1) & ~head["method"].isin(multi))]
    per_fold = single.groupby(["method", "horizon", "fold_id"])["mae"].mean().reset_index()
    base = per_fold[per_fold["method"] == "persistence"].set_index(["horizon", "fold_id"])["mae"]
    reference = base.reindex(pd.MultiIndex.from_arrays([per_fold["horizon"], per_fold["fold_id"]])).to_numpy()
    per_fold["skill_fold"] = 1 - per_fold["mae"].to_numpy() / reference
    yearly = per_fold.groupby(["method", "horizon"])["skill_fold"].mean().rename("skill_yearly")
    return summary.merge(yearly.reset_index(), on=["method", "horizon"], how="left")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--n-boot", type=int, default=2000)
    arguments = parser.parse_args()

    baseline = load_pipeline()
    folds = build_benchmark_folds(baseline.folds_module, include_holdout=True)
    tensors = baseline.folds_module.load_tensors("v2")
    thresholds = thresholds_by_fold(tensors, folds, baseline.naive)
    truth = truth_frame(tensors)

    predictions = load_predictions(set(arguments.exclude))
    selected, thermal_record = thermal_selection(predictions, truth)
    if len(selected):
        predictions = pd.concat([predictions, selected], ignore_index=True)
    keys = ["fold_id", "split", "horizon", "target_period_id", "node_id"]

    # Common cells: the intersection over methods, so nothing is scored on weeks
    # another method never forecast.
    cell_sets = [set(map(tuple, g[keys].drop_duplicates().to_numpy()))
                 for _, g in predictions.groupby("method")]
    common = set.intersection(*cell_sets)
    cells = pd.DataFrame(sorted(common), columns=keys)
    for column in ("fold_id", "horizon", "target_period_id", "node_id"):
        cells[column] = cells[column].astype(int)
    predictions = predictions.merge(cells, on=keys, how="inner")
    test_cells = cells["split"] == "test"
    print(f"{predictions['method'].nunique()} methods, {len(cells):,} common cells "
          f"({int(test_cells.sum()):,} test, "
          f"{int((test_cells & (cells['fold_id'] == HOLDOUT_FOLD)).sum()):,} in the 2026 hold-out)")

    naive_frame = naive_predictions(cells, tensors, folds)
    seed_ensembles = seed_mean(predictions)
    table = pd.concat([predictions, seed_ensembles, naive_frame], ignore_index=True)
    table = table.merge(truth, on=["target_period_id", "node_id"], how="left")
    threshold_rows = pd.DataFrame([
        {"fold_id": f, "node_id": n, "threshold": t}
        for f, values in thresholds.items() for n, t in enumerate(values)])
    table = table.merge(threshold_rows, on=["fold_id", "node_id"], how="left")

    mixtures = mixture_ensembles(table)
    if len(mixtures):
        mixtures = mixtures.merge(truth, on=["target_period_id", "node_id"], how="left") \
            .merge(threshold_rows, on=["fold_id", "node_id"], how="left")
        table = pd.concat([table, mixtures], ignore_index=True)

    # One prediction per method and cell for ensembles, intervals and warnings.
    point = table[(table["seed"] == -2) | ((table["seed"] == -1) & ~table["method"].isin(
        table.loc[table["seed"] == -2, "method"].unique()))].copy()

    members = [m for m in ENSEMBLE_MEMBERS if m in point["method"].unique()]
    weights = pd.DataFrame()
    if len(members) >= 2:
        ensembles, weights = build_ensembles(point, members, folds)
        ensembles = ensembles.merge(truth, on=["target_period_id", "node_id"], how="left") \
            .merge(threshold_rows, on=["fold_id", "node_id"], how="left")
        point = pd.concat([point, ensembles], ignore_index=True)
        table = pd.concat([table, ensembles], ignore_index=True)

    top_k, top_k_members = build_top_k(point)
    top_k = top_k.merge(truth, on=["target_period_id", "node_id"], how="left")
    top_k = top_k.merge(threshold_rows, on=["fold_id", "node_id"], how="left")
    point = pd.concat([point, top_k], ignore_index=True)
    table = pd.concat([table, top_k], ignore_index=True)

    metrics = per_fold_metrics(table)
    summary = skill_columns(headline(metrics), metrics)

    # The case-only baseline must not move when the climate or the method set changes.
    persistence = summary[summary["method"] == "persistence"].set_index("horizon")["mae"]
    print("persistence headline MAE, h=1-4:", persistence.reindex([1, 2, 3, 4]).round(2).tolist(),
          "(expected 16.42, 20.16, 24.58, 28.47)")

    # Skill and significance against same-horizon persistence; Holm over the
    # declared primary family only.
    tests = []
    for (method, horizon) in summary[["method", "horizon"]].itertuples(index=False):
        if method == "persistence":
            continue
        for column in ("mae", "peak_mae"):
            result = paired(metrics, method, "persistence", horizon, column)
            family = "primary" if (method in PRIMARY and column == "mae") else "exploratory"
            tests.append({"method": method, "horizon": horizon, "metric": column,
                          "family": family, **result})
    tests = pd.DataFrame(tests)
    primary = tests["family"] == "primary"
    tests["p_holm"] = np.nan
    tests["wilcoxon_p_holm"] = np.nan
    tests.loc[primary, "p_holm"] = revision.holm(tests.loc[primary, "p"].to_numpy())
    tests.loc[primary, "wilcoxon_p_holm"] = revision.holm(tests.loc[primary, "wilcoxon_p"].to_numpy())

    horizons = sorted(summary["horizon"].unique())
    ablations = []
    for arm, reference, label, family in ABLATIONS:
        for horizon in horizons:
            for column in ("mae", "peak_mae"):
                if {arm, reference} <= set(metrics["method"]):
                    ablations.append({"comparison": label, "family": family, "arm": arm,
                                      "reference": reference, "horizon": horizon, "metric": column,
                                      **paired(metrics, arm, reference, horizon, column)})
    ablations = pd.DataFrame(ablations)

    # Time-block bootstrap: primary vs persistence, and every ablation, headline folds.
    present = set(table["method"])
    pairs = [(m, "persistence", h) for m in PRIMARY if m in present for h in horizons]
    pairs += [(a, r, h) for a, r, _, _ in ABLATIONS if {a, r} <= present for h in horizons]
    bootstrap = pd.DataFrame(bootstrap_rows(table, pairs, HEADLINE_FOLDS, "headline", arguments.n_boot))
    holdout_pairs = [(m, "persistence", h) for m in PRIMARY if m in present for h in horizons]
    bootstrap_holdout = pd.DataFrame(bootstrap_rows(table, holdout_pairs, (HOLDOUT_FOLD,),
                                                    "holdout_2026", arguments.n_boot))

    # Probabilistic. Previous-year conformal for every point method; rolling and
    # adaptive conformal for a fixed set; native and recalibrated quantiles.
    conformal = conformal_quantiles(point[~point["method"].isin(["ens_stacked", "ens_top3"])], folds)
    online = []
    for method in ONLINE_CONFORMAL:
        for horizon in horizons:
            group = point[(point["method"] == method) & (point["horizon"] == horizon)
                          & (point["split"] == "test")]
            if group.empty:
                continue
            for variant, gamma in (("conformal_rolling", 0.0), ("conformal_aci", ACI_GAMMA)):
                frame = revision.online_conformal(group, horizon, gamma=gamma)
                if len(frame):
                    frame["variant"] = variant
                    online.append(frame)
    native_source = table[(table["seed"].isin([-1, -2])) & table[QCOLS[0]].notna()]
    native_source = native_source[~native_source["method"].isin(
        native_source.loc[native_source["seed"] == -2, "method"].unique())
        | (native_source["seed"] == -2)]
    native_frames = [native_quantiles(native_source, folds, False),
                     native_quantiles(native_source, folds, True)]
    probabilistic = pd.concat([conformal] + online + native_frames, ignore_index=True)
    prob_scores = revision.interval_scores(probabilistic)
    prob_summary = prob_scores[prob_scores["fold_id"].isin(HEADLINE_FOLDS)].groupby(
        ["method", "variant", "horizon"])[["wis", "peak_wis", "cov50", "cov80", "cov95",
                                          "width80", "width95"]].mean().reset_index()

    nb_diag = pd.DataFrame()
    if "nb_mu" in table.columns:
        nb_rows = table[table["nb_mu"].notna() & (table["seed"] >= 0)]
        nb_diag = revision.nb_diagnostics(nb_rows)

    ews, onsets = early_warning(point, tensors, thresholds)
    ews_summary = ews.groupby(["method", "horizon"])[
        ["auc", "sensitivity", "precision", "false_alarm_rate"]].mean().reset_index()
    onset_summary = onsets.groupby(["method", "horizon"]).agg(
        onsets=("onsets", "sum"), detected=("detected", "sum"),
        detected_at_far=("detected_at_far", "sum"), onset_auc=("onset_auc", "mean"),
        realised_far=("realised_far", "mean")).reset_index()
    onset_summary["detection_rate"] = onset_summary["detected"] / onset_summary["onsets"]
    onset_summary["detection_at_far"] = onset_summary["detected_at_far"] / onset_summary["onsets"]

    holdout = metrics[metrics["fold_id"] == HOLDOUT_FOLD]
    multi = set(holdout.loc[holdout["seed"] >= 0, "method"])
    holdout = holdout[((holdout["seed"] >= 0) & holdout["method"].isin(multi))
                      | ((holdout["seed"] == -1) & ~holdout["method"].isin(multi))]
    holdout = holdout.groupby(["method", "horizon"])[["mae", "peak_mae", "cells"]].mean().reset_index()

    compute_files = sorted(COMPUTE_DIR.glob("*.csv"))
    compute = (revision.compute_table(pd.concat([pd.read_csv(f) for f in compute_files],
                                                ignore_index=True))
               if compute_files else pd.DataFrame())

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(TABLE_DIR / "per_fold_metrics.csv", index=False)
    summary.to_csv(TABLE_DIR / "headline_summary.csv", index=False)
    tests.to_csv(TABLE_DIR / "vs_persistence_tests.csv", index=False)
    ablations.to_csv(TABLE_DIR / "ablations.csv", index=False)
    bootstrap.to_csv(TABLE_DIR / "bootstrap.csv", index=False)
    bootstrap_holdout.to_csv(TABLE_DIR / "holdout_bootstrap.csv", index=False)
    holdout.to_csv(TABLE_DIR / "holdout_2026.csv", index=False)
    prob_scores.to_csv(TABLE_DIR / "probabilistic_per_fold.csv", index=False)
    prob_summary.to_csv(TABLE_DIR / "probabilistic_summary.csv", index=False)
    nb_diag.to_csv(TABLE_DIR / "nb_diagnostics.csv", index=False)
    thermal_record.to_csv(TABLE_DIR / "thermal_selection.csv", index=False)
    ews.to_csv(TABLE_DIR / "early_warning_per_fold.csv", index=False)
    ews_summary.to_csv(TABLE_DIR / "early_warning_summary.csv", index=False)
    onset_summary.to_csv(TABLE_DIR / "onset_detection.csv", index=False)
    compute.to_csv(TABLE_DIR / "compute.csv", index=False)
    if len(weights):
        weights.to_csv(TABLE_DIR / "stacking_weights.csv", index=False)
    top_k_members.to_csv(TABLE_DIR / "top_k_members.csv", index=False)

    print_summary(summary, tests, ablations, prob_summary, ews_summary, onset_summary, weights)
    return 0


def print_summary(summary, tests, ablations, prob_summary, ews_summary, onset_summary, weights):
    horizons = sorted(summary["horizon"].unique())
    persistence = summary[summary["method"] == "persistence"].set_index("horizon")

    def skill_table(column):
        lines = [f"{'method':<22}" + "".join(f"{'h' + str(h):>8}" for h in horizons)]
        for method, group in summary.groupby("method"):
            group = group.set_index("horizon")
            values = [(1 - group.loc[h, column] / persistence.loc[h, column.replace('_ens', '')]) * 100
                      if h in group.index else np.nan for h in horizons]
            lines.append(f"{method:<22}" + "".join(f"{v:8.1f}" for v in values))
        return "\n".join(lines)

    print("\nHeadline MAE by horizon")
    print(summary.pivot(index="method", columns="horizon", values="mae").round(2).to_string())
    print("\nMAE skill vs same-horizon persistence (%), single model")
    print(skill_table("mae"))
    print("\nMAE skill vs persistence (%), seed ensemble")
    print(skill_table("mae_ens"))
    print("\nPeak-MAE skill vs persistence (%)")
    print(skill_table("peak_mae"))
    print("\nPaired tests vs persistence (MAE): delta / wins / p")
    mae_tests = tests[tests["metric"] == "mae"]
    print(mae_tests.pivot(index="method", columns="horizon",
                          values="p").round(3).to_string())
    print("\nAblations")
    print(ablations[ablations["metric"] == "mae"].pivot(
        index="comparison", columns="horizon", values="delta").round(2).to_string())
    print(ablations[ablations["metric"] == "mae"].pivot(
        index="comparison", columns="horizon", values="p").round(3).to_string())
    if len(weights):
        print("\nStacking weights (mean over folds)")
        print(weights.groupby("horizon").mean().drop(columns="fold_id").round(2).to_string())
    print("\nWIS")
    print(prob_summary.pivot_table(index=["method", "variant"], columns="horizon",
                                   values="wis").round(2).to_string())
    print("\n95% coverage")
    print(prob_summary.pivot_table(index=["method", "variant"], columns="horizon",
                                   values="cov95").round(3).to_string())
    print("\nEarly-warning AUC")
    print(ews_summary.pivot(index="method", columns="horizon", values="auc").round(3).to_string())
    print("\nOnset detection rate, rule forecast >= threshold")
    print(onset_summary.pivot(index="method", columns="horizon",
                              values="detection_rate").round(3).to_string())
    print("\nOnset AUC (onset weeks vs quiet weeks)")
    print(onset_summary.pivot(index="method", columns="horizon",
                              values="onset_auc").round(3).to_string())
    print(f"\nOnsets detected at a {TARGET_FAR:.0%} false-alarm cutoff set on the previous fold")
    print(onset_summary.pivot(index="method", columns="horizon",
                              values="detection_at_far").round(3).to_string())
    print("\nRealised false-alarm rate at that cutoff")
    print(onset_summary.pivot(index="method", columns="horizon",
                              values="realised_far").round(3).to_string())


if __name__ == "__main__":
    raise SystemExit(main())

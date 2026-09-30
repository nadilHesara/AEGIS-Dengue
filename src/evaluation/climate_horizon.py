"""
Evaluator for the climate-horizon extension (plan §6.5).

It reads benchmark predictions strictly read-only
(`results/benchmark/predictions/*.parquet` is opened, never written) and
scores reference methods and extension arms on one shared set of cells.
Outputs go only under `results/climate_horizon/`.

Before a reference method is compared directly, its provenance is checked:

    data       climate, tensors and folds files hash-identical to
               results/benchmark/data_manifest.json, and the prediction file
               written after the manifest (after the climate freeze)
    targets    every predicted (target_period, node) is a real observed cell
    dates      target periods lie within their fold's split
    split rule each row's split label equals split_by_target of its target
               (the per-cell rule), and horizons 1-4 are present

A method that fails a check is still reported, but labelled "context only".
Fold 10 (2026) is never scored here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.long_horizon import BENCHMARK_DIR, PREDICTIONS_DIR, split_by_target

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "results" / "climate_horizon"
MANIFEST_PATH = BENCHMARK_DIR / "data_manifest.json"

REFERENCE_METHODS = ("gru_v2_nb", "nb_shared_v2", "lgbm_v2")
DEVELOPMENT_FOLDS = tuple(range(1, 10))
HEADLINE_FOLDS = (1, 2, 3, 6, 7, 8, 9)
KEYS = ["fold_id", "split", "horizon", "target_period_id", "node_id"]

# The short paper's persistence, valid only under its protocol (scripts/15
# windows, lookback 12, fit_end_period = end of validation year, folds 1-9).
PERSISTENCE_REFERENCE = {1: 16.42, 2: 20.16, 3: 24.58, 4: 28.47}
MANIFEST_KEYS = ("climate_daily", "tensors_v2", "folds", "dengue_canonical", "reporting_calendar")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_provenance() -> pd.DataFrame:
    """Hash each manifest-recorded input against its recorded value."""

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    rows = []
    for key in MANIFEST_KEYS:
        entry = manifest["files"][key]
        path = PROJECT_DIR / entry["path"]
        current = sha256(path) if path.exists() else None
        rows.append({"file": key, "path": entry["path"], "matches_manifest": current == entry["sha256"]})
    return pd.DataFrame(rows)


def load_reference(method: str, horizons=(1, 2, 3, 4)) -> pd.DataFrame:
    """Test rows of one benchmark method, folds 1-9, h <= 4, per seed. Read-only."""

    path = PREDICTIONS_DIR / f"{method}.parquet"
    frame = pd.read_parquet(path, columns=["method", *KEYS, "seed", "prediction"])
    frame = frame[(frame["method"] == method) & (frame["split"] == "test")
                  & frame["fold_id"].isin(DEVELOPMENT_FOLDS) & frame["horizon"].isin(horizons)]
    return frame.reset_index(drop=True)


def check_reference(method: str, frame: pd.DataFrame, folds: dict[int, dict],
                    truth: pd.DataFrame, manifest_time: pd.Timestamp) -> dict:
    """Provenance checks for one reference method (see module docstring)."""

    path = PREDICTIONS_DIR / f"{method}.parquet"
    written = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")

    labels_ok = True
    dates_ok = True
    for fold_id, group in frame.groupby("fold_id"):
        fold = folds[int(fold_id)]
        expected = split_by_target(group["target_period_id"].to_numpy(), fold)
        labels_ok &= bool((expected == group["split"].to_numpy()).all())
        dates_ok &= bool(group["target_period_id"].between(
            fold["test_start_period"], fold["test_end_period"]).all())

    joined = frame[["target_period_id", "node_id"]].drop_duplicates().merge(
        truth, on=["target_period_id", "node_id"], how="left")
    targets_ok = bool(joined["observed"].fillna(0).eq(1).all())

    horizons_ok = set(frame["horizon"].unique()) == {1, 2, 3, 4}
    seeds = sorted(int(s) for s in frame["seed"].unique())
    result = {
        "method": method,
        "written_after_manifest": bool(written >= manifest_time),
        "split_rule_per_target": labels_ok,
        "dates_within_test_year": dates_ok,
        "targets_observed": targets_ok,
        "horizons_1_4": horizons_ok,
        "seeds": ",".join(map(str, seeds)),
        "rows": len(frame),
    }
    checks = [k for k in ("written_after_manifest", "split_rule_per_target",
                          "dates_within_test_year", "targets_observed", "horizons_1_4")]
    result["comparison"] = "direct" if all(result[k] for k in checks) else "context only"
    return result


def persistence_on_cells(cells: pd.DataFrame, tensors: dict, folds: dict[int, dict]) -> pd.DataFrame:
    """Persistence y[origin] on given cells; unobserved origin -> district mean up to fit_end."""

    y = np.where(tensors["y_mask"] == 1, tensors["y"], np.nan).astype(np.float64)
    period_id = tensors["period_id"]
    index_of = {int(p): i for i, p in enumerate(period_id)}
    out = []
    for fold_id, group in cells.groupby("fold_id"):
        fold = folds[int(fold_id)]
        fallback = np.nanmean(y[period_id <= fold["fit_end_period"]], axis=0)
        filled = np.where(np.isnan(y), fallback[None, :], y)
        target = np.array([index_of[int(p)] for p in group["target_period_id"]])
        origin = target - group["horizon"].to_numpy().astype(int)
        frame = group[KEYS].copy()
        frame["method"] = "persistence"
        frame["seed"] = -1
        frame["prediction"] = filled[origin, group["node_id"].to_numpy().astype(int)]
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def score(frame: pd.DataFrame, truth: pd.DataFrame, thresholds: dict[int, np.ndarray]) -> pd.DataFrame:
    """Per method/seed/fold/horizon MAE and peak MAE on observed cells."""

    table = frame.merge(truth, on=["target_period_id", "node_id"], how="left")
    table = table[table["observed"] == 1]
    table["abs_error"] = (table["prediction"] - table["actual"]).abs()
    table["threshold"] = [thresholds[int(f)][int(n)] for f, n in zip(table["fold_id"], table["node_id"])]
    table["peak"] = table["actual"] >= table["threshold"]
    grouped = table.groupby(["method", "seed", "fold_id", "horizon"])
    metrics = grouped["abs_error"].mean().rename("mae").to_frame()
    metrics["peak_mae"] = table[table["peak"]].groupby(["method", "seed", "fold_id", "horizon"])["abs_error"].mean()
    metrics["cells"] = grouped.size()
    return metrics.reset_index()


def headline(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean over headline folds of the mean over seeds (the benchmark's `mae` estimand)."""

    per_fold = metrics.groupby(["method", "fold_id", "horizon"])[["mae", "peak_mae"]].mean().reset_index()
    per_fold = per_fold[per_fold["fold_id"].isin(HEADLINE_FOLDS)]
    return per_fold.groupby(["method", "horizon"])[["mae", "peak_mae"]].mean().reset_index()

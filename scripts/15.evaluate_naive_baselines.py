"""
Score the naive forecasts and define the metrics every later model reuses.

Weekly dengue counts are strongly autocorrelated and strongly seasonal, so two
forecasts that contain no model at all are hard to beat:

    persistence      next week equals this week
    seasonal naive   next week equals the same week last year

A graph neural network that does not beat both is not detecting spatial
structure; it is spending several thousand parameters to approximate a lag-1
copy. These numbers exist so that claim can be checked rather than assumed, and
they are computed on exactly the folds, windows and masks the trained model
will use.

The metric functions live here and are imported by the training script, so a
model and its baseline can never be scored differently.

Three metrics are reported:

    MAE, RMSE   on the original case scale, over observed cells only

    peak MAE    on cells above a per-district peak threshold, where the
                threshold is the 90th percentile of that district's cases over
                the fold's fitting history. Fitted per fold, so the definition
                of "high for this district" never uses the test year.

Peak MAE is the one that matters for an outbreak warning system. A model can
win on mean MAE by predicting the median every week, and it will be useless in
the only weeks anyone cares about.

Outputs:
    results/models/naive_baseline_report.md
    results/models/naive_baseline_metrics.csv
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "models"

FOLDS_PATH = PROCESSED_DIR / "folds.json"

REPORT_PATH = RESULTS_DIR / "naive_baseline_report.md"
METRICS_PATH = RESULTS_DIR / "naive_baseline_metrics.csv"

BASELINE_VERSION = "naive-v1"

DEFAULT_LOOKBACK = 12
DEFAULT_HORIZON = 1

# Periods in one seasonal cycle. The calendar has 52 or 53 periods per year, so
# this is approximate by construction; it is the standard seasonal-naive lag and
# the drift is at most one period per year.
SEASONAL_LAG = 52

PEAK_PERCENTILE = 90.0


def load_module(name: str, filename: str):
    """Load one of the numbered pipeline scripts as a module."""

    spec = importlib.util.spec_from_file_location(
        name, PROJECT_DIR / "scripts" / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


tensors_module = load_module("model_tensors", "12.build_model_tensors.py")
folds_module = load_module("build_folds", "14.build_folds.py")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def masked_mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute error over observed cells only."""

    selected = mask == 1
    if not selected.any():
        return float("nan")

    return float(np.abs(prediction[selected] - target[selected]).mean())


def masked_rmse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Root mean squared error over observed cells only."""

    selected = mask == 1
    if not selected.any():
        return float("nan")

    return float(np.sqrt(((prediction[selected] - target[selected]) ** 2).mean()))


def peak_thresholds(
    y: np.ndarray,
    y_mask: np.ndarray,
    fit_mask: np.ndarray,
    percentile: float = PEAK_PERCENTILE,
) -> np.ndarray:
    """Return the per-district peak threshold from a fold's fitting history.

    Fitted on history, never on the test year: the question is whether a week is
    unusually high *for that district*, and answering it with the test
    distribution would let the threshold move with the thing being measured.
    """

    history = y[fit_mask]
    history_mask = y_mask[fit_mask]

    thresholds = np.zeros(y.shape[1], dtype=np.float64)

    for node in range(y.shape[1]):
        observed = history[:, node][history_mask[:, node] == 1]
        observed = observed[~np.isnan(observed)]
        thresholds[node] = (
            np.percentile(observed, percentile) if len(observed) else np.inf
        )

    return thresholds


def peak_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[float, int]:
    """MAE restricted to observed cells at or above the per-district threshold."""

    selected = (mask == 1) & (target >= thresholds[None, :])

    if not selected.any():
        return float("nan"), 0

    return (
        float(np.abs(prediction[selected] - target[selected]).mean()),
        int(selected.sum()),
    )


def evaluate(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, float]:
    """Return the standard metric set for one set of predictions."""

    peak, peak_cells = peak_mae(prediction, target, mask, thresholds)

    return {
        "mae": masked_mae(prediction, target, mask),
        "rmse": masked_rmse(prediction, target, mask),
        "peak_mae": peak,
        "peak_cells": peak_cells,
        "cells": int((mask == 1).sum()),
    }


def per_district_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    names: list[str],
) -> pd.DataFrame:
    """Return MAE for each district separately."""

    records = []
    for node, name in enumerate(names):
        selected = mask[:, node] == 1
        error = (
            float(np.abs(prediction[selected, node] - target[selected, node]).mean())
            if selected.any()
            else float("nan")
        )
        records.append({"node_id": node, "canonical_name": name, "mae": error})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Naive forecasts
# ---------------------------------------------------------------------------

def build_naive_predictions(
    tensors: dict[str, np.ndarray],
    windows: dict[str, np.ndarray],
    fit_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return persistence and seasonal-naive predictions for every window.

    Both read the raw case series, not the scaled features, so they are
    unaffected by any preprocessing choice.

    Where the source cell is itself unobserved the forecast falls back to the
    district's mean over the fold's fitting history. That happens once, for the
    absent Puttalam record, and a fallback is preferable to letting a NaN
    propagate silently into a metric.
    """

    y = tensors["y"]
    y_mask = tensors["y_mask"]
    period_id = tensors["period_id"]

    index_of = {int(period): index for index, period in enumerate(period_id)}

    history = y[fit_mask]
    history_mask = y_mask[fit_mask]
    fallback = np.array(
        [
            np.nanmean(history[:, node][history_mask[:, node] == 1])
            for node in range(y.shape[1])
        ]
    )

    def gather(periods: np.ndarray) -> np.ndarray:
        rows = []
        for period in periods:
            index = index_of.get(int(period))
            if index is None:
                rows.append(fallback.copy())
                continue
            row = y[index].copy()
            row[np.isnan(row)] = fallback[np.isnan(row)]
            rows.append(row)
        return np.array(rows, dtype=np.float64)

    # Persistence reads the origin period, which is the last period the model
    # is allowed to see. Seasonal naive reads one year before the target.
    persistence = gather(windows["origin_period_id"])
    seasonal = gather(windows["target_period_id"] - SEASONAL_LAG)

    return {"persistence": persistence, "seasonal_naive": seasonal}


# ---------------------------------------------------------------------------
# Fold evaluation
# ---------------------------------------------------------------------------

def evaluate_folds(
    tensors: dict[str, np.ndarray],
    folds: list[dict],
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score both naive forecasts on every fold's test year."""

    windows = tensors_module.make_windows(
        tensors, lookback=lookback, horizon=horizon, drop_incomplete=False
    )

    names = [str(name) for name in tensors["canonical_name"]]

    rows = []
    district_rows = []

    for fold in folds:
        fit_mask = tensors["period_id"] <= fold["fit_end_period"]
        thresholds = peak_thresholds(tensors["y"], tensors["y_mask"], fit_mask)

        predictions = build_naive_predictions(tensors, windows, fit_mask)
        split = folds_module.assign_windows(windows["target_period_id"], fold)

        target = windows["y"][split["test"]]
        mask = windows["y_mask"][split["test"]]

        for model, prediction in predictions.items():
            scores = evaluate(prediction[split["test"]], target, mask, thresholds)

            rows.append(
                {
                    "model": model,
                    "fold_id": fold["fold_id"],
                    "test_year": fold["test_year"],
                    "covers_covid": fold["covers_covid"],
                    "headline": fold["headline"],
                    **scores,
                }
            )

            district = per_district_mae(
                prediction[split["test"]], target, mask, names
            )
            district["model"] = model
            district["fold_id"] = fold["fold_id"]
            district_rows.append(district)

    return pd.DataFrame(rows), pd.concat(district_rows, ignore_index=True)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average each model over the headline folds and over the COVID folds."""

    records = []

    for model, group in metrics.groupby("model"):
        headline = group[group["headline"]]
        covid = group[group["covers_covid"]]

        records.append(
            {
                "model": model,
                "headline_mae": headline["mae"].mean(),
                "headline_rmse": headline["rmse"].mean(),
                "headline_peak_mae": headline["peak_mae"].mean(),
                "covid_mae": covid["mae"].mean(),
                "covid_peak_mae": covid["peak_mae"].mean(),
                "epidemic_2017_mae": group.loc[
                    group["test_year"] == 2017, "mae"
                ].mean(),
                "epidemic_2017_peak_mae": group.loc[
                    group["test_year"] == 2017, "peak_mae"
                ].mean(),
            }
        )

    return pd.DataFrame(records).sort_values("headline_mae").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(metrics: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write the naive baseline report."""

    lines = [
        "# Naive baselines",
        "",
        f"Version: `{BASELINE_VERSION}`",
        "",
        f"Lookback {DEFAULT_LOOKBACK}, horizon {DEFAULT_HORIZON}. Metrics are on the",
        "original case scale over observed cells only. Peak MAE covers cells at or",
        f"above each district's {PEAK_PERCENTILE:.0f}th percentile, fitted on the",
        "fold's history.",
        "",
        "## Summary",
        "",
        "| Model | Headline MAE | Headline RMSE | Headline peak MAE | 2017 MAE | 2017 peak MAE | COVID MAE |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in summary.itertuples():
        lines.append(
            f"| {row.model} | {row.headline_mae:.2f} | {row.headline_rmse:.2f} | "
            f"{row.headline_peak_mae:.2f} | {row.epidemic_2017_mae:.2f} | "
            f"{row.epidemic_2017_peak_mae:.2f} | {row.covid_mae:.2f} |"
        )

    lines += [
        "",
        "Headline is the mean over folds 1, 2, 3, 6, 7, 8 and 9. The COVID folds",
        "are averaged separately and never folded into the headline.",
        "",
        "## Per fold",
        "",
        "| Model | Fold | Test year | MAE | RMSE | Peak MAE | Peak cells | Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in metrics.sort_values(["model", "fold_id"]).itertuples():
        note = "COVID" if row.covers_covid else ("epidemic" if row.test_year == 2017 else "")
        lines.append(
            f"| {row.model} | {row.fold_id} | {row.test_year} | {row.mae:.2f} | "
            f"{row.rmse:.2f} | {row.peak_mae:.2f} | {row.peak_cells} | {note} |"
        )

    lines += [
        "",
        "## What these numbers are for",
        "",
        "A trained model must beat both, on headline MAE **and** on peak MAE. Beating",
        "mean MAE alone is achievable by predicting close to the recent level every",
        "week, which is what persistence already does for free.",
        "",
        "## Output files",
        "",
        f"- `{METRICS_PATH.relative_to(PROJECT_DIR).as_posix()}`",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Score the naive baselines on every fold."""

    if not FOLDS_PATH.exists():
        print(f"Missing {FOLDS_PATH}. Run scripts/14.build_folds.py first.")
        return 1

    try:
        tensors = folds_module.load_tensors("v0")
    except FileNotFoundError as error:
        print(error)
        return 1

    nodes = pd.read_csv(PROCESSED_DIR / "nodes.csv").sort_values("node_id")
    tensors["canonical_name"] = nodes["canonical_name"].to_numpy(dtype=object)

    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))["folds"]

    metrics, districts = evaluate_folds(tensors, folds)
    summary = summarise(metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    districts.to_csv(
        RESULTS_DIR / "naive_baseline_district_metrics.csv", index=False
    )

    write_report(metrics, summary)

    print(f"Folds scored: {len(folds)}\n")
    print(f"{'model':<16} {'MAE':>8} {'RMSE':>8} {'peakMAE':>9}   (headline)")
    for row in summary.itertuples():
        print(
            f"{row.model:<16} {row.headline_mae:>8.2f} {row.headline_rmse:>8.2f} "
            f"{row.headline_peak_mae:>9.2f}"
        )

    print("\n2017 epidemic fold")
    for row in summary.itertuples():
        print(
            f"{row.model:<16} MAE {row.epidemic_2017_mae:>7.2f}   "
            f"peak MAE {row.epidemic_2017_peak_mae:>7.2f}"
        )

    print(f"\nWrote {REPORT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {METRICS_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

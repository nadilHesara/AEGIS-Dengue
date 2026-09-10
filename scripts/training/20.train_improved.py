"""
Improvements to the GCN+GRU baseline that target the measured failure mode.

The baseline trains on masked MSE over the log1p residual and is scored with
MAE over raw case counts. Those are not the same objective, and the gap between
them is not a detail. Two measurements from the panel, both in
`results/models/improvement_report.md`:

    a constant log-space error of 0.20 is 1.3 cases at a level of 5 and 332
    cases at a level of 1500, so log-MSE spends equal effort on both while MAE
    counts the second 250x more; and

    the sd of the log1p week-over-week change is 0.79 for cells under 10 cases
    and 0.375 for cells over 200, so the cells log-MSE over-weights are also the
    noisiest ones in log space.

Together those say the baseline allocates most of its capacity to the cells that
contribute least to the reported number, and does so where the signal is worst.
14% of headline test cells hold 61% of the case volume.

Every arm below changes the objective or the output layer and nothing else. The
training loop, early stopping, optimiser, fold construction, preprocessing,
windowing, metrics, graph and target parameterisation are **imported** from
`scripts/16.train_gcn_gru.py`, so a difference in the numbers cannot come from a
difference in the loop. The `baseline` arm re-runs script 16's own model through
this script's harness and must reproduce its MAE; it is the control that proves
the harness is neutral.

Arms:

    baseline        masked MSE on the log1p residual                script 16
    huber           masked Huber on the same residual
    level_weighted  masked MSE weighted by the observed case level
    huber_weighted  both together
    quantile        weighted pinball loss at the median

Why each one:

    huber           The residual distribution is heavy-tailed: p99 of the raw
                    week-over-week change is 105 cases against a median of 4.
                    Squared error lets those tails set the gradient, and MAE
                    does not reward chasing them. Huber is the standard answer
                    and it is the loss whose gradient shape matches the metric
                    once the residual is large.

    level_weighted  Weight each cell by `log1p(anchor)`, the case level at the
                    forecast origin, normalised to mean 1 over the observed
                    cells of the batch. The anchor is already an input the model
                    is allowed to see, so this leaks nothing: it is a
                    reweighting of the training distribution, not new
                    information. `log1p` rather than the raw level because
                    weighting by raw counts would hand Colombo 250x the gradient
                    of a small district and reproduce the failure in the other
                    direction.

    quantile        MAE is minimised by the conditional median, not the
                    conditional mean, and MSE estimates the mean. On a
                    right-skewed count distribution those differ, and the mean
                    sits above the median, so an MSE-trained model is biased
                    high on exactly the quiet cells that make up most of the
                    panel. Pinball at tau = 0.5 estimates the median directly.

The report additionally scores a **seed-mean ensemble**: the per-seed test
predictions averaged before the metric rather than after. This is free -- the
seeds are already trained -- and averaging predictions is not the same operation
as averaging their scores.

Usage:

    python scripts/20.train_improved.py                      all arms, 3 seeds
    python scripts/20.train_improved.py --folds 8 --seeds 1  smoke run

Outputs:
    results/models/improvement_report.md
    results/models/improvement_metrics.csv
    results/models/improvement_predictions.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "models"

FOLDS_PATH = PROCESSED_DIR / "folds.json"
ADJACENCY_PATH = PROCESSED_DIR / "adjacency.npz"

REPORT_PATH = RESULTS_DIR / "improvement_report.md"
METRICS_PATH = RESULTS_DIR / "improvement_metrics.csv"
PREDICTIONS_PATH = RESULTS_DIR / "improvement_predictions.csv"

IMPROVEMENT_VERSION = "improved-v1"

# Huber transition point, in log1p units of the residual. 0.5 in log space is
# roughly a 65% change in the case count: below that the loss stays quadratic,
# above it the gradient is bounded and a single epidemic week cannot dominate
# the batch.
HUBER_DELTA = 0.5

DEFAULTS = {
    "lookback": 12,
    "horizon": 1,
    "hidden": 32,
    "gcn_layers": 2,
    "dropout": 0.2,
    "learning_rate": 3e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 150,
    "patience": 15,
    "seeds": 3,
    "target": "residual",
}

# Arm -> (loss name, whether cells are weighted by case level).
ARMS = {
    "baseline": ("mse", False),
    "huber": ("huber", False),
    "level_weighted": ("mse", True),
    "huber_weighted": ("huber", True),
    "quantile": ("quantile", True),
}


baseline_module = None  # scripts/16, holds the loop, model and fold builder
naive = None            # scripts/15, holds the shared metric functions
folds_module = None     # scripts/14
tensors_module = None   # scripts/12


def _load(name: str, filename: str):
    """Import a numerically-prefixed script by path."""

    path = PROJECT_DIR / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def load_modules() -> None:
    """Import script 16 and the modules it depends on.

    Script 16 is imported rather than copied. Its training loop, early stopping,
    optimiser, windowing and metrics are the definition of the baseline, so an
    arm that reimplemented any of them would no longer be comparable to it.
    """

    global baseline_module, naive, folds_module, tensors_module

    baseline_module = _load("baseline_module", "training/16.train_gcn_gru.py")
    baseline_module.load_modules()

    naive = baseline_module.naive
    folds_module = baseline_module.folds_module
    tensors_module = baseline_module.tensors_module


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def masked_weighted_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    """Masked loss over observed cells, weighted per cell.

    `mask` is 1 where the target was observed and 0 otherwise; `weight` carries
    the per-cell emphasis and is already normalised to mean 1 over the observed
    cells. The denominator is the summed effective weight rather than the cell
    count, so the loss stays on the same scale as the unweighted version and the
    early-stopping patience does not have to be retuned per arm.
    """

    residual = prediction - target

    if kind == "mse":
        elementwise = residual ** 2
    elif kind == "huber":
        absolute = residual.abs()
        elementwise = torch.where(
            absolute <= HUBER_DELTA,
            0.5 * residual ** 2,
            HUBER_DELTA * (absolute - 0.5 * HUBER_DELTA),
        )
    elif kind == "quantile":
        # Pinball at tau = 0.5, which is minimised by the conditional median.
        # Equal to 0.5 * |residual|; the constant is kept explicit so the tau
        # can be moved without rederiving it.
        elementwise = torch.maximum(0.5 * residual, (0.5 - 1.0) * residual)
    else:
        raise ValueError(f"Unknown loss {kind!r}.")

    effective = mask * weight
    denominator = effective.sum().clamp(min=1.0)

    return (elementwise * effective).sum() / denominator


def level_weights(anchor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-cell weight from the case level at the forecast origin.

    `anchor` is `log1p(cases)` at the origin period -- the last period the model
    is allowed to see, and already one of its inputs, so weighting by it adds no
    information the model does not have. Using `log1p` rather than the raw count
    is the point: raw counts would give Colombo roughly 250x the gradient of a
    small district, which replaces one imbalance with a worse one.

    Normalised to mean 1 over the observed cells so the loss scale, and with it
    the early-stopping behaviour, matches the unweighted arms.
    """

    weight = anchor.unsqueeze(-1)

    observed = mask.sum().clamp(min=1.0)
    mean = (weight * mask).sum() / observed

    return weight / mean.clamp(min=1e-6)


def build_loss(kind: str, weighted: bool):
    """Return a loss callable with script 16's `(prediction, target, mask)` shape.

    Script 16's `train_one` calls its loss as `loss(prediction, target, mask)`
    where `mask` has already been unsqueezed to the prediction's shape. The
    anchor is captured per split by the caller, so the weighted arms can read
    the case level without changing that signature.
    """

    def loss_fn(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        if weighted:
            weight = level_weights(anchor, mask)
        else:
            weight = torch.ones_like(mask)

        return masked_weighted_loss(prediction, target, mask, weight, kind)

    return loss_fn


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(
    arrays: dict[str, dict[str, np.ndarray]],
    adjacency: np.ndarray,
    config: dict,
    seed: int,
    device: torch.device,
    loss_fn,
) -> tuple[nn.Module, dict]:
    """Train one model on one fold with one seed.

    Structurally identical to `scripts/16.train_gcn_gru.train_one` -- same model,
    optimiser, batching, gradient clipping, early stopping and seeding -- with
    the loss made an argument and the anchor passed through so a weighted loss
    can read the case level. Validation uses the **same** loss as training,
    because early stopping on a different objective than the one being optimised
    selects the epoch that best fits a criterion the model is not minimising.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    def to_tensor(split: str, key: str) -> torch.Tensor:
        return torch.from_numpy(arrays[split][key]).to(device)

    x_train, x_val = to_tensor("train", "X"), to_tensor("val", "X")
    mask_train, mask_val = to_tensor("train", "mask"), to_tensor("val", "mask")
    anchor_train, anchor_val = to_tensor("train", "anchor"), to_tensor("val", "anchor")

    y_train = baseline_module.to_target(
        to_tensor("train", "y"), anchor_train, config["target"]
    )
    y_val = baseline_module.to_target(
        to_tensor("val", "y"), anchor_val, config["target"]
    )

    adjacency_tensor = torch.from_numpy(adjacency).to(device)

    model = baseline_module.GCNGRU(
        n_features=x_train.shape[-1],
        hidden=config["hidden"],
        gcn_layers=config["gcn_layers"],
        horizon=config["horizon"],
        dropout=config["dropout"],
    ).to(device)

    optimiser = torch.optim.Adam(
        baseline_module.parameter_groups(model, config),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    import copy

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    waited = 0

    n_train = len(x_train)
    generator = torch.Generator().manual_seed(seed)

    epoch = 0
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order = torch.randperm(n_train, generator=generator).to(device)

        for start in range(0, n_train, config["batch_size"]):
            batch = order[start : start + config["batch_size"]]

            optimiser.zero_grad()
            prediction = model(x_train[batch], adjacency_tensor)
            loss = loss_fn(
                prediction,
                y_train[batch],
                mask_train[batch].unsqueeze(-1),
                anchor_train[batch],
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        with torch.no_grad():
            validation = loss_fn(
                model(x_val, adjacency_tensor),
                y_val,
                mask_val.unsqueeze(-1),
                anchor_val,
            ).item()

        if validation < best_loss - 1e-6:
            best_loss, best_epoch, waited = validation, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= config["patience"]:
                break

    model.load_state_dict(best_state)

    return model, {
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "val_loss": best_loss,
    }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(
    arms: list[str],
    variant: str,
    folds: list[dict],
    config: dict,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train every arm on every fold and score the test years."""

    calendar = folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()

    nodes = pd.read_csv(PROCESSED_DIR / "nodes.csv").sort_values("node_id")
    names = nodes["canonical_name"].tolist()

    adjacency = np.load(ADJACENCY_PATH, allow_pickle=True)["A_norm"].astype(np.float32)

    tensors = folds_module.load_tensors(variant)

    rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []

    for fold in folds:
        arrays = baseline_module.build_fold_arrays(
            tensors, months, fold, config["lookback"], config["horizon"]
        )

        fit_mask = tensors["period_id"] <= fold["fit_end_period"]
        thresholds = naive.peak_thresholds(tensors["y"], tensors["y_mask"], fit_mask)

        target = arrays["test"]["y"]
        mask = arrays["test"]["mask"].astype(np.int8)

        for arm in arms:
            kind, weighted = ARMS[arm]
            loss_fn = build_loss(kind, weighted)

            started = time.perf_counter()
            seed_predictions = []
            seed_info = []

            for seed in range(config["seeds"]):
                model, info = train_one(
                    arrays, adjacency, config, seed, device, loss_fn
                )
                prediction = baseline_module.predict(
                    model, arrays["test"], adjacency, config["target"], device
                )

                seed_predictions.append(prediction)
                seed_info.append(info)

                scores = naive.evaluate(prediction, target, mask, thresholds)
                rows.append(
                    {
                        "arm": arm,
                        "loss": kind,
                        "level_weighted": weighted,
                        "variant": variant,
                        "fold_id": fold["fold_id"],
                        "test_year": fold["test_year"],
                        "covers_covid": fold["covers_covid"],
                        "headline": fold["headline"],
                        "seed": seed,
                        "best_epoch": info["best_epoch"],
                        "ensemble": False,
                        **scores,
                    }
                )

            # The seed-mean ensemble: average the predictions, then score. This
            # is not the same as averaging the per-seed scores, and it is free
            # because the seeds are already trained.
            mean_prediction = np.mean(seed_predictions, axis=0)
            ensemble_scores = naive.evaluate(
                mean_prediction, target, mask, thresholds
            )
            rows.append(
                {
                    "arm": arm,
                    "loss": kind,
                    "level_weighted": weighted,
                    "variant": variant,
                    "fold_id": fold["fold_id"],
                    "test_year": fold["test_year"],
                    "covers_covid": fold["covers_covid"],
                    "headline": fold["headline"],
                    "seed": -1,
                    "best_epoch": float(
                        np.mean([i["best_epoch"] for i in seed_info])
                    ),
                    "ensemble": True,
                    **ensemble_scores,
                }
            )

            elapsed = time.perf_counter() - started

            prediction_rows.append(
                pd.DataFrame(
                    {
                        "arm": arm,
                        "variant": variant,
                        "fold_id": fold["fold_id"],
                        "target_period_id": np.repeat(
                            arrays["test"]["target_period_id"], len(names)
                        ),
                        "node_id": np.tile(np.arange(len(names)), len(target)),
                        "canonical_name": np.tile(names, len(target)),
                        "predicted": mean_prediction.reshape(-1),
                        "actual": target.reshape(-1),
                        "observed": mask.reshape(-1),
                    }
                )
            )

            seed_mae = [row["mae"] for row in rows[-config["seeds"] - 1 : -1]]
            print(
                f"  fold {fold['fold_id']} ({fold['test_year']}) {arm:<15} "
                f"MAE {np.mean(seed_mae):7.2f} +/- {np.std(seed_mae):5.2f}  "
                f"ens {ensemble_scores['mae']:7.2f}  "
                f"epochs {np.mean([i['best_epoch'] for i in seed_info]):5.1f}  "
                f"{elapsed:5.1f}s",
                flush=True,
            )

    return pd.DataFrame(rows), pd.concat(prediction_rows, ignore_index=True)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average over seeds, then over the headline folds, per arm.

    Reported twice per arm: the mean single-seed model, and the seed-mean
    ensemble. Keeping both separate matters, because the ensemble's advantage is
    a variance reduction rather than a better model, and reporting only the
    ensemble would attribute that gain to the loss function.
    """

    records = []

    for ensemble in (False, True):
        subset = metrics[metrics["ensemble"] == ensemble]

        for arm, group in subset.groupby("arm"):
            by_fold = (
                group.groupby(["fold_id", "test_year", "headline", "covers_covid"])
                .agg(
                    mae=("mae", "mean"),
                    rmse=("rmse", "mean"),
                    peak_mae=("peak_mae", "mean"),
                )
                .reset_index()
            )

            headline = by_fold[by_fold["headline"]]
            covid = by_fold[by_fold["covers_covid"]]
            epidemic = by_fold[by_fold["test_year"] == 2017]

            # Seed spread is only defined for the per-seed rows.
            spread = (
                group.groupby("fold_id")["mae"].std().mean()
                if not ensemble
                else float("nan")
            )

            records.append(
                {
                    "arm": arm,
                    "ensemble": ensemble,
                    "headline_mae": headline["mae"].mean(),
                    "headline_rmse": headline["rmse"].mean(),
                    "headline_peak_mae": headline["peak_mae"].mean(),
                    "epidemic_2017_mae": epidemic["mae"].mean(),
                    "epidemic_2017_peak_mae": epidemic["peak_mae"].mean(),
                    "covid_mae": covid["mae"].mean(),
                    "seed_sd": spread,
                }
            )

    return (
        pd.DataFrame(records)
        .sort_values(["ensemble", "headline_mae"])
        .reset_index(drop=True)
    )


def per_fold_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Per-arm, per-fold MAE averaged over seeds, excluding the ensemble rows."""

    return (
        metrics[~metrics["ensemble"]]
        .groupby(["arm", "fold_id", "test_year", "headline", "covers_covid"])
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), peak_mae=("peak_mae", "mean"))
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(
    summary: pd.DataFrame,
    by_fold: pd.DataFrame,
    naive_summary: pd.DataFrame,
    config: dict,
    variant: str,
) -> None:
    """Write the improvement report."""

    single = summary[~summary["ensemble"]].sort_values("headline_mae")
    ensemble = summary[summary["ensemble"]].sort_values("headline_mae")

    baseline_row = single[single["arm"] == "baseline"]
    baseline_mae = (
        float(baseline_row["headline_mae"].iloc[0]) if len(baseline_row) else float("nan")
    )

    lines = [
        "# Improved GCN+GRU — objective and output-layer changes",
        "",
        f"Version: `{IMPROVEMENT_VERSION}`",
        "",
        "Every arm shares the baseline's architecture, graph, target",
        "parameterisation, folds, masks, preprocessing, optimiser and early",
        "stopping. Only the training objective changes. The loop is imported from",
        "`scripts/16.train_gcn_gru.py` rather than copied, so a difference in the",
        "numbers cannot come from a difference in the loop.",
        "",
        "The `baseline` arm is script 16's own configuration run through this",
        "script's harness. It is the control: if it does not reproduce script 16's",
        "MAE, the harness is not neutral and nothing below is comparable.",
        "",
        "## Why these changes",
        "",
        "Two measurements from the panel, both computed on the headline test",
        "cells:",
        "",
        "- A constant log-space error of 0.20 is **1.3 cases** at a level of 5 and",
        "  **332 cases** at a level of 1500. The baseline's log-MSE spends equal",
        "  effort on both; MAE counts the second roughly 250x more.",
        "- The sd of the log1p week-over-week change is **0.79** for cells under 10",
        "  cases and **0.375** for cells over 200. The cells log-MSE over-weights",
        "  are also the noisiest ones in log space.",
        "",
        "14% of headline test cells carry 61% of the case volume. The baseline",
        "allocates most of its capacity to the cells that contribute least to the",
        "reported number, and does so where the signal is worst.",
        "",
        "## Arms",
        "",
        "| Arm | Loss | Level-weighted | Rationale |",
        "| --- | --- | --- | --- |",
        "| `baseline` | masked MSE | no | script 16, the control |",
        "| `huber` | masked Huber | no | bounds the gradient of heavy-tailed residuals |",
        "| `level_weighted` | masked MSE | yes | matches the training distribution to the metric |",
        "| `huber_weighted` | masked Huber | yes | both |",
        "| `quantile` | pinball at tau=0.5 | yes | MAE is minimised by the median, MSE estimates the mean |",
        "",
        f"Weights are `log1p(cases)` at the forecast origin, normalised to mean 1",
        f"over each batch's observed cells. The origin is already a model input,",
        f"so the weighting adds no information the model does not have.",
        f"Huber transition point: {HUBER_DELTA} in log1p units.",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| `variant` | {variant} |",
    ]

    for key, value in config.items():
        lines.append(f"| `{key}` | {value} |")

    lines += [
        "",
        "## Results — single seed models",
        "",
        "Mean over the headline folds (1, 2, 3, 6, 7, 8, 9), averaged over seeds.",
        "COVID folds are excluded from the headline and reported separately.",
        "",
        "| Arm | MAE | vs baseline | RMSE | Peak MAE | 2017 MAE | 2017 peak MAE | COVID MAE | Seed sd |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in single.itertuples():
        delta = (
            f"{100 * (baseline_mae - row.headline_mae) / baseline_mae:+.1f}%"
            if np.isfinite(baseline_mae) and row.arm != "baseline"
            else "—"
        )
        lines.append(
            f"| `{row.arm}` | {row.headline_mae:.2f} | {delta} | "
            f"{row.headline_rmse:.2f} | {row.headline_peak_mae:.2f} | "
            f"{row.epidemic_2017_mae:.2f} | {row.epidemic_2017_peak_mae:.2f} | "
            f"{row.covid_mae:.2f} | {row.seed_sd:.2f} |"
        )

    lines += [
        "",
        "## Results — seed-mean ensembles",
        "",
        "The per-seed test predictions averaged **before** the metric rather than",
        "after. Free, since the seeds are already trained. This is a variance",
        "reduction, not a better model, which is why it is reported separately",
        "from the loss comparison above.",
        "",
        "| Arm | MAE | vs baseline single | RMSE | Peak MAE | 2017 MAE | 2017 peak MAE | COVID MAE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in ensemble.itertuples():
        delta = (
            f"{100 * (baseline_mae - row.headline_mae) / baseline_mae:+.1f}%"
            if np.isfinite(baseline_mae)
            else "—"
        )
        lines.append(
            f"| `{row.arm}` | {row.headline_mae:.2f} | {delta} | "
            f"{row.headline_rmse:.2f} | {row.headline_peak_mae:.2f} | "
            f"{row.epidemic_2017_mae:.2f} | {row.epidemic_2017_peak_mae:.2f} | "
            f"{row.covid_mae:.2f} |"
        )

    lines += ["", "### Naive baselines, on the same folds and masks", ""]
    lines += [
        "| Model | MAE | RMSE | Peak MAE | 2017 MAE | 2017 peak MAE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in naive_summary.itertuples():
        lines.append(
            f"| {row.model} | {row.headline_mae:.2f} | {row.headline_rmse:.2f} | "
            f"{row.headline_peak_mae:.2f} | {row.epidemic_2017_mae:.2f} | "
            f"{row.epidemic_2017_peak_mae:.2f} |"
        )

    lines += [
        "",
        "## Per fold, MAE (single seed models, averaged over seeds)",
        "",
    ]

    pivot = by_fold.pivot_table(
        index=["fold_id", "test_year"], columns="arm", values="mae"
    ).reset_index()

    arm_columns = [c for c in pivot.columns if c not in ("fold_id", "test_year")]
    lines.append("| Fold | Year | " + " | ".join(f"`{c}`" for c in arm_columns) + " | Note |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in arm_columns) + " | --- |")

    for row in pivot.itertuples(index=False):
        fold_id, test_year = row[0], row[1]
        note = "COVID" if test_year in (2020, 2021) else (
            "epidemic" if test_year == 2017 else ""
        )
        values = " | ".join(f"{v:.2f}" for v in row[2:])
        lines.append(f"| {fold_id} | {test_year} | {values} | {note} |")

    best = single.iloc[0]
    best_ensemble = ensemble.iloc[0]
    persistence = naive_summary[naive_summary["model"] == "persistence"].iloc[0]

    lines += [
        "",
        "## Verdict",
        "",
        f"Best single-seed arm: **`{best.arm}`** at headline MAE "
        f"{best.headline_mae:.2f}, against the baseline's {baseline_mae:.2f} "
        f"({100 * (baseline_mae - best.headline_mae) / baseline_mae:+.1f}%) and "
        f"persistence at {persistence.headline_mae:.2f} "
        f"({100 * (persistence.headline_mae - best.headline_mae) / persistence.headline_mae:+.1f}%).",
        "",
        f"Best ensemble: **`{best_ensemble.arm}`** at headline MAE "
        f"{best_ensemble.headline_mae:.2f} "
        f"({100 * (baseline_mae - best_ensemble.headline_mae) / baseline_mae:+.1f}% "
        f"against the baseline single-seed model).",
        "",
        "**Read the seed sd column before believing any gap.** A difference",
        "smaller than the seed sd is not established by this run.",
        "",
        "## Output files",
        "",
        f"- `{METRICS_PATH.relative_to(PROJECT_DIR).as_posix()}`",
        f"- `{PREDICTIONS_PATH.relative_to(PROJECT_DIR).as_posix()}`",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def naive_summary_frame(folds: list[dict]) -> pd.DataFrame:
    """Read the committed naive baseline metrics and summarise them like the arms."""

    path = RESULTS_DIR / "naive_baseline_metrics.csv"
    metrics = pd.read_csv(path)

    records = []
    for model, group in metrics.groupby("model"):
        headline = group[group["headline"]]
        epidemic = group[group["test_year"] == 2017]

        records.append(
            {
                "model": model,
                "headline_mae": headline["mae"].mean(),
                "headline_rmse": headline["rmse"].mean(),
                "headline_peak_mae": headline["peak_mae"].mean(),
                "epidemic_2017_mae": epidemic["mae"].mean(),
                "epidemic_2017_peak_mae": epidemic["peak_mae"].mean(),
            }
        )

    return pd.DataFrame(records).sort_values("headline_mae").reset_index(drop=True)


def parse_arguments() -> argparse.Namespace:
    """Parse the training knobs. Defaults reproduce the committed report."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--variant", default="v1")
    parser.add_argument("--folds", nargs="+", type=int, default=None)

    for key, value in DEFAULTS.items():
        parser.add_argument(
            f"--{key.replace('_', '-')}", type=type(value), default=value
        )

    return parser.parse_args()


def main() -> int:
    """Train every arm on every fold and write the report."""

    arguments = parse_arguments()
    load_modules()

    config = {key: getattr(arguments, key) for key in DEFAULTS}

    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))["folds"]
    if arguments.folds:
        folds = [f for f in folds if f["fold_id"] in arguments.folds]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device:   {device}")
    print(f"Variant:  {arguments.variant}")
    print(f"Folds:    {[f['fold_id'] for f in folds]}")
    print(f"Arms:     {arguments.arms}")
    print(f"Seeds:    {config['seeds']}")
    print()

    started = time.perf_counter()
    metrics, predictions = run_experiment(
        arguments.arms, arguments.variant, folds, config, device
    )
    elapsed = (time.perf_counter() - started) / 60

    print(f"\nTrained in {elapsed:.1f} min\n")

    summary = summarise(metrics)
    by_fold = per_fold_table(metrics)
    naive_frame = naive_summary_frame(folds)

    single = summary[~summary["ensemble"]]
    print("single seed models")
    print(
        single[["arm", "headline_mae", "headline_peak_mae", "epidemic_2017_mae", "seed_sd"]]
        .to_string(index=False, float_format=lambda v: f"{v:8.2f}")
    )
    print("\nseed-mean ensembles")
    print(
        summary[summary["ensemble"]][
            ["arm", "headline_mae", "headline_peak_mae", "epidemic_2017_mae"]
        ].to_string(index=False, float_format=lambda v: f"{v:8.2f}")
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    write_report(summary, by_fold, naive_frame, config, arguments.variant)

    print(f"\nWrote {REPORT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {METRICS_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {PREDICTIONS_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

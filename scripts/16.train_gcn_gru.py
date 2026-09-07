"""
Train and evaluate the GCN+GRU baseline across the walk-forward folds.

The architecture is deliberately the plain one, arranged spatial-then-temporal:

    for each step in the input window
        S = relu(A_norm @ X @ W1)          graph convolution over 25 districts
        S = relu(A_norm @ S  @ W2)
    H = GRU(S_1 .. S_L)                    shared weights, one sequence per node
    y = Linear(H_L)                        cases for the next period

Spatial first, then temporal, because that is where a spatial-versus-temporal
fusion gate goes later. Swapping the order or fusing the two into a single
recurrent graph cell would make this baseline structurally unlike the model it
is supposed to be the baseline for.

A GRU-only control is trained alongside it, identical in every respect except
that the adjacency is replaced by the identity. It answers the question the
graph is there to answer: does spatial structure help, or is a per-district
recurrent model already enough. Without the control, a GCN+GRU that beats
persistence has shown nothing about the graph.

Training details that matter:

    target      the log1p change from the last observed period, not the level:

                    predict   d = log1p(y[t+h]) - log1p(y[t])
                    report    y_hat = expm1(log1p(y[t]) + d_hat)

                Counts run 0 to 2631 with a median of 10, so the raw scale puts
                the whole loss on Colombo, and log1p alone still asks the model
                to reconstruct a level it already has in its input. Anchoring
                makes an output of zero mean "same as last week", so the network
                only has to learn the correction to persistence.

                This is a target parameterisation, not an architecture change:
                the model is still exactly a graph convolution followed by a
                GRU. It was chosen by measurement, not by preference. On folds 1
                and 8, against the direct log1p target:

                    fold 1 (2017)   MAE 87.45 direct -> 52.91 anchored
                    fold 8 (2024)   MAE 17.58 direct -> 10.11 anchored

                and it converges in roughly a third of the epochs. Pass
                `--target direct` to train the unanchored version.

                The anchor is log1p of the case count at the forecast origin,
                which is the last period the model is allowed to see, so it
                carries no information the input window does not already have.

    loss        masked MSE. Unobserved district-periods are excluded, never
                imputed: an imputed target is a number the model is rewarded
                for reproducing that no one observed.

    stopping    on validation loss, where validation is the calendar year
                before the test year. Never on the test fold.

    seeds       every configuration is trained several times and reported as a
                mean over seeds. A single-seed neural network result on 500
                training windows is noise.

Preprocessing is refitted per fold by scripts/14, so nothing fitted on a test
year ever reaches the model.

Outputs:
    results/models/baseline_report.md
    results/models/baseline_metrics.csv
    results/models/baseline_predictions.csv
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "models"

FOLDS_PATH = PROCESSED_DIR / "folds.json"
ADJACENCY_PATH = PROCESSED_DIR / "adjacency.npz"

REPORT_PATH = RESULTS_DIR / "baseline_report.md"
METRICS_PATH = RESULTS_DIR / "baseline_metrics.csv"
PREDICTIONS_PATH = RESULTS_DIR / "baseline_predictions.csv"

BASELINE_VERSION = "gcn-gru-v1"

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
}


naive = None  # loaded in main, holds the shared metric functions
tensors_module = None
folds_module = None


def load_modules() -> None:
    """Load the pipeline scripts this one builds on."""

    global naive, tensors_module, folds_module

    naive = _load("naive_baselines", "15.evaluate_naive_baselines.py")
    tensors_module = naive.tensors_module
    folds_module = naive.folds_module


def _load(name: str, filename: str):
    """Load one numbered pipeline script as a module."""

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        name, PROJECT_DIR / "scripts" / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GraphConv(nn.Module):
    """One dense graph convolution over a fixed 25 x 25 adjacency.

    Dense, not sparse, and no PyTorch Geometric. At 25 nodes a dense matmul is
    faster than any sparse gather, and it keeps the dependency list to torch.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x is [batch, steps, nodes, features]
        projected = self.linear(x)

        return torch.einsum("ij,bljf->blif", adjacency, projected)


class GCNGRU(nn.Module):
    """Graph convolution over districts, then a GRU over reporting periods."""

    def __init__(
        self,
        n_features: int,
        hidden: int = DEFAULTS["hidden"],
        gcn_layers: int = DEFAULTS["gcn_layers"],
        horizon: int = DEFAULTS["horizon"],
        dropout: float = DEFAULTS["dropout"],
    ):
        super().__init__()

        sizes = [n_features] + [hidden] * gcn_layers
        self.graph_layers = nn.ModuleList(
            GraphConv(sizes[index], sizes[index + 1]) for index in range(gcn_layers)
        )

        self.dropout = nn.Dropout(dropout)
        # One shared GRU applied to every district's sequence. Node identity
        # reaches it through the graph and through the centroid features, not
        # through separate weights: 25 per-district GRUs on ~500 training
        # windows would memorise the training years outright.
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, _ = x.shape

        spatial = x
        for layer in self.graph_layers:
            spatial = torch.relu(layer(spatial, adjacency))
            spatial = self.dropout(spatial)

        # [batch, steps, nodes, hidden] -> one sequence per (batch, node)
        sequences = spatial.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)

        output, _ = self.gru(sequences)
        last = self.dropout(output[:, -1])

        return self.head(last).view(batch, nodes, -1)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean squared error over observed cells only."""

    error = (prediction - target) * mask
    denominator = mask.sum().clamp(min=1.0)

    return (error ** 2).sum() / denominator


# ---------------------------------------------------------------------------
# Fold data
# ---------------------------------------------------------------------------

def build_fold_arrays(
    tensors: dict[str, np.ndarray],
    months: np.ndarray,
    fold: dict,
    lookback: int,
    horizon: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Impute, scale and window one fold, returning arrays per split."""

    statistics = folds_module.fit_fold_statistics(
        tensors, months, tensors["period_id"] <= fold["fit_end_period"]
    )

    scaled = dict(tensors)
    scaled["X"] = folds_module.transform(tensors, months, statistics)

    windows = tensors_module.make_windows(
        scaled, lookback=lookback, horizon=horizon, drop_incomplete=False
    )
    split = folds_module.assign_windows(windows["target_period_id"], fold)

    anchor = build_anchor(tensors, windows, fold)

    arrays = {}
    for name, selector in split.items():
        target = windows["y"][selector]
        mask = (windows["y_mask"][selector] == 1).astype(np.float32)

        # NaN targets are masked out, but they must still be finite or the
        # multiplication by zero would produce nan rather than zero.
        target = np.nan_to_num(target, nan=0.0)

        arrays[name] = {
            "X": windows["X"][selector],
            "y": target,
            "mask": mask,
            "anchor": anchor[selector],
            "target_period_id": windows["target_period_id"][selector],
        }

    return arrays


def build_anchor(
    tensors: dict[str, np.ndarray],
    windows: dict[str, np.ndarray],
    fold: dict,
) -> np.ndarray:
    """Return log1p of the case count at each window's forecast origin.

    Taken from the raw case series rather than recovered from the scaled
    features, so the anchor does not depend on the preprocessing statistics.

    Where the origin cell is unobserved the district's mean over the fold's
    history stands in, which is what the persistence baseline does with the same
    gap. It happens once, for the absent Puttalam record.
    """

    y = tensors["y"]
    y_mask = tensors["y_mask"]

    fit_mask = tensors["period_id"] <= fold["fit_end_period"]
    history, history_mask = y[fit_mask], y_mask[fit_mask]

    fallback = np.array(
        [
            np.nanmean(history[:, node][history_mask[:, node] == 1])
            for node in range(y.shape[1])
        ]
    )

    index_of = {int(period): index for index, period in enumerate(tensors["period_id"])}
    origins = np.array(
        [index_of[int(period)] for period in windows["origin_period_id"]]
    )

    at_origin = y[origins]
    at_origin = np.where(np.isnan(at_origin), fallback[None, :], at_origin)

    return np.log1p(at_origin).astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def parameter_groups(model: nn.Module, config: dict) -> list[dict]:
    """Split off any lag-encoder parameters so they can take their own rate.

    The delay kernels sit behind a softmax and a Gaussian shape, so the gradient
    reaching them is far smaller than the gradient reaching the GRU. At a shared
    learning rate they barely move from their initialisation and the module
    looks like it does nothing. Models without an encoder are unaffected.
    """

    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return list(model.parameters())

    encoder_ids = {id(p) for p in encoder.parameters()}
    rest = [p for p in model.parameters() if id(p) not in encoder_ids]

    return [
        {"params": rest},
        {
            "params": list(encoder.parameters()),
            "lr": config.get("kernel_learning_rate", config["learning_rate"]),
            "weight_decay": 0.0,
        },
    ]


def train_one(
    arrays: dict[str, dict[str, np.ndarray]],
    adjacency: np.ndarray,
    config: dict,
    seed: int,
    device: torch.device,
    build_model=None,
) -> tuple[nn.Module, dict]:
    """Train one model on one fold with one seed, early stopping on validation.

    `build_model` takes the input feature count and returns the module to train.
    It defaults to the plain GCN+GRU. The hook exists so a variant can reuse this
    loop verbatim rather than copying it: an experiment that reimplements early
    stopping or the optimiser is no longer comparable to this baseline.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    def to_tensor(split: str, key: str) -> torch.Tensor:
        return torch.from_numpy(arrays[split][key]).to(device)

    x_train, x_val = to_tensor("train", "X"), to_tensor("val", "X")
    mask_train, mask_val = to_tensor("train", "mask"), to_tensor("val", "mask")

    # The model predicts the log1p change from the forecast origin, so an output
    # of zero reproduces persistence exactly. Metrics are inverted afterwards.
    y_train = to_target(
        to_tensor("train", "y"), to_tensor("train", "anchor"), config["target"]
    )
    y_val = to_target(
        to_tensor("val", "y"), to_tensor("val", "anchor"), config["target"]
    )

    adjacency_tensor = torch.from_numpy(adjacency).to(device)

    if build_model is None:
        model = GCNGRU(
            n_features=x_train.shape[-1],
            hidden=config["hidden"],
            gcn_layers=config["gcn_layers"],
            horizon=config["horizon"],
            dropout=config["dropout"],
        )
    else:
        model = build_model(x_train.shape[-1])

    model = model.to(device)

    optimiser = torch.optim.Adam(
        parameter_groups(model, config),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    waited = 0

    n_train = len(x_train)
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order = torch.randperm(n_train, generator=generator).to(device)

        for start in range(0, n_train, config["batch_size"]):
            batch = order[start : start + config["batch_size"]]

            optimiser.zero_grad()
            prediction = model(x_train[batch], adjacency_tensor)
            loss = masked_mse(
                prediction, y_train[batch], mask_train[batch].unsqueeze(-1)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        with torch.no_grad():
            validation = masked_mse(
                model(x_val, adjacency_tensor), y_val, mask_val.unsqueeze(-1)
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


def to_target(
    y: torch.Tensor, anchor: torch.Tensor, mode: str
) -> torch.Tensor:
    """Map raw counts to the quantity the model is trained to predict."""

    log_cases = torch.log1p(y)

    if mode == "residual":
        return (log_cases - anchor).unsqueeze(-1)

    if mode == "direct":
        return log_cases.unsqueeze(-1)

    raise ValueError(f"Unknown target mode {mode!r}.")


def predict(
    model: nn.Module,
    split: dict[str, np.ndarray],
    adjacency: np.ndarray,
    mode: str,
    device: torch.device,
) -> np.ndarray:
    """Return case-scale predictions for one split."""

    model.eval()
    with torch.no_grad():
        output = model(
            torch.from_numpy(split["X"]).to(device),
            torch.from_numpy(adjacency).to(device),
        )

    log_prediction = output.squeeze(-1).cpu().numpy()

    if mode == "residual":
        log_prediction = log_prediction + split["anchor"]

    # Invert log1p and clip: a negative case count is not a forecast.
    return np.clip(np.expm1(log_prediction), 0.0, None)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(
    variants: list[str],
    models: dict[str, np.ndarray],
    folds: list[dict],
    config: dict,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train every model on every variant and fold, and score the test years."""

    calendar = folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()

    nodes = pd.read_csv(PROCESSED_DIR / "nodes.csv").sort_values("node_id")
    names = nodes["canonical_name"].tolist()

    rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []

    for variant in variants:
        tensors = folds_module.load_tensors(variant)

        for fold in folds:
            arrays = build_fold_arrays(
                tensors, months, fold, config["lookback"], config["horizon"]
            )

            fit_mask = tensors["period_id"] <= fold["fit_end_period"]
            thresholds = naive.peak_thresholds(
                tensors["y"], tensors["y_mask"], fit_mask
            )

            target = arrays["test"]["y"]
            mask = arrays["test"]["mask"].astype(np.int8)

            for model_name, adjacency in models.items():
                started = time.perf_counter()
                seed_predictions = []
                seed_info = []

                for seed in range(config["seeds"]):
                    model, info = train_one(
                        arrays, adjacency, config, seed, device
                    )
                    seed_predictions.append(
                        predict(
                            model, arrays["test"], adjacency, config["target"], device
                        )
                    )
                    seed_info.append(info)

                    scores = naive.evaluate(
                        seed_predictions[-1], target, mask, thresholds
                    )
                    rows.append(
                        {
                            "model": model_name,
                            "variant": variant,
                            "fold_id": fold["fold_id"],
                            "test_year": fold["test_year"],
                            "covers_covid": fold["covers_covid"],
                            "headline": fold["headline"],
                            "seed": seed,
                            "best_epoch": info["best_epoch"],
                            **scores,
                        }
                    )

                mean_prediction = np.mean(seed_predictions, axis=0)
                elapsed = time.perf_counter() - started

                frame = pd.DataFrame(
                    {
                        "model": model_name,
                        "variant": variant,
                        "fold_id": fold["fold_id"],
                        "target_period_id": np.repeat(
                            arrays["test"]["target_period_id"], len(names)
                        ),
                        "node_id": np.tile(
                            np.arange(len(names)), len(target)
                        ),
                        "canonical_name": np.tile(names, len(target)),
                        "predicted": mean_prediction.reshape(-1),
                        "actual": target.reshape(-1),
                        "observed": mask.reshape(-1),
                    }
                )
                prediction_rows.append(frame)

                seed_mae = [
                    row["mae"]
                    for row in rows[-config["seeds"] :]
                ]
                print(
                    f"  {variant} fold {fold['fold_id']} ({fold['test_year']}) "
                    f"{model_name:<9} MAE {np.mean(seed_mae):7.2f} "
                    f"+/- {np.std(seed_mae):5.2f}  "
                    f"epochs {np.mean([i['best_epoch'] for i in seed_info]):5.1f}  "
                    f"{elapsed:5.1f}s"
                )

    return pd.DataFrame(rows), pd.concat(prediction_rows, ignore_index=True)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average over seeds, then over the headline folds."""

    by_fold = (
        metrics.groupby(["model", "variant", "fold_id", "test_year", "headline", "covers_covid"])
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), peak_mae=("peak_mae", "mean"))
        .reset_index()
    )

    records = []
    for (model, variant), group in by_fold.groupby(["model", "variant"]):
        headline = group[group["headline"]]
        covid = group[group["covers_covid"]]
        epidemic = group[group["test_year"] == 2017]

        seed_spread = metrics[
            (metrics["model"] == model) & (metrics["variant"] == variant)
        ]
        spread = seed_spread.groupby("fold_id")["mae"].std().mean()

        records.append(
            {
                "model": model,
                "variant": variant,
                "headline_mae": headline["mae"].mean(),
                "headline_rmse": headline["rmse"].mean(),
                "headline_peak_mae": headline["peak_mae"].mean(),
                "epidemic_2017_mae": epidemic["mae"].mean(),
                "epidemic_2017_peak_mae": epidemic["peak_mae"].mean(),
                "covid_mae": covid["mae"].mean(),
                "seed_sd": spread,
            }
        )

    return pd.DataFrame(records).sort_values("headline_mae").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(
    summary: pd.DataFrame,
    by_fold: pd.DataFrame,
    naive_summary: pd.DataFrame,
    config: dict,
    n_parameters: int,
) -> None:
    """Write the baseline report."""

    lines = [
        "# GCN+GRU baseline",
        "",
        f"Version: `{BASELINE_VERSION}`",
        "",
        "Graph convolution over the 25 districts at each step, then a shared GRU",
        "over the input window, then a linear head. Trained per walk-forward fold",
        "with preprocessing refitted on that fold's history alone.",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
    ]

    for key, value in config.items():
        lines.append(f"| `{key}` | {value} |")

    lines += [
        f"| trainable parameters | {n_parameters:,} |",
        "",
        "## Results",
        "",
        "Mean over the headline folds (1, 2, 3, 6, 7, 8, 9), averaged over seeds.",
        "COVID folds are excluded from the headline and reported separately.",
        "",
        "| Model | Features | MAE | RMSE | Peak MAE | 2017 MAE | 2017 peak MAE | COVID MAE | Seed sd |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in summary.itertuples():
        lines.append(
            f"| {row.model} | `{row.variant}` | {row.headline_mae:.2f} | "
            f"{row.headline_rmse:.2f} | {row.headline_peak_mae:.2f} | "
            f"{row.epidemic_2017_mae:.2f} | {row.epidemic_2017_peak_mae:.2f} | "
            f"{row.covid_mae:.2f} | {row.seed_sd:.2f} |"
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

    best = summary.iloc[0]
    persistence = naive_summary[naive_summary["model"] == "persistence"].iloc[0]

    lines += [
        "",
        "## Verdict",
        "",
        f"Best configuration: **{best.model}** on `{best.variant}`, "
        f"headline MAE {best.headline_mae:.2f} against persistence at "
        f"{persistence.headline_mae:.2f} "
        f"({100 * (persistence.headline_mae - best.headline_mae) / persistence.headline_mae:+.1f}%).",
        "",
        f"Peak MAE {best.headline_peak_mae:.2f} against persistence at "
        f"{persistence.headline_peak_mae:.2f} "
        f"({100 * (persistence.headline_peak_mae - best.headline_peak_mae) / persistence.headline_peak_mae:+.1f}%).",
        "",
        "## Per fold",
        "",
        "| Model | Features | Fold | Test year | MAE | RMSE | Peak MAE | Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in by_fold.sort_values(
        ["model", "variant", "fold_id"]
    ).itertuples():
        note = "COVID" if row.covers_covid else ("epidemic" if row.test_year == 2017 else "")
        lines.append(
            f"| {row.model} | `{row.variant}` | {row.fold_id} | {row.test_year} | "
            f"{row.mae:.2f} | {row.rmse:.2f} | {row.peak_mae:.2f} | {note} |"
        )

    lines += [
        "",
        "## Reading this",
        "",
        "`gcn_gru` uses the contiguity adjacency; `gru_only` is the identical model",
        "with the adjacency replaced by the identity. The difference between them is",
        "what the graph contributes. The difference between `v0` and `v1` is what",
        "hand-specified 4, 8 and 12 period climate lags contribute, and it is the",
        "number a learnable lag module has to beat.",
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

def parse_arguments() -> argparse.Namespace:
    """Parse the training knobs. Defaults reproduce the committed report."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--variants", nargs="+", default=["v0", "v1"])
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--no-control", action="store_true")
    parser.add_argument(
        "--target",
        choices=["residual", "direct"],
        default="residual",
        help="residual predicts the change from the forecast origin",
    )

    for key, value in DEFAULTS.items():
        parser.add_argument(
            f"--{key.replace('_', '-')}", type=type(value), default=value
        )

    return parser.parse_args()


def main() -> int:
    """Train and evaluate the baseline."""

    arguments = parse_arguments()
    load_modules()

    if not FOLDS_PATH.exists() or not ADJACENCY_PATH.exists():
        print("Missing folds.json or adjacency.npz. Run scripts 13 and 14 first.")
        return 1

    config = {key: getattr(arguments, key) for key in DEFAULTS}
    config["target"] = arguments.target

    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))["folds"]
    if arguments.folds:
        folds = [fold for fold in folds if fold["fold_id"] in arguments.folds]

    with np.load(ADJACENCY_PATH, allow_pickle=True) as data:
        adjacency = data["A_norm"].astype(np.float32)

    models = {"gcn_gru": adjacency}
    if not arguments.no_control:
        # The control keeps every other component identical, so the comparison
        # isolates the adjacency rather than the architecture.
        models["gru_only"] = np.eye(len(adjacency), dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device:   {device}")
    print(f"Folds:    {[fold['fold_id'] for fold in folds]}")
    print(f"Variants: {arguments.variants}")
    print(f"Models:   {list(models)}")
    print(f"Seeds:    {config['seeds']}\n")

    started = time.perf_counter()
    metrics, predictions = run_experiment(
        arguments.variants, models, folds, config, device
    )
    elapsed = time.perf_counter() - started

    summary = summarise(metrics)

    by_fold = (
        metrics.groupby(
            ["model", "variant", "fold_id", "test_year", "headline", "covers_covid"]
        )
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), peak_mae=("peak_mae", "mean"))
        .reset_index()
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    naive_metrics = pd.read_csv(RESULTS_DIR / "naive_baseline_metrics.csv")
    naive_summary = naive.summarise(naive_metrics)

    probe = GCNGRU(
        n_features=folds_module.load_tensors(arguments.variants[-1])["X"].shape[2],
        hidden=config["hidden"],
        gcn_layers=config["gcn_layers"],
        horizon=config["horizon"],
        dropout=config["dropout"],
    )

    write_report(
        summary, by_fold, naive_summary, config, count_parameters(probe)
    )

    print(f"\nTrained in {elapsed / 60:.1f} min\n")
    print(
        f"{'model':<10} {'feat':<5} {'MAE':>8} {'RMSE':>8} {'peakMAE':>9} "
        f"{'2017':>8} {'seed sd':>8}"
    )
    for row in summary.itertuples():
        print(
            f"{row.model:<10} {row.variant:<5} {row.headline_mae:>8.2f} "
            f"{row.headline_rmse:>8.2f} {row.headline_peak_mae:>9.2f} "
            f"{row.epidemic_2017_mae:>8.2f} {row.seed_sd:>8.2f}"
        )

    for row in naive_summary.itertuples():
        print(
            f"{row.model:<10} {'-':<5} {row.headline_mae:>8.2f} "
            f"{row.headline_rmse:>8.2f} {row.headline_peak_mae:>9.2f} "
            f"{row.epidemic_2017_mae:>8.2f} {'-':>8}"
        )

    print(f"\nWrote {REPORT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {METRICS_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {PREDICTIONS_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

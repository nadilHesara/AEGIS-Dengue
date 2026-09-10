"""
Forecast at h = 1, 2, 3, 4: separate models per horizon, and one shared model.

Every result in this repository is at horizon 1, and that is the standing
explanation for most of the negative ones. `docs/learnable_lags_results.md`
measured why: at one week ahead the previous period's case count carries nearly
all the signal, and dropping every climate channel costs +0.01 MAE. The learnable
lags found no gradient (§7), the objective fix only bit in epidemic conditions
(§8), the activation sweep was flat (§8b), the hyperparameter surface was flat
(§8c), and no graph beat no graph (§8e) -- all consistent with one cause.

This tests the explanation. The measured rainfall-to-dengue delay is 5-10 weeks
(`scripts/17`), so at h=4 climate should be closer to load-bearing than at h=1.
Two arms:

    separate    one model per horizon, each trained only on its own target.
                Four independent trunks, four times the parameters and the
                training cost. This is the baseline architecture run four times
                and is what "extend the model to h=1..4" means in the plainest
                reading.

    shared      one trunk, four linear heads over the same GRU state, trained on
                the summed masked loss across all four horizons
                (`src/models/multi_horizon.py`). One quarter the trunk
                parameters of `separate`, and the near horizons act as auxiliary
                supervision for the far ones.

**The reference is a per-horizon persistence, rescored here.** `scripts/15`
scores persistence at h=1 only (16.42 headline). Persistence degrades sharply as
the horizon grows -- the copied value is up to four weeks stale -- so comparing
an h=4 model against the h=1 persistence would flatter the model against a
baseline measured on a much easier task. Rescored on the same folds and masks:

    h=1  16.42    h=2  20.16    h=3  24.58    h=4  28.47

Each horizon's model is judged against its own row.

**The target parameterisation is unchanged.** The model predicts
`log1p(y[t+h]) - log1p(y[t])`, so an output of zero reproduces persistence *at
that horizon*. The anchor being four weeks stale at h=4 is exactly what makes the
task harder; it is the effect being measured, not a confound, and keeping the
parameterisation identical is what makes these numbers comparable to §6-§8e.

One alignment decision worth stating. Both arms use windows cut at the **longest**
horizon, so every horizon is scored on identical forecast origins. Trimming per
horizon instead would leave the h=1 column measured on 1000 windows and the h=4
column on 997 different ones, and the columns would not be comparable to each
other. It costs the h=1 column three windows against §6's numbers, which is why
`separate` at h=1 is close to but not bit-identical with the committed 18.98.

Everything else -- architecture, graph, folds, preprocessing, early stopping,
optimiser, gradient clipping, the masked-MSE objective and the metrics -- comes
from `scripts/16.train_gcn_gru.py`.

Usage:

    python scripts/27.train_multi_horizon.py                      both arms, 9 folds, 3 seeds
    python scripts/27.train_multi_horizon.py --folds 8 --seeds 1  smoke run
    python scripts/27.train_multi_horizon.py --arms shared --horizons 1 2 3 4 6 8

Outputs:
    results/models/multi_horizon_report.md
    results/models/multi_horizon_metrics.csv
    results/models/multi_horizon_naive.csv
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.models.multi_horizon import (  # noqa: E402
    MultiHorizonGCNGRU,
    make_multi_horizon_windows,
    masked_multi_horizon_mse,
)


PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "models"

FOLDS_PATH = PROCESSED_DIR / "folds.json"
ADJACENCY_PATH = PROCESSED_DIR / "adjacency.npz"

REPORT_PATH = RESULTS_DIR / "multi_horizon_report.md"
METRICS_PATH = RESULTS_DIR / "multi_horizon_metrics.csv"
NAIVE_PATH = RESULTS_DIR / "multi_horizon_naive.csv"

MULTI_HORIZON_VERSION = "multi-horizon-v1"

DEFAULTS = {
    "lookback": 12,
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

ARMS = ("separate", "shared")


baseline_module = None  # scripts/16
naive = None            # scripts/15
folds_module = None     # scripts/14


def _load(name: str, filename: str):
    """Import a numerically-prefixed pipeline script by path."""

    path = PROJECT_DIR / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def load_modules() -> None:
    """Import script 16 and the modules it stands on."""

    global baseline_module, naive, folds_module

    baseline_module = _load("baseline_module", "16.train_gcn_gru.py")
    baseline_module.load_modules()

    naive = baseline_module.naive
    folds_module = baseline_module.folds_module


# ---------------------------------------------------------------------------
# Fold data
# ---------------------------------------------------------------------------

def build_multi_horizon_arrays(
    tensors: dict[str, np.ndarray],
    months: np.ndarray,
    fold: dict,
    lookback: int,
    horizons: tuple[int, ...],
) -> dict[str, dict[str, np.ndarray]]:
    """Impute, scale and window one fold, carrying every horizon's target.

    Mirrors `scripts/16.build_fold_arrays` -- same per-fold statistics, same
    transform, same split assignment -- but uses the multi-horizon windower and
    therefore returns `y`, `mask` and `target_period_id` with a trailing horizon
    axis.

    Windows are assigned to a split by their **longest**-horizon target period.
    That is the conservative choice: a window whose h=1 target falls in the test
    year but whose h=4 target falls beyond it would otherwise straddle the
    boundary. Using the furthest target means no window is assigned to a split it
    does not entirely belong to.
    """

    statistics = folds_module.fit_fold_statistics(
        tensors, months, tensors["period_id"] <= fold["fit_end_period"]
    )

    scaled = dict(tensors)
    scaled["X"] = folds_module.transform(tensors, months, statistics)

    windows = make_multi_horizon_windows(
        scaled, lookback=lookback, horizons=horizons, drop_incomplete=False
    )

    longest_index = int(np.argmax(horizons))
    split = folds_module.assign_windows(
        windows["target_period_id"][:, longest_index], fold
    )

    anchor = build_anchor(tensors, windows, fold)

    arrays = {}
    for name, selector in split.items():
        target = windows["y"][selector]
        mask = (windows["y_mask"][selector] == 1).astype(np.float32)

        # NaN targets are masked out, but must still be finite or multiplying by
        # zero would give nan rather than zero.
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

    Identical in construction to `scripts/16.build_anchor` -- the anchor is a
    property of the origin, not of the horizon, so all horizons of a window share
    one anchor. Shape is [windows, nodes]; the training code broadcasts it across
    the horizon axis.
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

def to_multi_target(
    y: torch.Tensor, anchor: torch.Tensor, mode: str
) -> torch.Tensor:
    """Map raw counts to the trained quantity, per horizon.

    `y` is [batch, nodes, horizons] and `anchor` is [batch, nodes]; the anchor
    broadcasts across horizons because it is a property of the forecast origin,
    which every horizon of a window shares.
    """

    log_cases = torch.log1p(y)

    if mode == "residual":
        return log_cases - anchor.unsqueeze(-1)

    if mode == "direct":
        return log_cases

    raise ValueError(f"Unknown target mode {mode!r}.")


def train_shared(
    arrays: dict[str, dict[str, np.ndarray]],
    adjacency: np.ndarray,
    config: dict,
    horizons: tuple[int, ...],
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Train one shared multi-horizon model on one fold with one seed.

    Structurally identical to `scripts/16.train_one` -- same optimiser, batching,
    gradient clipping, early stopping and seeding -- with the loss summed over
    horizons and the model carrying one head per horizon. Early stopping watches
    the combined validation loss, because that is the objective being minimised;
    stopping on one horizon's loss would select the epoch that best fits a
    criterion the model is not optimising.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    def to_tensor(split: str, key: str) -> torch.Tensor:
        return torch.from_numpy(arrays[split][key]).to(device)

    x_train, x_val = to_tensor("train", "X"), to_tensor("val", "X")
    mask_train, mask_val = to_tensor("train", "mask"), to_tensor("val", "mask")
    anchor_train, anchor_val = to_tensor("train", "anchor"), to_tensor("val", "anchor")

    y_train = to_multi_target(to_tensor("train", "y"), anchor_train, config["target"])
    y_val = to_multi_target(to_tensor("val", "y"), anchor_val, config["target"])

    adjacency_tensor = torch.from_numpy(adjacency).to(device)

    backbone = baseline_module.GCNGRU(
        n_features=x_train.shape[-1],
        hidden=config["hidden"],
        gcn_layers=config["gcn_layers"],
        horizon=1,
        dropout=config["dropout"],
    )
    model = MultiHorizonGCNGRU(backbone, n_horizons=len(horizons)).to(device)

    optimiser = torch.optim.Adam(
        baseline_module.parameter_groups(model, config),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

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
            loss = masked_multi_horizon_mse(
                prediction, y_train[batch], mask_train[batch]
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        with torch.no_grad():
            validation = masked_multi_horizon_mse(
                model(x_val, adjacency_tensor), y_val, mask_val
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


def predict_multi(
    model: nn.Module,
    split: dict[str, np.ndarray],
    adjacency: np.ndarray,
    mode: str,
    device: torch.device,
) -> np.ndarray:
    """Return case-scale predictions [windows, nodes, horizons] for one split."""

    model.eval()
    with torch.no_grad():
        output = model(
            torch.from_numpy(split["X"]).to(device),
            torch.from_numpy(adjacency).to(device),
        )

    log_prediction = output.cpu().numpy()

    if mode == "residual":
        log_prediction = log_prediction + split["anchor"][..., None]

    return np.clip(np.expm1(log_prediction), 0.0, None)


def train_separate_one_horizon(
    arrays: dict[str, dict[str, np.ndarray]],
    adjacency: np.ndarray,
    config: dict,
    horizon_index: int,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Train one single-horizon model on the multi-horizon arrays.

    Takes the same windows as the shared arm and trains on one horizon's column
    only, so `separate` and `shared` see identical inputs and identical forecast
    origins. Reusing `scripts/16.train_one` directly is not possible here because
    its arrays carry a single target; slicing the horizon column and delegating
    would require rebuilding the array dict per horizon, which is what this does
    inline.
    """

    single = {}
    for name, split in arrays.items():
        single[name] = dict(split)
        single[name]["y"] = split["y"][..., horizon_index]
        single[name]["mask"] = split["mask"][..., horizon_index]

    return baseline_module.train_one(single, adjacency, config, seed, device)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(
    arms: tuple[str, ...],
    variant: str,
    folds: list[dict],
    config: dict,
    horizons: tuple[int, ...],
    device: torch.device,
) -> pd.DataFrame:
    """Train every arm on every fold and score each horizon separately."""

    calendar = folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()

    adjacency = np.load(ADJACENCY_PATH, allow_pickle=True)["A_norm"].astype(np.float32)
    identity = np.eye(len(adjacency), dtype=np.float32)

    tensors = folds_module.load_tensors(variant)

    rows: list[dict] = []

    for fold in folds:
        arrays = build_multi_horizon_arrays(
            tensors, months, fold, config["lookback"], horizons
        )

        fit_mask = tensors["period_id"] <= fold["fit_end_period"]
        thresholds = naive.peak_thresholds(tensors["y"], tensors["y_mask"], fit_mask)

        target = arrays["test"]["y"]
        mask = arrays["test"]["mask"].astype(np.int8)

        # §8e established the graph hurts, so the identity backbone is the one
        # worth extending. Both are run: the question here is the horizon, and a
        # horizon effect that only appeared on one backbone would be worth
        # knowing about.
        for backbone_name, backbone_adjacency in (
            ("identity", identity),
            ("contiguity", adjacency),
        ):
            for arm in arms:
                started = time.perf_counter()

                # [seeds, windows, nodes, horizons]
                seed_predictions = []
                seed_info = []

                for seed in range(config["seeds"]):
                    if arm == "shared":
                        model, info = train_shared(
                            arrays, backbone_adjacency, config, horizons, seed, device
                        )
                        prediction = predict_multi(
                            model,
                            arrays["test"],
                            backbone_adjacency,
                            config["target"],
                            device,
                        )
                        seed_info.append(info)
                    else:
                        columns = []
                        epochs = []
                        for index, _ in enumerate(horizons):
                            single_config = dict(config)
                            single_config["horizon"] = 1
                            model, info = train_separate_one_horizon(
                                arrays,
                                backbone_adjacency,
                                single_config,
                                index,
                                seed,
                                device,
                            )
                            single_split = dict(arrays["test"])
                            single_split["y"] = arrays["test"]["y"][..., index]
                            columns.append(
                                baseline_module.predict(
                                    model,
                                    single_split,
                                    backbone_adjacency,
                                    config["target"],
                                    device,
                                )
                            )
                            epochs.append(info["best_epoch"])

                        prediction = np.stack(columns, axis=-1)
                        seed_info.append({"best_epoch": float(np.mean(epochs))})

                    seed_predictions.append(prediction)

                    for index, horizon in enumerate(horizons):
                        scores = naive.evaluate(
                            prediction[..., index],
                            target[..., index],
                            mask[..., index],
                            thresholds,
                        )
                        rows.append(
                            {
                                "arm": arm,
                                "backbone": backbone_name,
                                "variant": variant,
                                "horizon": horizon,
                                "fold_id": fold["fold_id"],
                                "test_year": fold["test_year"],
                                "covers_covid": fold["covers_covid"],
                                "headline": fold["headline"],
                                "seed": seed,
                                "best_epoch": seed_info[-1]["best_epoch"],
                                "ensemble": False,
                                **scores,
                            }
                        )

                mean_prediction = np.mean(seed_predictions, axis=0)
                for index, horizon in enumerate(horizons):
                    scores = naive.evaluate(
                        mean_prediction[..., index],
                        target[..., index],
                        mask[..., index],
                        thresholds,
                    )
                    rows.append(
                        {
                            "arm": arm,
                            "backbone": backbone_name,
                            "variant": variant,
                            "horizon": horizon,
                            "fold_id": fold["fold_id"],
                            "test_year": fold["test_year"],
                            "covers_covid": fold["covers_covid"],
                            "headline": fold["headline"],
                            "seed": -1,
                            "best_epoch": float(
                                np.mean([i["best_epoch"] for i in seed_info])
                            ),
                            "ensemble": True,
                            **scores,
                        }
                    )

                elapsed = time.perf_counter() - started
                by_horizon = " ".join(
                    f"h{h}:{np.mean([r['mae'] for r in rows if r['horizon'] == h and not r['ensemble'] and r['fold_id'] == fold['fold_id'] and r['arm'] == arm and r['backbone'] == backbone_name]):6.2f}"
                    for h in horizons
                )
                print(
                    f"  fold {fold['fold_id']} ({fold['test_year']}) "
                    f"{backbone_name:<10} {arm:<8}  {by_horizon}  {elapsed:5.1f}s",
                    flush=True,
                )

    return pd.DataFrame(rows)


def naive_by_horizon(
    folds: list[dict], lookback: int, horizons: tuple[int, ...]
) -> pd.DataFrame:
    """Rescore the naive baselines at every horizon, on the same folds and masks.

    `scripts/15` writes h=1 only. Persistence degrades sharply as the horizon
    grows -- its copied value goes stale -- so every horizon needs its own row or
    the model's degradation is measured against a baseline that is not
    degrading.
    """

    tensors = folds_module.load_tensors("v0")
    nodes = pd.read_csv(PROCESSED_DIR / "nodes.csv").sort_values("node_id")
    tensors["canonical_name"] = nodes["canonical_name"].to_numpy(dtype=object)

    frames = []
    for horizon in horizons:
        metrics, _ = naive.evaluate_folds(
            tensors, folds, lookback=lookback, horizon=horizon
        )
        metrics["horizon"] = horizon
        frames.append(metrics)

    return pd.concat(frames, ignore_index=True)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Headline means per arm, backbone and horizon."""

    records = []

    for ensemble in (False, True):
        subset = metrics[metrics["ensemble"] == ensemble]

        for (arm, backbone, horizon), group in subset.groupby(
            ["arm", "backbone", "horizon"]
        ):
            by_fold = (
                group.groupby(["fold_id", "test_year", "headline"])
                .agg(
                    mae=("mae", "mean"),
                    rmse=("rmse", "mean"),
                    peak_mae=("peak_mae", "mean"),
                )
                .reset_index()
            )
            headline = by_fold[by_fold["headline"]]
            epidemic = by_fold[by_fold["test_year"] == 2017]

            spread = (
                group.groupby("fold_id")["mae"].std().mean()
                if not ensemble
                else float("nan")
            )

            records.append(
                {
                    "arm": arm,
                    "backbone": backbone,
                    "horizon": horizon,
                    "ensemble": ensemble,
                    "headline_mae": headline["mae"].mean(),
                    "headline_rmse": headline["rmse"].mean(),
                    "headline_peak_mae": headline["peak_mae"].mean(),
                    "epidemic_2017_mae": epidemic["mae"].mean(),
                    "seed_sd": spread,
                }
            )

    return pd.DataFrame(records).sort_values(
        ["ensemble", "backbone", "arm", "horizon"]
    ).reset_index(drop=True)


def summarise_naive(frame: pd.DataFrame) -> pd.DataFrame:
    """Headline means per naive model and horizon."""

    records = []
    for (model, horizon), group in frame.groupby(["model", "horizon"]):
        headline = group[group["headline"]]
        records.append(
            {
                "model": model,
                "horizon": horizon,
                "headline_mae": headline["mae"].mean(),
                "headline_rmse": headline["rmse"].mean(),
                "headline_peak_mae": headline["peak_mae"].mean(),
                "epidemic_2017_mae": group.loc[
                    group["test_year"] == 2017, "mae"
                ].mean(),
            }
        )

    return pd.DataFrame(records).sort_values(["model", "horizon"]).reset_index(drop=True)


def skill_table(summary: pd.DataFrame, naive_summary: pd.DataFrame) -> pd.DataFrame:
    """Each configuration's margin over same-horizon persistence.

    The question the brief asks -- does climate become more useful at longer
    horizons -- shows up here as a margin that grows with the horizon. A model
    whose raw MAE rises but whose margin over persistence also rises is getting
    *relatively* better, which is the effect worth reporting.
    """

    persistence = naive_summary[naive_summary["model"] == "persistence"].set_index(
        "horizon"
    )

    records = []
    for row in summary.itertuples():
        reference = persistence.loc[row.horizon]
        records.append(
            {
                "arm": row.arm,
                "backbone": row.backbone,
                "horizon": row.horizon,
                "ensemble": row.ensemble,
                "mae": row.headline_mae,
                "persistence_mae": reference["headline_mae"],
                "delta": row.headline_mae - reference["headline_mae"],
                "skill": 100
                * (reference["headline_mae"] - row.headline_mae)
                / reference["headline_mae"],
                "peak_mae": row.headline_peak_mae,
                "persistence_peak_mae": reference["headline_peak_mae"],
                "peak_skill": 100
                * (reference["headline_peak_mae"] - row.headline_peak_mae)
                / reference["headline_peak_mae"],
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(
    summary: pd.DataFrame,
    skill: pd.DataFrame,
    naive_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    config: dict,
    horizons: tuple[int, ...],
    variant: str,
) -> None:
    """Write the multi-horizon report."""

    lines = [
        "# Multi-horizon forecasting — h = " + ", ".join(str(h) for h in horizons),
        "",
        f"Version: `{MULTI_HORIZON_VERSION}`",
        "",
        "Two arms. `separate` trains one model per horizon, four independent",
        "trunks. `shared` trains one trunk with one linear head per horizon on the",
        "summed masked loss. Both see identical windows, cut at the longest",
        "horizon so every horizon is scored on the same forecast origins.",
        "",
        "**Every horizon is judged against its own persistence.** `scripts/15`",
        "scores persistence at h=1 only; it degrades sharply as the horizon grows,",
        "so each row below is compared to a persistence rescored at that horizon on",
        "the same folds and masks.",
        "",
        "## Naive baselines, rescored per horizon",
        "",
        "| Model | Horizon | MAE | RMSE | Peak MAE | 2017 MAE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in naive_summary.itertuples():
        lines.append(
            f"| {row.model} | {row.horizon} | {row.headline_mae:.2f} | "
            f"{row.headline_rmse:.2f} | {row.headline_peak_mae:.2f} | "
            f"{row.epidemic_2017_mae:.2f} |"
        )

    lines += [
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| `variant` | {variant} |",
        f"| `horizons` | {list(horizons)} |",
    ]
    for key, value in config.items():
        lines.append(f"| `{key}` | {value} |")

    single = summary[~summary["ensemble"]]

    lines += [
        "",
        "## Results by horizon — single seed models",
        "",
        "Mean over the headline folds (1, 2, 3, 6, 7, 8, 9), averaged over seeds.",
        "",
        "| Backbone | Arm | h | MAE | RMSE | Peak MAE | 2017 MAE | Seed sd | persistence MAE | Skill vs persistence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    skill_single = skill[~skill["ensemble"]].set_index(
        ["backbone", "arm", "horizon"]
    )

    for row in single.itertuples():
        entry = skill_single.loc[(row.backbone, row.arm, row.horizon)]
        lines.append(
            f"| `{row.backbone}` | `{row.arm}` | {row.horizon} | "
            f"{row.headline_mae:.2f} | {row.headline_rmse:.2f} | "
            f"{row.headline_peak_mae:.2f} | {row.epidemic_2017_mae:.2f} | "
            f"{row.seed_sd:.2f} | {entry['persistence_mae']:.2f} | "
            f"**{entry['skill']:+.1f}%** |"
        )

    lines += [
        "",
        "## The question: does skill improve with the horizon?",
        "",
        "Raw MAE necessarily rises with the horizon — the task gets harder. What",
        "matters is the margin over a persistence that is *also* getting worse. A",
        "skill score that grows with the horizon is the signature of climate, or",
        "any non-persistence information, becoming load-bearing.",
        "",
        "| Backbone | Arm | " + " | ".join(f"h={h} skill" for h in horizons) + " |",
        "| --- | --- | " + " | ".join("---" for _ in horizons) + " |",
    ]

    for (backbone, arm), group in skill[~skill["ensemble"]].groupby(
        ["backbone", "arm"]
    ):
        by_horizon = group.set_index("horizon")["skill"]
        cells = " | ".join(
            f"{by_horizon.get(h, float('nan')):+.1f}%" for h in horizons
        )
        lines.append(f"| `{backbone}` | `{arm}` | {cells} |")

    lines += [
        "",
        "### Peak MAE skill — the outbreak-warning criterion",
        "",
        "| Backbone | Arm | " + " | ".join(f"h={h}" for h in horizons) + " |",
        "| --- | --- | " + " | ".join("---" for _ in horizons) + " |",
    ]

    for (backbone, arm), group in skill[~skill["ensemble"]].groupby(
        ["backbone", "arm"]
    ):
        by_horizon = group.set_index("horizon")["peak_skill"]
        cells = " | ".join(
            f"{by_horizon.get(h, float('nan')):+.1f}%" for h in horizons
        )
        lines.append(f"| `{backbone}` | `{arm}` | {cells} |")

    lines += [
        "",
        "## Shared versus separate",
        "",
        "`shared` uses one trunk for all horizons against `separate`'s four, so it",
        "has roughly a quarter the trunk parameters and trains in roughly a",
        "quarter the time. Any horizon where it matches or beats `separate` is a",
        "horizon where the near targets are useful auxiliary supervision for the",
        "far ones.",
        "",
        "| Backbone | h | `separate` | `shared` | Δ (shared − separate) |",
        "| --- | --- | --- | --- | --- |",
    ]

    pivot = single.pivot_table(
        index=["backbone", "horizon"], columns="arm", values="headline_mae"
    )
    if "separate" in pivot.columns and "shared" in pivot.columns:
        for (backbone, horizon), row in pivot.iterrows():
            lines.append(
                f"| `{backbone}` | {horizon} | {row['separate']:.2f} | "
                f"{row['shared']:.2f} | {row['shared'] - row['separate']:+.2f} |"
            )

    lines += [
        "",
        "## Seed-mean ensembles",
        "",
        "| Backbone | Arm | h | MAE | Peak MAE | Skill vs persistence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    skill_ensemble = skill[skill["ensemble"]].set_index(
        ["backbone", "arm", "horizon"]
    )
    for row in summary[summary["ensemble"]].itertuples():
        entry = skill_ensemble.loc[(row.backbone, row.arm, row.horizon)]
        lines.append(
            f"| `{row.backbone}` | `{row.arm}` | {row.horizon} | "
            f"{row.headline_mae:.2f} | {row.headline_peak_mae:.2f} | "
            f"**{entry['skill']:+.1f}%** |"
        )

    lines += [
        "",
        "## Output files",
        "",
        f"- `{METRICS_PATH.relative_to(PROJECT_DIR).as_posix()}`",
        f"- `{NAIVE_PATH.relative_to(PROJECT_DIR).as_posix()}` — persistence and seasonal naive at every horizon",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse the training knobs."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--variant", default="v1", choices=["v0", "v1"])
    parser.add_argument("--folds", nargs="+", type=int, default=None)

    for key, value in DEFAULTS.items():
        parser.add_argument(
            f"--{key.replace('_', '-')}", type=type(value), default=value
        )

    return parser.parse_args()


def main() -> int:
    """Train both arms at every horizon and write the report."""

    arguments = parse_arguments()
    load_modules()

    if not FOLDS_PATH.exists() or not ADJACENCY_PATH.exists():
        print("Missing folds.json or adjacency.npz. Run scripts 13 and 14 first.")
        return 1

    config = {key: getattr(arguments, key) for key in DEFAULTS}
    horizons = tuple(sorted(set(arguments.horizons)))

    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))["folds"]
    if arguments.folds:
        folds = [f for f in folds if f["fold_id"] in arguments.folds]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device:    {device}")
    print(f"Variant:   {arguments.variant}")
    print(f"Folds:     {[f['fold_id'] for f in folds]}")
    print(f"Horizons:  {list(horizons)}")
    print(f"Arms:      {arguments.arms}")
    print(f"Seeds:     {config['seeds']}")
    print()

    print("Rescoring naive baselines at every horizon...")
    naive_frame = naive_by_horizon(folds, config["lookback"], horizons)
    naive_summary = summarise_naive(naive_frame)
    persistence = naive_summary[naive_summary["model"] == "persistence"]
    print(
        "  persistence headline MAE:  "
        + "  ".join(
            f"h{row.horizon}: {row.headline_mae:.2f}" for row in persistence.itertuples()
        )
    )
    print()

    started = time.perf_counter()
    metrics = run_experiment(
        tuple(arguments.arms), arguments.variant, folds, config, horizons, device
    )
    elapsed = (time.perf_counter() - started) / 60

    print(f"\nTrained in {elapsed:.1f} min\n")

    summary = summarise(metrics)
    skill = skill_table(summary, naive_summary)

    single = summary[~summary["ensemble"]]
    print("single seed models, headline MAE by horizon")
    print(
        single.pivot_table(
            index=["backbone", "arm"], columns="horizon", values="headline_mae"
        ).to_string(float_format=lambda v: f"{v:8.2f}")
    )

    print("\nskill vs same-horizon persistence (higher is better)")
    print(
        skill[~skill["ensemble"]]
        .pivot_table(index=["backbone", "arm"], columns="horizon", values="skill")
        .to_string(float_format=lambda v: f"{v:+7.1f}%")
    )

    print("\npeak MAE skill vs same-horizon persistence")
    print(
        skill[~skill["ensemble"]]
        .pivot_table(index=["backbone", "arm"], columns="horizon", values="peak_skill")
        .to_string(float_format=lambda v: f"{v:+7.1f}%")
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    naive_frame.to_csv(NAIVE_PATH, index=False)
    write_report(
        summary, skill, naive_summary, metrics, config, horizons, arguments.variant
    )

    print(f"\nWrote {REPORT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {METRICS_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {NAIVE_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

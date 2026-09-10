"""
Does layer normalisation before the prediction head help the GCN+GRU?

The trunk hands the GRU's final hidden state straight to a linear head, with
nothing controlling that state's scale: the recurrent cell can drift its
activation magnitude across the 12-period window, and the graph convolution
feeding it is an unnormalised linear map. Layer normalisation at that point is
the standard treatment for recurrent sequence models (Ba et al., 2016), and the
brief asks for it after the GRU representation and before the final layer, which
is where it is expected to matter most. Two arms:

    baseline    the committed architecture, untouched. `normalise=False` on the
                same class, which reproduces `GCNGRU` bit-for-bit -- asserted in
                `tests/test_layer_norm.py`, not assumed.

    layer_norm  `GRU -> LayerNorm(hidden) -> dropout -> head`. Statistics over
                the hidden axis, per (window, district) row, with the default
                learnable gain and bias (`src/models/layer_norm.py`).

The norm goes **before** the dropout the baseline already applies. Putting it
after would have it renormalise a vector whose moments dropout has distorted at
training time but not at evaluation time, widening the train/eval gap; before is
also the ordering the recurrent and transformer blocks this borrows from use.

**The honest prior is that this moves very little.** Every negative result in
this project shares one documented cause (`docs/learnable_lags_results.md`): at
one week ahead the previous period's case count carries nearly all the signal.
The learnable lags found no gradient (7), the activation sweep was flat (8b),
the hyperparameter surface was flat (8c) and no graph beat no graph (8e). Layer
normalisation addresses optimisation conditioning, and nothing on record says
conditioning is this model's binding constraint. It is measured anyway because
it is cheap, and because a flat result here is more evidence about where the
ceiling actually is.

**Two things are reported that a bare MAE column would hide.** First, the
learned gain and bias: a norm whose gain collapsed toward zero, or grew large to
undo the rescaling, did something different from one that sat near its
initialisation, and "normalising did not help" and "the model turned the
normalisation off" are different findings. Second, epochs to best -- the standard
claim for normalisation is faster and more stable convergence, which can be true
while the final MAE is unchanged, so convergence speed and seed spread are
tabled alongside the score rather than folded into it.

`--horizons` runs the comparison at more than one forecast horizon. The trunk has
to carry more at h=4 than at h=1, so if normalisation helps anywhere it should
help there first; the default is h=1 to stay comparable with the committed
results.

Everything else -- architecture, graph, folds, preprocessing, windowing, early
stopping, optimiser, gradient clipping, the anchored residual target, the
masked-MSE objective and the metrics -- is imported from
`scripts/16.train_gcn_gru.py`.

Usage:

    python scripts/28.train_layer_norm.py                        both arms, 9 folds, 3 seeds
    python scripts/28.train_layer_norm.py --folds 8 --seeds 1    smoke run
    python scripts/28.train_layer_norm.py --horizons 1 4         h=1 and h=4
    python scripts/28.train_layer_norm.py --model gru_only       identity adjacency

Outputs:
    results/models/layer_norm_report.md
    results/models/layer_norm_metrics.csv
    results/models/layer_norm_affine.csv
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
sys.path.insert(0, str(PROJECT_DIR))

from src.models.layer_norm import (  # noqa: E402
    LayerNormGCNGRU,
    count_parameters,
    norm_statistics,
)


PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "models"

FOLDS_PATH = PROCESSED_DIR / "folds.json"
ADJACENCY_PATH = PROCESSED_DIR / "adjacency.npz"

REPORT_PATH = RESULTS_DIR / "layer_norm_report.md"
METRICS_PATH = RESULTS_DIR / "layer_norm_metrics.csv"
AFFINE_PATH = RESULTS_DIR / "layer_norm_affine.csv"

LAYER_NORM_VERSION = "layer-norm-v1"

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
    "layer_norm_eps": 1e-5,
}

# Arm -> whether the norm is applied. One class serves both, so the control is
# the same code path with one flag flipped rather than a separate model that
# could drift away from it.
ARMS = {
    "baseline": False,
    "layer_norm": True,
}

CONTROL_ARM = "baseline"

# Adjacency choice. `gcn_gru` is the contiguity graph the brief's "GCN+GRU
# pipeline" names; `gru_only` substitutes the identity and is the stronger of
# the two on the committed sweep (docs/baseline.md), so both are offered.
MODEL_ADJACENCY = {
    "gcn_gru": "A_norm",
    "gru_only": None,
}


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

    baseline_module = _load("baseline_module", "training/16.train_gcn_gru.py")
    baseline_module.load_modules()

    naive = baseline_module.naive
    folds_module = baseline_module.folds_module


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_adjacency(model: str, n_nodes: int) -> np.ndarray:
    """Return the adjacency for the chosen backbone."""

    key = MODEL_ADJACENCY[model]

    if key is None:
        return np.eye(n_nodes, dtype=np.float32)

    with np.load(ADJACENCY_PATH, allow_pickle=True) as data:
        if key not in data.files:
            raise KeyError(
                f"{key!r} is not in adjacency.npz. Rerun scripts/13.build_adjacency.py."
            )
        return data[key].astype(np.float32)


def make_model_builder(arm: str, config: dict):
    """Return a `build_model(n_features) -> nn.Module` closure for one arm.

    Both arms construct the identical `GCNGRU` first, under the seed the
    training loop has already set, and then wrap it. Because the wrapper adopts
    the backbone's submodules by reference and adds the norm afterwards, the two
    arms' shared weights are drawn from the same generator in the same order --
    so a difference between them is the normalisation and not a different random
    initialisation.

    The head is always one wide. `horizon` is overloaded in `scripts/16`: it is
    the forecast lead in `build_fold_arrays` and the head *width* in the
    `GCNGRU` constructor, and the two coincide only at h=1, which is the only
    value script 16 ever runs. This experiment forecasts one horizon at a time --
    a separate model per lead, as in script 27's `separate` arm -- so the lead
    belongs to the windower and the head stays one wide, exactly as
    `scripts/27` builds its backbone.
    """

    normalise = ARMS[arm]

    def build_model(n_features: int) -> nn.Module:
        backbone = baseline_module.GCNGRU(
            n_features=n_features,
            hidden=config["hidden"],
            gcn_layers=config["gcn_layers"],
            horizon=1,
            dropout=config["dropout"],
        )

        return LayerNormGCNGRU(
            backbone,
            normalise=normalise,
            eps=config["layer_norm_eps"],
        )

    return build_model


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(
    arms: list[str],
    model_name: str,
    variant: str,
    folds: list[dict],
    horizons: list[int],
    config: dict,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train every arm on every fold and horizon, and score the test years."""

    calendar = folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()

    nodes = pd.read_csv(PROCESSED_DIR / "nodes.csv").sort_values("node_id")
    n_nodes = len(nodes)

    adjacency = load_adjacency(model_name, n_nodes)
    tensors = folds_module.load_tensors(variant)

    rows: list[dict] = []
    affine_rows: list[dict] = []

    for horizon in horizons:
        horizon_config = dict(config)
        horizon_config["horizon"] = horizon

        for fold in folds:
            arrays = baseline_module.build_fold_arrays(
                tensors, months, fold, horizon_config["lookback"], horizon
            )

            fit_mask = tensors["period_id"] <= fold["fit_end_period"]
            thresholds = naive.peak_thresholds(
                tensors["y"], tensors["y_mask"], fit_mask
            )

            target = arrays["test"]["y"]
            mask = arrays["test"]["mask"].astype(np.int8)

            for arm in arms:
                started = time.perf_counter()
                seed_predictions = []
                seed_info = []

                for seed in range(horizon_config["seeds"]):
                    model, info = baseline_module.train_one(
                        arrays,
                        adjacency,
                        horizon_config,
                        seed,
                        device,
                        build_model=make_model_builder(arm, horizon_config),
                    )
                    prediction = baseline_module.predict(
                        model,
                        arrays["test"],
                        adjacency,
                        horizon_config["target"],
                        device,
                    )

                    seed_predictions.append(prediction)
                    seed_info.append(info)

                    scores = naive.evaluate(prediction, target, mask, thresholds)
                    rows.append(
                        {
                            "arm": arm,
                            "model": model_name,
                            "variant": variant,
                            "horizon": horizon,
                            "fold_id": fold["fold_id"],
                            "test_year": fold["test_year"],
                            "covers_covid": fold["covers_covid"],
                            "headline": fold["headline"],
                            "seed": seed,
                            "best_epoch": info["best_epoch"],
                            "parameters": count_parameters(model),
                            "ensemble": False,
                            **scores,
                        }
                    )

                    # Keep the learned gain and bias, so a flat MAE can be told
                    # apart from a norm the model learned to switch off.
                    statistics = norm_statistics(model)
                    if statistics:
                        affine_rows.append(
                            {
                                "arm": arm,
                                "horizon": horizon,
                                "fold_id": fold["fold_id"],
                                "seed": seed,
                                **statistics,
                            }
                        )

                mean_prediction = np.mean(seed_predictions, axis=0)
                ensemble_scores = naive.evaluate(
                    mean_prediction, target, mask, thresholds
                )
                rows.append(
                    {
                        "arm": arm,
                        "model": model_name,
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
                        "parameters": count_parameters(model),
                        "ensemble": True,
                        **ensemble_scores,
                    }
                )

                elapsed = time.perf_counter() - started
                seed_mae = [
                    row["mae"] for row in rows[-horizon_config["seeds"] - 1 : -1]
                ]

                print(
                    f"  h={horizon} fold {fold['fold_id']} ({fold['test_year']}) "
                    f"{arm:<11} MAE {np.mean(seed_mae):7.2f} +/- "
                    f"{np.std(seed_mae):5.2f}  ens {ensemble_scores['mae']:7.2f}  "
                    f"epochs {np.mean([i['best_epoch'] for i in seed_info]):5.1f}  "
                    f"{elapsed:5.1f}s",
                    flush=True,
                )

    return pd.DataFrame(rows), pd.DataFrame(affine_rows)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average over seeds, then over the headline folds, per arm and horizon."""

    records = []

    for ensemble in (False, True):
        subset = metrics[metrics["ensemble"] == ensemble]

        for (arm, horizon), group in subset.groupby(["arm", "horizon"]):
            by_fold = (
                group.groupby(["fold_id", "test_year", "headline", "covers_covid"])
                .agg(
                    mae=("mae", "mean"),
                    rmse=("rmse", "mean"),
                    peak_mae=("peak_mae", "mean"),
                    best_epoch=("best_epoch", "mean"),
                )
                .reset_index()
            )

            headline = by_fold[by_fold["headline"]]
            covid = by_fold[by_fold["covers_covid"]]
            epidemic = by_fold[by_fold["test_year"] == 2017]

            spread = (
                group.groupby("fold_id")["mae"].std().mean()
                if not ensemble
                else float("nan")
            )

            records.append(
                {
                    "arm": arm,
                    "horizon": horizon,
                    "ensemble": ensemble,
                    "headline_mae": headline["mae"].mean(),
                    "headline_rmse": headline["rmse"].mean(),
                    "headline_peak_mae": headline["peak_mae"].mean(),
                    "epidemic_2017_mae": epidemic["mae"].mean(),
                    "covid_mae": covid["mae"].mean(),
                    "seed_sd": spread,
                    "best_epoch": headline["best_epoch"].mean(),
                    "parameters": int(group["parameters"].iloc[0]),
                }
            )

    return (
        pd.DataFrame(records)
        .sort_values(["ensemble", "horizon", "headline_mae"])
        .reset_index(drop=True)
    )


def compare_to_control(metrics: pd.DataFrame) -> pd.DataFrame:
    """Paired comparison of each arm against the un-normalised baseline.

    Paired over the headline folds, which is the right pairing: both arms see
    identical folds, identical windows, identical seeds and identical initial
    weights, so the fold-to-fold variation that dominates the raw spread cancels.

    Includes the fold-1 decomposition. Four separate changes in this project have
    produced a headline movement that turned out to be the 2017 epidemic fold and
    nothing else, so the split is reported by default rather than on request.
    """

    single = metrics[~metrics["ensemble"]]
    headline = single[single["headline"]]

    records = []

    for horizon, horizon_group in headline.groupby("horizon"):
        by_fold = horizon_group.groupby(["arm", "fold_id"])["mae"].mean().unstack(0)
        if CONTROL_ARM not in by_fold.columns:
            continue

        seed_sd = (
            single[single["horizon"] == horizon]
            .groupby(["arm", "fold_id"])["mae"]
            .std()
            .groupby("arm")
            .mean()
        )
        epochs = horizon_group.groupby("arm")["best_epoch"].mean()

        control = by_fold[CONTROL_ARM]
        others = [f for f in by_fold.index if f != 1]

        for arm in by_fold.columns:
            if arm == CONTROL_ARM:
                continue

            delta = by_fold[arm] - control
            total = delta.mean()

            try:
                from scipy import stats

                statistic, p_value = stats.ttest_rel(by_fold[arm], control)
            except Exception:
                statistic, p_value = float("nan"), float("nan")

            pooled = float(np.nanmean([seed_sd.get(arm), seed_sd.get(CONTROL_ARM)]))

            records.append(
                {
                    "arm": arm,
                    "horizon": horizon,
                    "headline_delta": total,
                    "folds_improved": int((delta < 0).sum()),
                    "folds_total": len(delta),
                    "t_statistic": float(statistic),
                    "p_value": float(p_value),
                    "pooled_seed_sd": pooled,
                    "delta_in_sd": abs(total) / pooled if pooled else float("nan"),
                    "fold1_delta": float(delta.loc[1])
                    if 1 in delta.index
                    else float("nan"),
                    "ex_fold1_delta": float(delta.loc[others].mean())
                    if others
                    else float("nan"),
                    "seed_sd_delta": float(
                        seed_sd.get(arm, np.nan) - seed_sd.get(CONTROL_ARM, np.nan)
                    ),
                    "epoch_delta": float(
                        epochs.get(arm, np.nan) - epochs.get(CONTROL_ARM, np.nan)
                    ),
                }
            )

    return pd.DataFrame(records)


def summarise_affine(affine: pd.DataFrame) -> pd.DataFrame:
    """Mean learned gain and bias per horizon, over folds and seeds."""

    if not len(affine):
        return pd.DataFrame()

    return (
        affine.groupby("horizon")
        .agg(
            gain_mean=("gain_mean", "mean"),
            gain_sd=("gain_sd", "mean"),
            gain_min=("gain_min", "mean"),
            gain_max=("gain_max", "mean"),
            bias_abs_mean=("bias_abs_mean", "mean"),
        )
        .reset_index()
    )


def naive_summary_frame(horizons: list[int]) -> pd.DataFrame:
    """Read the committed naive baseline metrics and summarise them like the arms.

    Script 15 scores persistence at h=1 only. When a run covers other horizons,
    `results/models/multi_horizon_naive.csv` from script 27 carries the rescored
    per-horizon rows; comparing an h=4 model against the h=1 persistence would
    flatter it against a baseline measured on a much easier task.
    """

    records = []

    multi_path = RESULTS_DIR / "multi_horizon_naive.csv"
    multi = pd.read_csv(multi_path) if multi_path.exists() else pd.DataFrame()

    for horizon in horizons:
        if horizon == 1:
            path = RESULTS_DIR / "naive_baseline_metrics.csv"
            if not path.exists():
                continue

            metrics = pd.read_csv(path)
            for model, group in metrics.groupby("model"):
                headline = group[group["headline"]]
                records.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "headline_mae": headline["mae"].mean(),
                        "headline_peak_mae": headline["peak_mae"].mean(),
                        "epidemic_2017_mae": group.loc[
                            group["test_year"] == 2017, "mae"
                        ].mean(),
                    }
                )
        elif len(multi) and "horizon" in multi.columns:
            subset = multi[multi["horizon"] == horizon]
            for model, group in subset.groupby("model"):
                headline = group[group["headline"]] if "headline" in group else group
                records.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "headline_mae": headline["mae"].mean(),
                        "headline_peak_mae": headline["peak_mae"].mean()
                        if "peak_mae" in headline
                        else float("nan"),
                        "epidemic_2017_mae": group.loc[
                            group["test_year"] == 2017, "mae"
                        ].mean()
                        if "test_year" in group
                        else float("nan"),
                    }
                )

    if not records:
        return pd.DataFrame()

    return (
        pd.DataFrame(records)
        .sort_values(["horizon", "headline_mae"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    affine_summary: pd.DataFrame,
    naive_summary: pd.DataFrame,
    config: dict,
    model_name: str,
    variant: str,
    horizons: list[int],
) -> None:
    """Write the layer-normalisation report."""

    lines = [
        "# Layer normalisation before the prediction head",
        "",
        f"Version: `{LAYER_NORM_VERSION}`",
        "",
        "One GCN+GRU, two arms, one code path. Architecture, training loop, early",
        "stopping, optimiser, folds, masks, preprocessing, windowing, target",
        "parameterisation, loss and metrics are imported from",
        "`scripts/16.train_gcn_gru.py`. Only the normalisation changes.",
        "",
        "| Arm | Forward pass |",
        "| --- | --- |",
        "| `baseline` | GRU -> dropout -> head (the committed architecture) |",
        "| `layer_norm` | GRU -> **LayerNorm** -> dropout -> head |",
        "",
        "The norm is applied to the last hidden state, over the hidden axis, per",
        "(window, district) row -- so each district is normalised on its own terms",
        "and a district in an outbreak is not rescaled by its neighbours. It sits",
        "before the dropout the baseline already applies: normalising after dropout",
        "would compute statistics on a vector whose moments are distorted at",
        "training time but not at evaluation time.",
        "",
        "`baseline` is the same class with `normalise=False`, which reproduces",
        "`GCNGRU` exactly -- `tests/test_layer_norm.py` asserts bit-identical",
        "outputs and an identical parameter count, so the control is the committed",
        "model and not a re-implementation of it.",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| `model` | {model_name} ({'contiguity graph' if model_name == 'gcn_gru' else 'identity adjacency'}) |",
        f"| `variant` | {variant} |",
        f"| `horizons` | {horizons} |",
    ]
    for key, value in config.items():
        if key == "horizon":
            continue
        lines.append(f"| `{key}` | {value} |")

    single = summary[~summary["ensemble"]]
    ensemble = summary[summary["ensemble"]]

    lines += [
        "",
        "## Results — single seed models",
        "",
        "Mean over the headline folds (1, 2, 3, 6, 7, 8, 9), averaged over seeds.",
        "`Epochs` is the mean epoch early stopping selected: the standard claim",
        "for normalisation is faster convergence, which can hold while the final",
        "score does not move.",
        "",
        "| h | Arm | MAE | RMSE | Peak MAE | 2017 MAE | COVID MAE | Seed sd | Epochs | Params |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in single.itertuples():
        lines.append(
            f"| {row.horizon} | `{row.arm}` | {row.headline_mae:.2f} | "
            f"{row.headline_rmse:.2f} | {row.headline_peak_mae:.2f} | "
            f"{row.epidemic_2017_mae:.2f} | {row.covid_mae:.2f} | "
            f"{row.seed_sd:.2f} | {row.best_epoch:.1f} | {row.parameters} |"
        )

    lines += [
        "",
        "## Results — seed-mean ensembles",
        "",
        "| h | Arm | MAE | RMSE | Peak MAE | 2017 MAE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in ensemble.itertuples():
        lines.append(
            f"| {row.horizon} | `{row.arm}` | {row.headline_mae:.2f} | "
            f"{row.headline_rmse:.2f} | {row.headline_peak_mae:.2f} | "
            f"{row.epidemic_2017_mae:.2f} |"
        )

    if len(naive_summary):
        lines += ["", "### Naive baselines, same folds and masks", ""]
        lines += [
            "| h | Model | MAE | Peak MAE | 2017 MAE |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in naive_summary.itertuples():
            lines.append(
                f"| {row.horizon} | {row.model} | {row.headline_mae:.2f} | "
                f"{row.headline_peak_mae:.2f} | {row.epidemic_2017_mae:.2f} |"
            )

    if len(comparison):
        lines += [
            "",
            "## Does layer normalisation beat the un-normalised baseline?",
            "",
            "Paired over the seven headline folds. Both arms see identical folds,",
            "windows, seeds and initial weights, so the fold-to-fold variation",
            "cancels. `Δ` is the headline MAE change; negative is better. `Δ in sd`",
            "compares the gap to the pooled seed standard deviation — this project",
            "treats anything under 2 sd as not established.",
            "",
            "| h | Arm | Δ vs baseline | Folds improved | t | p | Pooled seed sd | Δ in sd | Verdict |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]

        for row in comparison.itertuples():
            if row.headline_delta < 0 and row.delta_in_sd > 2 and row.p_value < 0.05:
                verdict = "**improves**"
            elif row.headline_delta > 0 and row.delta_in_sd > 2 and row.p_value < 0.05:
                verdict = "**worse**"
            else:
                verdict = "not established"

            lines.append(
                f"| {row.horizon} | `{row.arm}` | {row.headline_delta:+.2f} | "
                f"{row.folds_improved}/{row.folds_total} | {row.t_statistic:+.2f} | "
                f"{row.p_value:.3f} | {row.pooled_seed_sd:.2f} | "
                f"{row.delta_in_sd:.2f} | {verdict} |"
            )

        lines += [
            "",
            "### Convergence and stability",
            "",
            "The two claims made for normalisation that are not about the final",
            "score. Negative `Δ epochs` means early stopping fired sooner;",
            "negative `Δ seed sd` means the run-to-run spread narrowed.",
            "",
            "| h | Arm | Δ epochs to best | Δ seed sd |",
            "| --- | --- | --- | --- |",
        ]
        for row in comparison.itertuples():
            lines.append(
                f"| {row.horizon} | `{row.arm}` | {row.epoch_delta:+.1f} | "
                f"{row.seed_sd_delta:+.3f} |"
            )

        lines += [
            "",
            "### Where any movement comes from",
            "",
            "Several changes in this project have produced a headline movement",
            "that turned out to be the 2017 epidemic fold and nothing else, so",
            "the split is reported by default.",
            "",
            "| h | Arm | Fold 1 Δ | Δ over the other six folds |",
            "| --- | --- | --- | --- |",
        ]
        for row in comparison.itertuples():
            lines.append(
                f"| {row.horizon} | `{row.arm}` | {row.fold1_delta:+.2f} | "
                f"{row.ex_fold1_delta:+.3f} |"
            )

    if len(affine_summary):
        lines += [
            "",
            "## What the norm learned",
            "",
            "The gain and bias say whether the module normalised or learned to",
            "undo itself. A gain near its initialisation of 1.0 with a small bias",
            "is plain normalisation; a gain collapsed toward 0 means the head",
            "learned to ignore the representation; a large gain means capacity was",
            "spent reversing the rescaling. Mean over folds and seeds.",
            "",
            "| h | Gain mean | Gain sd | Gain min | Gain max | Mean abs bias |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in affine_summary.itertuples():
            lines.append(
                f"| {row.horizon} | {row.gain_mean:.3f} | {row.gain_sd:.3f} | "
                f"{row.gain_min:.3f} | {row.gain_max:.3f} | {row.bias_abs_mean:.3f} |"
            )

    lines += [
        "",
        "## Reproducing",
        "",
        "```bash",
        "python scripts/28.train_layer_norm.py \\",
        f"    --model {model_name} --variant {variant} \\",
        f"    --horizons {' '.join(str(h) for h in horizons)} "
        f"--seeds {config['seeds']}",
        "```",
        "",
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument(
        "--model",
        default="gcn_gru",
        choices=list(MODEL_ADJACENCY),
        help="backbone adjacency: the contiguity graph, or the identity control",
    )
    parser.add_argument("--variant", default="v1", choices=["v0", "v1"])
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1],
        help="forecast horizons to run the comparison at",
    )

    for key, value in DEFAULTS.items():
        if key == "horizon":
            continue
        parser.add_argument(
            f"--{key.replace('_', '-')}", type=type(value), default=value
        )

    return parser.parse_args()


def main() -> int:
    """Train both arms on every fold and write the report."""

    arguments = parse_arguments()
    load_modules()

    if not FOLDS_PATH.exists() or not ADJACENCY_PATH.exists():
        print("Missing folds.json or adjacency.npz. Run scripts 13 and 14 first.")
        return 1

    config = {key: getattr(arguments, key, DEFAULTS[key]) for key in DEFAULTS}

    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))["folds"]
    if arguments.folds:
        folds = [f for f in folds if f["fold_id"] in arguments.folds]

    horizons = list(arguments.horizons)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device:   {device}")
    print(f"Model:    {arguments.model}")
    print(f"Variant:  {arguments.variant}")
    print(f"Folds:    {[f['fold_id'] for f in folds]}")
    print(f"Horizons: {horizons}")
    print(f"Arms:     {arguments.arms}")
    print(f"Seeds:    {config['seeds']}")
    print(f"Control:  {CONTROL_ARM} (no normalisation)")
    print()

    started = time.perf_counter()
    metrics, affine = run_experiment(
        arguments.arms,
        arguments.model,
        arguments.variant,
        folds,
        horizons,
        config,
        device,
    )
    elapsed = (time.perf_counter() - started) / 60

    print(f"\nTrained in {elapsed:.1f} min\n")

    summary = summarise(metrics)
    comparison = compare_to_control(metrics)
    affine_summary = summarise_affine(affine)
    naive_frame = naive_summary_frame(horizons)

    single = summary[~summary["ensemble"]]
    print("single seed models")
    print(
        single[
            [
                "horizon",
                "arm",
                "headline_mae",
                "headline_peak_mae",
                "epidemic_2017_mae",
                "seed_sd",
                "best_epoch",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:8.2f}")
    )

    if len(comparison):
        print("\nvs the un-normalised baseline")
        print(
            comparison[
                [
                    "horizon",
                    "arm",
                    "headline_delta",
                    "folds_improved",
                    "t_statistic",
                    "p_value",
                    "delta_in_sd",
                    "epoch_delta",
                    "ex_fold1_delta",
                ]
            ].to_string(index=False, float_format=lambda v: f"{v:8.3f}")
        )

    if len(affine_summary):
        print("\nwhat the norm learned")
        print(
            affine_summary.to_string(
                index=False, float_format=lambda v: f"{v:8.3f}"
            )
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    if len(affine):
        affine.to_csv(AFFINE_PATH, index=False)

    write_report(
        summary,
        comparison,
        affine_summary,
        naive_frame,
        config,
        arguments.model,
        arguments.variant,
        horizons,
    )

    print(f"\nWrote {REPORT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {METRICS_PATH.relative_to(PROJECT_DIR)}")
    if len(affine):
        print(f"Wrote {AFFINE_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

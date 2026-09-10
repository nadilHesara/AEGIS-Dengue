"""
Does *any* graph beat no graph? Four adjacencies through one GCN+GRU.

The committed baseline sweep settled a question this project had been treating
as open: the queen-contiguity graph does not help, it hurts. `gru_only` -- the
identical model with the identity substituted for the adjacency -- beats
`gcn_gru` on all nine folds, by 1.9-2.3 MAE on the headline mean against a seed
sd of 0.30-0.68. `docs/baseline.md` records that as decided.

That result is about *one* graph. It does not say spatial structure is useless
here; it says queen contiguity is. Two readings survive it:

    the graph is wrong        contiguity is a poor prior for a country 430 km
                              long. Colombo and Galle are 100 km apart, share no
                              border, and are both wet-zone coastal; contiguity
                              scores that pair zero and scores Colombo-Ratnapura
                              full. On this reading a better graph should help.

    there is no spatial       whatever cross-district signal exists at one week
    signal to find            ahead is already inside each district's own case
                              history. On this reading no graph helps.

This script separates them. Four adjacencies, one architecture, one code path:

    identity      A = I. The graph convolution degenerates to a per-node linear
                  layer, so this arm *is* the `gru_only` control -- it must
                  reproduce script 16's committed 16.68, which makes it a
                  correctness check as well as the reference.
    contiguity    A_norm, the queen-contiguity graph. Reproduces `gcn_gru`.
    gaussian      A_gaussian_norm, exp(-(d/50km)^2) on centroid distance,
                  thresholded at 0.1 -- built by `scripts/13` and never yet
                  trained on. The competing fixed prior.
    adaptive      A learned end-to-end from the forecasting loss:
                  softmax(relu(E1 @ E2.T)) over two [25, 8] embedding tables
                  (`src/models/adaptive_graph.py`). Commits to no prior at all.

The adaptive arm is what makes the conclusion strong. A fixed alternative that
fails could always be the wrong fixed alternative. A graph given 400 free
parameters and the training objective itself, which still fails to beat the
identity, is evidence about the data rather than about anyone's choice of prior.

**The comparison the brief asks for is against the identity, not against
contiguity.** Beating contiguity is a low bar -- contiguity loses to doing
nothing. Every table below is therefore anchored on `identity`, and an arm only
counts as an improvement if it beats *that*.

Everything except the adjacency is imported from `scripts/16.train_gcn_gru.py`:
architecture, training loop, early stopping, optimiser, folds, preprocessing,
windowing, the anchored residual target, the masked-MSE objective and the
metrics. The `identity` and `contiguity` arms reproduce script 16's two
committed models exactly; if they do not, the harness is not neutral.

On the adaptive arm's learning rate, which is not a free choice. The embeddings
sit behind a relu and a softmax, so the gradient reaching them is far smaller
than the gradient reaching the GRU -- the same problem
`scripts/16.parameter_groups` already documents for the lag encoder ("at a
shared learning rate they barely move from their initialisation and the module
looks like it does nothing"). Measured on fold 8, that is exactly what happens:

    multiplier 1x    normalised row entropy 1.000 -- the graph never leaves its
                     uniform initialisation, and the arm scores 10.30, worse
                     than the identity control. This tests whether the
                     embeddings can move, not whether a learned graph helps.
    multiplier 3-10x the graph moves, and collapses to near one-hot rows
                     (entropy 0.06-0.09, largest weight ~1.0).

Neither extreme is obviously the right setting, and picking one by looking at
test scores would be exactly the mistake `docs/hyperparameter_tuning.md` was
written to avoid. So the multiplier is **swept and selected on the validation
split**, per fold, from `--graph-lr-grid`. The test number is then computed once
for the selected value. The adaptive arm therefore gets a genuine chance rather
than a hand-picked hyperparameter, and gets it without touching the test folds.

Usage:

    python scripts/26.train_graph_variants.py                    4 arms, 9 folds, 3 seeds
    python scripts/26.train_graph_variants.py --folds 8 --seeds 1   smoke run
    python scripts/26.train_graph_variants.py --arms identity adaptive

Outputs:
    results/models/graph_report.md
    results/models/graph_metrics.csv
    results/models/graph_learned_adjacency.csv
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


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.models.adaptive_graph import (  # noqa: E402
    AdaptiveAdjacency,
    AdaptiveGraphGCNGRU,
)


PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "models"

FOLDS_PATH = PROCESSED_DIR / "folds.json"
ADJACENCY_PATH = PROCESSED_DIR / "adjacency.npz"

REPORT_PATH = RESULTS_DIR / "graph_report.md"
METRICS_PATH = RESULTS_DIR / "graph_metrics.csv"
LEARNED_PATH = RESULTS_DIR / "graph_learned_adjacency.csv"

GRAPH_VERSION = "graph-v1"

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
    "embedding_dim": 8,
    "graph_lr_multiplier": 10.0,
}

# Candidate multipliers for the adaptive arm's graph parameters, selected per
# fold on the validation split. 1x is included so "the embeddings barely move"
# stays on the menu rather than being ruled out by assumption.
GRAPH_LR_GRID = [1.0, 3.0, 10.0, 30.0]

# Arm -> (key in adjacency.npz, or None for a graph this script constructs).
ARMS = {
    "identity": None,
    "contiguity": "A_norm",
    "gaussian": "A_gaussian_norm",
    "adaptive": "learned",
}

# The reference every arm is judged against. Not contiguity: contiguity already
# loses to this, so beating it would establish nothing.
CONTROL_ARM = "identity"


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
# Adjacencies
# ---------------------------------------------------------------------------

def load_adjacencies(n_nodes: int) -> dict[str, np.ndarray]:
    """Return the fixed adjacency for each arm that has one.

    The adaptive arm has no fixed matrix; it still receives one through the
    training loop's signature, and its model ignores it.
    """

    with np.load(ADJACENCY_PATH, allow_pickle=True) as data:
        available = {key: data[key] for key in data.files}

    identity = np.eye(n_nodes, dtype=np.float32)

    matrices: dict[str, np.ndarray] = {}
    for arm, key in ARMS.items():
        if key is None:
            matrices[arm] = identity
        elif key == "learned":
            # Placeholder, ignored by the adaptive model. Passing the identity
            # rather than a real graph means a bug that used it instead of the
            # learned one would show up as the identity's number, not as a
            # plausible-looking graph result.
            matrices[arm] = identity
        else:
            if key not in available:
                raise KeyError(
                    f"{key!r} is not in adjacency.npz. Rerun scripts/13.build_adjacency.py."
                )
            matrices[arm] = available[key].astype(np.float32)

    return matrices


def describe_adjacency(matrix: np.ndarray) -> dict:
    """Summary statistics for one adjacency, for the report."""

    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0.0)

    nonzero = off_diagonal != 0
    degrees = nonzero.sum(axis=1)

    return {
        "edges": int(nonzero.sum()),
        "mean_degree": float(degrees.mean()),
        "min_degree": int(degrees.min()),
        "max_degree": int(degrees.max()),
        "symmetric": bool(np.allclose(matrix, matrix.T, atol=1e-6)),
        "self_weight": float(np.diag(matrix).mean()),
        "off_diagonal_mass": float(off_diagonal.sum() / matrix.sum())
        if matrix.sum()
        else 0.0,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_model_builder(arm: str, config: dict, n_nodes: int, seed: int):
    """Return a `build_model(n_features) -> nn.Module` closure for one arm.

    Every arm builds the same `GCNGRU` with the same constructor arguments. The
    adaptive arm then wraps it so the adjacency comes from learned embeddings
    instead of the supplied matrix; the fixed arms return it unwrapped.
    """

    def build_model(n_features: int) -> nn.Module:
        backbone = baseline_module.GCNGRU(
            n_features=n_features,
            hidden=config["hidden"],
            gcn_layers=config["gcn_layers"],
            horizon=config["horizon"],
            dropout=config["dropout"],
        )

        if arm != "adaptive":
            return backbone

        return AdaptiveGraphGCNGRU(
            backbone,
            AdaptiveAdjacency(
                n_nodes=n_nodes,
                embedding_dim=config["embedding_dim"],
                seed=seed,
            ),
        )

    return build_model


def make_optimiser(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Adam, with the learned graph parameters on their own higher rate.

    The embeddings sit behind a relu and a softmax, so their gradient is far
    smaller than the GRU's -- the same asymmetry `scripts/16.parameter_groups`
    documents for the lag encoder and patches the same way. Models without an
    adaptive graph fall through to script 16's own optimiser exactly, so the
    fixed arms are unaffected by this function existing.
    """

    graph = getattr(model, "adjacency", None)

    if graph is None or not isinstance(graph, nn.Module):
        return baseline_module.build_optimiser(model, config)

    graph_ids = {id(p) for p in graph.parameters()}
    rest = [p for p in model.parameters() if id(p) not in graph_ids]

    return torch.optim.Adam(
        [
            {"params": rest},
            {
                "params": list(graph.parameters()),
                "lr": config["learning_rate"] * config["graph_lr_multiplier"],
                # No decay on the embeddings: shrinking them toward zero pushes
                # the softmax toward uniform, which is a prior on the graph
                # rather than the regularisation weight decay is meant to be.
                "weight_decay": 0.0,
            },
        ],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )


def select_graph_lr(
    arrays: dict,
    adjacency: np.ndarray,
    config: dict,
    n_nodes: int,
    thresholds: np.ndarray,
    grid: list[float],
    device: torch.device,
) -> tuple[float, list[dict]]:
    """Pick the adaptive arm's LR multiplier on the **validation** split.

    Trains one seed per candidate and scores the validation split -- never the
    test split -- returning the multiplier with the lowest validation MAE. One
    seed because this is a per-fold nuisance-parameter choice inside an already
    seeded experiment, not the headline measurement; the selected value is then
    run at full seeds for the number that gets reported.
    """

    trials = []

    for multiplier in grid:
        candidate = dict(config)
        candidate["graph_lr_multiplier"] = multiplier

        model, _ = baseline_module.train_one(
            arrays,
            adjacency,
            candidate,
            0,
            device,
            build_model=make_model_builder("adaptive", candidate, n_nodes, 0),
            make_optimiser=make_optimiser,
        )
        prediction = baseline_module.predict(
            model, arrays["val"], adjacency, candidate["target"], device
        )
        scores = naive.evaluate(
            prediction,
            arrays["val"]["y"],
            arrays["val"]["mask"].astype(np.int8),
            thresholds,
        )
        trials.append({"graph_lr_multiplier": multiplier, "val_mae": scores["mae"]})

    best = min(trials, key=lambda t: t["val_mae"])

    return best["graph_lr_multiplier"], trials


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
    n_nodes = len(names)

    adjacencies = load_adjacencies(n_nodes)
    tensors = folds_module.load_tensors(variant)

    rows: list[dict] = []
    learned_rows: list[dict] = []
    selection_rows: list[dict] = []

    for fold in folds:
        arrays = baseline_module.build_fold_arrays(
            tensors, months, fold, config["lookback"], config["horizon"]
        )

        fit_mask = tensors["period_id"] <= fold["fit_end_period"]
        thresholds = naive.peak_thresholds(tensors["y"], tensors["y_mask"], fit_mask)

        target = arrays["test"]["y"]
        mask = arrays["test"]["mask"].astype(np.int8)

        for arm in arms:
            adjacency = adjacencies[arm]

            started = time.perf_counter()
            seed_predictions = []
            seed_info = []

            # The adaptive arm's graph learning rate is a nuisance parameter
            # with a documented failure mode at both ends, so it is chosen per
            # fold on the validation split rather than fixed by hand.
            arm_config = config
            selected_lr = float("nan")
            if arm == "adaptive" and len(config["graph_lr_grid"]) > 1:
                selected_lr, lr_trials = select_graph_lr(
                    arrays,
                    adjacency,
                    config,
                    n_nodes,
                    thresholds,
                    config["graph_lr_grid"],
                    device,
                )
                arm_config = dict(config)
                arm_config["graph_lr_multiplier"] = selected_lr

                for trial in lr_trials:
                    selection_rows.append(
                        {"fold_id": fold["fold_id"], "selected": selected_lr, **trial}
                    )
            elif arm == "adaptive":
                selected_lr = config["graph_lr_multiplier"]

            for seed in range(config["seeds"]):
                model, info = baseline_module.train_one(
                    arrays,
                    adjacency,
                    arm_config,
                    seed,
                    device,
                    build_model=make_model_builder(arm, arm_config, n_nodes, seed),
                    make_optimiser=make_optimiser if arm == "adaptive" else None,
                )
                prediction = baseline_module.predict(
                    model, arrays["test"], adjacency, arm_config["target"], device
                )

                seed_predictions.append(prediction)
                seed_info.append(info)

                scores = naive.evaluate(prediction, target, mask, thresholds)
                rows.append(
                    {
                        "arm": arm,
                        "variant": variant,
                        "fold_id": fold["fold_id"],
                        "test_year": fold["test_year"],
                        "covers_covid": fold["covers_covid"],
                        "headline": fold["headline"],
                        "seed": seed,
                        "best_epoch": info["best_epoch"],
                        "graph_lr_multiplier": selected_lr,
                        "ensemble": False,
                        **scores,
                    }
                )

                # Keep the learned graph so it can be inspected rather than
                # assumed. A learned adjacency that stayed uniform would produce
                # the identity's answer for a reason worth knowing.
                if arm == "adaptive":
                    learned = model.learned_adjacency().cpu().numpy()
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            learned_rows.append(
                                {
                                    "fold_id": fold["fold_id"],
                                    "seed": seed,
                                    "source": names[i],
                                    "target": names[j],
                                    "weight": float(learned[i, j]),
                                }
                            )

            mean_prediction = np.mean(seed_predictions, axis=0)
            ensemble_scores = naive.evaluate(mean_prediction, target, mask, thresholds)
            rows.append(
                {
                    "arm": arm,
                    "variant": variant,
                    "fold_id": fold["fold_id"],
                    "test_year": fold["test_year"],
                    "covers_covid": fold["covers_covid"],
                    "headline": fold["headline"],
                    "seed": -1,
                    "best_epoch": float(np.mean([i["best_epoch"] for i in seed_info])),
                    "graph_lr_multiplier": selected_lr,
                    "ensemble": True,
                    **ensemble_scores,
                }
            )

            elapsed = time.perf_counter() - started
            seed_mae = [row["mae"] for row in rows[-config["seeds"] - 1 : -1]]

            chosen = "" if np.isnan(selected_lr) else f"  graphLR {selected_lr:4.0f}x"
            print(
                f"  fold {fold['fold_id']} ({fold['test_year']}) {arm:<11} "
                f"MAE {np.mean(seed_mae):7.2f} +/- {np.std(seed_mae):5.2f}  "
                f"ens {ensemble_scores['mae']:7.2f}  "
                f"epochs {np.mean([i['best_epoch'] for i in seed_info]):5.1f}"
                f"{chosen}  {elapsed:5.1f}s",
                flush=True,
            )

    return pd.DataFrame(rows), pd.DataFrame(learned_rows), pd.DataFrame(selection_rows)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average over seeds, then over the headline folds, per arm."""

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
                    "covid_mae": covid["mae"].mean(),
                    "seed_sd": spread,
                }
            )

    return (
        pd.DataFrame(records)
        .sort_values(["ensemble", "headline_mae"])
        .reset_index(drop=True)
    )


def compare_to_control(metrics: pd.DataFrame) -> pd.DataFrame:
    """Per-arm comparison against the identity control, with a paired test.

    Reports what the brief asks for: whether each graph improves over the
    GRU-only identity. Includes the fold-1 decomposition, because four separate
    changes in this project have produced a headline movement that turned out to
    be the 2017 epidemic fold and nothing else.
    """

    single = metrics[~metrics["ensemble"]]
    headline = single[single["headline"]]

    by_fold = headline.groupby(["arm", "fold_id"])["mae"].mean().unstack(0)
    if CONTROL_ARM not in by_fold.columns:
        return pd.DataFrame()

    seed_sd = single.groupby(["arm", "fold_id"])["mae"].std().groupby("arm").mean()

    control = by_fold[CONTROL_ARM]
    others = [f for f in by_fold.index if f != 1]

    records = []
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
                "headline_delta": total,
                "folds_improved": int((delta < 0).sum()),
                "folds_total": len(delta),
                "t_statistic": float(statistic),
                "p_value": float(p_value),
                "pooled_seed_sd": pooled,
                "delta_in_sd": abs(total) / pooled if pooled else float("nan"),
                "fold1_delta": float(delta.loc[1]) if 1 in delta.index else float("nan"),
                "ex_fold1_delta": float(delta.loc[others].mean()),
            }
        )

    return pd.DataFrame(records)


def summarise_learned(learned: pd.DataFrame, contiguity: np.ndarray, names: list[str]) -> dict:
    """What the learned graph converged to, and whether it resembles contiguity.

    A learned adjacency that stayed uniform is a different result from one that
    found sharp structure and still did not help, so the distinction is measured
    rather than left to the reader.
    """

    if not len(learned):
        return {}

    n = len(names)
    index = {name: i for i, name in enumerate(names)}

    # Mean learned matrix over folds and seeds.
    mean = np.zeros((n, n))
    grouped = learned.groupby(["source", "target"])["weight"].mean()
    for (source, target), weight in grouped.items():
        mean[index[source], index[target]] = weight

    uniform = 1.0 / n

    # Entropy per row, normalised so 1.0 is uniform and 0.0 is one-hot.
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(mean * np.log(np.where(mean > 0, mean, 1.0))).sum(axis=1)
    normalised_entropy = float(np.mean(entropy) / np.log(n))

    # Agreement with contiguity: correlation of the off-diagonal weights.
    off = ~np.eye(n, dtype=bool)
    correlation = float(np.corrcoef(mean[off], contiguity[off])[0, 1])

    # How concentrated: mass in each row's top 3 targets.
    top3 = float(np.mean(np.sort(mean, axis=1)[:, -3:].sum(axis=1)))

    return {
        "normalised_entropy": normalised_entropy,
        "uniform_weight": uniform,
        "max_weight": float(mean.max()),
        "top3_mass": top3,
        "contiguity_correlation": correlation,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    descriptions: dict[str, dict],
    learned_summary: dict,
    selection: pd.DataFrame,
    naive_summary: pd.DataFrame,
    config: dict,
    variant: str,
) -> None:
    """Write the graph-variant report."""

    single = summary[~summary["ensemble"]].sort_values("headline_mae")
    ensemble = summary[summary["ensemble"]].sort_values("headline_mae")

    control_row = single[single["arm"] == CONTROL_ARM]
    control_mae = (
        float(control_row["headline_mae"].iloc[0]) if len(control_row) else float("nan")
    )

    lines = [
        "# Graph representation — does any adjacency beat no adjacency?",
        "",
        f"Version: `{GRAPH_VERSION}`",
        "",
        "Four adjacencies through one GCN+GRU. Architecture, training loop, early",
        "stopping, optimiser, folds, masks, preprocessing, windowing, target",
        "parameterisation, loss and metrics are imported from",
        "`scripts/16.train_gcn_gru.py`. Only the adjacency changes.",
        "",
        "**The reference is `identity`, not `contiguity`.** Substituting the",
        "identity makes the graph convolution a per-node linear layer, so the",
        "`identity` arm *is* script 16's `gru_only` control. Contiguity already",
        "loses to it, so beating contiguity would establish nothing; an arm counts",
        "as an improvement only if it beats `identity`.",
        "",
        "## The graphs",
        "",
        "| Arm | Source | Edges | Mean degree | Symmetric | Off-diagonal mass |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    sources = {
        "identity": "`np.eye` — the control, no graph",
        "contiguity": "`A_norm`, queen contiguity from GADM polygons",
        "gaussian": "`A_gaussian_norm`, exp(-(d/50km)²) on centroid distance",
        "adaptive": "learned, softmax(relu(E1 @ E2ᵀ))",
    }

    for arm, description in descriptions.items():
        lines.append(
            f"| `{arm}` | {sources.get(arm, '')} | {description['edges']} | "
            f"{description['mean_degree']:.2f} | "
            f"{'yes' if description['symmetric'] else 'no'} | "
            f"{100 * description['off_diagonal_mass']:.0f}% |"
        )

    lines += [
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
        "",
        "| Arm | MAE | vs `identity` | RMSE | Peak MAE | 2017 MAE | COVID MAE | Seed sd |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in single.itertuples():
        delta = (
            f"{row.headline_mae - control_mae:+.2f}"
            if np.isfinite(control_mae) and row.arm != CONTROL_ARM
            else "— (control)"
        )
        lines.append(
            f"| `{row.arm}` | {row.headline_mae:.2f} | {delta} | "
            f"{row.headline_rmse:.2f} | {row.headline_peak_mae:.2f} | "
            f"{row.epidemic_2017_mae:.2f} | {row.covid_mae:.2f} | {row.seed_sd:.2f} |"
        )

    lines += [
        "",
        "## Results — seed-mean ensembles",
        "",
        "| Arm | MAE | RMSE | Peak MAE | 2017 MAE |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in ensemble.itertuples():
        lines.append(
            f"| `{row.arm}` | {row.headline_mae:.2f} | {row.headline_rmse:.2f} | "
            f"{row.headline_peak_mae:.2f} | {row.epidemic_2017_mae:.2f} |"
        )

    lines += ["", "### Naive baselines, same folds and masks", ""]
    lines += ["| Model | MAE | Peak MAE | 2017 MAE |", "| --- | --- | --- | --- |"]
    for row in naive_summary.itertuples():
        lines.append(
            f"| {row.model} | {row.headline_mae:.2f} | "
            f"{row.headline_peak_mae:.2f} | {row.epidemic_2017_mae:.2f} |"
        )

    if len(comparison):
        lines += [
            "",
            "## Does each graph beat the identity control?",
            "",
            "The question the brief asks, answered per arm. `Δ` is the headline MAE",
            "change against `identity`; negative is better. The paired t is over the",
            "seven headline folds. `Δ in sd` compares the gap to the pooled seed",
            "standard deviation — this project treats anything under 2 sd as not",
            "established.",
            "",
            "| Arm | Δ vs identity | Folds improved | t | p | Pooled seed sd | Δ in sd | Verdict |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]

        for row in comparison.itertuples():
            if row.headline_delta < 0 and row.delta_in_sd > 2 and row.p_value < 0.05:
                verdict = "**improves**"
            elif row.headline_delta < 0:
                verdict = "not established"
            else:
                verdict = "**worse**"

            lines.append(
                f"| `{row.arm}` | {row.headline_delta:+.2f} | "
                f"{row.folds_improved}/{row.folds_total} | {row.t_statistic:+.2f} | "
                f"{row.p_value:.3f} | {row.pooled_seed_sd:.2f} | "
                f"{row.delta_in_sd:.2f} | {verdict} |"
            )

        lines += [
            "",
            "### Where any movement comes from",
            "",
            "Four separate changes in this project have produced a headline",
            "movement that turned out to be the 2017 epidemic fold and nothing",
            "else (README §8, §8b, §8c, §8d). This splits each arm's change into",
            "fold 1 and the other six headline folds.",
            "",
            "| Arm | Fold 1 Δ | Δ over the other six folds |",
            "| --- | --- | --- |",
        ]
        for row in comparison.itertuples():
            lines.append(
                f"| `{row.arm}` | {row.fold1_delta:+.2f} | {row.ex_fold1_delta:+.3f} |"
            )

    if len(selection):
        lines += [
            "",
            "## The adaptive arm's graph learning rate",
            "",
            "The embeddings sit behind a relu and a softmax, so their gradient is",
            "far smaller than the GRU's — the asymmetry",
            "`scripts/16.parameter_groups` documents for the lag encoder. At a 1x",
            "multiplier the graph never leaves its uniform initialisation; at 10x",
            "it collapses to near one-hot rows. Neither is obviously right, and",
            "choosing by test score would be the mistake",
            "`docs/hyperparameter_tuning.md` exists to avoid — so the multiplier is",
            "**selected per fold on the validation split**.",
            "",
            "| Fold | " + " | ".join(f"{m:.0f}x" for m in config["graph_lr_grid"]) + " | Selected |",
            "| --- | " + " | ".join("---" for _ in config["graph_lr_grid"]) + " | --- |",
        ]

        for fold_id, group in selection.groupby("fold_id"):
            by_lr = group.set_index("graph_lr_multiplier")["val_mae"]
            cells = " | ".join(
                f"{by_lr.get(m, float('nan')):.2f}" for m in config["graph_lr_grid"]
            )
            lines.append(
                f"| {fold_id} | {cells} | **{group['selected'].iloc[0]:.0f}x** |"
            )

        lines.append("")
        lines.append("Validation MAE per candidate; the selected column is the row minimum.")

    if learned_summary:
        lines += [
            "",
            "## What did the adaptive graph learn?",
            "",
            "A learned adjacency that never left its uniform initialisation would",
            "reproduce the identity's answer for an uninteresting reason. These",
            "numbers distinguish that from a graph that found structure and still",
            "did not help.",
            "",
            "| Quantity | Value | Reading |",
            "| --- | --- | --- |",
            f"| Normalised row entropy | {learned_summary['normalised_entropy']:.3f} | "
            "1.0 = uniform, 0.0 = one-hot |",
            f"| Uniform weight (1/25) | {learned_summary['uniform_weight']:.4f} | "
            "what every entry would be if nothing was learned |",
            f"| Largest learned weight | {learned_summary['max_weight']:.4f} | "
            f"{learned_summary['max_weight'] / learned_summary['uniform_weight']:.1f}x uniform |",
            f"| Mass in each row's top 3 | {learned_summary['top3_mass']:.3f} | "
            "0.12 would be uniform |",
            f"| Correlation with contiguity | {learned_summary['contiguity_correlation']:+.3f} | "
            "did it rediscover the border graph? |",
        ]

    best = single.iloc[0]
    persistence = naive_summary[naive_summary["model"] == "persistence"].iloc[0]

    lines += [
        "",
        "## Verdict",
        "",
        f"Best arm: **`{best.arm}`** at headline MAE {best.headline_mae:.2f}. "
        f"The `identity` control is at {control_mae:.2f} and persistence at "
        f"{persistence.headline_mae:.2f}.",
        "",
    ]

    improving = (
        comparison[
            (comparison["headline_delta"] < 0)
            & (comparison["delta_in_sd"] > 2)
            & (comparison["p_value"] < 0.05)
        ]
        if len(comparison)
        else pd.DataFrame()
    )

    if len(improving):
        names = ", ".join(f"`{a}`" for a in improving["arm"])
        lines.append(
            f"**{names} beats the identity control by an established margin.** "
            "That answers the precondition the dual-graph work plan set for itself."
        )
    else:
        lines.append(
            "**No graph beats the identity control by an established margin.** "
            "Contiguity, a distance kernel and a graph learned end-to-end from the "
            "forecasting loss itself all fail to improve on having no graph at all. "
            "A fixed alternative failing could be the wrong fixed alternative; a "
            "learned one failing too is evidence about the data rather than about "
            "the choice of prior."
        )

    lines += [
        "",
        "## Per fold, MAE (single seed models, averaged over seeds)",
        "",
    ]

    pivot = metrics[~metrics["ensemble"]].pivot_table(
        index=["fold_id", "test_year"], columns="arm", values="mae"
    ).reset_index()

    arm_columns = [c for c in pivot.columns if c not in ("fold_id", "test_year")]
    lines.append("| Fold | Year | " + " | ".join(f"`{c}`" for c in arm_columns) + " | Note |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in arm_columns) + " | --- |")

    for row in pivot.itertuples(index=False):
        fold_id, test_year = row[0], row[1]
        note = (
            "COVID" if test_year in (2020, 2021)
            else ("epidemic" if test_year == 2017 else "")
        )
        values = " | ".join(f"{v:.2f}" for v in row[2:])
        lines.append(f"| {fold_id} | {test_year} | {values} | {note} |")

    lines += [
        "",
        "## Output files",
        "",
        f"- `{METRICS_PATH.relative_to(PROJECT_DIR).as_posix()}`",
        f"- `{LEARNED_PATH.relative_to(PROJECT_DIR).as_posix()}` — the learned adjacency, per fold and seed",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def naive_summary_frame() -> pd.DataFrame:
    """Read the committed naive baseline metrics and summarise them like the arms."""

    metrics = pd.read_csv(RESULTS_DIR / "naive_baseline_metrics.csv")

    records = []
    for model, group in metrics.groupby("model"):
        headline = group[group["headline"]]
        records.append(
            {
                "model": model,
                "headline_mae": headline["mae"].mean(),
                "headline_peak_mae": headline["peak_mae"].mean(),
                "epidemic_2017_mae": group.loc[
                    group["test_year"] == 2017, "mae"
                ].mean(),
            }
        )

    return pd.DataFrame(records).sort_values("headline_mae").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse the training knobs. Defaults reproduce the committed report."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--variant", default="v1", choices=["v0", "v1"])
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--graph-lr-grid",
        nargs="+",
        type=float,
        default=GRAPH_LR_GRID,
        help="adaptive-arm graph LR multipliers, selected per fold on validation",
    )

    for key, value in DEFAULTS.items():
        parser.add_argument(
            f"--{key.replace('_', '-')}", type=type(value), default=value
        )

    return parser.parse_args()


def main() -> int:
    """Train every graph arm on every fold and write the report."""

    arguments = parse_arguments()
    load_modules()

    if not FOLDS_PATH.exists() or not ADJACENCY_PATH.exists():
        print("Missing folds.json or adjacency.npz. Run scripts 13 and 14 first.")
        return 1

    config = {key: getattr(arguments, key) for key in DEFAULTS}
    config["graph_lr_grid"] = list(arguments.graph_lr_grid)

    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))["folds"]
    if arguments.folds:
        folds = [f for f in folds if f["fold_id"] in arguments.folds]

    nodes = pd.read_csv(PROCESSED_DIR / "nodes.csv").sort_values("node_id")
    names = nodes["canonical_name"].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device:   {device}")
    print(f"Variant:  {arguments.variant}")
    print(f"Folds:    {[f['fold_id'] for f in folds]}")
    print(f"Arms:     {arguments.arms}")
    print(f"Seeds:    {config['seeds']}")
    print(f"Control:  {CONTROL_ARM} (script 16's gru_only)")
    if "adaptive" in arguments.arms:
        print(
            f"Graph LR: {config['graph_lr_grid']} "
            "(selected per fold on the validation split)"
        )
    print()

    started = time.perf_counter()
    metrics, learned, selection = run_experiment(
        arguments.arms, arguments.variant, folds, config, device
    )
    elapsed = (time.perf_counter() - started) / 60

    print(f"\nTrained in {elapsed:.1f} min\n")

    summary = summarise(metrics)
    comparison = compare_to_control(metrics)
    naive_frame = naive_summary_frame()

    adjacencies = load_adjacencies(len(names))
    descriptions = {
        arm: describe_adjacency(adjacencies[arm])
        for arm in arguments.arms
        if arm != "adaptive"
    }

    learned_summary = {}
    if len(learned):
        with np.load(ADJACENCY_PATH, allow_pickle=True) as data:
            contiguity = data["A_norm"].astype(np.float32)

        learned_summary = summarise_learned(learned, contiguity, names)

        # Describe the mean learned graph alongside the fixed ones.
        n = len(names)
        index = {name: i for i, name in enumerate(names)}
        mean = np.zeros((n, n), dtype=np.float32)
        for (source, target), weight in (
            learned.groupby(["source", "target"])["weight"].mean().items()
        ):
            mean[index[source], index[target]] = weight
        descriptions["adaptive"] = describe_adjacency(mean)

    single = summary[~summary["ensemble"]]
    print("single seed models")
    print(
        single[
            ["arm", "headline_mae", "headline_peak_mae", "epidemic_2017_mae", "seed_sd"]
        ].to_string(index=False, float_format=lambda v: f"{v:8.2f}")
    )

    if len(comparison):
        print("\nvs the identity control")
        print(
            comparison[
                [
                    "arm",
                    "headline_delta",
                    "folds_improved",
                    "t_statistic",
                    "p_value",
                    "delta_in_sd",
                    "ex_fold1_delta",
                ]
            ].to_string(index=False, float_format=lambda v: f"{v:8.3f}")
        )

    if len(selection):
        print("\nadaptive graph LR, selected per fold on validation")
        chosen = selection.groupby("fold_id")["selected"].first()
        print("  " + "  ".join(f"fold {f}: {v:.0f}x" for f, v in chosen.items()))

    if learned_summary:
        print("\nthe learned graph")
        print(
            f"  normalised entropy      {learned_summary['normalised_entropy']:.3f} "
            "(1.0 = uniform)"
        )
        print(
            f"  largest weight          {learned_summary['max_weight']:.4f} "
            f"({learned_summary['max_weight'] / learned_summary['uniform_weight']:.1f}x uniform)"
        )
        print(
            f"  correlation w contiguity {learned_summary['contiguity_correlation']:+.3f}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    if len(learned):
        learned.to_csv(LEARNED_PATH, index=False)

    write_report(
        summary,
        metrics,
        comparison,
        descriptions,
        learned_summary,
        selection,
        naive_frame,
        config,
        arguments.variant,
    )

    print(f"\nWrote {REPORT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {METRICS_PATH.relative_to(PROJECT_DIR)}")
    if len(learned):
        print(f"Wrote {LEARNED_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

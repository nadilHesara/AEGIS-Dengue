"""
Optimiser and learning-rate schedule for the GCN+GRU, on the frozen fold protocol.

The baseline uses Adam at a fixed learning rate for the whole run. Two things
about that were never tested, and this script tests them:

    AdamW       Adam applies weight decay by adding `wd * w` to the gradient,
                which Adam's own per-parameter normalisation then rescales -- so
                the effective decay is not `wd`, it is `wd` divided by a running
                estimate of the gradient magnitude. Parameters with small
                gradients get decayed far harder than parameters with large ones.
                AdamW decouples the two, applying the decay directly to the
                weight. At the baseline's `weight_decay=1e-4` this is a real
                difference in what the regulariser does, not a reparameterisation.

    scheduling  A fixed rate has to be small enough for the end of training and
                is therefore too small for the start, or large enough for the
                start and too large for the end. `ReduceLROnPlateau` removes the
                choice: train at the base rate until validation stops improving,
                then halve it. This matters more here than usual because the
                hyperparameter search (`docs/hyperparameter_tuning.md`) found
                `learning_rate` is one of only two axes with any measurable
                signal, and that the baseline's 3e-3 is the *worst* of the four
                values tried -- so there is a documented reason to think the
                baseline trains at too high a rate.

Three arms, changing nothing but the optimiser and the schedule:

    adam                Adam, fixed rate                         the control
    adamw               AdamW, fixed rate
    adamw_scheduled     AdamW + ReduceLROnPlateau on validation MAE

Everything else -- architecture, graph, folds, masks, preprocessing, windowing,
the anchored residual target, the masked-MSE objective, gradient clipping, early
stopping and the metric -- is **imported** from `scripts/16.train_gcn_gru.py`
through the `make_optimiser` / `make_scheduler` hooks that script now exposes.
The loop is not copied. The `adam` arm is script 16's own optimiser passed back
in through the hook, so it must reproduce script 16's committed MAE; if it does
not, the harness is not neutral and nothing else here is comparable.

On the two patiences. Early stopping and `ReduceLROnPlateau` both watch the
validation loss, so their patiences interact. Early stopping keeps the
baseline's 15 -- changing it would break comparability with every committed
result -- and the scheduler gets 5, so it can fire roughly three times before
early stopping can trigger and a reduced rate gets a real chance to show a
benefit. The scheduler is stepped *after* early stopping has seen the epoch, so
adding it cannot change which epoch is selected as best, only what the optimiser
does next.

What "same folds and seeds" means here: every arm trains on the same nine folds
with the same seed range, and within a fold each arm sees identical data,
identical batch order (the shuffle is seeded per run, not per arm) and an
identically initialised model, because `train_one` reseeds from `seed` before
building anything. The arms differ in the optimiser and nothing else.

Usage:

    python scripts/25.train_optimisers.py                      3 arms, 9 folds, 3 seeds
    python scripts/25.train_optimisers.py --folds 8 --seeds 1  smoke run
    python scripts/25.train_optimisers.py --learning-rate 3e-4 start from the tuned rate

Outputs:
    results/models/optimiser_report.md
    results/models/optimiser_metrics.csv
    results/models/optimiser_lr_traces.csv
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

REPORT_PATH = RESULTS_DIR / "optimiser_report.md"
METRICS_PATH = RESULTS_DIR / "optimiser_metrics.csv"
LR_TRACES_PATH = RESULTS_DIR / "optimiser_lr_traces.csv"

OPTIMISER_VERSION = "optimiser-v1"

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
    # ReduceLROnPlateau. Half the rate after 5 epochs without a validation
    # improvement; never go below 1e-6, where training has effectively stopped
    # and the epochs are better spent letting early stopping fire.
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "scheduler_min_lr": 1e-6,
}

# Arm -> (optimiser name, whether a plateau scheduler is attached).
ARMS = {
    "adam": ("adam", False),
    "adamw": ("adamw", False),
    "adamw_scheduled": ("adamw", True),
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
# Optimisers and schedules
# ---------------------------------------------------------------------------

def make_optimiser(kind: str):
    """Return a `(model, config) -> Optimizer` factory for one arm.

    Both branches use `scripts/16.parameter_groups`, so the lag-encoder rate
    split -- which matters for models that have an encoder and is a no-op for
    those that do not -- behaves identically across arms. The only difference is
    the optimiser class.
    """

    def build(model: nn.Module, config: dict) -> torch.optim.Optimizer:
        groups = baseline_module.parameter_groups(model, config)

        if kind == "adam":
            return torch.optim.Adam(
                groups,
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )

        if kind == "adamw":
            return torch.optim.AdamW(
                groups,
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )

        raise ValueError(f"Unknown optimiser {kind!r}.")

    return build


def make_scheduler(optimiser: torch.optim.Optimizer, config: dict):
    """Attach ReduceLROnPlateau, watching the validation loss.

    `mode="min"` because the monitored quantity is the validation loss, which is
    the same quantity early stopping minimises. The scheduler therefore never
    disagrees with early stopping about what "better" means -- it just reacts
    sooner, at `scheduler_patience` rather than `patience`.
    """

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="min",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
        min_lr=config["scheduler_min_lr"],
    )


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

    adjacency = np.load(ADJACENCY_PATH, allow_pickle=True)["A_norm"].astype(np.float32)
    tensors = folds_module.load_tensors(variant)

    rows: list[dict] = []
    trace_rows: list[dict] = []

    for fold in folds:
        arrays = baseline_module.build_fold_arrays(
            tensors, months, fold, config["lookback"], config["horizon"]
        )

        fit_mask = tensors["period_id"] <= fold["fit_end_period"]
        thresholds = naive.peak_thresholds(tensors["y"], tensors["y_mask"], fit_mask)

        target = arrays["test"]["y"]
        mask = arrays["test"]["mask"].astype(np.int8)

        for arm in arms:
            kind, scheduled = ARMS[arm]

            started = time.perf_counter()
            seed_predictions = []
            seed_info = []

            for seed in range(config["seeds"]):
                model, info = baseline_module.train_one(
                    arrays,
                    adjacency,
                    config,
                    seed,
                    device,
                    make_optimiser=make_optimiser(kind),
                    make_scheduler=make_scheduler if scheduled else None,
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
                        "optimiser": kind,
                        "scheduled": scheduled,
                        "variant": variant,
                        "fold_id": fold["fold_id"],
                        "test_year": fold["test_year"],
                        "covers_covid": fold["covers_covid"],
                        "headline": fold["headline"],
                        "seed": seed,
                        "best_epoch": info["best_epoch"],
                        "epochs_run": info["epochs_run"],
                        "final_lr": info["final_lr"],
                        "ensemble": False,
                        **scores,
                    }
                )

                # The LR trace is the evidence the schedule did something. An
                # arm that reports a scheduler but never leaves its base rate is
                # an unscheduled arm with extra bookkeeping.
                for epoch, rate in enumerate(info["learning_rates"], start=1):
                    trace_rows.append(
                        {
                            "arm": arm,
                            "fold_id": fold["fold_id"],
                            "seed": seed,
                            "epoch": epoch,
                            "learning_rate": rate,
                        }
                    )

            mean_prediction = np.mean(seed_predictions, axis=0)
            ensemble_scores = naive.evaluate(mean_prediction, target, mask, thresholds)
            rows.append(
                {
                    "arm": arm,
                    "optimiser": kind,
                    "scheduled": scheduled,
                    "variant": variant,
                    "fold_id": fold["fold_id"],
                    "test_year": fold["test_year"],
                    "covers_covid": fold["covers_covid"],
                    "headline": fold["headline"],
                    "seed": -1,
                    "best_epoch": float(np.mean([i["best_epoch"] for i in seed_info])),
                    "epochs_run": float(np.mean([i["epochs_run"] for i in seed_info])),
                    "final_lr": float(
                        np.mean([i["final_lr"] for i in seed_info])
                    ),
                    "ensemble": True,
                    **ensemble_scores,
                }
            )

            elapsed = time.perf_counter() - started
            seed_mae = [row["mae"] for row in rows[-config["seeds"] - 1 : -1]]

            reductions = ""
            if scheduled:
                drops = [
                    sum(
                        1
                        for a, b in zip(i["learning_rates"], i["learning_rates"][1:])
                        if b < a
                    )
                    for i in seed_info
                ]
                reductions = f"  drops {np.mean(drops):3.1f}"

            print(
                f"  fold {fold['fold_id']} ({fold['test_year']}) {arm:<16} "
                f"MAE {np.mean(seed_mae):7.2f} +/- {np.std(seed_mae):5.2f}  "
                f"ens {ensemble_scores['mae']:7.2f}  "
                f"epochs {np.mean([i['best_epoch'] for i in seed_info]):5.1f}"
                f"{reductions}  {elapsed:5.1f}s",
                flush=True,
            )

    return pd.DataFrame(rows), pd.DataFrame(trace_rows)


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
                    "ensemble": ensemble,
                    "headline_mae": headline["mae"].mean(),
                    "headline_rmse": headline["rmse"].mean(),
                    "headline_peak_mae": headline["peak_mae"].mean(),
                    "epidemic_2017_mae": epidemic["mae"].mean(),
                    "epidemic_2017_peak_mae": epidemic["peak_mae"].mean(),
                    "covid_mae": covid["mae"].mean(),
                    "mean_best_epoch": headline["best_epoch"].mean(),
                    "seed_sd": spread,
                }
            )

    return (
        pd.DataFrame(records)
        .sort_values(["ensemble", "headline_mae"])
        .reset_index(drop=True)
    )


def fold_decomposition(metrics: pd.DataFrame, control: str = "adam") -> pd.DataFrame:
    """Split each arm's headline change into fold 1 and everything else.

    Every accuracy result in this project so far has turned out to be the 2017
    epidemic fold and nothing else (README §8, §8b, §8c). Printing the
    decomposition next to the headline number is the cheapest way to stop that
    being mistaken for a general improvement a fourth time.
    """

    single = metrics[~metrics["ensemble"]]
    by_fold = (
        single.groupby(["arm", "fold_id", "headline"])["mae"].mean().reset_index()
    )
    headline = by_fold[by_fold["headline"]]

    pivot = headline.pivot(index="fold_id", columns="arm", values="mae")
    if control not in pivot.columns:
        return pd.DataFrame()

    records = []
    others = [f for f in pivot.index if f != 1]

    for arm in pivot.columns:
        if arm == control:
            continue

        delta = pivot[arm] - pivot[control]
        total = delta.mean()
        fold_one = delta.loc[1] / len(pivot.index) if 1 in pivot.index else float("nan")

        records.append(
            {
                "arm": arm,
                "headline_delta": total,
                "fold1_share": 100 * fold_one / total if total else float("nan"),
                "ex_fold1_delta": delta.loc[others].mean(),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    traces: pd.DataFrame,
    decomposition: pd.DataFrame,
    naive_summary: pd.DataFrame,
    config: dict,
    variant: str,
) -> None:
    """Write the optimiser report."""

    single = summary[~summary["ensemble"]].sort_values("headline_mae")
    ensemble = summary[summary["ensemble"]].sort_values("headline_mae")

    control_row = single[single["arm"] == "adam"]
    control_mae = (
        float(control_row["headline_mae"].iloc[0]) if len(control_row) else float("nan")
    )
    control_sd = (
        float(control_row["seed_sd"].iloc[0]) if len(control_row) else float("nan")
    )

    lines = [
        "# Optimiser and learning-rate schedule",
        "",
        f"Version: `{OPTIMISER_VERSION}`",
        "",
        "Three arms that change **only** the optimiser and the learning-rate",
        "schedule. Architecture, graph, folds, masks, preprocessing, windowing,",
        "target parameterisation, loss, gradient clipping, early stopping and the",
        "metric are imported from `scripts/16.train_gcn_gru.py` through its",
        "`make_optimiser` / `make_scheduler` hooks. The loop is not copied.",
        "",
        "The `adam` arm is script 16's own optimiser passed back through the hook.",
        "It is the control: if it does not reproduce script 16's committed MAE,",
        "the harness is not neutral and nothing below is comparable.",
        "",
        "## Arms",
        "",
        "| Arm | Optimiser | Schedule |",
        "| --- | --- | --- |",
        "| `adam` | Adam | fixed rate — the control |",
        "| `adamw` | AdamW | fixed rate |",
        "| `adamw_scheduled` | AdamW | ReduceLROnPlateau on validation loss |",
        "",
        "**Why AdamW.** Adam adds `wd * w` to the gradient, which its own",
        "per-parameter normalisation then rescales, so the effective decay is",
        "`wd` divided by a running estimate of gradient magnitude — parameters",
        "with small gradients get decayed far harder than those with large ones.",
        "AdamW applies the decay directly to the weight. At the baseline's",
        f"`weight_decay={config['weight_decay']}` that is a real difference in what the",
        "regulariser does.",
        "",
        "**Why a schedule.** A fixed rate is either too small for the start of",
        "training or too large for the end. `docs/hyperparameter_tuning.md` found",
        "`learning_rate` is one of only two axes with measurable signal, and that",
        f"the baseline's {config['learning_rate']:.0e} is the *worst* of the four values tried —",
        "a documented reason to think the baseline trains too hot.",
        "",
        "**The two patiences.** Early stopping and the scheduler both watch the",
        f"validation loss. Early stopping keeps the baseline's {config['patience']};",
        f"the scheduler gets {config['scheduler_patience']}, so it can fire roughly",
        f"{config['patience'] // config['scheduler_patience']} times before early stopping can trigger.",
        "The scheduler is stepped **after** early stopping has seen the epoch, so",
        "it cannot change which epoch is selected as best — only what the",
        "optimiser does next.",
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
        "| Arm | MAE | vs `adam` | RMSE | Peak MAE | 2017 MAE | COVID MAE | Mean best epoch | Seed sd |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in single.itertuples():
        delta = (
            f"{row.headline_mae - control_mae:+.2f}"
            if np.isfinite(control_mae) and row.arm != "adam"
            else "—"
        )
        lines.append(
            f"| `{row.arm}` | {row.headline_mae:.2f} | {delta} | "
            f"{row.headline_rmse:.2f} | {row.headline_peak_mae:.2f} | "
            f"{row.epidemic_2017_mae:.2f} | {row.covid_mae:.2f} | "
            f"{row.mean_best_epoch:.1f} | {row.seed_sd:.2f} |"
        )

    lines += [
        "",
        "## Results — seed-mean ensembles",
        "",
        "| Arm | MAE | RMSE | Peak MAE | 2017 MAE | COVID MAE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in ensemble.itertuples():
        lines.append(
            f"| `{row.arm}` | {row.headline_mae:.2f} | {row.headline_rmse:.2f} | "
            f"{row.headline_peak_mae:.2f} | {row.epidemic_2017_mae:.2f} | "
            f"{row.covid_mae:.2f} |"
        )

    lines += ["", "### Naive baselines, on the same folds and masks", ""]
    lines += [
        "| Model | MAE | RMSE | Peak MAE | 2017 MAE |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in naive_summary.itertuples():
        lines.append(
            f"| {row.model} | {row.headline_mae:.2f} | {row.headline_rmse:.2f} | "
            f"{row.headline_peak_mae:.2f} | {row.epidemic_2017_mae:.2f} |"
        )

    # The fold decomposition, printed before the verdict rather than after.
    if len(decomposition):
        lines += [
            "",
            "## Where any change actually comes from",
            "",
            "Every accuracy result in this project so far has been the 2017",
            "epidemic fold and nothing else (README §8, §8b, §8c). This table",
            "splits each arm's headline movement into fold 1 and the other six",
            "headline folds, so a single-fold artefact cannot be read as a",
            "general improvement.",
            "",
            "| Arm | Headline Δ vs `adam` | Fold 1 share | Δ over the other six folds |",
            "| --- | --- | --- | --- |",
        ]
        for row in decomposition.itertuples():
            lines.append(
                f"| `{row.arm}` | {row.headline_delta:+.3f} | "
                f"{row.fold1_share:.0f}% | {row.ex_fold1_delta:+.3f} |"
            )

    # Evidence the schedule moved.
    if len(traces):
        lines += [
            "",
            "## Did the schedule actually fire?",
            "",
            "An arm that reports a scheduler but never leaves its base rate is an",
            "unscheduled arm with extra bookkeeping. Per-run learning-rate traces",
            "are in `optimiser_lr_traces.csv`; the summary:",
            "",
            "| Arm | Runs | Mean reductions per run | Base LR | Mean final LR |",
            "| --- | --- | --- | --- | --- |",
        ]

        for arm, group in traces.groupby("arm"):
            per_run = group.groupby(["fold_id", "seed"])["learning_rate"]
            drops = per_run.apply(
                lambda s: int((s.diff().fillna(0) < 0).sum())
            )
            finals = per_run.last()
            lines.append(
                f"| `{arm}` | {len(drops)} | {drops.mean():.1f} | "
                f"{config['learning_rate']:.0e} | {finals.mean():.2e} |"
            )

    best = single.iloc[0]
    persistence = naive_summary[naive_summary["model"] == "persistence"].iloc[0]

    lines += [
        "",
        "## Verdict",
        "",
        f"Best arm: **`{best.arm}`** at headline MAE {best.headline_mae:.2f}, "
        f"against the `adam` control at {control_mae:.2f} "
        f"({best.headline_mae - control_mae:+.2f}) and persistence at "
        f"{persistence.headline_mae:.2f}.",
        "",
    ]

    # State the seed-sd test explicitly rather than leaving it to the reader.
    gap = abs(best.headline_mae - control_mae)
    pooled_sd = np.nanmean([control_sd, best.seed_sd])
    if np.isfinite(pooled_sd) and pooled_sd > 0:
        lines.append(
            f"That gap is {gap:.2f} MAE against a pooled seed sd of "
            f"{pooled_sd:.2f} — "
            + (
                f"**{gap / pooled_sd:.1f} seed-sd, which clears this project's usual bar.**"
                if gap > 2 * pooled_sd
                else f"**{gap / pooled_sd:.1f} seed-sd, which does not clear two seed-sd "
                "and is therefore not established by this run.**"
            )
        )
        lines.append("")

    lines += [
        "**Read the seed sd and the fold decomposition before believing any gap.**",
        "A difference smaller than the seed sd is not established by this run, and",
        "a difference that lives entirely in fold 1 is an epidemic-conditions",
        "result, not a general one.",
        "",
        "## Per fold, MAE (single seed models, averaged over seeds)",
        "",
    ]

    single_metrics = metrics[~metrics["ensemble"]]
    pivot = single_metrics.pivot_table(
        index=["fold_id", "test_year"], columns="arm", values="mae"
    ).reset_index()

    arm_columns = [c for c in pivot.columns if c not in ("fold_id", "test_year")]
    lines.append(
        "| Fold | Year | " + " | ".join(f"`{c}`" for c in arm_columns) + " | Note |"
    )
    lines.append("| --- | --- | " + " | ".join("---" for _ in arm_columns) + " | --- |")

    for row in pivot.itertuples(index=False):
        fold_id, test_year = row[0], row[1]
        note = (
            "COVID"
            if test_year in (2020, 2021)
            else ("epidemic" if test_year == 2017 else "")
        )
        values = " | ".join(f"{v:.2f}" for v in row[2:])
        lines.append(f"| {fold_id} | {test_year} | {values} | {note} |")

    lines += [
        "",
        "## Output files",
        "",
        f"- `{METRICS_PATH.relative_to(PROJECT_DIR).as_posix()}`",
        f"- `{LR_TRACES_PATH.relative_to(PROJECT_DIR).as_posix()}`",
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
                "headline_rmse": headline["rmse"].mean(),
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

    for key, value in DEFAULTS.items():
        parser.add_argument(
            f"--{key.replace('_', '-')}", type=type(value), default=value
        )

    return parser.parse_args()


def main() -> int:
    """Train every arm on every fold and write the report."""

    arguments = parse_arguments()
    load_modules()

    if not FOLDS_PATH.exists() or not ADJACENCY_PATH.exists():
        print("Missing folds.json or adjacency.npz. Run scripts 13 and 14 first.")
        return 1

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
    print(f"Base LR:  {config['learning_rate']:.0e}")
    print(
        f"Schedule: ReduceLROnPlateau factor {config['scheduler_factor']}, "
        f"patience {config['scheduler_patience']}, min {config['scheduler_min_lr']:.0e}"
    )
    print()

    started = time.perf_counter()
    metrics, traces = run_experiment(
        arguments.arms, arguments.variant, folds, config, device
    )
    elapsed = (time.perf_counter() - started) / 60

    print(f"\nTrained in {elapsed:.1f} min\n")

    summary = summarise(metrics)
    decomposition = fold_decomposition(metrics)
    naive_frame = naive_summary_frame()

    single = summary[~summary["ensemble"]]
    print("single seed models")
    print(
        single[
            [
                "arm",
                "headline_mae",
                "headline_peak_mae",
                "epidemic_2017_mae",
                "mean_best_epoch",
                "seed_sd",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:8.2f}")
    )

    print("\nseed-mean ensembles")
    print(
        summary[summary["ensemble"]][
            ["arm", "headline_mae", "headline_peak_mae", "epidemic_2017_mae"]
        ].to_string(index=False, float_format=lambda v: f"{v:8.2f}")
    )

    if len(decomposition):
        print("\nwhere the change comes from (vs adam)")
        print(decomposition.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    if len(traces):
        print("\nlearning-rate schedule")
        for arm, group in traces.groupby("arm"):
            per_run = group.groupby(["fold_id", "seed"])["learning_rate"]
            drops = per_run.apply(lambda s: int((s.diff().fillna(0) < 0).sum()))
            print(
                f"  {arm:<16} {len(drops)} runs, "
                f"{drops.mean():.1f} reductions/run, "
                f"final LR {per_run.last().mean():.2e}"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    traces.to_csv(LR_TRACES_PATH, index=False)
    write_report(
        summary, metrics, traces, decomposition, naive_frame, config, arguments.variant
    )

    print(f"\nWrote {REPORT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {METRICS_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {LR_TRACES_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

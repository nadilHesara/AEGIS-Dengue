"""
Figures for the baseline's training dynamics and hyperparameter sensitivity.

Trains nothing. Everything is read back from what `scripts/22` wrote, so this
runs in seconds and always draws the committed numbers rather than a fresh
random seed -- the same arrangement `scripts/19` uses.

Four figures, each answering one question:

    1. training curves     is the early stop premature, or has it converged?
    2. stopping point      how much training does each arm actually get?
    3. sensitivity         which knobs is the error steep in, and which flat?
    4. schedule effect     does fixing the schedule change the graph verdict?

Figure 1 is the diagnosis and figure 4 is the consequence. The middle two are
what turns "the graph hurts" into a claim that can be checked: if `gcn_gru` and
`gru_only` are selected at very different points on their curves, the committed
comparison was never between two converged models.

Run:
    python scripts/23.optimisation_figures.py

Outputs:
    results/figures/opt1_training_curves.png
    results/figures/opt2_stopping_point.png
    results/figures/opt3_parameter_sensitivity.png
    results/figures/opt4_schedule_effect.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parent.parent

METRICS_PATH = PROJECT_DIR / "results" / "models" / "optimisation_metrics.csv"
CURVES_PATH = PROJECT_DIR / "results" / "models" / "optimisation_curves.csv"
NAIVE_PATH = PROJECT_DIR / "results" / "models" / "naive_baseline_metrics.csv"
FIGURE_DIR = PROJECT_DIR / "results" / "figures"

# Categorical slots 1-2, in fixed order, from the same validated reference
# palette scripts/19 draws on. Validated for this pair on the adjacent
# pairlist: CVD dE 24.7 (protan), normal-vision dE 33.6, both well clear of the
# floors, and both above 3:1 against the surface -- so no relief label is
# required here, unlike scripts/19's four-colour set.
MODEL_COLOURS = {"gcn_gru": "#2a78d6", "gru_only": "#eb6834"}
MODEL_LABELS = {"gcn_gru": "GCN+GRU", "gru_only": "GRU only (identity A)"}

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2df"
SURFACE = "#fcfcfb"
REFERENCE = "#52514e"

# The knobs scripts/22 sweeps, with the axis treatment each one needs and the
# committed baseline value, which every panel marks so the sweep reads as
# "what moving this knob costs" rather than as an unanchored curve.
KNOBS = {
    "learning_rate": ("learning rate", "log", 3e-3),
    "batch_size": ("batch size", "log", 64),
    "hidden": ("hidden width", "log", 32),
    "gcn_layers": ("graph layers", "linear", 2),
    "dropout": ("dropout", "linear", 0.2),
    "patience": ("patience", "linear", 15),
}


def style() -> None:
    """A recessive default: thin marks, quiet axes, no chart junk."""

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.size": 9,
            "figure.dpi": 140,
        }
    )


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the artifacts, with a usable error if one is missing."""

    missing = [
        path.relative_to(PROJECT_DIR).as_posix()
        for path in (METRICS_PATH, CURVES_PATH, NAIVE_PATH)
        if not path.exists()
    ]
    if missing:
        raise SystemExit(
            "Missing "
            + ", ".join(missing)
            + ".\nRun scripts/15.evaluate_naive_baselines.py and "
            "scripts/22.optimise_baseline.py first."
        )

    return (
        pd.read_csv(METRICS_PATH),
        pd.read_csv(CURVES_PATH),
        pd.read_csv(NAIVE_PATH),
    )


def persistence_mae(naive: pd.DataFrame, fold_ids) -> float:
    """Mean persistence MAE over the folds a figure covers."""

    selected = naive[
        (naive["model"] == "persistence") & (naive["fold_id"].isin(fold_ids))
    ]

    return float(selected["mae"].mean())


# ---------------------------------------------------------------------------
# Figure 1 -- the diagnosis
# ---------------------------------------------------------------------------

def figure_training_curves(curves: pd.DataFrame) -> Path:
    """Validation loss against epoch, with the selected epoch marked.

    One panel per fold. The marker is where early stopping picked the model
    that gets scored; the question the panel answers is whether the curve was
    still descending at that point.
    """

    committed = curves[curves["label"] == "committed"]
    if committed.empty:
        committed = curves[curves["label"] == curves["label"].iloc[0]]

    fold_ids = sorted(committed["fold_id"].unique())

    figure, axes = plt.subplots(
        1, len(fold_ids), figsize=(4.4 * len(fold_ids), 3.6), squeeze=False
    )

    for axis, fold_id in zip(axes[0], fold_ids):
        panel = committed[committed["fold_id"] == fold_id]

        for model, group in panel.groupby("model"):
            group = group.sort_values("epoch")
            colour = MODEL_COLOURS[model]

            axis.plot(
                group["epoch"],
                group["val_loss"],
                color=colour,
                linewidth=1.6,
                label=MODEL_LABELS[model],
            )

            best = int(group["best_epoch"].iloc[0])
            at_best = group[group["epoch"] == best]["val_loss"]
            if len(at_best):
                axis.plot(
                    [best],
                    [at_best.iloc[0]],
                    marker="o",
                    markersize=7,
                    color=colour,
                    markeredgecolor=SURFACE,
                    markeredgewidth=2,
                    zorder=5,
                )
                axis.annotate(
                    f"epoch {best}",
                    xy=(best, at_best.iloc[0]),
                    xytext=(6, 8),
                    textcoords="offset points",
                    fontsize=8,
                    color=INK_SOFT,
                )

        axis.set_title(f"Fold {fold_id}")
        axis.set_xlabel("epoch")
        axis.set_ylabel("validation loss (masked MSE, log residual)")

    axes[0][0].legend(loc="upper right")

    figure.suptitle(
        "Where early stopping selects each model",
        fontsize=12,
        fontweight="bold",
        color=INK,
        y=1.0,
    )
    figure.text(
        0.5,
        -0.03,
        "Marker = the epoch whose weights are restored and scored. "
        "A marker on a still-descending curve is a premature stop.",
        ha="center",
        fontsize=8,
        color=INK_SOFT,
    )

    figure.tight_layout()
    path = FIGURE_DIR / "opt1_training_curves.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

    return path


# ---------------------------------------------------------------------------
# Figure 2 -- how much training each arm gets
# ---------------------------------------------------------------------------

def figure_stopping_point(metrics: pd.DataFrame) -> Path:
    """Optimiser steps behind the selected model, by schedule arm.

    Steps rather than epochs, because an epoch is a different amount of
    training at different batch sizes and on different folds -- the whole point
    of the batch-size arm is that it changes steps per epoch.
    """

    schedule = metrics[metrics["kind"] == "schedule"]
    if schedule.empty:
        return None

    grouped = (
        schedule.groupby(["label", "model"])["steps_to_best"].mean().reset_index()
    )
    labels = [
        label for label in schedule["label"].drop_duplicates() if label in
        set(grouped["label"])
    ]

    figure, axis = plt.subplots(figsize=(7.6, 3.8))

    width = 0.38
    positions = np.arange(len(labels))

    for offset, model in zip((-width / 2, width / 2), ("gcn_gru", "gru_only")):
        values = [
            grouped[(grouped["label"] == label) & (grouped["model"] == model)][
                "steps_to_best"
            ].mean()
            for label in labels
        ]
        bars = axis.bar(
            positions + offset,
            values,
            width,
            color=MODEL_COLOURS[model],
            label=MODEL_LABELS[model],
            zorder=3,
        )
        axis.bar_label(
            bars, fmt="%.0f", fontsize=7, color=INK_SOFT, padding=2
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("optimiser steps behind the selected model")
    axis.set_xlabel("schedule arm")
    axis.set_title("How much training the scored model actually received")
    axis.legend(loc="upper left")

    figure.text(
        0.5,
        -0.06,
        "Mean over folds and seeds. A large gap between the two bars means the "
        "arms are not being compared at the same point in training.",
        ha="center",
        fontsize=8,
        color=INK_SOFT,
    )

    figure.tight_layout()
    path = FIGURE_DIR / "opt2_stopping_point.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

    return path


# ---------------------------------------------------------------------------
# Figure 3 -- sensitivity
# ---------------------------------------------------------------------------

def figure_sensitivity(metrics: pd.DataFrame, naive: pd.DataFrame) -> Path:
    """Test MAE against each swept knob, one panel per knob.

    Small multiples rather than one crowded axis: the knobs share a y-scale
    meaning but not an x-scale, and overlaying them would need a second axis,
    which is never the right answer.
    """

    sweeps = metrics[metrics["kind"].str.startswith("sweep:")]
    if sweeps.empty:
        return None

    present = [knob for knob in KNOBS if f"sweep:{knob}" in set(sweeps["kind"])]
    if not present:
        return None

    reference = persistence_mae(naive, sweeps["fold_id"].unique())

    columns = 3
    rows = int(np.ceil(len(present) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.0 * columns, 3.2 * rows), squeeze=False
    )

    for index, knob in enumerate(present):
        axis = axes[index // columns][index % columns]
        panel = sweeps[sweeps["kind"] == f"sweep:{knob}"]
        title, scale, committed_value = KNOBS[knob]

        for model, group in panel.groupby("model"):
            curve = group.groupby(knob)["mae"].agg(["mean", "std"]).reset_index()
            curve = curve.sort_values(knob)

            axis.plot(
                curve[knob],
                curve["mean"],
                color=MODEL_COLOURS[model],
                linewidth=1.8,
                marker="o",
                markersize=5,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=MODEL_LABELS[model],
            )
            # Seed spread as a band, so a difference smaller than the spread
            # reads as what it is rather than as a trend.
            if curve["std"].notna().any():
                axis.fill_between(
                    curve[knob],
                    curve["mean"] - curve["std"].fillna(0),
                    curve["mean"] + curve["std"].fillna(0),
                    color=MODEL_COLOURS[model],
                    alpha=0.13,
                    linewidth=0,
                )

        axis.axhline(
            reference, color=REFERENCE, linestyle=(0, (4, 3)), linewidth=1.2
        )
        axis.annotate(
            f"persistence {reference:.1f}",
            xy=(0.98, reference),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="right",
            fontsize=7,
            color=INK_SOFT,
        )

        axis.axvline(
            committed_value, color=GRID, linewidth=1.4, zorder=1
        )
        axis.annotate(
            "baseline",
            xy=(committed_value, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -10),
            textcoords="offset points",
            fontsize=7,
            color=INK_SOFT,
        )

        if scale == "log":
            axis.set_xscale("log")
            axis.set_xticks(sorted(panel[knob].unique()))
            axis.get_xaxis().set_major_formatter(
                matplotlib.ticker.ScalarFormatter()
            )

        axis.set_title(title)
        axis.set_xlabel(title)
        axis.set_ylabel("test MAE (cases)")

    for index in range(len(present), rows * columns):
        axes[index // columns][index % columns].axis("off")

    axes[0][0].legend(loc="best")

    figure.suptitle(
        "Where the error is steep, and where tuning buys nothing",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.5,
        -0.02,
        "One factor at a time; every other knob held at the fixed schedule. "
        "Band = seed spread. A curve flatter than its band is not a trend.",
        ha="center",
        fontsize=8,
        color=INK_SOFT,
    )

    figure.tight_layout()
    path = FIGURE_DIR / "opt3_parameter_sensitivity.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

    return path


# ---------------------------------------------------------------------------
# Figure 4 -- the consequence
# ---------------------------------------------------------------------------

def figure_schedule_effect(metrics: pd.DataFrame, naive: pd.DataFrame) -> Path:
    """Test MAE by schedule arm, against persistence."""

    schedule = metrics[metrics["kind"] == "schedule"]
    if schedule.empty:
        return None

    reference = persistence_mae(naive, schedule["fold_id"].unique())
    labels = list(schedule["label"].drop_duplicates())

    figure, axis = plt.subplots(figsize=(7.6, 3.8))

    width = 0.38
    positions = np.arange(len(labels))

    for offset, model in zip((-width / 2, width / 2), ("gcn_gru", "gru_only")):
        subset = schedule[schedule["model"] == model]
        means = [
            subset[subset["label"] == label]["mae"].mean() for label in labels
        ]
        spread = [
            subset[subset["label"] == label].groupby("fold_id")["mae"].std().mean()
            for label in labels
        ]

        bars = axis.bar(
            positions + offset,
            means,
            width,
            yerr=spread,
            error_kw={"ecolor": INK_SOFT, "elinewidth": 0.9, "capsize": 2},
            color=MODEL_COLOURS[model],
            label=MODEL_LABELS[model],
            zorder=3,
        )
        axis.bar_label(bars, fmt="%.2f", fontsize=7, color=INK_SOFT, padding=3)

    axis.axhline(reference, color=REFERENCE, linestyle=(0, (4, 3)), linewidth=1.3)
    axis.annotate(
        f"persistence {reference:.2f}",
        xy=(0.99, reference),
        xycoords=("axes fraction", "data"),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=INK_SOFT,
    )

    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("test MAE (cases)")
    axis.set_xlabel("schedule arm")
    axis.set_title("Does fixing the schedule change the graph verdict?")
    axis.legend(loc="upper left")

    figure.text(
        0.5,
        -0.06,
        "Mean over folds and seeds; error bar is the seed spread. "
        "Lower is better. Bars below the dashed line beat persistence.",
        ha="center",
        fontsize=8,
        color=INK_SOFT,
    )

    figure.tight_layout()
    path = FIGURE_DIR / "opt4_schedule_effect.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_figures(
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    naive: pd.DataFrame,
) -> list[Path]:
    """Generate all available optimization figures from loaded artifacts."""

    written = [
        figure_training_curves(curves),
        figure_stopping_point(metrics),
        figure_sensitivity(metrics, naive),
        figure_schedule_effect(metrics, naive),
    ]

    for path in written:
        if path is not None:
            print(f"Wrote {path.relative_to(PROJECT_DIR)}")

    return [path for path in written if path is not None]


def main() -> int:
    style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    metrics, curves, naive = load()
    generate_figures(metrics, curves, naive)

    return 0


if __name__ == "__main__":
    sys.exit(main())

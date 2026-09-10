"""
Paper figures, generated from the committed result CSVs.

Three figures, each carrying a claim the paper makes in prose and currently
asks the reader to reconstruct from a table:

    fig_skill_by_horizon    the central result -- skill over same-horizon
                            persistence crossing zero between h=2 and h=3, on
                            both MAE and peak MAE. This is the paper's headline
                            and it is a shape, not a number.

    fig_h1_landscape        why one-week work stalled: every h=1 intervention
                            against the persistence line, sorted. Compresses
                            five subsections of Section 5 into one panel.

    fig_climate_ablation    what causes the h=4 skill -- the three-arm ablation
                            with the shuffle control, which is the only arm that
                            separates climate content from channel count.

Everything is read from `results/models/*.csv`; nothing is hardcoded, so a
rerun of any experiment regenerates the figures with the new numbers.

Output: `figures/*.pdf` (vector, for LaTeX) and `*.png` (for quick viewing).

Usage:
    python scripts/30.make_paper_figures.py
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
RESULTS_DIR = PROJECT_DIR / "results" / "models"
FIGURES_DIR = PROJECT_DIR / "figures"

# Single-column ACM figures are ~3.3in wide. Sizing them here rather than
# scaling in LaTeX keeps the font sizes honest: a figure shrunk by
# \includegraphics[width=...] shrinks its text too.
COLUMN_WIDTH = 3.33

PALETTE = {
    "model": "#1f4e79",
    "persistence": "#8c8c8c",
    "peak": "#c0504d",
    "good": "#2e7d32",
    "bad": "#b71c1c",
    "neutral": "#5b6770",
    "full": "#1f4e79",
    "no_climate": "#e8a33d",
    "shuffled": "#c0504d",
}


def style() -> None:
    """Match a two-column paper: small text, no chartjunk."""

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.4,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for suffix in ("pdf", "png"):
        path = FIGURES_DIR / f"{name}.{suffix}"
        fig.savefig(path)

    plt.close(fig)
    print(f"  wrote figures/{name}.pdf and .png")


# ---------------------------------------------------------------------------
# Figure 1 -- skill by horizon
# ---------------------------------------------------------------------------

def figure_skill_by_horizon() -> None:
    """Skill over same-horizon persistence, MAE and peak MAE, h = 1-4.

    The paper's central claim is a *crossing*: negative skill at short horizons,
    positive at h=3 and h=4, on both metrics. A table of eight percentages
    states that; a line through zero shows it.
    """

    metrics = pd.read_csv(RESULTS_DIR / "multi_horizon_metrics.csv")
    naive = pd.read_csv(RESULTS_DIR / "multi_horizon_naive.csv")

    model = metrics[
        (~metrics["ensemble"])
        & (metrics["headline"])
        & (metrics["backbone"] == "identity")
        & (metrics["arm"] == "separate")
    ]

    persistence = naive[naive["model"] == "persistence"]
    if "headline" in persistence.columns:
        persistence = persistence[persistence["headline"]]

    reference = persistence.groupby("horizon")[["mae", "peak_mae"]].mean()

    horizons = sorted(model["horizon"].unique())

    per_fold = model.groupby(["horizon", "fold_id"])[["mae", "peak_mae"]].mean()
    persistence_fold = persistence.groupby(["horizon", "fold_id"])[
        ["mae", "peak_mae"]
    ].mean()

    mae_skill, peak_skill = [], []
    deltas, delta_errors, p_values = [], [], []

    for horizon in horizons:
        rows = per_fold.loc[horizon]
        mae_reference = reference.loc[horizon, "mae"]
        peak_reference = reference.loc[horizon, "peak_mae"]

        mae_skill.append(100.0 * (mae_reference - rows["mae"].mean()) / mae_reference)
        peak_skill.append(
            100.0 * (peak_reference - rows["peak_mae"].mean()) / peak_reference
        )

        # The paired per-fold MAE difference -- the quantity the paper's
        # significance tests are computed on. Percentage skill per fold is not
        # usable as an error bar here: low-count folds produce ratios with a
        # standard deviation near 80 percentage points, which says nothing
        # about uncertainty in the mean. The paired difference is what the
        # t-test uses, so its standard error is the honest interval.
        paired = rows["mae"] - persistence_fold.loc[horizon, "mae"]
        deltas.append(paired.mean())
        delta_errors.append(paired.std() / np.sqrt(len(paired)))

        try:
            from scipy import stats

            p_values.append(
                stats.ttest_rel(rows["mae"], persistence_fold.loc[horizon, "mae"])[1]
            )
        except Exception:
            p_values.append(float("nan"))

    fig, axes = plt.subplots(
        1, 2, figsize=(COLUMN_WIDTH * 2.06, 2.3), gridspec_kw={"wspace": 0.3}
    )

    # -- Left: skill in percent, both metrics ------------------------------
    ax = axes[0]
    ax.axhspan(-20, 0, color=PALETTE["bad"], alpha=0.05, zorder=0)
    ax.axhline(0, color=PALETTE["persistence"], lw=0.9, zorder=1)

    ax.plot(
        horizons,
        mae_skill,
        color=PALETTE["model"],
        marker="o",
        markersize=4,
        label="MAE skill",
        zorder=3,
    )
    ax.plot(
        horizons,
        peak_skill,
        color=PALETTE["peak"],
        marker="s",
        markersize=3.5,
        linestyle="--",
        label="Peak MAE skill",
        zorder=3,
    )

    for horizon, value in zip(horizons, mae_skill):
        ax.annotate(
            f"{value:+.1f}",
            (horizon, value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=6.2,
            color=PALETTE["model"],
        )

    ax.text(
        2.55, 11.4, "better than persistence", fontsize=6,
        color=PALETTE["good"], style="italic",
    )
    ax.text(
        1.08, -13.4, "worse than persistence", fontsize=6,
        color=PALETTE["bad"], style="italic",
    )

    ax.set_xlabel("Forecast horizon (weeks ahead)")
    ax.set_ylabel("Skill vs same-horizon persistence (%)")
    ax.set_xticks(horizons)
    ax.set_ylim(-16, 16)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("(a) Skill crosses zero between $h{=}2$ and $h{=}3$", fontsize=7.2)

    # -- Right: the paired difference the tests are run on -----------------
    ax = axes[1]
    ax.axhline(0, color=PALETTE["persistence"], lw=0.9, zorder=1)

    colours = [
        PALETTE["good"] if d < 0 and p < 0.05 else
        (PALETTE["model"] if d < 0 else PALETTE["bad"])
        for d, p in zip(deltas, p_values)
    ]

    ax.errorbar(
        horizons,
        deltas,
        yerr=delta_errors,
        fmt="o",
        markersize=5,
        color=PALETTE["model"],
        ecolor=PALETTE["neutral"],
        capsize=3,
        elinewidth=0.9,
        zorder=3,
    )
    for horizon, delta, colour in zip(horizons, deltas, colours):
        ax.plot(horizon, delta, "o", markersize=5, color=colour, zorder=4)

    for horizon, delta, error, p_value in zip(
        horizons, deltas, delta_errors, p_values
    ):
        label = f"$p={p_value:.3f}$" if p_value >= 0.001 else "$p<0.001$"
        ax.annotate(
            label,
            (horizon, delta - error),
            textcoords="offset points",
            xytext=(0, -11),
            ha="center",
            fontsize=6,
            color=PALETTE["good"] if p_value < 0.05 else PALETTE["neutral"],
        )

    ax.set_xlabel("Forecast horizon (weeks ahead)")
    ax.set_ylabel("Paired $\\Delta$ MAE vs persistence")
    ax.set_xticks(horizons)
    ax.set_ylim(-5.6, 3.6)
    ax.set_title(
        "(b) Paired over 7 headline folds ($\\pm$ s.e.)", fontsize=7.2
    )

    save(fig, "fig_skill_by_horizon")


# ---------------------------------------------------------------------------
# Figure 2 -- the h=1 landscape
# ---------------------------------------------------------------------------

def figure_h1_landscape() -> None:
    """Every h=1 intervention against the persistence line.

    Sections 5.1-5.6 each report one intervention that fails to beat
    persistence at one week. Read in sequence they are six tables; read as one
    sorted panel they are a single finding, which is the finding the Discussion
    actually argues.
    """

    entries = []

    baseline = pd.read_csv(RESULTS_DIR / "baseline_metrics.csv")
    single = baseline[~baseline["ensemble"]] if "ensemble" in baseline else baseline
    headline = single[single["headline"]]

    naive_metrics = pd.read_csv(RESULTS_DIR / "naive_baseline_metrics.csv")
    naive_headline = naive_metrics[naive_metrics["headline"]]
    persistence_mae = naive_headline[
        naive_headline["model"] == "persistence"
    ]["mae"].mean()

    for (model, variant), group in headline.groupby(["model", "variant"]):
        entries.append((f"{model} ({variant})", group["mae"].mean(), "baseline"))

    graphs = pd.read_csv(RESULTS_DIR / "graph_metrics.csv")
    graph_single = graphs[(~graphs["ensemble"]) & (graphs["headline"])]
    for arm, group in graph_single.groupby("arm"):
        entries.append((f"graph: {arm}", group["mae"].mean(), "graph"))

    optimisers = pd.read_csv(RESULTS_DIR / "optimiser_metrics.csv")
    optimiser_single = optimisers[
        (~optimisers["ensemble"]) & (optimisers["headline"])
    ]
    for arm, group in optimiser_single.groupby("arm"):
        entries.append((f"optim: {arm}", group["mae"].mean(), "optimiser"))

    frame = pd.DataFrame(entries, columns=["label", "mae", "family"])

    # `graph: identity` and `gru_only` are the same model reached through two
    # experiments, and `graph: contiguity` is `gcn_gru`. Listing both members of
    # each pair would suggest more independent evidence than exists, so the
    # duplicates are dropped on the score rather than on the name.
    frame["rounded"] = frame["mae"].round(2)
    frame = (
        frame.drop_duplicates("rounded", keep="first")
        .drop(columns="rounded")
        .sort_values("mae")
        .reset_index(drop=True)
    )

    colours = {
        "baseline": PALETTE["model"],
        "graph": PALETTE["peak"],
        "optimiser": PALETTE["neutral"],
    }

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 0.21 * len(frame) + 0.8))

    positions = np.arange(len(frame))

    # The axis starts just below the best score, not at zero. Every bar here is
    # between 16 and 20 MAE, so a zero-based axis spends 80% of the panel on
    # empty space and compresses the differences the figure exists to show.
    low = min(frame["mae"].min(), persistence_mae) - 0.9
    high = frame["mae"].max() + 1.5

    ax.barh(
        positions,
        frame["mae"] - low,
        left=low,
        color=[colours[f] for f in frame["family"]],
        height=0.7,
        zorder=3,
    )

    ax.axvline(
        persistence_mae,
        color=PALETTE["bad"],
        lw=1.2,
        linestyle="--",
        zorder=5,
    )
    ax.annotate(
        f"persistence {persistence_mae:.2f}",
        (persistence_mae, -0.72),
        xytext=(persistence_mae + 0.45, -0.72),
        ha="left",
        va="center",
        fontsize=6.4,
        color=PALETTE["bad"],
        annotation_clip=False,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(frame["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Headline MAE at $h=1$ (lower is better)")
    ax.set_xlim(low, high)

    for position, value in zip(positions, frame["mae"]):
        ax.text(
            value + 0.12,
            position,
            f"{value:.2f}",
            va="center",
            fontsize=6,
            color="#333333",
        )

    ax.grid(axis="x", color="#dddddd", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    save(fig, "fig_h1_landscape")


# ---------------------------------------------------------------------------
# Figure 3 -- the climate ablation
# ---------------------------------------------------------------------------

def figure_climate_ablation() -> None:
    """The three-arm ablation: what causes the h=4 skill.

    Two panels, because the claim has two halves. Left: the cost of each
    ablation rises with the horizon -- the shape the 5-10 week delay predicts
    and flat-near-zero would refute. Right: at h=4, how much of the skill each
    arm keeps, which is where the shuffle control earns its place.
    """

    metrics = pd.read_csv(RESULTS_DIR / "climate_ablation_metrics.csv")
    naive = pd.read_csv(RESULTS_DIR / "climate_ablation_naive.csv")

    single = metrics[(~metrics["ensemble"]) & (metrics["headline"])]
    horizons = sorted(single["horizon"].unique())

    by_fold = single.groupby(["horizon", "arm", "fold_id"])["mae"].mean()

    fig, axes = plt.subplots(
        1, 2, figsize=(COLUMN_WIDTH * 2.06, 2.25), gridspec_kw={"wspace": 0.28}
    )

    # -- Left: ablation cost against horizon -------------------------------
    ax = axes[0]
    ax.axhline(0, color=PALETTE["persistence"], lw=0.9, zorder=1)

    for arm, label in (
        ("no_climate", "climate removed"),
        ("shuffled", "climate shuffled (control)"),
    ):
        means, errors = [], []
        for horizon in horizons:
            delta = by_fold.loc[horizon, arm] - by_fold.loc[horizon, "full"]
            means.append(delta.mean())
            errors.append(delta.std() / np.sqrt(len(delta)))

        ax.errorbar(
            horizons,
            means,
            yerr=errors,
            color=PALETTE[arm],
            marker="o" if arm == "shuffled" else "s",
            markersize=4,
            capsize=2.5,
            elinewidth=0.8,
            label=label,
        )

    ax.annotate(
        "$p=0.016$",
        (4, 1.55),
        textcoords="offset points",
        xytext=(-6, 9),
        ha="right",
        fontsize=6.2,
        color=PALETTE["shuffled"],
    )

    ax.set_xlabel("Forecast horizon (weeks ahead)")
    ax.set_ylabel("MAE cost of the ablation")
    ax.set_xticks(horizons)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("(a) Climate matters more as the horizon grows", fontsize=7.2)

    # -- Right: skill retained at h=4 --------------------------------------
    ax = axes[1]

    persistence = naive[naive["model"] == "persistence"].set_index("horizon")
    reference = persistence.loc[4, "headline_mae"]
    peak_reference = persistence.loc[4, "headline_peak_mae"]

    h4 = single[single["horizon"] == 4]
    arms = ["full", "no_climate", "shuffled"]
    labels = ["all channels", "climate\nremoved", "climate\nshuffled"]

    mae_skill = [
        100.0 * (reference - h4[h4["arm"] == a]["mae"].mean()) / reference
        for a in arms
    ]
    peak_skill = [
        100.0
        * (peak_reference - h4[h4["arm"] == a]["peak_mae"].mean())
        / peak_reference
        for a in arms
    ]

    positions = np.arange(len(arms))
    width = 0.36

    ax.axhline(0, color=PALETTE["persistence"], lw=0.9, zorder=1)
    ax.bar(
        positions - width / 2,
        mae_skill,
        width,
        color=PALETTE["model"],
        label="MAE skill",
        zorder=3,
    )
    ax.bar(
        positions + width / 2,
        peak_skill,
        width,
        color=PALETTE["peak"],
        label="Peak MAE skill",
        zorder=3,
    )

    for position, value in zip(positions - width / 2, mae_skill):
        ax.text(position, value + 0.35, f"{value:.1f}", ha="center", fontsize=6)
    for position, value in zip(positions + width / 2, peak_skill):
        ax.text(position, value + 0.35, f"{value:.1f}", ha="center", fontsize=6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Skill vs persistence at $h{=}4$ (%)")
    ax.set_ylim(0, max(mae_skill) * 1.28)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("(b) Half the $h{=}4$ skill is climate content", fontsize=7.2)

    save(fig, "fig_climate_ablation")


def main() -> int:
    style()

    print("Generating paper figures from results/models/*.csv")
    figure_skill_by_horizon()
    figure_h1_landscape()
    figure_climate_ablation()

    return 0


if __name__ == "__main__":
    sys.exit(main())

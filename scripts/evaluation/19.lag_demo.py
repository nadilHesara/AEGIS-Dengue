"""
The learnable-lag component in one command, for a supervisor walkthrough.

Trains nothing. Everything below is read back from the artifacts scripts 17 and
18 already wrote, so this runs in a couple of seconds in front of an audience
and always shows the committed numbers rather than a fresh random seed.

It prints the argument in five steps and writes four figures:

    1. the question        why a fixed lag is the wrong assumption
    2. the measurement     what the delay actually is        (scripts/17)
    3. the mechanism       how the model learns one instead  (src/models)
    4. the result          whether learning it helped        (scripts/18)
    5. the diagnosis       why not, and what is left to try

Run:
    python scripts/19.lag_demo.py            # narration + figures
    python scripts/19.lag_demo.py --quiet    # figures only

Outputs:
    results/figures/fig1_rainfall_lag_by_district.png
    results/figures/fig2_colombo_measured_vs_learned.png
    results/figures/fig3_measured_vs_learned_scatter.png
    results/figures/fig4_accuracy_by_arm.png
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parents[2]

SCAN_PATH = PROJECT_DIR / "results" / "eda" / "lag_correlation.csv"
METRICS_PATH = PROJECT_DIR / "results" / "models" / "lag_metrics.csv"
KERNELS_PATH = PROJECT_DIR / "results" / "models" / "learned_lags.csv"
FIGURE_DIR = PROJECT_DIR / "results" / "figures"

RAINFALL = "rainfall_daily_mean_mm"
FOCUS_DISTRICT = "Colombo"

# The fold every figure is drawn on: the longest history, and the one whose
# learned kernels had the most training data behind them.
REPORT_FOLD = 9

# Categorical slots 1-4, in fixed order, from the validated reference palette.
# Validated for this four-colour set on the adjacent pairlist: worst CVD dE 9.1,
# worst normal-vision dE 22.9. Aqua and yellow sit below 3:1 against the surface,
# so every bar built from them carries a direct value label -- that is the
# relief rule, not decoration.
ARM_COLOURS = {
    "no_lags_v0": "#2a78d6",
    "hand_lags_v1": "#eb6834",
    "learned_lags": "#1baf7a",
    "no_climate": "#eda100",
}

ARM_LABELS = {
    "no_lags_v0": "no lags (v0)",
    "hand_lags_v1": "hand lags (v1)",
    "learned_lags": "learned lags",
    "no_climate": "no climate",
}

MEASURED = "#2a78d6"
LEARNED = "#eb6834"

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2df"
SURFACE = "#fcfcfb"


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
    """Read the three artifacts, with a usable error if one is missing."""

    missing = [
        path.relative_to(PROJECT_DIR).as_posix()
        for path in (SCAN_PATH, METRICS_PATH, KERNELS_PATH)
        if not path.exists()
    ]

    if missing:
        raise SystemExit(
            "Missing "
            + ", ".join(missing)
            + ".\nRun scripts/17.lag_correlation_scan.py and "
            "scripts/18.train_lag_gcn_gru.py first."
        )

    return (
        pd.read_csv(SCAN_PATH),
        pd.read_csv(METRICS_PATH),
        pd.read_csv(KERNELS_PATH),
    )


def clean_axes(axes) -> None:
    """Drop the top and right spines; keep the rest recessive."""

    for side in ("top", "right"):
        axes.spines[side].set_visible(False)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_rainfall_lag_by_district(scan: pd.DataFrame) -> Path:
    """Measured rainfall delay per district: the component's standing result.

    One measure, 25 categories, so a sorted horizontal bar -- magnitude read
    against a common baseline, labels horizontal and legible.
    """

    data = (
        scan[(scan["fold_id"] == REPORT_FOLD) & (scan["feature"] == RAINFALL)]
        .drop_duplicates("node_id")
        .sort_values("peak_lag_deseasonalised")
    )

    figure, axes = plt.subplots(figsize=(6.4, 6.2))

    positions = np.arange(len(data))
    axes.barh(
        positions,
        data["peak_lag_deseasonalised"],
        color=MEASURED,
        height=0.68,
    )

    for position, lag in zip(positions, data["peak_lag_deseasonalised"]):
        axes.text(
            lag + 0.18, position, f"{lag:.0f}", va="center", ha="left",
            fontsize=7.5, color=INK_SOFT,
        )

    axes.set_yticks(positions)
    axes.set_yticklabels(data["canonical_name"], fontsize=8)
    axes.set_xlabel("weeks between rainfall and reported cases")
    axes.set_title(
        "Measured rainfall-to-dengue delay, by district", loc="left", pad=12
    )
    axes.set_xlim(0, max(data["peak_lag_deseasonalised"]) + 1.6)
    axes.grid(axis="y", visible=False)
    clean_axes(axes)

    figure.text(
        0.01, 0.005,
        f"Cross-correlation, seasonality removed, fold {REPORT_FOLD} training "
        "periods only. All 25 districts fall in a 5-10 week band.",
        fontsize=7, color=INK_SOFT,
    )

    return save(figure, "fig1_rainfall_lag_by_district.png")


def figure_colombo(scan: pd.DataFrame, kernels: pd.DataFrame) -> Path:
    """What was measured against what was learned, for one district.

    Two stacked panels sharing the lag axis rather than two y-scales on one
    plot: a correlation and a kernel weight are different quantities, and
    overlaying them on twin axes would let the choice of scaling decide whether
    the reader sees agreement.
    """

    measured = scan[
        (scan["fold_id"] == REPORT_FOLD)
        & (scan["canonical_name"] == FOCUS_DISTRICT)
        & (scan["feature"] == RAINFALL)
    ].sort_values("lag")

    learned = kernels[
        (kernels["fold_id"] == REPORT_FOLD)
        & (kernels["canonical_name"] == FOCUS_DISTRICT)
        & (kernels["feature"] == RAINFALL)
    ].sort_values("lag")

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.6, 5.0), sharex=True,
        gridspec_kw={"hspace": 0.28},
    )

    top.axhline(0, color=GRID, linewidth=1)
    top.plot(
        measured["lag"], measured["correlation_deseasonalised"],
        color=MEASURED, linewidth=2, marker="o", markersize=4,
    )

    peak = measured.loc[measured["correlation_deseasonalised"].idxmax()]
    top.annotate(
        f"measured peak\nlag {peak['lag']:.0f} weeks",
        xy=(peak["lag"], peak["correlation_deseasonalised"]),
        xytext=(peak["lag"] + 3.2, peak["correlation_deseasonalised"] * 0.98),
        fontsize=8, color=INK,
        arrowprops={"arrowstyle": "->", "color": INK_SOFT, "linewidth": 0.9},
    )

    top.set_ylabel("correlation with cases")
    top.set_title(
        f"{FOCUS_DISTRICT}: the delay measured, and the delay learned",
        loc="left", pad=10,
    )
    clean_axes(top)

    bottom.bar(
        learned["lag"], learned["weight"], color=LEARNED, width=0.68,
    )

    heaviest = learned.loc[learned["weight"].idxmax()]
    bottom.annotate(
        f"learned mass piles up\nat the {int(learned['lag'].max())}-week edge",
        xy=(heaviest["lag"], heaviest["weight"]),
        xytext=(heaviest["lag"] - 13.5, heaviest["weight"] * 0.86),
        fontsize=8, color=INK,
        arrowprops={"arrowstyle": "->", "color": INK_SOFT, "linewidth": 0.9},
    )

    bottom.set_ylabel("learned kernel weight")
    bottom.set_xlabel("lag (reporting periods before the forecast week)")
    clean_axes(bottom)

    figure.text(
        0.01, 0.005,
        "Top: measured by cross-correlation. Bottom: learned end-to-end by the "
        "encoder. They disagree, and the measurement is the one with a "
        "mechanism behind it.",
        fontsize=7, color=INK_SOFT,
    )

    return save(figure, "fig2_colombo_measured_vs_learned.png")


def figure_scatter(scan: pd.DataFrame, kernels: pd.DataFrame) -> tuple[Path, float]:
    """Every district: measured delay against learned delay.

    The check the whole component turns on. If the encoder recovered real
    structure the points track the diagonal; scatter means it did not.
    """

    measured = (
        scan[(scan["fold_id"] == REPORT_FOLD) & (scan["feature"] == RAINFALL)]
        .drop_duplicates("node_id")
        .set_index("node_id")["peak_lag_deseasonalised"]
    )

    learned = (
        kernels[(kernels["fold_id"] == REPORT_FOLD) & (kernels["feature"] == RAINFALL)]
        .drop_duplicates("node_id")
        .set_index("node_id")["peak_lag"]
    )

    paired = pd.DataFrame({"measured": measured, "learned": learned}).dropna()
    correlation = paired["measured"].corr(paired["learned"])

    figure, axes = plt.subplots(figsize=(5.6, 5.2))

    limit = 27
    axes.plot(
        [0, limit], [0, limit], color=INK_SOFT, linewidth=1,
        linestyle="--", label="perfect agreement",
    )
    axes.scatter(
        paired["measured"], paired["learned"],
        s=52, color=MEASURED, edgecolor=SURFACE, linewidth=1.2, zorder=3,
        label="district",
    )

    axes.set_xlim(0, limit)
    axes.set_ylim(0, limit)
    axes.set_xlabel("measured delay (weeks)")
    axes.set_ylabel("learned delay (weeks)")
    axes.set_title("Learned delay against measured delay", loc="left", pad=12)
    axes.legend(loc="lower right")
    clean_axes(axes)

    axes.text(
        0.03, 0.95,
        f"r = {correlation:+.2f}\nn = {len(paired)} districts",
        transform=axes.transAxes, fontsize=8.5, color=INK, va="top",
    )

    figure.text(
        0.01, 0.005,
        f"Rainfall, fold {REPORT_FOLD}. Points on the dashed line would mean "
        "the encoder recovered the measured delay.",
        fontsize=7, color=INK_SOFT,
    )

    return save(figure, "fig3_measured_vs_learned_scatter.png"), correlation


def figure_accuracy(metrics: pd.DataFrame) -> Path:
    """Did any of it help? Headline error, then the same thing per fold.

    Left panel is the summary a reader wants first; right panel is the honesty
    check, because the headline mean averages folds whose case volumes differ
    five-fold and 2017 therefore dominates it.
    """

    headline = metrics[metrics["headline"]]
    summary = (
        headline.groupby("arm")["mae"].mean().sort_values(ascending=False)
    )

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(10.4, 4.4), gridspec_kw={"width_ratios": [1, 1.45]}
    )

    positions = np.arange(len(summary))
    left.barh(
        positions, summary.to_numpy(),
        color=[ARM_COLOURS[arm] for arm in summary.index], height=0.62,
    )

    for position, value in zip(positions, summary.to_numpy()):
        left.text(
            value + 0.12, position, f"{value:.2f}", va="center", ha="left",
            fontsize=8.5, color=INK,
        )

    left.set_yticks(positions)
    left.set_yticklabels([ARM_LABELS[arm] for arm in summary.index], fontsize=8.5)
    left.set_xlabel("headline MAE (lower is better)")
    left.set_title("Mean error over the 7 headline folds", loc="left", pad=12)
    left.set_xlim(0, summary.max() * 1.16)
    left.grid(axis="y", visible=False)
    clean_axes(left)

    # Per fold, relative to the best arm rather than as raw MAE. Fold error
    # ranges from 9 to 48 across the seven folds, so raw bars are a chart about
    # which year had more dengue; the arms only become comparable once the fold
    # scale is divided out.
    reference = "no_lags_v0"
    order = ["hand_lags_v1", "learned_lags", "no_climate"]
    folds = sorted(headline["fold_id"].unique())
    table = headline.pivot_table(index="fold_id", columns="arm", values="mae")
    relative = 100 * (table.div(table[reference], axis=0) - 1)

    width = 0.26
    base = np.arange(len(folds))

    right.axhline(0, color=INK_SOFT, linewidth=1)

    for index, arm in enumerate(order):
        right.bar(
            base + (index - 1) * width,
            relative[arm].reindex(folds).to_numpy(),
            width=width * 0.88,
            color=ARM_COLOURS[arm],
            label=ARM_LABELS[arm],
        )

    right.set_xticks(base)
    right.set_xticklabels(
        [f"{fold}\n{int(headline[headline.fold_id == fold].test_year.iloc[0])}"
         for fold in folds]
    )
    right.set_ylabel(f"% MAE vs {ARM_LABELS[reference]}")
    right.set_xlabel("fold / test year")
    right.set_title("Per fold, relative to the best arm", loc="left", pad=26)
    right.legend(
        ncol=3, loc="lower left", bbox_to_anchor=(0, 1.005),
    )
    right.grid(axis="x", visible=False)
    clean_axes(right)

    # Data-driven limits, not symmetric ones: almost nothing lands below zero,
    # and a symmetric axis would spend half the panel on empty space and shrink
    # the bars that carry the result.
    values = relative[order].to_numpy()
    right.set_ylim(min(values.min() * 1.6, -3.0), values.max() * 1.18)
    right.text(
        0.985, 0.03, "worse above the line",
        transform=right.transAxes, ha="right", fontsize=7.5, color=INK_SOFT,
    )

    figure.text(
        0.01, 0.005,
        "Right panel divides out the fold scale: error ranges from 9 to 48 MAE "
        "across these folds, so raw bars would compare years, not arms.",
        fontsize=7, color=INK_SOFT,
    )

    return save(figure, "fig4_accuracy_by_arm.png")


def save(figure, name: str) -> Path:
    """Write one figure and return where it went."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name

    figure.tight_layout(rect=(0, 0.028, 1, 1))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)

    return path


# ---------------------------------------------------------------------------
# Narration
# ---------------------------------------------------------------------------

def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n  {title}\n{'=' * 74}")


def say(text: str) -> None:
    print(textwrap.fill(textwrap.dedent(text).strip(), width=74))
    print()


def narrate(scan, metrics, kernels, agreement: float) -> None:
    """Walk an audience through the argument, in the order it has to be made."""

    rule("1. THE QUESTION")
    say("""
        Dengue does not follow the rain immediately. Water pools, mosquitoes
        breed, people are bitten, fall ill, and only then reach a hospital. The
        baseline handles that with trailing 4, 8 and 12 week rainfall averages
        -- three windows chosen by hand, identical for all 25 districts.
    """)
    say("""
        Colombo is wet, dense and urban; Jaffna is dry and dispersed. There is
        no reason the delay should be the same in both. The question: can the
        model learn a per-district delay, and does that forecast better?
    """)

    rule("2. THE MEASUREMENT  (scripts/17.lag_correlation_scan.py)")

    rainfall = (
        scan[(scan["fold_id"] == REPORT_FOLD) & (scan["feature"] == RAINFALL)]
        .drop_duplicates("node_id")
    )
    focus = rainfall[rainfall["canonical_name"] == FOCUS_DISTRICT].iloc[0]

    say("""
        Before trusting any learned delay, measure the real one. Shift rainfall
        back 0 to 25 weeks, correlate against cases, take the best-aligned
        shift. Seasonality is removed first, so this is "extra rain leads to
        extra dengue", not "both peak in the same month". Training periods
        only. No model, 3.5 seconds.
    """)
    print(
        f"  {FOCUS_DISTRICT}: {focus['peak_lag_deseasonalised']:.0f} weeks "
        f"(r = {focus['peak_correlation_deseasonalised']:+.3f})"
    )
    print(
        f"  All 25 districts: {rainfall['peak_lag_deseasonalised'].min():.0f}"
        f"-{rainfall['peak_lag_deseasonalised'].max():.0f} weeks, "
        f"median {rainfall['peak_lag_deseasonalised'].median():.0f}\n"
    )
    say("""
        That is a real, stable, leakage-free result and it stands on its own:
        a per-district delay map is useful to a control programme whether or
        not any neural network improves. -> figure 1
    """)

    rule("3. THE MECHANISM  (src/models/lag_encoder.py)")
    say("""
        Each district gets a weighting curve over the previous 26 weeks, and
        the model learns it while it learns to forecast -- same loss, same
        backward pass. A freely chosen curve would be 26 x 7 x 25 = 4,550
        weights on ~800 training windows, which memorises. So each curve is
        built from 6 smooth bumps and only the bump positions, widths and a
        per-district mixture are learned: 548 parameters, non-negative, summing
        to one, and causal by construction.
    """)
    say("""
        The controlled test first: plant known delays of 3 to 19 weeks in
        synthetic data and the encoder recovers every district within 2 weeks,
        on 3 seeds. tests/test_lag_encoder.py -- 9 passed. The module works.
    """)

    rule("4. THE RESULT  (scripts/18.train_lag_gcn_gru.py)")

    headline = metrics[metrics["headline"]]
    summary = headline.groupby("arm")["mae"].mean().sort_values()

    say("""
        Four arms, identical folds, seeds, masks, graph, GRU, head and loss.
        Only the climate handling changes.
    """)
    for arm, value in summary.items():
        print(f"  {ARM_LABELS[arm]:<16} {value:6.2f} MAE")

    hand = summary["hand_lags_v1"]
    learned = summary["learned_lags"]
    print(
        f"\n  learned vs hand-coded: {100 * (hand - learned) / hand:+.1f}%  "
        "(negative = worse)\n"
    )

    say("""
        The learnable encoder is the WORST arm. It loses to hand-picked
        windows, to raw climate, and to dropping climate entirely. That is the
        finding, and it is not a bug -- the recovery test passes. -> figure 4
    """)

    rule("5. THE DIAGNOSIS")

    no_climate = summary["no_climate"]
    no_lags = summary["no_lags_v0"]
    recent = headline[headline["fold_id"].isin([6, 8, 9])]
    recent_gap = (
        recent[recent.arm == "no_climate"]["mae"].mean()
        - recent[recent.arm == "no_lags_v0"]["mae"].mean()
    )

    say(f"""
        The climate-free control explains it. Dropping every weather channel
        costs {100 * (no_climate - no_lags) / no_lags:+.1f}% overall, and on
        the recent normal years (2022, 2024, 2025) it costs
        {recent_gap:+.2f} MAE -- nothing. At one week ahead, last week's case
        count carries almost all the information.
    """)
    say(f"""
        A jointly-learned delay only gets a gradient if delayed climate reduces
        the loss. When climate contributes nothing, that gradient carries no
        information about delay, so the kernels drift to the boundary: measured
        vs learned agreement across districts is r = {agreement:+.2f}.
        -> figures 2 and 3
    """)
    say("""
        What is left to test: the horizon. At h=1 the previous case count
        dominates by construction, which is exactly why climate looks
        worthless. Delay structure should only start paying further out:

            python scripts/18.train_lag_gcn_gru.py --horizon 4 --seeds 1 --no-control
    """)


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="figures only")
    arguments = parser.parse_args()

    style()
    scan, metrics, kernels = load()

    paths = [
        figure_rainfall_lag_by_district(scan),
        figure_colombo(scan, kernels),
    ]
    scatter_path, agreement = figure_scatter(scan, kernels)
    paths.append(scatter_path)
    paths.append(figure_accuracy(metrics))

    if not arguments.quiet:
        narrate(scan, metrics, kernels, agreement)

    print(f"\n{'=' * 74}\n  FIGURES\n{'=' * 74}")
    for path in paths:
        print(f"  {path.relative_to(PROJECT_DIR).as_posix()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

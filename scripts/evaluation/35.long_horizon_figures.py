"""
Figures for the long-horizon benchmark, from scripts/34's tables only.

    fig_lh_skill.png            MAE and peak-MAE skill over same-horizon persistence
    fig_lh_climate.png          value of climate by horizon, in three model families
    fig_lh_probabilistic.png    WIS skill over conformal persistence, and 95% coverage
    fig_lh_early_warning.png    outbreak-week AUC and onset AUC

Colours are the validated categorical slots in fixed order, one per method and
the same method in every figure, never reassigned by rank. Every series also
carries its own marker and a legend entry, because three slots sit below 3:1
contrast on white.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.evaluation.long_horizon import BENCHMARK_DIR  # noqa: E402

TABLES = BENCHMARK_DIR / "tables"
FIGURES = BENCHMARK_DIR / "figures"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# method: (label, colour slot, marker). Fixed across every figure.
STYLE = {
    "gru_v2_quantile_lw": ("GRU, level-weighted quantile", "#2a78d6", "o"),
    "nb_shared_v2": ("NB, shared trunk (v2)", "#7b4fd6", "h"),
    "lgbm_v2": ("LightGBM", "#eb6834", "s"),
    "chronos2_joint": ("Chronos-2 joint, zero-shot", "#1baf7a", "^"),
    "ens_equal": ("Ensemble, equal weights", "#eda100", "D"),
    "chronos2_ft": ("Chronos-2, fine-tuned", "#008300", "P"),
}
PERSISTENCE = ("Persistence", MUTED, "x")


def style_axes(ax, ylabel, horizons):
    ax.set_ylabel(ylabel, color=INK)
    ax.set_xlabel("Forecast horizon (weeks ahead)", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.set_xticks(horizons)
    ax.set_xlim(horizons[0] - 0.5, horizons[-1] + 0.5)


def plot_series(ax, frame, value, horizons, keys, style):
    for key in keys:
        rows = frame[frame["key"] == key].set_index("horizon").reindex(horizons)
        if rows[value].isna().all():
            continue
        label, colour, marker = style[key]
        dashed = key == "persistence"
        ax.plot(horizons, rows[value], color=colour, marker=marker, markersize=6, linewidth=2,
                linestyle="--" if dashed else "-", label=label)
    ax.legend(loc="best", fontsize=8, frameon=False)


def title(ax, text):
    ax.set_title(text, color=INK, fontsize=10, loc="left")


def skill_figure(summary):
    persistence = summary[summary["method"] == "persistence"].set_index("horizon")
    horizons = sorted(summary["horizon"].unique())
    frame = summary.copy()
    frame["key"] = frame["method"]
    frame["mae_skill"] = [100 * (1 - r.mae / persistence.loc[r.horizon, "mae"]) for r in frame.itertuples()]
    frame["peak_skill"] = [100 * (1 - r.peak_mae / persistence.loc[r.horizon, "peak_mae"])
                           for r in frame.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, value, name in ((axes[0], "mae_skill", "MAE"), (axes[1], "peak_skill", "Peak-week MAE")):
        plot_series(ax, frame, value, horizons, list(STYLE), STYLE)
        ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
        title(ax, f"{name} skill vs same-horizon persistence (7 headline folds)")
        style_axes(ax, "Skill (%), above 0 beats persistence", horizons)
    fig.tight_layout()
    return fig


def climate_figure(ablations):
    # Value of observed weather over a training-years climatology of weather
    # (positive = observed weather helps beyond the seasonal calendar), in three
    # families, left; the same model's cost of removing climate, right.
    left = [("GRU: climate -> training climatology", 1.0, "GRU", "#2a78d6", "o"),
            ("NB: climate -> training climatology", 1.0, "NB shared", "#7b4fd6", "h"),
            ("LightGBM: climate -> training climatology", 1.0, "LightGBM", "#eb6834", "s")]
    right = [("GRU: climate removed", 1.0, "GRU", "#2a78d6", "o"),
             ("NB: climate removed", 1.0, "NB shared", "#7b4fd6", "h"),
             ("LightGBM: climate removed", 1.0, "LightGBM", "#eb6834", "s"),
             ("Chronos-2: climate covariates added", -1.0, "Chronos-2 (covariates added)", "#1baf7a", "^")]
    horizons = sorted(ablations["horizon"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    panels = ((axes[0], left, "Observed weather vs training climatology"),
              (axes[1], right, "Weather vs no weather"))
    for ax, specs, name in panels:
        metric = "mae"
        for comparison, sign, label, colour, marker in specs:
            rows = ablations[(ablations["comparison"] == comparison) & (ablations["metric"] == metric)]
            if rows.empty:
                continue
            rows = rows.sort_values("horizon")
            value = sign * rows["delta"]
            se = (rows["delta"] / rows["t"]).abs()
            ax.errorbar(rows["horizon"], value, yerr=se, color=colour, marker=marker, markersize=6,
                        linewidth=2, capsize=3, label=label)
            for h, v, p in zip(rows["horizon"], value, rows["p"]):
                if p < 0.05:
                    ax.annotate("*", (h, v), xytext=(0, 7), textcoords="offset points",
                                ha="center", color=INK, fontsize=12)
        ax.axhline(0, color=MUTED, linewidth=1, linestyle="--")
        title(ax, f"{name} (MAE, ± s.e., * p<0.05)")
        style_axes(ax, "MAE reduction (cases/week)", horizons)
        ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def probabilistic_figure(prob):
    series = {
        ("gru_v2_quantile_lw", "native"): STYLE["gru_v2_quantile_lw"],
        ("nb_shared_v2", "native"): STYLE["nb_shared_v2"],
        ("chronos2_joint", "native"): STYLE["chronos2_joint"],
        ("mix_nb_chronos", "native"): ("Mixture NB + Chronos-2", "#e87ba4", "v"),
        ("lgbm_v2", "conformal"): ("LightGBM + conformal", "#eb6834", "s"),
        ("lgbm_v2", "conformal_aci"): ("LightGBM + adaptive conformal", "#eda100", "D"),
        ("persistence", "conformal"): ("Persistence + conformal", MUTED, "x"),
    }
    frame = prob.copy()
    frame["key"] = list(zip(frame["method"], frame["variant"]))
    reference = frame[frame["key"] == ("persistence", "conformal")].set_index("horizon")["wis"]
    frame["wis_skill"] = [100 * (1 - r.wis / reference.loc[r.horizon]) for r in frame.itertuples()]
    horizons = sorted(frame["horizon"].unique())
    style = {k: v for k, v in series.items()}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    keys = [k for k in series if k != ("persistence", "conformal")]
    for key in keys:
        rows = frame[frame["key"] == key].set_index("horizon").reindex(horizons)
        label, colour, marker = style[key]
        axes[0].plot(horizons, rows["wis_skill"], color=colour, marker=marker, markersize=6,
                     linewidth=2, label=label)
    axes[0].axhline(0, color=MUTED, linewidth=1, linestyle="--")
    title(axes[0], "WIS skill vs persistence with conformal intervals")
    style_axes(axes[0], "Skill (%)", horizons)
    axes[0].legend(loc="best", fontsize=8, frameon=False)
    for key in series:
        rows = frame[frame["key"] == key].set_index("horizon").reindex(horizons)
        label, colour, marker = style[key]
        axes[1].plot(horizons, rows["cov95"], color=colour, marker=marker, markersize=6,
                     linewidth=2, label=label, linestyle="--" if key[0] == "persistence" else "-")
    axes[1].axhline(0.95, color=INK, linewidth=0.8, linestyle=":")
    title(axes[1], "Empirical coverage of the 95% interval (target 0.95)")
    style_axes(axes[1], "Coverage", horizons)
    axes[1].legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def early_warning_figure(ews, onset_task):
    """Left: outbreak-week ROC-AUC by horizon. Right: the onset task (at-risk
    origins, first crossing within K weeks), PR-AUC against the positive rate."""

    style = dict(STYLE)
    style["persistence"] = PERSISTENCE
    keys = list(STYLE) + ["persistence"]
    ews = ews.assign(key=ews["method"])
    horizons = sorted(ews["horizon"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    plot_series(axes[0], ews, "auc", horizons, keys, style)
    title(axes[0], "Outbreak weeks vs all other weeks (ROC-AUC)")
    style_axes(axes[0], "AUC", horizons)

    labels = {"persistence": "Persistence", "climatology_rate": "Seasonal base rate",
              "onset_lgbm": "Onset classifier", "onset_lgbm_noclimate": "Onset classifier, no climate",
              "lgbm_v2": "LightGBM", "gru_v2": "GRU v2", "gru_v2_quantile_lw": "GRU LW quantile",
              "nb_shared_v2": "NB shared", "chronos2_joint": "Chronos-2 joint",
              "chronos2_joint_ft": "Chronos-2 joint FT"}
    order = [m for m in labels if m in set(onset_task["method"])]
    width = 0.38
    for offset, (window, colour) in enumerate(((4, "#2a78d6"), (8, "#eb6834"))):
        rows = onset_task[onset_task["window"] == window].set_index("method").reindex(order)
        positions = [i + (offset - 0.5) * width for i in range(len(order))]
        axes[1].barh(positions, rows["pr_auc"], height=width, color=colour, label=f"K = {window} weeks")
        base = rows["positive_rate"].mean()
        axes[1].axvline(base, color=colour, linewidth=1, linestyle=":")
    axes[1].set_yticks(range(len(order)))
    axes[1].set_yticklabels([labels[m] for m in order])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("PR-AUC (dotted: positive rate = chance)", color=INK)
    for side in ("top", "right"):
        axes[1].spines[side].set_visible(False)
    axes[1].set_xlim(0, 0.8)
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    title(axes[1], "Onset: first crossing within K weeks, at-risk origins")
    fig.tight_layout()
    return fig


def coverage_by_year_figure(prob_per_fold):
    """95% coverage by test year at h = 4 and 12 for three interval constructions."""

    series = {("lgbm_v2", "conformal"): ("previous-year conformal", "#eb6834", "s"),
              ("lgbm_v2", "conformal_rolling"): ("rolling conformal", "#2a78d6", "o"),
              ("lgbm_v2", "conformal_aci"): ("adaptive conformal", "#1baf7a", "^"),
              ("nb_shared_v2", "native"): ("NB native", "#7b4fd6", "h")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, horizon in zip(axes, (4, 12)):
        subset = prob_per_fold[prob_per_fold["horizon"] == horizon]
        for (method, variant), (label, colour, marker) in series.items():
            rows = subset[(subset["method"] == method) & (subset["variant"] == variant)].sort_values("fold_id")
            if rows.empty:
                continue
            ax.plot(rows["fold_id"] + 2016, rows["cov95"], color=colour, marker=marker, linewidth=2,
                    label=label)
        ax.axhline(0.95, color=INK, linewidth=0.8, linestyle=":")
        ax.axvspan(2019.5, 2021.5, color=GRID, alpha=0.6, linewidth=0)
        title(ax, f"95% coverage by test year, h = {horizon}")
        ax.set_ylabel("Coverage", color=INK)
        ax.set_xlabel("Test year (2020-21 shaded: COVID; 2026: hold-out, Jan-May)", color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold"})
    figures = {
        "fig_lh_skill": skill_figure(pd.read_csv(TABLES / "headline_summary.csv")),
        "fig_lh_climate": climate_figure(pd.read_csv(TABLES / "ablations.csv")),
        "fig_lh_probabilistic": probabilistic_figure(pd.read_csv(TABLES / "probabilistic_summary.csv")),
        "fig_lh_early_warning": early_warning_figure(
            pd.read_csv(TABLES / "early_warning_summary.csv"),
            pd.read_csv(BENCHMARK_DIR / "onset_task" / "summary.csv")),
        "fig_lh_coverage_by_year": coverage_by_year_figure(
            pd.read_csv(TABLES / "probabilistic_per_fold.csv")),
    }
    for name, fig in figures.items():
        for suffix in ("png", "pdf"):
            fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {FIGURES.relative_to(PROJECT_DIR)}/{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

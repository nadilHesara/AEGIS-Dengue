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
    "lgbm_v2": ("LightGBM", "#eb6834", "s"),
    "chronos2": ("Chronos-2, zero-shot", "#1baf7a", "^"),
    "ens_top3": ("Ensemble, top-3 by prior year", "#eda100", "D"),
    "gru_v1": ("GRU v1 (short paper)", "#e87ba4", "v"),
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
    # Value of climate = error without climate minus error with it (positive = climate helps).
    specs = [("GRU: climate shuffled", 1.0, "GRU: shuffled minus full", "#2a78d6", "o"),
             ("LightGBM: climate removed", 1.0, "LightGBM: removed minus full", "#eb6834", "s"),
             ("Chronos-2: climate covariates added", -1.0, "Chronos-2: without minus with covariates",
              "#1baf7a", "^")]
    horizons = sorted(ablations["horizon"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, metric, name in ((axes[0], "mae", "MAE"), (axes[1], "peak_mae", "Peak-week MAE")):
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
        title(ax, f"Value of climate, {name} (± s.e., 7 folds; * paired p<0.05)")
        style_axes(ax, "Error reduction from climate (cases/week)", horizons)
        ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def probabilistic_figure(prob):
    series = {
        ("gru_v2_quantile_lw", "native"): STYLE["gru_v2_quantile_lw"],
        ("lgbm_v2", "conformal"): ("LightGBM + conformal", "#eb6834", "s"),
        ("chronos2", "native"): STYLE["chronos2"],
        ("gru_v1", "conformal"): ("GRU v1 + conformal", "#e87ba4", "v"),
        ("chronos2_ft", "native"): STYLE["chronos2_ft"],
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


def early_warning_figure(ews, onsets):
    style = dict(STYLE)
    style["persistence"] = PERSISTENCE
    keys = list(STYLE) + ["persistence"]
    ews = ews.assign(key=ews["method"])
    onsets = onsets.assign(key=onsets["method"])
    horizons = sorted(ews["horizon"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    plot_series(axes[0], ews, "auc", horizons, keys, style)
    title(axes[0], "Outbreak weeks vs all other weeks (ROC-AUC)")
    style_axes(axes[0], "AUC", horizons)
    plot_series(axes[1], onsets, "onset_auc", horizons, keys, style)
    axes[1].axhline(0.5, color=INK, linewidth=0.8, linestyle=":")
    title(axes[1], "Outbreak onsets vs quiet weeks (ROC-AUC; 0.5 = chance)")
    style_axes(axes[1], "AUC", horizons)
    fig.tight_layout()
    return fig


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold"})
    figures = {
        "fig_lh_skill": skill_figure(pd.read_csv(TABLES / "headline_summary.csv")),
        "fig_lh_climate": climate_figure(pd.read_csv(TABLES / "ablations.csv")),
        "fig_lh_probabilistic": probabilistic_figure(pd.read_csv(TABLES / "probabilistic_summary.csv")),
        "fig_lh_early_warning": early_warning_figure(pd.read_csv(TABLES / "early_warning_summary.csv"),
                                                     pd.read_csv(TABLES / "onset_detection.csv")),
    }
    for name, fig in figures.items():
        for suffix in ("png", "pdf"):
            fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {FIGURES.relative_to(PROJECT_DIR)}/{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Compare ERA5-Land and CHIRPS daily rainfall for the canonical districts.

Neither source is treated as truth. ERA5-Land is a reanalysis: physically
consistent, complete by construction, but modelled. CHIRPS is satellite
infrared calibrated against rain gauges: closer to observation, finer at 0.05
degrees, but with gaps where the satellite retrieval fails and a gauge
network that is sparse in places.

The comparison is diagnostic. This script never modifies either rainfall
series and never substitutes one for the other. Its output is a recommendation
for a human to act on, and the recommendation is deliberately conservative:
replacing the rainfall source is a modelling decision, not a validation one.

Focus districts span Sri Lanka's rainfall regimes, so that agreement in one
regime is not mistaken for agreement everywhere:

    Colombo       wet zone, coastal, highest dengue burden
    Gampaha       wet zone, adjacent to Colombo
    Ratnapura     wet zone, inland and orographic
    Jaffna        dry zone, northern peninsula
    Anuradhapura  dry zone, interior; the largest district, so grid
                  resolution is least likely to confound the comparison

Usage:

    python scripts/8.compare_era5_chirps.py
    python scripts/8.compare_era5_chirps.py --all-districts

Outputs (under results/climate/era5_chirps/):
    era5_chirps_comparison.md
    district_rainfall_comparison.csv
    monthly_rainfall_comparison.csv
    heavy_rainfall_events.csv
    plots/<district>_comparison.png
    plots/all_districts_summary.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent

ERA5_PATH = PROJECT_DIR / "data" / "raw" / "climate_daily_district.csv"
CHIRPS_PATH = PROJECT_DIR / "data" / "raw" / "chirps_daily_rainfall.csv"
NODES_PATH = PROJECT_DIR / "data" / "processed" / "nodes.csv"

RESULTS_DIR = PROJECT_DIR / "results" / "climate" / "era5_chirps"
PLOTS_DIR = RESULTS_DIR / "plots"

REPORT_PATH = RESULTS_DIR / "era5_chirps_comparison.md"
DISTRICT_COMPARISON_PATH = RESULTS_DIR / "district_rainfall_comparison.csv"
MONTHLY_COMPARISON_PATH = RESULTS_DIR / "monthly_rainfall_comparison.csv"
HEAVY_EVENTS_PATH = RESULTS_DIR / "heavy_rainfall_events.csv"

FOCUS_DISTRICTS = [
    "Colombo",
    "Gampaha",
    "Ratnapura",
    "Jaffna",
    "Anuradhapura",
]

# A day counts as wet at 1 mm rather than at any trace. Satellite and
# reanalysis products both produce large numbers of near-zero values that are
# numerical drizzle rather than real rain, and counting those would make the
# wet-day comparison a comparison of rounding behaviour.
WET_DAY_THRESHOLD_MM = 1.0

# Heavy-rainfall thresholds. The lower is a meaningful wet-season day; the
# upper is the kind of event that drives vector breeding and flooding.
HEAVY_RAINFALL_MM = 50.0
EXTREME_RAINFALL_MM = 100.0

# Correlation below this on daily values means the two products disagree
# about individual days even if their totals match.
STRONG_DAILY_CORRELATION = 0.7
MODERATE_DAILY_CORRELATION = 0.5

# Relative difference in mean rainfall beyond which the products carry a
# systematic bias rather than noise.
MEAN_BIAS_TOLERANCE = 0.20


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_paired_rainfall(
    era5_path: Path = ERA5_PATH,
    chirps_path: Path = CHIRPS_PATH,
) -> pd.DataFrame:
    """
    Load both rainfall series and pair them on district and date.

    An outer join is used deliberately: an inner join would hide exactly the
    missingness the comparison is meant to measure.
    """

    missing = [
        path for path in [era5_path, chirps_path] if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "These inputs do not exist:\n"
            + "\n".join(f"  {path}" for path in missing)
            + "\n\nRun scripts/5.extract_era5_daily.py and "
            "scripts/7.extract_chirps_daily.py first."
        )

    era5 = pd.read_csv(era5_path)
    chirps = pd.read_csv(chirps_path)

    era5["date"] = pd.to_datetime(era5["date"])
    chirps["date"] = pd.to_datetime(chirps["date"])

    era5_subset = era5[
        ["date", "node_id", "canonical_name", "rainfall_mm"]
    ].rename(columns={"rainfall_mm": "rainfall_era5"})

    chirps_subset = chirps[
        ["date", "node_id", "canonical_name", "rainfall_mm_chirps"]
    ].rename(columns={"rainfall_mm_chirps": "rainfall_chirps"})

    paired = era5_subset.merge(
        chirps_subset,
        on=["date", "node_id", "canonical_name"],
        how="outer",
    )

    return paired.sort_values(["canonical_name", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compare_district(paired: pd.DataFrame, district: str) -> dict:
    """
    Compute the comparison statistics for one district.

    Correlations use only days where both products report a value, since a
    correlation cannot be computed across a gap. The missingness columns
    record how many days that excluded, so a high correlation computed on a
    small overlap cannot be mistaken for broad agreement.
    """

    from scipy.stats import pearsonr, spearmanr

    district_rows = paired.loc[paired["canonical_name"].eq(district)]

    total_days = len(district_rows)

    era5_missing = int(district_rows["rainfall_era5"].isna().sum())
    chirps_missing = int(district_rows["rainfall_chirps"].isna().sum())

    both = district_rows.dropna(subset=["rainfall_era5", "rainfall_chirps"])

    record = {
        "canonical_name": district,
        "total_days": total_days,
        "days_both_present": len(both),
        "era5_missing_days": era5_missing,
        "chirps_missing_days": chirps_missing,
    }

    if len(both) < 30:
        # Too little overlap for any statistic to mean anything.
        record.update(
            {
                "pearson_r": np.nan,
                "spearman_r": np.nan,
                "era5_mean_mm": np.nan,
                "chirps_mean_mm": np.nan,
                "mean_difference_mm": np.nan,
                "relative_bias": np.nan,
                "era5_wet_days": np.nan,
                "chirps_wet_days": np.nan,
                "era5_heavy_days": np.nan,
                "chirps_heavy_days": np.nan,
                "era5_extreme_days": np.nan,
                "chirps_extreme_days": np.nan,
                "note": "fewer than 30 overlapping days",
            }
        )

        return record

    era5_values = both["rainfall_era5"].to_numpy()
    chirps_values = both["rainfall_chirps"].to_numpy()

    pearson = pearsonr(era5_values, chirps_values)
    spearman = spearmanr(era5_values, chirps_values)

    era5_mean = float(era5_values.mean())
    chirps_mean = float(chirps_values.mean())

    record.update(
        {
            # Pearson measures agreement on magnitude; Spearman on ordering.
            # Rainfall is heavily skewed, so Spearman is usually the fairer
            # summary and a large gap between the two signals that a few
            # extreme days dominate the Pearson value.
            "pearson_r": round(float(pearson[0]), 4),
            "spearman_r": round(float(spearman[0]), 4),
            "era5_mean_mm": round(era5_mean, 3),
            "chirps_mean_mm": round(chirps_mean, 3),
            "mean_difference_mm": round(era5_mean - chirps_mean, 3),
            "relative_bias": round(
                (era5_mean - chirps_mean) / chirps_mean, 4
            )
            if chirps_mean > 0
            else np.nan,
            "era5_wet_days": int((era5_values >= WET_DAY_THRESHOLD_MM).sum()),
            "chirps_wet_days": int(
                (chirps_values >= WET_DAY_THRESHOLD_MM).sum()
            ),
            "era5_heavy_days": int((era5_values >= HEAVY_RAINFALL_MM).sum()),
            "chirps_heavy_days": int(
                (chirps_values >= HEAVY_RAINFALL_MM).sum()
            ),
            "era5_extreme_days": int(
                (era5_values >= EXTREME_RAINFALL_MM).sum()
            ),
            "chirps_extreme_days": int(
                (chirps_values >= EXTREME_RAINFALL_MM).sum()
            ),
            "note": "",
        }
    )

    return record


def build_district_comparison(
    paired: pd.DataFrame,
    districts: list[str],
) -> pd.DataFrame:
    """Compare every requested district."""

    return pd.DataFrame(
        [compare_district(paired, district) for district in districts]
    )


def build_monthly_comparison(
    paired: pd.DataFrame,
    districts: list[str],
) -> pd.DataFrame:
    """
    Compare the mean seasonal cycle, by calendar month.

    Sri Lanka has two monsoons, so a product that reproduces the annual total
    but misplaces the seasonal peak would be misleading for a dengue model
    driven by lagged rainfall. This table exposes that.
    """

    subset = paired.loc[paired["canonical_name"].isin(districts)].copy()

    subset["month"] = subset["date"].dt.month

    monthly = (
        subset.groupby(["canonical_name", "month"])
        .agg(
            era5_mean_mm=("rainfall_era5", "mean"),
            chirps_mean_mm=("rainfall_chirps", "mean"),
            era5_days=("rainfall_era5", "count"),
            chirps_days=("rainfall_chirps", "count"),
        )
        .reset_index()
    )

    monthly["difference_mm"] = (
        monthly["era5_mean_mm"] - monthly["chirps_mean_mm"]
    )

    for column in ["era5_mean_mm", "chirps_mean_mm", "difference_mm"]:
        monthly[column] = monthly[column].round(3)

    return monthly


def build_heavy_events(
    paired: pd.DataFrame,
    districts: list[str],
) -> pd.DataFrame:
    """
    List days where either product reports heavy rainfall.

    Kept as individual days rather than a count, because whether the two
    products agree on the *same* heavy days matters more than whether they
    report the same number of them.
    """

    subset = paired.loc[paired["canonical_name"].isin(districts)].copy()

    heavy = subset.loc[
        subset["rainfall_era5"].ge(HEAVY_RAINFALL_MM).fillna(False)
        | subset["rainfall_chirps"].ge(HEAVY_RAINFALL_MM).fillna(False)
    ].copy()

    if heavy.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "canonical_name",
                "rainfall_era5",
                "rainfall_chirps",
                "difference_mm",
                "agreement",
            ]
        )

    heavy["difference_mm"] = (
        heavy["rainfall_era5"] - heavy["rainfall_chirps"]
    ).round(2)

    def _agreement(row):
        era5_heavy = (
            pd.notna(row["rainfall_era5"])
            and row["rainfall_era5"] >= HEAVY_RAINFALL_MM
        )
        chirps_heavy = (
            pd.notna(row["rainfall_chirps"])
            and row["rainfall_chirps"] >= HEAVY_RAINFALL_MM
        )

        if era5_heavy and chirps_heavy:
            return "both"

        return "era5 only" if era5_heavy else "chirps only"

    heavy["agreement"] = heavy.apply(_agreement, axis=1)

    return heavy[
        [
            "date",
            "canonical_name",
            "rainfall_era5",
            "rainfall_chirps",
            "difference_mm",
            "agreement",
        ]
    ].sort_values(["canonical_name", "date"])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_district(paired: pd.DataFrame, district: str, path: Path) -> None:
    """
    Draw the four-panel comparison for one district.

    Monthly totals, the seasonal cycle, a daily scatter and the wet-day
    distribution together show whether a disagreement is a bias, a seasonal
    misplacement, or confined to extremes.
    """

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    rows = paired.loc[paired["canonical_name"].eq(district)].copy()

    both = rows.dropna(subset=["rainfall_era5", "rainfall_chirps"])

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    figure.suptitle(
        f"{district} — ERA5-Land and CHIRPS daily rainfall",
        fontsize=13,
    )

    # Monthly totals through time.
    monthly = rows.set_index("date").resample("ME").agg(
        {"rainfall_era5": "sum", "rainfall_chirps": "sum"}
    )

    axes[0][0].plot(
        monthly.index, monthly["rainfall_era5"], label="ERA5-Land", linewidth=1
    )
    axes[0][0].plot(
        monthly.index, monthly["rainfall_chirps"], label="CHIRPS", linewidth=1
    )
    axes[0][0].set_title("Monthly rainfall total")
    axes[0][0].set_ylabel("mm/month")
    axes[0][0].legend()

    # Mean seasonal cycle.
    seasonal = rows.groupby(rows["date"].dt.month).agg(
        {"rainfall_era5": "mean", "rainfall_chirps": "mean"}
    )

    width = 0.4
    months = seasonal.index.to_numpy()

    axes[0][1].bar(
        months - width / 2, seasonal["rainfall_era5"], width, label="ERA5-Land"
    )
    axes[0][1].bar(
        months + width / 2, seasonal["rainfall_chirps"], width, label="CHIRPS"
    )
    axes[0][1].set_title("Mean seasonal cycle")
    axes[0][1].set_xlabel("Month")
    axes[0][1].set_ylabel("Mean mm/day")
    axes[0][1].set_xticks(range(1, 13))
    axes[0][1].legend()

    # Daily agreement.
    if not both.empty:
        axes[1][0].scatter(
            both["rainfall_chirps"],
            both["rainfall_era5"],
            s=4,
            alpha=0.25,
            edgecolors="none",
        )

        limit = float(
            max(
                both["rainfall_chirps"].max(),
                both["rainfall_era5"].max(),
                1.0,
            )
        )

        axes[1][0].plot([0, limit], [0, limit], linestyle="--", linewidth=1)
        axes[1][0].set_xlim(0, limit)
        axes[1][0].set_ylim(0, limit)

    axes[1][0].set_title("Daily rainfall, ERA5 against CHIRPS")
    axes[1][0].set_xlabel("CHIRPS (mm/day)")
    axes[1][0].set_ylabel("ERA5-Land (mm/day)")

    # Wet-day distribution, log scale because rainfall is heavily skewed.
    wet_era5 = both.loc[
        both["rainfall_era5"] >= WET_DAY_THRESHOLD_MM, "rainfall_era5"
    ]
    wet_chirps = both.loc[
        both["rainfall_chirps"] >= WET_DAY_THRESHOLD_MM, "rainfall_chirps"
    ]

    if len(wet_era5) and len(wet_chirps):
        bins = np.logspace(
            0,
            np.log10(
                max(float(wet_era5.max()), float(wet_chirps.max()), 10.0)
            ),
            40,
        )

        axes[1][1].hist(
            wet_era5, bins=bins, alpha=0.55, label="ERA5-Land"
        )
        axes[1][1].hist(wet_chirps, bins=bins, alpha=0.55, label="CHIRPS")
        axes[1][1].set_xscale("log")
        axes[1][1].legend()

    axes[1][1].set_title(
        f"Wet-day rainfall distribution (>= {WET_DAY_THRESHOLD_MM} mm)"
    )
    axes[1][1].set_xlabel("mm/day")
    axes[1][1].set_ylabel("Days")

    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)

    plt.close(figure)


def plot_summary(comparison: pd.DataFrame, path: Path) -> None:
    """Draw the cross-district summary of correlation and bias."""

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    usable = comparison.dropna(subset=["pearson_r"])

    if usable.empty:
        return

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    figure.suptitle("ERA5-Land and CHIRPS agreement by district", fontsize=13)

    names = usable["canonical_name"].tolist()
    positions = np.arange(len(names))
    width = 0.4

    axes[0].bar(
        positions - width / 2, usable["pearson_r"], width, label="Pearson"
    )
    axes[0].bar(
        positions + width / 2, usable["spearman_r"], width, label="Spearman"
    )
    axes[0].axhline(
        STRONG_DAILY_CORRELATION, linestyle="--", linewidth=1, color="grey"
    )
    axes[0].set_title("Daily correlation")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].legend()

    axes[1].bar(
        positions - width / 2, usable["era5_mean_mm"], width, label="ERA5-Land"
    )
    axes[1].bar(
        positions + width / 2, usable["chirps_mean_mm"], width, label="CHIRPS"
    )
    axes[1].set_title("Mean daily rainfall")
    axes[1].set_ylabel("mm/day")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].legend()

    axes[2].bar(
        positions - width / 2, usable["era5_wet_days"], width, label="ERA5-Land"
    )
    axes[2].bar(
        positions + width / 2,
        usable["chirps_wet_days"],
        width,
        label="CHIRPS",
    )
    axes[2].set_title(f"Wet days (>= {WET_DAY_THRESHOLD_MM} mm)")
    axes[2].set_ylabel("Days")
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(names, rotation=30, ha="right")
    axes[2].legend()

    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)

    plt.close(figure)


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def build_recommendation(comparison: pd.DataFrame) -> dict:
    """
    Derive a recommendation from the measured agreement.

    Three outcomes, in increasing order of how much they change the model:

    quality_cross_check
        The default. CHIRPS is kept as an independent check on ERA5 and does
        not enter the model.

    additional_channel
        The two disagree enough to carry information the other lacks, but
        both are individually plausible. Feed both, and let the model weigh
        them.

    replacement_candidate
        ERA5 shows a large systematic bias against the gauge-calibrated
        product. Even here the recommendation is only a candidate: swapping
        the rainfall source is a modelling decision that needs a held-out
        comparison of dengue forecast skill, not a rainfall statistic.
    """

    usable = comparison.dropna(subset=["pearson_r"])

    if usable.empty:
        return {
            "recommendation": "insufficient_data",
            "reasons": [
                "No district had enough overlapping days to compare."
            ],
        }

    median_pearson = float(usable["pearson_r"].median())
    median_spearman = float(usable["spearman_r"].median())
    median_bias = float(usable["relative_bias"].abs().median())

    strongly_biased = usable.loc[
        usable["relative_bias"].abs() > MEAN_BIAS_TOLERANCE
    ]

    weakly_correlated = usable.loc[
        usable["pearson_r"] < MODERATE_DAILY_CORRELATION
    ]

    reasons = [
        f"Median daily Pearson correlation {median_pearson:.3f}, "
        f"Spearman {median_spearman:.3f}.",
        f"Median absolute relative bias in mean rainfall {median_bias:.1%}.",
    ]

    if len(strongly_biased):
        reasons.append(
            f"{len(strongly_biased)} of {len(usable)} districts differ in "
            f"mean rainfall by more than {MEAN_BIAS_TOLERANCE:.0%}: "
            f"{', '.join(strongly_biased['canonical_name'])}."
        )

    if len(weakly_correlated):
        reasons.append(
            f"{len(weakly_correlated)} of {len(usable)} districts correlate "
            f"below {MODERATE_DAILY_CORRELATION} on daily values: "
            f"{', '.join(weakly_correlated['canonical_name'])}."
        )

    # Correlation and bias answer different questions, and only correlation
    # decides whether CHIRPS carries information ERA5 lacks.
    #
    # A high correlation with a large bias means one product is close to a
    # rescaling of the other: they agree about which days are wet and differ
    # only in level. That is a calibration difference, correctable by
    # scaling, and it does NOT make CHIRPS an independent channel — feeding
    # both would largely duplicate one signal. It is still worth reporting,
    # because a systematic bias matters for any absolute rainfall threshold.
    #
    # A low correlation is the case that genuinely matters: the products
    # disagree about which days it rained, and no rescaling reconciles them.
    if median_pearson >= STRONG_DAILY_CORRELATION:
        recommendation = "quality_cross_check"

        if median_bias <= MEAN_BIAS_TOLERANCE:
            reasons.append(
                "The two products agree closely on both daily values and "
                "totals, so CHIRPS adds little the model cannot already see "
                "in ERA5. Keep it as an independent check."
            )
        else:
            reasons.append(
                f"Daily correlation is strong ({median_pearson:.3f}) but the "
                f"mean levels differ by {median_bias:.1%}. The products "
                "agree about which days are wet and differ mainly in level, "
                "so this is a calibration difference rather than independent "
                "information: one series is close to a rescaling of the "
                "other. Feeding both would largely duplicate one signal. "
                "Keep CHIRPS as a cross-check, and note the bias when "
                "choosing any absolute rainfall threshold."
            )
    elif median_pearson >= MODERATE_DAILY_CORRELATION:
        recommendation = "additional_channel"

        reasons.append(
            "The products agree on the broad signal but differ enough on "
            "individual days that each carries information the other does "
            "not, and the difference is not a simple rescaling. Carrying "
            "both as separate channels lets the model use that difference "
            "rather than forcing a choice."
        )
    else:
        recommendation = "replacement_candidate"

        reasons.append(
            "Daily agreement is weak. Because CHIRPS is gauge-calibrated and "
            "roughly five times finer, it is the more credible rainfall "
            "estimate where the two conflict. Treat replacement as a "
            "candidate to be tested against dengue forecast skill on a "
            "held-out period, never as an automatic substitution."
        )

    return {
        "recommendation": recommendation,
        "median_pearson": round(median_pearson, 4),
        "median_spearman": round(median_spearman, 4),
        "median_absolute_bias": round(median_bias, 4),
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(
    comparison: pd.DataFrame,
    monthly: pd.DataFrame,
    heavy: pd.DataFrame,
    recommendation: dict,
    districts: list[str],
) -> str:
    """Build the markdown comparison report."""

    labels = {
        "quality_cross_check": "Quality cross-check only",
        "additional_channel": "Additional climate channel",
        "replacement_candidate": "Replacement candidate, to be tested",
        "insufficient_data": "Insufficient data",
    }

    lines = [
        "# ERA5-Land and CHIRPS rainfall comparison",
        "",
        "Neither source is treated as truth. ERA5-Land is a reanalysis: "
        "physically consistent and complete by construction, but modelled. "
        "CHIRPS is satellite infrared calibrated against rain gauges: closer "
        "to observation and finer at 0.05 degrees, but with retrieval gaps "
        "and an uneven gauge network.",
        "",
        "**This comparison changes nothing on its own.** ERA5 rainfall "
        "remains the extracted series; no substitution is applied.",
        "",
        "## Recommendation",
        "",
        f"**{labels.get(recommendation['recommendation'], 'Unknown')}**",
        "",
    ]

    for reason in recommendation["reasons"]:
        lines.append(f"- {reason}")

    lines += [
        "",
        "### What each outcome would mean",
        "",
        "| Outcome | Meaning |",
        "| --- | --- |",
        "| Quality cross-check | CHIRPS stays outside the model and is used "
        "to audit ERA5 rainfall. |",
        "| Additional channel | Both series feed the model as separate "
        "features; the model weighs them. |",
        "| Replacement candidate | CHIRPS *may* be the better rainfall "
        "source, to be decided by held-out dengue forecast skill, never by "
        "rainfall statistics alone. |",
        "",
        "## Districts compared",
        "",
        "Chosen to span the rainfall regimes, so that agreement in one "
        "regime is not mistaken for agreement everywhere.",
        "",
        "| District | Regime | Why included |",
        "| --- | --- | --- |",
        "| Colombo | Wet zone, coastal | Highest dengue burden; smaller than "
        "one native ERA5 cell |",
        "| Gampaha | Wet zone, coastal | Adjacent to Colombo; checks local "
        "consistency |",
        "| Ratnapura | Wet zone, inland | Orographic rainfall, which "
        "reanalysis and satellite treat differently |",
        "| Jaffna | Dry zone, northern | Low totals, where relative "
        "differences are largest |",
        "| Anuradhapura | Dry zone, interior | Largest district, so grid "
        "resolution is least likely to confound |",
        "",
        "## Comparison statistics",
        "",
    ]

    display_columns = [
        column
        for column in [
            "canonical_name",
            "days_both_present",
            "pearson_r",
            "spearman_r",
            "era5_mean_mm",
            "chirps_mean_mm",
            "relative_bias",
            "era5_wet_days",
            "chirps_wet_days",
        ]
        if column in comparison.columns
    ]

    lines += [
        comparison[display_columns].to_markdown(index=False),
        "",
        "Pearson measures agreement on magnitude and Spearman on ordering. "
        "Rainfall is heavily skewed, so a large gap between the two means a "
        "few extreme days dominate the Pearson value.",
        "",
        "## Missingness",
        "",
    ]

    missing_columns = [
        column
        for column in [
            "canonical_name",
            "total_days",
            "days_both_present",
            "era5_missing_days",
            "chirps_missing_days",
        ]
        if column in comparison.columns
    ]

    lines += [
        comparison[missing_columns].to_markdown(index=False),
        "",
        "Correlations use only days where both products report a value. "
        "These counts show how many days that excluded, so a high "
        "correlation on a small overlap is not mistaken for broad agreement.",
        "",
        "## Heavy rainfall events",
        "",
        f"Days where either product reported at least {HEAVY_RAINFALL_MM} mm. "
        "Whether the two agree on the *same* heavy days matters more than "
        "whether they report the same number of them.",
        "",
    ]

    if heavy.empty:
        lines += ["No heavy-rainfall days in the compared districts.", ""]
    else:
        counts = heavy["agreement"].value_counts()

        lines += [
            "| Agreement | Days |",
            "| --- | --- |",
        ]

        for label in ["both", "era5 only", "chirps only"]:
            lines.append(f"| {label} | {int(counts.get(label, 0))} |")

        lines += [
            "",
            f"Full list: `heavy_rainfall_events.csv` "
            f"({len(heavy)} rows).",
            "",
        ]

    lines += [
        "## Seasonal cycle",
        "",
        "Sri Lanka has two monsoons. A product that reproduces the annual "
        "total but misplaces the seasonal peak would mislead a dengue model "
        "driven by lagged rainfall, so the monthly means are compared "
        "separately in `monthly_rainfall_comparison.csv`.",
        "",
    ]

    if not monthly.empty:
        pivot = monthly.pivot_table(
            index="month",
            columns="canonical_name",
            values="difference_mm",
        ).round(2)

        lines += [
            "Mean daily difference by month (ERA5 minus CHIRPS, mm/day):",
            "",
            pivot.to_markdown(),
            "",
        ]

    lines += [
        "## Plots",
        "",
        "One four-panel figure per district under `plots/`: monthly totals, "
        "mean seasonal cycle, daily scatter against the 1:1 line, and the "
        "wet-day distribution. Together these separate a constant bias from "
        "a seasonal misplacement from a disagreement confined to extremes.",
        "",
        "## Method notes",
        "",
        f"- A day counts as wet at {WET_DAY_THRESHOLD_MM} mm. Both products "
        "produce many near-zero values that are numerical drizzle rather "
        "than rain; counting those would compare rounding behaviour.",
        f"- Heavy rainfall is {HEAVY_RAINFALL_MM} mm and extreme is "
        f"{EXTREME_RAINFALL_MM} mm.",
        "- Both series are aggregated over identical GADM polygons at "
        "identical scale, so a difference is a property of the rainfall "
        "products, not of the extraction.",
        "- The pairing is an outer join. An inner join would hide the "
        "missingness this comparison is meant to measure.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_arguments():
    """Parse the command line."""

    parser = argparse.ArgumentParser(
        description="Compare ERA5-Land and CHIRPS daily rainfall."
    )

    parser.add_argument(
        "--all-districts",
        action="store_true",
        help="Compare all 25 districts instead of the five focus districts.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip the figures.",
    )

    return parser.parse_args()


def main() -> int:
    """Compare the two rainfall products and write the report."""

    arguments = parse_arguments()

    try:
        paired = load_paired_rainfall()
    except FileNotFoundError as error:
        print(error)
        return 1

    districts = (
        sorted(paired["canonical_name"].dropna().unique())
        if arguments.all_districts
        else FOCUS_DISTRICTS
    )

    print(f"Paired rows:  {len(paired)}")
    print(f"Districts:    {len(districts)}")

    comparison = build_district_comparison(paired, districts)
    monthly = build_monthly_comparison(paired, districts)
    heavy = build_heavy_events(paired, districts)

    recommendation = build_recommendation(comparison)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(DISTRICT_COMPARISON_PATH, index=False)
    monthly.to_csv(MONTHLY_COMPARISON_PATH, index=False)

    heavy_output = heavy.copy()

    if not heavy_output.empty:
        heavy_output["date"] = heavy_output["date"].dt.strftime("%Y-%m-%d")

    heavy_output.to_csv(HEAVY_EVENTS_PATH, index=False)

    if not arguments.no_plots:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        for district in districts:
            plot_district(
                paired,
                district,
                PLOTS_DIR / f"{district.replace(' ', '_')}_comparison.png",
            )

        plot_summary(comparison, PLOTS_DIR / "all_districts_summary.png")

        print(f"Plots:        {len(districts) + 1}")

    REPORT_PATH.write_text(
        build_report(comparison, monthly, heavy, recommendation, districts),
        encoding="utf-8",
    )

    print("\nComparison")
    print("-" * 62)

    display = [
        column
        for column in [
            "canonical_name",
            "pearson_r",
            "spearman_r",
            "era5_mean_mm",
            "chirps_mean_mm",
            "relative_bias",
        ]
        if column in comparison.columns
    ]

    print(comparison[display].to_string(index=False))

    print(f"\nRecommendation: {recommendation['recommendation']}")

    for reason in recommendation["reasons"]:
        print(f"  - {reason}")

    print(f"\nWrote {RESULTS_DIR.relative_to(PROJECT_DIR)}")

    print(
        "\nERA5 rainfall is unchanged. No substitution has been applied."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

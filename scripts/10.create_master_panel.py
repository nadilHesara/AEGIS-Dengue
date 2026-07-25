"""
Build the master weekly modelling panel.

The panel is a complete structural grid of every reporting period crossed with
all 25 canonical districts. Observed dengue cases and reporting-period
climate are then left-joined onto that grid using period_id and node_id only.

Missing observations are preserved as missing. No dengue target is interpolated
and no synthetic weather is created.

Outputs:
    data/processed/panel_weekly.parquet
    results/data_validation/master_panel_report.md
    results/data_validation/master_panel_missingness.csv
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent

CALENDAR_PATH = PROJECT_DIR / "data" / "interim" / "reporting_calendar.csv"
NODES_PATH = PROJECT_DIR / "data" / "processed" / "nodes.csv"
DENGUE_PATH = PROJECT_DIR / "data" / "interim" / "dengue_weekly_canonical.parquet"
CLIMATE_PATH = PROJECT_DIR / "data" / "interim" / "climate_by_dengue_period.parquet"

OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "panel_weekly.parquet"
REPORT_PATH = PROJECT_DIR / "results" / "data_validation" / "master_panel_report.md"
MISSINGNESS_PATH = PROJECT_DIR / "results" / "data_validation" / "master_panel_missingness.csv"

PANEL_VERSION = "data-v1"

CALENDAR_COLUMNS = [
    "period_id",
    "source_year",
    "source_week",
    "start_date",
    "end_date",
    "reporting_days",
    "weekday_convention",
    "is_weekday_convention_change",
    "is_irregular_period",
    "calendar_gap_before_days",
    "has_calendar_gap_before",
]

NODE_COLUMNS = ["node_id", "canonical_name", "province"]

DENGUE_COLUMNS = [
    "period_id",
    "node_id",
    "cases",
    "case_observed",
    "source_reporting_areas_used",
    "source_row_count",
]

CLIMATE_COLUMNS = [
    "period_id",
    "node_id",
    "rainfall_sum_mm",
    "rainfall_daily_mean_mm",
    "rainy_days",
    "temperature_mean_c",
    "temperature_min_c",
    "temperature_max_c",
    "dewpoint_mean_c",
    "relative_humidity_mean",
    "wind_speed_mean",
    "weather_days_available",
    "expected_weather_days",
    "weather_coverage_ratio",
    "weather_complete",
]

PANEL_COLUMNS = [
    "period_id",
    "source_year",
    "source_week",
    "start_date",
    "end_date",
    "reporting_days",
    "weekday_convention",
    "is_weekday_convention_change",
    "is_irregular_period",
    "calendar_gap_before_days",
    "has_calendar_gap_before",
    "node_id",
    "canonical_name",
    "province",
    "cases",
    "case_observed",
    "rainfall_sum_mm",
    "rainfall_daily_mean_mm",
    "rainy_days",
    "temperature_mean_c",
    "temperature_min_c",
    "temperature_max_c",
    "dewpoint_mean_c",
    "relative_humidity_mean",
    "wind_speed_mean",
    "weather_days_available",
    "weather_coverage_ratio",
    "weather_complete",
    "row_has_complete_cases",
    "row_has_complete_climate",
    "row_is_fully_observed",
    "is_covid_window",
    "is_2017_outbreak",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_calendar(path: Path = CALENDAR_PATH) -> pd.DataFrame:
    """Load the reporting calendar used as the master time index."""

    calendar = pd.read_csv(path)

    for column in ["start_date", "end_date"]:
        calendar[column] = pd.to_datetime(calendar[column])

    return calendar[CALENDAR_COLUMNS].copy()


def load_nodes(path: Path = NODES_PATH) -> pd.DataFrame:
    """Load the canonical district registry."""

    return pd.read_csv(path)[NODE_COLUMNS].copy()


def load_dengue(path: Path = DENGUE_PATH) -> pd.DataFrame:
    """Load the canonical weekly dengue observations."""

    dengue = pd.read_parquet(path)

    dengue["start_date"] = pd.to_datetime(dengue["start_date"])
    dengue["end_date"] = pd.to_datetime(dengue["end_date"])
    dengue["cases"] = pd.to_numeric(dengue["cases"], errors="coerce")
    dengue["case_observed"] = pd.to_numeric(
        dengue["case_observed"], errors="coerce"
    ).fillna(0).astype(int)

    return dengue[DENGUE_COLUMNS].copy()


def load_climate(path: Path = CLIMATE_PATH) -> pd.DataFrame:
    """Load the reporting-period climate observations."""

    climate = pd.read_parquet(path)

    climate["start_date"] = pd.to_datetime(climate["start_date"])
    climate["end_date"] = pd.to_datetime(climate["end_date"])

    return climate[CLIMATE_COLUMNS].copy()


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def build_structural_grid(calendar: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Return every period_id × canonical district combination."""

    grid = calendar.merge(nodes, how="cross")
    grid = grid.sort_values(["period_id", "node_id"]).reset_index(drop=True)

    expected_rows = len(calendar) * len(nodes)
    if len(grid) != expected_rows:
        raise ValueError(
            f"Structural grid has {len(grid)} rows, expected {expected_rows}."
        )

    return grid


def merge_dengue(grid: pd.DataFrame, dengue: pd.DataFrame) -> pd.DataFrame:
    """Left-join observed dengue cases onto the structural grid."""

    return grid.merge(
        dengue,
        on=["period_id", "node_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_dengue"),
    )


def merge_climate(panel: pd.DataFrame, climate: pd.DataFrame) -> pd.DataFrame:
    """Left-join reporting-period climate onto the panel."""

    return panel.merge(
        climate,
        on=["period_id", "node_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_climate"),
    )


def add_flags(panel: pd.DataFrame) -> pd.DataFrame:
    """Add observation flags and modeling windows."""

    result = panel.copy()

    result["case_observed"] = result["cases"].notna().astype(int)
    result["row_has_complete_cases"] = result["case_observed"].astype(int)
    result["row_has_complete_climate"] = result["weather_complete"].fillna(
        False
    ).astype(int)
    result["row_is_fully_observed"] = (
        result["row_has_complete_cases"].astype(bool)
        & result["row_has_complete_climate"].astype(bool)
    ).astype(int)

    result["is_covid_window"] = result["start_date"].between(
        pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"), inclusive="both"
    ).astype(int)

    result["is_2017_outbreak"] = result["start_date"].between(
        pd.Timestamp("2017-01-01"), pd.Timestamp("2017-12-31"), inclusive="both"
    ).astype(int)

    return result


def build_master_panel(
    calendar: pd.DataFrame,
    nodes: pd.DataFrame,
    dengue: pd.DataFrame,
    climate: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full modelling panel with structural missing rows preserved."""

    grid = build_structural_grid(calendar, nodes)
    panel = merge_dengue(grid, dengue)
    panel = merge_climate(panel, climate)
    panel = add_flags(panel)

    duplicates = panel.duplicated(subset=["period_id", "node_id"])
    if duplicates.any():
        raise ValueError("The master panel contains duplicate period_id × node_id rows.")

    period_sizes = panel.groupby("period_id")["node_id"].size()
    if not period_sizes.eq(len(nodes)).all():
        bad_periods = period_sizes.loc[~period_sizes.eq(len(nodes))]
        raise ValueError(
            "Some periods do not contain exactly 25 structural rows: "
            f"{bad_periods.to_dict()}"
        )

    case_counts = panel.groupby("period_id")["case_observed"].sum()
    if not case_counts.eq(len(nodes)).all():
        incomplete = case_counts.loc[~case_counts.eq(len(nodes))]
        if len(incomplete) != 1:
            raise ValueError(
                "Observed case counts are incomplete for unexpected periods: "
                f"{incomplete.to_dict()}"
            )

    expected_period = calendar.loc[
        (calendar["source_year"].eq(2026)) & (calendar["source_week"].eq(7))
    ]

    if not expected_period.empty:
        week_7_period_id = int(expected_period.iloc[0]["period_id"])
        week_7 = panel.loc[panel["period_id"].eq(week_7_period_id)]

        if len(week_7) != len(nodes):
            raise ValueError("2026 week 7 does not contain 25 structural rows.")

        if int(week_7["case_observed"].sum()) != 24:
            raise ValueError(
                "2026 week 7 does not contain 24 observed case rows."
            )

        puttalam = week_7.loc[week_7["canonical_name"].eq("Puttalam")].iloc[0]
        if pd.notna(puttalam["cases"]):
            raise ValueError(
                "Puttalam in 2026 week 7 must remain missing, not zero."
            )

        if int(puttalam["case_observed"]) != 0:
            raise ValueError(
                "Puttalam in 2026 week 7 must have case_observed = 0."
            )

    panel = panel[PANEL_COLUMNS].sort_values(["period_id", "node_id"]).reset_index(
        drop=True
    )

    return panel


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def get_git_commit_hash() -> str:
    """Return the current git commit hash for the frozen panel version."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"

    return completed.stdout.strip()


def build_missingness_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Return one row per structurally present but incomplete observation."""

    missing = panel.loc[~panel["row_is_fully_observed"].eq(1)].copy()

    if missing.empty:
        return pd.DataFrame(
            columns=[
                "period_id",
                "source_year",
                "source_week",
                "node_id",
                "canonical_name",
                "cases",
                "case_observed",
                "weather_complete",
                "row_has_complete_cases",
                "row_has_complete_climate",
                "row_is_fully_observed",
                "missing_components",
            ]
        )

    missing_components = []
    for _, row in missing.iterrows():
        components = []
        if pd.isna(row["cases"]):
            components.append("cases")
        if not bool(row["row_has_complete_climate"]):
            components.append("climate")
        missing_components.append("|".join(components))

    missing["missing_components"] = missing_components

    return missing[
        [
            "period_id",
            "source_year",
            "source_week",
            "node_id",
            "canonical_name",
            "cases",
            "case_observed",
            "weather_complete",
            "row_has_complete_cases",
            "row_has_complete_climate",
            "row_is_fully_observed",
            "missing_components",
        ]
    ].sort_values(["period_id", "node_id"]).reset_index(drop=True)


def write_report(
    panel: pd.DataFrame,
    missingness: pd.DataFrame,
    commit_hash: str,
) -> None:
    """Write the markdown summary for the frozen master panel."""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    complete_case_rows = int(panel["row_has_complete_cases"].sum())
    complete_climate_rows = int(panel["row_has_complete_climate"].sum())
    fully_observed_rows = int(panel["row_is_fully_observed"].sum())

    week_7 = panel.loc[
        panel["source_year"].eq(2026) & panel["source_week"].eq(7)
    ]

    week_7_note = "not present in the provided calendar"
    if not week_7.empty:
        week_7_note = str(int(week_7["case_observed"].sum()))

    lines = [
        "# Master weekly modelling panel",
        "",
        f"Frozen version: `{PANEL_VERSION}`",
        f"Creation commit: `{commit_hash}`",
        "",
        f"Structural rows: {len(panel)}",
        f"Reporting periods: {panel['period_id'].nunique()}",
        f"Canonical districts: {panel['node_id'].nunique()}",
        f"Observed case rows: {complete_case_rows}",
        f"Observed climate rows: {complete_climate_rows}",
        f"Fully observed rows: {fully_observed_rows}",
        f"Incomplete rows written to: `{MISSINGNESS_PATH.relative_to(PROJECT_DIR)}`",
        "",
        "## Required checks",
        "",
        f"- Every period contains exactly 25 structural rows: `{bool(panel.groupby('period_id')['node_id'].size().eq(25).all())}`",
        f"- 2026 week 7 observed case rows: `{week_7_note}`",
        "",
        "## Output files",
        "",
        f"- `{OUTPUT_PATH.relative_to(PROJECT_DIR)}`",
        f"- `{MISSINGNESS_PATH.relative_to(PROJECT_DIR)}`",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Build the master modelling panel and write the outputs."""

    calendar = load_calendar()
    nodes = load_nodes()
    dengue = load_dengue()
    climate = load_climate()

    panel = build_master_panel(calendar, nodes, dengue, climate)
    missingness = build_missingness_report(panel)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT_PATH, index=False)

    missingness.to_csv(MISSINGNESS_PATH, index=False)

    write_report(panel, missingness, get_git_commit_hash())

    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {MISSINGNESS_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

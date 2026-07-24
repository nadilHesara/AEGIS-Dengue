"""
Aggregate daily district weather onto the dengue reporting periods.

Stage 2 of the climate pipeline. Daily rows from the ERA5 extraction are
grouped onto the exact reporting intervals defined by the calendar, producing
one row per period_id and canonical district.

The alignment rule is the whole point of this script:

    start_date <= date <= end_date

on the closed interval of each reporting period, taken from the calendar.
Nothing here derives a week from a label. Not the raw year and week, not the
ISO week, not a fixed Saturday-Friday or Monday-Sunday window. Those all
disagree with the source calendar somewhere: two 2009 periods are 8 and 6
days long, the reporting weekday changes permanently in 2026, and the
interval labelled 2026 week 53 falls in December 2025.

Two consequences follow, and both are handled explicitly:

    Interval length varies, so summed rainfall is not comparable across
    periods without knowing the length. Both the sum and the daily mean are
    retained, along with reporting_days.

    2025-12-27 and 2025-12-28 belong to no reporting period. They stay in the
    raw daily file and are simply never selected here, because no interval
    contains them. They are not attributed to either neighbouring period.

No lag features and no shift() in this stage: every value describes its own
period. Lags belong to stage 3, where they are aligned to a forecast origin.

Missing weather is reported, never filled.

Outputs:
    data/interim/climate_by_dengue_period.parquet
    results/data_validation/climate_period_coverage.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent

CLIMATE_PATH = PROJECT_DIR / "data" / "raw" / "climate_daily_district.csv"
CALENDAR_PATH = PROJECT_DIR / "data" / "interim" / "reporting_calendar.csv"
NODES_PATH = PROJECT_DIR / "data" / "processed" / "nodes.csv"

INTERIM_DIR = PROJECT_DIR / "data" / "interim"
RESULTS_DIR = PROJECT_DIR / "results" / "data_validation"

OUTPUT_PATH = INTERIM_DIR / "climate_by_dengue_period.parquet"
COVERAGE_PATH = RESULTS_DIR / "climate_period_coverage.csv"

EXPECTED_DISTRICTS = 25

# A day counts as rainy at 1 mm or more. Both ERA5 and CHIRPS produce many
# near-zero values that are numerical drizzle rather than rain, so counting
# any trace would make rainy_days a measure of rounding behaviour. The same
# threshold is used in 8.compare_era5_chirps.py, so "rainy day" means one
# thing across the project.
RAINY_DAY_THRESHOLD_MM = 1.0

# A period below this coverage is reported as incomplete. It is still
# written, with weather_complete False, so the decision to use or drop it
# stays with the modelling stage rather than being made silently here.
COMPLETE_COVERAGE_RATIO = 1.0

CALENDAR_COLUMNS = [
    "period_id",
    "source_year",
    "source_week",
    "start_date",
    "end_date",
    "reporting_days",
    "weekday_convention",
    "is_irregular_period",
    "calendar_gap_before_days",
]

DAILY_COLUMNS = [
    "date",
    "node_id",
    "canonical_name",
    "rainfall_mm",
    "temperature_mean_c",
    "temperature_min_c",
    "temperature_max_c",
    "dewpoint_mean_c",
    "relative_humidity_mean",
    "wind_speed_mean",
]

OUTPUT_COLUMNS = [
    "period_id",
    "source_year",
    "source_week",
    "start_date",
    "end_date",
    "reporting_days",
    "weekday_convention",
    "is_irregular_period",
    "calendar_gap_before_days",
    "node_id",
    "canonical_name",
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

# Columns that must be present on a daily row for it to count towards
# coverage. A row missing any of these is not a usable weather day.
REQUIRED_DAILY_VALUES = [
    "rainfall_mm",
    "temperature_mean_c",
    "temperature_min_c",
    "temperature_max_c",
    "dewpoint_mean_c",
    "relative_humidity_mean",
    "wind_speed_mean",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_daily_climate(path: Path = CLIMATE_PATH) -> pd.DataFrame:
    """Load the daily district weather."""

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist.\n"
            "Run scripts/5.extract_era5_daily.py first."
        )

    climate = pd.read_csv(path)

    climate["date"] = pd.to_datetime(climate["date"])

    return climate


def load_calendar(path: Path = CALENDAR_PATH) -> pd.DataFrame:
    """Load the reporting calendar."""

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist.\n"
            "Run scripts/2.create_calendar.py first."
        )

    calendar = pd.read_csv(path)

    for column in ["start_date", "end_date"]:
        calendar[column] = pd.to_datetime(calendar[column])

    return calendar[CALENDAR_COLUMNS].copy()


def load_nodes(path: Path = NODES_PATH) -> pd.DataFrame:
    """Load the permanent district registry, in canonical node order."""

    nodes = pd.read_csv(path)

    return nodes[["node_id", "canonical_name"]].sort_values(
        "node_id"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Interval assignment
# ---------------------------------------------------------------------------

def assign_periods(
    daily: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach period_id to each daily row by exact interval containment.

    A daily row belongs to the period whose closed interval contains its
    date. Because the calendar's intervals never overlap, a date matches at
    most one period; because the intervals do not tile the calendar, some
    dates match none. Both properties are asserted downstream.

    Dates outside every interval, including 2025-12-27 and 2025-12-28, are
    dropped here. That is the intended behaviour: they belong to no dengue
    reporting period and must not be attributed to a neighbouring one.
    """

    # merge_asof matches each date to the latest period starting on or before
    # it, which is the only candidate an ordered, non-overlapping calendar can
    # have. The end_date test then rejects dates that fall in a gap.
    ordered_daily = daily.sort_values("date")
    ordered_calendar = calendar.sort_values("start_date")

    matched = pd.merge_asof(
        ordered_daily,
        ordered_calendar,
        left_on="date",
        right_on="start_date",
        direction="backward",
    )

    inside = (
        matched["start_date"].notna()
        & matched["date"].ge(matched["start_date"])
        & matched["date"].le(matched["end_date"])
    )

    return matched.loc[inside].copy()


def find_unassigned_dates(
    daily: pd.DataFrame,
    assigned: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the dates present in the daily data that no period covers.

    Expected rather than exceptional: weather is extracted for a window that
    starts before the first reporting period, and the two December 2025 days
    fall in the documented calendar gap.
    """

    all_dates = set(daily["date"].dropna().unique())
    used_dates = set(assigned["date"].dropna().unique())

    unassigned = sorted(all_dates - used_dates)

    if not unassigned:
        return pd.DataFrame(columns=["date", "reason"])

    calendar_start = assigned["start_date"].min()
    calendar_end = assigned["end_date"].max()

    def _reason(date):
        if pd.notna(calendar_start) and date < calendar_start:
            return "before the first reporting period"

        if pd.notna(calendar_end) and date > calendar_end:
            return "after the last reporting period"

        return "inside a documented calendar gap"

    return pd.DataFrame(
        {
            "date": unassigned,
            "reason": [_reason(date) for date in unassigned],
        }
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_to_periods(
    assigned: pd.DataFrame,
    calendar: pd.DataFrame,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reduce the assigned daily rows to one row per period and district.

    Each variable takes the statistic that is correct for it, not a blanket
    mean:

        rainfall            summed, because rainfall accumulates over the
                            interval, and separately averaged per day so the
                            6, 7 and 8-day periods stay comparable
        rainy_days          counted above the documented threshold
        temperature min/max the extreme over the interval, not the mean of
                            the daily extremes, which would understate the
                            range
        everything else     the mean of the daily values

    Only days carrying every required value contribute. A day with a null
    temperature is not counted as available, so weather_coverage_ratio
    reflects usable days rather than merely present rows.
    """

    usable = assigned.dropna(subset=REQUIRED_DAILY_VALUES).copy()

    usable["is_rainy_day"] = usable["rainfall_mm"].ge(
        RAINY_DAY_THRESHOLD_MM
    )

    aggregated = (
        usable.groupby(["period_id", "canonical_name"], as_index=False)
        .agg(
            rainfall_sum_mm=("rainfall_mm", "sum"),
            rainfall_daily_mean_mm=("rainfall_mm", "mean"),
            rainy_days=("is_rainy_day", "sum"),
            temperature_mean_c=("temperature_mean_c", "mean"),
            temperature_min_c=("temperature_min_c", "min"),
            temperature_max_c=("temperature_max_c", "max"),
            dewpoint_mean_c=("dewpoint_mean_c", "mean"),
            relative_humidity_mean=("relative_humidity_mean", "mean"),
            wind_speed_mean=("wind_speed_mean", "mean"),
            weather_days_available=("date", "nunique"),
        )
    )

    aggregated["rainy_days"] = aggregated["rainy_days"].astype(int)

    # The output grid is every period crossed with every district, so a
    # district with no usable weather in a period still appears, with null
    # values and a coverage ratio of zero. Silently omitting it would leave a
    # hole that a later join could not distinguish from a missing period.
    grid = calendar.merge(nodes, how="cross")

    result = grid.merge(
        aggregated, on=["period_id", "canonical_name"], how="left"
    )

    result["weather_days_available"] = (
        result["weather_days_available"].fillna(0).astype(int)
    )

    result["rainy_days"] = result["rainy_days"].fillna(0).astype(int)

    result["expected_weather_days"] = result["reporting_days"]

    result["weather_coverage_ratio"] = (
        result["weather_days_available"] / result["expected_weather_days"]
    ).round(6)

    result["weather_complete"] = result["weather_coverage_ratio"].ge(
        COMPLETE_COVERAGE_RATIO
    )

    # period_id is the chronological key and node_id is the permanent model
    # order, so the output is sorted by both.
    return result[OUTPUT_COLUMNS].sort_values(
        ["period_id", "node_id"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def build_coverage_report(climate: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise weather coverage per reporting period.

    Keyed by period so that a period with patchy weather across many
    districts is as visible as one district missing entirely.
    """

    coverage = (
        climate.groupby(
            [
                "period_id",
                "source_year",
                "source_week",
                "start_date",
                "end_date",
                "reporting_days",
                "is_irregular_period",
            ],
            as_index=False,
        )
        .agg(
            districts=("canonical_name", "nunique"),
            districts_complete=("weather_complete", "sum"),
            weather_days_available=("weather_days_available", "sum"),
            min_coverage_ratio=("weather_coverage_ratio", "min"),
            mean_coverage_ratio=("weather_coverage_ratio", "mean"),
        )
    )

    coverage["expected_weather_days"] = (
        coverage["reporting_days"] * coverage["districts"]
    )

    coverage["districts_incomplete"] = (
        coverage["districts"] - coverage["districts_complete"]
    )

    coverage["period_complete"] = coverage["districts_incomplete"].eq(0)

    for column in ["min_coverage_ratio", "mean_coverage_ratio"]:
        coverage[column] = coverage[column].round(6)

    return coverage[
        [
            "period_id",
            "source_year",
            "source_week",
            "start_date",
            "end_date",
            "reporting_days",
            "is_irregular_period",
            "districts",
            "districts_complete",
            "districts_incomplete",
            "weather_days_available",
            "expected_weather_days",
            "min_coverage_ratio",
            "mean_coverage_ratio",
            "period_complete",
        ]
    ].sort_values("period_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def run_assertions(
    daily: pd.DataFrame,
    assigned: pd.DataFrame,
    climate: pd.DataFrame,
    calendar: pd.DataFrame,
    nodes: pd.DataFrame,
) -> None:
    """
    Verify the alignment before the output is written.

    These check the property the whole script exists to guarantee: that every
    aggregated value came from inside its own reporting interval.
    """

    # One row per period and district, with no duplicates and none missing.
    assert not climate.duplicated(
        subset=["period_id", "canonical_name"]
    ).any(), "a period and district pair is duplicated"

    assert len(climate) == len(calendar) * len(nodes), (
        f"expected {len(calendar) * len(nodes)} rows, found {len(climate)}"
    )

    # Every district follows the canonical node order.
    assert set(climate["canonical_name"]) == set(nodes["canonical_name"]), (
        "the districts differ from the node registry"
    )

    node_lookup = dict(zip(nodes["canonical_name"], nodes["node_id"]))

    mapped = climate["canonical_name"].map(node_lookup)

    assert climate["node_id"].equals(mapped.astype(climate["node_id"].dtype)), (
        "a district carries the wrong node_id"
    )

    for period_id, group in climate.groupby("period_id"):
        assert group["node_id"].is_monotonic_increasing, (
            f"period {period_id} is not in canonical node order"
        )

    # Every assigned daily row lies inside its period's closed interval.
    if not assigned.empty:
        outside = assigned.loc[
            assigned["date"].lt(assigned["start_date"])
            | assigned["date"].gt(assigned["end_date"])
        ]

        assert outside.empty, (
            f"{len(outside)} daily rows fall outside their reporting interval"
        )

        # A date can belong to at most one period.
        per_date = assigned.groupby(["date", "canonical_name"])[
            "period_id"
        ].nunique()

        assert per_date.le(1).all(), (
            "a date was assigned to more than one reporting period"
        )

    # The documented gap is never attributed to a period.
    for gap_date in ["2025-12-27", "2025-12-28"]:
        stamp = pd.Timestamp(gap_date)

        if stamp in set(daily["date"]):
            assert stamp not in set(assigned["date"]), (
                f"{gap_date} lies in the calendar gap but was assigned "
                "to a reporting period"
            )

    # Availability can never exceed the interval length.
    assert climate["weather_days_available"].le(
        climate["expected_weather_days"]
    ).all(), "a period holds more weather days than its interval has"

    assert climate["expected_weather_days"].equals(
        climate["reporting_days"]
    ), "expected_weather_days does not equal reporting_days"

    # Irregular periods keep their true lengths.
    for start, end, days in [
        ("2009-04-18", "2009-04-25", 8),
        ("2009-05-24", "2009-05-29", 6),
    ]:
        rows = climate.loc[climate["start_date"].eq(pd.Timestamp(start))]

        if rows.empty:
            continue

        assert rows["reporting_days"].eq(days).all(), (
            f"the period starting {start} is not {days} days"
        )

        assert rows["weather_days_available"].le(days).all(), (
            f"the period starting {start} holds more than {days} weather days"
        )

    # Rainy days cannot exceed the days that were actually available.
    assert climate["rainy_days"].le(
        climate["weather_days_available"]
    ).all(), "a period reports more rainy days than available weather days"

    # A complete period's daily mean must reconstruct its sum.
    complete = climate.loc[
        climate["weather_complete"] & climate["rainfall_sum_mm"].notna()
    ]

    if not complete.empty:
        reconstructed = (
            complete["rainfall_daily_mean_mm"]
            * complete["weather_days_available"]
        )

        assert (
            (reconstructed - complete["rainfall_sum_mm"]).abs() < 1e-6
        ).all(), "rainfall sum and daily mean are inconsistent"

    # Nothing was filled: a row with no weather carries nulls, not zeros.
    empty = climate.loc[climate["weather_days_available"].eq(0)]

    if not empty.empty:
        assert empty["temperature_mean_c"].isna().all(), (
            "a period with no weather days carries a temperature value"
        )

        assert empty["rainfall_sum_mm"].isna().all(), (
            "a period with no weather days carries a rainfall value"
        )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_outputs(climate: pd.DataFrame, coverage: pd.DataFrame) -> None:
    """Write the aggregated climate and the coverage report."""

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    climate.to_parquet(OUTPUT_PATH, index=False)

    formatted = coverage.copy()

    for column in ["start_date", "end_date"]:
        formatted[column] = formatted[column].dt.strftime("%Y-%m-%d")

    formatted.to_csv(COVERAGE_PATH, index=False)


def print_summary(
    climate: pd.DataFrame,
    coverage: pd.DataFrame,
    unassigned: pd.DataFrame,
) -> None:
    """Print what the aggregation produced."""

    print("\nClimate by dengue period")
    print("-" * 62)

    print(f"Rows:                     {len(climate)}")
    print(f"Reporting periods:        {climate['period_id'].nunique()}")
    print(f"Districts:                {climate['canonical_name'].nunique()}")

    print(
        f"\nDate range:               "
        f"{climate['start_date'].min().date()} to "
        f"{climate['end_date'].max().date()}"
    )

    print("\nInterval lengths")

    for days, count in (
        climate.groupby("reporting_days")["period_id"]
        .nunique()
        .sort_index()
        .items()
    ):
        marker = "  <- irregular" if days != 7 else ""
        print(f"  {days} days: {count} periods{marker}")

    complete = int(climate["weather_complete"].sum())
    incomplete = len(climate) - complete

    print(f"\nComplete district-periods: {complete}")
    print(f"Incomplete:                {incomplete}")

    if incomplete:
        worst = climate.loc[~climate["weather_complete"]].nsmallest(
            5, "weather_coverage_ratio"
        )

        print("\nLowest coverage")

        for _, row in worst.iterrows():
            print(
                f"  period {row['period_id']:>4} "
                f"{row['canonical_name']:<14} "
                f"{row['weather_days_available']}/"
                f"{row['expected_weather_days']} days "
                f"({row['weather_coverage_ratio']:.2f})"
            )

    if not unassigned.empty:
        print(f"\nDaily dates not assigned to any period: {len(unassigned)}")

        for reason, group in unassigned.groupby("reason"):
            print(f"  {reason}: {len(group)}")

            if reason == "inside a documented calendar gap":
                for date in group["date"]:
                    print(f"    {pd.Timestamp(date).date()}")

    print(
        "\nWeather is aggregated on the exact reporting interval "
        "start_date <= date <= end_date.\nNo lags were created and no "
        "missing weather was filled."
    )


def main() -> int:
    """Aggregate daily weather onto the dengue reporting periods."""

    calendar = load_calendar()
    nodes = load_nodes()

    try:
        daily = load_daily_climate()
    except FileNotFoundError as error:
        print(error)
        return 1

    print(f"Daily weather rows:  {len(daily)}")
    print(f"Reporting periods:   {len(calendar)}")
    print(f"Districts:           {len(nodes)}")

    assigned = assign_periods(daily, calendar)

    unassigned = find_unassigned_dates(daily, assigned)

    climate = aggregate_to_periods(assigned, calendar, nodes)

    coverage = build_coverage_report(climate)

    run_assertions(daily, assigned, climate, calendar, nodes)

    write_outputs(climate, coverage)

    print_summary(climate, coverage, unassigned)

    print(f"\nWrote {OUTPUT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {COVERAGE_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

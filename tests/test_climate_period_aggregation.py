"""
Tests for aggregating daily weather onto the dengue reporting periods.

The alignment rule is the thing under test: every aggregated value must come
from inside its own reporting interval, and only from there. The tests use
the real reporting calendar so that the two 2009 irregular periods and the
documented December 2025 gap are exercised as they actually occur, rather
than as invented approximations of themselves.

Daily weather is synthesised with hand-checkable values, so a test can assert
the exact sum rather than merely that a number was produced.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "aggregate_climate",
    PROJECT_DIR / "scripts" / "9.aggregate_climate_to_periods.py",
)
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)


# The two confirmed irregular periods, and the period following the gap.
EIGHT_DAY_START = pd.Timestamp("2009-04-18")
EIGHT_DAY_END = pd.Timestamp("2009-04-25")

SIX_DAY_START = pd.Timestamp("2009-05-24")
SIX_DAY_END = pd.Timestamp("2009-05-29")

GAP_DATES = [pd.Timestamp("2025-12-27"), pd.Timestamp("2025-12-28")]

AFTER_GAP_START = pd.Timestamp("2025-12-29")


@pytest.fixture(scope="module")
def calendar():
    """The real reporting calendar."""

    return aggregate.load_calendar()


@pytest.fixture(scope="module")
def nodes():
    """The real district registry, in canonical node order."""

    return aggregate.load_nodes()


def make_daily(nodes, start, end, rainfall=2.0, districts=None):
    """
    Build synthetic daily weather over a date range.

    rainfall may be a constant or a callable taking the date, which lets a
    test construct values whose aggregate it can compute by hand.
    """

    chosen = (
        nodes if districts is None
        else nodes.loc[nodes["canonical_name"].isin(districts)]
    )

    records = []

    for day in pd.date_range(start, end, freq="D"):
        value = rainfall(day) if callable(rainfall) else rainfall

        for _, node in chosen.iterrows():
            records.append(
                {
                    "date": day,
                    "node_id": int(node["node_id"]),
                    "canonical_name": node["canonical_name"],
                    "rainfall_mm": float(value),
                    "temperature_mean_c": 27.5,
                    "temperature_min_c": 23.8,
                    "temperature_max_c": 31.4,
                    "dewpoint_mean_c": 23.1,
                    "relative_humidity_mean": 77.5,
                    "wind_speed_mean": 3.2,
                }
            )

    return pd.DataFrame(records)


def run(daily, calendar, nodes):
    """Assign and aggregate, returning both stages."""

    assigned = aggregate.assign_periods(daily, calendar)

    return assigned, aggregate.aggregate_to_periods(assigned, calendar, nodes)


# ---------------------------------------------------------------------------
# The alignment rule
# ---------------------------------------------------------------------------

def test_only_dates_inside_the_interval_are_included(calendar, nodes):
    """
    Every assigned daily row lies within its period's closed interval.

    This is the property the whole script exists to guarantee.
    """

    daily = make_daily(nodes, "2015-01-01", "2015-06-30")

    assigned, _ = run(daily, calendar, nodes)

    assert not assigned.empty

    inside = (
        assigned["date"].ge(assigned["start_date"])
        & assigned["date"].le(assigned["end_date"])
    )

    assert inside.all(), "a daily row fell outside its reporting interval"


def test_a_date_belongs_to_at_most_one_period(calendar, nodes):
    """The intervals do not overlap, so no date is counted twice."""

    daily = make_daily(nodes, "2015-01-01", "2015-12-31")

    assigned, _ = run(daily, calendar, nodes)

    per_date = assigned.groupby(["date", "canonical_name"])[
        "period_id"
    ].nunique()

    assert per_date.eq(1).all()


def test_aggregation_uses_the_exact_interval_not_a_fixed_week(
    calendar, nodes
):
    """
    A period's weather comes from its own dates, not a seven-day window.

    Rainfall is set to the day of the month, so the expected sum for any
    interval can be computed directly and compared.
    """

    daily = make_daily(
        nodes, "2009-04-11", "2009-05-02", rainfall=lambda day: day.day
    )

    _, climate = run(daily, calendar, nodes)

    row = climate.loc[
        climate["start_date"].eq(EIGHT_DAY_START)
        & climate["canonical_name"].eq("Colombo")
    ].iloc[0]

    # 2009-04-18 to 2009-04-25 inclusive.
    expected = sum(range(18, 26))

    assert row["rainfall_sum_mm"] == pytest.approx(expected)
    assert row["rainfall_daily_mean_mm"] == pytest.approx(expected / 8)


def test_weather_before_the_first_period_is_not_assigned(calendar, nodes):
    """History extracted before the calendar starts joins to nothing."""

    first_start = calendar["start_date"].min()

    daily = make_daily(
        nodes,
        first_start - pd.Timedelta(days=10),
        first_start + pd.Timedelta(days=6),
    )

    assigned, _ = run(daily, calendar, nodes)

    assert assigned["date"].min() >= first_start


# ---------------------------------------------------------------------------
# Irregular periods
# ---------------------------------------------------------------------------

def test_eight_day_interval_uses_eight_daily_records(calendar, nodes):
    """The 2009 week 17 period aggregates over eight days."""

    daily = make_daily(nodes, "2009-04-11", "2009-05-02")

    assigned, climate = run(daily, calendar, nodes)

    rows = climate.loc[climate["start_date"].eq(EIGHT_DAY_START)]

    assert len(rows) == len(nodes)
    assert rows["reporting_days"].eq(8).all()
    assert rows["expected_weather_days"].eq(8).all()
    assert rows["weather_days_available"].eq(8).all()
    assert rows["weather_coverage_ratio"].eq(1.0).all()
    assert rows["weather_complete"].all()
    assert rows["is_irregular_period"].all()

    # The dates used are exactly the eight in the interval.
    used = assigned.loc[
        assigned["start_date"].eq(EIGHT_DAY_START), "date"
    ].unique()

    assert len(used) == 8
    assert used.min() == EIGHT_DAY_START
    assert used.max() == EIGHT_DAY_END


def test_six_day_interval_uses_six_daily_records(calendar, nodes):
    """The 2009 week 22 period aggregates over six days."""

    daily = make_daily(nodes, "2009-05-17", "2009-06-05")

    assigned, climate = run(daily, calendar, nodes)

    rows = climate.loc[climate["start_date"].eq(SIX_DAY_START)]

    assert len(rows) == len(nodes)
    assert rows["reporting_days"].eq(6).all()
    assert rows["expected_weather_days"].eq(6).all()
    assert rows["weather_days_available"].eq(6).all()
    assert rows["weather_complete"].all()

    used = assigned.loc[
        assigned["start_date"].eq(SIX_DAY_START), "date"
    ].unique()

    assert len(used) == 6
    assert used.min() == SIX_DAY_START
    assert used.max() == SIX_DAY_END


def test_seven_day_interval_uses_seven_daily_records(calendar, nodes):
    """An ordinary period aggregates over seven days."""

    daily = make_daily(nodes, "2015-03-07", "2015-03-13")

    _, climate = run(daily, calendar, nodes)

    rows = climate.loc[climate["start_date"].eq(pd.Timestamp("2015-03-07"))]

    assert rows["reporting_days"].eq(7).all()
    assert rows["weather_days_available"].eq(7).all()
    assert not rows["is_irregular_period"].any()


def test_rainfall_sum_and_mean_differ_across_interval_lengths(
    calendar, nodes
):
    """
    A constant daily rainfall gives different sums but equal daily means.

    This is why both columns are retained: the sum carries interval length
    and the mean does not.
    """

    daily = make_daily(nodes, "2009-04-11", "2009-06-05", rainfall=10.0)

    _, climate = run(daily, calendar, nodes)

    eight = climate.loc[
        climate["start_date"].eq(EIGHT_DAY_START)
        & climate["canonical_name"].eq("Colombo")
    ].iloc[0]

    six = climate.loc[
        climate["start_date"].eq(SIX_DAY_START)
        & climate["canonical_name"].eq("Colombo")
    ].iloc[0]

    assert eight["rainfall_sum_mm"] == pytest.approx(80.0)
    assert six["rainfall_sum_mm"] == pytest.approx(60.0)

    # The daily means are identical, which the sums are not.
    assert eight["rainfall_daily_mean_mm"] == pytest.approx(10.0)
    assert six["rainfall_daily_mean_mm"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# The documented calendar gap
# ---------------------------------------------------------------------------

def test_uncovered_dates_are_not_assigned(calendar, nodes):
    """2025-12-27 and 2025-12-28 belong to no reporting period."""

    daily = make_daily(nodes, "2025-12-20", "2026-01-04")

    assigned, _ = run(daily, calendar, nodes)

    assigned_dates = set(assigned["date"])

    for gap_date in GAP_DATES:
        assert gap_date not in assigned_dates, (
            f"{gap_date.date()} was assigned to a reporting period"
        )


def test_gap_dates_are_reported_as_unassigned(calendar, nodes):
    """The uncovered dates are named, not silently dropped."""

    daily = make_daily(nodes, "2025-12-20", "2026-01-04")

    assigned, _ = run(daily, calendar, nodes)

    unassigned = aggregate.find_unassigned_dates(daily, assigned)

    gap_rows = unassigned.loc[
        unassigned["reason"].eq("inside a documented calendar gap")
    ]

    assert set(pd.to_datetime(gap_rows["date"])) == set(GAP_DATES)


def test_period_after_the_gap_uses_only_its_own_seven_days(
    calendar, nodes
):
    """
    The period starting 2025-12-29 does not absorb the two gap days.

    It carries calendar_gap_before_days = 2, so a naive implementation that
    aggregated from the previous period's end would give it nine days.
    """

    daily = make_daily(nodes, "2025-12-20", "2026-01-11")

    assigned, climate = run(daily, calendar, nodes)

    rows = climate.loc[climate["start_date"].eq(AFTER_GAP_START)]

    assert rows["calendar_gap_before_days"].eq(2).all()
    assert rows["reporting_days"].eq(7).all()
    assert rows["weather_days_available"].eq(7).all()

    used = assigned.loc[
        assigned["start_date"].eq(AFTER_GAP_START), "date"
    ].unique()

    assert len(used) == 7
    assert used.min() == AFTER_GAP_START

    for gap_date in GAP_DATES:
        assert gap_date not in set(used)


def test_period_before_the_gap_ends_at_its_own_end_date(calendar, nodes):
    """The period preceding the gap does not extend into it."""

    daily = make_daily(nodes, "2025-12-15", "2026-01-04")

    assigned, _ = run(daily, calendar, nodes)

    before = assigned.loc[
        assigned["end_date"].eq(pd.Timestamp("2025-12-26"))
    ]

    assert before["date"].max() == pd.Timestamp("2025-12-26")


def test_total_assigned_days_equals_total_reporting_days(calendar, nodes):
    """
    Across the whole calendar, assigned days equal the sum of reporting_days.

    A single district is used so the arithmetic is direct. If any gap day
    were attributed, or any interval double-counted, this total would move.
    """

    daily = make_daily(
        nodes,
        calendar["start_date"].min(),
        calendar["end_date"].max(),
        districts=["Colombo"],
    )

    assigned, _ = run(daily, calendar, nodes)

    assert len(assigned) == int(calendar["reporting_days"].sum())


# ---------------------------------------------------------------------------
# Node order and grid shape
# ---------------------------------------------------------------------------

def test_every_district_follows_the_canonical_node_order(calendar, nodes):
    """Within each period the districts appear in node_id order."""

    daily = make_daily(nodes, "2015-03-07", "2015-03-27")

    _, climate = run(daily, calendar, nodes)

    for period_id, group in climate.groupby("period_id"):
        assert group["node_id"].is_monotonic_increasing, period_id
        assert group["node_id"].tolist() == list(range(len(nodes)))


def test_node_id_matches_the_registry(calendar, nodes):
    """Each district carries its registered node_id."""

    daily = make_daily(nodes, "2015-03-07", "2015-03-13")

    _, climate = run(daily, calendar, nodes)

    lookup = dict(zip(nodes["canonical_name"], nodes["node_id"]))

    expected = climate["canonical_name"].map(lookup)

    assert climate["node_id"].tolist() == expected.tolist()


def test_one_row_per_period_and_district(calendar, nodes):
    """The output is the full period-by-district grid, without duplicates."""

    daily = make_daily(nodes, "2015-03-07", "2015-03-27")

    _, climate = run(daily, calendar, nodes)

    assert len(climate) == len(calendar) * len(nodes)

    assert not climate.duplicated(
        subset=["period_id", "canonical_name"]
    ).any()


def test_periods_without_weather_still_appear(calendar, nodes):
    """
    A period with no weather is present with nulls, not omitted.

    A district can have weather where the dengue observation is missing, and
    the reverse, so the grid is kept complete on both sides.
    """

    daily = make_daily(nodes, "2015-03-07", "2015-03-13")

    _, climate = run(daily, calendar, nodes)

    empty = climate.loc[climate["weather_days_available"].eq(0)]

    assert not empty.empty

    assert empty["rainfall_sum_mm"].isna().all()
    assert empty["temperature_mean_c"].isna().all()
    assert empty["weather_coverage_ratio"].eq(0.0).all()
    assert not empty["weather_complete"].any()


# ---------------------------------------------------------------------------
# Aggregation statistics
# ---------------------------------------------------------------------------

def test_temperature_extremes_are_interval_extremes(calendar, nodes):
    """
    Minimum and maximum are taken over the interval, not averaged.

    Averaging the daily extremes would understate the range the district
    actually experienced.
    """

    daily = make_daily(nodes, "2015-03-07", "2015-03-13")

    # Give each day a distinct temperature range.
    daily["temperature_min_c"] = 20.0 + daily["date"].dt.day * 0.5
    daily["temperature_max_c"] = 30.0 + daily["date"].dt.day * 0.5

    _, climate = run(daily, calendar, nodes)

    row = climate.loc[
        climate["start_date"].eq(pd.Timestamp("2015-03-07"))
        & climate["canonical_name"].eq("Colombo")
    ].iloc[0]

    # Days 7 to 13 of March.
    assert row["temperature_min_c"] == pytest.approx(20.0 + 7 * 0.5)
    assert row["temperature_max_c"] == pytest.approx(30.0 + 13 * 0.5)


def test_rainy_days_use_the_documented_threshold(calendar, nodes):
    """Only days at or above the threshold count as rainy."""

    assert aggregate.RAINY_DAY_THRESHOLD_MM == 1.0

    daily = make_daily(
        nodes,
        "2015-03-07",
        "2015-03-13",
        # Four wet days, three at a trace below the threshold.
        rainfall=lambda day: 5.0 if day.day <= 10 else 0.2,
    )

    _, climate = run(daily, calendar, nodes)

    row = climate.loc[
        climate["start_date"].eq(pd.Timestamp("2015-03-07"))
        & climate["canonical_name"].eq("Colombo")
    ].iloc[0]

    assert row["rainy_days"] == 4
    assert row["weather_days_available"] == 7


def test_rainy_days_never_exceed_available_days(calendar, nodes):
    """The rainy-day count is bounded by the days actually present."""

    daily = make_daily(nodes, "2015-03-07", "2015-03-27", rainfall=20.0)

    _, climate = run(daily, calendar, nodes)

    assert climate["rainy_days"].le(climate["weather_days_available"]).all()


def test_sum_and_daily_mean_are_consistent(calendar, nodes):
    """The daily mean multiplied by the day count returns the sum."""

    daily = make_daily(
        nodes, "2015-03-07", "2015-03-27", rainfall=lambda day: day.day
    )

    _, climate = run(daily, calendar, nodes)

    present = climate.loc[climate["weather_days_available"].gt(0)]

    reconstructed = (
        present["rainfall_daily_mean_mm"] * present["weather_days_available"]
    )

    assert np.allclose(reconstructed, present["rainfall_sum_mm"])


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_expected_weather_days_equals_reporting_days(calendar, nodes):
    """The expected day count is the interval length, always."""

    daily = make_daily(nodes, "2009-04-11", "2009-06-05")

    _, climate = run(daily, calendar, nodes)

    assert climate["expected_weather_days"].equals(climate["reporting_days"])


def test_partial_coverage_is_reported(calendar, nodes):
    """A district missing days is incomplete, and its ratio reflects that."""

    daily = make_daily(nodes, "2015-03-07", "2015-03-13")

    dropped = [pd.Timestamp("2015-03-09"), pd.Timestamp("2015-03-10")]

    daily = daily.loc[
        ~(
            daily["canonical_name"].eq("Colombo")
            & daily["date"].isin(dropped)
        )
    ]

    _, climate = run(daily, calendar, nodes)

    period = climate.loc[
        climate["start_date"].eq(pd.Timestamp("2015-03-07"))
    ]

    colombo = period.loc[period["canonical_name"].eq("Colombo")].iloc[0]
    galle = period.loc[period["canonical_name"].eq("Galle")].iloc[0]

    assert colombo["weather_days_available"] == 5
    assert colombo["weather_coverage_ratio"] == pytest.approx(5 / 7)
    assert not colombo["weather_complete"]

    assert galle["weather_days_available"] == 7
    assert galle["weather_complete"]


def test_a_null_value_disqualifies_the_day(calendar, nodes):
    """
    A present row with a null weather value is not a usable day.

    Coverage measures usable days, not merely present rows.
    """

    daily = make_daily(nodes, "2015-03-07", "2015-03-13")

    daily.loc[
        daily["canonical_name"].eq("Kandy")
        & daily["date"].eq(pd.Timestamp("2015-03-11")),
        "temperature_mean_c",
    ] = np.nan

    _, climate = run(daily, calendar, nodes)

    kandy = climate.loc[
        climate["start_date"].eq(pd.Timestamp("2015-03-07"))
        & climate["canonical_name"].eq("Kandy")
    ].iloc[0]

    assert kandy["weather_days_available"] == 6
    assert not kandy["weather_complete"]


def test_missing_weather_is_never_filled(calendar, nodes):
    """No zero, mean or carried value replaces absent weather."""

    daily = make_daily(nodes, "2015-03-07", "2015-03-13")

    _, climate = run(daily, calendar, nodes)

    empty = climate.loc[climate["weather_days_available"].eq(0)]

    for column in [
        "rainfall_sum_mm",
        "rainfall_daily_mean_mm",
        "temperature_mean_c",
        "temperature_min_c",
        "temperature_max_c",
        "dewpoint_mean_c",
        "relative_humidity_mean",
        "wind_speed_mean",
    ]:
        assert empty[column].isna().all(), column

    assert not (empty["rainfall_sum_mm"] == 0).any()


def test_coverage_report_covers_every_period(calendar, nodes):
    """The coverage report holds one row per reporting period."""

    daily = make_daily(nodes, "2015-03-07", "2015-03-27")

    _, climate = run(daily, calendar, nodes)

    coverage = aggregate.build_coverage_report(climate)

    assert len(coverage) == len(calendar)
    assert coverage["districts"].eq(len(nodes)).all()

    complete = coverage.loc[coverage["period_complete"]]

    assert complete["districts_incomplete"].eq(0).all()


# ---------------------------------------------------------------------------
# No lags in this stage
# ---------------------------------------------------------------------------

def test_no_shift_is_used_in_aggregation():
    """
    The aggregation must not use shift(), because lags belong to stage 3.

    Asserted against the source so that a later edit introducing a lag here
    fails loudly rather than quietly leaking a value across periods.
    """

    source = (
        PROJECT_DIR / "scripts" / "9.aggregate_climate_to_periods.py"
    ).read_text(encoding="utf-8")

    assert ".shift(" not in source


def test_every_value_comes_from_its_own_period(calendar, nodes):
    """
    A period's values depend only on its own dates.

    Rainfall is set to zero everywhere except one interval; every other
    period must report zero, which would fail if any value leaked across a
    boundary.
    """

    target_start = pd.Timestamp("2015-03-14")
    target_end = pd.Timestamp("2015-03-20")

    daily = make_daily(
        nodes,
        "2015-03-07",
        "2015-03-27",
        rainfall=lambda day: 10.0 if target_start <= day <= target_end else 0.0,
    )

    _, climate = run(daily, calendar, nodes)

    present = climate.loc[climate["weather_days_available"].gt(0)]

    target = present.loc[present["start_date"].eq(target_start)]
    others = present.loc[~present["start_date"].eq(target_start)]

    assert target["rainfall_sum_mm"].eq(70.0).all()
    assert others["rainfall_sum_mm"].eq(0.0).all()


# ---------------------------------------------------------------------------
# Assertions and inputs
# ---------------------------------------------------------------------------

def test_full_assertions_pass_on_clean_input(calendar, nodes):
    """The script's own assertions hold for a well-formed extraction."""

    daily = make_daily(nodes, "2009-04-11", "2009-06-05")

    assigned, climate = run(daily, calendar, nodes)

    aggregate.run_assertions(daily, assigned, climate, calendar, nodes)


def test_missing_climate_file_names_the_extraction_script():
    """An absent input points at the script that produces it."""

    with pytest.raises(FileNotFoundError, match="5.extract_era5_daily"):
        aggregate.load_daily_climate(
            PROJECT_DIR / "data" / "raw" / "does_not_exist.csv"
        )

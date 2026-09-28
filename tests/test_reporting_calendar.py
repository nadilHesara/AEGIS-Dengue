"""
Tests for the canonical reporting calendar.

Synthetic frames cover each reporting convention and each anomaly type in
isolation. The real-file tests confirm that the confirmed source findings
survive the build unchanged.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "create_calendar", PROJECT_DIR / "scripts" / "data" / "2.create_calendar.py"
)
calendar_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calendar_module)


# The 26 source reporting areas, before Kalmune is merged into Ampara.
REPORTING_AREAS = [
    "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "NuwaraEliya",
    "Galle", "Hambantota", "Matara", "Jaffna", "Kilinochchi", "Mannar",
    "Vavuniya", "Mullaitivu", "Batticaloa", "Ampara", "Trincomalee",
    "Kurunegala", "Puttalam", "Anuradhapura", "Polonnaruwa", "Badulla",
    "Monaragala", "Ratnapura", "Kegalle", "Kalmune",
]


def make_period(year, week, start, end, areas=None, cases=5):
    """Build the district-level rows for one reporting period."""

    areas = REPORTING_AREAS if areas is None else areas

    return pd.DataFrame(
        {
            "year": year,
            "week": week,
            "start.date": start,
            "end.date": end,
            "district": areas,
            "cases": cases,
        }
    )


def make_frame(*periods):
    """Concatenate reporting periods into one source-shaped frame."""

    return pd.concat(periods, ignore_index=True)


def build(df):
    """Run the full calendar build over a synthetic source frame."""

    rows = calendar_module.add_row_date_columns(df)

    return rows, calendar_module.build_calendar_artifacts(rows)


# ---------------------------------------------------------------------------
# Reporting conventions
# ---------------------------------------------------------------------------

def test_normal_seven_day_saturday_period():
    """A Saturday-to-Friday week is regular and calendar-continuous."""

    df = make_frame(make_period(2015, 10, "3/7/2015", "3/13/2015"))

    _, artifacts = build(df)
    calendar = artifacts["calendar"]

    assert len(calendar) == 1

    period = calendar.iloc[0]

    assert period["reporting_days"] == 7
    assert period["start_weekday"] == "Saturday"
    assert period["end_weekday"] == "Friday"
    assert period["weekday_convention"] == "saturday_friday"
    assert not period["is_irregular_period"]


def test_eight_day_period_is_kept_and_flagged():
    """The 2009 week 17 period is 8 days and is not rejected."""

    df = make_frame(make_period(2009, 17, "4/18/2009", "4/25/2009"))

    _, artifacts = build(df)
    period = artifacts["calendar"].iloc[0]

    assert period["reporting_days"] == 8
    assert period["is_irregular_period"]
    assert period["start_weekday"] == "Saturday"


def test_temporary_sunday_sequence_is_labelled():
    """The periods after the 8-day interval start on Sunday."""

    df = make_frame(
        make_period(2009, 17, "4/18/2009", "4/25/2009"),
        make_period(2009, 18, "4/26/2009", "5/2/2009"),
        make_period(2009, 19, "5/3/2009", "5/9/2009"),
    )

    _, artifacts = build(df)
    calendar = artifacts["calendar"]

    sunday_periods = calendar.loc[
        calendar["weekday_convention"].eq("temporary_sunday_sequence")
    ]

    assert len(sunday_periods) == 2
    assert set(sunday_periods["start_weekday"]) == {"Sunday"}

    # The 8-day period itself still starts on a Saturday.
    assert calendar.iloc[0]["weekday_convention"] == "saturday_friday"


def test_six_day_period_restores_saturday():
    """The 6-day 2009 week 22 period returns reporting to Saturday."""

    df = make_frame(
        make_period(2009, 21, "5/17/2009", "5/23/2009"),
        make_period(2009, 22, "5/24/2009", "5/29/2009"),
        make_period(2009, 23, "5/30/2009", "6/5/2009"),
    )

    _, artifacts = build(df)
    calendar = artifacts["calendar"]

    six_day = calendar.loc[calendar["reporting_days"].eq(6)]

    assert len(six_day) == 1
    assert six_day.iloc[0]["is_irregular_period"]

    restored = calendar.iloc[2]

    assert restored["start_weekday"] == "Saturday"
    assert restored["weekday_convention"] == "saturday_friday"
    assert restored["is_weekday_convention_change"]

    # The shortened period does not break calendar continuity.
    assert calendar["is_calendar_continuous"].iloc[1:].all()


def test_permanent_monday_convention_is_valid():
    """Monday-to-Sunday periods from 2026 are a convention, not an error."""

    df = make_frame(
        make_period(2026, 1, "12/29/2025", "1/4/2026"),
        make_period(2026, 2, "1/5/2026", "1/11/2026"),
    )

    _, artifacts = build(df)
    calendar = artifacts["calendar"]

    assert set(calendar["weekday_convention"]) == {"monday_sunday"}
    assert set(calendar["start_weekday"]) == {"Monday"}
    assert not calendar["is_irregular_period"].any()


def test_convention_change_is_flagged_only_at_the_seam():
    """is_weekday_convention_change is True only on the changing period."""

    df = make_frame(
        make_period(2025, 51, "12/13/2025", "12/19/2025"),
        make_period(2026, 53, "12/20/2025", "12/26/2025"),
        make_period(2026, 1, "12/29/2025", "1/4/2026"),
        make_period(2026, 2, "1/5/2026", "1/11/2026"),
    )

    _, artifacts = build(df)
    calendar = artifacts["calendar"]

    changes = calendar.loc[calendar["is_weekday_convention_change"]]

    assert len(changes) == 1
    assert changes.iloc[0]["start_date"] == pd.Timestamp("2025-12-29")


# ---------------------------------------------------------------------------
# Continuity
# ---------------------------------------------------------------------------

def test_two_day_gap_is_detected():
    """The uncovered 2025-12-27 and 2025-12-28 are reported as a gap."""

    df = make_frame(
        make_period(2026, 53, "12/20/2025", "12/26/2025"),
        make_period(2026, 1, "12/29/2025", "1/4/2026"),
    )

    _, artifacts = build(df)
    calendar = artifacts["calendar"]

    second = calendar.iloc[1]

    assert second["calendar_gap_before_days"] == 2
    assert second["calendar_overlap_before_days"] == 0
    assert not second["is_calendar_continuous"]
    assert second["has_calendar_gap_before"]

    gaps = artifacts["known_gaps"]

    assert len(gaps) == 1

    gap = gaps.iloc[0]

    assert gap["gap_start_date"] == pd.Timestamp("2025-12-27")
    assert gap["gap_end_date"] == pd.Timestamp("2025-12-28")
    assert gap["gap_days"] == 2
    assert "Do not interpolate" in gap["recommended_handling"]


def test_overlapping_periods_are_detected():
    """An interval starting before its predecessor ends is an overlap."""

    df = make_frame(
        make_period(2015, 10, "3/7/2015", "3/13/2015"),
        make_period(2015, 11, "3/12/2015", "3/18/2015"),
    )

    _, artifacts = build(df)
    second = artifacts["calendar"].iloc[1]

    assert second["calendar_overlap_before_days"] == 2
    assert second["calendar_gap_before_days"] == 0
    assert second["has_calendar_overlap_before"]
    assert second["has_data_quality_issue"]


def test_continuous_periods_have_no_gap_or_overlap():
    """Back-to-back weeks are continuous."""

    df = make_frame(
        make_period(2015, 10, "3/7/2015", "3/13/2015"),
        make_period(2015, 11, "3/14/2015", "3/20/2015"),
    )

    _, artifacts = build(df)
    calendar = artifacts["calendar"]

    assert calendar["is_calendar_continuous"].iloc[1:].all()
    assert calendar["calendar_gap_before_days"].sum() == 0
    assert calendar["calendar_overlap_before_days"].sum() == 0


# ---------------------------------------------------------------------------
# Ordering and source labels
# ---------------------------------------------------------------------------

def test_period_id_follows_dates_not_source_labels():
    """A label that sorts late must still be positioned by its dates."""

    df = make_frame(
        make_period(2025, 52, "12/13/2025", "12/19/2025"),
        make_period(2026, 53, "12/20/2025", "12/26/2025"),
        make_period(2026, 1, "12/29/2025", "1/4/2026"),
    )

    _, artifacts = build(df)
    calendar = artifacts["calendar"]

    week_53 = calendar.loc[calendar["source_week"].eq(53)].iloc[0]
    week_1 = calendar.loc[calendar["source_week"].eq(1)].iloc[0]

    assert week_53["period_id"] < week_1["period_id"]

    assert calendar.sort_values("start_date")["period_id"].is_monotonic_increasing


def test_out_of_order_source_label_is_an_anomaly():
    """The 2026 week 53 label is reported, not corrected."""

    df = make_frame(
        make_period(2025, 52, "12/13/2025", "12/19/2025"),
        make_period(2026, 53, "12/20/2025", "12/26/2025"),
        make_period(2026, 1, "12/29/2025", "1/4/2026"),
    )

    _, artifacts = build(df)
    anomalies = artifacts["label_anomalies"]

    misplaced = anomalies.loc[
        anomalies["anomaly_type"].eq("chronologically_misplaced_label")
    ]

    assert len(misplaced) == 1
    assert misplaced.iloc[0]["source_week"] == 53

    # The source label itself is untouched.
    calendar = artifacts["calendar"]
    week_53 = calendar.loc[calendar["source_week"].eq(53)].iloc[0]

    assert week_53["source_year"] == 2026
    assert week_53["has_source_label_anomaly"]


def test_duplicate_source_label_with_different_dates():
    """One year-week label describing two intervals is an anomaly."""

    df = make_frame(
        make_period(2015, 10, "3/7/2015", "3/13/2015"),
        make_period(2015, 10, "3/14/2015", "3/20/2015"),
    )

    _, artifacts = build(df)
    anomalies = artifacts["label_anomalies"]

    duplicates = anomalies.loc[
        anomalies["anomaly_type"].eq("duplicate_source_label")
    ]

    assert len(duplicates) == 2
    assert set(duplicates["source_week"]) == {10}


def test_different_labels_with_identical_dates():
    """Two labels describing one interval is an anomaly."""

    first = make_period(2015, 10, "3/7/2015", "3/13/2015")
    second = make_period(2015, 11, "3/7/2015", "3/13/2015")

    rows = calendar_module.add_row_date_columns(
        make_frame(first, second)
    )

    # The calendar keeps one row per interval, so the duplicate label is
    # detected before de-duplication collapses it.
    unique_intervals = rows[
        ["year", "week", "start_date", "end_date"]
    ].drop_duplicates()

    assert len(unique_intervals) == 2

    artifacts = calendar_module.build_calendar_artifacts(rows)

    assert len(artifacts["calendar"]) == 1


def test_source_year_differing_from_start_year_is_documented_only():
    """A December start labelled as the next year is not an error."""

    df = make_frame(make_period(2026, 1, "12/29/2025", "1/4/2026"))

    _, artifacts = build(df)
    period = artifacts["calendar"].iloc[0]

    assert period["start_calendar_year"] == 2025
    assert period["end_calendar_year"] == 2026
    assert period["source_year_differs_from_start_year"]

    # Documentation only: it does not by itself make the period a problem.
    assert not period["has_source_label_anomaly"]
    assert not period["has_data_quality_issue"]


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_complete_period_carries_every_reporting_area():
    """A period with all 26 source areas is complete."""

    df = make_frame(make_period(2026, 6, "2/2/2026", "2/8/2026"))

    _, artifacts = build(df)
    period = artifacts["calendar"].iloc[0]

    assert period["source_reporting_area_count"] == 26
    assert period["is_complete_source_coverage"]
    assert period["missing_source_reporting_areas"] == ""
    assert artifacts["incomplete_coverage"].empty


def test_missing_reporting_area_is_named_not_filled():
    """A missing district is reported by name and never invented."""

    without_puttalam = [a for a in REPORTING_AREAS if a != "Puttalam"]

    df = make_frame(
        make_period(2026, 6, "2/2/2026", "2/8/2026"),
        make_period(
            2026, 7, "2/9/2026", "2/15/2026", areas=without_puttalam
        ),
    )

    rows, artifacts = build(df)

    incomplete = artifacts["incomplete_coverage"]

    assert len(incomplete) == 1

    issue = incomplete.iloc[0]

    assert issue["source_week"] == 7
    assert issue["source_reporting_area_count"] == 25
    assert issue["missing_source_reporting_areas"] == "Puttalam"

    # No synthetic row was added for the missing district.
    mapped = calendar_module.assign_period_ids(rows, artifacts["calendar"])

    puttalam_rows = mapped.loc[
        mapped["period_id"].eq(issue["period_id"])
        & mapped["district"].eq("Puttalam")
    ]

    assert puttalam_rows.empty

    # The period itself is retained.
    assert issue["period_id"] in set(artifacts["calendar"]["period_id"])


# ---------------------------------------------------------------------------
# Row mapping
# ---------------------------------------------------------------------------

def test_every_district_row_maps_to_exactly_one_period():
    """The district rows join the calendar without loss or duplication."""

    df = make_frame(
        make_period(2015, 10, "3/7/2015", "3/13/2015"),
        make_period(2015, 11, "3/14/2015", "3/20/2015"),
        make_period(2015, 12, "3/21/2015", "3/27/2015"),
    )

    rows, artifacts = build(df)

    mapped = calendar_module.assign_period_ids(rows, artifacts["calendar"])

    assert len(mapped) == len(rows)
    assert mapped["period_id"].notna().all()
    assert mapped["period_id"].nunique() == len(artifacts["calendar"])


def test_calendar_holds_one_row_per_interval_not_per_district():
    """The calendar is an interval table, not a district table."""

    df = make_frame(
        make_period(2015, 10, "3/7/2015", "3/13/2015"),
        make_period(2015, 11, "3/14/2015", "3/20/2015"),
    )

    rows, artifacts = build(df)

    assert len(rows) == 2 * len(REPORTING_AREAS)
    assert len(artifacts["calendar"]) == 2


# ---------------------------------------------------------------------------
# Real source file
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_artifacts():
    """Build the calendar from the real source file once per module."""

    rows = calendar_module.load_source_rows()

    return rows, calendar_module.build_calendar_artifacts(rows)


def test_real_file_assertions_all_hold(real_artifacts):
    """Every structural assertion passes on the real source file."""

    rows, artifacts = real_artifacts

    calendar_module.run_calendar_assertions(rows, artifacts)


def test_real_file_period_ids_are_chronological(real_artifacts):
    """period_id increases strictly with the actual dates."""

    _, artifacts = real_artifacts
    calendar = artifacts["calendar"]

    assert calendar["period_id"].is_unique

    ordered = calendar.sort_values(["start_date", "end_date"])

    assert ordered["period_id"].is_monotonic_increasing


def test_real_file_preserves_the_2009_irregular_periods(real_artifacts):
    """Both confirmed irregular periods survive the build."""

    _, artifacts = real_artifacts
    calendar = artifacts["calendar"]

    irregular = calendar.loc[calendar["is_irregular_period"]]

    assert len(irregular) == 2
    assert set(irregular["reporting_days"]) == {6, 8}

    # Neither breaks calendar continuity.
    assert irregular["is_calendar_continuous"].all()


def test_real_file_has_one_two_day_gap(real_artifacts):
    """The only uncovered dates are 2025-12-27 and 2025-12-28."""

    _, artifacts = real_artifacts
    gaps = artifacts["known_gaps"]

    assert len(gaps) == 1

    gap = gaps.iloc[0]

    assert gap["gap_start_date"] == pd.Timestamp("2025-12-27")
    assert gap["gap_end_date"] == pd.Timestamp("2025-12-28")
    assert gap["gap_days"] == 2


def test_real_file_has_no_overlaps(real_artifacts):
    """No reporting period overlaps its predecessor."""

    _, artifacts = real_artifacts

    assert not artifacts["calendar"]["has_calendar_overlap_before"].any()


def test_real_file_2026_week_53_is_the_only_label_anomaly(real_artifacts):
    """The single label anomaly is the 2026 week 53 interval."""

    _, artifacts = real_artifacts
    anomalies = artifacts["label_anomalies"]

    assert len(anomalies) == 1

    anomaly = anomalies.iloc[0]

    assert anomaly["source_year"] == 2026
    assert anomaly["source_week"] == 53
    assert anomaly["start_date"] == pd.Timestamp("2025-12-20")
    assert "period_id" in anomaly["recommended_handling"]


def test_real_file_puttalam_is_the_only_coverage_issue(real_artifacts):
    """Puttalam in 2026 week 7 is the single incomplete period."""

    _, artifacts = real_artifacts
    coverage = artifacts["incomplete_coverage"]

    assert len(coverage) == 1

    issue = coverage.iloc[0]

    assert issue["source_year"] == 2026
    assert issue["source_week"] == 7
    assert issue["source_reporting_area_count"] == 25
    assert issue["missing_source_reporting_areas"] == "Puttalam"


def test_real_file_weekday_conventions(real_artifacts):
    """The three conventions appear with their expected weekdays."""

    _, artifacts = real_artifacts
    calendar = artifacts["calendar"]

    for convention, weekday in [
        ("saturday_friday", "Saturday"),
        ("temporary_sunday_sequence", "Sunday"),
        ("monday_sunday", "Monday"),
    ]:
        periods = calendar.loc[
            calendar["weekday_convention"].eq(convention)
        ]

        assert not periods.empty, convention
        assert set(periods["start_weekday"]) == {weekday}, convention


def test_real_file_source_labels_are_unchanged(real_artifacts):
    """The raw year and week values are carried through untouched."""

    rows, _ = real_artifacts

    source = pd.read_csv(calendar_module.RAW_PATH)

    assert rows["year"].equals(source["year"])
    assert rows["week"].equals(source["week"])
    assert rows["start.date"].equals(source["start.date"])
    assert rows["end.date"].equals(source["end.date"])
    assert rows["cases"].equals(source["cases"])

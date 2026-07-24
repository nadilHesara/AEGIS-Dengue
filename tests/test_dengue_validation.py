"""
Unit tests for the dengue reporting-period validator.

Each test builds a small synthetic frame so that the behaviour under test is
isolated from the real source file. The real file is exercised separately in
the integration tests at the end, which assert the confirmed findings.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = PROJECT_DIR / "data" / "raw" / "srilanka_weekly_data.csv"


def _load_validator():
    """Import the validator module, whose filename is not a valid identifier."""

    module_path = PROJECT_DIR / "scripts" / "1.dataset_validate.py"

    spec = importlib.util.spec_from_file_location(
        "dengue_validator", module_path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


validator = _load_validator()


REPORTING_AREAS = [
    "Ampara",
    "Anuradhapura",
    "Badulla",
    "Batticaloa",
    "Colombo",
    "Galle",
    "Gampaha",
    "Hambanthota",
    "Jaffna",
    "Kalmune",
    "Kalutara",
    "Kandy",
    "Kegalle",
    "Kilinochchi",
    "Kurunegala",
    "Mannar",
    "Matale",
    "Matara",
    "Monaragala",
    "Mullaitivu",
    "NuwaraEliya",
    "Polonnaruwa",
    "Puttalam",
    "Ratnapura",
    "Trincomalee",
    "Vavuniya",
]


def make_period(
    year: int,
    week: int,
    start: str,
    end: str,
    areas: "list[str] | None" = None,
    cases: int = 5,
) -> list[dict]:
    """Build the district rows for one reporting period."""

    if areas is None:
        areas = REPORTING_AREAS

    return [
        {
            "year": year,
            "week": week,
            "start.date": start,
            "end.date": end,
            "district": area,
            "cases": cases,
        }
        for area in areas
    ]


def make_frame(*periods: list[dict]) -> pd.DataFrame:
    """Build a source-shaped frame from one or more periods."""

    rows = [row for period in periods for row in period]

    return pd.DataFrame(rows, columns=validator.SOURCE_COLUMNS)


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns."""

    return validator.add_derived_columns(df)


def periods_of(df: pd.DataFrame) -> pd.DataFrame:
    """Build the chronological period table."""

    return validator.build_reporting_periods(derive(df))


# ---------------------------------------------------------------------------
# Normal seven-day Saturday reporting
# ---------------------------------------------------------------------------

def test_normal_seven_day_saturday_period_is_regular():
    """A Saturday-to-Friday week is 7 inclusive days and not irregular."""

    df = make_frame(
        make_period(2015, 10, "3/7/2015", "3/13/2015"),
    )

    periods = periods_of(df)
    row = periods.iloc[0]

    assert row["reporting_days"] == 7
    assert not row["is_irregular_period"]
    assert row["start_weekday"] == "Saturday"
    assert row["expected_start_weekday"] == "Saturday"


def test_normal_saturday_reporting_produces_no_findings():
    """Consecutive complete Saturday weeks pass every check."""

    df = make_frame(
        make_period(2015, 10, "3/7/2015", "3/13/2015"),
        make_period(2015, 11, "3/14/2015", "3/20/2015"),
        make_period(2015, 12, "3/21/2015", "3/27/2015"),
    )

    validation = validator.validate_weekly_dengue_data(df)

    statuses = {
        name: validator.classify_check(severity, findings)
        for name, (severity, findings) in validation["checks"].items()
    }

    assert set(statuses.values()) == {validator.PASS}, statuses
    assert validator.count_failed_checks(validation) == 0


def test_source_columns_are_preserved_unchanged():
    """Derived columns are added without altering the source columns."""

    df = make_frame(make_period(2015, 10, "3/7/2015", "3/13/2015"))
    original = df.copy()

    derived = derive(df)

    pd.testing.assert_frame_equal(df, original)
    pd.testing.assert_frame_equal(
        derived[validator.SOURCE_COLUMNS], original
    )

    for column in [
        "start_date",
        "end_date",
        "reporting_days",
        "start_weekday",
        "is_irregular_period",
    ]:
        assert column in derived.columns


# ---------------------------------------------------------------------------
# The 2009 episode
# ---------------------------------------------------------------------------

def test_2009_eight_day_period_is_a_warning_not_a_failure():
    """The 8-day 2009 week 17 period is irregular but usable."""

    df = make_frame(
        make_period(2009, 17, "4/18/2009", "4/25/2009"),
    )

    periods = periods_of(df)
    row = periods.iloc[0]

    assert row["reporting_days"] == 8
    assert row["is_irregular_period"]

    irregular = validator.find_irregular_reporting_periods(periods)

    assert len(irregular) == 1
    assert bool(irregular.iloc[0]["is_documented"])

    # An 8-day period is not an impossible range.
    assert validator.find_impossible_date_ranges(derive(df)).empty

    # The check is classified WARNING, so it never fails the run.
    validation = validator.validate_weekly_dengue_data(df)
    severity, findings = validation["checks"]["irregular_reporting_periods"]

    assert validator.classify_check(severity, findings) == validator.WARNING


def test_temporary_sunday_start_sequence_is_expected():
    """Weeks 18 to 22 of 2009 start on Sunday by convention, not by error."""

    df = make_frame(
        make_period(2009, 18, "4/26/2009", "5/2/2009"),
        make_period(2009, 19, "5/3/2009", "5/9/2009"),
        make_period(2009, 20, "5/10/2009", "5/16/2009"),
        make_period(2009, 21, "5/17/2009", "5/23/2009"),
    )

    periods = periods_of(df)

    assert set(periods["start_weekday"]) == {"Sunday"}
    assert set(periods["expected_start_weekday"]) == {"Sunday"}
    assert validator.find_unexpected_start_weekdays(periods).empty


def test_2009_six_day_period_is_a_warning_not_a_failure():
    """The 6-day 2009 week 22 period is irregular but usable."""

    df = make_frame(
        make_period(2009, 22, "5/24/2009", "5/29/2009"),
    )

    periods = periods_of(df)
    row = periods.iloc[0]

    assert row["reporting_days"] == 6
    assert row["is_irregular_period"]
    assert row["start_weekday"] == "Sunday"
    assert row["expected_start_weekday"] == "Sunday"

    irregular = validator.find_irregular_reporting_periods(periods)

    assert len(irregular) == 1
    assert bool(irregular.iloc[0]["is_documented"])
    assert validator.find_impossible_date_ranges(derive(df)).empty


def test_2009_episode_keeps_the_calendar_continuous():
    """
    The 8-day and 6-day periods absorb the phase change.

    No calendar gap or overlap is produced across the whole episode, and the
    2009 start-date shifts are not counted as failures.
    """

    df = make_frame(
        make_period(2009, 16, "4/11/2009", "4/17/2009"),
        make_period(2009, 17, "4/18/2009", "4/25/2009"),
        make_period(2009, 18, "4/26/2009", "5/2/2009"),
        make_period(2009, 19, "5/3/2009", "5/9/2009"),
        make_period(2009, 20, "5/10/2009", "5/16/2009"),
        make_period(2009, 21, "5/17/2009", "5/23/2009"),
        make_period(2009, 22, "5/24/2009", "5/29/2009"),
        make_period(2009, 23, "5/30/2009", "6/5/2009"),
    )

    periods = periods_of(df)
    continuity = validator.build_calendar_continuity(periods)

    assert set(continuity["classification"]) == {validator.PASS}
    assert validator.find_calendar_gaps(continuity).empty
    assert validator.find_calendar_overlaps(continuity).empty
    assert validator.find_unexpected_start_weekdays(periods).empty


def test_saturday_reporting_is_restored_after_2009_week_22():
    """From 2009 week 23 the expected start weekday is Saturday again."""

    df = make_frame(
        make_period(2009, 23, "5/30/2009", "6/5/2009"),
        make_period(2009, 24, "6/6/2009", "6/12/2009"),
    )

    periods = periods_of(df)

    assert set(periods["start_weekday"]) == {"Saturday"}
    assert set(periods["expected_start_weekday"]) == {"Saturday"}
    assert validator.find_unexpected_start_weekdays(periods).empty


# ---------------------------------------------------------------------------
# The 2026 Monday convention
# ---------------------------------------------------------------------------

def test_2026_monday_convention_is_not_flagged():
    """Monday-start periods from 2026 week 1 are expected, not invalid."""

    df = make_frame(
        make_period(2026, 1, "12/29/2025", "1/4/2026"),
        make_period(2026, 2, "1/5/2026", "1/11/2026"),
        make_period(2026, 3, "1/12/2026", "1/18/2026"),
    )

    periods = periods_of(df)

    assert set(periods["start_weekday"]) == {"Monday"}
    assert set(periods["expected_start_weekday"]) == {"Monday"}
    assert set(periods["reporting_days"]) == {7}
    assert validator.find_unexpected_start_weekdays(periods).empty


def test_weekday_convention_change_is_reported_as_a_warning():
    """The Saturday-to-Monday change is reported separately from errors."""

    df = make_frame(
        make_period(2026, 53, "12/20/2025", "12/26/2025"),
        make_period(2026, 1, "12/29/2025", "1/4/2026"),
    )

    periods = periods_of(df)

    changes = validator.find_weekday_convention_changes(periods)

    assert len(changes) == 1

    change = changes.iloc[0]

    assert change["previous_start_weekday"] == "Saturday"
    assert change["start_weekday"] == "Monday"

    # Reported as a convention change, never as an unexpected weekday.
    assert validator.find_unexpected_start_weekdays(periods).empty

    validation = validator.validate_weekly_dengue_data(df)
    severity, findings = validation["checks"]["weekday_convention_changes"]

    assert validator.classify_check(severity, findings) == validator.WARNING


def test_a_saturday_start_inside_the_monday_convention_is_flagged():
    """A weekday that breaks its own convention is still detected."""

    df = make_frame(
        make_period(2026, 2, "1/3/2026", "1/9/2026"),
    )

    periods = periods_of(df)

    unexpected = validator.find_unexpected_start_weekdays(periods)

    assert len(unexpected) == 1
    assert unexpected.iloc[0]["start_weekday"] == "Saturday"
    assert unexpected.iloc[0]["expected_start_weekday"] == "Monday"


# ---------------------------------------------------------------------------
# Chronological placement of 2026 week 53
# ---------------------------------------------------------------------------

def test_2026_week_53_is_placed_before_2026_week_1():
    """
    period_id follows the dates, not the source label.

    2026 week 53 covers 2025-12-20 to 2025-12-26 and must therefore precede
    2026 week 1, which starts 2025-12-29.
    """

    df = make_frame(
        make_period(2026, 1, "12/29/2025", "1/4/2026"),
        make_period(2026, 53, "12/20/2025", "12/26/2025"),
        make_period(2026, 2, "1/5/2026", "1/11/2026"),
    )

    periods = periods_of(df).set_index(["source_year", "source_week"])

    week_53_id = periods.loc[(2026, 53), "period_id"]
    week_1_id = periods.loc[(2026, 1), "period_id"]
    week_2_id = periods.loc[(2026, 2), "period_id"]

    assert week_53_id < week_1_id < week_2_id


def test_period_ids_are_sequential_and_follow_date_order():
    """period_id is a stable sequential key aligned with the date order."""

    df = make_frame(
        make_period(2026, 2, "1/5/2026", "1/11/2026"),
        make_period(2026, 53, "12/20/2025", "12/26/2025"),
        make_period(2026, 1, "12/29/2025", "1/4/2026"),
    )

    periods = periods_of(df)

    assert list(periods["period_id"]) == [1, 2, 3]
    assert periods["start_date"].is_monotonic_increasing


def test_out_of_order_source_week_is_detected_without_rewriting_the_label():
    """The 2026 week 53 label is reported but left untouched."""

    df = make_frame(
        make_period(2026, 53, "12/20/2025", "12/26/2025"),
        make_period(2026, 1, "12/29/2025", "1/4/2026"),
    )

    periods = periods_of(df)

    out_of_order = validator.find_out_of_order_source_weeks(periods)

    assert len(out_of_order) == 1

    flagged = out_of_order.iloc[0]

    assert flagged["source_year"] == 2026
    assert flagged["source_week"] == 1
    assert flagged["previous_source_week"] == 53

    # The source label survives unchanged.
    week_53 = periods.loc[periods["source_week"].eq(53)].iloc[0]

    assert week_53["source_year"] == 2026
    assert week_53["source_week"] == 53


def test_source_year_label_mismatch_flags_2026_week_53():
    """
    2026 week 53 covers days that mostly belong to 2025.

    2026 week 1 (2025-12-29 to 2026-01-04) covers four days in 2026 and three
    in 2025, so its majority year matches its label and it is not flagged.
    """

    df = make_frame(
        make_period(2026, 53, "12/20/2025", "12/26/2025"),
        make_period(2026, 1, "12/29/2025", "1/4/2026"),
    )

    mismatches = validator.find_year_label_mismatches(periods_of(df))

    assert list(mismatches["source_week"]) == [53]

    flagged = mismatches.iloc[0]

    assert flagged["majority_calendar_year"] == 2025
    assert not bool(flagged["is_expected_year_rollover"])


def test_week_1_starting_in_december_is_an_expected_rollover():
    """A week 1 whose days mostly fall in the previous year is expected."""

    df = make_frame(
        make_period(2024, 1, "12/23/2023", "12/29/2023"),
    )

    mismatches = validator.find_year_label_mismatches(periods_of(df))

    assert len(mismatches) == 1

    flagged = mismatches.iloc[0]

    assert flagged["majority_calendar_year"] == 2023
    assert bool(flagged["is_expected_year_rollover"])


# ---------------------------------------------------------------------------
# The two-day uncovered calendar gap
# ---------------------------------------------------------------------------

def test_two_day_calendar_gap_is_detected_with_exact_dates():
    """2025-12-27 and 2025-12-28 are reported as uncovered."""

    df = make_frame(
        make_period(2026, 53, "12/20/2025", "12/26/2025"),
        make_period(2026, 1, "12/29/2025", "1/4/2026"),
    )

    periods = periods_of(df)
    continuity = validator.build_calendar_continuity(periods)

    gaps = validator.find_calendar_gaps(continuity)

    assert len(gaps) == 1
    assert gaps.iloc[0]["calendar_gap_days"] == 2

    known = validator.build_known_calendar_gaps(periods, continuity)

    assert len(known) == 1

    gap = known.iloc[0]

    assert gap["gap_start"] == pd.Timestamp("2025-12-27")
    assert gap["gap_end"] == pd.Timestamp("2025-12-28")
    assert gap["gap_days"] == 2
    assert gap["previous_period_end"] == pd.Timestamp("2025-12-26")
    assert gap["next_period_start"] == pd.Timestamp("2025-12-29")

    assert list(known.columns) == [
        "previous_period_start",
        "previous_period_end",
        "next_period_start",
        "next_period_end",
        "gap_start",
        "gap_end",
        "gap_days",
        "reason",
    ]


def test_expected_next_start_is_the_day_after_the_previous_end():
    """Continuity is measured from the previous end date."""

    df = make_frame(
        make_period(2026, 53, "12/20/2025", "12/26/2025"),
        make_period(2026, 1, "12/29/2025", "1/4/2026"),
    )

    periods = periods_of(df)
    row = periods.iloc[1]

    assert row["previous_end_date"] == pd.Timestamp("2025-12-26")
    assert row["expected_next_start"] == pd.Timestamp("2025-12-27")
    assert row["calendar_gap_days"] == 2
    assert row["calendar_overlap_days"] == 0
    assert row["calendar_gap_before_days"] == 2


def test_undocumented_gap_is_a_failure():
    """A gap with no accepted explanation is classified FAIL."""

    df = make_frame(
        make_period(2015, 10, "3/7/2015", "3/13/2015"),
        make_period(2015, 12, "3/21/2015", "3/27/2015"),
    )

    periods = periods_of(df)
    continuity = validator.build_calendar_continuity(periods)

    gaps = validator.find_calendar_gaps(continuity)

    assert len(gaps) == 1
    assert gaps.iloc[0]["calendar_gap_days"] == 7
    assert gaps.iloc[0]["classification"] == validator.FAIL


def test_overlapping_periods_are_a_failure():
    """Double-counted days are classified FAIL."""

    df = make_frame(
        make_period(2015, 10, "3/7/2015", "3/13/2015"),
        make_period(2015, 11, "3/12/2015", "3/18/2015"),
    )

    periods = periods_of(df)
    continuity = validator.build_calendar_continuity(periods)

    overlaps = validator.find_calendar_overlaps(continuity)

    assert len(overlaps) == 1
    assert overlaps.iloc[0]["calendar_overlap_days"] == 2
    assert overlaps.iloc[0]["classification"] == validator.FAIL


# ---------------------------------------------------------------------------
# Reporting-area coverage
# ---------------------------------------------------------------------------

def test_missing_puttalam_is_reported_as_a_missing_source_record():
    """A period with 25 of 26 reporting areas is incomplete, not repaired."""

    without_puttalam = [
        area for area in REPORTING_AREAS if area != "Puttalam"
    ]

    df = make_frame(
        make_period(2026, 6, "2/2/2026", "2/8/2026"),
        make_period(2026, 7, "2/9/2026", "2/15/2026", areas=without_puttalam),
        make_period(2026, 8, "2/16/2026", "2/22/2026"),
    )

    derived = derive(df)
    periods = validator.build_reporting_periods(derived)

    coverage = validator.build_reporting_area_coverage(derived, periods)

    incomplete = validator.find_incomplete_reporting_area_coverage(coverage)

    assert len(incomplete) == 1

    row = incomplete.iloc[0]

    assert row["source_year"] == 2026
    assert row["source_week"] == 7
    assert row["reporting_area_count"] == 25
    assert row["missing_reporting_areas"] == "Puttalam"
    assert not bool(row["is_complete_source_coverage"])

    # The period is not dropped and no Puttalam row is invented.
    assert len(periods) == 3
    assert len(derived) == len(df)

    puttalam_rows = derived.loc[
        derived["district"].eq("Puttalam")
        & derived["week"].eq(7)
    ]

    assert puttalam_rows.empty


def test_complete_period_has_twenty_six_source_reporting_areas():
    """Coverage is validated against 26 areas, before Kalmune is merged."""

    assert validator.EXPECTED_SOURCE_REPORTING_AREAS == 26

    df = make_frame(make_period(2026, 6, "2/2/2026", "2/8/2026"))

    derived = derive(df)
    periods = validator.build_reporting_periods(derived)

    coverage = validator.build_reporting_area_coverage(derived, periods)

    assert coverage.iloc[0]["reporting_area_count"] == 26
    assert bool(coverage.iloc[0]["is_complete_source_coverage"])
    assert coverage.iloc[0]["missing_reporting_areas"] == ""


# ---------------------------------------------------------------------------
# Impossible intervals, duplicates and case values
# ---------------------------------------------------------------------------

def test_reversed_date_interval_is_impossible():
    """A start date after the end date is a FAIL, not a warning."""

    df = make_frame(
        make_period(2015, 10, "3/13/2015", "3/7/2015"),
    )

    derived = derive(df)

    impossible = validator.find_impossible_date_ranges(derived)

    assert len(impossible) == len(REPORTING_AREAS)
    assert (impossible["reporting_days"] < 0).all()

    validation = validator.validate_weekly_dengue_data(df)
    severity, findings = validation["checks"]["impossible_date_ranges"]

    assert validator.classify_check(severity, findings) == validator.FAIL


def test_unparseable_date_is_a_failure():
    """A date that does not match the source format is a FAIL."""

    df = make_frame(
        make_period(2015, 10, "not-a-date", "3/13/2015"),
    )

    derived = derive(df)

    assert len(validator.find_invalid_start_dates(derived)) == len(
        REPORTING_AREAS
    )
    assert validator.find_invalid_end_dates(derived).empty


def test_duplicate_reporting_area_period_is_a_failure():
    """A reporting area appearing twice in one interval is a FAIL."""

    df = make_frame(
        make_period(2015, 10, "3/7/2015", "3/13/2015"),
        make_period(2015, 10, "3/7/2015", "3/13/2015", areas=["Colombo"]),
    )

    derived = derive(df)

    duplicates = validator.find_duplicate_reporting_area_periods(derived)

    assert len(duplicates) == 2
    assert set(duplicates["district"]) == {"Colombo"}

    validation = validator.validate_weekly_dengue_data(df)
    severity, findings = validation["checks"][
        "duplicate_reporting_area_periods"
    ]

    assert validator.classify_check(severity, findings) == validator.FAIL


def test_duplicate_source_year_week_with_different_intervals_is_a_failure():
    """One year-week label must not describe two different intervals."""

    df = make_frame(
        make_period(2015, 10, "3/7/2015", "3/13/2015"),
        make_period(2015, 10, "3/14/2015", "3/20/2015"),
    )

    periods = periods_of(df)

    duplicates = validator.find_duplicate_source_year_weeks(periods)

    assert len(duplicates) == 2
    assert set(duplicates["source_week"]) == {10}


def test_identical_interval_with_different_labels_is_a_failure():
    """One interval must not carry two different year-week labels."""

    df = make_frame(
        make_period(2015, 10, "3/7/2015", "3/13/2015"),
        make_period(2015, 11, "3/7/2015", "3/13/2015"),
    )

    duplicates = validator.find_duplicate_date_intervals(derive(df))

    assert len(duplicates) == 2
    assert set(duplicates["week"]) == {10, 11}


@pytest.mark.parametrize("bad_value", [-1, -25, "abc", 3.5, None])
def test_invalid_case_values_are_detected(bad_value):
    """Negative, fractional, text and missing case values are FAIL."""

    df = make_frame(
        make_period(2015, 10, "3/7/2015", "3/13/2015", areas=["Colombo"]),
    )

    df["cases"] = df["cases"].astype("object")
    df.loc[0, "cases"] = bad_value

    invalid = validator.find_invalid_case_values(derive(df))

    assert len(invalid) == 1


def test_valid_zero_case_count_is_accepted():
    """Zero is a legitimate observed count and must not be flagged."""

    df = make_frame(
        make_period(2015, 10, "3/7/2015", "3/13/2015", cases=0),
    )

    assert validator.find_invalid_case_values(derive(df)).empty


# ---------------------------------------------------------------------------
# Model-support columns
# ---------------------------------------------------------------------------

def test_model_support_columns_are_attached_without_losing_rows():
    """The support columns join back onto every source row exactly once."""

    without_puttalam = [
        area for area in REPORTING_AREAS if area != "Puttalam"
    ]

    df = make_frame(
        make_period(2026, 53, "12/20/2025", "12/26/2025"),
        make_period(2026, 1, "12/29/2025", "1/4/2026"),
        make_period(2026, 2, "1/5/2026", "1/11/2026", areas=without_puttalam),
    )

    validation = validator.validate_weekly_dengue_data(df)

    supported = validator.attach_model_support_columns(validation)

    assert len(supported) == len(df)

    for column in [
        "reporting_days",
        "is_irregular_period",
        "calendar_gap_before_days",
        "source_weekday_convention",
        "is_incomplete_reporting_area_period",
    ]:
        assert column in supported.columns

    # Source columns survive the round trip.
    pd.testing.assert_frame_equal(
        supported[validator.SOURCE_COLUMNS].reset_index(drop=True),
        df.reset_index(drop=True),
    )

    gap_flagged = supported.loc[
        supported["start_date"].eq(pd.Timestamp("2025-12-29"))
    ]

    assert (gap_flagged["calendar_gap_before_days"] == 2).all()

    incomplete = supported.loc[
        supported["is_incomplete_reporting_area_period"]
    ]

    assert set(incomplete["week"]) == {2}


def test_validation_does_not_compute_weather_lags():
    """Weather lags belong to preprocessing, not to validation."""

    df = make_frame(make_period(2015, 10, "3/7/2015", "3/13/2015"))

    validation = validator.validate_weekly_dengue_data(df)

    supported = validator.attach_model_support_columns(validation)

    weather_columns = [
        column
        for column in supported.columns
        if "lag" in column.lower()
        or any(
            token in column.lower()
            for token in ("rain", "temp", "humid", "precip", "weather")
        )
    ]

    assert weather_columns == []


# ---------------------------------------------------------------------------
# Integration: the confirmed findings on the real source file
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_validation():
    """Validate the real source file once for the integration tests."""

    if not RAW_PATH.exists():
        pytest.skip(f"Raw dengue file not available: {RAW_PATH}")

    return validator.validate_weekly_dengue_data(pd.read_csv(RAW_PATH))


def status_of(validation: dict, check: str) -> str:
    """Return the classification of one check."""

    severity, findings = validation["checks"][check]

    return validator.classify_check(severity, findings)


def test_real_file_has_twenty_six_source_reporting_areas(real_validation):
    """Kalmune is still separate before canonical district merging."""

    areas = real_validation["rows"]["district"].nunique()

    assert areas == validator.EXPECTED_SOURCE_REPORTING_AREAS == 26
    assert "Kalmune" in set(real_validation["rows"]["district"])


def test_real_file_dates_all_parse(real_validation):
    """Every date parses with the source format."""

    assert status_of(real_validation, "invalid_start_dates") == validator.PASS
    assert status_of(real_validation, "invalid_end_dates") == validator.PASS


def test_real_file_has_no_impossible_date_ranges(real_validation):
    """No interval is reversed or zero-length."""

    assert (
        status_of(real_validation, "impossible_date_ranges")
        == validator.PASS
    )


def test_real_file_has_exactly_two_irregular_periods(real_validation):
    """The confirmed 8-day and 6-day periods, reported as WARNING."""

    severity, findings = real_validation["checks"][
        "irregular_reporting_periods"
    ]

    assert validator.classify_check(severity, findings) == validator.WARNING
    assert len(findings) == 2
    assert findings["is_documented"].all()

    intervals = set(zip(findings["start_date"], findings["end_date"]))

    assert intervals == {
        (pd.Timestamp("2009-04-18"), pd.Timestamp("2009-04-25")),
        (pd.Timestamp("2009-05-24"), pd.Timestamp("2009-05-29")),
    }

    assert sorted(findings["reporting_days"]) == [6, 8]


def test_real_file_has_no_unexpected_weekdays(real_validation):
    """Convention-aware weekday validation passes on the real file."""

    assert status_of(real_validation, "unexpected_weekdays") == validator.PASS

    assert (
        status_of(real_validation, "weekday_convention_changes")
        == validator.WARNING
    )


def test_real_file_has_one_two_day_calendar_gap(real_validation):
    """The only gap is 2025-12-27 to 2025-12-28."""

    gaps = real_validation["known_calendar_gaps"]

    assert len(gaps) == 1

    gap = gaps.iloc[0]

    assert gap["gap_start"] == pd.Timestamp("2025-12-27")
    assert gap["gap_end"] == pd.Timestamp("2025-12-28")
    assert gap["gap_days"] == 2

    assert status_of(real_validation, "calendar_overlaps") == validator.PASS


def test_real_file_2026_week_53_precedes_2026_week_1(real_validation):
    """The mislabelled interval is positioned by its dates."""

    periods = real_validation["periods"].set_index(
        ["source_year", "source_week"]
    )

    week_53 = periods.loc[(2026, 53)]
    week_1 = periods.loc[(2026, 1)]

    assert week_53["start_date"] == pd.Timestamp("2025-12-20")
    assert week_53["end_date"] == pd.Timestamp("2025-12-26")
    assert week_53["period_id"] < week_1["period_id"]


def test_real_file_missing_puttalam_in_2026_week_7(real_validation):
    """The single coverage failure is Puttalam in 2026 week 7."""

    severity, findings = real_validation["checks"][
        "incomplete_reporting_area_coverage"
    ]

    assert validator.classify_check(severity, findings) == validator.FAIL
    assert len(findings) == 1

    row = findings.iloc[0]

    assert row["source_year"] == 2026
    assert row["source_week"] == 7
    assert row["missing_reporting_areas"] == "Puttalam"
    assert row["reporting_area_count"] == 25


def test_real_file_documented_failures_do_not_break_the_build(real_validation):
    """
    The two confirmed source defects are reported but are not regressions.

    They are permanent properties of the source, so the run must not fail on
    them every time.
    """

    assert validator.count_failed_checks(real_validation) == 2
    assert validator.count_unexpected_failures(real_validation) == 0


def test_an_extra_coverage_failure_counts_as_a_regression():
    """A second incomplete period beyond the documented one fails the run."""

    without_puttalam = [
        area for area in REPORTING_AREAS if area != "Puttalam"
    ]
    without_colombo = [
        area for area in REPORTING_AREAS if area != "Colombo"
    ]

    df = make_frame(
        make_period(2026, 6, "2/2/2026", "2/8/2026", areas=without_colombo),
        make_period(2026, 7, "2/9/2026", "2/15/2026", areas=without_puttalam),
    )

    validation = validator.validate_weekly_dengue_data(df)

    assert validator.count_unexpected_failures(validation) == 1


def test_an_undocumented_gap_counts_as_a_regression():
    """A calendar gap outside the documented seam fails the run."""

    df = make_frame(
        make_period(2015, 10, "3/7/2015", "3/13/2015"),
        make_period(2015, 12, "3/21/2015", "3/27/2015"),
    )

    validation = validator.validate_weekly_dengue_data(df)

    assert validator.count_unexpected_failures(validation) >= 1


def test_real_file_has_no_duplicates_or_invalid_cases(real_validation):
    """No duplicate records and no invalid case values."""

    for check in [
        "duplicate_reporting_area_periods",
        "duplicate_source_year_weeks",
        "duplicate_date_intervals",
        "missing_reporting_areas",
        "invalid_case_values",
    ]:
        assert status_of(real_validation, check) == validator.PASS, check

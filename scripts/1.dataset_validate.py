"""
Validate the raw weekly dengue data.

This script never modifies, drops, or rewrites the source CSV. It keeps the
original columns intact, adds derived columns, and classifies every finding
as PASS, WARNING or FAIL.

WARNING is used for documented source conventions that are usable and can be
carried into modelling. FAIL is reserved for genuinely missing, impossible,
duplicate or conflicting records.

Outputs:
    results/data_validation/dengue_validation_summary.md
    results/data_validation/irregular_reporting_periods.csv
    results/data_validation/weekday_conventions.csv

The known-gaps and incomplete-coverage tables report the same two documented
defects keyed by period_id, so they are written by 2.create_calendar.py.
"""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
RESULTS_DIR = PROJECT_DIR / "results" / "data_validation"

RAW_PATH = RAW_DATA_DIR / "srilanka_weekly_data.csv"

SUMMARY_PATH = RESULTS_DIR / "dengue_validation_summary.md"
IRREGULAR_PERIODS_PATH = RESULTS_DIR / "irregular_reporting_periods.csv"
WEEKDAY_CONVENTIONS_PATH = RESULTS_DIR / "weekday_conventions.csv"

DATE_FORMAT = "%m/%d/%Y"

EXPECTED_REPORTING_DAYS = 7

# The raw file carries 26 reporting areas because Kalmune is reported
# separately. Kalmune is merged into Ampara later to produce the 25 canonical
# districts. Coverage is therefore validated against 26 at this stage.
EXPECTED_SOURCE_REPORTING_AREAS = 26

SOURCE_COLUMNS = [
    "year",
    "week",
    "start.date",
    "end.date",
    "district",
    "cases",
]

PASS = "PASS"
WARNING = "WARNING"
FAIL = "FAIL"


# ---------------------------------------------------------------------------
# Documented source conventions
# ---------------------------------------------------------------------------

# Observed start-weekday conventions, in chronological order. valid_to is
# inclusive. The 2009 episode is a matched pair: an 8-day period (week 17)
# pushed reporting onto Sunday, and a 6-day period (week 22) restored
# Saturday. Week 17 itself still starts on a Saturday, so the Sunday window
# begins at week 18.
WEEKDAY_CONVENTIONS = [
    {
        "convention_id": 1,
        "valid_from": pd.Timestamp("2006-12-23"),
        "valid_to": pd.Timestamp("2009-04-25"),
        "expected_start_weekday": "Saturday",
        "description": (
            "Original Saturday-to-Friday convention, including the 8-day "
            "2009 week 17 period which still starts on a Saturday."
        ),
    },
    {
        "convention_id": 2,
        "valid_from": pd.Timestamp("2009-04-26"),
        "valid_to": pd.Timestamp("2009-05-29"),
        "expected_start_weekday": "Sunday",
        "description": (
            "Temporary Sunday start caused by the 8-day 2009 week 17 "
            "period. Ends with the 6-day 2009 week 22 period, which "
            "restores the Saturday phase."
        ),
    },
    {
        "convention_id": 3,
        "valid_from": pd.Timestamp("2009-05-30"),
        "valid_to": pd.Timestamp("2025-12-26"),
        "expected_start_weekday": "Saturday",
        "description": (
            "Saturday convention restored from 2009 week 23 until the last "
            "Saturday-start period, 2025-12-20 to 2025-12-26."
        ),
    },
    {
        "convention_id": 4,
        "valid_from": pd.Timestamp("2025-12-29"),
        "valid_to": pd.NaT,
        "expected_start_weekday": "Monday",
        "description": (
            "Permanent Monday-to-Sunday convention from the period labelled "
            "2026 week 1 onward."
        ),
    },
]

# Calendar discontinuities that are documented source behaviour. Anything not
# listed here is reported as a FAIL. The two 2025 days are genuinely absent
# from the source, so the gap is recorded rather than filled.
ACCEPTED_CALENDAR_GAPS = [
    {
        "previous_period_end": pd.Timestamp("2025-12-26"),
        "next_period_start": pd.Timestamp("2025-12-29"),
        "reason": (
            "Seam between the Saturday and Monday reporting conventions. "
            "2025-12-27 and 2025-12-28 are not covered by any reporting "
            "period in the source. Carried as an unobserved interval, never "
            "interpolated."
        ),
        "classification": FAIL,
    },
]

# Number of confirmed incomplete-coverage periods in the source: the missing
# Puttalam record in 2026 week 7. A higher count is a regression.
EXPECTED_COVERAGE_ISSUES = 1

# Reporting periods whose inclusive length is not seven days but which are
# confirmed source behaviour rather than data errors.
KNOWN_IRREGULAR_PERIODS = [
    {
        "start_date": pd.Timestamp("2009-04-18"),
        "end_date": pd.Timestamp("2009-04-25"),
        "reporting_days": 8,
        "reason": (
            "Source extended 2009 week 17 by one day, moving reporting "
            "starts from Saturday to Sunday."
        ),
    },
    {
        "start_date": pd.Timestamp("2009-05-24"),
        "end_date": pd.Timestamp("2009-05-29"),
        "reporting_days": 6,
        "reason": (
            "Source shortened 2009 week 22 by one day, restoring Saturday "
            "reporting from week 23."
        ),
    },
]


# ---------------------------------------------------------------------------
# Loading and derived columns
# ---------------------------------------------------------------------------

def validate_required_columns(df: pd.DataFrame) -> None:
    """Ensure that all required source columns exist."""

    missing_columns = set(SOURCE_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )


def add_derived_columns(
    df: pd.DataFrame,
    date_format: str = DATE_FORMAT,
) -> pd.DataFrame:
    """
    Add derived columns without touching the source columns.

    The source columns year, week, start.date, end.date, district and cases
    are preserved exactly as read. Everything the validator needs is added
    alongside them.
    """

    result = df.copy()

    result["start_date"] = pd.to_datetime(
        result["start.date"],
        format=date_format,
        errors="coerce",
    )

    result["end_date"] = pd.to_datetime(
        result["end.date"],
        format=date_format,
        errors="coerce",
    )

    # Inclusive length: a Saturday-to-Friday week is 7 days.
    result["reporting_days"] = (
        result["end_date"] - result["start_date"]
    ).dt.days + 1

    result["start_weekday"] = result["start_date"].dt.day_name()

    result["is_irregular_period"] = result["reporting_days"].ne(
        EXPECTED_REPORTING_DAYS
    )

    return result


def build_weekday_convention_table() -> pd.DataFrame:
    """Return the documented start-weekday conventions as a table."""

    return pd.DataFrame(WEEKDAY_CONVENTIONS)[
        [
            "convention_id",
            "valid_from",
            "valid_to",
            "expected_start_weekday",
            "description",
        ]
    ]


def expected_start_weekday(start_date: pd.Timestamp) -> str | None:
    """
    Return the expected start weekday for a date, or None if unknown.

    Dates before the first convention window return None so that they are
    not silently validated against the wrong convention.
    """

    if pd.isna(start_date):
        return None

    for convention in WEEKDAY_CONVENTIONS:
        after_start = start_date >= convention["valid_from"]

        before_end = (
            pd.isna(convention["valid_to"])
            or start_date <= convention["valid_to"]
        )

        if after_start and before_end:
            return convention["expected_start_weekday"]

    return None


def build_reporting_periods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one record per unique reporting interval, in chronological order.

    period_id is a stable sequential identifier assigned by sorting on
    start_date then end_date. It is the primary key for sorting, merging,
    lagging and model windowing. The source year and week are retained as
    source_year and source_week and are never overwritten.
    """

    periods = (
        df.loc[
            df["start_date"].notna() & df["end_date"].notna(),
            ["year", "week", "start.date", "end.date", "start_date", "end_date"],
        ]
        .drop_duplicates(subset=["start_date", "end_date"])
        .sort_values(["start_date", "end_date"])
        .reset_index(drop=True)
    )

    periods = periods.rename(
        columns={
            "year": "source_year",
            "week": "source_week",
            "start.date": "source_start_date",
            "end.date": "source_end_date",
        }
    )

    periods["period_id"] = range(1, len(periods) + 1)

    periods["reporting_days"] = (
        periods["end_date"] - periods["start_date"]
    ).dt.days + 1

    periods["is_irregular_period"] = periods["reporting_days"].ne(
        EXPECTED_REPORTING_DAYS
    )

    periods["start_weekday"] = periods["start_date"].dt.day_name()
    periods["end_weekday"] = periods["end_date"].dt.day_name()

    periods["expected_start_weekday"] = periods["start_date"].map(
        expected_start_weekday
    )

    periods["source_weekday_convention"] = periods["start_date"].map(
        _convention_id_for
    )

    # Calendar continuity, measured from the previous period's end date
    # rather than from the previous start date. A change in reporting length
    # must not be reported a second time as a spacing error.
    periods["previous_end_date"] = periods["end_date"].shift(1)

    periods["expected_next_start"] = periods["previous_end_date"] + pd.Timedelta(
        days=1
    )

    days_since_previous_end = (
        periods["start_date"] - periods["previous_end_date"]
    ).dt.days

    periods["calendar_gap_days"] = (
        (days_since_previous_end - 1).clip(lower=0).astype("Int64")
    )

    periods["calendar_overlap_days"] = (
        (1 - days_since_previous_end).clip(lower=0).astype("Int64")
    )

    periods["calendar_gap_before_days"] = periods["calendar_gap_days"]

    # Human-readable chronological labels. period_id remains the key used
    # for ordering; these exist only for reading and plotting.
    periods["reporting_sequence_year"] = periods["start_date"].dt.year

    periods["reporting_sequence_number"] = (
        periods.groupby("reporting_sequence_year").cumcount() + 1
    )

    return periods


def _convention_id_for(start_date: pd.Timestamp) -> "int | None":
    """Return the convention_id covering a start date, or None."""

    if pd.isna(start_date):
        return None

    for convention in WEEKDAY_CONVENTIONS:
        after_start = start_date >= convention["valid_from"]

        before_end = (
            pd.isna(convention["valid_to"])
            or start_date <= convention["valid_to"]
        )

        if after_start and before_end:
            return convention["convention_id"]

    return None


# ---------------------------------------------------------------------------
# Row-level date checks
# ---------------------------------------------------------------------------

def find_invalid_start_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where start.date could not be parsed. FAIL."""

    return df.loc[
        df["start_date"].isna(),
        ["year", "week", "district", "start.date"],
    ].copy()


def find_invalid_end_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where end.date could not be parsed. FAIL."""

    return df.loc[
        df["end_date"].isna(),
        ["year", "week", "district", "end.date"],
    ].copy()


def find_impossible_date_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows whose end date does not follow their start date.

    Only genuinely impossible intervals are reported here. Periods that are
    simply not seven days long are legitimate source behaviour and are
    reported separately as irregular reporting periods.
    """

    valid_dates = (
        df["start_date"].notna()
        & df["end_date"].notna()
    )

    day_difference = (
        df["end_date"] - df["start_date"]
    ).dt.days

    invalid_mask = valid_dates & day_difference.lt(0)

    return df.loc[
        invalid_mask,
        [
            "year",
            "week",
            "district",
            "start.date",
            "end.date",
            "start_date",
            "end_date",
            "reporting_days",
        ],
    ].copy()


def find_invalid_case_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows whose case count is missing, non-numeric or negative.

    Case counts are weekly notification totals, so any value that is not a
    non-negative whole number is a source defect.
    """

    numeric_cases = pd.to_numeric(df["cases"], errors="coerce")

    invalid_mask = (
        numeric_cases.isna()
        | numeric_cases.lt(0)
        | numeric_cases.ne(numeric_cases.round())
    )

    result = df.loc[
        invalid_mask,
        ["year", "week", "start_date", "end_date", "district", "cases"],
    ].copy()

    result["numeric_cases"] = numeric_cases.loc[invalid_mask]

    return result


def find_irregular_reporting_periods(periods: pd.DataFrame) -> pd.DataFrame:
    """
    Return unique reporting periods whose length is not seven days. WARNING.

    These periods are usable. Each is annotated with a reason when it matches
    a documented source irregularity.
    """

    irregular = periods.loc[periods["is_irregular_period"]].copy()

    if irregular.empty:
        return irregular.assign(is_documented=[], reason=[])

    known = {
        (item["start_date"], item["end_date"]): item["reason"]
        for item in KNOWN_IRREGULAR_PERIODS
    }

    irregular["reason"] = [
        known.get(
            (row["start_date"], row["end_date"]),
            "Undocumented irregular reporting length. Review before use.",
        )
        for _, row in irregular.iterrows()
    ]

    irregular["is_documented"] = [
        (row["start_date"], row["end_date"]) in known
        for _, row in irregular.iterrows()
    ]

    return irregular[
        [
            "period_id",
            "source_year",
            "source_week",
            "start_date",
            "end_date",
            "reporting_days",
            "start_weekday",
            "end_weekday",
            "is_documented",
            "reason",
        ]
    ].copy()


# ---------------------------------------------------------------------------
# Calendar continuity
# ---------------------------------------------------------------------------

def _accepted_gap_reason(
    previous_end: pd.Timestamp,
    next_start: pd.Timestamp,
) -> "dict | None":
    """Return the accepted-gap entry matching a discontinuity, or None."""

    for accepted in ACCEPTED_CALENDAR_GAPS:
        if (
            accepted["previous_period_end"] == previous_end
            and accepted["next_period_start"] == next_start
        ):
            return accepted

    return None


def build_calendar_continuity(periods: pd.DataFrame) -> pd.DataFrame:
    """
    Classify calendar continuity between consecutive reporting periods.

    PASS when a period starts the day after the previous one ends.
    WARNING when a documented convention change explains the difference.
    FAIL when days are uncovered or double-counted without an explanation.

    The 2009 start-date shifts are not counted here at all: the calendar is
    continuous across them because the 8-day and 6-day periods absorb the
    phase change.
    """

    continuity = periods.loc[
        periods["previous_end_date"].notna(),
        [
            "period_id",
            "source_year",
            "source_week",
            "start_date",
            "end_date",
            "previous_end_date",
            "expected_next_start",
            "calendar_gap_days",
            "calendar_overlap_days",
        ],
    ].copy()

    def classify(row: pd.Series) -> "tuple[str, str]":
        gap = int(row["calendar_gap_days"])
        overlap = int(row["calendar_overlap_days"])

        if gap == 0 and overlap == 0:
            return PASS, "Continuous with the previous reporting period."

        accepted = _accepted_gap_reason(
            row["previous_end_date"], row["start_date"]
        )

        if accepted is not None:
            return accepted["classification"], accepted["reason"]

        if gap > 0:
            return (
                FAIL,
                f"{gap} calendar day(s) uncovered after "
                f"{row['previous_end_date'].date()} with no accepted "
                "explanation.",
            )

        return (
            FAIL,
            f"Overlaps the previous period by {overlap} day(s) with no "
            "accepted explanation.",
        )

    if continuity.empty:
        return continuity.assign(
            classification=pd.Series(dtype="object"),
            reason=pd.Series(dtype="object"),
        )

    classified = continuity.apply(classify, axis=1, result_type="expand")

    continuity["classification"] = classified[0]
    continuity["reason"] = classified[1]

    return continuity


def find_calendar_gaps(continuity: pd.DataFrame) -> pd.DataFrame:
    """Return discontinuities that leave calendar days uncovered."""

    return continuity.loc[
        continuity["calendar_gap_days"].fillna(0).gt(0)
    ].copy()


def find_calendar_overlaps(continuity: pd.DataFrame) -> pd.DataFrame:
    """Return discontinuities where reporting periods overlap."""

    return continuity.loc[
        continuity["calendar_overlap_days"].fillna(0).gt(0)
    ].copy()


def build_known_calendar_gaps(
    periods: pd.DataFrame,
    continuity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the known-calendar-gaps table.

    One row per uncovered interval, with the exact uncovered dates so that
    downstream models can treat the interval as unobserved.
    """

    gaps = find_calendar_gaps(continuity)

    previous_starts = periods.set_index("end_date")["start_date"]

    records = []

    for _, row in gaps.iterrows():
        previous_end = row["previous_end_date"]

        gap_start = previous_end + pd.Timedelta(days=1)
        gap_end = row["start_date"] - pd.Timedelta(days=1)

        records.append(
            {
                "previous_period_start": previous_starts.get(previous_end),
                "previous_period_end": previous_end,
                "next_period_start": row["start_date"],
                "next_period_end": row["end_date"],
                "gap_start": gap_start,
                "gap_end": gap_end,
                "gap_days": int(row["calendar_gap_days"]),
                "reason": row["reason"],
            }
        )

    columns = [
        "previous_period_start",
        "previous_period_end",
        "next_period_start",
        "next_period_end",
        "gap_start",
        "gap_end",
        "gap_days",
        "reason",
    ]

    return pd.DataFrame(records, columns=columns)


# ---------------------------------------------------------------------------
# Weekday validation
# ---------------------------------------------------------------------------

def find_unexpected_start_weekdays(periods: pd.DataFrame) -> pd.DataFrame:
    """
    Return periods whose start weekday breaks its own convention. FAIL.

    Each period is compared with the convention in force on its start date,
    so the 2009 Sunday episode and the 2026 Monday periods are expected
    rather than flagged.
    """

    comparable = periods["expected_start_weekday"].notna()

    invalid_mask = comparable & periods["start_weekday"].ne(
        periods["expected_start_weekday"]
    )

    return periods.loc[
        invalid_mask,
        [
            "period_id",
            "source_year",
            "source_week",
            "start_date",
            "end_date",
            "start_weekday",
            "expected_start_weekday",
            "source_weekday_convention",
        ],
    ].copy()


def find_weekday_convention_changes(periods: pd.DataFrame) -> pd.DataFrame:
    """
    Return the points where the start-weekday convention changes. WARNING.

    Convention changes are reported separately from unexpected weekdays so
    that a documented change is never mistaken for an invalid date.
    """

    ordered = periods.sort_values("period_id")

    changed = ordered["source_weekday_convention"].ne(
        ordered["source_weekday_convention"].shift(1)
    ) & ordered["source_weekday_convention"].shift(1).notna()

    result = ordered.loc[
        changed,
        [
            "period_id",
            "source_year",
            "source_week",
            "start_date",
            "start_weekday",
            "source_weekday_convention",
        ],
    ].copy()

    result["previous_convention"] = (
        ordered["source_weekday_convention"].shift(1).loc[result.index]
    )

    result["previous_start_weekday"] = (
        ordered["start_weekday"].shift(1).loc[result.index]
    )

    return result


# ---------------------------------------------------------------------------
# Source year-week label checks
# ---------------------------------------------------------------------------

def find_year_label_mismatches(periods: pd.DataFrame) -> pd.DataFrame:
    """
    Return periods whose dates mostly belong to another calendar year.

    A week 1 that starts in late December is the source's normal convention,
    so the majority calendar year is compared and week 1 is treated as
    expected. Labels are reported, never rewritten.
    """

    def majority_year(row: pd.Series) -> "int | None":
        covered = pd.date_range(row["start_date"], row["end_date"])

        # An impossible interval covers no days. It is reported by the
        # impossible-date-range check, not reinterpreted here.
        if len(covered) == 0:
            return None

        return int(covered.year.value_counts().idxmax())

    result = periods.copy()

    if result.empty:
        result["majority_calendar_year"] = pd.Series(dtype="int64")
        result["is_expected_year_rollover"] = pd.Series(dtype="bool")

        return result[
            [
                "period_id",
                "source_year",
                "source_week",
                "start_date",
                "end_date",
                "majority_calendar_year",
                "is_expected_year_rollover",
            ]
        ]

    result["majority_calendar_year"] = result.apply(majority_year, axis=1)

    mismatched = result.loc[
        result["majority_calendar_year"].notna()
        & result["majority_calendar_year"].ne(result["source_year"])
    ].copy()

    mismatched["is_expected_year_rollover"] = mismatched["source_week"].eq(1)

    return mismatched[
        [
            "period_id",
            "source_year",
            "source_week",
            "start_date",
            "end_date",
            "majority_calendar_year",
            "is_expected_year_rollover",
        ]
    ]


def find_duplicate_source_year_weeks(periods: pd.DataFrame) -> pd.DataFrame:
    """Return source year-week labels reused for different date intervals."""

    duplicated = periods.duplicated(
        subset=["source_year", "source_week"],
        keep=False,
    )

    return periods.loc[
        duplicated,
        [
            "period_id",
            "source_year",
            "source_week",
            "start_date",
            "end_date",
        ],
    ].sort_values(["source_year", "source_week", "start_date"]).copy()


def find_duplicate_date_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Return identical date intervals carrying different year-week labels."""

    intervals = (
        df.loc[
            df["start_date"].notna() & df["end_date"].notna(),
            ["year", "week", "start_date", "end_date"],
        ]
        .drop_duplicates()
    )

    duplicated = intervals.duplicated(
        subset=["start_date", "end_date"],
        keep=False,
    )

    return intervals.loc[duplicated].sort_values(
        ["start_date", "year", "week"]
    ).copy()


def find_duplicate_reporting_area_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Return reporting areas recorded more than once in the same period."""

    valid = df.loc[
        df["start_date"].notna() & df["end_date"].notna(),
        ["year", "week", "start_date", "end_date", "district", "cases"],
    ]

    duplicated = valid.duplicated(
        subset=["start_date", "end_date", "district"],
        keep=False,
    )

    return valid.loc[duplicated].sort_values(
        ["start_date", "district"]
    ).copy()


def find_missing_reporting_areas(
    df: pd.DataFrame,
    expected_area_count: int = EXPECTED_SOURCE_REPORTING_AREAS,
) -> pd.DataFrame:
    """
    Return reporting areas missing from the source as a whole.

    This is the structural check: the source should carry every expected
    reporting area somewhere. An area that is present but absent from
    individual periods is a coverage issue, not a missing area, and is
    reported by build_reporting_area_coverage instead.
    """

    observed = sorted(df["district"].dropna().unique())

    if len(observed) >= expected_area_count:
        return pd.DataFrame(columns=["district", "reason"])

    return pd.DataFrame(
        {
            "district": ["<unknown>"],
            "reason": [
                f"Source carries {len(observed)} reporting areas, "
                f"expected {expected_area_count}."
            ],
        }
    )


def find_out_of_order_source_weeks(periods: pd.DataFrame) -> pd.DataFrame:
    """
    Return periods whose source week number breaks chronological order.

    Within one source year the week number should increase with the start
    date. The interval labelled 2026 week 53 covering 2025-12-20 to
    2025-12-26 breaks this and is positioned by its dates, before 2026
    week 1.
    """

    ordered = periods.sort_values("period_id").copy()

    ordered["previous_source_year"] = ordered["source_year"].shift(1)
    ordered["previous_source_week"] = ordered["source_week"].shift(1)

    same_year = ordered["source_year"].eq(ordered["previous_source_year"])

    not_increasing = ordered["source_week"].le(
        ordered["previous_source_week"]
    )

    return ordered.loc[
        same_year & not_increasing,
        [
            "period_id",
            "source_year",
            "source_week",
            "start_date",
            "end_date",
            "previous_source_year",
            "previous_source_week",
        ],
    ].copy()


# ---------------------------------------------------------------------------
# Reporting-area coverage
# ---------------------------------------------------------------------------

def build_reporting_area_coverage(
    df: pd.DataFrame,
    periods: pd.DataFrame,
    expected_area_count: int = EXPECTED_SOURCE_REPORTING_AREAS,
) -> pd.DataFrame:
    """
    Return one row per reporting period with its reporting-area count.

    Every period should carry one record per source reporting area. The
    count is taken over distinct district values so that a duplicated
    district cannot mask a missing one.
    """

    present = df.loc[
        df["start_date"].notna() & df["end_date"].notna(),
        ["start_date", "end_date", "district"],
    ].drop_duplicates()

    all_areas = set(present["district"].unique())

    observed = (
        present.groupby(["start_date", "end_date"])["district"]
        .agg(
            reporting_area_count="size",
            missing_reporting_areas=lambda areas: ", ".join(
                sorted(all_areas - set(areas))
            ),
        )
        .reset_index()
    )

    coverage = periods[
        [
            "period_id",
            "source_year",
            "source_week",
            "start_date",
            "end_date",
        ]
    ].merge(observed, on=["start_date", "end_date"], how="left")

    coverage["reporting_area_count"] = (
        coverage["reporting_area_count"].fillna(0).astype(int)
    )

    coverage["missing_reporting_areas"] = coverage[
        "missing_reporting_areas"
    ].fillna("")

    coverage["expected_reporting_area_count"] = expected_area_count

    coverage["missing_reporting_area_count"] = (
        expected_area_count - coverage["reporting_area_count"]
    )

    coverage["is_complete_source_coverage"] = coverage[
        "reporting_area_count"
    ].eq(expected_area_count)

    return coverage.sort_values("period_id").reset_index(drop=True)


def find_incomplete_reporting_area_coverage(
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Return reporting periods that do not carry every reporting area."""

    return coverage.loc[
        ~coverage["is_complete_source_coverage"],
        [
            "period_id",
            "source_year",
            "source_week",
            "start_date",
            "end_date",
            "reporting_area_count",
            "expected_reporting_area_count",
            "missing_reporting_area_count",
            "missing_reporting_areas",
            "is_complete_source_coverage",
        ],
    ].copy()


def find_non_consecutive_weeks(periods: pd.DataFrame) -> pd.DataFrame:
    """
    Return reporting periods whose start dates are not seven days apart.

    The check is performed once per unique reporting period instead of
    once per district.
    """

    ordered = periods.sort_values("period_id").copy()

    ordered["previous_start_date"] = ordered["start_date"].shift(1)

    ordered["start_gap_days"] = (
        ordered["start_date"] - ordered["previous_start_date"]
    ).dt.days

    invalid_mask = (
        ordered["previous_start_date"].notna()
        & ordered["start_gap_days"].ne(EXPECTED_REPORTING_DAYS)
    )

    return ordered.loc[
        invalid_mask,
        [
            "period_id",
            "source_year",
            "source_week",
            "previous_start_date",
            "start_date",
            "start_gap_days",
        ],
    ].copy()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def validate_weekly_dengue_data(
    df: pd.DataFrame,
    date_format: str = DATE_FORMAT,
) -> dict:
    """
    Run every validation check.

    Returns a dictionary with the derived row-level frame, the chronological
    period table, the coverage table, and one findings frame per check
    together with its severity.
    """

    validate_required_columns(df)

    derived = add_derived_columns(df, date_format=date_format)

    periods = build_reporting_periods(derived)
    continuity = build_calendar_continuity(periods)
    coverage = build_reporting_area_coverage(derived, periods)

    checks = {
        "invalid_start_dates": (
            FAIL,
            find_invalid_start_dates(derived),
        ),
        "invalid_end_dates": (
            FAIL,
            find_invalid_end_dates(derived),
        ),
        "impossible_date_ranges": (
            FAIL,
            find_impossible_date_ranges(derived),
        ),
        "irregular_reporting_periods": (
            WARNING,
            find_irregular_reporting_periods(periods),
        ),
        "calendar_gaps": (
            FAIL,
            find_calendar_gaps(continuity),
        ),
        "calendar_overlaps": (
            FAIL,
            find_calendar_overlaps(continuity),
        ),
        "unexpected_weekdays": (
            FAIL,
            find_unexpected_start_weekdays(periods),
        ),
        "weekday_convention_changes": (
            WARNING,
            find_weekday_convention_changes(periods),
        ),
        "source_year_label_mismatches": (
            WARNING,
            find_year_label_mismatches(periods),
        ),
        "duplicate_source_year_weeks": (
            FAIL,
            find_duplicate_source_year_weeks(periods),
        ),
        "duplicate_date_intervals": (
            FAIL,
            find_duplicate_date_intervals(derived),
        ),
        "out_of_order_source_weeks": (
            WARNING,
            find_out_of_order_source_weeks(periods),
        ),
        "duplicate_reporting_area_periods": (
            FAIL,
            find_duplicate_reporting_area_periods(derived),
        ),
        "missing_reporting_areas": (
            FAIL,
            find_missing_reporting_areas(derived),
        ),
        "incomplete_reporting_area_coverage": (
            FAIL,
            find_incomplete_reporting_area_coverage(coverage),
        ),
        "invalid_case_values": (
            FAIL,
            find_invalid_case_values(derived),
        ),
    }

    return {
        "rows": derived,
        "periods": periods,
        "continuity": continuity,
        "coverage": coverage,
        "weekday_conventions": build_weekday_convention_table(),
        "known_calendar_gaps": build_known_calendar_gaps(periods, continuity),
        "checks": checks,
    }


def classify_check(severity: str, findings: pd.DataFrame) -> str:
    """Return PASS when a check found nothing, otherwise its severity."""

    return PASS if findings.empty else severity


def count_failed_checks(validation: dict) -> int:
    """Return the number of checks classified as FAIL."""

    return sum(
        1
        for severity, findings in validation["checks"].values()
        if classify_check(severity, findings) == FAIL
    )


def count_unexpected_failures(validation: dict) -> int:
    """
    Return the number of FAIL findings that are not already documented.

    The two confirmed source defects — the 2-day calendar gap at the
    convention seam and the missing Puttalam record in 2026 week 7 — are
    permanent properties of the source. They are reported as FAIL because
    they are genuine data-quality issues, but they must not break the build
    on every run. Anything beyond them is a regression and does.
    """

    checks = validation["checks"]

    unexpected = 0

    for name, (severity, findings) in checks.items():
        if classify_check(severity, findings) != FAIL:
            continue

        if name == "calendar_gaps":
            # Only gaps matching an accepted entry are documented. The
            # known-gaps table is derived from the detected gaps, so it
            # cannot be used to discount them.
            undocumented = [
                row
                for _, row in findings.iterrows()
                if _accepted_gap_reason(
                    row["previous_end_date"], row["start_date"]
                )
                is None
            ]

            unexpected += len(undocumented)
        elif name == "incomplete_reporting_area_coverage":
            unexpected += max(len(findings) - EXPECTED_COVERAGE_ISSUES, 0)
        else:
            unexpected += len(findings)

    return unexpected


def attach_model_support_columns(validation: dict) -> pd.DataFrame:
    """
    Return the row-level frame with the model-support columns attached.

    These are the derived columns that later preprocessing should retain.
    Weather lags are deliberately not calculated here.
    """

    periods = validation["periods"]
    coverage = validation["coverage"]

    support = periods[
        [
            "period_id",
            "start_date",
            "end_date",
            "reporting_days",
            "is_irregular_period",
            "calendar_gap_before_days",
            "source_weekday_convention",
        ]
    ].merge(
        coverage[["period_id", "is_complete_source_coverage"]],
        on="period_id",
        how="left",
    )

    support["is_incomplete_reporting_area_period"] = ~support[
        "is_complete_source_coverage"
    ]

    support = support.drop(columns=["is_complete_source_coverage"])

    rows = validation["rows"].drop(
        columns=["reporting_days", "is_irregular_period"]
    )

    return rows.merge(support, on=["start_date", "end_date"], how="left")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_summary_table(validation: dict) -> pd.DataFrame:
    """Return one row per check with its classification and finding count."""

    records = [
        {
            "check": name,
            "status": classify_check(severity, findings),
            "findings": len(findings),
        }
        for name, (severity, findings) in validation["checks"].items()
    ]

    return pd.DataFrame(records)


def print_validation_summary(validation: dict) -> None:
    """Print the classification of each validation check."""

    print("\nValidation summary")
    print("-" * 62)

    for name, (severity, findings) in validation["checks"].items():
        status = classify_check(severity, findings)
        count = len(findings)

        if status == PASS:
            print(f"{PASS:<8} {name}")
        else:
            print(f"{status:<8} {name} - {count} finding(s)")


def write_validation_outputs(validation: dict) -> None:
    """Write the summary table and the supporting reference CSVs."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = build_summary_table(validation)

    SUMMARY_PATH.write_text(
        "# Dengue validation summary\n\n"
        + summary.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )

    validation["checks"]["irregular_reporting_periods"][1].to_csv(
        IRREGULAR_PERIODS_PATH, index=False
    )

    validation["weekday_conventions"].to_csv(
        WEEKDAY_CONVENTIONS_PATH, index=False
    )

    # known_calendar_gaps.csv and incomplete_reporting_area_coverage.csv are
    # written by 2.create_calendar.py, which reports the same two documented
    # defects against period_id. The checks stay here; only the files moved.


def main() -> int:
    """
    Validate the raw weekly dengue data.

    Returns a process exit code. Documented source defects are reported but
    do not fail the build; anything beyond them is a regression and does.
    """

    df = pd.read_csv(RAW_PATH)

    validation = validate_weekly_dengue_data(df)

    print_validation_summary(validation)

    write_validation_outputs(validation)

    unexpected = count_unexpected_failures(validation)

    if unexpected:
        print(f"\n{unexpected} undocumented validation failure(s).")
        return 1

    print("\nNo undocumented validation failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
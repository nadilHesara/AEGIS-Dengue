"""
Build the canonical chronological reporting calendar.

The source dengue CSV is never modified. This script reads it, derives one
row per unique reporting interval, sorts those intervals by their actual
dates, and assigns period_id as the primary chronological key.

period_id exists because the source year and week labels cannot order the
data. The interval labelled 2026 week 53 covers 2025-12-20 to 2025-12-26 and
therefore falls before 2026 week 1. Sorting on the raw labels would place it
almost a year too late.

The confirmed source findings are imported from the validator so that the
documented conventions, irregular periods and accepted gaps have exactly one
definition in the project.

Outputs:
    data/interim/reporting_calendar.csv
    results/data_validation/known_calendar_gaps.csv
    results/data_validation/source_week_label_anomalies.csv
    results/data_validation/incomplete_reporting_area_coverage.csv
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = PROJECT_DIR / "data" / "raw" / "srilanka_weekly_data.csv"

INTERIM_DIR = PROJECT_DIR / "data" / "interim"
RESULTS_DIR = PROJECT_DIR / "results" / "data_validation"

CALENDAR_PATH = INTERIM_DIR / "reporting_calendar.csv"
KNOWN_GAPS_PATH = RESULTS_DIR / "known_calendar_gaps.csv"
LABEL_ANOMALIES_PATH = RESULTS_DIR / "source_week_label_anomalies.csv"
COVERAGE_ISSUES_PATH = RESULTS_DIR / "incomplete_reporting_area_coverage.csv"

DATE_FORMAT = "%m/%d/%Y"
OUTPUT_DATE_FORMAT = "%Y-%m-%d"

EXPECTED_REPORTING_DAYS = 7


def _load_validator():
    """
    Import the validator module despite its non-identifier file name.

    The confirmed source findings live there as module constants. Importing
    them keeps this script from restating facts that are already documented
    and tested elsewhere.
    """

    path = PROJECT_DIR / "scripts" / "1.dataset_validate.py"

    spec = importlib.util.spec_from_file_location("dataset_validate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


validator = _load_validator()

WEEKDAY_CONVENTIONS = validator.WEEKDAY_CONVENTIONS
ACCEPTED_CALENDAR_GAPS = validator.ACCEPTED_CALENDAR_GAPS
EXPECTED_SOURCE_REPORTING_AREAS = validator.EXPECTED_SOURCE_REPORTING_AREAS

# Human-readable convention labels, keyed by the validator's convention_id.
CONVENTION_LABELS = {
    1: "saturday_friday",
    2: "temporary_sunday_sequence",
    3: "saturday_friday",
    4: "monday_sunday",
}

GAP_RECOMMENDED_HANDLING = (
    "Retain as unobserved calendar dates. Do not create a synthetic dengue "
    "reporting period. Do not assign zero dengue cases. Do not interpolate "
    "dengue cases."
)

LABEL_RECOMMENDED_HANDLING = (
    "Retain the source label unchanged as metadata. Use start_date, end_date "
    "and period_id for all chronological operations."
)


# ---------------------------------------------------------------------------
# 1. Load and parse the source data
# ---------------------------------------------------------------------------

def load_source_rows(
    path: Path = RAW_PATH,
    date_format: str = DATE_FORMAT,
) -> pd.DataFrame:
    """
    Load the district-level source rows and add parsed date columns.

    The original columns are preserved exactly as read. The source CSV is
    never written back.
    """

    df = pd.read_csv(path)

    return add_row_date_columns(df, date_format=date_format)


def add_row_date_columns(
    df: pd.DataFrame,
    date_format: str = DATE_FORMAT,
) -> pd.DataFrame:
    """Add parsed dates and period-shape columns to the district rows."""

    result = df.copy()

    result["start_date"] = pd.to_datetime(
        result["start.date"], format=date_format, errors="coerce"
    )

    result["end_date"] = pd.to_datetime(
        result["end.date"], format=date_format, errors="coerce"
    )

    # Inclusive length: a Saturday-to-Friday week is 7 days.
    result["reporting_days"] = (
        result["end_date"] - result["start_date"]
    ).dt.days + 1

    result["start_weekday"] = result["start_date"].dt.day_name()
    result["end_weekday"] = result["end_date"].dt.day_name()

    result["is_irregular_period"] = result["reporting_days"].ne(
        EXPECTED_REPORTING_DAYS
    )

    return result


# ---------------------------------------------------------------------------
# 2 and 3. One row per interval, ordered by actual dates
# ---------------------------------------------------------------------------

def build_reporting_calendar(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per unique reporting interval, in chronological order.

    Sorting is on start_date then end_date. The source labels are retained
    as source_year and source_week and are never used to order the calendar.
    """

    calendar = (
        rows.loc[
            rows["start_date"].notna() & rows["end_date"].notna(),
            [
                "year",
                "week",
                "start_date",
                "end_date",
                "reporting_days",
                "start_weekday",
                "end_weekday",
                "is_irregular_period",
            ],
        ]
        .drop_duplicates(subset=["start_date", "end_date"])
        .sort_values(["start_date", "end_date"])
        .reset_index(drop=True)
    )

    calendar = calendar.rename(
        columns={"year": "source_year", "week": "source_week"}
    )

    # period_id is assigned after the chronological sort, so it increases
    # strictly with the actual dates. It is the primary time key.
    calendar.insert(0, "period_id", range(1, len(calendar) + 1))

    return calendar


# ---------------------------------------------------------------------------
# 4. Calendar continuity
# ---------------------------------------------------------------------------

def add_calendar_continuity(calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Add the gap and overlap fields describing each period's predecessor.

    The first period has no predecessor, so its continuity fields are null
    and its gap and overlap counts are zero.
    """

    result = calendar.sort_values("period_id").copy()

    result["previous_period_id"] = result["period_id"].shift(1)
    result["previous_start_date"] = result["start_date"].shift(1)
    result["previous_end_date"] = result["end_date"].shift(1)

    result["expected_start_date"] = result[
        "previous_end_date"
    ] + pd.Timedelta(days=1)

    difference = (
        result["start_date"] - result["expected_start_date"]
    ).dt.days

    result["calendar_gap_before_days"] = (
        difference.clip(lower=0).fillna(0).astype(int)
    )

    result["calendar_overlap_before_days"] = (
        (-difference).clip(lower=0).fillna(0).astype(int)
    )

    result["is_calendar_continuous"] = result["start_date"].eq(
        result["expected_start_date"]
    )

    return result


# ---------------------------------------------------------------------------
# 5. Weekday conventions
# ---------------------------------------------------------------------------

def weekday_convention_for(start_date: pd.Timestamp) -> str | None:
    """Return the convention label covering a start date, or None."""

    if pd.isna(start_date):
        return None

    for convention in WEEKDAY_CONVENTIONS:
        after_start = start_date >= convention["valid_from"]

        before_end = (
            pd.isna(convention["valid_to"])
            or start_date <= convention["valid_to"]
        )

        if after_start and before_end:
            return CONVENTION_LABELS[convention["convention_id"]]

    return None


def add_weekday_conventions(calendar: pd.DataFrame) -> pd.DataFrame:
    """Label each period's weekday convention and mark the changes."""

    result = calendar.sort_values("period_id").copy()

    result["weekday_convention"] = result["start_date"].map(
        weekday_convention_for
    )

    previous_convention = result["weekday_convention"].shift(1)

    # The first period has no predecessor, so it is not a change.
    result["is_weekday_convention_change"] = (
        previous_convention.notna()
        & result["weekday_convention"].ne(previous_convention)
    )

    return result


# ---------------------------------------------------------------------------
# 6. Source-label anomalies
# ---------------------------------------------------------------------------

def add_source_label_fields(calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Add the calendar-year fields used to inspect source labels.

    source_year_differs_from_start_year is documentation only. A period
    starting in December and labelled as the following year is a normal
    consequence of week numbering, not an error.
    """

    result = calendar.copy()

    result["start_calendar_year"] = result["start_date"].dt.year
    result["end_calendar_year"] = result["end_date"].dt.year

    result["source_year_differs_from_start_year"] = result[
        "source_year"
    ].ne(result["start_calendar_year"])

    return result


def find_source_label_anomalies(calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Return the source year-week labels that cannot be trusted for ordering.

    Three anomaly types are detected:

    duplicate_source_label
        One source_year and source_week maps to more than one interval.

    duplicate_interval_label
        Different source labels map to the same interval.

    chronologically_misplaced_label
        Ordering by the source labels disagrees with ordering by dates.
        The interval labelled 2026 week 53 is detected here.
    """

    records = []

    duplicate_labels = calendar[
        calendar.duplicated(
            subset=["source_year", "source_week"], keep=False
        )
    ]

    for _, row in duplicate_labels.iterrows():
        records.append(
            {
                "period_id": row["period_id"],
                "source_year": row["source_year"],
                "source_week": row["source_week"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "anomaly_type": "duplicate_source_label",
                "description": (
                    f"Source label {row['source_year']} week "
                    f"{row['source_week']} describes more than one distinct "
                    f"reporting interval."
                ),
            }
        )

    duplicate_intervals = calendar[
        calendar.duplicated(subset=["start_date", "end_date"], keep=False)
    ]

    for _, row in duplicate_intervals.iterrows():
        records.append(
            {
                "period_id": row["period_id"],
                "source_year": row["source_year"],
                "source_week": row["source_week"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "anomaly_type": "duplicate_interval_label",
                "description": (
                    "More than one source label describes the interval "
                    f"{row['start_date'].date()} to {row['end_date'].date()}."
                ),
            }
        )

    # A label is misplaced when it breaks the otherwise increasing label
    # sequence. Comparing sorted positions instead would flag every period
    # after the offender, since one displaced row shifts all that follow.
    ordered = calendar.sort_values("period_id").reset_index(drop=True)

    # The source year alone is enough to place a period on the timeline:
    # every period's label year should match its neighbours' progression.
    # 2026 week 53 sits between 2025 week 52 and 2026 week 1, so its label
    # year jumps forward and back again. That round trip identifies it as
    # the outlier rather than the periods that merely follow it.
    misplaced_ids = []

    for index in range(1, len(ordered) - 1):
        previous_year = ordered.loc[index - 1, "source_year"]
        current_year = ordered.loc[index, "source_year"]
        next_year = ordered.loc[index + 1, "source_year"]

        current_week = ordered.loc[index, "source_week"]
        next_week = ordered.loc[index + 1, "source_week"]

        # A label whose year runs ahead of the period before it while the
        # period after it restarts that same year's week numbering is
        # labelled out of sequence.
        if (
            current_year > previous_year
            and next_year == current_year
            and next_week < current_week
        ):
            misplaced_ids.append(ordered.loc[index, "period_id"])

    for period_id in misplaced_ids:
        row = calendar.loc[calendar["period_id"].eq(period_id)].iloc[0]

        records.append(
            {
                "period_id": row["period_id"],
                "source_year": row["source_year"],
                "source_week": row["source_week"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "anomaly_type": "chronologically_misplaced_label",
                "description": (
                    f"Source label {row['source_year']} week "
                    f"{row['source_week']} sorts differently from the actual "
                    f"interval {row['start_date'].date()} to "
                    f"{row['end_date'].date()}."
                ),
            }
        )

    columns = [
        "period_id",
        "source_year",
        "source_week",
        "start_date",
        "end_date",
        "anomaly_type",
        "description",
    ]

    anomalies = pd.DataFrame(records, columns=columns)

    if anomalies.empty:
        anomalies["recommended_handling"] = pd.Series(dtype="object")
        return anomalies

    anomalies["recommended_handling"] = LABEL_RECOMMENDED_HANDLING

    return anomalies.drop_duplicates(
        subset=["period_id", "anomaly_type"]
    ).sort_values(["period_id", "anomaly_type"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. Reporting-area coverage
# ---------------------------------------------------------------------------

def add_reporting_area_coverage(
    calendar: pd.DataFrame,
    rows: pd.DataFrame,
    expected_area_count: int = EXPECTED_SOURCE_REPORTING_AREAS,
) -> pd.DataFrame:
    """
    Attach the reporting-area coverage of each period.

    The expected area list is every area observed anywhere in the source, so
    a district that vanishes for one period is named rather than silently
    counted. Missing records are never filled with zeros.
    """

    present = rows.loc[
        rows["start_date"].notna() & rows["end_date"].notna(),
        ["start_date", "end_date", "district"],
    ].drop_duplicates()

    all_areas = set(present["district"].dropna().unique())

    observed = (
        present.groupby(["start_date", "end_date"])["district"]
        .agg(
            source_reporting_area_count="size",
            source_reporting_areas=lambda areas: ", ".join(sorted(areas)),
            missing_source_reporting_areas=lambda areas: ", ".join(
                sorted(all_areas - set(areas))
            ),
        )
        .reset_index()
    )

    result = calendar.merge(observed, on=["start_date", "end_date"], how="left")

    result["source_reporting_area_count"] = (
        result["source_reporting_area_count"].fillna(0).astype(int)
    )

    for column in ["source_reporting_areas", "missing_source_reporting_areas"]:
        result[column] = result[column].fillna("")

    result["is_complete_source_coverage"] = result[
        "source_reporting_area_count"
    ].eq(expected_area_count)

    return result


# ---------------------------------------------------------------------------
# 8. Quality flags
# ---------------------------------------------------------------------------

def add_quality_flags(
    calendar: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the derived quality flags used to screen periods downstream.

    has_data_quality_issue marks periods that need a deliberate decision
    before use. The 2009 irregular periods are flagged as irregular but are
    fully usable, so irregularity alone does not set it.
    """

    result = calendar.copy()

    result["has_calendar_gap_before"] = result[
        "calendar_gap_before_days"
    ].gt(0)

    result["has_calendar_overlap_before"] = result[
        "calendar_overlap_before_days"
    ].gt(0)

    flagged = set(anomalies["period_id"]) if not anomalies.empty else set()

    result["has_source_label_anomaly"] = result["period_id"].isin(flagged)

    result["has_impossible_date_range"] = result["reporting_days"].le(0)

    result["has_data_quality_issue"] = (
        result["has_calendar_gap_before"]
        | result["has_calendar_overlap_before"]
        | ~result["is_complete_source_coverage"]
        | result["has_source_label_anomaly"]
        | result["has_impossible_date_range"]
    )

    return result


# ---------------------------------------------------------------------------
# 9. Known gaps
# ---------------------------------------------------------------------------

def _accepted_gap_reason(
    previous_end_date: pd.Timestamp,
    start_date: pd.Timestamp,
) -> str | None:
    """Return the documented reason for a gap, or None if undocumented."""

    for gap in ACCEPTED_CALENDAR_GAPS:
        if (
            gap["previous_period_end"] == previous_end_date
            and gap["next_period_start"] == start_date
        ):
            return gap["reason"]

    return None


def build_known_calendar_gaps(calendar: pd.DataFrame) -> pd.DataFrame:
    """Return one row per stretch of calendar dates no period covers."""

    gapped = calendar.loc[calendar["calendar_gap_before_days"].gt(0)]

    records = []

    for _, row in gapped.iterrows():
        reason = _accepted_gap_reason(
            row["previous_end_date"], row["start_date"]
        )

        records.append(
            {
                "previous_period_id": row["previous_period_id"],
                "previous_start_date": row["previous_start_date"],
                "previous_end_date": row["previous_end_date"],
                "next_period_id": row["period_id"],
                "next_start_date": row["start_date"],
                "next_end_date": row["end_date"],
                "gap_start_date": row["expected_start_date"],
                "gap_end_date": row["start_date"] - pd.Timedelta(days=1),
                "gap_days": row["calendar_gap_before_days"],
                "reason": reason or "Undocumented calendar gap.",
                "recommended_handling": GAP_RECOMMENDED_HANDLING,
            }
        )

    columns = [
        "previous_period_id",
        "previous_start_date",
        "previous_end_date",
        "next_period_id",
        "next_start_date",
        "next_end_date",
        "gap_start_date",
        "gap_end_date",
        "gap_days",
        "reason",
        "recommended_handling",
    ]

    return pd.DataFrame(records, columns=columns)


def build_incomplete_coverage(calendar: pd.DataFrame) -> pd.DataFrame:
    """Return the periods that do not carry every source reporting area."""

    return calendar.loc[
        ~calendar["is_complete_source_coverage"],
        [
            "period_id",
            "source_year",
            "source_week",
            "start_date",
            "end_date",
            "source_reporting_area_count",
            "missing_source_reporting_areas",
        ],
    ].copy()


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

CALENDAR_COLUMNS = [
    "period_id",
    "source_year",
    "source_week",
    "start_date",
    "end_date",
    "reporting_days",
    "start_weekday",
    "end_weekday",
    "weekday_convention",
    "is_weekday_convention_change",
    "is_irregular_period",
    "previous_period_id",
    "previous_start_date",
    "previous_end_date",
    "expected_start_date",
    "calendar_gap_before_days",
    "calendar_overlap_before_days",
    "is_calendar_continuous",
    "has_calendar_gap_before",
    "has_calendar_overlap_before",
    "start_calendar_year",
    "end_calendar_year",
    "source_year_differs_from_start_year",
    "source_reporting_area_count",
    "source_reporting_areas",
    "missing_source_reporting_areas",
    "is_complete_source_coverage",
    "has_source_label_anomaly",
    "has_impossible_date_range",
    "has_data_quality_issue",
]

DATE_COLUMNS = [
    "start_date",
    "end_date",
    "previous_start_date",
    "previous_end_date",
    "expected_start_date",
]


def build_calendar_artifacts(rows: pd.DataFrame) -> dict:
    """Build the calendar and its companion tables from the source rows."""

    calendar = build_reporting_calendar(rows)
    calendar = add_calendar_continuity(calendar)
    calendar = add_weekday_conventions(calendar)
    calendar = add_source_label_fields(calendar)
    calendar = add_reporting_area_coverage(calendar, rows)

    anomalies = find_source_label_anomalies(calendar)

    calendar = add_quality_flags(calendar, anomalies)

    calendar = calendar[CALENDAR_COLUMNS].sort_values(
        "period_id"
    ).reset_index(drop=True)

    return {
        "calendar": calendar,
        "known_gaps": build_known_calendar_gaps(calendar),
        "label_anomalies": anomalies,
        "incomplete_coverage": build_incomplete_coverage(calendar),
    }


def assign_period_ids(rows: pd.DataFrame, calendar: pd.DataFrame):
    """
    Attach period_id to every district-level row.

    The merge is on the exact interval, so a row can only match the single
    period covering it.
    """

    return rows.merge(
        calendar[["period_id", "start_date", "end_date"]],
        on=["start_date", "end_date"],
        how="left",
    )


# ---------------------------------------------------------------------------
# 13. Validation assertions
# ---------------------------------------------------------------------------

def run_calendar_assertions(rows: pd.DataFrame, artifacts: dict) -> None:
    """
    Verify the calendar's structural guarantees.

    These are hard invariants. If any fails the calendar must not be used as
    a time index, so the script stops rather than writing a broken file.
    """

    calendar = artifacts["calendar"]

    assert calendar["period_id"].is_unique, "period_id is not unique"

    chronological = calendar.sort_values(["start_date", "end_date"])

    assert chronological["period_id"].is_monotonic_increasing, (
        "period_id does not follow chronological date order"
    )

    assert calendar["start_date"].le(calendar["end_date"]).all(), (
        "a period starts after it ends"
    )

    assert calendar["reporting_days"].gt(0).all(), (
        "a period has a non-positive reporting_days value"
    )

    assert not calendar.duplicated(
        subset=["start_date", "end_date"]
    ).any(), "a reporting interval is duplicated"

    mapped = assign_period_ids(rows, calendar)

    assert len(mapped) == len(rows), (
        "mapping period_id changed the district-level row count"
    )

    assert mapped["period_id"].notna().all(), (
        "a district-level row maps to no period_id"
    )

    assert calendar["source_reporting_area_count"].le(
        EXPECTED_SOURCE_REPORTING_AREAS
    ).all(), "a period carries more than the expected reporting areas"

    # The two confirmed 2009 irregular periods survive unchanged.
    for start, end, days in [
        ("2009-04-18", "2009-04-25", 8),
        ("2009-05-24", "2009-05-29", 6),
    ]:
        match = calendar.loc[
            calendar["start_date"].eq(pd.Timestamp(start))
            & calendar["end_date"].eq(pd.Timestamp(end))
        ]

        assert len(match) == 1, f"the {days}-day 2009 period is missing"

        assert match.iloc[0]["reporting_days"] == days, (
            f"the 2009 period {start} to {end} is no longer {days} days"
        )

    # The two-day seam between the Saturday and Monday conventions.
    gaps = artifacts["known_gaps"]

    seam = gaps.loc[
        gaps["gap_start_date"].eq(pd.Timestamp("2025-12-27"))
        & gaps["gap_end_date"].eq(pd.Timestamp("2025-12-28"))
    ]

    assert len(seam) == 1, "the 2025 two-day calendar gap was not detected"
    assert seam.iloc[0]["gap_days"] == 2, "the 2025 gap is not two days"

    # 2026 week 53 covers December 2025 and must precede 2026 week 1.
    week_53 = calendar.loc[
        calendar["source_year"].eq(2026) & calendar["source_week"].eq(53)
    ].iloc[0]

    week_1 = calendar.loc[
        calendar["source_year"].eq(2026) & calendar["source_week"].eq(1)
    ].iloc[0]

    assert week_53["period_id"] < week_1["period_id"], (
        "2026 week 53 does not precede 2026 week 1"
    )

    assert week_53["has_source_label_anomaly"], (
        "2026 week 53 is not reported as a source-label anomaly"
    )

    # Puttalam is reported missing, never invented.
    coverage = artifacts["incomplete_coverage"]

    puttalam = coverage.loc[
        coverage["source_year"].eq(2026) & coverage["source_week"].eq(7)
    ]

    assert len(puttalam) == 1, "2026 week 7 is not reported as incomplete"

    assert "Puttalam" in puttalam.iloc[0]["missing_source_reporting_areas"], (
        "Puttalam is not named as missing from 2026 week 7"
    )

    assert mapped.loc[
        mapped["period_id"].eq(puttalam.iloc[0]["period_id"])
        & mapped["district"].eq("Puttalam")
    ].empty, "a synthetic Puttalam row was created"

    # The source labels are carried through untouched.
    source = pd.read_csv(RAW_PATH)

    assert rows["year"].equals(source["year"]), "source year was modified"
    assert rows["week"].equals(source["week"]), "source week was modified"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _format_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Render every datetime column as an unambiguous YYYY-MM-DD string."""

    result = df.copy()

    for column in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = result[column].dt.strftime(OUTPUT_DATE_FORMAT)

    return result


def write_calendar_outputs(artifacts: dict) -> None:
    """Write the calendar and its companion tables."""

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    calendar = artifacts["calendar"].copy()

    # period_id is a stable key, so it is written as an integer rather than
    # the float that a shifted column would otherwise produce.
    calendar["previous_period_id"] = calendar["previous_period_id"].astype(
        "Int64"
    )

    _format_dates(calendar).to_csv(CALENDAR_PATH, index=False)

    gaps = artifacts["known_gaps"].copy()

    for column in ["previous_period_id", "next_period_id"]:
        gaps[column] = gaps[column].astype("Int64")

    _format_dates(gaps).to_csv(KNOWN_GAPS_PATH, index=False)

    _format_dates(artifacts["label_anomalies"]).to_csv(
        LABEL_ANOMALIES_PATH, index=False
    )

    _format_dates(artifacts["incomplete_coverage"]).to_csv(
        COVERAGE_ISSUES_PATH, index=False
    )


# ---------------------------------------------------------------------------
# 16. Summary
# ---------------------------------------------------------------------------

def print_calendar_summary(rows: pd.DataFrame, artifacts: dict) -> None:
    """Print what the calendar contains and how it must be used."""

    calendar = artifacts["calendar"]
    gaps = artifacts["known_gaps"]
    anomalies = artifacts["label_anomalies"]
    coverage = artifacts["incomplete_coverage"]

    print("\nReporting calendar summary")
    print("-" * 62)

    print(f"Unique reporting periods: {len(calendar)}")
    print(
        f"First reporting date:     "
        f"{calendar['start_date'].min().date()}"
    )
    print(
        f"Last reporting date:      "
        f"{calendar['end_date'].max().date()}"
    )

    print("\nReporting-period length distribution")
    for days, count in calendar["reporting_days"].value_counts().sort_index().items():
        print(f"  {days} days: {count}")

    print("\nWeekday-convention distribution")
    for label, count in calendar["weekday_convention"].value_counts().items():
        print(f"  {label}: {count}")

    print(f"\nIrregular periods:        {int(calendar['is_irregular_period'].sum())}")
    print(f"Calendar gaps:            {int(calendar['has_calendar_gap_before'].sum())}")
    print(
        f"Calendar overlaps:        "
        f"{int(calendar['has_calendar_overlap_before'].sum())}"
    )
    print(f"Source-label anomalies:   {len(anomalies)}")
    print(f"Incomplete periods:       {len(coverage)}")

    for _, gap in gaps.iterrows():
        print(
            f"\nCalendar gap: {gap['gap_start_date'].date()} to "
            f"{gap['gap_end_date'].date()} ({gap['gap_days']} days)"
        )
        print(f"  Between period {int(gap['previous_period_id'])} and "
              f"period {int(gap['next_period_id'])}.")
        print("  Retained as unobserved dates. Never interpolated.")

    for _, issue in coverage.iterrows():
        print(
            f"\nIncomplete coverage: {issue['source_year']} week "
            f"{issue['source_week']} "
            f"({issue['start_date'].date()} to {issue['end_date'].date()})"
        )
        print(
            f"  Missing: {issue['missing_source_reporting_areas']} "
            f"({issue['source_reporting_area_count']} of "
            f"{EXPECTED_SOURCE_REPORTING_AREAS} areas present)"
        )
        print("  Reported as missing. No synthetic row, no zero cases.")

    print(
        "\nperiod_id is the primary chronological key. Use it for sorting, "
        "lagging,\nwindowing and splitting. Join weather on start_date and "
        "end_date. The\nsource_year and source_week columns are metadata "
        "only."
    )


def main() -> int:
    """Build, verify and write the canonical reporting calendar."""

    rows = load_source_rows()

    artifacts = build_calendar_artifacts(rows)

    run_calendar_assertions(rows, artifacts)

    write_calendar_outputs(artifacts)

    print_calendar_summary(rows, artifacts)

    print(f"\nWrote {CALENDAR_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

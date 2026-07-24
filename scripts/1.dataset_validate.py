import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"

def validate_required_columns(df: pd.DataFrame) -> None:
    """Ensure that all required columns exist."""

    required_columns = {
        "year",
        "week",
        "start.date",
        "end.date",
        "district",
        "cases",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )


def parse_dates(
    df: pd.DataFrame,
    date_format: str = "%m/%d/%Y",
) -> pd.DataFrame:
    """
    Parse start.date and end.date.

    The denguedatahub CSV currently uses month/day/year,
    for example: 12/23/2006.
    """

    result = df.copy()

    result["start_date_parsed"] = pd.to_datetime(
        result["start.date"],
        format=date_format,
        errors="coerce",
    )

    result["end_date_parsed"] = pd.to_datetime(
        result["end.date"],
        format=date_format,
        errors="coerce",
    )

    return result


def find_invalid_start_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where start.date could not be parsed."""

    return df.loc[
        df["start_date_parsed"].isna(),
        [
            "year",
            "week",
            "district",
            "start.date",
        ],
    ].copy()


def find_invalid_end_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where end.date could not be parsed."""

    return df.loc[
        df["end_date_parsed"].isna(),
        [
            "year",
            "week",
            "district",
            "end.date",
        ],
    ].copy()


def find_invalid_date_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return unexpected reporting-period lengths.

    Two known source periods are intentionally irregular:
    - 2009 week 17: 8 days
    - 2009 week 22: 6 days
    """

    valid_dates = (
        df["start_date_parsed"].notna()
        & df["end_date_parsed"].notna()
    )

    day_difference = (
        df["end_date_parsed"] - df["start_date_parsed"]
    ).dt.days

    known_irregular_period = (
        (
            df["start_date_parsed"].eq(pd.Timestamp("2009-04-18"))
            & df["end_date_parsed"].eq(pd.Timestamp("2009-04-25"))
        )
        |
        (
            df["start_date_parsed"].eq(pd.Timestamp("2009-05-24"))
            & df["end_date_parsed"].eq(pd.Timestamp("2009-05-29"))
        )
    )

    invalid_mask = (
        valid_dates
        & day_difference.ne(6)
        & ~known_irregular_period
    )

    result = df.loc[
        invalid_mask,
        [
            "year",
            "week",
            "district",
            "start.date",
            "end.date",
            "start_date_parsed",
            "end_date_parsed",
        ],
    ].copy()

    result["date_difference_days"] = day_difference.loc[invalid_mask]

    return result


def find_missing_districts(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where district is missing or blank."""

    district_text = df["district"].astype("string").str.strip()

    invalid_mask = (
        district_text.isna()
        | district_text.eq("")
    )

    return df.loc[
        invalid_mask,
        [
            "year",
            "week",
            "district",
            "cases",
        ],
    ].copy()


def find_missing_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where cases are missing or blank."""

    case_text = df["cases"].astype("string").str.strip()

    invalid_mask = (
        case_text.isna()
        | case_text.eq("")
    )

    return df.loc[
        invalid_mask,
        [
            "year",
            "week",
            "district",
            "cases",
        ],
    ].copy()


def find_invalid_case_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows where cases are not non-negative whole numbers.

    Invalid examples:
    - text values
    - negative numbers
    - decimal numbers
    - missing values
    """

    numeric_cases = pd.to_numeric(
        df["cases"],
        errors="coerce",
    )

    invalid_mask = (
        numeric_cases.isna()
        | numeric_cases.lt(0)
        | numeric_cases.mod(1).ne(0)
    )

    result = df.loc[
        invalid_mask,
        [
            "year",
            "week",
            "district",
            "cases",
        ],
    ].copy()

    result["numeric_cases"] = numeric_cases.loc[invalid_mask]

    return result


def find_duplicate_district_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return duplicate district-week rows.

    A district should appear only once for each year and week.
    """

    district_clean = (
        df["district"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )

    duplicate_check = df.assign(
        district_clean=district_clean
    )

    duplicate_mask = duplicate_check.duplicated(
        subset=["year", "week", "district_clean"],
        keep=False,
    )

    return duplicate_check.loc[
        duplicate_mask,
        [
            "year",
            "week",
            "district",
            "start.date",
            "end.date",
            "cases",
        ],
    ].sort_values(
        ["year", "week", "district"]
    ).copy()


def find_unexpected_start_weekdays(
    df: pd.DataFrame,
    expected_weekday: str = "Saturday",
) -> pd.DataFrame:
    """
    Validate reporting weekdays using the known source conventions.

    Normal convention: Saturday
    2009-04-26 to 2009-05-24: Sunday
    From 2025-12-29 onward: Monday
    """

    start_dates = df["start_date_parsed"]
    valid_mask = start_dates.notna()

    actual_weekday = start_dates.dt.day_name()

    expected = pd.Series(
        expected_weekday,
        index=df.index,
        dtype="string",
    )

    sunday_period = start_dates.between(
        pd.Timestamp("2009-04-26"),
        pd.Timestamp("2009-05-24"),
    )

    monday_period = start_dates.ge(
        pd.Timestamp("2025-12-29")
    )

    expected.loc[sunday_period] = "Sunday"
    expected.loc[monday_period] = "Monday"

    invalid_mask = (
        valid_mask
        & actual_weekday.ne(expected)
    )

    result = df.loc[
        invalid_mask,
        [
            "year",
            "week",
            "district",
            "start.date",
            "start_date_parsed",
        ],
    ].copy()

    result["actual_weekday"] = actual_weekday.loc[invalid_mask]
    result["expected_weekday"] = expected.loc[invalid_mask]

    return result


def find_non_consecutive_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return unexpected gaps or overlaps between reporting periods.

    Continuity is checked using the previous end date instead of
    expecting every pair of start dates to be seven days apart.
    """

    weekly_dates = (
        df.loc[
            df["start_date_parsed"].notna()
            & df["end_date_parsed"].notna(),
            [
                "year",
                "week",
                "start_date_parsed",
                "end_date_parsed",
            ],
        ]
        .drop_duplicates(
            subset=[
                "start_date_parsed",
                "end_date_parsed",
            ]
        )
        .sort_values("start_date_parsed")
        .reset_index(drop=True)
    )

    weekly_dates["previous_start_date"] = (
        weekly_dates["start_date_parsed"].shift(1)
    )

    weekly_dates["previous_end_date"] = (
        weekly_dates["end_date_parsed"].shift(1)
    )

    weekly_dates["gap_days"] = (
        weekly_dates["start_date_parsed"]
        - weekly_dates["previous_start_date"]
    ).dt.days

    weekly_dates["days_since_previous_end"] = (
        weekly_dates["start_date_parsed"]
        - weekly_dates["previous_end_date"]
    ).dt.days

    # Known two-day source gap during the reporting convention change.
    known_calendar_gap = (
        weekly_dates["previous_end_date"].eq(
            pd.Timestamp("2025-12-26")
        )
        & weekly_dates["start_date_parsed"].eq(
            pd.Timestamp("2025-12-29")
        )
    )

    invalid_mask = (
        weekly_dates["previous_end_date"].notna()
        & weekly_dates["days_since_previous_end"].ne(1)
        & ~known_calendar_gap
    )

    return weekly_dates.loc[
        invalid_mask,
        [
            "year",
            "week",
            "previous_start_date",
            "start_date_parsed",
            "gap_days",
            "previous_end_date",
            "days_since_previous_end",
        ],
    ].copy()


def validate_weekly_dengue_data(
    df: pd.DataFrame,
    date_format: str = "%m/%d/%Y",
    expected_weekday: str = "Saturday",
) -> dict[str, pd.DataFrame]:
    """
    Run all validation checks.

    Returns a dictionary containing a DataFrame for each type of issue.
    An empty DataFrame means that the validation check passed.
    """

    validate_required_columns(df)

    validated_df = parse_dates(
        df,
        date_format=date_format,
    )

    results = {
        "invalid_start_dates": find_invalid_start_dates(validated_df),
        "invalid_end_dates": find_invalid_end_dates(validated_df),
        "invalid_date_ranges": find_invalid_date_ranges(validated_df),
        "missing_districts": find_missing_districts(validated_df),
        "missing_cases": find_missing_cases(validated_df),
        "invalid_case_values": find_invalid_case_values(validated_df),
        "duplicate_district_weeks": find_duplicate_district_weeks(
            validated_df
        ),
        "unexpected_start_weekdays": find_unexpected_start_weekdays(
            validated_df,
            expected_weekday=expected_weekday,
        ),
        "non_consecutive_weeks": find_non_consecutive_weeks(
            validated_df
        ),
    }

    return results


def print_validation_summary(
    validation_results: dict[str, pd.DataFrame],
) -> None:
    """Print the number of problems found by each validation check."""

    print("\nValidation summary")
    print("-" * 50)

    for check_name, issue_rows in validation_results.items():
        issue_count = len(issue_rows)

        if issue_count == 0:
            print(f"PASS: {check_name}")
        else:
            print(f"FAIL: {check_name} — {issue_count} issue(s)")


def count_failed_checks(
    validation_results: dict[str, pd.DataFrame],
) -> int:
    """Return the number of validation checks that found issues."""

    return sum(
        1
        for issue_rows in validation_results.values()
        if len(issue_rows) > 0
    )


def main() -> int:
    """
    Validate the raw weekly dengue data.

    Returns a process exit code so that continuous integration fails
    when any validation check finds issues.
    """

    df = pd.read_csv(RAW_DATA_DIR / "srilanka_weekly_data.csv")

    validation_results = validate_weekly_dengue_data(df)

    print_validation_summary(validation_results)

    failed_checks = count_failed_checks(validation_results)

    if failed_checks:
        print(
            f"\n{failed_checks} validation check(s) failed."
        )
        return 1

    print("\nAll validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
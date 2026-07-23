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
    """Return rows where end date is not six days after start date."""

    valid_dates = (
        df["start_date_parsed"].notna()
        & df["end_date_parsed"].notna()
    )

    day_difference = (
        df["end_date_parsed"] - df["start_date_parsed"]
    ).dt.days

    invalid_mask = valid_dates & day_difference.ne(6)

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
    Return rows whose start dates are not on the expected weekday.

    Example expected_weekday values:
    - "Saturday"
    - "Sunday"
    - "Monday"
    """

    valid_mask = df["start_date_parsed"].notna()

    actual_weekday = df["start_date_parsed"].dt.day_name()

    invalid_mask = (
        valid_mask
        & actual_weekday.ne(expected_weekday)
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
    result["expected_weekday"] = expected_weekday

    return result


def find_non_consecutive_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return reporting weeks that are not seven days apart.

    The check is performed once per unique year-week instead of once
    per district.
    """

    weekly_dates = (
        df.loc[
            df["start_date_parsed"].notna(),
            [
                "year",
                "week",
                "start_date_parsed",
            ],
        ]
        .drop_duplicates()
        .sort_values("start_date_parsed")
        .reset_index(drop=True)
    )

    weekly_dates["previous_start_date"] = (
        weekly_dates["start_date_parsed"].shift(1)
    )

    weekly_dates["gap_days"] = (
        weekly_dates["start_date_parsed"]
        - weekly_dates["previous_start_date"]
    ).dt.days

    invalid_mask = (
        weekly_dates["previous_start_date"].notna()
        & weekly_dates["gap_days"].ne(7)
    )

    return weekly_dates.loc[
        invalid_mask,
        [
            "year",
            "week",
            "previous_start_date",
            "start_date_parsed",
            "gap_days",
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

if __name__ == "__main__":
    
    df = pd.read_csv(RAW_DATA_DIR / "srilanka_weekly_data.csv")
    
    print_validation_summary(
        validate_weekly_dengue_data(df)
    )
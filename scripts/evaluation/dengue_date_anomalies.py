"""
Analyse reporting-period anomalies in the raw weekly dengue data.

This script is read-only with respect to the source data. It never
corrects, drops, or rewrites records. It classifies anomalies and
writes a detailed CSV plus a short Markdown report.

Outputs:
    results/data_validation/dengue_date_anomalies.csv
    results/data_validation/dengue_date_report.md
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "srilanka_weekly_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "data_validation"

ANOMALIES_PATH = RESULTS_DIR / "dengue_date_anomalies.csv"
REPORT_PATH = RESULTS_DIR / "dengue_date_report.md"

DATE_FORMAT = "%m/%d/%Y"

EXPECTED_REPORTING_DAYS = 7
EXPECTED_START_WEEKDAY = "Saturday"


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw dengue file and parse both date columns."""

    df = pd.read_csv(path)

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
            f"Raw dengue file is missing columns: {sorted(missing_columns)}"
        )

    df["start_date"] = pd.to_datetime(
        df["start.date"],
        format=DATE_FORMAT,
        errors="coerce",
    )

    df["end_date"] = pd.to_datetime(
        df["end.date"],
        format=DATE_FORMAT,
        errors="coerce",
    )

    unparsed = (
        df["start_date"].isna().sum()
        + df["end_date"].isna().sum()
    )

    if unparsed:
        raise ValueError(
            f"{unparsed} date value(s) could not be parsed with "
            f"format {DATE_FORMAT}."
        )

    # Inclusive reporting length: a Saturday-to-Friday week is 7 days.
    df["reporting_days"] = (
        df["end_date"] - df["start_date"]
    ).dt.days + 1

    return df


def build_week_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the district-level rows to one row per reporting week.

    Date anomalies are properties of a reporting week, not of an
    individual district, so the analysis is done at week level.
    """

    weeks = (
        df[
            [
                "year",
                "week",
                "start.date",
                "end.date",
                "start_date",
                "end_date",
                "reporting_days",
            ]
        ]
        .drop_duplicates()
        .sort_values("start_date")
        .reset_index(drop=True)
    )

    district_counts = (
        df.groupby(["year", "week"], as_index=False)
        .agg(
            affected_districts=("district", "nunique"),
            total_cases=("cases", "sum"),
        )
    )

    weeks = weeks.merge(
        district_counts,
        on=["year", "week"],
        how="left",
        validate="one_to_one",
    )

    weeks["start_weekday"] = weeks["start_date"].dt.day_name()
    weeks["end_weekday"] = weeks["end_date"].dt.day_name()

    # Gap between the start of this week and the start of the previous
    # week. The validator expects exactly 7.
    weeks["previous_start_date"] = weeks["start_date"].shift(1)

    weeks["start_gap_days"] = (
        weeks["start_date"] - weeks["previous_start_date"]
    ).dt.days

    # Gap between the end of the previous week and the start of this
    # week. A continuous calendar gives exactly 1.
    weeks["previous_end_date"] = weeks["end_date"].shift(1)

    weeks["days_since_previous_end"] = (
        weeks["start_date"] - weeks["previous_end_date"]
    ).dt.days

    # A label is "year-shifted" when the reporting year does not match
    # the calendar year the week starts in. This is normal at the
    # December/January boundary.
    weeks["label_year_matches_start"] = (
        weeks["year"] == weeks["start_date"].dt.year
    )

    return weeks


def find_irregular_length_weeks(weeks: pd.DataFrame) -> pd.DataFrame:
    """Return weeks whose inclusive length is not seven days."""

    irregular = weeks.loc[
        weeks["reporting_days"].ne(EXPECTED_REPORTING_DAYS)
    ].copy()

    irregular["anomaly_type"] = "irregular_reporting_length"

    irregular["description"] = irregular.apply(
        lambda row: (
            f"Reporting period covers {row['reporting_days']} days "
            f"instead of {EXPECTED_REPORTING_DAYS}."
        ),
        axis=1,
    )

    return irregular


def find_unexpected_start_weekdays(weeks: pd.DataFrame) -> pd.DataFrame:
    """Return weeks that do not start on the dominant weekday."""

    unexpected = weeks.loc[
        weeks["start_weekday"].ne(EXPECTED_START_WEEKDAY)
    ].copy()

    unexpected["anomaly_type"] = "unexpected_start_weekday"

    unexpected["description"] = unexpected.apply(
        lambda row: (
            f"Week starts on {row['start_weekday']} instead of "
            f"{EXPECTED_START_WEEKDAY}."
        ),
        axis=1,
    )

    return unexpected


def find_calendar_discontinuities(weeks: pd.DataFrame) -> pd.DataFrame:
    """
    Return weeks that do not begin the day after the previous week ends.

    This detects genuine calendar gaps and overlaps, which the
    start-to-start check cannot distinguish from length changes.
    """

    discontinuous = weeks.loc[
        weeks["days_since_previous_end"].notna()
        & weeks["days_since_previous_end"].ne(1)
    ].copy()

    discontinuous["anomaly_type"] = "calendar_discontinuity"

    def describe(row: pd.Series) -> str:
        gap = row["days_since_previous_end"]
        previous_end = row["previous_end_date"].date()

        if gap > 1:
            uncovered = int(gap) - 1
            return (
                f"{uncovered} calendar day(s) are not covered by any "
                f"reporting week after {previous_end}."
            )

        overlap = 1 - int(gap)
        return (
            f"Overlaps the previous period (ending {previous_end}) "
            f"by {overlap} day(s)."
        )

    discontinuous["description"] = discontinuous.apply(describe, axis=1)

    return discontinuous


def find_non_consecutive_weeks(weeks: pd.DataFrame) -> pd.DataFrame:
    """
    Return weeks flagged by the start-to-start seven-day rule.

    This reproduces the check in scripts/1.dataset_validate.py so the
    report can explain each failure it raises.
    """

    non_consecutive = weeks.loc[
        weeks["start_gap_days"].notna()
        & weeks["start_gap_days"].ne(7)
    ].copy()

    non_consecutive["anomaly_type"] = "non_consecutive_week"

    non_consecutive["description"] = non_consecutive.apply(
        lambda row: (
            f"Start date is {int(row['start_gap_days'])} days after the "
            "previous start date instead of 7."
        ),
        axis=1,
    )

    return non_consecutive


def find_year_label_anomalies(weeks: pd.DataFrame) -> pd.DataFrame:
    """
    Return weeks whose year/week label is out of sequence.

    A week that starts in late December but is labelled as week 1 of the
    following year is the normal convention in this source. A week
    labelled with a high week number but placed before week 1 of the
    same reporting year is not.
    """

    candidates = weeks.loc[~weeks["label_year_matches_start"]].copy()

    if candidates.empty:
        return candidates.assign(anomaly_type=[], description=[])

    first_week_start = (
        weeks.loc[weeks["week"].eq(1)]
        .set_index("year")["start_date"]
        .to_dict()
    )

    def is_out_of_sequence(row: pd.Series) -> bool:
        reference = first_week_start.get(row["year"])

        if reference is None:
            return False

        # Normal: week 1 itself starting in the previous December.
        if row["week"] == 1:
            return False

        # Abnormal: a later week number placed before its own week 1.
        return row["start_date"] < reference

    flagged = candidates.loc[
        candidates.apply(is_out_of_sequence, axis=1)
    ].copy()

    flagged["anomaly_type"] = "year_label_out_of_sequence"

    flagged["description"] = flagged.apply(
        lambda row: (
            f"Labelled year {row['year']} week {row['week']} but starts "
            f"{row['start_date'].date()}, before week 1 of "
            f"{row['year']}. The period belongs to the "
            f"{row['start_date'].year} reporting year."
        ),
        axis=1,
    )

    return flagged


def find_incomplete_district_coverage(
    weeks: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return weeks that do not contain every district in the file."""

    expected_districts = df["district"].nunique()

    incomplete = weeks.loc[
        weeks["affected_districts"].ne(expected_districts)
    ].copy()

    if incomplete.empty:
        return incomplete.assign(anomaly_type=[], description=[])

    all_districts = set(df["district"].unique())

    def describe(row: pd.Series) -> str:
        present = set(
            df.loc[
                df["year"].eq(row["year"]) & df["week"].eq(row["week"]),
                "district",
            ]
        )

        missing = sorted(all_districts - present)

        return (
            f"Only {row['affected_districts']} of {expected_districts} "
            f"districts reported. Missing: {', '.join(missing)}."
        )

    incomplete["anomaly_type"] = "incomplete_district_coverage"
    incomplete["description"] = incomplete.apply(describe, axis=1)

    return incomplete


def build_anomaly_table(
    weeks: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine every anomaly category into one long-format table."""

    parts = [
        find_irregular_length_weeks(weeks),
        find_unexpected_start_weekdays(weeks),
        find_calendar_discontinuities(weeks),
        find_non_consecutive_weeks(weeks),
        find_year_label_anomalies(weeks),
        find_incomplete_district_coverage(weeks, df),
    ]

    parts = [part for part in parts if not part.empty]

    anomalies = pd.concat(parts, ignore_index=True)

    output_columns = [
        "anomaly_type",
        "year",
        "week",
        "start.date",
        "end.date",
        "start_date",
        "end_date",
        "reporting_days",
        "start_weekday",
        "end_weekday",
        "affected_districts",
        "total_cases",
        "previous_start_date",
        "start_gap_days",
        "previous_end_date",
        "days_since_previous_end",
        "description",
    ]

    anomalies = anomalies[output_columns].sort_values(
        ["start_date", "anomaly_type"]
    )

    return anomalies.reset_index(drop=True)


def summarise_reporting_lengths(df: pd.DataFrame) -> dict[str, int]:
    """Count rows by reporting-period length bucket."""

    return {
        "rows_fewer_than_7_days": int(
            (df["reporting_days"] < EXPECTED_REPORTING_DAYS).sum()
        ),
        "rows_exactly_7_days": int(
            (df["reporting_days"] == EXPECTED_REPORTING_DAYS).sum()
        ),
        "rows_more_than_7_days": int(
            (df["reporting_days"] > EXPECTED_REPORTING_DAYS).sum()
        ),
    }


def build_report(
    df: pd.DataFrame,
    weeks: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> str:
    """Render the Markdown report."""

    buckets = summarise_reporting_lengths(df)

    irregular = (
        weeks.loc[weeks["reporting_days"].ne(EXPECTED_REPORTING_DAYS)]
        .sort_values("start_date")
    )

    weekday_counts = weeks["start_weekday"].value_counts()

    lines: list[str] = []

    lines.append("# Dengue reporting-date anomaly report")
    lines.append("")
    lines.append(f"- Source file: `{RAW_PATH.relative_to(PROJECT_ROOT).as_posix()}`")
    lines.append(f"- Rows: {len(df):,}")
    lines.append(f"- Unique reporting weeks: {len(weeks):,}")
    lines.append(
        f"- Coverage: {weeks['start_date'].min().date()} to "
        f"{weeks['end_date'].max().date()}"
    )
    lines.append(f"- Districts: {df['district'].nunique()}")
    lines.append(f"- Date parse failures: 0 (format `{DATE_FORMAT}`)")
    lines.append("")

    lines.append("## 1. Reporting-period length")
    lines.append("")
    lines.append(
        "`reporting_days = (end_date - start_date).days + 1` "
        "(inclusive)."
    )
    lines.append("")
    lines.append("| Reporting length | Rows |")
    lines.append("| --- | ---: |")
    lines.append(f"| Fewer than 7 days | {buckets['rows_fewer_than_7_days']:,} |")
    lines.append(f"| Exactly 7 days | {buckets['rows_exactly_7_days']:,} |")
    lines.append(f"| More than 7 days | {buckets['rows_more_than_7_days']:,} |")
    lines.append("")
    lines.append(
        f"All {buckets['rows_fewer_than_7_days'] + buckets['rows_more_than_7_days']} "
        f"irregular rows belong to just {len(irregular)} reporting weeks; "
        "every district in those weeks is affected identically, which "
        "confirms the anomaly is in the reporting calendar rather than "
        "in any district's data."
    )
    lines.append("")

    lines.append("## 2. Irregular reporting periods")
    lines.append("")
    lines.append(
        "| Year | Week | Start | End | Inclusive days | Districts affected |"
    )
    lines.append("| ---: | ---: | --- | --- | ---: | ---: |")

    for _, row in irregular.iterrows():
        lines.append(
            f"| {row['year']} | {row['week']} | {row['start.date']} | "
            f"{row['end.date']} | {int(row['reporting_days'])} | "
            f"{int(row['affected_districts'])} |"
        )

    lines.append("")

    lines.append("## 3. Reporting start weekdays")
    lines.append("")
    lines.append("| Start weekday | Reporting weeks |")
    lines.append("| --- | ---: |")

    for weekday, count in weekday_counts.items():
        lines.append(f"| {weekday} | {count:,} |")

    lines.append("")

    saturday_weeks = weeks.loc[
        weeks["start_weekday"].eq(EXPECTED_START_WEEKDAY)
    ]

    lines.append(
        f"Saturday is the dominant convention "
        f"({len(saturday_weeks):,} of {len(weeks):,} weeks). "
        "The non-Saturday weeks fall into two distinct episodes, "
        "described below."
    )
    lines.append("")

    lines.append("## 4. Convention changes")
    lines.append("")
    lines.append("### 4.1 April-May 2009: temporary shift to Sunday")
    lines.append("")
    lines.append(
        "Week 17 of 2009 (`4/18/2009`-`4/25/2009`) was extended to 8 "
        "inclusive days. That one-day extension pushed the following "
        "week onto a Sunday start, and weeks 18-22 of 2009 all begin on "
        "Sunday. Week 22 (`5/24/2009`-`5/29/2009`) was then shortened to "
        "6 inclusive days, which restored the Saturday convention from "
        "week 23 onward."
    )
    lines.append("")
    lines.append(
        "The two length anomalies are therefore a matched pair: a "
        "+1-day extension and a -1-day contraction that together "
        "return the calendar to its original phase. The calendar "
        "remains fully continuous throughout - no day is uncovered and "
        "no day is double-counted."
    )
    lines.append("")
    lines.append("### 4.2 From 2026: permanent shift to Monday")
    lines.append("")
    lines.append(
        "Every reporting week labelled 2026 week 1 onward begins on a "
        "Monday and ends on a Sunday. The last Saturday-start week is "
        "`12/20/2025`-`12/26/2025`. This is a deliberate change of "
        "reporting convention at the source, not a data error, and it "
        "is permanent rather than transient."
    )
    lines.append("")

    lines.append("## 5. The three non-consecutive-week failures")
    lines.append("")
    lines.append(
        "`scripts/1.dataset_validate.py` compares each week's start "
        "date with the previous week's start date and expects a gap of "
        "exactly 7 days. Three weeks fail. They have three different "
        "causes:"
    )
    lines.append("")
    lines.append(
        "| Year | Week | Previous start | Start | Gap | Cause |"
    )
    lines.append("| ---: | ---: | --- | --- | ---: | --- |")
    lines.append(
        "| 2009 | 18 | 2009-04-18 | 2009-04-26 | 8 | "
        "Consequence of the 8-day week 17. Not a missing week. |"
    )
    lines.append(
        "| 2009 | 23 | 2009-05-24 | 2009-05-30 | 6 | "
        "Consequence of the 6-day week 22. Not a duplicate. |"
    )
    lines.append(
        "| 2026 | 1 | 2025-12-20 | 2025-12-29 | 9 | "
        "Genuine 2-day calendar gap plus a mislabelled week. |"
    )
    lines.append("")
    lines.append(
        "The first two are artefacts of the validator's assumption. "
        "Because it measures start-to-start distance, any change in "
        "reporting length is reported a second time as a spacing "
        "error. The underlying calendar is continuous."
    )
    lines.append("")
    lines.append("The third is a real problem, with two separate defects:")
    lines.append("")
    lines.append(
        "1. **Mislabelled year.** The week covering `12/20/2025`-"
        "`12/26/2025` is labelled `year=2026, week=53`. Sorted by week "
        "number it appears at the end of 2026, but chronologically it "
        "precedes 2026 week 1. It is the final week of the 2025 "
        "reporting year and should be read as such."
    )
    lines.append(
        "2. **Two uncovered days.** No reporting week covers "
        "`2025-12-27` or `2025-12-28`. The Saturday-based calendar ends "
        "on 2025-12-26 and the Monday-based calendar begins on "
        "2025-12-29. These two days are the seam between the two "
        "conventions and are genuinely absent from the source."
    )
    lines.append("")
    lines.append(
        "This is the only true calendar gap in the entire 2006-2026 "
        "series."
    )
    lines.append("")

    lines.append("## 6. Checks that are not anomalies")
    lines.append("")
    lines.append(
        "- **Week 1 starting in December.** In 18 reporting years, week "
        "1 begins in the last days of the previous December. This is "
        "the source's normal convention and is not flagged."
    )
    lines.append(
        "- **2023 has 51 weeks.** 2023 week 52 does not exist because "
        "2024 week 1 starts on `12/23/2023`. The calendar is "
        "continuous across the boundary; only the label rolls over "
        "early."
    )
    lines.append(
        "- **2009, 2016 and 2021 have 53 weeks.** Expected for a "
        "53-week reporting year."
    )
    lines.append("")

    incomplete = anomalies.loc[
        anomalies["anomaly_type"].eq("incomplete_district_coverage")
    ]

    if not incomplete.empty:
        lines.append("## 7. Incomplete district coverage")
        lines.append("")
        lines.append(
            "One week does not contain every district. This is a "
            "completeness issue rather than a date issue, but it "
            "affects the same node-week grid:"
        )
        lines.append("")
        lines.append("| Year | Week | Start | Districts | Detail |")
        lines.append("| ---: | ---: | --- | ---: | --- |")

        for _, row in incomplete.iterrows():
            lines.append(
                f"| {row['year']} | {row['week']} | {row['start.date']} | "
                f"{int(row['affected_districts'])} | {row['description']} |"
            )

        lines.append("")

    lines.append("## 8. Recommended handling")
    lines.append("")
    lines.append(
        "No source record should be modified or deleted. Every "
        "recommendation below adds derived columns and keeps the "
        "original `year`, `week`, `start.date` and `end.date` values "
        "intact."
    )
    lines.append("")
    lines.append(
        "| Anomaly | Recommended handling |"
    )
    lines.append("| --- | --- |")
    lines.append(
        "| 8-day week (2009 w17) and 6-day week (2009 w22) | "
        "Keep as-is. Add `reporting_days` as a derived column and, for "
        "any rate or climate-lag modelling, use a per-day exposure "
        "offset rather than raw weekly counts. The pair is "
        "phase-neutral, so cumulative sums are unaffected. |"
    )
    lines.append(
        "| Sunday-start weeks (2009 w18-w22) | "
        "Keep as-is. Treat the reporting week as defined by "
        "`start_date`/`end_date`, never by weekday. Any climate "
        "aggregation must join on the actual date interval. |"
    )
    lines.append(
        "| Monday-start weeks (2026 w1 onward) | "
        "Keep as-is and record the convention change explicitly. "
        "Relax the validator's fixed-weekday rule to a "
        "convention-aware rule: Saturday before 2025-12-27, Monday "
        "after 2025-12-28. |"
    )
    lines.append(
        "| Mislabelled 2026 week 53 | "
        "Do not edit the source. Add a derived "
        "`reporting_year`/`reporting_week` pair, ordered by "
        "`start_date`, and use that for all joins and sorting. Never "
        "sort the series on the raw `year`/`week` columns. |"
    )
    lines.append(
        "| Uncovered days 2025-12-27 and 2025-12-28 | "
        "Leave absent. Do not interpolate or redistribute cases. "
        "Record the gap in a known-gaps table so downstream models "
        "treat the interval as unobserved rather than as zero. |"
    )
    lines.append(
        "| Missing Puttalam row, 2026 week 7 | "
        "Leave absent. Represent it as an explicit missing value on "
        "the node-week grid, not as `cases = 0`. |"
    )
    lines.append(
        "| Start-to-start spacing check | "
        "Replace with a continuity check on "
        "`start_date == previous_end_date + 1 day`. That test detects "
        "real gaps and overlaps without re-flagging legitimate "
        "changes in reporting length. |"
    )
    lines.append("")
    lines.append(
        "The single highest-value change is the derived chronological "
        "ordering. Until 2026 week 53 is placed correctly, any "
        "time-ordered join between dengue cases and weather data will "
        "silently misalign one week of the series."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Run the analysis and write both output files."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(RAW_PATH)
    weeks = build_week_calendar(df)
    anomalies = build_anomaly_table(weeks, df)

    anomalies.to_csv(ANOMALIES_PATH, index=False)

    report = build_report(df, weeks, anomalies)
    REPORT_PATH.write_text(report, encoding="utf-8")

    buckets = summarise_reporting_lengths(df)

    print("Reporting-period length (rows):")
    print(f"  fewer than 7 days: {buckets['rows_fewer_than_7_days']}")
    print(f"  exactly 7 days:    {buckets['rows_exactly_7_days']}")
    print(f"  more than 7 days:  {buckets['rows_more_than_7_days']}")

    print(f"\nAnomaly records written: {len(anomalies)}")
    print(anomalies["anomaly_type"].value_counts().to_string())

    print(f"\nWrote: {ANOMALIES_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Wrote: {REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()

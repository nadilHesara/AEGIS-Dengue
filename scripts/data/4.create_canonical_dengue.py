"""
Build the canonical district-level dengue dataset.

The source dengue CSV is never modified. This script joins every source row
to the reporting calendar, maps the 26 source reporting areas onto the 25
canonical districts, and aggregates the areas that share a district.

The only aggregation is Kalmune into Ampara. The source reports Kalmune
separately, but the graph and the model use the 25 canonical districts, so
the two source rows are summed into one Ampara observation.

Missing observations stay missing. The absent Puttalam record for 2026 week 7
is carried as an absent row, never as a zero. A zero would assert that
Puttalam observed no dengue cases that week, which the source does not say.

All sorting and merging use period_id from the reporting calendar. The raw
year and week labels cannot order the data.

Outputs:
    data/interim/dengue_weekly_canonical.parquet
    results/data_validation/canonical_district_coverage.csv
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]

RAW_PATH = PROJECT_DIR / "data" / "raw" / "srilanka_weekly_data.csv"
CALENDAR_PATH = PROJECT_DIR / "data" / "interim" / "reporting_calendar.csv"
NODES_PATH = PROJECT_DIR / "data" / "processed" / "nodes.csv"

INTERIM_DIR = PROJECT_DIR / "data" / "interim"
RESULTS_DIR = PROJECT_DIR / "results" / "data_validation"

CANONICAL_PATH = INTERIM_DIR / "dengue_weekly_canonical.parquet"
COVERAGE_PATH = RESULTS_DIR / "canonical_district_coverage.csv"

DATE_FORMAT = "%m/%d/%Y"

EXPECTED_CANONICAL_DISTRICTS = 25
EXPECTED_SOURCE_REPORTING_AREAS = 26


def _load_nodes_module():
    """
    Import the node-registry module despite its non-identifier file name.

    DENGUE_TO_CANONICAL is the project's explicit source-to-district
    mapping. Importing it keeps a single definition rather than a second
    copy that could drift.
    """

    path = PROJECT_DIR / "scripts" / "data" / "3.create_nodes.py"

    spec = importlib.util.spec_from_file_location("create_nodes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


try:
    _nodes_module = _load_nodes_module()
    DENGUE_TO_CANONICAL = _nodes_module.DENGUE_TO_CANONICAL
    CANONICAL_ORDER = _nodes_module.CANONICAL_ORDER
except ImportError:
    # 3.create_nodes.py imports geopandas, which is only needed to build the
    # node registry from polygons. The mapping itself is plain data, so it is
    # restated here to keep this script usable without the geospatial stack.
    # Any change must be made in 3.create_nodes.py first.
    DENGUE_TO_CANONICAL = {
        "Ampara": "Ampara",
        "Anuradhapura": "Anuradhapura",
        "Badulla": "Badulla",
        "Batticaloa": "Batticaloa",
        "Colombo": "Colombo",
        "Galle": "Galle",
        "Gampaha": "Gampaha",
        "Hambanthota": "Hambantota",
        "Jaffna": "Jaffna",
        "Kalmune": "Ampara",
        "Kalutara": "Kalutara",
        "Kandy": "Kandy",
        "Kegalle": "Kegalle",
        "Kilinochchi": "Kilinochchi",
        "Kurunegala": "Kurunegala",
        "Mannar": "Mannar",
        "Matale": "Matale",
        "Matara": "Matara",
        "Monaragala": "Moneragala",
        "Mullaitivu": "Mullaitivu",
        "NuwaraEliya": "Nuwara Eliya",
        "Polonnaruwa": "Polonnaruwa",
        "Puttalam": "Puttalam",
        "Ratnapura": "Ratnapura",
        "Trincomalee": "Trincomalee",
        "Vavuniya": "Vavuniya",
    }

    CANONICAL_ORDER = sorted(set(DENGUE_TO_CANONICAL.values()))


# The calendar columns carried onto every observation so that downstream
# code can filter on period shape without a second join.
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
    "province",
    "cases",
    "source_reporting_areas_used",
    "source_row_count",
    "case_observed",
]


def _ensure_source_calendar_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the explicit source calendar join keys exist."""

    result = df.copy()

    if "source_year" not in result.columns:
        result["source_year"] = result["year"]

    if "source_week" not in result.columns:
        result["source_week"] = result["week"]

    return result


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_dengue(
    path: Path = RAW_PATH,
    date_format: str = DATE_FORMAT,
) -> pd.DataFrame:
    """Load the source rows and parse the reporting dates."""

    df = pd.read_csv(path)

    result = _ensure_source_calendar_keys(df)

    result["start_date"] = pd.to_datetime(
        result["start.date"], format=date_format, errors="coerce"
    )

    result["end_date"] = pd.to_datetime(
        result["end.date"], format=date_format, errors="coerce"
    )

    result["district"] = result["district"].astype(str).str.strip()

    return result


def load_reporting_calendar(path: Path = CALENDAR_PATH) -> pd.DataFrame:
    """Load the canonical reporting calendar."""

    calendar = pd.read_csv(path)

    for column in ["start_date", "end_date"]:
        calendar[column] = pd.to_datetime(calendar[column])

    return calendar[CALENDAR_COLUMNS].copy()


def load_nodes(path: Path = NODES_PATH) -> pd.DataFrame:
    """Load the permanent district node registry."""

    return pd.read_csv(path)[
        ["node_id", "canonical_name", "province"]
    ].copy()


# ---------------------------------------------------------------------------
# Calendar join
# ---------------------------------------------------------------------------

def join_reporting_calendar(
    rows: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach period_id to every source row.

    The join uses all four keys, so a source row can only match the calendar
    entry describing exactly its label and its interval.
    """

    rows = _ensure_source_calendar_keys(rows)

    merged = rows.merge(
        calendar,
        on=["source_year", "source_week", "start_date", "end_date"],
        how="left",
        validate="many_to_one",
    )

    unmapped = merged["period_id"].isna()

    if unmapped.any():
        examples = merged.loc[
            unmapped, ["year", "week", "start.date", "end.date"]
        ].drop_duplicates().head(10)

        raise ValueError(
            "These source rows match no reporting period:\n"
            f"{examples.to_string(index=False)}"
        )

    merged["period_id"] = merged["period_id"].astype(int)

    return merged


# ---------------------------------------------------------------------------
# District mapping
# ---------------------------------------------------------------------------

def check_district_mapping(rows: pd.DataFrame) -> dict:
    """
    Report every discrepancy between the source names and the mapping.

    Four conditions are checked. Unmapped names are fatal because a silently
    dropped district would remove real cases. The rest are reported.
    """

    source_names = set(rows["district"].unique())
    mapped_names = set(DENGUE_TO_CANONICAL)

    unmapped = sorted(source_names - mapped_names)
    obsolete = sorted(mapped_names - source_names)

    # A canonical district fed by more than one source name. Kalmune into
    # Ampara is the one expected case; anything else needs a decision.
    canonical_sources = {}

    for source_name, canonical in DENGUE_TO_CANONICAL.items():
        canonical_sources.setdefault(canonical, []).append(source_name)

    duplicates = {
        canonical: sorted(names)
        for canonical, names in canonical_sources.items()
        if len(names) > 1
    }

    # Names differing only by case, spacing or punctuation are almost always
    # the same district spelled two ways.
    def _normalise(name: str) -> str:
        return "".join(name.lower().split()).replace("-", "")

    normalised = {}

    for name in sorted(source_names):
        normalised.setdefault(_normalise(name), []).append(name)

    inconsistent = {
        key: names for key, names in normalised.items() if len(names) > 1
    }

    return {
        "unmapped": unmapped,
        "obsolete": obsolete,
        "duplicates": duplicates,
        "inconsistent_spellings": inconsistent,
    }


def print_mapping_report(report: dict) -> None:
    """Print the district-mapping findings."""

    print("\nDistrict mapping report")
    print("-" * 62)

    if report["unmapped"]:
        print(f"Unmapped source names:      {report['unmapped']}")
    else:
        print("Unmapped source names:      none")

    if report["obsolete"]:
        print(f"Obsolete mappings:          {report['obsolete']}")
    else:
        print("Obsolete mappings:          none")

    if report["duplicates"]:
        for canonical, names in sorted(report["duplicates"].items()):
            print(
                f"Aggregated into {canonical}:  {names}"
            )
    else:
        print("Aggregated districts:       none")

    if report["inconsistent_spellings"]:
        for names in report["inconsistent_spellings"].values():
            print(f"Inconsistent spellings:     {names}")
    else:
        print("Inconsistent spellings:     none")


def add_canonical_names(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Map every source reporting area to its canonical district.

    The mapping is explicit. A name with no entry raises rather than being
    guessed at, because a wrong district silently corrupts the graph.
    """

    report = check_district_mapping(rows)

    if report["unmapped"]:
        raise ValueError(
            "These source reporting areas have no explicit mapping: "
            f"{report['unmapped']}"
        )

    result = rows.copy()

    result["canonical_name"] = result["district"].map(DENGUE_TO_CANONICAL)

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_to_canonical(
    rows: pd.DataFrame,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce one observed row per period_id and canonical district.

    Source areas sharing a canonical district are summed. Ampara's count is
    the sum of the Ampara and Kalmune source rows whenever both appear in
    the same period; when only one appears, the sum is that one row and
    source_row_count records it.

    Only observed rows are produced. A district absent from a period yields
    no row at all rather than a zero.
    """

    aggregated = (
        rows.groupby(["period_id", "canonical_name"], as_index=False)
        .agg(
            cases=("cases", "sum"),
            source_row_count=("cases", "size"),
            source_reporting_areas_used=(
                "district",
                lambda names: "|".join(sorted(names)),
            ),
        )
    )

    # Every row here came from at least one real source record.
    aggregated["case_observed"] = 1

    calendar_fields = rows[CALENDAR_COLUMNS].drop_duplicates(
        subset=["period_id"]
    )

    result = aggregated.merge(
        calendar_fields, on="period_id", how="left", validate="many_to_one"
    )

    result = result.merge(
        nodes, on="canonical_name", how="left", validate="many_to_one"
    )

    unknown = result["node_id"].isna()

    if unknown.any():
        names = sorted(result.loc[unknown, "canonical_name"].unique())

        raise ValueError(
            f"These canonical districts have no node_id: {names}"
        )

    result["node_id"] = result["node_id"].astype(int)

    # period_id is the chronological key, so the output is ordered by it.
    return result[OUTPUT_COLUMNS].sort_values(
        ["period_id", "node_id"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Canonical coverage
# ---------------------------------------------------------------------------

def build_canonical_coverage(
    canonical: pd.DataFrame,
    calendar: pd.DataFrame,
    expected_districts: int = EXPECTED_CANONICAL_DISTRICTS,
) -> pd.DataFrame:
    """
    Report the canonical district coverage of every reporting period.

    A period is complete when all 25 canonical districts are observed. The
    missing districts are named so that an absence is visible rather than
    inferred from a count.
    """

    all_districts = set(CANONICAL_ORDER)

    observed = (
        canonical.groupby("period_id")["canonical_name"]
        .agg(
            observed_canonical_districts="size",
            missing_canonical_districts=lambda names: ", ".join(
                sorted(all_districts - set(names))
            ),
        )
        .reset_index()
    )

    coverage = calendar[
        ["period_id", "source_year", "source_week"]
    ].merge(observed, on="period_id", how="left", indicator=True)

    # A period with no observations at all did not appear in the grouping,
    # so it is missing every canonical district. A complete period appears
    # with an empty missing list. The two cases are distinguished by the
    # merge indicator rather than by a null, which both would produce.
    unobserved = coverage["_merge"].eq("left_only")

    coverage["observed_canonical_districts"] = (
        coverage["observed_canonical_districts"].fillna(0).astype(int)
    )

    coverage["missing_canonical_districts"] = coverage[
        "missing_canonical_districts"
    ].fillna("")

    coverage.loc[unobserved, "missing_canonical_districts"] = ", ".join(
        sorted(all_districts)
    )

    coverage = coverage.drop(columns=["_merge"])

    coverage["expected_canonical_districts"] = expected_districts

    coverage["is_complete_canonical_coverage"] = coverage[
        "observed_canonical_districts"
    ].eq(expected_districts)

    return coverage[
        [
            "period_id",
            "source_year",
            "source_week",
            "expected_canonical_districts",
            "observed_canonical_districts",
            "missing_canonical_districts",
            "is_complete_canonical_coverage",
        ]
    ].sort_values("period_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def run_canonical_assertions(
    raw: pd.DataFrame,
    canonical: pd.DataFrame,
    coverage: pd.DataFrame,
    nodes: pd.DataFrame,
) -> None:
    """
    Verify the canonical dataset's guarantees before it is written.

    These are hard invariants. A failure means the dataset would mislead
    downstream modelling, so the script stops rather than saving it.
    """

    assert canonical["period_id"].notna().all(), (
        "a canonical row has no period_id"
    )

    assert canonical["node_id"].notna().all(), (
        "a canonical row has no node_id"
    )

    assert canonical["node_id"].isin(nodes["node_id"]).all(), (
        "a canonical row carries a node_id outside the node registry"
    )

    assert not canonical.duplicated(
        subset=["period_id", "canonical_name"]
    ).any(), "a period_id and canonical district pair is duplicated"

    # Complete periods carry every canonical district, never more.
    complete = coverage.loc[coverage["is_complete_canonical_coverage"]]

    assert complete["observed_canonical_districts"].eq(
        EXPECTED_CANONICAL_DISTRICTS
    ).all(), "a complete period does not hold 25 canonical districts"

    assert coverage["observed_canonical_districts"].le(
        EXPECTED_CANONICAL_DISTRICTS
    ).all(), "a period holds more than 25 canonical districts"

    # Kalmune is a source name only. It must not survive as a model node.
    assert "Kalmune" not in set(canonical["canonical_name"]), (
        "Kalmune remains a separate canonical district"
    )

    assert "Kalmune" not in set(nodes["canonical_name"]), (
        "Kalmune remains a separate model node"
    )

    # Ampara carries the Kalmune cases wherever both source rows exist.
    source_pairs = (
        raw.loc[raw["district"].isin(["Ampara", "Kalmune"])]
        .groupby("period_id")["district"]
        .nunique()
    )

    both_present = set(source_pairs.loc[source_pairs.eq(2)].index)

    ampara = canonical.loc[canonical["canonical_name"].eq("Ampara")]

    combined = ampara.loc[ampara["period_id"].isin(both_present)]

    assert combined["source_row_count"].eq(2).all(), (
        "an Ampara observation does not combine both source rows"
    )

    assert combined["source_reporting_areas_used"].eq(
        "Ampara|Kalmune"
    ).all(), "an Ampara observation does not name both source areas"

    expected_totals = (
        raw.loc[raw["district"].isin(["Ampara", "Kalmune"])]
        .groupby("period_id")["cases"]
        .sum()
    )

    actual_totals = ampara.set_index("period_id")["cases"]

    assert actual_totals.equals(
        expected_totals.reindex(actual_totals.index)
    ), "Ampara case totals do not match the sum of its source rows"

    # Puttalam stays absent for 2026 week 7. No row, and no zero.
    week_7 = coverage.loc[
        coverage["source_year"].eq(2026) & coverage["source_week"].eq(7)
    ]

    assert len(week_7) == 1, "2026 week 7 is missing from the coverage report"

    issue = week_7.iloc[0]

    assert issue["observed_canonical_districts"] == 24, (
        "2026 week 7 does not hold 24 observed canonical districts"
    )

    assert "Puttalam" in issue["missing_canonical_districts"], (
        "Puttalam is not reported missing from 2026 week 7"
    )

    assert not issue["is_complete_canonical_coverage"], (
        "2026 week 7 is marked as complete"
    )

    assert canonical.loc[
        canonical["period_id"].eq(issue["period_id"])
        & canonical["canonical_name"].eq("Puttalam")
    ].empty, "a Puttalam row was created for 2026 week 7"

    # Every observed row is a real observation, never a filled zero.
    assert canonical["case_observed"].eq(1).all(), (
        "a canonical row is not marked as observed"
    )

    assert canonical["cases"].notna().all(), (
        "a canonical row has a missing case value"
    )

    assert canonical["source_row_count"].ge(1).all(), (
        "a canonical row is backed by no source row"
    )

    # No cases were invented or lost in aggregation.
    assert canonical["cases"].sum() == raw["cases"].sum(), (
        "the canonical case total does not match the source total"
    )

    # The output is ordered by the chronological key.
    assert canonical["period_id"].is_monotonic_increasing, (
        "the canonical dataset is not sorted by period_id"
    )


# ---------------------------------------------------------------------------
# Output and summary
# ---------------------------------------------------------------------------

def write_outputs(canonical: pd.DataFrame, coverage: pd.DataFrame) -> None:
    """Write the canonical dataset and the coverage report."""

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    canonical.to_parquet(CANONICAL_PATH, index=False)

    coverage.to_csv(COVERAGE_PATH, index=False)


def print_summary(
    raw: pd.DataFrame,
    canonical: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    """Print what the canonical dataset contains."""

    print("\nCanonical dengue dataset summary")
    print("-" * 62)

    print(f"Source rows:              {len(raw)}")
    print(f"Canonical rows:           {len(canonical)}")
    print(f"Reporting periods:        {canonical['period_id'].nunique()}")
    print(
        f"Canonical districts:      "
        f"{canonical['canonical_name'].nunique()}"
    )

    print(
        f"\nDate range:               "
        f"{canonical['start_date'].min().date()} to "
        f"{canonical['end_date'].max().date()}"
    )

    print(f"Total cases:              {int(canonical['cases'].sum())}")

    aggregated = canonical.loc[canonical["source_row_count"].gt(1)]

    print(
        f"\nAggregated observations:  {len(aggregated)} "
        f"(Ampara = Ampara + Kalmune)"
    )

    incomplete = coverage.loc[~coverage["is_complete_canonical_coverage"]]

    print(f"Complete periods:         {len(coverage) - len(incomplete)}")
    print(f"Incomplete periods:       {len(incomplete)}")

    for _, issue in incomplete.iterrows():
        print(
            f"\nIncomplete: {issue['source_year']} week "
            f"{issue['source_week']} (period {issue['period_id']})"
        )
        print(
            f"  Observed {issue['observed_canonical_districts']} of "
            f"{issue['expected_canonical_districts']} districts."
        )
        print(f"  Missing: {issue['missing_canonical_districts']}")
        print("  Preserved as absent. Not converted to zero.")

    print(
        "\nperiod_id is the chronological key. The dataset is sorted by it "
        "and\nmust be sorted, lagged, windowed and split by it."
    )


def main() -> int:
    """Build, verify and write the canonical dengue dataset."""

    raw = load_raw_dengue()
    calendar = load_reporting_calendar()
    nodes = load_nodes()

    rows = join_reporting_calendar(raw, calendar)

    report = check_district_mapping(rows)
    print_mapping_report(report)

    rows = add_canonical_names(rows)

    canonical = aggregate_to_canonical(rows, nodes)
    coverage = build_canonical_coverage(canonical, calendar)

    run_canonical_assertions(rows, canonical, coverage, nodes)

    write_outputs(canonical, coverage)

    print_summary(raw, canonical, coverage)

    print(f"\nWrote {CANONICAL_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {COVERAGE_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

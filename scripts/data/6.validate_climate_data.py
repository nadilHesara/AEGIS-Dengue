"""
Validate the daily district climate data.

Reads data/raw/climate_daily_district.csv and checks it against the official
district registry in data/processed/nodes.csv and the reporting calendar.
The climate CSV is never modified.

Every finding is classified PASS, WARNING or FAIL:

    PASS      the check found nothing.
    WARNING   documented or tolerable: values that are unusual but physically
              possible, and history shorter than preferred but sufficient.
    FAIL      structurally broken or physically impossible: a wrong district,
              a duplicate key, a negative rainfall, a dewpoint above air
              temperature.

Four categories of problem are kept separate throughout, because they have
different causes and different remedies:

    1. missing rows          - a district-date absent from the file entirely
    2. present-but-null      - a row that exists with null weather values
    3. invalid values        - physically impossible, so a real defect
    4. suspicious extremes   - unusual but possible, so investigate not delete

Nothing is interpolated and no extreme is dropped. This script reports; it
does not repair.

Outputs (under results/data_validation/weather/):
    weather_validation_summary.md
    missing_district_dates.csv
    duplicate_district_dates.csv
    invalid_weather_values.csv
    weather_coverage_by_district.csv
    weather_coverage_by_year.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]

CLIMATE_PATH = PROJECT_DIR / "data" / "raw" / "climate_daily_district.csv"
NODES_PATH = PROJECT_DIR / "data" / "processed" / "nodes.csv"
CALENDAR_PATH = PROJECT_DIR / "data" / "interim" / "reporting_calendar.csv"

RESULTS_DIR = PROJECT_DIR / "results" / "data_validation" / "weather"

SUMMARY_PATH = RESULTS_DIR / "weather_validation_summary.md"
MISSING_DATES_PATH = RESULTS_DIR / "missing_district_dates.csv"
DUPLICATES_PATH = RESULTS_DIR / "duplicate_district_dates.csv"
INVALID_VALUES_PATH = RESULTS_DIR / "invalid_weather_values.csv"
COVERAGE_DISTRICT_PATH = RESULTS_DIR / "weather_coverage_by_district.csv"
COVERAGE_YEAR_PATH = RESULTS_DIR / "weather_coverage_by_year.csv"

PASS = "PASS"
WARNING = "WARNING"
FAIL = "FAIL"

EXPECTED_DISTRICTS = 25

REQUIRED_COLUMNS = [
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

WEATHER_COLUMNS = [
    "rainfall_mm",
    "temperature_mean_c",
    "temperature_min_c",
    "temperature_max_c",
    "dewpoint_mean_c",
    "relative_humidity_mean",
    "wind_speed_mean",
]

# Physically impossible beyond these bounds. A breach is a FAIL: it means a
# unit conversion was applied twice, or two bands were transposed.
INVALID_BOUNDS = {
    "rainfall_mm": (0.0, 2000.0),
    "temperature_mean_c": (-10.0, 60.0),
    "temperature_min_c": (-10.0, 60.0),
    "temperature_max_c": (-10.0, 60.0),
    "dewpoint_mean_c": (-20.0, 45.0),
    "relative_humidity_mean": (0.0, 100.0),
    "wind_speed_mean": (0.0, 75.0),
}

# Unusual for lowland tropical Sri Lanka but physically possible. A breach is
# a WARNING and must be investigated, never silently dropped: an extreme
# rainfall day during a monsoon is exactly the signal a dengue model needs.
SUSPICIOUS_BOUNDS = {
    "rainfall_mm": (0.0, 400.0),
    "temperature_mean_c": (12.0, 36.0),
    "temperature_min_c": (8.0, 34.0),
    "temperature_max_c": (15.0, 42.0),
    "dewpoint_mean_c": (5.0, 30.0),
    "relative_humidity_mean": (30.0, 100.0),
    "wind_speed_mean": (0.0, 25.0),
}

# Expected units, asserted by range rather than taken on trust. Kelvin
# temperatures and metre rainfall are the two conversions most often missed.
EXPECTED_UNITS = {
    "rainfall_mm": "mm/day",
    "temperature_mean_c": "degrees Celsius",
    "temperature_min_c": "degrees Celsius",
    "temperature_max_c": "degrees Celsius",
    "dewpoint_mean_c": "degrees Celsius",
    "relative_humidity_mean": "percent (0-100)",
    "wind_speed_mean": "m/s",
}

# History required before the first dengue reporting period, per the climate
# specification. Below the minimum is a FAIL; between the two is a WARNING.
PREFERRED_HISTORY_WEEKS = 52
MINIMUM_HISTORY_WEEKS = 26

# A missing run this long or longer is reported individually: a single absent
# day is a nuisance, but a fortnight breaks any lag window crossing it.
LONG_GAP_DAYS = 5


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_nodes(path: Path = NODES_PATH) -> pd.DataFrame:
    """Load the official district registry."""

    nodes = pd.read_csv(path)

    return nodes[["node_id", "canonical_name"]].copy()


def load_climate(path: Path = CLIMATE_PATH) -> pd.DataFrame:
    """
    Load the climate CSV and parse the dates.

    Unparseable dates are kept as null rather than dropped, so that the date
    check can count and report them.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist.\n"
            "Run scripts/5.extract_era5_daily.py first."
        )

    climate = pd.read_csv(path)

    if "date" in climate.columns:
        climate["date_parsed"] = pd.to_datetime(
            climate["date"], errors="coerce"
        )

    return climate


def load_first_period_start(path: Path = CALENDAR_PATH) -> pd.Timestamp | None:
    """Return the first dengue reporting period start, if the calendar exists."""

    if not path.exists():
        return None

    calendar = pd.read_csv(path)

    return pd.to_datetime(calendar["start_date"]).min()


# ---------------------------------------------------------------------------
# Schema and dates
# ---------------------------------------------------------------------------

def check_schema(climate: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Confirm every required column is present."""

    missing = [
        column for column in REQUIRED_COLUMNS if column not in climate.columns
    ]

    findings = pd.DataFrame({"missing_column": missing})

    return (FAIL if missing else PASS), findings


def check_dates(climate: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Confirm every date parses."""

    if "date_parsed" not in climate.columns:
        return FAIL, pd.DataFrame({"issue": ["the date column is absent"]})

    unparsed = climate.loc[
        climate["date_parsed"].isna(), ["date"]
    ].drop_duplicates()

    return (FAIL if len(unparsed) else PASS), unparsed


def check_districts(
    climate: pd.DataFrame,
    nodes: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    """
    Confirm the file holds exactly the 25 official districts.

    Both directions matter: a missing district loses a model node, and an
    unexpected name means the extraction used a different boundary source.
    """

    observed = set(climate["canonical_name"].dropna().unique())
    expected = set(nodes["canonical_name"])

    records = [
        {"canonical_name": name, "issue": "missing from the climate data"}
        for name in sorted(expected - observed)
    ] + [
        {"canonical_name": name, "issue": "not an official district"}
        for name in sorted(observed - expected)
    ]

    return (FAIL if records else PASS), pd.DataFrame(
        records, columns=["canonical_name", "issue"]
    )


def check_node_id_mapping(
    climate: pd.DataFrame,
    nodes: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    """
    Confirm each canonical_name carries its registered node_id.

    A name paired with the wrong id would silently attach one district's
    weather to another district's cases, which no later check would catch.
    """

    official = dict(zip(nodes["canonical_name"], nodes["node_id"]))

    pairs = climate[["canonical_name", "node_id"]].drop_duplicates().dropna()

    records = []

    for _, row in pairs.iterrows():
        name = row["canonical_name"]
        expected = official.get(name)

        if expected is None:
            continue

        if int(row["node_id"]) != int(expected):
            records.append(
                {
                    "canonical_name": name,
                    "node_id_in_file": int(row["node_id"]),
                    "node_id_expected": int(expected),
                }
            )

    # A name must also not appear under two different ids.
    for name, group in pairs.groupby("canonical_name"):
        if len(group) > 1:
            records.append(
                {
                    "canonical_name": name,
                    "node_id_in_file": sorted(
                        int(value) for value in group["node_id"]
                    ),
                    "node_id_expected": int(official.get(name, -1)),
                }
            )

    return (FAIL if records else PASS), pd.DataFrame(
        records,
        columns=["canonical_name", "node_id_in_file", "node_id_expected"],
    )


def check_duplicates(climate: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Detect repeated date and canonical_name pairs."""

    keyed = climate.dropna(subset=["date_parsed", "canonical_name"])

    duplicated = keyed.duplicated(
        subset=["date_parsed", "canonical_name"], keep=False
    )

    if not duplicated.any():
        return PASS, pd.DataFrame(
            columns=["date", "canonical_name", "node_id", "row_count"]
        )

    findings = (
        keyed.loc[duplicated]
        .groupby(["date_parsed", "canonical_name"], as_index=False)
        .agg(node_id=("node_id", "first"), row_count=("canonical_name", "size"))
        .rename(columns={"date_parsed": "date"})
        .sort_values(["date", "canonical_name"])
    )

    return FAIL, findings


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def build_expected_grid(
    climate: pd.DataFrame,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete date-by-district grid the file should contain.

    The span is taken from the file itself: this check is about internal
    completeness, not about whether the extraction window was long enough,
    which check_history covers separately.
    """

    dates = climate["date_parsed"].dropna()

    if dates.empty:
        return pd.DataFrame(columns=["date_parsed", "canonical_name"])

    full_range = pd.date_range(dates.min(), dates.max(), freq="D")

    return pd.MultiIndex.from_product(
        [full_range, sorted(nodes["canonical_name"])],
        names=["date_parsed", "canonical_name"],
    ).to_frame(index=False)


def check_missing_rows(
    climate: pd.DataFrame,
    nodes: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    """
    Detect absent district-date combinations.

    ERA5 is a reanalysis: it produces a value for every cell on every date by
    construction. A missing row therefore means the extraction failed, not
    that weather did not occur, so this is a FAIL rather than a WARNING.
    """

    expected = build_expected_grid(climate, nodes)

    if expected.empty:
        return FAIL, pd.DataFrame(columns=["date", "canonical_name", "node_id"])

    present = climate[["date_parsed", "canonical_name"]].drop_duplicates()

    merged = expected.merge(
        present, on=["date_parsed", "canonical_name"], how="left", indicator=True
    )

    missing = merged.loc[merged["_merge"].eq("left_only")].copy()

    if missing.empty:
        return PASS, pd.DataFrame(columns=["date", "canonical_name", "node_id"])

    node_lookup = dict(zip(nodes["canonical_name"], nodes["node_id"]))

    missing["node_id"] = missing["canonical_name"].map(node_lookup)

    findings = missing.rename(columns={"date_parsed": "date"})[
        ["date", "canonical_name", "node_id"]
    ].sort_values(["date", "canonical_name"])

    return FAIL, findings


def check_complete_days(
    climate: pd.DataFrame,
    nodes: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    """Identify dates that do not carry all 25 districts."""

    counts = (
        climate.dropna(subset=["date_parsed"])
        .groupby("date_parsed")["canonical_name"]
        .nunique()
    )

    incomplete = counts.loc[counts.ne(EXPECTED_DISTRICTS)]

    if incomplete.empty:
        return PASS, pd.DataFrame(
            columns=["date", "districts_present", "districts_expected"]
        )

    findings = pd.DataFrame(
        {
            "date": incomplete.index,
            "districts_present": incomplete.to_numpy(),
            "districts_expected": EXPECTED_DISTRICTS,
        }
    )

    return FAIL, findings


def find_long_missing_runs(
    climate: pd.DataFrame,
    nodes: pd.DataFrame,
    threshold: int = LONG_GAP_DAYS,
) -> pd.DataFrame:
    """
    Find consecutive stretches of absent or null data per district.

    Runs matter more than totals. Fifty scattered missing days are an
    annoyance; five consecutive days break every lag window that crosses
    them, so the runs are reported with their start, end and length.
    """

    dates = climate["date_parsed"].dropna()

    if dates.empty:
        return pd.DataFrame(
            columns=[
                "canonical_name",
                "gap_start",
                "gap_end",
                "gap_days",
                "reason",
            ]
        )

    full_range = pd.date_range(dates.min(), dates.max(), freq="D")

    records = []

    for name in sorted(nodes["canonical_name"]):
        district = climate.loc[
            climate["canonical_name"].eq(name)
        ].drop_duplicates(subset=["date_parsed"]).set_index("date_parsed")

        reindexed = district.reindex(full_range)

        # Usable means the row exists and carries every weather value.
        if all(column in reindexed.columns for column in WEATHER_COLUMNS):
            usable = reindexed[WEATHER_COLUMNS].notna().all(axis=1)
        else:
            usable = pd.Series(False, index=full_range)

        unusable = ~usable

        if not unusable.any():
            continue

        # Group consecutive unusable days into runs.
        run_id = (unusable != unusable.shift()).cumsum()

        for _, group in unusable.loc[unusable].groupby(run_id.loc[unusable]):
            if len(group) < threshold:
                continue

            start = group.index.min()
            end = group.index.max()

            absent = reindexed.loc[group.index, "canonical_name"].isna().all()

            records.append(
                {
                    "canonical_name": name,
                    "gap_start": start,
                    "gap_end": end,
                    "gap_days": len(group),
                    "reason": (
                        "rows absent" if absent else "rows present, values null"
                    ),
                }
            )

    return pd.DataFrame(
        records,
        columns=[
            "canonical_name",
            "gap_start",
            "gap_end",
            "gap_days",
            "reason",
        ],
    ).sort_values(["gap_days", "canonical_name"], ascending=[False, True])


def check_long_gaps(
    climate: pd.DataFrame,
    nodes: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    """Report long consecutive missing-data runs."""

    runs = find_long_missing_runs(climate, nodes)

    return (FAIL if len(runs) else PASS), runs


def check_history(
    climate: pd.DataFrame,
    first_period_start: pd.Timestamp | None,
) -> tuple[str, pd.DataFrame]:
    """
    Confirm enough weather precedes the first dengue reporting period.

    The first forecast origin needs a full lag window behind it. Below the
    26-week minimum is a FAIL; between 26 and 52 weeks is a WARNING, since
    a 52-week lookback could not then be tested.
    """

    if first_period_start is None:
        return WARNING, pd.DataFrame(
            {"issue": ["the reporting calendar is absent; history unchecked"]}
        )

    dates = climate["date_parsed"].dropna()

    if dates.empty:
        return FAIL, pd.DataFrame({"issue": ["no parseable dates"]})

    first_weather = dates.min()

    available_days = (first_period_start - first_weather).days
    available_weeks = available_days / 7

    preferred_start = first_period_start - pd.Timedelta(
        weeks=PREFERRED_HISTORY_WEEKS
    )
    minimum_start = first_period_start - pd.Timedelta(
        weeks=MINIMUM_HISTORY_WEEKS
    )

    findings = pd.DataFrame(
        [
            {
                "first_weather_date": first_weather.date(),
                "first_dengue_period_start": first_period_start.date(),
                "history_days": available_days,
                "history_weeks": round(available_weeks, 1),
                "preferred_start": preferred_start.date(),
                "minimum_start": minimum_start.date(),
            }
        ]
    )

    if first_weather <= preferred_start:
        return PASS, findings.iloc[0:0]

    if first_weather <= minimum_start:
        return WARNING, findings

    return FAIL, findings


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

def check_present_but_null(climate: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """
    Find rows that exist but carry null weather values.

    Kept separate from absent rows: a present-but-null row is an honest record
    of an unobserved value, whereas an absent row breaks the grid.
    """

    available = [
        column for column in WEATHER_COLUMNS if column in climate.columns
    ]

    if not available:
        return FAIL, pd.DataFrame(columns=["date", "canonical_name", "column"])

    null_mask = climate[available].isna().any(axis=1)

    if not null_mask.any():
        return PASS, pd.DataFrame(columns=["date", "canonical_name", "column"])

    records = []

    for column in available:
        rows = climate.loc[climate[column].isna()]

        for _, row in rows.iterrows():
            records.append(
                {
                    "date": row.get("date"),
                    "canonical_name": row.get("canonical_name"),
                    "column": column,
                }
            )

    return WARNING, pd.DataFrame(records)


def _value_findings(
    climate: pd.DataFrame,
    bounds: dict,
    severity: str,
) -> pd.DataFrame:
    """Collect values falling outside the given bounds."""

    records = []

    for column, (low, high) in bounds.items():
        if column not in climate.columns:
            continue

        values = pd.to_numeric(climate[column], errors="coerce")

        outside = climate.loc[values.notna() & ((values < low) | (values > high))]

        for _, row in outside.iterrows():
            records.append(
                {
                    "date": row.get("date"),
                    "canonical_name": row.get("canonical_name"),
                    "node_id": row.get("node_id"),
                    "column": column,
                    "value": row.get(column),
                    "lower_bound": low,
                    "upper_bound": high,
                    "severity": severity,
                    "issue": f"{column} outside {low} to {high}",
                }
            )

    return pd.DataFrame(records, columns=_VALUE_COLUMNS)


_VALUE_COLUMNS = [
    "date",
    "canonical_name",
    "node_id",
    "column",
    "value",
    "lower_bound",
    "upper_bound",
    "severity",
    "issue",
]


def check_invalid_values(climate: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """
    Find physically impossible values.

    Includes the cross-column relations, which catch transposed bands that
    per-column range checks pass straight through.
    """

    findings = [_value_findings(climate, INVALID_BOUNDS, FAIL)]

    def _relation(mask, column, issue):
        rows = climate.loc[mask]

        if rows.empty:
            return pd.DataFrame(columns=_VALUE_COLUMNS)

        return pd.DataFrame(
            [
                {
                    "date": row.get("date"),
                    "canonical_name": row.get("canonical_name"),
                    "node_id": row.get("node_id"),
                    "column": column,
                    "value": row.get(column),
                    "lower_bound": None,
                    "upper_bound": None,
                    "severity": FAIL,
                    "issue": issue,
                }
                for _, row in rows.iterrows()
            ],
            columns=_VALUE_COLUMNS,
        )

    has = lambda *columns: all(
        column in climate.columns for column in columns
    )

    if has("temperature_min_c", "temperature_max_c"):
        findings.append(
            _relation(
                climate["temperature_min_c"] > climate["temperature_max_c"],
                "temperature_min_c",
                "minimum temperature above maximum temperature",
            )
        )

    if has("temperature_mean_c", "temperature_min_c"):
        findings.append(
            _relation(
                climate["temperature_mean_c"] < climate["temperature_min_c"],
                "temperature_mean_c",
                "mean temperature below minimum temperature",
            )
        )

    if has("temperature_mean_c", "temperature_max_c"):
        findings.append(
            _relation(
                climate["temperature_mean_c"] > climate["temperature_max_c"],
                "temperature_mean_c",
                "mean temperature above maximum temperature",
            )
        )

    if has("dewpoint_mean_c", "temperature_mean_c"):
        # Dewpoint above air temperature implies supersaturation and normally
        # means the two bands were transposed. Half a degree of tolerance
        # allows for rounding in a saturated atmosphere.
        findings.append(
            _relation(
                climate["dewpoint_mean_c"]
                > climate["temperature_mean_c"] + 0.5,
                "dewpoint_mean_c",
                "dewpoint above air temperature",
            )
        )

    # Empty frames are dropped before concatenating: an all-NA frame would
    # otherwise influence the resulting dtypes.
    populated = [frame for frame in findings if not frame.empty]

    if not populated:
        return PASS, pd.DataFrame(columns=_VALUE_COLUMNS)

    # The relation checks carry no numeric bounds, so those columns are typed
    # explicitly rather than left for concat to infer from all-null values.
    for frame in populated:
        for column in ["lower_bound", "upper_bound"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    combined = pd.concat(populated, ignore_index=True)

    return (FAIL if len(combined) else PASS), combined


def check_suspicious_values(climate: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """
    Find unusual but physically possible values.

    Reported for investigation and never dropped. A 300 mm rainfall day is
    plausible in a Sri Lankan monsoon and is exactly the kind of event a
    dengue model needs to see.
    """

    findings = _value_findings(climate, SUSPICIOUS_BOUNDS, WARNING)

    # Values already reported as impossible are not repeated here.
    _, invalid = check_invalid_values(climate)

    if not invalid.empty and not findings.empty:
        invalid_keys = set(
            zip(invalid["date"], invalid["canonical_name"], invalid["column"])
        )

        keep = [
            (row["date"], row["canonical_name"], row["column"])
            not in invalid_keys
            for _, row in findings.iterrows()
        ]

        findings = findings.loc[keep]

    return (WARNING if len(findings) else PASS), findings


def check_units(climate: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """
    Infer whether the units match the schema.

    Units cannot be read from a CSV, so they are inferred from the value
    distributions. The two conversions most often missed have unmistakable
    signatures: Kelvin temperatures sit near 300 rather than 27, and rainfall
    left in metres is a thousand times too small.
    """

    records = []

    def _median(column):
        if column not in climate.columns:
            return None

        values = pd.to_numeric(climate[column], errors="coerce").dropna()

        return values.median() if len(values) else None

    for column in ["temperature_mean_c", "temperature_min_c", "temperature_max_c"]:
        median = _median(column)

        if median is None:
            continue

        if median > 100:
            records.append(
                {
                    "column": column,
                    "expected_units": EXPECTED_UNITS[column],
                    "median": round(float(median), 3),
                    "issue": (
                        "median above 100 suggests Kelvin; subtract 273.15"
                    ),
                }
            )

    median_rain = _median("rainfall_mm")

    if median_rain is not None:
        maximum_rain = pd.to_numeric(
            climate["rainfall_mm"], errors="coerce"
        ).max()

        if maximum_rain is not None and maximum_rain < 1.0:
            records.append(
                {
                    "column": "rainfall_mm",
                    "expected_units": EXPECTED_UNITS["rainfall_mm"],
                    "median": round(float(median_rain), 6),
                    "issue": (
                        "maximum below 1 suggests metres; multiply by 1000"
                    ),
                }
            )

    median_humidity = _median("relative_humidity_mean")

    if median_humidity is not None and median_humidity <= 1.0:
        records.append(
            {
                "column": "relative_humidity_mean",
                "expected_units": EXPECTED_UNITS["relative_humidity_mean"],
                "median": round(float(median_humidity), 6),
                "issue": "median at or below 1 suggests a fraction; scale by 100",
            }
        )

    findings = pd.DataFrame(
        records, columns=["column", "expected_units", "median", "issue"]
    )

    return (FAIL if len(findings) else PASS), findings


# ---------------------------------------------------------------------------
# Coverage reports
# ---------------------------------------------------------------------------

def build_coverage_by_district(
    climate: pd.DataFrame,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """Report per-district coverage and missing values."""

    dates = climate["date_parsed"].dropna()

    expected_days = (
        len(pd.date_range(dates.min(), dates.max(), freq="D"))
        if not dates.empty
        else 0
    )

    available = [
        column for column in WEATHER_COLUMNS if column in climate.columns
    ]

    records = []

    for _, node in nodes.iterrows():
        name = node["canonical_name"]

        district = climate.loc[climate["canonical_name"].eq(name)]

        observed_days = district["date_parsed"].nunique()

        record = {
            "node_id": node["node_id"],
            "canonical_name": name,
            "rows_present": len(district),
            "dates_present": observed_days,
            "dates_expected": expected_days,
            "dates_missing": expected_days - observed_days,
        }

        for column in available:
            record[f"null_{column}"] = int(district[column].isna().sum())

        record["rows_fully_populated"] = (
            int(district[available].notna().all(axis=1).sum())
            if available
            else 0
        )

        records.append(record)

    return pd.DataFrame(records)


def build_coverage_by_year(
    climate: pd.DataFrame,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """Report per-year coverage and missing values."""

    keyed = climate.dropna(subset=["date_parsed"]).copy()

    if keyed.empty:
        return pd.DataFrame()

    keyed["year"] = keyed["date_parsed"].dt.year

    available = [
        column for column in WEATHER_COLUMNS if column in keyed.columns
    ]

    first = keyed["date_parsed"].min()
    last = keyed["date_parsed"].max()

    records = []

    for year, group in keyed.groupby("year"):
        # Only the portion of the year the file actually spans is expected.
        year_start = max(first, pd.Timestamp(year=year, month=1, day=1))
        year_end = min(last, pd.Timestamp(year=year, month=12, day=31))

        expected_days = len(pd.date_range(year_start, year_end, freq="D"))
        expected_rows = expected_days * EXPECTED_DISTRICTS

        record = {
            "year": year,
            "rows_present": len(group),
            "rows_expected": expected_rows,
            "dates_present": group["date_parsed"].nunique(),
            "dates_expected": expected_days,
            "districts_present": group["canonical_name"].nunique(),
        }

        for column in available:
            record[f"null_{column}"] = int(group[column].isna().sum())

        records.append(record)

    return pd.DataFrame(records).sort_values("year")


def build_missing_by_variable(climate: pd.DataFrame) -> pd.DataFrame:
    """Report null counts per weather variable."""

    available = [
        column for column in WEATHER_COLUMNS if column in climate.columns
    ]

    row_count = len(climate)

    records = [
        {
            "column": column,
            "expected_units": EXPECTED_UNITS[column],
            "null_count": int(climate[column].isna().sum()),
            "null_share_percent": round(
                100 * climate[column].isna().sum() / row_count, 4
            )
            if row_count
            else 0.0,
        }
        for column in available
    ]

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def validate_climate_data(
    climate: pd.DataFrame,
    nodes: pd.DataFrame,
    first_period_start: pd.Timestamp | None = None,
) -> dict:
    """Run every check and return the findings with their classifications."""

    schema_status, schema_findings = check_schema(climate)

    # Without the key columns the remaining checks cannot run meaningfully.
    if schema_status == FAIL and any(
        column in schema_findings["missing_column"].tolist()
        for column in ["date", "node_id", "canonical_name"]
    ):
        return {
            "checks": {"schema": (FAIL, schema_findings)},
            "coverage_by_district": pd.DataFrame(),
            "coverage_by_year": pd.DataFrame(),
            "missing_by_variable": pd.DataFrame(),
            "aborted": True,
        }

    checks = {
        "schema": (schema_status, schema_findings),
        "date_parsing": check_dates(climate),
        "district_completeness": check_districts(climate, nodes),
        "node_id_mapping": check_node_id_mapping(climate, nodes),
        "duplicate_district_dates": check_duplicates(climate),
        "missing_district_dates": check_missing_rows(climate, nodes),
        "incomplete_days": check_complete_days(climate, nodes),
        "long_missing_runs": check_long_gaps(climate, nodes),
        "weather_history": check_history(climate, first_period_start),
        "present_but_null": check_present_but_null(climate),
        "invalid_values": check_invalid_values(climate),
        "suspicious_extremes": check_suspicious_values(climate),
        "units": check_units(climate),
    }

    return {
        "checks": checks,
        "coverage_by_district": build_coverage_by_district(climate, nodes),
        "coverage_by_year": build_coverage_by_year(climate, nodes),
        "missing_by_variable": build_missing_by_variable(climate),
        "aborted": False,
    }


def count_failures(validation: dict) -> int:
    """Return the number of checks classified FAIL."""

    return sum(
        1
        for status, _ in validation["checks"].values()
        if status == FAIL
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    """Write a findings frame, formatting any date columns."""

    result = frame.copy()

    for column in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = result[column].dt.strftime("%Y-%m-%d")

    result.to_csv(path, index=False)


def write_outputs(validation: dict) -> None:
    """Write the summary and the detailed issue files."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    checks = validation["checks"]

    def _findings(name):
        return checks.get(name, (PASS, pd.DataFrame()))[1]

    _write_frame(_findings("missing_district_dates"), MISSING_DATES_PATH)
    _write_frame(_findings("duplicate_district_dates"), DUPLICATES_PATH)

    invalid = _findings("invalid_values")
    suspicious = _findings("suspicious_extremes")

    # One file, with severity distinguishing impossible from merely unusual.
    populated = [
        frame for frame in [invalid, suspicious] if not frame.empty
    ]

    combined = (
        pd.concat(populated, ignore_index=True)
        if populated
        else pd.DataFrame(columns=_VALUE_COLUMNS)
    )

    _write_frame(combined, INVALID_VALUES_PATH)

    _write_frame(
        validation["coverage_by_district"], COVERAGE_DISTRICT_PATH
    )
    _write_frame(validation["coverage_by_year"], COVERAGE_YEAR_PATH)

    SUMMARY_PATH.write_text(build_summary_markdown(validation), encoding="utf-8")


def build_summary_markdown(validation: dict) -> str:
    """Build the markdown validation summary."""

    checks = validation["checks"]

    lines = [
        "# Weather validation summary",
        "",
        "Validation of `data/raw/climate_daily_district.csv` against the "
        "official district registry in `data/processed/nodes.csv`.",
        "",
        "The climate CSV is never modified. Nothing is interpolated and no "
        "extreme value is dropped.",
        "",
        "## Classifications",
        "",
        "| Check | Status | Findings |",
        "| --- | --- | --- |",
    ]

    for name, (status, findings) in checks.items():
        lines.append(f"| `{name}` | **{status}** | {len(findings)} |")

    failures = count_failures(validation)

    lines += [
        "",
        f"**{failures} check(s) classified FAIL.**"
        if failures
        else "**No checks classified FAIL.**",
        "",
        "## The four categories of problem",
        "",
        "These are kept separate because they have different causes and "
        "different remedies.",
        "",
        "| Category | Check | Meaning |",
        "| --- | --- | --- |",
        "| Missing rows | `missing_district_dates` | The district-date is "
        "absent from the file. ERA5 is a reanalysis, so this means the "
        "extraction failed. |",
        "| Present but null | `present_but_null` | The row exists and "
        "honestly records an unobserved value. |",
        "| Invalid values | `invalid_values` | Physically impossible. A real "
        "defect, usually a unit conversion or a transposed band. |",
        "| Suspicious extremes | `suspicious_extremes` | Unusual but "
        "possible. Investigate; never drop. |",
        "",
    ]

    if not validation["missing_by_variable"].empty:
        lines += [
            "## Missing values by variable",
            "",
            validation["missing_by_variable"].to_markdown(index=False),
            "",
        ]

    coverage_year = validation["coverage_by_year"]

    if not coverage_year.empty:
        columns = [
            column
            for column in [
                "year",
                "rows_present",
                "rows_expected",
                "dates_present",
                "dates_expected",
                "districts_present",
            ]
            if column in coverage_year.columns
        ]

        lines += [
            "## Coverage by year",
            "",
            coverage_year[columns].to_markdown(index=False),
            "",
        ]

    coverage_district = validation["coverage_by_district"]

    if not coverage_district.empty:
        columns = [
            column
            for column in [
                "node_id",
                "canonical_name",
                "dates_present",
                "dates_expected",
                "dates_missing",
                "rows_fully_populated",
            ]
            if column in coverage_district.columns
        ]

        lines += [
            "## Coverage by district",
            "",
            coverage_district[columns].to_markdown(index=False),
            "",
        ]

    history_status, history = checks.get("weather_history", (PASS, pd.DataFrame()))

    if not history.empty:
        lines += [
            "## Weather history before the first forecast origin",
            "",
            f"Status: **{history_status}**",
            "",
            history.to_markdown(index=False),
            "",
        ]

    runs = checks.get("long_missing_runs", (PASS, pd.DataFrame()))[1]

    if not runs.empty:
        lines += [
            f"## Long missing runs ({LONG_GAP_DAYS} days or more)",
            "",
            runs.head(50).to_markdown(index=False),
            "",
        ]

    lines += [
        "## Detailed issue files",
        "",
        "| File | Contents |",
        "| --- | --- |",
        "| `missing_district_dates.csv` | Absent district-date rows |",
        "| `duplicate_district_dates.csv` | Repeated date and district pairs |",
        "| `invalid_weather_values.csv` | Invalid and suspicious values, by "
        "`severity` |",
        "| `weather_coverage_by_district.csv` | Per-district coverage |",
        "| `weather_coverage_by_year.csv` | Per-year coverage |",
        "",
    ]

    return "\n".join(lines)


def print_summary(validation: dict) -> None:
    """Print the classification of each check."""

    print("\nWeather validation summary")
    print("-" * 62)

    for name, (status, findings) in validation["checks"].items():
        if status == PASS:
            print(f"{PASS:<8} {name}")
        else:
            print(f"{status:<8} {name} - {len(findings)} finding(s)")

    variables = validation["missing_by_variable"]

    if not variables.empty:
        print("\nMissing values by variable")

        for _, row in variables.iterrows():
            print(
                f"  {row['column']:<26} {row['null_count']:>8} "
                f"({row['null_share_percent']:.3f}%)"
            )


def main() -> int:
    """Validate the daily district climate data."""

    nodes = load_nodes()

    try:
        climate = load_climate()
    except FileNotFoundError as error:
        print(error)
        return 1

    print(f"Loaded {len(climate)} rows from "
          f"{CLIMATE_PATH.relative_to(PROJECT_DIR)}")
    print(f"District registry: {len(nodes)} districts")

    validation = validate_climate_data(
        climate, nodes, load_first_period_start()
    )

    if validation["aborted"]:
        print("\nFAIL     schema - key columns are absent, checks stopped")

        missing = validation["checks"]["schema"][1]["missing_column"].tolist()

        print(f"         missing: {missing}")

        return 1

    write_outputs(validation)

    print_summary(validation)

    failures = count_failures(validation)

    print(f"\nWrote {RESULTS_DIR.relative_to(PROJECT_DIR)}")

    if failures:
        print(f"\n{failures} check(s) classified FAIL.")
        return 1

    print("\nNo checks classified FAIL.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

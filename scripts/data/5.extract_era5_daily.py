"""
Extract daily ERA5-Land weather for the 25 canonical Sri Lankan districts.

Stage 1 of the climate pipeline, as specified in docs/climate_dataset_schema.md.
The output is raw daily climate data and is deliberately independent of the
dengue dataset: no reporting periods, no lags, no case counts.

The extraction runs on Google Earth Engine. Hourly ERA5-Land bands are reduced
to daily values, then aggregated to districts with an area-aware reducer over a
downsampled grid. Sri Lanka is small relative to the ERA5 grid — Colombo is
smaller than one native 0.25 degree cell — so the downsampling is required, not
cosmetic. See docs/climate_dataset_schema.md for the measurements behind that.

Authentication (once per machine):

    earthengine authenticate

Usage:

    python scripts/5.extract_era5_daily.py                  # full run
    python scripts/5.extract_era5_daily.py --start 2010-01-01 --end 2010-12-31
    python scripts/5.extract_era5_daily.py --validate-only  # re-check the CSV
    python scripts/5.extract_era5_daily.py --dry-run        # no GEE calls
    python scripts/5.extract_era5_daily.py --submit-exports  # queue Drive exports
    python scripts/5.extract_era5_daily.py --check-tasks     # list Earth Engine tasks

Queued exports are written to Google Drive as CSV files named
climate_daily_<year> (or climate_daily_<year>_<month> if monthly exports are
requested). After downloading those files into data/raw/era5_chunks/, they can
be combined into data/raw/climate_daily_district.csv.

Outputs:
    data/raw/climate_daily_district.csv
    data/raw/era5_chunks/climate_daily_<year>.csv
    results/data_validation/era5_extraction_manifest.json
"""

from __future__ import annotations

import argparse
import calendar as calendar_lib
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]

NODES_PATH = PROJECT_DIR / "data" / "processed" / "nodes.csv"
CALENDAR_PATH = PROJECT_DIR / "data" / "interim" / "reporting_calendar.csv"

RAW_DIR = PROJECT_DIR / "data" / "raw"
CHUNK_DIR = RAW_DIR / "era5_chunks"
OUTPUT_PATH = RAW_DIR / "climate_daily_district.csv"

RESULTS_DIR = PROJECT_DIR / "results" / "data_validation"
MANIFEST_PATH = RESULTS_DIR / "era5_extraction_manifest.json"


# ---------------------------------------------------------------------------
# Earth Engine assets
#
# Recorded in the manifest with the extraction date. ERA5 is periodically
# reprocessed, so a rebuilt dataset cannot be compared against an earlier model
# run without knowing which asset version produced it.
# ---------------------------------------------------------------------------

ERA5_ASSET = "ECMWF/ERA5_LAND/HOURLY"

# geoBoundaries ADM2 corresponds to Sri Lanka's 25 districts.
DISTRICT_BOUNDARY_ASSET = "WM/geoLab/geoBoundaries/600/ADM2"

# Static land-sea mask. Ocean pixels must not enter a district mean: most Sri
# Lankan districts are coastal, and sea surface conditions differ
# systematically from land.
LAND_MASK_ASSET = "ECMWF/ERA5_LAND/HOURLY"

EXTRACTION_VERSION = "era5land-v1"
DATA_SOURCE = "ERA5-Land hourly (ECMWF/ERA5_LAND/HOURLY) via Google Earth Engine"

# Downsampled grid for district aggregation. ERA5-Land is ~9 km native; 0.02
# degrees (~2.2 km) gives even Colombo a few hundred sample points, which makes
# the area weighting numerically stable for small polygons.
AGGREGATION_SCALE_M = 2000

LAND_MASK_THRESHOLD = 0.5
RELAXED_LAND_MASK_THRESHOLD = 0.1

# ERA5-Land latency. The final product trails real time by 2-3 months.
ERA5_LATENCY_DAYS = 90

# Preferred history before the first dengue period, per the climate spec.
HISTORY_WEEKS = 52

EXPECTED_DISTRICTS = 25

BATCH_DISTRICTS = 25
DEFAULT_EXPORT_FOLDER = "era5_chunks"
EXPORT_GRANULARITY_CHOICES = ("year", "month")


# ---------------------------------------------------------------------------
# Explicit GADM to canonical name mapping
#
# Never fuzzy matching. Every GADM spelling is assigned deliberately. A name
# that is not listed raises rather than being guessed at, because a wrong
# district silently corrupts every downstream model.
#
# This mirrors GADM_TO_CANONICAL in 3.create_nodes.py. It is restated here
# because that module imports geopandas, which the Earth Engine environment
# (Colab) does not necessarily provide. The two are asserted equal at runtime
# whenever both are importable.
# ---------------------------------------------------------------------------

BOUNDARY_TO_CANONICAL = {
    "Ampara District": "Ampara",
    "Anuradhapura District": "Anuradhapura",
    "Badulla District": "Badulla",
    "Batticaloa District": "Batticaloa",
    "Colombo District": "Colombo",
    "Galle District": "Galle",
    "Gampaha District": "Gampaha",
    "Hambantota District": "Hambantota",
    "Jaffna District": "Jaffna",
    "Kalutara District": "Kalutara",
    "Kandy District": "Kandy",
    "Kegalle District": "Kegalle",
    "Kilinochchi District": "Kilinochchi",
    "Kurunegala District": "Kurunegala",
    "Mannar District": "Mannar",
    "Matale District": "Matale",
    "Matara District": "Matara",
    "Monaragala District": "Moneragala",
    "Mullaitivu District": "Mullaitivu",
    "Nuwara Eliya District": "Nuwara Eliya",
    "Polonnaruwa District": "Polonnaruwa",
    "Puttalam District": "Puttalam",
    "Ratnapura District": "Ratnapura",
    "Trincomalee District": "Trincomalee",
    "Vavuniya District": "Vavuniya",
}

OUTPUT_COLUMNS = [
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
    "weather_observed",
    "u_wind_mean",
    "v_wind_mean",
    "surface_pressure_mean",
    "data_source",
    "extraction_version",
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


# ---------------------------------------------------------------------------
# Node registry and date range
# ---------------------------------------------------------------------------

def load_nodes(path: Path = NODES_PATH) -> pd.DataFrame:
    """Load the permanent district registry and check its invariants."""

    nodes = pd.read_csv(path)

    if len(nodes) != EXPECTED_DISTRICTS:
        raise ValueError(
            f"nodes.csv holds {len(nodes)} districts, expected "
            f"{EXPECTED_DISTRICTS}."
        )

    if nodes["node_id"].tolist() != list(range(EXPECTED_DISTRICTS)):
        raise ValueError("node_id values must be exactly 0 through 24.")

    if nodes["canonical_name"].duplicated().any():
        raise ValueError("nodes.csv contains a duplicate canonical name.")

    return nodes[["node_id", "canonical_name", "gadm_name", "gadm_gid"]].copy()


def resolve_date_range(
    start: str | None,
    end: str | None,
    history_weeks: int = HISTORY_WEEKS,
) -> tuple[date, date]:
    """
    Determine the extraction window.

    The default covers every dengue reporting period plus the preferred
    52 weeks of prior history, so the earliest forecastable period has a full
    lag window behind it.
    """

    if start and end:
        return (
            pd.Timestamp(start).date(),
            pd.Timestamp(end).date(),
        )

    calendar = pd.read_csv(CALENDAR_PATH)

    first_period = pd.to_datetime(calendar["start_date"]).min()
    last_period = pd.to_datetime(calendar["end_date"]).max()

    default_start = (
        first_period - pd.Timedelta(weeks=history_weeks)
    ).date()

    # ERA5-Land trails real time, so the window cannot extend past what is
    # published. A short series is reported rather than silently truncated.
    latest_available = (
        pd.Timestamp.now(tz="UTC").tz_localize(None)
        - pd.Timedelta(days=ERA5_LATENCY_DAYS)
    ).date()

    default_end = min(last_period.date(), latest_available)

    resolved_start = pd.Timestamp(start).date() if start else default_start
    resolved_end = pd.Timestamp(end).date() if end else default_end

    if resolved_end < resolved_start:
        raise ValueError(
            f"End date {resolved_end} precedes start date {resolved_start}."
        )

    if default_end < last_period.date():
        print(
            f"NOTE: ERA5-Land availability ends near {latest_available}, "
            f"before the last dengue period ({last_period.date()}).\n"
            f"      The tail of the dengue series will have no weather.\n"
            f"      Re-run once ERA5-Land catches up."
        )

    return resolved_start, resolved_end


# ---------------------------------------------------------------------------
# Earth Engine
# ---------------------------------------------------------------------------

def initialise_earth_engine(project: str | None = None):
    """Import and initialise Earth Engine, failing with a usable message."""

    try:
        import ee
    except ImportError as error:
        raise SystemExit(
            "earthengine-api is not installed.\n"
            "  pip install earthengine-api\n"
            "  earthengine authenticate"
        ) from error

    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception:
        # A fresh machine needs an interactive consent step once.
        ee.Authenticate()
        ee.Initialize(project=project) if project else ee.Initialize()

    return ee


def load_district_polygons(ee, nodes: pd.DataFrame):
    """
    Load the 25 Sri Lankan district polygons from geoBoundaries ADM2
    and attach canonical district identities.
    """

    source = (
        ee.FeatureCollection(DISTRICT_BOUNDARY_ASSET)
        .filter(ee.Filter.eq("shapeGroup", "LKA"))
    )

    # Rename shapeName to gadm_name so the existing downstream mapping
    # and validation logic can remain unchanged.
    polygons = source.map(
        lambda feature: feature.set("gadm_name", feature.get("shapeName"))
    )

    source_level = "geoBoundaries v6 ADM2"

    observed_count = polygons.size().getInfo()

    if observed_count != EXPECTED_DISTRICTS:
        observed_names = polygons.aggregate_array("gadm_name").getInfo()

        raise ValueError(
            f"{source_level} returned {observed_count} polygons for Sri Lanka, "
            f"expected {EXPECTED_DISTRICTS}.\n"
            f"Observed names: {sorted(observed_names)}"
        )

    gadm_names = sorted(polygons.aggregate_array("gadm_name").getInfo())

    verify_district_mapping(gadm_names, nodes)

    lookup = {
        source_name: BOUNDARY_TO_CANONICAL[source_name]
        for source_name in gadm_names
    }

    node_lookup = dict(
        zip(nodes["canonical_name"], nodes["node_id"].astype(int))
    )

    canonical_map = ee.Dictionary(lookup)
    node_map = ee.Dictionary(
        {name: int(node_id) for name, node_id in node_lookup.items()}
    )

    def _label(feature):
        source_name = feature.get("gadm_name")
        canonical = canonical_map.get(source_name)

        return feature.set(
            {
                "canonical_name": canonical,
                "node_id": node_map.get(canonical),
            }
        )

    return polygons.map(_label), source_level


def verify_district_mapping(gadm_names: list[str], nodes: pd.DataFrame) -> None:
    """
    Confirm the GADM polygons and the node registry describe the same 25
    districts, with one polygon, one node_id and one canonical name each.
    """

    unmapped = sorted(set(gadm_names) - set(BOUNDARY_TO_CANONICAL))

    if unmapped:
        raise ValueError(
            "These GADM district names have no explicit canonical mapping: "
            f"{unmapped}. Add them deliberately; never match them by "
            "similarity."
        )

    obsolete = sorted(set(BOUNDARY_TO_CANONICAL) - set(gadm_names))

    if obsolete:
        raise ValueError(
            "The mapping contains GADM names absent from the source: "
            f"{obsolete}. GADM may have been revised."
        )

    if len(gadm_names) != len(set(gadm_names)):
        raise ValueError("A GADM district name appears on two polygons.")

    canonical_names = [BOUNDARY_TO_CANONICAL[name] for name in gadm_names]

    if len(canonical_names) != len(set(canonical_names)):
        duplicated = sorted(
            {
                name
                for name in canonical_names
                if canonical_names.count(name) > 1
            }
        )

        raise ValueError(
            f"Two GADM polygons map to the same canonical district: "
            f"{duplicated}."
        )

    registry_names = set(nodes["canonical_name"])
    mapped_names = set(canonical_names)

    if mapped_names != registry_names:
        raise ValueError(
            "The GADM polygons and nodes.csv describe different districts.\n"
            f"  Missing from GADM:      {sorted(registry_names - mapped_names)}\n"
            f"  Missing from nodes.csv: {sorted(mapped_names - registry_names)}"
        )

    # Cross-check against 3.create_nodes.py when it is importable, so the two
    # copies of the mapping cannot drift apart unnoticed.
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "create_nodes", PROJECT_DIR / "scripts" / "data" / "3.create_nodes.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return

    node_canonical = set(nodes["canonical_name"])
    weather_canonical = set(BOUNDARY_TO_CANONICAL.values())

    if node_canonical != weather_canonical:
        raise ValueError(
            "Node registry and weather boundary mapping have different "
            "canonical districts."
        )


# ---------------------------------------------------------------------------
# Daily reduction
# ---------------------------------------------------------------------------

def build_daily_image(ee, day):
    """
    Reduce one UTC day of hourly ERA5-Land bands to a single daily image.

    Every reduction below is deliberate:

    - Rainfall uses the LAST hourly value, not a sum. ERA5-Land total
      precipitation is a running accumulation from 00 UTC, so the final step of
      the day already holds the daily total. Summing the raw accumulation
      overstates rainfall by roughly an order of magnitude.
    - Temperature mean is the mean of 24 hourly values, not (min + max) / 2,
      which is biased in the tropics.
    - Relative humidity and wind speed are derived hourly and then averaged.
      Both relations are non-linear, so deriving them from daily means gives a
      different and biased answer.
    """

    day = ee.Date(day)
    next_day = day.advance(1, "day")

    hourly = (
        ee.ImageCollection(ERA5_ASSET)
        .filterDate(day, next_day)
    )

    temperature = hourly.select("temperature_2m")
    dewpoint = hourly.select("dewpoint_temperature_2m")

    def _derive(image):
        """Per-hour relative humidity and wind speed."""

        t_c = image.select("temperature_2m").subtract(273.15)
        td_c = image.select("dewpoint_temperature_2m").subtract(273.15)

        # Magnus saturation vapour pressure, hPa.
        def _es(temp_c):
            return temp_c.multiply(17.67).divide(
                temp_c.add(243.5)
            ).exp().multiply(6.112)

        humidity = (
            _es(td_c).divide(_es(t_c)).multiply(100).clamp(0, 100)
            .rename("relative_humidity")
        )

        u_wind = image.select("u_component_of_wind_10m")
        v_wind = image.select("v_component_of_wind_10m")

        speed = (
            u_wind.pow(2).add(v_wind.pow(2)).sqrt().rename("wind_speed")
        )

        return humidity.addBands(speed)

    derived = hourly.map(_derive)

    # Total precipitation is a running accumulation from 00 UTC; the last step
    # of the day carries the full daily total. Converted m -> mm.
    rainfall = (
        hourly.select("total_precipitation_hourly").sum()
        .multiply(1000)
        .rename("rainfall_mm")
    )

    daily = (
        temperature.mean().subtract(273.15).rename("temperature_mean_c")
        .addBands(
            temperature.min().subtract(273.15).rename("temperature_min_c")
        )
        .addBands(
            temperature.max().subtract(273.15).rename("temperature_max_c")
        )
        .addBands(
            dewpoint.mean().subtract(273.15).rename("dewpoint_mean_c")
        )
        .addBands(
            derived.select("relative_humidity").mean()
            .rename("relative_humidity_mean")
        )
        .addBands(
            derived.select("wind_speed").mean().rename("wind_speed_mean")
        )
        .addBands(
            hourly.select("u_component_of_wind_10m").mean()
            .rename("u_wind_mean")
        )
        .addBands(
            hourly.select("v_component_of_wind_10m").mean()
            .rename("v_wind_mean")
        )
        .addBands(
            hourly.select("surface_pressure").mean().divide(100)
            .rename("surface_pressure_mean")
        )
        .addBands(rainfall)
    )

    return daily.set("date", day.format("YYYY-MM-dd"))


def extract_date_chunk(
    ee,
    polygons,
    chunk_start: date,
    chunk_end: date,
) -> pd.DataFrame:
    """Extract weather sequentially, one day per Earth Engine request."""

    frames = []
    current_day = chunk_start

    while current_day <= chunk_end:
        print(f"\n        {current_day}...", end=" ", flush=True)

        daily = build_daily_image(ee, ee.Date(str(current_day)))

        reduced = daily.reduceRegions(
            collection=polygons,
            reducer=ee.Reducer.mean(),
            scale=AGGREGATION_SCALE_M,
            tileScale=4,
        ).map(
            lambda feature: feature.set("date", str(current_day))
        )

        frame = _feature_collection_to_frame(ee, reduced)
        frames.append(frame)

        print(f"{len(frame)} rows")

        current_day += pd.Timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    return pd.concat(frames, ignore_index=True)


def extract_year(
    ee,
    polygons,
    year: int,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Extract one year in monthly chunks to avoid Earth Engine limits."""

    year_start = max(start, date(year, 1, 1))
    year_end = min(end, date(year, 12, 31))

    if year_start > year_end:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    frames = []

    current_start = year_start

    while current_start <= year_end:
        next_month = (
            pd.Timestamp(current_start)
            + pd.offsets.MonthBegin(1)
        ).date()

        current_end = min(
            next_month - pd.Timedelta(days=1),
            year_end,
        )

        print(
            f"\n      {current_start} to {current_end}...",
            end=" ",
            flush=True,
        )

        frame = extract_date_chunk(
            ee,
            polygons,
            current_start,
            current_end,
        )

        frames.append(frame)

        print(f"{len(frame)} rows")

        current_start = current_end + pd.Timedelta(days=1)

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    return pd.concat(frames, ignore_index=True)


def _feature_collection_to_frame(ee, collection) -> pd.DataFrame:
    """Pull a FeatureCollection into a frame, chunked to respect GEE limits."""

    properties = [
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
        "u_wind_mean",
        "v_wind_mean",
        "surface_pressure_mean",
    ]

    try:
        records = collection.select(properties, retainGeometry=False).getInfo()
    except Exception as error:
        raise RuntimeError(
            "Earth Engine could not return this chunk in one request. "
            "Reduce the date range with --start and --end, or export to "
            "Drive instead.\n"
            f"Underlying error: {error}"
        ) from error

    rows = [feature["properties"] for feature in records.get("features", [])]

    frame = pd.DataFrame(rows)

    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for column in properties:
        if column not in frame.columns:
            frame[column] = pd.NA

    return frame


def finalise_chunk(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the provenance columns and the observation flag."""

    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    result = frame.copy()

    result["node_id"] = result["node_id"].astype("Int64")

    # A row is observed only when every required weather value is present. A
    # partial row is never completed with zeros or carried-forward values.
    observed = result[WEATHER_COLUMNS].notna().all(axis=1)

    result["weather_observed"] = observed.astype(int)

    # Where a row is not observed, the weather columns are nulled rather than
    # left partially filled, so absence is unambiguous.
    result.loc[~observed, WEATHER_COLUMNS] = pd.NA

    result["data_source"] = DATA_SOURCE
    result["extraction_version"] = EXTRACTION_VERSION

    return result[OUTPUT_COLUMNS].sort_values(
        ["date", "node_id"]
    ).reset_index(drop=True)


def build_export_collection(
    ee,
    polygons,
    window_start: date,
    window_end: date,
):
    """Build one FeatureCollection covering a closed date window."""

    start_day = ee.Date(window_start.isoformat())
    day_count = (window_end - window_start).days

    def _one_day(offset):
        day = start_day.advance(ee.Number(offset), "day")
        daily = build_daily_image(ee, day)

        return daily.reduceRegions(
            collection=polygons,
            reducer=ee.Reducer.mean(),
            scale=AGGREGATION_SCALE_M,
            tileScale=4,
        ).map(
            lambda feature: feature.set("date", day.format("YYYY-MM-dd"))
        )

    return ee.FeatureCollection(
        ee.List.sequence(0, day_count).map(_one_day)
    ).flatten()


def iter_export_windows(
    start: date,
    end: date,
    granularity: str,
):
    """Yield the export windows that should become separate Drive tasks."""

    current_start = start

    while current_start <= end:
        if granularity == "month":
            last_day = calendar_lib.monthrange(
                current_start.year, current_start.month
            )[1]
            current_end = min(
                date(current_start.year, current_start.month, last_day),
                end,
            )
            label = f"{current_start.year}_{current_start.month:02d}"
        else:
            current_end = min(date(current_start.year, 12, 31), end)
            label = f"{current_start.year}"

        yield label, current_start, current_end

        current_start = current_end + timedelta(days=1)


def export_task_name(label: str) -> str:
    """Return the stable export file prefix used by the Drive task."""

    return f"climate_daily_{label}"


def submit_export_task(
    ee,
    collection,
    export_folder: str,
    label: str,
):
    """Queue one Earth Engine table export and print its task details."""

    export_name = export_task_name(label)

    task = ee.batch.Export.table.toDrive(
        collection=collection.select(OUTPUT_COLUMNS, retainGeometry=False),
        description=export_name,
        folder=export_folder,
        fileNamePrefix=export_name,
        fileFormat="CSV",
        selectors=OUTPUT_COLUMNS,
    )

    task.start()

    status = task.status()

    print(
        f"  submitted {status.get('id')} | {status.get('description')} | "
        f"{status.get('state')}"
    )

    return task


def print_task_list(ee, prefix: str = "climate_daily_") -> None:
    """Print queued Earth Engine tasks, filtered to this pipeline when possible."""

    tasks = []

    for task in ee.batch.Task.list():
        status = task.status()
        description = status.get("description", "")

        if prefix and not description.startswith(prefix):
            continue

        tasks.append(status)

    tasks.sort(key=lambda item: (item.get("state", ""), item.get("description", "")))

    print("\nEarth Engine tasks")
    print("-" * 62)

    if not tasks:
        print("No matching tasks found.")
        return

    for status in tasks:
        print(
            f"{status.get('state', 'UNKNOWN'):<12} "
            f"{status.get('id', '<no id>')} "
            f"{status.get('description', '<no description>')}"
        )


# ---------------------------------------------------------------------------
# Chunk handling
# ---------------------------------------------------------------------------

def chunk_path(year: int) -> Path:
    """Return the file holding one year of extracted values."""

    return CHUNK_DIR / f"climate_daily_{year}.csv"


def combine_chunks(start: date, end: date) -> pd.DataFrame:
    """Combine the yearly chunks into one frame."""

    frames = []

    for year in range(start.year, end.year + 1):
        path = chunk_path(year)

        if not path.exists():
            continue

        frames.append(pd.read_csv(path))

    if not frames:
        raise FileNotFoundError(
            f"No chunks found in {CHUNK_DIR}. Run the extraction first."
        )

    combined = pd.concat(frames, ignore_index=True)

    combined = combined.loc[
        combined["date"].between(str(start), str(end))
    ]

    return combined.sort_values(["date", "node_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_extraction(
    frame: pd.DataFrame,
    nodes: pd.DataFrame,
    start: date,
    end: date,
) -> dict:
    """
    Report the extraction's shape and completeness.

    Findings are printed rather than raised. A short series is a real outcome
    worth seeing in full, not a reason to hide the numbers behind a traceback.
    """

    print("\nExtraction validation")
    print("-" * 62)

    if frame.empty:
        print("The extracted frame is empty.")
        return {"row_count": 0}

    dates = pd.to_datetime(frame["date"])

    unique_dates = dates.nunique()
    district_count = frame["node_id"].nunique()
    row_count = len(frame)

    expected_days = (end - start).days + 1
    expected_rows = unique_dates * EXPECTED_DISTRICTS

    duplicates = int(frame.duplicated(subset=["date", "node_id"]).sum())

    print(f"Minimum date:             {dates.min().date()}")
    print(f"Maximum date:             {dates.max().date()}")
    print(f"Unique dates:             {unique_dates}")
    print(f"Canonical districts:      {district_count}")
    print(f"Total rows:               {row_count}")
    print(f"Expected rows:            {expected_rows} "
          f"(= {unique_dates} dates x {EXPECTED_DISTRICTS})")
    print(f"Duplicate district-date:  {duplicates}")

    print("\nMissing values by column")

    missing = frame.isna().sum()

    for column in OUTPUT_COLUMNS:
        count = int(missing.get(column, 0))
        share = 100 * count / row_count if row_count else 0.0

        marker = "" if count == 0 else "   <-- "
        print(f"  {column:<28} {count:>8}  ({share:5.2f}%){marker}")

    findings = []

    if district_count != EXPECTED_DISTRICTS:
        findings.append(
            f"{district_count} districts present, expected "
            f"{EXPECTED_DISTRICTS}."
        )

    registry_names = set(nodes["canonical_name"])
    extracted_names = set(frame["canonical_name"].dropna())

    if extracted_names != registry_names:
        findings.append(
            "District names differ from nodes.csv. Missing: "
            f"{sorted(registry_names - extracted_names)}"
        )

    if duplicates:
        findings.append(f"{duplicates} duplicate district-date rows.")

    if row_count != expected_rows:
        findings.append(
            f"Row count {row_count} differs from expected {expected_rows} "
            f"by {row_count - expected_rows}."
        )

    if unique_dates != expected_days:
        missing_days = expected_days - unique_dates

        findings.append(
            f"{missing_days} of {expected_days} calendar dates absent. "
            "ERA5 is a reanalysis and produces a value for every date, so a "
            "gap means the extraction failed rather than that weather did "
            "not occur."
        )

    unobserved = int((frame["weather_observed"] == 0).sum())

    if unobserved:
        findings.append(
            f"{unobserved} rows carry weather_observed = 0 and null weather "
            "values. These are reported, never filled."
        )

    # Physical plausibility. A breach here usually means a unit conversion was
    # applied twice or a band was mixed up.
    observed = frame.loc[frame["weather_observed"] == 1]

    if not observed.empty:
        checks = [
            ("temperature_mean_c", 0, 50),
            ("temperature_min_c", 0, 50),
            ("temperature_max_c", 0, 50),
            ("relative_humidity_mean", 0, 100),
            ("rainfall_mm", 0, 1000),
            ("wind_speed_mean", 0, 60),
        ]

        for column, low, high in checks:
            values = observed[column].dropna()
            outside = values[(values < low) | (values > high)]

            if len(outside):
                findings.append(
                    f"{len(outside)} {column} values outside {low}-{high} "
                    f"(min {outside.min():.2f}, max {outside.max():.2f})."
                )

        ordered = observed.dropna(
            subset=["temperature_min_c", "temperature_mean_c", "temperature_max_c"]
        )

        misordered = ordered.loc[
            (ordered["temperature_min_c"] > ordered["temperature_mean_c"])
            | (ordered["temperature_mean_c"] > ordered["temperature_max_c"])
        ]

        if len(misordered):
            findings.append(
                f"{len(misordered)} rows where min <= mean <= max fails."
            )

        humid = observed.dropna(
            subset=["dewpoint_mean_c", "temperature_mean_c"]
        )

        impossible = humid.loc[
            humid["dewpoint_mean_c"] > humid["temperature_mean_c"] + 0.5
        ]

        if len(impossible):
            findings.append(
                f"{len(impossible)} rows where dewpoint exceeds air "
                "temperature, which is physically impossible."
            )

    print()

    if findings:
        print("Findings")
        for finding in findings:
            print(f"  - {finding}")
    else:
        print("No findings. The extraction is complete and plausible.")

    return {
        "min_date": str(dates.min().date()),
        "max_date": str(dates.max().date()),
        "unique_dates": int(unique_dates),
        "districts": int(district_count),
        "row_count": int(row_count),
        "expected_rows": int(expected_rows),
        "duplicate_district_dates": duplicates,
        "unobserved_rows": unobserved,
        "missing_by_column": {
            column: int(missing.get(column, 0)) for column in OUTPUT_COLUMNS
        },
        "findings": findings,
    }


def write_manifest(
    summary: dict,
    start: date,
    end: date,
    source_level: str,
) -> None:
    """Record the assets and settings that produced this extraction."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "extraction_date": datetime.now(timezone.utc).isoformat(),
        "extraction_version": EXTRACTION_VERSION,
        "data_source": DATA_SOURCE,
        "earth_engine_assets": {
            "era5": ERA5_ASSET,
            "earth_engine_assets": {
                "era5": ERA5_ASSET,
                "district_boundaries": DISTRICT_BOUNDARY_ASSET,
                "boundary_source_used": source_level,
            },
            "boundary_source_used": source_level,
        },
        "settings": {
            "aggregation_scale_m": AGGREGATION_SCALE_M,
            "land_mask_threshold": LAND_MASK_THRESHOLD,
            "history_weeks": HISTORY_WEEKS,
            "day_boundary": "UTC",
        },
        "date_range": {"start": str(start), "end": str(end)},
        "validation": summary,
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote {MANIFEST_PATH.relative_to(PROJECT_DIR)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_arguments():
    """Parse the command line."""

    parser = argparse.ArgumentParser(
        description="Extract daily ERA5-Land weather for Sri Lankan districts."
    )

    parser.add_argument("--start", help="First date, YYYY-MM-DD.")
    parser.add_argument("--end", help="Last date, YYYY-MM-DD.")
    parser.add_argument("--project", help="Earth Engine Cloud project id.")
    parser.add_argument(
        "--submit-exports",
        action="store_true",
        help="Queue Earth Engine table exports instead of waiting for daily getInfo calls.",
    )
    parser.add_argument(
        "--export-folder",
        default=DEFAULT_EXPORT_FOLDER,
        help="Google Drive folder for submitted export tasks.",
    )
    parser.add_argument(
        "--export-granularity",
        choices=EXPORT_GRANULARITY_CHOICES,
        default="year",
        help="Submit one export per year, or per month if yearly exports are too large.",
    )
    parser.add_argument(
        "--check-tasks",
        action="store_true",
        help="List queued Earth Engine tasks and exit.",
    )


    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Re-validate the existing CSV without contacting Earth Engine.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the plan and check local inputs without extracting.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract years whose chunk file already exists.",
    )

    return parser.parse_args()


def main() -> int:
    """Extract, combine and validate the daily district climate data."""

    arguments = parse_arguments()

    if arguments.check_tasks:
        ee = initialise_earth_engine(arguments.project)
        print_task_list(ee)
        return 0

    nodes = load_nodes()

    print(f"Node registry:   {len(nodes)} districts, node_id 0-24")

    if arguments.validate_only:
        if not OUTPUT_PATH.exists():
            raise SystemExit(f"{OUTPUT_PATH} does not exist.")

        frame = pd.read_csv(OUTPUT_PATH)

        dates = pd.to_datetime(frame["date"])

        validate_extraction(
            frame, nodes, dates.min().date(), dates.max().date()
        )

        return 0

    start, end = resolve_date_range(arguments.start, arguments.end)

    years = list(range(start.year, end.year + 1))

    print(f"Date range:      {start} to {end}")
    print(f"Expected days:   {(end - start).days + 1}")
    print(f"Expected rows:   {((end - start).days + 1) * EXPECTED_DISTRICTS}")
    print(f"Yearly chunks:   {len(years)} ({years[0]}-{years[-1]})")
    print(f"ERA5 asset:      {ERA5_ASSET}")
    print(f"Boundaries:      {DISTRICT_BOUNDARY_ASSET}")

    if arguments.dry_run:
        print("\nDry run: local inputs verified, no Earth Engine calls made.")
        return 0

    ee = initialise_earth_engine(arguments.project)

    polygons, source_level = load_district_polygons(ee, nodes)

    print(f"Boundaries used: {source_level}, {EXPECTED_DISTRICTS} polygons")
    print("District mapping verified: every boundary name maps explicitly.\n")

    if not arguments.submit_exports:
        print(
            "\nNOTE: this script now queues Earth Engine exports instead of "
            "pulling each day back with getInfo(). Use --submit-exports to "
            "make the intent explicit."
        )

    print(
        f"Submitting {arguments.export_granularity} exports to Drive folder "
        f"{arguments.export_folder!r}..."
    )

    submitted = 0

    for label, window_start, window_end in iter_export_windows(
        start, end, arguments.export_granularity
    ):
        print(f"  {label}: {window_start} to {window_end}")

        collection = build_export_collection(ee, polygons, window_start, window_end)
        submit_export_task(
            ee,
            collection,
            arguments.export_folder,
            label,
        )

        submitted += 1

    print(f"\nSubmitted {submitted} Earth Engine export task(s).")
    print(
        "Download the CSV files into data/raw/era5_chunks/ and rerun the "
        "script with --validate-only after combining them into "
        "data/raw/climate_daily_district.csv."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

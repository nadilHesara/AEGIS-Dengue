"""
Extract daily CHIRPS rainfall for the 25 canonical Sri Lankan districts.

Optional. CHIRPS is a second, independent rainfall estimate used to
cross-check ERA5-Land. It is never a silent replacement: the ERA5 extraction
and the canonical dengue dataset are unaffected by whether this script runs.

CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data) is
gauge-calibrated and satellite-derived, at 0.05 degrees. That is roughly five
times finer than ERA5-Land and calibrated against rain gauges rather than
modelled by a reanalysis, so it is a genuinely independent view of the same
quantity. Where the two agree, confidence in the rainfall signal rises; where
they disagree, the disagreement itself is informative.

The district polygons, node ordering, date range and aggregation method are
taken from scripts/5.extract_era5_daily.py so that the two rainfall series
are directly comparable. Any difference between them is then a property of
the rainfall products, not of the extraction.

Authentication (once per machine):

    earthengine authenticate

Usage:

    python scripts/7.extract_chirps_daily.py
    python scripts/7.extract_chirps_daily.py --start 2010-01-01 --end 2010-12-31
    python scripts/7.extract_chirps_daily.py --dry-run

Outputs:
    data/raw/chirps_daily_rainfall.csv
    data/raw/chirps_chunks/chirps_daily_<year>.csv
    results/climate/era5_chirps/chirps_extraction_manifest.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
CHUNK_DIR = RAW_DIR / "chirps_chunks"
OUTPUT_PATH = RAW_DIR / "chirps_daily_rainfall.csv"

RESULTS_DIR = PROJECT_DIR / "results" / "climate" / "era5_chirps"
MANIFEST_PATH = RESULTS_DIR / "chirps_extraction_manifest.json"


# CHIRPS daily, version 2.0, final product. Native resolution 0.05 degrees.
# The band is already millimetres per day, so no unit conversion is applied;
# this is asserted rather than assumed, in check_units below.
CHIRPS_ASSET = "UCSB-CHG/CHIRPS/DAILY"
CHIRPS_BAND = "precipitation"
CHIRPS_NATIVE_UNITS = "mm/day"

EXTRACTION_VERSION = "chirps-v2.0-final-v1"
DATA_SOURCE = "CHIRPS Daily v2.0 (UCSB-CHG/CHIRPS/DAILY) via Google Earth Engine"

# CHIRPS begins in 1981, comfortably before the ERA5 window, so the shared
# date range is never truncated at the start.
CHIRPS_FIRST_DATE = date(1981, 1, 1)

# CHIRPS final has a longer latency than ERA5-Land: roughly 1-2 months after
# month end. The preliminary product is faster but is revised.
CHIRPS_LATENCY_DAYS = 60

OUTPUT_COLUMNS = [
    "date",
    "node_id",
    "canonical_name",
    "rainfall_mm_chirps",
    "rainfall_observed",
    "data_source",
    "extraction_version",
]


def _load_era5_module():
    """
    Load the ERA5 extraction module for its shared helpers.

    Reusing load_nodes, load_district_polygons and resolve_date_range is what
    guarantees the two rainfall series describe identical districts over an
    identical window. Restating them here would let the two drift apart, and
    a drifted comparison is worse than no comparison.
    """

    path = PROJECT_DIR / "scripts" / "5.extract_era5_daily.py"

    spec = importlib.util.spec_from_file_location("era5_extract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


era5 = _load_era5_module()

EXPECTED_DISTRICTS = era5.EXPECTED_DISTRICTS
AGGREGATION_SCALE_M = era5.AGGREGATION_SCALE_M


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def build_daily_rainfall_image(ee, day):
    """
    Return one day of CHIRPS rainfall.

    CHIRPS daily is a single image per day already expressed in millimetres
    per day, so no accumulation differencing and no unit conversion are
    needed. This is the key difference from ERA5-Land, whose total
    precipitation is a running accumulation that must be handled carefully.
    """

    day = ee.Date(day)

    daily = (
        ee.ImageCollection(CHIRPS_ASSET)
        .filterDate(day, day.advance(1, "day"))
        .select(CHIRPS_BAND)
        .first()
    )

    return ee.Image(daily).rename("rainfall_mm_chirps").set(
        "date", day.format("YYYY-MM-dd")
    )


def extract_year(ee, polygons, year: int, start: date, end: date):
    """
    Extract one year of daily district rainfall.

    The reduction runs at the same scale as the ERA5 extraction so that the
    two series are aggregated identically over identical polygons.
    """

    year_start = max(start, date(year, 1, 1))
    year_end = min(end, date(year, 12, 31))

    if year_start > year_end:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    day_count = (year_end - year_start).days + 1

    days = ee.List.sequence(0, day_count - 1).map(
        lambda offset: ee.Date(str(year_start)).advance(offset, "day")
    )

    def _reduce_day(day):
        image = build_daily_rainfall_image(ee, day)

        reduced = image.reduceRegions(
            collection=polygons,
            reducer=ee.Reducer.mean(),
            scale=AGGREGATION_SCALE_M,
        )

        return reduced.map(
            lambda feature: feature.set(
                "date", ee.Date(day).format("YYYY-MM-dd")
            )
        )

    collection = ee.FeatureCollection(days.map(_reduce_day)).flatten()

    properties = ["date", "node_id", "canonical_name", "mean"]

    try:
        records = collection.select(properties, retainGeometry=False).getInfo()
    except Exception as error:
        raise RuntimeError(
            "Earth Engine could not return this chunk in one request. "
            "Narrow the range with --start and --end.\n"
            f"Underlying error: {error}"
        ) from error

    rows = [feature["properties"] for feature in records.get("features", [])]

    frame = pd.DataFrame(rows)

    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # reduceRegions names the single-band result "mean".
    if "mean" in frame.columns:
        frame = frame.rename(columns={"mean": "rainfall_mm_chirps"})

    return frame


def finalise_chunk(frame: pd.DataFrame) -> pd.DataFrame:
    """Add provenance and the observation flag."""

    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    result = frame.copy()

    if "rainfall_mm_chirps" not in result.columns:
        result["rainfall_mm_chirps"] = pd.NA

    result["node_id"] = pd.to_numeric(
        result["node_id"], errors="coerce"
    ).astype("Int64")

    observed = result["rainfall_mm_chirps"].notna()

    result["rainfall_observed"] = observed.astype(int)

    # Absence stays absence. A missing rainfall estimate is never zero: zero
    # asserts a dry day that was not observed.
    result.loc[~observed, "rainfall_mm_chirps"] = pd.NA

    result["data_source"] = DATA_SOURCE
    result["extraction_version"] = EXTRACTION_VERSION

    return result[OUTPUT_COLUMNS].sort_values(
        ["date", "node_id"]
    ).reset_index(drop=True)


def chunk_path(year: int) -> Path:
    """Return the file holding one year of extracted rainfall."""

    return CHUNK_DIR / f"chirps_daily_{year}.csv"


def combine_chunks(start: date, end: date) -> pd.DataFrame:
    """Combine the yearly chunks into one frame."""

    frames = []

    for year in range(start.year, end.year + 1):
        path = chunk_path(year)

        if path.exists():
            frames.append(pd.read_csv(path))

    if not frames:
        raise FileNotFoundError(
            f"No chunks found in {CHUNK_DIR}. Run the extraction first."
        )

    combined = pd.concat(frames, ignore_index=True)

    combined = combined.loc[combined["date"].between(str(start), str(end))]

    return combined.sort_values(["date", "node_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def check_units(frame: pd.DataFrame) -> list[str]:
    """
    Confirm the rainfall values are millimetres per day.

    CHIRPS is published in mm/day, but that is asserted here rather than
    trusted. A maximum below 1 would indicate metres; a median in the
    hundreds would indicate an accumulation rather than a daily rate.
    """

    findings = []

    values = pd.to_numeric(
        frame["rainfall_mm_chirps"], errors="coerce"
    ).dropna()

    if values.empty:
        return ["no rainfall values to check units against"]

    if values.max() < 1.0:
        findings.append(
            f"maximum rainfall {values.max():.6f} is below 1 mm, which "
            "suggests metres rather than millimetres"
        )

    if values.median() > 100:
        findings.append(
            f"median rainfall {values.median():.2f} mm/day is implausibly "
            "high, which suggests an accumulation rather than a daily rate"
        )

    if (values < 0).any():
        findings.append(f"{int((values < 0).sum())} negative rainfall values")

    return findings


def validate_extraction(
    frame: pd.DataFrame,
    nodes: pd.DataFrame,
    start: date,
    end: date,
) -> dict:
    """Report district-date coverage and unit findings."""

    print("\nCHIRPS extraction validation")
    print("-" * 62)

    if frame.empty:
        print("The extracted frame is empty.")
        return {"row_count": 0}

    dates = pd.to_datetime(frame["date"])

    unique_dates = dates.nunique()
    district_count = frame["canonical_name"].nunique()
    row_count = len(frame)

    expected_days = (end - start).days + 1
    expected_rows = unique_dates * EXPECTED_DISTRICTS

    duplicates = int(frame.duplicated(subset=["date", "canonical_name"]).sum())

    print(f"Minimum date:             {dates.min().date()}")
    print(f"Maximum date:             {dates.max().date()}")
    print(f"Unique dates:             {unique_dates} of {expected_days}")
    print(f"Canonical districts:      {district_count}")
    print(f"Total rows:               {row_count}")
    print(f"Expected rows:            {expected_rows}")
    print(f"Duplicate district-date:  {duplicates}")

    null_rainfall = int(frame["rainfall_mm_chirps"].isna().sum())

    print(f"Null rainfall values:     {null_rainfall}")

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
            f"Row count {row_count} differs from expected {expected_rows}."
        )

    if unique_dates != expected_days:
        findings.append(
            f"{expected_days - unique_dates} of {expected_days} dates absent."
        )

    findings.extend(check_units(frame))

    print()

    if findings:
        print("Findings")
        for finding in findings:
            print(f"  - {finding}")
    else:
        print("No findings. Coverage is complete and units are millimetres.")

    return {
        "min_date": str(dates.min().date()),
        "max_date": str(dates.max().date()),
        "unique_dates": int(unique_dates),
        "districts": int(district_count),
        "row_count": int(row_count),
        "expected_rows": int(expected_rows),
        "duplicate_district_dates": duplicates,
        "null_rainfall": null_rainfall,
        "findings": findings,
    }


def write_manifest(summary: dict, start: date, end: date) -> None:
    """Record the assets and settings that produced this extraction."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "extraction_date": datetime.now(timezone.utc).isoformat(),
        "extraction_version": EXTRACTION_VERSION,
        "data_source": DATA_SOURCE,
        "earth_engine_assets": {
            "chirps": CHIRPS_ASSET,
            "chirps_band": CHIRPS_BAND,
            "gadm_level1": era5.GADM_LEVEL1_ASSET,
        },
        "settings": {
            "native_units": CHIRPS_NATIVE_UNITS,
            "unit_conversion_applied": "none",
            "aggregation_scale_m": AGGREGATION_SCALE_M,
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
        description="Extract daily CHIRPS rainfall for Sri Lankan districts."
    )

    parser.add_argument("--start", help="First date, YYYY-MM-DD.")
    parser.add_argument("--end", help="Last date, YYYY-MM-DD.")
    parser.add_argument("--project", help="Earth Engine Cloud project id.")

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
    """Extract, combine and validate the daily district CHIRPS rainfall."""

    arguments = parse_arguments()

    nodes = era5.load_nodes()

    print(f"Node registry:   {len(nodes)} districts, node_id 0-24")

    start, end = era5.resolve_date_range(arguments.start, arguments.end)

    # CHIRPS begins in 1981 and trails real time more than ERA5-Land does.
    if start < CHIRPS_FIRST_DATE:
        print(
            f"NOTE: CHIRPS begins {CHIRPS_FIRST_DATE}; "
            f"start moved from {start}."
        )
        start = CHIRPS_FIRST_DATE

    chirps_latest = (
        pd.Timestamp.now(tz="UTC").tz_localize(None)
        - pd.Timedelta(days=CHIRPS_LATENCY_DAYS)
    ).date()

    if end > chirps_latest:
        print(
            f"NOTE: CHIRPS final trails roughly {CHIRPS_LATENCY_DAYS} days; "
            f"end moved from {end} to {chirps_latest}."
        )
        end = chirps_latest

    years = list(range(start.year, end.year + 1))

    print(f"Date range:      {start} to {end}")
    print(f"Expected rows:   {((end - start).days + 1) * EXPECTED_DISTRICTS}")
    print(f"Yearly chunks:   {len(years)}")
    print(f"CHIRPS asset:    {CHIRPS_ASSET}")
    print(f"Native units:    {CHIRPS_NATIVE_UNITS} (no conversion applied)")

    if arguments.dry_run:
        print("\nDry run: local inputs verified, no Earth Engine calls made.")
        return 0

    ee = era5.initialise_earth_engine(arguments.project)

    polygons, source_level = era5.load_district_polygons(ee, nodes)

    print(f"Boundaries:      {source_level}, {EXPECTED_DISTRICTS} polygons")
    print("Using the same polygons and scale as the ERA5 extraction.\n")

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    for year in years:
        path = chunk_path(year)

        if path.exists() and not arguments.overwrite:
            print(f"  {year}: already extracted, skipping")
            continue

        print(f"  {year}: extracting...", end=" ", flush=True)

        frame = finalise_chunk(extract_year(ee, polygons, year, start, end))

        frame.to_csv(path, index=False)

        print(f"{len(frame)} rows")

    combined = combine_chunks(start, end)

    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {OUTPUT_PATH.relative_to(PROJECT_DIR)}")

    summary = validate_extraction(combined, nodes, start, end)

    write_manifest(summary, start, end)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Extract daily CHIRPS rainfall for the 25 canonical Sri Lankan districts.

CHIRPS is a second, independent rainfall estimate used to cross-check
ERA5-Land. The district polygons, node registry, date range and aggregation
scale are shared with scripts/5.extract_era5_daily.py so the two rainfall
series stay directly comparable.

This version avoids synchronous yearly getInfo() calls. Instead it queues one
Earth Engine table export per year, or per month when monthly chunking is
requested, and leaves the CSV download/combination step to the user.

Outputs:
    data/raw/chirps_daily_rainfall.csv
    data/raw/chirps_chunks/chirps_daily_<year>.csv
    results/climate/era5_chirps/chirps_extraction_manifest.json
"""

from __future__ import annotations

import argparse
import calendar as calendar_lib
import importlib.util
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
CHUNK_DIR = RAW_DIR / "chirps_chunks"
OUTPUT_PATH = RAW_DIR / "chirps_daily_rainfall.csv"

RESULTS_DIR = PROJECT_DIR / "results" / "climate" / "era5_chirps"
MANIFEST_PATH = RESULTS_DIR / "chirps_extraction_manifest.json"


CHIRPS_ASSET = "UCSB-CHG/CHIRPS/DAILY"
CHIRPS_BAND = "precipitation"
CHIRPS_NATIVE_UNITS = "mm/day"

EXTRACTION_VERSION = "chirps-v2.0-final-v1"
DATA_SOURCE = "CHIRPS Daily v2.0 (UCSB-CHG/CHIRPS/DAILY) via Google Earth Engine"

CHIRPS_FIRST_DATE = date(1981, 1, 1)
CHIRPS_LATENCY_DAYS = 60
DEFAULT_EXPORT_FOLDER = "chirps_chunks"
EXPORT_GRANULARITY_CHOICES = ("year", "month")

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
    """Load the ERA5 module for shared nodes, polygons and date logic."""

    path = PROJECT_DIR / "scripts" / "5.extract_era5_daily.py"

    spec = importlib.util.spec_from_file_location("era5_extract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


era5 = _load_era5_module()

EXPECTED_DISTRICTS = era5.EXPECTED_DISTRICTS
AGGREGATION_SCALE_M = era5.AGGREGATION_SCALE_M


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def build_daily_rainfall_image(ee, day):
    """Return one day of CHIRPS rainfall."""

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


def build_export_collection(ee, polygons, window_start: date, window_end: date):
    """Build one FeatureCollection covering a closed date window."""

    start_day = ee.Date(window_start.isoformat())
    day_count = (window_end - window_start).days

    def _one_day(offset):
        day = start_day.advance(ee.Number(offset), "day")
        image = build_daily_rainfall_image(ee, day)

        reduced = image.reduceRegions(
            collection=polygons,
            reducer=ee.Reducer.mean(),
            scale=AGGREGATION_SCALE_M,
        )

        return reduced.map(
            lambda feature: feature.set(
                {
                    "date": day.format("YYYY-MM-dd"),
                    "rainfall_mm_chirps": feature.get("mean"),
                    "rainfall_observed": 1,
                    "data_source": DATA_SOURCE,
                    "extraction_version": EXTRACTION_VERSION,
                }
            )
        )

    return ee.FeatureCollection(
        ee.List.sequence(0, day_count).map(_one_day)
    ).flatten()


def iter_export_windows(start: date, end: date, granularity: str):
    """Yield the export windows that should become separate Drive tasks."""

    current_start = start

    while current_start <= end:
        if granularity == "month":
            last_day = calendar_lib.monthrange(
                current_start.year, current_start.month
            )[1]
            current_end = min(
                date(current_start.year, current_start.month, last_day), end
            )
            label = f"{current_start.year}_{current_start.month:02d}"
        else:
            current_end = min(date(current_start.year, 12, 31), end)
            label = f"{current_start.year}"

        yield label, current_start, current_end

        current_start = current_end + timedelta(days=1)


def export_task_name(label: str) -> str:
    """Return the stable export file prefix used by the Drive task."""

    return f"chirps_daily_{label}"


def submit_export_task(ee, collection, export_folder: str, label: str):
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


def print_task_list(ee, prefix: str = "chirps_daily_") -> None:
    """Print queued Earth Engine tasks, filtered to this pipeline when possible."""

    tasks = []

    for task in ee.batch.Task.list():
        status = task.status()
        description = status.get("description", "")

        if prefix and not description.startswith(prefix):
            continue

        tasks.append(status)

    tasks.sort(
        key=lambda item: (item.get("state", ""), item.get("description", ""))
    )

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


def combine_chunks(start: date, end: date) -> pd.DataFrame:
    """Combine downloaded CHIRPS chunks into one frame."""

    frames = []

    for path in sorted(CHUNK_DIR.glob("chirps_daily_*.csv")):
        frames.append(pd.read_csv(path))

    if not frames:
        raise FileNotFoundError(
            f"No chunks found in {CHUNK_DIR}. Run the extraction first."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")

    combined = combined.loc[
        combined["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    ]

    return combined.sort_values(["date", "node_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def check_units(frame: pd.DataFrame) -> list[str]:
    """Confirm the rainfall values are millimetres per day."""

    findings = []

    values = pd.to_numeric(frame["rainfall_mm_chirps"], errors="coerce").dropna()

    if values.empty:
        return ["no rainfall values to check units against"]

    if values.max() < 1.0:
        findings.append(
            f"maximum rainfall {values.max():.6f} is below 1 mm, which suggests metres rather than millimetres"
        )

    if values.median() > 100:
        findings.append(
            f"median rainfall {values.median():.2f} mm/day is implausibly high, which suggests an accumulation rather than a daily rate"
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
            f"{district_count} districts present, expected {EXPECTED_DISTRICTS}."
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


def write_manifest(summary: dict, start: date, end: date, source_level: str) -> None:
    """Record the assets and settings that produced this extraction."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "extraction_date": datetime.now(timezone.utc).isoformat(),
        "extraction_version": EXTRACTION_VERSION,
        "data_source": DATA_SOURCE,
        "earth_engine_assets": {
            "chirps": CHIRPS_ASSET,
            "chirps_band": CHIRPS_BAND,
            "district_boundaries": era5.DISTRICT_BOUNDARY_ASSET,
            "boundary_source_used": source_level,
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
        "--dry-run",
        action="store_true",
        help="Report the plan and check local inputs without extracting.",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Re-validate the existing combined CSV without contacting Earth Engine.",
    )

    return parser.parse_args()


def main() -> int:
    """Queue, combine and validate the daily district CHIRPS rainfall."""

    arguments = parse_arguments()

    if arguments.check_tasks:
        ee = era5.initialise_earth_engine(arguments.project)
        print_task_list(ee)
        return 0

    nodes = era5.load_nodes()

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

    start, end = era5.resolve_date_range(arguments.start, arguments.end)

    if start < CHIRPS_FIRST_DATE:
        print(
            f"NOTE: CHIRPS begins {CHIRPS_FIRST_DATE}; start moved from {start}."
        )
        start = CHIRPS_FIRST_DATE

    chirps_latest = (
        pd.Timestamp.now(tz="UTC").tz_localize(None)
        - pd.Timedelta(days=CHIRPS_LATENCY_DAYS)
    ).date()

    if end > chirps_latest:
        print(
            f"NOTE: CHIRPS final trails roughly {CHIRPS_LATENCY_DAYS} days; end moved from {end} to {chirps_latest}."
        )
        end = chirps_latest

    export_windows = list(
        iter_export_windows(start, end, arguments.export_granularity)
    )

    print(f"Date range:      {start} to {end}")
    print(f"Expected rows:   {((end - start).days + 1) * EXPECTED_DISTRICTS}")
    print(
        f"Export windows:  {len(export_windows)} ({arguments.export_granularity})"
    )
    print(f"CHIRPS asset:    {CHIRPS_ASSET}")
    print(f"Native units:    {CHIRPS_NATIVE_UNITS} (no conversion applied)")

    if arguments.dry_run:
        print("\nDry run: local inputs verified, no Earth Engine calls made.")
        return 0

    ee = era5.initialise_earth_engine(arguments.project)

    polygons, source_level = era5.load_district_polygons(ee, nodes)

    print(f"Boundaries:      {source_level}, {EXPECTED_DISTRICTS} polygons")
    print("Using the same polygons and scale as the ERA5 extraction.\n")

    if not arguments.submit_exports:
        print(
            "\nNOTE: this script now queues Earth Engine exports instead of pulling each day back with getInfo(). Use --submit-exports to make the intent explicit."
        )

    print(
        f"Submitting {arguments.export_granularity} exports to Drive folder {arguments.export_folder!r}..."
    )

    submitted = 0

    for label, window_start, window_end in export_windows:
        print(f"  {label}: {window_start} to {window_end}")
        collection = build_export_collection(
            ee, polygons, window_start, window_end
        )
        submit_export_task(
            ee,
            collection,
            arguments.export_folder,
            label,
        )
        submitted += 1

    print(f"\nSubmitted {submitted} Earth Engine export task(s).")
    print(
        "Download the CSV files into data/raw/chirps_chunks/ and rerun the script with --validate-only after combining them into data/raw/chirps_daily_rainfall.csv."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

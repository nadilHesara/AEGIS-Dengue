"""
Reconstruct the daily district weather file from Open-Meteo's ERA5 archive.

Fallback for scripts 5/5b when the Google Earth Engine exports are not
available locally. Writes the same file, with the same columns, that
5b.combine_era5_chunks.py writes, so every downstream stage (9, 10, 12, ...)
runs unchanged:

    data/raw/climate_daily_district.csv

What is the same as the GEE extraction
    - Reanalysis family: `era5_seamless` serves ERA5-Land (0.1 deg) for
      temperature, dew point and relative humidity, and ERA5 (0.25 deg) for
      precipitation and wind. ERA5-Land's own precipitation is ERA5's,
      downscaled, so rainfall is the same product at coarser grid spacing.
    - UTC days (timezone=GMT), as script 5 reduces UTC days.
    - Units: mm, deg C, %, m/s.
    - Relative humidity and wind speed are daily means of hourly derived
      values (Open-Meteo aggregates hourly), matching script 5's rule.

What differs, and must be stated wherever results built on this file appear
    - Spatial support: one point per district (the GADM polygon centroid from
      nodes.csv) rather than the polygon mean script 5 computes with
      reduceRegions. Large districts (Anuradhapura, Ampara) are less well
      represented by one point than by their mean.
    - u/v wind components and surface pressure are not fetched; those columns
      are written empty. No model tensor uses them.

The file records data_source = "open-meteo:era5_seamless:centroid" on every
row so a result can never be silently attributed to the GEE extraction.

Resumable: districts already present in the per-district cache are skipped,
and HTTP 429 (quota) backs off and retries.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
NODES_PATH = PROJECT_DIR / "data" / "processed" / "nodes.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "raw" / "climate_daily_district.csv"
CACHE_DIR = PROJECT_DIR / "data" / "raw" / "open_meteo_cache"

API_URL = "https://archive-api.open-meteo.com/v1/archive"
MODEL = "era5_seamless"
START_DATE = "2006-11-01"
END_DATE = "2026-05-31"
DATA_SOURCE = f"open-meteo:{MODEL}:centroid"
EXTRACTION_VERSION = "open-meteo-v1"

# Open-Meteo daily variable -> script 5 column.
VARIABLES = {
    "precipitation_sum": "rainfall_mm",
    "temperature_2m_mean": "temperature_mean_c",
    "temperature_2m_min": "temperature_min_c",
    "temperature_2m_max": "temperature_max_c",
    "dew_point_2m_mean": "dewpoint_mean_c",
    "relative_humidity_2m_mean": "relative_humidity_mean",
    "wind_speed_10m_mean": "wind_speed_mean",
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


def request_url(lat: float, lon: float) -> str:
    query = {
        "latitude": f"{lat:.5f}",
        "longitude": f"{lon:.5f}",
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(VARIABLES),
        "timezone": "GMT",
        "wind_speed_unit": "ms",
        "models": MODEL,
    }
    return API_URL + "?" + "&".join(f"{k}={v}" for k, v in query.items())


def fetch_json(url: str, max_wait_s: float = 3900.0) -> dict:
    """GET with exponential back-off on 429 and transient errors."""

    wait, waited = 30.0, 0.0
    while True:
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")[:200]
            if error.code != 429 and error.code < 500:
                raise RuntimeError(f"HTTP {error.code}: {body}") from error
            reason = f"HTTP {error.code}: {body}"
        except (urllib.error.URLError, TimeoutError) as error:
            reason = str(error)
        if waited >= max_wait_s:
            raise RuntimeError(f"gave up after {waited:.0f}s: {reason}")
        print(f"  retry in {wait:.0f}s ({reason})", flush=True)
        time.sleep(wait)
        waited += wait
        wait = min(wait * 2, 900.0)


def fetch_district(node: pd.Series) -> pd.DataFrame:
    payload = fetch_json(request_url(node["centroid_lat"], node["centroid_lon"]))
    daily = payload["daily"]
    frame = pd.DataFrame({"date": daily["time"]})
    for source, target in VARIABLES.items():
        frame[target] = pd.to_numeric(pd.Series(daily[source]), errors="coerce")
    frame["node_id"] = int(node["node_id"])
    frame["canonical_name"] = node["canonical_name"]
    frame["grid_latitude"] = payload.get("latitude")
    frame["grid_longitude"] = payload.get("longitude")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pause", type=float, default=4.0,
                        help="Seconds between districts (quota courtesy).")
    arguments = parser.parse_args()

    nodes = pd.read_csv(NODES_PATH)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for _, node in nodes.iterrows():
        cache = CACHE_DIR / f"node_{int(node['node_id']):02d}.csv"
        if cache.exists():
            continue
        print(f"Fetching {node['canonical_name']} "
              f"({node['centroid_lat']:.3f}, {node['centroid_lon']:.3f})", flush=True)
        fetch_district(node).to_csv(cache, index=False)
        time.sleep(arguments.pause)

    frames = [pd.read_csv(path) for path in sorted(CACHE_DIR.glob("node_*.csv"))]
    climate = pd.concat(frames, ignore_index=True)
    if climate["node_id"].nunique() != len(nodes):
        raise RuntimeError(
            f"only {climate['node_id'].nunique()} of {len(nodes)} districts cached"
        )

    weather = list(VARIABLES.values())
    climate["weather_observed"] = climate[weather].notna().all(axis=1)
    for column in ("u_wind_mean", "v_wind_mean", "surface_pressure_mean"):
        climate[column] = np.nan
    climate["data_source"] = DATA_SOURCE
    climate["extraction_version"] = EXTRACTION_VERSION

    climate = climate[OUTPUT_COLUMNS].sort_values(["date", "node_id"])
    climate.to_csv(OUTPUT_PATH, index=False)

    missing = int((~climate["weather_observed"]).sum())
    print(f"Wrote {OUTPUT_PATH.relative_to(PROJECT_DIR)}: {len(climate):,} rows, "
          f"{climate['date'].min()} to {climate['date'].max()}, "
          f"{missing} incomplete district-days, source {DATA_SOURCE}")


if __name__ == "__main__":
    main()

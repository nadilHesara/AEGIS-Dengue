"""
Freeze the benchmark's data: SHA-256 hashes and provenance of every input.

Writes results/benchmark/data_manifest.json (committed). A benchmark number is
reproducible only on the files listed there; scripts/34 prints the climate
hash it scored, so a result can never be attributed to the wrong extraction.

What is recorded
    - hashes and sizes of the raw dengue CSV, the daily climate CSV, the
      reporting calendar, the canonical dengue table, the master panel, the
      tensors (v0-v4), folds.json, nodes.csv and adjacency.npz;
    - the climate extraction settings (source, reanalysis model, spatial
      support, date range, per-district grid point actually served) and the
      date the file was fetched;
    - the known availability caveat: reanalysis values are final values, not
      what an operational system would have had at each origin.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT_DIR / "results" / "benchmark" / "data_manifest.json"

FILES = {
    "dengue_raw": "data/raw/srilanka_weekly_data.csv",
    "climate_daily": "data/raw/climate_daily_district.csv",
    "reporting_calendar": "data/interim/reporting_calendar.csv",
    "dengue_canonical": "data/interim/dengue_weekly_canonical.parquet",
    "nodes": "data/processed/nodes.csv",
    "adjacency": "data/processed/adjacency.npz",
    "folds": "data/processed/folds.json",
    **{f"tensors_{v}": f"data/processed/model_tensors_{v}.npz" for v in ("v0", "v1", "v2", "v3", "v4")},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    sys.path.insert(0, str(PROJECT_DIR / "scripts" / "data"))
    manifest: dict = {"written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      "files": {}}
    for key, relative in FILES.items():
        path = PROJECT_DIR / relative
        if path.exists():
            manifest["files"][key] = {"path": relative, "sha256": sha256(path),
                                      "bytes": path.stat().st_size,
                                      "modified_utc": datetime.fromtimestamp(
                                          path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")}
        else:
            manifest["files"][key] = {"path": relative, "missing": True}

    climate_path = PROJECT_DIR / FILES["climate_daily"]
    if climate_path.exists():
        climate = pd.read_csv(climate_path, usecols=["date", "node_id", "data_source", "extraction_version"])
        cache = sorted((PROJECT_DIR / "data" / "raw" / "open_meteo_cache").glob("node_*.csv"))
        grid = []
        for path in cache:
            head = pd.read_csv(path, nrows=1)
            grid.append({"node_id": int(head["node_id"].iloc[0]),
                         "grid_latitude": float(head["grid_latitude"].iloc[0]),
                         "grid_longitude": float(head["grid_longitude"].iloc[0]),
                         "fetched_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                         .isoformat(timespec="seconds")})
        manifest["climate"] = {
            "data_source": sorted(climate["data_source"].unique().tolist()),
            "extraction_version": sorted(climate["extraction_version"].unique().tolist()),
            "api": "https://archive-api.open-meteo.com/v1/archive",
            "model": "era5_seamless (ERA5-Land: temperature, dew point, humidity; ERA5: precipitation, wind)",
            "spatial_support": "one point per district: the GADM 4.1 polygon centroid (nodes.csv)",
            "day_boundary": "UTC",
            "first_date": str(climate["date"].min()), "last_date": str(climate["date"].max()),
            "districts": int(climate["node_id"].nunique()),
            "grid_points": grid,
            "availability_caveat": (
                "Values are final reanalysis values retrieved in September 2026. Open-Meteo's "
                "historical API documents a delay before recent days are available, and ERA5 is "
                "revised from its preliminary ERA5T release; an operational system would have seen "
                "neither the final values nor the most recent days at each forecast origin. "
                "nb_shared_v2_wxlag1 tests a one-week delay; case-reporting delays are not modelled."),
        }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

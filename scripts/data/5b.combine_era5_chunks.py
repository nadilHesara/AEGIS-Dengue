from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
CHUNK_DIR = PROJECT_DIR / "data" / "raw" / "era5_chunks"
OUTPUT_PATH = PROJECT_DIR / "data" / "raw" / "climate_daily_district.csv"



EXPECTED_COLUMNS = [
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


def main() -> None:
    files = sorted(CHUNK_DIR.glob("climate_daily_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No climate_daily_*.csv files found in {CHUNK_DIR}"
        )

    print(f"Found {len(files)} ERA5 chunk files.")

    frames = []


    for path in files:
        frame = pd.read_csv(path)
        weather_columns = [
            "rainfall_mm",
            "temperature_mean_c",
            "temperature_min_c",
            "temperature_max_c",
            "dewpoint_mean_c",
            "relative_humidity_mean",
            "wind_speed_mean",
        ]

        frame["weather_observed"] = (
            frame[weather_columns]
            .notna()
            .all(axis=1)
            .astype(int)
        )

        frame["data_source"] = (
            "ERA5-Land hourly (ECMWF/ERA5_LAND/HOURLY) "
            "via Google Earth Engine"
        )

        frame["extraction_version"] = "era5land-v1"

        # Earth Engine exports may include these extra columns.
        frame = frame.drop(
            columns=["system:index", ".geo"],
            errors="ignore",
        )
        frame["rainfall_mm"] = pd.to_numeric(
            frame["rainfall_mm"],
            errors="coerce",
        ).clip(lower=0)

        missing_columns = [
            column for column in EXPECTED_COLUMNS
            if column not in frame.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{path.name} is missing columns: {missing_columns}"
            )

        frame = frame[EXPECTED_COLUMNS].copy()
        frames.append(frame)

        print(f"  {path.name}: {len(frame)} rows")

    combined = pd.concat(frames, ignore_index=True)

    combined["date"] = pd.to_datetime(
        combined["date"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")

    combined["node_id"] = pd.to_numeric(
        combined["node_id"],
        errors="raise",
    ).astype(int)

    duplicate_count = combined.duplicated(
        subset=["date", "node_id"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate district-date rows."
        )

    combined = combined.sort_values(
        ["date", "node_id"]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print("\nCombined ERA5 dataset")
    print("-" * 50)
    print(f"Rows:       {len(combined)}")
    print(f"Dates:      {combined['date'].nunique()}")
    print(f"Districts:  {combined['node_id'].nunique()}")
    print(f"Start:      {combined['date'].min()}")
    print(f"End:        {combined['date'].max()}")
    print(f"Wrote:      {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

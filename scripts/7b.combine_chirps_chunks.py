from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
CHUNK_DIR = PROJECT_DIR / "data" / "raw" / "chirps_chunks"
OUTPUT_PATH = PROJECT_DIR / "data" / "raw" / "chirps_daily_rainfall.csv"


def main() -> None:
    files = sorted(CHUNK_DIR.glob("chirps_daily_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No chirps_daily_*.csv files found in {CHUNK_DIR}"
        )

    print(f"Found {len(files)} CHIRPS chunk files.")

    frames = []

    for path in files:
        frame = pd.read_csv(path)

        frame = frame.drop(
            columns=["system:index", ".geo"],
            errors="ignore",
        )

        required = [
            "date",
            "node_id",
            "canonical_name",
            "rainfall_mm_chirps",
        ]

        missing = [
            column
            for column in required
            if column not in frame.columns
        ]

        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {missing}"
            )

        frame["date"] = pd.to_datetime(
            frame["date"],
            errors="raise",
        ).dt.strftime("%Y-%m-%d")

        frame["node_id"] = pd.to_numeric(
            frame["node_id"],
            errors="raise",
        ).astype(int)

        frame["rainfall_mm_chirps"] = pd.to_numeric(
            frame["rainfall_mm_chirps"],
            errors="coerce",
        ).clip(lower=0)

        frames.append(frame)

        print(
            f"  {path.name}: "
            f"{frame['date'].min()} to {frame['date'].max()} "
            f"({len(frame)} rows)"
        )

    combined = pd.concat(frames, ignore_index=True)

    duplicate_count = int(
        combined.duplicated(
            subset=["date", "node_id"]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate district-date rows."
        )

    combined = combined.sort_values(
        ["date", "node_id"]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print("\nCombined CHIRPS dataset")
    print("-" * 50)
    print(f"Rows:       {len(combined)}")
    print(f"Dates:      {combined['date'].nunique()}")
    print(f"Districts:  {combined['node_id'].nunique()}")
    print(f"Start:      {combined['date'].min()}")
    print(f"End:        {combined['date'].max()}")
    print(f"Wrote:      {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
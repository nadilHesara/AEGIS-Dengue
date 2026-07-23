from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def get_csv_files() -> list[Path]:
    """Return all CSV files inside the data directory and subdirectories."""
    if not DATA_DIR.exists():
        return []

    return list(DATA_DIR.rglob("*.csv"))


def test_data_directory_exists():
    """Check that the data directory exists."""
    assert DATA_DIR.exists(), (
        f"Data directory does not exist: {DATA_DIR}"
    )


def test_csv_files_exist():
    """Check that at least one CSV file exists."""
    csv_files = get_csv_files()

    assert csv_files, (
        f"No CSV files found inside: {DATA_DIR}"
    )


def test_csv_files_can_be_read():
    """Check that every CSV file can be read by pandas."""
    csv_files = get_csv_files()

    assert csv_files, (
        f"No CSV files available to test inside: {DATA_DIR}"
    )

    errors = []

    for file_path in csv_files:
        try:
            pd.read_csv(file_path)
        except Exception as error:
            errors.append(f"{file_path}: {error}")

    assert not errors, (
        "Some CSV files could not be read:\n" + "\n".join(errors)
    )


def test_csv_files_have_rows_and_columns():
    """Check that every CSV file contains rows and columns."""
    csv_files = get_csv_files()

    assert csv_files, (
        f"No CSV files available to test inside: {DATA_DIR}"
    )

    errors = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)

        if df.empty:
            errors.append(f"{file_path.name} contains no rows")

        if len(df.columns) == 0:
            errors.append(f"{file_path.name} contains no columns")

    assert not errors, "\n".join(errors)


def test_report_missing_values():
    """
    Display missing-value information.

    This test reports missing values but does not fail because some datasets
    may legitimately contain missing values.
    """
    csv_files = get_csv_files()

    assert csv_files, (
        f"No CSV files available to test inside: {DATA_DIR}"
    )

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        missing = df.isna().sum()
        missing = missing[missing > 0]

        print(f"\nDataset: {file_path}")

        if missing.empty:
            print("No missing values found.")
        else:
            print("Missing values:")
            print(missing)
from pathlib import Path
import pandas as pd
import sys

DATA_DIR = Path(__file__).parent.parent / "data"


def check_dataset_exists():
    """Check if data directory exists."""
    if not DATA_DIR.exists():
        print(f"❌ Data directory does not exist: {DATA_DIR}")
        return False
    print(f"✓ Data directory found: {DATA_DIR}")
    return True


def check_dataset_files():
    """Check for expected dataset files."""
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found in data directory")
        return False
    print(f"✓ Found {len(csv_files)} CSV file(s)")
    for file in csv_files:
        print(f"  - {file.name}")
    return True


def check_dataset_structure(file_path):
    """Check dataset structure and basic criteria."""
    try:
        df = pd.read_csv(file_path)
        print(f"\n📊 Checking: {file_path.name}")
        print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Data types:\n{df.dtypes}")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print(f"  ⚠ Missing values:\n{missing[missing > 0]}")
        
        return True
    except Exception as e:
        print(f"❌ Error reading {file_path.name}: {e}")
        return False


def main():
    """Main function to check dataset criteria."""
    print("=" * 50)
    print("Dataset Check Script")
    print("=" * 50)
    
    if not check_dataset_exists():
        sys.exit(1)
    
    if not check_dataset_files():
        sys.exit(1)
    
    csv_files = list(DATA_DIR.glob("*.csv"))
    for csv_file in csv_files:
        check_dataset_structure(csv_file)
    
    print("\n" + "=" * 50)
    print("Dataset check completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()


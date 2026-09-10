import pandas as pd
from pathlib import Path
import tomllib

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_DIR / "configs" / "dataset.toml", "rb") as file:
    config = tomllib.load(file)



def save_to_csv(df:pd.DataFrame, filename:str):
    df.to_csv(RAW_DATA_DIR / filename, index=False)
    
    
def save_srilanka_weekly_data():
    url = config["dataset_urls"]["SRILANKA_WEEKLY_DATA"]
    df = pd.read_csv(url)
    save_to_csv(df, "srilanka_weekly_data.csv")
    return df


if __name__ == "__main__":
    df = save_srilanka_weekly_data()
    print(df.head())

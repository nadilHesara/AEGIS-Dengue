from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tomllib

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "data" / "processed" / "panel_weekly.parquet"
CONFIG_PATH = ROOT / "configs" / "imputation" / "panel_v1.toml"
OUTPUT_PATH = ROOT / "results" / "data_validation" / "imputation_comparison.md"
OUTPUT_DATA_PATH = ROOT / "results" / "data_validation" / "imputation_comparison.csv"
IMPUTED_PANEL_PATH = ROOT / "data" / "processed" / "panel_weekly_imputed.parquet"

TARGET_COLUMN = "cases"
DISTRICT_COLUMN = "canonical_name"
MONTH_COLUMN = "source_month"
WEEK_COLUMN = "source_week"
YEAR_COLUMN = "source_year"
PERIOD_COLUMN = "period_id"
TRAIN_FLAG_COLUMN = "is_train_split"
TARGET_OBSERVED_COLUMN = "case_observed"
CLIMATE_OBSERVED_COLUMN = "row_has_complete_climate"

TARGET_PREFIXES = (
    "cases",
    "case_",
)
CLIMATE_PREFIXES = (
    "era5_",
    "chirps_",
)

CLIMATE_COLUMNS = [
    "rainfall_sum_mm",
    "rainfall_daily_mean_mm",
    "rainy_days",
    "temperature_mean_c",
    "temperature_min_c",
    "temperature_max_c",
    "dewpoint_mean_c",
    "relative_humidity_mean",
    "wind_speed_mean",
]

NON_FEATURE_COLUMNS = {
    PERIOD_COLUMN,
    YEAR_COLUMN,
    WEEK_COLUMN,
    "start_date",
    "end_date",
    "reporting_days",
    "weekday_convention",
    "is_weekday_convention_change",
    "is_irregular_period",
    "calendar_gap_before_days",
    "has_calendar_gap_before",
    "node_id",
    DISTRICT_COLUMN,
    "province",
    TARGET_OBSERVED_COLUMN,
    CLIMATE_OBSERVED_COLUMN,
    "row_has_complete_cases",
    "row_has_complete_climate",
    "row_is_fully_observed",
    "is_covid_window",
    "is_2017_outbreak",
}


@dataclass(frozen=True)
class MethodSummary:
    method: str
    target_mae: float
    target_rmse: float
    target_bias: float
    target_missing_rate: float
    climate_mae: float
    climate_rmse: float
    climate_bias: float
    climate_missing_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare leakage-safe imputation methods on the frozen master panel.")
    parser.add_argument("--panel-path", type=Path, default=PANEL_PATH)
    parser.add_argument("--config-path", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--output-data-path", type=Path, default=OUTPUT_DATA_PATH)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required_columns = {
        PERIOD_COLUMN,
        YEAR_COLUMN,
        WEEK_COLUMN,
        DISTRICT_COLUMN,
        "start_date",
        TARGET_COLUMN,
        TARGET_OBSERVED_COLUMN,
        CLIMATE_OBSERVED_COLUMN,
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")
    frame = frame.copy()
    frame["start_date"] = pd.to_datetime(frame["start_date"], errors="coerce")
    if frame["start_date"].isna().any():
        raise ValueError("panel contains invalid start_date values")
    frame[MONTH_COLUMN] = frame["start_date"].dt.month.astype("int8")
    return frame


def split_frame(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    train_source_year_end = int(config["train_source_year_end"])
    validation_source_year_start = int(config["validation_source_year_start"])
    test_source_year_start = int(config["test_source_year_start"])

    split = np.where(
        frame[YEAR_COLUMN] <= train_source_year_end,
        "train",
        np.where(
            frame[YEAR_COLUMN] < test_source_year_start,
            "validation",
            "test",
        ),
    )
    frame = frame.copy()
    frame[TRAIN_FLAG_COLUMN] = split == "train"
    frame["split_name"] = split
    frame["split_name"] = pd.Categorical(frame["split_name"], categories=["train", "validation", "test"], ordered=True)

    if validation_source_year_start <= train_source_year_end:
        raise ValueError("validation_source_year_start must be after train_source_year_end")
    return frame


def target_columns(frame: pd.DataFrame) -> list[str]:
    return [TARGET_COLUMN]


def climate_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = [column for column in CLIMATE_COLUMNS if column in frame.columns]
    if columns:
        return columns
    columns = []
    for column in frame.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def observed_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")


def evaluate_numeric(actual: pd.DataFrame, predicted: pd.DataFrame, mask: pd.DataFrame) -> tuple[float, float, float, float]:
    values = (predicted - actual).where(mask)
    observed = values.stack()
    if observed.empty:
        return float("nan"), float("nan"), float("nan"), float("nan")
    absolute_error = observed.abs()
    squared_error = observed.pow(2)
    return (
        float(absolute_error.mean()),
        float(np.sqrt(squared_error.mean())),
        float(observed.mean()),
        float(1.0 - mask.stack().mean()),
    )


def forward_fill_by_group(frame: pd.DataFrame, columns: list[str], max_gap: int) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=frame.index)
    grouped = frame.sort_values([DISTRICT_COLUMN, PERIOD_COLUMN]).groupby(DISTRICT_COLUMN, sort=False)
    filled = grouped[columns].ffill(limit=max_gap)
    return filled.reindex(frame.index)


def district_median_fill(train_frame: pd.DataFrame, full_frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=full_frame.index)
    medians = train_frame.groupby(DISTRICT_COLUMN)[columns].median(numeric_only=True)
    global_medians = train_frame[columns].median(numeric_only=True)
    filled = full_frame[[DISTRICT_COLUMN]].merge(
        medians,
        left_on=DISTRICT_COLUMN,
        right_index=True,
        how="left",
    )
    for column in columns:
        filled[column] = filled[column].fillna(global_medians[column])
    return filled[columns]


def district_month_climatology(train_frame: pd.DataFrame, full_frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=full_frame.index)
    climatology = train_frame.groupby([DISTRICT_COLUMN, MONTH_COLUMN])[columns].mean(numeric_only=True)
    global_month = train_frame.groupby(MONTH_COLUMN)[columns].mean(numeric_only=True)
    global_mean = train_frame[columns].mean(numeric_only=True)
    merged = full_frame[[DISTRICT_COLUMN, MONTH_COLUMN]].merge(
        climatology,
        left_on=[DISTRICT_COLUMN, MONTH_COLUMN],
        right_index=True,
        how="left",
    )
    month_means = full_frame[[MONTH_COLUMN]].merge(
        global_month,
        left_on=MONTH_COLUMN,
        right_index=True,
        how="left",
    )
    for column in columns:
        merged[column] = merged[column].fillna(month_means[column])
        merged[column] = merged[column].fillna(global_mean[column])
    return merged[columns]


def seasonal_group(month: int, mapping: dict[str, str]) -> str:
    return mapping[str(int(month))]


def seasonal_climatology(train_frame: pd.DataFrame, full_frame: pd.DataFrame, columns: list[str], mapping: dict[str, str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=full_frame.index)
    train_frame = train_frame.copy()
    full_frame = full_frame.copy()
    train_frame["seasonal_group"] = train_frame[MONTH_COLUMN].map(lambda month: seasonal_group(int(month), mapping))
    full_frame["seasonal_group"] = full_frame[MONTH_COLUMN].map(lambda month: seasonal_group(int(month), mapping))
    climatology = train_frame.groupby([DISTRICT_COLUMN, "seasonal_group"])[columns].mean(numeric_only=True)
    global_season = train_frame.groupby("seasonal_group")[columns].mean(numeric_only=True)
    global_mean = train_frame[columns].mean(numeric_only=True)
    merged = full_frame[[DISTRICT_COLUMN, "seasonal_group"]].merge(
        climatology,
        left_on=[DISTRICT_COLUMN, "seasonal_group"],
        right_index=True,
        how="left",
    )
    seasonal_means = full_frame[["seasonal_group"]].merge(
        global_season,
        left_on="seasonal_group",
        right_index=True,
        how="left",
    )
    for column in columns:
        merged[column] = merged[column].fillna(seasonal_means[column])
        merged[column] = merged[column].fillna(global_mean[column])
    return merged[columns]


def add_missingness_indicators(frame: pd.DataFrame, columns: list[str], suffix: str) -> pd.DataFrame:
    indicators = {}
    for column in columns:
        indicators[f"{column}{suffix}"] = frame[column].isna().astype("int8")
    return pd.DataFrame(indicators, index=frame.index)


def build_method_predictions(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, dict[str, pd.DataFrame]]:
    train_frame = frame.loc[frame[TRAIN_FLAG_COLUMN]].copy()
    target_cols = target_columns(frame)
    climate_cols = climate_columns(frame)
    max_gap = int(config["forward_fill_max_gap_periods"])

    predictions: dict[str, dict[str, pd.DataFrame]] = {}

    target_ffill = forward_fill_by_group(train_frame, target_cols, max_gap=max_gap).reindex(frame.index)
    target_ffill_full = forward_fill_by_group(frame, target_cols, max_gap=max_gap)
    climate_ffill = forward_fill_by_group(train_frame, climate_cols, max_gap=max_gap).reindex(frame.index)
    climate_ffill_full = forward_fill_by_group(frame, climate_cols, max_gap=max_gap)

    district_target = district_median_fill(train_frame, frame, target_cols)
    district_climate = district_median_fill(train_frame, frame, climate_cols)
    district_month_target = district_month_climatology(train_frame, frame, target_cols)
    district_month_climate = district_month_climatology(train_frame, frame, climate_cols)
    seasonal_mapping = config["seasonal_climatology"]
    seasonal_target = seasonal_climatology(train_frame, frame, target_cols, seasonal_mapping)
    seasonal_climate = seasonal_climatology(train_frame, frame, climate_cols, seasonal_mapping)

    predictions["forward_fill"] = {"target": target_ffill_full[target_cols], "climate": climate_ffill_full[climate_cols]}
    predictions["district_median"] = {"target": district_target, "climate": district_climate}
    predictions["district_month_climatology"] = {"target": district_month_target, "climate": district_month_climate}
    predictions["seasonal_climatology"] = {"target": seasonal_target, "climate": seasonal_climate}
    predictions["forward_fill_train_only"] = {"target": target_ffill[target_cols], "climate": climate_ffill[climate_cols]}
    return predictions


def summarize_methods(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[MethodSummary]]:
    target_cols = target_columns(frame)
    climate_cols = climate_columns(frame)
    target_actual = observed_numeric(frame, target_cols)
    climate_actual = observed_numeric(frame, climate_cols)
    target_mask = frame[TARGET_OBSERVED_COLUMN].astype(bool)
    climate_mask = frame[CLIMATE_OBSERVED_COLUMN].astype(bool)

    predictions = build_method_predictions(frame, config)
    summaries: list[MethodSummary] = []
    rows: list[dict[str, Any]] = []

    for method, groups in predictions.items():
        target_pred = groups["target"].reindex(frame.index)
        climate_pred = groups["climate"].reindex(frame.index)

        target_mae, target_rmse, target_bias, target_missing_rate = evaluate_numeric(
            target_actual,
            target_pred,
            pd.DataFrame(np.repeat(target_mask.to_numpy()[:, None], len(target_cols), axis=1), columns=target_cols, index=frame.index),
        )
        climate_mae, climate_rmse, climate_bias, climate_missing_rate = evaluate_numeric(
            climate_actual,
            climate_pred,
            pd.DataFrame(np.repeat(climate_mask.to_numpy()[:, None], len(climate_cols), axis=1), columns=climate_cols, index=frame.index),
        )

        summaries.append(
            MethodSummary(
                method=method,
                target_mae=target_mae,
                target_rmse=target_rmse,
                target_bias=target_bias,
                target_missing_rate=target_missing_rate,
                climate_mae=climate_mae,
                climate_rmse=climate_rmse,
                climate_bias=climate_bias,
                climate_missing_rate=climate_missing_rate,
            )
        )
        rows.append(
            {
                "method": method,
                "target_mae": target_mae,
                "target_rmse": target_rmse,
                "target_bias": target_bias,
                "target_missing_rate": target_missing_rate,
                "climate_mae": climate_mae,
                "climate_rmse": climate_rmse,
                "climate_bias": climate_bias,
                "climate_missing_rate": climate_missing_rate,
            }
        )

    comparison = pd.DataFrame(rows).sort_values(["target_mae", "climate_mae"], na_position="last")
    return comparison, summaries


def write_markdown_report(path: Path, frame: pd.DataFrame, comparison: pd.DataFrame, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Imputation Comparison",
        "",
        f"- Panel rows: {len(frame):,}",
        f"- Train split source year end: {config['train_source_year_end']}",
        f"- Validation split source year start: {config['validation_source_year_start']}",
        f"- Test split source year start: {config['test_source_year_start']}",
        f"- Forward fill max gap: {config['forward_fill_max_gap_periods']}",
        "",
        "## Method Ranking",
        "",
        comparison.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Leakage Rules",
        "",
        "- Fit all statistics on the training split only.",
        "- Keep dengue target missingness separate from climate-feature missingness.",
        "- Preserve the frozen master panel unchanged; emit derived outputs only.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_data(path: Path, comparison: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(path, index=False)


def build_model_ready_panel(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    train_frame = frame.loc[frame[TRAIN_FLAG_COLUMN]].copy()
    climate_cols = climate_columns(frame)
    method = str(config["selected_method"])
    predictions = build_method_predictions(frame, config)

    if method not in predictions:
        raise ValueError(f"selected_method {method!r} is not available")

    model_ready = frame.copy()
    climate_prediction = predictions[method]["climate"].reindex(frame.index)
    for column in climate_cols:
        model_ready[f"{column}_missing"] = model_ready[column].isna().astype("int8")
        model_ready[column] = climate_prediction[column].where(climate_prediction[column].notna(), model_ready[column])

    model_ready[f"{TARGET_COLUMN}_missing"] = model_ready[TARGET_COLUMN].isna().astype("int8")
    model_ready["climate_imputation_method"] = method
    model_ready["target_imputation_method"] = "none"
    model_ready["imputation_training_rows"] = int(len(train_frame))
    return model_ready


def write_imputed_panel(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)
    frame = split_frame(load_panel(args.panel_path), config)
    comparison, summaries = summarize_methods(frame, config)
    write_markdown_report(args.output_path, frame, comparison, config)
    write_comparison_data(args.output_data_path, comparison)
    write_imputed_panel(IMPUTED_PANEL_PATH, build_model_ready_panel(frame, config))

    payload = {
        "selected_method": config["selected_method"],
        "methods": [summary.__dict__ for summary in summaries],
        "imputed_panel_path": str(IMPUTED_PANEL_PATH.relative_to(ROOT)),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

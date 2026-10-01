"""Experiment B-weather: TimesFM XReg with causal district weather covariates.

At each origin the arm uses observed weather through that origin. Each future
weather value is replaced by that district's same-week-of-year climatology
fitted from weather observed through the origin. Thus target-week actual
weather is never read. Calendar sine/cosine features remain known future
covariates. Output: results/benchmark/predictions/timesfm2p5_weather_xreg.parquet.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.evaluation.long_horizon import (  # noqa: E402
    BENCHMARK_DIR, HORIZONS, QUANTILES, build_benchmark_folds, cell_frame,
    load_script, save_predictions,
)


def week_of_year(calendar: pd.DataFrame, future_steps: int = 0) -> np.ndarray:
    """Project-consistent 0..51 week buckets, extended for forecast tails."""

    starts = pd.DatetimeIndex(pd.to_datetime(calendar["start_date"]))
    if future_steps:
        starts = starts.append(starts[-1] + pd.to_timedelta(np.arange(1, future_steps + 1) * 7, unit="D"))
    starts = starts.to_numpy(dtype="datetime64[D]")
    year_start = starts.astype("datetime64[Y]")
    return np.minimum(((starts - year_start).astype(int) // 7), 51).astype(np.int16)


def load_weather(calendar: pd.DataFrame, path: Path, columns: list[str]) -> np.ndarray:
    """Load a complete [period, district, weather-feature] climate panel."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Experiment B-weather requires a complete prepared climate panel; "
            "it will not request or download weather automatically. Restore this file from a "
            "previous workspace, or explicitly resume weather preparation with "
            "`scripts/data/5c.fetch_open_meteo_climate.py` followed by "
            "`scripts/data/9.aggregate_climate_to_periods.py`."
        )
    climate = pd.read_parquet(path)
    required = {"period_id", "node_id", "weather_complete", *columns}
    missing = required.difference(climate.columns)
    if missing:
        raise ValueError(f"Climate panel is missing columns: {sorted(missing)}")
    period_ids = calendar["period_id"].to_numpy()
    node_ids = np.arange(int(climate["node_id"].max()) + 1)
    if len(node_ids) != 25:
        raise ValueError(f"Expected 25 district nodes in climate panel, found {len(node_ids)}")
    arrays = []
    for column in columns:
        matrix = climate.pivot(index="period_id", columns="node_id", values=column)
        arrays.append(matrix.reindex(index=period_ids, columns=node_ids).to_numpy(dtype=np.float32))
    values = np.stack(arrays, axis=-1)
    complete = climate.pivot(index="period_id", columns="node_id", values="weather_complete")
    complete = complete.reindex(index=period_ids, columns=node_ids).to_numpy()
    if not np.all(complete == 1) or not np.isfinite(values).all():
        raise ValueError(
            "Climate panel is incomplete. Finish/recover the 25-district climate panel before "
            "running Experiment B-weather; partial-weather results would not be comparable."
        )
    return values


def causal_weather_path(values: np.ndarray, weeks: np.ndarray, origin: int, node: int,
                        horizon: int) -> np.ndarray:
    """Observed past weather + origin-fitted week-of-year climatology future."""

    history = values[:origin + 1, node]
    table = np.full((52, values.shape[-1]), np.nan, dtype=np.float32)
    for week in range(52):
        rows = history[weeks[:origin + 1] == week]
        if len(rows):
            table[week] = np.nanmean(rows, axis=0)
    fallback = np.nanmean(history, axis=0)
    table = np.where(np.isfinite(table), table, fallback[None, :])
    future = table[weeks[origin + 1:origin + 1 + horizon]]
    return np.concatenate([history, future], axis=0)


def forecast_origins(model, log_cases: np.ndarray, weather: np.ndarray, season: dict[str, np.ndarray],
                     weeks: np.ndarray, origins: np.ndarray, horizon: int, chunk_size: int,
                     config: dict, adapter, calendar_adapter) -> np.ndarray:
    n_nodes = log_cases.shape[1]
    inputs = []
    dynamic = {name: [] for name in (*season.keys(), *config["weather_columns"])}
    for origin in origins:
        stop = origin + 1 + horizon
        for node in range(n_nodes):
            inputs.append(calendar_adapter.causal_fill(log_cases[:origin + 1, node]))
            for name, values in season.items():
                dynamic[name].append(values[:stop])
            weather_path = causal_weather_path(weather, weeks, origin, node, horizon)
            for index, name in enumerate(config["weather_columns"]):
                dynamic[name].append(weather_path[:, index])
    chunks = []
    for start in range(0, len(inputs), chunk_size):
        stop = start + chunk_size
        _, native = model.forecast_with_covariates(
            inputs=inputs[start:stop],
            dynamic_numerical_covariates={name: rows[start:stop] for name, rows in dynamic.items()},
            xreg_mode=config["xreg_mode"], normalize_xreg_target_per_input=True,
            ridge=float(config["ridge"]), force_on_cpu=bool(config["force_xreg_on_cpu"]),
        )
        chunks.append(adapter.extract_quantiles(np.asarray(native)))
    return np.concatenate(chunks).reshape(len(origins), n_nodes, horizon, len(QUANTILES))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "timesfm_weather_covariates.toml")
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    parser.add_argument("--suffix", default="")
    parser.add_argument("--no-holdout", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    adapter = load_script("timesfm_weather_adapter", "training/46.timesfm_zero_shot.py")
    calendar_adapter = load_script("timesfm_calendar_adapter", "training/47.timesfm_covariates.py")
    config = adapter.load_config(arguments.config)
    horizons = tuple(arguments.horizons)
    if not horizons or min(horizons) < 1:
        raise ValueError("At least one positive horizon is required")
    max_horizon = max(horizons)
    folds_module = load_script("timesfm_weather_folds_module", "features/14.build_folds.py")
    folds = build_benchmark_folds(folds_module, include_holdout=not arguments.no_holdout)
    if arguments.folds:
        folds = [fold for fold in folds if fold["fold_id"] in arguments.folds]
    if not folds:
        raise ValueError("No requested folds are available")
    calendar = folds_module.load_calendar()
    cases = adapter.load_canonical_cases(calendar, PROJECT_DIR / "data" / "interim" / "dengue_weekly_canonical.parquet")
    weather = load_weather(calendar, PROJECT_DIR / "data" / "interim" / "climate_by_dengue_period.parquet", config["weather_columns"])
    period_id = calendar["period_id"].to_numpy()
    index_of = {int(period): index for index, period in enumerate(period_id)}
    first = min(fold["val_start_period"] for fold in folds) - max_horizon
    last = max(fold["test_end_period"] for fold in folds) - 1
    origins = np.arange(max(index_of[first], 52), index_of[last] + 1)
    expected_cells = sum(sum(fold[f"{s}_end_period"] - fold[f"{s}_start_period"] + 1 for s in ("val", "test")) * len(horizons) * cases.shape[1] for fold in folds)
    print(f"Experiment B-weather: {len(origins)} origins × {cases.shape[1]} districts; {expected_cells:,} cells; covariates=calendar + weather.", flush=True)
    if arguments.dry_run:
        return 0
    try:
        import timesfm
        import timesfm.utils.xreg_lib  # noqa: F401
    except ImportError as error:
        raise SystemExit("TimesFM XReg requires JAX. Run `.venv\\Scripts\\python.exe -m pip install --upgrade -r requirements-timesfm.txt`.") from error
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(config["checkpoint"])
    model.compile(timesfm.ForecastConfig(max_context=int(config["max_context"]), max_horizon=max_horizon, normalize_inputs=True, per_core_batch_size=int(config["per_core_batch_size"]), use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True, return_backcast=True))
    started = time.perf_counter()
    forecast = forecast_origins(model, np.log1p(cases).astype(np.float32), weather, calendar_adapter.calendar_covariates(calendar, max_horizon), week_of_year(calendar, max_horizon), origins, max_horizon, int(config["call_chunk_size"]), config, adapter, calendar_adapter)
    forecast = np.clip(np.expm1(np.sort(forecast, axis=-1)), 0.0, None)
    positions = {int(origin): index for index, origin in enumerate(origins)}
    frames = []
    for fold in folds:
        for split in ("val", "test"):
            lo, hi = fold[f"{split}_start_period"], fold[f"{split}_end_period"]
            targets = np.array([p for p in period_id if lo <= p <= hi])
            target_indices = np.array([index_of[int(p)] for p in targets])
            for horizon in horizons:
                quantiles = forecast[np.array([positions[int(target - horizon)] for target in target_indices]), :, horizon - 1]
                frames.append(cell_frame(config["arm"], fold, split, horizon, targets, quantiles[..., QUANTILES.index(0.5)], -1, quantiles))
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != expected_cells or frame["prediction"].isna().any():
        raise RuntimeError("TimesFM weather output does not cover every benchmark cell")
    path = save_predictions([frame], config["arm"] + arguments.suffix)
    compute_dir = BENCHMARK_DIR / "compute"; compute_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"arm": config["arm"], "checkpoint": config["checkpoint"], "train_seconds": 0.0, "parameters": int(config["model_parameters"]), "inference_seconds": time.perf_counter() - started, "origins": len(origins), "covariates": "calendar + observed-past weather + causal weather climatology future", "xreg_mode": config["xreg_mode"]}]).to_csv(compute_dir / f"{config['arm']}{arguments.suffix}.csv", index=False)
    print(f"Wrote {path.relative_to(PROJECT_DIR)} in {time.perf_counter() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

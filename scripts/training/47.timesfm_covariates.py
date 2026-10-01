"""Experiment B: TimesFM 2.5 + future-known calendar covariates (XReg).

This outside comparator gives TimesFM the district's case history plus only
calendar seasonality known at every forecast origin: sin/cos of week-of-year.
It does not train or fine-tune TimesFM and does not use climate, district,
spatial, or future case data.  The XReg component is fit independently inside
each forecast context, as provided by the official TimesFM API.

Output: results/benchmark/predictions/timesfm2p5_calendar_xreg*.parquet.
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
    BENCHMARK_DIR,
    HORIZONS,
    QUANTILES,
    build_benchmark_folds,
    cell_frame,
    load_script,
    save_predictions,
)


def causal_fill(values: np.ndarray) -> np.ndarray:
    """Forward-fill missing case history without borrowing a future value."""

    filled = np.asarray(values, dtype=np.float32).copy()
    last = 0.0
    for index, value in enumerate(filled):
        if np.isfinite(value):
            last = value
        else:
            filled[index] = last
    return filled


def calendar_covariates(calendar: pd.DataFrame, max_horizon: int) -> dict[str, np.ndarray]:
    """Known weekly seasonal covariates, extended only for unscored tail steps."""

    starts = pd.DatetimeIndex(pd.to_datetime(calendar["start_date"]))
    extension = starts[-1] + pd.to_timedelta(np.arange(1, max_horizon + 1) * 7, unit="D")
    dates = starts.append(pd.DatetimeIndex(extension))
    angle = 2.0 * np.pi * dates.dayofyear.to_numpy(dtype=np.float32) / 365.25
    return {"season_sin": np.sin(angle).astype(np.float32),
            "season_cos": np.cos(angle).astype(np.float32)}


def forecast_origins(model, log_cases: np.ndarray, covariates: dict[str, np.ndarray],
                     origins: np.ndarray, horizon: int, call_chunk_size: int,
                     config: dict, adapter) -> np.ndarray:
    """Run an origin-by-district XReg forecast using only origin-available data."""

    n_nodes = log_cases.shape[1]
    inputs, dynamic = [], {name: [] for name in covariates}
    for origin in origins:
        stop = origin + 1 + horizon
        for node in range(n_nodes):
            inputs.append(causal_fill(log_cases[:origin + 1, node]))
            for name, values in covariates.items():
                dynamic[name].append(values[:stop])

    chunks = []
    for start in range(0, len(inputs), call_chunk_size):
        stop = start + call_chunk_size
        _, native = model.forecast_with_covariates(
            inputs=inputs[start:stop],
            dynamic_numerical_covariates={name: values[start:stop] for name, values in dynamic.items()},
            xreg_mode=config["xreg_mode"],
            normalize_xreg_target_per_input=True,
            ridge=float(config["ridge"]),
            force_on_cpu=bool(config["force_xreg_on_cpu"]),
        )
        chunks.append(adapter.extract_quantiles(np.asarray(native)))
    return np.concatenate(chunks, axis=0).reshape(
        len(origins), n_nodes, horizon, len(QUANTILES)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path,
                        default=PROJECT_DIR / "configs" / "timesfm_covariates.toml")
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    parser.add_argument("--suffix", default="", help="Optional output suffix, e.g. _pilot.")
    parser.add_argument("--no-holdout", action="store_true", help="Leave out fold 10 (2026).")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading TimesFM/XReg.")
    arguments = parser.parse_args()

    adapter = load_script("timesfm_zero_shot_adapter", "training/46.timesfm_zero_shot.py")
    config = adapter.load_config(arguments.config)
    horizons = tuple(arguments.horizons)
    if not horizons or min(horizons) < 1:
        raise ValueError("At least one positive horizon is required")
    max_horizon = max(horizons)

    folds_module = load_script("timesfm_covariates_folds_module", "features/14.build_folds.py")
    folds = build_benchmark_folds(folds_module, include_holdout=not arguments.no_holdout)
    if arguments.folds:
        folds = [fold for fold in folds if fold["fold_id"] in arguments.folds]
    if not folds:
        raise ValueError("No requested folds are available")
    calendar = folds_module.load_calendar()
    cases = adapter.load_canonical_cases(
        calendar, PROJECT_DIR / "data" / "interim" / "dengue_weekly_canonical.parquet"
    )
    period_id = calendar["period_id"].to_numpy()
    index_of = {int(period): index for index, period in enumerate(period_id)}
    first = min(fold["val_start_period"] for fold in folds) - max_horizon
    last = max(fold["test_end_period"] for fold in folds) - 1
    origins = np.arange(max(index_of[first], 52), index_of[last] + 1)
    expected_cells = sum(
        sum(fold[f"{split}_end_period"] - fold[f"{split}_start_period"] + 1
            for split in ("val", "test")) * len(horizons) * cases.shape[1]
        for fold in folds
    )
    print(f"Experiment B: {len(origins)} origins × {cases.shape[1]} districts; "
          f"{expected_cells:,} benchmark cells; covariates=calendar seasonality.", flush=True)
    if arguments.dry_run:
        return 0

    try:
        import timesfm
        import timesfm.utils.xreg_lib  # noqa: F401 -- fail early with a useful install message
    except ImportError as error:
        raise SystemExit(
            "TimesFM XReg requires a working CPU JAX runtime. Run `.venv\\Scripts\\python.exe "
            "-m pip install --upgrade -r requirements-timesfm.txt` first."
        ) from error

    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(config["checkpoint"])
    model.compile(timesfm.ForecastConfig(
        max_context=int(config["max_context"]), max_horizon=max_horizon,
        normalize_inputs=True, per_core_batch_size=int(config["per_core_batch_size"]),
        use_continuous_quantile_head=True, force_flip_invariance=True,
        infer_is_positive=True, fix_quantile_crossing=True, return_backcast=True,
    ))
    started = time.perf_counter()
    forecasts = forecast_origins(
        model, np.log1p(cases).astype(np.float32), calendar_covariates(calendar, max_horizon),
        origins, max_horizon, int(config["call_chunk_size"]), config, adapter,
    )
    forecasts = np.clip(np.expm1(np.sort(forecasts, axis=-1)), 0.0, None)
    position = {int(origin): index for index, origin in enumerate(origins)}
    frames = []
    for fold in folds:
        for split in ("val", "test"):
            lo, hi = fold[f"{split}_start_period"], fold[f"{split}_end_period"]
            targets = np.array([period for period in period_id if lo <= period <= hi])
            target_index = np.array([index_of[int(period)] for period in targets])
            for horizon in horizons:
                rows = np.array([position[int(target - horizon)] for target in target_index])
                quantiles = forecasts[rows, :, horizon - 1, :]
                frames.append(cell_frame(config["arm"], fold, split, horizon, targets,
                                         quantiles[..., QUANTILES.index(0.5)], -1, quantiles))
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != expected_cells or frame["prediction"].isna().any():
        raise RuntimeError("TimesFM XReg output does not cover every benchmark cell with a prediction")
    path = save_predictions([frame], config["arm"] + arguments.suffix)
    compute_dir = BENCHMARK_DIR / "compute"
    compute_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "arm": config["arm"], "checkpoint": config["checkpoint"], "train_seconds": 0.0,
        "parameters": int(config["model_parameters"]), "inference_seconds": time.perf_counter() - started,
        "origins": len(origins), "max_context": int(config["max_context"]),
        "input_transform": config["input_transform"], "covariates": "future-known season_sin, season_cos",
        "xreg_mode": config["xreg_mode"], "ridge": float(config["ridge"]),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }]).to_csv(compute_dir / f"{config['arm']}{arguments.suffix}.csv", index=False)
    print(f"Wrote {path.relative_to(PROJECT_DIR)} in {time.perf_counter() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

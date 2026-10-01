"""Experiment A: TimesFM 2.5 zero-shot forecasts on benchmark cells.

This is an outside comparator, not a trained AEGIS-Dengue model.  For every
district and forecast origin it receives only that district's observed dengue
case history through the origin.  It is never fit, fine-tuned, given climate,
calendar, spatial, or district-label inputs, and emits the same long-format
per-cell predictions as the established long-horizon benchmark.

Output: results/benchmark/predictions/timesfm2p5_zero_shot*.parquet and the
matching compute CSV.  Model weights are downloaded by TimesFM on first run.
"""

from __future__ import annotations

import argparse
import sys
import time
import tomllib
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


def load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    if tuple(config["quantiles"]) != QUANTILES:
        raise ValueError("TimesFM quantiles must exactly match benchmark QUANTILES")
    if config["input_transform"] != "log1p":
        raise ValueError("Experiment A is frozen to the benchmark's log1p case scale")
    return config


def extract_quantiles(native: np.ndarray) -> np.ndarray:
    """Map TimesFM 2.5's fixed q10..q90 grid onto benchmark quantiles.

    TimesFM 3.0.2 returns mean, then q10 through q90 in steps of 10.  The
    benchmark additionally needs q2.5, q25, q75 and q97.5.  We linearly
    interpolate on the model's log1p output scale; q2.5 and q97.5 use the
    nearest grid segment for a fixed, deterministic linear extrapolation.
    """

    native = np.asarray(native, dtype=np.float32)
    if native.ndim != 3:
        raise ValueError(f"Expected TimesFM [batch, horizon, quantile] output, got {native.shape}")
    native_levels = np.arange(0.1, 1.0, 0.1, dtype=np.float32)
    if native.shape[-1] != len(native_levels) + 1:
        raise ValueError(
            f"TimesFM returned {native.shape[-1]} forecast columns; expected mean + "
            "native q10..q90. Check the installed TimesFM API."
        )
    grid = native[..., 1:]
    result = []
    for level in QUANTILES:
        upper = int(np.searchsorted(native_levels, level, side="right"))
        upper = min(max(upper, 1), len(native_levels) - 1)
        lower = upper - 1
        weight = (level - native_levels[lower]) / (native_levels[upper] - native_levels[lower])
        result.append(grid[..., lower] + weight * (grid[..., upper] - grid[..., lower]))
    return np.stack(result, axis=-1).astype(np.float32)


def load_canonical_cases(calendar: pd.DataFrame, path: Path) -> np.ndarray:
    """Return [period, district] cases from the climate-independent dengue file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/data/0.load_dataset.py, "
            "scripts/data/2.create_calendar.py, and "
            "scripts/data/4.create_canonical_dengue.py first."
        )
    dengue = pd.read_parquet(path)
    required = {"period_id", "node_id", "cases", "case_observed"}
    missing = required.difference(dengue.columns)
    if missing:
        raise ValueError(f"Canonical dengue file is missing columns: {sorted(missing)}")
    dengue = dengue.copy()
    dengue.loc[dengue["case_observed"] != 1, "cases"] = np.nan
    matrix = dengue.pivot(index="period_id", columns="node_id", values="cases")
    node_ids = np.arange(int(matrix.columns.max()) + 1)
    if not np.array_equal(matrix.columns.to_numpy(), node_ids):
        raise ValueError("Canonical dengue node ids must be contiguous from 0")
    period_ids = calendar["period_id"].to_numpy()
    return matrix.reindex(index=period_ids, columns=node_ids).to_numpy(dtype=np.float32)


def forecast_origins(model, log_cases: np.ndarray, origins: np.ndarray, horizon: int,
                     call_chunk_size: int) -> np.ndarray:
    """Forecast every (origin, district) pair without reading after its origin."""

    n_nodes = log_cases.shape[1]
    inputs = [log_cases[: origin + 1, node].copy()
              for origin in origins for node in range(n_nodes)]
    chunks = []
    for start in range(0, len(inputs), call_chunk_size):
        _, native = model.forecast(horizon=horizon, inputs=inputs[start:start + call_chunk_size])
        chunks.append(extract_quantiles(native))
    return np.concatenate(chunks, axis=0).reshape(
        len(origins), n_nodes, horizon, len(QUANTILES)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path,
                        default=PROJECT_DIR / "configs" / "timesfm_zero_shot.toml")
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    parser.add_argument("--suffix", default="", help="Optional output suffix, e.g. _pilot.")
    parser.add_argument("--no-holdout", action="store_true", help="Leave out fold 10 (2026).")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and report cells without loading TimesFM.")
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    horizons = tuple(arguments.horizons)
    if not horizons or min(horizons) < 1:
        raise ValueError("At least one positive horizon is required")
    max_horizon = max(horizons)

    folds_module = load_script("timesfm_folds_module", "features/14.build_folds.py")
    folds = build_benchmark_folds(folds_module, include_holdout=not arguments.no_holdout)
    if arguments.folds:
        folds = [fold for fold in folds if fold["fold_id"] in arguments.folds]
    if not folds:
        raise ValueError("No requested folds are available")

    calendar = folds_module.load_calendar()
    cases = load_canonical_cases(
        calendar, PROJECT_DIR / "data" / "interim" / "dengue_weekly_canonical.parquet"
    )
    log_cases = np.log1p(cases).astype(np.float32)
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
    print(f"Experiment A: {len(origins)} origins × {cases.shape[1]} districts; "
          f"{expected_cells:,} benchmark cells.", flush=True)
    if arguments.dry_run:
        return 0

    try:
        import timesfm
    except ImportError as error:
        raise SystemExit(
            "TimesFM is not installed. Run `.venv\\Scripts\\python.exe -m pip install "
            "-r requirements-timesfm.txt` first."
        ) from error

    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(config["checkpoint"])
    model.compile(timesfm.ForecastConfig(
        max_context=int(config["max_context"]),
        max_horizon=max_horizon,
        normalize_inputs=True,
        per_core_batch_size=int(config["per_core_batch_size"]),
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    ))

    started = time.perf_counter()
    forecasts = forecast_origins(model, log_cases, origins, max_horizon,
                                 int(config["call_chunk_size"]))
    # The model forecast is on log1p(cases). expm1 preserves quantile order.
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
        raise RuntimeError("TimesFM output does not cover every benchmark cell with a prediction")
    path = save_predictions([frame], config["arm"] + arguments.suffix)

    compute_dir = BENCHMARK_DIR / "compute"
    compute_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "arm": config["arm"], "checkpoint": config["checkpoint"],
        "train_seconds": 0.0, "parameters": int(config["model_parameters"]),
        "inference_seconds": time.perf_counter() - started, "origins": len(origins),
        "max_context": int(config["max_context"]), "input_transform": config["input_transform"],
        "covariates": "none", "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }]).to_csv(compute_dir / f"{config['arm']}{arguments.suffix}.csv", index=False)
    print(f"Wrote {path.relative_to(PROJECT_DIR)} in {time.perf_counter() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

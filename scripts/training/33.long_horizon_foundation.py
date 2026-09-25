"""
Pretrained time-series foundation models, zero-shot, on the benchmark's cells.

Chronos-Bolt (Ansari et al. 2024) and Chronos-2 (Oct 2025) are forecasters
pretrained on large corpora of unrelated series. Zero-shot, they see a
district's own case history up to the forecast origin and nothing else about
dengue or Sri Lanka. They are the strongest "no domain modelling at all"
reference a 2026 paper can use, and they answer a question the short paper
could not: how much of the GRU's long-horizon skill is anything a generic
pretrained forecaster would also find in the case series alone?

Arms (all zero-shot, no fine-tuning, deterministic):
    chronos_bolt         Chronos-Bolt-small, univariate per district.
    chronos2             Chronos-2, univariate per district.
    chronos2_climate     Chronos-2 with the district's past 4-week rain,
                         temperature and humidity means as past-only
                         covariates (no future weather is given -- it would be
                         unknown at the origin) and the target week's season as
                         a known future covariate.
    chronos2_joint       Chronos-2 forecasting all 25 districts as one
                         multivariate series, so group attention can share
                         information across districts -- the foundation-model
                         analogue of a spatial graph.
    chronos2_ft          chronos2, LoRA fine-tuned per fold on each district's
    chronos2_joint_ft    series up to the fold's train_end (the validation and
                         test years are never seen), then forecast as its
                         zero-shot counterpart. Fold-specific, so each fold
                         forecasts only its own val/test origins.

Series are modelled on log1p(cases); quantiles map back through expm1, which
is exact for quantiles. Unobserved weeks are passed as NaN, which Chronos
masks. Because a zero-shot forecast depends only on data up to its origin, each
origin is forecast once and mapped onto every fold whose val or test year
contains the target -- there is nothing fold-specific to refit.

Output: results/benchmark/predictions/<arm>.parquet (seed -1, with quantiles).
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
    HORIZONS,
    QUANTILES,
    build_benchmark_folds,
    cell_frame,
    load_pipeline,
    save_predictions,
)

ARMS = ("chronos_bolt", "chronos2", "chronos2_climate", "chronos2_joint",
        "chronos2_ft", "chronos2_joint_ft")
FINE_TUNED = {"chronos2_ft": "chronos2", "chronos2_joint_ft": "chronos2_joint"}

# Fine-tuning settings, fixed before any headline fold is seen: LoRA (the
# package's recommendation for small data, with its recommended 1e-5 rate) on
# a 512-week context. Steps are the one knob, chosen on fold 0's validation
# year (2015) only -- see --ft-steps.
FT_SETTINGS = {"finetune_mode": "lora", "learning_rate": 1e-5, "context_length": 512,
               "batch_size": 64}
MODELS = {"chronos_bolt": "amazon/chronos-bolt-small"}
COVARIATES = ("rainfall_daily_mean_mm_roll4", "temperature_mean_c_roll4",
              "relative_humidity_mean_roll4")


def forecast_origins(pipeline, arm, log_cases, raw_x, names, season, origins, max_h):
    """Quantile forecasts [origins, nodes, max_h, q] on the log1p scale."""

    n_nodes = log_cases.shape[1]
    levels = list(QUANTILES)
    out = np.full((len(origins), n_nodes, max_h, len(levels)), np.nan, dtype=np.float32)

    if arm == "chronos_bolt":
        inputs = [torch.tensor(log_cases[: o + 1, n]) for o in origins for n in range(n_nodes)]
        chunks = []
        for start in range(0, len(inputs), 512):  # Bolt runs a list in one batch
            quantiles, _ = pipeline.predict_quantiles(inputs[start : start + 512],
                                                      prediction_length=max_h,
                                                      quantile_levels=levels)
            chunks.append(quantiles.numpy())
        return np.concatenate(chunks).reshape(len(origins), n_nodes, max_h, len(levels))

    if arm == "chronos2_joint":
        inputs = [log_cases[: o + 1].T.copy() for o in origins]  # [25, T]
        quantiles, _ = pipeline.predict_quantiles(inputs, prediction_length=max_h,
                                                  quantile_levels=levels, batch_size=64)
        for i, q in enumerate(quantiles):  # q: [25, max_h, q]
            out[i] = q.numpy()
        return out

    if arm == "chronos2":
        inputs = [log_cases[: o + 1, n].copy() for o in origins for n in range(n_nodes)]
    else:  # chronos2_climate
        index = {name: i for i, name in enumerate(names)}
        inputs = []
        for o in origins:
            for n in range(n_nodes):
                past = {c: raw_x[: o + 1, n, index[c]].astype(np.float32) for c in COVARIATES}
                past["season_sin"] = season[0][: o + 1].astype(np.float32)
                past["season_cos"] = season[1][: o + 1].astype(np.float32)
                future = {"season_sin": season[0][o + 1 : o + 1 + max_h].astype(np.float32),
                          "season_cos": season[1][o + 1 : o + 1 + max_h].astype(np.float32)}
                inputs.append({"target": log_cases[: o + 1, n].copy(),
                               "past_covariates": past, "future_covariates": future})
    quantiles, _ = pipeline.predict_quantiles(inputs, prediction_length=max_h,
                                              quantile_levels=levels, batch_size=32)
    stacked = np.stack([q.numpy().reshape(max_h, len(levels)) for q in quantiles])
    return stacked.reshape(len(origins), n_nodes, max_h, len(levels))


def fine_tune(base, arm, log_cases, fold, index_of, steps, output_dir):
    """A LoRA-fine-tuned copy of `base`, trained only on periods <= train_end."""

    end = index_of[int(fold["train_end_period"])] + 1
    if FINE_TUNED[arm] == "chronos2_joint":
        inputs = [log_cases[:end].T.copy()]
    else:
        inputs = [log_cases[:end, n].copy() for n in range(log_cases.shape[1])]
    return base.fit(inputs, prediction_length=12, num_steps=steps,
                    output_dir=str(output_dir / f"{arm}_fold{fold['fold_id']}"),
                    remove_printer_callback=True, report_to="none", **FT_SETTINGS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    parser.add_argument("--ft-steps", type=int, default=1000)  # chosen on fold 0 val (2015): 1000 < zero-shot < 100 < 300
    parser.add_argument("--suffix", default="")
    arguments = parser.parse_args()

    from chronos import BaseChronosPipeline

    baseline = load_pipeline()
    folds_module = baseline.folds_module
    folds = build_benchmark_folds(folds_module)
    if arguments.folds:
        folds = [f for f in folds if f["fold_id"] in arguments.folds]
    horizons = tuple(arguments.horizons)
    max_h = max(horizons)

    tensors = folds_module.load_tensors("v2")
    names = [str(n) for n in tensors["feature_names"]]
    y = np.where(tensors["y_mask"] == 1, tensors["y"], np.nan)
    log_cases = np.log1p(y).astype(np.float32)
    period_id = tensors["period_id"]

    start = tensors["start_date"].astype("datetime64[D]")
    doy = (start - start.astype("datetime64[Y]")).astype(int) + 1
    angle = 2.0 * np.pi * doy / 365.25
    # Season is a calendar fact, known for future weeks; extend past the panel end.
    extended = np.concatenate([angle, angle[-1] + 2.0 * np.pi * 7 / 365.25 * np.arange(1, max_h + 1)])
    season = (np.sin(extended), np.cos(extended))

    # Every origin whose target, at some horizon, falls in a val or test year.
    first = min(f["val_start_period"] for f in folds) - max_h
    last = max(f["test_end_period"] for f in folds) - 1
    index_of = {int(p): i for i, p in enumerate(period_id)}
    origins = np.arange(max(index_of[first], 52), index_of[last] + 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for arm in [a for a in arguments.arms if a in FINE_TUNED]:
        started = time.perf_counter()
        base = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=device,
                                                   torch_dtype=torch.float32)
        frames = []
        for fold in folds:
            tuned = fine_tune(base, arm, log_cases, fold, index_of, arguments.ft_steps,
                              PROJECT_DIR / "models" / "checkpoints" / "chronos2_ft")
            lo = index_of[int(fold["val_start_period"])] - max_h
            hi = index_of[int(fold["test_end_period"])] - 1
            fold_origins = np.arange(lo, hi + 1)
            forecasts = forecast_origins(tuned, FINE_TUNED[arm], log_cases, tensors["X"], names,
                                         season, fold_origins, max_h)
            forecasts = np.clip(np.expm1(np.sort(forecasts, axis=-1)), 0.0, None)
            position = {int(o): i for i, o in enumerate(fold_origins)}
            for split in ("val", "test"):
                lo_p, hi_p = fold[f"{split}_start_period"], fold[f"{split}_end_period"]
                targets = np.array([p for p in period_id if lo_p <= p <= hi_p])
                target_index = np.array([index_of[int(p)] for p in targets])
                for horizon in horizons:
                    rows = np.array([position[int(t - horizon)] for t in target_index])
                    quantiles = forecasts[rows, :, horizon - 1, :]
                    frames.append(cell_frame(arm, fold, split, horizon, targets,
                                             quantiles[..., QUANTILES.index(0.5)], -1, quantiles))
            print(f"  {arm} fold {fold['fold_id']} done "
                  f"({time.perf_counter() - started:.0f}s elapsed)", flush=True)
            del tuned
            torch.cuda.empty_cache()
        path = save_predictions(frames, arm + arguments.suffix)
        print(f"{arm}: wrote {path.relative_to(PROJECT_DIR)} in "
              f"{time.perf_counter() - started:.0f}s", flush=True)

    for arm in [a for a in arguments.arms if a not in FINE_TUNED]:
        started = time.perf_counter()
        name = MODELS.get(arm, "amazon/chronos-2")
        pipeline = BaseChronosPipeline.from_pretrained(name, device_map=device,
                                                       torch_dtype=torch.float32)
        forecasts = forecast_origins(pipeline, arm, log_cases, tensors["X"], names,
                                     season, origins, max_h)
        forecasts = np.clip(np.expm1(np.sort(forecasts, axis=-1)), 0.0, None)
        position = {int(o): i for i, o in enumerate(origins)}

        frames = []
        for fold in folds:
            for split in ("val", "test"):
                lo, hi = fold[f"{split}_start_period"], fold[f"{split}_end_period"]
                targets = np.array([p for p in period_id if lo <= p <= hi])
                target_index = np.array([index_of[int(p)] for p in targets])
                for horizon in horizons:
                    rows = np.array([position[int(t - horizon)] for t in target_index])
                    quantiles = forecasts[rows, :, horizon - 1, :]
                    median = quantiles[..., QUANTILES.index(0.5)]
                    frames.append(cell_frame(arm, fold, split, horizon, targets, median,
                                             -1, quantiles))
        path = save_predictions(frames, arm)
        print(f"{arm}: {len(origins)} origins, wrote {path.relative_to(PROJECT_DIR)} in "
              f"{time.perf_counter() - started:.0f}s", flush=True)
        del pipeline
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

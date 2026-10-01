"""Experiment C: fold-specific LoRA fine-tuning of TimesFM 2.5.

Each fold trains a fresh LoRA adapter on only that fold's canonical dengue
history through ``train_end_period``. No validation/test cases enter training.
The frozen adapter forecasts that fold's validation and test origins. This is
the official Transformers + PEFT route for TimesFM 2.5; it is deliberately
separate from the package's inference-only PyTorch API.
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
from torch.utils.data import DataLoader, Dataset

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.evaluation.long_horizon import (  # noqa: E402
    BENCHMARK_DIR, HORIZONS, build_benchmark_folds, cell_frame, load_script, save_predictions,
)


class RandomWindowDataset(Dataset):
    """Deterministic random context/future windows, entirely inside fold training."""

    def __init__(self, series: list[np.ndarray], context: int, horizon: int, samples: int, seed: int):
        self.series, self.context, self.horizon = series, context, horizon
        minimum = context + horizon
        valid = [index for index, values in enumerate(series) if len(values) >= minimum]
        if not valid:
            raise ValueError(f"No district history has {minimum} observations for LoRA training")
        rng = np.random.default_rng(seed)
        self.samples = []
        for _ in range(samples):
            index = int(rng.choice(valid))
            start = int(rng.integers(0, len(series[index]) - minimum + 1))
            self.samples.append((index, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        series_index, start = self.samples[index]
        values = self.series[series_index]
        return (torch.tensor(values[start:start + self.context], dtype=torch.float32),
                torch.tensor(values[start + self.context:start + self.context + self.horizon], dtype=torch.float32))


def causal_fill(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    last = 0.0
    for index, value in enumerate(out):
        if np.isfinite(value):
            last = value
        else:
            out[index] = last
    return out


def fold_series(cases: np.ndarray, end_index: int) -> list[np.ndarray]:
    """Raw count series through a fold's training boundary only."""

    return [causal_fill(cases[:end_index + 1, node]) for node in range(cases.shape[1])]


def train_adapter(series: list[np.ndarray], config: dict, output: Path, device: str):
    """Train and save one LoRA adapter without external data normalisation."""

    from peft import LoraConfig, get_peft_model
    try:
        from transformers import TimesFm2_5ModelForPrediction
    except ImportError as error:
        raise RuntimeError(
            "TimesFM 2.5 support is unavailable in this Transformers installation. "
            "Install the upstream build required by Experiment C with "
            "`.venv\\Scripts\\python.exe -m pip install --upgrade "
            "-r requirements-timesfm-finetune.txt`, then rerun this command."
        ) from error

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base = TimesFm2_5ModelForPrediction.from_pretrained(config["checkpoint"], torch_dtype=dtype).to(device)
    model = get_peft_model(base, LoraConfig(
        r=int(config["lora_rank"]), lora_alpha=int(config["lora_alpha"]),
        target_modules="all-linear", lora_dropout=float(config["lora_dropout"]), bias="none",
    ))
    dataset = RandomWindowDataset(series, int(config["context_length"]), int(config["horizon_length"]),
                                  int(config["training_samples"]), int(config["seed"]))
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True, drop_last=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=0.01)
    model.train()
    for _ in range(int(config["epochs"])):
        for context, future in loader:
            output_values = model(past_values=context.to(device), future_values=future.to(device),
                                  forecast_context_len=int(config["context_length"]))
            output_values.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step(); optimiser.zero_grad()
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    return model


def forecast_origins(model, cases: np.ndarray, origins: np.ndarray, horizon: int, context: int,
                     batch_size: int, device: str) -> np.ndarray:
    """Mean point forecasts for every district/origin after a fold-specific fit."""

    inputs = [causal_fill(cases[max(0, origin - context + 1):origin + 1, node])
              for origin in origins for node in range(cases.shape[1])]
    if any(len(values) != context for values in inputs):
        raise ValueError("Every benchmark origin must provide a full fine-tuning context")
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            values = torch.tensor(np.stack(inputs[start:start + batch_size]), dtype=torch.float32, device=device)
            prediction = model(past_values=values, forecast_context_len=context).mean_predictions
            output.append(prediction[:, :horizon].float().cpu().numpy())
    return np.concatenate(output).reshape(len(origins), cases.shape[1], horizon)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "timesfm_lora.toml")
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    parser.add_argument("--suffix", default="")
    parser.add_argument("--no-holdout", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    with arguments.config.open("rb") as handle:
        config = tomllib.load(handle)
    horizons = tuple(arguments.horizons)
    if max(horizons) > int(config["horizon_length"]):
        raise ValueError("horizon_length must cover every requested benchmark horizon")
    if int(config["context_length"]) % 32:
        raise ValueError("TimesFM context_length must be divisible by 32")

    adapter = load_script("timesfm_lora_data", "training/46.timesfm_zero_shot.py")
    folds_module = load_script("timesfm_lora_folds", "features/14.build_folds.py")
    folds = build_benchmark_folds(folds_module, include_holdout=not arguments.no_holdout)
    if arguments.folds:
        folds = [fold for fold in folds if fold["fold_id"] in arguments.folds]
    calendar = folds_module.load_calendar()
    cases = adapter.load_canonical_cases(calendar, PROJECT_DIR / "data" / "interim" / "dengue_weekly_canonical.parquet")
    period_id = calendar["period_id"].to_numpy(); index_of = {int(p): i for i, p in enumerate(period_id)}
    print(f"Experiment C: {len(folds)} fold-specific LoRA adapters; {cases.shape[1]} districts; "
          f"training through each fold's train_end only.", flush=True)
    if arguments.dry_run:
        return 0
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as error:
        raise SystemExit("Experiment C dependencies are missing. Run `.venv\\Scripts\\python.exe -m pip install -r requirements-timesfm-finetune.txt`.") from error
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames, compute = [], []
    started = time.perf_counter()
    for fold in folds:
        train_end = index_of[int(fold["train_end_period"])]
        fit_started = time.perf_counter()
        model = train_adapter(fold_series(cases, train_end), config,
                              PROJECT_DIR / "models" / "checkpoints" / "timesfm_lora" / f"fold{fold['fold_id']}", device)
        fit_seconds = time.perf_counter() - fit_started
        first = index_of[int(fold["val_start_period"])] - max(horizons)
        last = index_of[int(fold["test_end_period"])] - 1
        origins = np.arange(max(first, int(config["context_length"]) - 1), last + 1)
        infer_started = time.perf_counter()
        forecast = np.clip(forecast_origins(model, cases, origins, max(horizons), int(config["context_length"]), int(config["batch_size"]), device), 0.0, None)
        positions = {int(origin): index for index, origin in enumerate(origins)}
        for split in ("val", "test"):
            lo, hi = fold[f"{split}_start_period"], fold[f"{split}_end_period"]
            targets = np.array([p for p in period_id if lo <= p <= hi])
            target_indices = np.array([index_of[int(p)] for p in targets])
            for horizon in horizons:
                prediction = forecast[np.array([positions[int(target - horizon)] for target in target_indices]), :, horizon - 1]
                frames.append(cell_frame(config["arm"], fold, split, horizon, targets, prediction, -1))
        compute.append({"fold_id": fold["fold_id"], "train_seconds": fit_seconds,
                        "inference_seconds": time.perf_counter() - infer_started, "origins": len(origins)})
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    path = save_predictions(frames, config["arm"] + arguments.suffix)
    compute_dir = BENCHMARK_DIR / "compute"; compute_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(compute); table["arm"] = config["arm"]; table["checkpoint"] = config["checkpoint"]; table["device"] = device; table["wall_seconds"] = time.perf_counter() - started
    table.to_csv(compute_dir / f"{config['arm']}{arguments.suffix}.csv", index=False)
    print(f"Wrote {path.relative_to(PROJECT_DIR)} in {time.perf_counter() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

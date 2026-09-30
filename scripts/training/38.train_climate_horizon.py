"""
Train the climate-horizon model (plan §6.4).

Step 4 implements the baseline arm only, `hcd_uniform` (climate-gradient
weights [1, 1, 1, 1]). `--smoke` runs a tiny check (one fold, one seed, few
epochs) and writes to results/climate_horizon/smoke/. That directory is not
read by the evaluator.

    python scripts/training/38.train_climate_horizon.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.data.climate_horizon import (  # noqa: E402
    VARIANT, build_arrays, development_last_period, extension_fold, split_channels,
)
from src.evaluation.long_horizon import load_script, months_for  # noqa: E402
from src.training import climate_horizon as trainer  # noqa: E402

OUTPUT_DIR = PROJECT_DIR / "results" / "climate_horizon"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fold", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lag-mode", default="target", choices=["target", "origin"])
    parser.add_argument("--common-support", action="store_true")
    args = parser.parse_args()
    if not args.smoke:
        raise SystemExit("Only --smoke is implemented in this step.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fm = load_script("build_folds", "features/14.build_folds.py")
    calendar = fm.load_calendar()
    tensors = fm.load_tensors(VARIANT)
    channels = split_channels(tensors["feature_names"])
    folds = {f["fold_id"]: f for f in json.loads(
        (PROJECT_DIR / "data" / "processed" / "folds.json").read_text())["folds"]}
    fold = extension_fold(folds[args.fold])

    arrays = build_arrays(tensors, months_for(fm), fold, fm, channels,
                          last_target_period=development_last_period(calendar))
    config = {**trainer.DEFAULTS, "lag_mode": args.lag_mode, "common_support": args.common_support}
    tag = args.lag_mode + ("_common" if args.common_support else "")
    model, info = trainer.train(arrays, config, args.seed, device,
                                weights=trainer.UNIFORM_WEIGHTS, max_epochs=args.epochs)

    out = OUTPUT_DIR / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    test = arrays["test"]
    result = trainer.predict(model, test, device)
    gate_off = trainer.predict(model, test, device, gate_off=True)
    frame = trainer.prediction_frame("hcd_uniform", args.fold, "test", test, result, args.seed)
    frame.to_parquet(out / f"hcd_uniform_{tag}_smoke.parquet", index=False)

    observed = test["mask"] == 1
    mae = np.abs(result["prediction"] - test["y"])[observed].mean()
    summary = {
        "fold": args.fold, "seed": args.seed, "device": str(device), "epochs": args.epochs,
        **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in info.items() if k != "gradient_log"},
        "climate_weights": trainer.UNIFORM_WEIGHTS,
        "case_channels": channels.case_names(), "climate_channels": channels.climate_names(),
        "train_origins": int(len(arrays["train"]["y"])), "test_cells": int(observed.sum()),
        "test_mae_median": float(mae),
        "all_finite": bool(all(np.isfinite(result[k]).all() for k in ("mu", "alpha", "gate", "prediction"))),
        "gate_mean_by_horizon": result["gate"].mean(axis=(0, 1)).round(4).tolist(),
        "abs_climate_correction_by_horizon": np.abs(result["delta_climate"]).mean(axis=(0, 1)).round(5).tolist(),
        "lag_mode": args.lag_mode, "common_support": args.common_support,
        "max_delay": model.encoder.max_delay, "history": model.encoder.history,
        "available_mass_by_horizon": model.available_mass().mean(dim=(1, 2)).detach().cpu().numpy().round(4).tolist(),
        "peak_delay_rainfall_by_horizon": model.encoder.peak_delays()[:, :, 0].mean(-1).detach().cpu().numpy().round(2).tolist(),
        "gate_off_changes_mu": bool(not np.allclose(result["mu"], gate_off["mu"])),
    }
    (out / f"smoke_summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

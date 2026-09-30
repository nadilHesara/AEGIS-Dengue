"""
End-to-end checks before the full sweep.

A. Synthetic NB panels (src/data/climate_horizon_synthetic.py) with a
   planted climate delay (beta > 0) or no climate signal (beta = 0). The
   equal-weight model is trained and the script reports: finite losses and
   predictions, which parameter groups updated, MAE with real versus
   shuffled climate on the synthetic test period, gates and climate
   correction sizes, and the learned rainfall delay (descriptive only).

B. A short pre-2026 development comparison on outer fold 9: equal-weight
   versus fold-9 shuffle weights. Both arms share architecture,
   initialisation, seeds, data order, budget, NB head, loss and checkpoint
   rule. They are scored on the **validation year (2024) only**. That year
   selects the checkpoint, so this is not a held-out estimate. The test year
   (2025) is not scored, to keep the declared primary comparison untouched.
   Checkpoints are saved, reloaded and compared prediction by prediction.

    python scripts/evaluation/43.climate_horizon_end_to_end_checks.py

Outputs: results/climate_horizon/end_to_end/, results/climate_horizon/checkpoints/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.data.climate_horizon import (  # noqa: E402
    HORIZONS, VARIANT, build_arrays, development_last_period, extension_fold, split_channels,
)
from src.data.climate_horizon_synthetic import DELAY, make_panel  # noqa: E402
from src.evaluation.climate_horizon import data_provenance  # noqa: E402
from src.evaluation.long_horizon import load_script, months_for  # noqa: E402
from src.training import climate_horizon as tr  # noqa: E402
from src.training.climate_horizon_pilots import array_fingerprint  # noqa: E402

OUT = PROJECT_DIR / "results" / "climate_horizon" / "end_to_end"
CKPT = PROJECT_DIR / "results" / "climate_horizon" / "checkpoints"
SEEDS = (0, 1)
N_SHUFFLES = 5


def masked_mae(prediction, part):
    live = part["mask"] == 1
    return [float(np.abs(prediction - part["y"])[..., h][live[..., h]].mean()) for h in range(len(HORIZONS))]


def correction_summary(result, part) -> dict:
    live = part["mask"] == 1
    out = {}
    for key in ("gate", "delta_climate", "gated_climate", "delta_case"):
        values = np.abs(result[key]) if key != "gate" else result[key]
        out[key] = [float(values[..., h][live[..., h]].mean()) for h in range(len(HORIZONS))]
    return out


def changed_fraction(before: dict, model) -> dict:
    climate = {id(p) for p in model.climate_parameters()}
    rows = {"climate": [0, 0], "other": [0, 0]}
    for name, p in model.named_parameters():
        group = "climate" if id(p) in climate else "other"
        rows[group][0] += int((p.detach().cpu() != before[name]).any())
        rows[group][1] += 1
    return {g: f"{a}/{b} tensors changed" for g, (a, b) in rows.items()}


def synthetic_checks(fm, device) -> list[dict]:
    rows = []
    for condition, beta in (("delayed_climate_signal", 0.6), ("no_climate_signal", 0.0)):
        tensors, months, fold = make_panel(beta=beta, seed=0)
        channels = split_channels(tensors["feature_names"])
        arrays = build_arrays(tensors, months, fold, fm, channels)
        for seed in SEEDS:
            torch.manual_seed(seed)
            initial = {n: p.detach().clone() for n, p in tr.build_model(arrays, tr.DEFAULTS).named_parameters()}
            model, info = tr.train(arrays, dict(tr.DEFAULTS), seed, device)
            test = arrays["test"]
            real = tr.predict(model, test, device)
            rng = np.random.default_rng(100 + seed)
            shuffled = []
            for _ in range(N_SHUFFLES):
                x = test["X_climate"].copy()
                for n in range(x.shape[2]):
                    x[:, :, n] = x[rng.permutation(len(x)), :, n]
                shuffled.append(masked_mae(tr.predict(model, test, device, x_climate=x)["prediction"], test))
            gate_off = tr.predict(model, test, device, gate_off=True)
            mae_real = masked_mae(real["prediction"], test)
            mae_shuf = np.mean(shuffled, axis=0).tolist()
            log = info["gradient_log"]
            rain = channels.climate_names().index("rainfall_daily_mean_mm")
            rows.append({
                "condition": condition, "beta": beta, "seed": seed,
                "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
                "all_losses_finite": bool(all(np.isfinite(g["loss_equal"]) for g in log)),
                "predictions_finite": bool(all(np.isfinite(real[k]).all() for k in ("mu", "alpha", "prediction"))),
                "parameters_updated": changed_fraction(initial, model),
                "mae_real": mae_real, "mae_shuffled": mae_shuf,
                "shuffle_gain": [s - r for s, r in zip(mae_shuf, mae_real)],
                "mae_gate_off": masked_mae(gate_off["prediction"], test),
                "corrections": correction_summary(real, test),
                "learned_rain_delay_by_h": model.encoder.peak_delays()[:, :, rain].mean(-1).detach().cpu().numpy().round(2).tolist(),
                "planted_delay": DELAY if beta else None,
            })
            print(condition, seed, "gain", np.round(rows[-1]["shuffle_gain"], 3), "delay", rows[-1]["learned_rain_delay_by_h"], flush=True)
    return rows


def development_comparison(fm, device) -> tuple[list[dict], list[dict]]:
    calendar = fm.load_calendar()
    months = months_for(fm)
    tensors = fm.load_tensors(VARIANT)
    channels = split_channels(tensors["feature_names"])
    committed = json.loads((PROJECT_DIR / "data" / "processed" / "folds.json").read_text())["folds"]
    fold = extension_fold({f["fold_id"]: f for f in committed}[9])
    arrays = build_arrays(tensors, months, fold, fm, channels,
                          last_target_period=development_last_period(calendar))
    stats = fm.fit_fold_statistics(tensors, months, tensors["period_id"] <= fold["fit_end_period"])
    weight_file = PROJECT_DIR / "results" / "climate_horizon" / "weights" / "fold9_shuffle.json"
    fold9 = json.loads(weight_file.read_text())["weights"]
    fingerprints = {"data_manifest_match": bool(data_provenance()["matches_manifest"].all()),
                    "train": array_fingerprint(arrays["train"]["y"], arrays["train"]["mask"]),
                    "val": array_fingerprint(arrays["val"]["target_period_id"], arrays["val"]["mask"])}

    rows, reload_rows = [], []
    for arm, weights, source in (("hcd_uniform", tr.UNIFORM_WEIGHTS, "uniform"),
                                 ("hcd_informed", fold9, str(weight_file.relative_to(PROJECT_DIR)))):
        for seed in SEEDS:
            path = CKPT / f"dev_fold9_{arm}_seed{seed}.pt"
            metadata = tr.checkpoint_metadata(channels, stats, HORIZONS, fold, weights, source, fingerprints,
                                              {"lag_mode": "target", "history": 26, "max_delay": 29,
                                               "common_support": False})
            model, info = tr.train(arrays, dict(tr.DEFAULTS), seed, device, weights,
                                   checkpoint_path=path, metadata=metadata)
            val = arrays["val"]
            result = tr.predict(model, val, device)
            reloaded, _ = tr.load_checkpoint(path, device)
            again = tr.predict(reloaded, val, device)
            reload_rows.append({"arm": arm, "seed": seed, "checkpoint": path.name,
                                "predictions_identical": bool(np.array_equal(result["prediction"], again["prediction"])
                                                              and np.array_equal(result["mu"], again["mu"]))})
            log = pd.DataFrame(info["gradient_log"])
            rows.append({
                "arm": arm, "seed": seed, "weights": list(weights),
                "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
                "val_nll": info["val_loss"], "val_mae": masked_mae(result["prediction"], val),
                "corrections": correction_summary(result, val),
                "raw_norm_climate_mean": float(log["raw_norm_climate"].mean()),
                "raw_norm_other_mean": float(log["raw_norm_other"].mean()),
                "clip_fraction": float((log["clip_coefficient"] < 1).mean()),
                "cosine_weighted_vs_equal_mean": float(log["cosine_climate_weighted_vs_equal"].dropna().mean()),
                "train_seconds": info["train_seconds"],
            })
            print(arm, seed, "val nll", round(info["val_loss"], 4), "mae", np.round(rows[-1]["val_mae"], 2), flush=True)
    return rows, reload_rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fm = load_script("build_folds", "features/14.build_folds.py")
    synthetic = synthetic_checks(fm, device)
    (OUT / "synthetic_checks.json").write_text(json.dumps(synthetic, indent=2), encoding="utf-8")
    dev, reloads = development_comparison(fm, device)
    (OUT / "dev_comparison_fold9_val2024.json").write_text(json.dumps(dev, indent=2), encoding="utf-8")
    (OUT / "checkpoint_reload.json").write_text(json.dumps(reloads, indent=2), encoding="utf-8")
    print(json.dumps(reloads))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

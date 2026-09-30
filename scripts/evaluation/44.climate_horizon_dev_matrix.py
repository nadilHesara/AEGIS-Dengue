"""
Pre-2026 development matrix on outer fold 9 (validation year 2024 only).

    A  origin-relative lags,  equal climate weights
    B  target-relative lags,  equal climate weights      (main baseline)
    C  origin-relative lags,  measured weights (fold9_shuffle.json)
    D  target-relative lags,  measured weights (fold9_shuffle.json)   (proposed)
    E  target, fixed increasing weights [1, 2, 3, 4]
    F  target, h=4-only climate gradients [0, 0, 0, 1]
    G  target, case-only-utility weights (fold9_case_only.json)
    H  target, GradNorm on the climate group (src/training/climate_gradnorm.py)

In every arm the case branch, gate and heads learn from all horizons
(L_equal). Only the climate group's loss changes. The NB head, loss,
preprocessing (fitted on training years), origins, initialisation (the same
seed builds the same weights), data order and budget are the same
throughout. Seeds are 0 and 1.

Scored on 2024, the year that also selects each checkpoint, so these are
**not held-out estimates**. The 2025 test year is not scored. Stored
benchmark predictions for the same 2024 cells (persistence recomputed,
gru_v2_nb, nb_shared_v2, lgbm_v2) are read only; they also selected their
checkpoints on 2024. Chronos models are not used.

    python scripts/evaluation/44.climate_horizon_dev_matrix.py

Outputs: results/climate_horizon/dev_matrix/ (resumable per arm/seed).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.data.climate_horizon import (  # noqa: E402
    HORIZONS, VARIANT, build_arrays, development_last_period, extension_fold, split_channels,
)
from src.evaluation import climate_horizon as ev  # noqa: E402
from src.evaluation.long_horizon import PREDICTIONS_DIR, load_pipeline, months_for, split_by_target, truth_frame  # noqa: E402
from src.models.climate_horizon import nb_loss_terms  # noqa: E402
from src.training import climate_horizon as tr  # noqa: E402

OUT = PROJECT_DIR / "results" / "climate_horizon" / "dev_matrix"
WEIGHTS = PROJECT_DIR / "results" / "climate_horizon" / "weights"
SEEDS = (0, 1)
FOLD = 9


def arms() -> dict:
    measured = json.loads((WEIGHTS / f"fold{FOLD}_shuffle.json").read_text())["weights"]
    case_only = json.loads((WEIGHTS / f"fold{FOLD}_case_only.json").read_text())["weights"]
    return {
        "A_origin_equal": ("origin", (1, 1, 1, 1), False),
        "B_target_equal": ("target", (1, 1, 1, 1), False),
        "C_origin_measured": ("origin", measured, False),
        "D_target_measured": ("target", measured, False),
        "E_target_fixed_1234": ("target", (1, 2, 3, 4), False),
        "F_target_h4_only": ("target", (0, 0, 0, 1), False),
        "G_target_caseonly_utility": ("target", case_only, False),
        "H_target_gradnorm": ("target", (1, 1, 1, 1), True),
    }


def nll_by_horizon(result, part) -> list[float]:
    num, den = nb_loss_terms(torch.from_numpy(result["mu"]), torch.from_numpy(result["alpha"]),
                             torch.from_numpy(part["y"]), torch.from_numpy(part["mask"]))
    return (num / den).tolist()


def main() -> int:
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = load_pipeline()
    fm = baseline.folds_module
    calendar = fm.load_calendar()
    tensors = fm.load_tensors(VARIANT)
    channels = split_channels(tensors["feature_names"])
    committed = json.loads((PROJECT_DIR / "data" / "processed" / "folds.json").read_text())["folds"]
    outer = {f["fold_id"]: f for f in committed}[FOLD]
    fold = extension_fold(outer)
    arrays = build_arrays(tensors, months_for(fm), fold, fm, channels,
                          last_target_period=development_last_period(calendar))
    val = arrays["val"]
    thresholds = baseline.naive.peak_thresholds(tensors["y"], tensors["y_mask"],
                                                tensors["period_id"] <= fold["train_end_period"])

    rows, frames = [], []
    for arm, (mode, weights, gradnorm) in arms().items():
        config = {**tr.DEFAULTS, "lag_mode": mode, "gradnorm": gradnorm}
        for seed in SEEDS:
            path = OUT / "predictions" / f"{arm}_seed{seed}.parquet"
            meta_path = path.with_suffix(".json")
            if path.exists() and meta_path.exists():
                frames.append(pd.read_parquet(path))
                rows.append(json.loads(meta_path.read_text()))
                continue
            started = time.perf_counter()
            model, info = tr.train(arrays, config, seed, device, weights)
            result = tr.predict(model, val, device)
            frame = tr.prediction_frame(arm, FOLD, "val", val, result, seed)
            frame.to_parquet(path, index=False)
            live = val["mask"] == 1
            log = pd.DataFrame(info["gradient_log"])
            rain = channels.climate_names().index("rainfall_daily_mean_mm")
            weights_mass = model.available_mass().detach().cpu().numpy()
            row = {
                "arm": arm, "lag_mode": mode, "seed": seed, "weights": [float(w) for w in weights],
                "gradnorm": gradnorm, "parameters": info["parameters"], "best_epoch": info["best_epoch"],
                "epochs_run": info["epochs_run"], "val_nll_total": info["val_loss"],
                "val_nll_by_h": nll_by_horizon(result, val),
                "mean_gate_by_h": [float(result["gate"][..., h][live[..., h]].mean()) for h in range(4)],
                "mean_abs_delta_climate_by_h": [float(np.abs(result["delta_climate"][..., h])[live[..., h]].mean()) for h in range(4)],
                "mean_abs_gated_climate_by_h": [float(np.abs(result["gated_climate"][..., h])[live[..., h]].mean()) for h in range(4)],
                "available_mass_by_h": weights_mass.mean(axis=(1, 2)).round(4).tolist(),
                "rain_peak_delay_by_h": model.encoder.peak_delays()[:, :, rain].mean(-1).detach().cpu().numpy().round(2).tolist(),
                "raw_norm_climate_mean": float(log["raw_norm_climate"].mean()),
                "cosine_weighted_vs_equal_mean": float(log["cosine_climate_weighted_vs_equal"].dropna().mean()),
                "climate_steps_skipped": int(log["climate_skipped"].sum()),
                "final_gradnorm_weights": ([float(log[f"gradnorm_w{h}"].dropna().iloc[-1]) for h in HORIZONS]
                                           if gradnorm else None),
                "seconds": time.perf_counter() - started,
            }
            meta_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            rows.append(row)
            frames.append(frame)
            print(arm, seed, "nll", round(info["val_loss"], 4), flush=True)
    runs = pd.DataFrame(rows)
    runs.to_json(OUT / "runs.json", orient="records", indent=2)

    # Stored references on the same 2024 cells (read-only), plus persistence.
    truth = truth_frame(tensors)
    cells = pd.concat(frames)[ev.KEYS].drop_duplicates()
    refs = []
    provenance = []
    for method in ev.REFERENCE_METHODS:
        frame = pd.read_parquet(PREDICTIONS_DIR / f"{method}.parquet",
                                columns=["method", *ev.KEYS, "seed", "prediction"])
        frame = frame[(frame["method"] == method) & (frame["fold_id"] == FOLD) & (frame["split"] == "val")
                      & frame["horizon"].isin(HORIZONS)]
        labels_ok = bool((split_by_target(frame["target_period_id"].to_numpy(), outer) == "val").all())
        provenance.append({"method": method, "split_rule_per_target": labels_ok,
                           "rows": len(frame), "seeds": sorted(frame["seed"].unique().tolist())})
        refs.append(frame)
    pd.DataFrame(provenance).to_csv(OUT / "reference_provenance.csv", index=False)
    references = pd.concat(refs + [ev.persistence_on_cells(cells, tensors, {FOLD: outer})], ignore_index=True)
    extension = pd.concat(frames)[["method", *ev.KEYS, "seed", "prediction"]]
    table = pd.concat([extension, references], ignore_index=True)
    common = cells
    for method, group in table.groupby("method"):
        common = common.merge(group[ev.KEYS].drop_duplicates(), on=ev.KEYS)
    table = table.merge(common, on=ev.KEYS)
    metrics = ev.score(table, truth, {FOLD: thresholds})
    metrics.to_csv(OUT / "metrics_by_seed.csv", index=False)
    summary = metrics.groupby(["method", "horizon"])[["mae", "peak_mae"]].mean().reset_index()
    spread = metrics.groupby(["method", "horizon"])["mae"].agg(lambda x: float(x.max() - x.min())).rename("mae_seed_range")
    summary = summary.merge(spread.reset_index(), on=["method", "horizon"])
    summary.to_csv(OUT / "summary.csv", index=False)
    mae = summary.pivot(index="method", columns="horizon", values="mae").round(3)
    peak = summary.pivot(index="method", columns="horizon", values="peak_mae").round(3)
    print(f"common 2024 cells per horizon: {len(common) // 4}")
    print(mae.to_string())
    print(peak.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

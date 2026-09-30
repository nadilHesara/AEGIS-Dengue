"""
Tiny smoke check of routing v2 (not a result). Outer fold 9, seed 0, 5 epochs,
arms B (equal), D (fold-9 frozen shuffle weights), F (h=4 only). Scored on the
2024 validation year only; no test year, no 2026.

    python scripts/evaluation/51.climate_horizon_routing_smoke.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.data.climate_horizon import VARIANT, build_arrays, development_last_period, extension_fold, split_channels  # noqa: E402
from src.evaluation.long_horizon import build_benchmark_folds, load_pipeline, months_for  # noqa: E402
from src.models.climate_horizon import ROUTING_VERSION  # noqa: E402
from src.training import climate_horizon as tr  # noqa: E402

OUT = PROJECT_DIR / "results" / "climate_horizon" / "correction_2026-09-30" / "routing_smoke"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fm = load_pipeline().folds_module
    calendar = fm.load_calendar()
    tensors = fm.load_tensors(VARIANT)
    channels = split_channels(tensors["feature_names"])
    fold = extension_fold({f["fold_id"]: f for f in build_benchmark_folds(fm)}[9])
    arrays = build_arrays(tensors, months_for(fm), fold, fm, channels,
                          last_target_period=development_last_period(calendar))
    measured = json.loads((PROJECT_DIR / "results/climate_horizon/weights_final/fold9_shuffle.json").read_text())["weights"]
    report = {"routing_version": ROUTING_VERSION, "fold": 9, "seed": 0, "epochs": 5, "scored": "validation 2024"}
    for arm, weights in (("B", (1, 1, 1, 1)), ("D", measured), ("F", (0, 0, 0, 1))):
        model, info = tr.train(arrays, dict(tr.DEFAULTS), 0, device, weights, max_epochs=5)
        val = arrays["val"]
        result = tr.predict(model, val, device)
        live = val["mask"] == 1
        head = model.climate_delta.weight.detach().abs().amax(dim=1).cpu().numpy()
        report[arm] = {
            "weights": list(weights),
            "finite": bool(np.isfinite(result["mu"]).all() and np.isfinite(result["prediction"]).all()),
            "val_nll": info["val_loss"],
            "val_mae_by_h": [float(np.abs(result["prediction"] - val["y"])[..., h][live[..., h]].mean()) for h in range(4)],
            "climate_head_row_max_abs": head.round(5).tolist(),
            "mean_abs_gated_climate_by_h": [float(np.abs(result["gated_climate"][..., h])[live[..., h]].mean()) for h in range(4)],
            "empty_batches": int(sum(g.get("empty_batch", False) for g in info["gradient_log"])),
            "climate_steps_skipped": int(sum(g["climate_skipped"] for g in info["gradient_log"])),
        }
    (OUT / "routing_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

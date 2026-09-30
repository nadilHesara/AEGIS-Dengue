"""
Minimal corrected (routing v2) development check before the reruns.

Fold 9, seed 0, full training budget, arms B, D (frozen fold-9 weights) and F
(h=4 only). Scored on the 2024 validation year only. Also tests whether B
under v2 reproduces the saved v1 B model (fold 9, seed 0), which is the
evidence needed to reuse the uniform-weight runs and pilots.

    python scripts/evaluation/53.climate_horizon_correction_dev_check.py
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

ROOT = PROJECT_DIR / "results" / "climate_horizon"
OUT = ROOT / "correction_2026-09-30" / "dev_check"


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
    val = arrays["val"]
    live = val["mask"] == 1
    measured = json.loads((ROOT / "weights_final" / "fold9_shuffle.json").read_text())["weights"]

    report = {"routing_version": ROUTING_VERSION, "fold": 9, "seed": 0, "scored": "validation 2024 only"}
    predictions = {}
    for arm, weights in (("B", (1, 1, 1, 1)), ("D", measured), ("F", (0, 0, 0, 1))):
        model, info = tr.train(arrays, dict(tr.DEFAULTS), 0, device, weights,
                               checkpoint_path=OUT / f"{arm}_fold9_seed0.pt")
        result = tr.predict(model, val, device)
        predictions[arm] = result
        report[arm] = {
            "weights": list(weights), "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
            "val_nll": info["val_loss"],
            "val_mae_by_h": [float(np.abs(result["prediction"] - val["y"])[..., h][live[..., h]].mean()) for h in range(4)],
            "finite": bool(np.isfinite(result["mu"]).all()),
            "climate_head_row_max_abs": model.climate_delta.weight.detach().abs().amax(1).cpu().numpy().round(4).tolist(),
            "empty_batches": int(sum(g.get("empty_batch", False) for g in info["gradient_log"])),
        }

    # Equivalence of B under v1 vs v2: the v1 sweep checkpoint for the same unit.
    v1_model, v1_saved = tr.load_checkpoint(ROOT / "checkpoints" / "sweep" / "B_target_equal" / "fold9_seed0.pt", device)
    v1 = tr.predict(v1_model, val, device)
    v2 = predictions["B"]
    v1_log = v1_saved["training_state"]["gradient_log"]
    report["B_v1_vs_v2"] = {
        "v1_best_epoch": v1_saved["training_state"]["best_epoch"],
        "v2_best_epoch": report["B"]["best_epoch"],
        "max_abs_mu_difference": float(np.abs(v1["mu"] - v2["mu"]).max()),
        "max_abs_median_difference": float(np.abs(v1["prediction"] - v2["prediction"]).max()),
        "v1_val_nll": v1_saved["training_state"]["best_loss"], "v2_val_nll": report["B"]["val_nll"],
        "v1_zero_loss_steps": int(sum(1 for g in v1_log if g["loss_equal"] == 0)),
        "v1_steps": len(v1_log), "v2_steps": len(tr.load_checkpoint(OUT / "B_fold9_seed0.pt", device)[1]
                                                     ["training_state"]["gradient_log"]),
    }
    checks = {
        "all_finite": all(report[a]["finite"] for a in "BDF"),
        "F_heads_learn_h1_3": min(report["F"]["climate_head_row_max_abs"][:3]) > 0,
        "no_empty_batches": all(report[a]["empty_batches"] == 0 for a in "BDF"),
        "B_reproduces_v1": report["B_v1_vs_v2"]["max_abs_mu_difference"] < 1e-3,
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())
    (OUT / "dev_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("B_v1_vs_v2", "checks", "passed")}, indent=1))
    for a in "BDF":
        print(a, round(report[a]["val_nll"], 4), [round(x, 2) for x in report[a]["val_mae_by_h"]], report[a]["climate_head_row_max_abs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

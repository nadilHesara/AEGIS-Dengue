"""
Run-status table for the routing-v2 correction.

    python scripts/evaluation/55.climate_horizon_run_status.py

Lists every expected (arm, fold, seed) unit, where its predictions come from
(reused v1 or corrected v2), and whether it is done, failed or missing.
Writes results/climate_horizon/correction_2026-09-30/run_status.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.training import climate_horizon_arms as spec  # noqa: E402

ROOT = PROJECT_DIR / "results" / "climate_horizon"
CORR = ROOT / "correction_2026-09-30"


def main() -> int:
    manifest = json.loads((CORR / "correction_manifest.json").read_text())
    rows = []
    for arm in spec.ARMS:
        rerun = arm in manifest["reruns"]
        root = (CORR if rerun else ROOT) / "predictions" / arm
        folds = list(spec.RETROSPECTIVE_FOLDS) + ([spec.HOLDOUT_FOLD] if spec.ARMS[arm][3] else [])
        for fold in folds:
            for seed in spec.seeds_for(arm):
                unit = root / f"fold{fold}_seed{seed}"
                meta = json.loads(unit.with_suffix(".json").read_text()) if unit.with_suffix(".json").exists() else {}
                if unit.with_suffix(".parquet").exists() and meta:
                    status = "done"
                elif unit.with_suffix(".error.json").exists():
                    status = "error"
                else:
                    status = "missing"
                rows.append({"arm": arm, "fold": fold, "seed": seed,
                             "source": "rerun (routing v2)" if rerun else "reused (v1 = v2 for equal weights)",
                             "status": status, "routing_version": meta.get("routing_version", "v1 (pre-correction)"),
                             "seconds": meta.get("seconds"), "peak_gpu_mb": meta.get("peak_gpu_mb"),
                             "path": str(unit.relative_to(PROJECT_DIR))})
    table = pd.DataFrame(rows)
    table.to_csv(CORR / "run_status.csv", index=False)
    print(table.groupby(["source", "status"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

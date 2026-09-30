"""
Regenerate scripts/17's climate-to-dengue lag scan for the extension, using
training data only.

scripts/17 fits each fold on `period_id <= fit_end_period`, which includes
the validation year. Here each fold's history is cut at `train_end_period`
instead, and the output goes to results/climate_horizon/diagnostics/ (the
original results/eda/ path is not written). The scan's own functions are
reused unchanged.

The peaks are **descriptive correlations** on the Open-Meteo reconstruction.
They are not known biological delays; scripts/17 itself warns when a peak
correlation is below its resolvability threshold.

    python scripts/evaluation/40.climate_horizon_delay_scan.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT = PROJECT_DIR / "results" / "climate_horizon" / "diagnostics"


def main() -> int:
    spec = importlib.util.spec_from_file_location(
        "lag_scan", PROJECT_DIR / "scripts" / "evaluation" / "17.lag_correlation_scan.py")
    scan_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scan_module)

    tensors = scan_module.folds_module.load_tensors("v0")
    nodes = pd.read_csv(PROJECT_DIR / "data" / "processed" / "nodes.csv").sort_values("node_id")
    tensors["canonical_name"] = nodes["canonical_name"].to_numpy(dtype=object)
    names = list(tensors["feature_names"])
    features = {name: names.index(name) for name in scan_module.CLIMATE_FEATURES}
    weeks = scan_module.week_of_year(tensors["start_date"])

    folds = json.loads((PROJECT_DIR / "data" / "processed" / "folds.json").read_text())["folds"]
    for fold in folds:
        fold["fit_end_period"] = fold["train_end_period"]  # training data only

    scan = pd.concat([scan_module.scan_fold(tensors, weeks, fold, features, scan_module.DEFAULT_MAX_LAG)
                      for fold in folds], ignore_index=True)
    scan = scan_module.add_peaks(scan)
    alerts = scan_module.check_alignment(scan)

    OUT.mkdir(parents=True, exist_ok=True)
    scan.to_csv(OUT / "lag_correlation_train_only.csv", index=False)
    summaries = pd.concat([scan_module.feature_summary(scan, fold["fold_id"]).assign(fold_id=fold["fold_id"])
                           for fold in folds], ignore_index=True)
    summaries.to_csv(OUT / "lag_summary_by_fold.csv", index=False)
    (OUT / "alerts.txt").write_text("\n".join(alerts), encoding="utf-8")
    print(summaries[summaries["fold_id"] == max(f["fold_id"] for f in folds)].to_string(index=False))
    print("\n".join(alerts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

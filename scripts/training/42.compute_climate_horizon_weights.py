"""
Freeze fold-specific climate-gradient weights from the inner utility pilots.

    python scripts/training/42.compute_climate_horizon_weights.py

Reads results/climate_horizon/pilots/ only. Writes
results/climate_horizon/weights/fold{f}_{shuffle,case_only}.json and
weights_summary.csv. An existing weight file is never overwritten, so weights
stay frozen once written (delete one by hand to recompute it).
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.training import climate_horizon_weights as cw  # noqa: E402

PILOTS = PROJECT_DIR / "results" / "climate_horizon" / "pilots"
OUT = PROJECT_DIR / "results" / "climate_horizon" / "weights"


def main() -> int:
    import argparse

    global OUT
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--require-blocks", nargs="*", type=int, default=None,
                        help="only use these inner blocks, and require all of them for a fold")
    parser.add_argument("--require-seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    unit_paths = sorted(PILOTS.glob("fold*_block*_seed*.parquet"))
    if args.require_blocks is not None or args.require_seeds is not None:
        blocks = set(args.require_blocks or [])
        seeds = set(args.require_seeds or [])
        def keep(path):
            parts = path.stem.split("_")
            return int(parts[1][5:]) in blocks and int(parts[2][4:]) in seeds
        unit_paths = [p for p in unit_paths if keep(p)]
        complete = {}
        for p in unit_paths:
            complete.setdefault(int(p.stem.split("_")[0][4:]), set()).add(p.stem.split("_", 1)[1])
        needed = {f"block{b}_seed{s}" for b in blocks for s in seeds}
        unit_paths = [p for p in unit_paths if complete[int(p.stem.split("_")[0][4:])] == needed]
        skipped = sorted(f for f, units in complete.items() if units != needed)
        if skipped:
            print(f"incomplete pilot evidence, no weights written for folds {skipped}")
    frame = pd.concat([pd.read_parquet(p) for p in unit_paths], ignore_index=True)
    units = cw.unit_errors(frame)
    agg = cw.aggregate(units)
    meta = {p.stem: json.loads(p.with_suffix(".json").read_text()) for p in unit_paths}

    rows = []
    for fold_id, fold_agg in agg.groupby("fold_id"):
        fold_agg = fold_agg.sort_values("horizon")
        sources = {k: v for k, v in meta.items() if v["fold_id"] == fold_id}
        for utility, column, gain in (("shuffle", "u_shuffle", "gain_shuffle"),
                                      ("case_only", "u_case_only", "gain_case_only_rel")):
            weights, status = cw.weights_from_utilities(fold_agg[column].to_numpy())
            record = {
                "rule": cw.RULE_VERSION, "utility": utility, "fold_id": int(fold_id),
                "rho": cw.RHO, "status": status,
                "weights": [round(float(w), 6) for w in weights],
                "mean_weight": float(weights.mean()),
                "distance_from_uniform": cw.distance_from_uniform(weights),
                "signed_gain": fold_agg[gain].round(6).tolist(),
                "utility_clipped": fold_agg[column].round(6).tolist(),
                "aggregated_errors": fold_agg[["horizon", "mae_real", "mae_shuffled", "mae_case_only",
                                               "cells", "units"]].to_dict("records"),
                "variability": fold_agg[["horizon", "gain_shuffle_unit_mean", "gain_shuffle_unit_sd",
                                         "gain_shuffle_unit_min", "gain_shuffle_unit_max",
                                         "gain_shuffle_units_positive", "gain_case_rel_unit_sd",
                                         "gain_case_rel_units_positive"]].to_dict("records"),
                "unit_errors": units[units["fold_id"] == fold_id].to_dict("records"),
                "source_runs": sorted(sources),
                "inner_blocks": {k: {"score_year": v["block"]["score_year"],
                                     "utility_periods": [v["block"]["test_start_period"], v["block"]["test_end_period"]],
                                     "early_stop_periods": [v["block"]["val_start_period"], v["block"]["val_end_period"]],
                                     "train_end_period": v["block"]["train_end_period"]} for k, v in sources.items()},
                "fingerprints": {k: v["fingerprints"] for k, v in sources.items()},
                "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "interpretation": "shuffle reliance of a frozen equal-weight model" if utility == "shuffle"
                                  else "incremental value over a size-matched case-only model (sensitivity arm only)",
            }
            path = OUT / f"fold{fold_id}_{utility}.json"
            if path.exists():
                record = json.loads(path.read_text())
                print(f"kept frozen {path.name}")
            else:
                path.write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
            rows.append({"fold_id": int(fold_id), "utility": utility, "status": record["status"],
                         **{f"w_h{h}": w for h, w in zip((1, 2, 3, 4), record["weights"])},
                         **{f"gain_h{h}": g for h, g in zip((1, 2, 3, 4), record["signed_gain"])},
                         "l1_from_uniform": record["distance_from_uniform"]["l1"],
                         "max_over_min": record["distance_from_uniform"]["max_over_min"]})

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "weights_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(agg.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

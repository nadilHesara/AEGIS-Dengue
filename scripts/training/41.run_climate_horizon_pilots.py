"""
Run utility pilots inside an outer fold's training period (resumable).

Each unit is one (outer fold, inner block, seed): an equal-weight two-branch
pilot, frozen and scored on the utility year with real climate and 5 donor
perturbations, plus a size-matched case-only pilot. Existing units are
skipped. See src/training/climate_horizon_pilots.py for the donor rule.

    python scripts/training/41.run_climate_horizon_pilots.py --folds 1 2 3 4 5 6 7 8 9 10

Outputs: results/climate_horizon/pilots/ (per-unit parquet + json, donor
mappings, utility_summary.csv). Never reads 2026 targets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.data.climate_horizon import VARIANT, extension_fold, inner_blocks, split_channels  # noqa: E402
from src.evaluation.climate_horizon import data_provenance  # noqa: E402
from src.evaluation.long_horizon import build_benchmark_folds, load_script, months_for  # noqa: E402
from src.training import climate_horizon as trainer  # noqa: E402
from src.training import climate_horizon_pilots as pilots  # noqa: E402

OUT = PROJECT_DIR / "results" / "climate_horizon" / "pilots"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--folds", nargs="+", type=int, default=[9],
                        help="outer folds 1-10; fold 10 pilots use only periods up to 2024")
    parser.add_argument("--blocks", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(pilots.PILOT_SEEDS))
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fm = load_script("build_folds", "features/14.build_folds.py")
    calendar = fm.load_calendar()
    months = months_for(fm)
    tensors = fm.load_tensors(VARIANT)
    channels = split_channels(tensors["feature_names"])
    all_folds = {f["fold_id"]: f for f in build_benchmark_folds(fm, include_holdout=True)}

    data = data_provenance()
    if not data["matches_manifest"].all():
        raise SystemExit("Input data do not match results/benchmark/data_manifest.json")
    provenance = {"data_manifest_match": True,
                  "variant": VARIANT, "channels": pilots.fingerprint(channels.names)}

    for fold_id in args.folds:
        fold = extension_fold(all_folds[fold_id])
        blocks = {b["block"]: b for b in inner_blocks(fold, calendar)}
        for k in args.blocks:
            block = blocks[k]
            for seed in args.seeds:
                path = pilots.run_unit(OUT, fold, block, seed, tensors, months, calendar, fm,
                                       channels, device, dict(trainer.DEFAULTS), provenance)
                print(f"fold {fold_id} block {k} (score {block['score_year']}) seed {seed}: {path.name}",
                      flush=True)

    frames = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("fold*_block*_seed*.parquet"))],
                       ignore_index=True)
    summary = pilots.utility_summary(frames)
    summary.to_csv(OUT / "utility_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

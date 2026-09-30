"""
Final sweep of the climate-horizon extension (frozen spec: src/training/climate_horizon_arms.py).

Retrospective (folds 1-9, test years 2017-2025):
    python scripts/training/45.run_climate_horizon_sweep.py --folds 1 2 3 4 5 6 7 8 9

2026 hold-out (fold 10). Refused unless the run manifest's hash matches and
--confirm-holdout is given. Only the declared arms (B, D) are scored:
    python scripts/training/45.run_climate_horizon_sweep.py --folds 10 --holdout --confirm-holdout

Each (arm, fold, seed) unit writes
results/climate_horizon/predictions/<arm>/fold{f}_seed{s}.{parquet,json} and is
skipped when both files exist (resumable). Test cells only; no tuning happens
here. Preprocessing is fitted on training years only; early stopping uses the
fold's validation year.
"""

from __future__ import annotations

import argparse
import hashlib
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
    VARIANT, build_arrays, development_last_period, extension_fold, split_channels,
)
from src.evaluation.climate_horizon import data_provenance  # noqa: E402
from src.evaluation.long_horizon import build_benchmark_folds, load_pipeline, months_for  # noqa: E402
from src.models.climate_horizon import ROUTING_VERSION  # noqa: E402
from src.training import climate_horizon as tr  # noqa: E402
from src.training import climate_horizon_arms as spec  # noqa: E402
from src.training.climate_horizon_pilots import array_fingerprint  # noqa: E402

OUT = PROJECT_DIR / "results" / "climate_horizon" / "predictions"
MANIFEST = PROJECT_DIR / "results" / "climate_horizon" / "run_manifest_2026-09-30.json"
MANIFEST_HASH = PROJECT_DIR / "results" / "climate_horizon" / "run_manifest_2026-09-30.sha256"
CHECKPOINTS = PROJECT_DIR / "results" / "climate_horizon" / "checkpoints" / "sweep"

# Routing-v2 correction (docs/climate_horizon_correction_audit.md): corrected runs
# go to their own dated directory and are checked against the correction manifest.
CORRECTION = PROJECT_DIR / "results" / "climate_horizon" / "correction_2026-09-30"
CORRECTION_MANIFEST = CORRECTION / "correction_manifest.json"
CORRECTION_HASH = CORRECTION / "correction_manifest.sha256"


def check_manifest(path: Path = MANIFEST, hash_path: Path = MANIFEST_HASH) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if hash_path.read_text().split()[0] != digest:
        raise SystemExit(f"{path.name} changed since it was frozen")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--folds", nargs="+", type=int, required=True)
    parser.add_argument("--arms", nargs="+", default=list(spec.ARMS), choices=list(spec.ARMS))
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--confirm-holdout", action="store_true")
    parser.add_argument("--correction", action="store_true",
                        help="routing-v2 reruns: write under correction_2026-09-30/, check the correction manifest")
    args = parser.parse_args()

    global OUT, CHECKPOINTS
    if args.correction:
        manifest_sha = check_manifest(CORRECTION_MANIFEST, CORRECTION_HASH)
        allowed = json.loads(CORRECTION_MANIFEST.read_text())["reruns"]
        OUT, CHECKPOINTS = CORRECTION / "predictions", CORRECTION / "checkpoints" / "sweep"
        for arm in args.arms:
            if arm not in allowed:
                raise SystemExit(f"{arm} is reused, not rerun, under the correction manifest")
    else:
        manifest_sha = check_manifest()
    if spec.HOLDOUT_FOLD in args.folds and not (args.holdout and args.confirm_holdout):
        raise SystemExit("fold 10 (2026) needs --holdout --confirm-holdout")
    if args.holdout and set(args.folds) != {spec.HOLDOUT_FOLD}:
        raise SystemExit("--holdout runs fold 10 only")
    if not data_provenance()["matches_manifest"].all():
        raise SystemExit("input data do not match the data manifest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = load_pipeline()
    fm = baseline.folds_module
    calendar = fm.load_calendar()
    months = months_for(fm)
    tensors = fm.load_tensors(VARIANT)
    channels = split_channels(tensors["feature_names"])
    folds = {f["fold_id"]: f for f in build_benchmark_folds(fm, include_holdout=True)}

    for fold_id in args.folds:
        fold = extension_fold(folds[fold_id])
        last = None if fold_id == spec.HOLDOUT_FOLD else development_last_period(calendar)
        arrays = build_arrays(tensors, months, fold, fm, channels, last_target_period=last)
        split_fp = {"train": array_fingerprint(arrays["train"]["y"], arrays["train"]["mask"]),
                    "val": array_fingerprint(arrays["val"]["target_period_id"], arrays["val"]["mask"]),
                    "test": array_fingerprint(arrays["test"]["target_period_id"], arrays["test"]["mask"])}
        for arm in args.arms:
            if fold_id == spec.HOLDOUT_FOLD and not spec.ARMS[arm][3]:
                continue
            weights, source = spec.weights_for(arm, fold_id)
            config = spec.config_for(arm, tr.DEFAULTS)
            for seed in spec.seeds_for(arm):
                unit = OUT / arm / f"fold{fold_id}_seed{seed}"
                if unit.with_suffix(".parquet").exists() and unit.with_suffix(".json").exists():
                    continue
                unit.parent.mkdir(parents=True, exist_ok=True)
                checkpoint = CHECKPOINTS / arm / f"fold{fold_id}_seed{seed}.pt"
                resume = checkpoint.exists() and not config.get("gradnorm")
                started = time.perf_counter()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                try:
                    model, info = tr.train(
                        arrays, config, seed, device, weights, checkpoint_path=checkpoint, resume=resume,
                        metadata={"arm": arm, "fold": fold, "weights_source": source,
                                  "manifest_sha256": manifest_sha, "split_fingerprints": split_fp,
                                  "feature_names": channels.names, "case_channels": channels.case_names(),
                                  "climate_channels": channels.climate_names(), "horizons": [1, 2, 3, 4]})
                    result = tr.predict(model, arrays["test"], device)
                except Exception as error:  # recorded, never silently dropped
                    (unit.with_suffix(".error.json")).write_text(json.dumps(
                        {"arm": arm, "fold_id": fold_id, "seed": seed, "error": repr(error),
                         "command": " ".join(sys.argv)}, indent=2), encoding="utf-8")
                    print(f"ERROR {arm} fold {fold_id} seed {seed}: {error!r}", flush=True)
                    continue
                frame = tr.prediction_frame(arm, fold_id, "test", arrays["test"], result, seed)
                frame.to_parquet(unit.with_suffix(".parquet"), index=False)
                log = pd.DataFrame(info["gradient_log"])
                meta = {"arm": arm, "fold_id": fold_id, "seed": seed, "weights": list(weights),
                        "routing_version": ROUTING_VERSION, "correction_run": args.correction,
                        "weights_source": source, "config": config, "manifest_sha256": manifest_sha,
                        "split_fingerprints": split_fp, "best_epoch": info["best_epoch"],
                        "epochs_run": info["epochs_run"], "val_nll": info["val_loss"],
                        "parameters": info["parameters"],
                        "raw_norm_climate_mean": float(log["raw_norm_climate"].mean()),
                        "raw_norm_other_mean": float(log["raw_norm_other"].mean()),
                        "clip_fraction": float((log["clip_coefficient"] < 1).mean()),
                        "climate_steps_skipped": int(log["climate_skipped"].sum()),
                        "seconds": time.perf_counter() - started, "device": str(device),
                        "resumed_from_checkpoint": resume, "checkpoint": str(checkpoint.relative_to(PROJECT_DIR)),
                        "peak_gpu_mb": (torch.cuda.max_memory_allocated(device) / 2 ** 20
                                        if device.type == "cuda" else None),
                        "test_cells": int((arrays["test"]["mask"] == 1).sum()),
                        "test_target_periods": [int(arrays["test"]["target_period_id"][arrays["test"]["mask"].any(axis=1)].min()),
                                                int(arrays["test"]["target_period_id"][arrays["test"]["mask"].any(axis=1)].max())]}
                unit.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
                print(f"{arm} fold {fold_id} seed {seed}: val nll {info['val_loss']:.4f} "
                      f"({meta['seconds']:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

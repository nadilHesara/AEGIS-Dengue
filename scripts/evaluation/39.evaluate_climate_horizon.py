"""
Evaluate the climate-horizon extension's comparison set.

Step 3 (data interface): builds the extension's eligible test cells, checks
the provenance of the benchmark reference methods, checks persistence under
the original protocol, and scores persistence and the references on the
extension's cell intersection. Extension arms are added in later steps: any
`results/climate_horizon/predictions/*.parquet` in `cell_frame` format is
picked up automatically.

Read-only inputs: data/processed, results/benchmark/predictions,
results/benchmark/data_manifest.json.
Outputs: results/climate_horizon/evaluation/*.csv and report.md.

    python scripts/evaluation/39.evaluate_climate_horizon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.data.climate_horizon import (  # noqa: E402
    HORIZONS, VARIANT, WINDOW, build_arrays, cell_keys, development_last_period,
    inner_blocks, split_channels,
)
from src.evaluation import climate_horizon as ev  # noqa: E402
from src.evaluation.long_horizon import load_pipeline, months_for, truth_frame  # noqa: E402

OUT = ev.OUTPUT_DIR / "evaluation"
EXTENSION_PREDICTIONS = ev.OUTPUT_DIR / "predictions"


def original_protocol_persistence(baseline) -> pd.DataFrame:
    """scripts/15 at each horizon: v0 windows, lookback 12, committed folds 1-9."""

    import json

    tensors = baseline.folds_module.load_tensors("v0")
    nodes = pd.read_csv(PROJECT_DIR / "data" / "processed" / "nodes.csv").sort_values("node_id")
    tensors["canonical_name"] = nodes["canonical_name"].to_numpy(dtype=object)
    folds = json.loads((PROJECT_DIR / "data" / "processed" / "folds.json").read_text())["folds"]
    rows = []
    for horizon in HORIZONS:
        metrics, _ = baseline.naive.evaluate_folds(tensors, folds, lookback=12, horizon=horizon)
        metrics = metrics[(metrics["model"] == "persistence") & metrics["headline"]]
        value = float(metrics["mae"].mean())
        rows.append({"horizon": horizon, "persistence_mae": round(value, 4),
                     "reference": ev.PERSISTENCE_REFERENCE[horizon],
                     "matches_2dp": round(value, 2) == ev.PERSISTENCE_REFERENCE[horizon],
                     "cells": int(metrics["cells"].sum())})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = load_pipeline()
    fm = baseline.folds_module
    calendar = fm.load_calendar()
    months = months_for(fm)
    tensors = fm.load_tensors(VARIANT)
    channels = split_channels(tensors["feature_names"])
    last_period = development_last_period(calendar)
    truth = truth_frame(tensors)

    import json
    committed = json.loads((PROJECT_DIR / "data" / "processed" / "folds.json").read_text())["folds"]
    folds = {int(f["fold_id"]): f for f in committed}

    # 1. Data provenance.
    data = ev.data_provenance()
    data.to_csv(OUT / "data_provenance.csv", index=False)
    print(data.to_string(index=False))

    # 2. Persistence under the original protocol.
    protocol = original_protocol_persistence(baseline)
    protocol.to_csv(OUT / "persistence_original_protocol.csv", index=False)
    print(protocol.to_string(index=False))

    # 3. Extension cells and inner blocks.
    cell_frames, block_rows = [], []
    for fold_id in ev.DEVELOPMENT_FOLDS:
        fold = folds[fold_id]
        arrays = build_arrays(tensors, months, fold, fm, channels,
                              last_target_period=last_period)
        cell_frames.append(cell_keys(arrays, fold_id))
        for block in inner_blocks(fold, calendar):
            block_rows.append(block)
    extension_cells = pd.concat(cell_frames, ignore_index=True)
    pd.DataFrame(block_rows).to_csv(OUT / "inner_blocks.csv", index=False)

    # 4. References: provenance and cells.
    manifest_time = pd.Timestamp(json.loads(ev.MANIFEST_PATH.read_text())["written_utc"])
    references, checks = {}, []
    for method in ev.REFERENCE_METHODS:
        frame = ev.load_reference(method)
        references[method] = frame
        checks.append(ev.check_reference(method, frame, folds, truth, manifest_time))
    checks = pd.DataFrame(checks)
    data_ok = bool(data["matches_manifest"].all())
    checks["data_hashes_match"] = data_ok
    if not data_ok:
        checks["comparison"] = "context only"
    checks.to_csv(OUT / "reference_provenance.csv", index=False)
    print(checks.to_string(index=False))

    extension_arms = {}
    if EXTENSION_PREDICTIONS.exists():
        for arm_dir in sorted(p for p in EXTENSION_PREDICTIONS.iterdir() if p.is_dir()):
            frame = pd.concat([pd.read_parquet(f) for f in sorted(arm_dir.glob("fold*_seed*.parquet"))],
                              ignore_index=True)
            # Retrospective folds only; 2026 (fold 10) is scored by a separate, declared step.
            frame = frame[(frame["split"] == "test") & frame["fold_id"].isin(ev.DEVELOPMENT_FOLDS)]
            extension_arms[arm_dir.name] = frame

    # 5. Intersection and coverage.
    sets = {"extension_eligible": extension_cells}
    sets.update({m: f[ev.KEYS].drop_duplicates() for m, f in references.items()
                 if checks.set_index("method").loc[m, "comparison"] == "direct"})
    sets.update({m: f[ev.KEYS].drop_duplicates() for m, f in extension_arms.items()})
    common = extension_cells
    for name, keys in sets.items():
        common = common.merge(keys, on=ev.KEYS, how="inner")
    coverage = []
    for name, keys in {**sets, **{m: f[ev.KEYS].drop_duplicates() for m, f in references.items()}}.items():
        for horizon in HORIZONS:
            n = int((keys["horizon"] == horizon).sum())
            c = int((common["horizon"] == horizon).sum())
            coverage.append({"set": name, "horizon": horizon, "cells": n, "common": c,
                             "coverage": c / n if n else np.nan})
    coverage = pd.DataFrame(coverage).drop_duplicates()
    coverage.to_csv(OUT / "coverage.csv", index=False)

    # 6. Score on the common cells.
    thresholds = {fid: baseline.naive.peak_thresholds(
        tensors["y"], tensors["y_mask"], tensors["period_id"] <= folds[fid]["fit_end_period"])
        for fid in ev.DEVELOPMENT_FOLDS}
    frames = [ev.persistence_on_cells(common, tensors, folds)]
    for method, frame in {**references, **extension_arms}.items():
        frames.append(frame.merge(common, on=ev.KEYS, how="inner"))
    metrics = ev.score(pd.concat(frames, ignore_index=True), truth, thresholds)
    metrics.to_csv(OUT / "per_fold_metrics.csv", index=False)
    summary = ev.headline(metrics)
    summary = summary.merge(checks[["method", "comparison"]], on="method", how="left")
    summary["comparison"] = summary["comparison"].fillna("direct")
    summary.to_csv(OUT / "headline.csv", index=False)
    pivot = summary.pivot(index="method", columns="horizon", values="mae").round(2)
    print(pivot.to_string())

    lines = [
        "# Climate-horizon extension: comparison set (step 3)", "",
        "Generated by `scripts/evaluation/39.evaluate_climate_horizon.py`. Benchmark files read only.", "",
        "## Data provenance (hash vs results/benchmark/data_manifest.json)", "",
        data.to_markdown(index=False), "",
        "## Persistence under the original protocol (scripts/15, lookback 12, folds 1-9)", "",
        protocol.to_markdown(index=False), "",
        "## Reference provenance", "", checks.to_markdown(index=False), "",
        f"## Extension cells (window {WINDOW}, horizons {list(HORIZONS)}, targets <= period {last_period})", "",
        coverage.to_markdown(index=False), "",
        "## Headline MAE on the common cells (folds 1,2,3,6,7,8,9)", "",
        pivot.to_markdown(), "",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

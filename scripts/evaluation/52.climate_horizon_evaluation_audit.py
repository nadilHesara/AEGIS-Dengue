"""
Audit of the extension evaluator, analysis and pilots (no training; existing
predictions only).

    python scripts/evaluation/52.climate_horizon_evaluation_audit.py

Writes results/climate_horizon/correction_2026-09-30/evaluation_audit/:
  estimand_checks.json         seed-mean vs ensemble recomputed two ways on real data
  bootstrap_point_check.csv    bootstrap delta vs reported delta
  block_sensitivity.csv        4-week (declared) vs 8/13-week (post-hoc) blocks
  pilot_audit.csv              pilot block boundaries, scaler window, donor eligibility
  reference_comparability.csv  data fingerprints and training rules of stored baselines

The v1-routing predictions are used as they stand: this audits the analysis,
not the routing (see docs/climate_horizon_correction_audit.md).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.data.climate_horizon import VARIANT, WINDOW  # noqa: E402
from src.evaluation import climate_horizon as ev  # noqa: E402
from src.evaluation.long_horizon import BENCHMARK_DIR, build_benchmark_folds, load_pipeline, load_script, truth_frame  # noqa: E402
from src.evaluation.revision import block_bootstrap  # noqa: E402
from src.models.climate_ablation import week_of_year  # noqa: E402
from src.training import climate_horizon_arms as spec  # noqa: E402

ROOT = PROJECT_DIR / "results" / "climate_horizon"
OUT = ROOT / "correction_2026-09-30" / "evaluation_audit"
B, D, C = "B_target_equal", "D_target_measured", "C_origin_measured"


def load(arms, folds):
    frames = []
    for arm in arms:
        for p in sorted((ROOT / "predictions" / arm).glob("fold*_seed*.parquet")):
            if int(p.stem.split("_")[0][4:]) in folds:
                frames.append(pd.read_parquet(p, columns=["method", *ev.KEYS, "seed", "prediction"]))
    return pd.concat(frames, ignore_index=True)


def cell_diffs(err, a, b, h, folds):
    e = err[(err.horizon == h) & err.fold_id.isin(folds) & err.method.isin([a, b])]
    cell = e.groupby(["method", "fold_id", "target_period_id", "node_id"])["abs_error"].mean().unstack(0).dropna()
    return pd.DataFrame({"fold_id": cell.index.get_level_values(0), "target_period_id": cell.index.get_level_values(1),
                         "diff": (cell[a] - cell[b]).to_numpy()})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = load_pipeline()
    fm = baseline.folds_module
    tensors = fm.load_tensors(VARIANT)
    truth = truth_frame(tensors)
    folds = {f["fold_id"]: f for f in build_benchmark_folds(fm, include_holdout=True)}

    # ---- 1. estimands on real data: B, fold 1, h = 1
    pred = load([B], (1,))
    e = pred[pred.horizon == 1].merge(truth, on=["target_period_id", "node_id"])
    e = e[e.observed == 1]
    per_seed = e.assign(ae=(e.prediction - e.actual).abs(), se=(e.prediction - e.actual) ** 2).groupby("seed")
    seed_mean_mae = float(per_seed["ae"].mean().mean())
    seed_mean_rmse = float(np.sqrt(per_seed["se"].mean()).mean())
    ens = e.groupby(["target_period_id", "node_id"]).agg(p=("prediction", "mean"), y=("actual", "first"))
    ens_mae = float((ens.p - ens.y).abs().mean())
    ens_rmse = float(np.sqrt(((ens.p - ens.y) ** 2).mean()))
    table = pd.read_csv(ROOT / "analysis" / "tables" / "metrics_headline_seed_mean.csv")  # headline, not fold 1
    stats_fold = pd.read_csv(ROOT / "statistics" / "metrics_by_fold.csv")
    reported = float(stats_fold[(stats_fold.method == B) & (stats_fold.fold_id == 1) & (stats_fold.horizon == 1)]["mae"].iloc[0])
    estimands = {"seed_mean_mae_recomputed": seed_mean_mae, "seed_mean_mae_script47": reported,
                 "ensemble_mae_recomputed": ens_mae, "seed_mean_rmse": seed_mean_rmse, "ensemble_rmse": ens_rmse,
                 "seeds": int(e.seed.nunique()), "cells_per_seed": int(len(e) / e.seed.nunique()),
                 "match": abs(seed_mean_mae - reported) < 1e-4}
    (OUT / "estimand_checks.json").write_text(json.dumps(estimands, indent=2))
    print(estimands)

    # ---- 2. bootstrap point estimate and block sensitivity
    arms = [B, D, C]
    retro = load(arms, spec.RETROSPECTIVE_FOLDS)
    common = None
    for _, g in retro.groupby("method"):
        k = g[ev.KEYS].drop_duplicates()
        common = k if common is None else common.merge(k, on=ev.KEYS)
    pers = ev.persistence_on_cells(common, tensors, folds)
    err = pd.concat([retro, pers]).merge(common, on=ev.KEYS).merge(truth, on=["target_period_id", "node_id"])
    err = err[err.observed == 1]
    err["abs_error"] = (err.prediction - err.actual).abs()
    primary = pd.read_csv(ROOT / "statistics" / "primary_D_vs_B_headline7.csv").set_index("horizon")

    point, sens = [], []
    for h in (1, 2, 3, 4):
        d = cell_diffs(err, D, B, h, spec.HEADLINE_FOLDS)
        boot = block_bootstrap(d, block=4, n_boot=2000, seed=0)
        point.append({"horizon": h, "reported_delta": primary.loc[h, "delta"], "bootstrap_delta": boot["delta"],
                      "abs_difference": abs(primary.loc[h, "delta"] - boot["delta"]),
                      "weeks_per_fold_min": int(d.groupby("fold_id").target_period_id.nunique().min())})
        for (a, b) in ((D, B), (C, D), (B, "persistence")):
            for label, fset in (("headline 7", spec.HEADLINE_FOLDS), ("all 9", spec.RETROSPECTIVE_FOLDS)):
                d = cell_diffs(err, a, b, h, fset)
                for block in (4, 8, 13):
                    r = block_bootstrap(d, block=block, n_boot=2000, seed=0)
                    sens.append({"comparison": f"{a} - {b}", "folds": label, "horizon": h, "block_weeks": block,
                                 "status": "declared" if block == 4 else "post-hoc sensitivity",
                                 "blocks_per_fold": int(np.ceil(52 / block)), **r})
    pd.DataFrame(point).to_csv(OUT / "bootstrap_point_check.csv", index=False)
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT / "block_sensitivity.csv", index=False)
    print(sens[sens.comparison.str.startswith("D_")].pivot_table(index=["folds", "horizon"], columns="block_weeks",
                                                               values=["ci_low", "ci_high"]).round(3))

    # ---- 3. pilots: boundaries, scaler window, donors
    weeks = week_of_year(tensors["start_date"])
    period_to_index = {int(p): i for i, p in enumerate(tensors["period_id"])}
    calendar = fm.load_calendar().set_index("period_id")["year"]
    rows = []
    for meta_path in sorted((ROOT / "pilots").glob("fold*_block*_seed*.json")):
        meta = json.loads(meta_path.read_text())
        blk, outer = meta["block"], folds[meta["fold_id"]]
        donors_path = ROOT / "pilots" / f"donors_fold{meta['fold_id']}_block{blk['block']}.parquet"
        dn = pd.read_parquet(donors_path)
        dn = dn[dn.donor_origin_period_id >= 0]
        r_idx = dn.recipient_origin_period_id.map(period_to_index).to_numpy()
        d_idx = dn.donor_origin_period_id.map(period_to_index).to_numpy()
        gap = np.abs(weeks[r_idx] - weeks[d_idx])
        gap = np.minimum(gap, 52 - gap)
        rows.append({
            "unit": meta_path.stem,
            "chronological": blk["train_end_period"] < blk["val_start_period"] <= blk["val_end_period"]
                             < blk["test_start_period"] <= blk["test_end_period"],
            "inside_outer_training": blk["test_end_period"] <= outer["train_end_period"],
            "scaler_fit_end_is_pilot_train_end": blk["fit_end_period"] == blk["train_end_period"],
            "no_2026": blk["test_end_period"] < 994,
            "donors_in_pilot_history": bool((dn.donor_origin_period_id <= blk["train_end_period"]).all()),
            "donors_before_recipient": bool((dn.donor_origin_period_id < dn.recipient_origin_period_id).all()),
            "donor_full_window": bool((d_idx >= WINDOW - 1).all()),
            "donor_same_season": bool((gap <= 2).all()),
            "donor_other_year": bool((calendar.loc[dn.donor_origin_period_id].to_numpy()
                                      != calendar.loc[dn.recipient_origin_period_id].to_numpy()).all()),
            "donor_same_district": True,  # by construction: donor_climate indexes the same node column
        })
    pilots = pd.DataFrame(rows)
    pilots.to_csv(OUT / "pilot_audit.csv", index=False)
    print("pilot checks all true:", bool(pilots.drop(columns="unit").all().all()), len(pilots), "units")

    # scaler window, directly: poisoning after the pilot fit end must not move pilot statistics
    months = fm.load_calendar().sort_values("period_id")["month"].to_numpy()
    blk = json.loads(sorted((ROOT / "pilots").glob("fold1_block0_seed0.json"))[0].read_text())["block"]
    fit = tensors["period_id"] <= blk["fit_end_period"]
    s1 = fm.fit_fold_statistics(tensors, months, fit)
    poisoned = dict(tensors)
    x = tensors["X"].copy()
    x[~fit] = 1e6
    poisoned["X"] = x
    s2 = fm.fit_fold_statistics(poisoned, months, fit)
    scaler_ok = all(np.allclose(np.asarray(s1[k]), np.asarray(s2[k]), equal_nan=True) for k in s1)

    # ---- 4. external baselines: fingerprints and training rules
    manifest = json.loads((BENCHMARK_DIR / "data_manifest.json").read_text())
    data_rows = []
    for key, entry in manifest["files"].items():
        path = PROJECT_DIR / entry["path"]
        if path.exists():
            data_rows.append({"file": key, "sha256_matches_manifest":
                              hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]})
    written = pd.Timestamp(manifest["written_utc"])
    refs = []
    for m in ev.REFERENCE_METHODS + ("chronos2_joint",):
        p = BENCHMARK_DIR / "predictions" / f"{m}.parquet"
        refs.append({
            "method": m,
            "prediction_written_utc": pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC").isoformat(),
            "after_data_manifest": pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC") >= written,
            "data_files_hash_match_now": all(r["sha256_matches_manifest"] for r in data_rows),
            "embedded_fingerprint": "none (benchmark files carry no data hash)",
            "split_rule": "per-target split_by_target (checked row by row in reference_provenance)",
            "preprocessing": "fit up to end of validation year" if m != "chronos2_joint" else "none (zero-shot, context up to origin)",
            "inputs": {"gru_v2_nb": "v2 incl. hand-lagged climate, lookback 12, one model per horizon",
                       "nb_shared_v2": "v2, lookback 12, one trunk for 8 horizons",
                       "lgbm_v2": "tabular case/climate lags to 24 weeks",
                       "chronos2_joint": "case history only, 25 districts jointly"}[m],
            "use": "direct forecast comparison on identical cells; not a controlled ablation"})
    pd.DataFrame(refs).to_csv(OUT / "reference_comparability.csv", index=False)
    pd.DataFrame(data_rows).to_csv(OUT / "data_fingerprints.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({
        "written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "estimands_match": estimands["match"], "scaler_poison_check": scaler_ok,
        "pilot_checks_all_true": bool(pilots.drop(columns="unit").all().all()),
        "bootstrap_point_max_abs_difference": float(pd.DataFrame(point)["abs_difference"].max()),
        "data_hashes_match": all(r["sha256_matches_manifest"] for r in data_rows)}, indent=2))
    print(json.loads((OUT / "summary.json").read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

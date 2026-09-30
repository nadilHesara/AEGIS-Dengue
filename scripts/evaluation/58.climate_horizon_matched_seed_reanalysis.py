"""
Reanalysis of existing routing-v2 predictions (no training):

1. Exploratory arms vs B on MATCHED seeds {0, 1, 2} and matched cells
   (fold, district, origin = target - h, target, horizon, observed mask).
   The corrected `controls_vs_B` table compared each arm's own seed set
   (B: 0-4, others: 0-2); this recomputes every exploratory contrast on
   the three seed IDs all arms share. Unadjusted, exploratory.
2. Clipping over COMPLETE training logs (every saved unit), not only the
   sampled diagnostic steps.
3. Angle aggregation for the post-hoc gradient diagnostics.
4. Properties of the 2026 interval (blocks, level, adjustment).

    python scripts/evaluation/58.climate_horizon_matched_seed_reanalysis.py

Outputs: results/climate_horizon/correction_2026-09-30/matched_seed/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.data.climate_horizon import VARIANT  # noqa: E402
from src.evaluation import climate_horizon as ev  # noqa: E402
from src.evaluation.long_horizon import load_pipeline, truth_frame  # noqa: E402
from src.evaluation.revision import block_bootstrap  # noqa: E402
from src.training import climate_horizon_arms as spec  # noqa: E402

ROOT = PROJECT_DIR / "results" / "climate_horizon"
CORR = ROOT / "correction_2026-09-30"
OUT = CORR / "matched_seed"
SEEDS = (0, 1, 2)
RERUNS = json.loads((CORR / "correction_manifest.json").read_text())["reruns"]


def root_of(arm):
    return CORR if arm in RERUNS else ROOT


def load(arm, folds, seeds=SEEDS):
    frames = [pd.read_parquet(p, columns=["method", *ev.KEYS, "seed", "prediction"])
              for p in sorted((root_of(arm) / "predictions" / arm).glob("fold*_seed*.parquet"))
              if int(p.stem.split("_")[0][4:]) in folds and int(p.stem.split("_seed")[1]) in seeds]
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tensors = load_pipeline().folds_module.load_tensors(VARIANT)
    truth = truth_frame(tensors)

    # ---- 1. matched-seed exploratory contrasts
    arms = list(spec.ARMS)
    frames = {a: load(a, spec.RETROSPECTIVE_FOLDS) for a in arms}
    seeds_present = {a: sorted(f.seed.unique().tolist()) for a, f in frames.items()}
    common = None
    for f in frames.values():
        k = f[ev.KEYS].drop_duplicates()
        common = k if common is None else common.merge(k, on=ev.KEYS)
    data = pd.concat(frames.values()).merge(common, on=ev.KEYS).merge(truth, on=["target_period_id", "node_id"])
    data = data[data.observed == 1]
    data["ae"] = (data.prediction - data.actual).abs()
    # per cell, per arm: mean over the SAME seeds {0,1,2}
    counts = data.groupby(["method", "fold_id", "horizon", "target_period_id", "node_id"]).seed.nunique()
    assert (counts == len(SEEDS)).all(), "an arm is missing a seed at some cell"
    cell = data.groupby(["method", "fold_id", "horizon", "target_period_id", "node_id"])["ae"].mean().unstack(0)

    rows, means = [], []
    pairs = [(a, "B_target_equal") for a in arms if a != "B_target_equal"] + [("C_origin_measured", "D_target_measured")]
    for fold_label, folds in (("headline 7", spec.HEADLINE_FOLDS), ("all 9", spec.RETROSPECTIVE_FOLDS)):
        for h in (1, 2, 3, 4):
            c = cell.xs(h, level="horizon")
            c = c[c.index.get_level_values("fold_id").isin(folds)]
            per_fold = c.groupby(level="fold_id").mean()
            for arm in arms:
                means.append({"folds": fold_label, "horizon": h, "arm": arm, "seeds": "0,1,2",
                              "mae_matched": float(per_fold[arm].mean())})
            for a, b in pairs:
                errors = pd.DataFrame({"fold_id": c.index.get_level_values("fold_id"),
                                       "target_period_id": c.index.get_level_values("target_period_id"),
                                       "diff": (c[a] - c[b]).to_numpy()})
                boot = block_bootstrap(errors, block=4, n_boot=2000, seed=0)
                d = per_fold[a] - per_fold[b]
                w = stats.wilcoxon(d) if (d != 0).any() else None
                rows.append({"folds": fold_label, "a": a, "b": b, "horizon": h, "seeds": "0,1,2 (both)",
                             "mae_a": float(per_fold[a].mean()), "mae_b": float(per_fold[b].mean()),
                             "delta": boot["delta"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                             "folds_a_better": int((d < 0).sum()), "n_folds": len(d),
                             "p_t_unadjusted": float(stats.ttest_1samp(d, 0).pvalue),
                             "p_wilcoxon_unadjusted": float(w.pvalue) if w else np.nan,
                             "status": "exploratory, unadjusted"})
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(OUT / "matched_seed_contrasts.csv", index=False)
    pd.DataFrame(means).to_csv(OUT / "matched_seed_arm_means.csv", index=False)

    # ---- 1b. block-length sensitivity for the CORRECTED primary D-B (5 seeds, headline 7)
    five = {a: load(a, spec.HEADLINE_FOLDS, seeds=spec.PRIMARY_SEEDS) for a in ("B_target_equal", "D_target_measured")}
    k5 = five["B_target_equal"][ev.KEYS].drop_duplicates().merge(five["D_target_measured"][ev.KEYS].drop_duplicates(), on=ev.KEYS)
    d5 = pd.concat(five.values()).merge(k5, on=ev.KEYS).merge(truth, on=["target_period_id", "node_id"])
    d5 = d5[d5.observed == 1]
    d5["ae"] = (d5.prediction - d5.actual).abs()
    c5 = d5.groupby(["method", "fold_id", "horizon", "target_period_id", "node_id"])["ae"].mean().unstack(0)
    sens = []
    for h in (1, 2, 3, 4):
        c = c5.xs(h, level="horizon")
        errors = pd.DataFrame({"fold_id": c.index.get_level_values("fold_id"),
                               "target_period_id": c.index.get_level_values("target_period_id"),
                               "diff": (c["D_target_measured"] - c["B_target_equal"]).to_numpy()})
        for block in (4, 8, 13):
            sens.append({"horizon": h, "block_weeks": block,
                         "status": "declared" if block == 4 else "post-hoc sensitivity",
                         **block_bootstrap(errors, block=block, n_boot=2000, seed=0)})
    pd.DataFrame(sens).to_csv(OUT / "primary_block_sensitivity_v2.csv", index=False)
    print(pd.DataFrame(sens).round(3).to_string(index=False))

    # ---- 2. clipping in complete training logs (unit metadata carries the per-run fraction)
    clip = []
    for arm in arms:
        for p in (root_of(arm) / "predictions" / arm).glob("fold*_seed*.json"):
            m = json.loads(p.read_text())
            clip.append({"arm": arm, "fold": m["fold_id"], "seed": m["seed"], "clip_fraction": m["clip_fraction"]})
    clip = pd.DataFrame(clip)
    clip_summary = clip.groupby("arm").agg(runs=("clip_fraction", "size"),
                                           runs_with_any_clipping=("clip_fraction", lambda x: int((x > 0).sum())),
                                           mean_fraction=("clip_fraction", "mean"),
                                           max_fraction=("clip_fraction", "max")).reset_index()
    clip_summary.to_csv(OUT / "clipping_full_training_logs.csv", index=False)

    # ---- 3. angles from the diagnostic comparisons
    diag = pd.read_csv(CORR / "diagnostics_posthoc" / "paired_step_comparisons.csv")
    ang = {}
    for label, col in (("raw_gradient", "raw_climate_cosine"), ("optimizer_update", "update_climate_cosine")):
        c = diag[col].clip(-1, 1)
        ang[label] = {"mean_cosine": float(c.mean()),
                      "angle_of_mean_cosine_deg": float(np.degrees(np.arccos(c.mean()))),
                      "mean_angle_deg": float(np.degrees(np.arccos(c)).mean()),
                      "median_angle_deg": float(np.median(np.degrees(np.arccos(c)))),
                      "min_angle_deg": float(np.degrees(np.arccos(c.max()))),
                      "max_angle_deg": float(np.degrees(np.arccos(c.min())))}
    rel = diag["update_climate_rel_diff"]
    ang["update_relative_difference"] = {
        "definition": "||u_measured - u_equal||_2 / ||u_equal||_2 over all climate-encoder parameters, "
                      "one Adam step, per paired comparison",
        "mean": float(rel.mean()), "median": float(rel.median()), "min": float(rel.min()), "max": float(rel.max()),
        "n": int(len(rel))}
    ang["sampled_clipping"] = {"comparisons": int(len(diag)),
                               "clipped_equal": int(diag.clipped_equal.sum()),
                               "clipped_measured": int(diag.clipped_measured.sum())}

    # ---- 4. 2026 interval properties
    hb = pd.read_csv(CORR / "holdout_2026" / "bootstrap_2026.csv")
    weeks = 19
    ang["holdout_2026_interval"] = {
        "method": "src/evaluation/revision.block_bootstrap: percentile interval, 2000 resamples, seed 0",
        "block_weeks": 4, "weeks": weeks, "blocks": int(np.ceil(weeks / 4)),
        "level": "95% two-sided, pointwise per horizon and comparison",
        "multiplicity_adjustment": "none",
        "D_minus_B": hb[(hb.a == "D_target_measured") & (hb.b == "B_target_equal")][
            ["horizon", "delta", "ci_low", "ci_high"]].round(3).to_dict("records")}
    (OUT / "diagnostic_facts.json").write_text(json.dumps(ang, indent=2), encoding="utf-8")

    pd.set_option("display.width", 220)
    print("seeds present:", seeds_present, "| cells per horizon:", int(len(cell) / 4))
    print(contrasts[contrasts.folds == "headline 7"][["a", "horizon", "mae_a", "mae_b", "delta", "ci_low", "ci_high",
                                                     "folds_a_better", "p_wilcoxon_unadjusted"]].round(3).to_string(index=False))
    print(clip_summary.round(4).to_string(index=False))
    print(json.dumps(ang, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

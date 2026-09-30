"""
Primary statistics for the climate-horizon extension (spec frozen in the run manifest).

Retrospective (folds 1-9):
    python scripts/evaluation/47.climate_horizon_statistics.py
2026 hold-out (fold 10, B and D only; run after the retrospective report):
    python scripts/evaluation/47.climate_horizon_statistics.py --holdout

Reads results/climate_horizon/predictions/<arm>/fold*_seed*.parquet and the
benchmark references read-only (gru_v2_nb, nb_shared_v2, lgbm_v2; h = 1-4).
Scores every method on the intersection of their cells, and recomputes
persistence on that support. Writes results/climate_horizon/statistics/
(retrospective) or results/climate_horizon/holdout_2026/ (fold 10).

Primary: D_target_measured vs B_target_equal, headline MAE (folds 1,2,3,6,7,8,9)
per horizon. Statistics: paired t-test and Wilcoxon over folds on seed-mean
MAE; Holm across the 4 horizons; 4-week time-block bootstrap 95% CI of the
seed-mean difference (src/evaluation/revision.block_bootstrap).
Decision: improvement if D - B <= -0.30 at h = 3 or 4 with Holm p < 0.05 and
CI < 0; non-inferior at h = 1, 2 if CI upper <= +0.30; otherwise no
measurable difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.data.climate_horizon import HORIZONS, VARIANT  # noqa: E402
from src.evaluation import climate_horizon as ev  # noqa: E402
from src.evaluation.long_horizon import (  # noqa: E402
    PREDICTIONS_DIR, QUANTILES, build_benchmark_folds, load_pipeline, nb_quantiles, truth_frame,
    weighted_interval_score,
)
from src.evaluation.revision import block_bootstrap, holm  # noqa: E402
from src.training import climate_horizon_arms as spec  # noqa: E402

ROOT = PROJECT_DIR / "results" / "climate_horizon"
PRIMARY, BASELINE = "D_target_measured", "B_target_equal"
MARGIN = 0.30
N_BOOT = 2000


def load_arms(folds: tuple[int, ...], arms) -> pd.DataFrame:
    frames = []
    for arm in arms:
        paths = sorted((ROOT / "predictions" / arm).glob("fold*_seed*.parquet"))
        paths = [p for p in paths if int(p.stem.split("_")[0][4:]) in folds]
        frames += [pd.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_refs(folds, split="test") -> pd.DataFrame:
    frames = []
    for method in ev.REFERENCE_METHODS:
        f = pd.read_parquet(PREDICTIONS_DIR / f"{method}.parquet", columns=["method", *ev.KEYS, "seed", "prediction"])
        frames.append(f[(f["method"] == method) & (f["split"] == split) & f["fold_id"].isin(folds)
                        & f["horizon"].isin(HORIZONS)])
    return pd.concat(frames, ignore_index=True)


def paired(per_fold: pd.DataFrame, a: str, b: str, folds) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        x = per_fold[(per_fold["method"] == a) & (per_fold["horizon"] == h)].set_index("fold_id")["mae"]
        y = per_fold[(per_fold["method"] == b) & (per_fold["horizon"] == h)].set_index("fold_id")["mae"]
        common = [f for f in folds if f in x.index and f in y.index]
        d = (x[common] - y[common]).to_numpy()
        t = stats.ttest_rel(x[common], y[common]) if len(d) > 1 else None
        w = stats.wilcoxon(d) if len(d) > 1 and np.any(d != 0) else None
        rows.append({"a": a, "b": b, "horizon": h, "folds": len(d), "delta": float(d.mean()) if len(d) else np.nan,
                     "t": float(t.statistic) if t else np.nan, "p_t": float(t.pvalue) if t else np.nan,
                     "p_wilcoxon": float(w.pvalue) if w else np.nan, "wins_a": int((d < 0).sum())})
    out = pd.DataFrame(rows)
    out["p_t_holm"] = holm(out["p_t"].to_numpy())
    out["p_wilcoxon_holm"] = holm(out["p_wilcoxon"].to_numpy())
    return out


def bootstrap(table: pd.DataFrame, a: str, b: str, folds) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        sub = table[(table["horizon"] == h) & table["fold_id"].isin(folds) & table["method"].isin([a, b])]
        cell = sub.groupby(["method", "fold_id", "target_period_id", "node_id"])["abs_error"].mean().unstack(0)
        cell = cell.dropna()
        errors = pd.DataFrame({"fold_id": cell.index.get_level_values(0),
                               "target_period_id": cell.index.get_level_values(1),
                               "diff": (cell[a] - cell[b]).to_numpy()})
        rows.append({"a": a, "b": b, "horizon": h, **block_bootstrap(errors, n_boot=N_BOOT, seed=0)})
    return pd.DataFrame(rows)


def decide(tests: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    merged = tests.merge(boot[["horizon", "ci_low", "ci_high", "p_boot"]], on="horizon")
    verdict = []
    for r in merged.itertuples():
        if r.horizon in (3, 4) and r.delta <= -MARGIN and r.p_t_holm < 0.05 and r.ci_high < 0:
            verdict.append("practical improvement")
        elif r.horizon in (1, 2) and r.ci_high <= MARGIN:
            verdict.append("non-inferior (margin 0.30)")
        elif r.horizon in (1, 2):
            verdict.append("non-inferiority not shown")
        else:
            verdict.append("no measurable difference")
    merged["verdict"] = verdict
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args()

    baseline = load_pipeline()
    fm = baseline.folds_module
    tensors = fm.load_tensors(VARIANT)
    truth = truth_frame(tensors)
    all_folds = {f["fold_id"]: f for f in build_benchmark_folds(fm, include_holdout=True)}

    if args.holdout:
        folds, arms, out = (spec.HOLDOUT_FOLD,), spec.PRIMARY_ARMS, ROOT / "holdout_2026"
    else:
        folds, arms, out = spec.RETROSPECTIVE_FOLDS, tuple(spec.ARMS), ROOT / "statistics"
    out.mkdir(parents=True, exist_ok=True)

    ext = load_arms(folds, arms)
    if ext.empty:
        raise SystemExit("no extension predictions for these folds")
    refs = load_refs(folds)

    # Common support: every present arm and reference.
    methods = {m: g[ev.KEYS].drop_duplicates() for m, g in ext.groupby("method")}
    methods.update({m: g[ev.KEYS].drop_duplicates() for m, g in refs.groupby("method")})
    common = None
    for keys in methods.values():
        common = keys if common is None else common.merge(keys, on=ev.KEYS)
    coverage = pd.DataFrame([{"method": m, "horizon": h, "cells": int((k["horizon"] == h).sum()),
                              "common": int((common["horizon"] == h).sum())}
                             for m, k in methods.items() for h in HORIZONS])
    coverage["coverage"] = coverage["common"] / coverage["cells"]
    coverage.to_csv(out / "coverage.csv", index=False)

    units = ext.groupby(["method", "fold_id"])["seed"].nunique().unstack(0)
    units.to_csv(out / "seeds_per_fold.csv")

    persistence = ev.persistence_on_cells(common, tensors, all_folds)
    table = pd.concat([ext[["method", *ev.KEYS, "seed", "prediction"]], refs, persistence], ignore_index=True)
    table = table.merge(common, on=ev.KEYS).merge(truth, on=["target_period_id", "node_id"])
    table = table[table["observed"] == 1]
    table["abs_error"] = (table["prediction"] - table["actual"]).abs()
    thresholds = {f: baseline.naive.peak_thresholds(tensors["y"], tensors["y_mask"],
                                                    tensors["period_id"] <= all_folds[f]["fit_end_period"])
                  for f in folds}
    metrics = ev.score(table[["method", *ev.KEYS, "seed", "prediction"]], truth, thresholds)
    metrics.to_csv(out / "metrics_by_seed.csv", index=False)
    per_fold = metrics.groupby(["method", "fold_id", "horizon"])[["mae", "peak_mae"]].mean().reset_index()
    per_fold.to_csv(out / "metrics_by_fold.csv", index=False)

    # WIS from NB quantiles (extension arms; seed-level, then averaged).
    wis_rows = []
    for (m, f, s, h), g in ext.merge(common, on=ev.KEYS).groupby(["method", "fold_id", "seed", "horizon"]):
        g = g.merge(truth, on=["target_period_id", "node_id"])
        g = g[g["observed"] == 1]
        q = nb_quantiles(g["mu"].to_numpy(), g["alpha"].to_numpy())
        wis_rows.append({"method": m, "fold_id": f, "seed": s, "horizon": h,
                         "wis": float(weighted_interval_score(g["actual"].to_numpy(), q, QUANTILES).mean()),
                         "cov95": float(((g["actual"] >= q[:, 0]) & (g["actual"] <= q[:, -1])).mean())})
    wis = pd.DataFrame(wis_rows)
    wis.to_csv(out / "wis_by_seed.csv", index=False)

    if args.holdout:
        summary = per_fold.pivot(index="method", columns="horizon", values="mae")
        summary.to_csv(out / "mae_2026.csv")
        boot = pd.concat([bootstrap(table, PRIMARY, BASELINE, folds),
                          bootstrap(table, PRIMARY, "persistence", folds),
                          bootstrap(table, BASELINE, "persistence", folds)], ignore_index=True)
        boot["same_sign_as_retrospective"] = np.nan
        boot.to_csv(out / "bootstrap_2026.csv", index=False)
        print(summary.round(3).to_string())
        print(boot.round(3).to_string(index=False))
        return 0

    report = {}
    for label, subset in (("headline_7", spec.HEADLINE_FOLDS), ("all_9", spec.RETROSPECTIVE_FOLDS),
                          ("covid_4_5", (4, 5))):
        pf = per_fold[per_fold["fold_id"].isin(subset)]
        report[label] = pf.groupby(["method", "horizon"])[["mae", "peak_mae"]].mean().reset_index()
        report[label].to_csv(out / f"summary_{label}.csv", index=False)
    tests = paired(per_fold, PRIMARY, BASELINE, spec.HEADLINE_FOLDS)
    boot = bootstrap(table, PRIMARY, BASELINE, spec.HEADLINE_FOLDS)
    primary = decide(tests, boot)
    primary.to_csv(out / "primary_D_vs_B_headline7.csv", index=False)
    decide(paired(per_fold, PRIMARY, BASELINE, spec.RETROSPECTIVE_FOLDS),
           bootstrap(table, PRIMARY, BASELINE, spec.RETROSPECTIVE_FOLDS)).to_csv(
        out / "secondary_D_vs_B_all9.csv", index=False)

    exploratory = []
    for a, b in (("A_origin_equal", BASELINE), ("C_origin_measured", PRIMARY),
                 ("E_target_fixed_1234", BASELINE), ("F_target_h4_only", BASELINE),
                 ("G_target_caseonly_utility", BASELINE), ("H_target_gradnorm", BASELINE),
                 (BASELINE, "persistence"), (PRIMARY, "persistence"),
                 (BASELINE, "nb_shared_v2"), (PRIMARY, "nb_shared_v2"),
                 (BASELINE, "gru_v2_nb"), (BASELINE, "lgbm_v2")):
        if a in per_fold["method"].unique() and b in per_fold["method"].unique():
            exploratory.append(paired(per_fold, a, b, spec.HEADLINE_FOLDS))
    pd.concat(exploratory, ignore_index=True).to_csv(out / "exploratory_headline7.csv", index=False)

    print(report["headline_7"].pivot(index="method", columns="horizon", values="mae").round(3).to_string())
    print(primary.round(4).to_string(index=False))
    (out / "provenance.json").write_text(json.dumps({"n_boot": N_BOOT, "margin": MARGIN,
                                                     "common_cells_per_h": int(len(common) / 4)}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

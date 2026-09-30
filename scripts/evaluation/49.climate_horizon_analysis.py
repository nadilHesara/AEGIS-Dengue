"""
Analysis of the completed climate-horizon predictions (frozen plan; nothing retrained).

    python scripts/evaluation/49.climate_horizon_analysis.py

Reads results/climate_horizon/{predictions,checkpoints/sweep,pilots,weights_final,
diagnostics} and, read-only, benchmark references with verified provenance.
Writes results/climate_horizon/analysis/{tables,figures}/ (CSV + LaTeX, PNG + PDF).

Uncertainty: paired 4-week blocks of contiguous forecast origins. For a fixed
horizon, contiguous target weeks are contiguous origin weeks shifted by h. All
25 districts of a block move together; blocks are resampled within each fold;
the headline statistic is the mean over folds. Seeds are averaged per cell
before resampling, so repeated seeds are not treated as independent
observations. Horizons are analysed separately (an origin's four targets
overlap in time). Folds share expanding training histories and are therefore
not independent. Fold-level t/Wilcoxon p-values are reported as in the
frozen plan but are optimistic for that reason.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.data.climate_horizon import HORIZONS, VARIANT  # noqa: E402
from src.evaluation import climate_horizon as ev  # noqa: E402
from src.evaluation.long_horizon import (  # noqa: E402
    PREDICTIONS_DIR, build_benchmark_folds, load_pipeline, nb_quantiles, truth_frame,
)
from src.evaluation.revision import block_bootstrap  # noqa: E402
from src.training import climate_horizon as tr  # noqa: E402
from src.training import climate_horizon_arms as spec  # noqa: E402

ROOT = PROJECT_DIR / "results" / "climate_horizon"
OUT = ROOT / "analysis"
TAB, FIG = OUT / "tables", OUT / "figures"
CORR = ROOT / "correction_2026-09-30"
ARM_ROOT: dict[str, Path] = {}      # arm -> results root for predictions and checkpoints (--correction)
STATS = ROOT / "statistics"


def arm_root(arm: str) -> Path:
    return ARM_ROOT.get(arm, ROOT)
B, D = "B_target_equal", "D_target_measured"
LABELS = {"A_origin_equal": "A origin, equal", "B_target_equal": "B target, equal (baseline)",
          "C_origin_measured": "C origin, measured", "D_target_measured": "D target, measured (proposed)",
          "E_target_fixed_1234": "E fixed 1,2,3,4", "F_target_h4_only": "F h=4 only",
          "G_target_caseonly_utility": "G case-only utility", "H_target_gradnorm": "H GradNorm (climate group)",
          "persistence": "Persistence", "gru_v2_nb": "GRU-NB, separate (stored)",
          "nb_shared_v2": "GRU-NB, shared (stored)", "lgbm_v2": "LightGBM (stored)",
          "chronos2_joint": "Chronos-2 joint, zero-shot (outside comparator)"}
REFS = ("gru_v2_nb", "nb_shared_v2", "lgbm_v2", "chronos2_joint")
N_BOOT = 2000


def save(frame: pd.DataFrame, name: str, caption: str, index=False) -> None:
    frame.to_csv(TAB / f"{name}.csv", index=index)
    body = frame.to_latex(index=index, float_format=lambda v: f"{v:.2f}", escape=True)
    (TAB / f"{name}.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\small\n\\caption{" + caption + "}\n\\label{tab:" + name.replace("_", "-")
        + "}\n" + body + "\\end{table}\n", encoding="utf-8")


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=160)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def load_extension(folds) -> pd.DataFrame:
    frames = []
    for arm in spec.ARMS:
        for p in sorted((arm_root(arm) / "predictions" / arm).glob("fold*_seed*.parquet")):
            if int(p.stem.split("_")[0][4:]) in folds:
                frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True)


def provenance(folds_by_id, truth, manifest_time) -> pd.DataFrame:
    rows = []
    for m in REFS:
        f = ev.load_reference(m)
        r = ev.check_reference(m, f, folds_by_id, truth, manifest_time)
        rows.append(r)
    return pd.DataFrame(rows)


def cell_errors(table: pd.DataFrame) -> pd.DataFrame:
    """Seed-level absolute and squared errors on observed cells."""

    t = table[table["observed"] == 1].copy()
    t["abs_error"] = (t["prediction"] - t["actual"]).abs()
    t["sq_error"] = (t["prediction"] - t["actual"]) ** 2
    return t


def horizon_table(err: pd.DataFrame, folds, estimand: str) -> pd.DataFrame:
    """Headline metrics per method and horizon for one estimand.

    seed_mean: mean over seeds of per-seed fold metrics (the frozen primary estimand).
    ensemble:  metrics of the per-cell mean of the seed forecasts.
    """

    e = err[err["fold_id"].isin(folds)]
    if estimand == "ensemble":
        e = e.groupby(["method", "fold_id", "horizon", "target_period_id", "node_id"], as_index=False).agg(
            prediction=("prediction", "mean"), actual=("actual", "first"), threshold=("threshold", "first"))
        e["abs_error"] = (e["prediction"] - e["actual"]).abs()
        e["sq_error"] = (e["prediction"] - e["actual"]) ** 2
        e["seed"] = -1
    e = e.assign(peak=e["actual"] >= e["threshold"])
    per_seed = e.groupby(["method", "fold_id", "horizon", "seed"]).agg(
        mae=("abs_error", "mean"), mse=("sq_error", "mean")).reset_index()
    peak = e[e["peak"]].groupby(["method", "fold_id", "horizon", "seed"])["abs_error"].mean().rename("peak_mae")
    per_seed = per_seed.merge(peak.reset_index(), on=["method", "fold_id", "horizon", "seed"], how="left")
    per_seed["rmse"] = np.sqrt(per_seed["mse"])
    per_fold = per_seed.groupby(["method", "fold_id", "horizon"])[["mae", "rmse", "peak_mae"]].mean().reset_index()
    head = per_fold.groupby(["method", "horizon"])[["mae", "rmse", "peak_mae"]].mean().reset_index()
    pers = head[head["method"] == "persistence"].set_index("horizon")
    if len(pers):
        head["skill_mae"] = 1 - head["mae"] / head["horizon"].map(pers["mae"])
        head["skill_peak"] = 1 - head["peak_mae"] / head["horizon"].map(pers["peak_mae"])
    seeds = err.groupby("method")["seed"].nunique()
    head["seeds"] = head["method"].map(seeds)
    return head, per_fold


def paired_boot(err, a, b, folds, estimand) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        e = err[(err["horizon"] == h) & err["fold_id"].isin(folds) & err["method"].isin([a, b])]
        if estimand == "seed_mean":
            cell = e.groupby(["method", "fold_id", "target_period_id", "node_id"])["abs_error"].mean().unstack(0)
        else:
            pred = e.groupby(["method", "fold_id", "target_period_id", "node_id"]).agg(
                p=("prediction", "mean"), y=("actual", "first"))
            pred["abs_error"] = (pred["p"] - pred["y"]).abs()
            cell = pred["abs_error"].unstack(0)
        cell = cell.dropna()
        errors = pd.DataFrame({"fold_id": cell.index.get_level_values(0),
                               "target_period_id": cell.index.get_level_values(1),
                               "diff": (cell[a] - cell[b]).to_numpy()})
        rows.append({"comparison": f"{a} − {b}", "estimand": estimand, "horizon": h,
                     **block_bootstrap(errors, n_boot=N_BOOT, seed=0)})
    return pd.DataFrame(rows)


def main() -> int:
    import argparse

    global OUT, TAB, FIG, STATS
    parser = argparse.ArgumentParser()
    parser.add_argument("--correction", action="store_true",
                        help="routing v2: rerun arms from correction_2026-09-30, output to correction_2026-09-30/analysis")
    args = parser.parse_args()
    if args.correction:
        for arm in json.loads((CORR / "correction_manifest.json").read_text())["reruns"]:
            ARM_ROOT[arm] = CORR
        OUT = CORR / "analysis"
        TAB, FIG, STATS = OUT / "tables", OUT / "figures", CORR / "statistics"
    for d in (TAB, FIG):
        d.mkdir(parents=True, exist_ok=True)
    baseline = load_pipeline()
    fm = baseline.folds_module
    tensors = fm.load_tensors(VARIANT)
    truth = truth_frame(tensors)
    folds_by_id = {f["fold_id"]: f for f in build_benchmark_folds(fm, include_holdout=True)}
    nodes = pd.read_csv(PROJECT_DIR / "data" / "processed" / "nodes.csv").sort_values("node_id")

    # ---- provenance of references (retrospective folds)
    manifest_time = pd.Timestamp(json.loads(ev.MANIFEST_PATH.read_text())["written_utc"])
    prov = provenance({k: v for k, v in folds_by_id.items() if k in spec.RETROSPECTIVE_FOLDS}, truth, manifest_time)
    save(prov[["method", "written_after_manifest", "split_rule_per_target", "dates_within_test_year",
               "targets_observed", "horizons_1_4", "seeds", "comparison"]],
         "reference_provenance", "Provenance checks for stored reference forecasts (folds 1--9, h=1--4).")
    direct = [m for m in REFS if prov.set_index("method").loc[m, "comparison"] == "direct"]

    results = {}
    for period, folds in (("retrospective", spec.RETROSPECTIVE_FOLDS), ("holdout", (spec.HOLDOUT_FOLD,))):
        ext = load_extension(folds)
        refs = []
        for m in direct:
            f = pd.read_parquet(PREDICTIONS_DIR / f"{m}.parquet", columns=["method", *ev.KEYS, "seed", "prediction"])
            refs.append(f[(f["method"] == m) & (f["split"] == "test") & f["fold_id"].isin(folds)
                          & f["horizon"].isin(HORIZONS)])
        refs = pd.concat(refs, ignore_index=True)
        common = ext[ev.KEYS].drop_duplicates()
        for m, g in list(ext.groupby("method")) + list(refs.groupby("method")):
            common = common.merge(g[ev.KEYS].drop_duplicates(), on=ev.KEYS)
        pers = ev.persistence_on_cells(common, tensors, folds_by_id)
        table = pd.concat([ext[["method", *ev.KEYS, "seed", "prediction"]], refs, pers], ignore_index=True)
        table = table.merge(common, on=ev.KEYS).merge(truth, on=["target_period_id", "node_id"])
        thr = pd.DataFrame([{"fold_id": f, "node_id": n, "threshold": t} for f in folds
                            for n, t in enumerate(baseline.naive.peak_thresholds(
                                tensors["y"], tensors["y_mask"], tensors["period_id"] <= folds_by_id[f]["fit_end_period"]))])
        err = cell_errors(table.merge(thr, on=["fold_id", "node_id"]))
        results[period] = (ext, err, common)

    ext, err, common = results["retrospective"]
    cov = pd.DataFrame([{"period": p, "horizon": h, "common_cells": int((c["horizon"] == h).sum())}
                        for p, (_, _, c) in results.items() for h in HORIZONS])
    save(cov, "coverage", "Common scored cells per horizon (all methods and persistence scored on these).")

    # ---- 1. primary paired comparison
    primary = pd.read_csv(STATS / "primary_D_vs_B_headline7.csv")
    boots = pd.concat([paired_boot(err, D, B, spec.HEADLINE_FOLDS, "seed_mean"),
                       paired_boot(err, D, B, spec.HEADLINE_FOLDS, "ensemble"),
                       paired_boot(err, D, B, spec.RETROSPECTIVE_FOLDS, "seed_mean")], ignore_index=True)
    boots["folds"] = ["headline 7"] * 8 + ["all 9"] * 4
    t1 = primary[["horizon", "delta", "p_t", "p_t_holm", "p_wilcoxon", "wins_a", "ci_low", "ci_high", "verdict"]].rename(
        columns={"delta": "D − B (MAE)", "p_t": "t p", "p_t_holm": "Holm p", "p_wilcoxon": "Wilcoxon p",
                 "wins_a": "D wins /7", "ci_low": "CI low", "ci_high": "CI high"})
    save(t1, "primary_D_vs_B", "Primary paired comparison: utility-weighted (D) minus equal-weight (B) two-branch "
         "NB model, headline folds, 5 seeds, 4-week block bootstrap 95\\% CI.")
    save(boots[["folds", "estimand", "horizon", "delta", "ci_low", "ci_high", "p_boot"]], "primary_bootstrap_estimands",
         "D − B under both estimands and fold sets (paired 4-week origin blocks, all districts together).")

    # ---- 2. metrics by horizon, both estimands
    for estimand in ("seed_mean", "ensemble"):
        for label, folds in (("headline", spec.HEADLINE_FOLDS), ("all9", spec.RETROSPECTIVE_FOLDS),
                             ("covid", (4, 5))):
            head, per_fold = horizon_table(err, folds, estimand)
            head["method"] = head["method"].map(LABELS)
            wide = head.pivot(index="method", columns="horizon", values=["mae", "rmse", "peak_mae", "skill_mae"])
            wide.columns = [f"{m} h{h}" for m, h in wide.columns]
            save(wide.reset_index(), f"metrics_{label}_{estimand}",
                 f"MAE, RMSE, peak MAE and MAE skill vs persistence by horizon ({label} folds, {estimand.replace('_', '-')} estimand).")
            if label == "headline" and estimand == "seed_mean":
                per_fold_headline = per_fold
    pf = per_fold_headline
    fold_tab = pf[pf["method"].isin([B, D, "persistence"])].pivot_table(index=["fold_id", "horizon"], columns="method",
                                                                       values="mae").reset_index()
    fold_tab["D − B"] = fold_tab[D] - fold_tab[B]
    save(fold_tab, "per_fold_B_D", "Per-fold MAE (seed mean) for B, D and persistence, headline folds.")

    # ---- 3. holdout
    hext, herr, hcommon = results["holdout"]
    hhead, _ = horizon_table(herr, (spec.HOLDOUT_FOLD,), "seed_mean")
    hhead["method"] = hhead["method"].map(LABELS)
    hw = hhead.pivot(index="method", columns="horizon", values=["mae", "rmse", "peak_mae", "skill_mae"])
    hw.columns = [f"{m} h{h}" for m, h in hw.columns]
    save(hw.reset_index(), "holdout_2026_metrics", "2026 hold-out (19 weeks, B and D only among extension arms).")
    hb = pd.concat([paired_boot(herr, D, B, (10,), "seed_mean"), paired_boot(herr, D, "persistence", (10,), "seed_mean"),
                    paired_boot(herr, B, "persistence", (10,), "seed_mean")], ignore_index=True)
    save(hb[["comparison", "horizon", "delta", "ci_low", "ci_high", "p_boot"]], "holdout_2026_bootstrap",
         "2026 hold-out paired differences, 4-week block bootstrap.")

    # ---- 4. controls vs B
    # Matched seeds: every arm (B included) restricted to the seed IDs all arms
    # share, {0, 1, 2}, on the same cells. (Before 2026-09-30 21:00 this table
    # compared B's 5 seeds with the other arms' 3.)
    rows = []
    shared = sorted(set.intersection(*[set(err[err.method == a].seed.unique()) for a in spec.ARMS]))
    head, per_fold = horizon_table(err[err.method.isin(list(spec.ARMS)) & err.seed.isin(shared)],
                                   spec.HEADLINE_FOLDS, "seed_mean")
    for arm in spec.ARMS:
        if arm == B:
            continue
        for h in HORIZONS:
            x = per_fold[(per_fold["method"] == arm) & (per_fold["horizon"] == h)].set_index("fold_id")["mae"]
            y = per_fold[(per_fold["method"] == B) & (per_fold["horizon"] == h)].set_index("fold_id")["mae"]
            d = x - y
            rows.append({"arm": LABELS[arm], "horizon": h, "delta_vs_B": d.mean(), "wins": int((d < 0).sum())})
    ctrl = pd.DataFrame(rows).pivot(index="arm", columns="horizon", values="delta_vs_B")
    ctrl.columns = [f"h{h}" for h in ctrl.columns]
    save(ctrl.reset_index(), "controls_vs_B", "Architecture and utility controls: headline MAE difference vs B "
         f"(negative = better), matched seeds {shared} for every arm. Exploratory, unadjusted.")

    # ---- 5. compute
    units = pd.DataFrame([json.loads(p.read_text()) for arm in spec.ARMS
                          for p in (arm_root(arm) / "predictions" / arm).glob("fold*_seed*.json")])
    pil = pd.DataFrame([json.loads(p.read_text()) for p in (ROOT / "pilots").glob("fold*_block*_seed*.json")])
    comp = units.groupby("arm").agg(parameters=("parameters", "first"), fits=("seconds", "size"),
                                    mean_s=("seconds", "mean"), total_min=("seconds", lambda s: s.sum() / 60),
                                    peak_gpu_mb=("peak_gpu_mb", "max"), best_epoch=("best_epoch", "mean")).reset_index()
    comp["arm"] = comp["arm"].map(LABELS)
    pilot_row = pd.DataFrame([{"arm": "Utility pilots (60 units, 2 fits each)", "parameters": np.nan,
                               "fits": 2 * len(pil), "mean_s": pil["runtime_s"].map(lambda r: r["total_s"]).mean() / 2,
                               "total_min": pil["runtime_s"].map(lambda r: r["total_s"]).sum() / 60,
                               "peak_gpu_mb": np.nan, "best_epoch": np.nan}])
    save(pd.concat([comp, pilot_row], ignore_index=True), "compute",
         "Compute cost on an RTX 4070 Laptop GPU (retrospective + hold-out fits of the arms shown; pilots listed separately).")

    # ---- 6. weights and utilities
    wrows, urows = [], []
    for f in range(1, 11):
        for util in ("shuffle", "case_only"):
            rec = json.loads((ROOT / "weights_final" / f"fold{f}_{util}.json").read_text())
            for h, (w, g) in enumerate(zip(rec["weights"], rec["signed_gain"]), start=1):
                wrows.append({"fold": f, "utility": util, "horizon": h, "weight": w, "signed_gain": g})
            if util == "shuffle":
                for v in rec["variability"]:
                    urows.append({"fold": f, **v})
    W = pd.DataFrame(wrows)
    save(W.pivot_table(index=["utility", "fold"], columns="horizon", values="weight").reset_index(),
         "weights_by_fold", "Frozen climate-gradient weights per fold (mean one; uniform = 1).")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, util, title in ((axes[0], "shuffle", "Primary (shuffle reliance)"), (axes[1], "case_only", "Sensitivity (case-only)")):
        for f in range(1, 11):
            s = W[(W.utility == util) & (W.fold == f)]
            ax.plot(s.horizon, s.weight, marker="o", alpha=0.7, label=f"fold {f}")
        ax.axhline(1.0, color="k", ls="--", lw=1, label="uniform")
        ax.set(title=title, xlabel="horizon (weeks)", ylabel="weight", xticks=list(HORIZONS))
        if util == "case_only":
            ax.set_yscale("log")
    axes[0].legend(fontsize=6, ncol=2)
    savefig(fig, "weights_vs_uniform")

    U = pd.DataFrame(urows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for f in range(1, 11):
        s = W[(W.utility == "shuffle") & (W.fold == f)]
        axes[0].plot(s.horizon, s.signed_gain, marker="o", alpha=0.7, label=f"fold {f}")
        u = U[U.fold == f]
        axes[0].vlines(u.horizon + (f - 5.5) * 0.02, u.gain_shuffle_unit_min, u.gain_shuffle_unit_max, alpha=0.3)
        s = W[(W.utility == "case_only") & (W.fold == f)]
        axes[1].plot(s.horizon, s.signed_gain, marker="o", alpha=0.7)
    for ax, t, y in ((axes[0], "Shuffle gain (MAE_shuffled − MAE_real)", "MAE"),
                     (axes[1], "Case-only relative gain", "relative MAE gain")):
        ax.axhline(0, color="k", lw=1)
        ax.set(title=t, xlabel="horizon (weeks)", ylabel=y, xticks=list(HORIZONS))
    axes[0].legend(fontsize=6, ncol=2)
    savefig(fig, "signed_utilities")

    # ---- 7. kernels (from saved checkpoints), gates and corrections
    device = torch.device("cpu")
    krows = []
    for arm in ("A_origin_equal", B, "C_origin_measured", D):
        for p in sorted((arm_root(arm) / "checkpoints" / "sweep" / arm).glob("fold*_seed*.pt")):
            fold = int(p.stem.split("_")[0][4:])
            if fold not in spec.RETROSPECTIVE_FOLDS:
                continue
            model, saved = tr.load_checkpoint(p, device)
            with torch.no_grad():
                weights, mass = model.encoder.weights()                      # [H, N, K, history]
                peaks = model.encoder.peak_delays()                          # [H, N, K]
            rain = saved["metadata"]["climate_channels"].index("rainfall_daily_mean_mm")
            for hi, h in enumerate(HORIZONS):
                curve = weights[hi, :, rain].mean(0).numpy()
                for k, v in enumerate(curve):
                    krows.append({"arm": arm, "fold": fold, "seed": int(p.stem.split("_")[1][4:]), "horizon": h,
                                  "delay": h + k, "weight": float(v), "mass": float(mass[hi, :, rain].mean()),
                                  "peak": float(peaks[hi, :, rain].mean())})
                for n in range(peaks.shape[1]):
                    krows.append({"arm": arm, "fold": fold, "seed": int(p.stem.split("_")[1][4:]), "horizon": h,
                                  "node_id": n, "district_peak": float(peaks[hi, n, rain])})
    K = pd.DataFrame(krows)
    Kc = K[K["delay"].notna()]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for hi, h in enumerate(HORIZONS):
        for arm, style in ((B, "-"), (D, "--"), ("A_origin_equal", ":"), ("C_origin_measured", "-.")):
            c = Kc[(Kc.arm == arm) & (Kc.horizon == h)].groupby("delay")["weight"].mean()
            axes[hi].plot(c.index, c.values, style, label=LABELS[arm])
        axes[hi].set(title=f"h = {h}", xlabel="delay d = h + k (weeks)")
    axes[0].set_ylabel("rainfall kernel weight")
    axes[0].legend(fontsize=6)
    savefig(fig, "rainfall_delay_kernels")
    peak_tab = Kc.groupby(["arm", "horizon"]).agg(peak_delay=("peak", "mean"), usable_mass=("mass", "mean")).reset_index()
    peak_tab["arm"] = peak_tab["arm"].map(LABELS)
    peak_wide = peak_tab.pivot(index="arm", columns="horizon", values=["peak_delay", "usable_mass"]).round(3)
    peak_wide.columns = [f"{m} h{h}" for m, h in peak_wide.columns]
    save(peak_wide.reset_index(),
         "rainfall_kernel_summary", "Learned rainfall kernel: centre of mass as a target-relative delay and usable mass.")

    scan = pd.read_csv(ROOT / "diagnostics" / "lag_correlation_train_only.csv")
    scan = scan[(scan.feature == "rainfall_daily_mean_mm") & (scan.lag == 0)][["fold_id", "node_id", "peak_lag_deseasonalised"]]
    dp = K[K["district_peak"].notna()].groupby(["arm", "fold", "horizon", "node_id"])["district_peak"].mean().reset_index()
    dp = dp.merge(scan, left_on=["fold", "node_id"], right_on=["fold_id", "node_id"])
    corr = dp.groupby(["arm", "horizon", "fold"]).apply(
        lambda g: g["district_peak"].corr(g["peak_lag_deseasonalised"], method="spearman"),
        include_groups=False).groupby(["arm", "horizon"]).agg(["mean", "std"]).reset_index()
    corr["arm"] = corr["arm"].map(LABELS)
    save(corr, "kernel_vs_scan_spearman", "Spearman correlation across districts between learned rainfall delay and "
         "the training-only cross-correlation peak (descriptive; the scan peak is below its resolvability threshold).")

    g = ext[ext["method"].isin([B, D]) & ext["fold_id"].isin(spec.HEADLINE_FOLDS)]
    gc = g.groupby(["method", "horizon"]).agg(gate=("gate", "mean"), abs_delta_climate=("delta_climate", lambda x: x.abs().mean()),
                                               abs_gated_climate=("gated_climate", lambda x: x.abs().mean()),
                                               abs_delta_case=("delta_case", lambda x: x.abs().mean())).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for m, st in ((B, "-o"), (D, "--s")):
        s = gc[gc.method == m]
        axes[0].plot(s.horizon, s.gate, st, label=LABELS[m])
        axes[1].plot(s.horizon, s.abs_gated_climate, st, label=f"|gated climate| {m[0]}")
        axes[1].plot(s.horizon, s.abs_delta_case, st, alpha=0.5, label=f"|case correction| {m[0]}")
    axes[0].set(title="Mean gate", xlabel="horizon", xticks=list(HORIZONS))
    axes[1].set(title="Correction magnitudes (log-mean scale)", xlabel="horizon", xticks=list(HORIZONS))
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=6)
    savefig(fig, "gates_and_corrections")
    gc["method"] = gc["method"].map(LABELS)
    save(gc, "gates_and_corrections", "Mean gate and absolute correction magnitudes, headline test cells, all seeds.")

    # ---- 8. forecasts selected by a fixed rule
    # Rule (fixed before plotting): the district with the highest mean cases in
    # fold 1's training period; folds 1 (2017) and 9 (2025); h = 4; B and D seed-mean
    # medians with D's seed-0 NB 95% interval; persistence.
    y_hist = np.where(tensors["y_mask"] == 1, tensors["y"], np.nan)[tensors["period_id"] <= folds_by_id[1]["train_end_period"]]
    node = int(np.nanargmean(y_hist, axis=0)) if hasattr(np, "nanargmean") else int(np.nanargmax(np.nanmean(y_hist, axis=0)))
    name = nodes.set_index("node_id").loc[node, "canonical_name"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, fold in zip(axes, (1, 9)):
        e = err[(err.fold_id == fold) & (err.horizon == 4) & (err.node_id == node)]
        for m, st in ((B, "-"), (D, "--"), ("persistence", ":")):
            s = e[e.method == m].groupby("target_period_id")["prediction"].mean()
            ax.plot(s.index, s.values, st, label=LABELS[m])
        a = e[e.method == B].groupby("target_period_id")["actual"].first()
        ax.plot(a.index, a.values, "k.", label="observed")
        s0 = ext[(ext.method == D) & (ext.seed == 0) & (ext.fold_id == fold) & (ext.horizon == 4) & (ext.node_id == node)]
        s0 = s0.sort_values("target_period_id")
        q = nb_quantiles(s0["mu"].to_numpy(), s0["alpha"].to_numpy())
        ax.fill_between(s0["target_period_id"], q[:, 0], q[:, -1], alpha=0.15, label="D 95% NB interval (seed 0)")
        ax.set(title=f"{name}, fold {fold} ({folds_by_id[fold]['test_year']}), h = 4", xlabel="target period", ylabel="cases")
    axes[0].legend(fontsize=6)
    savefig(fig, "selected_forecasts")

    # ---- 9. primary per-fold differences
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for h in HORIZONS:
        s = fold_tab[fold_tab.horizon == h]
        ax.plot(s.fold_id + (h - 2.5) * 0.08, s["D − B"], "o", label=f"h={h}")
    pb = boots[(boots.folds == "headline 7") & (boots.estimand == "seed_mean")]
    for _, r in pb.iterrows():
        ax.errorbar(10 + (r.horizon - 2.5) * 0.15, r.delta, yerr=[[r.delta - r.ci_low], [r.ci_high - r.delta]], fmt="ks", ms=4)
    ax.axhline(0, color="k", lw=1)
    ax.set(xlabel="fold (10 = headline mean with 95% block CI)", ylabel="D − B MAE", xticks=[1, 2, 3, 6, 7, 8, 9, 10])
    ax.legend(fontsize=7)
    savefig(fig, "primary_per_fold_differences")

    (OUT / "analysis_meta.json").write_text(json.dumps({
        "selected_district": name, "selection_rule": "highest mean cases in fold 1 training period",
        "direct_references": direct, "n_boot": N_BOOT}, indent=2), encoding="utf-8")
    print("direct references:", direct, "| selected district:", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

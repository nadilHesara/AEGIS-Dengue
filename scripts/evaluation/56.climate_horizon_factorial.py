"""
Exploratory 2x2 factorial analysis: lag alignment x climate weighting.

    python scripts/evaluation/56.climate_horizon_factorial.py

Arms (routing-correct predictions):
  A origin, equal     results/climate_horizon/predictions/            (v1 = v2 for equal weights)
  B target, equal     results/climate_horizon/predictions/            (v1 = v2, verified bit-identical)
  C origin, measured  results/climate_horizon/correction_2026-09-30/predictions/  (routing v2)
  D target, measured  results/climate_horizon/correction_2026-09-30/predictions/  (routing v2)

Seeds 0, 1, 2 for every arm (the seed IDs all four share); the 5-seed B/D
primary comparison is reported elsewhere and not mixed in. Cells: the
intersection of all four arms' observed test cells.

Contrasts per horizon (negative = first term better):
  C-A  weighting with origin lags        D-B  weighting with target lags
  B-A  alignment with equal weights      D-C  alignment with measured weights
  (D-B)-(C-A)  interaction

Statistic: mean over folds of the per-fold mean per-cell contrast of
seed-mean absolute errors. Uncertainty: 4-week block bootstrap of contiguous
origins, districts kept together, within folds. Fold-level paired t and
Wilcoxon. Multiplicity: Holm over all 20 contrast x horizon tests within
each fold set. The whole analysis is exploratory and post-hoc.

Outputs: results/climate_horizon/correction_2026-09-30/factorial/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.data.climate_horizon import VARIANT  # noqa: E402
from src.evaluation import climate_horizon as ev  # noqa: E402
from src.evaluation.long_horizon import load_pipeline, truth_frame  # noqa: E402
from src.evaluation.revision import block_bootstrap, holm  # noqa: E402
from src.training import climate_horizon as tr  # noqa: E402
from src.training import climate_horizon_arms as spec  # noqa: E402

ROOT = PROJECT_DIR / "results" / "climate_horizon"
CORR = ROOT / "correction_2026-09-30"
OUT = CORR / "factorial"
SEEDS = (0, 1, 2)
ARMS = {"A": ("A_origin_equal", ROOT), "B": ("B_target_equal", ROOT),
        "C": ("C_origin_measured", CORR), "D": ("D_target_measured", CORR)}
CONTRASTS = {"C-A (weighting | origin)": {"C": 1, "A": -1},
             "D-B (weighting | target)": {"D": 1, "B": -1},
             "B-A (alignment | equal)": {"B": 1, "A": -1},
             "D-C (alignment | measured)": {"D": 1, "C": -1},
             "(D-B)-(C-A) interaction": {"D": 1, "B": -1, "C": -1, "A": 1}}


def load(label):
    arm, root = ARMS[label]
    frames = [pd.read_parquet(p, columns=["method", *ev.KEYS, "seed", "prediction"])
              for p in sorted((root / "predictions" / arm).glob("fold*_seed*.parquet"))
              if int(p.stem.split("_")[0][4:]) in spec.RETROSPECTIVE_FOLDS and int(p.stem.split("_seed")[1]) in SEEDS]
    f = pd.concat(frames, ignore_index=True)
    f["arm"] = label
    return f


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = load_pipeline()
    tensors = baseline.folds_module.load_tensors(VARIANT)
    truth = truth_frame(tensors)

    frames = {k: load(k) for k in ARMS}
    seeds = {k: sorted(f.seed.unique().tolist()) for k, f in frames.items()}
    common = None
    for f in frames.values():
        keys = f[ev.KEYS].drop_duplicates()
        common = keys if common is None else common.merge(keys, on=ev.KEYS)
    data = pd.concat(frames.values()).merge(common, on=ev.KEYS).merge(truth, on=["target_period_id", "node_id"])
    data = data[data.observed == 1]
    data["ae"] = (data.prediction - data.actual).abs()
    cell = data.groupby(["arm", "fold_id", "horizon", "target_period_id", "node_id"])["ae"].mean().unstack(0).dropna()
    coverage = {"cells_per_horizon": int(len(cell) / 4), "seeds": seeds,
                "complete_seed_sets": all(v == list(SEEDS) for v in seeds.values())}

    arm_mae = (cell.groupby(level=["fold_id", "horizon"]).mean())      # per-fold seed-mean MAE per arm
    rows = []
    for fold_label, folds in (("all 9", spec.RETROSPECTIVE_FOLDS), ("headline 7", spec.HEADLINE_FOLDS)):
        for name, coef in CONTRASTS.items():
            for h in (1, 2, 3, 4):
                c = cell.xs(h, level="horizon")
                c = c[c.index.get_level_values("fold_id").isin(folds)]
                diff = sum(w * c[k] for k, w in coef.items())
                errors = pd.DataFrame({"fold_id": c.index.get_level_values("fold_id"),
                                       "target_period_id": c.index.get_level_values("target_period_id"),
                                       "diff": diff.to_numpy()})
                boot = block_bootstrap(errors, block=4, n_boot=2000, seed=0)
                per_fold = errors.groupby("fold_id")["diff"].mean()
                t = stats.ttest_1samp(per_fold, 0.0)
                w = stats.wilcoxon(per_fold) if (per_fold != 0).any() else None
                b_mae = arm_mae.xs(h, level="horizon").loc[list(folds), "B"].mean()
                rows.append({"folds": fold_label, "contrast": name, "horizon": h, "delta": boot["delta"],
                             "delta_pct_of_B": 100 * boot["delta"] / b_mae,
                             "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                             "dz": float(per_fold.mean() / per_fold.std(ddof=1)),
                             "folds_negative": int((per_fold < 0).sum()), "n_folds": len(per_fold),
                             "p_t": float(t.pvalue), "p_wilcoxon": float(w.pvalue) if w else np.nan})
    table = pd.DataFrame(rows)
    for fold_label in table.folds.unique():
        m = table.folds == fold_label
        table.loc[m, "p_t_holm20"] = holm(table.loc[m, "p_t"].to_numpy())
        table.loc[m, "p_wilcoxon_holm20"] = holm(table.loc[m, "p_wilcoxon"].to_numpy())
    table["status"] = "exploratory, post-hoc"
    table.to_csv(OUT / "factorial_contrasts.csv", index=False)
    arm_table = arm_mae.groupby(level="horizon").apply(
        lambda g: g[g.index.get_level_values("fold_id").isin(spec.HEADLINE_FOLDS)].mean())
    arm_table.to_csv(OUT / "arm_mae_headline7_3seeds.csv")

    # Lag support from the saved checkpoints (same seeds).
    support = []
    for label, (arm, root) in ARMS.items():
        for p in sorted((root / "checkpoints" / "sweep" / arm).glob("fold*_seed*.pt")):
            fold, seed = int(p.stem.split("_")[0][4:]), int(p.stem.split("_seed")[1])
            if fold not in spec.RETROSPECTIVE_FOLDS or seed not in SEEDS:
                continue
            model, saved = tr.load_checkpoint(p, torch.device("cpu"))
            rain = saved["metadata"]["climate_channels"].index("rainfall_daily_mean_mm")
            with torch.no_grad():
                weights, mass = model.encoder.weights()
                peak = model.encoder.peak_delays()
            for hi in range(4):
                support.append({"arm": label, "mode": model.encoder.mode, "fold": fold, "seed": seed,
                                "horizon": hi + 1, "history_weeks": model.encoder.history,
                                "max_delay": model.encoder.max_delay,
                                "usable_mass_all_climate": float(mass[hi].mean()),
                                "rain_peak_delay": float(peak[hi, :, rain].mean())})
    support = pd.DataFrame(support).groupby(["arm", "mode", "horizon"]).mean(numeric_only=True).reset_index()
    support.drop(columns=["fold", "seed"]).to_csv(OUT / "lag_support.csv", index=False)

    pd.Series(coverage).to_json(OUT / "coverage.json", indent=2)
    print(coverage)
    print(arm_table.round(2))
    print(table[table.folds == "all 9"][["contrast", "horizon", "delta", "ci_low", "ci_high", "folds_negative",
                                         "p_wilcoxon", "p_t_holm20"]].round(3).to_string(index=False))
    print(table[table.folds == "headline 7"][["contrast", "horizon", "delta", "ci_low", "ci_high",
                                              "folds_negative", "p_wilcoxon", "p_t_holm20"]].round(3).to_string(index=False))
    print(support.drop(columns=["fold", "seed"]).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

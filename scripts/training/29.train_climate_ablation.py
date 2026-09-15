"""
Is climate what causes the h=3-4 skill? A three-arm ablation across horizons.

`docs/multi_horizon.md` (README 8f) established the first real improvement in
this project: at h=3 and h=4 the model beats same-horizon persistence on MAE and
on peak MAE, 7/7 folds at h=4, p < 0.01, and the gain is not the 2017 artefact.
The explanation offered for it is mechanistic -- the measured rainfall-to-dengue
delay is 5-10 weeks (`scripts/17`), so as the horizon grows the forecast origin
stops dominating and climate gets room to matter.

**That explanation has never been tested.** Skill appearing at h=4 is equally
consistent with a duller reading: persistence degrades faster than the model
does, so the relative margin grows while climate contributes nothing at all. The
two readings predict opposite things about removing climate, and this script
measures which one holds.

Three arms, one backbone, four horizons:

    full        every channel -- the 8f model.
    no_climate  the 16 climate-derived channels sliced out of v1's 23, leaving
                case history, seasonality, the observation flags and centroids.
    shuffled    all 23 channels, climate time-shuffled within each district.

**The shuffled arm is what makes this conclusive**, and it is the piece 6.3
was missing. That section ran the two-arm ablation at h=1, found +0.01 MAE in
recent normal years, and flagged its own result as confounded: dropping channels
changes the input width, the first layer's parameter count and the effective
regularisation together, so a `no_climate` penalty need not be about weather.
The shuffled arm holds every one of those fixed -- identical width, identical
parameter count, identical per-fold scaler statistics, identical marginal
distribution per channel per district -- and destroys only the alignment between
weather and time. The three arms then read as:

    full ~ shuffled ~ no_climate     climate contributes nothing. The h=4 skill
                                     is persistence decaying, not weather.
    full < shuffled ~ no_climate     climate's temporal content is load-bearing.
                                     The mechanism claim survives.
    full ~ shuffled < no_climate     the channels help as width, not as weather.

**The horizon trend is the actual test, not the h=4 number.** The mechanism claim
is not "climate matters" but "climate matters *more as the horizon grows*". One
horizon cannot show that. Running h = 1, 2, 3, 4 turns a point estimate into a
curve with a predicted shape, and h=1 doubles as a correctness check: 6.3
measured the cost of dropping climate at h=1 as approximately zero in recent
normal years, so an h=1 column that shows a large cost means the harness is
wrong before any h=4 number is worth reading.

**Every horizon is judged against its own persistence**, rescored here on the
same folds and masks. Persistence goes stale as the horizon grows (16.42 at h=1,
28.47 at h=4), so comparing an h=4 arm to the h=1 reference would flatter it
against a much easier task.

The backbone is `identity`. 8e established that no graph beats the identity,
and 8f's positive result is on the identity backbone; running the ablation that
explains that result on the same backbone keeps it attributable.

Everything else -- architecture, training loop, early stopping, optimiser,
folds, preprocessing, windowing, the anchored residual target, the masked-MSE
objective and the metrics -- is imported from `scripts/16.train_gcn_gru.py`.

Usage:

    python scripts/29.train_climate_ablation.py                    3 arms, h=1-4, 9 folds, 3 seeds
    python scripts/29.train_climate_ablation.py --folds 8 --seeds 1  smoke run
    python scripts/29.train_climate_ablation.py --horizons 4
    python scripts/29.train_climate_ablation.py --model contiguity

Outputs:
    results/models/climate_ablation_report.md
    results/models/climate_ablation_metrics.csv
    results/models/climate_ablation_naive.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.models.climate_ablation import (  # noqa: E402
    FeatureSubsetGCNGRU,
    climate_indices,
    non_climate_indices,
    shuffle_split_climate,
)


PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "models"

FOLDS_PATH = PROCESSED_DIR / "folds.json"
ADJACENCY_PATH = PROCESSED_DIR / "adjacency.npz"

REPORT_PATH = RESULTS_DIR / "climate_ablation_report.md"
METRICS_PATH = RESULTS_DIR / "climate_ablation_metrics.csv"
NAIVE_PATH = RESULTS_DIR / "climate_ablation_naive.csv"

CLIMATE_ABLATION_VERSION = "climate-ablation-v1"

DEFAULTS = {
    "lookback": 12,
    "hidden": 32,
    "gcn_layers": 2,
    "dropout": 0.2,
    "learning_rate": 3e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 150,
    "patience": 15,
    "seeds": 3,
    "target": "residual",
}

ARMS = ("full", "no_climate", "shuffled")

# The arm every other is measured against: the model 8f actually ran.
CONTROL_ARM = "full"

MODEL_ADJACENCY = {
    "identity": None,
    "contiguity": "A_norm",
}


baseline_module = None  # scripts/16
naive = None            # scripts/15
folds_module = None     # scripts/14


def _load(name: str, filename: str):
    """Import a numerically-prefixed pipeline script by path."""

    path = PROJECT_DIR / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def load_modules() -> None:
    """Import script 16 and the modules it stands on."""

    global baseline_module, naive, folds_module

    baseline_module = _load("baseline_module", "training/16.train_gcn_gru.py")
    baseline_module.load_modules()

    naive = baseline_module.naive
    folds_module = baseline_module.folds_module


def load_adjacency(model: str, n_nodes: int) -> np.ndarray:
    """Return the adjacency for the chosen backbone."""

    key = MODEL_ADJACENCY[model]

    if key is None:
        return np.eye(n_nodes, dtype=np.float32)

    with np.load(ADJACENCY_PATH, allow_pickle=True) as data:
        if key not in data.files:
            raise KeyError(
                f"{key!r} is not in adjacency.npz. Rerun scripts/13.build_adjacency.py."
            )
        return data[key].astype(np.float32)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_model_builder(arm: str, config: dict, keep: list[int]):
    """Return a `build_model(n_features) -> nn.Module` closure for one arm.

    `full` and `shuffled` get the plain backbone at the full input width -- they
    differ in their *data*, not their architecture, which is the whole point of
    the shuffled control. `no_climate` gets a backbone sized for the surviving
    channels behind a slicing wrapper.

    The head is always one wide. `horizon` is overloaded in `scripts/16` -- the
    forecast lead in `build_fold_arrays`, the head width in the `GCNGRU`
    constructor -- and they coincide only at h=1. `scripts/18` documents the
    same trap. One model per lead here, so the lead belongs to the windower.
    """

    def build_model(n_features: int) -> nn.Module:
        width = len(keep) if arm == "no_climate" else n_features

        backbone = baseline_module.GCNGRU(
            n_features=width,
            hidden=config["hidden"],
            gcn_layers=config["gcn_layers"],
            horizon=1,
            dropout=config["dropout"],
        )

        if arm != "no_climate":
            return backbone

        return FeatureSubsetGCNGRU(backbone, keep)

    return build_model


# ---------------------------------------------------------------------------
# Persistence, rescored per horizon
# ---------------------------------------------------------------------------

def rescore_persistence(
    variant: str, folds: list[dict], horizons: list[int], lookback: int
) -> pd.DataFrame:
    """Score the naive baselines at each horizon on the same folds and masks."""

    tensors = dict(folds_module.load_tensors(variant))

    # `evaluate_folds` reports per-district errors and so needs the district
    # names, which `load_tensors` does not carry. `scripts/15.main` attaches
    # them from nodes.csv before calling; do the same rather than reimplementing
    # the scoring.
    nodes = pd.read_csv(PROCESSED_DIR / "nodes.csv").sort_values("node_id")
    tensors["canonical_name"] = nodes["canonical_name"].to_numpy(dtype=object)

    records = []
    for horizon in horizons:
        metrics, _ = naive.evaluate_folds(
            tensors, folds, lookback=lookback, horizon=horizon
        )

        for model, group in metrics.groupby("model"):
            headline = group[group["headline"]]
            records.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "headline_mae": headline["mae"].mean(),
                    "headline_peak_mae": headline["peak_mae"].mean(),
                    "epidemic_2017_mae": group.loc[
                        group["test_year"] == 2017, "mae"
                    ].mean(),
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(
    arms: list[str],
    model_name: str,
    variant: str,
    folds: list[dict],
    horizons: list[int],
    config: dict,
    device: torch.device,
) -> pd.DataFrame:
    """Train every arm on every fold and horizon, and score the test years."""

    calendar = folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()

    nodes = pd.read_csv(PROCESSED_DIR / "nodes.csv").sort_values("node_id")
    n_nodes = len(nodes)

    adjacency = load_adjacency(model_name, n_nodes)
    tensors = folds_module.load_tensors(variant)

    names = [str(name) for name in tensors["feature_names"]]
    dropped = climate_indices(names)
    keep = non_climate_indices(names)

    print(f"Channels: {len(names)} total, {len(dropped)} climate, {len(keep)} kept")
    print(f"  dropped: {[names[i] for i in dropped]}")
    print()

    rows: list[dict] = []

    for horizon in horizons:
        for fold in folds:
            base_arrays = baseline_module.build_fold_arrays(
                tensors, months, fold, config["lookback"], horizon
            )

            fit_mask = tensors["period_id"] <= fold["fit_end_period"]
            thresholds = naive.peak_thresholds(
                tensors["y"], tensors["y_mask"], fit_mask
            )

            target = base_arrays["test"]["y"]
            mask = base_arrays["test"]["mask"].astype(np.int8)

            for arm in arms:
                started = time.perf_counter()
                seed_predictions = []
                seed_info = []

                for seed in range(config["seeds"]):
                    # The shuffled arm re-draws its permutation per seed, so the
                    # reported spread includes the shuffle draw rather than
                    # treating one arbitrary permutation as the answer.
                    if arm == "shuffled":
                        arrays = shuffle_split_climate(
                            base_arrays, dropped, seed=10_000 * horizon + seed
                        )
                    else:
                        arrays = base_arrays

                    model, info = baseline_module.train_one(
                        arrays,
                        adjacency,
                        config,
                        seed,
                        device,
                        build_model=make_model_builder(arm, config, keep),
                    )
                    prediction = baseline_module.predict(
                        model, arrays["test"], adjacency, config["target"], device
                    )

                    seed_predictions.append(prediction)
                    seed_info.append(info)

                    scores = naive.evaluate(prediction, target, mask, thresholds)
                    rows.append(
                        {
                            "arm": arm,
                            "model": model_name,
                            "variant": variant,
                            "horizon": horizon,
                            "fold_id": fold["fold_id"],
                            "test_year": fold["test_year"],
                            "covers_covid": fold["covers_covid"],
                            "headline": fold["headline"],
                            "seed": seed,
                            "best_epoch": info["best_epoch"],
                            "ensemble": False,
                            **scores,
                        }
                    )

                mean_prediction = np.mean(seed_predictions, axis=0)
                ensemble_scores = naive.evaluate(
                    mean_prediction, target, mask, thresholds
                )
                rows.append(
                    {
                        "arm": arm,
                        "model": model_name,
                        "variant": variant,
                        "horizon": horizon,
                        "fold_id": fold["fold_id"],
                        "test_year": fold["test_year"],
                        "covers_covid": fold["covers_covid"],
                        "headline": fold["headline"],
                        "seed": -1,
                        "best_epoch": float(
                            np.mean([i["best_epoch"] for i in seed_info])
                        ),
                        "ensemble": True,
                        **ensemble_scores,
                    }
                )

                elapsed = time.perf_counter() - started
                seed_mae = [row["mae"] for row in rows[-config["seeds"] - 1 : -1]]

                print(
                    f"  h={horizon} fold {fold['fold_id']} ({fold['test_year']}) "
                    f"{arm:<11} MAE {np.mean(seed_mae):7.2f} +/- "
                    f"{np.std(seed_mae):5.2f}  ens {ensemble_scores['mae']:7.2f}  "
                    f"{elapsed:5.1f}s",
                    flush=True,
                )

    return pd.DataFrame(rows)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average over seeds, then over the headline folds, per arm and horizon."""

    records = []

    for ensemble in (False, True):
        subset = metrics[metrics["ensemble"] == ensemble]

        for (arm, horizon), group in subset.groupby(["arm", "horizon"]):
            by_fold = (
                group.groupby(["fold_id", "test_year", "headline", "covers_covid"])
                .agg(
                    mae=("mae", "mean"),
                    rmse=("rmse", "mean"),
                    peak_mae=("peak_mae", "mean"),
                )
                .reset_index()
            )

            headline = by_fold[by_fold["headline"]]
            epidemic = by_fold[by_fold["test_year"] == 2017]
            normal = by_fold[by_fold["fold_id"].isin([6, 8, 9])]

            spread = (
                group.groupby("fold_id")["mae"].std().mean()
                if not ensemble
                else float("nan")
            )

            records.append(
                {
                    "arm": arm,
                    "horizon": horizon,
                    "ensemble": ensemble,
                    "headline_mae": headline["mae"].mean(),
                    "headline_rmse": headline["rmse"].mean(),
                    "headline_peak_mae": headline["peak_mae"].mean(),
                    "epidemic_2017_mae": epidemic["mae"].mean(),
                    # The subset 6.3 reported its +0.01 MAE on, carried
                    # forward so the h=1 column is directly comparable to it.
                    "normal_years_mae": normal["mae"].mean(),
                    "seed_sd": spread,
                }
            )

    return (
        pd.DataFrame(records)
        .sort_values(["ensemble", "horizon", "headline_mae"])
        .reset_index(drop=True)
    )


def compare_to_control(metrics: pd.DataFrame) -> pd.DataFrame:
    """Paired comparison of each ablated arm against `full`, per horizon.

    Positive `headline_delta` means the ablation *hurt* -- removing or scrambling
    climate cost accuracy, which is the direction the mechanism claim predicts.
    Paired over the headline folds: both arms see identical folds, windows and
    seeds, so fold-to-fold variation cancels.
    """

    single = metrics[~metrics["ensemble"]]
    headline = single[single["headline"]]

    records = []

    for horizon, horizon_group in headline.groupby("horizon"):
        by_fold = horizon_group.groupby(["arm", "fold_id"])["mae"].mean().unstack(0)
        if CONTROL_ARM not in by_fold.columns:
            continue

        peak_by_fold = (
            horizon_group.groupby(["arm", "fold_id"])["peak_mae"].mean().unstack(0)
        )
        seed_sd = (
            single[single["horizon"] == horizon]
            .groupby(["arm", "fold_id"])["mae"]
            .std()
            .groupby("arm")
            .mean()
        )

        control = by_fold[CONTROL_ARM]
        others = [f for f in by_fold.index if f != 1]

        for arm in by_fold.columns:
            if arm == CONTROL_ARM:
                continue

            delta = by_fold[arm] - control
            total = delta.mean()

            try:
                from scipy import stats

                statistic, p_value = stats.ttest_rel(by_fold[arm], control)
            except Exception:
                statistic, p_value = float("nan"), float("nan")

            pooled = float(np.nanmean([seed_sd.get(arm), seed_sd.get(CONTROL_ARM)]))

            records.append(
                {
                    "arm": arm,
                    "horizon": horizon,
                    "headline_delta": total,
                    "headline_delta_pct": 100.0 * total / control.mean(),
                    "peak_delta": float(
                        (peak_by_fold[arm] - peak_by_fold[CONTROL_ARM]).mean()
                    ),
                    "folds_hurt": int((delta > 0).sum()),
                    "folds_total": len(delta),
                    "t_statistic": float(statistic),
                    "p_value": float(p_value),
                    "pooled_seed_sd": pooled,
                    "delta_in_sd": abs(total) / pooled if pooled else float("nan"),
                    "fold1_delta": float(delta.loc[1])
                    if 1 in delta.index
                    else float("nan"),
                    "ex_fold1_delta": float(delta.loc[others].mean())
                    if others
                    else float("nan"),
                }
            )

    return pd.DataFrame(records)


def skill_table(summary: pd.DataFrame, naive_frame: pd.DataFrame) -> pd.DataFrame:
    """Each arm's margin over same-horizon persistence.

    The 8f headline is a skill number, so the ablation has to be read on the
    same scale: the question is whether removing climate removes the *skill*,
    not merely whether it moves the raw MAE.
    """

    persistence = naive_frame[naive_frame["model"] == "persistence"].set_index(
        "horizon"
    )

    records = []
    for row in summary[~summary["ensemble"]].itertuples():
        if row.horizon not in persistence.index:
            continue

        reference = persistence.loc[row.horizon]

        records.append(
            {
                "arm": row.arm,
                "horizon": row.horizon,
                "mae": row.headline_mae,
                "persistence_mae": reference["headline_mae"],
                "skill_pct": 100.0
                * (reference["headline_mae"] - row.headline_mae)
                / reference["headline_mae"],
                "peak_mae": row.headline_peak_mae,
                "persistence_peak_mae": reference["headline_peak_mae"],
                "peak_skill_pct": 100.0
                * (reference["headline_peak_mae"] - row.headline_peak_mae)
                / reference["headline_peak_mae"],
            }
        )

    return pd.DataFrame(records).sort_values(["horizon", "arm"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    skill: pd.DataFrame,
    naive_frame: pd.DataFrame,
    dropped_names: list[str],
    config: dict,
    model_name: str,
    variant: str,
    horizons: list[int],
) -> None:
    """Write the climate-ablation report."""

    lines = [
        "# Climate ablation — is climate what causes the h=3–4 skill?",
        "",
        f"Version: `{CLIMATE_ABLATION_VERSION}`",
        "",
        "README §8f established that the model beats same-horizon persistence at",
        "h=3 and h=4, on MAE and on peak MAE, and attributed it to the 5–10 week",
        "rainfall-to-dengue delay. That attribution was never measured. Skill at",
        "h=4 is equally consistent with persistence decaying faster than the model",
        "does, with climate contributing nothing. This separates the two.",
        "",
        "## The arms",
        "",
        "| Arm | Inputs |",
        "| --- | --- |",
        "| `full` | every channel — the §8f model |",
        f"| `no_climate` | the {len(dropped_names)} climate-derived channels sliced out |",
        "| `shuffled` | every channel, climate time-shuffled within each district |",
        "",
        "**The `shuffled` arm is what makes this conclusive.**",
        "`docs/learnable_lags_results.md` §6.3 ran the two-arm version at h=1 and",
        "flagged its own result as confounded: dropping channels changes the input",
        "width, the first layer's parameter count and the effective regularisation",
        "at once, so a `no_climate` penalty need not be about weather. The shuffled",
        "arm holds all of those fixed — identical width, identical parameter count,",
        "identical per-fold scaler statistics, identical marginal distribution per",
        "channel per district — and destroys only the alignment between weather and",
        "time. Reading the three together:",
        "",
        "| Pattern | Conclusion |",
        "| --- | --- |",
        "| `full` ≈ `shuffled` ≈ `no_climate` | climate contributes nothing; the skill is persistence decaying |",
        "| `full` < `shuffled` ≈ `no_climate` | climate's temporal content is load-bearing |",
        "| `full` ≈ `shuffled` < `no_climate` | the channels help as width, not as weather |",
        "",
        "The shuffle is within-district and within-split, permuting whole windows,",
        "with an independent permutation per district and per channel. It never",
        "touches `y`, the mask or the anchor.",
        "",
        f"Dropped channels: {', '.join(f'`{n}`' for n in dropped_names)}.",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| `model` | {model_name} |",
        f"| `variant` | {variant} |",
        f"| `horizons` | {horizons} |",
    ]
    for key, value in config.items():
        lines.append(f"| `{key}` | {value} |")

    lines += [
        "",
        "## Does removing climate cost accuracy?",
        "",
        "Paired over the seven headline folds, against `full`. **Positive Δ means",
        "the ablation hurt** — the direction the mechanism claim predicts. `Δ in sd`",
        "compares the gap to the pooled seed sd; this project treats anything under",
        "2 sd as not established.",
        "",
        "| h | Arm | Δ MAE | Δ % | Δ peak MAE | Folds hurt | t | p | Δ in sd | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in comparison.itertuples():
        if row.headline_delta > 0 and row.delta_in_sd > 2 and row.p_value < 0.05:
            verdict = "**climate matters**"
        elif row.headline_delta < 0 and row.delta_in_sd > 2 and row.p_value < 0.05:
            verdict = "**ablation helped**"
        else:
            verdict = "not established"

        lines.append(
            f"| {row.horizon} | `{row.arm}` | {row.headline_delta:+.2f} | "
            f"{row.headline_delta_pct:+.1f}% | {row.peak_delta:+.2f} | "
            f"{row.folds_hurt}/{row.folds_total} | {row.t_statistic:+.2f} | "
            f"{row.p_value:.3f} | {row.delta_in_sd:.2f} | {verdict} |"
        )

    lines += [
        "",
        "### The horizon trend — the actual test",
        "",
        "The mechanism claim is not that climate matters, but that it matters",
        "**more as the horizon grows**. A single horizon cannot show that. If the",
        "claim holds, the `no_climate` cost should rise with h; if the skill is",
        "persistence decay, the column should stay flat near zero.",
        "",
        "| Arm | " + " | ".join(f"h={h}" for h in horizons) + " |",
        "| --- | " + " | ".join("---" for _ in horizons) + " |",
    ]

    for arm in [a for a in ARMS if a != CONTROL_ARM]:
        cells = []
        for horizon in horizons:
            match = comparison[
                (comparison["arm"] == arm) & (comparison["horizon"] == horizon)
            ]
            cells.append(
                f"{match['headline_delta'].iloc[0]:+.2f}" if len(match) else "—"
            )
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "**The h=1 column is a correctness check.** §6.3 measured the cost of",
        "dropping climate at h=1 as ≈ +0.01 MAE in recent normal years. An h=1",
        "column showing a large cost means the harness is wrong before any h=4",
        "number is worth reading.",
        "",
        "### Where any movement comes from",
        "",
        "| h | Arm | Fold 1 Δ | Δ over the other six folds |",
        "| --- | --- | --- | --- |",
    ]
    for row in comparison.itertuples():
        lines.append(
            f"| {row.horizon} | `{row.arm}` | {row.fold1_delta:+.2f} | "
            f"{row.ex_fold1_delta:+.3f} |"
        )

    lines += [
        "",
        "## Skill over same-horizon persistence",
        "",
        "The §8f headline is a skill number, so the ablation is read on the same",
        "scale: does removing climate remove the *skill*? Positive is better than",
        "persistence.",
        "",
        "| h | Arm | MAE | persistence | Skill | Peak MAE | persistence peak | Peak skill |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in skill.itertuples():
        lines.append(
            f"| {row.horizon} | `{row.arm}` | {row.mae:.2f} | "
            f"{row.persistence_mae:.2f} | {row.skill_pct:+.1f}% | "
            f"{row.peak_mae:.2f} | {row.persistence_peak_mae:.2f} | "
            f"{row.peak_skill_pct:+.1f}% |"
        )

    single = summary[~summary["ensemble"]]
    lines += [
        "",
        "## Full results — single seed models",
        "",
        "| h | Arm | MAE | RMSE | Peak MAE | 2017 MAE | Normal years | Seed sd |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in single.itertuples():
        lines.append(
            f"| {row.horizon} | `{row.arm}` | {row.headline_mae:.2f} | "
            f"{row.headline_rmse:.2f} | {row.headline_peak_mae:.2f} | "
            f"{row.epidemic_2017_mae:.2f} | {row.normal_years_mae:.2f} | "
            f"{row.seed_sd:.2f} |"
        )

    lines += [
        "",
        "## Reproducing",
        "",
        "```bash",
        "python scripts/29.train_climate_ablation.py \\",
        f"    --model {model_name} --variant {variant} \\",
        f"    --horizons {' '.join(str(h) for h in horizons)} "
        f"--seeds {config['seeds']}",
        "```",
        "",
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument(
        "--model", default="identity", choices=list(MODEL_ADJACENCY)
    )
    parser.add_argument("--variant", default="v1", choices=["v0", "v1"])
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3, 4])

    for key, value in DEFAULTS.items():
        parser.add_argument(
            f"--{key.replace('_', '-')}", type=type(value), default=value
        )

    return parser.parse_args()


def main() -> int:
    """Run the ablation and write the report."""

    arguments = parse_arguments()
    load_modules()

    if not FOLDS_PATH.exists() or not ADJACENCY_PATH.exists():
        print("Missing folds.json or adjacency.npz. Run scripts 13 and 14 first.")
        return 1

    config = {key: getattr(arguments, key) for key in DEFAULTS}

    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))["folds"]
    if arguments.folds:
        folds = [f for f in folds if f["fold_id"] in arguments.folds]

    horizons = list(arguments.horizons)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device:   {device}")
    print(f"Model:    {arguments.model}")
    print(f"Variant:  {arguments.variant}")
    print(f"Folds:    {[f['fold_id'] for f in folds]}")
    print(f"Horizons: {horizons}")
    print(f"Arms:     {arguments.arms}")
    print(f"Seeds:    {config['seeds']}")
    print(f"Control:  {CONTROL_ARM}")
    print()

    started = time.perf_counter()
    metrics = run_experiment(
        arguments.arms,
        arguments.model,
        arguments.variant,
        folds,
        horizons,
        config,
        device,
    )
    elapsed = (time.perf_counter() - started) / 60

    print(f"\nTrained in {elapsed:.1f} min\n")

    naive_frame = rescore_persistence(
        arguments.variant, folds, horizons, config["lookback"]
    )

    summary = summarise(metrics)
    comparison = compare_to_control(metrics)
    skill = skill_table(summary, naive_frame)

    tensors = folds_module.load_tensors(arguments.variant)
    names = [str(name) for name in tensors["feature_names"]]
    dropped_names = [names[i] for i in climate_indices(names)]

    print("single seed models")
    print(
        summary[~summary["ensemble"]][
            [
                "horizon",
                "arm",
                "headline_mae",
                "headline_peak_mae",
                "epidemic_2017_mae",
                "normal_years_mae",
                "seed_sd",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:8.2f}")
    )

    if len(comparison):
        print("\nvs the full-climate model (positive = the ablation hurt)")
        print(
            comparison[
                [
                    "horizon",
                    "arm",
                    "headline_delta",
                    "headline_delta_pct",
                    "peak_delta",
                    "folds_hurt",
                    "p_value",
                    "delta_in_sd",
                    "ex_fold1_delta",
                ]
            ].to_string(index=False, float_format=lambda v: f"{v:8.3f}")
        )

    print("\nskill over same-horizon persistence")
    print(
        skill[["horizon", "arm", "mae", "persistence_mae", "skill_pct",
               "peak_skill_pct"]].to_string(
            index=False, float_format=lambda v: f"{v:8.2f}"
        )
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    naive_frame.to_csv(NAIVE_PATH, index=False)

    write_report(
        summary,
        comparison,
        skill,
        naive_frame,
        dropped_names,
        config,
        arguments.model,
        arguments.variant,
        horizons,
    )

    print(f"\nWrote {REPORT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {METRICS_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {NAIVE_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

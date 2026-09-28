"""
Does the lag encoder's basis mixture want a different simplex activation?

The encoder builds each delay curve as a mixture of six Gaussian bumps and puts
the mixture weights on the simplex with a softmax. That softmax was never a
decision -- it is the default way onto a simplex -- and there are three specific
reasons to think it is the wrong one here, all of which this repository has
already recorded symptoms of:

    saturation      Softmax's Jacobian is diag(p) - p p^T, so as one bump comes
                    to dominate the gradient reaching the basis logits dies.
                    scripts/16's `parameter_groups` says so in as many words --
                    "the gradient reaching them is far smaller than the gradient
                    reaching the GRU" -- and works around it with a 10x learning
                    rate instead of changing the activation. Measured in
                    tests/test_simplex_activations.py: the gradient norm falls
                    from 2.2e-01 to 4.1e-09 as the logits scale by 20.

    scale           The logits are a product of two 0.1*randn tensors. Nothing
                    fixes their scale, so the effective temperature is an
                    accident of initialisation that drifts during training.

    density         Softmax cannot return zero, so every learned kernel is a
                    blend of all six bumps. A biological delay is one peak.
                    Blending six of them widens the curve and drags its centre
                    of mass toward the middle of the reach, and the centre of
                    mass is exactly the quantity docs/learnable_lags_results.md
                    compares against the measured delay -- finding r = -0.16.

So this script varies the activation and nothing else. Every arm shares
scripts/16's training loop, folds, masks, graph, architecture, seeds and
preprocessing, and scripts/18's encoder wiring, all by import rather than by
copy. The `softmax` arm is the control and must reproduce scripts/18's
`learned_lags` number; if it does not, the harness is not neutral and no other
number here means anything.

    softmax           the control -- what the encoder does today
    temp_softmax      softmax(z / tau), tau learned. Saturation and scale.
    sparsemax         Euclidean projection onto the simplex. Density.
    entmax15          alpha-entmax at alpha = 1.5, between the two.
    floored_entmax15  entmax15 with 2% uniform mass mixed back in, so a
                      collapsed mixture keeps a live gradient.
    gumbel_softmax    annealed stochastic sampling, against boundary drift.

What this can and cannot show. README §7 established that at h=1 the previous
period's case count carries nearly all the signal, so no delay treatment has
much room to move the headline MAE -- dropping every climate channel costs
+0.01 MAE. A better activation cannot repair that; it is a property of the
forecasting problem, not of the encoder. What it can do is make the encoder
recover the delay that scripts/17 measures independently, and that is why this
script reports the correlation between learned and measured lags as a first-
class result alongside MAE rather than as a diagnostic afterthought. An arm
that lifts r from -0.16 towards a positive number while holding MAE flat has
demonstrated something real about the parameterisation, and the honest way to
report it is exactly that -- not as a forecasting improvement.

Outputs:
    results/models/activation_report.md
    results/models/activation_metrics.csv
    results/models/activation_kernels.csv
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


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.models.lag_encoder import (  # noqa: E402
    DEFAULT_CENTRES,
    LagGCNGRU,
    LearnableLagEncoder,
)
from src.models.simplex_activations import SimplexActivation  # noqa: E402


RESULTS_DIR = PROJECT_DIR / "results" / "models"

REPORT_PATH = RESULTS_DIR / "activation_report.md"
METRICS_PATH = RESULTS_DIR / "activation_metrics.csv"
KERNELS_PATH = RESULTS_DIR / "activation_kernels.csv"

# The independently measured delays scripts/17 produces, used as the ground
# truth the learned kernels are scored against.
LAG_SCAN_PATH = PROJECT_DIR / "results" / "eda" / "lag_correlation.csv"

MODEL_LOOKBACK = 12
LAG_REACH = 26

ARM_NAMES = (
    "softmax",
    "temp_softmax",
    "sparsemax",
    "entmax15",
    "floored_entmax15",
    "gumbel_softmax",
)


def load_script(name: str, filename: str):
    """Import a numbered pipeline script, whose name is not a valid module."""

    spec = importlib.util.spec_from_file_location(name, PROJECT_DIR / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def build_arms(baseline, config: dict, n_nodes: int, lagged_indices: list[int], names):
    """Return {activation name: model builder}.

    Every builder produces the identical architecture and differs only in the
    string handed to the encoder, which is the whole design of this experiment.
    """

    def build(activation: str):
        def builder(n_features: int):
            backbone = baseline.GCNGRU(
                n_features=n_features,
                hidden=config["hidden"],
                gcn_layers=config["gcn_layers"],
                horizon=1,
                dropout=config["dropout"],
            )

            return LagGCNGRU(
                backbone=backbone,
                n_nodes=n_nodes,
                lagged_indices=lagged_indices,
                lag_reach=config["lag_reach"],
                n_basis=len(DEFAULT_CENTRES),
                embedding_dim=config["embedding_dim"],
                activation=activation,
            )

        return builder

    return {name: build(name) for name in names}


def measured_lags() -> pd.DataFrame | None:
    """Load scripts/17's cross-correlation delays, if they have been produced.

    Returns None rather than raising: the MAE comparison is still valid without
    them, and a missing optional input should degrade the report, not stop the
    run.
    """

    if not LAG_SCAN_PATH.exists():
        return None

    scan = pd.read_csv(LAG_SCAN_PATH)

    # The scan is long over `lag`, but `peak_lag` is a per-district constant
    # repeated down every row of the group, so one row per district-feature is
    # all that is wanted. Folds are averaged: the delay is a property of the
    # district, and a per-fold estimate would make the target move.
    rainfall = scan[scan["feature"] == "rainfall_daily_mean_mm"]
    if rainfall.empty:
        return None

    collapsed = (
        rainfall.groupby(["fold_id", "node_id"])["peak_lag"].first().reset_index()
    )

    return collapsed.groupby("node_id")["peak_lag"].mean().reset_index()


def run(baseline, arms, models, folds, config, device, feature_names, lagged_indices):
    """Train every arm on every backbone and fold, and score the test years."""

    calendar = baseline.folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()

    window = config["lag_reach"] + MODEL_LOOKBACK - 1

    rows: list[dict] = []
    kernel_rows: list[pd.DataFrame] = []

    tensors = baseline.folds_module.load_tensors("v0")

    for arm, build_model in arms.items():
        for fold in folds:
            arrays = baseline.build_fold_arrays(
                tensors, months, fold, window, config["horizon"]
            )

            fit_mask = tensors["period_id"] <= fold["fit_end_period"]
            thresholds = baseline.naive.peak_thresholds(
                tensors["y"], tensors["y_mask"], fit_mask
            )

            target = arrays["test"]["y"]
            mask = arrays["test"]["mask"].astype(np.int8)

            for backbone, adjacency in models.items():
                started = time.perf_counter()
                seed_mae = []

                for seed in range(config["seeds"]):
                    model, info = baseline.train_one(
                        arrays, adjacency, config, seed, device, build_model
                    )
                    prediction = baseline.predict(
                        model, arrays["test"], adjacency, config["target"], device
                    )
                    scores = baseline.naive.evaluate(
                        prediction, target, mask, thresholds
                    )

                    rows.append(
                        {
                            "arm": arm,
                            "backbone": backbone,
                            "fold_id": fold["fold_id"],
                            "test_year": fold["test_year"],
                            "headline": fold["headline"],
                            "covers_covid": fold["covers_covid"],
                            "seed": seed,
                            "train_windows": len(arrays["train"]["X"]),
                            "best_epoch": info["best_epoch"],
                            **scores,
                        }
                    )
                    seed_mae.append(scores["mae"])

                    kernel_rows.append(
                        kernel_frame(model, arm, backbone, fold, seed, feature_names, lagged_indices)
                    )

                print(
                    f"  {arm:<17} {backbone:<9} fold {fold['fold_id']} "
                    f"({fold['test_year']})  MAE {np.mean(seed_mae):7.2f} "
                    f"+/- {np.std(seed_mae):5.2f}  "
                    f"{time.perf_counter() - started:5.1f}s"
                )

    kernels = pd.concat(kernel_rows, ignore_index=True) if kernel_rows else None

    return pd.DataFrame(rows), kernels


def kernel_frame(model, arm, backbone, fold, seed, feature_names, lagged_indices):
    """Flatten one trained model's delay curves into long form.

    Records the support size alongside the peak lag, because support size is
    the mechanism this experiment varies: an arm that changes MAE without
    changing support has not changed what the hypothesis says it changes.
    """

    with torch.no_grad():
        encoder = model.encoder
        weights = encoder.mixture_weights().detach().cpu()
        peaks = encoder.peak_lags().detach().cpu().numpy()
        kernels = encoder.kernels().detach().cpu()

        tau = torch.arange(encoder.lag_reach, dtype=torch.float32)
        centre = (kernels * tau).sum(-1, keepdim=True)
        spread = (kernels * (tau - centre) ** 2).sum(-1).sqrt().numpy()

    support = (weights > 1e-6).sum(-1).numpy()
    names = [feature_names[index] for index in lagged_indices]

    records = []
    for node in range(peaks.shape[0]):
        for k, name in enumerate(names):
            records.append(
                {
                    "arm": arm,
                    "backbone": backbone,
                    "fold_id": fold["fold_id"],
                    "seed": seed,
                    "node_index": node,
                    "feature": name,
                    "peak_lag": float(peaks[node, k]),
                    "kernel_sd": float(spread[node, k]),
                    "support_size": int(support[node, k]),
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-fold, per-seed scores to one row per arm and backbone."""

    headline = metrics[metrics["headline"]]

    per_seed = (
        headline.groupby(["arm", "backbone", "seed"])["mae"].mean().reset_index()
    )
    seed_sd = per_seed.groupby(["arm", "backbone"])["mae"].std().rename("seed_sd")

    summary = (
        headline.groupby(["arm", "backbone"])
        .agg(
            headline_mae=("mae", "mean"),
            headline_rmse=("rmse", "mean"),
            headline_peak_mae=("peak_mae", "mean"),
        )
        .join(seed_sd)
        .reset_index()
    )

    return summary.sort_values(["backbone", "headline_mae"])


def kernel_summary(kernels: pd.DataFrame, measured: pd.DataFrame | None) -> pd.DataFrame:
    """One row per arm: how sparse, how narrow, and how well it recovers delay.

    The correlation is computed against scripts/17's independently measured
    per-district rainfall delay, averaged over folds and seeds so that a single
    unlucky fit does not decide the number.
    """

    rows = []

    for (arm, backbone), group in kernels.groupby(["arm", "backbone"]):
        record = {
            "arm": arm,
            "backbone": backbone,
            "mean_support": group["support_size"].mean(),
            "mean_kernel_sd": group["kernel_sd"].mean(),
            "mean_peak_lag": group["peak_lag"].mean(),
            "peak_lag_spread": group.groupby("node_index")["peak_lag"].mean().std(),
        }

        record["lag_correlation"] = correlate_with_measured(group, measured)
        rows.append(record)

    return pd.DataFrame(rows).sort_values(["backbone", "arm"])


def correlate_with_measured(group: pd.DataFrame, measured: pd.DataFrame | None):
    """Pearson r between learned and measured per-district rainfall delay.

    Returns NaN when the scan output is absent or its columns do not match, so
    a missing optional input shows as a blank cell rather than as a wrong
    number or a crash.
    """

    if measured is None:
        return float("nan")

    rainfall = group[group["feature"].str.contains("rainfall", case=False)]
    if rainfall.empty:
        return float("nan")

    learned = rainfall.groupby("node_index")["peak_lag"].mean()

    if not {"node_id", "peak_lag"} <= set(measured.columns):
        return float("nan")

    truth = measured.set_index("node_id")["peak_lag"]
    shared = learned.index.intersection(truth.index)

    if len(shared) < 3:
        return float("nan")

    values = np.corrcoef(learned.loc[shared].to_numpy(), truth.loc[shared].to_numpy())

    return float(values[0, 1])


def epidemic_share(metrics: pd.DataFrame, backbone: str, arm: str):
    """How much of an arm's headline gain comes from fold 1 alone.

    Returns (percentage of the gain contributed by fold 1, mean gain over the
    other headline folds), or (None, None) if the comparison cannot be made.

    This exists because the seed-sd test on its own has been misleading in this
    project before. README §8's `level_weighted` cleared the seed sd comfortably
    and was still, on inspection, one fold: 100.1% of the improvement came from
    2017 and the other six folds moved by -0.004 MAE. An activation change is
    at least as likely to have that shape, since the epidemic fold is where the
    wide case distribution makes any change to the climate path matter at all.
    """

    headline = metrics[metrics["headline"] & (metrics["backbone"] == backbone)]
    by_fold = headline.groupby(["arm", "fold_id"])["mae"].mean().unstack(0)

    if "softmax" not in by_fold.columns or arm not in by_fold.columns:
        return None, None

    gain = by_fold["softmax"] - by_fold[arm]
    if 1 not in gain.index or gain.sum() == 0:
        return None, None

    return float(gain.loc[1] / gain.sum() * 100.0), float(gain.drop(index=1).mean())


def write_report(summary, kernel_stats, metrics, config, elapsed) -> None:
    """Write the markdown report, stating limits in the same breath as results."""

    lines: list[str] = []
    add = lines.append

    add("# Simplex activations for the lag encoder")
    add("")
    add(
        f"Generated by `scripts/22.train_simplex_activations.py` in "
        f"{elapsed / 60:.1f} min. "
        f"{config['seeds']} seeds, horizon {config['horizon']}, "
        f"lag reach {config['lag_reach']}."
    )
    add("")
    add(
        "Every arm is the same architecture, training loop, folds, masks, "
        "seeds and preprocessing. The only difference is how the six basis "
        "logits are mapped onto the simplex. `softmax` is the control and is "
        "what the encoder does today."
    )
    add("")

    add("## Forecast accuracy")
    add("")
    add("Headline = mean over folds 1, 2, 3, 6, 7, 8, 9. Lower is better.")
    add("")
    add("| Backbone | Arm | MAE | RMSE | Peak MAE | Seed sd |")
    add("|---|---|---|---|---|---|")
    for row in summary.itertuples():
        add(
            f"| {row.backbone} | `{row.arm}` | {row.headline_mae:.2f} | "
            f"{row.headline_rmse:.2f} | {row.headline_peak_mae:.2f} | "
            f"{row.seed_sd:.2f} |"
        )
    add("")

    add("## Kernel shape and delay recovery")
    add("")
    add(
        "`mean support` is how many of the six bumps carry mass -- the "
        "mechanism this experiment varies. `kernel sd` is the width of the "
        "learned delay curve in periods. `r vs measured` is the Pearson "
        "correlation between the learned per-district rainfall delay and the "
        "independent cross-correlation delay from `scripts/17`; "
        "`docs/learnable_lags_results.md` reports **-0.16** for the softmax "
        "encoder, and moving that number is the point of this experiment."
    )
    add("")
    add("| Backbone | Arm | Mean support | Kernel sd | Mean peak lag | Across-district sd | r vs measured |")
    add("|---|---|---|---|---|---|---|")
    for row in kernel_stats.itertuples():
        correlation = (
            "n/a" if np.isnan(row.lag_correlation) else f"{row.lag_correlation:+.2f}"
        )
        add(
            f"| {row.backbone} | `{row.arm}` | {row.mean_support:.2f} | "
            f"{row.mean_kernel_sd:.2f} | {row.mean_peak_lag:.2f} | "
            f"{row.peak_lag_spread:.2f} | {correlation} |"
        )
    add("")

    add("## Reading this")
    add("")

    for backbone, group in summary.groupby("backbone"):
        indexed = group.set_index("arm")
        if "softmax" not in indexed.index:
            continue

        control = indexed.loc["softmax", "headline_mae"]
        sd = indexed.loc["softmax", "seed_sd"]
        best_arm = group.iloc[0]["arm"]
        best = group.iloc[0]["headline_mae"]

        add(f"**{backbone}.** Control (`softmax`) {control:.2f} MAE, seed sd {sd:.2f}.")

        if best_arm == "softmax":
            add(
                f"No alternative activation beats it. The best alternative is "
                f"{group.iloc[1]['arm']} at {group.iloc[1]['headline_mae']:.2f}."
            )
        else:
            margin = control - best
            verdict = (
                "larger than the control's seed sd"
                if margin > sd
                else "**smaller than the control's seed sd, so it is not "
                "distinguishable from seed noise**"
            )
            add(f"Best is `{best_arm}` at {best:.2f}, a margin of {margin:+.2f} -- {verdict}.")

            # Beating the seed sd is necessary and nowhere near sufficient. In
            # this project a headline gain has twice turned out to be one
            # epidemic fold carrying the mean (README §8), so the mean is
            # decomposed here rather than left for a reader to check.
            share, without = epidemic_share(metrics, backbone, best_arm)
            if share is not None:
                add("")
                add(
                    f"Decomposed by fold, however, fold 1 (the 2017 epidemic) "
                    f"contributes **{share:.0f}%** of that margin, and the mean "
                    f"over the other headline folds moves by {without:+.3f} MAE. "
                    + (
                        "**The gain is one fold, not a general improvement.**"
                        if abs(without) < 0.1
                        else "The remaining folds still move, so the gain is "
                        "not purely epidemic."
                    )
                )

        add("")

    add(
        "**What this experiment cannot show.** README §7 established that at "
        "horizon 1 the previous period's case count carries nearly all the "
        "forecasting signal -- removing every climate channel costs about "
        "+0.01 MAE on recent normal folds. No change to how the climate "
        "channels are smoothed can move the headline much against that, "
        "whatever the activation. A flat MAE column here is the expected "
        "result and is not evidence against the activation; the column that "
        "carries information at h=1 is delay recovery, not MAE. Run this at "
        "`--horizon 4` before concluding anything about accuracy."
    )
    add("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--lag-reach", type=int, default=LAG_REACH)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--kernel-learning-rate", type=float, default=2e-2)
    parser.add_argument("--arms", nargs="+", default=list(ARM_NAMES))
    parser.add_argument("--no-control", action="store_true")
    arguments = parser.parse_args()

    unknown = set(arguments.arms) - set(SimplexActivation.NAMES)
    if unknown:
        parser.error(f"unknown arms {sorted(unknown)}; expected {SimplexActivation.NAMES}")

    lag_module = load_script("train_lag_gcn_gru", "training/18.train_lag_gcn_gru.py")
    baseline = lag_module.load_baseline()

    config = dict(baseline.DEFAULTS)
    config["target"] = "residual"
    config["seeds"] = arguments.seeds
    config["lag_reach"] = arguments.lag_reach
    config["embedding_dim"] = arguments.embedding_dim
    config["kernel_learning_rate"] = arguments.kernel_learning_rate
    if arguments.horizon is not None:
        config["horizon"] = arguments.horizon

    folds = json.loads(baseline.FOLDS_PATH.read_text(encoding="utf-8"))["folds"]
    if arguments.folds:
        folds = [fold for fold in folds if fold["fold_id"] in arguments.folds]

    with np.load(baseline.ADJACENCY_PATH, allow_pickle=True) as data:
        adjacency = data["A_norm"].astype(np.float32)

    models = {"gcn_gru": adjacency}
    if not arguments.no_control:
        models["gru_only"] = np.eye(len(adjacency), dtype=np.float32)

    tensors_v0 = baseline.folds_module.load_tensors("v0")
    feature_names = list(tensors_v0["feature_names"])
    lagged_indices = [
        feature_names.index(name) for name in lag_module.LAGGED_FEATURES
    ]
    n_nodes = len(adjacency)

    arms = build_arms(baseline, config, n_nodes, lagged_indices, arguments.arms)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe = LearnableLagEncoder(
        n_nodes=n_nodes,
        n_features=len(lag_module.LAGGED_FEATURES),
        lag_reach=config["lag_reach"],
        embedding_dim=config["embedding_dim"],
    )

    print(f"Device:            {device}")
    print(f"Folds:             {[fold['fold_id'] for fold in folds]}")
    print(f"Arms:              {list(arms)}")
    print(f"Backbones:         {list(models)}")
    print(f"Seeds:             {config['seeds']}")
    print(f"Horizon:           {config['horizon']}")
    print(f"Encoder params:    {sum(p.numel() for p in probe.parameters())}")
    print(f"Input window:      {config['lag_reach'] + MODEL_LOOKBACK - 1} periods\n")

    started = time.perf_counter()
    metrics, kernels = run(
        baseline, arms, models, folds, config, device, feature_names, lagged_indices
    )
    elapsed = time.perf_counter() - started

    summary = summarise(metrics)
    stats = kernel_summary(kernels, measured_lags())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    kernels.to_csv(KERNELS_PATH, index=False)

    write_report(summary, stats, metrics, config, elapsed)

    print(f"\nTrained in {elapsed / 60:.1f} min\n")
    print(f"{'backbone':<10} {'arm':<18} {'MAE':>8} {'peakMAE':>9} {'seed sd':>8} {'support':>8} {'kern sd':>8}")

    merged = summary.merge(stats, on=["arm", "backbone"], how="left")
    for row in merged.itertuples():
        print(
            f"{row.backbone:<10} {row.arm:<18} {row.headline_mae:>8.2f} "
            f"{row.headline_peak_mae:>9.2f} {row.seed_sd:>8.2f} "
            f"{row.mean_support:>8.2f} {row.mean_kernel_sd:>8.2f}"
        )

    print(f"\nWrote {REPORT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {METRICS_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {KERNELS_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

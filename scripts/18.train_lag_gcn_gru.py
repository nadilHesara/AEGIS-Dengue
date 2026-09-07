"""
Train the GCN+GRU with learnable climate lags, against the hand-coded windows.

The question this script answers is narrow on purpose:

    the baseline is handed trailing 4, 8 and 12 period means of rainfall,
    temperature and humidity, windows chosen by hand and identical for all 25
    districts. Does letting the model learn a district-specific delay instead
    do better than that?

So the comparison is run three ways on the same folds, the same seeds and the
same masks:

    v1              the baseline's hand-coded windows, no encoder
    v0 + lags       raw features, learnable per-district delay kernels
    v0              raw features, no lag information at all -- the floor

Everything except the delay handling is held fixed. The graph, the GRU, the
head, the anchored target, the masked loss, the early stopping and the per-fold
refitted preprocessing all come from scripts/16 by import, not by copy, so a
difference in the numbers cannot come from a difference in the training loop.

The graph is deliberately untouched here. In the baseline report `gru_only`
beats `gcn_gru`, which is a real problem, but it is a problem about the graph.
Changing the delay and the graph in the same experiment would leave neither
result attributable, so the encoder is measured on both backbones and the graph
is left for its own phase.

One thing does change, and it has to: the input window. A 26-period lag reach
needs 26 periods of history behind the first step the backbone sees, so the
window is 12 + 26 - 1 = 37 periods rather than 12. That costs about 25 windows
per fold, and the count is reported so the cost is visible rather than implied.

Outputs:
    results/models/lag_report.md
    results/models/lag_metrics.csv
    results/models/learned_lags.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.models.lag_encoder import (  # noqa: E402
    DEFAULT_CENTRES,
    LagGCNGRU,
    LearnableLagEncoder,
)


PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "models"

REPORT_PATH = RESULTS_DIR / "lag_report.md"
METRICS_PATH = RESULTS_DIR / "lag_metrics.csv"
KERNELS_PATH = RESULTS_DIR / "learned_lags.csv"

MODEL_LOOKBACK = 12
LAG_REACH = 26

# The climate channels. Case history, seasonality, the observation flags and
# the centroids are passed through unsmoothed -- see LagGCNGRU.
LAGGED_FEATURES = (
    "rainfall_daily_mean_mm",
    "rainy_days_frac",
    "temperature_mean_c",
    "diurnal_range_c",
    "dewpoint_mean_c",
    "relative_humidity_mean",
    "wind_speed_mean",
)


def load_baseline():
    """Load scripts/16 and everything it already loaded."""

    spec = importlib.util.spec_from_file_location(
        "baseline", PROJECT_DIR / "scripts" / "16.train_gcn_gru.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_modules()

    return module


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------

class FeatureSubsetGCNGRU(nn.Module):
    """The baseline backbone, shown only some of the input channels.

    The floor beneath the floor. `no_lags_v0` still receives every climate
    feature at the current period, so it answers "does *delayed* climate help",
    not "does climate help". Dropping the climate channels entirely answers the
    second question, and the second question is the one that decides whether
    any delay treatment could ever have won: if case history, seasonality and
    geography alone score what the climate arms score, then the lag structure
    is not the thing standing between this model and a better number.

    Implemented by slicing rather than by zeroing. A zeroed channel still costs
    the first graph convolution its weights and still passes through the
    per-fold scaler, so it is not the same experiment.
    """

    def __init__(self, backbone: nn.Module, keep_indices: list[int]):
        super().__init__()

        self.backbone = backbone
        self.register_buffer(
            "keep_indices", torch.tensor(keep_indices, dtype=torch.long)
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.backbone(x[..., self.keep_indices], adjacency)


def build_arms(baseline, config: dict, n_nodes: int, lagged_indices: list[int]):
    """Return {arm name: (variant, lookback, model builder)}.

    The builder takes the input feature count and returns a fresh module, which
    is what scripts/16's `train_one` asks for.
    """

    def plain(n_features):
        # The head emits one number, not `config["horizon"]` of them.
        #
        # `horizon` in this pipeline means "how far ahead the target sits", not
        # "how many steps to emit": scripts/12's `make_windows` returns a single
        # target at t + horizon, and scripts/16's `predict` does `squeeze(-1)`.
        # Sizing the head by the horizon would leave h - 1 extra outputs trained
        # against the same target by broadcast and then break `predict`, both
        # silently. Passing the horizon through to the window builder below is
        # what actually moves the forecast further out.
        return baseline.GCNGRU(
            n_features=n_features,
            hidden=config["hidden"],
            gcn_layers=config["gcn_layers"],
            horizon=1,
            dropout=config["dropout"],
        )

    def with_lags(n_features):
        return LagGCNGRU(
            backbone=plain(n_features),
            n_nodes=n_nodes,
            lagged_indices=lagged_indices,
            lag_reach=config["lag_reach"],
            n_basis=len(DEFAULT_CENTRES),
            embedding_dim=config["embedding_dim"],
        )

    def without_climate(n_features):
        # `n_features` is the full v0 width; the backbone is sized for what is
        # left after the climate channels come out, so the control differs from
        # `no_lags_v0` in its inputs and in nothing else.
        dropped = set(lagged_indices)
        kept = [index for index in range(n_features) if index not in dropped]

        return FeatureSubsetGCNGRU(backbone=plain(len(kept)), keep_indices=kept)

    window = config["lag_reach"] + MODEL_LOOKBACK - 1

    return {
        "hand_lags_v1": ("v1", MODEL_LOOKBACK, plain),
        "learned_lags": ("v0", window, with_lags),
        "no_lags_v0": ("v0", MODEL_LOOKBACK, plain),
        "no_climate": ("v0", MODEL_LOOKBACK, without_climate),
    }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run(baseline, arms, models, folds, config, device):
    """Train every arm on every backbone and fold, and score the test years."""

    calendar = baseline.folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()

    rows: list[dict] = []
    kernel_rows: list[pd.DataFrame] = []

    for arm, (variant, lookback, build_model) in arms.items():
        tensors = baseline.folds_module.load_tensors(variant)

        for fold in folds:
            arrays = baseline.build_fold_arrays(
                tensors, months, fold, lookback, config["horizon"]
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
                            "variant": variant,
                            "lookback": lookback,
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

                    if isinstance(model, LagGCNGRU):
                        kernel_rows.append(
                            kernel_frame(model, arm, backbone, fold, seed)
                        )

                print(
                    f"  {arm:<13} {backbone:<9} fold {fold['fold_id']} "
                    f"({fold['test_year']})  MAE {np.mean(seed_mae):7.2f} "
                    f"+/- {np.std(seed_mae):5.2f}  "
                    f"n={len(arrays['train']['X']):4d}  "
                    f"{time.perf_counter() - started:5.1f}s"
                )

    kernels = pd.concat(kernel_rows, ignore_index=True) if kernel_rows else None

    return pd.DataFrame(rows), kernels


def kernel_frame(model, arm, backbone, fold, seed) -> pd.DataFrame:
    """Flatten one trained model's delay curves into long form.

    Saved per fold and per seed rather than averaged. A kernel that is stable
    across seeds is a finding; one that is not is a different finding, and
    averaging first would hide which of the two happened.
    """

    with torch.no_grad():
        kernels = model.encoder.kernels().cpu().numpy()
        peaks = model.encoder.peak_lags().cpu().numpy()

    n_nodes, n_features, lag_reach = kernels.shape
    nodes = pd.read_csv(PROCESSED_DIR / "nodes.csv").sort_values("node_id")
    names = nodes["canonical_name"].tolist()

    return pd.DataFrame(
        {
            "arm": arm,
            "backbone": backbone,
            "fold_id": fold["fold_id"],
            "test_year": fold["test_year"],
            "seed": seed,
            "node_id": np.repeat(np.arange(n_nodes), n_features * lag_reach),
            "canonical_name": np.repeat(names, n_features * lag_reach),
            "feature": np.tile(
                np.repeat(list(LAGGED_FEATURES), lag_reach), n_nodes
            ),
            "lag": np.tile(np.arange(lag_reach), n_nodes * n_features),
            "weight": kernels.reshape(-1),
            "peak_lag": np.repeat(peaks.reshape(-1), lag_reach),
        }
    )


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average over seeds, then over the headline folds."""

    by_fold = (
        metrics.groupby(
            ["arm", "backbone", "fold_id", "test_year", "headline", "covers_covid"]
        )
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), peak_mae=("peak_mae", "mean"))
        .reset_index()
    )

    records = []
    for (arm, backbone), group in by_fold.groupby(["arm", "backbone"]):
        headline = group[group["headline"]]
        epidemic = group[group["test_year"] == 2017]
        spread = (
            metrics[(metrics["arm"] == arm) & (metrics["backbone"] == backbone)]
            .groupby("fold_id")["mae"]
            .std()
            .mean()
        )

        records.append(
            {
                "arm": arm,
                "backbone": backbone,
                "headline_mae": headline["mae"].mean(),
                "headline_rmse": headline["rmse"].mean(),
                "headline_peak_mae": headline["peak_mae"].mean(),
                "epidemic_2017_mae": epidemic["mae"].mean(),
                "seed_sd": spread,
            }
        )

    return pd.DataFrame(records).sort_values(["backbone", "headline_mae"])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(summary, metrics, kernels, config) -> None:
    """Write the lag report."""

    notes = {
        "hand_lags_v1": (
            "`v1`",
            "trailing 4, 8, 12 period means, hand-chosen, same for all districts",
        ),
        "learned_lags": (
            "`v0`",
            f"learned per-district kernels over {config['lag_reach']} periods",
        ),
        "no_lags_v0": ("`v0`", "none -- climate at the current period only"),
        "no_climate": (
            "`v0` minus climate",
            "no climate channels at all -- cases, seasonality and geography only",
        ),
    }

    present = [arm for arm in notes if arm in set(metrics["arm"])]

    lines = [
        "# Learnable climate lags",
        "",
        f"{len(present)} arms on the same folds, seeds and masks. Only the climate",
        "handling differs; the graph, GRU, head, target and loss are the baseline's.",
        "",
        "| Arm | Features | Climate handling |",
        "| --- | --- | --- |",
    ]

    lines += [
        f"| `{arm}` | {notes[arm][0]} | {notes[arm][1]} |" for arm in present
    ]

    lines += [
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
    ]

    for key, value in config.items():
        lines.append(f"| `{key}` | {value} |")

    lines += [
        "",
        "## Results",
        "",
        "Mean over the headline folds, averaged over seeds.",
        "",
        "| Backbone | Arm | MAE | RMSE | Peak MAE | 2017 MAE | Seed sd |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in summary.itertuples():
        lines.append(
            f"| {row.backbone} | `{row.arm}` | {row.headline_mae:.2f} | "
            f"{row.headline_rmse:.2f} | {row.headline_peak_mae:.2f} | "
            f"{row.epidemic_2017_mae:.2f} | {row.seed_sd:.2f} |"
        )

    lines += ["", "## Verdict", ""]

    for backbone in summary["backbone"].unique():
        arms = summary[summary["backbone"] == backbone].set_index("arm")
        if {"learned_lags", "hand_lags_v1"} <= set(arms.index):
            learned = arms.loc["learned_lags", "headline_mae"]
            hand = arms.loc["hand_lags_v1", "headline_mae"]
            delta = 100 * (hand - learned) / hand
            verdict = "better than" if delta > 0 else "worse than"
            lines.append(
                f"- **{backbone}**: learned lags {learned:.2f} MAE, "
                f"{verdict} hand-coded windows at {hand:.2f} ({delta:+.1f}%)."
            )

        # The prior question. If dropping climate entirely costs nothing, the
        # gap between the delay arms above is not a result about delays.
        if {"no_climate", "no_lags_v0"} <= set(arms.index):
            blind = arms.loc["no_climate", "headline_mae"]
            seeing = arms.loc["no_lags_v0", "headline_mae"]
            cost = 100 * (blind - seeing) / seeing
            lines.append(
                f"- **{backbone}**: dropping climate entirely costs "
                f"{cost:+.1f}% MAE ({seeing:.2f} to {blind:.2f}). Read the "
                "delay arms against this: a delay treatment cannot be worth "
                "more than climate itself is."
            )

    windows = metrics.groupby("arm")["train_windows"].max()
    lines += [
        "",
        "## Cost of the longer window",
        "",
        "The 26-period reach needs 37 input periods rather than 12, which costs",
        "training windows in every fold.",
        "",
        "| Arm | Max train windows |",
        "| --- | --- |",
    ]
    for arm, count in windows.items():
        lines.append(f"| `{arm}` | {count} |")

    if kernels is not None:
        peaks = (
            kernels.drop_duplicates(["fold_id", "seed", "node_id", "feature"])
            .groupby("feature")["peak_lag"]
            .agg(["mean", "min", "max"])
            .round(1)
        )

        lines += [
            "",
            "## Learned delays",
            "",
            "Centre of mass of each kernel, in reporting periods, over all folds,",
            "seeds and districts. The spread across districts is the point: a",
            "narrow range would mean the per-district embedding learned nothing.",
            "",
            "| Feature | Mean lag | Min | Max |",
            "| --- | --- | --- | --- |",
        ]
        for feature, row in peaks.iterrows():
            lines.append(
                f"| `{feature}` | {row['mean']:.1f} | {row['min']:.1f} | {row['max']:.1f} |"
            )

    lines += [
        "",
        "## Output files",
        "",
        f"- `{METRICS_PATH.relative_to(PROJECT_DIR).as_posix()}`",
        f"- `{KERNELS_PATH.relative_to(PROJECT_DIR).as_posix()}`",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=3)
    # The horizon is the one setting under which the encoder is still expected
    # to win. At h=1 the previous period's case count carries nearly all the
    # information and the climate channels are close to redundant; the further
    # ahead the forecast reaches, the more it has to lean on delayed climate.
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--lag-reach", type=int, default=LAG_REACH)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--kernel-learning-rate", type=float, default=2e-2)
    parser.add_argument("--no-control", action="store_true")
    parser.add_argument("--arms", nargs="+", default=None)
    arguments = parser.parse_args()

    baseline = load_baseline()

    config = dict(baseline.DEFAULTS)
    config["target"] = "residual"
    config["seeds"] = arguments.seeds
    config["lag_reach"] = arguments.lag_reach
    config["embedding_dim"] = arguments.embedding_dim
    config["kernel_learning_rate"] = arguments.kernel_learning_rate
    if arguments.horizon is not None:
        config["horizon"] = arguments.horizon

    import json

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
    lagged_indices = [feature_names.index(name) for name in LAGGED_FEATURES]
    n_nodes = len(adjacency)

    arms = build_arms(baseline, config, n_nodes, lagged_indices)
    if arguments.arms:
        arms = {name: arms[name] for name in arguments.arms}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe = LearnableLagEncoder(
        n_nodes=n_nodes,
        n_features=len(LAGGED_FEATURES),
        lag_reach=config["lag_reach"],
        embedding_dim=config["embedding_dim"],
    )
    encoder_parameters = sum(p.numel() for p in probe.parameters())

    print(f"Device:            {device}")
    print(f"Folds:             {[fold['fold_id'] for fold in folds]}")
    print(f"Arms:              {list(arms)}")
    print(f"Backbones:         {list(models)}")
    print(f"Lagged features:   {len(LAGGED_FEATURES)} of {len(feature_names)}")
    print(f"Encoder params:    {encoder_parameters}")
    print(f"Input window:      {config['lag_reach'] + MODEL_LOOKBACK - 1} periods\n")

    started = time.perf_counter()
    metrics, kernels = run(baseline, arms, models, folds, config, device)
    elapsed = time.perf_counter() - started

    summary = summarise(metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    if kernels is not None:
        kernels.to_csv(KERNELS_PATH, index=False)

    write_report(summary, metrics, kernels, config)

    print(f"\nTrained in {elapsed / 60:.1f} min\n")
    print(f"{'backbone':<10} {'arm':<14} {'MAE':>8} {'RMSE':>8} {'peakMAE':>9} {'seed sd':>8}")
    for row in summary.itertuples():
        print(
            f"{row.backbone:<10} {row.arm:<14} {row.headline_mae:>8.2f} "
            f"{row.headline_rmse:>8.2f} {row.headline_peak_mae:>9.2f} "
            f"{row.seed_sd:>8.2f}"
        )

    print(f"\nWrote {REPORT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {METRICS_PATH.relative_to(PROJECT_DIR)}")
    if kernels is not None:
        print(f"Wrote {KERNELS_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

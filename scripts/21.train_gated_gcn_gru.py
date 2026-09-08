"""
Train the GCN+GRU with a per-district spatial/temporal fusion gate.

The question is narrow. In the committed baseline sweep `gru_only` beats
`gcn_gru` on every headline fold -- cutting the contiguity graph out entirely
lowers the error. That is a statement about the average district. This script
asks whether letting each district choose, per district, how much of the graph
convolution to keep does better than either fixed choice.

    fused[i] = g[i] * spatial[i] + (1 - g[i]) * temporal[i]

`spatial` is the baseline's graph convolution over the real adjacency;
`temporal` is the same convolution with the same weights over the identity;
`g[i] = sigmoid(gate_logit[i])` is one learned scalar per district. With every
g = 1 the model is exactly `gcn_gru`; with every g = 0 it is exactly
`gru_only`. See `src/models/gated_fusion.py`.

Four arms on the same folds, seeds and masks:

    gcn_gru         the baseline, real adjacency                 the control
    gru_only        the baseline, identity adjacency             the number to beat
    gated           GatedGCNGRU, per-district gate learned        the contribution
    gated_uniform   GatedGCNGRU, gate frozen at 0.5              ablation: does *learning* the gate matter

`gcn_gru` and `gru_only` are re-run here through this script's harness rather
than read from the committed report: if they do not reproduce `scripts/16`'s
numbers, the harness is not neutral and nothing else here is comparable. The
training loop, fold construction, preprocessing, windowing, anchored target,
masked loss and metrics are all **imported** from `scripts/16.train_gcn_gru.py`,
so a difference in the numbers cannot come from a difference in the loop.

The graph itself is untouched -- this is the fusion half of workplan Phase 5,
not the multiplex-relation half. Only the spatial/temporal blend moves.

Outputs:
    results/models/gated_report.md
    results/models/gated_metrics.csv
    results/models/gated_gates.csv
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


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.models.gated_fusion import GatedGCNGRU  # noqa: E402


PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "models"

REPORT_PATH = RESULTS_DIR / "gated_report.md"
METRICS_PATH = RESULTS_DIR / "gated_metrics.csv"
GATES_PATH = RESULTS_DIR / "gated_gates.csv"

GATED_VERSION = "gated-v1"

# arm -> (adjacency key, model builder tag). "real" uses adjacency.npz's A_norm,
# "identity" uses the 25x25 identity -- matching how scripts/16 defines the
# gcn_gru / gru_only pair.
ARMS = {
    "gcn_gru": ("real", "plain"),
    "gru_only": ("identity", "plain"),
    "gated": ("real", "gated"),
    "gated_uniform": ("real", "gated_uniform"),
}


def load_baseline():
    """Load scripts/16 and everything it already loaded.

    Mirrors scripts/18.load_baseline exactly: import the module by path, then
    call its load_modules() so naive / folds_module / tensors_module are wired.
    """

    spec = importlib.util.spec_from_file_location(
        "baseline", PROJECT_DIR / "scripts" / "16.train_gcn_gru.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_modules()

    return module


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def build_model_builders(baseline, config: dict, n_nodes: int):
    """Return {tag: build_model callable}.

    Each callable takes the input feature count and returns a fresh module, the
    signature scripts/16's train_one expects for its `build_model` hook.
    """

    def plain(n_features):
        return baseline.GCNGRU(
            n_features=n_features,
            hidden=config["hidden"],
            gcn_layers=config["gcn_layers"],
            horizon=config["horizon"],
            dropout=config["dropout"],
        )

    def gated(n_features):
        return GatedGCNGRU(backbone=plain(n_features), n_nodes=n_nodes)

    def gated_uniform(n_features):
        return GatedGCNGRU(
            backbone=plain(n_features), n_nodes=n_nodes, freeze_gate=0.5
        )

    return {"plain": plain, "gated": gated, "gated_uniform": gated_uniform}


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run(baseline, arms, adjacencies, builders, folds, config, device):
    """Train every arm on every fold and score the test years."""

    calendar = baseline.folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()

    nodes = pd.read_csv(PROCESSED_DIR / "nodes.csv").sort_values("node_id")
    names = nodes["canonical_name"].tolist()

    rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    gate_rows: list[pd.DataFrame] = []

    for variant in config["variants"]:
        tensors = baseline.folds_module.load_tensors(variant)

        for fold in folds:
            arrays = baseline.build_fold_arrays(
                tensors, months, fold, config["lookback"], config["horizon"]
            )

            fit_mask = tensors["period_id"] <= fold["fit_end_period"]
            thresholds = baseline.naive.peak_thresholds(
                tensors["y"], tensors["y_mask"], fit_mask
            )

            target = arrays["test"]["y"]
            mask = arrays["test"]["mask"].astype(np.int8)

            for arm, (adjacency_key, builder_tag) in arms.items():
                adjacency = adjacencies[adjacency_key]
                build_model = builders[builder_tag]

                started = time.perf_counter()
                seed_predictions = []
                seed_mae = []
                seed_epochs = []

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
                            "variant": variant,
                            "fold_id": fold["fold_id"],
                            "test_year": fold["test_year"],
                            "headline": fold["headline"],
                            "covers_covid": fold["covers_covid"],
                            "seed": seed,
                            "best_epoch": info["best_epoch"],
                            **scores,
                        }
                    )
                    seed_predictions.append(prediction)
                    seed_mae.append(scores["mae"])
                    seed_epochs.append(info["best_epoch"])

                    if isinstance(model, GatedGCNGRU) and arm == "gated":
                        gate_rows.append(
                            gate_frame(model, arm, variant, fold, seed, names)
                        )

                mean_prediction = np.mean(seed_predictions, axis=0)
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "arm": arm,
                            "variant": variant,
                            "fold_id": fold["fold_id"],
                            "target_period_id": np.repeat(
                                arrays["test"]["target_period_id"], len(names)
                            ),
                            "node_id": np.tile(np.arange(len(names)), len(target)),
                            "canonical_name": np.tile(names, len(target)),
                            "predicted": mean_prediction.reshape(-1),
                            "actual": target.reshape(-1),
                            "observed": mask.reshape(-1),
                        }
                    )
                )

                print(
                    f"  {variant} fold {fold['fold_id']} ({fold['test_year']}) "
                    f"{arm:<14} MAE {np.mean(seed_mae):7.2f} "
                    f"+/- {np.std(seed_mae):5.2f}  "
                    f"epochs {np.mean(seed_epochs):5.1f}  "
                    f"{time.perf_counter() - started:5.1f}s",
                    flush=True,
                )

    gates = pd.concat(gate_rows, ignore_index=True) if gate_rows else None

    return (
        pd.DataFrame(rows),
        pd.concat(prediction_rows, ignore_index=True),
        gates,
    )


def gate_frame(model, arm, variant, fold, seed, names) -> pd.DataFrame:
    """One trained gated model's per-district gate values, long form.

    Saved per fold and per seed rather than averaged. A gate that is stable
    across seeds is a finding; one that is not is a different finding, and
    averaging first would hide which happened. (Same reasoning as
    scripts/18.kernel_frame.)
    """

    values = model.gate_values()

    return pd.DataFrame(
        {
            "arm": arm,
            "variant": variant,
            "fold_id": fold["fold_id"],
            "test_year": fold["test_year"],
            "seed": seed,
            "node_id": np.arange(len(names)),
            "canonical_name": names,
            "gate": values,
        }
    )


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average over seeds, then over the headline folds. Mirrors scripts/16.summarise."""

    by_fold = (
        metrics.groupby(
            ["arm", "variant", "fold_id", "test_year", "headline", "covers_covid"]
        )
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), peak_mae=("peak_mae", "mean"))
        .reset_index()
    )

    records = []
    for (arm, variant), group in by_fold.groupby(["arm", "variant"]):
        headline = group[group["headline"]]
        covid = group[group["covers_covid"]]
        epidemic = group[group["test_year"] == 2017]

        spread = (
            metrics[(metrics["arm"] == arm) & (metrics["variant"] == variant)]
            .groupby("fold_id")["mae"]
            .std()
            .mean()
        )

        records.append(
            {
                "arm": arm,
                "variant": variant,
                "headline_mae": headline["mae"].mean(),
                "headline_rmse": headline["rmse"].mean(),
                "headline_peak_mae": headline["peak_mae"].mean(),
                "epidemic_2017_mae": epidemic["mae"].mean(),
                "epidemic_2017_peak_mae": epidemic["peak_mae"].mean(),
                "covid_mae": covid["mae"].mean(),
                "seed_sd": spread,
            }
        )

    return pd.DataFrame(records).sort_values("headline_mae").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(summary, by_fold, gates, config) -> None:
    """Write the gated-fusion report, in the shape of lag_report.md."""

    lines = [
        "# GCN+GRU with a spatial/temporal fusion gate",
        "",
        f"Version: `{GATED_VERSION}`",
        "",
        "Per district, a learned scalar `g` blends the graph convolution over the",
        "real adjacency against the same convolution over the identity. `g = 1`",
        "for every district is `gcn_gru`; `g = 0` is `gru_only`. Only the blend",
        "moves; the graph, GRU, head, target and loss are the baseline's, imported",
        "from `scripts/16`.",
        "",
        "| Arm | Adjacency | Gate |",
        "| --- | --- | --- |",
        "| `gcn_gru` | contiguity | none -- the control |",
        "| `gru_only` | identity | none -- the number to beat |",
        "| `gated` | contiguity | per-district, learned |",
        "| `gated_uniform` | contiguity | frozen at 0.5 |",
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
        "Mean over the headline folds (1, 2, 3, 6, 7, 8, 9), averaged over seeds.",
        "COVID folds are excluded from the headline and reported separately.",
        "",
        "| Arm | Features | MAE | RMSE | Peak MAE | 2017 MAE | COVID MAE | Seed sd |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in summary.itertuples():
        lines.append(
            f"| {row.arm} | `{row.variant}` | {row.headline_mae:.2f} | "
            f"{row.headline_rmse:.2f} | {row.headline_peak_mae:.2f} | "
            f"{row.epidemic_2017_mae:.2f} | {row.covid_mae:.2f} | {row.seed_sd:.2f} |"
        )

    # Verdict: gated against gru_only, the specific failure it targets.
    by_arm = summary.set_index("arm")
    if {"gated", "gru_only"} <= set(by_arm.index):
        gated_mae = by_arm.loc["gated", "headline_mae"]
        gru_mae = by_arm.loc["gru_only", "headline_mae"]
        gcn_mae = by_arm.loc["gcn_gru", "headline_mae"] if "gcn_gru" in by_arm.index else float("nan")
        delta = 100 * (gru_mae - gated_mae) / gru_mae
        verdict = "beats" if delta > 0 else "does not beat"
        lines += [
            "",
            "## Verdict",
            "",
            f"`gated` headline MAE {gated_mae:.2f} {verdict} `gru_only` at "
            f"{gru_mae:.2f} ({delta:+.1f}%); `gcn_gru` is {gcn_mae:.2f}.",
            "",
            "**Read the seed sd column before believing any gap.** A difference",
            "smaller than the seed sd is not established by this run.",
        ]

    lines += [
        "",
        "## Per fold, MAE",
        "",
        "| Arm | Features | Fold | Test year | MAE | RMSE | Peak MAE | Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in by_fold.sort_values(["arm", "variant", "fold_id"]).itertuples():
        note = "COVID" if row.covers_covid else ("epidemic" if row.test_year == 2017 else "")
        lines.append(
            f"| {row.arm} | `{row.variant}` | {row.fold_id} | {row.test_year} | "
            f"{row.mae:.2f} | {row.rmse:.2f} | {row.peak_mae:.2f} | {note} |"
        )

    if gates is not None:
        lines += [
            "",
            "## Learned gate per district",
            "",
            "`g` near 1 means the district keeps the graph convolution; near 0",
            "means it routes past it. Mean over folds and seeds, with the spread.",
            "The check is whether the low-degree northern districts (Jaffna,",
            "Mannar, Mullaitivu, Kilinochchi) gate lower than the well-connected",
            "ones.",
            "",
            "| District | Mean g | Min | Max |",
            "| --- | --- | --- | --- |",
        ]
        per_district = (
            gates.groupby("canonical_name")["gate"]
            .agg(["mean", "min", "max"])
            .sort_values("mean")
        )
        for name, row in per_district.iterrows():
            lines.append(
                f"| {name} | {row['mean']:.3f} | {row['min']:.3f} | {row['max']:.3f} |"
            )

    lines += [
        "",
        "## Output files",
        "",
        f"- `{METRICS_PATH.relative_to(PROJECT_DIR).as_posix()}`",
        f"- `{GATES_PATH.relative_to(PROJECT_DIR).as_posix()}`",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--variants", nargs="+", default=["v1"])
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument(
        "--arms", nargs="+", default=list(ARMS), choices=list(ARMS)
    )
    parser.add_argument(
        "--target", choices=["residual", "direct"], default="residual"
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    baseline = load_baseline()

    if not baseline.FOLDS_PATH.exists() or not baseline.ADJACENCY_PATH.exists():
        print("Missing folds.json or adjacency.npz. Run scripts 13 and 14 first.")
        return 1

    config = dict(baseline.DEFAULTS)
    config["target"] = arguments.target
    config["seeds"] = arguments.seeds
    config["variants"] = arguments.variants
    if arguments.horizon is not None:
        config["horizon"] = arguments.horizon

    folds = json.loads(baseline.FOLDS_PATH.read_text(encoding="utf-8"))["folds"]
    if arguments.folds:
        folds = [fold for fold in folds if fold["fold_id"] in arguments.folds]

    with np.load(baseline.ADJACENCY_PATH, allow_pickle=True) as data:
        real_adjacency = data["A_norm"].astype(np.float32)

    n_nodes = len(real_adjacency)
    adjacencies = {
        "real": real_adjacency,
        "identity": np.eye(n_nodes, dtype=np.float32),
    }

    arms = {name: ARMS[name] for name in arguments.arms}
    builders = build_model_builders(baseline, config, n_nodes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device:   {device}")
    print(f"Folds:    {[fold['fold_id'] for fold in folds]}")
    print(f"Variants: {config['variants']}")
    print(f"Arms:     {list(arms)}")
    print(f"Seeds:    {config['seeds']}\n")

    started = time.perf_counter()
    metrics, predictions, gates = run(
        baseline, arms, adjacencies, builders, folds, config, device
    )
    elapsed = time.perf_counter() - started

    summary = summarise(metrics)

    by_fold = (
        metrics.groupby(
            ["arm", "variant", "fold_id", "test_year", "headline", "covers_covid"]
        )
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), peak_mae=("peak_mae", "mean"))
        .reset_index()
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    predictions.to_csv(PROJECT_DIR / "results" / "models" / "gated_predictions.csv", index=False)
    if gates is not None:
        gates.to_csv(GATES_PATH, index=False)

    write_report(summary, by_fold, gates, config)

    print(f"\nTrained in {elapsed / 60:.1f} min\n")
    print(
        f"{'arm':<14} {'feat':<5} {'MAE':>8} {'RMSE':>8} {'peakMAE':>9} "
        f"{'2017':>8} {'seed sd':>8}"
    )
    for row in summary.itertuples():
        print(
            f"{row.arm:<14} {row.variant:<5} {row.headline_mae:>8.2f} "
            f"{row.headline_rmse:>8.2f} {row.headline_peak_mae:>9.2f} "
            f"{row.epidemic_2017_mae:>8.2f} {row.seed_sd:>8.2f}"
        )

    if gates is not None:
        print("\nlearned gate, mean over folds/seeds:")
        per_district = (
            gates.groupby("canonical_name")["gate"].mean().sort_values()
        )
        for name, value in per_district.items():
            print(f"  {name:<16} {value:.3f}")

    print(f"\nWrote {REPORT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {METRICS_PATH.relative_to(PROJECT_DIR)}")
    if gates is not None:
        print(f"Wrote {GATES_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

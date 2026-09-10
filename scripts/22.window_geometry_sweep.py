"""
Sweep the input-window geometry: lookback L and forecast horizon h.

Every other project result is at L = 12, h = 1. Two architecture changes — the
learnable lag module (item #6) and the spatial/temporal fusion gate (item #8) —
have now failed for the same reason: at h = 1 the anchored residual target is
carried almost entirely by the previous period's case count, so the forecasting
gradient tells the model almost nothing about lag structure or graph structure.
The question here is whether that is a property of the h = 1 task specifically.

    the graph hurts at h = 1: gru_only beats gcn_gru on every headline fold.
    does that gap shrink, vanish or reverse as the forecast reaches further
    out, where last week's count is a worse answer and spatial spillover has
    more to contribute?

So the thing this experiment exists to measure is **the gcn_gru - gru_only gap,
and how it moves with h**. Everything else is scaffolding for reading that
number honestly.

Only the window moves. The graph, the GRU, the head, the anchored target, the
masked loss, the folds, the masks and the per-fold refitted preprocessing are
all the baseline's, imported from `scripts/16` — `train_one`, `build_fold_arrays`,
`predict` and the metric functions, never `scripts/16`'s own `run_experiment` or
`main`. A difference in the numbers cannot come from a difference in the loop.

Two things are handled with care:

    the head stays size 1 at every horizon. `horizon` in this pipeline means
    "how far ahead the target sits", not "how many steps to emit": `make_windows`
    returns a single target at t + h and `predict` does `squeeze(-1)`. Sizing the
    head by h would train h - 1 extra outputs against the same target by
    broadcast and then break `predict`, both silently. The horizon only ever
    reaches `build_fold_arrays` / `make_windows`. This is the fix from
    `scripts/18.build_arms`.

    the persistence baseline is recomputed for each (L, h). The committed
    `naive_baseline_metrics.csv` is an h = 1 file; comparing an h = 3 model
    against an h = 1 persistence number would flatter the model, since
    persistence gets much worse as h grows. `baseline.naive.evaluate_folds` is
    horizon-correct and is called per cell of the sweep.

Outputs:
    results/models/window_report.md
    results/models/window_metrics.csv
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


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "models"

REPORT_PATH = RESULTS_DIR / "window_report.md"
METRICS_PATH = RESULTS_DIR / "window_metrics.csv"

WINDOW_VERSION = "window-geometry-v1"

HEADLINE_FOLDS = (1, 2, 3, 6, 7, 8, 9)


def load_baseline():
    """Load scripts/16 by path and wire the modules it depends on.

    Identical to `scripts/18.load_baseline` and `scripts/21.load_baseline`.
    """

    spec = importlib.util.spec_from_file_location(
        "baseline", PROJECT_DIR / "scripts" / "16.train_gcn_gru.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_modules()

    return module


# ---------------------------------------------------------------------------
# Model builder — the head is size 1 at every horizon
# ---------------------------------------------------------------------------

def make_build_model(baseline, config: dict):
    """Return a `build_model(n_features) -> GCNGRU` for `train_one`'s hook.

    `horizon=1` is hardcoded: the head emits one number regardless of the
    forecast horizon. The horizon reaches the windowing, not the head. See the
    module docstring and `scripts/18.build_arms`.
    """

    def build_model(n_features: int):
        return baseline.GCNGRU(
            n_features=n_features,
            hidden=config["hidden"],
            gcn_layers=config["gcn_layers"],
            horizon=1,
            dropout=config["dropout"],
        )

    return build_model


# ---------------------------------------------------------------------------
# Naive baseline, recomputed per (lookback, horizon)
# ---------------------------------------------------------------------------

def naive_for_geometry(
    baseline, tensors: dict, folds: list[dict], lookback: int, horizon: int
) -> pd.DataFrame:
    """Persistence and seasonal-naive scored at this exact (lookback, horizon).

    `evaluate_folds` returns (metrics, per_district); only the metrics frame is
    needed here. Its rows are keyed by model and fold_id, one set per fold.
    """

    metrics, _ = baseline.naive.evaluate_folds(
        tensors, folds, lookback=lookback, horizon=horizon
    )
    metrics = metrics.copy()
    metrics["lookback"] = lookback
    metrics["horizon"] = horizon

    return metrics


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run(baseline, geometries, folds, adjacencies, arms, config, device):
    """Train every arm on every (lookback, horizon, fold, seed) cell."""

    calendar = baseline.folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()

    build_model = make_build_model(baseline, config)

    model_rows: list[dict] = []
    naive_rows: list[pd.DataFrame] = []

    nodes = pd.read_csv(PROCESSED_DIR / "nodes.csv").sort_values("node_id")

    for variant in config["variants"]:
        tensors = baseline.folds_module.load_tensors(variant)
        # `load_tensors` unpacks the .npz, which carries no district names.
        # `baseline.naive.evaluate_folds` needs `tensors["canonical_name"]` for
        # its per-district breakdown; scripts/15.main attaches it the same way
        # right before its own call. Match that contract rather than touch
        # evaluate_folds.
        tensors["canonical_name"] = nodes["canonical_name"].to_numpy(dtype=object)

        for lookback, horizon in geometries:
            naive_metrics = naive_for_geometry(
                baseline, tensors, folds, lookback, horizon
            )
            naive_metrics["variant"] = variant
            naive_rows.append(naive_metrics)

            for fold in folds:
                arrays = baseline.build_fold_arrays(
                    tensors, months, fold, lookback, horizon
                )

                fit_mask = tensors["period_id"] <= fold["fit_end_period"]
                thresholds = baseline.naive.peak_thresholds(
                    tensors["y"], tensors["y_mask"], fit_mask
                )

                target = arrays["test"]["y"]
                mask = arrays["test"]["mask"].astype(np.int8)
                train_windows = len(arrays["train"]["X"])

                for arm, adjacency in arms.items():
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

                        model_rows.append(
                            {
                                "arm": arm,
                                "variant": variant,
                                "lookback": lookback,
                                "horizon": horizon,
                                "fold_id": fold["fold_id"],
                                "test_year": fold["test_year"],
                                "headline": fold["headline"],
                                "covers_covid": fold["covers_covid"],
                                "seed": seed,
                                "train_windows": train_windows,
                                "best_epoch": info["best_epoch"],
                                **scores,
                            }
                        )
                        seed_mae.append(scores["mae"])

                    print(
                        f"  {variant} L={lookback:<2} h={horizon} "
                        f"fold {fold['fold_id']} ({fold['test_year']}) "
                        f"{arm:<9} MAE {np.mean(seed_mae):7.2f} "
                        f"+/- {np.std(seed_mae):5.2f}  "
                        f"n={train_windows:4d}  "
                        f"{time.perf_counter() - started:5.1f}s",
                        flush=True,
                    )

    return pd.DataFrame(model_rows), pd.concat(naive_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarise(model_metrics: pd.DataFrame, naive_metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per (variant, lookback, horizon, arm-or-naive): headline + 2017 MAE.

    Headline is the mean over the headline folds that are present in the run.
    With a scoped fold list (e.g. just 1 and 8) it is the mean over whichever of
    those are headline folds — stated in the report so a partial headline is
    never mistaken for the real one.
    """

    records = []

    # Model arms.
    per_fold = (
        model_metrics.groupby(
            ["variant", "lookback", "horizon", "arm", "fold_id",
             "test_year", "headline", "covers_covid"]
        )
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), peak_mae=("peak_mae", "mean"))
        .reset_index()
    )

    for (variant, lookback, horizon, arm), group in per_fold.groupby(
        ["variant", "lookback", "horizon", "arm"]
    ):
        headline = group[group["headline"]]
        epidemic = group[group["test_year"] == 2017]
        covid = group[group["covers_covid"]]

        spread = (
            model_metrics[
                (model_metrics["variant"] == variant)
                & (model_metrics["lookback"] == lookback)
                & (model_metrics["horizon"] == horizon)
                & (model_metrics["arm"] == arm)
            ]
            .groupby("fold_id")["mae"]
            .std()
            .mean()
        )

        records.append(
            {
                "variant": variant,
                "lookback": lookback,
                "horizon": horizon,
                "series": arm,
                "kind": "model",
                "headline_mae": headline["mae"].mean(),
                "headline_peak_mae": headline["peak_mae"].mean(),
                "epidemic_2017_mae": epidemic["mae"].mean(),
                "covid_mae": covid["mae"].mean(),
                "seed_sd": spread,
            }
        )

    # Naive baselines, at the matching geometry.
    for (variant, lookback, horizon, model), group in naive_metrics.groupby(
        ["variant", "lookback", "horizon", "model"]
    ):
        headline = group[group["headline"]]
        epidemic = group[group["test_year"] == 2017]
        covid = group[group["covers_covid"]]

        records.append(
            {
                "variant": variant,
                "lookback": lookback,
                "horizon": horizon,
                "series": model,
                "kind": "naive",
                "headline_mae": headline["mae"].mean(),
                "headline_peak_mae": headline["peak_mae"].mean(),
                "epidemic_2017_mae": epidemic["mae"].mean(),
                "covid_mae": covid["mae"].mean(),
                "seed_sd": float("nan"),
            }
        )

    return pd.DataFrame(records)


def gap_table(summary: pd.DataFrame) -> pd.DataFrame:
    """The gcn_gru - gru_only gap, per (variant, lookback, horizon).

    Positive gap means gru_only is better (the graph hurts). This is the number
    the experiment exists to produce; whether it moves with horizon is the
    finding.
    """

    wide = summary[summary["kind"] == "model"].pivot_table(
        index=["variant", "lookback", "horizon"],
        columns="series",
        values="headline_mae",
    )

    rows = []
    for (variant, lookback, horizon), row in wide.iterrows():
        gcn = row.get("gcn_gru", float("nan"))
        gru = row.get("gru_only", float("nan"))
        persistence = summary[
            (summary["variant"] == variant)
            & (summary["lookback"] == lookback)
            & (summary["horizon"] == horizon)
            & (summary["series"] == "persistence")
        ]["headline_mae"]
        persistence = float(persistence.iloc[0]) if len(persistence) else float("nan")

        rows.append(
            {
                "variant": variant,
                "lookback": lookback,
                "horizon": horizon,
                "gcn_gru": gcn,
                "gru_only": gru,
                "gap_gcn_minus_gru": gcn - gru,
                "persistence": persistence,
                "best_model_minus_persistence": min(gcn, gru) - persistence,
            }
        )

    return pd.DataFrame(rows).sort_values(["variant", "lookback", "horizon"])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(summary, gaps, model_metrics, config, folds_used, partial_headline) -> None:
    """Write the window-geometry report, in the shape of lag_report.md."""

    lines = [
        "# Input-window geometry sweep",
        "",
        f"Version: `{WINDOW_VERSION}`",
        "",
        "Lookback `L` and forecast horizon `h`, swept against the frozen",
        "baseline. Only the window moves — the graph, GRU, head, anchored target,",
        "masked loss, folds and per-fold preprocessing are the baseline's,",
        "imported from `scripts/16`. The head is size 1 at every horizon; `h`",
        "reaches the windowing only. Persistence is recomputed at each `(L, h)`.",
        "",
        "The number this exists to produce is **`gcn_gru - gru_only`**, and",
        "whether it moves with `h`. At `h = 1` the graph hurts (`gru_only`",
        "wins); if the gap closes as `h` grows, spatial structure matters more",
        "at longer range and the architecture work is not dead, only mis-tested.",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| `variants` | {config['variants']} |",
        f"| `seeds` | {config['seeds']} |",
        f"| `target` | {config['target']} |",
        f"| `folds` | {sorted(folds_used)} |",
        f"| `hidden` | {config['hidden']} |",
        f"| `gcn_layers` | {config['gcn_layers']} |",
        f"| `dropout` | {config['dropout']} |",
        f"| `learning_rate` | {config['learning_rate']} |",
        f"| `batch_size` | {config['batch_size']} |",
        f"| `max_epochs` | {config['max_epochs']} |",
        f"| `patience` | {config['patience']} |",
    ]

    if partial_headline:
        lines += [
            "",
            "> **Partial headline.** This run does not cover all seven headline",
            f"> folds (1, 2, 3, 6, 7, 8, 9). \"Headline MAE\" below is the mean over",
            f"> {sorted(f for f in folds_used if f in HEADLINE_FOLDS)} only and is not",
            "> comparable to the committed baseline's headline number.",
        ]

    # The gap table, first, because it is the point.
    lines += [
        "",
        "## The graph gap, by geometry",
        "",
        "`gap = gcn_gru - gru_only` headline MAE. Positive means the identity",
        "beats the contiguity graph — the graph is costing accuracy. Persistence",
        "is at the matching horizon.",
        "",
        "| Variant | L | h | gcn_gru | gru_only | gap (gcn − gru) | persistence | best − persistence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in gaps.itertuples():
        lines.append(
            f"| {row.variant} | {row.lookback} | {row.horizon} | "
            f"{row.gcn_gru:.2f} | {row.gru_only:.2f} | "
            f"**{row.gap_gcn_minus_gru:+.2f}** | {row.persistence:.2f} | "
            f"{row.best_model_minus_persistence:+.2f} |"
        )

    # Full headline table, all series.
    lines += [
        "",
        "## Headline MAE, all series",
        "",
        "| Variant | L | h | Series | MAE | Peak MAE | 2017 MAE | COVID MAE | Seed sd |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary.sort_values(
        ["variant", "lookback", "horizon", "kind", "headline_mae"]
    ).itertuples():
        sd = "—" if not np.isfinite(row.seed_sd) else f"{row.seed_sd:.2f}"
        lines.append(
            f"| {row.variant} | {row.lookback} | {row.horizon} | {row.series} | "
            f"{row.headline_mae:.2f} | {row.headline_peak_mae:.2f} | "
            f"{row.epidemic_2017_mae:.2f} | {row.covid_mae:.2f} | {sd} |"
        )

    # Per fold.
    per_fold = (
        model_metrics.groupby(
            ["variant", "lookback", "horizon", "arm", "fold_id", "test_year",
             "covers_covid", "train_windows"]
        )
        .agg(mae=("mae", "mean"), peak_mae=("peak_mae", "mean"))
        .reset_index()
    )
    lines += [
        "",
        "## Per fold, MAE (model arms, mean over seeds)",
        "",
        "| Variant | L | h | Arm | Fold | Year | MAE | Peak MAE | Train windows | Note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in per_fold.sort_values(
        ["variant", "lookback", "horizon", "arm", "fold_id"]
    ).itertuples():
        note = "COVID" if row.covers_covid else ("epidemic" if row.test_year == 2017 else "")
        lines.append(
            f"| {row.variant} | {row.lookback} | {row.horizon} | {row.arm} | "
            f"{row.fold_id} | {row.test_year} | {row.mae:.2f} | {row.peak_mae:.2f} | "
            f"{row.train_windows} | {note} |"
        )

    # Verdict.
    lines += ["", "## Verdict", ""]

    base = gaps[(gaps["lookback"] == 12) & (gaps["horizon"] == 1)]
    if len(base):
        b = base.iloc[0]
        lines.append(
            f"At L = 12, h = 1 the gap is {b.gap_gcn_minus_gru:+.2f} "
            f"(gcn_gru {b.gcn_gru:.2f}, gru_only {b.gru_only:.2f}) — "
            f"{'the graph hurts' if b.gap_gcn_minus_gru > 0 else 'the graph helps'}, "
            "as the baseline sweep found."
        )

    horizons_run = sorted(gaps[gaps["lookback"] == 12]["horizon"].unique())
    if len(horizons_run) > 1:
        trend = gaps[gaps["lookback"] == 12].sort_values("horizon")
        seq = ", ".join(
            f"h={int(r.horizon)}: {r.gap_gcn_minus_gru:+.2f}" for r in trend.itertuples()
        )
        lines += [
            "",
            f"Across horizons at L = 12, the gap moves: {seq}.",
            "",
            "Read this against the seed sd column — a change in the gap smaller",
            "than the pooled seed sd is not established by this run. If the gap",
            "narrows monotonically toward zero as h grows, the graph earns its",
            "place at longer range and items #7 and #8 should be re-run at the",
            "horizon where it does. If the gap is flat, the h = 1 diagnosis",
            "generalises and the ceiling is the objective, not the architecture.",
        ]

    lookbacks_run = sorted(gaps["lookback"].unique())
    if len(lookbacks_run) > 1:
        l12 = gaps[(gaps["lookback"] == 12) & (gaps["horizon"] == 1)]
        l26 = gaps[(gaps["lookback"] == 26) & (gaps["horizon"] == 1)]
        if len(l12) and len(l26):
            a, b = l12.iloc[0], l26.iloc[0]
            lines += [
                "",
                f"Doubling the lookback (12 → 26) at h = 1 moves the best model "
                f"from {a.best_model_minus_persistence:+.2f} to "
                f"{b.best_model_minus_persistence:+.2f} against persistence, and "
                f"the graph gap from {a.gap_gcn_minus_gru:+.2f} to "
                f"{b.gap_gcn_minus_gru:+.2f}. A longer lookback is the cheapest "
                "test of whether lag structure is being missed; if it does not "
                "move the number, more history is not the constraint.",
            ]

    lines += [
        "",
        "## Output files",
        "",
        f"- `{METRICS_PATH.relative_to(PROJECT_DIR).as_posix()}`",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_geometries(lookbacks: list[int], horizons: list[int], pairs: list[str] | None):
    """Return the list of (lookback, horizon) cells to run.

    Default is the cross product of `--lookbacks` and `--horizons`. `--pairs`
    overrides it with an explicit list like `12:1 12:2 26:1`, for a scoped run
    that is not a full grid.
    """

    if pairs:
        out = []
        for token in pairs:
            left, right = token.split(":")
            out.append((int(left), int(right)))
        return out

    return [(lb, h) for lb in lookbacks for h in horizons]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--lookbacks", nargs="+", type=int, default=[12, 26])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="explicit L:h cells, e.g. 12:1 12:2 26:1 — overrides the cross product",
    )
    parser.add_argument("--variants", nargs="+", default=["v1"])
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["gcn_gru", "gru_only"],
        choices=["gcn_gru", "gru_only"],
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

    folds = json.loads(baseline.FOLDS_PATH.read_text(encoding="utf-8"))["folds"]
    if arguments.folds:
        folds = [fold for fold in folds if fold["fold_id"] in arguments.folds]

    folds_used = [fold["fold_id"] for fold in folds]
    headline_present = [f for f in folds_used if f in HEADLINE_FOLDS]
    partial_headline = sorted(headline_present) != sorted(HEADLINE_FOLDS)

    with np.load(baseline.ADJACENCY_PATH, allow_pickle=True) as data:
        real_adjacency = data["A_norm"].astype(np.float32)

    n_nodes = len(real_adjacency)
    all_arms = {
        "gcn_gru": real_adjacency,
        "gru_only": np.eye(n_nodes, dtype=np.float32),
    }
    arms = {name: all_arms[name] for name in arguments.arms}

    geometries = parse_geometries(
        arguments.lookbacks, arguments.horizons, arguments.pairs
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device:      {device}")
    print(f"Geometries:  {geometries}   (lookback, horizon)")
    print(f"Variants:    {config['variants']}")
    print(f"Folds:       {folds_used}")
    print(f"Arms:        {list(arms)}")
    print(f"Seeds:       {config['seeds']}")
    if partial_headline:
        print(f"NOTE: partial headline — only folds {sorted(headline_present)} of the seven")
    print()

    started = time.perf_counter()
    model_metrics, naive_metrics = run(
        baseline, geometries, folds, all_arms, arms, config, device
    )
    elapsed = time.perf_counter() - started

    summary = summarise(model_metrics, naive_metrics)
    gaps = gap_table(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_metrics.to_csv(METRICS_PATH, index=False)
    naive_metrics.to_csv(
        RESULTS_DIR / "window_naive_metrics.csv", index=False
    )
    write_report(summary, gaps, model_metrics, config, folds_used, partial_headline)

    print(f"\nSwept in {elapsed / 60:.1f} min\n")
    print("gap = gcn_gru - gru_only headline MAE  (positive = graph hurts)")
    print(
        f"{'var':<4} {'L':>3} {'h':>2} {'gcn_gru':>9} {'gru_only':>9} "
        f"{'gap':>8} {'persist':>9} {'best-persist':>13}"
    )
    for row in gaps.itertuples():
        print(
            f"{row.variant:<4} {row.lookback:>3} {row.horizon:>2} "
            f"{row.gcn_gru:>9.2f} {row.gru_only:>9.2f} "
            f"{row.gap_gcn_minus_gru:>+8.2f} {row.persistence:>9.2f} "
            f"{row.best_model_minus_persistence:>+13.2f}"
        )

    print(f"\nWrote {REPORT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {METRICS_PATH.relative_to(PROJECT_DIR)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

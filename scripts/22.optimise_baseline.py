"""
Find how much graph the GCN+GRU actually wants, and what each knob is worth.

The question this script started from was whether `gcn_gru` was being stopped
prematurely: at the committed settings it is selected at roughly a quarter of
the optimiser steps `gru_only` gets. Running both to 200 epochs with early
stopping disabled answered it, and the answer was no.

    fold 8  gcn_gru   best validation loss 0.5148 at epoch 11, of 200
    fold 9  gcn_gru   best validation loss 0.2525 at epoch 20, of 200

The stop is correct. Training longer makes it monotonically worse. But the same
curves showed something the MAE table did not:

    epoch 11    gcn_gru train 0.4949    gru_only train 0.3947
    epoch 200   gcn_gru train 0.4376    gru_only train 0.3592

`gru_only`'s *training* loss at epoch 11 beats `gcn_gru`'s at epoch 200. The two
arms have identical parameter counts and differ only in whether A is the
contiguity matrix or the identity, so this is not capacity and not
regularisation: the propagation is destroying district-level signal, and more
training does not recover it. That is over-smoothing.

So the experiment moves to the propagation itself.

    graph       A(alpha) = (1 - alpha) I + alpha A_norm, swept from 0 to 1.
                alpha = 0 is exactly `gru_only`, alpha = 1 is exactly
                `gcn_gru`, and the sweep asks where in between the error is
                lowest. This is the per-district gate of `scripts/21` reduced
                to one country-wide scalar that can be swept rather than
                learned, which is what makes it readable as a dose-response
                curve. A_gaussian_norm is swept the same way, because
                scripts/13 builds it and nothing has ever consumed it.

    schedule    `committed` against `patience40`, kept only to document the
                negative result above rather than to find a win.

    sensitivity one factor at a time: graph depth and dropout, which
                over-smoothing predicts should move the error, plus learning
                rate and hidden width as controls that it predicts should not.

Nothing about the architecture, the target, the loss, the folds, the
preprocessing or the metrics moves. The training loop, model, fold builder and
scoring are **imported** from `scripts/16.train_gcn_gru.py`, so a difference in
the numbers cannot come from a difference in the loop.

Outputs:
    results/models/optimisation_metrics.csv
    results/models/optimisation_curves.csv
    results/figures/opt1_training_curves.png
    results/figures/opt2_stopping_point.png
    results/figures/opt3_parameter_sensitivity.png
    results/figures/opt4_schedule_effect.png

The figures are generated automatically at the end of this script by the
plotting functions in `scripts/23.optimisation_figures.py`. Run that script
alone to redraw figures from existing artifacts, or pass `--skip-figures` when
running a partial optimization stage.
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

RESULTS_DIR = PROJECT_DIR / "results" / "models"
FIGURE_DIR = PROJECT_DIR / "results" / "figures"

METRICS_PATH = RESULTS_DIR / "optimisation_metrics.csv"
CURVES_PATH = RESULTS_DIR / "optimisation_curves.csv"
REPORT_PATH = RESULTS_DIR / "optimisation_report.md"

OPTIMISATION_VERSION = "optimise-v1"

# The folds this runs on by default. 8 and 9 are the two longest histories and
# the ones the early-stopping asymmetry was first measured on.
#
# Fold 1 is deliberately NOT here. It is the 2017 epidemic, where persistence
# scores 36.08 MAE against roughly 16 elsewhere, so it dominates any mean it is
# averaged into and it roughly doubles the run. Conclusions drawn here are
# about ordinary years; run `--folds 1 8 9` before claiming anything about
# epidemic years.
DEFAULT_FOLDS = (8, 9)

# Sweeps run by default. The other three knobs in SENSITIVITY are defined and
# runnable via `--sweep`, but they are not in the default set: on a CPU each
# knob is roughly 20 minutes, and these three are where the early-stopping
# diagnosis says the error should actually move.
DEFAULT_SWEEPS = ("gcn_layers", "dropout", "learning_rate")

# Schedule arms. Each is a patch applied over scripts/16's DEFAULTS, so an arm
# is exactly the committed configuration plus the keys named here.
SCHEDULE_ARMS: dict[str, dict] = {
    # The control. Must reproduce scripts/16 exactly.
    "committed": {},
    # The measured negative result, kept so the report carries it rather than
    # asserting it: with patience 40 the selected epoch does not move, because
    # the validation minimum is genuinely at epoch 11-20 and everything after
    # it is worse. This arm exists to be seen failing to help.
    "patience40": {"patience": 40},
}

# Blend between the identity and the real adjacency. alpha = 0 reproduces
# `gru_only` exactly and alpha = 1 reproduces `gcn_gru` exactly, so the two
# committed arms are the endpoints of this sweep rather than separate models.
GRAPH_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)

# One-factor-at-a-time sweeps, applied over the winning schedule arm. Values
# are chosen to bracket the baseline setting on both sides -- a sweep that only
# goes one way cannot show a minimum.
SENSITIVITY: dict[str, tuple] = {
    # Over-smoothing predicts these two move the error.
    "gcn_layers": (1, 2, 3, 4),
    "dropout": (0.0, 0.2, 0.4, 0.6),
    # ... and that these two do not. They are the controls: a hypothesis that
    # only predicts what it wants to find is not being tested.
    "learning_rate": (3e-4, 1e-3, 3e-3, 1e-2),
    "hidden": (8, 16, 32, 64),
    "batch_size": (16, 32, 64, 128),
    "patience": (5, 15, 40, 80),
}


baseline = None  # scripts/16, holds the loop, model, fold builder and metrics


def load_baseline():
    """Load scripts/16 and wire up everything it depends on."""

    spec = importlib.util.spec_from_file_location(
        "baseline", PROJECT_DIR / "scripts" / "16.train_gcn_gru.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_modules()

    return module


def load_figure_generator():
    """Load the plotting module without importing a non-identifier filename."""

    spec = importlib.util.spec_from_file_location(
        "optimisation_figures", PROJECT_DIR / "scripts" / "23.optimisation_figures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ---------------------------------------------------------------------------
# Running one configuration
# ---------------------------------------------------------------------------

def blended_adjacency(adjacency: np.ndarray, alpha: float) -> np.ndarray:
    """Return (1 - alpha) I + alpha A.

    A one-parameter family joining the two committed arms. At alpha = 0 every
    node sees only itself and the graph convolution is a per-district MLP,
    which is exactly `gru_only`; at alpha = 1 it is exactly `gcn_gru`. The
    blend is taken on the already-normalised matrix rather than re-normalising
    afterwards, so the endpoints reproduce the committed arms bit for bit
    instead of approximately.
    """

    identity = np.eye(len(adjacency), dtype=np.float64)
    blended = (1.0 - alpha) * identity + alpha * adjacency.astype(np.float64)

    return blended.astype(np.float32)


def prepare_folds(variant: str, fold_ids: tuple[int, ...], config: dict):
    """Build the per-fold arrays and peak thresholds once, and reuse them.

    Imputation and scaling are refitted per fold inside `build_fold_arrays`,
    which is the expensive part and does not depend on any training knob. Doing
    it once per (variant, batch geometry) rather than once per arm is the only
    reason a sweep this size finishes on a CPU.
    """

    all_folds = json.loads(baseline.FOLDS_PATH.read_text(encoding="utf-8"))["folds"]
    folds = [f for f in all_folds if f["fold_id"] in fold_ids]

    calendar = baseline.folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()
    tensors = baseline.folds_module.load_tensors(variant)

    prepared = {}
    for fold in folds:
        arrays = baseline.build_fold_arrays(
            tensors, months, fold, config["lookback"], config["horizon"]
        )
        fit_mask = tensors["period_id"] <= fold["fit_end_period"]
        thresholds = baseline.naive.peak_thresholds(
            tensors["y"], tensors["y_mask"], fit_mask
        )
        prepared[fold["fold_id"]] = {
            "fold": fold,
            "arrays": arrays,
            "thresholds": thresholds,
        }

    return prepared


def run_configuration(
    label: str,
    kind: str,
    config: dict,
    prepared: dict,
    models: dict[str, np.ndarray],
    seeds: int,
    device: torch.device,
    keep_curves: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Train one configuration on every fold, model and seed."""

    rows: list[dict] = []
    curves: list[dict] = []

    for fold_id, bundle in prepared.items():
        arrays = bundle["arrays"]
        target = arrays["test"]["y"]
        mask = arrays["test"]["mask"].astype(np.int8)

        for model_name, adjacency in models.items():
            started = time.perf_counter()

            for seed in range(seeds):
                model, info = baseline.train_one(
                    arrays, adjacency, config, seed, device
                )
                prediction = baseline.predict(
                    model, arrays["test"], adjacency, config["target"], device
                )
                scores = baseline.naive.evaluate(
                    prediction, target, mask, bundle["thresholds"]
                )

                # Optimiser steps behind the *selected* model, which is what is
                # actually scored -- not the steps the run happened to take.
                steps_per_epoch = int(
                    np.ceil(len(arrays["train"]["X"]) / config["batch_size"])
                )

                rows.append(
                    {
                        "kind": kind,
                        "label": label,
                        "model": model_name,
                        "fold_id": fold_id,
                        "test_year": bundle["fold"]["test_year"],
                        "seed": seed,
                        "best_epoch": info["best_epoch"],
                        "epochs_run": info["epochs_run"],
                        "steps_per_epoch": steps_per_epoch,
                        "steps_to_best": info["best_epoch"] * steps_per_epoch,
                        "val_loss": info["val_loss"],
                        **{k: config[k] for k in SENSITIVITY},
                        **scores,
                    }
                )

                if keep_curves and seed == 0:
                    history = info["history"]
                    for epoch, train_loss, val_loss in zip(
                        history["epoch"], history["train_loss"], history["val_loss"]
                    ):
                        curves.append(
                            {
                                "label": label,
                                "model": model_name,
                                "fold_id": fold_id,
                                "epoch": epoch,
                                "train_loss": train_loss,
                                "val_loss": val_loss,
                                "best_epoch": info["best_epoch"],
                            }
                        )

            recent = [r["mae"] for r in rows[-seeds:]]
            print(
                f"  {kind:<11} {label:<14} fold {fold_id} {model_name:<9} "
                f"MAE {np.mean(recent):7.2f} +/- {np.std(recent):5.2f}  "
                f"steps {rows[-1]['steps_to_best']:>5}  "
                f"{time.perf_counter() - started:5.1f}s",
                flush=True,
            )

    return rows, curves


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--variant", default="v1")
    parser.add_argument(
        "--folds", nargs="+", type=int, default=list(DEFAULT_FOLDS)
    )
    parser.add_argument("--seeds", type=int, default=2)
    # The sweeps run at one seed by default. They are looking for the *shape*
    # of the curve -- which knobs the error is flat in -- and the seed band is
    # drawn from the schedule arms, which do carry repeats.
    parser.add_argument("--sensitivity-seeds", type=int, default=1)
    parser.add_argument(
        "--stage",
        nargs="+",
        default=["graph", "schedule", "sensitivity"],
        choices=["graph", "schedule", "sensitivity"],
    )
    parser.add_argument(
        "--arms", nargs="+", default=None, help="subset of the schedule arms"
    )
    parser.add_argument(
        "--sweep", nargs="+", default=None, help="subset of the sensitivity knobs"
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="write optimization artifacts without generating figures",
    )

    return parser.parse_args()


def main() -> int:
    global baseline

    arguments = parse_arguments()
    baseline = load_baseline()

    if not baseline.FOLDS_PATH.exists() or not baseline.ADJACENCY_PATH.exists():
        print("Missing folds.json or adjacency.npz. Run scripts 13 and 14 first.")
        return 1

    with np.load(baseline.ADJACENCY_PATH, allow_pickle=True) as data:
        adjacency = data["A_norm"].astype(np.float32)
        # Built by scripts/13 and, until now, consumed by nothing.
        gaussian = (
            data["A_gaussian_norm"].astype(np.float32)
            if "A_gaussian_norm" in data
            else None
        )

    models = {
        "gcn_gru": adjacency,
        "gru_only": np.eye(len(adjacency), dtype=np.float32),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_ids = tuple(arguments.folds)

    base_config = dict(baseline.DEFAULTS)
    base_config["target"] = "residual"

    print(f"Device:  {device}")
    print(f"Folds:   {list(fold_ids)}")
    print(f"Variant: {arguments.variant}")
    print(f"Seeds:   {arguments.seeds}\n")

    prepared = prepare_folds(arguments.variant, fold_ids, base_config)

    all_rows: list[dict] = []
    all_curves: list[dict] = []

    started = time.perf_counter()

    if "graph" in arguments.stage:
        print("Graph-strength sweep: A(alpha) = (1 - alpha) I + alpha A")
        for matrix_name, matrix in (
            ("contiguity", adjacency),
            ("gaussian", gaussian),
        ):
            if matrix is None:
                continue
            for alpha in GRAPH_ALPHAS:
                # alpha = 0 is the same matrix for both, so run it once.
                if alpha == 0.0 and matrix_name == "gaussian":
                    continue
                config = dict(base_config)
                rows, _ = run_configuration(
                    f"alpha={alpha:.2f}",
                    f"graph:{matrix_name}",
                    config,
                    prepared,
                    {f"{matrix_name}": blended_adjacency(matrix, alpha)},
                    arguments.seeds,
                    device,
                )
                for row in rows:
                    row["alpha"] = alpha
                    row["matrix"] = matrix_name
                all_rows += rows

    if "schedule" in arguments.stage:
        arms = arguments.arms or list(SCHEDULE_ARMS)
        print("Schedule arms")
        for label in arms:
            config = dict(base_config)
            config.update(SCHEDULE_ARMS[label])
            rows, curves = run_configuration(
                label, "schedule", config, prepared, models,
                arguments.seeds, device, keep_curves=True,
            )
            all_rows += rows
            all_curves += curves

    if "sensitivity" in arguments.stage:
        winner = pick_winner(pd.DataFrame(all_rows)) if all_rows else "cosine"
        print(f"\nSensitivity sweeps, on top of the '{winner}' schedule")

        knobs = arguments.sweep or list(DEFAULT_SWEEPS)
        for knob in knobs:
            for value in SENSITIVITY[knob]:
                config = dict(base_config)
                config.update(SCHEDULE_ARMS[winner])
                config[knob] = value
                rows, _ = run_configuration(
                    f"{knob}={value}", f"sweep:{knob}", config, prepared,
                    models, arguments.sensitivity_seeds, device,
                )
                all_rows += rows

    elapsed = time.perf_counter() - started

    metrics = pd.DataFrame(all_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)

    if all_curves:
        pd.DataFrame(all_curves).to_csv(CURVES_PATH, index=False)

    print(f"\nRan in {elapsed / 60:.1f} min")
    print(f"Wrote {METRICS_PATH.relative_to(PROJECT_DIR)}")
    if all_curves:
        print(f"Wrote {CURVES_PATH.relative_to(PROJECT_DIR)}")

    if arguments.skip_figures:
        print("Skipped optimization figures (--skip-figures).")
    elif not all_curves:
        print("Skipped optimization figures: schedule curves were not generated.")
    else:
        figure_generator = load_figure_generator()
        figure_generator.main()

    return 0


def pick_winner(metrics: pd.DataFrame) -> str:
    """Return the schedule arm with the lowest mean MAE for gcn_gru."""

    schedule = metrics[
        (metrics["kind"] == "schedule") & (metrics["model"] == "gcn_gru")
    ]
    if schedule.empty:
        return "cosine"

    return schedule.groupby("label")["mae"].mean().idxmin()


if __name__ == "__main__":
    sys.exit(main())

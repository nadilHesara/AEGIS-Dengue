"""
Hyperparameter search for the full GCN+GRU model, selected on validation folds.

The baseline in `scripts/16.train_gcn_gru.py` fixes every training knob by hand
(`DEFAULTS`). Those values were chosen once, on fold 8, before the full sweep
existed. This script asks whether a systematic search over the joint space finds
a configuration the hand-picked one missed -- and answers it honestly, which
here means: **selection touches the validation split only, and the winning
configuration is then run once through the frozen test protocol for a verdict.**

Search space (9 axes, 4x3x2x3x2x4x4x3x4 = 27 648 points):

    lookback         input window, periods       [8, 12, 16, 24]
    gcn_hidden       graph-conv width            [16, 32, 64]
    gcn_layers       graph-conv depth            [1, 2]
    gru_hidden       GRU hidden width           [32, 64, 128]
    gru_layers       GRU depth                  [1, 2]
    dropout                                     [0.0, 0.1, 0.2, 0.3]
    learning_rate                               [1e-4, 3e-4, 1e-3, 3e-3]
    batch_size                                  [16, 32, 64]
    weight_decay                                [0.0, 1e-5, 1e-4, 1e-3]

The baseline's `GCNGRU` ties the GRU width to the graph-conv width and hardcodes
a single GRU layer. `gru_hidden` and `gru_layers` therefore need a model the
baseline does not provide, so this script defines `TunableGCNGRU`: the same
spatial-then-temporal arrangement, with a linear projection inserted when
`gcn_hidden != gru_hidden` and `num_layers` passed through to `nn.GRU`. When the
search lands on `gcn_hidden == gru_hidden` and `gru_layers == 1` it is
byte-for-byte the baseline architecture; a test asserts the parameter count
matches in that case.

Everything else -- the training loop, early stopping, optimiser, gradient
clipping, seeding, fold construction, preprocessing, windowing, the anchored
residual target, and the masked metrics -- is **imported** from
`scripts/16.train_gcn_gru.py`, unchanged. A search that reimplemented any of them
would be tuning a different model than the one it reports a number for.

Method:

    --method random   (default) uniform random search with a seeded RNG, so the
                      exact trial list is reproducible from `--search-seed` and
                      `--n-trials` alone. No third-party dependency.
    --method optuna   TPE sampler, if `optuna` is installed. Same objective, same
                      search space, same validation-only selection. Falls back to
                      random with a printed warning if the import fails.

Objective (minimised): mean masked MAE on the **validation** split of each
tuning fold, on the raw case scale, seeds averaged. `--tune-folds` picks which
folds contribute; the default is the seven headline folds. The test split of
every fold is never read during search -- `evaluate_split` is pointed at "val".

After the search the best configuration is retrained with `--final-seeds` seeds
on all nine folds and scored on the **test** split by the same
`scripts/15.evaluate_naive_baselines.evaluate` the baseline uses, so the
tuned number and the baseline number are directly comparable.

Usage:

    python scripts/24.tune_hyperparameters.py                       80 random trials
    python scripts/24.tune_hyperparameters.py --n-trials 200
    python scripts/24.tune_hyperparameters.py --method optuna --n-trials 150
    python scripts/24.tune_hyperparameters.py --tune-folds 8 --n-trials 20 --trial-seeds 1   smoke

Outputs:
    results/models/tuning_report.md
    results/models/tuning_trials.csv
    results/models/tuning_best_config.json
    results/models/tuning_test_metrics.csv
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

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "models"

FOLDS_PATH = PROCESSED_DIR / "folds.json"
ADJACENCY_PATH = PROCESSED_DIR / "adjacency.npz"

REPORT_PATH = RESULTS_DIR / "tuning_report.md"
TRIALS_PATH = RESULTS_DIR / "tuning_trials.csv"
BEST_CONFIG_PATH = RESULTS_DIR / "tuning_best_config.json"
TEST_METRICS_PATH = RESULTS_DIR / "tuning_test_metrics.csv"

TUNING_VERSION = "tuning-v1"

# The search space, exactly as commissioned. Order here is the order the report
# and the CSV columns use.
SEARCH_SPACE: dict[str, list] = {
    "lookback": [8, 12, 16, 24],
    "gcn_hidden": [16, 32, 64],
    "gcn_layers": [1, 2],
    "gru_hidden": [32, 64, 128],
    "gru_layers": [1, 2],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3],
    "batch_size": [16, 32, 64],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
}

# Knobs that are held at the baseline's value throughout the search. horizon and
# target are part of the problem definition, not things being tuned here;
# max_epochs and patience govern the budget and are deliberately shared so every
# trial gets the same one.
FIXED = {
    "horizon": 1,
    "target": "residual",
    "max_epochs": 150,
    "patience": 15,
}

# Headline folds -- the ones the README averages. COVID folds (4, 5) are left out
# of selection for the same reason they are left out of the headline.
HEADLINE_FOLDS = [1, 2, 3, 6, 7, 8, 9]


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

    baseline_module = _load("baseline_module", "16.train_gcn_gru.py")
    baseline_module.load_modules()

    naive = baseline_module.naive
    folds_module = baseline_module.folds_module


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TunableGCNGRU(nn.Module):
    """The baseline GCN+GRU with GRU width and depth decoupled from the graph conv.

    Identical in arrangement to `scripts/16.train_gcn_gru.GCNGRU` -- graph
    convolution over the 25 districts at every step, then one shared GRU over the
    window, then a linear head. Two knobs the baseline hardcodes are exposed:

        gru_hidden   the GRU's hidden size. When it differs from the graph
                     conv's output width a linear projection bridges them; when
                     it matches, no projection is added and the module reduces to
                     the baseline exactly (checked by test).
        gru_layers   passed straight to `nn.GRU(num_layers=...)`. Inter-layer
                     dropout is applied by `nn.GRU` itself only when
                     `gru_layers > 1`, matching torch's own convention.

    The graph conv reuses the baseline's `GraphConv` so the spatial half is
    literally the same code.
    """

    def __init__(
        self,
        n_features: int,
        gcn_hidden: int,
        gcn_layers: int,
        gru_hidden: int,
        gru_layers: int,
        dropout: float,
        horizon: int,
    ):
        super().__init__()

        graph_conv = baseline_module.GraphConv

        sizes = [n_features] + [gcn_hidden] * gcn_layers
        self.graph_layers = nn.ModuleList(
            graph_conv(sizes[i], sizes[i + 1]) for i in range(gcn_layers)
        )

        self.dropout = nn.Dropout(dropout)

        # Bridge the graph-conv width to the GRU width only when they differ.
        # Identity otherwise, so the byte-for-byte-baseline case adds no params.
        if gcn_hidden != gru_hidden:
            self.project = nn.Linear(gcn_hidden, gru_hidden)
        else:
            self.project = nn.Identity()

        self.gru = nn.GRU(
            gru_hidden,
            gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Linear(gru_hidden, horizon)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, _ = x.shape

        spatial = x
        for layer in self.graph_layers:
            spatial = torch.relu(layer(spatial, adjacency))
            spatial = self.dropout(spatial)

        spatial = self.project(spatial)

        sequences = spatial.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)

        output, _ = self.gru(sequences)
        last = self.dropout(output[:, -1])

        return self.head(last).view(batch, nodes, -1)


def make_model_builder(config: dict):
    """Return a `build_model(n_features) -> nn.Module` closure for `train_one`.

    `scripts/16.train_gcn_gru.train_one` already accepts a `build_model` hook so a
    variant can reuse its loop verbatim. This is that hook.
    """

    def build_model(n_features: int) -> nn.Module:
        return TunableGCNGRU(
            n_features=n_features,
            gcn_hidden=config["gcn_hidden"],
            gcn_layers=config["gcn_layers"],
            gru_hidden=config["gru_hidden"],
            gru_layers=config["gru_layers"],
            dropout=config["dropout"],
            horizon=config["horizon"],
        )

    return build_model


# ---------------------------------------------------------------------------
# Scoring one configuration
# ---------------------------------------------------------------------------

def evaluate_split(
    model: nn.Module,
    split: dict[str, np.ndarray],
    adjacency: np.ndarray,
    mode: str,
    thresholds: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    """Score one split with the baseline's own predict + evaluate.

    Pointed at the validation split during search and at the test split for the
    final verdict. The prediction path and the metric are `scripts/16`'s and
    `scripts/15`'s respectively, so a tuned score and a baseline score mean the
    same thing.
    """

    prediction = baseline_module.predict(model, split, adjacency, mode, device)
    target = split["y"]
    mask = split["mask"].astype(np.int8)

    return naive.evaluate(prediction, target, mask, thresholds)


def score_config(
    config: dict,
    fold_arrays: dict[int, dict],
    fold_thresholds: dict[int, np.ndarray],
    adjacency: np.ndarray,
    tune_folds: list[int],
    seeds: int,
    split: str,
    device: torch.device,
) -> dict:
    """Train `config` on every tuning fold and return its mean MAE on `split`.

    `split` is "val" during the search and "test" only for the final rerun. Seeds
    are averaged. The per-fold MAEs are returned too, so the report can show
    where a configuration's score comes from rather than only the mean.
    """

    build_model = make_model_builder(config)

    per_fold_mae: dict[int, float] = {}
    per_fold_peak: dict[int, float] = {}

    for fold_id in tune_folds:
        arrays = fold_arrays[fold_id]
        thresholds = fold_thresholds[fold_id]

        seed_scores = []
        for seed in range(seeds):
            model, _ = baseline_module.train_one(
                arrays, adjacency, config, seed, device, build_model=build_model
            )
            seed_scores.append(
                evaluate_split(
                    model, arrays[split], adjacency, config["target"], thresholds, device
                )
            )

        per_fold_mae[fold_id] = float(np.mean([s["mae"] for s in seed_scores]))
        per_fold_peak[fold_id] = float(np.mean([s["peak_mae"] for s in seed_scores]))

    return {
        "mean_mae": float(np.mean(list(per_fold_mae.values()))),
        "mean_peak_mae": float(np.nanmean(list(per_fold_peak.values()))),
        "per_fold_mae": per_fold_mae,
        "per_fold_peak_mae": per_fold_peak,
    }


# ---------------------------------------------------------------------------
# Fold data, built once
# ---------------------------------------------------------------------------

def build_all_fold_arrays(
    variant: str,
    folds: list[dict],
    lookback: int,
    horizon: int,
) -> tuple[dict[int, dict], dict[int, np.ndarray]]:
    """Window every fold at one lookback and return arrays + peak thresholds.

    Lookback changes the windowing, so this is called once per distinct lookback
    the search visits and the result cached by the caller. Everything downstream
    of windowing is lookback-independent.
    """

    calendar = folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()

    tensors = folds_module.load_tensors(variant)

    fold_arrays: dict[int, dict] = {}
    fold_thresholds: dict[int, np.ndarray] = {}

    for fold in folds:
        fold_arrays[fold["fold_id"]] = baseline_module.build_fold_arrays(
            tensors, months, fold, lookback, horizon
        )
        fit_mask = tensors["period_id"] <= fold["fit_end_period"]
        fold_thresholds[fold["fold_id"]] = naive.peak_thresholds(
            tensors["y"], tensors["y_mask"], fit_mask
        )

    return fold_arrays, fold_thresholds


class FoldCache:
    """Lazily window each fold once per distinct lookback the search asks for."""

    def __init__(self, variant: str, folds: list[dict], horizon: int):
        self.variant = variant
        self.folds = folds
        self.horizon = horizon
        self._by_lookback: dict[int, tuple[dict, dict]] = {}

    def get(self, lookback: int) -> tuple[dict[int, dict], dict[int, np.ndarray]]:
        if lookback not in self._by_lookback:
            self._by_lookback[lookback] = build_all_fold_arrays(
                self.variant, self.folds, lookback, self.horizon
            )
        return self._by_lookback[lookback]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def sample_random(rng, space: dict[str, list]) -> dict:
    """One uniform draw from the discrete grid."""

    return {key: rng.choice(values) for key, values in space.items()}


def full_config(sampled: dict) -> dict:
    """Merge a sampled point with the fixed knobs into a complete train config."""

    config = dict(FIXED)
    config.update(sampled)
    # train_one reads seeds off the config in the baseline; the search drives the
    # seed loop itself, so this is only here for shape-compatibility.
    config.setdefault("seeds", 1)
    return config


def run_random_search(
    n_trials: int,
    search_seed: int,
    cache: FoldCache,
    adjacency: np.ndarray,
    tune_folds: list[int],
    trial_seeds: int,
    device: torch.device,
) -> list[dict]:
    """Uniform random search. Reproducible from `search_seed` and `n_trials`."""

    import random

    rng = random.Random(search_seed)
    trials: list[dict] = []
    seen: set[tuple] = set()

    trial_index = 0
    attempts = 0
    max_attempts = n_trials * 50

    while trial_index < n_trials and attempts < max_attempts:
        attempts += 1
        sampled = sample_random(rng, SEARCH_SPACE)
        key = tuple(sampled[k] for k in SEARCH_SPACE)
        if key in seen:
            continue
        seen.add(key)

        config = full_config(sampled)
        fold_arrays, fold_thresholds = cache.get(config["lookback"])

        started = time.perf_counter()
        result = score_config(
            config, fold_arrays, fold_thresholds, adjacency,
            tune_folds, trial_seeds, "val", device,
        )
        elapsed = time.perf_counter() - started

        row = {"trial": trial_index, **sampled, **_flat_result(result), "seconds": elapsed}
        trials.append(row)
        trial_index += 1

        print(
            f"  trial {trial_index:3d}/{n_trials}  val MAE {result['mean_mae']:7.3f}  "
            f"L{sampled['lookback']:<2} gcn{sampled['gcn_hidden']}x{sampled['gcn_layers']} "
            f"gru{sampled['gru_hidden']}x{sampled['gru_layers']} "
            f"do{sampled['dropout']} lr{sampled['learning_rate']:.0e} "
            f"bs{sampled['batch_size']} wd{sampled['weight_decay']:.0e}  {elapsed:5.1f}s",
            flush=True,
        )

    return trials


def run_optuna_search(
    n_trials: int,
    search_seed: int,
    cache: FoldCache,
    adjacency: np.ndarray,
    tune_folds: list[int],
    trial_seeds: int,
    device: torch.device,
) -> list[dict]:
    """TPE search over the same grid, selecting on validation MAE.

    Only reached when `import optuna` succeeds; `main` falls back to random
    otherwise. The search space is `suggest_categorical` over the identical
    lists, so a TPE run and a random run are exploring the same 27 648 points.
    """

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    trials: list[dict] = []

    def objective(trial: "optuna.Trial") -> float:
        sampled = {
            key: trial.suggest_categorical(key, values)
            for key, values in SEARCH_SPACE.items()
        }
        config = full_config(sampled)
        fold_arrays, fold_thresholds = cache.get(config["lookback"])

        started = time.perf_counter()
        result = score_config(
            config, fold_arrays, fold_thresholds, adjacency,
            tune_folds, trial_seeds, "val", device,
        )
        elapsed = time.perf_counter() - started

        trials.append(
            {
                "trial": trial.number,
                **sampled,
                **_flat_result(result),
                "seconds": elapsed,
            }
        )
        print(
            f"  trial {trial.number + 1:3d}/{n_trials}  val MAE {result['mean_mae']:7.3f}  "
            f"L{sampled['lookback']:<2} gcn{sampled['gcn_hidden']}x{sampled['gcn_layers']} "
            f"gru{sampled['gru_hidden']}x{sampled['gru_layers']}  {elapsed:5.1f}s",
            flush=True,
        )
        return result["mean_mae"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=search_seed),
    )
    study.optimize(objective, n_trials=n_trials)

    return trials


def _flat_result(result: dict) -> dict:
    """Flatten score_config's nested per-fold dicts into CSV-friendly columns."""

    flat = {"val_mae": result["mean_mae"], "val_peak_mae": result["mean_peak_mae"]}
    for fold_id, mae in result["per_fold_mae"].items():
        flat[f"val_mae_fold{fold_id}"] = mae
    return flat


# ---------------------------------------------------------------------------
# Final test-set verdict
# ---------------------------------------------------------------------------

def score_on_test(
    config: dict,
    cache: FoldCache,
    adjacency: np.ndarray,
    folds: list[dict],
    final_seeds: int,
    device: torch.device,
) -> pd.DataFrame:
    """Retrain the winning config on all nine folds and score the test split.

    This is the only place the test split is read. Same seeds, same metric, same
    predict path as the baseline, so the row this produces drops straight into
    the same comparison table.
    """

    fold_arrays, fold_thresholds = cache.get(config["lookback"])
    build_model = make_model_builder(config)

    rows: list[dict] = []
    for fold in folds:
        fold_id = fold["fold_id"]
        arrays = fold_arrays[fold_id]
        thresholds = fold_thresholds[fold_id]

        for seed in range(final_seeds):
            model, info = baseline_module.train_one(
                arrays, adjacency, config, seed, device, build_model=build_model
            )
            scores = evaluate_split(
                model, arrays["test"], adjacency, config["target"], thresholds, device
            )
            rows.append(
                {
                    "fold_id": fold_id,
                    "test_year": fold["test_year"],
                    "headline": fold["headline"],
                    "covers_covid": fold["covers_covid"],
                    "seed": seed,
                    "best_epoch": info["best_epoch"],
                    **scores,
                }
            )

    return pd.DataFrame(rows)


def summarise_test(metrics: pd.DataFrame) -> dict:
    """Headline / 2017 / COVID means from the test-split metrics, seeds averaged."""

    by_fold = (
        metrics.groupby(["fold_id", "test_year", "headline", "covers_covid"])
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), peak_mae=("peak_mae", "mean"))
        .reset_index()
    )
    headline = by_fold[by_fold["headline"]]
    covid = by_fold[by_fold["covers_covid"]]
    epidemic = by_fold[by_fold["test_year"] == 2017]

    seed_sd = metrics.groupby("fold_id")["mae"].std().mean()

    return {
        "headline_mae": float(headline["mae"].mean()),
        "headline_rmse": float(headline["rmse"].mean()),
        "headline_peak_mae": float(headline["peak_mae"].mean()),
        "epidemic_2017_mae": float(epidemic["mae"].mean()),
        "epidemic_2017_peak_mae": float(epidemic["peak_mae"].mean()),
        "covid_mae": float(covid["mae"].mean()),
        "seed_sd": float(seed_sd),
        "by_fold": by_fold,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def load_baseline_reference() -> dict | None:
    """The frozen baseline's headline numbers, read from its metrics CSV if present.

    `scripts/16` writes `baseline_metrics.csv`. If it has not been run this
    session the report still writes, just without the side-by-side row.
    """

    path = RESULTS_DIR / "baseline_metrics.csv"
    if not path.exists():
        return None

    metrics = pd.read_csv(path)
    # The README's headline baseline is gru_only on v1; the tuned model here is a
    # gcn_gru (it uses the real adjacency), so gcn_gru on v1 is the honest
    # like-for-like comparison. Both are reported.
    out = {}
    for (model, variant), group in metrics.groupby(["model", "variant"]):
        by_fold = group.groupby(["fold_id", "test_year", "headline"]).agg(
            mae=("mae", "mean"), peak_mae=("peak_mae", "mean")
        ).reset_index()
        headline = by_fold[by_fold["headline"]]
        out[f"{model}/{variant}"] = {
            "headline_mae": float(headline["mae"].mean()),
            "headline_peak_mae": float(headline["peak_mae"].mean()),
            "epidemic_2017_mae": float(
                by_fold.loc[by_fold["test_year"] == 2017, "mae"].mean()
            ),
        }
    return out


def load_persistence_reference() -> dict | None:
    """Persistence's headline numbers from the committed naive metrics CSV."""

    path = RESULTS_DIR / "naive_baseline_metrics.csv"
    if not path.exists():
        return None

    metrics = pd.read_csv(path)
    persistence = metrics[metrics["model"] == "persistence"]
    headline = persistence[persistence["headline"]]

    return {
        "headline_mae": float(headline["mae"].mean()),
        "headline_peak_mae": float(headline["peak_mae"].mean()),
        "epidemic_2017_mae": float(
            persistence.loc[persistence["test_year"] == 2017, "mae"].mean()
        ),
    }


def write_report(
    trials: pd.DataFrame,
    best_config: dict,
    best_val: dict,
    test_summary: dict,
    args: argparse.Namespace,
    baseline_ref: dict | None,
    persistence_ref: dict | None,
) -> None:
    """Write the tuning report."""

    ranked = trials.sort_values("val_mae").reset_index(drop=True)

    lines = [
        "# Hyperparameter search — full GCN+GRU",
        "",
        f"Version: `{TUNING_VERSION}`",
        "",
        f"Method: **{args.method}**, {len(trials)} trials, "
        f"{args.trial_seeds} seed(s) per trial during search, "
        f"{args.final_seeds} seeds for the test rerun.",
        f"Selection metric: mean masked MAE on the **validation** split of folds "
        f"{args.tune_folds}, seeds averaged. The test split is not read during "
        f"search.",
        f"Feature variant: `{args.variant}`. Search RNG seed: {args.search_seed}.",
        "",
        "## Search space",
        "",
        "| Axis | Values |",
        "| --- | --- |",
    ]
    for key, values in SEARCH_SPACE.items():
        lines.append(f"| `{key}` | {values} |")

    lines += [
        "",
        f"Grid size: {_grid_size():,} points. Held fixed at the baseline: "
        + ", ".join(f"`{k}`={v}" for k, v in FIXED.items())
        + ".",
        "",
        "## Best configuration (selected on validation folds only)",
        "",
        "| Axis | Value | Baseline default |",
        "| --- | --- | --- |",
    ]

    baseline_defaults = {
        "lookback": 12, "gcn_hidden": 32, "gcn_layers": 2, "gru_hidden": 32,
        "gru_layers": 1, "dropout": 0.2, "learning_rate": 3e-3,
        "batch_size": 64, "weight_decay": 1e-4,
    }
    for key in SEARCH_SPACE:
        moved = "" if best_config[key] == baseline_defaults[key] else "  ← moved"
        lines.append(
            f"| `{key}` | {best_config[key]}{moved} | {baseline_defaults[key]} |"
        )

    lines += [
        "",
        f"Validation MAE of this configuration: **{best_val['mean_mae']:.3f}** "
        f"(mean over folds {args.tune_folds}).",
        "",
        "## Verdict — the winning configuration on the frozen test protocol",
        "",
        "Retrained with "
        f"{args.final_seeds} seeds on all nine folds, scored on the **test** "
        "split by the same metric the baseline uses.",
        "",
        "| Model | Headline MAE | Headline peak MAE | 2017 MAE | Seed sd |",
        "| --- | --- | --- | --- | --- |",
    ]

    lines.append(
        f"| **tuned `gcn_gru` `{args.variant}`** | {test_summary['headline_mae']:.2f} | "
        f"{test_summary['headline_peak_mae']:.2f} | {test_summary['epidemic_2017_mae']:.2f} | "
        f"{test_summary['seed_sd']:.2f} |"
    )
    if baseline_ref and f"gcn_gru/{args.variant}" in baseline_ref:
        b = baseline_ref[f"gcn_gru/{args.variant}"]
        lines.append(
            f"| baseline `gcn_gru` `{args.variant}` (script 16 defaults) | "
            f"{b['headline_mae']:.2f} | {b['headline_peak_mae']:.2f} | "
            f"{b['epidemic_2017_mae']:.2f} | — |"
        )
    if baseline_ref and f"gru_only/{args.variant}" in baseline_ref:
        b = baseline_ref[f"gru_only/{args.variant}"]
        lines.append(
            f"| baseline `gru_only` `{args.variant}` (README headline best) | "
            f"{b['headline_mae']:.2f} | {b['headline_peak_mae']:.2f} | "
            f"{b['epidemic_2017_mae']:.2f} | — |"
        )
    if persistence_ref:
        p = persistence_ref
        lines.append(
            f"| persistence | {p['headline_mae']:.2f} | {p['headline_peak_mae']:.2f} | "
            f"{p['epidemic_2017_mae']:.2f} | — |"
        )

    # The honest reading.
    verdict_lines = ["", "### What this establishes", ""]
    if baseline_ref and f"gcn_gru/{args.variant}" in baseline_ref:
        b = baseline_ref[f"gcn_gru/{args.variant}"]
        delta = b["headline_mae"] - test_summary["headline_mae"]
        pct = 100 * delta / b["headline_mae"]
        sd = test_summary["seed_sd"]
        established = abs(delta) > 2 * sd
        verdict_lines.append(
            f"- Against the like-for-like baseline (`gcn_gru` `{args.variant}`, "
            f"script 16's hand-picked defaults), tuning moves headline MAE by "
            f"**{delta:+.2f} ({pct:+.1f}%)**, against a seed sd of {sd:.2f}. "
            + (
                "This exceeds two seed-sd and is a real effect."
                if established
                else "This is **within** two seed-sd — not established by this run."
            )
        )
    if persistence_ref:
        p = persistence_ref
        gap = test_summary["headline_mae"] - p["headline_mae"]
        verdict_lines.append(
            f"- The tuned model {'still ' if gap > 0 else ''}"
            f"{'loses to' if gap > 0 else 'beats'} persistence on the headline "
            f"mean ({test_summary['headline_mae']:.2f} vs {p['headline_mae']:.2f}, "
            f"{gap:+.2f})."
        )
        peak_gap = test_summary["headline_peak_mae"] - p["headline_peak_mae"]
        verdict_lines.append(
            f"- Peak MAE: {test_summary['headline_peak_mae']:.2f} vs persistence "
            f"{p['headline_peak_mae']:.2f} ({peak_gap:+.2f}). "
            + ("Still loses." if peak_gap > 0 else "Beats persistence on peak.")
        )
    verdict_lines.append(
        "- Selection used validation folds only; the test numbers above were "
        "computed once, after the configuration was fixed."
    )
    lines += verdict_lines

    lines += [
        "",
        "## Per-fold test MAE, tuned model (seeds averaged)",
        "",
        "| Fold | Year | MAE | RMSE | Peak MAE | Note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in test_summary["by_fold"].sort_values("fold_id").itertuples():
        note = "COVID" if row.covers_covid else ("epidemic" if row.test_year == 2017 else "")
        lines.append(
            f"| {row.fold_id} | {row.test_year} | {row.mae:.2f} | {row.rmse:.2f} | "
            f"{row.peak_mae:.2f} | {note} |"
        )

    lines += [
        "",
        f"## Top {min(15, len(ranked))} trials by validation MAE",
        "",
        "| Rank | val MAE | lookback | gcn h×L | gru h×L | dropout | lr | batch | wd |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank, row in enumerate(ranked.head(15).itertuples(), start=1):
        lines.append(
            f"| {rank} | {row.val_mae:.3f} | {row.lookback} | "
            f"{row.gcn_hidden}×{row.gcn_layers} | {row.gru_hidden}×{row.gru_layers} | "
            f"{row.dropout} | {row.learning_rate:.0e} | {row.batch_size} | "
            f"{row.weight_decay:.0e} |"
        )

    lines += [
        "",
        "## What the search space looks like around the optimum",
        "",
        "Mean validation MAE by the value taken on each axis, over all trials. A "
        "flat column is an axis the search found little signal on.",
        "",
    ]
    for axis in SEARCH_SPACE:
        grouped = trials.groupby(axis)["val_mae"].mean().sort_index()
        cells = "  ".join(f"{v}: {m:.2f}" for v, m in grouped.items())
        lines.append(f"- `{axis}` — {cells}")

    lines += [
        "",
        "## Output files",
        "",
        f"- `{TRIALS_PATH.relative_to(PROJECT_DIR).as_posix()}` — every trial, all axes and per-fold val MAE",
        f"- `{BEST_CONFIG_PATH.relative_to(PROJECT_DIR).as_posix()}` — the selected configuration as JSON",
        f"- `{TEST_METRICS_PATH.relative_to(PROJECT_DIR).as_posix()}` — per-fold per-seed test metrics for the tuned model",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _grid_size() -> int:
    size = 1
    for values in SEARCH_SPACE.values():
        size *= len(values)
    return size


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Parse the search knobs."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--method", choices=["random", "optuna"], default="random",
        help="random search (no dependency) or Optuna TPE if installed",
    )
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument(
        "--trial-seeds", type=int, default=2,
        help="seeds averaged per configuration during the search",
    )
    parser.add_argument(
        "--final-seeds", type=int, default=3,
        help="seeds for the best config's test-set rerun",
    )
    parser.add_argument(
        "--tune-folds", nargs="+", type=int, default=HEADLINE_FOLDS,
        help="folds whose validation split contributes to selection",
    )
    parser.add_argument("--variant", default="v1", choices=["v0", "v1"])

    return parser.parse_args()


def main() -> int:
    """Run the search, pick the best config on validation, verdict it on test."""

    args = parse_arguments()
    load_modules()

    if not FOLDS_PATH.exists() or not ADJACENCY_PATH.exists():
        print("Missing folds.json or adjacency.npz. Run scripts 13 and 14 first.")
        return 1

    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))["folds"]
    fold_ids = {f["fold_id"] for f in folds}
    missing = set(args.tune_folds) - fold_ids
    if missing:
        print(f"Unknown fold ids in --tune-folds: {sorted(missing)}")
        return 1

    with np.load(ADJACENCY_PATH, allow_pickle=True) as data:
        adjacency = data["A_norm"].astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    method = args.method
    if method == "optuna":
        try:
            import optuna  # noqa: F401
        except ImportError:
            print("optuna not installed — falling back to --method random.")
            print("Install with: .venv\\Scripts\\pip install optuna")
            method = "random"

    print(f"Device:      {device}")
    print(f"Method:      {method}")
    print(f"Trials:      {args.n_trials}")
    print(f"Tune folds:  {args.tune_folds}  (validation split only)")
    print(f"Variant:     {args.variant}")
    print(f"Grid size:   {_grid_size():,}")
    print(f"Seeds:       {args.trial_seeds}/trial in search, {args.final_seeds} for test rerun")
    print()

    cache = FoldCache(args.variant, folds, FIXED["horizon"])

    started = time.perf_counter()
    if method == "optuna":
        trial_rows = run_optuna_search(
            args.n_trials, args.search_seed, cache, adjacency,
            args.tune_folds, args.trial_seeds, device,
        )
    else:
        trial_rows = run_random_search(
            args.n_trials, args.search_seed, cache, adjacency,
            args.tune_folds, args.trial_seeds, device,
        )
    search_elapsed = (time.perf_counter() - started) / 60

    trials = pd.DataFrame(trial_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trials.to_csv(TRIALS_PATH, index=False)

    best_row = trials.sort_values("val_mae").iloc[0]
    best_sampled = {key: _coerce(best_row[key], SEARCH_SPACE[key]) for key in SEARCH_SPACE}
    best_config = full_config(best_sampled)

    print(f"\nSearch done in {search_elapsed:.1f} min.")
    print(f"Best validation MAE: {best_row['val_mae']:.3f}")
    print(f"Best config: {best_sampled}\n")

    BEST_CONFIG_PATH.write_text(
        json.dumps(
            {
                "sampled": best_sampled,
                "fixed": FIXED,
                "validation_mae": float(best_row["val_mae"]),
                "tune_folds": args.tune_folds,
                "variant": args.variant,
                "method": method,
                "n_trials": int(args.n_trials),
                "search_seed": args.search_seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best_val = {"mean_mae": float(best_row["val_mae"])}

    print("Retraining the best config on all nine folds for the test verdict...")
    test_metrics = score_on_test(
        best_config, cache, adjacency, folds, args.final_seeds, device
    )
    test_metrics.to_csv(TEST_METRICS_PATH, index=False)
    test_summary = summarise_test(test_metrics)

    baseline_ref = load_baseline_reference()
    persistence_ref = load_persistence_reference()

    write_report(
        trials, best_config, best_val, test_summary, args, baseline_ref, persistence_ref
    )

    print(f"\nTuned model on the test protocol:")
    print(f"  headline MAE      {test_summary['headline_mae']:.2f}")
    print(f"  headline peak MAE {test_summary['headline_peak_mae']:.2f}")
    print(f"  2017 MAE          {test_summary['epidemic_2017_mae']:.2f}")
    print(f"  seed sd           {test_summary['seed_sd']:.2f}")
    if baseline_ref and f"gcn_gru/{args.variant}" in baseline_ref:
        b = baseline_ref[f"gcn_gru/{args.variant}"]
        print(
            f"  vs baseline gcn_gru {args.variant}: "
            f"{b['headline_mae']:.2f} -> {test_summary['headline_mae']:.2f} "
            f"({test_summary['headline_mae'] - b['headline_mae']:+.2f})"
        )

    print(f"\nWrote {REPORT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {TRIALS_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {BEST_CONFIG_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {TEST_METRICS_PATH.relative_to(PROJECT_DIR)}")

    return 0


def _coerce(value, allowed: list):
    """Match a value read back from the CSV to its canonical entry in `allowed`.

    Round-tripping through pandas turns ints into numpy ints and can nudge a
    float; this snaps the value back to the exact object in the search list so
    the retrain uses `int`/`float`, not `np.int64`.
    """

    for candidate in allowed:
        if isinstance(candidate, bool):
            if bool(value) == candidate:
                return candidate
        elif isinstance(candidate, int) and not isinstance(candidate, bool):
            if int(value) == candidate:
                return candidate
        elif isinstance(candidate, float):
            if abs(float(value) - candidate) < 1e-12:
                return candidate
        elif value == candidate:
            return candidate
    return value


if __name__ == "__main__":
    sys.exit(main())

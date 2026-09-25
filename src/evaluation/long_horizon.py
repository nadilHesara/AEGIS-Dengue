"""
Shared protocol for the long-horizon benchmark (scripts 31-34).

Every method in the benchmark -- trained GRUs, gradient-boosted trees, a
pretrained foundation model, and the naive baselines -- is scored on exactly
the same cells: one cell is a (fold, split, horizon, target period, district),
and a method's prediction for it may use nothing after the forecast origin
`target - horizon`. Methods write long-format prediction frames keyed on those
columns; `scripts/evaluation/34` joins them and scores every method on the
intersection, so a method that silently covered different weeks could not look
better or worse for it.

Folds are the project's walk-forward folds (scripts/14) plus one extra,
**fold 0 = test year 2016**. It is never part of a headline number. It exists
so that fold 1 (2017) has a genuinely out-of-sample calibration year: interval
calibration and ensemble weights for fold f are fitted on fold f-1's *test*
predictions -- forecasts for the year before f's test year, made by models that
never trained on it. Using a model's own validation predictions instead would
be mildly optimistic, because early stopping selected the epoch on them.

Horizons run to 12 weeks. The measured rainfall-to-dengue delay is 5-10 weeks
(scripts/17; Liu et al. 2025 find 7-9 weeks for Sri Lanka with a different
model), so a horizon curve that stops at 4 cannot say whether climate's
contribution keeps rising, peaks near the delay, or fades.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
BENCHMARK_DIR = PROJECT_DIR / "results" / "benchmark"
PREDICTIONS_DIR = BENCHMARK_DIR / "predictions"

HORIZONS = (1, 2, 3, 4, 6, 8, 10, 12)

# Fold 0 (2016) is the calibration fold for fold 1; it is excluded from every
# headline number. 2017..2025 keep the ids 1..9 they have everywhere else.
FIRST_TEST_YEAR = 2016
LAST_TEST_YEAR = 2025
CALIBRATION_FOLD = 0

HEADLINE_FOLDS = (1, 2, 3, 6, 7, 8, 9)

# Quantile levels for probabilistic forecasts: the median plus the 50%, 80%
# and 95% central intervals, which is the level set WIS is computed over.
QUANTILES = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)

KEY_COLUMNS = ["fold_id", "split", "horizon", "target_period_id", "node_id"]


def load_script(name: str, relative: str):
    """Import a numerically-prefixed pipeline script by path."""

    path = PROJECT_DIR / "scripts" / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_pipeline():
    """Return script 16 with its naive/fold/tensor modules loaded."""

    baseline = load_script("baseline_module", "training/16.train_gcn_gru.py")
    baseline.load_modules()
    return baseline


def build_benchmark_folds(folds_module) -> list[dict]:
    """The project's folds, extended back one year to a calibration fold 0.

    Built by scripts/14's own `build_folds`, so fold 1..9 are identical to
    `data/processed/folds.json`; only the id offset is applied here.
    """

    calendar = folds_module.load_calendar()
    regime = folds_module.load_regime_flags()
    folds = folds_module.build_folds(
        calendar, regime, first_test_year=FIRST_TEST_YEAR, last_test_year=LAST_TEST_YEAR
    )
    for fold in folds:
        fold["fold_id"] = int(fold["test_year"] - FIRST_TEST_YEAR)
        fold["calibration_only"] = fold["fold_id"] == CALIBRATION_FOLD
        fold["headline"] = fold["fold_id"] in HEADLINE_FOLDS
    return folds


def check_against_committed_folds(folds: list[dict]) -> None:
    """Fail if folds 1..9 differ from scripts/14's committed folds.json."""

    import json

    committed = json.loads((PROCESSED_DIR / "folds.json").read_text())
    committed = committed["folds"] if isinstance(committed, dict) else committed
    by_id = {int(f["fold_id"]): f for f in committed}
    keys = ("train_end_period", "val_start_period", "val_end_period",
            "test_start_period", "test_end_period", "fit_end_period")
    for fold in folds:
        if fold["fold_id"] == CALIBRATION_FOLD:
            continue
        reference = by_id[fold["fold_id"]]
        for key in keys:
            if int(reference[key]) != int(fold[key]):
                raise ValueError(
                    f"fold {fold['fold_id']} {key}: {fold[key]} != committed {reference[key]}"
                )


def months_for(folds_module) -> np.ndarray:
    calendar = folds_module.load_calendar()
    return calendar.sort_values("period_id")["month"].to_numpy()


def cell_frame(
    method: str,
    fold: dict,
    split: str,
    horizon: int,
    target_period_id: np.ndarray,
    prediction: np.ndarray,
    seed: int = -1,
    quantiles: np.ndarray | None = None,
) -> pd.DataFrame:
    """Long-format predictions for one method/fold/split/horizon/seed.

    `prediction` is [windows, nodes] on the case scale; `quantiles`, when
    given, is [windows, nodes, len(QUANTILES)] on the case scale.
    """

    n_windows, n_nodes = prediction.shape
    frame = pd.DataFrame(
        {
            "method": method,
            "fold_id": int(fold["fold_id"]),
            "split": split,
            "horizon": int(horizon),
            "seed": int(seed),
            "target_period_id": np.repeat(np.asarray(target_period_id, dtype=np.int32), n_nodes),
            "node_id": np.tile(np.arange(n_nodes, dtype=np.int16), n_windows),
            "prediction": prediction.reshape(-1).astype(np.float32),
        }
    )
    if quantiles is not None:
        flat = quantiles.reshape(-1, quantiles.shape[-1]).astype(np.float32)
        for index, level in enumerate(QUANTILES):
            frame[quantile_column(level)] = flat[:, index]
    return frame


def quantile_column(level: float) -> str:
    return f"q{level:.3f}"


def truth_frame(tensors: dict[str, np.ndarray]) -> pd.DataFrame:
    """Observed cases per (period, district), for joining onto predictions."""

    y, mask = tensors["y"], tensors["y_mask"]
    periods = tensors["period_id"]
    return pd.DataFrame(
        {
            "target_period_id": np.repeat(periods.astype(np.int32), y.shape[1]),
            "node_id": np.tile(np.arange(y.shape[1], dtype=np.int16), y.shape[0]),
            "actual": y.reshape(-1).astype(np.float32),
            "observed": mask.reshape(-1).astype(np.int8),
        }
    )


def save_predictions(frames: list[pd.DataFrame], name: str) -> Path:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"{name}.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def weighted_interval_score(
    actual: np.ndarray, quantiles: np.ndarray, levels=QUANTILES
) -> np.ndarray:
    """WIS per cell (Bracher et al. 2021), from symmetric quantile pairs.

    `quantiles` is [cells, len(levels)] with the median in the middle. With K
    central intervals, WIS = (0.5*|y-m| + sum_k alpha_k/2 * IS_alpha_k) / (K+0.5).
    """

    levels = np.asarray(levels)
    median_index = int(np.argmin(np.abs(levels - 0.5)))
    median = quantiles[:, median_index]
    total = 0.5 * np.abs(actual - median)
    n_intervals = 0
    for lower_index in range(median_index):
        upper_index = len(levels) - 1 - lower_index
        alpha = 2.0 * levels[lower_index]
        lower, upper = quantiles[:, lower_index], quantiles[:, upper_index]
        interval = (
            (upper - lower)
            + (2.0 / alpha) * np.clip(lower - actual, 0.0, None)
            + (2.0 / alpha) * np.clip(actual - upper, 0.0, None)
        )
        total = total + (alpha / 2.0) * interval
        n_intervals += 1
    return total / (n_intervals + 0.5)

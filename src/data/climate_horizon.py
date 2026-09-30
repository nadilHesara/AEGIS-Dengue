"""
Data interface for the climate-horizon extension.

Plan: docs/climate_horizon_implementation.md §6. This module builds the
arrays both the baseline (`hcd_uniform`) and the proposed model
(`hcd_informed`) train on. The two arms therefore see exactly the same
origins, cells, preprocessing and masks.

Inputs are the existing Open-Meteo tensors (`data/processed/model_tensors_v2.npz`,
recorded in `results/benchmark/data_manifest.json`) and the committed
walk-forward folds. Nothing is rebuilt.

Channels
    case      case history, calendar and static channels -- everything that is
              not climate or derived from climate
    climate   the 7 instantaneous weather channels, which feed the delay
              encoder
    excluded  channels derived from climate (the v1 rolling means and any v4
              lags). They go to neither branch, so the encoder is the only
              path through which delayed weather reaches the model.

Availability assumptions (stated, not verified against operations)
    - The case count for week t is known at origin t. Reporting delay and
      back-fill are not modelled; the source file is treated as final.
    - Reconstructed weather for week t is also known at origin t. Open-Meteo
      era5_seamless has a lag of a few days to a week, so this may be one
      week optimistic. The benchmark's `nb_shared_v2_wxlag1` control (weather
      one week late) changed MAE by at most +0.27 at h = 1-4
      (results/benchmark/tables/ablations.csv).
    - Calendar and static channels (day of year, centroids, observation flags)
      are known in advance.
    - Outbreak-history channels (national wave rank, trailing 52-week cases)
      use only cases up to week t.

Splits
    Each (origin, horizon) cell belongs to the split of its own target period
    (`split_by_target`). An origin enters a split's array when at least one of
    its cells does; its other cells are masked. The eligible origins are the
    same for every fold and arm: every origin with the full `WINDOW` of input
    history behind it (the longest lag reach plus the model lookback).

Inner blocks (for utility estimation)
    For outer fold f, whose training period ends in calendar year Y, block k
    scores year s = Y - K + 1 + k. Its early-stopping year is s - 1, and pilot
    training uses targets up to the end of year s - 2. Utility scoring and
    checkpoint selection are therefore different years, and every block ends
    at or before the outer training cut-off. Preprocessing is fitted only on
    the pilot-training periods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.evaluation.long_horizon import split_by_target
from src.models.climate_ablation import CLIMATE_FEATURES, climate_indices


LOOKBACK = 12
LAG_REACH = 26
WINDOW = LOOKBACK + LAG_REACH - 1  # 37 periods of input history per origin
HORIZONS = (1, 2, 3, 4)
VARIANT = "v2"
N_INNER_BLOCKS = 3

# Development and utility estimation stop at the end of 2025 (period 993).
# Fold 10 (2026) is covered only by the hold-out addendum of 2026-09-30.
DEVELOPMENT_LAST_YEAR = 2025

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelSplit:
    case: list[int]
    climate: list[int]
    excluded: list[int]
    names: list[str]

    def case_names(self) -> list[str]:
        return [self.names[i] for i in self.case]

    def climate_names(self) -> list[str]:
        return [self.names[i] for i in self.climate]


def split_channels(feature_names) -> ChannelSplit:
    """Separate case/calendar/static channels from climate and climate-derived ones."""

    names = [str(name) for name in feature_names]
    derived_or_raw = set(climate_indices(names))
    climate = [names.index(name) for name in CLIMATE_FEATURES if name in names]
    if len(climate) != len(CLIMATE_FEATURES):
        missing = sorted(set(CLIMATE_FEATURES) - set(names))
        raise ValueError(f"Missing climate channels: {missing}")
    excluded = sorted(derived_or_raw - set(climate))
    case = [i for i in range(len(names)) if i not in derived_or_raw]
    return ChannelSplit(case=case, climate=climate, excluded=excluded, names=names)


# ---------------------------------------------------------------------------
# Folds and inner blocks
# ---------------------------------------------------------------------------

def year_of_period(calendar, period_id: int) -> int:
    row = calendar.loc[calendar["period_id"] == period_id, "year"]
    return int(row.iloc[0])


def development_last_period(calendar) -> int:
    return int(calendar.loc[calendar["year"] == DEVELOPMENT_LAST_YEAR, "period_id"].max())


def inner_blocks(fold: dict, calendar, n_blocks: int = N_INNER_BLOCKS) -> list[dict]:
    """Pilot-train / early-stop / utility-score blocks inside one outer training period.

    Each block is a fold-shaped dict, so the existing helpers accept it:
    `train_end_period` (pilot training), `val_*` (early stopping), `test_*`
    (utility scoring), and `fit_end_period` = pilot-training end.
    """

    def bounds(year: int) -> tuple[int, int]:
        rows = calendar[calendar["year"] == year]["period_id"]
        if rows.empty:
            raise ValueError(f"No reporting periods in {year}.")
        return int(rows.min()), int(rows.max())

    last_train_year = year_of_period(calendar, fold["train_end_period"])
    blocks = []
    for k in range(n_blocks):
        score_year = last_train_year - n_blocks + 1 + k
        stop_year = score_year - 1
        score_start, score_end = bounds(score_year)
        stop_start, stop_end = bounds(stop_year)
        _, train_end = bounds(stop_year - 1)
        block = {
            "fold_id": fold["fold_id"],
            "block": k,
            "score_year": score_year,
            "train_end_period": train_end,
            "val_start_period": stop_start,
            "val_end_period": stop_end,
            "test_start_period": score_start,
            "test_end_period": score_end,
            "fit_end_period": train_end,
        }
        if score_end > fold["train_end_period"]:
            raise AssertionError(f"inner block {k} of fold {fold['fold_id']} leaves the training period")
        blocks.append(block)
    return blocks


# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------

def eligible_origins(n_periods: int, window: int = WINDOW, horizons=HORIZONS) -> np.ndarray:
    """Origin indices with a full input window and at least the shortest target inside the data."""

    return np.arange(window - 1, n_periods - min(horizons))


def build_anchor(y: np.ndarray, y_mask: np.ndarray, period_id: np.ndarray,
                 origins: np.ndarray, fit_end_period: int) -> np.ndarray:
    """log1p(cases at origin). An unobserved origin cell uses the district's mean
    over periods <= fit_end_period (scripts/16.build_anchor's rule)."""

    history = period_id <= fit_end_period
    observed = np.where(y_mask[history] == 1, y[history], np.nan)
    fallback = np.nanmean(observed, axis=0)
    at_origin = np.where(y_mask[origins] == 1, y[origins], np.nan)
    at_origin = np.where(np.isnan(at_origin), fallback[None, :], at_origin)
    return np.log1p(at_origin).astype(np.float32)


def scaled_features(tensors, months, fold: dict, folds_module) -> np.ndarray:
    """Impute and scale every period with statistics from `period_id <= fit_end_period`."""

    statistics = folds_module.fit_fold_statistics(
        tensors, months, tensors["period_id"] <= fold["fit_end_period"]
    )
    return folds_module.transform(tensors, months, statistics).astype(np.float32)


def build_arrays(
    tensors: dict[str, np.ndarray],
    months: np.ndarray,
    fold: dict,
    folds_module,
    channels: ChannelSplit,
    horizons=HORIZONS,
    window: int = WINDOW,
    last_target_period: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Scale, window and split one fold (or inner block) for the two-branch model.

    Preprocessing is fitted on `period_id <= fold["fit_end_period"]`. Cells
    whose target lies after `last_target_period` (2026 during development)
    are dropped entirely. Returns per split: `X_case [o, lookback, N, Fc]`,
    `X_climate [o, window, N, 7]`, `y`/`mask [o, N, H]` (y finite, 0 where
    masked), `anchor [o, N]`, `origin_period_id [o]`,
    `target_period_id [o, H]` and `cells [o, H]`.
    """

    period_id = tensors["period_id"]
    n_periods = len(period_id)
    x = scaled_features(tensors, months, fold, folds_module)
    if np.isnan(x[..., channels.case + channels.climate]).any():
        raise ValueError("NaN left in model inputs after per-fold imputation")

    origins = eligible_origins(n_periods, window, horizons)
    horizon_array = np.asarray(horizons)
    targets = origins[:, None] + horizon_array[None, :]
    inside = targets < n_periods
    clipped = np.minimum(targets, n_periods - 1)
    target_period = np.where(inside, period_id[clipped], -1)
    if last_target_period is not None:
        target_period = np.where(target_period <= last_target_period, target_period, -1)
    labels = split_by_target(target_period, fold)
    labels[target_period < 0] = ""

    y_raw = tensors["y"][clipped].transpose(0, 2, 1)                   # [o, N, H]
    observed = (tensors["y_mask"][clipped] == 1).transpose(0, 2, 1)
    observed &= ~np.isnan(y_raw)
    anchor = build_anchor(tensors["y"], tensors["y_mask"], period_id, origins,
                          fold["fit_end_period"])

    case_steps = np.arange(-LOOKBACK + 1, 1)
    climate_steps = np.arange(-window + 1, 1)
    bounds = {
        "train": (-np.inf, fold["train_end_period"]),
        "val": (fold["val_start_period"], fold["val_end_period"]),
        "test": (fold["test_start_period"], fold["test_end_period"]),
    }

    arrays = {}
    for split in SPLITS:
        cells = labels == split                                         # [o, H]
        keep = cells.any(axis=1)
        mask = observed & cells[:, None, :]                             # [o, N, H]
        live = np.broadcast_to(target_period[:, None, :], mask.shape)[mask]
        low, high = bounds[split]
        if live.size and (live.min() < low or live.max() > high):
            raise AssertionError(f"{split} cells cross the split boundary in fold {fold['fold_id']}")
        o = origins[keep]
        arrays[split] = {
            "X_case": x[o[:, None] + case_steps[None, :]][..., channels.case],
            "X_climate": x[o[:, None] + climate_steps[None, :]][..., channels.climate],
            "y": np.where(mask[keep], np.nan_to_num(y_raw[keep], nan=0.0), 0.0).astype(np.float32),
            "mask": mask[keep].astype(np.float32),
            "anchor": anchor[keep],
            "origin_period_id": period_id[o],
            "target_period_id": target_period[keep],
            "cells": cells[keep],
        }
    return arrays


def cell_keys(arrays: dict, fold_id: int, horizons=HORIZONS, split: str = "test"):
    """(fold, split, horizon, target_period_id, node_id) of every observed cell."""

    import pandas as pd

    part = arrays[split]
    o, n, h = np.nonzero(part["mask"] == 1)
    return pd.DataFrame({
        "fold_id": fold_id,
        "split": split,
        "horizon": np.asarray(horizons)[h],
        "target_period_id": part["target_period_id"][o, h],
        "node_id": n,
    })


def extension_fold(fold: dict) -> dict:
    """An outer fold with preprocessing fitted on its training portion only.

    The benchmark fits scaling and imputation up to the end of the validation
    year (`fit_end_period`). The extension fits them up to `train_end_period`,
    for both arms and all pilots. Persistence and the target are unaffected.
    """

    return {**fold, "fit_end_period": fold["train_end_period"]}

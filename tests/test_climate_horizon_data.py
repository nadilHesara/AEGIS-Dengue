"""Data interface, loss, conversion and evaluator checks for the climate-horizon extension."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from src.data.climate_horizon import (
    HORIZONS, WINDOW, build_arrays, eligible_origins, inner_blocks, split_channels,
)
from src.evaluation import climate_horizon as ev
from src.evaluation.long_horizon import load_script, split_by_target
from src.models.climate_ablation import CLIMATE_FEATURES
from src.models.climate_horizon import anchored_mean, nb_loss_by_horizon, nb_point_forecast
from src.models.negative_binomial import NegBinHead, masked_negative_binomial_loss

V2_NAMES = [
    "cases_log1p", "rainfall_daily_mean_mm", "rainy_days_frac", "temperature_mean_c",
    "diurnal_range_c", "dewpoint_mean_c", "relative_humidity_mean", "wind_speed_mean",
    "doy_sin", "doy_cos", "weather_observed", "case_observed", "centroid_lat", "centroid_lon",
    "rainfall_daily_mean_mm_roll4", "rainfall_daily_mean_mm_roll8", "rainfall_daily_mean_mm_roll12",
    "temperature_mean_c_roll4", "temperature_mean_c_roll8", "temperature_mean_c_roll12",
    "relative_humidity_mean_roll4", "relative_humidity_mean_roll8", "relative_humidity_mean_roll12",
    "national_wave_rank", "trailing_52_cumulative_cases",
]


@pytest.fixture(scope="module")
def folds_module():
    return load_script("build_folds_test", "features/14.build_folds.py")


def synthetic(n_periods=200, n_nodes=4, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.poisson(20, size=(n_periods, n_nodes)).astype(np.float64)
    y_mask = np.ones_like(y, dtype=np.int8)
    y[150, 1] = np.nan
    y_mask[150, 1] = 0
    y[60, 2] = np.nan
    y_mask[60, 2] = 0
    x = rng.normal(size=(n_periods, n_nodes, len(V2_NAMES))).astype(np.float32)
    x[..., 0] = np.log1p(np.nan_to_num(y))
    x[10, 0, 1] = np.nan  # a missing weather cell, imputed per fold
    tensors = {
        "X": x, "y": y, "y_mask": y_mask,
        "period_id": np.arange(1, n_periods + 1),
        "feature_names": np.array(V2_NAMES),
    }
    months = (np.arange(n_periods) // 4) % 12 + 1
    fold = {"fold_id": 1, "train_end_period": 120, "val_start_period": 121, "val_end_period": 150,
            "test_start_period": 151, "test_end_period": 190, "fit_end_period": 120}
    return tensors, months, fold


# --------------------------------------------------------------------------- channels

def test_channels_separate_case_from_all_climate():
    split = split_channels(V2_NAMES)
    assert split.climate_names() == list(CLIMATE_FEATURES)
    assert not set(split.case) & (set(split.climate) | set(split.excluded))
    assert sorted(split.case + split.climate + split.excluded) == list(range(len(V2_NAMES)))
    assert all("roll" in V2_NAMES[i] for i in split.excluded) and len(split.excluded) == 9
    assert "cases_log1p" in split.case_names() and "national_wave_rank" in split.case_names()
    assert not any(n in split.case_names() for n in CLIMATE_FEATURES)


# --------------------------------------------------------------------------- inner blocks

def test_inner_blocks_are_chronological_and_inside_training(folds_module):
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "processed" / "folds.json"
    if not path.exists():
        pytest.skip("folds.json not built")
    calendar = folds_module.load_calendar()
    for fold in json.loads(path.read_text())["folds"]:
        blocks = inner_blocks(fold, calendar)
        assert len({b["score_year"] for b in blocks}) == 3
        for b in blocks:
            assert b["train_end_period"] < b["val_start_period"] <= b["val_end_period"] < b["test_start_period"]
            assert b["test_end_period"] <= fold["train_end_period"] < fold["val_start_period"]
            assert b["fit_end_period"] == b["train_end_period"]
            # utility scoring is never the early-stopping year
            assert b["test_start_period"] > b["val_end_period"]
        assert blocks[-1]["test_end_period"] == fold["train_end_period"]


# --------------------------------------------------------------------------- arrays

def test_split_uses_every_target_date(folds_module):
    tensors, months, fold = synthetic()
    arrays = build_arrays(tensors, months, fold, folds_module, split_channels(V2_NAMES))
    bounds = {"train": (1, 120), "val": (121, 150), "test": (151, 190)}
    for split, (low, high) in bounds.items():
        part = arrays[split]
        live = part["target_period_id"][:, None, :].repeat(4, 1)[part["mask"] == 1]
        assert live.min() >= low and live.max() <= high
        labels = split_by_target(part["target_period_id"], fold)
        assert np.array_equal(labels == split, part["cells"])


def test_origins_are_shared_and_leave_room_for_the_lag(folds_module):
    tensors, months, fold = synthetic()
    arrays = build_arrays(tensors, months, fold, folds_module, split_channels(V2_NAMES))
    first = min(a["origin_period_id"].min() for a in arrays.values())
    assert first == WINDOW  # period_id of index WINDOW - 1
    assert eligible_origins(200)[0] == WINDOW - 1


def test_scaling_is_fitted_only_on_fit_end(folds_module):
    tensors, months, fold = synthetic()
    channels = split_channels(V2_NAMES)
    before = build_arrays(tensors, months, fold, folds_module, channels)
    poisoned = dict(tensors)
    x = tensors["X"].copy()
    x[fold["fit_end_period"]:] = 1e6
    poisoned["X"] = x
    after = build_arrays(poisoned, months, fold, folds_module, channels)
    assert np.allclose(before["train"]["X_case"], after["train"]["X_case"])
    assert np.allclose(before["train"]["X_climate"], after["train"]["X_climate"])


def test_masks_targets_and_district_order(folds_module):
    tensors, months, fold = synthetic()
    arrays = build_arrays(tensors, months, fold, folds_module, split_channels(V2_NAMES))
    index = {int(p): i for i, p in enumerate(tensors["period_id"])}
    for part in arrays.values():
        assert np.isfinite(part["y"]).all() and np.isfinite(part["X_case"]).all()
        o, n, h = np.nonzero(part["mask"] == 1)
        periods = part["target_period_id"][o, h]
        expected = tensors["y"][[index[int(p)] for p in periods], n]
        assert np.array_equal(part["y"][o, n, h], expected)
        assert (part["y"][part["mask"] == 0] == 0).all()
    # the unobserved cell (period 151, node 1) is never live
    test = arrays["test"]
    assert ((test["target_period_id"] == 151).any())
    assert not ((test["target_period_id"][:, None, :] == 151) & (np.arange(4)[None, :, None] == 1)
                & (test["mask"] == 1)).any()


def test_climate_window_ends_at_origin(folds_module):
    tensors, months, fold = synthetic()
    channels = split_channels(V2_NAMES)
    arrays = build_arrays(tensors, months, fold, folds_module, channels)
    stats_ = folds_module.fit_fold_statistics(tensors, months, tensors["period_id"] <= fold["fit_end_period"])
    scaled = folds_module.transform(tensors, months, stats_)
    part = arrays["test"]
    origin_index = part["origin_period_id"] - 1
    assert np.allclose(part["X_climate"][:, -1], scaled[origin_index][..., channels.climate])
    assert np.allclose(part["X_climate"][:, 0], scaled[origin_index - WINDOW + 1][..., channels.climate])
    assert np.allclose(part["X_case"][:, -1], scaled[origin_index][..., channels.case])


def test_anchor_is_log1p_origin_with_history_fallback(folds_module):
    tensors, months, fold = synthetic()
    arrays = build_arrays(tensors, months, fold, folds_module, split_channels(V2_NAMES))
    part = arrays["train"]
    origin = part["origin_period_id"] - 1
    raw = tensors["y"][origin]
    live = ~np.isnan(raw)
    assert np.allclose(part["anchor"][live], np.log1p(raw[live]))
    row = np.nonzero(part["origin_period_id"] == 61)[0][0]
    fallback = np.nanmean(tensors["y"][:120, 2])
    assert np.isclose(part["anchor"][row, 2], np.log1p(fallback))


def test_last_target_period_drops_later_cells(folds_module):
    tensors, months, fold = synthetic()
    arrays = build_arrays(tensors, months, fold, folds_module, split_channels(V2_NAMES),
                          last_target_period=170)
    live = arrays["test"]["target_period_id"][:, None, :].repeat(4, 1)[arrays["test"]["mask"] == 1]
    assert live.max() <= 170


# --------------------------------------------------------------------------- loss and conversion

def test_per_horizon_loss_sums_to_pooled_loss_and_is_nan_safe():
    torch.manual_seed(0)
    mu = torch.rand(5, 4, 4) * 30 + 1
    alpha = torch.rand(5, 4, 4) + 0.1
    y = torch.poisson(torch.full((5, 4, 4), 15.0))
    mask = (torch.rand(5, 4, 4) > 0.2).float()
    pooled = masked_negative_binomial_loss(mu, alpha, y, mask)
    assert torch.isclose(nb_loss_by_horizon(mu, alpha, y, mask).sum(), pooled)

    delta = torch.zeros(5, 4, 4, requires_grad=True)
    y_nan = y.clone()
    y_nan[mask == 0] = float("nan")
    loss = nb_loss_by_horizon(anchored_mean(torch.log1p(y.mean(-1)), delta), alpha, y_nan, mask).sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(delta.grad).all()
    assert (delta.grad[mask == 0] == 0).all()


def test_nb_head_is_anchored_on_the_actual_origin_count():
    counts = torch.tensor([[0.0, 3.0, 250.0]])
    anchor = torch.log1p(counts)
    head = NegBinHead(hidden_dim=8, horizon=4)
    mu, _ = head(torch.randn(3, 8), anchor)
    assert torch.allclose(mu, (counts + 1).unsqueeze(-1).expand(1, 3, 4))
    assert torch.allclose(anchored_mean(anchor, torch.zeros(1, 3, 4)), mu)


def test_point_forecast_is_the_nb_median():
    mu = np.array([[1.0, 12.0, 300.0]])
    alpha = np.array([[0.5, 0.2, 0.05]])
    r = 1 / alpha
    expected = stats.nbinom.ppf(0.5, r, r / (r + mu))
    assert np.allclose(nb_point_forecast(mu, alpha), expected)


# --------------------------------------------------------------------------- evaluator

def test_persistence_on_cells_copies_the_origin():
    tensors, _, fold = synthetic()
    cells = pd.DataFrame({"fold_id": 1, "split": "test", "horizon": [1, 4],
                          "target_period_id": [160, 160], "node_id": [0, 3]})
    out = ev.persistence_on_cells(cells, tensors, {1: fold})
    assert out["prediction"].tolist() == [tensors["y"][158, 0], tensors["y"][155, 3]]


def test_check_reference_flags_a_wrong_split_label(tmp_path, monkeypatch):
    tensors, _, fold = synthetic()
    monkeypatch.setattr(ev, "PREDICTIONS_DIR", tmp_path)
    frame = pd.DataFrame({"method": "m", "fold_id": 1, "split": "test", "horizon": [1, 2, 3, 4],
                          "target_period_id": [151, 152, 153, 140], "node_id": 0, "seed": 0,
                          "prediction": 1.0})
    frame.to_parquet(tmp_path / "m.parquet")
    truth = pd.DataFrame({"target_period_id": range(1, 201), "node_id": 0, "observed": 1})
    result = ev.check_reference("m", frame, {1: fold}, truth, pd.Timestamp("2000-01-01", tz="UTC"))
    assert not result["split_rule_per_target"] and result["comparison"] == "context only"


def test_loading_references_does_not_modify_them():
    path = ev.PREDICTIONS_DIR / "nb_shared_v2.parquet"
    if not path.exists():
        pytest.skip("benchmark predictions not present")
    before = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime)
    frame = ev.load_reference("nb_shared_v2")
    assert set(frame["horizon"]) == set(HORIZONS) and (frame["fold_id"] <= 9).all()
    assert (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime) == before


def test_extension_fold_fits_on_training_only():
    from src.data.climate_horizon import extension_fold

    _, _, fold = synthetic()
    moved = extension_fold({**fold, "fit_end_period": 150})
    assert moved["fit_end_period"] == moved["train_end_period"] == 120

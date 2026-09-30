"""
Neural arms of the long-horizon benchmark: identity-backbone GRUs at h = 1..12.

The identity backbone (`gru_only`) is the model the short paper's positive
result rests on (README §8f), so it is the one extended here. Every arm reuses
scripts/16 verbatim -- `build_fold_arrays` for per-fold imputation, scaling and
windowing, `train_one` for Adam, clipping, early stopping on the validation
year and seeding -- except the quantile and level-weighted arms, which need a
different loss and so run `train_custom`, a line-for-line copy of `train_one`
with the loss swapped. The plain MSE arms never touch it.

Arms
    gru_v1            the paper's model: v1 features (climate + 4/8/12-week
                      rolling climate means), MSE on the anchored log residual.
    gru_v2            v1 + outbreak-history features (national wave rank,
                      trailing 52-week cumulative cases; docs/outbreak_history_features.md).
    gru_v2_shuffled   gru_v2 with every climate-derived channel time-shuffled
                      within district and split (src/models/climate_ablation.py).
                      Holds width, parameters and scaler statistics fixed and
                      destroys only weather-to-time alignment -- the §8h control,
                      now at every horizon to 12.
    gru_v2_quantile   gru_v2 with 7 outputs trained by pinball loss at
                      QUANTILES; its median is the point forecast and its
                      quantiles are a native predictive distribution.
    gcn_v2            contiguity graph instead of identity, v2 features: the
                      graph question (§8e) re-asked at long horizons.
    gru_v2_lw         gru_v2 with scripts/20's level weighting (per-cell weight
                      log1p(cases at origin), mean 1 per batch): the only change
                      that moved the 2017 fold at h=1 (README §8), never tried
                      beyond h=1.
    gru_v2_quantile_lw  gru_v2_quantile with the same level weighting.
    adaptive_v2       a graph learned end-to-end (Graph WaveNet adjacency,
                      src/models/adaptive_graph.py) exactly as scripts/26 trains
                      it, graph LR multiplier selected per fold and horizon on
                      the validation year. README §9 item 1: the one graph
                      question still open at long horizons.

Negative Binomial arms (src/models/negative_binomial.py; revision of 29 Sep 2026)
    gru_v2_nb         gru_v2 with the MSE head replaced by an NB2 head
                      (mean, dispersion), one model per horizon. With gru_v2
                      and gru_v2_quantile it is the head ablation: same
                      features, backbone, optimiser, seeds and stopping rule.
    nb_shared_<v>     one GRU trunk with a parallel NB head per horizon, all
                      eight horizons trained jointly. Each (origin, horizon)
                      cell is assigned to the split of its own target, so a
                      training or validation origin never carries a target
                      from a later split (the purge the earlier four-horizon
                      script lacked). v2/v3/v4 is the feature ablation.
    nb_shared_v2_{shuffled,climatology,noclimate}
                      climate controls on the shared NB model: time-shuffled
                      climate, climate replaced by each district's
                      week-of-year climatology from the training years only,
                      and climate removed.
    nb_shared_v2_wxlag1
                      weather delayed by one reporting period (the most recent
                      week unavailable at the origin), an input-availability
                      check.
    nb_shared_v4_t<c> v4 with the thermal-suitability channel re-centred at
                      c degC (26..30; the tensor's own is 28), for the
                      sensitivity analysis; scripts/34 selects c per fold on
                      the validation year.
    gru_v2_{climatology,noclimate}
                      the same two controls on the MSE GRU, beside the
                      existing shuffle arm.

NB arms write two methods: `<arm>` (point = NB median, seven NB quantiles,
and the mean and dispersion as columns `nb_mu`, `nb_alpha`) and `<arm>_mean`
(point = NB mean). The mean link is log mu = log(1 + y_origin) + delta, so at
initialisation (delta = 0) mu = y_origin + 1, one case above persistence; see
src/models/negative_binomial.py.

Separate model per horizon for the non-shared arms, so each horizon's windows
are assigned to splits by their own target period and every horizon's test set
is exactly the weeks of the test year.

Output: results/benchmark/predictions/<arm>.parquet, per seed, val and test;
results/benchmark/compute/<arm>.csv, training time, epochs and parameters per
fit, peak GPU memory and inference time per forecast origin.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.evaluation.long_horizon import (  # noqa: E402
    BENCHMARK_DIR,
    HORIZONS,
    QUANTILES,
    build_benchmark_folds,
    cell_frame,
    load_pipeline,
    months_for,
    nb_quantiles,
    save_predictions,
    split_by_target,
)
from src.models.adaptive_graph import AdaptiveAdjacency, AdaptiveGraphGCNGRU  # noqa: E402
from src.models.climate_ablation import (  # noqa: E402
    climate_indices,
    climatology_climate,
    delay_climate,
    drop_climate,
    shuffle_split_climate,
)
from src.models.negative_binomial import NegBinGCNGRU, masked_negative_binomial_loss  # noqa: E402
from src.models.stgnn import GCNGRU  # noqa: E402

ARMS = {
    # arm: (variant, backbone, kind[, tensor control])
    "gru_v1": ("v1", "identity", "mse"),
    "gru_v2": ("v2", "identity", "mse"),
    "gru_v2_shuffled": ("v2", "identity", "shuffled"),
    "gru_v2_quantile": ("v2", "identity", "quantile"),
    "gcn_v2": ("v2", "contiguity", "mse"),
    "gru_v2_lw": ("v2", "identity", "mse_lw"),
    "gru_v2_quantile_lw": ("v2", "identity", "quantile_lw"),
    "adaptive_v2": ("v2", "identity", "adaptive"),
    "gru_v2_climatology": ("v2", "identity", "mse", "climatology"),
    "gru_v2_noclimate": ("v2", "identity", "mse", "noclimate"),
    "gru_v2_nb": ("v2", "identity", "nb"),
    "nb_shared_v2": ("v2", "identity", "nb_shared"),
    "nb_shared_v3": ("v3", "identity", "nb_shared"),
    "nb_shared_v4": ("v4", "identity", "nb_shared"),
    "nb_shared_v2_shuffled": ("v2", "identity", "nb_shared", "shuffled"),
    "nb_shared_v2_climatology": ("v2", "identity", "nb_shared", "climatology"),
    "nb_shared_v2_noclimate": ("v2", "identity", "nb_shared", "noclimate"),
    "nb_shared_v2_wxlag1": ("v2", "identity", "nb_shared", "wxlag1"),
    **{f"nb_shared_v4_t{c}": ("v4", "identity", "nb_shared", f"thermal{c}")
       for c in (26, 27, 29, 30)},
}

THERMAL_WIDTH_C = 4.0  # scripts/12.add_biological_features' sigma
COMPUTE_DIR = BENCHMARK_DIR / "compute"

# scripts/26's adaptive-graph recipe: embedding dim 8, graph parameters on their
# own learning rate (no decay), the multiplier chosen per fold on validation MAE.
GRAPH_LR_GRID = (1.0, 3.0, 10.0, 30.0)
EMBEDDING_DIM = 8

baseline = None  # scripts/16


def level_weight(anchor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """scripts/20.level_weights: log1p level at the origin, mean 1 over observed."""

    weight = anchor.unsqueeze(-1)
    mean = (weight * mask).sum() / mask.sum().clamp(min=1.0)
    return weight / mean.clamp(min=1e-6)


def custom_loss(prediction, target, mask, anchor, levels, weighted):
    """Masked pinball (levels given) or MSE (levels None), optionally level-weighted.

    prediction [b, n, k], target/mask [b, n, 1], anchor [b, n].
    """

    weight = level_weight(anchor, mask) if weighted else torch.ones_like(mask)
    effective = mask * weight
    error = target - prediction
    if levels is None:
        elementwise = error ** 2
        count = effective.sum()
    else:
        elementwise = torch.maximum(levels * error, (levels - 1.0) * error)
        count = effective.sum() * len(levels)
    return (elementwise * effective).sum() / count.clamp(min=1.0)


def train_custom(arrays, adjacency, config, seed, device, quantile, weighted):
    """scripts/16.train_one with the loss swapped: pinball and/or level weighting.

    Same seeding, Adam via scripts/16.build_optimiser, batching, gradient
    clipping, early stopping on the validation loss and best-state restore.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    def to_tensor(split, key):
        return torch.from_numpy(arrays[split][key]).to(device)

    x_train, x_val = to_tensor("train", "X"), to_tensor("val", "X")
    mask_train, mask_val = to_tensor("train", "mask"), to_tensor("val", "mask")
    anchor_train, anchor_val = to_tensor("train", "anchor"), to_tensor("val", "anchor")
    y_train = baseline.to_target(to_tensor("train", "y"), anchor_train, "residual")
    y_val = baseline.to_target(to_tensor("val", "y"), anchor_val, "residual")
    adjacency_tensor = torch.from_numpy(adjacency).to(device)
    levels = torch.tensor(QUANTILES, dtype=torch.float32, device=device) if quantile else None

    model = GCNGRU(
        n_features=x_train.shape[-1],
        hidden=config["hidden"],
        gcn_layers=config["gcn_layers"],
        horizon=len(QUANTILES) if quantile else 1,
        dropout=config["dropout"],
    ).to(device)
    optimiser = baseline.build_optimiser(model, config)

    best_loss, best_epoch, waited = float("inf"), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    n_train = len(x_train)
    generator = torch.Generator().manual_seed(seed)

    epoch = 0
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order = torch.randperm(n_train, generator=generator).to(device)
        for start in range(0, n_train, config["batch_size"]):
            batch = order[start : start + config["batch_size"]]
            optimiser.zero_grad()
            loss = custom_loss(model(x_train[batch], adjacency_tensor), y_train[batch],
                               mask_train[batch].unsqueeze(-1), anchor_train[batch],
                               levels, weighted)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        with torch.no_grad():
            validation = custom_loss(model(x_val, adjacency_tensor), y_val,
                                     mask_val.unsqueeze(-1), anchor_val, levels,
                                     weighted).item()
        if validation < best_loss - 1e-6:
            best_loss, best_epoch, waited = validation, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= config["patience"]:
                break

    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "epochs_run": epoch, "val_loss": best_loss}


def adaptive_builder(config, n_nodes, seed):
    def build(n_features):
        backbone = GCNGRU(n_features=n_features, hidden=config["hidden"],
                          gcn_layers=config["gcn_layers"], horizon=1,
                          dropout=config["dropout"])
        return AdaptiveGraphGCNGRU(backbone, AdaptiveAdjacency(
            n_nodes=n_nodes, embedding_dim=EMBEDDING_DIM, seed=seed))
    return build


def adaptive_optimiser(model, config):
    """scripts/26.make_optimiser: the learned graph on its own rate, no decay."""

    graph_ids = {id(p) for p in model.adjacency.parameters()}
    rest = [p for p in model.parameters() if id(p) not in graph_ids]
    return torch.optim.Adam(
        [{"params": rest},
         {"params": list(model.adjacency.parameters()),
          "lr": config["learning_rate"] * config["graph_lr_multiplier"], "weight_decay": 0.0}],
        lr=config["learning_rate"], weight_decay=config["weight_decay"])


def train_adaptive(arrays, graph, config, seed, device, multiplier):
    candidate = dict(config, graph_lr_multiplier=multiplier)
    return baseline.train_one(arrays, graph, candidate, seed, device,
                              build_model=adaptive_builder(candidate, graph.shape[0], seed),
                              make_optimiser=adaptive_optimiser)


def select_multiplier(arrays, graph, config, device) -> float:
    """scripts/26.select_graph_lr: one seed per candidate, lowest validation MAE."""

    scores = []
    for multiplier in GRAPH_LR_GRID:
        model, _ = train_adaptive(arrays, graph, config, 0, device, multiplier)
        prediction = baseline.predict(model, arrays["val"], graph, "residual", device)
        mask = arrays["val"]["mask"] == 1
        scores.append((float(np.abs(prediction - arrays["val"]["y"])[mask].mean()), multiplier))
    return min(scores)[1]


def predict_quantiles(model, split, adjacency, device) -> np.ndarray:
    """Case-scale quantiles [windows, nodes, q], sorted to remove crossing."""

    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(split["X"]).to(device),
                       torch.from_numpy(adjacency).to(device)).cpu().numpy()
    output = np.sort(output, axis=-1) + split["anchor"][..., None]
    return np.clip(np.expm1(output), 0.0, None)


# ---------------------------------------------------------------------------
# Negative Binomial arms
# ---------------------------------------------------------------------------

def train_nb(arrays, adjacency, config, seed, device, n_heads):
    """Masked NB2 likelihood, otherwise scripts/16.train_one's loop.

    Same seeding, Adam via scripts/16.build_optimiser, batching, gradient
    clipping, early stopping on the validation loss (here the NB negative
    log-likelihood over the validation split's own cells) and best-state
    restore. `arrays[split]` holds y and mask as [windows, nodes, n_heads].
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    def to_tensor(split, key):
        return torch.from_numpy(arrays[split][key]).to(device)

    x_train, x_val = to_tensor("train", "X"), to_tensor("val", "X")
    y_train, y_val = to_tensor("train", "y"), to_tensor("val", "y")
    mask_train, mask_val = to_tensor("train", "mask"), to_tensor("val", "mask")
    anchor_train, anchor_val = to_tensor("train", "anchor"), to_tensor("val", "anchor")
    adjacency_tensor = torch.from_numpy(adjacency).to(device)

    backbone = GCNGRU(n_features=x_train.shape[-1], hidden=config["hidden"],
                      gcn_layers=config["gcn_layers"], horizon=n_heads,
                      dropout=config["dropout"])
    model = NegBinGCNGRU(backbone=backbone, horizon=n_heads).to(device)
    optimiser = baseline.build_optimiser(model, config)

    best_loss, best_epoch, waited = float("inf"), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    n_train = len(x_train)
    generator = torch.Generator().manual_seed(seed)

    epoch = 0
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order = torch.randperm(n_train, generator=generator).to(device)
        for start in range(0, n_train, config["batch_size"]):
            batch = order[start : start + config["batch_size"]]
            optimiser.zero_grad()
            mu, alpha = model(x_train[batch], adjacency_tensor, anchor_train[batch])
            loss = masked_negative_binomial_loss(mu, alpha, y_train[batch], mask_train[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        with torch.no_grad():
            mu, alpha = model(x_val, adjacency_tensor, anchor_val)
            validation = masked_negative_binomial_loss(mu, alpha, y_val, mask_val).item()
        if validation < best_loss - 1e-6:
            best_loss, best_epoch, waited = validation, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= config["patience"]:
                break

    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "epochs_run": epoch, "val_loss": best_loss}


def predict_nb(model, split, adjacency, device) -> tuple[np.ndarray, np.ndarray]:
    """NB mean and dispersion, [windows, nodes, heads]."""

    model.eval()
    with torch.no_grad():
        mu, alpha = model(torch.from_numpy(split["X"]).to(device),
                          torch.from_numpy(adjacency).to(device),
                          torch.from_numpy(split["anchor"]).to(device))
    return mu.cpu().numpy(), alpha.cpu().numpy()


def nb_frames(arm, fold, split, horizon, target_period_id, mu, alpha, seed):
    """The two NB methods for one set of cells: median (+ quantiles) and mean."""

    quantiles = nb_quantiles(mu, alpha)
    median = quantiles[..., QUANTILES.index(0.5)]
    extra = {"nb_mu": mu, "nb_alpha": alpha}
    return [cell_frame(arm, fold, split, horizon, target_period_id, median, seed, quantiles, extra),
            cell_frame(arm + "_mean", fold, split, horizon, target_period_id, mu, seed)]


def build_shared_arrays(tensors, months, fold, lookback, horizons):
    """Per-fold arrays for one trunk predicting every horizon from one origin.

    Imputation and scaling exactly as scripts/16.build_fold_arrays. Every
    origin with at least one target in a split enters that split's array, but
    only the (origin, horizon) cells whose own target lies in the split are
    unmasked: split_by_target, the rule every separate-horizon arm follows.
    """

    folds_module = baseline.folds_module
    period_id = tensors["period_id"]
    statistics = folds_module.fit_fold_statistics(tensors, months, period_id <= fold["fit_end_period"])
    x = folds_module.transform(tensors, months, statistics)

    n_periods = len(period_id)
    origins = np.arange(lookback - 1, n_periods - 1)
    windows_x = x[origins[:, None] + np.arange(-lookback + 1, 1)[None, :]]

    targets = origins[:, None] + np.asarray(horizons)[None, :]
    inside = targets < n_periods
    clipped = np.minimum(targets, n_periods - 1)
    y = np.nan_to_num(tensors["y"][clipped], nan=0.0).transpose(0, 2, 1)       # [o, n, H]
    observed = (tensors["y_mask"][clipped] == 1) & inside[:, :, None]            # [o, H, n]
    target_period = np.where(inside, period_id[clipped], -1)                     # [o, H]
    labels = split_by_target(target_period, fold)

    anchor = baseline.build_anchor(tensors, {"origin_period_id": period_id[origins]}, fold)

    arrays = {}
    bounds = {"train": (-np.inf, fold["train_end_period"]),
              "val": (fold["val_start_period"], fold["val_end_period"]),
              "test": (fold["test_start_period"], fold["test_end_period"])}
    for split in ("train", "val", "test"):
        cells = labels == split                                                   # [o, H]
        keep = cells.any(axis=1)
        mask = (observed & cells[:, :, None]).transpose(0, 2, 1)                  # [o, n, H]
        # The purge, asserted: no unmasked cell has a target outside its split.
        live = target_period[mask.any(axis=1)]
        low, high = bounds[split]
        if live.size and (live.min() < low or live.max() > high):
            raise AssertionError(f"{split} cells with targets outside the split in fold {fold['fold_id']}")
        arrays[split] = {
            "X": windows_x[keep].astype(np.float32),
            "y": y[keep].astype(np.float32),
            "mask": mask[keep].astype(np.float32),
            "anchor": anchor[keep],
            "cells": cells[keep],
            "target_period_id": target_period[keep],
        }
    return arrays


def apply_control(tensors, control, fold):
    """Tensor-level climate controls, applied before per-fold imputation and scaling."""

    if control in (None, "shuffled"):
        return tensors
    names = [str(n) for n in tensors["feature_names"]]
    climate = climate_indices(names)
    if control == "climatology":
        return climatology_climate(tensors, climate, fold["train_end_period"])
    if control == "noclimate":
        return drop_climate(tensors, climate)
    if control == "wxlag1":
        return delay_climate(tensors, climate, 1)
    if control.startswith("thermal"):
        centre = float(control[len("thermal"):])
        out = dict(tensors)
        x = tensors["X"].copy()
        temperature = x[..., names.index("temperature_mean_c")]
        x[..., names.index("thermal_suitability")] = np.exp(
            -0.5 * ((temperature - centre) / THERMAL_WIDTH_C) ** 2)
        out["X"] = x
        return out
    raise ValueError(control)


def shuffle_shared(arrays, climate, seed):
    """shuffle_split_climate for shared arrays: each split among its own windows."""

    return shuffle_split_climate(arrays, climate, seed)


def run_arm(arm, folds, horizons, seeds, device, lookback) -> tuple[list[pd.DataFrame], list[dict]]:
    variant, backbone, kind, *rest = ARMS[arm]
    control = rest[0] if rest else ("shuffled" if kind == "shuffled" else None)
    folds_module = baseline.folds_module
    base_tensors = folds_module.load_tensors(variant)
    months = months_for(folds_module)

    adjacency = np.load(PROJECT_DIR / "data" / "processed" / "adjacency.npz",
                        allow_pickle=True)["A_norm"].astype(np.float32)
    graph = np.eye(len(adjacency), dtype=np.float32) if backbone == "identity" else adjacency

    config = dict(baseline.DEFAULTS)
    config.update({"lookback": lookback, "horizon": 1, "target": "residual"})

    median_index = QUANTILES.index(0.5)
    frames: list[pd.DataFrame] = []
    compute: list[dict] = []

    def log_fit(fold, horizon, seed, model, info, seconds, n_origins):
        compute.append({"arm": arm, "fold_id": fold["fold_id"], "horizon": horizon, "seed": seed,
                        "train_seconds": seconds, "epochs_run": info["epochs_run"],
                        "best_epoch": info["best_epoch"], "train_windows": n_origins,
                        "parameters": sum(p.numel() for p in model.parameters())})

    def timed_inference(fn):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return result, time.perf_counter() - started

    for fold in folds:
        started = time.perf_counter()
        tensors = apply_control(base_tensors, control, fold)
        climate = climate_indices([str(name) for name in tensors["feature_names"]])
        line = []

        if kind == "nb_shared":
            arrays = build_shared_arrays(tensors, months, fold, lookback, horizons)
            maes = {h: [] for h in horizons}
            for seed in range(seeds):
                seed_arrays = shuffle_shared(arrays, climate, seed) if control == "shuffled" else arrays
                fit_started = time.perf_counter()
                model, info = train_nb(seed_arrays, graph, config, seed, device, len(horizons))
                log_fit(fold, 0, seed, model, info, time.perf_counter() - fit_started,
                        len(seed_arrays["train"]["X"]))
                for split in ("val", "test"):
                    data = seed_arrays[split]
                    (mu, alpha), seconds = timed_inference(lambda: predict_nb(model, data, graph, device))
                    compute[-1][f"{split}_inference_seconds"] = seconds
                    compute[-1][f"{split}_origins"] = len(data["X"])
                    for index, horizon in enumerate(horizons):
                        rows = data["cells"][:, index]
                        if not rows.any():
                            continue
                        frames.extend(nb_frames(arm, fold, split, horizon,
                                                data["target_period_id"][rows, index],
                                                mu[rows, :, index], alpha[rows, :, index], seed))
                        if split == "test":
                            observed = data["mask"][rows, :, index] == 1
                            median = nb_quantiles(mu[rows, :, index], alpha[rows, :, index])[..., median_index]
                            maes[horizon].append(float(np.abs(median - data["y"][rows, :, index])[observed].mean()))
            line = [f"h{h}:{np.mean(v):6.2f}" for h, v in maes.items() if v]
        else:
            for horizon in horizons:
                arrays = baseline.build_fold_arrays(tensors, months, fold, lookback, horizon)
                if kind == "nb":
                    for split in arrays.values():
                        split["y"] = split["y"][..., None]
                        split["mask"] = split["mask"][..., None]
                multiplier = select_multiplier(arrays, graph, config, device) if kind == "adaptive" else None
                maes = []
                for seed in range(seeds):
                    seed_arrays = (
                        shuffle_split_climate(arrays, climate, seed) if control == "shuffled" else arrays
                    )
                    fit_started = time.perf_counter()
                    if kind == "adaptive":
                        model, info = train_adaptive(seed_arrays, graph, config, seed, device, multiplier)
                    elif kind == "nb":
                        model, info = train_nb(seed_arrays, graph, config, seed, device, 1)
                    elif kind in ("quantile", "quantile_lw", "mse_lw"):
                        model, info = train_custom(seed_arrays, graph, config, seed, device,
                                                   quantile=kind.startswith("quantile"),
                                                   weighted=kind.endswith("_lw"))
                    else:
                        model, info = baseline.train_one(seed_arrays, graph, config, seed, device)
                    log_fit(fold, horizon, seed, model, info, time.perf_counter() - fit_started,
                            len(seed_arrays["train"]["X"]))

                    for split in ("val", "test"):
                        data = seed_arrays[split]
                        if kind == "nb":
                            (mu, alpha), seconds = timed_inference(
                                lambda: predict_nb(model, data, graph, device))
                            mu, alpha = mu[..., 0], alpha[..., 0]
                            frames.extend(nb_frames(arm, fold, split, horizon,
                                                    data["target_period_id"], mu, alpha, seed))
                            point = nb_quantiles(mu, alpha)[..., median_index]
                            target = data["y"][..., 0]
                            mask = data["mask"][..., 0] == 1
                        else:
                            if kind.startswith("quantile"):
                                quantiles, seconds = timed_inference(
                                    lambda: predict_quantiles(model, data, graph, device))
                                point = quantiles[..., median_index]
                            else:
                                quantiles = None
                                point, seconds = timed_inference(
                                    lambda: baseline.predict(model, data, graph, "residual", device))
                            frames.append(cell_frame(arm, fold, split, horizon,
                                                     data["target_period_id"], point, seed, quantiles))
                            target, mask = data["y"], data["mask"] == 1
                        compute[-1][f"{split}_inference_seconds"] = seconds
                        compute[-1][f"{split}_origins"] = len(data["X"])
                        if split == "test":
                            maes.append(float(np.abs(point - target)[mask].mean()))
                line.append(f"h{horizon}:{np.mean(maes):6.2f}")
        print(f"  {arm:<16} fold {fold['fold_id']} ({fold['test_year']}) "
              f"{' '.join(line)}  {time.perf_counter() - started:5.1f}s", flush=True)
    return frames, compute


def main() -> int:
    global baseline
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    parser.add_argument("--no-holdout", action="store_true",
                        help="Leave out fold 10 (2026 hold-out).")
    arguments = parser.parse_args()

    if arguments.threads:
        torch.set_num_threads(arguments.threads)
    baseline = load_pipeline()
    device = torch.device(arguments.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    folds = build_benchmark_folds(baseline.folds_module, include_holdout=not arguments.no_holdout)
    if arguments.folds:
        folds = [f for f in folds if f["fold_id"] in arguments.folds]

    print(f"device {device}, folds {[f['fold_id'] for f in folds]}, "
          f"horizons {arguments.horizons}, seeds {arguments.seeds}", flush=True)
    COMPUTE_DIR.mkdir(parents=True, exist_ok=True)
    for arm in arguments.arms:
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        frames, compute = run_arm(arm, folds, tuple(arguments.horizons), arguments.seeds,
                                  device, arguments.lookback)
        path = save_predictions(frames, arm + arguments.suffix)
        compute = pd.DataFrame(compute)
        compute["device"] = str(device)
        compute["hardware"] = (torch.cuda.get_device_name(0) if device.type == "cuda"
                               else f"CPU, {torch.get_num_threads()} threads")
        compute["peak_gpu_mb"] = (torch.cuda.max_memory_allocated() / 2**20
                                  if device.type == "cuda" else np.nan)
        compute["arm_wall_seconds"] = time.perf_counter() - started
        compute.to_csv(COMPUTE_DIR / f"{arm}{arguments.suffix}.csv", index=False)
        print(f"{arm}: wrote {path.relative_to(PROJECT_DIR)} "
              f"in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

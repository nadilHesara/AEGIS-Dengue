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

Separate model per horizon (README §8f: separate ~ shared on this backbone),
so each horizon's windows are assigned to splits by their own target period and
every horizon's test set is exactly the weeks of the test year.

Output: results/benchmark/predictions/<arm>.parquet, per seed, val and test.
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
    HORIZONS,
    QUANTILES,
    build_benchmark_folds,
    cell_frame,
    load_pipeline,
    months_for,
    save_predictions,
)
from src.models.adaptive_graph import AdaptiveAdjacency, AdaptiveGraphGCNGRU  # noqa: E402
from src.models.climate_ablation import climate_indices, shuffle_split_climate  # noqa: E402
from src.models.stgnn import GCNGRU  # noqa: E402

ARMS = {
    # arm: (variant, backbone, kind)
    "gru_v1": ("v1", "identity", "mse"),
    "gru_v2": ("v2", "identity", "mse"),
    "gru_v2_shuffled": ("v2", "identity", "shuffled"),
    "gru_v2_quantile": ("v2", "identity", "quantile"),
    "gcn_v2": ("v2", "contiguity", "mse"),
    "gru_v2_lw": ("v2", "identity", "mse_lw"),
    "gru_v2_quantile_lw": ("v2", "identity", "quantile_lw"),
    "adaptive_v2": ("v2", "identity", "adaptive"),
}

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


def run_arm(arm, folds, horizons, seeds, device, lookback) -> list[pd.DataFrame]:
    variant, backbone, kind = ARMS[arm]
    folds_module = baseline.folds_module
    tensors = folds_module.load_tensors(variant)
    months = months_for(folds_module)

    adjacency = np.load(PROJECT_DIR / "data" / "processed" / "adjacency.npz",
                        allow_pickle=True)["A_norm"].astype(np.float32)
    graph = np.eye(len(adjacency), dtype=np.float32) if backbone == "identity" else adjacency

    config = dict(baseline.DEFAULTS)
    config.update({"lookback": lookback, "horizon": 1, "target": "residual"})

    climate = climate_indices([str(name) for name in tensors["feature_names"]])
    median_index = QUANTILES.index(0.5)

    frames: list[pd.DataFrame] = []
    for fold in folds:
        started = time.perf_counter()
        line = []
        for horizon in horizons:
            arrays = baseline.build_fold_arrays(tensors, months, fold, lookback, horizon)
            multiplier = select_multiplier(arrays, graph, config, device) if kind == "adaptive" else None
            maes = []
            for seed in range(seeds):
                seed_arrays = (
                    shuffle_split_climate(arrays, climate, seed) if kind == "shuffled" else arrays
                )
                if kind == "adaptive":
                    model, info = train_adaptive(seed_arrays, graph, config, seed, device, multiplier)
                elif kind in ("quantile", "quantile_lw", "mse_lw"):
                    model, info = train_custom(seed_arrays, graph, config, seed, device,
                                               quantile=kind.startswith("quantile"),
                                               weighted=kind.endswith("_lw"))
                else:
                    model, info = baseline.train_one(seed_arrays, graph, config, seed, device)

                for split in ("val", "test"):
                    data = seed_arrays[split]
                    if kind.startswith("quantile"):
                        quantiles = predict_quantiles(model, data, graph, device)
                        point = quantiles[..., median_index]
                    else:
                        quantiles = None
                        point = baseline.predict(model, data, graph, "residual", device)
                    frames.append(cell_frame(arm, fold, split, horizon,
                                             data["target_period_id"], point, seed, quantiles))
                    if split == "test":
                        mask = data["mask"] == 1
                        maes.append(float(np.abs(point - data["y"])[mask].mean()))
            line.append(f"h{horizon}:{np.mean(maes):6.2f}")
        print(f"  {arm:<16} fold {fold['fold_id']} ({fold['test_year']}) "
              f"{' '.join(line)}  {time.perf_counter() - started:5.1f}s", flush=True)
    return frames


def main() -> int:
    global baseline
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--horizons", nargs="*", type=int, default=list(HORIZONS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--suffix", default="")
    arguments = parser.parse_args()

    baseline = load_pipeline()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = build_benchmark_folds(baseline.folds_module)
    if arguments.folds:
        folds = [f for f in folds if f["fold_id"] in arguments.folds]

    print(f"device {device}, folds {[f['fold_id'] for f in folds]}, "
          f"horizons {arguments.horizons}, seeds {arguments.seeds}", flush=True)
    for arm in arguments.arms:
        started = time.perf_counter()
        frames = run_arm(arm, folds, tuple(arguments.horizons), arguments.seeds,
                         device, arguments.lookback)
        path = save_predictions(frames, arm + arguments.suffix)
        print(f"{arm}: wrote {path.relative_to(PROJECT_DIR)} "
              f"in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

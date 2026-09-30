"""
Training and prediction for the climate-horizon model (plan §6.3).

Follows `scripts/training/31.long_horizon_neural.train_nb` line for line:
seeded, Adam, batch 64, gradient-norm clip 1.0, early stopping (patience 15)
on the **unweighted** validation NB NLL, best-state restore. Horizon weights
act only through the climate-only loss (src/training/climate_weighting.py),
so every arm stops on the same objective.

Optimiser groups: the delay encoder takes `kernel_learning_rate` with no
weight decay (scripts/16.parameter_groups' rule). The rest of the climate
branch and every other parameter take the base settings.

Checkpoints (`save_checkpoint`, written after every epoch when a path is
given) hold the config, model dimensions, seed, utility weights, model and
optimiser state, early-stopping state, the RNG states (torch, CUDA, numpy,
batch-order generator) and caller metadata (feature schema, scaler
statistics, horizon order, delay settings, fingerprints). `resume=True`
continues training from the file.
"""

from __future__ import annotations

import copy
import pathlib
import time

import numpy as np
import torch

from src.data.climate_horizon import HORIZONS
from src.models.climate_horizon import (
    CaseOnlyNB, ClimateHorizonNB, case_only_hidden, nb_loss_by_horizon, nb_point_forecast,
)
from src.training import climate_weighting

DEFAULTS = {
    "hidden": 32, "gcn_layers": 2, "climate_hidden": 16, "dropout": 0.2,
    "learning_rate": 3e-3, "kernel_learning_rate": 2e-2, "weight_decay": 1e-4,
    "batch_size": 64, "max_epochs": 150, "patience": 15,
    # target-relative lags for both main arms; "origin" is the ablation,
    # common_support=True the sensitivity check.
    "lag_mode": "target", "common_support": False, "max_delay": None,
}

UNIFORM_WEIGHTS = (1.0, 1.0, 1.0, 1.0)


def model_dims(arrays: dict) -> dict:
    part = arrays["train"]
    return {"n_nodes": int(part["X_case"].shape[2]), "n_case": int(part["X_case"].shape[-1]),
            "n_climate": int(part["X_climate"].shape[-1]), "n_horizons": int(part["y"].shape[-1]),
            "lookback": int(part["X_case"].shape[1]),
            "lag_reach": int(part["X_climate"].shape[1] - part["X_case"].shape[1] + 1)}


def build_model(arrays_or_dims: dict, config: dict):
    """Build from arrays (training) or from saved `model_dims` (checkpoint reload)."""

    dims = arrays_or_dims if "n_nodes" in arrays_or_dims else model_dims(arrays_or_dims)
    if config.get("case_only"):
        target = sum(p.numel() for p in build_model(dims, {**config, "case_only": False}).parameters())
        hidden = case_only_hidden(target, dims["n_nodes"], dims["n_case"], dims["n_horizons"],
                                  config["gcn_layers"])
        return CaseOnlyNB(dims["n_nodes"], dims["n_case"], dims["n_horizons"], hidden=hidden,
                          gcn_layers=config["gcn_layers"], dropout=config["dropout"])
    return ClimateHorizonNB(
        n_nodes=dims["n_nodes"], n_case=dims["n_case"], n_climate=dims["n_climate"],
        n_horizons=dims["n_horizons"], hidden=config["hidden"], gcn_layers=config["gcn_layers"],
        climate_hidden=config["climate_hidden"], dropout=config["dropout"], lag_reach=dims["lag_reach"],
        lag_mode=config.get("lag_mode", "target"), common_support=config.get("common_support", False),
        max_delay=config.get("max_delay"),
    )


def build_optimiser(model, config: dict) -> torch.optim.Optimizer:
    encoder_module = getattr(model, "encoder", None)
    encoder = {id(p) for p in encoder_module.parameters()} if encoder_module is not None else set()
    groups = [{"params": [p for p in model.parameters() if id(p) not in encoder]}]
    if encoder:
        groups.append({"params": [p for p in model.parameters() if id(p) in encoder],
                       "lr": config["kernel_learning_rate"], "weight_decay": 0.0})
    return torch.optim.Adam(groups, lr=config["learning_rate"], weight_decay=config["weight_decay"])


def _tensors(part: dict, device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(part[k]).to(device)
            for k in ("X_case", "X_climate", "anchor", "y", "mask")}


def _rng_state(generator: torch.Generator) -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "generator": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(state: dict, generator: torch.Generator) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    generator.set_state(state["generator"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train(arrays: dict, config: dict, seed: int, device, weights=UNIFORM_WEIGHTS,
          max_epochs: int | None = None, cosine_every: int = 20,
          checkpoint_path=None, resume: bool = False, metadata: dict | None = None):
    """Train one model. `weights` are climate-only loss weights.

    With `checkpoint_path`, the full training state is saved after every
    epoch. With `resume=True`, training continues from that file up to
    `max_epochs`; on the same device the result equals an uninterrupted run.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_t, val_t = _tensors(arrays["train"], device), _tensors(arrays["val"], device)
    dims = model_dims(arrays)
    weight_list = [float(w) for w in weights]
    loss_weights = torch.tensor(weight_list, dtype=torch.float32, device=device)
    if loss_weights.shape != (dims["n_horizons"],) or (loss_weights < 0).any():
        raise ValueError("weights must be non-negative, one per horizon")

    model = build_model(dims, config).to(device)
    optimiser = build_optimiser(model, config)
    gradnorm = None
    if config.get("gradnorm"):
        if resume:
            raise ValueError("the GradNorm control is not resumable")
        from src.training.climate_gradnorm import ClimateGradNorm
        gradnorm = ClimateGradNorm(dims["n_horizons"], device)
    generator = torch.Generator().manual_seed(seed)
    state = {"epoch": 0, "best_loss": float("inf"), "best_epoch": 0, "waited": 0,
             "best_state": copy.deepcopy(model.state_dict()), "gradient_log": [],
             "global_step": 0, "train_seconds": 0.0, "stopped": False}

    if resume:
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved["seed"] != seed or saved["config"] != config or saved["weights"] != weight_list:
            raise ValueError("checkpoint was written for a different seed, config or weights")
        model.load_state_dict(saved["model_state"])
        optimiser.load_state_dict(saved["optimiser_state"])
        state = saved["training_state"]
        _set_rng_state(saved["rng_state"], generator)

    n_train = len(train_t["y"])
    epochs = max_epochs or config["max_epochs"]
    started = time.perf_counter()

    for epoch in range(state["epoch"] + 1, epochs + 1):
        if state["stopped"]:
            break
        model.train()
        order = torch.randperm(n_train, generator=generator).to(device)
        for start in range(0, n_train, config["batch_size"]):
            b = order[start:start + config["batch_size"]]
            batch = {k: v[b] for k, v in train_t.items()}
            step_weights = gradnorm.weights() if gradnorm is not None else loss_weights
            log = climate_weighting.step(
                model, optimiser, batch, step_weights, clip=1.0,
                log_cosine=cosine_every > 0 and state["global_step"] % cosine_every == 0, hook=gradnorm)
            state["gradient_log"].append({"epoch": epoch, "step": state["global_step"], **log})
            state["global_step"] += 1

        model.eval()
        with torch.no_grad():
            out = model(val_t["X_case"], val_t["X_climate"], val_t["anchor"])
            validation = nb_loss_by_horizon(out["mu"], out["alpha"], val_t["y"], val_t["mask"]).sum().item()
        state["epoch"] = epoch
        if validation < state["best_loss"] - 1e-6:
            state.update(best_loss=validation, best_epoch=epoch, waited=0,
                         best_state=copy.deepcopy(model.state_dict()))
        else:
            state["waited"] += 1
            if state["waited"] >= config["patience"]:
                state["stopped"] = True
        if checkpoint_path is not None:
            state["train_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            save_checkpoint(checkpoint_path, model, optimiser, config, dims, seed, weight_list, state,
                            _rng_state(generator), metadata or {})
        if state["stopped"]:
            break

    state["train_seconds"] += time.perf_counter() - started
    model.load_state_dict(state["best_state"])
    return model, {"best_epoch": state["best_epoch"], "epochs_run": state["epoch"],
                   "val_loss": state["best_loss"], "train_seconds": state["train_seconds"],
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "gradient_log": state["gradient_log"]}


def save_checkpoint(path, model, optimiser, config, dims, seed, weights, state, rng_state, metadata) -> None:
    """Atomic write of everything needed to reload predictions or resume training."""

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"format": "climate-horizon-checkpoint-v1", "config": config, "model_dims": dims,
                "seed": seed, "weights": list(weights),
                "model_state": model.state_dict(), "optimiser_state": optimiser.state_dict(),
                "training_state": state, "rng_state": rng_state, "metadata": metadata}, tmp)
    # Windows can briefly lock the target (indexer/antivirus): retry the atomic replace.
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25 * (attempt + 1))


def load_checkpoint(path, device, best: bool = True):
    """Rebuild the model from a checkpoint; `best=True` loads the early-stopping state."""

    saved = torch.load(path, map_location=device, weights_only=False)
    model = build_model(saved["model_dims"], saved["config"]).to(device)
    model.load_state_dict(saved["training_state"]["best_state"] if best else saved["model_state"])
    return model, saved


def checkpoint_metadata(channels, statistics: dict, horizons, fold: dict, weights, weights_source: str,
                        fingerprints: dict, delay_settings: dict) -> dict:
    """Feature schema, scalers, horizon order, delay settings, utility weights, fingerprints."""

    return {"feature_names": channels.names, "case_channels": channels.case_names(),
            "climate_channels": channels.climate_names(),
            "excluded_channels": [channels.names[i] for i in channels.excluded],
            "scaler_statistics": {k: np.asarray(v) for k, v in statistics.items()},
            "horizons": list(horizons), "delay_settings": delay_settings, "fold": fold,
            "utility_weights": [float(w) for w in weights], "weights_source": weights_source,
            "fingerprints": fingerprints}


def predict(model, part: dict, device, gate_off: bool = False,
            x_climate: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Count-scale NB median, NB mean and dispersion, gates and corrections [o, N, H].

    The point forecast is the NB median of (mu, alpha). mu is already a count
    mean, so no expm1 is applied.
    """

    model.eval()
    t = _tensors(part, device)
    if x_climate is not None:  # inference-time perturbation (pilot shuffle)
        t["X_climate"] = torch.from_numpy(x_climate).to(device)
    with torch.no_grad():
        out = model(t["X_case"], t["X_climate"], t["anchor"], gate_off=gate_off)
    result = {k: v.cpu().numpy() for k, v in out.items()}
    result["prediction"] = nb_point_forecast(result["mu"], result["alpha"])
    return result


def prediction_frame(method: str, fold_id: int, split: str, part: dict, result: dict,
                     seed: int, horizons=HORIZONS):
    """Long frame of observed cells with prediction, mu, alpha, gate and corrections."""

    import pandas as pd

    o, n, h = np.nonzero(part["mask"] == 1)
    frame = pd.DataFrame({
        "method": method, "fold_id": fold_id, "split": split,
        "horizon": np.asarray(horizons)[h], "seed": seed,
        "target_period_id": part["target_period_id"][o, h], "node_id": n,
        "prediction": result["prediction"][o, n, h],
    })
    for key in ("mu", "alpha", "gate", "delta_case", "delta_climate", "gated_climate"):
        frame[key] = result[key][o, n, h].astype(np.float32)
    return frame

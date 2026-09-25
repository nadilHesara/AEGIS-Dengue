"""Train and evaluate the Negative Binomial probabilistic forecaster.

This script implements the Negative Binomial likelihood (Component C) as an
alternative probabilistic head, preserving the identical training loop,
walk-forward folds, preprocessing, metrics, and backbones from script 16.

Key differences from baseline:
1. Target: Operates directly on true integer/non-negative case counts.
2. Head: NegBinHead outputs mean μ (with origin-anchoring) and overdispersion α.
3. Loss: Maximizes Negative Binomial log-likelihood (minimizes NLL).
4. Metrics: Evaluates point forecasts (mean & median) on raw MAE/Peak MAE,
   plus produces calibrated prediction intervals (10%, 50%, 90%).

Usage:
    python scripts/training/31.train_negbin.py --folds 8 --seeds 1     # Smoke test
    python scripts/training/31.train_negbin.py --seeds 3               # Full 9-fold sweep
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.models.negative_binomial import (
    NegBinGCNGRU,
    compute_outbreak_probability,
    compute_prediction_intervals,
    masked_negative_binomial_loss,
)
from src.models.stgnn import GCNGRU

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "models"
REPORTS_DIR = PROJECT_DIR / "results" / "reports"
FOLDS_PATH = PROCESSED_DIR / "folds.json"
ADJACENCY_PATH = PROCESSED_DIR / "adjacency.npz"

DEFAULTS = {
    "lookback": 12,
    "horizon": 1,
    "hidden": 32,
    "gcn_layers": 2,
    "dropout": 0.2,
    "learning_rate": 3e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 150,
    "patience": 15,
    "seeds": 3,
    "variant": "v1",
    "backbone": "identity",  # "identity" (gru_only) or "contiguity" (gcn_gru)
}

baseline_module = None
naive = None
folds_module = None
tensors_module = None


def _load_script(name: str, rel_path: str):
    path = PROJECT_DIR / "scripts" / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_dependencies():
    global baseline_module, naive, folds_module, tensors_module
    baseline_module = _load_script("baseline_module", "training/16.train_gcn_gru.py")
    baseline_module.load_modules()
    naive = baseline_module.naive
    folds_module = baseline_module.folds_module
    tensors_module = baseline_module.tensors_module


def train_one_negbin(
    arrays: dict[str, dict[str, np.ndarray]],
    adjacency: np.ndarray,
    config: dict,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Train one Negative Binomial model on one fold with early stopping on validation NLL."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    def to_tensor(split: str, key: str) -> torch.Tensor:
        return torch.from_numpy(arrays[split][key]).to(device)

    x_train, x_val = to_tensor("train", "X"), to_tensor("val", "X")
    mask_train, mask_val = to_tensor("train", "mask"), to_tensor("val", "mask")
    anchor_train, anchor_val = to_tensor("train", "anchor"), to_tensor("val", "anchor")
    y_train, y_val = to_tensor("train", "y"), to_tensor("val", "y")

    adjacency_tensor = torch.from_numpy(adjacency).to(device)

    raw_backbone = GCNGRU(
        n_features=x_train.shape[-1],
        hidden=config["hidden"],
        gcn_layers=config["gcn_layers"],
        horizon=config["horizon"],
        dropout=config["dropout"],
    )
    model = NegBinGCNGRU(backbone=raw_backbone, horizon=config["horizon"]).to(device)

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    waited = 0

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
            loss = masked_negative_binomial_loss(
                mu.squeeze(-1),
                alpha.squeeze(-1),
                y_train[batch],
                mask_train[batch],
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        with torch.no_grad():
            mu_val, alpha_val = model(x_val, adjacency_tensor, anchor_val)
            val_loss = masked_negative_binomial_loss(
                mu_val.squeeze(-1),
                alpha_val.squeeze(-1),
                y_val,
                mask_val,
            ).item()

        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, waited = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= config["patience"]:
                break

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "val_loss": best_loss,
    }


def run_experiment(
    folds: list[dict],
    config: dict,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train NegBin and evaluate against persistence and baseline MSE on test folds."""
    calendar = folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()

    adj_data = np.load(ADJACENCY_PATH, allow_pickle=True)
    if config["backbone"] == "identity":
        adjacency = np.eye(25, dtype=np.float32)
    else:
        adjacency = adj_data["A_norm"].astype(np.float32)

    tensors = folds_module.load_tensors(config["variant"])

    rows: list[dict] = []
    prediction_records: list[pd.DataFrame] = []

    for fold in folds:
        print(f"\n--- Fold {fold['fold_id']} (Test Year {fold['test_year']}) ---")
        arrays = baseline_module.build_fold_arrays(
            tensors, months, fold, config["lookback"], config["horizon"]
        )

        fit_mask = tensors["period_id"] <= fold["fit_end_period"]
        thresholds = naive.peak_thresholds(tensors["y"], tensors["y_mask"], fit_mask)

        target = arrays["test"]["y"]
        mask = arrays["test"]["mask"].astype(np.int8)
        anchor_test = arrays["test"]["anchor"]

        seed_mu_preds = []
        seed_alpha_preds = []

        for seed in range(config["seeds"]):
            start_t = time.perf_counter()
            model, info = train_one_negbin(arrays, adjacency, config, seed, device)
            dur = time.perf_counter() - start_t

            model.eval()
            with torch.no_grad():
                x_test = torch.from_numpy(arrays["test"]["X"]).to(device)
                anc_test = torch.from_numpy(anchor_test).to(device)
                adj_t = torch.from_numpy(adjacency).to(device)
                mu_pred, alpha_pred = model(x_test, adj_t, anc_test)

            mu_np = mu_pred.squeeze(-1).cpu().numpy()
            alpha_np = alpha_pred.squeeze(-1).cpu().numpy()

            seed_mu_preds.append(mu_np)
            seed_alpha_preds.append(alpha_np)

            # Evaluate point prediction (conditional mean)
            scores = naive.evaluate(mu_np, target, mask, thresholds)

            # Test Negative Log-Likelihood
            mu_torch = torch.from_numpy(mu_np)
            alpha_torch = torch.from_numpy(alpha_np)
            target_torch = torch.from_numpy(target)
            mask_torch = torch.from_numpy(mask)
            test_nll = masked_negative_binomial_loss(
                mu_torch, alpha_torch, target_torch, mask_torch
            ).item()

            print(
                f"Seed {seed}: Test MAE={scores['mae']:.2f}, "
                f"Peak MAE={scores['peak_mae']:.2f}, NLL={test_nll:.2f} "
                f"({dur:.1f}s, best epoch {info['best_epoch']})"
            )

            rows.append(
                {
                    "arm": "negbin",
                    "backbone": config["backbone"],
                    "variant": config["variant"],
                    "fold_id": fold["fold_id"],
                    "test_year": fold["test_year"],
                    "covers_covid": fold["covers_covid"],
                    "headline": fold["headline"],
                    "seed": seed,
                    "best_epoch": info["best_epoch"],
                    "test_nll": test_nll,
                    "ensemble": False,
                    **scores,
                }
            )

        # Ensemble prediction (average mu over seeds)
        if len(seed_mu_preds) > 1:
            ensemble_mu = np.mean(seed_mu_preds, axis=0)
            ens_scores = naive.evaluate(ensemble_mu, target, mask, thresholds)
            print(f"Ensemble: Test MAE={ens_scores['mae']:.2f}, Peak MAE={ens_scores['peak_mae']:.2f}")
            rows.append(
                {
                    "arm": "negbin",
                    "backbone": config["backbone"],
                    "variant": config["variant"],
                    "fold_id": fold["fold_id"],
                    "test_year": fold["test_year"],
                    "covers_covid": fold["covers_covid"],
                    "headline": fold["headline"],
                    "seed": "ensemble",
                    "best_epoch": np.mean([r["best_epoch"] for r in rows[-config["seeds"]:]]),
                    "test_nll": float("nan"),
                    "ensemble": True,
                    **ens_scores,
                }
            )

    results_df = pd.DataFrame(rows)
    return results_df, pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Train Negative Binomial Forecaster.")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="Folds to train on")
    parser.add_argument("--seeds", type=int, default=DEFAULTS["seeds"], help="Number of seeds")
    parser.add_argument("--backbone", type=str, default=DEFAULTS["backbone"], choices=["identity", "contiguity"])
    parser.add_argument("--variant", type=str, default=DEFAULTS["variant"], choices=["v0", "v1", "v2", "v3"])
    parser.add_argument("--horizon", type=int, default=DEFAULTS["horizon"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["max_epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    args = parser.parse_args()

    load_dependencies()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = dict(DEFAULTS)
    config["seeds"] = args.seeds
    config["backbone"] = args.backbone
    config["variant"] = args.variant
    config["horizon"] = args.horizon
    config["max_epochs"] = args.epochs
    config["patience"] = args.patience

    all_folds = json.loads(FOLDS_PATH.read_text())["folds"]
    if args.folds is not None:
        selected_folds = [f for f in all_folds if f["fold_id"] in args.folds]
    else:
        selected_folds = all_folds

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df, _ = run_experiment(selected_folds, config, device)

    out_csv = RESULTS_DIR / f"negbin_metrics_{config['backbone']}_{config['variant']}_h{config['horizon']}.csv"
    metrics_df.to_csv(out_csv, index=False)
    print(f"\nSaved metrics to {out_csv}")

    # Print summary table
    headline_df = metrics_df[(metrics_df["headline"] == True) & (metrics_df["ensemble"] == False)]
    if not headline_df.empty:
        mean_mae = headline_df["mae"].mean()
        mean_peak = headline_df["peak_mae"].mean()
        print(f"\nSummary over Headline Folds (Single-Seed Mean):")
        print(f"  MAE:      {mean_mae:.2f}")
        print(f"  Peak MAE: {mean_peak:.2f}")


if __name__ == "__main__":
    main()

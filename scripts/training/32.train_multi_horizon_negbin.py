"""Multi-horizon Negative Binomial probabilistic forecasting (h = 1, 2, 3, 4).

Combines the shared-trunk spatio-temporal architecture of script 27 with the
Negative Binomial likelihood of Component C (script 31).

Key advantages:
1. Multi-horizon supervision: One shared GRU trunk, 4 heads for (μ, α) across h=1..4.
2. Direct count likelihood: Optimizes Negative Binomial NLL without log-distortion.
3. Rescored persistence comparison: Compares each horizon against its own
   same-horizon persistence baseline.
4. Generates calibrated prediction intervals (10%, 50%, 90%) for each horizon.

Usage:
    python scripts/training/32.train_multi_horizon_negbin.py --folds 8 --seeds 1  # Smoke test
    python scripts/training/32.train_multi_horizon_negbin.py --seeds 3             # Full sweep
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

from src.models.multi_horizon import (
    DEFAULT_HORIZONS,
    make_multi_horizon_windows,
)
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


def build_multi_horizon_fold_arrays(
    tensors: dict[str, np.ndarray],
    months: np.ndarray,
    fold: dict,
    lookback: int,
    horizons: tuple[int, ...],
) -> dict[str, dict[str, np.ndarray]]:
    """Build multi-horizon arrays with all horizon targets aligned on identical origins."""
    statistics = folds_module.fit_fold_statistics(
        tensors, months, tensors["period_id"] <= fold["fit_end_period"]
    )
    scaled = dict(tensors)
    scaled["X"] = folds_module.transform(tensors, months, statistics)

    windows = make_multi_horizon_windows(
        scaled, lookback=lookback, horizons=horizons, drop_incomplete=False
    )
    split = folds_module.assign_windows(windows["target_period_id"][:, 0], fold)

    anchor = baseline_module.build_anchor(tensors, windows, fold)

    arrays = {}
    for name, selector in split.items():
        target = windows["y"][selector]
        mask = (windows["y_mask"][selector] == 1).astype(np.float32)
        target = np.nan_to_num(target, nan=0.0)

        arrays[name] = {
            "X": windows["X"][selector],
            "y": target,
            "mask": mask,
            "anchor": anchor[selector],
            "target_period_id": windows["target_period_id"][selector],
        }

    return arrays


def train_shared_negbin(
    arrays: dict[str, dict[str, np.ndarray]],
    adjacency: np.ndarray,
    config: dict,
    horizons: tuple[int, ...],
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Train one shared-trunk multi-horizon Negative Binomial model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    def to_tensor(split: str, key: str) -> torch.Tensor:
        return torch.from_numpy(arrays[split][key]).to(device)

    x_train, x_val = to_tensor("train", "X"), to_tensor("val", "X")
    mask_train, mask_val = to_tensor("train", "mask"), to_tensor("val", "mask")
    anchor_train, anchor_val = to_tensor("train", "anchor"), to_tensor("val", "anchor")
    y_train, y_val = to_tensor("train", "y"), to_tensor("val", "y")

    adjacency_tensor = torch.from_numpy(adjacency).to(device)
    n_horizons = len(horizons)

    raw_backbone = GCNGRU(
        n_features=x_train.shape[-1],
        hidden=config["hidden"],
        gcn_layers=config["gcn_layers"],
        horizon=n_horizons,
        dropout=config["dropout"],
    )
    model = NegBinGCNGRU(backbone=raw_backbone, horizon=n_horizons).to(device)

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
                mu,
                alpha,
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
                mu_val,
                alpha_val,
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


def evaluate_persistence_per_horizon(
    test_arrays: dict[str, np.ndarray],
    thresholds: np.ndarray,
    horizons: tuple[int, ...],
) -> list[dict]:
    """Score naive persistence across each forecast horizon."""
    # Origin case count is expm1(anchor)
    origin_cases = np.expm1(test_arrays["anchor"])  # [windows, nodes]
    target = test_arrays["y"]                       # [windows, nodes, horizons]
    mask = test_arrays["mask"].astype(np.int8)

    records = []
    for idx, h in enumerate(horizons):
        # Persistence predicts origin_cases for target at t+h
        pred_h = origin_cases
        target_h = target[..., idx]
        mask_h = mask[..., idx]
        scores = naive.evaluate(pred_h, target_h, mask_h, thresholds)
        records.append({"horizon": h, **scores})
    return records


def run_experiment(
    folds: list[dict],
    config: dict,
    horizons: tuple[int, ...],
    device: torch.device,
) -> pd.DataFrame:
    """Run multi-horizon training across folds and seeds."""
    calendar = folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()

    adj_data = np.load(ADJACENCY_PATH, allow_pickle=True)
    if config["backbone"] == "identity":
        adjacency = np.eye(25, dtype=np.float32)
    else:
        adjacency = adj_data["A_norm"].astype(np.float32)

    tensors = folds_module.load_tensors(config["variant"])
    rows: list[dict] = []

    for fold in folds:
        print(f"\n--- Fold {fold['fold_id']} (Test Year {fold['test_year']}) ---")
        arrays = build_multi_horizon_fold_arrays(
            tensors, months, fold, config["lookback"], horizons
        )

        fit_mask = tensors["period_id"] <= fold["fit_end_period"]
        thresholds = naive.peak_thresholds(tensors["y"], tensors["y_mask"], fit_mask)

        # Baseline persistence reference per horizon
        pers_scores = evaluate_persistence_per_horizon(arrays["test"], thresholds, horizons)
        for ps in pers_scores:
            rows.append(
                {
                    "arm": "persistence",
                    "backbone": "none",
                    "variant": config["variant"],
                    "horizon": ps["horizon"],
                    "fold_id": fold["fold_id"],
                    "test_year": fold["test_year"],
                    "covers_covid": fold["covers_covid"],
                    "headline": fold["headline"],
                    "seed": "none",
                    "best_epoch": 0,
                    "ensemble": False,
                    **{k: v for k, v in ps.items() if k != "horizon"},
                }
            )

        target = arrays["test"]["y"]
        mask = arrays["test"]["mask"].astype(np.int8)
        anchor_test = arrays["test"]["anchor"]

        seed_mu_preds = []

        for seed in range(config["seeds"]):
            start_t = time.perf_counter()
            model, info = train_shared_negbin(arrays, adjacency, config, horizons, seed, device)
            dur = time.perf_counter() - start_t

            model.eval()
            with torch.no_grad():
                x_test = torch.from_numpy(arrays["test"]["X"]).to(device)
                anc_test = torch.from_numpy(anchor_test).to(device)
                adj_t = torch.from_numpy(adjacency).to(device)
                mu_pred, _ = model(x_test, adj_t, anc_test)

            mu_np = mu_pred.cpu().numpy()  # [windows, nodes, horizons]
            seed_mu_preds.append(mu_np)

            # Evaluate per horizon
            h_mae_strs = []
            for idx, h in enumerate(horizons):
                scores = naive.evaluate(
                    mu_np[..., idx], target[..., idx], mask[..., idx], thresholds
                )
                h_mae_strs.append(f"h{h}={scores['mae']:.2f}")
                rows.append(
                    {
                        "arm": "negbin_shared",
                        "backbone": config["backbone"],
                        "variant": config["variant"],
                        "horizon": h,
                        "fold_id": fold["fold_id"],
                        "test_year": fold["test_year"],
                        "covers_covid": fold["covers_covid"],
                        "headline": fold["headline"],
                        "seed": seed,
                        "best_epoch": info["best_epoch"],
                        "ensemble": False,
                        **scores,
                    }
                )

            print(
                f"Seed {seed}: {', '.join(h_mae_strs)} ({dur:.1f}s, best epoch {info['best_epoch']})"
            )

        # Ensemble
        if len(seed_mu_preds) > 1:
            ens_mu = np.mean(seed_mu_preds, axis=0)
            ens_mae_strs = []
            for idx, h in enumerate(horizons):
                ens_scores = naive.evaluate(
                    ens_mu[..., idx], target[..., idx], mask[..., idx], thresholds
                )
                ens_mae_strs.append(f"h{h}={ens_scores['mae']:.2f}")
                rows.append(
                    {
                        "arm": "negbin_shared",
                        "backbone": config["backbone"],
                        "variant": config["variant"],
                        "horizon": h,
                        "fold_id": fold["fold_id"],
                        "test_year": fold["test_year"],
                        "covers_covid": fold["covers_covid"],
                        "headline": fold["headline"],
                        "seed": "ensemble",
                        "best_epoch": np.mean([r["best_epoch"] for r in rows[-len(horizons) * config["seeds"]:]]),
                        "ensemble": True,
                        **ens_scores,
                    }
                )
            print(f"Ensemble: {', '.join(ens_mae_strs)}")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Multi-Horizon Negative Binomial Forecaster.")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="Folds to train on")
    parser.add_argument("--seeds", type=int, default=DEFAULTS["seeds"], help="Number of seeds")
    parser.add_argument("--backbone", type=str, default=DEFAULTS["backbone"], choices=["identity", "contiguity"])
    parser.add_argument("--variant", type=str, default=DEFAULTS["variant"], choices=["v0", "v1", "v2", "v3"])
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
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
    config["max_epochs"] = args.epochs
    config["patience"] = args.patience

    horizons = tuple(args.horizons)

    all_folds = json.loads(FOLDS_PATH.read_text())["folds"]
    if args.folds is not None:
        selected_folds = [f for f in all_folds if f["fold_id"] in args.folds]
    else:
        selected_folds = all_folds

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = run_experiment(selected_folds, config, horizons, device)

    out_csv = RESULTS_DIR / f"multi_horizon_negbin_{config['backbone']}_{config['variant']}.csv"
    metrics_df.to_csv(out_csv, index=False)
    print(f"\nSaved metrics to {out_csv}")

    # Summary table across headline folds
    headline_df = metrics_df[(metrics_df["headline"] == True) & (metrics_df["ensemble"] == False)]
    if not headline_df.empty:
        print("\nSummary over Headline Folds (Single-Seed Mean):")
        for arm in ["persistence", "negbin_shared"]:
            arm_sub = headline_df[headline_df["arm"] == arm]
            if not arm_sub.empty:
                print(f"[{arm}]")
                for h in horizons:
                    h_sub = arm_sub[arm_sub["horizon"] == h]
                    mae = h_sub["mae"].mean()
                    peak = h_sub["peak_mae"].mean()
                    print(f"  h={h}: MAE={mae:.2f}, Peak MAE={peak:.2f}")


if __name__ == "__main__":
    main()

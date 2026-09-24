"""Generate publication figures for Negative Binomial probabilistic forecasts.

Plots observed case counts, predicted mean, and shaded 80% prediction intervals
(10th to 90th percentiles) for high-burden districts during the 2017 epidemic.

Output:
    figures/fig_probabilistic_forecast.pdf
    figures/fig_probabilistic_forecast.png
    images/fig_probabilistic_forecast.png

Usage:
    python scripts/evaluation/33.plot_probabilistic_forecasts.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.models.negative_binomial import (
    NegBinGCNGRU,
    compute_prediction_intervals,
)
from src.models.stgnn import GCNGRU

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FIGURES_DIR = PROJECT_DIR / "figures"
IMAGES_DIR = PROJECT_DIR / "images"
FOLDS_PATH = PROCESSED_DIR / "folds.json"
ADJACENCY_PATH = PROCESSED_DIR / "adjacency.npz"

COLUMN_WIDTH = 3.33  # Standard single-column ACM format


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.2,
            "figure.dpi": 400,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
        }
    )


def _load_script(name: str, rel_path: str):
    path = PROJECT_DIR / "scripts" / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    style()
    train_script = _load_script("train_negbin_script", "training/31.train_negbin.py")
    train_script.load_dependencies()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_folds = json.loads(FOLDS_PATH.read_text())["folds"]
    fold_1 = next(f for f in all_folds if f["fold_id"] == 1)

    calendar = train_script.folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()
    tensors = train_script.folds_module.load_tensors("v1")

    nodes = pd.read_csv(PROCESSED_DIR / "nodes.csv").sort_values("node_id")
    district_names = nodes["canonical_name"].tolist()

    colombo_id = district_names.index("Colombo") if "Colombo" in district_names else 0
    gampaha_id = district_names.index("Gampaha") if "Gampaha" in district_names else 1

    config = dict(train_script.DEFAULTS)
    config["max_epochs"] = 70
    config["patience"] = 15

    adjacency = np.eye(25, dtype=np.float32)

    print("Training NegBin model on Fold 1 for visualization...")
    arrays = train_script.baseline_module.build_fold_arrays(
        tensors, months, fold_1, config["lookback"], config["horizon"]
    )

    model, _ = train_script.train_one_negbin(arrays, adjacency, config, seed=0, device=device)

    model.eval()
    with torch.no_grad():
        x_test = torch.from_numpy(arrays["test"]["X"]).to(device)
        anc_test = torch.from_numpy(arrays["test"]["anchor"]).to(device)
        adj_t = torch.from_numpy(adjacency).to(device)
        mu_pred, alpha_pred = model(x_test, adj_t, anc_test)

    mu_np = mu_pred.squeeze(-1).cpu().numpy()        # [weeks, 25]
    alpha_np = alpha_pred.squeeze(-1).cpu().numpy()  # [weeks, 25]
    actual_y = arrays["test"]["y"]                   # [weeks, 25]
    persistence = np.expm1(arrays["test"]["anchor"]) # [weeks, 25]

    # Compute 10th and 90th percentiles for the 80% interval
    intervals = compute_prediction_intervals(mu_np, alpha_np, quantiles=(0.10, 0.90))
    lower_80 = intervals[0.10]
    upper_80 = intervals[0.90]

    weeks = np.arange(1, actual_y.shape[0] + 1)

    # Two-panel figure: Colombo (top) & Gampaha (bottom)
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH * 1.5, 3.6), sharex=True)

    districts_to_plot = [(colombo_id, "Colombo (Epicenter)", axes[0]),
                         (gampaha_id, "Gampaha (High-Burden)", axes[1])]

    for node_idx, title, ax in districts_to_plot:
        y_obs = actual_y[:, node_idx]
        y_hat = mu_np[:, node_idx]
        y_pers = persistence[:, node_idx]
        lo = lower_80[:, node_idx]
        hi = upper_80[:, node_idx]

        # 80% Prediction interval
        ax.fill_between(
            weeks, lo, hi, color="#1f4e79", alpha=0.18, label="80% Prediction Interval"
        )
        # Persistence baseline
        ax.plot(
            weeks, y_pers, color="#8c8c8c", linestyle="--", linewidth=1.0, label="Persistence"
        )
        # Predicted Mean
        ax.plot(
            weeks, y_hat, color="#1f4e79", linewidth=1.4, label="NegBin Predicted Mean (μ)"
        )
        # Observed cases
        ax.scatter(
            weeks, y_obs, color="#b71c1c", s=10, zorder=5, label="Observed Cases"
        )

        ax.set_title(f"{title} — 2017 National Epidemic", fontsize=8, pad=3)
        ax.set_ylabel("Weekly Cases")
        ax.grid(True, linestyle=":", alpha=0.4, linewidth=0.5)

    axes[1].set_xlabel("Epidemiological Week of 2017")
    axes[0].legend(loc="upper left", frameon=False, fontsize=6.5)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    pdf_path = FIGURES_DIR / "fig_probabilistic_forecast.pdf"
    png_path = FIGURES_DIR / "fig_probabilistic_forecast.png"
    img_png_path = IMAGES_DIR / "fig_probabilistic_forecast.png"

    fig.savefig(pdf_path)
    fig.savefig(png_path)
    fig.savefig(img_png_path)
    plt.close(fig)

    print(f"Saved figures to:\n  {pdf_path}\n  {png_path}\n  {img_png_path}")


if __name__ == "__main__":
    main()

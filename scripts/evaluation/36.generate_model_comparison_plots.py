"""Generate comparison plots (2017-2025) between:
1. Actual Observed Cases
2. gru_v2_quantile_lw (3-Seed Ensemble)
3. Multi-Horizon NegBin v4 (3-Seed Ensemble)

Generates 9-panel figures (one panel per year 2017 to 2025) for:
- h=1 (1 week ahead operational forecast)
- h=4 (4 weeks ahead outbreak early warning forecast)

Outputs:
    figures/fig_comparison_2017_2025_h1.png
    figures/fig_comparison_2017_2025_h1.pdf
    figures/fig_comparison_2017_2025_h4.png
    figures/fig_comparison_2017_2025_h4.pdf
    results/benchmark/tables/model_comparison_forecasts_2017_2025.csv
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Use 8 CPU threads for fast training
torch.set_num_threads(8)

from src.evaluation.long_horizon import (
    QUANTILES,
    build_benchmark_folds,
    load_pipeline,
    months_for,
)
from src.models.negative_binomial import (
    NegBinGCNGRU,
    compute_level_weights,
    masked_negative_binomial_loss,
)
from src.models.stgnn import GCNGRU

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
FIGURES_DIR = PROJECT_DIR / "figures"
TABLES_DIR = PROJECT_DIR / "results" / "benchmark" / "tables"
ARTIFACTS_DIR = Path(r"C:\Users\DELL\.gemini\antigravity\brain\b047b28e-edb8-4751-a9fe-3b62341b22c7")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Import pipeline modules
# ---------------------------------------------------------------------------
spec_lhn = importlib.util.spec_from_file_location("lhn", PROJECT_DIR / "scripts/training/31.long_horizon_neural.py")
lhn = importlib.util.module_from_spec(spec_lhn)
spec_lhn.loader.exec_module(lhn)

spec_mh = importlib.util.spec_from_file_location("mh", PROJECT_DIR / "scripts/training/32.train_multi_horizon_negbin.py")
mh = importlib.util.module_from_spec(spec_mh)
spec_mh.loader.exec_module(mh)

pipeline = load_pipeline()
lhn.baseline = pipeline
mh.load_dependencies()


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def predict_quantiles(model, split, adjacency, device) -> np.ndarray:
    """Predict median point forecast for gru_v2_quantile_lw."""
    model.eval()
    with torch.no_grad():
        output = model(
            torch.from_numpy(split["X"]).to(device),
            torch.from_numpy(adjacency).to(device),
        ).cpu().numpy()
    output = np.sort(output, axis=-1) + split["anchor"][..., None]
    median_idx = QUANTILES.index(0.5)
    point = np.clip(np.expm1(output[..., median_idx]), 0.0, None)
    return point


def main():
    style()
    device = torch.device("cpu")
    print(f"Using device: {device} with {torch.get_num_threads()} CPU threads\n")

    folds_all = build_benchmark_folds(pipeline.folds_module)
    test_folds = [f for f in folds_all if f["fold_id"] in (1, 2, 3, 4, 5, 6, 7, 8, 9)]

    tensors_v2 = pipeline.folds_module.load_tensors("v2")
    tensors_v4 = mh.folds_module.load_tensors("v4")
    calendar = pipeline.folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()

    adjacency = np.eye(25, dtype=np.float32)

    # Dictionary to collect national forecasts per year and horizon
    # Key: (year, horizon) -> {'weeks': [...], 'actual': [...], 'gru_qlw': [...], 'negbin_v4': [...]}
    results = {}

    config_negbin = dict(mh.DEFAULTS)
    config_negbin["variant"] = "v4"
    config_negbin["max_epochs"] = 35
    config_negbin["patience"] = 10
    config_negbin["seeds"] = 3

    config_qlw = dict(pipeline.DEFAULTS)
    config_qlw["max_epochs"] = 45
    config_qlw["patience"] = 10
    config_qlw["batch_size"] = 64

    for fold in test_folds:
        fid = fold["fold_id"]
        year = fold["test_year"]
        print(f"=== Processing Fold {fid} (Test Year {year}) ===")
        start_fold_time = time.perf_counter()

        # -------------------------------------------------------------------
        # 1. Multi-Horizon NegBin v4 (trains h=1..4 jointly)
        # -------------------------------------------------------------------
        print("  Training Multi-Horizon NegBin v4 (3 seeds)...", end="", flush=True)
        mh_arrays = mh.build_multi_horizon_fold_arrays(
            tensors_v4, months, fold, config_negbin["lookback"], (1, 2, 3, 4)
        )
        x_test_mh = torch.from_numpy(mh_arrays["test"]["X"]).to(device)
        anc_test_mh = torch.from_numpy(mh_arrays["test"]["anchor"]).to(device)
        adj_t = torch.from_numpy(adjacency).to(device)

        negbin_seed_preds = []
        for seed in range(3):
            model_nb, _ = mh.train_shared_negbin(
                mh_arrays, adjacency, config_negbin, (1, 2, 3, 4), seed, device
            )
            model_nb.eval()
            with torch.no_grad():
                mu, _ = model_nb(x_test_mh, adj_t, anc_test_mh)
                negbin_seed_preds.append(mu.cpu().numpy())

        # Average across 3 seeds: [n_windows, 25, 4]
        negbin_ens_mu = np.mean(negbin_seed_preds, axis=0)
        print(" done.", flush=True)

        # -------------------------------------------------------------------
        # 2. gru_v2_quantile_lw for h=1 and h=4
        # -------------------------------------------------------------------
        for h_idx, h in [(0, 1), (3, 4)]:
            print(f"  Training gru_v2_quantile_lw h={h} (3 seeds)...", end="", flush=True)
            q_arrays = pipeline.build_fold_arrays(
                tensors_v2, months, fold, lookback=12, horizon=h
            )
            qlw_seed_preds = []
            for seed in range(3):
                model_qlw, _ = lhn.train_custom(
                    q_arrays,
                    adjacency,
                    config_qlw,
                    seed,
                    device,
                    quantile=True,
                    weighted=True,
                )
                pred = predict_quantiles(model_qlw, q_arrays["test"], adjacency, device)
                qlw_seed_preds.append(pred)

            # Average across 3 seeds: [n_windows, 25]
            qlw_ens = np.mean(qlw_seed_preds, axis=0)
            print(" done.", flush=True)

            # ---------------------------------------------------------------
            # 3. Align targets and national totals
            # ---------------------------------------------------------------
            # Target periods and ground truth cases
            test_target_periods = q_arrays["test"]["target_period_id"]
            y_actual = q_arrays["test"]["y"]  # [n_windows, 25]
            y_mask = q_arrays["test"]["mask"]  # [n_windows, 25]

            # In Multi-Horizon NegBin:
            # Map mh predictions by target_period_id
            mh_periods_h = mh_arrays["test"]["target_period_id"][:, h_idx]
            mh_mu_h = negbin_ens_mu[..., h_idx]  # [n_windows, 25]

            # Common alignment dictionary per period
            mh_dict = {p: mh_mu_h[i] for i, p in enumerate(mh_periods_h)}

            weeks = []
            actual_national = []
            qlw_national = []
            nb_national = []

            for i, p in enumerate(test_target_periods):
                week_no = i + 1
                weeks.append(week_no)
                actual_national.append(float(np.sum(y_actual[i] * y_mask[i])))
                qlw_national.append(float(np.sum(qlw_ens[i])))
                if p in mh_dict:
                    nb_national.append(float(np.sum(mh_dict[p])))
                else:
                    nb_national.append(np.nan)

            results[(year, h)] = {
                "weeks": np.array(weeks),
                "actual": np.array(actual_national),
                "gru_qlw": np.array(qlw_national),
                "negbin_v4": np.array(nb_national),
            }

        dur_fold = time.perf_counter() - start_fold_time
        print(f"  Fold {fid} completed in {dur_fold:.1f}s\n", flush=True)

    # -----------------------------------------------------------------------
    # Save CSV of forecasts
    # -----------------------------------------------------------------------
    csv_rows = []
    for (year, h), data in results.items():
        for w, act, qlw, nb in zip(data["weeks"], data["actual"], data["gru_qlw"], data["negbin_v4"]):
            csv_rows.append(
                {
                    "year": year,
                    "horizon": h,
                    "week": w,
                    "actual_cases": act,
                    "gru_v2_quantile_lw_ens": qlw,
                    "negbin_v4_ens": nb,
                }
            )

    df_out = pd.DataFrame(csv_rows)
    csv_path = TABLES_DIR / "model_comparison_forecasts_2017_2025.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"Saved forecast dataset to {csv_path}")

    # -----------------------------------------------------------------------
    # Plot 9-panel figures for h=1 and h=4
    # -----------------------------------------------------------------------
    for h in [1, 4]:
        fig, axes = plt.subplots(3, 3, figsize=(14, 9.5), sharex=True)
        axes = axes.flatten()

        for idx, fold in enumerate(test_folds):
            year = fold["test_year"]
            ax = axes[idx]
            data = results[(year, h)]

            w = data["weeks"]
            act = data["actual"]
            qlw = data["gru_qlw"]
            nb = data["negbin_v4"]

            # Compute year-level MAE for title
            valid_mask = ~np.isnan(nb) & ~np.isnan(qlw)
            mae_qlw = np.mean(np.abs(qlw[valid_mask] - act[valid_mask]))
            mae_nb = np.mean(np.abs(nb[valid_mask] - act[valid_mask]))

            # Plot lines
            ax.plot(w, act, color="#111827", label="Observed Cases", linewidth=1.8, zorder=4)
            ax.plot(
                w,
                qlw,
                color="#f59e0b",
                linestyle="--",
                label="gru_v2_quantile_lw (Ens)",
                linewidth=1.5,
                zorder=3,
            )
            ax.plot(
                w,
                nb,
                color="#0284c7",
                linestyle="-",
                label="Multi-Horizon NegBin v4 (Ens)",
                linewidth=1.6,
                zorder=2,
            )

            # Styling
            is_epidemic = year == 2017
            is_covid = year in (2020, 2021)
            tag = " (Epidemic)" if is_epidemic else (" (COVID)" if is_covid else "")

            ax.set_title(
                f"{year}{tag} — NegBin: {mae_nb:.1f} | Quantile: {mae_qlw:.1f} MAE",
                fontweight="bold" if is_epidemic else "normal",
                color="#991b1b" if is_epidemic else "#111827",
            )
            ax.grid(True, linestyle=":", alpha=0.45)

            if idx >= 6:
                ax.set_xlabel("Epidemiological Week of Year")
            if idx % 3 == 0:
                ax.set_ylabel("National Total Cases")

            if idx == 0:
                ax.legend(frameon=True, facecolor="white", edgecolor="#e5e7eb", loc="upper right")

        lead_desc = "1 Week Ahead (Operational)" if h == 1 else "4 Weeks Ahead (Early Warning)"
        fig.suptitle(
            f"National Dengue Forecast Comparison Across Sri Lanka: 2017–2025 (Lead Time h={h}, {lead_desc})\n"
            f"Observed Cases vs. gru_v2_quantile_lw (Ens) vs. Multi-Horizon NegBin v4 (Ens)",
            fontsize=11.5,
            fontweight="bold",
            y=0.99,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        out_png = FIGURES_DIR / f"fig_comparison_2017_2025_h{h}.png"
        out_pdf = FIGURES_DIR / f"fig_comparison_2017_2025_h{h}.pdf"
        fig.savefig(out_png, dpi=300)
        fig.savefig(out_pdf)
        plt.close(fig)

        # Also copy to artifacts directory
        import shutil
        shutil.copy(out_png, ARTIFACTS_DIR / out_png.name)
        print(f"Saved figure for h={h} to {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()

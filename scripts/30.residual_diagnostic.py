"""
Is there capturable structure left in the model's errors, or are they noise?

Every training-procedure and architecture change tried so far has moved the h=1
headline by less than the seed noise (README 8-8e). One hypothesis for why: the
ceiling is the feature set, not the model. Before engineering a new feature this
script asks the prior question -- do the best model's residuals correlate with
anything, or do they look like irreducible noise?

The best configuration is the identity backbone, one model per horizon (the
`separate` arm of 8f), on `v1`. It is trained here at h=1 and h=4: h=4 is where
the model has real skill (8f, 8h), h=1 is the contrast where it does not.
Seven headline folds, three seeds, test predictions captured in-process.

Four diagnostics on the per-district-period test residuals e = y_hat - y:

    A  residual autocorrelation, lag 1-8, per district. White -> the temporal
       signal is extracted. Positive at short lag -> short-range dynamics left
       on the table.

    B  residual vs every one of the 23 v1 channels at the forecast origin, plus
       the case level. A channel that correlates with |e| is one the model is
       not using well -- an architecture or loss lead, not a new-feature lead.

    C  residual vs three candidate missing drivers, each computed on the fly
       from the existing case history, no new data source:
         - periods since this district's last outbreak (>= its 90th pct,
           threshold fitted on the fold's history only)
         - trailing 52-period cumulative cases (a crude depletion proxy)
         - the district's rank within the current national outbreak wave
       A driver that correlates is a green light to engineer it as a feature.

    D  variance decomposition: between-district vs within-district-over-time;
       the share explained by day-of-year; the share concentrated in outbreak
       periods vs endemic ones.

Everything -- the loop, the folds, the masks, the anchored target, the
preprocessing, `predict` -- is imported from `scripts/16`. This trains models
but changes nothing about how they are trained.

Outputs:
    docs/residual_analysis.md
    results/models/residual_diagnostics.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RESULTS_DIR = PROJECT_DIR / "results" / "models"
DOCS_DIR = PROJECT_DIR / "docs"

FOLDS_PATH = PROCESSED_DIR / "folds.json"

DIAGNOSTICS_PATH = RESULTS_DIR / "residual_diagnostics.csv"
REPORT_PATH = DOCS_DIR / "residual_analysis.md"

HEADLINE_FOLDS = (1, 2, 3, 6, 7, 8, 9)
HORIZONS = (1, 4)
SEEDS = 3
PEAK_PERCENTILE = 90.0
DEPLETION_WINDOW = 52  # periods, ~1 year

# Candidate missing-driver proxy names, for the diagnostics table.
DRIVER_COLUMNS = (
    "periods_since_outbreak",
    "trailing_52_cumulative_cases",
    "national_wave_rank",
)


baseline_module = None
naive = None
folds_module = None


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_DIR / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_modules() -> None:
    global baseline_module, naive, folds_module
    baseline_module = _load("baseline_module", "16.train_gcn_gru.py")
    baseline_module.load_modules()
    naive = baseline_module.naive
    folds_module = baseline_module.folds_module


# ---------------------------------------------------------------------------
# Train the best config and capture per-cell test residuals
# ---------------------------------------------------------------------------

def build_model_identity(config: dict):
    """`build_model(n_features) -> GCNGRU`, head width 1 (the scripts/18 trap)."""

    def build_model(n_features: int):
        return baseline_module.GCNGRU(
            n_features=n_features,
            hidden=config["hidden"],
            gcn_layers=config["gcn_layers"],
            horizon=1,
            dropout=config["dropout"],
        )

    return build_model


def collect_residuals(folds: list[dict], config: dict, device: torch.device) -> pd.DataFrame:
    """Train identity/separate at each horizon and return per-cell test residuals.

    One row per (horizon, fold, node_id, target_period_id): the seed-mean
    prediction, the actual, the residual, and whether the cell was observed.
    Only observed cells are kept -- an imputed target is not a real error.
    """

    calendar = folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()

    nodes = pd.read_csv(PROCESSED_DIR / "nodes.csv").sort_values("node_id")
    names = nodes["canonical_name"].tolist()
    n_nodes = len(names)

    adjacency = np.eye(n_nodes, dtype=np.float32)
    tensors = folds_module.load_tensors("v1")

    build_model = build_model_identity(config)

    rows: list[pd.DataFrame] = []

    for horizon in HORIZONS:
        for fold in folds:
            arrays = baseline_module.build_fold_arrays(
                tensors, months, fold, config["lookback"], horizon
            )

            test = arrays["test"]
            target = test["y"]                       # [windows, nodes]
            mask = test["mask"].astype(np.int8)      # [windows, nodes]
            target_period = test["target_period_id"]  # [windows]

            seed_predictions = []
            for seed in range(config["seeds"]):
                model, _ = baseline_module.train_one(
                    arrays, adjacency, config, seed, device, build_model
                )
                seed_predictions.append(
                    baseline_module.predict(
                        model, test, adjacency, config["target"], device
                    )
                )

            prediction = np.mean(seed_predictions, axis=0)  # [windows, nodes]

            n_windows = prediction.shape[0]
            frame = pd.DataFrame(
                {
                    "horizon": horizon,
                    "fold_id": fold["fold_id"],
                    "test_year": fold["test_year"],
                    "target_period_id": np.repeat(target_period, n_nodes),
                    "node_id": np.tile(np.arange(n_nodes), n_windows),
                    "canonical_name": np.tile(names, n_windows),
                    "predicted": prediction.reshape(-1),
                    "actual": target.reshape(-1),
                    "observed": mask.reshape(-1),
                }
            )
            frame["residual"] = frame["predicted"] - frame["actual"]
            frame["abs_residual"] = frame["residual"].abs()
            rows.append(frame)

            print(
                f"  h={horizon} fold {fold['fold_id']} ({fold['test_year']}) "
                f"{n_windows} windows",
                flush=True,
            )

    residuals = pd.concat(rows, ignore_index=True)
    return residuals[residuals["observed"] == 1].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Origin-period feature panel (part B) and driver proxies (part C)
# ---------------------------------------------------------------------------

def origin_feature_panel(horizon: int) -> pd.DataFrame:
    """The 23 v1 channels at each window's forecast origin, long by (period, node).

    The residual is aligned to the TARGET period; the model's inputs end at the
    ORIGIN, which is `horizon` periods earlier. Part B asks whether the error
    correlates with what the model actually saw, so features are read at the
    origin and then joined onto the residual by target_period_id.
    """

    tensors = folds_module.load_tensors("v1")
    x = tensors["X"]                       # [periods, nodes, 23], unscaled, NaN where unobserved
    period_id = tensors["period_id"]
    feature_names = [str(n) for n in tensors["feature_names"]]
    n_periods, n_nodes, _ = x.shape

    index_of = {int(p): i for i, p in enumerate(period_id)}

    records = []
    for t_index, target_period in enumerate(period_id):
        origin_period = int(target_period) - horizon
        origin_index = index_of.get(origin_period)
        if origin_index is None:
            continue
        for node in range(n_nodes):
            row = {
                "target_period_id": int(target_period),
                "node_id": node,
            }
            for f_index, f_name in enumerate(feature_names):
                row[f"origin__{f_name}"] = x[origin_index, node, f_index]
            records.append(row)

    return pd.DataFrame(records)


def driver_proxies(folds: list[dict]) -> pd.DataFrame:
    """The three candidate missing-driver proxies, per (target_period_id, node_id).

    Computed from the raw case series only. The outbreak threshold is fitted per
    fold on `period_id <= fit_end_period` so the proxy carries no future
    information into a training window; the proxy is emitted once per fold with
    that fold's threshold, and later joined to residuals on (fold_id, period,
    node).
    """

    tensors = folds_module.load_tensors("v1")
    y = tensors["y"].astype(float)          # [periods, nodes] raw counts, NaN where unobserved
    y_mask = tensors["y_mask"]
    period_id = tensors["period_id"].astype(int)
    n_periods, n_nodes = y.shape

    # A filled series for the cumulative / recency arithmetic: missing -> 0 cases
    # observed, which is what persistence does with the same gap and keeps the
    # running quantities finite. The residual join later restricts to observed
    # target cells anyway.
    y_filled = np.where(np.isnan(y), 0.0, y)

    # National wave rank: within each period, rank districts by that period's
    # case count (1 = highest). A district high in the ranking is "leading" the
    # current national wave. Period-local, so no fold dependence.
    order = np.argsort(-y_filled, axis=1, kind="stable")
    wave_rank = np.empty_like(order)
    rows = np.arange(n_periods)[:, None]
    wave_rank[rows, order] = np.arange(1, n_nodes + 1)[None, :]

    # Trailing 52-period cumulative cases, inclusive of the current period.
    cumulative = np.zeros_like(y_filled)
    for t in range(n_periods):
        lo = max(0, t - DEPLETION_WINDOW + 1)
        cumulative[t] = y_filled[lo : t + 1].sum(axis=0)

    frames = []
    for fold in folds:
        fit_mask = period_id <= fold["fit_end_period"]
        history = y[fit_mask]
        history_mask = y_mask[fit_mask]

        # Per-district 90th-percentile threshold on the fold's history only.
        thresholds = np.array(
            [
                np.nanpercentile(
                    history[:, node][history_mask[:, node] == 1], PEAK_PERCENTILE
                )
                if (history_mask[:, node] == 1).any()
                else np.inf
                for node in range(n_nodes)
            ]
        )

        is_outbreak = y_filled >= thresholds[None, :]

        # periods_since_outbreak[t, node] = t - (last t' <= t with an outbreak).
        # Before the first outbreak in the series it is the distance from t=0,
        # which is a defensible "no outbreak seen yet, and this long into the
        # record" encoding.
        since = np.full((n_periods, n_nodes), np.nan)
        last = np.full(n_nodes, -1)
        for t in range(n_periods):
            fired = is_outbreak[t]
            last = np.where(fired, t, last)
            seen = last >= 0
            since[t, seen] = t - last[seen]

        for t in range(n_periods):
            for node in range(n_nodes):
                frames.append(
                    {
                        "fold_id": fold["fold_id"],
                        "target_period_id": int(period_id[t]),
                        "node_id": node,
                        "periods_since_outbreak": since[t, node],
                        "trailing_52_cumulative_cases": cumulative[t, node],
                        "national_wave_rank": int(wave_rank[t, node]),
                    }
                )

    return pd.DataFrame(frames)


# ---------------------------------------------------------------------------
# Part A -- residual autocorrelation
# ---------------------------------------------------------------------------

def autocorrelation(series: np.ndarray, lag: int) -> float:
    """Pearson autocorrelation of `series` at `lag`, ignoring NaN-padded gaps."""

    if len(series) <= lag + 2:
        return float("nan")
    a = series[:-lag]
    b = series[lag:]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return float("nan")
    if np.std(a[ok]) < 1e-9 or np.std(b[ok]) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def part_a(residuals: pd.DataFrame) -> pd.DataFrame:
    """Lag 1-8 residual ACF, per horizon, averaged over districts.

    Each district's residual series is ordered by target_period_id and its ACF
    computed separately, then averaged across the 25 districts. A district-level
    series can have gaps (unobserved target periods); `autocorrelation` skips
    pairs that straddle one.
    """

    records = []
    for horizon, hgroup in residuals.groupby("horizon"):
        for lag in range(1, 9):
            per_district = []
            for _, dgroup in hgroup.groupby("node_id"):
                ordered = dgroup.sort_values("target_period_id")
                per_district.append(
                    autocorrelation(ordered["residual"].to_numpy(), lag)
                )
            per_district = np.array(per_district, dtype=float)
            finite = per_district[np.isfinite(per_district)]

            # Two-sided test that the mean district-level ACF is zero.
            if len(finite) >= 3:
                t_stat, p_value = stats.ttest_1samp(finite, 0.0)
            else:
                t_stat, p_value = float("nan"), float("nan")

            records.append(
                {
                    "part": "A_residual_acf",
                    "horizon": horizon,
                    "quantity": f"acf_lag_{lag}",
                    "value": float(np.nanmean(per_district)),
                    "value_sd": float(np.nanstd(per_district)),
                    "n": int(len(finite)),
                    "p_value": float(p_value),
                    "note": "mean over districts of per-district residual ACF",
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Parts B and C -- residual vs a candidate quantity
# ---------------------------------------------------------------------------

def correlate_block(
    merged: pd.DataFrame, columns: list[str], part: str
) -> pd.DataFrame:
    """Pearson and Spearman of residual and |residual| against each column.

    One row per (part, horizon, column, target) where target is `residual`
    (signed -- does the model run high or low) or `abs_residual` (magnitude --
    is the model less certain here). NaN cells are dropped pairwise.
    """

    records = []
    for horizon, hgroup in merged.groupby("horizon"):
        for column in columns:
            if column not in hgroup:
                continue
            x = hgroup[column].to_numpy(dtype=float)
            for target_name in ("residual", "abs_residual"):
                y = hgroup[target_name].to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 30 or np.std(x[ok]) < 1e-12:
                    pear = spear = p_pear = float("nan")
                else:
                    pear, p_pear = stats.pearsonr(x[ok], y[ok])
                    spear, _ = stats.spearmanr(x[ok], y[ok])
                records.append(
                    {
                        "part": part,
                        "horizon": horizon,
                        "quantity": column,
                        "target": target_name,
                        "pearson_r": float(pear),
                        "spearman_r": float(spear),
                        "p_value": float(p_pear),
                        "n": int(ok.sum()),
                    }
                )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Part D -- variance decomposition
# ---------------------------------------------------------------------------

def part_d(residuals: pd.DataFrame) -> pd.DataFrame:
    """Where the residual variance lives.

    - between-district: variance of per-district mean residual / total variance
      of the residual (a one-way ANOVA eta^2 on district).
    - seasonal: R^2 of residual on [doy_sin, doy_cos] built from target_period_id
      via the calendar.
    - outbreak concentration: mean |residual| in outbreak target cells vs
      endemic ones, and the share of total absolute error carried by outbreak
      cells (which are a minority).
    """

    calendar = folds_module.load_calendar()
    doy = calendar.set_index("period_id")["start_date"]
    doy = pd.to_datetime(doy).dt.dayofyear

    tensors = folds_module.load_tensors("v1")
    y = tensors["y"].astype(float)
    y_mask = tensors["y_mask"]
    period_id = tensors["period_id"].astype(int)
    index_of = {int(p): i for i, p in enumerate(period_id)}

    # A fold-free global 90th-pct outbreak flag for the concentration split.
    # This is descriptive, not fed to any model, so a global threshold is fine.
    global_threshold = np.array(
        [
            np.nanpercentile(y[:, node][y_mask[:, node] == 1], PEAK_PERCENTILE)
            for node in range(y.shape[1])
        ]
    )

    records = []
    for horizon, hgroup in residuals.groupby("horizon"):
        e = hgroup["residual"].to_numpy(dtype=float)
        total_var = float(np.nanvar(e))

        # Between-district eta^2.
        district_means = hgroup.groupby("node_id")["residual"].mean()
        counts = hgroup.groupby("node_id")["residual"].size()
        grand = float(np.nanmean(e))
        ss_between = float(
            (counts * (district_means - grand) ** 2).sum()
        )
        ss_total = float(np.nansum((e - grand) ** 2))
        eta2_district = ss_between / ss_total if ss_total > 0 else float("nan")

        # Seasonal R^2.
        periods = hgroup["target_period_id"].map(lambda p: doy.get(p, np.nan))
        angle = 2.0 * np.pi * periods.to_numpy(dtype=float) / 365.25
        design = np.column_stack(
            [np.ones_like(angle), np.sin(angle), np.cos(angle)]
        )
        ok = np.isfinite(angle) & np.isfinite(e)
        if ok.sum() > 10:
            coef, *_ = np.linalg.lstsq(design[ok], e[ok], rcond=None)
            fitted = design[ok] @ coef
            ss_res = float(np.sum((e[ok] - fitted) ** 2))
            ss_tot = float(np.sum((e[ok] - np.mean(e[ok])) ** 2))
            seasonal_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        else:
            seasonal_r2 = float("nan")

        # Outbreak concentration.
        is_outbreak = np.array(
            [
                row.actual >= global_threshold[int(row.node_id)]
                for row in hgroup.itertuples()
            ]
        )
        abs_e = hgroup["abs_residual"].to_numpy(dtype=float)
        outbreak_share_of_cells = float(is_outbreak.mean())
        outbreak_share_of_abs_error = (
            float(abs_e[is_outbreak].sum() / abs_e.sum()) if abs_e.sum() > 0 else float("nan")
        )
        mae_outbreak = float(np.nanmean(abs_e[is_outbreak])) if is_outbreak.any() else float("nan")
        mae_endemic = float(np.nanmean(abs_e[~is_outbreak])) if (~is_outbreak).any() else float("nan")

        for quantity, value in [
            ("total_residual_variance", total_var),
            ("between_district_eta2", eta2_district),
            ("seasonal_r2", seasonal_r2),
            ("outbreak_share_of_cells", outbreak_share_of_cells),
            ("outbreak_share_of_abs_error", outbreak_share_of_abs_error),
            ("mae_outbreak_cells", mae_outbreak),
            ("mae_endemic_cells", mae_endemic),
        ]:
            records.append(
                {
                    "part": "D_variance_decomposition",
                    "horizon": horizon,
                    "quantity": quantity,
                    "value": value,
                    "n": int(len(hgroup)),
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(diagnostics: pd.DataFrame, config: dict) -> None:
    """Write docs/residual_analysis.md -- raw numbers, no interpretation.

    Interpretation is added by hand after review, per the task. This writes the
    tables and a stub verdict section.
    """

    def block(part: str) -> pd.DataFrame:
        return diagnostics[diagnostics["part"] == part]

    lines = [
        "# Residual diagnostic",
        "",
        "Does the best model's error carry capturable structure, or is it noise?",
        "",
        "Config: identity backbone, one model per horizon (8f `separate`), `v1`, "
        f"{config['seeds']} seeds, headline folds {list(HEADLINE_FOLDS)}, "
        f"horizons {list(HORIZONS)}. Residual `e = y_hat - y` on observed test "
        "cells only. Trained via `scripts/16`'s imported loop.",
        "",
        "Reproduce: `python scripts/30.residual_diagnostic.py`",
        "",
        "/ RAW NUMBERS BELOW -- interpretation section is filled in after review. /",
        "",
        "## A. Residual autocorrelation",
        "",
        "Mean over the 25 districts of each district's own residual ACF, by lag. "
        "`p_value` tests that the across-district mean is zero.",
        "",
        "| Horizon | Lag | Mean ACF | sd | n districts | p |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in block("A_residual_acf").itertuples():
        lag = row.quantity.replace("acf_lag_", "")
        lines.append(
            f"| {row.horizon} | {lag} | {row.value:+.3f} | {row.value_sd:.3f} | "
            f"{row.n} | {row.p_value:.3f} |"
        )

    for part, title, note in [
        (
            "B_residual_vs_feature",
            "B. Residual vs each v1 channel at the forecast origin",
            "`residual` signed (model runs high/low); `abs_residual` magnitude. "
            "Sorted by |pearson_r| within horizon.",
        ),
        (
            "C_residual_vs_driver",
            "C. Residual vs candidate missing-driver proxies",
            "Computed from case history only. `periods_since_outbreak` uses a "
            "per-fold 90th-pct threshold.",
        ),
    ]:
        lines += ["", f"## {title}", "", note, ""]
        lines += [
            "| Horizon | Quantity | Target | Pearson r | Spearman r | p | n |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        b = block(part).copy()
        b["abs_r"] = b["pearson_r"].abs()
        for row in b.sort_values(
            ["horizon", "abs_r"], ascending=[True, False]
        ).itertuples():
            lines.append(
                f"| {row.horizon} | `{row.quantity}` | {row.target} | "
                f"{row.pearson_r:+.3f} | {row.spearman_r:+.3f} | "
                f"{row.p_value:.3f} | {row.n} |"
            )

    lines += [
        "",
        "## D. Variance decomposition",
        "",
        "| Horizon | Quantity | Value |",
        "| --- | --- | --- |",
    ]
    for row in block("D_variance_decomposition").itertuples():
        lines.append(f"| {row.horizon} | `{row.quantity}` | {row.value:.4f} |")

    lines += [
        "",
        "## Verdict",
        "",
        "_To be written after review of the numbers above._",
        "",
        "The decision table this feeds:",
        "",
        "| Finding | Implication |",
        "| --- | --- |",
        "| Residuals ~ white, correlate with nothing | Ceiling is the data. Stop feature engineering; move to likelihood/calibration. |",
        "| Residuals correlate with a channel already in the tensor | Architecture/loss lead, not a feature lead. |",
        "| Residuals correlate with an outbreak-history / susceptibility proxy | Green light to engineer that feature. |",
        "| Residual has spatial structure (neighbour errors correlate) | The graph question is not as closed as 8e says; revisit at h=4. |",
        "",
        "## Output files",
        "",
        f"- `{DIAGNOSTICS_PATH.relative_to(PROJECT_DIR).as_posix()}`",
    ]

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, default=list(HEADLINE_FOLDS))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument("--seeds", type=int, default=SEEDS)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()

    load_modules()

    if not FOLDS_PATH.exists() or not (PROCESSED_DIR / "adjacency.npz").exists():
        print("Missing folds.json or adjacency.npz. Run scripts 13 and 14 first.")
        return 1

    global HORIZONS
    HORIZONS = tuple(arguments.horizons)

    config = dict(baseline_module.DEFAULTS)
    config["target"] = "residual"
    config["seeds"] = arguments.seeds

    all_folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))["folds"]
    folds = [f for f in all_folds if f["fold_id"] in arguments.folds]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device:   {device}")
    print(f"Horizons: {list(HORIZONS)}")
    print(f"Folds:    {[f['fold_id'] for f in folds]}")
    print(f"Seeds:    {config['seeds']}")
    print(f"Config:   identity backbone, separate arm, v1\n")

    started = time.perf_counter()

    print("Training and collecting residuals...")
    residuals = collect_residuals(folds, config, device)
    print(f"  {len(residuals)} observed test cells\n")

    # Part B: join the origin-period feature panel per horizon.
    print("Part B: residual vs origin features")
    b_blocks = []
    for horizon in HORIZONS:
        panel = origin_feature_panel(horizon)
        merged = residuals[residuals["horizon"] == horizon].merge(
            panel, on=["target_period_id", "node_id"], how="left"
        )
        feature_columns = [c for c in merged.columns if c.startswith("origin__")]
        b_blocks.append(correlate_block(merged, feature_columns, "B_residual_vs_feature"))
    part_b = pd.concat(b_blocks, ignore_index=True)

    # Part C: join the driver proxies per (fold, period, node).
    print("Part C: residual vs driver proxies")
    proxies = driver_proxies(folds)
    merged_c = residuals.merge(
        proxies, on=["fold_id", "target_period_id", "node_id"], how="left"
    )
    part_c = correlate_block(merged_c, list(DRIVER_COLUMNS), "C_residual_vs_driver")

    print("Part A: residual autocorrelation")
    part_a_block = part_a(residuals)

    print("Part D: variance decomposition")
    part_d_block = part_d(residuals)

    diagnostics = pd.concat(
        [part_a_block, part_b, part_c, part_d_block], ignore_index=True
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(DIAGNOSTICS_PATH, index=False)
    write_report(diagnostics, config)

    elapsed = (time.perf_counter() - started) / 60
    print(f"\nDone in {elapsed:.1f} min")
    print(f"Wrote {DIAGNOSTICS_PATH.relative_to(PROJECT_DIR)}")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_DIR)}")

    # Console summary of the headline numbers.
    print("\n--- Part A: residual ACF (mean over districts) ---")
    for row in part_a_block.itertuples():
        flag = " *" if np.isfinite(row.p_value) and row.p_value < 0.05 else ""
        print(f"  h={row.horizon} {row.quantity:<10} {row.value:+.3f}  p={row.p_value:.3f}{flag}")

    print("\n--- Part C: residual vs driver proxies (top by |r|) ---")
    cc = part_c.copy()
    cc["abs_r"] = cc["pearson_r"].abs()
    for row in cc.sort_values("abs_r", ascending=False).head(12).itertuples():
        flag = " *" if np.isfinite(row.p_value) and row.p_value < 0.05 else ""
        print(
            f"  h={row.horizon} {row.quantity:<28} vs {row.target:<12} "
            f"r={row.pearson_r:+.3f}  p={row.p_value:.3f}{flag}"
        )

    print("\n--- Part B: strongest residual-vs-feature correlations ---")
    bb = part_b.copy()
    bb["abs_r"] = bb["pearson_r"].abs()
    for row in bb.sort_values("abs_r", ascending=False).head(12).itertuples():
        flag = " *" if np.isfinite(row.p_value) and row.p_value < 0.05 else ""
        print(
            f"  h={row.horizon} {row.quantity:<32} vs {row.target:<12} "
            f"r={row.pearson_r:+.3f}  p={row.p_value:.3f}{flag}"
        )

    print("\n--- Part D ---")
    for row in part_d_block.itertuples():
        print(f"  h={row.horizon} {row.quantity:<32} {row.value:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

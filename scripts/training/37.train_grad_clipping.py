"""Gradient clipping ablation for the multi-horizon Negative Binomial model.

Does clipping the global gradient norm before every ``optimiser.step()`` make
the current model (script 32: shared GRU trunk, 4 NegBin heads, tensor v3)
train more stably or forecast better, and are there exploding gradients for it
to prevent in the first place?

The arms change only ``clip_max_norm``; everything else is script 32:

    no_clip    max_norm = None   gradients untouched (norm still logged)
    clip_0.25  max_norm = 0.25   near the median step norm, so it binds
    clip_1     max_norm = 1.0    what script 32 has always done
    clip_5     max_norm = 5.0    a looser clip

The training loop is not copied: script 32's ``train_shared_negbin`` takes the
threshold through ``config["clip_max_norm"]`` and returns the pre-clip global
norm of every step. Within a (fold, seed) every arm gets identical data, batch
order and initialisation, so the arms differ in the clip and nothing else.

Runtime. Two things keep the sweep short on a CPU:

1. The unclipped control runs first. Its logged norms say whether a threshold
   could ever have fired: if every step's norm was below ``max_norm`` the
   clipped run is bit-identical (the clip factor is exactly 1 on every step),
   so it is copied instead of retrained and flagged ``inferred=True``. On this
   model 5.0 never fires and 1.0 fires in 9 of 21 runs, so 33 of 84 runs are
   copied. ``--no-skip`` trains every arm anyway.
2. The model is ~8k parameters, so per-batch overhead dominates and PyTorch's
   default thread count is slower than 2-4 threads. Jobs run in ``--workers``
   processes of ``--threads`` threads (default: cpu_count/2 workers x 2).

Default sweep (7 headline folds x 3 seeds x 4 arms, 51 of 84 runs trained):
28.7 min on a 12-thread CPU, against ~47 min without the skip and ~1.5 h
serially at the default thread count. ``--quick`` (folds 1 and 8, 1
seed): ~3 min.

Outputs (results/models/):
    grad_clipping_metrics.csv   test metrics per arm / fold / seed / horizon
    grad_clipping_runs.csv      per-run training-stability statistics
    grad_clipping_epochs.csv    per-epoch losses and gradient-norm summaries
    grad_clipping_steps.csv     pre-clip gradient norm of every optimiser step
    grad_clipping_report.md     summary tables and verdict inputs
results/figures/grad_clipping_norms.png

Usage:
    python scripts/training/37.train_grad_clipping.py --quick     # smoke test
    python scripts/training/37.train_grad_clipping.py             # full sweep
    python scripts/training/37.train_grad_clipping.py --max-norms none 0.5 1 5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

RESULTS_DIR = PROJECT_DIR / "results" / "models"
FIGURES_DIR = PROJECT_DIR / "results" / "figures"
FOLDS_PATH = PROJECT_DIR / "data" / "processed" / "folds.json"

HEADLINE_FOLDS = (1, 2, 3, 6, 7, 8, 9)
QUICK_FOLDS = (1, 8)
# 1.0 is script 32's committed clip, 5.0 a looser one. 0.25 sits near the median
# pre-clip norm of this model (~0.2), so unlike the other two it actually binds.
DEFAULT_MAX_NORMS = (None, 0.25, 1.0, 5.0)
CONTROL_ARM = "no_clip"

# The current model's configuration (docs/new_findings_report.md, v3 row).
MODEL_OVERRIDES = {"variant": "v3", "backbone": "identity", "head": "parallel"}

negbin = None  # script 32, loaded per process
_worker_state: dict = {}


def _load_script(name: str, rel_path: str):
    path = PROJECT_DIR / "scripts" / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_modules() -> None:
    global negbin
    if negbin is None:
        negbin = _load_script("negbin_module", "training/32.train_multi_horizon_negbin.py")
        negbin.load_dependencies()


def arm_name(max_norm: float | None) -> str:
    return CONTROL_ARM if max_norm is None else f"clip_{max_norm:g}"


def parse_max_norm(value: str) -> float | None:
    load_modules()
    return negbin.parse_max_norm(value)


def make_config(max_norm: float | None, seeds: int, **overrides) -> dict:
    config = dict(negbin.DEFAULTS)
    config.update(MODEL_OVERRIDES)
    config.update(seeds=seeds, weighted=False, clip_max_norm=max_norm)
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# Per-process data (built once per worker, fold arrays cached)
# ---------------------------------------------------------------------------

def init_worker(threads: int | None, variant: str, backbone: str) -> None:
    import torch

    if threads:
        torch.set_num_threads(threads)
    load_modules()
    folds_module = negbin.folds_module

    calendar = folds_module.load_calendar()
    if backbone == "identity":
        adjacency = np.eye(25, dtype=np.float32)
    else:
        adjacency = np.load(negbin.ADJACENCY_PATH, allow_pickle=True)["A_norm"].astype(np.float32)

    _worker_state.clear()
    _worker_state.update(
        months=calendar.sort_values("period_id")["month"].to_numpy(),
        tensors=folds_module.load_tensors(variant),
        adjacency=adjacency,
        folds={f["fold_id"]: f for f in json.loads(FOLDS_PATH.read_text())["folds"]},
        arrays={},
    )


def fold_arrays(fold_id: int, config: dict, horizons: tuple[int, ...]):
    cache = _worker_state["arrays"]
    if fold_id not in cache:
        cache[fold_id] = negbin.build_multi_horizon_fold_arrays(
            _worker_state["tensors"],
            _worker_state["months"],
            _worker_state["folds"][fold_id],
            config["lookback"],
            horizons,
        )
    return cache[fold_id]


def fold_thresholds(fold_id: int) -> np.ndarray:
    tensors = _worker_state["tensors"]
    fold = _worker_state["folds"][fold_id]
    fit_mask = tensors["period_id"] <= fold["fit_end_period"]
    return negbin.naive.peak_thresholds(tensors["y"], tensors["y_mask"], fit_mask)


# ---------------------------------------------------------------------------
# One job = one (fold, seed, arm) training run
# ---------------------------------------------------------------------------

def stability_stats(grad_norms: np.ndarray, history: list[dict], max_norm: float | None) -> dict:
    """Summarise how smoothly one run trained."""

    finite = grad_norms[np.isfinite(grad_norms)]
    median = float(np.median(finite)) if finite.size else float("nan")
    val = np.array([h["val_loss"] for h in history])
    train = np.array([h["train_loss"] for h in history])
    val_steps = np.diff(val)
    return {
        "steps": int(grad_norms.size),
        "grad_norm_median": median,
        "grad_norm_p99": float(np.percentile(finite, 99)) if finite.size else float("nan"),
        "grad_norm_max": float(finite.max()) if finite.size else float("nan"),
        # A spike is a step whose norm is >10x the run's median.
        "spike_ratio": float(finite.max() / median) if finite.size else float("nan"),
        "spike_steps": int((finite > 10 * median).sum()),
        "nonfinite_steps": int((~np.isfinite(grad_norms)).sum()),
        "clipped_fraction": (
            float((grad_norms > max_norm).mean()) if max_norm is not None else 0.0
        ),
        "first_epoch_grad_norm": float(np.mean(grad_norms[: max(1, grad_norms.size // len(history))])),
        "val_up_fraction": float((val_steps > 0).mean()) if val_steps.size else 0.0,
        "val_jitter": float(np.median(np.abs(val_steps))) if val_steps.size else 0.0,
        "train_jitter": float(np.median(np.abs(np.diff(train)))) if train.size > 1 else 0.0,
        "final_train_loss": float(train[-1]),
    }


def run_job(job: dict) -> dict:
    import torch

    fold_id, seed, max_norm = job["fold_id"], job["seed"], job["max_norm"]
    config, horizons = job["config"], tuple(job["horizons"])
    fold = _worker_state["folds"][fold_id]
    arrays = fold_arrays(fold_id, config, horizons)
    thresholds = fold_thresholds(fold_id)
    device = torch.device("cpu")

    started = time.perf_counter()
    model, info = negbin.train_shared_negbin(
        arrays, _worker_state["adjacency"], config, horizons, seed, device
    )
    seconds = time.perf_counter() - started

    model.eval()
    with torch.no_grad():
        mu, _ = model(
            torch.from_numpy(arrays["test"]["X"]),
            torch.from_numpy(_worker_state["adjacency"]),
            torch.from_numpy(arrays["test"]["anchor"]),
        )
    mu = mu.numpy()

    arm = arm_name(max_norm)
    base = {
        "arm": arm,
        "max_norm": max_norm,
        "fold_id": fold_id,
        "test_year": fold["test_year"],
        "headline": fold["headline"],
        "seed": seed,
    }
    target = arrays["test"]["y"]
    mask = arrays["test"]["mask"].astype(np.int8)
    metrics = [
        {**base, "horizon": h, "ensemble": False, "best_epoch": info["best_epoch"],
         **negbin.naive.evaluate(mu[..., i], target[..., i], mask[..., i], thresholds)}
        for i, h in enumerate(horizons)
    ]
    grad_norms = info["grad_norms"]
    run = {
        **base,
        "best_epoch": info["best_epoch"],
        "epochs_run": info["epochs_run"],
        "best_val_loss": info["val_loss"],
        "seconds": seconds,
        **stability_stats(grad_norms, info["history"], max_norm),
    }
    epochs = [{**base, **row} for row in info["history"]]
    steps_per_epoch = int(np.ceil(len(arrays["train"]["X"]) / config["batch_size"]))
    steps = pd.DataFrame(
        {
            "arm": arm,
            "fold_id": fold_id,
            "seed": seed,
            "step": np.arange(grad_norms.size),
            "epoch": np.arange(grad_norms.size) // steps_per_epoch + 1,
            "grad_norm": grad_norms,
        }
    )
    return {"metrics": metrics, "run": run, "epochs": epochs, "steps": steps, "mu": mu}


def persistence_rows(fold_ids, config, horizons) -> list[dict]:
    rows = []
    for fold_id in fold_ids:
        fold = _worker_state["folds"][fold_id]
        arrays = fold_arrays(fold_id, config, horizons)
        scores = negbin.evaluate_persistence_per_horizon(
            arrays["test"], fold_thresholds(fold_id), horizons
        )
        for score in scores:
            rows.append({
                "arm": "persistence", "max_norm": None, "fold_id": fold_id,
                "test_year": fold["test_year"], "headline": fold["headline"],
                "seed": "none", "ensemble": False, "best_epoch": 0, **score,
            })
    return rows


def ensemble_rows(results: list[dict], config: dict, horizons) -> list[dict]:
    """Seed-mean ensembles per (arm, fold)."""

    groups: dict[tuple, list[dict]] = {}
    for result in results:
        groups.setdefault((result["run"]["arm"], result["run"]["fold_id"]), []).append(result)

    rows = []
    for (arm, fold_id), members in groups.items():
        if len(members) < 2:
            continue
        run = members[0]["run"]
        fold = _worker_state["folds"][fold_id]
        arrays = fold_arrays(fold_id, config, horizons)
        target = arrays["test"]["y"]
        mask = arrays["test"]["mask"].astype(np.int8)
        thresholds = fold_thresholds(fold_id)
        mu = np.mean([m["mu"] for m in members], axis=0)
        best_epoch = float(np.mean([m["run"]["best_epoch"] for m in members]))
        for i, h in enumerate(horizons):
            rows.append({
                "arm": arm, "max_norm": run["max_norm"], "fold_id": fold_id,
                "test_year": fold["test_year"], "headline": fold["headline"],
                "seed": "ensemble", "ensemble": True, "best_epoch": best_epoch,
                "horizon": h,
                **negbin.naive.evaluate(mu[..., i], target[..., i], mask[..., i], thresholds),
            })
    return rows


def clip_never_fires(control_run: dict, max_norm: float) -> bool:
    """True when clipping at ``max_norm`` could not have touched the control run.

    ``clip_grad_norm_`` scales gradients by ``min(1, max_norm / (norm + 1e-6))``.
    If every step of the unclipped run had ``norm + 1e-6 <= max_norm`` the
    factor is exactly 1 on every step, so the clipped run follows the identical
    trajectory and training it again would reproduce the control bit-for-bit.
    """
    return (
        control_run["nonfinite_steps"] == 0
        and control_run["grad_norm_max"] + 1e-6 <= max_norm
    )


def inferred_result(control: dict, max_norm: float) -> dict:
    """The clipped run's result, copied from a control run it provably equals."""

    arm = arm_name(max_norm)
    relabel = {"arm": arm, "max_norm": max_norm}
    steps = control["steps"].copy()
    steps["arm"] = arm
    return {
        "metrics": [{**row, **relabel} for row in control["metrics"]],
        "run": {**control["run"], **relabel, "inferred": True, "seconds": 0.0},
        "epochs": [{**row, **relabel} for row in control["epochs"]],
        "steps": steps,
        "mu": control["mu"],
    }


def run_experiment(
    fold_ids, seeds: int, max_norms, horizons, workers: int, threads: int | None,
    config_overrides: dict | None = None, skip_identical: bool = True,
) -> dict[str, pd.DataFrame]:
    """Train the control first; train each clipped arm only if it can differ.

    Control (no-clip) jobs are queued first, longest folds first. When one
    finishes, every clipped arm for that (fold, seed) is either queued or, if
    its threshold is above every gradient norm the control ever saw, recorded
    as identical to the control without retraining (``inferred=True``).
    """
    config_overrides = config_overrides or {}
    base = make_config(None, seeds, **config_overrides)
    init_worker(threads, base["variant"], base["backbone"])

    def job(fold_id, seed, max_norm):
        return {
            "fold_id": fold_id, "seed": seed, "max_norm": max_norm,
            "config": make_config(max_norm, seeds, **config_overrides),
            "horizons": list(horizons),
        }

    clipped = [m for m in max_norms if m is not None]
    controls = [
        job(fold_id, seed, None)
        for fold_id in sorted(fold_ids, reverse=True)
        for seed in range(seeds)
    ]
    total = len(controls) * len(max_norms)
    print(f"{total} runs ({len(controls)} controls first) on {workers} worker(s) "
          f"x {threads or 'default'} thread(s)", flush=True)

    results: list[dict] = []
    started = time.perf_counter()

    def record(result: dict) -> list[dict]:
        """Store a finished run and return the follow-up jobs it unlocks."""
        results.append(result)
        run = result["run"]
        run.setdefault("inferred", False)
        maes = " ".join(f"h{m['horizon']}={m['mae']:.2f}" for m in result["metrics"])
        how = "copied (clip never fires)" if run["inferred"] else f"{run['seconds']:.0f}s"
        print(
            f"[{len(results):>3}/{total}] fold {run['fold_id']} seed {run['seed']} "
            f"{run['arm']:<9} {maes} | best ep {run['best_epoch']:>3} "
            f"| |g| med {run['grad_norm_median']:.2f} max {run['grad_norm_max']:.2f} "
            f"clipped {run['clipped_fraction']:.0%} | {how} "
            f"(elapsed {time.perf_counter() - started:.0f}s)",
            flush=True,
        )
        if run["max_norm"] is not None and not pd.isna(run["max_norm"]):
            return []
        follow_up = []
        for max_norm in clipped:
            if skip_identical and clip_never_fires(run, max_norm):
                follow_up.extend(record(inferred_result(result, max_norm)))
            else:
                follow_up.append(job(run["fold_id"], run["seed"], max_norm))
        return follow_up

    if workers <= 1:
        queue = list(controls)
        while queue:
            queue.extend(record(run_job(queue.pop(0))))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            initargs=(threads, base["variant"], base["backbone"]),
        ) as pool:
            pending = {pool.submit(run_job, j) for j in controls}
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    for follow in record(future.result()):
                        pending.add(pool.submit(run_job, follow))

    metrics = pd.DataFrame(
        persistence_rows(fold_ids, base, horizons)
        + [row for r in results for row in r["metrics"]]
        + ensemble_rows(results, base, horizons)
    )
    sort = ["arm", "fold_id", "seed"]
    return {
        "metrics": metrics,
        "runs": pd.DataFrame([r["run"] for r in results]).sort_values(sort, key=lambda c: c.astype(str)),
        "epochs": pd.DataFrame([row for r in results for row in r["epochs"]]),
        "steps": pd.concat([r["steps"] for r in results], ignore_index=True),
        "wall_seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarise(metrics: pd.DataFrame, runs: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    """Headline-fold MAE per arm and horizon, plus paired tests against no_clip."""

    from scipy import stats

    head = metrics[metrics["headline"].astype(bool)]
    single = head[~head["ensemble"].astype(bool)]
    ens = head[head["ensemble"].astype(bool)]
    fold_mean = single.groupby(["arm", "horizon", "fold_id"])["mae"].mean()

    rows = []
    for arm in ["persistence", *arms]:
        for horizon in sorted(single["horizon"].unique()):
            sub = single[(single["arm"] == arm) & (single["horizon"] == horizon)]
            if sub.empty:
                continue
            row = {
                "arm": arm,
                "horizon": horizon,
                "mae": sub["mae"].mean(),
                "peak_mae": sub["peak_mae"].mean(),
                "mae_2017": sub.loc[sub["fold_id"] == 1, "mae"].mean(),
                "seed_sd": sub.groupby("seed")["mae"].mean().std(ddof=1)
                if sub["seed"].nunique() > 1 else np.nan,
                "ensemble_mae": ens.loc[(ens["arm"] == arm) & (ens["horizon"] == horizon), "mae"].mean(),
            }
            if arm not in ("persistence", CONTROL_ARM) and (CONTROL_ARM, horizon) in fold_mean.index.droplevel(2):
                a = fold_mean.loc[(arm, horizon)]
                b = fold_mean.loc[(CONTROL_ARM, horizon)].reindex(a.index)
                delta = a - b
                row["delta_vs_no_clip"] = delta.mean()
                row["folds_improved"] = f"{int((delta < 0).sum())}/{len(delta)}"
                if len(delta) > 2 and delta.std() > 0:
                    t, p = stats.ttest_rel(a, b)
                    row["paired_t"], row["paired_p"] = t, p
            rows.append(row)
    return pd.DataFrame(rows)


def stability_summary(runs: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    cols = [
        "best_epoch", "epochs_run", "grad_norm_median", "grad_norm_p99",
        "grad_norm_max", "spike_ratio", "spike_steps", "nonfinite_steps",
        "clipped_fraction", "val_up_fraction", "val_jitter", "train_jitter",
        "best_val_loss", "seconds", "inferred",
    ]
    table = runs.groupby("arm")[cols].mean().reindex(arms)
    table["inferred"] = table["inferred"].astype(float)
    table["grad_norm_max"] = runs.groupby("arm")["grad_norm_max"].max().reindex(arms)
    return table.rename(columns={"grad_norm_max": "grad_norm_max (worst run)"})


def norm_quantiles(steps: pd.DataFrame) -> pd.DataFrame:
    """Quantiles of the control's step norms per fold: how big do gradients get?"""

    control = steps[steps["arm"] == CONTROL_ARM]
    q = control.groupby("fold_id")["grad_norm"].quantile([0.5, 0.9, 0.99, 1.0]).unstack()
    q.columns = ["p50", "p90", "p99", "max"]
    q["share > 0.25"] = control.groupby("fold_id")["grad_norm"].apply(lambda g: (g > 0.25).mean())
    q["share > 1"] = control.groupby("fold_id")["grad_norm"].apply(lambda g: (g > 1.0).mean())
    return q.reset_index()


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".2f") -> str:
    frame = frame.copy()
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "|" + "---|" * len(frame.columns)
    lines = [header, rule]
    for row in frame.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("—" if np.isnan(value) else format(value, floatfmt))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_norms(steps: pd.DataFrame, epochs: pd.DataFrame, arms: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Validated categorical order (dataviz reference palette); line style is a
    # second encoding so arms never rely on colour alone.
    colors = dict(zip(arms, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
    styles = dict(zip(arms, ["-", "--", "-.", ":", (0, (5, 1))]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    # Each run is scaled by its own epoch-1 validation loss (folds have
    # different NLL levels), and an epoch is drawn only while at least half of
    # an arm's runs are still training, so early stopping does not show up as
    # jumps in the median.
    epochs = epochs.copy()
    run_key = ["arm", "fold_id", "seed"]
    epochs["val_rel"] = epochs["val_loss"] / epochs.groupby(run_key)["val_loss"].transform("first")

    def band(arm: str, column: str) -> pd.DataFrame:
        sub = epochs[epochs["arm"] == arm]
        n_runs = sub.groupby(run_key).ngroups
        grouped = sub.groupby("epoch")[column]
        out = pd.DataFrame({
            "p25": grouped.quantile(0.25), "p50": grouped.median(),
            "p75": grouped.quantile(0.75), "n": grouped.size(),
        })
        return out[out["n"] >= n_runs / 2]

    panels = [
        (axes[0], "grad_norm_mean", "Pre-clip gradient norm (mean per epoch)",
         "Gradient norm by epoch"),
        (axes[1], "val_rel", "Validation NLL / epoch-1 validation NLL",
         "Validation loss by epoch"),
    ]
    for ax, column, ylabel, title in panels:
        for arm in arms:
            b = band(arm, column)
            ax.fill_between(b.index, b["p25"], b["p75"], color=colors[arm], alpha=0.12, lw=0)
            ax.plot(b.index, b["p50"], color=colors[arm], ls=styles[arm], lw=2, label=arm)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title + " — median and IQR over runs", loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=9, loc="lower right" if column == "grad_norm_mean" else "upper right")

    ax = axes[0]
    for arm in arms:
        max_norm = epochs.loc[epochs["arm"] == arm, "max_norm"].iloc[0]
        if pd.notna(max_norm):
            ax.axhline(max_norm, color=colors[arm], lw=0.8, alpha=0.6)
            ax.annotate(f"max_norm={max_norm:g}", (0.01, max_norm), xycoords=("axes fraction", "data"),
                        ha="left", va="bottom", fontsize=8, color="#555555")
    ax.set_yscale("log")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(outputs: dict, summary: pd.DataFrame, stability: pd.DataFrame,
                 args: argparse.Namespace, arms: list[str], path: Path) -> None:
    runs = outputs["runs"]
    fmt = summary.copy()
    for col in ("paired_p",):
        if col in fmt:
            fmt[col] = fmt[col].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
    stab = stability.reset_index().rename(columns={"index": "arm"})

    lines = [
        "# Gradient clipping ablation — multi-horizon NegBin (script 37)",
        "",
        f"Model: script 32, variant `{MODEL_OVERRIDES['variant']}`, backbone "
        f"`{MODEL_OVERRIDES['backbone']}`, parallel heads. Folds "
        f"{sorted(runs['fold_id'].unique().tolist())}, seeds {args.seeds}, "
        f"arms {arms}. {len(runs)} runs in {outputs['wall_seconds'] / 60:.1f} min "
        f"({args.workers} workers x {args.threads} threads).",
        "",
        "Gradient norms are the global L2 norm **before** clipping, logged at every",
        "optimiser step, so the arms are comparable on the same quantity.",
        "",
        "## Forecast skill (headline folds, single-seed mean)",
        "",
        markdown_table(fmt),
        "",
        "`delta_vs_no_clip` and the paired t-test use per-fold seed-mean MAE.",
        "",
        "## Training stability (mean over runs)",
        "",
        markdown_table(stab, ".3f"),
        "",
        "- `spike_ratio`: largest step norm / median step norm of the run.",
        "- `spike_steps`: steps whose norm exceeded 10x the run median.",
        "- `clipped_fraction`: share of steps whose pre-clip norm exceeded max_norm.",
        "- `val_up_fraction` / `val_jitter`: share of epochs where validation loss",
        "  rose, and median |epoch-to-epoch change| in validation loss.",
        "- `inferred`: share of runs copied from the control because the control's",
        "  largest step norm was below the threshold (clip provably never fires).",
        "",
        "## Pre-clip gradient norm distribution (no_clip arm, all steps)",
        "",
        markdown_table(norm_quantiles(outputs["steps"]), ".3f"),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--folds", type=int, nargs="+", default=list(HEADLINE_FOLDS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-norms", type=parse_max_norm, nargs="+",
                        default=list(DEFAULT_MAX_NORMS),
                        help="Clip thresholds to compare; 'none' is the unclipped control")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--threads", type=int, default=2, help="Torch threads per worker")
    parser.add_argument("--no-skip", action="store_true",
                        help="Train clipped arms even when the control proves they are identical")
    parser.add_argument("--quick", action="store_true",
                        help=f"Smoke test: folds {QUICK_FOLDS}, 1 seed")
    parser.add_argument("--tag", default="", help="Suffix for output file names")
    args = parser.parse_args(argv)
    if args.quick:
        args.folds, args.seeds = list(QUICK_FOLDS), 1
    if None not in args.max_norms:
        parser.error("--max-norms must include 'none' (the control arm)")
    return args


def main(argv=None) -> int:
    args = parse_arguments(argv)
    load_modules()
    arms = [arm_name(m) for m in args.max_norms]
    horizons = tuple(args.horizons)

    outputs = run_experiment(args.folds, args.seeds, args.max_norms, horizons,
                             args.workers, args.threads, skip_identical=not args.no_skip)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ("_quick" if args.quick else "")
    for name in ("metrics", "runs", "epochs", "steps"):
        outputs[name].to_csv(RESULTS_DIR / f"grad_clipping_{name}{tag}.csv", index=False)

    summary = summarise(outputs["metrics"], outputs["runs"], arms)
    stability = stability_summary(outputs["runs"], arms)
    write_report(outputs, summary, stability, args, arms,
                 RESULTS_DIR / f"grad_clipping_report{tag}.md")
    plot_norms(outputs["steps"], outputs["epochs"], arms,
               FIGURES_DIR / f"grad_clipping_norms{tag}.png")

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("\nForecast skill (headline folds):")
        print(summary.round(3).to_string(index=False))
        print("\nTraining stability:")
        print(stability.round(3).to_string())
    print(f"\nWall time {outputs['wall_seconds'] / 60:.1f} min. "
          f"Report: {RESULTS_DIR / f'grad_clipping_report{tag}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

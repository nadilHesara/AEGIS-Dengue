"""
Pilots for horizon-utility estimation (plan §6.2, declared 2026-09-30).

Inside one outer fold's training period, an inner block gives three years:
pilot training, early stopping (checkpoint selection) and utility scoring.

Primary utility (inference perturbation). Train the equal-weight two-branch
NB model on real climate and keep the checkpoint chosen on the
early-stopping year. Freeze it, then score the utility year twice: with real
climate, and with donor climate (5 predeclared permutation seeds). No
shuffled model is ever trained.

Donor rule. For recipient origin r in the utility year and district n, the
entire 37-week x 7-variable climate window of (r, n) is replaced by the
window of the **same district** at a donor origin o, where
    - o <= the pilot-training cut-off (permitted training history, which is
      necessarily before r, so available by the recipient's origin);
    - o has a full 37-week window;
    - o's start week lies within +-SEASON_WEEKS of r's week of year
      (circular), i.e. a comparable season;
    - o falls in a different calendar year from r.
The window moves as one block, so its internal time order and
cross-variable structure are preserved. Cases, targets, masks and anchors
are never touched. The model reads no climate-derived features: the rolling
means are excluded channels, so there is nothing to recompute. Donors are
drawn independently per (permutation, origin, district) and saved.

Sensitivity utility. A size-matched case-only pilot (`CaseOnlyNB`; hidden
size chosen to match the parameter count) is trained on the same data and
budget and scored on the same cells.

Matched cells. Every arm is scored on the cells whose recipient (origin,
district) had an eligible donor under all five permutations.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.climate_horizon import (
    HORIZONS, WINDOW, build_arrays, development_last_period, scaled_features,
)
from src.models.climate_ablation import week_of_year
from src.training import climate_horizon as trainer

PERMUTATION_SEEDS = (11, 23, 37, 41, 53)   # declared before any pilot was run
PILOT_SEEDS = (0, 1)
SEASON_WEEKS = 2
HOLDOUT_FIRST_PERIOD = 994                 # 2026-01-05


def fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def array_fingerprint(*arrays) -> str:
    digest = hashlib.sha256()
    for a in arrays:
        digest.update(np.ascontiguousarray(a).tobytes())
    return digest.hexdigest()[:16]


def eligible_donors(tensors: dict, calendar, block: dict) -> dict[int, np.ndarray]:
    """Recipient origin index -> donor origin indices (comparable season, training history)."""

    period_id = tensors["period_id"]
    weeks = week_of_year(tensors["start_date"])
    years = calendar.sort_values("period_id")["year"].to_numpy()
    candidates = np.nonzero((np.arange(len(period_id)) >= WINDOW - 1)
                            & (period_id <= block["train_end_period"]))[0]
    out = {}
    recipients = np.nonzero((period_id >= block["test_start_period"] - max(HORIZONS))
                            & (period_id <= block["test_end_period"]))[0]
    for r in recipients:
        gap = np.abs(weeks[candidates] - weeks[r])
        gap = np.minimum(gap, 52 - gap)
        ok = (gap <= SEASON_WEEKS) & (years[candidates] != years[r]) & (candidates < r)
        out[int(r)] = candidates[ok]
    return out


def draw_donors(recipient_index: np.ndarray, donors: dict[int, np.ndarray], n_nodes: int,
                perm_seed: int) -> np.ndarray:
    """Donor origin index per (recipient, district); -1 where none is eligible."""

    rng = np.random.default_rng(perm_seed)
    chosen = np.full((len(recipient_index), n_nodes), -1, dtype=np.int64)
    for i, r in enumerate(recipient_index):
        pool = donors.get(int(r), np.empty(0, dtype=np.int64))
        if len(pool):
            chosen[i] = rng.choice(pool, size=n_nodes, replace=True)
    return chosen


def donor_climate(x_scaled: np.ndarray, climate_idx: list[int], chosen: np.ndarray,
                  original: np.ndarray) -> np.ndarray:
    """Replace each (origin, district) climate window with its donor's whole window."""

    out = original.copy()
    steps = np.arange(-WINDOW + 1, 1)
    for n in range(chosen.shape[1]):
        ok = chosen[:, n] >= 0
        idx = chosen[ok, n][:, None] + steps[None, :]                    # [o, WINDOW]
        out[ok, :, n, :] = x_scaled[idx, n][..., climate_idx]
    return out


def cell_frame(arm: str, perm: int, seed: int, part: dict, prediction: np.ndarray,
               mu: np.ndarray, alpha: np.ndarray, matched: np.ndarray) -> pd.DataFrame:
    o, n, h = np.nonzero(part["mask"] == 1)
    return pd.DataFrame({
        "arm": arm, "permutation": perm, "seed": seed,
        "origin_period_id": part["origin_period_id"][o],
        "target_period_id": part["target_period_id"][o, h],
        "node_id": n, "horizon": np.asarray(HORIZONS)[h],
        "prediction": prediction[o, n, h], "nb_mu": mu[o, n, h], "nb_alpha": alpha[o, n, h],
        "actual": part["y"][o, n, h], "observed": part["mask"][o, n, h].astype(np.int8),
        "matched": matched[o, n],
    })


def run_unit(out_dir: Path, fold: dict, block: dict, seed: int, tensors, months, calendar, fm,
             channels, device, config: dict, provenance: dict) -> Path:
    """One (outer fold, inner block, seed): pilots, frozen scoring, perturbations. Resumable."""

    tag = f"fold{fold['fold_id']}_block{block['block']}_seed{seed}"
    path = out_dir / f"{tag}.parquet"
    if path.exists() and (out_dir / f"{tag}.json").exists():
        return path

    started = time.perf_counter()
    arrays = build_arrays(tensors, months, block, fm, channels,
                          last_target_period=development_last_period(calendar))
    part = arrays["test"]
    for split in arrays.values():
        live = split["target_period_id"][:, None, :].repeat(split["mask"].shape[1], 1)[split["mask"] == 1]
        if live.size and live.max() >= HOLDOUT_FIRST_PERIOD:
            raise AssertionError("a 2026 target reached the pilot")

    x_scaled = scaled_features(tensors, months, block, fm)
    period_index = {int(p): i for i, p in enumerate(tensors["period_id"])}
    recipients = np.array([period_index[int(p)] for p in part["origin_period_id"]])

    donors_path = out_dir / f"donors_fold{fold['fold_id']}_block{block['block']}.parquet"
    donors = eligible_donors(tensors, calendar, block)
    draws = {p: draw_donors(recipients, donors, part["mask"].shape[1], p) for p in PERMUTATION_SEEDS}
    matched = np.all(np.stack([d >= 0 for d in draws.values()]), axis=0)           # [o, N]
    if not donors_path.exists():
        rows = [pd.DataFrame({"permutation": p, "recipient_origin_period_id": np.repeat(part["origin_period_id"], d.shape[1]),
                              "node_id": np.tile(np.arange(d.shape[1]), len(d)),
                              "donor_origin_period_id": np.where(d.reshape(-1) >= 0, tensors["period_id"][np.maximum(d.reshape(-1), 0)], -1)})
                for p, d in draws.items()]
        pd.concat(rows, ignore_index=True).to_parquet(donors_path, index=False)

    frames, timings = [], {}
    t0 = time.perf_counter()
    model, info = trainer.train(arrays, config, seed, device, weights=trainer.UNIFORM_WEIGHTS)
    timings["climate_pilot_train_s"] = time.perf_counter() - t0
    real = trainer.predict(model, part, device)
    frames.append(cell_frame("real", -1, seed, part, real["prediction"], real["mu"], real["alpha"], matched))
    for p, chosen in draws.items():
        perturbed = donor_climate(x_scaled, channels.climate, chosen, part["X_climate"])
        shuffled = trainer.predict(model, part, device, x_climate=perturbed)
        frames.append(cell_frame("shuffled", p, seed, part, shuffled["prediction"], shuffled["mu"],
                                 shuffled["alpha"], matched))

    t0 = time.perf_counter()
    case_model, case_info = trainer.train(arrays, {**config, "case_only": True}, seed, device)
    timings["case_pilot_train_s"] = time.perf_counter() - t0
    case = trainer.predict(case_model, part, device)
    frames.append(cell_frame("case_only", -1, seed, part, case["prediction"], case["mu"], case["alpha"], matched))

    frame = pd.concat(frames, ignore_index=True)
    frame.insert(0, "fold_id", fold["fold_id"])
    frame.insert(1, "block", block["block"])
    frame.to_parquet(path, index=False)

    meta = {
        "fold_id": fold["fold_id"], "block": block, "seed": seed,
        "permutation_seeds": list(PERMUTATION_SEEDS), "season_weeks": SEASON_WEEKS,
        "climate_pilot": {k: float(v) for k, v in info.items() if k != "gradient_log"},
        "case_pilot": {k: float(v) for k, v in case_info.items() if k != "gradient_log"},
        "runtime_s": {**timings, "total_s": time.perf_counter() - started},
        "cells_scored": int((part["mask"] == 1).sum()),
        "matched_origin_districts": int(matched.sum()), "origin_districts": int(matched.size),
        "fingerprints": {
            **provenance,
            "block": fingerprint(block),
            "train_arrays": array_fingerprint(arrays["train"]["X_case"], arrays["train"]["X_climate"],
                                              arrays["train"]["y"], arrays["train"]["mask"]),
            "utility_cells": array_fingerprint(part["target_period_id"], part["mask"], part["y"]),
            "config": fingerprint(config),
        },
        "device": str(device),
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path


def utility_summary(frames: pd.DataFrame) -> pd.DataFrame:
    """Per fold/block/seed/horizon: MAE of each arm on matched cells, and the two utilities."""

    live = frames[(frames["observed"] == 1) & frames["matched"]].copy()
    live["abs_error"] = (live["prediction"] - live["actual"]).abs()
    keys = ["fold_id", "block", "seed", "horizon"]
    real = live[live["arm"] == "real"].groupby(keys)["abs_error"].mean().rename("mae_real")
    case = live[live["arm"] == "case_only"].groupby(keys)["abs_error"].mean().rename("mae_case_only")
    shuffled = live[live["arm"] == "shuffled"].groupby(keys + ["permutation"])["abs_error"].mean()
    table = pd.concat([real, case, shuffled.groupby(keys).mean().rename("mae_shuffled"),
                       shuffled.groupby(keys).std().rename("mae_shuffled_perm_sd")], axis=1).reset_index()
    table["cells"] = live[live["arm"] == "real"].groupby(keys).size().to_numpy()
    table["shuffle_gain"] = (table["mae_shuffled"] - table["mae_real"]) / table["mae_shuffled"]
    table["case_only_gain"] = (table["mae_case_only"] - table["mae_real"]) / table["mae_case_only"]
    return table

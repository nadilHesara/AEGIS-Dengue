"""
Fold-specific climate-gradient weights from the inner utility pilots.

Inputs: only the pilots' utility-scoring predictions
(results/climate_horizon/pilots/). No test year, no benchmark weights.

Primary (shuffle reliance of a frozen equal-weight pilot)
    g_h = MAE_shuffled_h - MAE_real_h      (MAE_shuffled = mean over the 5 declared permutations)
    u_h = max(g_h, 0)

Sensitivity (incremental value over a size-matched case-only pilot; separate arm)
    u_case_h = max(0, (MAE_case_h - MAE_real_h) / (MAE_case_h + 1e-8))

Weights (either utility)
    sum(u) > 0:  q_h = H * u_h / sum(u);  w_h = (1 - rho) + rho * q_h,   rho = 0.95 (declared)
    otherwise:   w = [1, 1, 1, 1]
    mean(w) = 1 exactly in both cases.

Fixed aggregation (declared here, before any weight was computed)
    1. Per unit (block, seed) and horizon: MAE over the matched observed cells.
       MAE_shuffled is the mean of the five per-permutation MAEs.
    2. Average each MAE over seeds within a block.
    3. Average over inner blocks with equal weight.
    4. Form g_h (or u_case_h) from the aggregated MAEs.
    Per-unit signed gains are reported for variability. They are not used
    to form the weights.

Missing evidence: a fold with no pilot units gets status "missing_evidence"
and no weight file. Callers must not train the weighted arm for it.

What the weights measure. g_h is how much a trained model **relies** on
correctly aligned weather at horizon h (a perturbation of a frozen model).
It is not the incremental or causal value of climate: a model can rely on
an input that a model without it could replace (see the case-only
utility), and the shuffle removes weather-time alignment, not weather's
causal effect. Mean-one weights keep the average weight fixed. They do not
guarantee identical gradient norms or learning dynamics, because the
per-horizon losses differ in size and in direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RHO = 0.95
N_HORIZONS = 4
EPSILON = 1e-8
RULE_VERSION = "climate-horizon-weights-v1 (2026-09-30)"


def weights_from_utilities(utilities, rho: float = RHO) -> tuple[np.ndarray, str]:
    """Mixed, mean-one weights from non-negative utilities. Returns (w, status)."""

    u = np.asarray(utilities, dtype=np.float64)
    if u.ndim != 1 or len(u) == 0:
        raise ValueError("utilities must be a non-empty vector")
    if np.isnan(u).any():
        return np.ones(len(u)), "missing_evidence"
    if (u < 0).any():
        raise ValueError("utilities must be clipped at zero before weighting")
    total = u.sum()
    if total <= 0:
        return np.ones(len(u)), "all_nonpositive_uniform"
    q = len(u) * u / total
    return (1.0 - rho) + rho * q, "informed"


def unit_errors(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (fold, block, seed, horizon): MAE of real / shuffled (per permutation) / case-only."""

    live = frame[(frame["observed"] == 1) & frame["matched"]].copy()
    live["abs_error"] = (live["prediction"] - live["actual"]).abs()
    keys = ["fold_id", "block", "seed", "horizon"]
    real = live[live["arm"] == "real"].groupby(keys)["abs_error"].agg(["mean", "size"])
    real.columns = ["mae_real", "cells"]
    case = live[live["arm"] == "case_only"].groupby(keys)["abs_error"].mean().rename("mae_case_only")
    per_perm = live[live["arm"] == "shuffled"].groupby(keys + ["permutation"])["abs_error"].mean()
    shuffled = per_perm.groupby(keys).agg(["mean", "std", "count"])
    shuffled.columns = ["mae_shuffled", "mae_shuffled_perm_sd", "permutations"]
    table = pd.concat([real, case, shuffled], axis=1).reset_index()
    table["gain_shuffle"] = table["mae_shuffled"] - table["mae_real"]
    table["gain_case_only_rel"] = (table["mae_case_only"] - table["mae_real"]) / (table["mae_case_only"] + EPSILON)
    return table


def aggregate(units: pd.DataFrame) -> pd.DataFrame:
    """Seeds within block, then blocks equally; utilities from the aggregated MAEs."""

    cols = ["mae_real", "mae_shuffled", "mae_case_only"]
    by_block = units.groupby(["fold_id", "block", "horizon"])[cols].mean()
    agg = by_block.groupby(["fold_id", "horizon"]).mean()
    spread = units.groupby(["fold_id", "horizon"]).agg(
        gain_shuffle_unit_mean=("gain_shuffle", "mean"), gain_shuffle_unit_sd=("gain_shuffle", "std"),
        gain_shuffle_unit_min=("gain_shuffle", "min"), gain_shuffle_unit_max=("gain_shuffle", "max"),
        gain_shuffle_units_positive=("gain_shuffle", lambda g: int((g > 0).sum())),
        gain_case_rel_unit_sd=("gain_case_only_rel", "std"),
        gain_case_rel_units_positive=("gain_case_only_rel", lambda g: int((g > 0).sum())),
        units=("gain_shuffle", "size"), cells=("cells", "sum"))
    agg = agg.join(spread).reset_index()
    agg["gain_shuffle"] = agg["mae_shuffled"] - agg["mae_real"]
    agg["u_shuffle"] = agg["gain_shuffle"].clip(lower=0)
    agg["gain_case_only_rel"] = (agg["mae_case_only"] - agg["mae_real"]) / (agg["mae_case_only"] + EPSILON)
    agg["u_case_only"] = agg["gain_case_only_rel"].clip(lower=0)
    return agg


def distance_from_uniform(w) -> dict[str, float]:
    w = np.asarray(w, dtype=np.float64)
    return {"l1": float(np.abs(w - 1).sum()), "max_abs": float(np.abs(w - 1).max()),
            "max_over_min": float(w.max() / w.min())}

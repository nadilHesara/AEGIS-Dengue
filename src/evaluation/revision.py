"""
Statistics and calibration added in the 29 September 2026 revision of the
long-horizon benchmark (scripts/34 calls these; tests/test_revision.py checks
them).

    holm                  Holm step-down adjustment over a declared family.
    block_bootstrap       time-block bootstrap of a paired MAE difference:
                          blocks of consecutive target weeks are resampled
                          within each test year, jointly across all districts,
                          because districts and overlapping horizons in the
                          same weeks are not independent samples.
    rolling_conformal     split-conformal intervals whose calibration set is
    adaptive_conformal    the latest out-of-sample residuals *already observed
                          at the forecast origin* (a horizon-h residual enters
                          only once its target week has been reported), and
                          the adaptive version of Gibbs and Candes (2021),
                          which moves each interval's miscoverage level online.
    nb_diagnostics        dispersion, zero-count and district-level
                          calibration of Negative Binomial forecasts.
    select_by_validation  pick one arm per fold from a set of variants by a
                          validation-year score, never the test year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.long_horizon import HEADLINE_FOLDS, QUANTILES, quantile_column, weighted_interval_score

QCOLS = [quantile_column(q) for q in QUANTILES]
INTERVALS = ((0.5, 2, 4), (0.8, 1, 5), (0.95, 0, 6))  # (coverage, lower col, upper col)


# ---------------------------------------------------------------------------
# Multiple comparisons and resampling
# ---------------------------------------------------------------------------

def holm(p_values) -> np.ndarray:
    """Holm-adjusted p-values (NaN entries are left NaN and not counted)."""

    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.where(~np.isnan(p))[0]
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[index]))
        out[index] = running
    return out


def block_bootstrap(errors: pd.DataFrame, block: int = 4, n_boot: int = 2000,
                    seed: int = 0) -> dict:
    """Paired time-block bootstrap of a headline MAE difference.

    `errors` has one row per cell with columns fold_id, target_period_id and
    `diff` = |error of A| - |error of B| (seed-averaged absolute errors, so the
    estimand is the same seed-mean MAE the headline tables report). The
    statistic is the mean over folds of each fold's mean `diff`, which is the
    headline difference. Within each fold, consecutive blocks of `block`
    target weeks are resampled with replacement; a block carries every
    district's cells for those weeks, so spatial correlation is kept.
    """

    rng = np.random.default_rng(seed)
    per_fold = []
    for fold_id, group in errors.groupby("fold_id"):
        weeks = np.sort(group["target_period_id"].unique())
        week_sum = group.groupby("target_period_id")["diff"].sum().reindex(weeks).to_numpy()
        week_n = group.groupby("target_period_id")["diff"].size().reindex(weeks).to_numpy()
        starts = np.arange(0, len(weeks), block)
        per_fold.append((week_sum, week_n, starts))

    observed = np.mean([s.sum() / n.sum() for s, n, _ in per_fold])
    draws = np.empty(n_boot)
    for b in range(n_boot):
        values = []
        for week_sum, week_n, starts in per_fold:
            chosen = rng.choice(starts, size=len(starts), replace=True)
            total, count = 0.0, 0.0
            for start in chosen:
                total += week_sum[start : start + block].sum()
                count += week_n[start : start + block].sum()
            values.append(total / count)
        draws[b] = np.mean(values)
    low, high = np.quantile(draws, [0.025, 0.975])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return {"delta": float(observed), "ci_low": float(low), "ci_high": float(high),
            "p_boot": float(min(1.0, p))}


# ---------------------------------------------------------------------------
# Conformal intervals under drift
# ---------------------------------------------------------------------------

def _interval_quantiles(point_log: np.ndarray, residuals: np.ndarray,
                        miscoverage: dict[float, float]) -> np.ndarray:
    """Seven log-scale quantiles from a residual sample and per-interval levels."""

    q = np.empty((len(point_log), len(QUANTILES)))
    q[:, 3] = point_log + np.quantile(residuals, 0.5)
    for nominal, lower, upper in INTERVALS:
        a = float(np.clip(miscoverage[nominal], 1e-3, 0.999))
        q[:, lower] = point_log + np.quantile(residuals, a / 2)
        q[:, upper] = point_log + np.quantile(residuals, 1 - a / 2)
    return np.sort(q, axis=1)


def online_conformal(group: pd.DataFrame, horizon: int, window: int = 52,
                     gamma: float = 0.0, min_calibration: int = 200) -> pd.DataFrame:
    """Rolling (gamma = 0) or adaptive (gamma > 0) conformal quantiles for one
    method and horizon.

    `group` holds the method's out-of-sample test cells for consecutive folds
    (columns fold_id, target_period_id, node_id, prediction, actual, observed,
    threshold). For target week T the forecast was issued at T - h, so the
    calibration sample is the log residuals of cells with target in
    [T - h - window + 1, T - h] that were observed -- nothing later. With
    gamma > 0 each interval's miscoverage level a is updated after every week
    whose outcomes have become observable: a <- a + gamma * (nominal_a - miss),
    miss being that week's miscoverage fraction across districts (Gibbs and
    Candes 2021), applied with the h-week reporting delay.
    """

    group = group.sort_values(["target_period_id", "node_id"])
    weeks = np.sort(group["target_period_id"].unique())
    residual = np.log1p(group["actual"].to_numpy()) - np.log1p(group["prediction"].to_numpy())
    observed = group["observed"].to_numpy() == 1
    target = group["target_period_id"].to_numpy()
    by_week = {w: np.where(target == w)[0] for w in weeks}

    level = {nominal: 1 - nominal for nominal, _, _ in INTERVALS}
    pending: list[tuple[int, dict]] = []  # (week becomes observable, per-interval miss)
    frames = []
    for week in weeks:
        issue = week - horizon
        # Update the levels with every earlier week whose outcome is now known.
        ready = [item for item in pending if item[0] <= issue]
        pending = [item for item in pending if item[0] > issue]
        for _, miss in sorted(ready, key=lambda item: item[0]):
            for nominal, _, _ in INTERVALS:
                level[nominal] += gamma * ((1 - nominal) - miss[nominal])

        calibration = observed & (target <= issue) & (target > issue - window)
        rows = by_week[week]
        if calibration.sum() < min_calibration:
            continue
        point_log = np.log1p(group["prediction"].to_numpy()[rows])
        q = _interval_quantiles(point_log, residual[calibration], level)
        frame = group.iloc[rows][["method", "fold_id", "horizon", "target_period_id", "node_id",
                                  "actual", "observed", "threshold"]].copy()
        frame[QCOLS] = np.clip(np.expm1(q), 0.0, None)
        frames.append(frame)

        seen = observed[rows]
        if seen.any():
            actual = group["actual"].to_numpy()[rows][seen]
            values = np.expm1(q[seen])
            miss = {nominal: float(np.mean((actual < values[:, lower]) | (actual > values[:, upper])))
                    for nominal, lower, upper in INTERVALS}
            pending.append((week, miss))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def interval_scores(quantiles: pd.DataFrame, by=("method", "variant", "horizon", "fold_id")) -> pd.DataFrame:
    """WIS, coverage at 50/80/95 and mean 95% width, per group, observed cells."""

    rows = []
    observed = quantiles[quantiles["observed"] == 1]
    for keys, group in observed.groupby(list(by)):
        q = group[QCOLS].to_numpy()
        actual = group["actual"].to_numpy()
        record = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        wis = weighted_interval_score(actual, q)
        record["wis"] = float(wis.mean())
        peak = actual >= group["threshold"].to_numpy()
        record["peak_wis"] = float(wis[peak].mean()) if peak.any() else np.nan
        for nominal, lower, upper in INTERVALS:
            record[f"cov{int(nominal * 100)}"] = float(np.mean((actual >= q[:, lower]) & (actual <= q[:, upper])))
        record["width95"] = float(np.mean(q[:, 6] - q[:, 0]))
        record["width80"] = float(np.mean(q[:, 5] - q[:, 1]))
        record["cells"] = len(group)
        rows.append(record)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Negative Binomial diagnostics
# ---------------------------------------------------------------------------

def nb_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    """Calibration checks for NB forecasts on observed headline test cells.

    `frame` has per-seed rows with nb_mu, nb_alpha, quantile columns, actual,
    threshold. Returned per method and horizon:
      - mean dispersion alpha (median over cells);
      - predicted vs observed share of zero counts (P(Y=0) = (1 + alpha mu)^(-1/alpha));
      - 80% and 95% coverage overall, on peak cells and on 2017;
      - the spread of 80% coverage across districts (min, max).
    """

    rows = []
    data = frame[(frame["observed"] == 1) & frame["fold_id"].isin(HEADLINE_FOLDS)
                 & (frame["split"] == "test")]
    for (method, horizon), group in data.groupby(["method", "horizon"]):
        mu = group["nb_mu"].to_numpy(dtype=float)
        alpha = np.maximum(group["nb_alpha"].to_numpy(dtype=float), 1e-6)
        p_zero = np.power(1.0 + alpha * mu, -1.0 / alpha)
        actual = group["actual"].to_numpy()
        q = group[QCOLS].to_numpy()
        inside80 = (actual >= q[:, 1]) & (actual <= q[:, 5])
        inside95 = (actual >= q[:, 0]) & (actual <= q[:, 6])
        peak = actual >= group["threshold"].to_numpy()
        epidemic = group["fold_id"].to_numpy() == 1
        district = pd.Series(inside80).groupby(group["node_id"].to_numpy()).mean()
        rows.append({
            "method": method, "horizon": horizon,
            "alpha_median": float(np.median(alpha)),
            "zero_share_predicted": float(p_zero.mean()),
            "zero_share_observed": float((actual == 0).mean()),
            "cov80": float(inside80.mean()), "cov95": float(inside95.mean()),
            "cov80_peak": float(inside80[peak].mean()) if peak.any() else np.nan,
            "cov95_peak": float(inside95[peak].mean()) if peak.any() else np.nan,
            "cov80_2017": float(inside80[epidemic].mean()) if epidemic.any() else np.nan,
            "cov95_2017": float(inside95[epidemic].mean()) if epidemic.any() else np.nan,
            "cov80_district_min": float(district.min()),
            "cov80_district_max": float(district.max()),
        })
    return pd.DataFrame(rows)


def nb_validation_nll(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean NB negative log-likelihood on validation cells, per method and fold."""

    from scipy import stats

    data = frame[(frame["split"] == "val") & (frame["observed"] == 1)]
    rows = []
    for (method, fold_id), group in data.groupby(["method", "fold_id"]):
        mu = group["nb_mu"].to_numpy(dtype=float)
        alpha = np.maximum(group["nb_alpha"].to_numpy(dtype=float), 1e-6)
        r = 1.0 / alpha
        p = r / (r + mu)
        nll = -stats.nbinom.logpmf(group["actual"].to_numpy().round(), r, p)
        rows.append({"method": method, "fold_id": fold_id, "val_nll": float(np.mean(nll))})
    return pd.DataFrame(rows)


def select_by_validation(scores: pd.DataFrame, candidates: list[str]) -> dict[int, str]:
    """For each fold, the candidate with the lowest validation score."""

    subset = scores[scores["method"].isin(candidates)]
    return {int(fold_id): str(group.loc[group["val_nll"].idxmin(), "method"])
            for fold_id, group in subset.groupby("fold_id")}


# ---------------------------------------------------------------------------
# Distribution mixtures
# ---------------------------------------------------------------------------

def quantile_cdf(x_log: np.ndarray, q_log: np.ndarray, levels=QUANTILES) -> np.ndarray:
    """CDF of a quantile forecast at x (both on the log1p scale), per row.

    Piecewise linear between the forecast quantiles; the tails are closed
    linearly to 0 and 1 at two inter-quantile spacings beyond the outer
    quantiles. `q_log` is [cells, levels] sorted; `x_log` is [cells].
    """

    levels = np.asarray(levels, dtype=float)
    low = q_log[:, 0] - 2 * np.maximum(q_log[:, 1] - q_log[:, 0], 1e-3)
    high = q_log[:, -1] + 2 * np.maximum(q_log[:, -1] - q_log[:, -2], 1e-3)
    knots = np.column_stack([low, q_log, high])
    probs = np.concatenate([[0.0], levels, [1.0]])
    out = np.where(x_log >= knots[:, -1], 1.0, 0.0)
    for k in range(knots.shape[1] - 1):
        left, right = knots[:, k], knots[:, k + 1]
        inside = (x_log >= left) & (x_log < right)
        span = np.maximum(right - left, 1e-9)
        value = probs[k] + (probs[k + 1] - probs[k]) * (x_log - left) / span
        out = np.where(inside, value, out)
    return out


def nb_cdf(count: np.ndarray, mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    from scipy import stats

    r = 1.0 / np.maximum(alpha, 1e-6)
    return stats.nbinom.cdf(np.floor(count), r, r / (r + np.maximum(mu, 1e-6)))


def mixture_quantiles(nb_mu: np.ndarray, nb_alpha: np.ndarray, other_q: np.ndarray,
                      weight_nb: float = 0.5, levels=QUANTILES, iterations: int = 40) -> np.ndarray:
    """Quantiles of the mixture weight_nb * NB-seed-mixture + (1 - weight_nb) * other.

    `nb_mu`, `nb_alpha` are [cells, seeds] (an equal-weight mixture over seeds);
    `other_q` is [cells, levels] on the count scale. Solved by bisection on the
    log1p scale, so the result is the quantile of the mixed *distribution*, not
    an average of the members' quantiles.
    """

    other_log = np.log1p(np.sort(np.maximum(other_q, 0.0), axis=1))
    upper = np.log1p(np.maximum(nb_mu.max(axis=1) * 20 + 50, np.expm1(other_log[:, -1]) * 4 + 50))
    out = np.empty((len(other_q), len(levels)))
    for j, level in enumerate(levels):
        lo, hi = np.zeros(len(other_q)), upper.copy()
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            count = np.expm1(mid)
            f_nb = np.mean([nb_cdf(count, nb_mu[:, s], nb_alpha[:, s]) for s in range(nb_mu.shape[1])], axis=0)
            f = weight_nb * f_nb + (1 - weight_nb) * quantile_cdf(mid, other_log)
            below = f < level
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        out[:, j] = np.expm1(hi)
    return np.sort(out, axis=1)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

def compute_table(compute: pd.DataFrame) -> pd.DataFrame:
    """One row per arm: parameters, fits, training time, inference latency, memory."""

    rows = []
    for arm, group in compute.groupby("arm"):
        origins = group.filter(like="origins").sum(axis=1).sum()
        inference = group.filter(like="inference_seconds").sum(axis=1).sum()
        rows.append({
            "arm": arm,
            "parameters": int(group["parameters"].max()) if "parameters" in group and group["parameters"].notna().any() else np.nan,
            "fits": int(len(group)),
            "train_seconds_total": float(group["train_seconds"].sum()),
            "train_seconds_per_fit": float(group["train_seconds"].mean()),
            "inference_ms_per_origin": float(1000 * inference / origins) if origins else np.nan,
            "peak_gpu_mb": float(group["peak_gpu_mb"].max()) if "peak_gpu_mb" in group else np.nan,
            "hardware": str(group["hardware"].iloc[0]) if "hardware" in group else "",
        })
    return pd.DataFrame(rows)

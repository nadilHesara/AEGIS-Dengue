"""
Synthetic NB panels with a known climate delay, for end-to-end checks only.

For district n and week t:

    x[t, n, k]  AR(1) climate (rho 0.7) plus an annual cycle, 7 channels
    r[t, n]     = c_n + phi (r[t-1, n] - c_n) + beta * rain[t - DELAY, n]
    y[t, n]     ~ NB2(mean exp(r), alpha)

So cases depend on their own history (the case component) and on rainfall
DELAY weeks before the week being forecast. In target-relative terms, the
informative delay is d = DELAY for every horizon. With beta = 0 there is no
climate signal. The feature layout matches the extension's channel split:
cases_log1p, the 7 climate channels, doy_sin, doy_cos.

Success here says the code can learn and use a planted delayed signal. It
says nothing about dengue biology.
"""

from __future__ import annotations

import numpy as np

from src.models.climate_ablation import CLIMATE_FEATURES

DELAY = 8
NAMES = ["cases_log1p", *CLIMATE_FEATURES, "doy_sin", "doy_cos"]


def make_panel(beta: float, n_periods: int = 420, n_nodes: int = 6, alpha: float = 0.05,
               phi: float = 0.6, seed: int = 0) -> tuple[dict, np.ndarray, dict]:
    """Return (tensors, months, fold) in the layout `build_arrays` expects."""

    rng = np.random.default_rng(seed)
    t = np.arange(n_periods)
    season = np.sin(2 * np.pi * t / 52.18)
    k = len(CLIMATE_FEATURES)
    x = np.zeros((n_periods, n_nodes, k))
    noise = rng.normal(size=(n_periods, n_nodes, k))
    for i in range(1, n_periods):
        x[i] = 0.7 * x[i - 1] + noise[i]
    x = x / x.std(axis=0, keepdims=True) + 0.5 * season[:, None, None]
    rain = x[..., CLIMATE_FEATURES.index("rainfall_daily_mean_mm")]

    level = rng.uniform(2.0, 4.0, size=n_nodes)
    r = np.tile(level, (n_periods, 1)).astype(float)
    for i in range(1, n_periods):
        driver = rain[i - DELAY] if i >= DELAY else 0.0
        r[i] = level + phi * (r[i - 1] - level) + beta * driver
    mu = np.exp(r)
    size = 1.0 / alpha
    y = rng.negative_binomial(size, size / (size + mu)).astype(float)

    start = (np.datetime64("2000-01-03") + 7 * t).astype("datetime64[D]")
    doy = (start - start.astype("datetime64[Y]")).astype(int)
    angle = 2 * np.pi * doy / 365.25
    features = np.concatenate([np.log1p(y)[..., None], x,
                               np.broadcast_to(np.sin(angle)[:, None, None], (n_periods, n_nodes, 1)),
                               np.broadcast_to(np.cos(angle)[:, None, None], (n_periods, n_nodes, 1))], axis=-1)
    tensors = {"X": features.astype(np.float32), "y": y, "y_mask": np.ones_like(y, dtype=np.int8),
               "period_id": t + 1, "start_date": start, "feature_names": np.array(NAMES)}
    months = start.astype("datetime64[M]").astype(int) % 12 + 1
    fold = {"fold_id": 0, "train_end_period": 300, "val_start_period": 301, "val_end_period": 350,
            "test_start_period": 351, "test_end_period": 400, "fit_end_period": 300}
    return tensors, months, fold

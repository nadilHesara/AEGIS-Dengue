"""Tests for the 29 September 2026 revision: statistics, conformal, NB helpers,
climate controls and the onset task definitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.evaluation import onset, revision
from src.evaluation.long_horizon import (
    QUANTILES,
    nb_exceedance,
    nb_quantiles,
    quantile_column,
    quantile_exceedance,
    split_by_target,
)
from src.models.climate_ablation import (
    climate_indices,
    climatology_climate,
    delay_climate,
    drop_climate,
    week_of_year,
)

QCOLS = [quantile_column(q) for q in QUANTILES]


# --- statistics --------------------------------------------------------------

def test_holm_matches_hand_computation():
    adjusted = revision.holm([0.01, 0.04, 0.03, np.nan])
    # sorted 0.01, 0.03, 0.04 with m = 3: 0.03, 0.06, max(0.06, 0.04) = 0.06
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)
    assert np.isnan(adjusted[3])


def test_block_bootstrap_centre_and_interval():
    rng = np.random.default_rng(1)
    rows = []
    for fold_id in (1, 2, 3):
        for week in range(52):
            for node in range(5):
                rows.append({"fold_id": fold_id, "target_period_id": 100 * fold_id + week,
                             "diff": -1.0 + rng.normal(0, 0.5)})
    frame = pd.DataFrame(rows)
    result = revision.block_bootstrap(frame, n_boot=300)
    expected = frame.groupby("fold_id")["diff"].mean().mean()
    assert result["delta"] == pytest.approx(expected)
    assert result["ci_low"] < result["delta"] < result["ci_high"]
    assert result["ci_high"] < 0 and result["p_boot"] < 0.05


# --- conformal ---------------------------------------------------------------

def _conformal_group(n_weeks=120, later_shock=False):
    rows = []
    rng = np.random.default_rng(0)
    for week in range(n_weeks):
        for node in range(10):
            actual = float(rng.poisson(20))
            if later_shock and week >= 100:
                actual = 5000.0
            rows.append({"method": "m", "fold_id": 1, "horizon": 4, "target_period_id": week,
                         "node_id": node, "prediction": 20.0, "actual": actual,
                         "observed": 1, "threshold": 50.0})
    return pd.DataFrame(rows)


def test_online_conformal_uses_only_observed_residuals():
    calm = revision.online_conformal(_conformal_group(), horizon=4, min_calibration=50)
    shocked = revision.online_conformal(_conformal_group(later_shock=True), horizon=4, min_calibration=50)
    # Weeks <= 103 are forecast at origins <= 99: the shock (targets >= 100) is not yet observed.
    early_calm = calm[calm["target_period_id"] <= 103][QCOLS].to_numpy()
    early_shocked = shocked[shocked["target_period_id"] <= 103][QCOLS].to_numpy()
    np.testing.assert_allclose(early_calm, early_shocked)
    # From week 104 on, the week-100 shock is observable and widens the interval.
    late = shocked[shocked["target_period_id"] == 110]
    assert (late[QCOLS[-1]] > calm[calm["target_period_id"] == 110][QCOLS[-1]].to_numpy()).all()


def test_adaptive_conformal_widens_after_misses():
    group = _conformal_group(later_shock=True)
    rolling = revision.online_conformal(group, horizon=1, gamma=0.0, min_calibration=50)
    adaptive = revision.online_conformal(group, horizon=1, gamma=0.05, min_calibration=50)
    week = adaptive["target_period_id"].max()
    assert adaptive.loc[adaptive["target_period_id"] == week, QCOLS[-1]].mean() >= \
        rolling.loc[rolling["target_period_id"] == week, QCOLS[-1]].mean()


def test_interval_scores_coverage():
    frame = pd.DataFrame({"method": "m", "variant": "v", "horizon": 1, "fold_id": 1,
                          "observed": 1, "actual": [5.0, 50.0], "threshold": 40.0,
                          **{c: [q, q] for c, q in zip(QCOLS, (1, 2, 3, 5, 8, 10, 20))}})
    scores = revision.interval_scores(frame)
    assert scores["cov95"].iloc[0] == pytest.approx(0.5)
    assert scores["width95"].iloc[0] == pytest.approx(19.0)


# --- NB helpers --------------------------------------------------------------

def test_nb_quantiles_and_exceedance_match_scipy():
    mu, alpha = np.array([10.0, 200.0]), np.array([0.3, 0.1])
    r = 1 / alpha
    p = r / (r + mu)
    q = nb_quantiles(mu, alpha)
    np.testing.assert_allclose(q[:, QUANTILES.index(0.5)], stats.nbinom.ppf(0.5, r, p))
    np.testing.assert_allclose(nb_exceedance(mu, alpha, np.array([15.0, 250.0])),
                               stats.nbinom.sf([14, 249], r, p))


def test_nb_zero_residual_is_one_above_persistence():
    """The documented behaviour of the mean link: delta = 0 gives mu = y + 1."""

    import torch

    from src.models.negative_binomial import NegBinHead

    head = NegBinHead(hidden_dim=4, horizon=1)
    mu, alpha = head(torch.zeros(3, 4), torch.log1p(torch.tensor([[0.0, 9.0, 99.0]])))
    np.testing.assert_allclose(mu.detach().numpy().ravel(), [1.0, 10.0, 100.0], rtol=1e-6)
    assert torch.all(alpha > 0)


def test_quantile_exceedance_is_monotone():
    q = np.array([[1, 2, 4, 8, 16, 32, 64]], dtype=float)
    values = [quantile_exceedance(q, np.array([t]))[0] for t in (0.5, 3, 8, 30, 100)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert values[2] == pytest.approx(0.5, abs=1e-6)


def test_mixture_of_identical_components_recovers_them():
    mu = np.array([[30.0, 30.0]])
    alpha = np.array([[0.2, 0.2]])
    q_nb = nb_quantiles(mu[:, 0], alpha[:, 0])
    mixed = revision.mixture_quantiles(mu, alpha, q_nb, weight_nb=1.0)
    np.testing.assert_allclose(mixed[0, 3], q_nb[0, 3], atol=0.6)


# --- splits and climate controls ---------------------------------------------

def test_split_by_target():
    fold = {"train_end_period": 10, "val_start_period": 11, "val_end_period": 20,
            "test_start_period": 21, "test_end_period": 30}
    labels = split_by_target(np.array([[9, 12], [19, 22], [30, 31]]), fold)
    assert labels.tolist() == [["train", "val"], ["val", "test"], ["test", ""]]


def _tensors():
    periods = 160
    start = np.datetime64("2010-01-02") + np.arange(periods) * 7
    names = ["cases_log1p", "rainfall_daily_mean_mm", "rainfall_daily_mean_mm_roll4",
             "doy_sin", "thermal_suitability"]
    rng = np.random.default_rng(3)
    x = rng.normal(size=(periods, 2, len(names))).astype(np.float32)
    return {"X": x, "period_id": np.arange(1, periods + 1), "start_date": start,
            "feature_names": np.array(names, dtype=object)}


def test_climate_indices_include_v4_derived_channels():
    names = list(_tensors()["feature_names"])
    assert climate_indices(names) == [1, 2, 4]


def test_climatology_uses_training_years_only():
    tensors = _tensors()
    indices = climate_indices(list(tensors["feature_names"]))
    a = climatology_climate(tensors, indices, train_end_period=104)
    changed = dict(tensors)
    changed["X"] = tensors["X"].copy()
    changed["X"][104:, :, indices] += 100.0  # alter only validation/test-period weather
    b = climatology_climate(changed, indices, train_end_period=104)
    np.testing.assert_allclose(a["X"][..., indices], b["X"][..., indices])
    np.testing.assert_allclose(a["X"][..., 0], tensors["X"][..., 0])  # non-climate untouched
    week = week_of_year(tensors["start_date"])
    same_week = np.where(week == week[0])[0]
    assert np.allclose(a["X"][same_week, 0, 1], a["X"][same_week[0], 0, 1])


def test_drop_and_delay_climate():
    tensors = _tensors()
    indices = climate_indices(list(tensors["feature_names"]))
    dropped = drop_climate(tensors, indices)
    assert list(dropped["feature_names"]) == ["cases_log1p", "doy_sin"]
    delayed = delay_climate(tensors, indices, 1)
    np.testing.assert_allclose(delayed["X"][5, :, 1], tensors["X"][4, :, 1])
    assert np.isnan(delayed["X"][0, :, 1]).all()
    np.testing.assert_allclose(delayed["X"][..., 0], tensors["X"][..., 0])


# --- onset task --------------------------------------------------------------

def _series():
    # one district, threshold 10: quiet, quiet, ..., crossing at index 8
    y = np.array([[1], [2], [1], [3], [2], [1], [2], [4], [12], [15], [3], [2], [1], [2], [1], [1]], dtype=float)
    return y, np.ones_like(y, dtype=np.int8), np.array([10.0])


def test_onset_definitions():
    y, mask, threshold = _series()
    risk = onset.at_risk(y, mask, threshold)
    assert risk[:3, 0].tolist() == [False, False, False] and risk[3, 0] and risk[7, 0]
    assert not risk[8, 0] and not risk[11, 0] and risk[13, 0]
    label, complete = onset.window_label(y, mask, threshold, window=4)
    assert label[4, 0] and label[7, 0] and not label[3, 0]
    assert not complete[-4:, 0].any()
    events = onset.onset_events(y, mask, threshold)
    assert np.nonzero(events[:, 0])[0].tolist() == [8]


def test_event_detection_counts_each_onset_once_with_lead_time():
    y, mask, threshold = _series()
    period_id = np.arange(1, len(y) + 1)
    eligible = onset.eligible_origins(y, mask, threshold, 4, period_id, 1, len(y))
    alarm = np.zeros_like(eligible)
    alarm[5, 0] = True
    alarm[7, 0] = True
    events = onset.onset_events(y, mask, threshold)
    result = onset.event_detection(alarm, eligible, events, 4, np.ones(len(y), dtype=bool))
    assert len(result) == 1
    assert result["detected"].iloc[0] and result["lead_weeks"].iloc[0] == 3


def test_alarm_cutoff_budget():
    cutoff = onset.alarm_cutoff(np.arange(100, dtype=float), budget=0.1)
    assert (np.arange(100) >= cutoff).mean() == pytest.approx(0.1, abs=0.011)

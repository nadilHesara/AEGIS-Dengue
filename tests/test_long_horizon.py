"""Tests for the long-horizon benchmark's shared protocol (src/evaluation/long_horizon.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.long_horizon import (
    CALIBRATION_FOLD,
    HEADLINE_FOLDS,
    QUANTILES,
    cell_frame,
    quantile_column,
    weighted_interval_score,
)


def test_wis_is_zero_for_a_point_mass_on_the_truth():
    actual = np.array([5.0, 12.0])
    quantiles = np.repeat(actual[:, None], len(QUANTILES), axis=1)
    assert np.allclose(weighted_interval_score(actual, quantiles), 0.0)


def test_wis_of_a_median_only_miss_is_half_the_absolute_error_scaled():
    # All quantiles at 10, truth 14: every interval misses by 4 on the upper side.
    actual = np.array([14.0])
    quantiles = np.full((1, len(QUANTILES)), 10.0)
    k = 3
    expected = 0.5 * 4.0
    for alpha in (0.05, 0.2, 0.5):
        expected += (alpha / 2.0) * (2.0 / alpha) * 4.0
    expected /= k + 0.5
    assert weighted_interval_score(actual, quantiles)[0] == pytest.approx(expected)


def test_wis_rewards_a_sharper_calibrated_interval():
    rng = np.random.default_rng(0)
    actual = rng.normal(100.0, 10.0, size=4000)
    z = np.array([-1.959964, -1.281552, -0.674490, 0.0, 0.674490, 1.281552, 1.959964])
    right = 100.0 + 10.0 * z[None, :].repeat(len(actual), axis=0)
    wide = 100.0 + 30.0 * z[None, :].repeat(len(actual), axis=0)
    assert weighted_interval_score(actual, right).mean() < weighted_interval_score(actual, wide).mean()


def test_cell_frame_is_long_format_with_quantile_columns():
    fold = {"fold_id": 3}
    prediction = np.arange(6, dtype=np.float32).reshape(2, 3)
    quantiles = np.repeat(prediction[..., None], len(QUANTILES), axis=-1)
    frame = cell_frame("m", fold, "test", 4, np.array([100, 101]), prediction, 0, quantiles)
    assert len(frame) == 6
    assert list(frame["target_period_id"]) == [100, 100, 100, 101, 101, 101]
    assert list(frame["node_id"]) == [0, 1, 2, 0, 1, 2]
    assert np.allclose(frame["prediction"], frame[quantile_column(0.5)])


def test_calibration_fold_is_never_a_headline_fold():
    assert CALIBRATION_FOLD not in HEADLINE_FOLDS
    assert HEADLINE_FOLDS == (1, 2, 3, 6, 7, 8, 9)

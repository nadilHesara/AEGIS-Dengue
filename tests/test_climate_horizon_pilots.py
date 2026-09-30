"""Donor-based climate perturbation for the utility pilots."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.climate_horizon import WINDOW
from src.training.climate_horizon_pilots import (
    PERMUTATION_SEEDS, SEASON_WEEKS, donor_climate, draw_donors, eligible_donors,
)


def setup(n_periods=300, n_nodes=3):
    start = np.datetime64("2010-01-04") + 7 * np.arange(n_periods)
    tensors = {"period_id": np.arange(1, n_periods + 1), "start_date": start.astype("datetime64[D]")}
    calendar = pd.DataFrame({"period_id": tensors["period_id"],
                             "year": start.astype("datetime64[Y]").astype(int) + 1970})
    block = {"train_end_period": 200, "test_start_period": 240, "test_end_period": 280}
    return tensors, calendar, block


def test_donors_are_past_training_history_same_season_other_year():
    tensors, calendar, block = setup()
    donors = eligible_donors(tensors, calendar, block)
    weeks = (np.asarray(tensors["start_date"]) - np.asarray(tensors["start_date"]).astype("datetime64[Y]")).astype(int) // 7
    years = calendar["year"].to_numpy()
    assert donors
    for r, pool in donors.items():
        assert (pool < r).all() and (tensors["period_id"][pool] <= block["train_end_period"]).all()
        assert (pool >= WINDOW - 1).all()
        gap = np.abs(np.minimum(weeks[pool], 51) - np.minimum(weeks[r], 51))
        assert (np.minimum(gap, 52 - gap) <= SEASON_WEEKS).all()
        assert (years[pool] != years[r]).all()


def test_draws_are_reproducible_and_flag_missing_donors():
    tensors, calendar, block = setup()
    donors = eligible_donors(tensors, calendar, block)
    recipients = np.array(sorted(donors)[:5] + [10])   # index 10 has no donor entry
    a = draw_donors(recipients, donors, 3, PERMUTATION_SEEDS[0])
    b = draw_donors(recipients, donors, 3, PERMUTATION_SEEDS[0])
    assert np.array_equal(a, b) and (a[-1] == -1).all()
    assert not np.array_equal(a, draw_donors(recipients, donors, 3, PERMUTATION_SEEDS[1]))


def test_whole_window_moves_within_district_and_nothing_else_changes():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 3, 10)).astype(np.float32)
    climate = [1, 2, 3]
    origins = np.array([60, 70])
    original = np.stack([x[o - WINDOW + 1:o + 1][..., climate] for o in origins])
    chosen = np.array([[40, 50, -1], [45, 40, 41]])
    out = donor_climate(x, climate, chosen, original)
    for i in range(2):
        for n in range(3):
            if chosen[i, n] < 0:
                assert np.array_equal(out[i, :, n], original[i, :, n])
            else:
                d = chosen[i, n]
                assert np.array_equal(out[i, :, n], x[d - WINDOW + 1:d + 1, n][:, climate])
    assert out.shape == original.shape

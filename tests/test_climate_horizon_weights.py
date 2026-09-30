"""Weight rule and aggregation for the climate-horizon extension."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.climate_horizon_weights import (
    RHO, aggregate, distance_from_uniform, unit_errors, weights_from_utilities,
)


def test_uniform_utilities_give_uniform_weights():
    w, status = weights_from_utilities([0.3, 0.3, 0.3, 0.3])
    assert status == "informed" and np.allclose(w, 1.0)


def test_all_zero_falls_back_to_uniform():
    w, status = weights_from_utilities([0.0, 0.0, 0.0, 0.0])
    assert status == "all_nonpositive_uniform" and np.array_equal(w, np.ones(4))


def test_negative_gains_are_clipped_before_weighting():
    gains = np.array([-0.4, -0.1, -0.2, -0.3])
    w, status = weights_from_utilities(np.maximum(gains, 0))
    assert status == "all_nonpositive_uniform" and np.array_equal(w, np.ones(4))
    with pytest.raises(ValueError):
        weights_from_utilities(gains)


def test_single_positive_horizon():
    w, _ = weights_from_utilities([0.0, 0.0, 0.0, 0.2])
    assert np.allclose(w, [1 - RHO, 1 - RHO, 1 - RHO, (1 - RHO) + RHO * 4])
    assert np.isclose(w.mean(), 1.0) and (w > 0).all()


def test_missing_evidence_is_flagged_and_uniform():
    w, status = weights_from_utilities([np.nan, 0.1, 0.2, 0.3])
    assert status == "missing_evidence" and np.array_equal(w, np.ones(4))


@pytest.mark.parametrize("u", [[0.1, 0.2, 0.3, 0.4], [1e-6, 0, 0, 5.0], [2, 2, 0, 0]])
def test_weights_average_one_and_follow_the_formula(u):
    u = np.asarray(u, dtype=float)
    w, _ = weights_from_utilities(u)
    assert np.isclose(w.mean(), 1.0)
    assert np.allclose(w, (1 - RHO) + RHO * 4 * u / u.sum())


def test_distance_from_uniform():
    d = distance_from_uniform([0.5, 1.0, 1.0, 1.5])
    assert d == {"l1": 1.0, "max_abs": 0.5, "max_over_min": 3.0}


def pilot_rows():
    rows = []
    for block in (0, 1):
        for seed in (0, 1):
            for arm, perm, err in (("real", -1, 1.0), ("case_only", -1, 1.5),
                                   ("shuffled", 11, 1.2), ("shuffled", 23, 1.4)):
                for h in (1, 2):
                    rows.append({"fold_id": 9, "block": block, "seed": seed, "arm": arm, "permutation": perm,
                                 "horizon": h, "prediction": err * h + block, "actual": 0.0,
                                 "observed": 1, "matched": True})
    rows.append({**rows[0], "matched": False, "prediction": 1e6})   # unmatched: ignored
    return pd.DataFrame(rows)


def test_aggregation_averages_permutations_then_seeds_then_blocks():
    units = unit_errors(pilot_rows())
    agg = aggregate(units).set_index("horizon")
    # real = h + block, shuffled = mean(1.2h, 1.4h) + block = 1.3h + block; blocks average
    assert np.isclose(agg.loc[1, "mae_real"], 1.5) and np.isclose(agg.loc[1, "mae_shuffled"], 1.8)
    assert np.isclose(agg.loc[2, "gain_shuffle"], 0.6)
    assert np.isclose(agg.loc[1, "gain_case_only_rel"], (2.0 - 1.5) / 2.0, atol=1e-6)
    assert (agg["units"] == 4).all()

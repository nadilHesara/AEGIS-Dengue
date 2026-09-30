"""Estimand and bootstrap checks for the extension's evaluation code."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import climate_horizon as ev
from src.evaluation.long_horizon import load_script
from src.evaluation.revision import block_bootstrap


@pytest.fixture(scope="module")
def analysis():
    return load_script("analysis_49", "evaluation/49.climate_horizon_analysis.py")


def errors(rows):
    frame = pd.DataFrame(rows)
    frame["abs_error"] = (frame.prediction - frame.actual).abs()
    frame["sq_error"] = (frame.prediction - frame.actual) ** 2
    return frame


def two_seed_frame():
    base = {"fold_id": 1, "horizon": 1, "target_period_id": 100, "node_id": 0, "actual": 10.0, "threshold": 5.0}
    rows = [{**base, "method": "m", "seed": 0, "prediction": 0.0},
            {**base, "method": "m", "seed": 1, "prediction": 20.0},
            {**base, "method": "persistence", "seed": -1, "prediction": 10.0}]
    return errors(rows)


def test_seed_mean_versus_ensemble_target_10_predictions_0_and_20(analysis):
    frame = two_seed_frame()
    seed_mean, _ = analysis.horizon_table(frame, (1,), "seed_mean")
    ensemble, _ = analysis.horizon_table(frame, (1,), "ensemble")
    s, e = seed_mean.set_index("method").loc["m"], ensemble.set_index("method").loc["m"]
    assert s["mae"] == 10.0 and e["mae"] == 0.0
    assert s["rmse"] == 10.0 and e["rmse"] == 0.0            # per-seed RMSE averaged vs RMSE of the mean
    assert s["peak_mae"] == 10.0 and e["peak_mae"] == 0.0     # actual 10 >= threshold 5


def test_score_and_headline_average_per_seed_errors():
    frame = pd.DataFrame({"method": "m", "fold_id": 1, "split": "test", "horizon": 1, "target_period_id": 100,
                          "node_id": 0, "seed": [0, 1], "prediction": [0.0, 20.0]})
    truth = pd.DataFrame({"target_period_id": [100], "node_id": [0], "actual": [10.0], "observed": [1]})
    metrics = ev.score(frame, truth, {1: np.array([5.0])})
    assert metrics["mae"].tolist() == [10.0, 10.0]
    per_fold = metrics.groupby(["method", "fold_id", "horizon"])["mae"].mean()
    assert per_fold.iloc[0] == 10.0


def test_bootstrap_point_estimate_is_the_fold_weighted_mean():
    rows = []
    for fold, shift in ((1, 1.0), (2, 3.0)):
        for week in range(20):
            for node in range(5 if fold == 1 else 25):    # unequal cell counts: folds still weigh equally
                rows.append({"fold_id": fold, "target_period_id": week, "diff": shift})
    result = block_bootstrap(pd.DataFrame(rows), block=4, n_boot=200, seed=0)
    assert result["delta"] == pytest.approx(2.0)
    assert result["ci_low"] == pytest.approx(2.0) and result["ci_high"] == pytest.approx(2.0)


def test_bootstrap_keeps_districts_of_a_week_together():
    rows = [{"fold_id": 1, "target_period_id": w, "diff": (1.0 if w < 4 else -1.0)}
            for w in range(8) for _ in range(25)]
    result = block_bootstrap(pd.DataFrame(rows), block=4, n_boot=500, seed=1)
    # only two blocks exist (all +1 or all -1), so every draw is -1, 0 or +1
    assert result["ci_low"] >= -1 - 1e-12 and result["ci_high"] <= 1 + 1e-12

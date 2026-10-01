"""Focused checks for causal weather construction in Experiment B-weather."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "training" / "48.timesfm_weather_covariates.py"
SPEC = importlib.util.spec_from_file_location("timesfm_weather_covariates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_causal_weather_path_uses_only_history_for_future_climatology():
    values = np.array([[[10.0]], [[20.0]], [[999.0]], [[999.0]]], dtype=np.float32)
    weeks = np.array([0, 1, 0, 1], dtype=np.int16)
    path = MODULE.causal_weather_path(values, weeks, origin=1, node=0, horizon=2)
    assert np.allclose(path[:, 0], [10.0, 20.0, 10.0, 20.0])


def test_week_of_year_has_project_week_range():
    calendar = pd.DataFrame({"start_date": pd.to_datetime(["2025-01-01", "2025-12-31"])})
    weeks = MODULE.week_of_year(calendar, future_steps=2)
    assert np.array_equal(weeks, [0, 51, 0, 1])

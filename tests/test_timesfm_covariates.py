"""Focused tests for Experiment B's causal covariate preparation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "training" / "47.timesfm_covariates.py"
SPEC = importlib.util.spec_from_file_location("timesfm_covariates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_causal_fill_never_uses_a_later_observation():
    assert np.array_equal(MODULE.causal_fill(np.array([np.nan, 2.0, np.nan, 5.0, np.nan])),
                          np.array([0.0, 2.0, 2.0, 5.0, 5.0]))


def test_calendar_covariates_cover_the_requested_future_horizon():
    calendar = pd.DataFrame({"start_date": pd.to_datetime(["2025-12-20", "2025-12-27"])})
    covariates = MODULE.calendar_covariates(calendar, max_horizon=12)
    assert set(covariates) == {"season_sin", "season_cos"}
    assert all(values.shape == (14,) for values in covariates.values())
    assert all(np.isfinite(values).all() for values in covariates.values())

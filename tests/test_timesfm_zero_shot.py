"""Focused checks for Experiment A's optional TimesFM adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "training" / "46.timesfm_zero_shot.py"
SPEC = importlib.util.spec_from_file_location("timesfm_zero_shot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_extract_quantiles_maps_timesfm_native_grid_to_benchmark_levels():
    native = np.array([[[999.0, 10.0, 20.0, 30.0, 40.0, 50.0,
                         60.0, 70.0, 80.0, 90.0]]], dtype=np.float32)
    extracted = MODULE.extract_quantiles(native)
    assert extracted.shape == (1, 1, 7)
    assert np.allclose(extracted[0, 0], [2.5, 10.0, 25.0, 50.0, 75.0, 90.0, 97.5])


def test_frozen_config_requests_the_benchmark_quantiles():
    config = MODULE.load_config(Path(__file__).resolve().parents[1] / "configs" / "timesfm_zero_shot.toml")
    assert tuple(config["quantiles"]) == MODULE.QUANTILES
    assert config["input_transform"] == "log1p"


def test_load_canonical_cases_reindexes_missing_periods_and_masks_unobserved(tmp_path):
    calendar = MODULE.pd.DataFrame({"period_id": [1, 2, 3]})
    source = tmp_path / "canonical.parquet"
    MODULE.pd.DataFrame({
        "period_id": [1, 1, 3, 3], "node_id": [0, 1, 0, 1],
        "cases": [2.0, 4.0, 6.0, 8.0], "case_observed": [1, 1, 1, 0],
    }).to_parquet(source)
    cases = MODULE.load_canonical_cases(calendar, source)
    assert cases.shape == (3, 2)
    assert np.allclose(cases[0], [2.0, 4.0])
    assert np.isnan(cases[1]).all()
    assert cases[2, 0] == 6.0 and np.isnan(cases[2, 1])


def test_extract_quantiles_rejects_an_unexpected_timesfm_output_shape():
    with pytest.raises(ValueError, match="forecast columns"):
        MODULE.extract_quantiles(np.zeros((1, 2, 4), dtype=np.float32))

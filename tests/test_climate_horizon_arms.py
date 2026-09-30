"""Frozen arm specification and weight lookup."""

from __future__ import annotations

import json

import pytest

from src.training import climate_horizon_arms as spec


def test_primary_pair_and_seeds():
    assert spec.PRIMARY_ARMS == ("B_target_equal", "D_target_measured")
    assert spec.seeds_for("D_target_measured") == (0, 1, 2, 3, 4)
    assert spec.seeds_for("A_origin_equal") == (0, 1, 2)
    assert [a for a, v in spec.ARMS.items() if v[3]] == list(spec.PRIMARY_ARMS)


def test_weight_lookup(tmp_path):
    assert spec.weights_for("B_target_equal", 3, tmp_path) == ((1.0, 1.0, 1.0, 1.0), "equal")
    assert spec.weights_for("E_target_fixed_1234", 3, tmp_path)[1] == "fixed"
    with pytest.raises(FileNotFoundError):
        spec.weights_for("D_target_measured", 3, tmp_path)
    (tmp_path / "fold3_shuffle.json").write_text(json.dumps({"status": "missing_evidence", "weights": [1, 1, 1, 1]}))
    with pytest.raises(ValueError):
        spec.weights_for("D_target_measured", 3, tmp_path)


def test_configs_differ_only_in_lag_mode_and_gradnorm():
    base = {"x": 1}
    assert spec.config_for("A_origin_equal", base) == {"x": 1, "lag_mode": "origin", "gradnorm": False}
    assert spec.config_for("H_target_gradnorm", base)["gradnorm"] is True

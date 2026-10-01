"""Focused data-boundary tests for Experiment C."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "training" / "49.timesfm_lora_finetune.py"
SPEC = importlib.util.spec_from_file_location("timesfm_lora", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fold_series_stops_at_the_declared_training_boundary():
    cases = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [999.0, 999.0]])
    series = MODULE.fold_series(cases, end_index=2)
    assert len(series) == 2
    assert np.array_equal(series[0], [1.0, 2.0, 3.0])
    assert 999.0 not in series[0]

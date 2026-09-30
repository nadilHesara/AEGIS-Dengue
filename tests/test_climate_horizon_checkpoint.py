"""Checkpoint save / reload / resume for the climate-horizon trainer (CPU, synthetic)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.climate_horizon import build_arrays, split_channels
from src.data.climate_horizon_synthetic import make_panel
from src.evaluation.long_horizon import load_script
from src.training import climate_horizon as tr

CONFIG = {**tr.DEFAULTS, "hidden": 8, "climate_hidden": 4, "batch_size": 32, "patience": 3}
WEIGHTS = (0.5, 0.8, 1.2, 1.5)


@pytest.fixture(scope="module")
def arrays():
    fm = load_script("build_folds_ckpt", "features/14.build_folds.py")
    tensors, months, fold = make_panel(beta=0.5, n_periods=200, n_nodes=3)
    fold = {**fold, "train_end_period": 130, "val_start_period": 131, "val_end_period": 160,
            "test_start_period": 161, "test_end_period": 195, "fit_end_period": 130}
    channels = split_channels(tensors["feature_names"])
    stats = fm.fit_fold_statistics(tensors, months, tensors["period_id"] <= fold["fit_end_period"])
    return build_arrays(tensors, months, fold, fm, channels), channels, stats, fold


def meta(channels, stats, fold):
    return tr.checkpoint_metadata(channels, stats, (1, 2, 3, 4), fold, WEIGHTS, "test",
                                  {"data": "synthetic"}, {"history": 26, "mode": "target"})


def test_resume_equals_an_uninterrupted_run(arrays, tmp_path):
    data, channels, stats, fold = arrays
    cpu = torch.device("cpu")
    straight, info = tr.train(data, CONFIG, 3, cpu, WEIGHTS, max_epochs=4)
    path = tmp_path / "run.pt"
    tr.train(data, CONFIG, 3, cpu, WEIGHTS, max_epochs=2, checkpoint_path=path, metadata=meta(channels, stats, fold))
    resumed, rinfo = tr.train(data, CONFIG, 3, cpu, WEIGHTS, max_epochs=4, checkpoint_path=path, resume=True)
    for (n, a), (_, b) in zip(straight.state_dict().items(), resumed.state_dict().items()):
        assert torch.equal(a, b), n
    assert info["best_epoch"] == rinfo["best_epoch"] and info["val_loss"] == rinfo["val_loss"]
    assert len(info["gradient_log"]) == len(rinfo["gradient_log"])


def test_reload_reproduces_predictions_and_carries_metadata(arrays, tmp_path):
    data, channels, stats, fold = arrays
    cpu = torch.device("cpu")
    path = tmp_path / "run.pt"
    model, _ = tr.train(data, CONFIG, 1, cpu, WEIGHTS, max_epochs=3, checkpoint_path=path,
                        metadata=meta(channels, stats, fold))
    reloaded, saved = tr.load_checkpoint(path, cpu)
    a, b = tr.predict(model, data["test"], cpu), tr.predict(reloaded, data["test"], cpu)
    for key in ("mu", "alpha", "prediction", "gate", "delta_climate"):
        assert np.array_equal(a[key], b[key]), key
    m = saved["metadata"]
    assert m["climate_channels"] == channels.climate_names() and m["horizons"] == [1, 2, 3, 4]
    assert m["utility_weights"] == list(WEIGHTS) and saved["weights"] == list(WEIGHTS)
    assert set(m["scaler_statistics"]) == set(stats) and m["delay_settings"]["mode"] == "target"
    assert saved["config"] == CONFIG and saved["rng_state"]["generator"] is not None


def test_resume_refuses_a_mismatched_run(arrays, tmp_path):
    data, *_ = arrays
    path = tmp_path / "run.pt"
    tr.train(data, CONFIG, 1, torch.device("cpu"), WEIGHTS, max_epochs=1, checkpoint_path=path)
    with pytest.raises(ValueError):
        tr.train(data, CONFIG, 1, torch.device("cpu"), (1, 1, 1, 1), max_epochs=2, checkpoint_path=path, resume=True)

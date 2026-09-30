"""
Tests for configurable gradient clipping in script 32 and the ablation in
script 37.

What has to hold for the ablation to mean anything:

  - **The default is the committed model.** Script 32 always clipped at 1.0;
    the new ``clip_max_norm`` hook must default to exactly that.
  - **``None`` really means no clipping**, while still logging the same
    pre-clip global norm a clipped run logs.
  - **The skip rule is sound.** Script 37 copies a clipped arm from the control
    instead of training it when the control's largest step norm is below the
    threshold. That is only legitimate if such a run is bit-identical, so it is
    asserted on a real training loop, together with the converse (a binding
    threshold does change the model).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load(name: str, filename: str):
    """Import a numerically-prefixed script by path."""

    path = PROJECT_DIR / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


clipping = _load("grad_clipping_module", "training/37.train_grad_clipping.py")
clipping.load_modules()
negbin = clipping.negbin


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HORIZONS = (1, 2)


def make_config(**overrides):
    config = dict(negbin.DEFAULTS)
    config.update(
        hidden=8, gcn_layers=1, dropout=0.0, batch_size=16,
        max_epochs=3, patience=10, weighted=False,
    )
    config.update(overrides)
    return config


def make_arrays(n=40, nodes=25, features=5, steps=6, seed=0):
    rng = np.random.default_rng(seed)

    def split(count):
        return {
            "X": rng.normal(size=(count, steps, nodes, features)).astype(np.float32),
            "y": rng.poisson(20.0, size=(count, nodes, len(HORIZONS))).astype(np.float32),
            "mask": np.ones((count, nodes, len(HORIZONS)), dtype=np.float32),
            "anchor": np.log1p(rng.poisson(20.0, size=(count, nodes))).astype(np.float32),
            "target_period_id": np.arange(count, dtype=np.int64),
        }

    return {"train": split(n), "val": split(n // 2), "test": split(n // 2)}


def train(max_norm, **overrides):
    return negbin.train_shared_negbin(
        make_arrays(), np.eye(25, dtype=np.float32),
        make_config(clip_max_norm=max_norm, **overrides),
        HORIZONS, seed=0, device=torch.device("cpu"),
    )


def model_with_gradients(scale=1.0):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))
    (model(torch.randn(16, 4)).pow(2).sum() * scale).backward()
    return model


def grads(model):
    return [p.grad.clone() for p in model.parameters()]


# ---------------------------------------------------------------------------
# The hook
# ---------------------------------------------------------------------------

def test_default_is_the_committed_clip_of_one():
    assert negbin.DEFAULTS["clip_max_norm"] == 1.0


def test_clip_gradients_matches_torch_clip_grad_norm_exactly():
    ours, reference = model_with_gradients(50.0), model_with_gradients(50.0)
    norm = negbin.clip_gradients(ours, 1.0)
    expected = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    assert norm == pytest.approx(float(expected))
    for a, b in zip(grads(ours), grads(reference)):
        assert torch.equal(a, b)


def test_none_leaves_gradients_untouched_but_reports_the_same_norm():
    model = model_with_gradients(50.0)
    before = grads(model)
    norm = negbin.clip_gradients(model, None)
    for a, b in zip(before, grads(model)):
        assert torch.equal(a, b)
    clipped = model_with_gradients(50.0)
    assert norm == pytest.approx(negbin.clip_gradients(clipped, 1.0))
    assert norm > 1.0


def test_a_binding_clip_bounds_the_post_clip_norm():
    model = model_with_gradients(50.0)
    negbin.clip_gradients(model, 0.5)
    after = torch.nn.utils.get_total_norm([p.grad for p in model.parameters()])
    assert float(after) <= 0.5 + 1e-5


@pytest.mark.parametrize("text, expected", [("none", None), ("None", None), ("1", 1.0), ("0.25", 0.25)])
def test_parse_max_norm(text, expected):
    assert negbin.parse_max_norm(text) == expected


@pytest.mark.parametrize("text", ["0", "-1"])
def test_parse_max_norm_rejects_non_positive(text):
    with pytest.raises(Exception):
        negbin.parse_max_norm(text)


# ---------------------------------------------------------------------------
# Logging from the real training loop
# ---------------------------------------------------------------------------

def test_every_optimiser_step_logs_a_finite_norm():
    _, info = train(None)
    steps_per_epoch = int(np.ceil(40 / 16))
    assert info["grad_norms"].shape == (info["epochs_run"] * steps_per_epoch,)
    assert np.isfinite(info["grad_norms"]).all()
    assert len(info["history"]) == info["epochs_run"]
    row = info["history"][0]
    assert {"train_loss", "val_loss", "grad_norm_mean", "grad_norm_max",
            "clipped_fraction", "nonfinite_steps"} <= set(row)


def test_clipped_fraction_is_zero_without_a_threshold():
    _, info = train(None)
    assert all(h["clipped_fraction"] == 0.0 for h in info["history"])


# ---------------------------------------------------------------------------
# Soundness of script 37's skip rule
# ---------------------------------------------------------------------------

def test_a_threshold_above_every_norm_is_bit_identical_to_no_clip():
    control, control_info = train(None)
    threshold = float(control_info["grad_norms"].max()) * 2
    clipped, clipped_info = train(threshold)

    np.testing.assert_array_equal(control_info["grad_norms"], clipped_info["grad_norms"])
    for (name, a), b in zip(control.state_dict().items(), clipped.state_dict().values()):
        assert torch.equal(a, b), name


def test_a_binding_threshold_changes_the_model():
    control, control_info = train(None)
    threshold = float(np.median(control_info["grad_norms"])) / 2
    clipped, clipped_info = train(threshold)

    assert any(h["clipped_fraction"] > 0 for h in clipped_info["history"])
    assert any(
        not torch.equal(a, b)
        for a, b in zip(control.state_dict().values(), clipped.state_dict().values())
    )


def test_clip_never_fires_needs_the_whole_run_below_threshold():
    run = {"grad_norm_max": 0.93, "nonfinite_steps": 0}
    assert clipping.clip_never_fires(run, 1.0)
    assert not clipping.clip_never_fires(run, 0.93)
    assert not clipping.clip_never_fires(run, 0.5)
    assert not clipping.clip_never_fires({**run, "nonfinite_steps": 1}, 5.0)


def test_inferred_result_relabels_everything_and_is_flagged():
    control = {
        "metrics": [{"arm": "no_clip", "max_norm": None, "mae": 3.0}],
        "run": {"arm": "no_clip", "max_norm": None, "seconds": 12.0, "inferred": False},
        "epochs": [{"arm": "no_clip", "max_norm": None, "epoch": 1}],
        "steps": pd.DataFrame({"arm": ["no_clip"], "grad_norm": [0.2]}),
        "mu": np.zeros(2),
    }
    copied = clipping.inferred_result(control, 5.0)
    assert copied["run"]["inferred"] and copied["run"]["seconds"] == 0.0
    assert {r["arm"] for r in copied["metrics"] + copied["epochs"]} == {"clip_5"}
    assert set(copied["steps"]["arm"]) == {"clip_5"}
    assert copied["metrics"][0]["mae"] == 3.0
    assert control["run"]["arm"] == "no_clip"  # the control is not mutated


def test_arm_names():
    assert clipping.arm_name(None) == "no_clip"
    assert clipping.arm_name(1.0) == "clip_1"
    assert clipping.arm_name(0.25) == "clip_0.25"


def test_the_control_is_always_part_of_the_sweep():
    assert None in clipping.DEFAULT_MAX_NORMS
    with pytest.raises(SystemExit):
        clipping.parse_arguments(["--max-norms", "1", "5"])

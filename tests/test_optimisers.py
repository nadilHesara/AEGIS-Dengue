"""
Tests for the optimiser/schedule arms in scripts/25.train_optimisers.py.

The comparison is only meaningful if four things hold, so those are what is
asserted:

  - **The `adam` arm is bit-identical to script 16's native path.** It is the
    control; if passing Adam back through the new `make_optimiser` hook changes
    a single prediction, the hook is not neutral and no arm is comparable to any
    committed result.
  - **Adam and AdamW actually differ.** They differ only in how weight decay is
    applied, so at `weight_decay=0` they must agree and at the baseline's
    `weight_decay=1e-4` they must not. A test that only checked "AdamW runs"
    would pass on a copy-paste that built Adam twice.
  - **The scheduler cannot move the selected epoch.** It is stepped after early
    stopping has seen the epoch, so attaching one must not change which epoch is
    recorded as best for a run that never reduces the rate.
  - **The schedule is observable.** `train_one` returns the per-epoch rates, and
    a reduction has to show up in them, or the report's "did the schedule fire"
    table is unfalsifiable.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


PROJECT_DIR = Path(__file__).resolve().parent.parent

FOLDS_PATH = PROJECT_DIR / "data" / "processed" / "folds.json"
TENSOR_PATH = PROJECT_DIR / "data" / "processed" / "model_tensors_v1.npz"


def _load(name: str, filename: str):
    """Import a numerically-prefixed script by path."""

    path = PROJECT_DIR / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


optimisers = _load("optimiser_module", "training/25.train_optimisers.py")
optimisers.load_modules()
baseline = optimisers.baseline_module


requires_real_data = pytest.mark.skipif(
    not (FOLDS_PATH.exists() and TENSOR_PATH.exists()),
    reason="Run scripts 12 and 14 first.",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_config(**overrides):
    """A small, fast training config with the baseline's shape."""

    config = {
        "lookback": 12,
        "horizon": 1,
        "hidden": 8,
        "gcn_layers": 1,
        "dropout": 0.0,
        "learning_rate": 3e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "max_epochs": 6,
        "patience": 6,
        "target": "residual",
        "scheduler_patience": 1,
        "scheduler_factor": 0.5,
        "scheduler_min_lr": 1e-6,
    }
    config.update(overrides)
    return config


def make_arrays(n=40, nodes=25, features=6, steps=12, seed=0):
    """Synthetic fold arrays with the shapes `train_one` expects."""

    rng = np.random.default_rng(seed)

    def split(count):
        return {
            "X": rng.normal(size=(count, steps, nodes, features)).astype(np.float32),
            "y": rng.gamma(3.0, 5.0, size=(count, nodes)).astype(np.float32),
            "mask": np.ones((count, nodes), dtype=np.float32),
            "anchor": np.log1p(
                rng.gamma(3.0, 5.0, size=(count, nodes))
            ).astype(np.float32),
            "target_period_id": np.arange(count, dtype=np.int64),
        }

    return {"train": split(n), "val": split(n // 2), "test": split(n // 2)}


# ---------------------------------------------------------------------------
# The control identity
# ---------------------------------------------------------------------------

def test_adam_through_the_hook_matches_script_16_exactly():
    """Passing Adam back through `make_optimiser` must change nothing.

    This is the identity the whole comparison rests on: the `adam` arm is the
    control, and it is only a control if the hook is a no-op for it.
    """

    arrays = make_arrays()
    adjacency = np.eye(25, dtype=np.float32)
    config = make_config()
    device = torch.device("cpu")

    native, native_info = baseline.train_one(arrays, adjacency, config, 0, device)
    hooked, hooked_info = baseline.train_one(
        arrays,
        adjacency,
        config,
        0,
        device,
        make_optimiser=optimisers.make_optimiser("adam"),
    )

    assert native_info["best_epoch"] == hooked_info["best_epoch"]

    native_prediction = baseline.predict(
        native, arrays["test"], adjacency, "residual", device
    )
    hooked_prediction = baseline.predict(
        hooked, arrays["test"], adjacency, "residual", device
    )

    np.testing.assert_array_equal(native_prediction, hooked_prediction)


@requires_real_data
def test_the_control_reproduces_the_baseline_on_a_real_fold():
    """Same check on real fold data, where the model and windows are the real ones."""

    import json

    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))["folds"]
    fold = [f for f in folds if f["fold_id"] == 8][0]

    adjacency = np.load(
        PROJECT_DIR / "data" / "processed" / "adjacency.npz", allow_pickle=True
    )["A_norm"].astype(np.float32)

    calendar = baseline.folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()
    tensors = baseline.folds_module.load_tensors("v1")

    config = {
        "lookback": 12, "horizon": 1, "hidden": 32, "gcn_layers": 2,
        "dropout": 0.2, "learning_rate": 3e-3, "weight_decay": 1e-4,
        "batch_size": 64, "max_epochs": 30, "patience": 15, "target": "residual",
    }
    arrays = baseline.build_fold_arrays(tensors, months, fold, 12, 1)
    device = torch.device("cpu")

    native, _ = baseline.train_one(arrays, adjacency, config, 0, device)
    hooked, _ = baseline.train_one(
        arrays, adjacency, config, 0, device,
        make_optimiser=optimisers.make_optimiser("adam"),
    )

    np.testing.assert_array_equal(
        baseline.predict(native, arrays["test"], adjacency, "residual", device),
        baseline.predict(hooked, arrays["test"], adjacency, "residual", device),
    )


# ---------------------------------------------------------------------------
# Adam vs AdamW
# ---------------------------------------------------------------------------

def test_the_two_optimisers_are_different_classes():
    model = baseline.GCNGRU(n_features=6, hidden=8, gcn_layers=1, horizon=1)
    config = make_config()

    adam = optimisers.make_optimiser("adam")(model, config)
    adamw = optimisers.make_optimiser("adamw")(model, config)

    assert isinstance(adam, torch.optim.Adam)
    assert isinstance(adamw, torch.optim.AdamW)
    assert not isinstance(adam, torch.optim.AdamW)


def test_adam_and_adamw_diverge_when_weight_decay_is_nonzero():
    """The arms must actually train differently, or the comparison is empty.

    Adam folds `wd * w` into the gradient, where its own normalisation rescales
    it; AdamW applies the decay to the weight directly. At a nonzero decay those
    are different updates and the trained models must differ.
    """

    arrays = make_arrays()
    adjacency = np.eye(25, dtype=np.float32)
    config = make_config(weight_decay=1e-4)
    device = torch.device("cpu")

    adam, _ = baseline.train_one(
        arrays, adjacency, config, 0, device,
        make_optimiser=optimisers.make_optimiser("adam"),
    )
    adamw, _ = baseline.train_one(
        arrays, adjacency, config, 0, device,
        make_optimiser=optimisers.make_optimiser("adamw"),
    )

    a = baseline.predict(adam, arrays["test"], adjacency, "residual", device)
    w = baseline.predict(adamw, arrays["test"], adjacency, "residual", device)

    assert not np.array_equal(a, w)


def test_adam_and_adamw_agree_when_weight_decay_is_zero():
    """With no decay the two updates are the same, which is the sanity check.

    If they disagreed at `weight_decay=0` the difference would be coming from
    something other than the decay coupling, and the arm would not be measuring
    what it claims.
    """

    arrays = make_arrays()
    adjacency = np.eye(25, dtype=np.float32)
    config = make_config(weight_decay=0.0)
    device = torch.device("cpu")

    adam, _ = baseline.train_one(
        arrays, adjacency, config, 0, device,
        make_optimiser=optimisers.make_optimiser("adam"),
    )
    adamw, _ = baseline.train_one(
        arrays, adjacency, config, 0, device,
        make_optimiser=optimisers.make_optimiser("adamw"),
    )

    np.testing.assert_allclose(
        baseline.predict(adam, arrays["test"], adjacency, "residual", device),
        baseline.predict(adamw, arrays["test"], adjacency, "residual", device),
        rtol=1e-5,
        atol=1e-6,
    )


def test_an_unknown_optimiser_is_rejected():
    """A typo in the arm table must fail loudly rather than silently pick one."""

    model = baseline.GCNGRU(n_features=6, hidden=8, gcn_layers=1, horizon=1)

    with pytest.raises(ValueError, match="Unknown optimiser"):
        optimisers.make_optimiser("nesterov_sgd_maybe")(model, make_config())


# ---------------------------------------------------------------------------
# The scheduler
# ---------------------------------------------------------------------------

def test_scheduler_is_reduce_lr_on_plateau_watching_a_minimum():
    model = baseline.GCNGRU(n_features=6, hidden=8, gcn_layers=1, horizon=1)
    config = make_config()

    optimiser = optimisers.make_optimiser("adamw")(model, config)
    scheduler = optimisers.make_scheduler(optimiser, config)

    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.mode == "min"
    assert scheduler.factor == config["scheduler_factor"]
    assert scheduler.patience == config["scheduler_patience"]


def test_the_rate_actually_falls_on_a_plateau():
    """Feed the scheduler a flat loss and the rate must halve.

    Cheap, but it is the property the whole arm depends on: a scheduler that
    never fires is an unscheduled run with extra bookkeeping.
    """

    model = baseline.GCNGRU(n_features=6, hidden=8, gcn_layers=1, horizon=1)
    config = make_config(scheduler_patience=1)

    optimiser = optimisers.make_optimiser("adamw")(model, config)
    scheduler = optimisers.make_scheduler(optimiser, config)

    start = optimiser.param_groups[0]["lr"]
    for _ in range(6):
        scheduler.step(1.0)

    assert optimiser.param_groups[0]["lr"] < start


def test_the_rate_never_falls_below_the_floor():
    model = baseline.GCNGRU(n_features=6, hidden=8, gcn_layers=1, horizon=1)
    config = make_config(scheduler_patience=0, scheduler_min_lr=1e-4)

    optimiser = optimisers.make_optimiser("adamw")(model, config)
    scheduler = optimisers.make_scheduler(optimiser, config)

    for _ in range(50):
        scheduler.step(1.0)

    assert optimiser.param_groups[0]["lr"] >= config["scheduler_min_lr"]


def test_train_one_records_the_learning_rate_trace_only_when_scheduled():
    """The trace is the report's evidence; unscheduled runs must report none."""

    arrays = make_arrays()
    adjacency = np.eye(25, dtype=np.float32)
    config = make_config()
    device = torch.device("cpu")

    _, plain = baseline.train_one(
        arrays, adjacency, config, 0, device,
        make_optimiser=optimisers.make_optimiser("adamw"),
    )
    _, scheduled = baseline.train_one(
        arrays, adjacency, config, 0, device,
        make_optimiser=optimisers.make_optimiser("adamw"),
        make_scheduler=optimisers.make_scheduler,
    )

    assert plain["learning_rates"] == []
    assert np.isnan(plain["final_lr"])

    assert len(scheduled["learning_rates"]) > 0
    assert scheduled["learning_rates"][0] == pytest.approx(config["learning_rate"])
    assert np.isfinite(scheduled["final_lr"])


def test_attaching_a_scheduler_does_not_move_the_selected_epoch_before_it_fires():
    """Early stopping sees each epoch before the scheduler steps.

    So for the epochs up to the first reduction, a scheduled run and an
    unscheduled one must select the same best epoch. If the order were reversed,
    adding a scheduler would silently change which model gets kept and the arm
    would not be isolating the schedule.
    """

    arrays = make_arrays()
    adjacency = np.eye(25, dtype=np.float32)
    # A patience high enough that no reduction happens within max_epochs.
    config = make_config(scheduler_patience=99, max_epochs=5)
    device = torch.device("cpu")

    _, plain = baseline.train_one(
        arrays, adjacency, config, 0, device,
        make_optimiser=optimisers.make_optimiser("adamw"),
    )
    _, scheduled = baseline.train_one(
        arrays, adjacency, config, 0, device,
        make_optimiser=optimisers.make_optimiser("adamw"),
        make_scheduler=optimisers.make_scheduler,
    )

    assert plain["best_epoch"] == scheduled["best_epoch"]
    # And the rate never moved, so the two runs are the same run.
    assert scheduled["learning_rates"] == [config["learning_rate"]] * len(
        scheduled["learning_rates"]
    )


# ---------------------------------------------------------------------------
# Arm wiring
# ---------------------------------------------------------------------------

def test_every_arm_is_declared_with_a_known_optimiser():
    for arm, (kind, scheduled) in optimisers.ARMS.items():
        assert kind in ("adam", "adamw"), arm
        assert isinstance(scheduled, bool), arm


def test_the_control_arm_is_unscheduled_adam():
    """`adam` must be script 16's configuration, or it is not a control."""

    assert optimisers.ARMS["adam"] == ("adam", False)


def test_only_the_scheduled_arm_is_scheduled():
    assert optimisers.ARMS["adamw"] == ("adamw", False)
    assert optimisers.ARMS["adamw_scheduled"] == ("adamw", True)


def test_baseline_defaults_are_unchanged_by_this_script():
    """Script 25 must not have moved the numbers script 16 is committed to."""

    for key in ("learning_rate", "weight_decay", "batch_size", "patience"):
        assert optimisers.DEFAULTS[key] == baseline.DEFAULTS[key], key


# ---------------------------------------------------------------------------
# Fold decomposition
# ---------------------------------------------------------------------------

def test_fold_decomposition_splits_fold_one_from_the_rest():
    """The guard against reading a single-fold artefact as a general result."""

    import pandas as pd

    # adam flat at 10; adamw better by 7 on fold 1 only.
    rows = []
    for fold_id in [1, 2, 3, 6, 7, 8, 9]:
        rows.append(
            {"arm": "adam", "fold_id": fold_id, "headline": True,
             "mae": 10.0, "ensemble": False, "seed": 0}
        )
        rows.append(
            {"arm": "adamw", "fold_id": fold_id, "headline": True,
             "mae": 3.0 if fold_id == 1 else 10.0, "ensemble": False, "seed": 0}
        )

    decomposition = optimisers.fold_decomposition(pd.DataFrame(rows))
    row = decomposition[decomposition["arm"] == "adamw"].iloc[0]

    assert row["headline_delta"] == pytest.approx(-1.0)
    assert row["fold1_share"] == pytest.approx(100.0)
    assert row["ex_fold1_delta"] == pytest.approx(0.0)

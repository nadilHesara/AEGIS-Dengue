"""
Tests for the hyperparameter search in scripts/24.tune_hyperparameters.py.

The search only means something if three things hold, so those are what is
asserted here:

  - **Selection never reads the test split.** `score_config` is called with
    `split="val"` during the search; a regression that pointed it at "test"
    would turn the whole exercise into leakage. The test checks the call path
    explicitly.
  - **The tunable model reduces to the baseline** when the search lands on
    `gcn_hidden == gru_hidden` and `gru_layers == 1` — same architecture, same
    parameter count. Otherwise a "tuned" win could just be a bigger model
    reported against a smaller baseline without saying so.
  - **Random search is reproducible** from its seed, and its trial list is drawn
    from exactly the commissioned grid — no value outside the nine lists can be
    proposed.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
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


tuning = _load("tuning_module", "24.tune_hyperparameters.py")
tuning.load_modules()
baseline = tuning.baseline_module


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def test_search_space_is_exactly_what_was_commissioned():
    """The nine axes and their value lists, verbatim from the brief."""

    assert tuning.SEARCH_SPACE == {
        "lookback": [8, 12, 16, 24],
        "gcn_hidden": [16, 32, 64],
        "gcn_layers": [1, 2],
        "gru_hidden": [32, 64, 128],
        "gru_layers": [1, 2],
        "dropout": [0.0, 0.1, 0.2, 0.3],
        "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3],
        "batch_size": [16, 32, 64],
        "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    }


def test_grid_size_is_the_product_of_the_axis_lengths():
    assert tuning._grid_size() == 4 * 3 * 2 * 3 * 2 * 4 * 4 * 3 * 4


def test_horizon_and_target_are_held_fixed_not_searched():
    """These are the problem definition, not knobs. They must not be in the grid."""

    assert "horizon" not in tuning.SEARCH_SPACE
    assert "target" not in tuning.SEARCH_SPACE
    assert tuning.FIXED["horizon"] == 1
    assert tuning.FIXED["target"] == "residual"


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------

def test_random_sampling_only_ever_proposes_grid_values():
    """Every sampled point lies on the commissioned grid, on every axis."""

    import random

    rng = random.Random(0)
    for _ in range(500):
        sampled = tuning.sample_random(rng, tuning.SEARCH_SPACE)
        assert set(sampled) == set(tuning.SEARCH_SPACE)
        for axis, value in sampled.items():
            assert value in tuning.SEARCH_SPACE[axis], (axis, value)


def test_random_search_trial_list_is_reproducible_from_the_seed():
    """Same seed -> same ordered list of configurations proposed."""

    import random

    def first_n(seed, n):
        rng = random.Random(seed)
        return [
            tuple(tuning.sample_random(rng, tuning.SEARCH_SPACE).values())
            for _ in range(n)
        ]

    assert first_n(0, 30) == first_n(0, 30)
    assert first_n(0, 30) != first_n(1, 30)


def test_full_config_merges_fixed_knobs_without_dropping_sampled_ones():
    sampled = tuning.sample_random(__import__("random").Random(0), tuning.SEARCH_SPACE)
    config = tuning.full_config(sampled)

    for key in tuning.SEARCH_SPACE:
        assert config[key] == sampled[key]
    for key, value in tuning.FIXED.items():
        assert config[key] == value


# ---------------------------------------------------------------------------
# The tunable model
# ---------------------------------------------------------------------------

def test_reduces_to_the_baseline_when_widths_match_and_gru_is_one_layer():
    """gcn_hidden == gru_hidden, gru_layers == 1 -> identical parameter count.

    This is the identity the honesty of the comparison rests on: in that corner
    of the grid the "tunable" model must be the baseline, not a near-copy with a
    stray projection layer.
    """

    n_features = 23

    tunable = tuning.TunableGCNGRU(
        n_features=n_features,
        gcn_hidden=32,
        gcn_layers=2,
        gru_hidden=32,
        gru_layers=1,
        dropout=0.2,
        horizon=1,
    )
    reference = baseline.GCNGRU(
        n_features=n_features, hidden=32, gcn_layers=2, horizon=1, dropout=0.2
    )

    assert baseline.count_parameters(tunable) == baseline.count_parameters(reference)

    # And the projection really is a no-op module in that case.
    assert isinstance(tunable.project, torch.nn.Identity)


def test_a_projection_is_added_only_when_the_widths_differ():
    same = tuning.TunableGCNGRU(23, 32, 1, 32, 1, 0.0, 1)
    differ = tuning.TunableGCNGRU(23, 16, 1, 128, 1, 0.0, 1)

    assert isinstance(same.project, torch.nn.Identity)
    assert isinstance(differ.project, torch.nn.Linear)
    assert differ.project.in_features == 16
    assert differ.project.out_features == 128


def test_gru_depth_and_width_are_independent_of_the_graph_conv():
    model = tuning.TunableGCNGRU(
        n_features=14,
        gcn_hidden=64,
        gcn_layers=1,
        gru_hidden=128,
        gru_layers=2,
        dropout=0.1,
        horizon=1,
    )

    assert model.gru.hidden_size == 128
    assert model.gru.num_layers == 2
    assert model.graph_layers[0].linear.out_features == 64


def test_forward_returns_batch_nodes_horizon():
    model = tuning.TunableGCNGRU(14, 16, 2, 64, 1, 0.0, 1)

    batch, steps, nodes, feats = 5, 8, 25, 14
    x = torch.randn(batch, steps, nodes, feats)
    adjacency = torch.eye(nodes)

    out = model(x, adjacency)

    assert out.shape == (batch, nodes, 1)
    assert torch.isfinite(out).all()


def test_model_builder_closure_reads_the_config():
    config = tuning.full_config(
        {
            "lookback": 16,
            "gcn_hidden": 16,
            "gcn_layers": 1,
            "gru_hidden": 128,
            "gru_layers": 2,
            "dropout": 0.3,
            "learning_rate": 1e-4,
            "batch_size": 32,
            "weight_decay": 1e-3,
        }
    )
    model = tuning.make_model_builder(config)(n_features=23)

    assert isinstance(model, tuning.TunableGCNGRU)
    assert model.gru.hidden_size == 128
    assert model.gru.num_layers == 2
    assert model.graph_layers[0].linear.out_features == 16
    assert len(model.graph_layers) == 1


# ---------------------------------------------------------------------------
# Selection is validation-only
# ---------------------------------------------------------------------------

def test_score_config_reads_the_split_it_is_told_to_and_no_other(monkeypatch):
    """`score_config(split="val")` must touch only the "val" arrays, never "test".

    The guard against turning the search into test-set optimisation. We stub
    `train_one` and `evaluate_split` and record which split key each fold's
    arrays are indexed with.
    """

    seen_splits = []

    fake_model = object()
    monkeypatch.setattr(
        tuning.baseline_module,
        "train_one",
        lambda *a, **k: (fake_model, {"best_epoch": 1}),
    )

    def fake_evaluate(model, split, adjacency, mode, thresholds, device):
        # `split` is the dict arrays[<key>]; identify it by a marker we planted.
        seen_splits.append(split["__which__"])
        return {"mae": 1.0, "peak_mae": 2.0, "rmse": 3.0}

    monkeypatch.setattr(tuning, "evaluate_split", fake_evaluate)

    fold_arrays = {
        7: {
            "val": {"__which__": "val", "y": None, "mask": np.zeros((1, 1))},
            "test": {"__which__": "test", "y": None, "mask": np.zeros((1, 1))},
        }
    }
    fold_thresholds = {7: np.zeros(25)}

    tuning.score_config(
        config=tuning.full_config(
            tuning.sample_random(__import__("random").Random(0), tuning.SEARCH_SPACE)
        ),
        fold_arrays=fold_arrays,
        fold_thresholds=fold_thresholds,
        adjacency=np.eye(25, dtype=np.float32),
        tune_folds=[7],
        seeds=2,
        split="val",
        device=torch.device("cpu"),
    )

    assert seen_splits == ["val", "val"]  # two seeds, both on val
    assert "test" not in seen_splits


def test_headline_folds_exclude_the_covid_years():
    """The default tuning folds are the README's headline set, no COVID folds."""

    assert tuning.HEADLINE_FOLDS == [1, 2, 3, 6, 7, 8, 9]
    assert 4 not in tuning.HEADLINE_FOLDS
    assert 5 not in tuning.HEADLINE_FOLDS


# ---------------------------------------------------------------------------
# CSV round-trip
# ---------------------------------------------------------------------------

def test_coerce_snaps_csv_values_back_to_native_grid_types():
    """Values read back from pandas must become int/float, not numpy scalars."""

    assert tuning._coerce(np.int64(16), tuning.SEARCH_SPACE["gcn_hidden"]) == 16
    assert type(tuning._coerce(np.int64(16), tuning.SEARCH_SPACE["gcn_hidden"])) is int

    wd = tuning._coerce(np.float64(1e-5), tuning.SEARCH_SPACE["weight_decay"])
    assert wd == 1e-5
    assert type(wd) is float

    assert tuning._coerce(np.float64(0.0), tuning.SEARCH_SPACE["dropout"]) == 0.0

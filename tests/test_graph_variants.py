"""
Tests for the adaptive graph and the graph-variant comparison.

The comparison answers "does any graph beat no graph", so the properties that
matter are the ones that would make that answer wrong:

  - **The identity arm must be the `gru_only` control.** Substituting `I` for the
    adjacency makes the graph convolution a per-node linear layer. If it does
    not -- if some normalisation or self-loop crept in -- the reference the whole
    report is anchored on is not the control it claims to be.
  - **The adaptive adjacency must be a valid adjacency.** Row-stochastic, finite,
    non-negative, and differentiable. A graph whose rows do not sum to 1 changes
    the feature scale, which would confound the comparison with a scaling effect.
  - **The adaptive model must actually use its learned graph** and ignore the one
    the training loop hands it. A wrapper that silently used the passed matrix
    would report the identity's number under the adaptive arm's name.
  - **Selection must read the validation split, never test.**
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.models.adaptive_graph import (  # noqa: E402
    AdaptiveAdjacency,
    AdaptiveGraphGCNGRU,
    count_parameters,
)


def _load(name: str, filename: str):
    """Import a numerically-prefixed script by path."""

    path = PROJECT_DIR / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


graphs = _load("graph_module", "26.train_graph_variants.py")
graphs.load_modules()
baseline = graphs.baseline_module


# ---------------------------------------------------------------------------
# The adaptive adjacency
# ---------------------------------------------------------------------------

def test_rows_sum_to_one():
    """Row-stochastic, so the convolution averages rather than sums.

    A graph whose rows summed to something else would rescale the features, and
    the comparison against the identity would be measuring that scaling as much
    as the graph.
    """

    adjacency = AdaptiveAdjacency(n_nodes=25, embedding_dim=8, seed=0)()

    torch.testing.assert_close(
        adjacency.sum(dim=1), torch.ones(25), rtol=1e-5, atol=1e-6
    )


def test_weights_are_non_negative_and_finite():
    adjacency = AdaptiveAdjacency(n_nodes=25, embedding_dim=8, seed=0)()

    assert torch.isfinite(adjacency).all()
    assert (adjacency >= 0).all()


def test_shape_is_nodes_by_nodes():
    for n in (5, 25, 40):
        assert AdaptiveAdjacency(n_nodes=n, embedding_dim=4, seed=0)().shape == (n, n)


def test_parameter_count_is_two_embedding_tables():
    """400 parameters at 25 nodes and k=8, against ~8,200 for the backbone.

    The count is the reason this parameterisation was chosen over a free 25x25
    matrix, so it is worth pinning.
    """

    module = AdaptiveAdjacency(n_nodes=25, embedding_dim=8)

    assert count_parameters(module) == 2 * 25 * 8 == 400


def test_initialisation_is_close_to_uniform():
    """The initial graph must not commit to arbitrary district pairings.

    The embeddings are scaled down at init so the pre-softmax logits start small.
    A sharp random graph at epoch 0 would bias what the loss can subsequently
    learn.
    """

    adjacency = AdaptiveAdjacency(n_nodes=25, embedding_dim=8, seed=0)()
    uniform = 1.0 / 25

    assert adjacency.max().item() < 2 * uniform


def test_the_graph_is_differentiable():
    """It has to train, or the arm tests nothing."""

    module = AdaptiveAdjacency(n_nodes=10, embedding_dim=4, seed=0)
    module().sum().backward()

    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_the_graph_can_become_asymmetric():
    """Asymmetry is a stated advantage over both fixed graphs, so check it exists.

    At initialisation the embeddings are near-uniform; after a gradient step
    pushing one direction the matrix must be able to differ from its transpose.
    """

    module = AdaptiveAdjacency(n_nodes=6, embedding_dim=4, seed=0)

    target = torch.zeros(6, 6)
    target[0, 1] = 1.0

    optimiser = torch.optim.Adam(module.parameters(), lr=0.5)
    for _ in range(50):
        optimiser.zero_grad()
        ((module() - target) ** 2).sum().backward()
        optimiser.step()

    adjacency = module()
    assert not torch.allclose(adjacency, adjacency.T, atol=1e-3)


def test_seeding_makes_initialisation_reproducible():
    a = AdaptiveAdjacency(n_nodes=25, embedding_dim=8, seed=7)()
    b = AdaptiveAdjacency(n_nodes=25, embedding_dim=8, seed=7)()
    c = AdaptiveAdjacency(n_nodes=25, embedding_dim=8, seed=8)()

    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------

def make_backbone(n_features=6, hidden=8):
    return baseline.GCNGRU(
        n_features=n_features, hidden=hidden, gcn_layers=1, horizon=1, dropout=0.0
    )


def test_the_wrapper_ignores_the_adjacency_it_is_passed():
    """The learned graph must win, or the arm reports the wrong model's number.

    Feeding two completely different fixed matrices must give identical output,
    because neither is used.
    """

    model = AdaptiveGraphGCNGRU(
        make_backbone(), AdaptiveAdjacency(n_nodes=25, embedding_dim=8, seed=0)
    )
    model.eval()

    x = torch.randn(3, 12, 25, 6)

    with torch.no_grad():
        with_identity = model(x, torch.eye(25))
        with_ones = model(x, torch.ones(25, 25))

    torch.testing.assert_close(with_identity, with_ones)


def test_the_wrapper_output_changes_when_the_learned_graph_changes():
    """The complement of the previous test: the learned graph must matter."""

    model = AdaptiveGraphGCNGRU(
        make_backbone(), AdaptiveAdjacency(n_nodes=25, embedding_dim=8, seed=0)
    )
    model.eval()

    x = torch.randn(3, 12, 25, 6)

    with torch.no_grad():
        before = model(x, torch.eye(25))
        model.adjacency.source.mul_(50.0)
        after = model(x, torch.eye(25))

    assert not torch.allclose(before, after)


def test_the_wrapper_preserves_the_output_shape():
    model = AdaptiveGraphGCNGRU(
        make_backbone(), AdaptiveAdjacency(n_nodes=25, embedding_dim=8, seed=0)
    )

    output = model(torch.randn(4, 12, 25, 6), torch.eye(25))

    assert output.shape == (4, 25, 1)


def test_learned_adjacency_is_detached():
    model = AdaptiveGraphGCNGRU(
        make_backbone(), AdaptiveAdjacency(n_nodes=25, embedding_dim=8, seed=0)
    )

    learned = model.learned_adjacency()

    assert not learned.requires_grad
    assert learned.shape == (25, 25)


# ---------------------------------------------------------------------------
# The identity control
# ---------------------------------------------------------------------------

def test_identity_adjacency_makes_the_graph_conv_per_node():
    """The claim the whole report is anchored on.

    With A = I, `einsum("ij,bljf->blif", I, projected)` is `projected`, so the
    graph convolution is a plain linear layer applied independently per district
    -- which is exactly what `gru_only` means.
    """

    layer = baseline.GraphConv(6, 8)
    x = torch.randn(2, 12, 25, 6)

    with torch.no_grad():
        convolved = layer(x, torch.eye(25))
        plain = layer.linear(x)

    torch.testing.assert_close(convolved, plain)


def test_the_identity_arm_uses_the_identity():
    matrices = graphs.load_adjacencies(25)

    np.testing.assert_array_equal(matrices["identity"], np.eye(25, dtype=np.float32))


def test_the_control_arm_is_identity_not_contiguity():
    """Beating contiguity would establish nothing; contiguity loses to identity."""

    assert graphs.CONTROL_ARM == "identity"


def test_fixed_arms_load_the_expected_matrices():
    matrices = graphs.load_adjacencies(25)

    with np.load(graphs.ADJACENCY_PATH, allow_pickle=True) as data:
        np.testing.assert_array_equal(
            matrices["contiguity"], data["A_norm"].astype(np.float32)
        )
        np.testing.assert_array_equal(
            matrices["gaussian"], data["A_gaussian_norm"].astype(np.float32)
        )


def test_the_adaptive_arm_gets_a_placeholder_that_is_the_identity():
    """So a bug that used the passed matrix would report the control's number.

    A plausible-looking wrong graph would be much harder to notice than the
    identity's number turning up twice.
    """

    matrices = graphs.load_adjacencies(25)

    np.testing.assert_array_equal(matrices["adaptive"], np.eye(25, dtype=np.float32))


# ---------------------------------------------------------------------------
# Optimiser routing
# ---------------------------------------------------------------------------

def test_fixed_arms_fall_through_to_the_baseline_optimiser():
    """A fixed-graph model must get script 16's own optimiser, unmodified."""

    config = {"learning_rate": 3e-3, "weight_decay": 1e-4, "graph_lr_multiplier": 10.0}

    optimiser = graphs.make_optimiser(make_backbone(), config)

    assert isinstance(optimiser, torch.optim.Adam)
    assert len(optimiser.param_groups) == 1
    assert optimiser.param_groups[0]["lr"] == config["learning_rate"]


def test_the_adaptive_arm_gets_a_separate_higher_rate_for_the_graph():
    """Without this the embeddings barely move -- measured, not assumed."""

    config = {"learning_rate": 3e-3, "weight_decay": 1e-4, "graph_lr_multiplier": 10.0}

    model = AdaptiveGraphGCNGRU(
        make_backbone(), AdaptiveAdjacency(n_nodes=25, embedding_dim=8, seed=0)
    )
    optimiser = graphs.make_optimiser(model, config)

    assert len(optimiser.param_groups) == 2

    graph_group = optimiser.param_groups[1]
    assert graph_group["lr"] == config["learning_rate"] * 10.0
    # No decay on the embeddings: shrinking them pushes the softmax to uniform,
    # which is a prior on the graph rather than regularisation.
    assert graph_group["weight_decay"] == 0.0


def test_every_graph_parameter_is_in_exactly_one_group():
    config = {"learning_rate": 3e-3, "weight_decay": 1e-4, "graph_lr_multiplier": 10.0}

    model = AdaptiveGraphGCNGRU(
        make_backbone(), AdaptiveAdjacency(n_nodes=25, embedding_dim=8, seed=0)
    )
    optimiser = graphs.make_optimiser(model, config)

    grouped = sum(len(group["params"]) for group in optimiser.param_groups)

    assert grouped == len(list(model.parameters()))


# ---------------------------------------------------------------------------
# Selection is validation-only
# ---------------------------------------------------------------------------

def test_graph_lr_selection_reads_validation_never_test(monkeypatch):
    """The nuisance parameter is chosen on validation, or the arm is leaking."""

    seen = []

    monkeypatch.setattr(
        graphs.baseline_module,
        "train_one",
        lambda *a, **k: (object(), {"best_epoch": 1}),
    )
    monkeypatch.setattr(
        graphs.baseline_module,
        "predict",
        lambda model, split, adjacency, mode, device: (
            seen.append(split["__which__"]) or np.zeros((2, 25))
        ),
    )
    monkeypatch.setattr(
        graphs.naive, "evaluate", lambda *a, **k: {"mae": 1.0, "peak_mae": 1.0}
    )

    arrays = {
        "val": {"__which__": "val", "y": np.zeros((2, 25)), "mask": np.ones((2, 25))},
        "test": {"__which__": "test", "y": np.zeros((2, 25)), "mask": np.ones((2, 25))},
    }

    graphs.select_graph_lr(
        arrays,
        np.eye(25, dtype=np.float32),
        {**{k: v for k, v in graphs.DEFAULTS.items()}, "graph_lr_grid": [1.0, 10.0]},
        25,
        np.zeros(25),
        [1.0, 10.0],
        torch.device("cpu"),
    )

    assert seen == ["val", "val"]
    assert "test" not in seen


def test_graph_lr_selection_returns_the_lowest_validation_mae(monkeypatch):
    scores = {1.0: 12.0, 3.0: 9.0, 10.0: 11.0}

    monkeypatch.setattr(
        graphs.baseline_module,
        "train_one",
        lambda arrays, adj, config, seed, device, **k: (
            config["graph_lr_multiplier"],
            {"best_epoch": 1},
        ),
    )
    monkeypatch.setattr(
        graphs.baseline_module, "predict", lambda model, *a, **k: model
    )
    monkeypatch.setattr(
        graphs.naive,
        "evaluate",
        lambda prediction, *a, **k: {"mae": scores[prediction], "peak_mae": 1.0},
    )

    arrays = {"val": {"y": None, "mask": np.ones((2, 25))}}

    selected, trials = graphs.select_graph_lr(
        arrays,
        np.eye(25, dtype=np.float32),
        dict(graphs.DEFAULTS),
        25,
        np.zeros(25),
        [1.0, 3.0, 10.0],
        torch.device("cpu"),
    )

    assert selected == 3.0
    assert len(trials) == 3


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def test_describe_adjacency_counts_off_diagonal_edges_only():
    matrix = np.eye(4, dtype=np.float32)

    description = graphs.describe_adjacency(matrix)

    assert description["edges"] == 0
    assert description["mean_degree"] == 0.0
    assert description["symmetric"]


def test_describe_adjacency_reports_asymmetry():
    matrix = np.zeros((3, 3), dtype=np.float32)
    matrix[0, 1] = 1.0

    assert not graphs.describe_adjacency(matrix)["symmetric"]


def test_comparison_is_anchored_on_the_control_arm():
    """Deltas must be against identity, and the control must not compare to itself."""

    import pandas as pd

    rows = []
    for arm, mae in [("identity", 10.0), ("contiguity", 12.0)]:
        for fold_id in [1, 2, 3, 6, 7, 8, 9]:
            rows.append(
                {
                    "arm": arm,
                    "fold_id": fold_id,
                    "headline": True,
                    "mae": mae,
                    "ensemble": False,
                    "seed": 0,
                }
            )

    comparison = graphs.compare_to_control(pd.DataFrame(rows))

    assert list(comparison["arm"]) == ["contiguity"]
    assert comparison["headline_delta"].iloc[0] == pytest.approx(2.0)
    assert comparison["folds_improved"].iloc[0] == 0

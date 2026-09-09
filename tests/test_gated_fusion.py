"""
Tests for the per-district spatial/temporal fusion gate.

Self-contained: every tensor here is constructed in-process. Nothing reads
``data/processed/``, ``folds.json`` or the model tensors, so the suite runs
without the pipeline having been built.

The two tests that matter are the boundary conditions. ``GatedGCNGRU`` exists
to interpolate between ``scripts/16``'s ``gcn_gru`` (graph convolution over the
real adjacency) and its ``gru_only`` control (the identity in place of the
adjacency). If the gate is pinned to 1 it must *be* ``gcn_gru``, and if pinned
to 0 it must *be* ``gru_only`` -- exactly, not approximately -- or the headline
comparison is measuring an architecture change on top of the gate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.models.gated_fusion import GatedGCNGRU  # noqa: E402


def _load_baseline_gcngru():
    """Import the ``GCNGRU`` class from ``scripts/16.train_gcn_gru.py``.

    The module is imported for its class definitions only; importing it does not
    touch any data file (``load_modules`` is never called).
    """

    spec = importlib.util.spec_from_file_location(
        "baseline_for_test", PROJECT_DIR / "scripts" / "16.train_gcn_gru.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.GCNGRU


GCNGRU = _load_baseline_gcngru()


# ---------------------------------------------------------------------------
# Fixtures -- all synthetic
# ---------------------------------------------------------------------------

N_NODES = 25
N_FEATURES = 14
LOOKBACK = 12
HIDDEN = 32
GCN_LAYERS = 2
HORIZON = 1
BATCH = 4


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def inputs(rng):
    """A batch of input windows, [batch, steps, nodes, features]."""

    array = rng.standard_normal(
        (BATCH, LOOKBACK, N_NODES, N_FEATURES)
    ).astype(np.float32)

    return torch.from_numpy(array)


@pytest.fixture
def adjacency(rng):
    """A plausible normalised 25x25 adjacency: symmetric, non-negative, D^-1/2 (A+I) D^-1/2.

    Not the real contiguity matrix -- a random sparse symmetric graph put
    through the same normalisation ``scripts/13`` uses. Shape and dtype are
    what matters for these tests.
    """

    raw = rng.random((N_NODES, N_NODES)).astype(np.float32)
    raw = (raw + raw.T) / 2.0
    raw[raw < 0.7] = 0.0  # sparsify
    np.fill_diagonal(raw, 0.0)

    a_tilde = raw + np.eye(N_NODES, dtype=np.float32)
    degree = a_tilde.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    normalised = d_inv_sqrt @ a_tilde @ d_inv_sqrt

    return torch.from_numpy(normalised.astype(np.float32))


@pytest.fixture
def identity():
    return torch.eye(N_NODES, dtype=torch.float32)


def _make_backbone(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return GCNGRU(
        n_features=N_FEATURES,
        hidden=HIDDEN,
        gcn_layers=GCN_LAYERS,
        horizon=HORIZON,
        dropout=0.2,
    )


def _gated_from(backbone: nn.Module, freeze_gate=None) -> GatedGCNGRU:
    return GatedGCNGRU(backbone=backbone, n_nodes=N_NODES, freeze_gate=freeze_gate)


# ---------------------------------------------------------------------------
# Boundary conditions -- the two tests the whole design turns on
# ---------------------------------------------------------------------------

def test_gate_one_reproduces_gcn_gru(inputs, adjacency):
    """g == 1 for every district must equal plain GCNGRU over the real adjacency."""

    backbone = _make_backbone(seed=1)
    gated = _gated_from(backbone)

    # Force the gate wide open.
    with torch.no_grad():
        gated.gate_logit.fill_(30.0)  # sigmoid(30) == 1.0 in float32

    backbone.eval()
    gated.eval()

    with torch.no_grad():
        reference = backbone(inputs, adjacency)
        actual = gated(inputs, adjacency)

    assert actual.shape == reference.shape == (BATCH, N_NODES, HORIZON)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)


def test_gate_zero_reproduces_gru_only(inputs, adjacency, identity):
    """g == 0 for every district must equal plain GCNGRU with the identity adjacency."""

    backbone = _make_backbone(seed=2)
    gated = _gated_from(backbone)

    with torch.no_grad():
        gated.gate_logit.fill_(-30.0)  # sigmoid(-30) == 0.0 in float32

    backbone.eval()
    gated.eval()

    with torch.no_grad():
        reference = backbone(inputs, identity)  # this is exactly gru_only
        actual = gated(inputs, adjacency)       # gate routes past the adjacency

    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)


def test_frozen_uniform_gate_is_half(inputs, adjacency):
    """freeze_gate=0.5 holds every district at an even blend and does not learn."""

    backbone = _make_backbone(seed=3)
    gated = _gated_from(backbone, freeze_gate=0.5)

    np.testing.assert_allclose(gated.gate_values(), np.full(N_NODES, 0.5), atol=1e-6)

    # The frozen logit is a buffer, not a parameter.
    parameter_names = {name for name, _ in gated.named_parameters()}
    assert "gate_logit" not in parameter_names
    assert "gate_logit" in dict(gated.named_buffers())


def test_frozen_uniform_matches_manual_blend(inputs, adjacency, identity):
    """freeze_gate=0.5 output equals 0.5*(gcn_gru repr) + 0.5*(gru_only repr) fed through the tail.

    Checked by comparing against a hand-rolled 50/50 blend built from the
    backbone's own layers, so the test does not just restate the implementation.
    """

    backbone = _make_backbone(seed=4)
    backbone.eval()
    gated = _gated_from(backbone, freeze_gate=0.5)
    gated.eval()

    def graph_stack(x, a):
        spatial = x
        for layer in backbone.graph_layers:
            spatial = torch.relu(layer(spatial, a))
            spatial = backbone.dropout(spatial)
        return spatial

    with torch.no_grad():
        blend = 0.5 * graph_stack(inputs, adjacency) + 0.5 * graph_stack(inputs, identity)
        b, n = BATCH, N_NODES
        sequences = blend.permute(0, 2, 1, 3).reshape(b * n, LOOKBACK, -1)
        out, _ = backbone.gru(sequences)
        expected = backbone.head(backbone.dropout(out[:, -1])).view(b, n, -1)

        actual = gated(inputs, adjacency)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Gradients, shapes, ranges
# ---------------------------------------------------------------------------

def test_gate_logit_receives_gradient(inputs, adjacency):
    """One backward pass must leave a non-zero gradient on the learnable gate."""

    backbone = _make_backbone(seed=5)
    gated = _gated_from(backbone)
    gated.train()

    output = gated(inputs, adjacency)
    target = torch.zeros_like(output)
    loss = ((output - target) ** 2).mean()
    loss.backward()

    assert gated.gate_logit.grad is not None
    assert torch.isfinite(gated.gate_logit.grad).all()
    assert gated.gate_logit.grad.abs().sum() > 0.0


def test_frozen_gate_receives_no_gradient(inputs, adjacency):
    """A frozen gate is a buffer; it must not accumulate a gradient."""

    backbone = _make_backbone(seed=6)
    gated = _gated_from(backbone, freeze_gate=0.5)
    gated.train()

    output = gated(inputs, adjacency)
    output.pow(2).mean().backward()

    # buffers have no .grad; assert the backbone still trained
    assert not hasattr(gated.gate_logit, "grad") or gated.gate_logit.grad is None
    assert any(p.grad is not None for p in gated.backbone.parameters())


def test_output_shape(inputs, adjacency):
    backbone = _make_backbone(seed=7)
    gated = _gated_from(backbone)
    gated.eval()

    with torch.no_grad():
        output = gated(inputs, adjacency)

    assert output.shape == (BATCH, N_NODES, HORIZON)


def test_gate_values_in_unit_interval(inputs, adjacency):
    backbone = _make_backbone(seed=8)
    gated = _gated_from(backbone)

    # Push the logits around and confirm the reported gate stays in (0, 1).
    with torch.no_grad():
        gated.gate_logit.copy_(torch.linspace(-8.0, 8.0, N_NODES))

    values = gated.gate_values()
    assert values.shape == (N_NODES,)
    assert np.all(values > 0.0) and np.all(values < 1.0)


def test_parameter_count_is_backbone_plus_25(inputs, adjacency):
    """The gate adds exactly n_nodes parameters and nothing else."""

    backbone = _make_backbone(seed=9)
    backbone_params = sum(p.numel() for p in backbone.parameters())

    gated = _gated_from(backbone)
    gated_params = sum(p.numel() for p in gated.parameters())

    assert gated_params == backbone_params + N_NODES


def test_no_submodule_named_encoder(inputs, adjacency):
    """scripts/16.parameter_groups keys off a submodule literally named `encoder`.

    The gate wants the ordinary learning rate, not the lag encoder's raised one,
    so it must not present an `.encoder` attribute.
    """

    backbone = _make_backbone(seed=10)
    gated = _gated_from(backbone)

    assert getattr(gated, "encoder", None) is None


def test_wrong_node_count_raises(adjacency, rng):
    backbone = _make_backbone(seed=11)
    gated = _gated_from(backbone)

    bad = torch.from_numpy(
        rng.standard_normal((BATCH, LOOKBACK, N_NODES + 1, N_FEATURES)).astype(np.float32)
    )
    bad_adjacency = torch.eye(N_NODES + 1, dtype=torch.float32)

    with pytest.raises(ValueError):
        gated(bad, bad_adjacency)

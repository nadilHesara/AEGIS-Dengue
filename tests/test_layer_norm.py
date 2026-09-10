"""
Tests for layer normalisation before the prediction head.

The experiment claims that a difference between the two arms is the
normalisation and nothing else. These are the properties that would make that
claim false:

  - **The control must be the committed baseline, exactly.** `normalise=False`
    has to reproduce `GCNGRU` bit-for-bit -- same outputs, same parameter count,
    same state_dict keys. If the wrapper perturbs the forward pass or the
    initialisation even slightly, the reference the whole report is anchored on
    is not the committed model.
  - **The two arms must start from the same weights.** Both builders construct
    the backbone under the same seed in the same order, so the shared parameters
    are drawn identically. If they diverge, a measured delta is partly a
    different random initialisation.
  - **The norm must go where it is claimed to go.** After the GRU, before the
    dropout, before the head; over the hidden axis, per (window, district) row.
    A norm over the wrong axis would mix districts together.
  - **The norm must be a real, differentiable normalisation** that produces zero
    mean and unit variance at initialisation and passes gradient to its affine
    parameters.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.models.layer_norm import (  # noqa: E402
    LayerNormGCNGRU,
    count_parameters,
    norm_statistics,
)


def _load(name: str, filename: str):
    """Import a numerically-prefixed script by path."""

    path = PROJECT_DIR / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


layer_norm_module = _load("layer_norm_module", "28.train_layer_norm.py")
layer_norm_module.load_modules()
baseline = layer_norm_module.baseline_module


N_NODES = 25
N_FEATURES = 6
HIDDEN = 32
LOOKBACK = 12


def make_backbone(seed: int = 0, dropout: float = 0.0) -> nn.Module:
    """A `GCNGRU` with the committed shape, under a fixed seed."""

    torch.manual_seed(seed)

    return baseline.GCNGRU(
        n_features=N_FEATURES,
        hidden=HIDDEN,
        gcn_layers=2,
        horizon=1,
        dropout=dropout,
    )


def make_inputs(seed: int = 0):
    """A batch and an adjacency of the shapes the training loop passes."""

    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(4, LOOKBACK, N_NODES, N_FEATURES, generator=generator)
    adjacency = torch.eye(N_NODES)

    return x, adjacency


# ---------------------------------------------------------------------------
# The control arm must be the committed baseline
# ---------------------------------------------------------------------------

def test_control_arm_matches_baseline_exactly():
    """`normalise=False` reproduces `GCNGRU` bit-for-bit.

    The whole comparison is anchored on this arm being the committed model. Not
    `allclose` -- identical: the wrapper adopts the backbone's submodules by
    reference and adds nothing, so the two forward passes are the same
    arithmetic in the same order.
    """

    backbone = make_backbone(seed=0)
    wrapped = LayerNormGCNGRU(backbone, normalise=False)

    x, adjacency = make_inputs()

    backbone.eval()
    wrapped.eval()

    with torch.no_grad():
        expected = backbone(x, adjacency)
        actual = wrapped(x, adjacency)

    assert torch.equal(actual, expected)


def test_control_arm_adds_no_parameters():
    """The control must not carry a norm it does not use."""

    backbone = make_backbone(seed=0)
    before = count_parameters(backbone)

    wrapped = LayerNormGCNGRU(backbone, normalise=False)

    assert count_parameters(wrapped) == before
    assert wrapped.norm is None


def test_normalised_arm_adds_exactly_two_hidden_vectors():
    """LayerNorm(32) adds a gain and a bias: 64 parameters, and no more."""

    backbone = make_backbone(seed=0)
    before = count_parameters(backbone)

    wrapped = LayerNormGCNGRU(backbone, normalise=True)

    assert count_parameters(wrapped) == before + 2 * HIDDEN


def test_backbone_submodules_are_adopted_not_copied():
    """The wrapper must reuse the backbone's modules, not rebuild them.

    Rebuilding would draw fresh weights and make the arms incomparable.
    """

    backbone = make_backbone(seed=0)
    wrapped = LayerNormGCNGRU(backbone, normalise=True)

    assert wrapped.gru is backbone.gru
    assert wrapped.head is backbone.head
    assert wrapped.dropout is backbone.dropout
    assert wrapped.graph_layers is backbone.graph_layers


# ---------------------------------------------------------------------------
# The two arms must start from the same weights
# ---------------------------------------------------------------------------

def test_both_arms_share_initial_backbone_weights():
    """Same seed, same builder order -> identical shared parameters.

    This is what makes the paired comparison paired. If the arms started from
    different initialisations, part of any measured delta would be the seed.
    """

    config = dict(layer_norm_module.DEFAULTS)

    torch.manual_seed(7)
    control = layer_norm_module.make_model_builder("baseline", config)(N_FEATURES)

    torch.manual_seed(7)
    normalised = layer_norm_module.make_model_builder("layer_norm", config)(N_FEATURES)

    control_state = control.state_dict()

    for key, value in normalised.state_dict().items():
        if key.startswith("norm."):
            continue

        assert key in control_state, f"{key} missing from the control arm"
        assert torch.equal(value, control_state[key]), f"{key} differs between arms"


def test_head_is_one_wide_at_every_horizon():
    """The forecast lead must not widen the head.

    `horizon` is overloaded in `scripts/16`: the forecast lead in
    `build_fold_arrays`, the head width in the `GCNGRU` constructor. They
    coincide at h=1, the only value script 16 runs, so passing the lead through
    to the constructor looks correct until h>1 -- where it builds an h-wide head
    whose output `scripts/16.predict` cannot broadcast against the anchor. This
    experiment trains one model per lead, so the head is always one wide.
    """

    for horizon in (1, 2, 3, 4):
        config = dict(layer_norm_module.DEFAULTS)
        config["horizon"] = horizon

        model = layer_norm_module.make_model_builder("layer_norm", config)(N_FEATURES)

        assert model.head.out_features == 1, f"h={horizon} widened the head"

        x, adjacency = make_inputs()
        with torch.no_grad():
            assert model(x, adjacency).shape == (4, N_NODES, 1)


def test_arms_map_to_the_expected_flag():
    """The control arm normalises nothing; the treatment arm normalises."""

    assert layer_norm_module.ARMS["baseline"] is False
    assert layer_norm_module.ARMS["layer_norm"] is True
    assert layer_norm_module.CONTROL_ARM == "baseline"


# ---------------------------------------------------------------------------
# The norm goes where it is claimed to go
# ---------------------------------------------------------------------------

def test_normalisation_changes_the_output():
    """With the norm on, the answer must actually differ from the baseline.

    A wrapper that registered the norm but never called it would silently
    report the baseline's number under the treatment arm's name.
    """

    backbone = make_backbone(seed=0)
    wrapped = LayerNormGCNGRU(backbone, normalise=True)

    x, adjacency = make_inputs()

    backbone.eval()
    wrapped.eval()

    with torch.no_grad():
        assert not torch.allclose(wrapped(x, adjacency), backbone(x, adjacency))


def test_norm_is_over_the_hidden_axis_only():
    """Statistics are per (window, district) row, not across districts.

    Normalising across the node axis would rescale one district by what its
    neighbours are doing, which is a different model from the one described.
    """

    backbone = make_backbone(seed=0)
    wrapped = LayerNormGCNGRU(backbone, normalise=True)

    assert wrapped.norm.normalized_shape == (HIDDEN,)


def test_norm_precedes_dropout():
    """The norm must see the clean GRU state, not the dropout-masked one.

    Probed behaviourally: with dropout at 1.0 in training mode, everything
    after the dropout is zeroed, so the head sees zeros and the output is the
    head bias regardless of the input. If the norm ran *after* the dropout it
    would renormalise that all-zero vector into something input-independent but
    non-zero, and the output would not equal the bias.
    """

    backbone = make_backbone(seed=0, dropout=1.0)
    wrapped = LayerNormGCNGRU(backbone, normalise=True)
    wrapped.train()

    x, adjacency = make_inputs()

    with torch.no_grad():
        output = wrapped(x, adjacency)

    expected = backbone.head.bias.expand_as(output.reshape(-1, 1)).reshape(output.shape)

    torch.testing.assert_close(output, expected)


# ---------------------------------------------------------------------------
# The norm must be a real normalisation
# ---------------------------------------------------------------------------

def test_normalises_to_zero_mean_unit_variance_at_initialisation():
    """At init the gain is 1 and the bias 0, so the output is standardised."""

    norm = nn.LayerNorm(HIDDEN)
    hidden = torch.randn(64, HIDDEN) * 17.0 + 5.0

    normalised = norm(hidden)

    torch.testing.assert_close(
        normalised.mean(dim=-1), torch.zeros(64), atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(
        normalised.std(dim=-1, unbiased=False), torch.ones(64), atol=1e-3, rtol=1e-3
    )


def test_gradient_reaches_the_affine_parameters():
    """The gain and bias must be trained, not frozen decoration."""

    backbone = make_backbone(seed=0)
    wrapped = LayerNormGCNGRU(backbone, normalise=True)

    x, adjacency = make_inputs()
    wrapped(x, adjacency).sum().backward()

    assert wrapped.norm.weight.grad is not None
    assert wrapped.norm.bias.grad is not None
    assert torch.isfinite(wrapped.norm.weight.grad).all()
    assert wrapped.norm.weight.grad.abs().sum() > 0


def test_gradient_still_reaches_the_backbone():
    """The norm must not cut the gradient path to the GRU and graph layers."""

    backbone = make_backbone(seed=0)
    wrapped = LayerNormGCNGRU(backbone, normalise=True)

    x, adjacency = make_inputs()
    wrapped(x, adjacency).sum().backward()

    for name, parameter in wrapped.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has a non-finite gradient"


def test_output_shape_is_unchanged():
    """The norm must not change the contract the training loop depends on."""

    for normalise in (False, True):
        wrapped = LayerNormGCNGRU(make_backbone(seed=0), normalise=normalise)
        x, adjacency = make_inputs()

        with torch.no_grad():
            assert wrapped(x, adjacency).shape == (4, N_NODES, 1)


def test_output_is_finite_on_a_constant_input():
    """A constant hidden state has zero variance; eps must keep it finite.

    Hidden width is 32 here, small enough that a degenerate row is possible.
    """

    norm = nn.LayerNorm(HIDDEN, eps=1e-5)
    constant = torch.full((8, HIDDEN), 3.0)

    assert torch.isfinite(norm(constant)).all()


# ---------------------------------------------------------------------------
# Reporting the affine parameters
# ---------------------------------------------------------------------------

def test_norm_statistics_reports_initialisation():
    """At init the gain is 1 and the bias 0, and the summary must say so."""

    wrapped = LayerNormGCNGRU(make_backbone(seed=0), normalise=True)
    statistics = norm_statistics(wrapped)

    assert statistics["gain_mean"] == pytest.approx(1.0)
    assert statistics["gain_sd"] == pytest.approx(0.0, abs=1e-6)
    assert statistics["bias_abs_mean"] == pytest.approx(0.0, abs=1e-6)


def test_norm_statistics_is_empty_without_a_norm():
    """The control arm has no affine parameters to report."""

    assert norm_statistics(LayerNormGCNGRU(make_backbone(), normalise=False)) == {}


def test_norm_statistics_tracks_a_collapsed_gain():
    """A gain driven toward zero must be visible, not averaged away.

    "The norm did not help" and "the model switched the norm off" are different
    findings, and this is the measurement that separates them.
    """

    wrapped = LayerNormGCNGRU(make_backbone(seed=0), normalise=True)

    with torch.no_grad():
        wrapped.norm.weight.fill_(0.01)

    assert norm_statistics(wrapped)["gain_mean"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# The experiment harness
# ---------------------------------------------------------------------------

def test_identity_adjacency_is_the_gru_only_control():
    """`--model gru_only` must hand the training loop a true identity."""

    adjacency = layer_norm_module.load_adjacency("gru_only", N_NODES)

    assert adjacency.shape == (N_NODES, N_NODES)
    np.testing.assert_array_equal(adjacency, np.eye(N_NODES, dtype=np.float32))


def test_comparison_is_anchored_on_the_baseline_arm():
    """`compare_to_control` must difference against `baseline`, per horizon."""

    rows = []
    for arm, mae in (("baseline", 20.0), ("layer_norm", 18.0)):
        for fold in range(1, 4):
            for seed in range(2):
                rows.append(
                    {
                        "arm": arm,
                        "horizon": 1,
                        "fold_id": fold,
                        "test_year": 2010 + fold,
                        "headline": True,
                        "covers_covid": False,
                        "seed": seed,
                        "best_epoch": 10.0,
                        "mae": mae + 0.1 * seed,
                        "ensemble": False,
                    }
                )

    import pandas as pd

    comparison = layer_norm_module.compare_to_control(pd.DataFrame(rows))

    assert list(comparison["arm"]) == ["layer_norm"]
    assert comparison["headline_delta"].iloc[0] == pytest.approx(-2.0)
    assert comparison["folds_improved"].iloc[0] == 3


def test_comparison_ignores_ensemble_rows():
    """Seed-mean rows must not be pooled with the single-seed rows.

    Mixing them would understate the spread the verdict is judged against.
    """

    import pandas as pd

    rows = []
    for arm, mae in (("baseline", 20.0), ("layer_norm", 19.0)):
        for fold in range(1, 4):
            rows.append(
                {
                    "arm": arm,
                    "horizon": 1,
                    "fold_id": fold,
                    "test_year": 2010 + fold,
                    "headline": True,
                    "covers_covid": False,
                    "seed": 0,
                    "best_epoch": 10.0,
                    "mae": mae,
                    "ensemble": False,
                }
            )
            # An ensemble row with an absurd score: if it leaked into the
            # comparison the delta would not be -1.0.
            rows.append(
                {
                    "arm": arm,
                    "horizon": 1,
                    "fold_id": fold,
                    "test_year": 2010 + fold,
                    "headline": True,
                    "covers_covid": False,
                    "seed": -1,
                    "best_epoch": 10.0,
                    "mae": 999.0,
                    "ensemble": True,
                }
            )

    comparison = layer_norm_module.compare_to_control(pd.DataFrame(rows))

    assert comparison["headline_delta"].iloc[0] == pytest.approx(-1.0)

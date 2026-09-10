"""
Tests for the simplex activations behind the lag encoder's basis mixture.

Two things are being protected here.

The first is correctness against a definition rather than against the
implementation's own output. `sparsemax` is checked against a separately
written Euclidean projection, and `entmax15` -- which is solved by bisection,
so there is no closed form to compare against -- is checked against the KKT
stationarity condition its solution must satisfy. A test that only asserted
"sums to one, non-negative" would pass for any number of wrong functions.

The second is the claim that motivates the module at all: that softmax
saturates and the sparse alternatives do not. That is measured, in
`test_softmax_gradient_vanishes_while_entmax_does_not`, because it is the
reason to prefer one of these over the default and it should fail loudly if it
ever stops being true.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.lag_encoder import LearnableLagEncoder  # noqa: E402
from src.models.simplex_activations import (  # noqa: E402
    SimplexActivation,
    entmax15,
    floored_entmax15,
    sparsemax,
)


ALL_NAMES = SimplexActivation.NAMES
SPARSE_NAMES = ("sparsemax", "entmax15")


def reference_projection(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto the simplex, written out the slow way.

    Deliberately a different implementation from `sparsemax` -- a Python loop
    over the sorted coordinates rather than a vectorised gather -- so that the
    two agreeing means the algorithm is right, not that the same expression was
    typed twice.
    """

    sorted_values, _ = torch.sort(v, descending=True)
    cumulative = sorted_values.cumsum(0)

    rho = 0
    for index in range(len(v)):
        if sorted_values[index] - (cumulative[index] - 1) / (index + 1) > 0:
            rho = index

    threshold = (cumulative[rho] - 1) / (rho + 1)

    return torch.clamp(v - threshold, min=0.0)


# ---------------------------------------------------------------------------
# Simplex membership
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_NAMES)
def test_output_lies_on_the_simplex(name):
    """Every scheme must return non-negative rows summing to one.

    This is what the encoder relies on: `kernels()` mixes the bumps with these
    weights and calls the result a weighted average. If a row summed to 1.2 the
    kernel would silently rescale the climate channel it smooths.
    """

    torch.manual_seed(0)
    activation = SimplexActivation(name).eval()
    weights = activation(torch.randn(64, 6) * 3.0)

    assert torch.all(weights >= 0.0)
    assert torch.allclose(weights.sum(-1), torch.ones(64), atol=1e-5)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_shape_and_dim_are_preserved(name):
    """The activation must work on the encoder's real [N, K, B] shape."""

    torch.manual_seed(0)
    activation = SimplexActivation(name).eval()
    weights = activation(torch.randn(25, 7, 6), dim=-1)

    assert weights.shape == (25, 7, 6)
    assert torch.allclose(weights.sum(-1), torch.ones(25, 7), atol=1e-5)


def test_unknown_activation_is_rejected():
    with pytest.raises(ValueError, match="Unknown activation"):
        SimplexActivation("softmax_but_better")


# ---------------------------------------------------------------------------
# Correctness against a definition
# ---------------------------------------------------------------------------

def test_sparsemax_matches_an_independent_projection():
    """sparsemax must equal the Euclidean projection it claims to be."""

    torch.manual_seed(1)
    logits = torch.randn(100, 6, dtype=torch.float64) * 2.0

    ours = sparsemax(logits)
    reference = torch.stack([reference_projection(row) for row in logits])

    assert torch.allclose(ours, reference, atol=1e-12)


def test_entmax15_satisfies_its_stationarity_condition():
    """On the support, p^(1/2) - z/2 must be the same constant for every i.

    This is the KKT condition for the alpha-entmax problem at alpha = 1.5, and
    it is what makes the bisection's answer the right answer rather than merely
    a point on the simplex.
    """

    torch.manual_seed(2)
    logits = torch.randn(50, 6, dtype=torch.float64) * 2.0
    weights = entmax15(logits)

    for row, probabilities in zip(logits, weights):
        support = probabilities > 1e-9
        offsets = probabilities[support].sqrt() - row[support] / 2.0

        assert (offsets - offsets.mean()).abs().max() < 1e-6


def test_softmax_name_is_exactly_torch_softmax():
    """The control arm must be the unchanged function, not a near copy.

    If this drifted, every comparison in the experiment would be against a
    baseline that is not the baseline.
    """

    torch.manual_seed(3)
    logits = torch.randn(32, 6)

    assert torch.equal(
        SimplexActivation("softmax")(logits), torch.softmax(logits, dim=-1)
    )


# ---------------------------------------------------------------------------
# Sparsity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SPARSE_NAMES)
def test_sparse_schemes_produce_exact_zeros(name):
    """The reason to use these at all: a bump can be given no mass whatsoever.

    Softmax cannot do this, which is why every softmax kernel is a blend of all
    six bumps rather than the single peak a biological delay looks like.
    """

    torch.manual_seed(4)
    logits = torch.randn(200, 6) * 3.0
    weights = SimplexActivation(name).eval()(logits)

    assert (weights == 0.0).any()
    assert (weights > 0).sum(-1).float().mean() < 6.0


def test_sparsity_is_ordered_softmax_then_entmax_then_sparsemax():
    """entmax15 must sit strictly between the dense and the sparsest scheme.

    That ordering is the reason both are in the sweep: if entmax collapsed as
    hard as sparsemax there would be no point running both.
    """

    torch.manual_seed(5)
    logits = torch.randn(500, 6) * 3.0

    def support(name):
        return (SimplexActivation(name).eval()(logits) > 1e-9).sum(-1).float().mean()

    assert support("softmax") == 6.0
    assert support("sparsemax") < support("entmax15") < support("softmax")


def test_floored_entmax_keeps_every_coordinate_alive():
    """The floored variant must never hand back a hard zero.

    That is its entire purpose -- a zero coordinate in a piecewise-linear map
    is a dead gradient that nothing can revive.
    """

    torch.manual_seed(6)
    logits = torch.randn(200, 6) * 10.0
    weights = floored_entmax15(logits)

    assert torch.all(weights > 0.0)
    assert torch.allclose(weights.sum(-1), torch.ones(200), atol=1e-5)


def test_floor_must_be_a_fraction():
    with pytest.raises(ValueError, match="floor must be in"):
        floored_entmax15(torch.randn(4, 6), floor=1.5)


# ---------------------------------------------------------------------------
# The gradient claim
# ---------------------------------------------------------------------------

def test_softmax_gradient_vanishes_while_entmax_does_not():
    """The measured justification for this whole module.

    As the logits grow, softmax's Jacobian diag(p) - p p^T goes to zero and the
    gradient reaching the basis logits dies. entmax15 is a projection on its
    support, so its gradient does not decay the same way. scripts/16 works
    around softmax's version of this with a 10x learning rate; this test is the
    evidence that the workaround is treating a real symptom.
    """

    logits = torch.tensor([2.0, 1.0, 0.5, 0.0, -0.5, -1.0])

    def gradient_norm(function, scale):
        scaled = (logits * scale).clone().requires_grad_(True)
        function(scaled, dim=-1).pow(2).sum().backward()
        return scaled.grad.norm().item()

    mild = gradient_norm(torch.softmax, 1.0)
    saturated = gradient_norm(torch.softmax, 20.0)
    extreme = gradient_norm(torch.softmax, 50.0)

    # Not merely smaller: gone. Seven orders of magnitude by scale 20, and by
    # scale 50 the gradient has underflowed past anything an optimiser can use.
    assert saturated < mild * 1e-7
    assert extreme < 1e-20

    # At a comparable, non-degenerate operating point entmax carries far more
    # gradient than softmax does.
    logits_soft = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    def gradient_at(function):
        point = logits_soft.clone().requires_grad_(True)
        function(point, dim=-1).pow(2).sum().backward()
        return point.grad.norm().item()

    assert gradient_at(entmax15) > 5.0 * gradient_at(torch.softmax)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_gradients_flow_to_the_logits(name):
    """Every scheme must be trainable at a normal operating point."""

    torch.manual_seed(7)
    activation = SimplexActivation(name)
    logits = (torch.randn(25, 7, 6) * 0.1).requires_grad_(True)

    activation(logits, dim=-1).pow(2).sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0.0


def test_one_hot_collapse_is_a_dead_end_for_the_sparse_schemes():
    """Document the failure mode `floored_entmax15` exists to avoid.

    This is not a bug being asserted as correct behaviour: at support size one
    the output really is locally constant, so a zero gradient is the true
    derivative. The point of pinning it is that it is an absorbing state -- the
    reason the floored variant is in the sweep, and a fact that should not be
    rediscovered by a confusing training run.
    """

    extreme = (torch.tensor([50.0, 1.0, 0.5, 0.0, -0.5, -1.0])).requires_grad_(True)
    weights = entmax15(extreme, dim=-1)

    assert int((weights > 1e-9).sum()) == 1

    weights.pow(2).sum().backward()
    assert extreme.grad.abs().sum() == 0.0

    # The floored variant survives the same input.
    rescued = (torch.tensor([50.0, 1.0, 0.5, 0.0, -0.5, -1.0])).requires_grad_(True)
    floored_entmax15(rescued, dim=-1).pow(2).sum().backward()
    assert torch.isfinite(rescued.grad).all()


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------

def test_learnable_temperature_is_a_parameter_and_stays_positive():
    activation = SimplexActivation("temp_softmax", init_temperature=1.0)

    assert activation.temperature_raw is not None
    assert activation.temperature().item() == pytest.approx(1.1, abs=1e-4)

    # Driven hard negative, the softplus floor still keeps it above zero, so
    # the softmax cannot invert.
    with torch.no_grad():
        activation.temperature_raw.fill_(-50.0)

    assert activation.temperature().item() > 0.0


def test_temperature_sharpens_the_distribution():
    """A low temperature must concentrate mass, a high one must spread it."""

    logits = torch.tensor([[2.0, 1.0, 0.5, 0.0, -0.5, -1.0]])

    def peak_at(temperature):
        activation = SimplexActivation("temp_softmax", init_temperature=temperature)
        return activation(logits).max().item()

    assert peak_at(0.25) > peak_at(1.0) > peak_at(8.0)


def test_parameter_free_schemes_have_no_temperature():
    activation = SimplexActivation("entmax15")

    assert activation.temperature_raw is None
    assert sum(p.numel() for p in activation.parameters()) == 0

    with pytest.raises(AttributeError):
        activation.temperature()


def test_gumbel_is_stochastic_in_training_and_deterministic_in_eval():
    """Sampling noise at evaluation would make the kernel report meaningless.

    Two evaluation passes over the same district must give the same delay, or
    `learned_lags.csv` records noise rather than a learned quantity.
    """

    torch.manual_seed(8)
    activation = SimplexActivation("gumbel_softmax")
    logits = torch.randn(16, 6)

    activation.train()
    assert not torch.equal(activation(logits), activation(logits))

    activation.eval()
    assert torch.equal(activation(logits), activation(logits))


# ---------------------------------------------------------------------------
# Integration with the encoder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ALL_NAMES)
def test_encoder_accepts_every_activation_and_keeps_its_contract(name):
    """Swapping the activation must not break anything the encoder promises.

    Kernels still sum to one, output shape is still the input minus the reach,
    and the delay is still causal. Those are properties of the encoder, not of
    the activation, and none of them may depend on which scheme is chosen.
    """

    torch.manual_seed(9)
    encoder = LearnableLagEncoder(
        n_nodes=25, n_features=7, lag_reach=26, activation=name
    ).eval()

    kernels = encoder.kernels()
    assert kernels.shape == (25, 7, 26)
    assert torch.all(kernels >= 0.0)
    assert torch.allclose(kernels.sum(-1), torch.ones(25, 7), atol=1e-4)

    output = encoder(torch.randn(4, 37, 25, 7))
    assert output.shape == (4, 12, 25, 7)
    assert torch.isfinite(output).all()

    lags = encoder.peak_lags()
    assert lags.shape == (25, 7)
    assert torch.all(lags >= 0.0) and torch.all(lags <= 25.0)


def test_encoder_default_is_still_plain_softmax():
    """The default path must be untouched, or every past result is invalidated.

    An encoder built with no `activation=` argument has to produce exactly what
    it produced before this module existed.
    """

    torch.manual_seed(10)
    encoder = LearnableLagEncoder(n_nodes=25, n_features=7)

    logits = torch.einsum(
        "nd,kdb->nkb", encoder.node_embedding, encoder.feature_projection
    )

    assert encoder.activation.name == "softmax"
    assert torch.equal(encoder.mixture_weights(), torch.softmax(logits, dim=-1))


@pytest.mark.parametrize("name", SPARSE_NAMES)
def test_sparse_activations_give_narrower_kernels(name):
    """The mechanism the experiment is testing, asserted at the kernel level.

    A sparse mixture drops bumps, so the resulting delay curve concentrates
    instead of spreading across the whole reach. If this stopped holding, the
    experiment would be varying something that no longer changes the kernel.
    """

    def kernel_spread(activation):
        torch.manual_seed(11)
        encoder = LearnableLagEncoder(
            n_nodes=25, n_features=7, activation=activation
        ).eval()

        # Push the logits off uniform, which is where they start; at
        # initialisation every scheme is near-uniform by construction.
        with torch.no_grad():
            encoder.node_embedding.mul_(30.0)
            encoder.feature_projection.mul_(30.0)

        kernels = encoder.kernels()
        tau = torch.arange(encoder.lag_reach, dtype=torch.float32)
        mean = (kernels * tau).sum(-1, keepdim=True)
        variance = (kernels * (tau - mean) ** 2).sum(-1)

        return variance.mean().item()

    assert kernel_spread(name) < kernel_spread("softmax")


def test_encoder_stays_causal_under_every_activation():
    """No future period may reach the output, whatever the mixture looks like.

    Perturbing a period after the window must leave the output untouched. This
    is structural in `forward`, but it is the one property whose violation
    would be silent and would invalidate every forecast.
    """

    for name in ALL_NAMES:
        torch.manual_seed(12)
        encoder = LearnableLagEncoder(
            n_nodes=5, n_features=3, lag_reach=8, activation=name
        ).eval()

        x = torch.randn(1, 12, 5, 3)
        before = encoder(x)

        future = x.clone()
        future[:, -1] += 100.0

        # The last input period feeds only the last output step, so every
        # earlier step must be bit-identical.
        assert torch.allclose(encoder(future)[:, :-1], before[:, :-1], atol=1e-6)

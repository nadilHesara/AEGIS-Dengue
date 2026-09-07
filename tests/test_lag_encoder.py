"""
Tests for the learnable lag encoder.

The one that matters is `test_recovers_planted_delays`. Everything else in this
component is judged by a change in MAE, which is a number that can move for a
dozen reasons. The recovery test is the only check that asks the module the
question it exists to answer: when a delay is definitely there, do you find it?
Fake data is built where district 0's cases depend on rainfall 4 periods ago,
district 12's on rainfall 11 periods ago and district 24's on rainfall 18
periods ago, and the encoder has to report those delays back.

If that test fails, no result from this component means anything, whichever way
the MAE moved.

The rest guard the structural properties the parameterisation claims: the curves
are normalised, and they cannot see the future.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.models.lag_encoder import LearnableLagEncoder  # noqa: E402


LAG_REACH = 26


def make_encoder(n_nodes=25, n_features=1, **kwargs):
    return LearnableLagEncoder(
        n_nodes=n_nodes, n_features=n_features, lag_reach=LAG_REACH, **kwargs
    )


# ---------------------------------------------------------------------------
# The one that matters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_recovers_planted_delays(seed):
    """Plant a different known delay in each district and recover it."""

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n_nodes, n_periods = 25, 900
    true_delays = np.linspace(3, 19, n_nodes).round().astype(int)

    # A slow, autocorrelated driver. White noise would let the encoder locate
    # the delay from a single spike, which is easier than the real problem.
    noise = rng.normal(size=(n_periods + 200, n_nodes))
    driver = np.stack(
        [np.convolve(noise[:, i], np.ones(5) / 5, mode="same") for i in range(n_nodes)],
        axis=1,
    )[200:]

    target = np.stack(
        [np.roll(driver[:, i], true_delays[i]) for i in range(n_nodes)], axis=1
    )
    target += 0.05 * rng.normal(size=target.shape)

    x = torch.tensor(driver, dtype=torch.float32)[None, :, :, None]
    y = torch.tensor(target, dtype=torch.float32)[None, LAG_REACH - 1 :, :]

    encoder = make_encoder(n_nodes=n_nodes, n_features=1)
    scale = torch.nn.Parameter(torch.ones(1))

    optimiser = torch.optim.Adam(
        [{"params": encoder.parameters()}, {"params": [scale]}], lr=0.05
    )

    for _ in range(600):
        optimiser.zero_grad()
        prediction = encoder(x).squeeze(-1) * scale
        loss = torch.mean((prediction - y) ** 2)
        loss.backward()
        optimiser.step()

    recovered = encoder.peak_lags().detach().squeeze(-1).numpy()
    error = np.abs(recovered - true_delays)

    assert error.max() <= 2.0, (
        f"Worst district off by {error.max():.1f} periods. "
        f"planted {true_delays.tolist()}, recovered {recovered.round(1).tolist()}"
    )


def test_recovers_different_delays_per_feature():
    """Two features in the same district can hold different delays."""

    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    n_nodes, n_periods = 4, 800
    fast_delay, slow_delay = 4, 16

    driver = rng.normal(size=(n_periods, n_nodes, 2)).astype(np.float32)
    driver = np.stack(
        [
            np.apply_along_axis(
                lambda v: np.convolve(v, np.ones(5) / 5, mode="same"), 0, driver[..., k]
            )
            for k in range(2)
        ],
        axis=-1,
    )

    target = np.roll(driver[..., 0], fast_delay, axis=0) + np.roll(
        driver[..., 1], slow_delay, axis=0
    )

    x = torch.tensor(driver, dtype=torch.float32)[None]
    y = torch.tensor(target, dtype=torch.float32)[None, LAG_REACH - 1 :]

    encoder = make_encoder(n_nodes=n_nodes, n_features=2)
    optimiser = torch.optim.Adam(encoder.parameters(), lr=0.05)

    for _ in range(600):
        optimiser.zero_grad()
        loss = torch.mean((encoder(x).sum(-1) - y) ** 2)
        loss.backward()
        optimiser.step()

    peaks = encoder.peak_lags().detach().numpy()

    assert np.abs(peaks[:, 0] - fast_delay).max() <= 2.5
    assert np.abs(peaks[:, 1] - slow_delay).max() <= 2.5


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------

def test_kernels_are_normalised_and_non_negative():
    """Every curve is a weighting, so it sums to one and never goes negative."""

    encoder = make_encoder(n_features=7)
    kernels = encoder.kernels()

    assert kernels.shape == (25, 7, LAG_REACH)
    assert torch.all(kernels >= 0)
    assert torch.allclose(kernels.sum(-1), torch.ones(25, 7), atol=1e-5)


def test_output_shape_trims_the_incomplete_history():
    """37 periods in, 12 out: the first lag_reach - 1 have no full history."""

    encoder = make_encoder(n_features=7)
    output = encoder(torch.randn(8, 37, 25, 7))

    assert output.shape == (8, 12, 25, 7)


def test_cannot_see_the_future():
    """Changing a period after the output window changes no output."""

    torch.manual_seed(0)
    encoder = make_encoder(n_features=3)

    x = torch.randn(2, 30, 25, 3)
    baseline = encoder(x)

    # The output covers periods LAG_REACH - 1 .. 29. Perturbing the last input
    # period may only touch the last output step, never any earlier one.
    perturbed = x.clone()
    perturbed[:, -1] += 10.0

    changed = encoder(perturbed)

    assert torch.allclose(baseline[:, :-1], changed[:, :-1], atol=1e-5)
    assert not torch.allclose(baseline[:, -1], changed[:, -1], atol=1e-5)


def test_all_parameter_groups_receive_gradient():
    """A parameter with no gradient is a parameter that is not being learned."""

    encoder = make_encoder(n_features=3)
    encoder(torch.randn(4, 30, 25, 3)).sum().backward()

    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert parameter.grad.abs().sum() > 0, f"{name} received a zero gradient"


def test_districts_are_free_to_differ():
    """Two districts with different embeddings get different curves.

    The point of the component. If the embedding did not reach the kernel this
    would pass silently while the model learned one national delay.
    """

    encoder = make_encoder(n_features=2)
    with torch.no_grad():
        encoder.node_embedding[0] = torch.randn(8)
        encoder.node_embedding[1] = -encoder.node_embedding[0]

    kernels = encoder.kernels()

    assert not torch.allclose(kernels[0], kernels[1], atol=1e-4)

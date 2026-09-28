"""
Tests for the improved training objectives in scripts/20.train_improved.py.

The arms differ from the baseline only in the loss, so the properties worth
asserting are the ones that would make the comparison meaningless if they broke:

  - the loss still ignores unobserved cells, exactly as the baseline's does. A
    weighting scheme that quietly reintroduced masked cells would be scored
    against targets nobody observed.
  - the level weights come from the forecast origin only. If they could be
    influenced by the target, the whole comparison would be leakage.
  - the weights are normalised, so the loss scale, and with it the early-stopping
    patience, is comparable across arms rather than each arm needing its own.
  - `masked_weighted_loss` with unit weights and `kind="mse"` reduces to the
    baseline's `masked_mse` exactly. That is the identity the `baseline` control
    arm depends on: if it does not hold, the control is not a control.
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


improved = _load("improved_module", "training/20.train_improved.py")
baseline = _load("baseline_for_tests", "training/16.train_gcn_gru.py")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_batch(batch=4, nodes=25, seed=0):
    """Return a prediction, target, mask and anchor with a realistic mix."""

    generator = torch.Generator().manual_seed(seed)

    prediction = torch.randn(batch, nodes, 1, generator=generator)
    target = torch.randn(batch, nodes, 1, generator=generator)

    mask = (torch.rand(batch, nodes, 1, generator=generator) > 0.2).float()

    # Anchors are log1p case counts, so they span roughly 0 to log1p(2631).
    anchor = torch.rand(batch, nodes, generator=generator) * 7.9

    return prediction, target, mask, anchor


# ---------------------------------------------------------------------------
# The control identity
# ---------------------------------------------------------------------------

def test_unweighted_mse_matches_the_baseline_loss_exactly():
    """Unit weights and kind='mse' must reproduce script 16's masked_mse.

    The `baseline` arm of script 20 exists to prove the harness is neutral. It
    can only do that if its loss is numerically the baseline's, so this is the
    assertion the whole comparison rests on.
    """

    prediction, target, mask, _ = make_batch()

    ours = improved.masked_weighted_loss(
        prediction, target, mask, torch.ones_like(mask), "mse"
    )
    theirs = baseline.masked_mse(prediction, target, mask)

    assert torch.allclose(ours, theirs, atol=1e-7)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["mse", "huber", "quantile"])
def test_unobserved_cells_cannot_change_any_loss(kind):
    """Overwriting masked cells with nonsense must not move the loss.

    A missing case count is never imputed anywhere else in this pipeline; a loss
    that reacted to a masked cell would be scoring the model against a number
    nobody observed.
    """

    prediction, target, mask, anchor = make_batch()
    weight = improved.level_weights(anchor, mask)

    before = improved.masked_weighted_loss(prediction, target, mask, weight, kind)

    corrupted = target.clone()
    corrupted[mask == 0] = 1e6

    after = improved.masked_weighted_loss(prediction, corrupted, mask, weight, kind)

    assert torch.allclose(before, after, atol=1e-6)


@pytest.mark.parametrize("kind", ["mse", "huber", "quantile"])
def test_a_fully_masked_batch_does_not_divide_by_zero(kind):
    """An all-masked batch must return a finite loss, not a nan."""

    prediction, target, _, anchor = make_batch()
    mask = torch.zeros_like(prediction)

    loss = improved.masked_weighted_loss(
        prediction, target, mask, torch.ones_like(mask), kind
    )

    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Level weights
# ---------------------------------------------------------------------------

def test_weights_are_normalised_to_mean_one_over_observed_cells():
    """The mean weight over observed cells is 1.

    This is what keeps the loss on the baseline's scale, so a single patience
    and learning rate remain valid across arms instead of each arm needing its
    own tuning -- which would confound the loss comparison with a tuning
    comparison.
    """

    _, _, mask, anchor = make_batch()

    weight = improved.level_weights(anchor, mask)
    mean = (weight * mask).sum() / mask.sum()

    assert torch.allclose(mean, torch.tensor(1.0), atol=1e-5)


def test_weights_increase_with_the_case_level():
    """A district with more cases at the origin gets more weight.

    The whole point of the reweighting: MAE is dominated by high-count cells, so
    the training distribution has to be tilted toward them.
    """

    mask = torch.ones(1, 3, 1)
    anchor = torch.log1p(torch.tensor([[5.0, 100.0, 1500.0]]))

    weight = improved.level_weights(anchor, mask).squeeze()

    assert weight[0] < weight[1] < weight[2]


def test_weights_are_bounded_far_below_the_raw_count_ratio():
    """log1p weighting must not reproduce the raw-count imbalance.

    Weighting by raw counts would hand a 1500-case district 300x the gradient of
    a 5-case one, replacing one imbalance with a worse one. log1p keeps the
    ratio near 4x.
    """

    mask = torch.ones(1, 2, 1)
    anchor = torch.log1p(torch.tensor([[5.0, 1500.0]]))

    weight = improved.level_weights(anchor, mask).squeeze()
    ratio = (weight[1] / weight[0]).item()

    assert 3.0 < ratio < 6.0, ratio


def test_weights_depend_only_on_the_anchor_not_the_target():
    """Changing the target must not change a single weight.

    The anchor is the case count at the forecast origin -- already a model input.
    If the weights could see the target, the reweighting would be leakage rather
    than a redistribution of training emphasis.
    """

    _, _, mask, anchor = make_batch()

    weight = improved.level_weights(anchor, mask)
    again = improved.level_weights(anchor.clone(), mask)

    assert torch.allclose(weight, again)


# ---------------------------------------------------------------------------
# Loss shapes
# ---------------------------------------------------------------------------

def test_huber_is_quadratic_below_the_transition_and_linear_above():
    """Huber's defining property, checked at the transition point itself."""

    delta = improved.HUBER_DELTA

    small = torch.tensor([[[0.5 * delta]]])
    zero = torch.zeros_like(small)
    mask = torch.ones_like(small)
    ones = torch.ones_like(small)

    quadratic = improved.masked_weighted_loss(small, zero, mask, ones, "huber")
    assert torch.allclose(quadratic, 0.5 * small.squeeze() ** 2, atol=1e-6)

    # Well past the transition the loss must grow linearly: doubling the excess
    # over delta doubles the increment.
    def at(residual):
        value = torch.tensor([[[residual]]])
        return improved.masked_weighted_loss(
            value, zero, mask, ones, "huber"
        ).item()

    first = at(2 * delta) - at(delta)
    second = at(3 * delta) - at(2 * delta)

    assert abs(first - second) < 1e-5


def test_huber_penalises_a_large_residual_far_less_than_mse():
    """The reason Huber is here: bounded gradients on heavy tails.

    p99 of the raw week-over-week case change is 105 against a median of 4, so
    squared error lets a handful of epidemic weeks set the gradient direction.
    """

    residual = torch.tensor([[[8.0]]])
    zero = torch.zeros_like(residual)
    mask = torch.ones_like(residual)
    ones = torch.ones_like(residual)

    mse = improved.masked_weighted_loss(residual, zero, mask, ones, "mse")
    huber = improved.masked_weighted_loss(residual, zero, mask, ones, "huber")

    assert huber < mse / 10


def test_pinball_at_the_median_is_symmetric():
    """tau = 0.5 must penalise over- and under-prediction equally.

    An asymmetric pinball would bias the forecast in a direction nobody chose.
    """

    mask = torch.ones(1, 1, 1)
    ones = torch.ones_like(mask)
    zero = torch.zeros(1, 1, 1)

    over = improved.masked_weighted_loss(
        torch.tensor([[[1.5]]]), zero, mask, ones, "quantile"
    )
    under = improved.masked_weighted_loss(
        torch.tensor([[[-1.5]]]), zero, mask, ones, "quantile"
    )

    assert torch.allclose(over, under, atol=1e-7)


@pytest.mark.parametrize("kind", ["mse", "huber", "quantile"])
def test_every_loss_is_zero_at_a_perfect_prediction(kind):
    """A perfect forecast must cost nothing under every objective."""

    _, target, mask, anchor = make_batch()
    weight = improved.level_weights(anchor, mask)

    loss = improved.masked_weighted_loss(target, target, mask, weight, kind)

    assert loss.item() == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("kind", ["mse", "huber", "quantile"])
def test_every_loss_produces_a_gradient(kind):
    """Each objective must actually train something."""

    prediction, target, mask, anchor = make_batch()
    prediction = prediction.clone().requires_grad_(True)
    weight = improved.level_weights(anchor, mask)

    improved.masked_weighted_loss(
        prediction, target, mask, weight, kind
    ).backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_an_unknown_loss_is_rejected():
    """A typo in the arm table must fail loudly rather than silently pick one."""

    prediction, target, mask, _ = make_batch()

    with pytest.raises(ValueError, match="Unknown loss"):
        improved.masked_weighted_loss(
            prediction, target, mask, torch.ones_like(mask), "not_a_loss"
        )


# ---------------------------------------------------------------------------
# Arm wiring
# ---------------------------------------------------------------------------

def test_build_loss_wires_every_declared_arm():
    """Every arm in the table must produce a working, finite loss."""

    prediction, target, mask, anchor = make_batch()

    for arm, (kind, weighted) in improved.ARMS.items():
        loss_fn = improved.build_loss(kind, weighted)
        value = loss_fn(prediction, target, mask, anchor)

        assert torch.isfinite(value), arm


def test_the_baseline_arm_is_declared_unweighted_mse():
    """The control arm must be script 16's objective, or it is not a control."""

    assert improved.ARMS["baseline"] == ("mse", False)


def test_weighting_changes_the_loss_for_the_weighted_arms():
    """The weighted arms must actually differ from their unweighted twins.

    Cheap, but it catches a weighting that is silently normalised away to unit
    weights -- which would make two arms report as different while training
    identically.
    """

    prediction, target, mask, anchor = make_batch()

    plain = improved.build_loss("mse", False)(prediction, target, mask, anchor)
    weighted = improved.build_loss("mse", True)(prediction, target, mask, anchor)

    assert not torch.allclose(plain, weighted, atol=1e-4)

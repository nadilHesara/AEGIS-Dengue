"""Tests for the Negative Binomial module."""

import numpy as np
import pytest
import scipy.stats as stats
import torch

from src.models.negative_binomial import (
    NegBinGCNGRU,
    NegBinHead,
    compute_level_weights,
    compute_outbreak_probability,
    compute_prediction_intervals,
    masked_negative_binomial_loss,
    negative_binomial_nll,
)
from src.models.stgnn import GCNGRU


def test_nll_matches_scipy_exact():
    """Verify our PyTorch NLL implementation matches scipy.stats.nbinom."""
    mu_vals = [2.0, 15.0, 120.0]
    alpha_vals = [0.1, 0.5, 2.0]
    targets = [0, 5, 25, 100]

    for mu in mu_vals:
        for alpha in alpha_vals:
            for y in targets:
                r = 1.0 / alpha
                p = r / (r + mu)
                # scipy logpmf
                scipy_logpmf = stats.nbinom.logpmf(y, r, p)

                # torch nll
                mu_tensor = torch.tensor([mu], dtype=torch.float32)
                alpha_tensor = torch.tensor([alpha], dtype=torch.float32)
                target_tensor = torch.tensor([y], dtype=torch.float32)

                torch_nll = negative_binomial_nll(mu_tensor, alpha_tensor, target_tensor)
                expected_nll = -scipy_logpmf

                assert np.isclose(torch_nll.item(), expected_nll, atol=1e-4), (
                    f"Mismatch for mu={mu}, alpha={alpha}, y={y}: "
                    f"torch={torch_nll.item()}, scipy={expected_nll}"
                )


def test_nll_handles_zeros_without_nan():
    """Target counts of 0 must evaluate smoothly without NaN."""
    mu = torch.tensor([0.1, 5.0, 50.0], dtype=torch.float32)
    alpha = torch.tensor([0.2, 0.5, 1.0], dtype=torch.float32)
    target = torch.zeros(3, dtype=torch.float32)

    nll = negative_binomial_nll(mu, alpha, target)
    assert torch.isfinite(nll).all()
    assert (nll > 0).all()


def test_masked_loss():
    """Masked loss must ignore unobserved cells and normalize by observed count."""
    mu = torch.tensor([10.0, 20.0], dtype=torch.float32)
    alpha = torch.tensor([0.5, 0.5], dtype=torch.float32)
    target = torch.tensor([10.0, 500.0], dtype=torch.float32)
    # Mask out the second element (500.0)
    mask = torch.tensor([1.0, 0.0], dtype=torch.float32)

    loss = masked_negative_binomial_loss(mu, alpha, target, mask)

    # Expected: loss of only element 0
    single_nll = negative_binomial_nll(mu[:1], alpha[:1], target[:1])
    assert np.isclose(loss.item(), single_nll.item(), atol=1e-5)


def test_head_anchoring_initialization():
    """NegBinHead should initialize near the persistence anchor."""
    head = NegBinHead(hidden_dim=32, horizon=1)
    hidden = torch.zeros(25, 32)
    cases_origin = torch.tensor([0.0, 5.0, 50.0, 200.0] + [10.0] * 21)
    anchor = torch.log1p(cases_origin)

    mu, alpha = head(hidden, anchor)

    # Since weights & bias are 0, mu should be exp(log1p(cases)) = cases + 1
    expected_mu = cases_origin + 1.0
    assert torch.allclose(mu.squeeze(-1), expected_mu, atol=1e-4)

    # alpha should be softplus(0) + 1e-4 ~ log(2) ~ 0.693
    assert (alpha > 0).all()
    assert torch.allclose(alpha.squeeze(-1), torch.full_like(cases_origin, 0.6932), atol=1e-2)


def test_gradient_flow():
    """Gradients must flow to both mu and alpha parameters."""
    head = NegBinHead(hidden_dim=16, horizon=1)
    hidden = torch.randn(10, 16, requires_grad=True)
    anchor = torch.log1p(torch.tensor([5.0] * 10))
    target = torch.tensor([8.0] * 10)
    mask = torch.ones(10)

    mu, alpha = head(hidden, anchor)
    loss = masked_negative_binomial_loss(mu.squeeze(-1), alpha.squeeze(-1), target, mask)
    loss.backward()

    assert head.mu_linear.weight.grad is not None
    assert head.alpha_linear.weight.grad is not None
    assert torch.isfinite(head.mu_linear.weight.grad).all()
    assert torch.isfinite(head.alpha_linear.weight.grad).all()


def test_negbin_gcngru_forward():
    """Test full forward pass of NegBinGCNGRU."""
    backbone = GCNGRU(n_features=23, hidden=16, gcn_layers=1, horizon=1)
    model = NegBinGCNGRU(backbone=backbone, horizon=1)

    batch_size = 4
    steps = 12
    nodes = 25
    features = 23

    x = torch.randn(batch_size, steps, nodes, features)
    adjacency = torch.eye(nodes)
    anchor = torch.log1p(torch.randint(0, 100, (batch_size, nodes)).float())

    mu, alpha = model(x, adjacency, anchor)

    assert mu.shape == (batch_size, nodes, 1)
    assert alpha.shape == (batch_size, nodes, 1)
    assert (mu > 0).all()
    assert (alpha > 0).all()


def test_prediction_intervals_and_outbreak_prob():
    """Verify intervals and exceedance probability calculation."""
    mu = np.array([20.0, 50.0])
    alpha = np.array([0.5, 0.2])

    intervals = compute_prediction_intervals(mu, alpha, quantiles=(0.1, 0.5, 0.9))
    assert 0.1 in intervals and 0.5 in intervals and 0.9 in intervals
    assert (intervals[0.1] <= intervals[0.5]).all()
    assert (intervals[0.5] <= intervals[0.9]).all()

    prob = compute_outbreak_probability(mu, alpha, threshold=40.0)
    assert (prob >= 0.0).all() and (prob <= 1.0).all()
    # District with mean 50 should have higher outbreak probability than district with mean 20
    assert prob[1] > prob[0]


def test_level_weighted_loss():
    """Verify level weights normalize to mean 1 and scale loss appropriately."""
    anchor = torch.log1p(torch.tensor([[10.0, 50.0], [5.0, 100.0]]))  # [2, 2]
    mask = torch.ones(2, 2)

    weights = compute_level_weights(anchor, mask)
    assert weights.shape == (2, 2)
    assert torch.allclose(weights.mean(), torch.tensor(1.0), atol=1e-5)
    # Higher anchor must receive higher weight
    assert weights[1, 1] > weights[1, 0]

    mu = torch.full((2, 2), 20.0)
    alpha = torch.full((2, 2), 0.5)
    target = torch.tensor([[15.0, 60.0], [8.0, 120.0]])

    unweighted_loss = masked_negative_binomial_loss(mu, alpha, target, mask)
    weighted_loss = masked_negative_binomial_loss(mu, alpha, target, mask, weight=weights)

    assert torch.isfinite(weighted_loss)
    assert weighted_loss > 0


def test_cumulative_trajectory_head():
    """Verify cumulative trajectory parameterization cascades multipliers across horizons."""
    head = NegBinHead(hidden_dim=16, horizon=4, cumulative=True)
    # Zero init should produce persistence exactly across all horizons
    hidden = torch.zeros(10, 16)
    anchor = torch.log1p(torch.tensor([[25.0]] * 10))
    mu, alpha = head(hidden, anchor)

    assert mu.shape == (10, 1, 4)
    # At zero init, mu == exp(anchor) == 26.0 for all horizons
    assert torch.allclose(mu, torch.tensor(26.0), atol=1e-5)

    # Set non-zero weights: positive growth rate of +0.1 per week
    with torch.no_grad():
        head.mu_linear.bias.copy_(torch.tensor([0.1, 0.1, 0.1, 0.1]))

    mu_growth, _ = head(hidden, anchor)
    # Check that each horizon strictly grows: mu[h=1] < mu[h=2] < mu[h=3] < mu[h=4]
    for h in range(3):
        assert (mu_growth[..., h] < mu_growth[..., h + 1]).all()

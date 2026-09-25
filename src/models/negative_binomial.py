"""Negative Binomial distribution module for probabilistic dengue forecasting.

Epidemiological count data exhibits extreme overdispersion (Var(Y) >> E[Y])
and zero-inflation that Gaussian/MSE assumptions distort. This module
implements a Negative Binomial (NB2) parameterization:

    E[Y] = μ
    Var(Y) = μ + α * μ²

where:
    μ > 0 is the predicted mean count, anchored to the forecast origin:
        log(μ) = log(1 + y_origin) + Δ_μ
    α > 0 is the overdispersion parameter (α -> 0 recovers Poisson).
    r = 1 / α is the dispersion (failure/shape) parameter.
"""

from __future__ import annotations

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F


def negative_binomial_nll(
    mu: torch.Tensor,
    alpha: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute elementwise Negative Binomial (NB2) negative log-likelihood.

    Log-likelihood formula:
        log P(Y = y | μ, α) =
            log Γ(y + r) - log Γ(r) - log Γ(y + 1)
            - r * log1p(α * μ)
            + y * log(α * μ) - y * log1p(α * μ)
    where r = 1 / α.

    Args:
        mu: Predicted mean count, shape [...] (strictly positive).
        alpha: Predicted overdispersion parameter, shape [...] (strictly positive).
        target: Observed count, shape [...] (non-negative).
        eps: Small constant for numerical stability.

    Returns:
        Elementwise negative log-likelihood with the same shape as target.
    """
    mu = mu.clamp(min=eps)
    alpha = alpha.clamp(min=eps, max=1e4)
    r = 1.0 / alpha

    # α * μ
    alpha_mu = (alpha * mu).clamp(min=eps)

    # Log terms using log1p for stability
    log1p_alpha_mu = torch.log1p(alpha_mu)

    # y * log(α * μ): for y == 0, y * log(...) is mathematically 0
    y_log_alpha_mu = torch.where(
        target > 0,
        target * torch.log(alpha_mu),
        torch.zeros_like(target),
    )

    log_prob = (
        torch.lgamma(target + r)
        - torch.lgamma(r)
        - torch.lgamma(target + 1.0)
        - r * log1p_alpha_mu
        + y_log_alpha_mu
        - target * log1p_alpha_mu
    )

    return -log_prob


def masked_negative_binomial_loss(
    mu: torch.Tensor,
    alpha: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Masked Negative Binomial negative log-likelihood over observed cells.

    Args:
        mu: Predicted mean counts.
        alpha: Predicted overdispersion parameter.
        target: Target case counts.
        mask: Binary observation mask (1 for observed, 0 for missing).
        eps: Small constant for denominator clamp.

    Returns:
        Scalar loss normalized by number of observed cells.
    """
    nll = negative_binomial_nll(mu, alpha, target, eps=eps)
    effective = nll * mask
    denominator = mask.sum().clamp(min=1.0)
    return effective.sum() / denominator


class NegBinHead(nn.Module):
    """Output head predicting (μ, α) with epidemiological origin anchoring.

    The mean μ is parameterized via log-multiplicative adjustment to the
    anchor case count at the forecast origin:
        log(μ) = anchor + Δ_μ
        μ = exp(anchor + Δ_μ) = (1 + y_origin) * exp(Δ_μ)

    When Δ_μ = 0, the predicted mean exactly reproduces the persistence anchor.
    Overdispersion α is constrained strictly positive via softplus.
    """

    def __init__(self, hidden_dim: int, horizon: int = 1):
        super().__init__()
        self.horizon = horizon
        self.mu_linear = nn.Linear(hidden_dim, horizon)
        self.alpha_linear = nn.Linear(hidden_dim, horizon)

        # Initialize mu weights near zero so model starts near persistence baseline
        nn.init.zeros_(self.mu_linear.weight)
        nn.init.zeros_(self.mu_linear.bias)

        # Initialize alpha bias to 0.0 -> softplus(0) = log(2) ~ 0.69 (sensible overdispersion start)
        nn.init.zeros_(self.alpha_linear.weight)
        nn.init.zeros_(self.alpha_linear.bias)

    def forward(
        self, hidden: torch.Tensor, anchor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            hidden: Sequence representation [batch * nodes, hidden_dim]
            anchor: Log1p case counts at origin [batch, nodes] or [batch, nodes, 1]

        Returns:
            mu: Predicted mean count [batch, nodes, horizon]
            alpha: Predicted overdispersion [batch, nodes, horizon]
        """
        is_1d = (anchor.ndim == 1)
        if anchor.ndim == 1:
            anchor_expanded = anchor.unsqueeze(0).unsqueeze(-1)
        elif anchor.ndim == 2:
            anchor_expanded = anchor.unsqueeze(-1)
        else:
            anchor_expanded = anchor

        batch, nodes = anchor_expanded.shape[0], anchor_expanded.shape[1]

        delta_mu = self.mu_linear(hidden)
        alpha_raw = self.alpha_linear(hidden)

        delta_mu = delta_mu.view(batch, nodes, self.horizon)
        # Clamp delta to prevent float exp overflow: [-10, 10] covers factors from 4.5e-5 to 2.2e4
        delta_mu = delta_mu.clamp(min=-10.0, max=10.0)

        # Compute anchored mean μ: exp(anchor + Δ)
        mu = torch.exp(anchor_expanded + delta_mu)

        # Overdispersion α must be strictly positive: softplus + eps
        alpha = F.softplus(alpha_raw.view(batch, nodes, self.horizon)) + 1e-4

        if is_1d:
            return mu.squeeze(0), alpha.squeeze(0)

        return mu, alpha


class NegBinGCNGRU(nn.Module):
    """GCN-GRU with Negative Binomial probabilistic output head."""

    def __init__(
        self,
        backbone: nn.Module,
        horizon: int = 1,
    ):
        super().__init__()
        self.backbone = backbone
        self.horizon = horizon
        hidden_dim = backbone.gru.hidden_size
        self.head = NegBinHead(hidden_dim, horizon=horizon)

        # Bypass backbone's original linear head to keep parameter count honest
        self.backbone.head = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through spatio-temporal backbone and NegBin head.

        Args:
            x: Input windows [batch, steps, nodes, features]
            adjacency: District adjacency matrix [nodes, nodes]
            anchor: Log1p cases at forecast origin [batch, nodes]

        Returns:
            mu: Mean predicted cases [batch, nodes, horizon]
            alpha: Overdispersion [batch, nodes, horizon]
        """
        batch, steps, nodes, _ = x.shape
        spatial = x
        for layer in self.backbone.graph_layers:
            spatial = self.backbone.dropout(torch.relu(layer(spatial, adjacency)))

        sequences = spatial.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)
        output, _ = self.backbone.gru(sequences)
        last_hidden = self.backbone.dropout(output[:, -1])

        return self.head(last_hidden, anchor)


def compute_prediction_intervals(
    mu: np.ndarray,
    alpha: np.ndarray,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[float, np.ndarray]:
    """Calculate prediction intervals from fitted Negative Binomial parameters.

    Args:
        mu: Mean array [...].
        alpha: Overdispersion array [...].
        quantiles: Tuple of quantiles (e.g., 0.1 for 10th percentile).

    Returns:
        Dict mapping quantile float to predicted case counts array.
    """
    r = 1.0 / np.maximum(alpha, 1e-6)
    p = r / (r + np.maximum(mu, 1e-6))
    # In scipy.stats.nbinom, n = r (number of successes), p = success probability
    intervals = {}
    for q in quantiles:
        intervals[q] = stats.nbinom.ppf(q, r, p)
    return intervals


def compute_outbreak_probability(
    mu: np.ndarray,
    alpha: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Calculate probability of exceeding a given epidemic case threshold P(Y >= threshold).

    Args:
        mu: Mean array.
        alpha: Overdispersion array.
        threshold: Outbreak cutoff value.

    Returns:
        Probability array in [0, 1].
    """
    r = 1.0 / np.maximum(alpha, 1e-6)
    p = r / (r + np.maximum(mu, 1e-6))
    # P(Y >= threshold) = 1 - P(Y <= threshold - 1) = sf(threshold - 1)
    k = max(0, int(threshold) - 1)
    return stats.nbinom.sf(k, r, p)

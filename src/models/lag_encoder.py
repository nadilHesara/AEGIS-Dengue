"""
Learnable, per-district climate lag kernels.

The baseline hands the model three trailing means of rainfall, temperature and
humidity over 4, 8 and 12 periods. Those windows were chosen by hand, and they
are the same for all 25 districts. Dengue does not work that way: rain has to
pool, mosquitoes have to breed, people have to be bitten, fall ill and reach a
hospital, and how long that chain takes depends on where you are.

So learn the delay instead of asserting it. For district i and climate feature
k, the encoder builds a weighting curve over the previous L periods,

    x_smoothed[i, k, t] = sum over tau of  w[i, k, tau] * x[i, k, t - tau]

and applies it as a causal convolution.

The parameterisation is the whole point. A free curve is L * K * N = 26 * 7 * 25
= 4,550 weights, learned from roughly 500 training windows, which memorises the
training years and produces noise that cannot be read. Instead every curve is a
mixture of B smooth Gaussian bumps:

    w[i, k] = sum over b of  alpha[i, k, b] * basis[b]

where only the bump centres and widths (2B numbers), a district embedding
(N * D) and a per-feature projection (K * D * B) are learned. About 550
parameters, and each curve is smooth, non-negative and sums to one by
construction -- which is what a biological delay actually looks like, and what
makes the learned kernel readable as "district i responds to rainfall about 11
weeks later".

Causality is structural, not a convention: tau only ever indexes backwards, so
no future period can reach the output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# Bump centres in reporting periods, spread across the reach.
#
# Six, not four, and the count was chosen by measurement. A mixture can only
# interpolate between its bumps, never past the outermost one, and the centres
# contract towards wherever the data puts its mass rather than spreading to
# cover the reach. With four bumps the synthetic recovery test quantised: every
# district wanting a delay beyond the last centre was pinned to it, and the
# worst district came out 2.3 periods wrong. Six bumps give a fine enough grid
# to interpolate any delay in the range (worst district 1.7, mean 0.2) for
# about 120 extra parameters.
DEFAULT_CENTRES = (0.0, 4.0, 8.0, 13.0, 18.0, 24.0)

DEFAULT_LAG_REACH = 26


def _inverse_softplus(value: float) -> float:
    """Return the raw parameter whose softplus is `value`."""

    return float(torch.log(torch.expm1(torch.tensor(value))))


class LearnableLagEncoder(nn.Module):
    """Smooth, causal, per-district, per-feature delay kernels.

    Input  [batch, steps, nodes, features]
    Output [batch, steps - lag_reach + 1, nodes, features]

    The output is shorter than the input because the first `lag_reach - 1`
    periods have no complete history behind them. Feed a window of
    `model_lookback + lag_reach - 1` periods to get `model_lookback` back.
    """

    def __init__(
        self,
        n_nodes: int,
        n_features: int,
        lag_reach: int = DEFAULT_LAG_REACH,
        n_basis: int = 6,
        embedding_dim: int = 8,
        init_centres: tuple[float, ...] = DEFAULT_CENTRES,
        init_width: float = 3.0,
    ):
        super().__init__()

        if len(init_centres) != n_basis:
            raise ValueError(
                f"Got {len(init_centres)} centres for {n_basis} bases."
            )

        self.lag_reach = lag_reach
        self.n_nodes = n_nodes
        self.n_features = n_features

        self.centre_raw = nn.Parameter(torch.tensor(init_centres, dtype=torch.float32))
        self.width_raw = nn.Parameter(
            torch.full((n_basis,), _inverse_softplus(init_width))
        )

        # District identity. This is what makes the delay district-specific:
        # everything else in the module is shared across the country.
        self.node_embedding = nn.Parameter(torch.randn(n_nodes, embedding_dim) * 0.1)

        # One projection per climate feature, so rainfall and temperature can
        # settle on different mixtures for the same district.
        self.feature_projection = nn.Parameter(
            torch.randn(n_features, embedding_dim, n_basis) * 0.1
        )

    def basis(self) -> torch.Tensor:
        """Return the B smooth bumps over lag, each summing to one. [B, L]"""

        tau = torch.arange(
            self.lag_reach, device=self.centre_raw.device, dtype=torch.float32
        )

        centre = self.centre_raw.clamp(0.0, self.lag_reach - 1.0)
        # The floor keeps a bump from shrinking onto a single lag, which is the
        # degenerate case the smooth parameterisation exists to avoid.
        width = F.softplus(self.width_raw) + 0.5

        bumps = torch.exp(
            -0.5 * ((tau[None, :] - centre[:, None]) / width[:, None]) ** 2
        )

        return bumps / bumps.sum(dim=-1, keepdim=True)

    def mixture_weights(self) -> torch.Tensor:
        """Return how much of each bump each district-feature pair uses. [N, K, B]"""

        logits = torch.einsum(
            "nd,kdb->nkb", self.node_embedding, self.feature_projection
        )

        return torch.softmax(logits, dim=-1)

    def kernels(self) -> torch.Tensor:
        """Return the delay curves. [nodes, features, lag_reach], rows sum to one."""

        return torch.einsum("nkb,bl->nkl", self.mixture_weights(), self.basis())

    def peak_lags(self) -> torch.Tensor:
        """Return the centre of mass of each curve, in periods. [nodes, features]

        The centre of mass rather than the argmax: it is what the curve
        actually weights, it moves smoothly, and it is comparable against a
        cross-correlation peak.
        """

        tau = torch.arange(
            self.lag_reach, device=self.centre_raw.device, dtype=torch.float32
        )

        return (self.kernels() * tau).sum(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the learned delay to every district and feature."""

        steps = x.shape[1]
        if steps < self.lag_reach:
            raise ValueError(
                f"Need at least {self.lag_reach} steps, got {steps}."
            )

        # [batch, out_steps, nodes, features, lag_reach], oldest period first.
        windows = x.unfold(dimension=1, size=self.lag_reach, step=1)

        # The kernel is indexed by tau (0 = most recent), the window by
        # position (0 = oldest), so one of them has to be reversed. Getting
        # this backwards is silent and would invert the learned delay.
        kernel = self.kernels().flip(-1)

        return torch.einsum("btnkl,nkl->btnk", windows, kernel)


class LagGCNGRU(nn.Module):
    """The baseline GCN+GRU with a learnable lag encoder bolted on the front.

    The backbone is untouched -- same graph convolutions, same shared GRU, same
    linear head, same number of input features. Only the climate channels are
    rewritten on the way in, so a difference in the results is a difference in
    the delay handling and nothing else.

    Non-climate channels (case history, seasonality, observation flags,
    centroids) are passed through unsmoothed. Smoothing the case history would
    blur exactly the signal the anchored target leans on, and smoothing a
    constant like latitude is meaningless.
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_nodes: int,
        lagged_indices: list[int],
        lag_reach: int = DEFAULT_LAG_REACH,
        **encoder_kwargs,
    ):
        super().__init__()

        self.backbone = backbone
        self.lag_reach = lag_reach
        self.register_buffer(
            "lagged_indices", torch.tensor(lagged_indices, dtype=torch.long)
        )

        self.encoder = LearnableLagEncoder(
            n_nodes=n_nodes,
            n_features=len(lagged_indices),
            lag_reach=lag_reach,
            **encoder_kwargs,
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        smoothed = self.encoder(x[..., self.lagged_indices])

        # Trim the pass-through channels to the same periods the encoder could
        # produce, so the backbone sees one aligned window.
        out = x[:, self.lag_reach - 1 :].clone()
        out[..., self.lagged_indices] = smoothed

        return self.backbone(out, adjacency)

    def kernel_parameters(self):
        """The encoder's parameters, which want a higher learning rate."""

        return self.encoder.parameters()

    def backbone_parameters(self):
        """Everything else."""

        return self.backbone.parameters()

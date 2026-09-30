"""
Loss and forecast conversion for the climate-horizon extension.

The two-branch model and the gradient router come in a later step (plan §6.1).
This module holds what the data interface and the evaluator already need:

- `nb_loss_by_horizon` splits `masked_negative_binomial_loss` into
  per-horizon terms L_h, with sum_h L_h equal to the pooled loss. The router
  weights the L_h terms separately.
- `anchored_mean` is the NB mean link, log mu = log1p(y_origin) + delta.
- `nb_point_forecast` gives the NB median, the extension's point forecast.
"""

from __future__ import annotations

import numpy as np
import torch

from src.evaluation.long_horizon import nb_quantiles
from src.models.negative_binomial import negative_binomial_nll


def nb_loss_by_horizon(
    mu: torch.Tensor, alpha: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Per-horizon masked NB2 NLL, [H], each normalised by the pooled observed count.

    NaN-safe: masked cells are replaced before the NLL is evaluated, and their
    terms are removed with `torch.where` rather than by multiplying by zero. So
    a NaN or garbage target at a masked cell cannot reach the loss or its
    gradient.
    """

    observed = mask > 0
    safe_target = torch.where(observed, torch.nan_to_num(target, nan=0.0), torch.zeros_like(target))
    nll = negative_binomial_nll(mu, alpha, safe_target)
    nll = torch.where(observed, nll, torch.zeros_like(nll))
    denominator = observed.sum().clamp(min=1).to(nll.dtype)
    dims = tuple(range(nll.ndim - 1))
    return nll.sum(dim=dims) / denominator


def anchored_mean(anchor: torch.Tensor, delta: torch.Tensor, clamp: float = 10.0) -> torch.Tensor:
    """mu = exp(log1p(y_origin) + delta), with NegBinHead's clamp on delta."""

    if anchor.ndim == delta.ndim - 1:
        anchor = anchor.unsqueeze(-1)
    return torch.exp(anchor + delta.clamp(-clamp, clamp))


def nb_point_forecast(mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """NB median on the count scale (the benchmark's point forecast for NB arms)."""

    return nb_quantiles(mu, alpha, levels=(0.5,))[..., 0]


# ---------------------------------------------------------------------------
# Two-branch model (plan §6.1)
# ---------------------------------------------------------------------------

import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from src.models.horizon_lag_encoder import HorizonLagEncoder  # noqa: E402
from src.models.lag_encoder import DEFAULT_LAG_REACH  # noqa: E402
from src.models.stgnn import GCNGRU  # noqa: E402


# Gradient-routing version recorded with every run. v1: climate head in the
# weighted group (as run until 2026-09-30 18:44). v2: heads unweighted.
ROUTING_VERSION = "v2-heads-unweighted"


class _GradScale(torch.autograd.Function):
    """Identity forward; the gradient of column h is multiplied by weights[h]."""

    @staticmethod
    def forward(ctx, x, weights):
        ctx.save_for_backward(weights)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        (weights,) = ctx.saved_tensors
        return grad * weights, None


class ClimateHorizonNB(nn.Module):
    """Shared case-history GRU + small climate branch, NB2 output for H horizons.

    For horizon h, in NB log-mean space:

        delta_total_h = delta_case_h + sigmoid(gate_logit_h) * delta_climate_h
        log mu_h      = log1p(y_origin) + clamp(delta_total_h, -10, 10)
        alpha_h       = softplus(case head) + 1e-4       (dispersion is case-based)

    Case branch: `GCNGRU` run with an identity adjacency, so no spatial message
    passing. Climate branch: `LearnableLagEncoder` (per-district, per-feature
    delay kernels) -> GRU -> one correction per horizon. The gate reads the
    case state, a district embedding and a per-horizon bias. Nothing forces it
    to increase with the horizon.

    `climate_weights` [H] scales, in the backward pass only, the gradient each
    horizon sends into the climate branch (`delta_climate_h` is the only path
    from horizon h to it). With [1, 1, 1, 1] it is the identity, which is the
    baseline `hcd_uniform`.

    All corrections are zero at initialisation, so mu = y_origin + 1, exactly
    as `NegBinHead`.
    """

    def __init__(
        self,
        n_nodes: int,
        n_case: int,
        n_climate: int,
        n_horizons: int = 4,
        hidden: int = 32,
        gcn_layers: int = 2,
        climate_hidden: int = 16,
        lag_reach: int = DEFAULT_LAG_REACH,
        embedding_dim: int = 8,
        dropout: float = 0.2,
        lag_mode: str = "target",
        common_support: bool = False,
        max_delay: int | None = None,
    ):
        super().__init__()
        self.n_horizons = n_horizons
        self.lag_reach = lag_reach

        # Case branch: the baseline trunk (per-node layers, shared GRU).
        self.case_trunk = GCNGRU(n_case, hidden=hidden, gcn_layers=gcn_layers,
                                 horizon=1, dropout=dropout)
        self.case_trunk.head = nn.Identity()
        self.register_buffer("identity", torch.eye(n_nodes))
        self.case_delta = nn.Linear(hidden, n_horizons)
        self.case_alpha = nn.Linear(hidden, n_horizons)

        # Gate: case state + district + horizon.
        self.gate_state = nn.Linear(hidden, n_horizons)
        self.gate_district = nn.Parameter(torch.zeros(n_nodes, n_horizons))
        self.gate_horizon = nn.Parameter(torch.zeros(n_horizons))

        # Climate branch.
        # `lag_reach` is the history length each kernel may read.
        self.encoder = HorizonLagEncoder(n_nodes, n_climate, horizons=tuple(range(1, n_horizons + 1)),
                                         history=lag_reach, max_delay=max_delay, mode=lag_mode,
                                         common_support=common_support, embedding_dim=embedding_dim)
        self.climate_gru = nn.GRU(n_climate, climate_hidden, batch_first=True)
        self.climate_delta = nn.Linear(climate_hidden, n_horizons)
        self.dropout = nn.Dropout(dropout)

        for layer in (self.case_delta, self.case_alpha, self.climate_delta, self.gate_state):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.register_buffer("climate_weights", torch.ones(n_horizons))

    # -- parameter groups ---------------------------------------------------

    def climate_parameters(self) -> list[nn.Parameter]:
        """Group 1 (weighted loss): the climate feature encoder -- lag kernels and climate GRU.

        Routing v2 (correction 2026-09-30, docs/climate_horizon_correction_audit.md):
        the climate correction head `climate_delta` is a prediction head and is
        no longer in this group. No parameter is shared between the groups; the
        encoder still receives its weighted gradient *through* the head.
        """

        return [*self.encoder.parameters(), *self.climate_gru.parameters()]

    def other_parameters(self) -> list[nn.Parameter]:
        """Group 2 (unweighted loss): case branch, gate, every prediction head, dispersion."""

        climate = {id(p) for p in self.climate_parameters()}
        return [p for p in self.parameters() if id(p) not in climate]

    def set_climate_weights(self, weights) -> None:
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.climate_weights.device)
        if weights.shape != self.climate_weights.shape or (weights < 0).any():
            raise ValueError("climate weights must be non-negative, one per horizon")
        self.climate_weights.copy_(weights)

    # -- forward --------------------------------------------------------------

    def forward(self, x_case: torch.Tensor, x_climate: torch.Tensor, anchor: torch.Tensor,
                gate_off: bool = False) -> dict[str, torch.Tensor]:
        """x_case [B, L, N, Fc]; x_climate [B, L + reach - 1, N, K]; anchor [B, N]."""

        batch, steps, nodes, _ = x_case.shape
        state = self.case_trunk(x_case, self.identity).view(batch, nodes, -1)   # dropout applied

        delayed = self.encoder(x_climate)                                  # [B, L, N, K, H]
        if delayed.shape[1] != steps:
            raise ValueError(f"climate window gives {delayed.shape[1]} steps, case window {steps}")
        horizons = delayed.shape[-1]
        # One climate stream per horizon through the same GRU and head, so
        # both lag modes have identical parameters.
        sequences = delayed.permute(0, 2, 4, 1, 3).reshape(batch * nodes * horizons, steps, -1)
        climate_state, _ = self.climate_gru(sequences)
        climate_state = self.dropout(climate_state[:, -1]).view(batch, nodes, horizons, -1)
        per_stream = self.climate_delta(climate_state)                     # [B, N, H(stream), H(head)]
        delta_climate = per_stream.diagonal(dim1=-2, dim2=-1)              # stream h -> head h
        routed = _GradScale.apply(delta_climate, self.climate_weights)

        gate_logit = self.gate_state(state) + self.gate_district + self.gate_horizon
        gate = torch.zeros_like(gate_logit) if gate_off else torch.sigmoid(gate_logit)

        delta_case = self.case_delta(state)
        gated_climate = gate * routed
        delta_total = delta_case + gated_climate
        mu = anchored_mean(anchor, delta_total)
        alpha = F.softplus(self.case_alpha(state)) + 1e-4
        return {"mu": mu, "alpha": alpha, "gate": gate, "delta_case": delta_case,
                "delta_climate": delta_climate, "gated_climate": gated_climate,
                "delta_total": delta_total}

    def available_mass(self) -> torch.Tensor:
        """Kernel mass on each horizon's usable delay support. [H, N, K]"""

        return self.encoder.weights()[1]


class CaseOnlyNB(nn.Module):
    """Size-matched case-only pilot: the case branch and NB heads, no climate path.

    Same outputs as `ClimateHorizonNB` (the climate terms and gate are zero).
    `hidden` is chosen by `case_only_hidden` so the parameter count is as
    close as possible to the two-branch model's.
    """

    def __init__(self, n_nodes: int, n_case: int, n_horizons: int = 4, hidden: int = 32,
                 gcn_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.case_trunk = GCNGRU(n_case, hidden=hidden, gcn_layers=gcn_layers, horizon=1, dropout=dropout)
        self.case_trunk.head = nn.Identity()
        self.register_buffer("identity", torch.eye(n_nodes))
        self.case_delta = nn.Linear(hidden, n_horizons)
        self.case_alpha = nn.Linear(hidden, n_horizons)
        for layer in (self.case_delta, self.case_alpha):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def climate_parameters(self) -> list[nn.Parameter]:
        return []

    def other_parameters(self) -> list[nn.Parameter]:
        return list(self.parameters())

    def set_climate_weights(self, weights) -> None:
        """No climate path, so the weights have nothing to act on."""

    def forward(self, x_case, x_climate, anchor, gate_off: bool = False):
        batch, _, nodes, _ = x_case.shape
        state = self.case_trunk(x_case, self.identity).view(batch, nodes, -1)
        delta_case = self.case_delta(state)
        zeros = torch.zeros_like(delta_case)
        return {"mu": anchored_mean(anchor, delta_case),
                "alpha": F.softplus(self.case_alpha(state)) + 1e-4,
                "gate": zeros, "delta_case": delta_case, "delta_climate": zeros,
                "gated_climate": zeros, "delta_total": delta_case}


def case_only_hidden(target_parameters: int, n_nodes: int, n_case: int, n_horizons: int = 4,
                     gcn_layers: int = 2) -> int:
    """Hidden size whose CaseOnlyNB parameter count is closest to `target_parameters`."""

    def count(hidden):
        return sum(p.numel() for p in CaseOnlyNB(n_nodes, n_case, n_horizons, hidden, gcn_layers).parameters())

    return min(range(8, 129), key=lambda h: abs(count(h) - target_parameters))


def nb_loss_terms(mu: torch.Tensor, alpha: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-horizon NB2 NLL numerator N_h (sum over observed cells) and denominator D_h (count). [H], [H]

    The verified pooled reduction is sum_h N_h / sum_h D_h
    (= `masked_negative_binomial_loss` = `nb_loss_by_horizon(...).sum()`).
    Masked cells are removed with `torch.where`, as in `nb_loss_by_horizon`.
    """

    observed = mask > 0
    safe_target = torch.where(observed, torch.nan_to_num(target, nan=0.0), torch.zeros_like(target))
    nll = negative_binomial_nll(mu, alpha, safe_target)
    nll = torch.where(observed, nll, torch.zeros_like(nll))
    dims = tuple(range(nll.ndim - 1))
    return nll.sum(dim=dims), observed.sum(dim=dims).to(nll.dtype)

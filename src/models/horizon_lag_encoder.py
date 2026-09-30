"""
Horizon-aware climate-lag kernels: origin-relative and target-relative modes.

Notation. Origin t, horizon h, climate observed at t - k (k = 0 is the origin
week). The delay between that observation and the target week t + h is

    d = h + k.

origin-relative (ablation)
    One kernel over the origin lag k = 0 .. history - 1, the same for every
    horizon. This is the existing `LearnableLagEncoder` behaviour. In delay
    terms a kernel peak at k covers d = h + k, so the implied delay drifts
    with the horizon.

target-relative (main model, both equal-weight and weighted arms)
    One kernel over the delay d = 0 .. max_delay, shared by every horizon.
    For horizon h it selects x[t + h - d] = x[t - k] with k = d - h. Only
    delays with h <= d (the observation is available by t) and
    d - h <= history - 1 (inside the input window) are usable. The kernel is
    masked to that support and renormalised over it. The mass it had there
    before renormalising, the "available mass", is reported. If the support
    is empty the weights are all zero, the output is zero and the mass is 0,
    with no division by zero.

Explicit sizes
    history    number of observed weeks each kernel may read (k = 0 .. history-1)
    max_delay  largest target-relative delay; default history - 1 + max(h),
               so every horizon can reach its full history window.

Both modes reuse `LearnableLagEncoder`'s smooth parameterisation unchanged:
6 Gaussian bumps with learnable centres and widths, and a simplex mixture
from a district embedding times a per-feature projection. They have
**identical parameters and parameter counts**; only the grid the bumps live
on differs (k in origin mode, d in target mode).

Available lag support
    - Observations: both modes read exactly x[t - k] for k = 0 .. history - 1,
      whatever the horizon.
    - Delays: origin mode covers d in [h, h + history - 1], which moves with h.
      Target mode covers the same observations, but its kernel mass on
      d < h is unusable for horizon h, so the available mass is at most 1.
    - Sensitivity check `common_support=True`: in target mode, restrict every
      horizon to d in [max(h), min(h) + history - 1], the delays usable by all
      so every horizon's kernel lives on the same support.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.lag_encoder import DEFAULT_CENTRES, LearnableLagEncoder

MODES = ("target", "origin")


def delay_weights(kernel: torch.Tensor, horizons, history: int, mode: str,
                  common_support: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn a kernel over its grid into per-horizon weights over origin lags.

    kernel [N, K, G] with non-negative rows. G = history in origin mode and
    max_delay + 1 in target mode.

    Returns weights [H, N, K, history], indexed by origin lag k (0 = the
    origin week) and summing to 1 over valid support or 0 when it is empty,
    and mass [H, N, K], the kernel mass on that support before renormalising.
    """

    horizons = [int(h) for h in horizons]
    n, f, grid = kernel.shape
    k = torch.arange(history, device=kernel.device)
    weights, masses = [], []
    for h in horizons:
        if mode == "origin":
            if grid != history:
                raise ValueError("origin mode needs a kernel over exactly `history` lags")
            raw = kernel
            valid = torch.ones(history, dtype=torch.bool, device=kernel.device)
        elif mode == "target":
            d = h + k                                  # delay of each usable observation
            if common_support:  # delays usable by every horizon: [max h, min h + history - 1]
                low, high = max(horizons), min(grid - 1, min(horizons) + history - 1)
            else:
                low, high = h, grid - 1
            valid = (d >= low) & (d <= high)
            raw = kernel[..., d.clamp(max=grid - 1)] * valid
        else:
            raise ValueError(f"unknown lag mode {mode!r}")
        mass = raw.sum(-1)
        safe = torch.where(mass > 0, mass, torch.ones_like(mass))
        weights.append(torch.where(mass[..., None] > 0, raw / safe[..., None], torch.zeros_like(raw)))
        masses.append(mass)
    return torch.stack(weights), torch.stack(masses)


class HorizonLagEncoder(nn.Module):
    """Per-district, per-feature delay kernels applied separately for each horizon.

    Input  [B, T, N, K] with T >= history. Output [B, T - history + 1, N, K, H].
    """

    def __init__(self, n_nodes: int, n_features: int, horizons=(1, 2, 3, 4), history: int = 26,
                 max_delay: int | None = None, mode: str = "target", common_support: bool = False,
                 n_basis: int = 6, embedding_dim: int = 8, init_centres=DEFAULT_CENTRES,
                 init_width: float = 3.0, activation: str = "softmax"):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if common_support and mode != "target":
            raise ValueError("common_support applies to target mode only")
        self.horizons = tuple(int(h) for h in horizons)
        self.history = history
        self.max_delay = history - 1 + max(self.horizons) if max_delay is None else max_delay
        if self.max_delay < min(self.horizons):
            raise ValueError("max_delay leaves no usable delay for any horizon")
        self.mode = mode
        self.common_support = common_support
        grid = history if mode == "origin" else self.max_delay + 1
        # The existing smooth mixture, reused as the kernel generator over its grid.
        self.mixture = LearnableLagEncoder(n_nodes, n_features, lag_reach=grid, n_basis=n_basis,
                                           embedding_dim=embedding_dim, init_centres=init_centres,
                                           init_width=init_width, activation=activation)

    def kernels(self) -> torch.Tensor:
        """Kernel over the mode's own grid (k or d). [N, K, G]"""

        return self.mixture.kernels()

    def weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-horizon weights over origin lags [H, N, K, history] and available mass [H, N, K]."""

        return delay_weights(self.kernels(), self.horizons, self.history, self.mode, self.common_support)

    def peak_delays(self) -> torch.Tensor:
        """Centre of mass of each horizon's usable kernel, as a target-relative delay d. [H, N, K]"""

        weights, _ = self.weights()
        k = torch.arange(self.history, device=weights.device, dtype=weights.dtype)
        h = torch.tensor(self.horizons, device=weights.device, dtype=weights.dtype)
        return (weights * k).sum(-1) + h[:, None, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.history:
            raise ValueError(f"Need at least {self.history} steps, got {x.shape[1]}.")
        windows = x.unfold(dimension=1, size=self.history, step=1)   # [B, S, N, K, history], oldest first
        weights, _ = self.weights()
        return torch.einsum("bsnkl,hnkl->bsnkh", windows, weights.flip(-1))

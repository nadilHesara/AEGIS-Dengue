"""
Simplex activations for the lag encoder's basis mixture.

The lag encoder builds a delay curve as a mixture of B Gaussian bumps. The
mixture weights have to lie on the simplex -- non-negative, summing to one --
so that a kernel is a weighted average of bumps and stays readable as "district
i responds to rainfall about 11 weeks later" rather than as an arbitrary filter.

Softmax is the default way onto that simplex, and it is a poor fit here for
three reasons that this repository's own results already show symptoms of:

1.  It saturates. The Jacobian is diag(p) - p p^T, so once one coordinate
    dominates every gradient goes to zero. scripts/16's `parameter_groups`
    documents the symptom -- "the gradient reaching them is far smaller than
    the gradient reaching the GRU" -- and patches it with a 10x learning rate
    rather than addressing the activation.

2.  Its scale is unconstrained. The logits are a product of two 0.1*randn
    tensors, so they start near zero, the mixture starts near uniform, and the
    effective temperature drifts during training with nothing holding it.

3.  It has full support. Softmax never returns an exact zero, so every learned
    curve is a blend of all six bumps. A biological delay is one peak, not six
    overlapping ones; blending them widens the curve and pulls its centre of
    mass towards the middle of the reach. docs/learnable_lags_results.md
    reports learned and measured delays correlating at r = -0.16, and a curve
    that structurally cannot concentrate is one mechanism for that.

The alternatives here all map R^B onto the same simplex, so each is drop-in and
the rest of the encoder is unchanged:

    softmax         the control
    temp_softmax    softmax(z / tau), tau learned -- addresses (1) and (2)
    sparsemax       Euclidean projection onto the simplex -- addresses (3)
    entmax15        alpha-entmax at alpha = 1.5, between the two
    gumbel_softmax  annealed stochastic sampling -- addresses boundary drift

`sparsemax` and `entmax15` return exact zeros, which is the point: a district
can put all of its mass on one bump and none on the rest, and the kernel that
results is a single readable peak. Their gradients do not vanish as the
distribution sharpens, because on the support they behave like a projection
rather than like a saturating squash.

References:
    Martins & Astudillo (2016), "From Softmax to Sparsemax".
    Peters, Niculae & Martins (2019), "Sparse Sequence-to-Sequence Models".

The implementations are written directly against torch autograd rather than
taken from the `entmax` package, to avoid a dependency for two functions and to
keep the numerics explicit and testable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Euclidean projection of `logits` onto the probability simplex.

    Returns argmin_p ||p - z||^2 subject to p >= 0 and sum p = 1. Unlike
    softmax this is exactly zero off the support, and it is piecewise linear,
    so its gradient on the support stays constant instead of shrinking as the
    distribution sharpens.

    The threshold is found in closed form by sorting, which is cheap here: the
    sorted dimension is the basis count, B = 6.
    """

    sorted_logits, _ = torch.sort(logits, dim=dim, descending=True)
    cumulative = sorted_logits.cumsum(dim) - 1.0

    rank = torch.arange(
        1, logits.shape[dim] + 1, device=logits.device, dtype=logits.dtype
    )
    shape = [1] * logits.dim()
    shape[dim] = -1
    rank = rank.view(shape)

    # The support is every coordinate still above the running threshold.
    support = (rank * sorted_logits > cumulative).to(logits.dtype)
    size = support.sum(dim=dim, keepdim=True)
    threshold = cumulative.gather(dim, (size.long() - 1).clamp(min=0)) / size

    return torch.clamp(logits - threshold, min=0.0)


def entmax15(logits: torch.Tensor, dim: int = -1, n_iter: int = 30) -> torch.Tensor:
    """alpha-entmax at alpha = 1.5, by bisection on the threshold.

    Sits between softmax (alpha = 1, dense) and sparsemax (alpha = 2, most
    sparse): it produces exact zeros, but concentrates less abruptly than
    sparsemax, so a mixture can still straddle two neighbouring bumps when the
    true delay falls between their centres. That matters here, because
    DEFAULT_CENTRES is a coarse grid and interpolating between centres is how
    the encoder reaches a delay the grid does not name.

    p = [(z - tau)_+]^2 after scaling the logits by 1/(alpha - 1) = 2. tau is
    the unique value making the sum one, found by bisection; 30 iterations is
    well past float32 resolution for B = 6.
    """

    logits = logits / 2.0
    logits = logits - logits.max(dim=dim, keepdim=True).values

    lower = logits.max(dim=dim, keepdim=True).values - 1.0
    upper = logits.max(dim=dim, keepdim=True).values

    for _ in range(n_iter):
        middle = (lower + upper) / 2.0
        total = torch.clamp(logits - middle, min=0.0).pow(2).sum(dim=dim, keepdim=True)
        # A sum above one means the threshold is still too low.
        too_low = (total > 1.0).to(logits.dtype)
        lower = too_low * middle + (1.0 - too_low) * lower
        upper = too_low * upper + (1.0 - too_low) * middle

    probabilities = torch.clamp(logits - (lower + upper) / 2.0, min=0.0).pow(2)

    # Bisection leaves a residual well under float32 noise, but nothing
    # downstream renormalises and a kernel that does not sum to one stops being
    # a weighted average, so normalise explicitly.
    return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp(min=1e-12)


def floored_entmax15(
    logits: torch.Tensor, dim: int = -1, floor: float = 0.02
) -> torch.Tensor:
    """entmax15 mixed with a little uniform mass, so no bump is ever dead.

    Sparsity buys a readable kernel and a gradient that does not vanish as the
    mixture sharpens, but it buys them with an absorbing state. When the
    support collapses to a single bump the output is one-hot, locally constant,
    and its gradient with respect to the logits is exactly zero -- measured, in
    tests/test_simplex_activations.py, not assumed. Nothing can then move it,
    whatever the loss does, and this encoder trains at a 10x learning rate on
    parameters that start near zero, so reaching that state is not remote.

    Mixing in `floor` of uniform mass keeps every coordinate strictly positive
    and every partial derivative alive, while leaving the sparsity pattern
    intact -- the off-support bumps sit at floor/B, which is 0.3% of the mass
    at the default, small enough that the kernel still reads as one peak.

    This is the arm to prefer if the sparse arms train well but unstably.
    """

    if not 0.0 <= floor < 1.0:
        raise ValueError(f"floor must be in [0, 1), got {floor}.")

    sparse = entmax15(logits, dim=dim)
    uniform = torch.full_like(sparse, 1.0 / sparse.shape[dim])

    return (1.0 - floor) * sparse + floor * uniform


class SimplexActivation(nn.Module):
    """Map logits onto the simplex by one of the named schemes.

    `temp_softmax` carries a learnable temperature; every other scheme is
    parameter-free, so this module adds either one parameter or none and the
    encoder's parameter count is unchanged to within that one.

    The temperature is stored as a raw parameter behind a softplus and a floor,
    for the same reason the encoder's bump widths are: an unconstrained
    temperature can cross zero, and at tau <= 0 softmax inverts.
    """

    NAMES = (
        "softmax",
        "temp_softmax",
        "sparsemax",
        "entmax15",
        "floored_entmax15",
        "gumbel_softmax",
    )

    def __init__(
        self,
        name: str = "softmax",
        init_temperature: float = 1.0,
        gumbel_temperature: float = 1.0,
        floor: float = 0.02,
    ):
        super().__init__()

        if name not in self.NAMES:
            raise ValueError(
                f"Unknown activation {name!r}; expected one of {self.NAMES}."
            )

        self.name = name
        self.gumbel_temperature = gumbel_temperature
        self.floor = floor

        if name == "temp_softmax":
            raw = float(torch.log(torch.expm1(torch.tensor(init_temperature))))
            self.temperature_raw = nn.Parameter(torch.tensor(raw))
        else:
            self.register_parameter("temperature_raw", None)

    def temperature(self) -> torch.Tensor:
        """The positive temperature, floored away from zero."""

        if self.temperature_raw is None:
            raise AttributeError(f"{self.name} has no temperature.")

        return F.softplus(self.temperature_raw) + 0.1

    def forward(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.name == "softmax":
            return torch.softmax(logits, dim=dim)

        if self.name == "temp_softmax":
            return torch.softmax(logits / self.temperature(), dim=dim)

        if self.name == "sparsemax":
            return sparsemax(logits, dim=dim)

        if self.name == "entmax15":
            return entmax15(logits, dim=dim)

        if self.name == "floored_entmax15":
            return floored_entmax15(logits, dim=dim, floor=self.floor)

        if self.name == "gumbel_softmax":
            # Stochastic only while training. At evaluation the sampling noise
            # would make the same district score differently on two passes,
            # which would make the kernel report meaningless.
            if not self.training:
                return torch.softmax(logits / self.gumbel_temperature, dim=dim)

            return F.gumbel_softmax(
                logits, tau=self.gumbel_temperature, hard=False, dim=dim
            )

        raise AssertionError(f"unreachable: {self.name}")

    def extra_repr(self) -> str:
        return f"name={self.name}"

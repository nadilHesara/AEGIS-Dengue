"""
Layer normalisation between the GRU representation and the prediction head.

The GCN+GRU trunk hands its final hidden state straight to a linear head. That
state is the output of a recurrent cell run over 12 reporting periods, and its
scale is not controlled by anything: the GRU's gates can drift the activation
magnitude across the sequence, and the graph convolution feeding it applies an
unnormalised linear map on top of features whose per-fold standardisation was
fitted on the training years only. Layer normalisation at that point fixes the
per-sample mean and variance of the representation the head reads, which is the
standard treatment for recurrent sequence models (Ba et al., 2016) and is where
it is expected to matter most.

The placement is `GRU -> LayerNorm -> dropout -> head`, not the other order.
Normalising before the dropout means the statistics are computed on the clean
representation; putting the norm after would have it renormalise a vector whose
moments dropout has already distorted at training time but not at evaluation
time, widening the train/eval gap that dropout already creates. Before-dropout
is also the ordering used in the recurrent and transformer blocks this borrows
from.

**Why this might do nothing here, stated up front.** Every negative result in
this project so far -- the learnable lags finding no gradient, the flat
activation sweep, the flat hyperparameter surface, no graph beating no graph --
has the same documented cause: at one week ahead the previous period's case
count carries nearly all the signal, and the model is a shallow one on ~500
training windows. Layer normalisation addresses optimisation conditioning, and
there is no standing evidence that conditioning is what limits this model. The
honest prior is that it changes little on the headline. It is worth measuring
anyway because it is cheap, because the trunk is deeper at h=4 where the
recurrence has to carry more, and because "we normalised and it did not move" is
itself a fact about where the ceiling comes from.

`elementwise_affine` is left at torch's default (on), so the module can learn to
undo the normalisation through its gain and bias if that is what the loss wants.
Turning it off would be imposing unit scale as a constraint rather than offering
normalisation as a reparameterisation; the affine version is what "add
LayerNorm" conventionally means, and it is strictly the more general of the two.

The module owns only the placement. The backbone is the unmodified `GCNGRU` from
`scripts/16.train_gcn_gru.py` -- same graph convolutions, same shared GRU, same
head, same anchored residual target -- so a normalised run differs from a
baseline run in the norm and nothing else.
"""

from __future__ import annotations

import torch
from torch import nn


class LayerNormGCNGRU(nn.Module):
    """A `GCNGRU` with layer normalisation before the prediction head.

    Reimplements the baseline forward pass rather than wrapping it, because the
    insertion point is *inside* that pass -- between the GRU's last hidden state
    and the head -- and there is no hook there. To keep the two arms honestly
    comparable the submodules are taken from the supplied backbone by reference
    rather than rebuilt: the graph layers, dropout, GRU and head are the very
    objects the baseline constructor made, with the same shapes, the same
    initialisation and the same seed-dependence. The only additions are the
    `nn.LayerNorm` and, when `normalise=False`, nothing at all.

    `normalise=False` must reproduce the baseline **exactly**, not merely
    closely: it is the control arm of the comparison. The test suite asserts
    bit-identical outputs against `GCNGRU` for that setting.

    Parameters
    ----------
    backbone:
        A constructed `GCNGRU`. Its submodules are adopted, not copied.
    normalise:
        Whether to apply the norm. `False` makes this the untouched baseline and
        adds no parameters, which is what lets one code path serve both arms.
    eps:
        `nn.LayerNorm` epsilon. Exposed because the hidden width here is 32,
        small enough that the variance estimate is noisy.
    """

    def __init__(
        self,
        backbone: nn.Module,
        normalise: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.normalise = normalise

        # Adopted by reference: same objects, same initialised weights.
        self.graph_layers = backbone.graph_layers
        self.dropout = backbone.dropout
        self.gru = backbone.gru
        self.head = backbone.head

        hidden = self.gru.hidden_size

        # Registered only when used, so the control arm's parameter count and
        # its state_dict keys match the baseline's exactly.
        self.norm = nn.LayerNorm(hidden, eps=eps) if normalise else None

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, _ = x.shape

        spatial = x
        for layer in self.graph_layers:
            spatial = torch.relu(layer(spatial, adjacency))
            spatial = self.dropout(spatial)

        # [batch, steps, nodes, hidden] -> one sequence per (batch, node)
        sequences = spatial.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)

        output, _ = self.gru(sequences)
        last = output[:, -1]

        # The norm sits here: on the clean final hidden state, before dropout
        # perturbs it and before the head reads it. Statistics are over the
        # hidden axis, per (window, district) row -- so each district's
        # representation is normalised on its own terms, and a district with a
        # large outbreak is not rescaled by what its neighbours are doing.
        if self.norm is not None:
            last = self.norm(last)

        last = self.dropout(last)

        return self.head(last).view(batch, nodes, -1)

    def extra_repr(self) -> str:
        return f"normalise={self.normalise}"


def count_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def norm_statistics(module: nn.Module) -> dict[str, float]:
    """Summarise a trained `LayerNorm`'s learned gain and bias.

    The affine parameters say what the norm actually did. A gain that stayed
    near its initialisation of 1 with a bias near 0 means the module is doing
    plain normalisation; a gain that collapsed toward 0 means the head learned
    to ignore the representation; a gain that grew large means the model spent
    capacity undoing the rescaling, which is evidence the normalisation was not
    wanted. Distinguishing those is the difference between "normalising did not
    help" and "the model turned the normalisation off".
    """

    norm = getattr(module, "norm", None)

    if norm is None or norm.weight is None:
        return {}

    with torch.no_grad():
        gain = norm.weight.detach()
        bias = norm.bias.detach()

        return {
            "gain_mean": float(gain.mean()),
            "gain_sd": float(gain.std()),
            "gain_min": float(gain.min()),
            "gain_max": float(gain.max()),
            "bias_mean": float(bias.mean()),
            "bias_abs_mean": float(bias.abs().mean()),
        }

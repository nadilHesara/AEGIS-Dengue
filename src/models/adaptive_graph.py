"""
A learnable adjacency for the GCN+GRU, from node embeddings.

The committed baseline sweep found the queen-contiguity graph is not neutral but
harmful: `gru_only` -- the identical model with the identity in place of the
adjacency -- beats `gcn_gru` on all nine folds. Two readings of that are
possible, and they call for different responses:

    the graph is wrong        contiguity is a poor prior for a country 430 km
                              long where Colombo and Galle are 100 km apart,
                              share no border, and are both wet-zone coastal.
                              Then a *better* graph should help.

    spatial structure is not  whatever cross-district signal exists at one week
    there to be had           ahead is already carried by each district's own
                              case history. Then no graph helps, and the search
                              for one is misdirected.

A fixed alternative graph (`A_gaussian`) tests the first reading against one
specific competing prior. This module tests it without committing to any prior
at all: let the model learn the adjacency from the forecasting loss itself. If a
graph learned end-to-end also fails to beat the identity, the second reading is
the surviving one, and that is a much stronger statement than any single fixed
graph can support.

The parameterisation is the Graph WaveNet adaptive adjacency (Wu et al., 2019):

    A = softmax(relu(E1 @ E2.T))

with two learned embedding tables `E1, E2` of shape [nodes, k]. Three properties
matter here:

    asymmetric      `E1 @ E2.T` is not symmetric, so the module can learn
                    directed influence -- district i affecting j more than j
                    affects i. Both fixed graphs in this project are symmetric
                    by construction, and disease spread need not be.

    row-normalised  the softmax is over the source axis, so every row sums to 1
                    and the convolution is an average over neighbours rather
                    than a sum. This replaces the `D^-1/2 (A+I) D^-1/2`
                    normalisation the fixed graphs get from `scripts/13`; it is
                    a different normalisation, but the property that matters --
                    that the convolution cannot blow up the feature scale -- is
                    the same, and it is the one the published method uses.

    small           2 * nodes * k parameters. At k=8 and 25 nodes that is 400,
                    against roughly 8,200 for the rest of the model, on ~500
                    training windows. A free 25x25 matrix (625 parameters) was
                    the obvious alternative and was rejected on that count:
                    it would be more parameters than the backbone, learned from
                    fewer samples than it has entries.

`relu` before the softmax is the published form and it matters: it zeroes the
negative similarities, so a pair of districts with no learned affinity
contributes nothing before normalisation rather than a small positive weight.
The result is sparser than a bare softmax would give.

The module owns only the adjacency. It is deliberately not a model: the training
script hands the produced matrix to the *unmodified* `GCNGRU`, so an adaptive-
graph run differs from a fixed-graph run in the adjacency and nothing else.
"""

from __future__ import annotations

import torch
from torch import nn


class AdaptiveAdjacency(nn.Module):
    """A learnable [nodes, nodes] adjacency from two node embedding tables.

    Produces `softmax(relu(E1 @ E2.T))`, row-stochastic and generally
    asymmetric. Call it with no arguments; it takes no input because the graph
    is a property of the districts, not of the batch.

    Parameters
    ----------
    n_nodes:
        Number of districts. 25 here.
    embedding_dim:
        Width of each embedding table. The published default for a graph this
        size is 10; 8 is used here to keep the parameter count at 400.
    seed:
        Optional generator seed for the initial embeddings, so a run is
        reproducible independently of global torch state.
    """

    def __init__(self, n_nodes: int, embedding_dim: int = 8, seed: int | None = None):
        super().__init__()

        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim

        if seed is None:
            source = torch.randn(n_nodes, embedding_dim)
            target = torch.randn(n_nodes, embedding_dim)
        else:
            generator = torch.Generator().manual_seed(seed)
            source = torch.randn(n_nodes, embedding_dim, generator=generator)
            target = torch.randn(n_nodes, embedding_dim, generator=generator)

        # Scaled down at initialisation so the pre-softmax logits start small and
        # the initial adjacency is close to uniform. Starting from a sharp random
        # graph would commit the model to arbitrary district pairings before the
        # loss has said anything about them.
        self.source = nn.Parameter(source * 0.1)
        self.target = nn.Parameter(target * 0.1)

    def forward(self) -> torch.Tensor:
        """Return the current [nodes, nodes] adjacency."""

        similarity = torch.relu(self.source @ self.target.T)

        return torch.softmax(similarity, dim=1)

    def extra_repr(self) -> str:
        return f"n_nodes={self.n_nodes}, embedding_dim={self.embedding_dim}"


def count_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class AdaptiveGraphGCNGRU(nn.Module):
    """A `GCNGRU` whose adjacency is learned rather than supplied.

    Wraps an existing `GCNGRU` instance -- built by the caller with exactly the
    baseline's constructor arguments -- and an `AdaptiveAdjacency`. The forward
    signature keeps the `(x, adjacency)` shape the training loop expects, but
    the passed adjacency is **ignored**: the learned one is used instead. That
    lets `scripts/16.train_one` drive this model with no special-casing, while
    keeping the substitution explicit rather than hidden in a default argument.

    The backbone is untouched -- same `GraphConv` layers, same shared GRU, same
    head, same anchored residual target -- so a difference against a fixed-graph
    run is the adjacency and nothing else.
    """

    def __init__(self, backbone: nn.Module, adjacency: AdaptiveAdjacency):
        super().__init__()

        self.backbone = backbone
        self.adjacency = adjacency

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # `adjacency` is accepted and discarded: the training loop passes the
        # fixed matrix for every model, and this one supplies its own.
        del adjacency

        return self.backbone(x, self.adjacency())

    def learned_adjacency(self) -> torch.Tensor:
        """Return the current learned adjacency, detached, for inspection."""

        with torch.no_grad():
            return self.adjacency().detach()

"""
Per-district spatial-versus-temporal fusion gate for the GCN+GRU baseline.

In the committed baseline sweep `gru_only` beats `gcn_gru` on every headline
fold: replacing the contiguity adjacency with the identity -- cutting the graph
out entirely -- lowers the error. The graph is not earning its place. But that
is a statement about the *average* district. Colombo borders six districts and
sits on the main road network; Jaffna borders one. There is no reason the graph
should help or hurt all 25 equally.

So instead of choosing between `gcn_gru` and `gru_only` for the whole country,
let each district choose. For district i,

    fused[i] = g[i] * spatial[i] + (1 - g[i]) * temporal[i]

where `spatial` is the baseline's graph convolution over the real adjacency and
`temporal` is *the same convolution with the same weights* applied with the
identity in place of the adjacency. `g[i] = sigmoid(gate_logit[i])` is one
learned scalar per district.

The shared weights are the whole point. With g[i] = 1 for all i the module is
exactly `gcn_gru`; with g[i] = 0 it is exactly `gru_only`; nothing new sits
between those two except 25 numbers deciding, per district, which one to be. A
separate temporal branch with its own parameters would add capacity and blur
what the gate is measuring. This way the gate value is directly readable as
"how much district i wants the graph", and the interpretability check is
whether it goes low for the isolated northern districts (Jaffna, degree 1;
Mannar, Mullaitivu, Kilinochchi) that the contiguity graph has the least to
say about.

The backbone is untouched. Same `GraphConv` layers, same shared GRU, same
linear head, same input feature count, same anchored residual target. A
difference in the numbers is the gate and nothing else.
"""

from __future__ import annotations

import torch
from torch import nn


class GatedGCNGRU(nn.Module):
    """The baseline GCN+GRU with a per-district spatial/temporal fusion gate.

    Wraps a plain ``GCNGRU`` instance (passed in, exactly the way
    ``src.models.lag_encoder.LagGCNGRU`` wraps a backbone) and reuses its graph
    layers, GRU and head. The only new parameters are ``gate_logit``: one scalar
    per node, initialised to zero so the gate starts at ``sigmoid(0) = 0.5``, an
    even blend with no bias toward either branch.

    Input   [batch, steps, nodes, features]
    Output  [batch, nodes, horizon]

    Parameters
    ----------
    backbone:
        A ``scripts/16`` ``GCNGRU``. Its ``graph_layers``, ``dropout``, ``gru``
        and ``head`` are used directly; its ``forward`` is not called.
    n_nodes:
        Number of districts. Must match the adjacency passed to ``forward``.
    freeze_gate:
        If set, ``gate_logit`` is registered as a buffer at this constant value
        rather than as a learnable parameter. ``freeze_gate=0.5`` gives the
        ``gated_uniform`` ablation -- the identical architecture with the gate
        held at an even blend instead of learned -- so any gap between it and
        the learned gate is attributable to the gate and not to the extra
        branch.
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_nodes: int,
        freeze_gate: float | None = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.n_nodes = n_nodes
        self.freeze_gate = freeze_gate

        if freeze_gate is None:
            self.gate_logit = nn.Parameter(torch.zeros(n_nodes))
        else:
            # A frozen gate is a fixed probability, so store the logit that
            # sigmoid maps to it. 0.5 -> 0.0, and the general case is the
            # logit of the requested probability.
            probability = torch.tensor(float(freeze_gate)).clamp(1e-6, 1 - 1e-6)
            logit = torch.log(probability) - torch.log1p(-probability)
            self.register_buffer("gate_logit", logit.expand(n_nodes).clone())

    def gate(self) -> torch.Tensor:
        """Return the per-district gate values g in (0, 1). [n_nodes]"""

        return torch.sigmoid(self.gate_logit)

    def gate_values(self):
        """Return the gate as a detached numpy array, for the results dump."""

        with torch.no_grad():
            return self.gate().cpu().numpy()

    def _graph_stack(
        self, x: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        """Run the backbone's graph-convolution stack. [batch, steps, nodes, hidden]

        Byte-for-byte the loop from ``GCNGRU.forward`` -- ``relu`` then the
        backbone's own dropout after each ``GraphConv`` layer -- so that
        ``g = 1`` reproduces the baseline exactly rather than approximately.
        """

        spatial = x
        for layer in self.backbone.graph_layers:
            spatial = torch.relu(layer(spatial, adjacency))
            spatial = self.backbone.dropout(spatial)

        return spatial

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, _ = x.shape

        if nodes != self.n_nodes:
            raise ValueError(
                f"Gate has {self.n_nodes} nodes, input has {nodes}."
            )

        identity = torch.eye(nodes, dtype=adjacency.dtype, device=adjacency.device)

        spatial = self._graph_stack(x, adjacency)
        temporal = self._graph_stack(x, identity)

        # g broadcasts over [batch, steps, nodes, hidden] from [nodes].
        g = self.gate().view(1, 1, nodes, 1)
        fused = g * spatial + (1.0 - g) * temporal

        # The baseline's tail, unchanged: one GRU sequence per (batch, node).
        sequences = fused.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)

        output, _ = self.backbone.gru(sequences)
        last = self.backbone.dropout(output[:, -1])

        return self.backbone.head(last).view(batch, nodes, -1)

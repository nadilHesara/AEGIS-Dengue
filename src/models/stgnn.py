"""Spatio-temporal GCN-GRU model used by AEGIS-Dengue experiments."""

from __future__ import annotations

import torch
from torch import nn

from src.models.gnn import GraphConv


class GCNGRU(nn.Module):
    """Graph convolution over districts followed by a shared temporal GRU."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 32,
        gcn_layers: int = 2,
        horizon: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        sizes = [n_features] + [hidden] * gcn_layers
        self.graph_layers = nn.ModuleList(
            GraphConv(sizes[index], sizes[index + 1])
            for index in range(gcn_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, steps, nodes, _ = x.shape
        spatial = x
        for layer in self.graph_layers:
            spatial = self.dropout(torch.relu(layer(spatial, adjacency)))
        sequences = spatial.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)
        output, _ = self.gru(sequences)
        return self.head(self.dropout(output[:, -1])).view(batch, nodes, -1)


def count_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters in a module."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

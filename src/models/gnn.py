"""Reusable dense graph-convolution building block for district graphs."""

from __future__ import annotations

import torch
from torch import nn


class GraphConv(nn.Module):
    """One dense graph convolution over a fixed adjacency matrix."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        projected = self.linear(x)
        return torch.einsum("ij,bljf->blif", adjacency, projected)

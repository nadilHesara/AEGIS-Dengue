"""Reusable checkpoint inference for trained AEGIS-Dengue PyTorch models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


def load_state_dict(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load either a plain state dict or a standard wrapped training checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Checkpoint must be a state dict or contain model_state_dict/state_dict.")


@torch.inference_mode()
def predict(model: nn.Module, inputs: np.ndarray, adjacency: np.ndarray, device: torch.device) -> np.ndarray:
    """Run a model on `[samples, steps, nodes, features]` inputs."""

    if inputs.ndim != 4:
        raise ValueError("inputs must have shape [samples, steps, nodes, features].")
    if adjacency.shape != (inputs.shape[2], inputs.shape[2]):
        raise ValueError("adjacency shape must match the input node axis.")
    model.eval()
    values = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    graph = torch.as_tensor(adjacency, dtype=torch.float32, device=device)
    return model(values, graph).detach().cpu().numpy()

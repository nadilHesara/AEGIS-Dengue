"""Run a trained GCN-GRU checkpoint on prepared model-input windows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.inference.predictor import load_state_dict, predict
from src.models.stgnn import GCNGRU


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="Prepared .npy input windows.")
    parser.add_argument("--adjacency", type=Path, default=PROJECT_DIR / "data" / "processed" / "adjacency.npz")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "results" / "predictions" / "dengue_predictions.csv")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--gcn-layers", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    inputs = np.load(args.inputs)
    archive = np.load(args.adjacency)
    adjacency = archive["A_norm"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNGRU(inputs.shape[-1], args.hidden, args.gcn_layers, args.horizon).to(device)
    model.load_state_dict(load_state_dict(args.checkpoint, device))
    predictions = predict(model, inputs, adjacency, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions.reshape(predictions.shape[0], -1)).to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

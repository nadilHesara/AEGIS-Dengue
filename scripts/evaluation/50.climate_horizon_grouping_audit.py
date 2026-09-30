"""
Audit of the climate-horizon parameter grouping (read-only; nothing is trained).

    python scripts/evaluation/50.climate_horizon_grouping_audit.py

Writes results/climate_horizon/correction_2026-09-30/grouping_audit/:
  parameter_table.csv      component, parameter, gradient source, lr, weight decay
  h4_only_head_rows.csv    climate-head rows in the saved F (h=4-only) checkpoints
  head_row_scaling.json    effect of a per-row gradient scale on one Adam update
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.training import climate_horizon as tr  # noqa: E402

OUT = PROJECT_DIR / "results" / "climate_horizon" / "correction_2026-09-30" / "grouping_audit"
CKPT = PROJECT_DIR / "results" / "climate_horizon" / "checkpoints" / "sweep"

COMPONENT = [
    ("encoder.", "climate delay encoder (HorizonLagEncoder)"),
    ("climate_gru.", "climate feature GRU"),
    ("climate_delta.", "climate correction head"),
    ("case_trunk.", "case branch (GCNGRU, identity adjacency)"),
    ("case_delta.", "case correction head"),
    ("case_alpha.", "dispersion head (case-based)"),
    ("gate_", "gate"),
]


def component(name: str) -> str:
    return next(label for prefix, label in COMPONENT if name.startswith(prefix))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dims = {"n_nodes": 25, "n_case": 9, "n_climate": 7, "n_horizons": 4, "lookback": 12, "lag_reach": 26}
    config = dict(tr.DEFAULTS)
    model = tr.build_model(dims, config)
    optimiser = tr.build_optimiser(model, config)
    climate = {id(p) for p in model.climate_parameters()}
    group_of = {}
    for index, group in enumerate(optimiser.param_groups):
        for p in group["params"]:
            group_of[id(p)] = (group["lr"], group["weight_decay"])

    rows = []
    for name, p in model.named_parameters():
        lr, wd = group_of[id(p)]
        rows.append({"component": component(name), "parameter": name, "shape": tuple(p.shape),
                     "numel": p.numel(),
                     "gradient_source (as run)": "grad of L_climate (weighted)" if id(p) in climate
                     else "grad of L_equal (unweighted)",
                     "lr": lr, "weight_decay": wd})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "parameter_table.csv", index=False)
    print(table.groupby(["component", "gradient_source (as run)", "lr", "weight_decay"])["numel"].sum().to_string())

    # F arm: rows of the climate head for h = 1..3 in the saved checkpoints.
    head_rows = []
    for path in sorted((CKPT / "F_target_h4_only").glob("fold*_seed*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        state = saved["training_state"]["best_state"]
        w, b = state["climate_delta.weight"], state["climate_delta.bias"]
        for h in range(4):
            head_rows.append({"checkpoint": path.name, "horizon": h + 1,
                              "max_abs_weight": float(w[h].abs().max()), "abs_bias": float(b[h].abs())})
    head = pd.DataFrame(head_rows)
    head.to_csv(OUT / "h4_only_head_rows.csv", index=False)
    print(head.groupby("horizon")[["max_abs_weight", "abs_bias"]].max().to_string())

    # A head row receives gradient only from its own horizon, so a weight w_h
    # scales that row's gradient by a constant. Adam normalises per element,
    # so apart from eps, weight decay and global clipping, the update is
    # unchanged. The encoder and GRU mix horizons and are affected.
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(4, 16))
    g = torch.randn(4, 16) * 1e-2
    updates = {}
    for label, scale in (("unit", torch.ones(4, 1)), ("scaled", torch.tensor([[0.41], [0.75], [1.19], [1.65]]))):
        q = torch.nn.Parameter(p.detach().clone())
        opt = torch.optim.Adam([q], lr=3e-3, weight_decay=0.0)
        q.grad = g * scale
        opt.step()
        updates[label] = q.detach().clone()
    rel = float((updates["unit"] - updates["scaled"]).abs().max() / updates["unit"].abs().max())
    (OUT / "head_row_scaling.json").write_text(json.dumps({"max_relative_update_difference": rel}, indent=2))
    print("Adam, one step, per-row scale 0.41-1.65: max relative update difference", rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

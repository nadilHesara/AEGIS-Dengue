"""
Climate-only gradient weighting from one forward pass (supersedes the
backward-only `_GradScale` router, which stays at weight 1).

The baseline reduction is the pooled masked NB loss, sum_h N_h / sum_h D_h,
where N_h is horizon h's summed NLL over observed cells and D_h its observed
cell count. Over the active horizons (D_h > 0 in the batch):

    L_h       = N_h / D_h
    a_h       = D_h / sum_h D_h
    L_equal   = sum_h a_h L_h                        (= the pooled baseline loss)
    L_climate = sum_h a_h w_h L_h / sum_h a_h w_h

The gradient of L_climate goes to the climate group
(`model.climate_parameters()`: lag encoder and climate GRU; routing v2). The
gradient of L_equal goes to every other parameter, including all prediction heads. Each gradient is assigned
once. The optimiser's global-norm clipping (max 1.0, over all parameters,
as in the baseline) is then applied, followed by one update.

With w = 1, L_climate = L_equal exactly, so the gradients, the clipping and
the update are the baseline's. The weights are detached constants. If the
active horizons have sum a_h w_h = 0, the climate parameters get no
gradient this step (grad None, so Adam skips them) and the other parameters
still train.

The mean-one weights fix the average weight, not the gradient norms.
`step` therefore logs raw (pre-clip) and clipped norms per group and,
occasionally, the cosine between the climate group's weighted and equal
gradients.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.climate_horizon import nb_loss_terms


def weighted_losses(numerator: torch.Tensor, denominator: torch.Tensor,
                    weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    """L_equal and L_climate (None when the active climate weight is zero)."""

    active = denominator > 0
    safe = torch.where(active, denominator, torch.ones_like(denominator))
    per_h = torch.where(active, numerator / safe, torch.zeros_like(numerator))
    a = torch.where(active, denominator, torch.zeros_like(denominator)) / denominator.sum().clamp(min=1)
    w = weights.detach().to(per_h.dtype)
    loss_equal = (a * per_h).sum()
    climate_mass = (a * w).sum()
    loss_climate = (a * w * per_h).sum() / climate_mass if climate_mass > 0 else None
    return loss_equal, loss_climate, {"per_horizon": per_h.detach(), "a": a.detach(),
                                      "climate_mass": float(climate_mass)}


def _norm(grads) -> float:
    grads = [g for g in grads if g is not None]
    return float(torch.norm(torch.stack([g.norm() for g in grads]))) if grads else 0.0


def _cosine(a, b) -> float:
    pairs = [(x.flatten(), y.flatten()) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    x = torch.cat([p[0] for p in pairs])
    y = torch.cat([p[1] for p in pairs])
    return float(torch.nn.functional.cosine_similarity(x, y, dim=0))


def compute_gradients(model, batch: dict, weights: torch.Tensor, log_cosine: bool = False,
                      hook=None) -> dict:
    """One forward pass; returns the climate and other gradients (not yet assigned) and logs.

    `hook(model, per_horizon_losses, active)` runs on the live graph before it is
    released (used by the GradNorm control); it must not touch parameter grads.
    """

    out = model(batch["X_case"], batch["X_climate"], batch["anchor"])
    numerator, denominator = nb_loss_terms(out["mu"], out["alpha"], batch["y"], batch["mask"])
    loss_equal, loss_climate, info = weighted_losses(numerator, denominator, weights)

    climate = list(model.climate_parameters())
    other = list(model.other_parameters())
    if float(denominator.sum()) == 0:
        # No observed target in the batch: no gradient for anyone, and `step`
        # skips the optimiser so weight decay cannot move parameters on zeros.
        return {"climate": climate, "climate_grads": None, "other": other, "other_grads": None,
                "log": {"loss_equal": 0.0, "loss_climate": float("nan"), "climate_mass": 0.0,
                        "raw_norm_climate": 0.0, "raw_norm_other": 0.0, "climate_skipped": True,
                        "empty_batch": True}}
    climate_grads = None
    if climate and loss_climate is not None:
        climate_grads = torch.autograd.grad(loss_climate, climate, retain_graph=True, allow_unused=True)
    equal_climate = None
    if log_cosine and climate:
        equal_climate = torch.autograd.grad(loss_equal, climate, retain_graph=True, allow_unused=True)
    hook_log = {}
    if hook is not None:
        active = denominator > 0
        per_h = torch.where(active, numerator / torch.where(active, denominator, torch.ones_like(denominator)),
                            torch.zeros_like(numerator))
        hook_log = hook(model, per_h, active) or {}
    other_grads = torch.autograd.grad(loss_equal, other, allow_unused=True)

    log = {**hook_log, "loss_equal": float(loss_equal.detach()),
           "loss_climate": float(loss_climate.detach()) if loss_climate is not None else float("nan"),
           "climate_mass": info["climate_mass"],
           "raw_norm_climate": _norm(climate_grads) if climate_grads is not None else 0.0,
           "raw_norm_other": _norm(other_grads),
           "climate_skipped": climate_grads is None and bool(climate)}
    if equal_climate is not None:
        log["raw_norm_climate_equal"] = _norm(equal_climate)
        log["cosine_climate_weighted_vs_equal"] = (_cosine(climate_grads, equal_climate)
                                                   if climate_grads is not None else float("nan"))
    return {"climate": climate, "climate_grads": climate_grads, "other": other,
            "other_grads": other_grads, "log": log}


def step(model, optimiser, batch: dict, weights: torch.Tensor, clip: float = 1.0,
         log_cosine: bool = False, hook=None) -> dict:
    """Assign each gradient once, clip globally, and make one optimiser update."""

    optimiser.zero_grad(set_to_none=True)
    result = compute_gradients(model, batch, weights, log_cosine, hook)
    for params, grads in ((result["climate"], result["climate_grads"]), (result["other"], result["other_grads"])):
        if grads is None:
            continue
        for p, g in zip(params, grads):
            p.grad = None if g is None else g.detach().clone()
    if result["other_grads"] is None and result["climate_grads"] is None:
        log = result["log"]
        log.update(raw_norm_total=0.0, clip_coefficient=1.0, clipped_norm_climate=0.0,
                   clipped_norm_other=0.0, optimiser_step_skipped=True)
        return log
    total = float(nn.utils.clip_grad_norm_(model.parameters(), clip))
    log = result["log"]
    log["raw_norm_total"] = total
    log["clip_coefficient"] = min(1.0, clip / (total + 1e-6))
    log["clipped_norm_climate"] = _norm([p.grad for p in result["climate"]])
    log["clipped_norm_other"] = _norm([p.grad for p in result["other"]])
    optimiser.step()
    return log

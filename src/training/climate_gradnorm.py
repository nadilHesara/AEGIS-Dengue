"""
GradNorm control restricted to the climate parameter group.

Core update of GradNorm (Chen, Badrinarayanan, Lee & Rabinovich, ICML 2018),
written from the paper's algorithm description (re-check against the paper
before citing). Per training step, with task weights w_i and task losses L_i:

    G_i   = || grad_W (w_i L_i) ||           W = the last shared layer
    G_bar = mean_i G_i
    r_i   = (L_i(t) / L_i(0)) / mean_j (L_j(t) / L_j(0))
    L_grad = sum_i | G_i - [G_bar * r_i^alpha] |     (bracket held constant)

The gradient of L_grad updates only the w_i, which are then renormalised so
that sum_i w_i = T (the number of tasks).

How this control differs from the paper (labelled):
- The weights act only on the climate-only loss L_climate. The case branch,
  gate and heads still learn from the unweighted L_equal. In the paper the
  weighted loss trains the whole network.
- W is the climate GRU's recurrent weight matrix, the last layer shared by
  all horizons inside the climate branch. The paper uses the last shared
  layer of the whole network.
- alpha = 1.5 and weight learning rate 0.025 are fixed a priori, not tuned.
  The paper tunes alpha per task set.
- L_climate is normalised by sum a_h w_h, so it is invariant to the scale
  of w. The renormalisation to sum T therefore affects only the logs.
"""

from __future__ import annotations

import torch

ALPHA = 1.5
WEIGHT_LR = 0.025


class ClimateGradNorm:
    def __init__(self, n_horizons: int, device, alpha: float = ALPHA, lr: float = WEIGHT_LR):
        self.w = torch.ones(n_horizons, device=device, requires_grad=True)
        self.optimiser = torch.optim.Adam([self.w], lr=lr)
        self.alpha = alpha
        self.initial = None
        self.n = n_horizons

    def weights(self) -> torch.Tensor:
        return self.w.detach().clone()

    def __call__(self, model, per_h: torch.Tensor, active: torch.Tensor) -> dict:
        shared = model.climate_gru.weight_hh_l0
        if not bool(active.all()):
            return {"gradnorm_skipped": True}
        if self.initial is None:
            self.initial = per_h.detach().clamp(min=1e-8)
        norms = []
        for h in range(self.n):
            g = torch.autograd.grad(per_h[h], shared, retain_graph=True)[0]
            norms.append(self.w[h] * g.norm())          # ||grad(w_h L_h)|| = w_h ||grad L_h||
        norms = torch.stack(norms)
        ratio = per_h.detach() / self.initial
        relative = ratio / ratio.mean()
        target = (norms.mean() * relative ** self.alpha).detach()
        loss = (norms - target).abs().sum()
        self.optimiser.zero_grad()
        self.w.grad = torch.autograd.grad(loss, self.w)[0]
        self.optimiser.step()
        with torch.no_grad():
            self.w.clamp_(min=1e-4)
            self.w.mul_(self.n / self.w.sum())
        return {"gradnorm_loss": float(loss.detach()),
                **{f"gradnorm_w{h + 1}": float(v) for h, v in enumerate(self.w.detach())}}

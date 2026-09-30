"""GradNorm climate-group control."""

from __future__ import annotations

import torch

from src.models.climate_horizon import ClimateHorizonNB
from src.training import climate_weighting as cw
from src.training.climate_gradnorm import ClimateGradNorm

N, FC, K, L, R, H, B = 4, 5, 3, 12, 26, 4, 6


def setup():
    torch.manual_seed(0)
    m = ClimateHorizonNB(N, FC, K, lag_reach=R, dropout=0.0)
    with torch.no_grad():
        for layer in (m.case_delta, m.case_alpha, m.climate_delta, m.gate_state):
            layer.weight.normal_(0, 0.2)
    g = torch.Generator().manual_seed(1)
    batch = {"X_case": torch.randn(B, L, N, FC, generator=g), "X_climate": torch.randn(B, L + R - 1, N, K, generator=g),
             "anchor": torch.log1p(torch.randint(0, 200, (B, N), generator=g).float()),
             "y": torch.randint(0, 300, (B, N, H), generator=g).float(), "mask": torch.ones(B, N, H)}
    return m, batch


def test_weights_stay_positive_and_sum_to_the_task_count():
    m, batch = setup()
    control = ClimateGradNorm(H, torch.device("cpu"))
    for _ in range(3):
        log = cw.compute_gradients(m, batch, control.weights(), hook=control)["log"]
    w = control.weights()
    assert torch.isclose(w.sum(), torch.tensor(4.0)) and (w > 0).all()
    assert "gradnorm_loss" in log and not torch.allclose(w, torch.ones(H))


def test_gradnorm_does_not_change_other_gradients():
    m, batch = setup()
    plain = cw.compute_gradients(m, batch, torch.ones(H))
    control = ClimateGradNorm(H, torch.device("cpu"))
    with_hook = cw.compute_gradients(m, batch, torch.ones(H), hook=control)
    for a, b in zip(plain["other_grads"], with_hook["other_grads"]):
        assert (a is None and b is None) or torch.equal(a, b)
    for a, b in zip(plain["climate_grads"], with_hook["climate_grads"]):
        assert (a is None and b is None) or torch.equal(a, b)

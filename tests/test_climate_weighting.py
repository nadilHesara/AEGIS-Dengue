"""Numerical checks for climate-only gradient weighting (full precision)."""

from __future__ import annotations

import copy

import torch
from torch import nn

from src.models.climate_horizon import ClimateHorizonNB, nb_loss_by_horizon, nb_loss_terms
from src.models.negative_binomial import masked_negative_binomial_loss
from src.training import climate_weighting as cw

D = torch.float64  # full precision, set per tensor so no global state leaks
N, FC, K, L, R, H, B = 4, 5, 3, 12, 26, 4, 6


def make_model(seed=0):
    torch.manual_seed(seed)
    m = ClimateHorizonNB(N, FC, K, n_horizons=H, lag_reach=R, dropout=0.0).double()
    with torch.no_grad():  # make every head live
        for layer in (m.case_delta, m.case_alpha, m.climate_delta, m.gate_state):
            layer.weight.normal_(0, 0.2)
            layer.bias.normal_(0, 0.2)
    return m


def make_batch(seed=1, drop=True):
    g = torch.Generator().manual_seed(seed)
    anchor = torch.log1p(torch.randint(0, 200, (B, N), generator=g).double())
    y = torch.randint(0, 300, (B, N, H), generator=g).double()
    mask = (torch.rand(B, N, H, generator=g, dtype=D) > (0.25 if drop else -1)).double()
    return {"X_case": torch.randn(B, L, N, FC, generator=g, dtype=D),
            "X_climate": torch.randn(B, L + R - 1, N, K, generator=g, dtype=D),
            "anchor": anchor, "y": y, "mask": mask}


def adam(model):
    enc = {id(p) for p in model.encoder.parameters()}
    return torch.optim.Adam([{"params": [p for p in model.parameters() if id(p) not in enc]},
                             {"params": [p for p in model.parameters() if id(p) in enc], "lr": 2e-2,
                              "weight_decay": 0.0}], lr=3e-3, weight_decay=1e-4)


def grads_of(model, loss):
    return {n: (g.clone() if g is not None else None) for (n, _), g in zip(
        model.named_parameters(), torch.autograd.grad(loss, list(model.parameters()), allow_unused=True))}


def test_loss_terms_preserve_the_pooled_reduction():
    m, batch = make_model(), make_batch()
    out = m(batch["X_case"], batch["X_climate"], batch["anchor"])
    num, den = nb_loss_terms(out["mu"], out["alpha"], batch["y"], batch["mask"])
    pooled = masked_negative_binomial_loss(out["mu"], out["alpha"], batch["y"], batch["mask"])
    assert torch.allclose(num.sum() / den.sum(), pooled)
    assert torch.allclose(num / den.sum(), nb_loss_by_horizon(out["mu"], out["alpha"], batch["y"], batch["mask"]))
    equal, climate, _ = cw.weighted_losses(num, den, torch.ones(H, dtype=D))
    assert torch.allclose(equal, pooled) and torch.allclose(climate, pooled)


def test_unit_weights_reproduce_baseline_loss_gradients_and_update():
    batch = make_batch()
    base, new = make_model(), make_model()
    opt_base, opt_new = adam(base), adam(new)

    opt_base.zero_grad()
    out = base(batch["X_case"], batch["X_climate"], batch["anchor"])
    loss = masked_negative_binomial_loss(out["mu"], out["alpha"], batch["y"], batch["mask"])
    loss.backward()
    base_grads = {n: p.grad.clone() for n, p in base.named_parameters() if p.grad is not None}
    nn.utils.clip_grad_norm_(base.parameters(), 1.0)
    opt_base.step()

    result = cw.compute_gradients(new, batch, torch.ones(H, dtype=D))
    raw = dict(zip([id(p) for p in result["climate"] + result["other"]],
                   list(result["climate_grads"]) + list(result["other_grads"])))
    for n, p in new.named_parameters():
        if n in base_grads:
            assert torch.allclose(raw[id(p)], base_grads[n], atol=1e-12, rtol=1e-10), n
    log = cw.step(new, opt_new, batch, torch.ones(H, dtype=D))
    assert abs(log["loss_equal"] - loss.item()) < 1e-12
    for (n, a), (_, b) in zip(base.named_parameters(), new.named_parameters()):
        assert torch.allclose(a, b, atol=1e-12, rtol=1e-10), n


def test_weights_change_climate_raw_gradients_only():
    m, batch = make_model(), make_batch()
    uniform = cw.compute_gradients(m, batch, torch.ones(H, dtype=D))
    skewed = cw.compute_gradients(m, batch, torch.tensor([0.5, 0.8, 1.2, 1.5], dtype=D))
    for a, b in zip(uniform["other_grads"], skewed["other_grads"]):
        assert (a is None and b is None) or torch.equal(a, b)
    assert any(not torch.allclose(a, b) for a, b in zip(uniform["climate_grads"], skewed["climate_grads"])
               if a is not None)


def test_climate_gradients_match_the_explicit_weighted_combination():
    m, batch = make_model(), make_batch()
    w = torch.tensor([0.05, 0.3, 1.6, 2.05], dtype=D)
    result = cw.compute_gradients(m, batch, w)
    out = m(batch["X_case"], batch["X_climate"], batch["anchor"])
    num, den = nb_loss_terms(out["mu"], out["alpha"], batch["y"], batch["mask"])
    a = den / den.sum()
    climate = list(m.climate_parameters())
    expected = [torch.zeros_like(p) for p in climate]
    for h in range(H):
        g_h = torch.autograd.grad(num[h] / den[h], climate, retain_graph=True, allow_unused=True)
        for i, g in enumerate(g_h):
            if g is not None:
                expected[i] += a[h] * w[h] * g / (a * w).sum()
    for got, exp in zip(result["climate_grads"], expected):
        assert torch.allclose(got if got is not None else torch.zeros_like(exp), exp, atol=1e-12)


def test_masked_targets_cannot_affect_gradients():
    m, batch = make_model(), make_batch()
    w = torch.tensor([0.5, 0.8, 1.2, 1.5], dtype=D)
    clean = cw.compute_gradients(m, batch, w)
    masked = batch["mask"] == 0
    huge = {**batch, "y": torch.where(masked, torch.full_like(batch["y"], 1e9), batch["y"])}
    nan = {**batch, "y": torch.where(masked, torch.full_like(batch["y"], float("nan")), batch["y"])}
    for variant in (huge, nan):
        noisy = cw.compute_gradients(m, variant, w)
        for a, b in zip(clean["climate_grads"] + clean["other_grads"], noisy["climate_grads"] + noisy["other_grads"]):
            assert (a is None and b is None) or (torch.isfinite(b).all() and torch.equal(a, b))


def test_zero_active_climate_weight_skips_only_the_climate_update():
    m, batch = make_model(), make_batch(drop=False)
    batch["mask"][..., 3] = 0.0                      # horizon 4 inactive in this batch
    before = copy.deepcopy(m.state_dict())
    log = cw.step(m, adam(m), batch, torch.tensor([0.0, 0.0, 0.0, 4.0], dtype=D))
    assert log["climate_skipped"]
    climate = {id(p) for p in m.climate_parameters()}
    for n, p in m.named_parameters():
        if id(p) in climate:
            assert torch.equal(p, before[n]), n
    assert any(not torch.equal(p, before[n]) for n, p in m.named_parameters() if id(p) not in climate)


def test_logs_distinguish_raw_and_clipped_norms():
    m, batch = make_model(), make_batch()
    log = cw.step(m, adam(m), batch, torch.tensor([0.5, 0.8, 1.2, 1.5], dtype=D), clip=1e-3, log_cosine=True)
    assert log["raw_norm_total"] > 1e-3 and log["clip_coefficient"] < 1
    clipped = (log["clipped_norm_climate"] ** 2 + log["clipped_norm_other"] ** 2) ** 0.5
    assert abs(clipped - 1e-3) < 1e-6
    assert -1 <= log["cosine_climate_weighted_vs_equal"] <= 1 and log["raw_norm_climate_equal"] > 0

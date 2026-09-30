"""Routing v2 (correction 2026-09-30): only the climate feature encoder is weighted.

Weighted loss L_climate -> lag encoder + climate GRU.
Unweighted loss L_equal -> case branch, gate, case head, climate head, dispersion.
All checks in float64 and after a short warm-up, because zero-initialised heads
make the encoder's gradient exactly zero at step 0.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from src.models.climate_horizon import ROUTING_VERSION, ClimateHorizonNB, nb_loss_terms
from src.models.negative_binomial import masked_negative_binomial_loss
from src.training import climate_horizon as tr
from src.training import climate_weighting as cw

D = torch.float64
N, FC, K, L, R, H, B = 4, 5, 3, 12, 26, 4, 8


def batch(seed=1, drop=0.25):
    g = torch.Generator().manual_seed(seed)
    return {"X_case": torch.randn(B, L, N, FC, generator=g, dtype=D),
            "X_climate": torch.randn(B, L + R - 1, N, K, generator=g, dtype=D),
            "anchor": torch.log1p(torch.randint(0, 200, (B, N), generator=g).double()),
            "y": torch.randint(0, 300, (B, N, H), generator=g).double(),
            "mask": (torch.rand(B, N, H, generator=g, dtype=D) > drop).double()}


def optimiser(m):
    return tr.build_optimiser(m, tr.DEFAULTS)


def warmed(seed=0, steps=5):
    """A model after a few uniform-weight steps, so every head is non-zero."""

    torch.manual_seed(seed)
    m = ClimateHorizonNB(N, FC, K, lag_reach=R, dropout=0.0).double()
    opt = optimiser(m)
    for s in range(steps):
        cw.step(m, opt, batch(seed=100 + s), torch.ones(H, dtype=D))
    assert m.climate_delta.weight.abs().max() > 0 and m.case_delta.weight.abs().max() > 0
    return m, opt


def test_groups_are_exact_and_learning_rates_unchanged():
    m = ClimateHorizonNB(N, FC, K, lag_reach=R)
    assert ROUTING_VERSION == "v2-heads-unweighted"
    names = {id(p): n for n, p in m.named_parameters()}
    climate = [names[id(p)] for p in m.climate_parameters()]
    other = [names[id(p)] for p in m.other_parameters()]
    assert all(n.startswith(("encoder.", "climate_gru.")) for n in climate)
    assert {"climate_delta.weight", "climate_delta.bias", "case_delta.weight", "case_alpha.weight",
            "gate_state.weight", "gate_district", "gate_horizon"} <= set(other)
    assert sorted(climate + other) == sorted(names.values())          # each exactly once
    opt = optimiser(m)
    setting = {id(p): (g["lr"], g["weight_decay"]) for g in opt.param_groups for p in g["params"]}
    for n, p in m.named_parameters():                                 # same as the v1 audit table
        expected = (2e-2, 0.0) if n.startswith("encoder.") else (3e-3, 1e-4)
        assert setting[id(p)] == expected, n


def test_uniform_weights_reproduce_baseline_gradients_and_update():
    base, opt_base = warmed()
    new = copy.deepcopy(base)
    opt_new = optimiser(new)
    # deepcopy: load_state_dict can share Adam moment tensors with the source.
    opt_new.load_state_dict(copy.deepcopy(opt_base.state_dict()))
    b = batch(seed=7)

    opt_base.zero_grad()
    out = base(b["X_case"], b["X_climate"], b["anchor"])
    masked_negative_binomial_loss(out["mu"], out["alpha"], b["y"], b["mask"]).backward()
    reference = {n: p.grad.clone() for n, p in base.named_parameters()}
    nn.utils.clip_grad_norm_(base.parameters(), 1.0)
    opt_base.step()

    result = cw.compute_gradients(new, b, torch.ones(H, dtype=D))
    raw = dict(zip([id(p) for p in result["climate"] + result["other"]],
                   list(result["climate_grads"]) + list(result["other_grads"])))
    for n, p in new.named_parameters():
        assert torch.allclose(raw[id(p)], reference[n], rtol=1e-10, atol=1e-13), n
    cw.step(new, opt_new, b, torch.ones(H, dtype=D))
    for (n, a), (_, c) in zip(base.named_parameters(), new.named_parameters()):
        assert torch.allclose(a, c, rtol=1e-10, atol=1e-13), n


def test_nonuniform_weights_change_only_climate_raw_gradients():
    m, _ = warmed()
    b = batch(seed=8)
    uniform = cw.compute_gradients(m, b, torch.ones(H, dtype=D))
    skewed = cw.compute_gradients(m, b, torch.tensor([0.4, 0.8, 1.2, 1.6], dtype=D))
    for a, c in zip(uniform["other_grads"], skewed["other_grads"]):   # includes both heads
        assert torch.equal(a, c)
    assert any(not torch.allclose(a, c) for a, c in zip(uniform["climate_grads"], skewed["climate_grads"]))
    # Only raw gradients are compared: global clipping may legitimately change
    # the other group's final update when the climate norm changes.


def test_routed_gradient_matches_per_horizon_combination():
    m, _ = warmed()
    b = batch(seed=9)
    w = torch.tensor([0.05, 0.6, 1.3, 2.05], dtype=D)
    got = cw.compute_gradients(m, b, w)["climate_grads"]
    out = m(b["X_case"], b["X_climate"], b["anchor"])
    num, den = nb_loss_terms(out["mu"], out["alpha"], b["y"], b["mask"])
    a = den / den.sum()
    expected = [torch.zeros_like(p) for p in m.climate_parameters()]
    for h in range(H):
        g_h = torch.autograd.grad(num[h] / den[h], m.climate_parameters(), retain_graph=True)
        for i, g in enumerate(g_h):
            expected[i] += a[h] * w[h] * g / (a * w).sum()
    for x, e in zip(got, expected):
        assert torch.allclose(x, e, atol=1e-12)
    assert any(x.abs().max() > 0 for x in got)                       # not trivially zero after warm-up


def test_h4_only_weighting_keeps_every_head_learning():
    torch.manual_seed(0)
    m = ClimateHorizonNB(N, FC, K, lag_reach=R, dropout=0.0).double()
    opt = optimiser(m)
    w = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=D)
    first = cw.compute_gradients(m, batch(seed=11), w)
    head = dict(zip([id(p) for p in first["other"]], first["other_grads"]))
    assert head[id(m.climate_delta.bias)][:3].abs().min() > 0        # h=1-3 climate head gets L_equal
    for s in range(10):
        cw.step(m, opt, batch(seed=20 + s), w)
    for layer in (m.climate_delta, m.case_delta, m.case_alpha):
        assert (layer.weight.abs().amax(dim=1) > 0).all(), layer     # all four rows learned


def test_only_h4_contributes_to_the_routed_encoder_gradient():
    m, _ = warmed()
    b = batch(seed=12)
    got = cw.compute_gradients(m, b, torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=D))["climate_grads"]
    out = m(b["X_case"], b["X_climate"], b["anchor"])
    num, den = nb_loss_terms(out["mu"], out["alpha"], b["y"], b["mask"])
    only_h4 = torch.autograd.grad(num[3] / den[3], m.climate_parameters())
    for x, e in zip(got, only_h4):
        assert torch.allclose(x, e, atol=1e-12)


def test_missing_targets_empty_batches_and_zero_climate_weight():
    m, opt = warmed()
    b = batch(seed=13)
    nan = {**b, "y": torch.where(b["mask"] > 0, b["y"], torch.full_like(b["y"], float("nan")))}
    clean = cw.compute_gradients(m, b, torch.ones(H, dtype=D))
    dirty = cw.compute_gradients(m, nan, torch.ones(H, dtype=D))
    for a, c in zip(clean["climate_grads"] + clean["other_grads"], dirty["climate_grads"] + dirty["other_grads"]):
        assert torch.isfinite(c).all() and torch.equal(a, c)

    empty = {**b, "mask": torch.zeros_like(b["mask"])}
    before = copy.deepcopy(m.state_dict())
    log = cw.step(m, opt, empty, torch.ones(H, dtype=D))
    assert log["empty_batch"] and log["optimiser_step_skipped"]
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in before.items())

    zero = {**b, "mask": b["mask"].clone()}
    zero["mask"][..., 3] = 0
    before = copy.deepcopy(m.state_dict())
    log = cw.step(m, opt, zero, torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=D))
    assert log["climate_skipped"] and not log.get("optimiser_step_skipped", False)
    climate = {id(p) for p in m.climate_parameters()}
    for n, p in m.named_parameters():
        if id(p) in climate:
            assert torch.equal(p, before[n]), n
    assert not torch.equal(m.climate_delta.weight, before["climate_delta.weight"])  # head still learns

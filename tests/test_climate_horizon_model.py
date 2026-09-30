"""Two-branch climate-horizon NB model: shapes, safety, parameter groups, gate, routing."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.climate_horizon import ClimateHorizonNB, nb_loss_by_horizon

N, FC, K, L, R, H = 5, 9, 7, 12, 26, 4


def inputs(batch=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, L, N, FC, generator=g), torch.randn(batch, L + R - 1, N, K, generator=g),
            torch.log1p(torch.randint(0, 300, (batch, N), generator=g).float()))


def model(seed=0, randomise=True):
    torch.manual_seed(seed)
    m = ClimateHorizonNB(N, FC, K, n_horizons=H, lag_reach=R, dropout=0.0)
    if randomise:  # move the zero-initialised heads so every path is live
        with torch.no_grad():
            for layer in (m.case_delta, m.case_alpha, m.climate_delta, m.gate_state):
                layer.weight.normal_(0, 0.1)
                layer.bias.normal_(0, 0.1)
    return m.eval()


def test_shapes_and_finite_outputs():
    out = model()(*inputs())
    for key in ("mu", "alpha", "gate", "delta_case", "delta_climate", "gated_climate", "delta_total"):
        assert out[key].shape == (3, N, H) and torch.isfinite(out[key]).all()
    assert (out["mu"] > 0).all() and (out["alpha"] > 0).all()
    assert ((out["gate"] > 0) & (out["gate"] < 1)).all()
    assert torch.allclose(out["gated_climate"], out["gate"] * out["delta_climate"])


def test_extreme_inputs_stay_finite():
    x_case, x_climate, anchor = inputs()
    out = model()(x_case * 1e3, x_climate * 1e3, anchor + 20)
    assert torch.isfinite(out["mu"]).all() and torch.isfinite(out["alpha"]).all()


def test_initial_mean_is_the_nb_anchor():
    x_case, x_climate, anchor = inputs()
    out = model(randomise=False)(x_case, x_climate, anchor)
    assert torch.allclose(out["mu"], torch.expm1(anchor).unsqueeze(-1).expand(-1, -1, H) + 1, rtol=1e-5)


def test_parameter_groups_are_disjoint_and_complete():
    m = model()
    climate = {id(p) for p in m.climate_parameters()}
    other = {id(p) for p in m.other_parameters()}
    everything = {id(p) for p in m.parameters() if p.requires_grad}
    assert not climate & other and climate | other == everything
    assert {id(p) for p in m.encoder.parameters()} <= climate
    for part in (m.case_trunk, m.case_delta, m.case_alpha, m.gate_state):
        assert {id(p) for p in part.parameters()} <= other
    assert id(m.gate_district) in other and id(m.gate_horizon) in other


def test_gate_off_removes_climate_influence():
    m = model()
    x_case, x_climate, anchor = inputs()
    off = m(x_case, x_climate, anchor, gate_off=True)
    off_other_weather = m(x_case, torch.randn_like(x_climate) * 5, anchor, gate_off=True)
    assert torch.equal(off["mu"], off_other_weather["mu"])
    assert torch.equal(off["alpha"], off_other_weather["alpha"])
    on = m(x_case, x_climate, anchor)
    assert not torch.allclose(on["mu"], off["mu"])

    y = torch.poisson(torch.expm1(anchor).unsqueeze(-1).expand(-1, -1, H) + 1)
    m.zero_grad()
    out = m(x_case, x_climate, anchor, gate_off=True)
    nb_loss_by_horizon(out["mu"], out["alpha"], y, torch.ones_like(y)).sum().backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in m.climate_parameters())


def test_dispersion_is_case_based():
    m = model()
    x_case, x_climate, anchor = inputs()
    a = m(x_case, x_climate, anchor)["alpha"]
    b = m(x_case, torch.randn_like(x_climate), anchor)["alpha"]
    assert torch.equal(a, b)


def _grads(m, weights, seed=1):
    m.set_climate_weights(weights)
    m.zero_grad()
    x_case, x_climate, anchor = inputs(seed=seed)
    out = m(x_case, x_climate, anchor)
    y = torch.round(out["mu"].detach() * 1.3)
    losses = nb_loss_by_horizon(out["mu"], out["alpha"], y, torch.ones_like(y))
    losses.sum().backward()
    return ([p.grad.clone() for p in m.climate_parameters()],
            [p.grad.clone() if p.grad is not None else None for p in m.other_parameters()], losses)


def test_uniform_weights_equal_plain_summed_loss():
    m = model()
    climate, other, _ = _grads(m, [1, 1, 1, 1])
    m.zero_grad()
    x_case, x_climate, anchor = inputs(seed=1)
    out = m(x_case, x_climate, anchor)
    y = torch.round(out["mu"].detach() * 1.3)
    # reference: the model with the router bypassed is the same graph with weights 1
    losses = nb_loss_by_horizon(out["mu"], out["alpha"], y, torch.ones_like(y))
    reference = torch.autograd.grad(losses.sum(), m.climate_parameters(), allow_unused=True)
    for got, ref in zip(climate, reference):
        assert ref is None or torch.allclose(got, ref, atol=1e-7)


def test_router_matches_two_pass_reference_and_leaves_other_groups_alone():
    # Legacy backward-only router (superseded by src/training/climate_weighting.py
    # and fixed at weight 1 in training). It scales the gradient at the climate
    # head's output, so under routing v2 it also scales the head's own gradient;
    # only the case branch, gate and dispersion are checked as untouched here.
    torch.manual_seed(0)
    m = model()
    weights = torch.tensor([0.0, 0.5, 1.5, 2.0])
    _, other_uniform, _ = _grads(m, [1, 1, 1, 1])
    climate, other, losses = _grads(m, weights)
    head = {id(p) for p in m.climate_delta.parameters()}
    for p, a, b in zip(m.other_parameters(), other, other_uniform):
        if id(p) in head:
            continue
        assert (a is None and b is None) or torch.allclose(a, b, atol=1e-7)

    m.set_climate_weights([1, 1, 1, 1])
    x_case, x_climate, anchor = inputs(seed=1)
    out = m(x_case, x_climate, anchor)
    y = torch.round(out["mu"].detach() * 1.3)
    ref_losses = nb_loss_by_horizon(out["mu"], out["alpha"], y, torch.ones_like(y))
    reference = torch.autograd.grad((weights * ref_losses).sum(), m.climate_parameters(), allow_unused=True)
    for got, ref in zip(climate, reference):
        assert torch.allclose(got, ref if ref is not None else torch.zeros_like(got), atol=1e-6)


def test_negative_or_wrong_length_weights_rejected():
    m = model()
    with pytest.raises(ValueError):
        m.set_climate_weights([1, -1, 1, 1])
    with pytest.raises(ValueError):
        m.set_climate_weights([1, 1, 1])

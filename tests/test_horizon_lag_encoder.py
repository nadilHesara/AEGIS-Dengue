"""Origin- vs target-relative climate-lag kernels: exact selection, boundaries, causality."""

from __future__ import annotations

import pytest
import torch

from src.models.climate_horizon import ClimateHorizonNB
from src.models.horizon_lag_encoder import HorizonLagEncoder, delay_weights

HISTORY, HORIZONS = 26, (1, 2, 3, 4)


def ramp(steps=37, nodes=2, features=3):
    """x[.., s, ..] = s, so an output value names the time index it read."""

    t = torch.arange(steps, dtype=torch.float32)
    return t[None, :, None, None].expand(1, steps, nodes, features).clone()


def one_hot_encoder(delay: int, mode="target", **kwargs) -> HorizonLagEncoder:
    enc = HorizonLagEncoder(2, 3, HORIZONS, history=HISTORY, mode=mode, **kwargs)
    grid = enc.kernels().shape[-1]
    kernel = torch.zeros(2, 3, grid)
    kernel[..., delay] = 1.0
    enc.kernels = lambda: kernel  # fixed kernel for the exact tests
    return enc


@pytest.mark.parametrize("h, expected_lag", [(1, 7), (2, 6), (4, 4)])
def test_target_relative_delay_8_selects_t_plus_h_minus_8(h, expected_lag):
    enc = one_hot_encoder(8)
    x = ramp()
    t = x.shape[1] - 1                                    # origin = last input week
    out = enc(x)[:, -1]                                   # forecast made at t, [1, N, K, H]
    assert torch.all(out[..., HORIZONS.index(h)] == t - expected_lag)
    _, mass = enc.weights()
    assert torch.all(mass[HORIZONS.index(h)] == 1.0)


def test_origin_relative_uses_the_same_origin_lag_for_every_horizon():
    enc = one_hot_encoder(8, mode="origin")
    x = ramp()
    out = enc(x)[:, -1]
    assert torch.all(out == x.shape[1] - 1 - 8)


def test_delay_shorter_than_horizon_is_unavailable_without_nans():
    enc = one_hot_encoder(2)                              # d = 2: usable for h = 1, 2 only
    out = enc(ramp())[:, -1]
    weights, mass = enc.weights()
    t = 36
    assert torch.all(out[..., 0] == t - 1) and torch.all(out[..., 1] == t)
    assert torch.all(out[..., 2:] == 0) and torch.isfinite(out).all()
    assert torch.all(mass[2:] == 0) and torch.all(weights[2:] == 0)


def test_boundaries_of_the_delay_grid():
    enc = one_hot_encoder(29)                             # max_delay = 25 + 4
    assert enc.max_delay == 29
    out = enc(ramp())[:, -1]
    _, mass = enc.weights()
    # only h = 4 reaches d = 29 (k = 25, the oldest week in the window)
    assert torch.all(out[..., 3] == 36 - 25) and torch.all(mass[3] == 1)
    assert torch.all(out[..., :3] == 0) and torch.all(mass[:3] == 0)
    enc = one_hot_encoder(0)                              # d = 0 is never available
    assert torch.all(enc(ramp()) == 0) and torch.all(enc.weights()[1] == 0)


def test_explicit_max_delay_limits_support_and_renormalises():
    enc = HorizonLagEncoder(2, 3, HORIZONS, history=HISTORY, max_delay=10)
    weights, mass = enc.weights()
    k = torch.arange(HISTORY)
    for i, h in enumerate(HORIZONS):
        assert torch.all(weights[i][..., h + k > 10] == 0)
        assert torch.allclose(weights[i].sum(-1), torch.ones(2, 3))
        assert torch.all((mass[i] > 0) & (mass[i] <= 1 + 1e-6))


def test_common_support_restricts_every_horizon_to_the_same_delays():
    enc = HorizonLagEncoder(2, 3, HORIZONS, history=HISTORY, common_support=True)
    weights, _ = enc.weights()
    k = torch.arange(HISTORY)
    for i, h in enumerate(HORIZONS):
        assert torch.all(weights[i][..., h + k < 4] == 0)
    assert torch.allclose(enc.peak_delays()[0], enc.peak_delays()[3], atol=1e-5)
    with pytest.raises(ValueError):
        HorizonLagEncoder(2, 3, HORIZONS, mode="origin", common_support=True)


def test_empty_support_is_zero_not_nan():
    kernel = torch.zeros(1, 1, 30)
    weights, mass = delay_weights(kernel, HORIZONS, HISTORY, "target")
    assert torch.isfinite(weights).all() and torch.all(weights == 0) and torch.all(mass == 0)


@pytest.mark.parametrize("mode", ["target", "origin"])
def test_data_after_the_origin_cannot_change_the_forecast(mode):
    enc = HorizonLagEncoder(2, 3, HORIZONS, history=HISTORY, mode=mode)
    x = torch.randn(1, 40, 2, 3)
    y = x.clone()
    y[:, 37:] = torch.randn_like(y[:, 37:]) * 100          # weeks after origin index 36
    before, after = enc(x), enc(y)
    assert torch.equal(before[:, : 37 - HISTORY + 1], after[:, : 37 - HISTORY + 1])

    torch.manual_seed(0)
    model = ClimateHorizonNB(2, 4, 3, lag_reach=HISTORY, lag_mode=mode, dropout=0.0).eval()
    with torch.no_grad():
        for layer in (model.climate_delta, model.case_delta):
            layer.weight.normal_(0, 0.3)
    x_case = torch.randn(1, 12, 2, 4)
    anchor = torch.zeros(1, 2)
    first = model(x_case, x[:, :37], anchor)["mu"]
    later = model(x_case, y[:, :37], anchor)["mu"]           # the window ending at t is unchanged
    assert torch.equal(first, later)


def test_modes_match_in_parameterisation_and_count():
    counts = {}
    for mode in ("target", "origin"):
        model = ClimateHorizonNB(25, 9, 7, lag_mode=mode)
        counts[mode] = (sum(p.numel() for p in model.parameters()),
                        sorted((n, tuple(p.shape)) for n, p in model.encoder.named_parameters()))
    assert counts["target"] == counts["origin"]


def test_target_mode_is_the_default_and_reports_mass():
    model = ClimateHorizonNB(25, 9, 7)
    assert model.encoder.mode == "target"
    mass = model.available_mass()
    assert mass.shape == (4, 25, 7) and torch.all((mass > 0) & (mass <= 1 + 1e-6))

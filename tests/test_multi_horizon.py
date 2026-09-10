"""
Tests for multi-horizon windowing and the shared-head model.

The forecast alignment is the thing that must not break. A window that leaked a
target period into its inputs would make every horizon look better than it is,
and the leak would be invisible in the metrics. So the alignment is asserted
directly against `period_id` at every horizon, the way
`tests/test_model_tensors.py` does for the single-horizon windower.

The rest:

  - **Every horizon shares the same origins.** The window count is set by the
    longest horizon so the per-horizon columns are comparable to each other. If
    a horizon were trimmed independently, h=1 and h=4 would be measured on
    different weeks.
  - **The shared model shares its trunk.** One GRU, H heads. A separate GRU per
    horizon would make "shared" H models in a trench coat and the comparison
    against separate runs would measure nothing.
  - **The anchor is horizon-independent.** It is a property of the forecast
    origin, so all horizons of a window use the same one.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.models.multi_horizon import (  # noqa: E402
    DEFAULT_HORIZONS,
    MultiHorizonGCNGRU,
    make_multi_horizon_windows,
    masked_multi_horizon_mse,
)


def _load(name: str, filename: str):
    """Import a numerically-prefixed script by path."""

    path = PROJECT_DIR / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


multi = _load("multi_horizon_module", "training/27.train_multi_horizon.py")
multi.load_modules()
baseline = multi.baseline_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_tensors(n_periods=120, n_nodes=3, n_features=4, seed=0):
    rng = np.random.default_rng(seed)

    return {
        "X": rng.normal(size=(n_periods, n_nodes, n_features)).astype(np.float32),
        "y": rng.gamma(3.0, 5.0, size=(n_periods, n_nodes)).astype(np.float32),
        "y_mask": np.ones((n_periods, n_nodes), dtype=np.int8),
        "period_id": np.arange(1, n_periods + 1, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Alignment — the property that must not break
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("horizons", [(1,), (1, 2), (1, 2, 3, 4), (2, 4)])
def test_targets_land_exactly_h_periods_after_the_origin(horizons):
    """target_period_id[:, i] == origin_period_id + horizons[i], exactly."""

    tensors = make_tensors()
    windows = make_multi_horizon_windows(tensors, lookback=6, horizons=horizons)

    for index, horizon in enumerate(horizons):
        np.testing.assert_array_equal(
            windows["target_period_id"][:, index],
            windows["origin_period_id"] + horizon,
        )


def test_no_input_period_reaches_the_forecast_origin_or_beyond():
    """Inputs end at the origin. Anything later would be leakage."""

    tensors = make_tensors()
    windows = make_multi_horizon_windows(tensors, lookback=6, horizons=(1, 2, 3, 4))

    assert (windows["input_period_id"][:, -1] == windows["origin_period_id"]).all()
    assert (
        windows["input_period_id"] <= windows["origin_period_id"][:, None]
    ).all()


def test_every_target_is_strictly_after_every_input():
    tensors = make_tensors()
    windows = make_multi_horizon_windows(tensors, lookback=6, horizons=(1, 2, 3, 4))

    latest_input = windows["input_period_id"].max(axis=1)

    for index in range(windows["target_period_id"].shape[1]):
        assert (windows["target_period_id"][:, index] > latest_input).all()


def test_the_target_values_match_the_source_series_at_those_periods():
    """Not just the ids -- the actual y values must come from the right rows."""

    tensors = make_tensors()
    horizons = (1, 2, 3, 4)
    windows = make_multi_horizon_windows(tensors, lookback=6, horizons=horizons)

    index_of = {int(p): i for i, p in enumerate(tensors["period_id"])}

    for index in range(len(horizons)):
        expected = np.array(
            [
                tensors["y"][index_of[int(p)]]
                for p in windows["target_period_id"][:, index]
            ]
        )
        np.testing.assert_allclose(windows["y"][..., index], expected)


# ---------------------------------------------------------------------------
# Shared origins
# ---------------------------------------------------------------------------

def test_window_count_is_set_by_the_longest_horizon():
    """So every horizon is scored on identical origins and stays comparable."""

    tensors = make_tensors(n_periods=100)

    short = make_multi_horizon_windows(tensors, lookback=6, horizons=(1,))
    long = make_multi_horizon_windows(tensors, lookback=6, horizons=(1, 4))

    # 100 periods, lookback 6: origins run 5 .. 100-h-1.
    assert len(short["X"]) == 100 - 6 + 1 - 1
    assert len(long["X"]) == 100 - 6 + 1 - 4
    assert len(long["X"]) < len(short["X"])


def test_all_horizons_share_one_set_of_origins():
    tensors = make_tensors()
    windows = make_multi_horizon_windows(tensors, lookback=6, horizons=(1, 2, 3, 4))

    # One origin column, four target columns -- not four independent origin sets.
    assert windows["origin_period_id"].ndim == 1
    assert windows["target_period_id"].shape == (
        len(windows["origin_period_id"]),
        4,
    )


def test_the_h1_column_matches_the_single_horizon_windower_on_shared_origins():
    """The multi-horizon h=1 column must be the single-horizon windower's output.

    Restricted to the origins both keep -- the multi-horizon version stops
    earlier because it needs room for h=4.
    """

    tensors = make_tensors()

    single = baseline.tensors_module.make_windows(
        tensors, lookback=6, horizon=1, drop_incomplete=False
    )
    multi_windows = make_multi_horizon_windows(
        tensors, lookback=6, horizons=(1, 2, 3, 4)
    )

    shared = len(multi_windows["X"])

    np.testing.assert_allclose(single["X"][:shared], multi_windows["X"])
    np.testing.assert_allclose(single["y"][:shared], multi_windows["y"][..., 0])


# ---------------------------------------------------------------------------
# Shapes and validation
# ---------------------------------------------------------------------------

def test_targets_and_masks_carry_a_horizon_axis():
    tensors = make_tensors(n_nodes=3)
    windows = make_multi_horizon_windows(tensors, lookback=6, horizons=(1, 2, 3, 4))

    n = len(windows["X"])
    assert windows["y"].shape == (n, 3, 4)
    assert windows["y_mask"].shape == (n, 3, 4)


def test_a_masked_cell_stays_masked_at_the_right_horizon():
    """Per-node, per-horizon masking -- one absent record must not drop a window."""

    tensors = make_tensors(n_periods=60, n_nodes=3)
    tensors["y_mask"][30, 1] = 0

    windows = make_multi_horizon_windows(tensors, lookback=6, horizons=(1, 2))

    for index, _ in enumerate((1, 2)):
        hit = windows["target_period_id"][:, index] == tensors["period_id"][30]
        if hit.any():
            assert (windows["y_mask"][hit, 1, index] == 0).all()
            # The other nodes in that window are untouched.
            assert (windows["y_mask"][hit, 0, index] == 1).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lookback": 0, "horizons": (1,)},
        {"lookback": 6, "horizons": ()},
        {"lookback": 6, "horizons": (0,)},
        {"lookback": 6, "horizons": (1, -2)},
    ],
)
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make_multi_horizon_windows(make_tensors(), **kwargs)


def test_a_horizon_longer_than_the_series_is_rejected():
    with pytest.raises(ValueError):
        make_multi_horizon_windows(
            make_tensors(n_periods=20), lookback=15, horizons=(1, 10)
        )


# ---------------------------------------------------------------------------
# The shared model
# ---------------------------------------------------------------------------

def make_model(n_horizons=4, n_features=4, hidden=8):
    backbone = baseline.GCNGRU(
        n_features=n_features, hidden=hidden, gcn_layers=1, horizon=1, dropout=0.0
    )
    return MultiHorizonGCNGRU(backbone, n_horizons=n_horizons)


def test_output_has_one_column_per_horizon():
    model = make_model(n_horizons=4)

    output = model(torch.randn(5, 12, 25, 4), torch.eye(25))

    assert output.shape == (5, 25, 4)
    assert torch.isfinite(output).all()


def test_the_trunk_is_shared_and_only_the_heads_are_per_horizon():
    """One GRU for all horizons. Otherwise 'shared' is four models in disguise."""

    model = make_model(n_horizons=4)

    assert isinstance(model.backbone.gru, torch.nn.GRU)
    assert len(model.heads) == 4

    gru_parameters = list(model.backbone.gru.parameters())
    assert len(gru_parameters) > 0

    # The heads are the only per-horizon parameters, and they are small.
    head_parameters = sum(p.numel() for p in model.heads.parameters())
    assert head_parameters == 4 * (8 + 1)


def test_the_heads_are_independent():
    """Different horizons must be able to predict different things."""

    model = make_model(n_horizons=4)
    model.eval()

    with torch.no_grad():
        model.heads[0].bias.fill_(5.0)
        model.heads[1].bias.fill_(-5.0)
        output = model(torch.randn(2, 12, 25, 4), torch.eye(25))

    assert (output[..., 0] > output[..., 1]).all()


def test_the_bypassed_backbone_head_is_not_a_trainable_parameter():
    """An unused head would inflate the reported parameter count."""

    model = make_model(n_horizons=4)

    assert isinstance(model.backbone.head, torch.nn.Identity)
    assert sum(p.numel() for p in model.backbone.head.parameters()) == 0


def test_the_model_is_cheaper_than_separate_models():
    """The claim `shared` rests on: one trunk, not H."""

    shared = make_model(n_horizons=4, hidden=8)
    single = baseline.GCNGRU(
        n_features=4, hidden=8, gcn_layers=1, horizon=1, dropout=0.0
    )

    shared_count = sum(p.numel() for p in shared.parameters())
    separate_count = 4 * sum(p.numel() for p in single.parameters())

    assert shared_count < separate_count


def test_a_zero_horizon_model_is_rejected():
    backbone = baseline.GCNGRU(n_features=4, hidden=8, gcn_layers=1, horizon=1)

    with pytest.raises(ValueError):
        MultiHorizonGCNGRU(backbone, n_horizons=0)


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------

def test_masked_cells_cannot_change_the_loss():
    """A missing target is never imputed anywhere in this pipeline."""

    generator = torch.Generator().manual_seed(0)
    prediction = torch.randn(4, 25, 4, generator=generator)
    target = torch.randn(4, 25, 4, generator=generator)
    mask = (torch.rand(4, 25, 4, generator=generator) > 0.3).float()

    before = masked_multi_horizon_mse(prediction, target, mask)

    corrupted = target.clone()
    corrupted[mask == 0] = 1e6

    after = masked_multi_horizon_mse(prediction, corrupted, mask)

    assert torch.allclose(before, after, atol=1e-6)


def test_the_loss_reduces_to_the_baseline_at_one_horizon():
    """With one horizon it must be script 16's masked_mse exactly."""

    generator = torch.Generator().manual_seed(0)
    prediction = torch.randn(4, 25, 1, generator=generator)
    target = torch.randn(4, 25, 1, generator=generator)
    mask = (torch.rand(4, 25, 1, generator=generator) > 0.3).float()

    ours = masked_multi_horizon_mse(prediction, target, mask)
    theirs = baseline.masked_mse(prediction, target, mask)

    assert torch.allclose(ours, theirs, atol=1e-7)


def test_every_horizon_contributes_to_the_loss():
    """Equal weighting -- a change at any horizon must move the total."""

    prediction = torch.zeros(2, 25, 4)
    target = torch.zeros(2, 25, 4)
    mask = torch.ones(2, 25, 4)

    base = masked_multi_horizon_mse(prediction, target, mask)
    assert base.item() == pytest.approx(0.0)

    for index in range(4):
        moved = target.clone()
        moved[..., index] = 1.0
        assert masked_multi_horizon_mse(prediction, moved, mask).item() > 0


def test_a_fully_masked_batch_is_finite():
    prediction = torch.randn(2, 25, 4)
    target = torch.randn(2, 25, 4)

    loss = masked_multi_horizon_mse(prediction, target, torch.zeros(2, 25, 4))

    assert torch.isfinite(loss)


def test_the_loss_produces_gradients_for_every_head():
    model = make_model(n_horizons=4)

    prediction = model(torch.randn(3, 12, 25, 4), torch.eye(25))
    masked_multi_horizon_mse(
        prediction, torch.randn(3, 25, 4), torch.ones(3, 25, 4)
    ).backward()

    for index, head in enumerate(model.heads):
        assert head.weight.grad is not None, index
        assert head.weight.grad.abs().sum() > 0, index


# ---------------------------------------------------------------------------
# The target parameterisation
# ---------------------------------------------------------------------------

def test_the_anchor_broadcasts_across_horizons():
    """One anchor per window-node, shared by every horizon of that window."""

    y = torch.rand(3, 25, 4) * 100
    anchor = torch.log1p(torch.rand(3, 25) * 100)

    residual = multi.to_multi_target(y, anchor, "residual")

    assert residual.shape == (3, 25, 4)
    for index in range(4):
        torch.testing.assert_close(
            residual[..., index], torch.log1p(y[..., index]) - anchor
        )


def test_the_residual_round_trip_is_exact():
    """expm1(log1p(y_t) + d) must return the original counts at every horizon."""

    y = torch.rand(3, 25, 4) * 100
    anchor = torch.log1p(torch.rand(3, 25) * 100)

    residual = multi.to_multi_target(y, anchor, "residual")
    recovered = torch.expm1(residual + anchor.unsqueeze(-1))

    torch.testing.assert_close(recovered, y, rtol=1e-4, atol=1e-4)


def test_a_zero_residual_reproduces_persistence_at_every_horizon():
    """The parameterisation's defining property, which longer horizons preserve."""

    anchor = torch.log1p(torch.tensor([[10.0, 200.0]]))
    zero = torch.zeros(1, 2, 4)

    recovered = torch.expm1(zero + anchor.unsqueeze(-1))

    for index in range(4):
        torch.testing.assert_close(
            recovered[..., index], torch.tensor([[10.0, 200.0]]), rtol=1e-4, atol=1e-4
        )


def test_an_unknown_target_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown target mode"):
        multi.to_multi_target(torch.rand(2, 3, 4), torch.rand(2, 3), "nonsense")


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_default_horizons_are_one_through_four():
    assert DEFAULT_HORIZONS == (1, 2, 3, 4)


def test_both_arms_are_declared():
    assert set(multi.ARMS) == {"separate", "shared"}

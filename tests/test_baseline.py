"""
Tests for the metrics, the naive baselines and the GCN+GRU model.

Two properties carry most of the weight here.

The metrics must respect the mask. An unobserved district-period that quietly
contributes a zero to the error would make every model look better than it is,
and would make the improvement over persistence unfalsifiable.

The model must not be able to see the future. The window alignment is already
tested in test_model_tensors, so what is checked here is the rest of the path:
that the anchor comes from the forecast origin, that the residual round trip is
exact, and that perturbing a period after the target changes no prediction.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


PROJECT_DIR = Path(__file__).resolve().parent.parent

FOLDS_PATH = PROJECT_DIR / "data" / "processed" / "folds.json"
TENSOR_PATH = PROJECT_DIR / "data" / "processed" / "model_tensors_v0.npz"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, PROJECT_DIR / "scripts" / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


naive = load("naive_baselines", "evaluation/15.evaluate_naive_baselines.py")
trainer = load("train_gcn_gru", "training/16.train_gcn_gru.py")
trainer.load_modules()


requires_real_data = pytest.mark.skipif(
    not (FOLDS_PATH.exists() and TENSOR_PATH.exists()),
    reason="Run scripts 12 and 14 first.",
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_masked_metrics_ignore_unobserved_cells():
    target = np.array([[10.0, 20.0], [30.0, 40.0]])
    prediction = np.array([[10.0, 999.0], [30.0, 40.0]])
    mask = np.array([[1, 0], [1, 1]], dtype=np.int8)

    assert naive.masked_mae(prediction, target, mask) == pytest.approx(0.0)
    assert naive.masked_rmse(prediction, target, mask) == pytest.approx(0.0)


def test_masked_metrics_match_the_unmasked_case():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(20, 5))
    prediction = target + rng.normal(size=(20, 5))
    mask = np.ones((20, 5), dtype=np.int8)

    assert naive.masked_mae(prediction, target, mask) == pytest.approx(
        np.abs(prediction - target).mean()
    )
    assert naive.masked_rmse(prediction, target, mask) == pytest.approx(
        np.sqrt(((prediction - target) ** 2).mean())
    )


def test_an_entirely_masked_split_is_nan_not_zero():
    target = np.ones((3, 2))
    mask = np.zeros((3, 2), dtype=np.int8)

    assert np.isnan(naive.masked_mae(target, target, mask))


def test_peak_thresholds_come_from_history_only():
    y = np.zeros((100, 2))
    y[:50] = 10.0
    y[50:] = 1000.0  # the "test" half, which must not move the threshold

    mask = np.ones_like(y, dtype=np.int8)
    fit_mask = np.arange(100) < 50

    thresholds = naive.peak_thresholds(y, mask, fit_mask)

    np.testing.assert_allclose(thresholds, [10.0, 10.0])


def test_peak_mae_covers_only_cells_above_the_threshold():
    target = np.array([[5.0, 100.0], [200.0, 5.0]])
    prediction = np.array([[0.0, 100.0], [200.0, 0.0]])
    mask = np.ones_like(target, dtype=np.int8)
    thresholds = np.array([50.0, 50.0])

    error, cells = naive.peak_mae(prediction, target, mask, thresholds)

    # only the two cells at 100 and 200 qualify, and both are predicted exactly
    assert cells == 2
    assert error == pytest.approx(0.0)


def test_per_district_mae_splits_by_node():
    target = np.array([[10.0, 10.0], [10.0, 10.0]])
    prediction = np.array([[10.0, 20.0], [10.0, 20.0]])
    mask = np.ones_like(target, dtype=np.int8)

    frame = naive.per_district_mae(prediction, target, mask, ["A", "B"])

    assert frame.loc[frame["canonical_name"] == "A", "mae"].iloc[0] == 0.0
    assert frame.loc[frame["canonical_name"] == "B", "mae"].iloc[0] == 10.0


# ---------------------------------------------------------------------------
# Naive forecasts
# ---------------------------------------------------------------------------

def make_tensors(n_periods=120, n_nodes=3, seed=0):
    rng = np.random.default_rng(seed)

    return {
        "y": rng.gamma(3.0, 5.0, size=(n_periods, n_nodes)).astype(np.float32),
        "y_mask": np.ones((n_periods, n_nodes), dtype=np.int8),
        "period_id": np.arange(1, n_periods + 1, dtype=np.int32),
        "X": rng.normal(size=(n_periods, n_nodes, 4)).astype(np.float32),
        "feature_names": np.array(["a", "b", "c", "d"], dtype=object),
    }


def test_persistence_copies_the_origin_period():
    tensors = make_tensors()
    windows = trainer.tensors_module.make_windows(
        tensors, lookback=4, horizon=1, drop_incomplete=False
    )
    fit_mask = tensors["period_id"] <= 80

    predictions = naive.build_naive_predictions(tensors, windows, fit_mask)

    index_of = {int(p): i for i, p in enumerate(tensors["period_id"])}
    expected = np.array(
        [tensors["y"][index_of[int(p)]] for p in windows["origin_period_id"]]
    )

    np.testing.assert_allclose(predictions["persistence"], expected)


def test_seasonal_naive_reads_one_year_before_the_target():
    tensors = make_tensors()
    windows = trainer.tensors_module.make_windows(
        tensors, lookback=4, horizon=1, drop_incomplete=False
    )
    fit_mask = tensors["period_id"] <= 80

    predictions = naive.build_naive_predictions(tensors, windows, fit_mask)

    index_of = {int(p): i for i, p in enumerate(tensors["period_id"])}
    late = windows["target_period_id"] > naive.SEASONAL_LAG

    expected = np.array(
        [
            tensors["y"][index_of[int(p) - naive.SEASONAL_LAG]]
            for p in windows["target_period_id"][late]
        ]
    )

    np.testing.assert_allclose(predictions["seasonal_naive"][late], expected)


def test_a_forecast_never_reads_its_own_target_period():
    """Corrupting the period being predicted must not change that prediction.

    Checked one window at a time. Corrupting every target period at once would
    prove nothing: at horizon 1 the target of one window is the origin of the
    next, which persistence is entitled to read.
    """

    tensors = make_tensors()
    windows = trainer.tensors_module.make_windows(
        tensors, lookback=4, horizon=1, drop_incomplete=False
    )
    fit_mask = tensors["period_id"] <= 80
    index_of = {int(p): i for i, p in enumerate(tensors["period_id"])}

    clean = naive.build_naive_predictions(tensors, windows, fit_mask)

    # Probe only periods after the fitting range. Corrupting a period inside it
    # would also move the fitted fallback mean, which is a different mechanism
    # and a legitimate one.
    probes = np.where(windows["target_period_id"] > 85)[0]

    for sample in probes[:: max(1, len(probes) // 4)]:
        corrupted = {key: value.copy() for key, value in tensors.items()}
        target = int(windows["target_period_id"][sample])
        corrupted["y"][index_of[target]] = 1e6

        dirty = naive.build_naive_predictions(corrupted, windows, fit_mask)

        for model in ("persistence", "seasonal_naive"):
            np.testing.assert_allclose(
                clean[model][sample], dirty[model][sample], err_msg=model
            )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def test_graph_convolution_mixes_only_along_the_node_axis():
    layer = trainer.GraphConv(4, 4)
    x = torch.randn(2, 3, 5, 4)

    identity = torch.eye(5)
    mixed = torch.ones(5, 5) / 5.0

    with torch.no_grad():
        unmixed_out = layer(x, identity)
        mixed_out = layer(x, mixed)

    # under the identity a node sees only itself
    with torch.no_grad():
        np.testing.assert_allclose(
            unmixed_out.numpy(), layer.linear(x).numpy(), rtol=1e-5
        )

    # under a uniform adjacency every node sees the same average
    first = mixed_out[:, :, 0]
    for node in range(1, 5):
        np.testing.assert_allclose(
            first.numpy(), mixed_out[:, :, node].numpy(), rtol=1e-5
        )


def test_model_output_shape():
    model = trainer.GCNGRU(n_features=7, hidden=8, gcn_layers=2, horizon=1)
    x = torch.randn(4, 12, 25, 7)
    adjacency = torch.eye(25)

    assert model(x, adjacency).shape == (4, 25, 1)


def test_model_is_permutation_consistent_with_its_adjacency():
    """Relabelling the districts and the graph together must not change output."""

    torch.manual_seed(0)
    model = trainer.GCNGRU(n_features=3, hidden=8, gcn_layers=1, horizon=1)
    model.eval()

    x = torch.randn(2, 5, 6, 3)
    adjacency = torch.rand(6, 6)
    adjacency = (adjacency + adjacency.T) / 2

    order = torch.tensor([3, 1, 5, 0, 4, 2])

    with torch.no_grad():
        base = model(x, adjacency)
        permuted = model(x[:, :, order], adjacency[order][:, order])

    np.testing.assert_allclose(
        base[:, order].numpy(), permuted.numpy(), rtol=1e-4, atol=1e-5
    )


def test_masked_loss_ignores_masked_cells():
    prediction = torch.tensor([[[1.0], [99.0]]])
    target = torch.tensor([[[1.0], [0.0]]])
    mask = torch.tensor([[[1.0], [0.0]]])

    assert trainer.masked_mse(prediction, target, mask).item() == pytest.approx(0.0)


def test_masked_loss_averages_over_observed_cells_only():
    prediction = torch.tensor([[[3.0], [99.0]]])
    target = torch.tensor([[[1.0], [0.0]]])
    mask = torch.tensor([[[1.0], [0.0]]])

    # one observed cell with an error of 2, so the mean squared error is 4
    assert trainer.masked_mse(prediction, target, mask).item() == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Target parameterisation
# ---------------------------------------------------------------------------

def test_residual_target_round_trips_exactly():
    y = torch.tensor([[0.0, 5.0, 100.0]])
    anchor = torch.log1p(torch.tensor([[2.0, 7.0, 90.0]]))

    encoded = trainer.to_target(y, anchor, "residual")

    split = {"X": np.zeros((1, 1, 3, 1), dtype=np.float32), "anchor": anchor.numpy()}

    class Echo(torch.nn.Module):
        def forward(self, x, adjacency):
            return encoded

    recovered = trainer.predict(
        Echo(), split, np.eye(3, dtype=np.float32), "residual", torch.device("cpu")
    )

    np.testing.assert_allclose(recovered, y.numpy(), rtol=1e-5)


def test_a_zero_residual_prediction_reproduces_persistence():
    """The model's zero point is the persistence forecast, by construction."""

    anchor = np.log1p(np.array([[4.0, 30.0, 700.0]], dtype=np.float32))
    split = {"X": np.zeros((1, 1, 3, 1), dtype=np.float32), "anchor": anchor}

    class Zero(torch.nn.Module):
        def forward(self, x, adjacency):
            return torch.zeros(1, 3, 1)

    prediction = trainer.predict(
        Zero(), split, np.eye(3, dtype=np.float32), "residual", torch.device("cpu")
    )

    np.testing.assert_allclose(prediction, [[4.0, 30.0, 700.0]], rtol=1e-4)


def test_direct_target_is_plain_log1p():
    y = torch.tensor([[0.0, 5.0, 100.0]])
    anchor = torch.zeros_like(y)

    encoded = trainer.to_target(y, anchor, "direct")

    np.testing.assert_allclose(
        encoded.squeeze(-1).numpy(), np.log1p(y.numpy()), rtol=1e-6
    )


def test_predictions_are_never_negative():
    split = {
        "X": np.zeros((2, 1, 3, 1), dtype=np.float32),
        "anchor": np.zeros((2, 3), dtype=np.float32),
    }

    class VeryNegative(torch.nn.Module):
        def forward(self, x, adjacency):
            return torch.full((2, 3, 1), -50.0)

    prediction = trainer.predict(
        VeryNegative(), split, np.eye(3, dtype=np.float32), "residual",
        torch.device("cpu"),
    )

    assert (prediction >= 0).all()


def test_unknown_target_mode_is_rejected():
    with pytest.raises(ValueError):
        trainer.to_target(torch.zeros(1, 2), torch.zeros(1, 2), "nonsense")


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

@requires_real_data
def test_the_anchor_is_the_case_count_at_the_forecast_origin():
    import json

    tensors = trainer.folds_module.load_tensors("v0")
    fold = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))["folds"][0]

    windows = trainer.tensors_module.make_windows(
        tensors, lookback=12, horizon=1, drop_incomplete=False
    )
    anchor = trainer.build_anchor(tensors, windows, fold)

    index_of = {int(p): i for i, p in enumerate(tensors["period_id"])}

    for sample in (0, 100, len(anchor) - 1):
        origin = int(windows["origin_period_id"][sample])
        observed = tensors["y"][index_of[origin]]
        finite = ~np.isnan(observed)

        np.testing.assert_allclose(
            anchor[sample][finite], np.log1p(observed[finite]), rtol=1e-5
        )

        # and it is strictly before the target
        assert origin < int(windows["target_period_id"][sample])


@requires_real_data
def test_fold_arrays_are_finite_and_split_by_target():
    import json

    tensors = trainer.folds_module.load_tensors("v0")
    calendar = trainer.folds_module.load_calendar()
    months = calendar.sort_values("period_id")["month"].to_numpy()
    fold = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))["folds"][0]

    arrays = trainer.build_fold_arrays(tensors, months, fold, 12, 1)

    for split in ("train", "val", "test"):
        assert np.isfinite(arrays[split]["X"]).all()
        assert np.isfinite(arrays[split]["y"]).all()
        assert np.isfinite(arrays[split]["anchor"]).all()

    assert arrays["train"]["target_period_id"].max() <= fold["train_end_period"]
    assert arrays["test"]["target_period_id"].min() >= fold["test_start_period"]
    assert arrays["test"]["target_period_id"].max() <= fold["test_end_period"]

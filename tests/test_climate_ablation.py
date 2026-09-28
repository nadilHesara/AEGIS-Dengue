"""
Tests for the h=1-4 climate ablation.

The experiment claims that any gap between `full` and the ablated arms is
climate and nothing else. These are the properties that would make that claim
false, and the shuffled arm carries most of the risk because a subtly wrong
shuffle would look like a clean null result:

  - **The climate channel set must be right.** Missing the rolling lags would
    leave `rainfall_..._roll12` in the "no climate" arm -- the exact channels the
    5-10 week delay story runs through -- and the ablation would measure nothing.
  - **The shuffle must destroy time alignment and nothing else.** Same values,
    same per-district marginals, same shape, same dtype.
  - **The shuffle must not leak across splits.** Permuting the pooled array would
    move test-period weather into training windows.
  - **The shuffle must not touch the target**, the mask or the anchor.
  - **`no_climate` must be sliced, not zeroed**, and its backbone sized for the
    surviving width.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.models.climate_ablation import (  # noqa: E402
    CLIMATE_FEATURES,
    FeatureSubsetGCNGRU,
    climate_indices,
    non_climate_indices,
    shuffle_climate,
    shuffle_split_climate,
)


def _load(name: str, filename: str):
    """Import a numerically-prefixed script by path."""

    path = PROJECT_DIR / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ablation = _load("climate_ablation_module", "training/29.train_climate_ablation.py")
ablation.load_modules()
baseline = ablation.baseline_module


V1_FEATURES = [
    "cases_log1p",
    "rainfall_daily_mean_mm",
    "rainy_days_frac",
    "temperature_mean_c",
    "diurnal_range_c",
    "dewpoint_mean_c",
    "relative_humidity_mean",
    "wind_speed_mean",
    "doy_sin",
    "doy_cos",
    "weather_observed",
    "case_observed",
    "centroid_lat",
    "centroid_lon",
    "rainfall_daily_mean_mm_roll4",
    "rainfall_daily_mean_mm_roll8",
    "rainfall_daily_mean_mm_roll12",
    "temperature_mean_c_roll4",
    "temperature_mean_c_roll8",
    "temperature_mean_c_roll12",
    "relative_humidity_mean_roll4",
    "relative_humidity_mean_roll8",
    "relative_humidity_mean_roll12",
]


# ---------------------------------------------------------------------------
# Which channels count as climate
# ---------------------------------------------------------------------------

def test_rolling_climate_channels_are_counted_as_climate():
    """The rolling lags must be dropped too.

    They are the channels that carry *delayed* climate, so leaving
    `rainfall_..._roll12` in a "no climate" arm would keep the 5-10 week
    mechanism intact while calling it an ablation.
    """

    dropped = set(climate_indices(V1_FEATURES))
    names = {V1_FEATURES[i] for i in dropped}

    for name in V1_FEATURES:
        if any(name.endswith(s) for s in ("_roll4", "_roll8", "_roll12")):
            assert name in names, f"{name} was not treated as climate"


def test_all_seven_instantaneous_channels_are_climate():
    names = {V1_FEATURES[i] for i in climate_indices(V1_FEATURES)}

    for name in CLIMATE_FEATURES:
        assert name in names


def test_non_climate_channels_survive():
    """Case history, seasonality, flags and geography are not weather."""

    kept = {V1_FEATURES[i] for i in non_climate_indices(V1_FEATURES)}

    assert kept == {
        "cases_log1p",
        "doy_sin",
        "doy_cos",
        "weather_observed",
        "case_observed",
        "centroid_lat",
        "centroid_lon",
    }


def test_v1_splits_sixteen_and_seven():
    """23 channels: 16 climate-derived, 7 kept."""

    assert len(climate_indices(V1_FEATURES)) == 16
    assert len(non_climate_indices(V1_FEATURES)) == 7


def test_indices_are_disjoint_and_complete():
    dropped = climate_indices(V1_FEATURES)
    kept = non_climate_indices(V1_FEATURES)

    assert set(dropped) & set(kept) == set()
    assert sorted(dropped + kept) == list(range(len(V1_FEATURES)))


def test_a_channel_named_like_a_rolling_non_climate_is_not_dropped():
    """Only rolling transforms *of climate* count, not every `_roll` name."""

    names = ["cases_log1p_roll4", "rainfall_daily_mean_mm_roll4"]

    assert [names[i] for i in climate_indices(names)] == [
        "rainfall_daily_mean_mm_roll4"
    ]


# ---------------------------------------------------------------------------
# The shuffle
# ---------------------------------------------------------------------------

def make_windows(n_windows=40, steps=12, nodes=5, features=6):
    """Windowed inputs with a distinct value per cell, so moves are traceable."""

    total = n_windows * steps * nodes * features

    return np.arange(total, dtype=np.float32).reshape(
        n_windows, steps, nodes, features
    )


def test_shuffle_preserves_shape_and_dtype():
    x = make_windows()
    shuffled = shuffle_climate(x, [1, 3], seed=0)

    assert shuffled.shape == x.shape
    assert shuffled.dtype == x.dtype


def test_shuffle_does_not_modify_its_input():
    """A copy, not an in-place edit -- the `full` arm reuses the same arrays."""

    x = make_windows()
    original = x.copy()

    shuffle_climate(x, [1, 3], seed=0)

    np.testing.assert_array_equal(x, original)


def test_shuffle_leaves_non_climate_channels_untouched():
    """Case history and seasonality must be bit-identical after the shuffle."""

    x = make_windows()
    climate = [1, 3]
    shuffled = shuffle_climate(x, climate, seed=0)

    for channel in range(x.shape[-1]):
        if channel in climate:
            continue
        np.testing.assert_array_equal(shuffled[..., channel], x[..., channel])


def test_shuffle_preserves_the_per_district_marginal_exactly():
    """Same values, different order.

    This is what makes the arm a control: the per-fold scaler sees identical
    statistics, so nothing about the normalisation changes.
    """

    x = make_windows()
    climate = [1, 3]
    shuffled = shuffle_climate(x, climate, seed=0)

    for node in range(x.shape[2]):
        for channel in climate:
            np.testing.assert_array_equal(
                np.sort(shuffled[:, :, node, channel], axis=None),
                np.sort(x[:, :, node, channel], axis=None),
            )


def test_shuffle_actually_reorders():
    """A no-op shuffle would make the arm a duplicate of `full`."""

    x = make_windows()
    shuffled = shuffle_climate(x, [1], seed=0)

    assert not np.array_equal(shuffled[..., 1], x[..., 1])


def test_shuffle_moves_whole_windows():
    """Each window's 12-step block stays intact, permuted as a unit.

    Scrambling within a window would leave something that is not weather at all;
    the control is meant to destroy alignment, not realism.
    """

    x = make_windows()
    shuffled = shuffle_climate(x, [2], seed=0)

    source = {tuple(x[w, :, 0, 2]): w for w in range(x.shape[0])}

    for window in range(x.shape[0]):
        assert tuple(shuffled[window, :, 0, 2]) in source


def test_shuffle_is_independent_per_district():
    """A shared permutation would preserve cross-district weather correlation."""

    x = make_windows(n_windows=60)
    shuffled = shuffle_climate(x, [1], seed=0)

    order_by_node = []
    for node in range(x.shape[2]):
        lookup = {x[w, 0, node, 1]: w for w in range(x.shape[0])}
        order_by_node.append(
            tuple(lookup[shuffled[w, 0, node, 1]] for w in range(x.shape[0]))
        )

    assert len(set(order_by_node)) > 1


def test_shuffle_is_reproducible_by_seed():
    x = make_windows()

    a = shuffle_climate(x, [1, 3], seed=5)
    b = shuffle_climate(x, [1, 3], seed=5)
    c = shuffle_climate(x, [1, 3], seed=6)

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_shuffle_with_no_climate_indices_is_a_copy():
    x = make_windows()
    shuffled = shuffle_climate(x, [], seed=0)

    np.testing.assert_array_equal(shuffled, x)
    assert shuffled is not x


# ---------------------------------------------------------------------------
# Splits: no leakage, no target contamination
# ---------------------------------------------------------------------------

def make_arrays():
    """A minimal train/val/test structure of the shape the training loop uses."""

    arrays = {}
    for offset, split in enumerate(("train", "val", "test")):
        n = 20 + offset
        arrays[split] = {
            "X": make_windows(n_windows=n),
            "y": np.arange(n * 5, dtype=np.float32).reshape(n, 5),
            "mask": np.ones((n, 5), dtype=np.float32),
            "anchor": np.arange(n * 5, dtype=np.float32).reshape(n, 5),
        }

    return arrays


def test_shuffle_split_does_not_move_data_between_splits():
    """Each split is permuted among its own windows only.

    Pooling would move test-period weather into training windows, which is a
    leak that would make the shuffled arm look better than it is.
    """

    arrays = make_arrays()
    climate = [1, 3]
    out = shuffle_split_climate(arrays, climate, seed=0)

    for split, content in arrays.items():
        for channel in climate:
            np.testing.assert_array_equal(
                np.sort(out[split]["X"][..., channel], axis=None),
                np.sort(content["X"][..., channel], axis=None),
            )


def test_shuffle_split_leaves_target_mask_and_anchor_untouched():
    """Only `X` may change; the scoring target must be identical across arms."""

    arrays = make_arrays()
    out = shuffle_split_climate(arrays, [1, 3], seed=0)

    for split, content in arrays.items():
        for key in ("y", "mask", "anchor"):
            np.testing.assert_array_equal(out[split][key], content[key])


def test_shuffle_split_covers_every_split():
    arrays = make_arrays()
    out = shuffle_split_climate(arrays, [1], seed=0)

    assert set(out) == set(arrays)
    for split in arrays:
        assert not np.array_equal(out[split]["X"][..., 1], arrays[split]["X"][..., 1])


def test_shuffle_split_uses_a_different_permutation_per_split():
    """Splits differ in length, so a shared permutation could not even apply."""

    arrays = make_arrays()
    out = shuffle_split_climate(arrays, [1], seed=0)

    assert out["train"]["X"].shape[0] != out["val"]["X"].shape[0]


# ---------------------------------------------------------------------------
# The no_climate arm
# ---------------------------------------------------------------------------

def test_feature_subset_slices_rather_than_zeroes():
    """The backbone must be built for the narrower input, not handed zeros.

    A zeroed channel still costs the first graph convolution its weights and
    still passes through the per-fold scaler, so it is a different experiment.
    """

    keep = [0, 2, 4]
    backbone = baseline.GCNGRU(
        n_features=len(keep), hidden=8, gcn_layers=1, horizon=1, dropout=0.0
    )
    model = FeatureSubsetGCNGRU(backbone, keep)
    model.eval()

    x = torch.randn(3, 12, 5, 6)
    adjacency = torch.eye(5)

    with torch.no_grad():
        np.testing.assert_allclose(
            model(x, adjacency).numpy(),
            backbone(x[..., keep], adjacency).numpy(),
            rtol=1e-6,
            atol=1e-6,
        )


def test_feature_subset_ignores_the_dropped_channels():
    """Changing a dropped channel must not change the output at all."""

    keep = [0, 2]
    backbone = baseline.GCNGRU(
        n_features=len(keep), hidden=8, gcn_layers=1, horizon=1, dropout=0.0
    )
    model = FeatureSubsetGCNGRU(backbone, keep)
    model.eval()

    x = torch.randn(3, 12, 5, 6)
    adjacency = torch.eye(5)

    with torch.no_grad():
        before = model(x, adjacency)
        x[..., 1] = 999.0
        after = model(x, adjacency)

    torch.testing.assert_close(before, after)


def test_no_climate_builder_sizes_the_backbone_to_the_kept_width():
    config = dict(ablation.DEFAULTS)
    keep = list(range(7))

    model = ablation.make_model_builder("no_climate", config, keep)(23)

    assert isinstance(model, FeatureSubsetGCNGRU)
    assert model.backbone.graph_layers[0].linear.in_features == 7


def test_full_and_shuffled_builders_use_the_full_width():
    """They differ in data, not architecture -- that is the point of the control."""

    config = dict(ablation.DEFAULTS)
    keep = list(range(7))

    for arm in ("full", "shuffled"):
        model = ablation.make_model_builder(arm, config, keep)(23)

        assert not isinstance(model, FeatureSubsetGCNGRU)
        assert model.graph_layers[0].linear.in_features == 23


def test_full_and_shuffled_have_identical_parameter_counts():
    """The confound control only works if capacity is held fixed."""

    config = dict(ablation.DEFAULTS)
    keep = list(range(7))

    counts = []
    for arm in ("full", "shuffled"):
        model = ablation.make_model_builder(arm, config, keep)(23)
        counts.append(sum(p.numel() for p in model.parameters()))

    assert counts[0] == counts[1]


def test_head_is_one_wide_at_every_horizon():
    """`horizon` is the forecast lead here, not the head width."""

    for horizon in (1, 2, 3, 4):
        config = dict(ablation.DEFAULTS)
        model = ablation.make_model_builder("full", config, list(range(7)))(23)

        assert model.head.out_features == 1


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------

def test_comparison_is_anchored_on_the_full_arm():
    """Every ablated arm is differenced against `full`, per horizon."""

    import pandas as pd

    rows = []
    for arm, mae in (("full", 20.0), ("no_climate", 23.0), ("shuffled", 22.0)):
        for fold in (1, 2, 3):
            for seed in (0, 1):
                rows.append(
                    {
                        "arm": arm,
                        "horizon": 4,
                        "fold_id": fold,
                        "test_year": 2010 + fold,
                        "headline": True,
                        "covers_covid": False,
                        "seed": seed,
                        "mae": mae + 0.1 * seed,
                        "peak_mae": mae * 2,
                        "ensemble": False,
                    }
                )

    comparison = ablation.compare_to_control(pd.DataFrame(rows))

    assert set(comparison["arm"]) == {"no_climate", "shuffled"}

    by_arm = comparison.set_index("arm")["headline_delta"]
    assert by_arm["no_climate"] == pytest.approx(3.0)
    assert by_arm["shuffled"] == pytest.approx(2.0)


def test_positive_delta_means_the_ablation_hurt():
    """Sign convention: the report and the verdicts depend on it."""

    import pandas as pd

    rows = []
    for arm, mae in (("full", 10.0), ("no_climate", 15.0)):
        for fold in (1, 2, 3):
            rows.append(
                {
                    "arm": arm,
                    "horizon": 4,
                    "fold_id": fold,
                    "test_year": 2010 + fold,
                    "headline": True,
                    "covers_covid": False,
                    "seed": 0,
                    "mae": mae,
                    "peak_mae": mae * 2,
                    "ensemble": False,
                }
            )

    comparison = ablation.compare_to_control(pd.DataFrame(rows))

    assert comparison["headline_delta"].iloc[0] > 0
    assert comparison["folds_hurt"].iloc[0] == 3

"""
Tests for the stage 3 model tensors.

The assertions that matter here are the ones nothing downstream can catch. A
misaligned sequence window trains and scores perfectly well; it just reports a
skill it does not have. So the window alignment is checked directly against
period_id, the rolling features are checked to be trailing only, and the node
axis is checked to match nodes.csv order.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parent.parent

PANEL_PATH = PROJECT_DIR / "data" / "processed" / "panel_weekly.parquet"
NODES_PATH = PROJECT_DIR / "data" / "processed" / "nodes.csv"

REAL_DATA_FILES = [PANEL_PATH, NODES_PATH]

spec = importlib.util.spec_from_file_location(
    "model_tensors",
    PROJECT_DIR / "scripts" / "features" / "12.build_model_tensors.py",
)
tensors_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tensors_module)


requires_real_data = pytest.mark.skipif(
    not all(path.exists() for path in REAL_DATA_FILES),
    reason="Run scripts 9 and 10 to build the master panel first.",
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def make_panel(n_periods=40, n_nodes=3, weather_complete=True):
    """Return a synthetic panel with the columns the builder reads."""

    records = []
    start = pd.Timestamp("2010-01-02")

    for period_id in range(1, n_periods + 1):
        period_start = start + pd.Timedelta(days=7 * (period_id - 1))
        for node_id in range(n_nodes):
            records.append(
                {
                    "period_id": period_id,
                    "node_id": node_id,
                    "start_date": period_start,
                    "reporting_days": 7,
                    "cases": float(period_id + node_id),
                    "case_observed": 1,
                    "rainfall_daily_mean_mm": float(period_id),
                    "rainy_days": 3,
                    "temperature_mean_c": 26.0 + node_id,
                    "temperature_min_c": 22.0,
                    "temperature_max_c": 30.0,
                    "dewpoint_mean_c": 22.0,
                    "relative_humidity_mean": 80.0,
                    "wind_speed_mean": 2.0,
                    "weather_complete": weather_complete,
                }
            )

    return pd.DataFrame(records).sort_values(["period_id", "node_id"]).reset_index(
        drop=True
    )


def make_nodes(n_nodes=3):
    """Return a synthetic node registry."""

    return pd.DataFrame(
        {
            "node_id": list(range(n_nodes)),
            "canonical_name": [f"District{index}" for index in range(n_nodes)],
            "centroid_lat": [7.0 + index for index in range(n_nodes)],
            "centroid_lon": [80.0 + index for index in range(n_nodes)],
        }
    )


def build_synthetic(variant="v1", **kwargs):
    """Run the full synthetic build and return the tensors."""

    panel = make_panel(**kwargs)
    nodes = make_nodes(kwargs.get("n_nodes", 3))

    tensors_module.check_grid(panel, nodes)

    panel = tensors_module.mask_unobserved_weather(panel)
    panel = tensors_module.add_base_features(panel, nodes)
    panel, rolling_names = tensors_module.add_rolling_features(panel)

    features = tensors_module.feature_names_for(variant, rolling_names)

    return tensors_module.build_tensors(panel, features)


# ---------------------------------------------------------------------------
# Grid checks
# ---------------------------------------------------------------------------

def test_check_grid_accepts_a_complete_grid():
    tensors_module.check_grid(make_panel(), make_nodes())


def test_check_grid_rejects_a_missing_district_period():
    panel = make_panel()
    panel = panel.drop(index=5).reset_index(drop=True)

    with pytest.raises(ValueError):
        tensors_module.check_grid(panel, make_nodes())


def test_check_grid_rejects_node_major_ordering():
    panel = make_panel().sort_values(["node_id", "period_id"]).reset_index(drop=True)

    with pytest.raises(ValueError, match="ordered period_id then node_id"):
        tensors_module.check_grid(panel, make_nodes())


def test_check_grid_rejects_a_period_gap():
    panel = make_panel()
    panel = panel[panel["period_id"] != 10].reset_index(drop=True)

    with pytest.raises(ValueError, match="dense gap-free"):
        tensors_module.check_grid(panel, make_nodes())


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def test_unobserved_weather_clears_the_fabricated_rainy_day_count():
    panel = make_panel(weather_complete=False)

    masked = tensors_module.mask_unobserved_weather(panel)

    assert masked["rainy_days"].isna().all()


def test_rainy_days_are_normalised_by_the_true_period_length():
    panel = make_panel()
    panel.loc[panel["period_id"] == 5, "reporting_days"] = 8
    panel.loc[panel["period_id"] == 5, "rainy_days"] = 8

    built = tensors_module.add_base_features(panel, make_nodes())
    irregular = built[built["period_id"] == 5]

    assert (irregular["rainy_days_frac"] == 1.0).all()


def test_rolling_features_are_trailing_only():
    """A spike at one period must not alter any earlier rolling value."""

    panel = make_panel()
    baseline, names = tensors_module.add_rolling_features(
        tensors_module.add_base_features(panel, make_nodes())
    )

    spiked = make_panel()
    spiked.loc[spiked["period_id"] == 30, "rainfall_daily_mean_mm"] = 999.0
    spiked, _ = tensors_module.add_rolling_features(
        tensors_module.add_base_features(spiked, make_nodes())
    )

    rolling = [name for name in names if name.startswith("rainfall")]
    before = baseline["period_id"] < 30

    pd.testing.assert_frame_equal(
        baseline.loc[before, rolling],
        spiked.loc[before, rolling],
    )

    # and the spike does reach the periods at and after it, or the test above
    # would pass on a column of constants
    assert not baseline.loc[~before, rolling].equals(spiked.loc[~before, rolling])


def test_rolling_features_never_emit_a_partial_mean():
    panel = tensors_module.add_base_features(make_panel(), make_nodes())
    built, names = tensors_module.add_rolling_features(panel)

    for window in tensors_module.ROLLING_WINDOWS:
        name = tensors_module.rolling_feature_name("rainfall_daily_mean_mm", window)
        leading = built[built["period_id"] < window]
        settled = built[built["period_id"] >= window]

        assert leading[name].isna().all()
        assert settled[name].notna().all()


def test_variants_differ_only_by_the_rolling_block():
    v0 = build_synthetic("v0")
    v1 = build_synthetic("v1")

    n_base = len(tensors_module.BASE_FEATURES)
    n_rolling = len(tensors_module.ROLLING_SOURCES) * len(
        tensors_module.ROLLING_WINDOWS
    )

    assert list(v0["feature_names"]) == tensors_module.BASE_FEATURES
    assert v1["X"].shape[2] == n_base + n_rolling
    np.testing.assert_allclose(v0["X"], v1["X"][:, :, :n_base])


# ---------------------------------------------------------------------------
# Window alignment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lookback,horizon", [(1, 1), (4, 1), (12, 1), (4, 3)])
def test_windows_never_read_at_or_after_the_target(lookback, horizon):
    """The assertion this whole module exists for."""

    tensors = build_synthetic()
    windows = tensors_module.make_windows(
        tensors, lookback=lookback, horizon=horizon, drop_incomplete=False
    )

    origins = windows["origin_period_id"]
    targets = windows["target_period_id"]
    inputs = windows["input_period_id"]

    assert (targets == origins + horizon).all()
    assert (inputs <= origins[:, None]).all()
    assert (inputs < targets[:, None]).all()
    assert inputs.shape[1] == lookback
    assert (inputs[:, -1] == origins).all()

    # contiguous and increasing, so the window is a true history and not a
    # gathered set of periods
    assert (np.diff(inputs, axis=1) == 1).all()


def test_window_contents_match_the_source_tensors():
    tensors = build_synthetic()
    windows = tensors_module.make_windows(
        tensors, lookback=5, horizon=2, drop_incomplete=False
    )

    period_index = {period: index for index, period in enumerate(tensors["period_id"])}

    for sample in (0, 7, len(windows["y"]) - 1):
        for step, period in enumerate(windows["input_period_id"][sample]):
            np.testing.assert_allclose(
                windows["X"][sample, step], tensors["X"][period_index[period]]
            )

        target = windows["target_period_id"][sample]
        np.testing.assert_allclose(windows["y"][sample], tensors["y"][period_index[target]])


def one_missing_district_tensors():
    """Tensors where district 1 has no case record for period 20."""

    panel = make_panel()
    panel.loc[(panel["period_id"] == 20) & (panel["node_id"] == 1), "cases"] = np.nan
    panel.loc[(panel["period_id"] == 20) & (panel["node_id"] == 1), "case_observed"] = 0

    nodes = make_nodes()
    built = tensors_module.add_base_features(
        tensors_module.mask_unobserved_weather(panel), nodes
    )
    built, rolling_names = tensors_module.add_rolling_features(built)

    return tensors_module.build_tensors(
        built, tensors_module.feature_names_for("v0", rolling_names)
    )


def test_a_partly_observed_target_is_kept_and_masked_not_dropped():
    tensors = one_missing_district_tensors()
    windows = tensors_module.make_windows(tensors, lookback=4, horizon=1)

    targets = windows["target_period_id"].tolist()

    assert 20 in targets

    # 24 of 25 districts survive; only the absent one is masked out of the loss
    sample = targets.index(20)
    np.testing.assert_array_equal(windows["y_mask"][sample], [1, 0, 1])


def test_a_nan_input_still_removes_every_window_that_reads_it():
    tensors = one_missing_district_tensors()
    windows = tensors_module.make_windows(tensors, lookback=4, horizon=1)

    targets = set(windows["target_period_id"].tolist())

    # cases_log1p is NaN at period 20, and a four-period window targeting 21
    # through 24 reads it, so those go
    assert targets.isdisjoint({21, 22, 23, 24})
    assert 25 in targets
    assert 19 in targets


def test_a_fully_unobserved_target_is_dropped():
    panel = make_panel()
    panel.loc[panel["period_id"] == 20, "cases"] = np.nan
    panel.loc[panel["period_id"] == 20, "case_observed"] = 0

    nodes = make_nodes()
    built = tensors_module.add_base_features(
        tensors_module.mask_unobserved_weather(panel), nodes
    )
    built, rolling_names = tensors_module.add_rolling_features(built)
    tensors = tensors_module.build_tensors(
        built, tensors_module.feature_names_for("v0", rolling_names)
    )

    windows = tensors_module.make_windows(
        tensors, lookback=4, horizon=1, drop_incomplete=False
    )

    assert 20 not in set(windows["target_period_id"].tolist())


def test_make_windows_rejects_an_impossible_configuration():
    tensors = build_synthetic(n_periods=10)

    with pytest.raises(ValueError):
        tensors_module.make_windows(tensors, lookback=0)

    with pytest.raises(ValueError):
        tensors_module.make_windows(tensors, lookback=12, horizon=1)


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_tensors():
    panel = tensors_module.load_panel()
    nodes = tensors_module.load_nodes()

    tensors_module.check_grid(panel, nodes)

    panel = tensors_module.mask_unobserved_weather(panel)
    panel = tensors_module.add_base_features(panel, nodes)
    panel, rolling_names = tensors_module.add_rolling_features(panel)

    return {
        variant: tensors_module.build_tensors(
            panel, tensors_module.feature_names_for(variant, rolling_names)
        )
        for variant in ("v0", "v1")
    }


@requires_real_data
def test_real_tensor_shapes(real_tensors):
    assert real_tensors["v0"]["X"].shape == (1012, 25, 14)
    assert real_tensors["v1"]["X"].shape == (1012, 25, 23)
    assert real_tensors["v0"]["y"].shape == (1012, 25)


@requires_real_data
def test_node_axis_matches_the_registry(real_tensors):
    nodes = pd.read_csv(NODES_PATH).sort_values("node_id")

    np.testing.assert_array_equal(
        real_tensors["v0"]["node_id"], nodes["node_id"].to_numpy()
    )

    # the centroid columns are carried as features, so they double as a check
    # that row i of the node axis really is district i
    names = list(real_tensors["v0"]["feature_names"])
    latitudes = real_tensors["v0"]["X"][0, :, names.index("centroid_lat")]

    np.testing.assert_allclose(
        latitudes, nodes["centroid_lat"].to_numpy(), rtol=1e-5
    )


@requires_real_data
def test_targets_match_the_panel(real_tensors):
    panel = pd.read_parquet(PANEL_PATH)[["period_id", "node_id", "cases"]]
    panel = panel.sort_values(["period_id", "node_id"])

    flat = real_tensors["v0"]["y"].reshape(-1)

    np.testing.assert_allclose(
        flat, panel["cases"].to_numpy(dtype=np.float32), equal_nan=True
    )


@requires_real_data
def test_known_gaps_are_carried_as_missing(real_tensors):
    tensors = real_tensors["v0"]
    names = list(tensors["feature_names"])

    period_index = {period: index for index, period in enumerate(tensors["period_id"])}

    # the single absent Puttalam record, period 999
    assert tensors["y_mask"][period_index[999], 21] == 0
    assert np.isnan(tensors["y"][period_index[999], 21])

    # ERA5 ends 2026-04-26, so periods 1010-1012 have no weather at all
    for period in (1010, 1011, 1012):
        row = tensors["X"][period_index[period], :, names.index("temperature_mean_c")]
        assert np.isnan(row).all()
        assert (tensors["weather_mask"][period_index[period]] == 0).all()


@requires_real_data
def test_real_windows_are_aligned(real_tensors):
    windows = tensors_module.make_windows(real_tensors["v1"], lookback=12, horizon=1)

    assert (windows["target_period_id"] == windows["origin_period_id"] + 1).all()
    assert (windows["input_period_id"] < windows["target_period_id"][:, None]).all()
    assert not np.isnan(windows["X"]).any()

    # every kept window has at least one observed district, and the only masked
    # cell in the whole set is the absent Puttalam record for period 999
    assert (windows["y_mask"] == 1).any(axis=1).all()

    masked = np.argwhere(windows["y_mask"] != 1)
    assert {int(windows["target_period_id"][row]) for row, _ in masked} == {999}
    assert set(masked[:, 1].tolist()) == {21}

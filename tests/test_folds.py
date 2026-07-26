"""
Tests for the walk-forward folds and the per-fold preprocessing.

The central test is test_statistics_ignore_everything_after_the_fit_boundary:
it corrupts every period from the test year onward and asserts the fitted
statistics do not move. That is leakage stated as an experiment rather than as
a code review, and it is the one property that cannot be checked by reading the
training curve.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parent.parent

CALENDAR_PATH = PROJECT_DIR / "data" / "interim" / "reporting_calendar.csv"
TENSOR_PATH = PROJECT_DIR / "data" / "processed" / "model_tensors_v1.npz"

spec = importlib.util.spec_from_file_location(
    "build_folds",
    PROJECT_DIR / "scripts" / "14.build_folds.py",
)
folds_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(folds_module)


requires_real_data = pytest.mark.skipif(
    not (CALENDAR_PATH.exists() and TENSOR_PATH.exists()),
    reason="Run scripts 9, 10 and 12 first.",
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def make_calendar(first_year=2010, last_year=2020, periods_per_year=52):
    """Return a synthetic calendar with whole years of weekly periods."""

    records = []
    period_id = 1

    for year in range(first_year, last_year + 1):
        start = pd.Timestamp(year=year, month=1, day=1)
        for index in range(periods_per_year):
            period_start = start + pd.Timedelta(days=7 * index)
            records.append(
                {
                    "period_id": period_id,
                    "start_date": period_start,
                    "end_date": period_start + pd.Timedelta(days=6),
                    "year": year,
                    "month": period_start.month,
                }
            )
            period_id += 1

    return pd.DataFrame(records)


def make_tensors(calendar, n_nodes=4, n_features=3, seed=0):
    """Return synthetic tensors aligned to a calendar."""

    rng = np.random.default_rng(seed)
    n_periods = len(calendar)

    return {
        "X": rng.normal(size=(n_periods, n_nodes, n_features)).astype(np.float32),
        "y": rng.gamma(2.0, 10.0, size=(n_periods, n_nodes)).astype(np.float32),
        "y_mask": np.ones((n_periods, n_nodes), dtype=np.int8),
        "period_id": calendar["period_id"].to_numpy(dtype=np.int32),
        "feature_names": np.array([f"f{i}" for i in range(n_features)], dtype=object),
    }


@pytest.fixture
def calendar():
    return make_calendar()


@pytest.fixture
def empty_regime():
    return pd.DataFrame(columns=["period_id", "is_covid_window", "is_2017_outbreak"])


@pytest.fixture
def synthetic_folds(calendar, empty_regime):
    return folds_module.build_folds(
        calendar, empty_regime, first_test_year=2015, last_test_year=2020
    )


# ---------------------------------------------------------------------------
# Fold structure
# ---------------------------------------------------------------------------

def test_folds_walk_forward_one_year_at_a_time(synthetic_folds):
    assert [fold["test_year"] for fold in synthetic_folds] == [
        2015,
        2016,
        2017,
        2018,
        2019,
        2020,
    ]

    for fold in synthetic_folds:
        assert fold["train_start_period"] == 1


def test_fold_boundaries_are_ordered_and_disjoint(synthetic_folds, calendar):
    folds_module.check_folds(synthetic_folds, calendar)

    for fold in synthetic_folds:
        assert fold["train_end_period"] < fold["val_start_period"]
        assert fold["val_end_period"] < fold["test_start_period"]
        assert fold["fit_end_period"] == fold["test_start_period"] - 1


def test_test_ranges_never_overlap(synthetic_folds):
    for earlier, later in zip(synthetic_folds, synthetic_folds[1:]):
        assert later["test_start_period"] > earlier["test_end_period"]


def test_validation_is_the_year_before_the_test_year(synthetic_folds, calendar):
    for fold in synthetic_folds:
        val_years = calendar.loc[
            calendar["period_id"].between(
                fold["val_start_period"], fold["val_end_period"]
            ),
            "year",
        ].unique()

        assert val_years.tolist() == [fold["test_year"] - 1]


def test_check_folds_rejects_a_fit_boundary_inside_the_test_year(
    synthetic_folds, calendar
):
    broken = [dict(fold) for fold in synthetic_folds]
    broken[0]["fit_end_period"] = broken[0]["test_start_period"] + 5

    with pytest.raises(ValueError, match="fits statistics into its test range"):
        folds_module.check_folds(broken, calendar)


def test_check_folds_rejects_overlapping_test_ranges(synthetic_folds, calendar):
    broken = [dict(fold) for fold in synthetic_folds]
    broken[1]["test_start_period"] = broken[0]["test_end_period"] - 1

    with pytest.raises(ValueError):
        folds_module.check_folds(broken, calendar)


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_statistics_ignore_everything_after_the_fit_boundary(
    calendar, synthetic_folds
):
    """Corrupt the future and the fitted statistics must not move."""

    tensors = make_tensors(calendar)
    months = calendar["month"].to_numpy()
    fold = synthetic_folds[0]

    fit_mask = tensors["period_id"] <= fold["fit_end_period"]
    clean = folds_module.fit_fold_statistics(tensors, months, fit_mask)

    corrupted = {key: value.copy() for key, value in tensors.items()}
    corrupted["X"][~fit_mask] = 1e6

    dirty = folds_module.fit_fold_statistics(corrupted, months, fit_mask)

    for key in clean:
        np.testing.assert_allclose(clean[key], dirty[key], rtol=1e-6)


def test_each_fold_sees_strictly_more_history_than_the_last(
    calendar, synthetic_folds
):
    tensors = make_tensors(calendar)
    period_id = tensors["period_id"]

    sizes = [
        int((period_id <= fold["fit_end_period"]).sum()) for fold in synthetic_folds
    ]

    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_windows_are_assigned_by_target_period(synthetic_folds):
    fold = synthetic_folds[0]
    targets = np.arange(1, fold["test_end_period"] + 1)

    masks = folds_module.assign_windows(targets, fold)

    assert targets[masks["train"]].max() <= fold["train_end_period"]
    assert targets[masks["val"]].min() >= fold["val_start_period"]
    assert targets[masks["test"]].min() >= fold["test_start_period"]

    # no window belongs to two splits
    overlap = masks["train"].astype(int) + masks["val"] + masks["test"]
    assert overlap.max() <= 1


# ---------------------------------------------------------------------------
# Imputation and scaling
# ---------------------------------------------------------------------------

def test_imputation_fills_from_the_matching_district_and_month(calendar):
    tensors = make_tensors(calendar)
    months = calendar["month"].to_numpy()

    n_nodes, n_features = tensors["X"].shape[1], tensors["X"].shape[2]
    climatology = np.arange(n_nodes * 12 * n_features, dtype=np.float32).reshape(
        n_nodes, 12, n_features
    )

    x = tensors["X"].copy()
    x[5, 2, 1] = np.nan

    filled = folds_module.impute(x, climatology, months)

    assert filled[5, 2, 1] == climatology[2, months[5] - 1, 1]
    assert not np.isnan(filled).any()


def test_imputation_leaves_observed_cells_untouched(calendar):
    tensors = make_tensors(calendar)
    months = calendar["month"].to_numpy()

    x = tensors["X"].copy()
    x[3, 1, 0] = np.nan

    climatology = np.zeros((x.shape[1], 12, x.shape[2]), dtype=np.float32)
    filled = folds_module.impute(x, climatology, months)

    observed = ~np.isnan(x)
    np.testing.assert_allclose(filled[observed], x[observed])


def test_scaling_standardises_the_fitting_history(calendar, synthetic_folds):
    tensors = make_tensors(calendar)
    months = calendar["month"].to_numpy()
    fold = synthetic_folds[2]

    fit_mask = tensors["period_id"] <= fold["fit_end_period"]
    statistics = folds_module.fit_fold_statistics(tensors, months, fit_mask)

    scaled = folds_module.transform(tensors, months, statistics)

    history = scaled[fit_mask]
    np.testing.assert_allclose(history.mean(axis=(0, 1)), 0.0, atol=1e-4)
    np.testing.assert_allclose(history.std(axis=(0, 1)), 1.0, atol=1e-4)


def test_scaling_survives_a_constant_feature(calendar, synthetic_folds):
    """A fully observed flag channel has zero variance and must not blow up."""

    tensors = make_tensors(calendar)
    tensors["X"][:, :, 0] = 1.0

    months = calendar["month"].to_numpy()
    fold = synthetic_folds[0]
    fit_mask = tensors["period_id"] <= fold["fit_end_period"]

    statistics = folds_module.fit_fold_statistics(tensors, months, fit_mask)
    scaled = folds_module.transform(tensors, months, statistics)

    assert statistics["std"][0] == 1.0
    assert np.isfinite(scaled).all()
    np.testing.assert_allclose(scaled[:, :, 0], 0.0, atol=1e-6)


def test_transform_removes_every_nan(calendar, synthetic_folds):
    tensors = make_tensors(calendar)
    rng = np.random.default_rng(1)

    holes = rng.random(tensors["X"].shape) < 0.05
    tensors["X"][holes] = np.nan

    months = calendar["month"].to_numpy()
    fold = synthetic_folds[0]
    fit_mask = tensors["period_id"] <= fold["fit_end_period"]

    statistics = folds_module.fit_fold_statistics(tensors, months, fit_mask)
    scaled = folds_module.transform(tensors, months, statistics)

    assert np.isfinite(scaled).all()


def test_a_district_month_with_no_history_falls_back(calendar, synthetic_folds):
    tensors = make_tensors(calendar)
    months = calendar["month"].to_numpy()
    fold = synthetic_folds[0]
    fit_mask = tensors["period_id"] <= fold["fit_end_period"]

    # wipe every January observation for district 2
    january = months == 1
    tensors["X"][np.ix_(january, [2])] = np.nan

    statistics = folds_module.fit_fold_statistics(tensors, months, fit_mask)

    assert not np.isnan(statistics["climatology"]).any()
    np.testing.assert_allclose(
        statistics["climatology"][2, 0, :], statistics["district_mean"][2, :]
    )


def test_the_target_is_never_imputed(calendar, synthetic_folds):
    tensors = make_tensors(calendar)
    tensors["y"][10, 1] = np.nan
    tensors["y_mask"][10, 1] = 0

    months = calendar["month"].to_numpy()
    fold = synthetic_folds[0]
    fit_mask = tensors["period_id"] <= fold["fit_end_period"]

    statistics = folds_module.fit_fold_statistics(tensors, months, fit_mask)
    folds_module.transform(tensors, months, statistics)

    assert np.isnan(tensors["y"][10, 1])
    assert tensors["y_mask"][10, 1] == 0


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_folds():
    calendar = folds_module.load_calendar()
    regime = folds_module.load_regime_flags()

    return calendar, folds_module.build_folds(calendar, regime)


@requires_real_data
def test_real_folds_match_the_calendar(real_folds):
    calendar, built = real_folds

    folds_module.check_folds(built, calendar)

    assert len(built) == 9
    assert built[0]["test_year"] == 2017
    assert built[-1]["test_year"] == 2025


@requires_real_data
def test_the_2017_epidemic_is_a_test_fold_and_never_training_data(real_folds):
    _, built = real_folds

    epidemic = [fold for fold in built if fold["covers_2017_outbreak"]]

    assert len(epidemic) == 1
    assert epidemic[0]["test_year"] == 2017
    # the epidemic fold must not have seen any of 2017 while training
    assert epidemic[0]["fit_end_period"] < epidemic[0]["test_start_period"]


@requires_real_data
def test_covid_folds_are_flagged_and_excluded_from_the_headline(real_folds):
    _, built = real_folds

    covid = [fold["test_year"] for fold in built if fold["covers_covid"]]
    headline = [fold["test_year"] for fold in built if fold["headline"]]

    assert covid == [2020, 2021]
    assert 2020 not in headline and 2021 not in headline
    assert 2017 in headline


@requires_real_data
def test_real_fold_statistics_are_finite_and_complete(real_folds):
    calendar, built = real_folds

    tensors = folds_module.load_tensors("v1")
    months = calendar.sort_values("period_id")["month"].to_numpy()

    for fold in (built[0], built[-1]):
        fit_mask = tensors["period_id"] <= fold["fit_end_period"]
        statistics = folds_module.fit_fold_statistics(tensors, months, fit_mask)

        assert np.isfinite(statistics["climatology"]).all()
        assert (statistics["std"] > 0).all()

        scaled = folds_module.transform(tensors, months, statistics)
        assert np.isfinite(scaled).all()


@requires_real_data
def test_every_fold_has_a_full_test_year_of_windows(real_folds):
    _, built = real_folds

    tensors = folds_module.load_tensors("v1")
    sizes = folds_module.size_folds(tensors, built)

    assert (sizes["test_windows"] >= 52).all()
    assert (sizes["train_windows"] > 400).all()
    # training history must grow monotonically across the walk
    assert sizes["train_windows"].is_monotonic_increasing

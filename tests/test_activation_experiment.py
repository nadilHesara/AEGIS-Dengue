"""
Tests for the activation experiment and its gradient probe.

These cover the parts of `scripts/22` and `scripts/23` that can be wrong
silently. Training is not exercised -- that needs the real tensors and a GPU
budget -- but the reporting and summary logic is, because a summary that
mislabels an arm or averages the wrong folds produces a plausible-looking table
that says something false, which is worse than a crash.

The specific thing being protected is the honesty of the comparison: that the
control is really the control, that the headline mean is really over headline
folds, and that a missing optional input degrades the report instead of
inventing a number for it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def load_script(name: str, filename: str):
    """Import a numbered script, whose filename is not a valid module name."""

    spec = importlib.util.spec_from_file_location(
        name, PROJECT_DIR / "scripts" / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


@pytest.fixture(scope="module")
def experiment():
    return load_script("activation_experiment", "training/22.train_simplex_activations.py")


@pytest.fixture(scope="module")
def probe():
    return load_script("activation_probe", "evaluation/23.activation_gradient_probe.py")


def make_metrics() -> pd.DataFrame:
    """Two arms over three folds, one of which is deliberately not headline.

    The non-headline fold carries a wild MAE so that any function which fails
    to exclude it produces an obviously wrong number rather than a subtly wrong
    one.
    """

    rows = []
    for arm, base in (("softmax", 20.0), ("entmax15", 18.0)):
        for fold_id, headline in ((1, True), (2, True), (4, False)):
            for seed in range(3):
                rows.append(
                    {
                        "arm": arm,
                        "backbone": "gcn_gru",
                        "fold_id": fold_id,
                        "test_year": 2016 + fold_id,
                        "headline": headline,
                        "covers_covid": not headline,
                        "seed": seed,
                        "mae": (base + seed) if headline else 999.0,
                        "rmse": base * 2,
                        "peak_mae": base * 1.5,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summarising
# ---------------------------------------------------------------------------

def test_summary_uses_headline_folds_only(experiment):
    """A COVID fold must never reach the headline mean.

    Folds 4 and 5 are reported separately throughout this project. If they
    leaked into the mean every arm would score about 999 and the ranking would
    be meaningless.
    """

    summary = experiment.summarise(make_metrics())
    softmax = summary[summary["arm"] == "softmax"].iloc[0]

    # Seeds 0, 1, 2 over folds 1 and 2, all at base 20 -> mean 21.
    assert softmax["headline_mae"] == pytest.approx(21.0)
    assert softmax["headline_mae"] < 100.0


def test_summary_reports_seed_spread_not_fold_spread(experiment):
    """`seed_sd` must be the spread of per-seed means across seeds.

    That is the number every claim in this repo is judged against -- "larger
    than the seed sd" is the bar -- so computing it over the wrong axis would
    silently change what counts as a real effect.
    """

    summary = experiment.summarise(make_metrics())
    softmax = summary[summary["arm"] == "softmax"].iloc[0]

    # Per-seed headline means are 20, 21, 22; their sample sd is 1.
    assert softmax["seed_sd"] == pytest.approx(1.0)


def test_summary_is_ordered_best_first(experiment):
    """The report reads `iloc[0]` as the best arm, so the sort must hold."""

    summary = experiment.summarise(make_metrics())

    assert summary.iloc[0]["arm"] == "entmax15"
    assert summary["headline_mae"].is_monotonic_increasing


def test_every_declared_arm_is_a_real_activation(experiment):
    """The script's arm list must not name an activation that does not exist.

    A typo here would fail only after the first arm had finished training,
    which on a full sweep is a long way in.
    """

    from src.models.simplex_activations import SimplexActivation

    assert set(experiment.ARM_NAMES) <= set(SimplexActivation.NAMES)
    assert "softmax" in experiment.ARM_NAMES, "the control must always be present"


# ---------------------------------------------------------------------------
# Delay recovery
# ---------------------------------------------------------------------------

def make_kernels(peak_by_node: dict[int, float]) -> pd.DataFrame:
    rows = []
    for node, peak in peak_by_node.items():
        for feature in ("rainfall_daily_mean_mm", "temperature_mean_c"):
            rows.append(
                {
                    "arm": "softmax",
                    "backbone": "gcn_gru",
                    "fold_id": 1,
                    "seed": 0,
                    "node_index": node,
                    "feature": feature,
                    # Only the rainfall row should be scored, so the
                    # temperature row is given a contradictory value.
                    "peak_lag": peak if "rain" in feature else 25.0 - peak,
                    "kernel_sd": 3.0,
                    "support_size": 6,
                }
            )

    return pd.DataFrame(rows)


def test_correlation_recovers_a_planted_relationship(experiment):
    """A learned delay that matches the measured one must score r = +1."""

    peaks = {node: float(node) for node in range(10)}
    measured = pd.DataFrame(
        {"node_id": list(range(10)), "peak_lag": [float(n) for n in range(10)]}
    )

    result = experiment.correlate_with_measured(make_kernels(peaks), measured)

    assert result == pytest.approx(1.0)


def test_correlation_detects_an_inverted_relationship(experiment):
    """The sign has to be real, or `r = -0.16` could not be read as bad news."""

    peaks = {node: float(9 - node) for node in range(10)}
    measured = pd.DataFrame(
        {"node_id": list(range(10)), "peak_lag": [float(n) for n in range(10)]}
    )

    result = experiment.correlate_with_measured(make_kernels(peaks), measured)

    assert result == pytest.approx(-1.0)


def test_correlation_scores_rainfall_not_the_other_features(experiment):
    """Only the rainfall channel has a measured delay to compare against.

    The fixture gives temperature the inverted value, so a function scoring the
    wrong feature -- or both -- cannot return +1.
    """

    peaks = {node: float(node) for node in range(10)}
    measured = pd.DataFrame(
        {"node_id": list(range(10)), "peak_lag": [float(n) for n in range(10)]}
    )

    assert experiment.correlate_with_measured(make_kernels(peaks), measured) > 0.99


def test_correlation_returns_nan_when_the_scan_is_missing(experiment):
    """A missing optional input must blank the cell, not fabricate a number."""

    peaks = {node: float(node) for node in range(10)}

    assert np.isnan(experiment.correlate_with_measured(make_kernels(peaks), None))


def test_correlation_returns_nan_on_unexpected_columns(experiment):
    """If the scan's schema changes, the cell must go blank rather than lie."""

    peaks = {node: float(node) for node in range(10)}
    wrong = pd.DataFrame({"district": range(10), "delay": range(10)})

    assert np.isnan(experiment.correlate_with_measured(make_kernels(peaks), wrong))


def test_correlation_needs_enough_shared_districts(experiment):
    """Two points always correlate perfectly; that is not evidence."""

    peaks = {node: float(node) for node in range(10)}
    measured = pd.DataFrame({"node_id": [0, 1], "peak_lag": [0.0, 1.0]})

    assert np.isnan(experiment.correlate_with_measured(make_kernels(peaks), measured))


def test_kernel_summary_reports_support_and_spread(experiment):
    """Support size is the mechanism, so it must survive into the summary."""

    peaks = {node: float(node) for node in range(10)}
    stats = experiment.kernel_summary(make_kernels(peaks), None)

    assert set(stats.columns) >= {
        "arm",
        "backbone",
        "mean_support",
        "mean_kernel_sd",
        "lag_correlation",
    }
    assert stats.iloc[0]["mean_support"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------

def test_probe_writes_one_file_per_fold(probe):
    """Two folds must not collide on one filename.

    They did in the first version of this script, and the symptom was a report
    labelled with one fold holding another fold's numbers.
    """

    first_csv, first_report = probe.output_paths(1)
    eighth_csv, eighth_report = probe.output_paths(8)

    assert first_csv != eighth_csv
    assert first_report != eighth_report
    assert "fold1" in first_csv.name and "fold8" in eighth_csv.name


def test_probe_report_survives_a_single_epoch(probe, tmp_path):
    """`write_report` reads first and last epoch; at one epoch they coincide."""

    frame = pd.DataFrame(
        [
            {
                "activation": "softmax",
                "epoch": 0,
                "loss": 1.0,
                "grad_norm": 1e-4,
                "mean_support": 6.0,
                "dead_fraction": 0.0,
                "mean_kernel_sd": 3.0,
                "mean_peak_lag": 8.0,
                "peak_lag_sd": 2.0,
            }
        ]
    )

    destination = tmp_path / "probe.md"
    probe.write_report(frame, epochs=1, fold_id=8, report_path=destination)

    text = destination.read_text(encoding="utf-8")
    assert "softmax" in text
    assert "1.00e-04" in text


def test_epidemic_share_detects_a_single_fold_gain(experiment):
    """A gain living entirely in fold 1 must be reported as such.

    This is the check the seed-sd test cannot make. In the real sweep every
    arm cleared or nearly cleared the seed sd while 96-117% of the gain came
    from fold 1 and the other six folds moved by hundredths of an MAE -- the
    same shape README §8 records for the loss work. Without this decomposition
    the report would call that a real improvement.
    """

    rows = []
    for fold_id in (1, 2, 3, 6, 7, 8, 9):
        for arm in ("softmax", "entmax15"):
            # The arms are identical everywhere except fold 1.
            mae = 20.0 if arm == "softmax" else (15.0 if fold_id == 1 else 20.0)
            rows.append(
                {
                    "arm": arm,
                    "backbone": "gru_only",
                    "fold_id": fold_id,
                    "headline": True,
                    "seed": 0,
                    "mae": mae,
                }
            )

    share, without = experiment.epidemic_share(
        pd.DataFrame(rows), "gru_only", "entmax15"
    )

    assert share == pytest.approx(100.0)
    assert without == pytest.approx(0.0)


def test_epidemic_share_recognises_a_broad_gain(experiment):
    """An arm that improves every fold must not be dismissed as one fold."""

    rows = []
    for fold_id in (1, 2, 3, 6, 7, 8, 9):
        for arm in ("softmax", "entmax15"):
            mae = 20.0 if arm == "softmax" else 19.0
            rows.append(
                {
                    "arm": arm,
                    "backbone": "gru_only",
                    "fold_id": fold_id,
                    "headline": True,
                    "seed": 0,
                    "mae": mae,
                }
            )

    share, without = experiment.epidemic_share(
        pd.DataFrame(rows), "gru_only", "entmax15"
    )

    assert share == pytest.approx(100.0 / 7.0)
    assert without == pytest.approx(1.0)


def test_epidemic_share_is_none_without_a_control(experiment):
    """No control means no comparison, not a fabricated one."""

    rows = [
        {
            "arm": "entmax15",
            "backbone": "gru_only",
            "fold_id": 1,
            "headline": True,
            "seed": 0,
            "mae": 15.0,
        }
    ]

    assert experiment.epidemic_share(pd.DataFrame(rows), "gru_only", "entmax15") == (
        None,
        None,
    )

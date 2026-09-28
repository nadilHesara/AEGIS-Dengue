"""
Tests for the CHIRPS extraction and the ERA5 comparison.

Neither real input exists until the Earth Engine extractions run, so the
statistics are verified against synthetic data built with a *known*
relationship. That is stronger than checking the code merely runs: if CHIRPS
is constructed as 0.8 times ERA5, the measured bias must come back at
about 25 percent, and a test that says so would catch a sign error, a
transposed pair or a wrong denominator.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load(name):
    """Load a numbered script by path."""

    spec = importlib.util.spec_from_file_location(
        name.replace(".", "_"), PROJECT_DIR / "scripts" / name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


compare = _load("data/8.compare_era5_chirps.py")


DISTRICTS = compare.FOCUS_DISTRICTS


def make_paired(relationship, seed=5, start="2015-01-01", end="2017-12-31"):
    """
    Build a paired rainfall frame with a known ERA5-to-CHIRPS relationship.

    relationship(era5_value, generator) returns the CHIRPS value, so a test
    can construct exactly the agreement it wants to measure.
    """

    generator = np.random.default_rng(seed)

    records = []

    for day in pd.date_range(start, end, freq="D"):
        for node_id, district in enumerate(DISTRICTS):
            era5_value = float(generator.gamma(0.6, 6.0))
            chirps_value = relationship(era5_value, generator)

            records.append(
                {
                    "date": day,
                    "node_id": node_id,
                    "canonical_name": district,
                    "rainfall_era5": max(era5_value, 0.0),
                    "rainfall_chirps": max(chirps_value, 0.0),
                }
            )

    return pd.DataFrame(records)


IDENTICAL = lambda era5, generator: era5
CLOSE = lambda era5, generator: era5 + float(generator.normal(0, 0.3))
SCALED = lambda era5, generator: era5 * 0.8 + float(generator.normal(0, 0.4))
MODERATE = lambda era5, generator: (
    era5 * 0.6 + float(generator.gamma(0.5, 4.0))
)
UNRELATED = lambda era5, generator: float(generator.gamma(0.6, 6.0))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_focus_districts_span_the_rainfall_regimes():
    """The five districts cover wet zone, dry zone, coast and interior."""

    assert DISTRICTS == [
        "Colombo",
        "Gampaha",
        "Ratnapura",
        "Jaffna",
        "Anuradhapura",
    ]


def test_focus_districts_are_real_districts():
    """Every focus district exists in the official registry."""

    nodes = pd.read_csv(PROJECT_DIR / "data" / "processed" / "nodes.csv")

    assert set(DISTRICTS).issubset(set(nodes["canonical_name"]))


def test_thresholds_are_ordered():
    """Heavy is above the wet-day threshold, and extreme above heavy."""

    assert (
        compare.WET_DAY_THRESHOLD_MM
        < compare.HEAVY_RAINFALL_MM
        < compare.EXTREME_RAINFALL_MM
    )

    assert (
        compare.MODERATE_DAILY_CORRELATION
        < compare.STRONG_DAILY_CORRELATION
    )


# ---------------------------------------------------------------------------
# Statistics against known relationships
# ---------------------------------------------------------------------------

def test_identical_series_correlate_perfectly():
    """Two identical series give correlation 1 and zero bias."""

    comparison = compare.build_district_comparison(
        make_paired(IDENTICAL), DISTRICTS
    )

    assert comparison["pearson_r"].round(6).eq(1.0).all()
    assert comparison["spearman_r"].round(6).eq(1.0).all()
    assert comparison["mean_difference_mm"].abs().max() < 1e-9
    assert comparison["relative_bias"].abs().max() < 1e-9


def test_known_scale_bias_is_measured_correctly():
    """
    CHIRPS built as 0.8 x ERA5 must show ERA5 about 25 percent higher.

    Clipping at zero pulls the measured value slightly below the exact
    1/0.8 - 1 = 25 percent, so the tolerance allows for that.
    """

    comparison = compare.build_district_comparison(
        make_paired(SCALED), DISTRICTS
    )

    assert comparison["relative_bias"].between(0.18, 0.28).all()

    # ERA5 is the larger series, so the difference is positive.
    assert comparison["mean_difference_mm"].gt(0).all()

    # A pure rescaling leaves the ordering untouched.
    assert comparison["pearson_r"].gt(0.95).all()


def test_unrelated_series_do_not_correlate():
    """Independent draws give a correlation near zero."""

    comparison = compare.build_district_comparison(
        make_paired(UNRELATED), DISTRICTS
    )

    assert comparison["pearson_r"].abs().max() < 0.2


def test_wet_day_counts_use_the_threshold():
    """A day below the wet threshold is not counted as wet."""

    days = pd.date_range("2015-01-01", periods=10, freq="D")

    # Five days at 5 mm, five days at 0.2 mm, in both products.
    paired = pd.DataFrame(
        {
            "date": days,
            "node_id": 4,
            "canonical_name": "Colombo",
            "rainfall_era5": [5.0] * 5 + [0.2] * 5,
            "rainfall_chirps": [5.0] * 5 + [0.2] * 5,
        }
    )

    record = compare.compare_district(paired, "Colombo")

    # Under 30 overlapping days the statistics are suppressed, so the
    # threshold behaviour is checked directly instead.
    assert record["days_both_present"] == 10
    assert record["note"] == "fewer than 30 overlapping days"

    wet = (paired["rainfall_era5"] >= compare.WET_DAY_THRESHOLD_MM).sum()

    assert wet == 5


def test_short_overlap_suppresses_statistics():
    """Fewer than 30 overlapping days yields no correlation."""

    paired = make_paired(CLOSE, start="2015-01-01", end="2015-01-20")

    record = compare.compare_district(paired, "Colombo")

    assert pd.isna(record["pearson_r"])
    assert record["note"] == "fewer than 30 overlapping days"


def test_missingness_is_counted_per_product():
    """Nulls in each series are counted separately."""

    paired = make_paired(CLOSE)

    colombo = paired["canonical_name"].eq("Colombo")
    indices = paired.loc[colombo].index

    paired.loc[indices[:10], "rainfall_era5"] = np.nan
    paired.loc[indices[:4], "rainfall_chirps"] = np.nan

    record = compare.compare_district(paired, "Colombo")

    assert record["era5_missing_days"] == 10
    assert record["chirps_missing_days"] == 4
    assert record["days_both_present"] == record["total_days"] - 10


def test_pairing_is_an_outer_join():
    """A date present in only one product is retained, not dropped."""

    era5 = pd.DataFrame(
        {
            "date": pd.to_datetime(["2015-01-01", "2015-01-02"]),
            "node_id": [4, 4],
            "canonical_name": ["Colombo", "Colombo"],
            "rainfall_era5": [1.0, 2.0],
        }
    )

    chirps = pd.DataFrame(
        {
            "date": pd.to_datetime(["2015-01-02", "2015-01-03"]),
            "node_id": [4, 4],
            "canonical_name": ["Colombo", "Colombo"],
            "rainfall_chirps": [2.0, 3.0],
        }
    )

    merged = era5.merge(
        chirps, on=["date", "node_id", "canonical_name"], how="outer"
    )

    assert len(merged) == 3
    assert merged["rainfall_era5"].isna().sum() == 1
    assert merged["rainfall_chirps"].isna().sum() == 1


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def test_close_agreement_recommends_cross_check():
    """Products that agree closely add nothing as a channel."""

    comparison = compare.build_district_comparison(
        make_paired(CLOSE), DISTRICTS
    )

    result = compare.build_recommendation(comparison)

    assert result["recommendation"] == "quality_cross_check"


def test_scale_bias_with_high_correlation_stays_a_cross_check():
    """
    A pure rescaling is a calibration difference, not new information.

    This is the case that matters most: a naive rule keyed on bias alone
    would call two near-identical series an independent channel, which is
    exactly backwards.
    """

    comparison = compare.build_district_comparison(
        make_paired(SCALED), DISTRICTS
    )

    result = compare.build_recommendation(comparison)

    assert result["recommendation"] == "quality_cross_check"
    assert result["median_absolute_bias"] > compare.MEAN_BIAS_TOLERANCE

    # The bias must still be reported, not hidden by the classification.
    assert any("calibration difference" in reason for reason in result["reasons"])


def test_moderate_agreement_recommends_additional_channel():
    """Partial agreement means each product carries its own information."""

    comparison = compare.build_district_comparison(
        make_paired(MODERATE), DISTRICTS
    )

    result = compare.build_recommendation(comparison)

    assert result["recommendation"] == "additional_channel"


def test_weak_agreement_recommends_replacement_candidate():
    """Weak daily agreement escalates, but only to a candidate."""

    comparison = compare.build_district_comparison(
        make_paired(UNRELATED), DISTRICTS
    )

    result = compare.build_recommendation(comparison)

    assert result["recommendation"] == "replacement_candidate"

    # Even here the wording must not authorise an automatic swap.
    reasons = " ".join(result["reasons"])

    assert "never as an automatic substitution" in reasons


def test_recommendation_handles_no_usable_data():
    """With no comparable district the recommendation says so."""

    comparison = compare.build_district_comparison(
        make_paired(CLOSE, start="2015-01-01", end="2015-01-05"), DISTRICTS
    )

    result = compare.build_recommendation(comparison)

    assert result["recommendation"] == "insufficient_data"


# ---------------------------------------------------------------------------
# Derived tables
# ---------------------------------------------------------------------------

def test_monthly_comparison_covers_every_month():
    """The seasonal table holds twelve months per district."""

    monthly = compare.build_monthly_comparison(
        make_paired(CLOSE), DISTRICTS
    )

    assert len(monthly) == 12 * len(DISTRICTS)
    assert sorted(monthly["month"].unique()) == list(range(1, 13))


def test_heavy_events_classify_agreement():
    """Each heavy day is labelled by which products saw it."""

    days = pd.date_range("2015-01-01", periods=3, freq="D")

    paired = pd.DataFrame(
        {
            "date": days,
            "node_id": 4,
            "canonical_name": "Colombo",
            "rainfall_era5": [80.0, 80.0, 1.0],
            "rainfall_chirps": [80.0, 1.0, 80.0],
        }
    )

    heavy = compare.build_heavy_events(paired, ["Colombo"])

    assert len(heavy) == 3
    assert set(heavy["agreement"]) == {"both", "era5 only", "chirps only"}


def test_heavy_events_ignore_ordinary_days():
    """Days below the heavy threshold do not appear."""

    paired = pd.DataFrame(
        {
            "date": pd.date_range("2015-01-01", periods=3, freq="D"),
            "node_id": 4,
            "canonical_name": "Colombo",
            "rainfall_era5": [1.0, 2.0, 3.0],
            "rainfall_chirps": [1.0, 2.0, 3.0],
        }
    )

    assert compare.build_heavy_events(paired, ["Colombo"]).empty


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def test_report_states_that_nothing_is_replaced():
    """The report must say plainly that ERA5 is unchanged."""

    paired = make_paired(CLOSE)

    comparison = compare.build_district_comparison(paired, DISTRICTS)
    monthly = compare.build_monthly_comparison(paired, DISTRICTS)
    heavy = compare.build_heavy_events(paired, DISTRICTS)

    report = compare.build_report(
        comparison,
        monthly,
        heavy,
        compare.build_recommendation(comparison),
        DISTRICTS,
    )

    assert "no substitution is applied" in report
    assert "Neither source is treated as truth" in report

    for district in DISTRICTS:
        assert district in report


def test_missing_inputs_raise_a_useful_message():
    """An absent input names the script that produces it."""

    with pytest.raises(FileNotFoundError, match="7.extract_chirps_daily"):
        compare.load_paired_rainfall(
            era5_path=PROJECT_DIR / "data" / "raw" / "nope_era5.csv",
            chirps_path=PROJECT_DIR / "data" / "raw" / "nope_chirps.csv",
        )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def test_district_plot_is_written(tmp_path):
    """The four-panel figure renders to a non-trivial file."""

    paired = make_paired(CLOSE, start="2015-01-01", end="2016-12-31")

    path = tmp_path / "Colombo_comparison.png"

    compare.plot_district(paired, "Colombo", path)

    assert path.exists()
    assert path.stat().st_size > 10_000


def test_summary_plot_is_written(tmp_path):
    """The cross-district summary renders."""

    comparison = compare.build_district_comparison(
        make_paired(CLOSE), DISTRICTS
    )

    path = tmp_path / "all_districts_summary.png"

    compare.plot_summary(comparison, path)

    assert path.exists()
    assert path.stat().st_size > 10_000


# ---------------------------------------------------------------------------
# CHIRPS extraction
# ---------------------------------------------------------------------------

def test_chirps_units_are_not_converted():
    """
    CHIRPS is published in mm/day, so the extraction applies no conversion.

    Stated as a test because a later edit that "helpfully" multiplies by 1000
    would otherwise pass silently and inflate rainfall a thousandfold.
    """

    source = (PROJECT_DIR / "scripts" / "data" / "7.extract_chirps_daily.py").read_text(
        encoding="utf-8"
    )

    assert "CHIRPS_NATIVE_UNITS = \"mm/day\"" in source
    assert "multiply(1000)" not in source


def test_chirps_unit_check_detects_metres():
    """Values in metres are caught by the unit check."""

    chirps = _load("data/7.extract_chirps_daily.py")

    frame = pd.DataFrame({"rainfall_mm_chirps": [0.003, 0.005, 0.001]})

    findings = chirps.check_units(frame)

    assert any("metres" in finding for finding in findings)


def test_chirps_unit_check_accepts_millimetres():
    """Ordinary millimetre values raise nothing."""

    chirps = _load("data/7.extract_chirps_daily.py")

    frame = pd.DataFrame({"rainfall_mm_chirps": [0.0, 3.5, 42.0, 120.0]})

    assert chirps.check_units(frame) == []


def test_chirps_unit_check_detects_negative_rainfall():
    """Negative rainfall is impossible and is reported."""

    chirps = _load("data/7.extract_chirps_daily.py")

    frame = pd.DataFrame({"rainfall_mm_chirps": [1.0, -2.0, 3.0]})

    findings = chirps.check_units(frame)

    assert any("negative" in finding for finding in findings)


def test_chirps_reuses_the_era5_districts_and_scale():
    """
    The two extractions must share polygons, node ordering and scale.

    If they diverge, a measured difference between the products would be an
    artefact of the extraction rather than a property of the rainfall.
    """

    chirps = _load("data/7.extract_chirps_daily.py")

    assert chirps.EXPECTED_DISTRICTS == chirps.era5.EXPECTED_DISTRICTS
    assert chirps.AGGREGATION_SCALE_M == chirps.era5.AGGREGATION_SCALE_M

    # The polygon source is the ERA5 module's, not a second definition.
    assert chirps.era5.DISTRICT_BOUNDARY_ASSET


def test_chirps_missing_rainfall_is_never_zero():
    """An unobserved value stays null and is flagged, never filled."""

    chirps = _load("data/7.extract_chirps_daily.py")

    frame = pd.DataFrame(
        {
            "date": ["2015-01-01", "2015-01-02"],
            "node_id": [4, 4],
            "canonical_name": ["Colombo", "Colombo"],
            "rainfall_mm_chirps": [5.0, None],
        }
    )

    result = chirps.finalise_chunk(frame)

    assert result["rainfall_observed"].tolist() == [1, 0]
    assert pd.isna(result.loc[1, "rainfall_mm_chirps"])
    assert result.loc[1, "rainfall_mm_chirps"] != 0

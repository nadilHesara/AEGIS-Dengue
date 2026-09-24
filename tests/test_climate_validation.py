"""
Tests for the daily district climate validator.

The climate CSV is produced by an Earth Engine extraction that cannot run in
CI, so these tests work on synthetic frames built from the real district
registry. Every check is exercised twice: once on clean data to confirm it
passes, and once on a deliberately broken frame to confirm it actually fires.

A validator that never fails is worse than no validator, so the negative cases
matter more than the positive ones.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "validate_climate", PROJECT_DIR / "scripts" / "data" / "6.validate_climate_data.py"
)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


PASS = validator.PASS
WARNING = validator.WARNING
FAIL = validator.FAIL


@pytest.fixture(scope="module")
def nodes():
    """The official district registry."""

    return validator.load_nodes()


def make_climate(nodes, start="2005-12-24", end="2006-06-30", seed=11):
    """
    Build a clean synthetic climate frame over the real 25 districts.

    The default span starts 52 weeks before the first dengue reporting period
    so that the history check passes.
    """

    generator = np.random.default_rng(seed)

    records = []

    for day in pd.date_range(start, end, freq="D"):
        for _, node in nodes.iterrows():
            records.append(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "node_id": int(node["node_id"]),
                    "canonical_name": node["canonical_name"],
                    "rainfall_mm": round(float(generator.gamma(1.2, 4.0)), 2),
                    "temperature_mean_c": 27.5,
                    "temperature_min_c": 23.8,
                    "temperature_max_c": 31.4,
                    "dewpoint_mean_c": 23.1,
                    "relative_humidity_mean": 77.5,
                    "wind_speed_mean": 3.2,
                }
            )

    frame = pd.DataFrame(records)
    frame["date_parsed"] = pd.to_datetime(frame["date"])

    return frame


@pytest.fixture
def clean(nodes):
    """A clean climate frame."""

    return make_climate(nodes)


def first_period_start():
    """The first dengue reporting period start, for the history check."""

    return validator.load_first_period_start()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_required_columns_are_specified():
    """The validator checks exactly the columns the schema requires."""

    assert validator.REQUIRED_COLUMNS == [
        "date",
        "node_id",
        "canonical_name",
        "rainfall_mm",
        "temperature_mean_c",
        "temperature_min_c",
        "temperature_max_c",
        "dewpoint_mean_c",
        "relative_humidity_mean",
        "wind_speed_mean",
    ]


def test_schema_passes_on_a_complete_frame(clean):
    """A frame with every required column passes."""

    status, findings = validator.check_schema(clean)

    assert status == PASS
    assert findings.empty


def test_schema_fails_on_a_missing_column(clean):
    """A dropped column is reported by name."""

    status, findings = validator.check_schema(
        clean.drop(columns=["relative_humidity_mean"])
    )

    assert status == FAIL
    assert findings["missing_column"].tolist() == ["relative_humidity_mean"]


def test_validation_aborts_when_a_key_column_is_absent(clean, nodes):
    """Without canonical_name the remaining checks cannot run."""

    result = validator.validate_climate_data(
        clean.drop(columns=["canonical_name"]), nodes
    )

    assert result["aborted"]
    assert result["checks"]["schema"][0] == FAIL


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

def test_dates_pass_when_all_parse(clean):
    """Well-formed dates pass."""

    status, findings = validator.check_dates(clean)

    assert status == PASS
    assert findings.empty


def test_unparseable_date_is_reported(clean):
    """A malformed date is a FAIL and is listed."""

    broken = clean.copy()
    broken.loc[broken.index[0], "date"] = "not-a-date"
    broken["date_parsed"] = pd.to_datetime(broken["date"], errors="coerce")

    status, findings = validator.check_dates(broken)

    assert status == FAIL
    assert "not-a-date" in findings["date"].tolist()


# ---------------------------------------------------------------------------
# Districts and node ids
# ---------------------------------------------------------------------------

def test_all_twenty_five_districts_pass(clean, nodes):
    """The real 25 districts pass the completeness check."""

    assert len(nodes) == 25

    status, findings = validator.check_districts(clean, nodes)

    assert status == PASS
    assert findings.empty


def test_missing_district_is_reported(clean, nodes):
    """A district absent from the file is a FAIL."""

    status, findings = validator.check_districts(
        clean.loc[~clean["canonical_name"].eq("Colombo")], nodes
    )

    assert status == FAIL
    assert "Colombo" in findings["canonical_name"].tolist()


def test_unexpected_district_is_reported(clean, nodes):
    """A name that is not an official district is a FAIL."""

    broken = clean.copy()
    broken.loc[broken.index[0], "canonical_name"] = "Atlantis"

    status, findings = validator.check_districts(broken, nodes)

    assert status == FAIL
    assert "Atlantis" in findings["canonical_name"].tolist()


def test_node_id_mapping_passes_on_correct_pairs(clean, nodes):
    """Correct name-to-id pairs pass."""

    status, findings = validator.check_node_id_mapping(clean, nodes)

    assert status == PASS
    assert findings.empty


def test_wrong_node_id_is_reported(clean, nodes):
    """A district carrying another district's id is a FAIL."""

    broken = clean.copy()

    colombo = broken["canonical_name"].eq("Colombo")
    broken.loc[colombo, "node_id"] = 99

    status, findings = validator.check_node_id_mapping(broken, nodes)

    assert status == FAIL
    assert "Colombo" in findings["canonical_name"].tolist()


def test_district_under_two_node_ids_is_reported(clean, nodes):
    """One name appearing under two ids is a FAIL."""

    broken = clean.copy()

    index = broken.loc[broken["canonical_name"].eq("Kandy")].index[0]
    broken.loc[index, "node_id"] = 77

    status, findings = validator.check_node_id_mapping(broken, nodes)

    assert status == FAIL
    assert "Kandy" in findings["canonical_name"].tolist()


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------

def test_no_duplicates_passes(clean):
    """A frame keyed uniquely by date and district passes."""

    status, findings = validator.check_duplicates(clean)

    assert status == PASS
    assert findings.empty


def test_duplicate_district_date_is_reported(clean):
    """A repeated date and district pair is a FAIL."""

    broken = pd.concat([clean, clean.iloc[[0]]], ignore_index=True)

    status, findings = validator.check_duplicates(broken)

    assert status == FAIL
    assert len(findings) == 1
    assert findings.iloc[0]["row_count"] == 2


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_complete_grid_has_no_missing_rows(clean, nodes):
    """A complete date-by-district grid passes."""

    status, findings = validator.check_missing_rows(clean, nodes)

    assert status == PASS
    assert findings.empty


def test_missing_district_date_is_reported(clean, nodes):
    """An absent district-date is a FAIL and is named."""

    broken = clean.loc[
        ~(
            clean["date"].eq("2006-03-10")
            & clean["canonical_name"].eq("Colombo")
        )
    ]

    status, findings = validator.check_missing_rows(broken, nodes)

    assert status == FAIL
    assert len(findings) == 1
    assert findings.iloc[0]["canonical_name"] == "Colombo"


def test_complete_days_pass(clean, nodes):
    """Every date carrying 25 districts passes."""

    status, findings = validator.check_complete_days(clean, nodes)

    assert status == PASS
    assert findings.empty


def test_incomplete_day_is_reported(clean, nodes):
    """A date missing a district is reported with its count."""

    broken = clean.loc[
        ~(
            clean["date"].eq("2006-03-10")
            & clean["canonical_name"].eq("Colombo")
        )
    ]

    status, findings = validator.check_complete_days(broken, nodes)

    assert status == FAIL
    assert findings.iloc[0]["districts_present"] == 24
    assert findings.iloc[0]["districts_expected"] == 25


def test_long_missing_run_is_reported(clean, nodes):
    """A consecutive gap is reported with its start, end and length."""

    gap = pd.date_range("2006-04-01", "2006-04-08").strftime("%Y-%m-%d")

    broken = clean.loc[~clean["date"].isin(gap)]

    status, findings = validator.check_long_gaps(broken, nodes)

    assert status == FAIL

    # Every district loses the same eight days.
    assert len(findings) == 25
    assert set(findings["gap_days"]) == {8}
    assert set(findings["reason"]) == {"rows absent"}


def test_short_gap_is_below_the_reporting_threshold(clean, nodes):
    """A gap shorter than the threshold is not reported as a run."""

    gap = pd.date_range("2006-04-01", "2006-04-02").strftime("%Y-%m-%d")

    broken = clean.loc[~clean["date"].isin(gap)]

    _, findings = validator.check_long_gaps(broken, nodes)

    assert findings.empty


def test_null_run_is_distinguished_from_absent_run(clean, nodes):
    """A run of present-but-null rows is labelled differently from absence."""

    broken = clean.copy()

    window = pd.date_range("2006-04-01", "2006-04-08").strftime("%Y-%m-%d")

    target = broken["date"].isin(window) & broken["canonical_name"].eq("Galle")

    broken.loc[target, "rainfall_mm"] = np.nan

    _, findings = validator.check_long_gaps(broken, nodes)

    galle = findings.loc[findings["canonical_name"].eq("Galle")]

    assert len(galle) == 1
    assert galle.iloc[0]["reason"] == "rows present, values null"


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def test_preferred_history_passes(nodes):
    """52 weeks of history before the first dengue period passes."""

    start = first_period_start() - pd.Timedelta(weeks=52)

    frame = make_climate(
        nodes,
        start=start.strftime("%Y-%m-%d"),
        end=(start + pd.Timedelta(days=400)).strftime("%Y-%m-%d"),
    )

    status, _ = validator.check_history(frame, first_period_start())

    assert status == PASS


def test_short_history_warns(nodes):
    """Between 26 and 52 weeks is a WARNING, not a failure."""

    start = first_period_start() - pd.Timedelta(weeks=30)

    frame = make_climate(
        nodes,
        start=start.strftime("%Y-%m-%d"),
        end=(start + pd.Timedelta(days=300)).strftime("%Y-%m-%d"),
    )

    status, findings = validator.check_history(frame, first_period_start())

    assert status == WARNING
    assert findings.iloc[0]["history_weeks"] == pytest.approx(30.0, abs=0.2)


def test_insufficient_history_fails(nodes):
    """Under the 26-week minimum is a FAIL."""

    start = first_period_start() - pd.Timedelta(weeks=4)

    frame = make_climate(
        nodes,
        start=start.strftime("%Y-%m-%d"),
        end=(start + pd.Timedelta(days=200)).strftime("%Y-%m-%d"),
    )

    status, findings = validator.check_history(frame, first_period_start())

    assert status == FAIL
    assert not findings.empty


# ---------------------------------------------------------------------------
# Value ranges
# ---------------------------------------------------------------------------

def test_plausible_values_pass(clean):
    """Ordinary tropical values raise nothing."""

    invalid_status, invalid = validator.check_invalid_values(clean)
    suspicious_status, suspicious = validator.check_suspicious_values(clean)

    assert invalid_status == PASS
    assert invalid.empty
    assert suspicious_status == PASS
    assert suspicious.empty


@pytest.mark.parametrize(
    "column,value",
    [
        ("rainfall_mm", -3.0),
        ("rainfall_mm", 2500.0),
        ("relative_humidity_mean", -5.0),
        ("relative_humidity_mean", 105.0),
        ("wind_speed_mean", -1.0),
        ("temperature_mean_c", 80.0),
    ],
)
def test_impossible_value_is_a_failure(clean, column, value):
    """Physically impossible values are FAIL."""

    broken = clean.copy()
    broken.loc[broken.index[0], column] = value

    status, findings = validator.check_invalid_values(broken)

    assert status == FAIL
    assert column in findings["column"].tolist()


def test_min_above_max_is_a_failure(clean):
    """Minimum temperature above maximum is impossible."""

    broken = clean.copy()
    broken.loc[broken.index[0], "temperature_min_c"] = 40.0

    status, findings = validator.check_invalid_values(broken)

    assert status == FAIL
    assert "minimum temperature above maximum temperature" in (
        findings["issue"].tolist()
    )


def test_mean_outside_min_and_max_is_a_failure(clean):
    """A mean outside the daily range is impossible."""

    broken = clean.copy()
    broken.loc[broken.index[0], "temperature_mean_c"] = 35.0

    status, findings = validator.check_invalid_values(broken)

    assert status == FAIL
    assert "mean temperature above maximum temperature" in (
        findings["issue"].tolist()
    )


def test_dewpoint_above_temperature_is_a_failure(clean):
    """Dewpoint above air temperature usually means transposed bands."""

    broken = clean.copy()
    broken.loc[broken.index[0], "dewpoint_mean_c"] = 35.0

    status, findings = validator.check_invalid_values(broken)

    assert status == FAIL
    assert "dewpoint above air temperature" in findings["issue"].tolist()


@pytest.mark.parametrize(
    "column,value",
    [
        ("rainfall_mm", 500.0),
        ("wind_speed_mean", 30.0),
    ],
)
def test_unusual_but_possible_value_is_a_warning(clean, column, value):
    """Extremes that are physically possible warn rather than fail."""

    broken = clean.copy()
    broken.loc[broken.index[0], column] = value

    invalid_status, _ = validator.check_invalid_values(broken)
    status, findings = validator.check_suspicious_values(broken)

    assert invalid_status == PASS, "a possible value must not be invalid"
    assert status == WARNING
    assert column in findings["column"].tolist()


def test_unusually_cold_but_coherent_day_is_a_warning(clean):
    """
    A cold day warns only when the whole temperature triple stays coherent.

    Lowering the mean alone would put it below the minimum, which is
    impossible rather than merely unusual, so the min and max move with it.
    """

    broken = clean.copy()
    index = broken.index[0]

    broken.loc[index, "temperature_min_c"] = 7.0
    broken.loc[index, "temperature_mean_c"] = 10.0
    broken.loc[index, "temperature_max_c"] = 14.0
    broken.loc[index, "dewpoint_mean_c"] = 6.0

    invalid_status, invalid = validator.check_invalid_values(broken)
    status, findings = validator.check_suspicious_values(broken)

    assert invalid_status == PASS, invalid.to_string(index=False)
    assert status == WARNING

    flagged = set(findings["column"])

    assert "temperature_mean_c" in flagged
    assert "temperature_min_c" in flagged


def test_impossible_value_is_not_double_reported(clean):
    """A value reported as impossible is not repeated as suspicious."""

    broken = clean.copy()
    broken.loc[broken.index[0], "rainfall_mm"] = 2500.0

    _, invalid = validator.check_invalid_values(broken)
    _, suspicious = validator.check_suspicious_values(broken)

    assert len(invalid) == 1
    assert suspicious.empty


def test_suspicious_bounds_sit_inside_invalid_bounds():
    """Every suspicious band must be narrower than its invalid band."""

    for column, (low, high) in validator.SUSPICIOUS_BOUNDS.items():
        invalid_low, invalid_high = validator.INVALID_BOUNDS[column]

        assert invalid_low <= low, column
        assert high <= invalid_high, column


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def test_correct_units_pass(clean):
    """Celsius, millimetres and percent pass."""

    status, findings = validator.check_units(clean)

    assert status == PASS
    assert findings.empty


def test_kelvin_temperatures_are_detected(clean):
    """Temperatures left in Kelvin are caught by their magnitude."""

    broken = clean.copy()

    for column in [
        "temperature_mean_c",
        "temperature_min_c",
        "temperature_max_c",
    ]:
        broken[column] = broken[column] + 273.15

    status, findings = validator.check_units(broken)

    assert status == FAIL
    assert "Kelvin" in " ".join(findings["issue"])


def test_rainfall_left_in_metres_is_detected(clean):
    """Rainfall in metres is a thousand times too small."""

    broken = clean.copy()
    broken["rainfall_mm"] = broken["rainfall_mm"] / 1000.0

    status, findings = validator.check_units(broken)

    assert status == FAIL
    assert "metres" in " ".join(findings["issue"])


def test_humidity_as_fraction_is_detected(clean):
    """Humidity on a 0-1 scale is caught."""

    broken = clean.copy()
    broken["relative_humidity_mean"] = 0.775

    status, findings = validator.check_units(broken)

    assert status == FAIL
    assert "fraction" in " ".join(findings["issue"])


# ---------------------------------------------------------------------------
# Present but null
# ---------------------------------------------------------------------------

def test_fully_populated_rows_pass(clean):
    """No nulls means nothing to report."""

    status, findings = validator.check_present_but_null(clean)

    assert status == PASS
    assert findings.empty


def test_null_value_in_a_present_row_warns(clean):
    """A present row with a null value is a WARNING, not a FAIL."""

    broken = clean.copy()
    broken.loc[broken.index[0], "rainfall_mm"] = np.nan

    status, findings = validator.check_present_but_null(broken)

    assert status == WARNING
    assert findings.iloc[0]["column"] == "rainfall_mm"


# ---------------------------------------------------------------------------
# Coverage reports
# ---------------------------------------------------------------------------

def test_coverage_by_district_covers_every_district(clean, nodes):
    """The district report holds one row per official district."""

    coverage = validator.build_coverage_by_district(clean, nodes)

    assert len(coverage) == 25
    assert set(coverage["canonical_name"]) == set(nodes["canonical_name"])
    assert coverage["dates_missing"].eq(0).all()


def test_coverage_by_district_counts_missing_dates(clean, nodes):
    """A district missing a date is counted."""

    broken = clean.loc[
        ~(
            clean["date"].eq("2006-03-10")
            & clean["canonical_name"].eq("Colombo")
        )
    ]

    coverage = validator.build_coverage_by_district(broken, nodes)

    colombo = coverage.loc[coverage["canonical_name"].eq("Colombo")].iloc[0]

    assert colombo["dates_missing"] == 1


def test_coverage_by_year_matches_expected_rows(clean, nodes):
    """A complete frame reports present rows equal to expected rows."""

    coverage = validator.build_coverage_by_year(clean, nodes)

    assert not coverage.empty
    assert (coverage["rows_present"] == coverage["rows_expected"]).all()
    assert coverage["districts_present"].eq(25).all()


def test_missing_by_variable_reports_each_column(clean):
    """The variable report covers every weather column with its units."""

    report = validator.build_missing_by_variable(clean)

    assert set(report["column"]) == set(validator.WEATHER_COLUMNS)
    assert report["null_count"].eq(0).all()
    assert report["expected_units"].notna().all()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def test_clean_frame_passes_every_check(clean, nodes):
    """A clean frame yields no FAIL and no WARNING."""

    result = validator.validate_climate_data(
        clean, nodes, first_period_start()
    )

    assert not result["aborted"]
    assert validator.count_failures(result) == 0

    statuses = {
        name: status for name, (status, _) in result["checks"].items()
    }

    assert set(statuses.values()) == {PASS}, statuses


def test_broken_frame_is_classified_fail(clean, nodes):
    """A frame with real defects reports them."""

    broken = clean.copy()
    broken.loc[broken.index[0], "rainfall_mm"] = -5.0

    result = validator.validate_climate_data(
        broken, nodes, first_period_start()
    )

    assert validator.count_failures(result) >= 1
    assert result["checks"]["invalid_values"][0] == FAIL


def test_summary_markdown_names_every_check(clean, nodes):
    """The summary lists each check and its classification."""

    result = validator.validate_climate_data(
        clean, nodes, first_period_start()
    )

    summary = validator.build_summary_markdown(result)

    for name in result["checks"]:
        assert f"`{name}`" in summary

    assert "PASS" in summary
    assert "never modified" in summary


def test_outputs_are_written(clean, nodes, tmp_path, monkeypatch):
    """Every required output file is produced."""

    monkeypatch.setattr(validator, "RESULTS_DIR", tmp_path)

    for attribute, name in [
        ("SUMMARY_PATH", "weather_validation_summary.md"),
        ("MISSING_DATES_PATH", "missing_district_dates.csv"),
        ("DUPLICATES_PATH", "duplicate_district_dates.csv"),
        ("INVALID_VALUES_PATH", "invalid_weather_values.csv"),
        ("COVERAGE_DISTRICT_PATH", "weather_coverage_by_district.csv"),
        ("COVERAGE_YEAR_PATH", "weather_coverage_by_year.csv"),
    ]:
        monkeypatch.setattr(validator, attribute, tmp_path / name)

    result = validator.validate_climate_data(
        clean, nodes, first_period_start()
    )

    validator.write_outputs(result)

    for name in [
        "weather_validation_summary.md",
        "missing_district_dates.csv",
        "duplicate_district_dates.csv",
        "invalid_weather_values.csv",
        "weather_coverage_by_district.csv",
        "weather_coverage_by_year.csv",
    ]:
        assert (tmp_path / name).exists(), name


def test_missing_climate_file_raises():
    """A absent climate CSV raises a message naming the extraction script."""

    with pytest.raises(FileNotFoundError, match="5.extract_era5_daily"):
        validator.load_climate(PROJECT_DIR / "data" / "raw" / "nope.csv")


# ---------------------------------------------------------------------------
# Real file, when it exists
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not validator.CLIMATE_PATH.exists(),
    reason="climate_daily_district.csv not extracted yet",
)
def test_real_climate_file_schema(nodes):
    """The real extracted file carries every required column."""

    climate = validator.load_climate()

    status, findings = validator.check_schema(climate)

    assert status == PASS, findings["missing_column"].tolist()


@pytest.mark.skipif(
    not validator.CLIMATE_PATH.exists(),
    reason="climate_daily_district.csv not extracted yet",
)
def test_real_climate_file_districts_and_keys(nodes):
    """The real file holds the 25 official districts, keyed uniquely."""

    climate = validator.load_climate()

    assert validator.check_districts(climate, nodes)[0] == PASS
    assert validator.check_node_id_mapping(climate, nodes)[0] == PASS
    assert validator.check_duplicates(climate)[0] == PASS


@pytest.mark.skipif(
    not validator.CLIMATE_PATH.exists(),
    reason="climate_daily_district.csv not extracted yet",
)
def test_real_climate_file_values_are_possible(nodes):
    """The real file holds no physically impossible value."""

    climate = validator.load_climate()

    status, findings = validator.check_invalid_values(climate)

    assert status == PASS, findings.head(20).to_string(index=False)


@pytest.mark.skipif(
    not validator.CLIMATE_PATH.exists(),
    reason="climate_daily_district.csv not extracted yet",
)
def test_real_climate_file_units(nodes):
    """The real file is in Celsius, millimetres and percent."""

    climate = validator.load_climate()

    status, findings = validator.check_units(climate)

    assert status == PASS, findings.to_string(index=False)

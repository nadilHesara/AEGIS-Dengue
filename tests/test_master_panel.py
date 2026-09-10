"""
Tests for the master weekly modelling panel.

Synthetic inputs verify the structural cross-join, the dengue/climate merge
keys, and the missing-data flags. A real-data integration test checks the
known Puttalam absence in 2026 week 7 and the full 25-district structural grid.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]

REAL_DATA_FILES = [
    PROJECT_ROOT / "data" / "interim" / "reporting_calendar.csv",
    PROJECT_ROOT / "data" / "processed" / "nodes.csv",
    PROJECT_ROOT / "data" / "interim" / "dengue_weekly_canonical.parquet",
    PROJECT_ROOT / "data" / "interim" / "climate_by_dengue_period.parquet",
]

spec = importlib.util.spec_from_file_location(
    "master_panel",
    PROJECT_DIR / "scripts" / "data" / "10.create_master_panel.py",
)
master = importlib.util.module_from_spec(spec)
spec.loader.exec_module(master)


@pytest.fixture(scope="module")
def calendar():
    return master.load_calendar()


@pytest.fixture(scope="module")
def nodes():
    return master.load_nodes()


@pytest.fixture(scope="module")
def dengue():
    return master.load_dengue()


@pytest.fixture(scope="module")
def climate():
    return master.load_climate()


def make_calendar(periods):
    records = []
    for period_id, source_year, source_week, start, end in periods:
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)
        records.append(
            {
                "period_id": period_id,
                "source_year": source_year,
                "source_week": source_week,
                "start_date": start_date,
                "end_date": end_date,
                "reporting_days": (end_date - start_date).days + 1,
                "weekday_convention": "saturday_friday",
                "is_weekday_convention_change": False,
                "is_irregular_period": False,
                "calendar_gap_before_days": 0,
                "has_calendar_gap_before": False,
            }
        )
    return pd.DataFrame(records)


def make_nodes():
    return pd.DataFrame(
        {
            "node_id": range(25),
            "canonical_name": [
                "Ampara",
                "Anuradhapura",
                "Badulla",
                "Batticaloa",
                "Colombo",
                "Galle",
                "Gampaha",
                "Hambantota",
                "Jaffna",
                "Kalutara",
                "Kandy",
                "Kegalle",
                "Kilinochchi",
                "Kurunegala",
                "Mannar",
                "Matale",
                "Matara",
                "Moneragala",
                "Mullaitivu",
                "Nuwara Eliya",
                "Polonnaruwa",
                "Puttalam",
                "Ratnapura",
                "Trincomalee",
                "Vavuniya",
            ],
            "province": ["Test"] * 25,
        }
    )


def make_dengue(calendar, nodes, missing_name=None):
    records = []
    for _, period in calendar.iterrows():
        for _, node in nodes.iterrows():
            if (
                missing_name == "Puttalam"
                and period["source_year"] == 2026
                and period["source_week"] == 7
                and node["canonical_name"] == "Puttalam"
            ):
                continue
            records.append(
                {
                    "period_id": int(period["period_id"]),
                    "node_id": int(node["node_id"]),
                    "cases": 1,
                    "case_observed": 1,
                    "source_reporting_areas_used": node["canonical_name"],
                    "source_row_count": 1,
                }
            )
    return pd.DataFrame(records)


def make_climate(calendar, nodes):
    records = []
    for _, period in calendar.iterrows():
        for _, node in nodes.iterrows():
            records.append(
                {
                    "period_id": int(period["period_id"]),
                    "node_id": int(node["node_id"]),
                    "rainfall_sum_mm": 10.0,
                    "rainfall_daily_mean_mm": 1.0,
                    "rainy_days": 2,
                    "temperature_mean_c": 27.0,
                    "temperature_min_c": 24.0,
                    "temperature_max_c": 31.0,
                    "dewpoint_mean_c": 23.0,
                    "relative_humidity_mean": 80.0,
                    "wind_speed_mean": 3.0,
                    "weather_days_available": 7,
                    "expected_weather_days": 7,
                    "weather_coverage_ratio": 1.0,
                    "weather_complete": True,
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Structural grid
# ---------------------------------------------------------------------------

def test_structural_grid_cross_joins_periods_and_nodes():
    cal = make_calendar(
        [
            (1, 2017, 1, "2017-01-01", "2017-01-07"),
            (2, 2017, 2, "2017-01-08", "2017-01-14"),
        ]
    )
    nds = make_nodes()

    grid = master.build_structural_grid(cal, nds)

    assert len(grid) == 50
    assert grid.groupby("period_id").size().eq(25).all()
    assert grid["period_id"].is_monotonic_increasing
    assert grid["node_id"].isin(range(25)).all()


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def test_master_panel_joins_on_period_id_and_node_id_only():
    cal = make_calendar(
        [
            (1, 1999, 99, "2017-01-01", "2017-01-07"),
        ]
    )
    nds = make_nodes()
    dengue_frame = make_dengue(cal, nds)
    climate_frame = make_climate(cal, nds)

    panel = master.build_master_panel(cal, nds, dengue_frame, climate_frame)

    assert len(panel) == 25
    assert panel["source_year"].eq(1999).all()
    assert panel["source_week"].eq(99).all()


# ---------------------------------------------------------------------------
# Missingness flags
# ---------------------------------------------------------------------------

def test_missing_puttalam_is_retained_as_missing_case():
    cal = make_calendar(
        [
            (1, 2026, 7, "2026-02-09", "2026-02-15"),
        ]
    )
    nds = make_nodes()
    dengue_frame = make_dengue(cal, nds, missing_name="Puttalam")
    climate_frame = make_climate(cal, nds)

    panel = master.build_master_panel(cal, nds, dengue_frame, climate_frame)

    assert len(panel) == 25

    week_7 = panel.loc[panel["period_id"].eq(1)]
    puttalam = week_7.loc[week_7["canonical_name"].eq("Puttalam")].iloc[0]

    assert pd.isna(puttalam["cases"])
    assert int(puttalam["case_observed"]) == 0
    assert int(week_7["case_observed"].sum()) == 24
    assert int(week_7["row_has_complete_cases"].sum()) == 24
    assert int(week_7["row_has_complete_climate"].sum()) == 25
    assert int(week_7["row_is_fully_observed"].sum()) == 24


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not all(path.exists() for path in REAL_DATA_FILES),
    reason="Full generated datasets are not available in CI",
)

def test_real_inputs_build_a_complete_structural_panel(calendar, nodes, dengue, climate):
    panel = master.build_master_panel(calendar, nodes, dengue, climate)

    assert len(panel) == len(calendar) * len(nodes)
    assert panel.groupby("period_id").size().eq(25).all()
    assert panel.equals(panel.sort_values(["period_id", "node_id"]).reset_index(drop=True))
    assert int(panel.loc[panel["source_year"].eq(2026) & panel["source_week"].eq(7), "case_observed"].sum()) == 24
    assert panel.loc[
        panel["source_year"].eq(2026)
        & panel["source_week"].eq(7)
        & panel["canonical_name"].eq("Puttalam")
    , "cases"].isna().all()
    assert panel.loc[
        panel["source_year"].eq(2026)
        & panel["source_week"].eq(7)
        & panel["canonical_name"].eq("Puttalam")
    , "case_observed"].eq(0).all()
    assert panel["row_is_fully_observed"].isin([0, 1]).all()
    assert panel["row_has_complete_climate"].isin([0, 1]).all()
    assert panel["row_has_complete_cases"].isin([0, 1]).all()

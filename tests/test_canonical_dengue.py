"""
Tests for the canonical district-level dengue dataset.

Synthetic frames cover the district mapping and the Kalmune aggregation in
isolation. The real-data tests confirm that the confirmed source findings
survive the build: Kalmune folded into Ampara, and Puttalam still absent from
2026 week 7.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location(
    "create_canonical",
    PROJECT_DIR / "scripts" / "4.create_canonical_dengue.py",
)
canonical_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canonical_module)


SOURCE_AREAS = sorted(canonical_module.DENGUE_TO_CANONICAL)


def make_period(year, week, start, end, areas=None, cases=5):
    """Build the source rows for one reporting period."""

    areas = SOURCE_AREAS if areas is None else areas

    return pd.DataFrame(
        {
            "year": year,
            "week": week,
            "start.date": start,
            "end.date": end,
            "district": areas,
            "cases": cases,
        }
    )


def make_calendar(*periods):
    """Build a calendar covering the given periods, keyed chronologically."""

    records = []

    for index, (year, week, start, end) in enumerate(periods, start=1):
        start_date = pd.Timestamp(start)
        end_date = pd.Timestamp(end)

        records.append(
            {
                "period_id": index,
                "source_year": year,
                "source_week": week,
                "start_date": start_date,
                "end_date": end_date,
                "reporting_days": (end_date - start_date).days + 1,
                "weekday_convention": "saturday_friday",
                "is_irregular_period": False,
                "calendar_gap_before_days": 0,
            }
        )

    return pd.DataFrame(records)


def make_nodes():
    """Build a node registry for the 25 canonical districts."""

    names = sorted(set(canonical_module.DENGUE_TO_CANONICAL.values()))

    return pd.DataFrame(
        {
            "node_id": range(len(names)),
            "canonical_name": names,
            "province": "Test Province",
        }
    )


def build(raw, calendar, nodes=None):
    """Run the full canonical build over synthetic inputs."""

    nodes = make_nodes() if nodes is None else nodes

    raw = raw.copy()
    raw["start_date"] = pd.to_datetime(raw["start.date"], format="%m/%d/%Y")
    raw["end_date"] = pd.to_datetime(raw["end.date"], format="%m/%d/%Y")
    raw["district"] = raw["district"].astype(str).str.strip()

    rows = canonical_module.join_reporting_calendar(raw, calendar)
    rows = canonical_module.add_canonical_names(rows)

    result = canonical_module.aggregate_to_canonical(rows, nodes)
    coverage = canonical_module.build_canonical_coverage(result, calendar)

    return rows, result, coverage


# ---------------------------------------------------------------------------
# District mapping
# ---------------------------------------------------------------------------

def test_mapping_covers_every_source_area():
    """All 26 source reporting areas have an explicit mapping."""

    assert len(canonical_module.DENGUE_TO_CANONICAL) == 26

    canonical_names = set(canonical_module.DENGUE_TO_CANONICAL.values())

    assert len(canonical_names) == 25


def test_required_explicit_mappings():
    """The spelling and merge mappings are exactly as specified."""

    mapping = canonical_module.DENGUE_TO_CANONICAL

    assert mapping["Ampara"] == "Ampara"
    assert mapping["Kalmune"] == "Ampara"
    assert mapping["Hambanthota"] == "Hambantota"
    assert mapping["Monaragala"] == "Moneragala"
    assert mapping["NuwaraEliya"] == "Nuwara Eliya"


def test_kalmune_is_not_a_canonical_district():
    """Kalmune is a source name only, never a model node."""

    assert "Kalmune" not in set(
        canonical_module.DENGUE_TO_CANONICAL.values()
    )


def test_unmapped_source_area_is_rejected():
    """An unknown reporting area raises rather than being guessed at."""

    df = make_period(
        2015, 10, "3/7/2015", "3/13/2015", areas=["Colombo", "Atlantis"]
    )

    df["start_date"] = pd.to_datetime(df["start.date"], format="%m/%d/%Y")
    df["end_date"] = pd.to_datetime(df["end.date"], format="%m/%d/%Y")

    with pytest.raises(ValueError, match="no explicit mapping"):
        canonical_module.add_canonical_names(df)


def test_mapping_report_names_unmapped_areas():
    """The report lists source areas with no mapping."""

    df = make_period(
        2015, 10, "3/7/2015", "3/13/2015", areas=["Colombo", "Atlantis"]
    )

    report = canonical_module.check_district_mapping(df)

    assert report["unmapped"] == ["Atlantis"]


def test_mapping_report_names_obsolete_mappings():
    """The report lists mapped areas absent from the source."""

    df = make_period(2015, 10, "3/7/2015", "3/13/2015", areas=["Colombo"])

    report = canonical_module.check_district_mapping(df)

    assert "Kalmune" in report["obsolete"]
    assert "Colombo" not in report["obsolete"]


def test_mapping_report_names_aggregated_districts():
    """The report shows Kalmune and Ampara sharing one district."""

    df = make_period(2015, 10, "3/7/2015", "3/13/2015")

    report = canonical_module.check_district_mapping(df)

    assert report["duplicates"] == {"Ampara": ["Ampara", "Kalmune"]}


def test_mapping_report_detects_inconsistent_spellings():
    """Names differing only by spacing or case are reported."""

    df = make_period(
        2015,
        10,
        "3/7/2015",
        "3/13/2015",
        areas=["NuwaraEliya", "Nuwara Eliya"],
    )

    report = canonical_module.check_district_mapping(df)

    assert report["inconsistent_spellings"]

    variants = list(report["inconsistent_spellings"].values())[0]

    assert set(variants) == {"NuwaraEliya", "Nuwara Eliya"}


def test_every_source_area_maps_to_a_node():
    """Each canonical name from the mapping exists in the node registry."""

    nodes = make_nodes()

    canonical_names = set(canonical_module.DENGUE_TO_CANONICAL.values())

    assert canonical_names == set(nodes["canonical_name"])


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_kalmune_is_summed_into_ampara():
    """Ampara's count is the sum of the Ampara and Kalmune source rows."""

    df = pd.concat(
        [
            make_period(
                2015, 10, "3/7/2015", "3/13/2015",
                areas=["Ampara"], cases=3,
            ),
            make_period(
                2015, 10, "3/7/2015", "3/13/2015",
                areas=["Kalmune"], cases=4,
            ),
        ],
        ignore_index=True,
    )

    calendar = make_calendar((2015, 10, "2015-03-07", "2015-03-13"))

    _, canonical, _ = build(df, calendar)

    ampara = canonical.loc[canonical["canonical_name"].eq("Ampara")]

    assert len(ampara) == 1

    row = ampara.iloc[0]

    assert row["cases"] == 7
    assert row["source_row_count"] == 2
    assert row["source_reporting_areas_used"] == "Ampara|Kalmune"
    assert row["case_observed"] == 1


def test_ampara_alone_is_not_treated_as_aggregated():
    """When only one source row exists the count reflects that."""

    df = make_period(
        2015, 10, "3/7/2015", "3/13/2015", areas=["Ampara"], cases=3
    )

    calendar = make_calendar((2015, 10, "2015-03-07", "2015-03-13"))

    _, canonical, _ = build(df, calendar)

    row = canonical.iloc[0]

    assert row["canonical_name"] == "Ampara"
    assert row["cases"] == 3
    assert row["source_row_count"] == 1
    assert row["source_reporting_areas_used"] == "Ampara"


def test_aggregation_is_scoped_to_one_period():
    """Kalmune cases never leak across reporting periods."""

    df = pd.concat(
        [
            make_period(
                2015, 10, "3/7/2015", "3/13/2015",
                areas=["Ampara", "Kalmune"], cases=1,
            ),
            make_period(
                2015, 11, "3/14/2015", "3/20/2015",
                areas=["Ampara", "Kalmune"], cases=10,
            ),
        ],
        ignore_index=True,
    )

    calendar = make_calendar(
        (2015, 10, "2015-03-07", "2015-03-13"),
        (2015, 11, "2015-03-14", "2015-03-20"),
    )

    _, canonical, _ = build(df, calendar)

    ampara = canonical.loc[
        canonical["canonical_name"].eq("Ampara")
    ].sort_values("period_id")

    assert list(ampara["cases"]) == [2, 20]


def test_spelling_variants_map_without_aggregating():
    """A renamed district keeps its own row."""

    df = make_period(
        2015,
        10,
        "3/7/2015",
        "3/13/2015",
        areas=["Hambanthota", "Monaragala", "NuwaraEliya"],
        cases=2,
    )

    calendar = make_calendar((2015, 10, "2015-03-07", "2015-03-13"))

    _, canonical, _ = build(df, calendar)

    names = set(canonical["canonical_name"])

    assert names == {"Hambantota", "Moneragala", "Nuwara Eliya"}
    assert canonical["source_row_count"].eq(1).all()
    assert canonical["cases"].eq(2).all()


def test_one_row_per_period_and_district():
    """The output is keyed by period_id and canonical district."""

    df = pd.concat(
        [
            make_period(2015, 10, "3/7/2015", "3/13/2015"),
            make_period(2015, 11, "3/14/2015", "3/20/2015"),
        ],
        ignore_index=True,
    )

    calendar = make_calendar(
        (2015, 10, "2015-03-07", "2015-03-13"),
        (2015, 11, "2015-03-14", "2015-03-20"),
    )

    _, canonical, _ = build(df, calendar)

    assert len(canonical) == 2 * 25

    assert not canonical.duplicated(
        subset=["period_id", "canonical_name"]
    ).any()


def test_output_is_sorted_by_period_id():
    """The dataset is ordered by the chronological key."""

    df = pd.concat(
        [
            make_period(2015, 11, "3/14/2015", "3/20/2015"),
            make_period(2015, 10, "3/7/2015", "3/13/2015"),
        ],
        ignore_index=True,
    )

    calendar = make_calendar(
        (2015, 10, "2015-03-07", "2015-03-13"),
        (2015, 11, "2015-03-14", "2015-03-20"),
    )

    _, canonical, _ = build(df, calendar)

    assert canonical["period_id"].is_monotonic_increasing


def test_case_totals_are_preserved():
    """Aggregation neither invents nor loses cases."""

    df = make_period(2015, 10, "3/7/2015", "3/13/2015", cases=3)

    calendar = make_calendar((2015, 10, "2015-03-07", "2015-03-13"))

    _, canonical, _ = build(df, calendar)

    assert canonical["cases"].sum() == df["cases"].sum()


# ---------------------------------------------------------------------------
# Missing observations
# ---------------------------------------------------------------------------

def test_missing_district_produces_no_row():
    """An absent district yields no observation, not a zero."""

    without_puttalam = [a for a in SOURCE_AREAS if a != "Puttalam"]

    df = make_period(
        2026, 7, "2/9/2026", "2/15/2026", areas=without_puttalam
    )

    calendar = make_calendar((2026, 7, "2026-02-09", "2026-02-15"))

    _, canonical, coverage = build(df, calendar)

    assert canonical.loc[
        canonical["canonical_name"].eq("Puttalam")
    ].empty

    assert len(canonical) == 24

    issue = coverage.iloc[0]

    assert issue["observed_canonical_districts"] == 24
    assert issue["missing_canonical_districts"] == "Puttalam"
    assert not issue["is_complete_canonical_coverage"]


def test_a_reported_zero_is_kept_as_an_observation():
    """A genuine zero is observed data and is retained."""

    df = make_period(2015, 10, "3/7/2015", "3/13/2015", cases=0)

    calendar = make_calendar((2015, 10, "2015-03-07", "2015-03-13"))

    _, canonical, coverage = build(df, calendar)

    assert len(canonical) == 25
    assert canonical["cases"].eq(0).all()
    assert canonical["case_observed"].eq(1).all()

    # A reported zero is complete coverage, unlike a missing record.
    assert coverage.iloc[0]["is_complete_canonical_coverage"]


def test_complete_period_reports_no_missing_districts():
    """A full period lists no missing districts."""

    df = make_period(2015, 10, "3/7/2015", "3/13/2015")

    calendar = make_calendar((2015, 10, "2015-03-07", "2015-03-13"))

    _, _, coverage = build(df, calendar)

    issue = coverage.iloc[0]

    assert issue["observed_canonical_districts"] == 25
    assert issue["missing_canonical_districts"] == ""
    assert issue["is_complete_canonical_coverage"]


# ---------------------------------------------------------------------------
# Calendar join
# ---------------------------------------------------------------------------

def test_every_row_maps_to_exactly_one_period():
    """The calendar join neither drops nor duplicates source rows."""

    df = pd.concat(
        [
            make_period(2015, 10, "3/7/2015", "3/13/2015"),
            make_period(2015, 11, "3/14/2015", "3/20/2015"),
        ],
        ignore_index=True,
    )

    calendar = make_calendar(
        (2015, 10, "2015-03-07", "2015-03-13"),
        (2015, 11, "2015-03-14", "2015-03-20"),
    )

    rows, _, _ = build(df, calendar)

    assert len(rows) == len(df)
    assert rows["period_id"].notna().all()
    assert rows["period_id"].nunique() == 2


def test_row_with_no_matching_period_is_rejected():
    """A source row outside the calendar raises rather than vanishing."""

    df = make_period(2015, 99, "6/6/2015", "6/12/2015")

    df["start_date"] = pd.to_datetime(df["start.date"], format="%m/%d/%Y")
    df["end_date"] = pd.to_datetime(df["end.date"], format="%m/%d/%Y")

    calendar = make_calendar((2015, 10, "2015-03-07", "2015-03-13"))

    with pytest.raises(ValueError, match="match no reporting period"):
        canonical_module.join_reporting_calendar(df, calendar)


def test_join_uses_dates_not_only_labels():
    """A label matching the wrong interval does not join."""

    df = make_period(2015, 10, "3/14/2015", "3/20/2015")

    df["start_date"] = pd.to_datetime(df["start.date"], format="%m/%d/%Y")
    df["end_date"] = pd.to_datetime(df["end.date"], format="%m/%d/%Y")

    # The calendar holds label 2015 week 10 against different dates.
    calendar = make_calendar((2015, 10, "2015-03-07", "2015-03-13"))

    with pytest.raises(ValueError, match="match no reporting period"):
        canonical_module.join_reporting_calendar(df, calendar)


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_build():
    """Build the canonical dataset from the real inputs once per module."""

    raw = canonical_module.load_raw_dengue()
    calendar = canonical_module.load_reporting_calendar()
    nodes = canonical_module.load_nodes()

    rows = canonical_module.join_reporting_calendar(raw, calendar)
    rows = canonical_module.add_canonical_names(rows)

    canonical = canonical_module.aggregate_to_canonical(rows, nodes)
    coverage = canonical_module.build_canonical_coverage(canonical, calendar)

    return raw, rows, canonical, coverage, nodes


def test_real_assertions_all_hold(real_build):
    """Every guarantee holds on the real data."""

    raw, rows, canonical, coverage, nodes = real_build

    canonical_module.run_canonical_assertions(
        rows, canonical, coverage, nodes
    )


def test_real_data_has_twenty_five_districts(real_build):
    """The canonical dataset holds exactly the 25 model districts."""

    _, _, canonical, _, nodes = real_build

    assert canonical["canonical_name"].nunique() == 25
    assert set(canonical["canonical_name"]) == set(nodes["canonical_name"])
    assert "Kalmune" not in set(canonical["canonical_name"])


def test_real_data_row_count(real_build):
    """Every period holds 25 districts, less the absent Puttalam record."""

    _, _, canonical, coverage, _ = real_build

    periods = canonical["period_id"].nunique()

    assert len(canonical) == periods * 25 - 1


def test_real_data_preserves_case_totals(real_build):
    """No case is lost or invented by the Kalmune aggregation."""

    raw, _, canonical, _, _ = real_build

    assert canonical["cases"].sum() == raw["cases"].sum()


def test_real_data_ampara_combines_both_source_areas(real_build):
    """Ampara is aggregated from both source rows in every period."""

    raw, _, canonical, _, _ = real_build

    ampara = canonical.loc[canonical["canonical_name"].eq("Ampara")]

    assert ampara["source_row_count"].eq(2).all()
    assert ampara["source_reporting_areas_used"].eq("Ampara|Kalmune").all()

    source_total = raw.loc[
        raw["district"].isin(["Ampara", "Kalmune"]), "cases"
    ].sum()

    assert ampara["cases"].sum() == source_total


def test_real_data_puttalam_absent_from_2026_week_7(real_build):
    """The missing Puttalam record stays missing."""

    _, _, canonical, coverage, _ = real_build

    issue = coverage.loc[
        coverage["source_year"].eq(2026) & coverage["source_week"].eq(7)
    ].iloc[0]

    assert issue["observed_canonical_districts"] == 24
    assert issue["missing_canonical_districts"] == "Puttalam"
    assert not issue["is_complete_canonical_coverage"]

    assert canonical.loc[
        canonical["period_id"].eq(issue["period_id"])
        & canonical["canonical_name"].eq("Puttalam")
    ].empty


def test_real_data_has_one_incomplete_period(real_build):
    """2026 week 7 is the only incomplete canonical period."""

    _, _, _, coverage, _ = real_build

    incomplete = coverage.loc[
        ~coverage["is_complete_canonical_coverage"]
    ]

    assert len(incomplete) == 1
    assert incomplete.iloc[0]["source_week"] == 7


def test_real_data_every_row_is_observed(real_build):
    """No row is a synthetic fill."""

    _, _, canonical, _, _ = real_build

    assert canonical["case_observed"].eq(1).all()
    assert canonical["cases"].notna().all()
    assert canonical["source_row_count"].ge(1).all()


def test_real_source_file_is_unchanged(real_build):
    """The raw source columns are carried through untouched."""

    raw, _, _, _, _ = real_build

    source = pd.read_csv(canonical_module.RAW_PATH)

    assert raw["year"].equals(source["year"])
    assert raw["week"].equals(source["week"])
    assert raw["cases"].equals(source["cases"])
    assert len(raw) == len(source)

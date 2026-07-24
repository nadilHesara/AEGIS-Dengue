# The canonical reporting calendar

`data/interim/reporting_calendar.csv` is the project's time index. It holds
one row per unique dengue reporting interval, ordered by the actual dates the
source reports, and assigns each interval a stable identifier: `period_id`.

Built by `scripts/2.create_calendar.py` from
`data/raw/srilanka_weekly_data.csv`. The source CSV is read only and is never
modified.

**All chronological sorting, lag generation, model windowing and temporal
splitting must use `period_id`.**

**Weather aggregation must use the actual `start_date` and `end_date` of each
reporting period.**

**The `source_year` and `source_week` columns are retained only as source
metadata.**

---

## Why the raw year and week cannot be the chronological key

The source `year` and `week` columns are labels, not positions. Sorting on
them puts the data in the wrong order.

The clearest case is the interval labelled **2026 week 53**, which actually
covers **2025-12-20 to 2025-12-26**. Sorted by label it lands after 2026 week
52, roughly a year late. Sorted by its dates it belongs immediately before
2026 week 1 (2025-12-29 to 2026-01-04), which is where it truly sits.

Sorting on the raw labels would therefore:

- place a December 2025 observation after May 2026;
- corrupt every lag calculated across that boundary;
- leak future data into training windows built by label order;
- put the wrong weather week against the wrong dengue week.

The labels are not wrong as *labels* — they are the source's own week
numbering. They are simply unusable as a **sort key**. The calendar keeps
them unchanged and adds a trustworthy key alongside them.

## Why `period_id` is required

`period_id` is assigned **after** sorting on `start_date` then `end_date`. It
therefore has the properties the raw labels lack:

- unique — exactly one row per reporting interval;
- strictly increasing in true chronological order;
- gap-free as a sequence, so `period_id - 1` is always the previous period;
- stable once the dataset is frozen.

Because it is a dense integer sequence, lags and sequence windows are simple
and correct: the observation `k` periods before `period_id = n` is
`period_id = n - k`, with no date arithmetic and no risk of the label
ordering intruding.

A caveat worth stating plainly: `period_id - 1` is the previous *reporting
period*, which is not always exactly seven days earlier. Three periods are
exceptions (see below). Where an exact time delta matters, use the dates.

## Why weather must be joined on exact dates

Reporting periods are not ISO weeks and are not uniformly seven days. Three
intervals break the weekly assumption, and the reporting weekday changes
twice across the series. Any weather join that assumes "week `w` of year `y`"
will silently misalign against these.

Weather must therefore be aggregated over the **closed interval
`start_date` to `end_date`** of each reporting period, taking the period's
actual length. Join the result on `period_id`.

## The 2009 irregular periods

Two intervals are not seven days long. Both are confirmed source behaviour,
not data errors, and both are retained.

| Source label | Interval | Days | Effect |
| --- | --- | --- | --- |
| 2009 week 17 | 2009-04-18 to 2009-04-25 | 8 | Pushed reporting starts from Saturday to Sunday |
| 2009 week 22 | 2009-05-24 to 2009-05-29 | 6 | Restored the Saturday convention |

They are a matched pair: the extra day is given back five weeks later. The
five periods between them start on Sunday and are labelled
`temporary_sunday_sequence`.

Crucially, **neither period breaks calendar continuity**. Each begins the day
after its predecessor ends, so `is_calendar_continuous` is `True` throughout
the episode. They are flagged by `is_irregular_period` and are fully usable.

Models sensitive to exposure length should use `reporting_days` as a weight
or offset rather than dropping these periods.

## The 2026 weekday convention change

From the period labelled 2026 week 1 (starting 2025-12-29) the source moves
permanently from Saturday–Friday to **Monday–Sunday** reporting.

This is a convention change, not an error. Monday-start periods are valid.
`weekday_convention` records which convention each period follows, and
`is_weekday_convention_change` is `True` only on the first period of a new
convention.

The three conventions in the series:

| Label | Period | Start weekday |
| --- | --- | --- |
| `saturday_friday` | 2006-12-23 to 2009-04-25, and 2009-05-30 to 2025-12-26 | Saturday |
| `temporary_sunday_sequence` | 2009-04-26 to 2009-05-29 | Sunday |
| `monday_sunday` | 2025-12-29 onward | Monday |

## The two uncovered dates

The convention change leaves a real hole. The last Saturday period ends
**2025-12-26**; the first Monday period begins **2025-12-29**. That leaves:

- **2025-12-27**
- **2025-12-28**

These two dates are covered by **no reporting period at all**. No dengue
counts exist for them.

They are recorded in `results/data_validation/known_calendar_gaps.csv` and
flagged on the following period via `calendar_gap_before_days = 2`. They are
**not** filled. Specifically: no synthetic reporting period is created, no
zero cases are assigned, and no interpolation is performed.

Weather aggregated over these two days has no dengue counterpart and must not
be attributed to either neighbouring period.

## Missing Puttalam data

**Puttalam has no record for 2026 week 7** (2026-02-09 to 2026-02-15). That
period carries 25 of the 26 source reporting areas.

This is represented as an explicit absence:

- the period is **retained** in the calendar;
- `is_complete_source_coverage` is `False`;
- `missing_source_reporting_areas` names `Puttalam`;
- `source_reporting_area_count` is 25;
- the detail is written to
  `results/data_validation/incomplete_reporting_area_coverage.csv`.

No Puttalam row is created for that period. The district-level data simply
has no row there.

## Why missing records are not converted to zero

A missing record means **"not reported"**. A zero means **"reported, and the
count was zero"**. These are different facts and must not be conflated.

Writing zero for Puttalam in 2026 week 7 would assert that Puttalam observed
no dengue cases that week. Nothing in the source supports that. The likely
explanation is a reporting or transmission failure, and during an outbreak a
fabricated zero would bias any model toward under-prediction precisely where
accuracy matters most.

The same reasoning governs the two uncovered December 2025 dates and any
future gap. Absence is carried as absence. Downstream code decides how to
handle it — masking, exclusion, or an explicit missing-data indicator — but
that decision is made in the open, not hidden inside the calendar.

## Columns later scripts must use

### Keys

| Column | Use |
| --- | --- |
| `period_id` | **The** chronological key. Sorting, lags, windows, splits. |
| `start_date`, `end_date` | Weather aggregation bounds. Exact interval join. |

### Source metadata — never use for ordering

| Column | Meaning |
| --- | --- |
| `source_year`, `source_week` | The source's own labels, unchanged. |
| `start_calendar_year`, `end_calendar_year` | Calendar years of the actual dates. |
| `source_year_differs_from_start_year` | Documentation only. A December start labelled as the next year is normal week numbering, not an error. |

### Period shape

| Column | Meaning |
| --- | --- |
| `reporting_days` | Inclusive length. 7 for almost all periods; 6 and 8 once each. |
| `start_weekday`, `end_weekday` | Weekday names of the interval bounds. |
| `weekday_convention` | `saturday_friday`, `temporary_sunday_sequence` or `monday_sunday`. |
| `is_weekday_convention_change` | `True` on the first period of a new convention. |
| `is_irregular_period` | `reporting_days != 7`. Usable, not an error. |

### Continuity

| Column | Meaning |
| --- | --- |
| `previous_period_id` | Previous period's key. Null for the first period. |
| `previous_start_date`, `previous_end_date` | Previous period's bounds. |
| `expected_start_date` | `previous_end_date + 1 day`. |
| `calendar_gap_before_days` | Uncovered days before this period. |
| `calendar_overlap_before_days` | Days overlapping the previous period. |
| `is_calendar_continuous` | `start_date == expected_start_date`. |

### Coverage and quality

| Column | Meaning |
| --- | --- |
| `source_reporting_area_count` | Distinct reporting areas present. Expected 26. |
| `source_reporting_areas` | The areas present, comma separated. |
| `missing_source_reporting_areas` | The areas absent, comma separated. |
| `is_complete_source_coverage` | All 26 areas present. |
| `has_calendar_gap_before`, `has_calendar_overlap_before` | Boolean forms of the day counts. |
| `has_source_label_anomaly` | The source label cannot be trusted for ordering. |
| `has_impossible_date_range` | `reporting_days <= 0`. None in the current source. |
| `has_data_quality_issue` | Any of: gap, overlap, incomplete coverage, label anomaly, impossible range. |

`has_data_quality_issue` marks periods needing a deliberate decision. Note
that `is_irregular_period` alone does **not** set it: the 2009 periods are
irregular but sound.

## Companion tables

| File | Contents |
| --- | --- |
| `results/data_validation/known_calendar_gaps.csv` | Uncovered calendar dates, with the recommended handling. |
| `results/data_validation/source_week_label_anomalies.csv` | Source labels unusable for ordering, including 2026 week 53. |
| `results/data_validation/incomplete_reporting_area_coverage.csv` | Periods missing reporting areas, including Puttalam in 2026 week 7. |

## Rebuilding

```bash
python scripts/0.load_dataset.py      # fetch the raw source CSV
python scripts/1.dataset_validate.py  # validate the raw source
python scripts/2.create_calendar.py   # build this calendar
```

`2.create_calendar.py` asserts its own guarantees before writing: unique and
chronological `period_id`, positive period lengths, no duplicate intervals,
every district row mapping to exactly one period, the 2009 periods preserved,
the two-day gap detected, 2026 week 53 positioned before 2026 week 1,
Puttalam reported missing, and the raw source labels unchanged. A failure
stops the script rather than writing a calendar that cannot be trusted.

## Summary of what the calendar must never do

- Modify the original dengue CSV.
- Force periods into ISO weeks.
- Reorder using the raw `year` and `week` labels.
- Create synthetic dengue reporting periods.
- Interpolate dengue cases.
- Convert missing dengue records to zero.
- Remove the 2009 irregular periods.
- Treat the 2026 Monday convention as invalid.

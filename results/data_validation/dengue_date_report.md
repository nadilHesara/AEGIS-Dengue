# Dengue reporting-date anomaly report

- Source file: `data/raw/srilanka_weekly_data.csv`
- Rows: 26,311
- Unique reporting weeks: 1,012
- Coverage: 2006-12-23 to 2026-05-17
- Districts: 26
- Date parse failures: 0 (format `%m/%d/%Y`)

## 1. Reporting-period length

`reporting_days = (end_date - start_date).days + 1` (inclusive).

| Reporting length | Rows |
| --- | ---: |
| Fewer than 7 days | 26 |
| Exactly 7 days | 26,259 |
| More than 7 days | 26 |

All 52 irregular rows belong to just 2 reporting weeks; every district in those weeks is affected identically, which confirms the anomaly is in the reporting calendar rather than in any district's data.

## 2. Irregular reporting periods

| Year | Week | Start | End | Inclusive days | Districts affected |
| ---: | ---: | --- | --- | ---: | ---: |
| 2009 | 17 | 4/18/2009 | 4/25/2009 | 8 | 26 |
| 2009 | 22 | 5/24/2009 | 5/29/2009 | 6 | 26 |

## 3. Reporting start weekdays

| Start weekday | Reporting weeks |
| --- | ---: |
| Saturday | 987 |
| Monday | 20 |
| Sunday | 5 |

Saturday is the dominant convention (987 of 1,012 weeks). The non-Saturday weeks fall into two distinct episodes, described below.

## 4. Convention changes

### 4.1 April-May 2009: temporary shift to Sunday

Week 17 of 2009 (`4/18/2009`-`4/25/2009`) was extended to 8 inclusive days. That one-day extension pushed the following week onto a Sunday start, and weeks 18-22 of 2009 all begin on Sunday. Week 22 (`5/24/2009`-`5/29/2009`) was then shortened to 6 inclusive days, which restored the Saturday convention from week 23 onward.

The two length anomalies are therefore a matched pair: a +1-day extension and a -1-day contraction that together return the calendar to its original phase. The calendar remains fully continuous throughout - no day is uncovered and no day is double-counted.

### 4.2 From 2026: permanent shift to Monday

Every reporting week labelled 2026 week 1 onward begins on a Monday and ends on a Sunday. The last Saturday-start week is `12/20/2025`-`12/26/2025`. This is a deliberate change of reporting convention at the source, not a data error, and it is permanent rather than transient.

## 5. The three non-consecutive-week failures

`scripts/1.dataset_validate.py` compares each week's start date with the previous week's start date and expects a gap of exactly 7 days. Three weeks fail. They have three different causes:

| Year | Week | Previous start | Start | Gap | Cause |
| ---: | ---: | --- | --- | ---: | --- |
| 2009 | 18 | 2009-04-18 | 2009-04-26 | 8 | Consequence of the 8-day week 17. Not a missing week. |
| 2009 | 23 | 2009-05-24 | 2009-05-30 | 6 | Consequence of the 6-day week 22. Not a duplicate. |
| 2026 | 1 | 2025-12-20 | 2025-12-29 | 9 | Genuine 2-day calendar gap plus a mislabelled week. |

The first two are artefacts of the validator's assumption. Because it measures start-to-start distance, any change in reporting length is reported a second time as a spacing error. The underlying calendar is continuous.

The third is a real problem, with two separate defects:

1. **Mislabelled year.** The week covering `12/20/2025`-`12/26/2025` is labelled `year=2026, week=53`. Sorted by week number it appears at the end of 2026, but chronologically it precedes 2026 week 1. It is the final week of the 2025 reporting year and should be read as such.
2. **Two uncovered days.** No reporting week covers `2025-12-27` or `2025-12-28`. The Saturday-based calendar ends on 2025-12-26 and the Monday-based calendar begins on 2025-12-29. These two days are the seam between the two conventions and are genuinely absent from the source.

This is the only true calendar gap in the entire 2006-2026 series.

## 6. Checks that are not anomalies

- **Week 1 starting in December.** In 18 reporting years, week 1 begins in the last days of the previous December. This is the source's normal convention and is not flagged.
- **2023 has 51 weeks.** 2023 week 52 does not exist because 2024 week 1 starts on `12/23/2023`. The calendar is continuous across the boundary; only the label rolls over early.
- **2009, 2016 and 2021 have 53 weeks.** Expected for a 53-week reporting year.

## 7. Incomplete district coverage

One week does not contain every district. This is a completeness issue rather than a date issue, but it affects the same node-week grid:

| Year | Week | Start | Districts | Detail |
| ---: | ---: | --- | ---: | --- |
| 2026 | 7 | 2/9/2026 | 25 | Only 25 of 26 districts reported. Missing: Puttalam. |

## 8. Recommended handling

No source record should be modified or deleted. Every recommendation below adds derived columns and keeps the original `year`, `week`, `start.date` and `end.date` values intact.

| Anomaly | Recommended handling |
| --- | --- |
| 8-day week (2009 w17) and 6-day week (2009 w22) | Keep as-is. Add `reporting_days` as a derived column and, for any rate or climate-lag modelling, use a per-day exposure offset rather than raw weekly counts. The pair is phase-neutral, so cumulative sums are unaffected. |
| Sunday-start weeks (2009 w18-w22) | Keep as-is. Treat the reporting week as defined by `start_date`/`end_date`, never by weekday. Any climate aggregation must join on the actual date interval. |
| Monday-start weeks (2026 w1 onward) | Keep as-is and record the convention change explicitly. Relax the validator's fixed-weekday rule to a convention-aware rule: Saturday before 2025-12-27, Monday after 2025-12-28. |
| Mislabelled 2026 week 53 | Do not edit the source. Add a derived `reporting_year`/`reporting_week` pair, ordered by `start_date`, and use that for all joins and sorting. Never sort the series on the raw `year`/`week` columns. |
| Uncovered days 2025-12-27 and 2025-12-28 | Leave absent. Do not interpolate or redistribute cases. Record the gap in a known-gaps table so downstream models treat the interval as unobserved rather than as zero. |
| Missing Puttalam row, 2026 week 7 | Leave absent. Represent it as an explicit missing value on the node-week grid, not as `cases = 0`. |
| Start-to-start spacing check | Replace with a continuity check on `start_date == previous_end_date + 1 day`. That test detects real gaps and overlaps without re-flagging legitimate changes in reporting length. |

The single highest-value change is the derived chronological ordering. Until 2026 week 53 is placed correctly, any time-ordered join between dengue cases and weather data will silently misalign one week of the series.

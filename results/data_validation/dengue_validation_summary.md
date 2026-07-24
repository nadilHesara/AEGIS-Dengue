# Dengue validation summary

- Source file: `data/raw/srilanka_weekly_data.csv`
- Rows: 26,311
- Unique reporting periods: 1,012
- Coverage: 2006-12-23 to 2026-05-17
- Source reporting areas: 26 (expected 26 before Kalmune is merged into Ampara)
- Date format: `%m/%d/%Y`

## Check results

| Check | Status | Findings |
| --- | --- | ---: |
| `invalid_start_dates` | PASS | 0 |
| `invalid_end_dates` | PASS | 0 |
| `impossible_date_ranges` | PASS | 0 |
| `irregular_reporting_periods` | WARNING | 2 |
| `calendar_gaps` | FAIL | 1 |
| `calendar_overlaps` | PASS | 0 |
| `unexpected_weekdays` | PASS | 0 |
| `weekday_convention_changes` | WARNING | 3 |
| `source_year_label_mismatches` | WARNING | 10 |
| `duplicate_source_year_weeks` | PASS | 0 |
| `duplicate_date_intervals` | PASS | 0 |
| `out_of_order_source_weeks` | WARNING | 1 |
| `duplicate_reporting_area_periods` | PASS | 0 |
| `missing_reporting_areas` | PASS | 0 |
| `incomplete_reporting_area_coverage` | FAIL | 1 |
| `invalid_case_values` | PASS | 0 |

PASS means no problem found. WARNING means a known source irregularity that can be retained and modelled. FAIL means a genuinely missing, impossible, duplicate or conflicting record.

## Irregular reporting periods (WARNING)

| Period | Source year | Source week | Start | End | Days | Reason |
| ---: | ---: | ---: | --- | --- | ---: | --- |
| 122 | 2009 | 17 | 2009-04-18 | 2009-04-25 | 8 | Source extended 2009 week 17 by one day, moving reporting starts from Saturday to Sunday. |
| 127 | 2009 | 22 | 2009-05-24 | 2009-05-29 | 6 | Source shortened 2009 week 22 by one day, restoring Saturday reporting from week 23. |

These periods are usable. The pair is phase-neutral: the 8-day period moved reporting starts to Sunday and the 6-day period restored Saturday, so no calendar day is lost or double-counted.

## Weekday conventions (WARNING on change)

| Convention | Valid from | Valid to | Expected start | Description |
| ---: | --- | --- | --- | --- |
| 1 | 2006-12-23 | 2009-04-25 | Saturday | Original Saturday-to-Friday convention, including the 8-day 2009 week 17 period which still starts on a Saturday. |
| 2 | 2009-04-26 | 2009-05-29 | Sunday | Temporary Sunday start caused by the 8-day 2009 week 17 period. Ends with the 6-day 2009 week 22 period, which restores the Saturday phase. |
| 3 | 2009-05-30 | 2025-12-26 | Saturday | Saturday convention restored from 2009 week 23 until the last Saturday-start period, 2025-12-20 to 2025-12-26. |
| 4 | 2025-12-29 | open | Monday | Permanent Monday-to-Sunday convention from the period labelled 2026 week 1 onward. |

## Calendar gaps

| Previous period end | Next period start | Gap start | Gap end | Days | Reason |
| --- | --- | --- | --- | ---: | --- |
| 2025-12-26 | 2025-12-29 | 2025-12-27 | 2025-12-28 | 2 | Seam between the Saturday and Monday reporting conventions. 2025-12-27 and 2025-12-28 are not covered by any reporting period in the source. Carried as an unobserved interval, never interpolated. |

## Reporting-area coverage

| Period | Source year | Source week | Start | End | Areas | Missing |
| ---: | ---: | ---: | --- | --- | ---: | --- |
| 999 | 2026 | 7 | 2026-02-09 | 2026-02-15 | 25 | Puttalam |

Missing records stay missing. Cases are not filled with zero, copied from a neighbouring period, or interpolated, and the period is not dropped.

## Source year-week labels

The following source labels do not follow chronological order. They are positioned by their actual dates through `period_id` and their source labels are left untouched.

| Period | Source year | Source week | Start | End | Previous label |
| ---: | ---: | ---: | --- | --- | --- |
| 993 | 2026 | 1 | 2025-12-29 | 2026-01-04 | 2026 week 53 |

The interval labelled 2026 week 53 covers 2025-12-20 to 2025-12-26 and is assigned `period_id` 992, before 2026 week 1. Its source label is reported but never rewritten.

## Chronological ordering

`period_id` is assigned by sorting unique reporting intervals on `start_date` then `end_date`. It is the primary key for sorting, merging, lagging and model windowing. `source_year` and `source_week` are retained for traceability only.

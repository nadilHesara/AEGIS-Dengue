# Dengue validation summary

| check                              | status   |   findings |
|:-----------------------------------|:---------|-----------:|
| invalid_start_dates                | PASS     |          0 |
| invalid_end_dates                  | PASS     |          0 |
| impossible_date_ranges             | PASS     |          0 |
| irregular_reporting_periods        | WARNING  |          2 |
| calendar_gaps                      | FAIL     |          1 |
| calendar_overlaps                  | PASS     |          0 |
| unexpected_weekdays                | PASS     |          0 |
| weekday_convention_changes         | WARNING  |          3 |
| source_year_label_mismatches       | WARNING  |         10 |
| duplicate_source_year_weeks        | PASS     |          0 |
| duplicate_date_intervals           | PASS     |          0 |
| out_of_order_source_weeks          | WARNING  |          1 |
| duplicate_reporting_area_periods   | PASS     |          0 |
| missing_reporting_areas            | PASS     |          0 |
| incomplete_reporting_area_coverage | FAIL     |          1 |
| invalid_case_values                | PASS     |          0 |

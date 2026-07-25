# Weather validation summary

Validation of `data/raw/climate_daily_district.csv` against the official district registry in `data/processed/nodes.csv`.

The climate CSV is never modified. Nothing is interpolated and no extreme value is dropped.

## Classifications

| Check | Status | Findings |
| --- | --- | --- |
| `schema` | **PASS** | 0 |
| `date_parsing` | **PASS** | 0 |
| `district_completeness` | **PASS** | 0 |
| `node_id_mapping` | **PASS** | 0 |
| `duplicate_district_dates` | **PASS** | 0 |
| `missing_district_dates` | **PASS** | 0 |
| `incomplete_days` | **PASS** | 0 |
| `long_missing_runs` | **PASS** | 0 |
| `weather_history` | **PASS** | 0 |
| `present_but_null` | **PASS** | 0 |
| `invalid_values` | **PASS** | 0 |
| `suspicious_extremes` | **PASS** | 0 |
| `units` | **PASS** | 0 |

**No checks classified FAIL.**

## The four categories of problem

These are kept separate because they have different causes and different remedies.

| Category | Check | Meaning |
| --- | --- | --- |
| Missing rows | `missing_district_dates` | The district-date is absent from the file. ERA5 is a reanalysis, so this means the extraction failed. |
| Present but null | `present_but_null` | The row exists and honestly records an unobserved value. |
| Invalid values | `invalid_values` | Physically impossible. A real defect, usually a unit conversion or a transposed band. |
| Suspicious extremes | `suspicious_extremes` | Unusual but possible. Investigate; never drop. |

## Missing values by variable

| column                 | expected_units   |   null_count |   null_share_percent |
|:-----------------------|:-----------------|-------------:|---------------------:|
| rainfall_mm            | mm/day           |            0 |                    0 |
| temperature_mean_c     | degrees Celsius  |            0 |                    0 |
| temperature_min_c      | degrees Celsius  |            0 |                    0 |
| temperature_max_c      | degrees Celsius  |            0 |                    0 |
| dewpoint_mean_c        | degrees Celsius  |            0 |                    0 |
| relative_humidity_mean | percent (0-100)  |            0 |                    0 |
| wind_speed_mean        | m/s              |            0 |                    0 |

## Coverage by year

|   year |   rows_present |   rows_expected |   dates_present |   dates_expected |   districts_present |
|-------:|---------------:|----------------:|----------------:|-----------------:|--------------------:|
|   2005 |            200 |             200 |               8 |                8 |                  25 |
|   2006 |           9125 |            9125 |             365 |              365 |                  25 |
|   2007 |           9125 |            9125 |             365 |              365 |                  25 |
|   2008 |           9150 |            9150 |             366 |              366 |                  25 |
|   2009 |           9125 |            9125 |             365 |              365 |                  25 |
|   2010 |           9125 |            9125 |             365 |              365 |                  25 |
|   2011 |           9125 |            9125 |             365 |              365 |                  25 |
|   2012 |           9150 |            9150 |             366 |              366 |                  25 |
|   2013 |           9125 |            9125 |             365 |              365 |                  25 |
|   2014 |           9125 |            9125 |             365 |              365 |                  25 |
|   2015 |           9125 |            9125 |             365 |              365 |                  25 |
|   2016 |           9150 |            9150 |             366 |              366 |                  25 |
|   2017 |           9125 |            9125 |             365 |              365 |                  25 |
|   2018 |           9125 |            9125 |             365 |              365 |                  25 |
|   2019 |           9125 |            9125 |             365 |              365 |                  25 |
|   2020 |           9150 |            9150 |             366 |              366 |                  25 |
|   2021 |           9125 |            9125 |             365 |              365 |                  25 |
|   2022 |           9125 |            9125 |             365 |              365 |                  25 |
|   2023 |           9125 |            9125 |             365 |              365 |                  25 |
|   2024 |           9150 |            9150 |             366 |              366 |                  25 |
|   2025 |           9125 |            9125 |             365 |              365 |                  25 |
|   2026 |           2900 |            2900 |             116 |              116 |                  25 |

## Coverage by district

|   node_id | canonical_name   |   dates_present |   dates_expected |   dates_missing |   rows_fully_populated |
|----------:|:-----------------|----------------:|-----------------:|----------------:|-----------------------:|
|         0 | Ampara           |            7429 |             7429 |               0 |                   7429 |
|         1 | Anuradhapura     |            7429 |             7429 |               0 |                   7429 |
|         2 | Badulla          |            7429 |             7429 |               0 |                   7429 |
|         3 | Batticaloa       |            7429 |             7429 |               0 |                   7429 |
|         4 | Colombo          |            7429 |             7429 |               0 |                   7429 |
|         5 | Galle            |            7429 |             7429 |               0 |                   7429 |
|         6 | Gampaha          |            7429 |             7429 |               0 |                   7429 |
|         7 | Hambantota       |            7429 |             7429 |               0 |                   7429 |
|         8 | Jaffna           |            7429 |             7429 |               0 |                   7429 |
|         9 | Kalutara         |            7429 |             7429 |               0 |                   7429 |
|        10 | Kandy            |            7429 |             7429 |               0 |                   7429 |
|        11 | Kegalle          |            7429 |             7429 |               0 |                   7429 |
|        12 | Kilinochchi      |            7429 |             7429 |               0 |                   7429 |
|        13 | Kurunegala       |            7429 |             7429 |               0 |                   7429 |
|        14 | Mannar           |            7429 |             7429 |               0 |                   7429 |
|        15 | Matale           |            7429 |             7429 |               0 |                   7429 |
|        16 | Matara           |            7429 |             7429 |               0 |                   7429 |
|        17 | Moneragala       |            7429 |             7429 |               0 |                   7429 |
|        18 | Mullaitivu       |            7429 |             7429 |               0 |                   7429 |
|        19 | Nuwara Eliya     |            7429 |             7429 |               0 |                   7429 |
|        20 | Polonnaruwa      |            7429 |             7429 |               0 |                   7429 |
|        21 | Puttalam         |            7429 |             7429 |               0 |                   7429 |
|        22 | Ratnapura        |            7429 |             7429 |               0 |                   7429 |
|        23 | Trincomalee      |            7429 |             7429 |               0 |                   7429 |
|        24 | Vavuniya         |            7429 |             7429 |               0 |                   7429 |

## Detailed issue files

| File | Contents |
| --- | --- |
| `missing_district_dates.csv` | Absent district-date rows |
| `duplicate_district_dates.csv` | Repeated date and district pairs |
| `invalid_weather_values.csv` | Invalid and suspicious values, by `severity` |
| `weather_coverage_by_district.csv` | Per-district coverage |
| `weather_coverage_by_year.csv` | Per-year coverage |

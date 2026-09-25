# Model tensors

Version: `tensors-v1`

Stage 3 arrays cut from `data/processed/panel_weekly.parquet`.
Values are unscaled and unimputed: NaN is preserved so that scaling
and imputation can be fitted on the training split alone.

## Shapes

| Variant | X | Features | Windows (L=12, h=1) |
| --- | --- | --- | --- |
| `v0` | 1012x25x14 | 14 | 987 of 1000 |
| `v1` | 1012x25x23 | 23 | 976 of 1000 |
| `v2` | 1012x25x25 | 25 | 936 of 1000 |
| `v3` | 1012x25x28 | 28 | 936 of 1000 |

## Features

| # | Name | Variant |
| --- | --- | --- |
| 0 | `cases_log1p` | v0, v1, v2 |
| 1 | `rainfall_daily_mean_mm` | v0, v1, v2 |
| 2 | `rainy_days_frac` | v0, v1, v2 |
| 3 | `temperature_mean_c` | v0, v1, v2 |
| 4 | `diurnal_range_c` | v0, v1, v2 |
| 5 | `dewpoint_mean_c` | v0, v1, v2 |
| 6 | `relative_humidity_mean` | v0, v1, v2 |
| 7 | `wind_speed_mean` | v0, v1, v2 |
| 8 | `doy_sin` | v0, v1, v2 |
| 9 | `doy_cos` | v0, v1, v2 |
| 10 | `weather_observed` | v0, v1, v2 |
| 11 | `case_observed` | v0, v1, v2 |
| 12 | `centroid_lat` | v0, v1, v2 |
| 13 | `centroid_lon` | v0, v1, v2 |
| 14 | `rainfall_daily_mean_mm_roll4` | v1, v2 |
| 15 | `rainfall_daily_mean_mm_roll8` | v1, v2 |
| 16 | `rainfall_daily_mean_mm_roll12` | v1, v2 |
| 17 | `temperature_mean_c_roll4` | v1, v2 |
| 18 | `temperature_mean_c_roll8` | v1, v2 |
| 19 | `temperature_mean_c_roll12` | v1, v2 |
| 20 | `relative_humidity_mean_roll4` | v1, v2 |
| 21 | `relative_humidity_mean_roll8` | v1, v2 |
| 22 | `relative_humidity_mean_roll12` | v1, v2 |
| 23 | `national_wave_rank` | v2 |
| 24 | `trailing_52_cumulative_cases` | v2 |

## Missing cells

### `v0`

Total NaN cells: 526

| Feature | NaN cells |
| --- | --- |
| `cases_log1p` | 1 |
| `rainfall_daily_mean_mm` | 75 |
| `rainy_days_frac` | 75 |
| `temperature_mean_c` | 75 |
| `diurnal_range_c` | 75 |
| `dewpoint_mean_c` | 75 |
| `relative_humidity_mean` | 75 |
| `wind_speed_mean` | 75 |

Usable target periods: 13 to 999

Windows dropped for a NaN input: 13. For a target with no observed district: 0. Kept with a partly masked target: 1.

Dropped target periods: 1000-1012

### `v1`

Total NaN cells: 2776

| Feature | NaN cells |
| --- | --- |
| `cases_log1p` | 1 |
| `rainfall_daily_mean_mm` | 75 |
| `rainy_days_frac` | 75 |
| `temperature_mean_c` | 75 |
| `diurnal_range_c` | 75 |
| `dewpoint_mean_c` | 75 |
| `relative_humidity_mean` | 75 |
| `wind_speed_mean` | 75 |
| `rainfall_daily_mean_mm_roll4` | 150 |
| `rainfall_daily_mean_mm_roll8` | 250 |
| `rainfall_daily_mean_mm_roll12` | 350 |
| `temperature_mean_c_roll4` | 150 |
| `temperature_mean_c_roll8` | 250 |
| `temperature_mean_c_roll12` | 350 |
| `relative_humidity_mean_roll4` | 150 |
| `relative_humidity_mean_roll8` | 250 |
| `relative_humidity_mean_roll12` | 350 |

Usable target periods: 24 to 999

Windows dropped for a NaN input: 24. For a target with no observed district: 0. Kept with a partly masked target: 1.

Dropped target periods: 13-23, 1000-1012

### `v2`

Total NaN cells: 4066

| Feature | NaN cells |
| --- | --- |
| `cases_log1p` | 1 |
| `rainfall_daily_mean_mm` | 75 |
| `rainy_days_frac` | 75 |
| `temperature_mean_c` | 75 |
| `diurnal_range_c` | 75 |
| `dewpoint_mean_c` | 75 |
| `relative_humidity_mean` | 75 |
| `wind_speed_mean` | 75 |
| `rainfall_daily_mean_mm_roll4` | 150 |
| `rainfall_daily_mean_mm_roll8` | 250 |
| `rainfall_daily_mean_mm_roll12` | 350 |
| `temperature_mean_c_roll4` | 150 |
| `temperature_mean_c_roll8` | 250 |
| `temperature_mean_c_roll12` | 350 |
| `relative_humidity_mean_roll4` | 150 |
| `relative_humidity_mean_roll8` | 250 |
| `relative_humidity_mean_roll12` | 350 |
| `national_wave_rank` | 1 |
| `trailing_52_cumulative_cases` | 1289 |

Usable target periods: 64 to 999

Windows dropped for a NaN input: 64. For a target with no observed district: 0. Kept with a partly masked target: 1.

Dropped target periods: 13-63, 1000-1012

### `v3`

Total NaN cells: 4091

| Feature | NaN cells |
| --- | --- |
| `cases_log1p` | 1 |
| `rainfall_daily_mean_mm` | 75 |
| `rainy_days_frac` | 75 |
| `temperature_mean_c` | 75 |
| `diurnal_range_c` | 75 |
| `dewpoint_mean_c` | 75 |
| `relative_humidity_mean` | 75 |
| `wind_speed_mean` | 75 |
| `rainfall_daily_mean_mm_roll4` | 150 |
| `rainfall_daily_mean_mm_roll8` | 250 |
| `rainfall_daily_mean_mm_roll12` | 350 |
| `temperature_mean_c_roll4` | 150 |
| `temperature_mean_c_roll8` | 250 |
| `temperature_mean_c_roll12` | 350 |
| `relative_humidity_mean_roll4` | 150 |
| `relative_humidity_mean_roll8` | 250 |
| `relative_humidity_mean_roll12` | 350 |
| `national_wave_rank` | 1 |
| `trailing_52_cumulative_cases` | 1289 |
| `neighbor_case_velocity` | 25 |

Usable target periods: 64 to 999

Windows dropped for a NaN input: 64. For a target with no observed district: 0. Kept with a partly masked target: 1.

Dropped target periods: 13-63, 1000-1012

## Output files

- `data/processed/model_tensors_v0.npz`
- `data/processed/model_tensors_v1.npz`
- `data/processed/model_tensors_v2.npz`

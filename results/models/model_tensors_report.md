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

## Features

| # | Name | Variant |
| --- | --- | --- |
| 0 | `cases_log1p` | v0, v1 |
| 1 | `rainfall_daily_mean_mm` | v0, v1 |
| 2 | `rainy_days_frac` | v0, v1 |
| 3 | `temperature_mean_c` | v0, v1 |
| 4 | `diurnal_range_c` | v0, v1 |
| 5 | `dewpoint_mean_c` | v0, v1 |
| 6 | `relative_humidity_mean` | v0, v1 |
| 7 | `wind_speed_mean` | v0, v1 |
| 8 | `doy_sin` | v0, v1 |
| 9 | `doy_cos` | v0, v1 |
| 10 | `weather_observed` | v0, v1 |
| 11 | `case_observed` | v0, v1 |
| 12 | `centroid_lat` | v0, v1 |
| 13 | `centroid_lon` | v0, v1 |
| 14 | `rainfall_daily_mean_mm_roll4` | v1 |
| 15 | `rainfall_daily_mean_mm_roll8` | v1 |
| 16 | `rainfall_daily_mean_mm_roll12` | v1 |
| 17 | `temperature_mean_c_roll4` | v1 |
| 18 | `temperature_mean_c_roll8` | v1 |
| 19 | `temperature_mean_c_roll12` | v1 |
| 20 | `relative_humidity_mean_roll4` | v1 |
| 21 | `relative_humidity_mean_roll8` | v1 |
| 22 | `relative_humidity_mean_roll12` | v1 |

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

## Output files

- `data/processed/model_tensors_v0.npz`
- `data/processed/model_tensors_v1.npz`

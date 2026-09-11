# Residual diagnostic

Does the best model's error carry capturable structure, or is it noise?

Config: identity backbone, one model per horizon (8f `separate`), `v1`, 3 seeds, headline folds [1, 2, 3, 6, 7, 8, 9], horizons [1, 4]. Residual `e = y_hat - y` on observed test cells only. Trained via `scripts/16`'s imported loop.

Reproduce: `python scripts/30.residual_diagnostic.py`

/ RAW NUMBERS BELOW -- interpretation section is filled in after review. /

## A. Residual autocorrelation

Mean over the 25 districts of each district's own residual ACF, by lag. `p_value` tests that the across-district mean is zero.

| Horizon | Lag | Mean ACF | sd | n districts | p |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | +0.169 | 0.248 | 25 | 0.003 |
| 1 | 2 | +0.295 | 0.187 | 25 | 0.000 |
| 1 | 3 | +0.205 | 0.165 | 25 | 0.000 |
| 1 | 4 | +0.149 | 0.161 | 25 | 0.000 |
| 1 | 5 | +0.108 | 0.157 | 25 | 0.003 |
| 1 | 6 | +0.067 | 0.117 | 25 | 0.010 |
| 1 | 7 | +0.057 | 0.117 | 25 | 0.026 |
| 1 | 8 | +0.046 | 0.110 | 25 | 0.051 |
| 4 | 1 | +0.665 | 0.137 | 25 | 0.000 |
| 4 | 2 | +0.533 | 0.140 | 25 | 0.000 |
| 4 | 3 | +0.370 | 0.146 | 25 | 0.000 |
| 4 | 4 | +0.117 | 0.194 | 25 | 0.007 |
| 4 | 5 | +0.116 | 0.159 | 25 | 0.002 |
| 4 | 6 | +0.068 | 0.139 | 25 | 0.025 |
| 4 | 7 | +0.039 | 0.140 | 25 | 0.180 |
| 4 | 8 | +0.035 | 0.122 | 25 | 0.179 |

## B. Residual vs each v1 channel at the forecast origin

`residual` signed (model runs high/low); `abs_residual` magnitude. Sorted by |pearson_r| within horizon.

| Horizon | Quantity | Target | Pearson r | Spearman r | p | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `origin__cases_log1p` | abs_residual | +0.417 | +0.651 | 0.000 | 9125 |
| 1 | `origin__cases_log1p` | residual | -0.222 | -0.097 | 0.000 | 9125 |
| 1 | `origin__relative_humidity_mean_roll8` | abs_residual | +0.147 | +0.294 | 0.000 | 9125 |
| 1 | `origin__rainfall_daily_mean_mm_roll8` | abs_residual | +0.144 | +0.276 | 0.000 | 9125 |
| 1 | `origin__rainfall_daily_mean_mm_roll12` | abs_residual | +0.140 | +0.293 | 0.000 | 9125 |
| 1 | `origin__relative_humidity_mean_roll4` | abs_residual | +0.138 | +0.270 | 0.000 | 9125 |
| 1 | `origin__relative_humidity_mean_roll12` | abs_residual | +0.137 | +0.292 | 0.000 | 9125 |
| 1 | `origin__centroid_lon` | abs_residual | -0.136 | -0.149 | 0.000 | 9125 |
| 1 | `origin__rainy_days_frac` | abs_residual | +0.132 | +0.203 | 0.000 | 9125 |
| 1 | `origin__relative_humidity_mean` | abs_residual | +0.118 | +0.228 | 0.000 | 9125 |
| 1 | `origin__centroid_lat` | abs_residual | -0.117 | -0.284 | 0.000 | 9125 |
| 1 | `origin__rainfall_daily_mean_mm_roll4` | abs_residual | +0.112 | +0.229 | 0.000 | 9125 |
| 1 | `origin__diurnal_range_c` | abs_residual | -0.106 | -0.137 | 0.000 | 9125 |
| 1 | `origin__dewpoint_mean_c` | abs_residual | +0.081 | +0.062 | 0.000 | 9125 |
| 1 | `origin__rainy_days_frac` | residual | -0.076 | -0.069 | 0.000 | 9125 |
| 1 | `origin__rainfall_daily_mean_mm` | abs_residual | +0.075 | +0.178 | 0.000 | 9125 |
| 1 | `origin__centroid_lon` | residual | +0.060 | +0.008 | 0.000 | 9125 |
| 1 | `origin__rainfall_daily_mean_mm_roll8` | residual | -0.059 | -0.040 | 0.000 | 9125 |
| 1 | `origin__relative_humidity_mean_roll4` | residual | -0.059 | -0.038 | 0.000 | 9125 |
| 1 | `origin__rainfall_daily_mean_mm_roll4` | residual | -0.056 | -0.063 | 0.000 | 9125 |
| 1 | `origin__relative_humidity_mean` | residual | -0.054 | -0.047 | 0.000 | 9125 |
| 1 | `origin__relative_humidity_mean_roll8` | residual | -0.053 | -0.016 | 0.000 | 9125 |
| 1 | `origin__wind_speed_mean` | abs_residual | -0.051 | -0.118 | 0.000 | 9125 |
| 1 | `origin__dewpoint_mean_c` | residual | -0.049 | -0.040 | 0.000 | 9125 |
| 1 | `origin__doy_cos` | abs_residual | -0.045 | +0.046 | 0.000 | 9125 |
| 1 | `origin__centroid_lat` | residual | +0.045 | +0.037 | 0.000 | 9125 |
| 1 | `origin__diurnal_range_c` | residual | +0.044 | +0.006 | 0.000 | 9125 |
| 1 | `origin__rainfall_daily_mean_mm_roll12` | residual | -0.041 | -0.021 | 0.000 | 9125 |
| 1 | `origin__relative_humidity_mean_roll12` | residual | -0.040 | -0.005 | 0.000 | 9125 |
| 1 | `origin__rainfall_daily_mean_mm` | residual | -0.039 | -0.065 | 0.000 | 9125 |
| 1 | `origin__doy_sin` | residual | -0.037 | -0.022 | 0.000 | 9125 |
| 1 | `origin__doy_cos` | residual | +0.035 | -0.014 | 0.001 | 9125 |
| 1 | `origin__wind_speed_mean` | residual | +0.034 | +0.065 | 0.001 | 9125 |
| 1 | `origin__temperature_mean_c_roll4` | abs_residual | -0.029 | -0.158 | 0.006 | 9125 |
| 1 | `origin__temperature_mean_c` | abs_residual | -0.028 | -0.151 | 0.007 | 9125 |
| 1 | `origin__temperature_mean_c_roll8` | abs_residual | -0.028 | -0.161 | 0.008 | 9125 |
| 1 | `origin__temperature_mean_c_roll12` | abs_residual | -0.023 | -0.159 | 0.027 | 9125 |
| 1 | `origin__doy_sin` | abs_residual | +0.014 | +0.001 | 0.186 | 9125 |
| 1 | `origin__temperature_mean_c_roll8` | residual | +0.005 | +0.023 | 0.654 | 9125 |
| 1 | `origin__temperature_mean_c_roll4` | residual | +0.005 | +0.017 | 0.661 | 9125 |
| 1 | `origin__temperature_mean_c_roll12` | residual | +0.004 | +0.029 | 0.735 | 9125 |
| 1 | `origin__temperature_mean_c` | residual | +0.003 | +0.012 | 0.747 | 9125 |
| 1 | `origin__weather_observed` | residual | +nan | +nan | nan | 9125 |
| 1 | `origin__weather_observed` | abs_residual | +nan | +nan | nan | 9125 |
| 1 | `origin__case_observed` | residual | +nan | +nan | nan | 9125 |
| 1 | `origin__case_observed` | abs_residual | +nan | +nan | nan | 9125 |
| 4 | `origin__cases_log1p` | abs_residual | +0.392 | +0.643 | 0.000 | 9125 |
| 4 | `origin__cases_log1p` | residual | -0.165 | -0.042 | 0.000 | 9125 |
| 4 | `origin__rainy_days_frac` | abs_residual | +0.149 | +0.242 | 0.000 | 9125 |
| 4 | `origin__relative_humidity_mean_roll4` | abs_residual | +0.145 | +0.283 | 0.000 | 9125 |
| 4 | `origin__centroid_lon` | abs_residual | -0.144 | -0.169 | 0.000 | 9125 |
| 4 | `origin__rainfall_daily_mean_mm_roll4` | abs_residual | +0.139 | +0.266 | 0.000 | 9125 |
| 4 | `origin__rainfall_daily_mean_mm_roll8` | abs_residual | +0.136 | +0.293 | 0.000 | 9125 |
| 4 | `origin__relative_humidity_mean` | abs_residual | +0.133 | +0.250 | 0.000 | 9125 |
| 4 | `origin__relative_humidity_mean_roll8` | abs_residual | +0.133 | +0.280 | 0.000 | 9125 |
| 4 | `origin__diurnal_range_c` | abs_residual | -0.124 | -0.171 | 0.000 | 9125 |
| 4 | `origin__rainfall_daily_mean_mm_roll12` | abs_residual | +0.119 | +0.285 | 0.000 | 9125 |
| 4 | `origin__relative_humidity_mean_roll12` | abs_residual | +0.116 | +0.255 | 0.000 | 9125 |
| 4 | `origin__centroid_lat` | abs_residual | -0.107 | -0.276 | 0.000 | 9125 |
| 4 | `origin__dewpoint_mean_c` | abs_residual | +0.099 | +0.108 | 0.000 | 9125 |
| 4 | `origin__rainfall_daily_mean_mm` | abs_residual | +0.088 | +0.206 | 0.000 | 9125 |
| 4 | `origin__rainy_days_frac` | residual | -0.070 | -0.056 | 0.000 | 9125 |
| 4 | `origin__rainfall_daily_mean_mm_roll4` | residual | -0.063 | -0.038 | 0.000 | 9125 |
| 4 | `origin__centroid_lon` | residual | +0.057 | +0.007 | 0.000 | 9125 |
| 4 | `origin__relative_humidity_mean` | residual | -0.056 | -0.033 | 0.000 | 9125 |
| 4 | `origin__doy_cos` | abs_residual | -0.055 | +0.024 | 0.000 | 9125 |
| 4 | `origin__relative_humidity_mean_roll4` | residual | -0.053 | -0.015 | 0.000 | 9125 |
| 4 | `origin__doy_sin` | residual | -0.052 | -0.035 | 0.000 | 9125 |
| 4 | `origin__centroid_lat` | residual | +0.049 | +0.062 | 0.000 | 9125 |
| 4 | `origin__wind_speed_mean` | abs_residual | -0.047 | -0.114 | 0.000 | 9125 |
| 4 | `origin__wind_speed_mean` | residual | +0.043 | +0.081 | 0.000 | 9125 |
| 4 | `origin__dewpoint_mean_c` | residual | -0.037 | -0.005 | 0.000 | 9125 |
| 4 | `origin__rainfall_daily_mean_mm` | residual | -0.036 | -0.045 | 0.001 | 9125 |
| 4 | `origin__diurnal_range_c` | residual | +0.035 | -0.017 | 0.001 | 9125 |
| 4 | `origin__rainfall_daily_mean_mm_roll8` | residual | -0.034 | -0.014 | 0.001 | 9125 |
| 4 | `origin__relative_humidity_mean_roll8` | residual | -0.034 | -0.005 | 0.001 | 9125 |
| 4 | `origin__doy_sin` | abs_residual | +0.028 | -0.006 | 0.008 | 9125 |
| 4 | `origin__temperature_mean_c` | abs_residual | -0.024 | -0.139 | 0.020 | 9125 |
| 4 | `origin__temperature_mean_c_roll4` | abs_residual | -0.023 | -0.143 | 0.029 | 9125 |
| 4 | `origin__relative_humidity_mean_roll12` | residual | -0.022 | -0.000 | 0.039 | 9125 |
| 4 | `origin__temperature_mean_c_roll8` | abs_residual | -0.016 | -0.137 | 0.137 | 9125 |
| 4 | `origin__rainfall_daily_mean_mm_roll12` | residual | -0.016 | -0.004 | 0.137 | 9125 |
| 4 | `origin__temperature_mean_c_roll4` | residual | +0.015 | +0.042 | 0.165 | 9125 |
| 4 | `origin__temperature_mean_c` | residual | +0.013 | +0.030 | 0.211 | 9125 |
| 4 | `origin__temperature_mean_c_roll12` | residual | +0.013 | +0.065 | 0.212 | 9125 |
| 4 | `origin__temperature_mean_c_roll12` | abs_residual | -0.012 | -0.131 | 0.266 | 9125 |
| 4 | `origin__doy_cos` | residual | +0.012 | -0.055 | 0.267 | 9125 |
| 4 | `origin__temperature_mean_c_roll8` | residual | +0.011 | +0.055 | 0.298 | 9125 |
| 4 | `origin__weather_observed` | residual | +nan | +nan | nan | 9125 |
| 4 | `origin__weather_observed` | abs_residual | +nan | +nan | nan | 9125 |
| 4 | `origin__case_observed` | residual | +nan | +nan | nan | 9125 |
| 4 | `origin__case_observed` | abs_residual | +nan | +nan | nan | 9125 |

## C. Residual vs candidate missing-driver proxies

Computed from case history only. `periods_since_outbreak` uses a per-fold 90th-pct threshold.

| Horizon | Quantity | Target | Pearson r | Spearman r | p | n |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `national_wave_rank` | abs_residual | -0.341 | -0.621 | 0.000 | 9125 |
| 1 | `trailing_52_cumulative_cases` | abs_residual | +0.334 | +0.548 | 0.000 | 9125 |
| 1 | `national_wave_rank` | residual | +0.249 | +0.325 | 0.000 | 9125 |
| 1 | `trailing_52_cumulative_cases` | residual | -0.181 | -0.107 | 0.000 | 9125 |
| 1 | `periods_since_outbreak` | abs_residual | -0.109 | -0.332 | 0.000 | 9125 |
| 1 | `periods_since_outbreak` | residual | +0.067 | +0.127 | 0.000 | 9125 |
| 4 | `trailing_52_cumulative_cases` | abs_residual | +0.384 | +0.586 | 0.000 | 9125 |
| 4 | `national_wave_rank` | abs_residual | -0.344 | -0.643 | 0.000 | 9125 |
| 4 | `national_wave_rank` | residual | +0.233 | +0.328 | 0.000 | 9125 |
| 4 | `trailing_52_cumulative_cases` | residual | -0.155 | -0.085 | 0.000 | 9125 |
| 4 | `periods_since_outbreak` | abs_residual | -0.116 | -0.346 | 0.000 | 9125 |
| 4 | `periods_since_outbreak` | residual | +0.071 | +0.171 | 0.000 | 9125 |

## D. Variance decomposition

| Horizon | Quantity | Value |
| --- | --- | --- |
| 1 | `total_residual_variance` | 2049.7751 |
| 1 | `between_district_eta2` | 0.0392 |
| 1 | `seasonal_r2` | 0.0026 |
| 1 | `outbreak_share_of_cells` | 0.2152 |
| 1 | `outbreak_share_of_abs_error` | 0.5710 |
| 1 | `mae_outbreak_cells` | 44.5306 |
| 1 | `mae_endemic_cells` | 9.1745 |
| 4 | `total_residual_variance` | 5297.0272 |
| 4 | `between_district_eta2` | 0.0336 |
| 4 | `seasonal_r2` | 0.0028 |
| 4 | `outbreak_share_of_cells` | 0.2152 |
| 4 | `outbreak_share_of_abs_error` | 0.6068 |
| 4 | `mae_outbreak_cells` | 72.1774 |
| 4 | `mae_endemic_cells` | 12.8256 |

## Verdict

_To be written after review of the numbers above._

The decision table this feeds:

| Finding | Implication |
| --- | --- |
| Residuals ~ white, correlate with nothing | Ceiling is the data. Stop feature engineering; move to likelihood/calibration. |
| Residuals correlate with a channel already in the tensor | Architecture/loss lead, not a feature lead. |
| Residuals correlate with an outbreak-history / susceptibility proxy | Green light to engineer that feature. |
| Residual has spatial structure (neighbour errors correlate) | The graph question is not as closed as 8e says; revisit at h=4. |

## Output files

- `results/models/residual_diagnostics.csv`

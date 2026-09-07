# Measured climate lags

Cross-correlation of each climate feature at `t - lag` against
`log1p(cases)` at `t`, per district. Nothing is trained here; this is
the yardstick the hand-coded windows and the learned kernels are both
checked against.

Version: `lag-scan-v1`

## Configuration

| Setting | Value |
| --- | --- |
| Lags scanned | 0 to 25 |
| Folds | [1, 2, 3, 4, 5, 6, 7, 8, 9] |
| Districts | 25 |
| Features | 7 |
| Minimum pairs | 30 |

Every correlation uses `period_id <= fit_end_period` for its fold, so
no test year contributes to any number below.

## Warnings

- `rainfall_daily_mean_mm`: median deseasonalised peak correlation is only +0.036. Below 0.10 the peak lag is not resolvable and should not be quoted as a delay.
- `rainy_days_frac`: median deseasonalised peak correlation is only +0.075. Below 0.10 the peak lag is not resolvable and should not be quoted as a delay.
- `relative_humidity_mean`: median deseasonalised peak correlation is only +0.059. Below 0.10 the peak lag is not resolvable and should not be quoted as a delay.
- `wind_speed_mean`: median deseasonalised peak correlation is only +0.073. Below 0.10 the peak lag is not resolvable and should not be quoted as a delay.

## Peak lag by feature

Fold 9 (the longest history). `raw` is the correlation as-is;
`deseasonalised` removes the district x week-of-year mean from both
series first. Where the two disagree, the raw peak is the seasonal
cycles lining up rather than a delay.

`peak` is the most positive association, which is the hypothesis for
rainfall, temperature and humidity; `trough` is the most negative, and
is where a feature like wind speed should be read.

| Feature | Raw peak | Deseas. peak | Deseas. range | Mean r | Trough | Mean r |
| --- | --- | --- | --- | --- | --- | --- |
| `dewpoint_mean_c` | 9 | 5 | 2-10 | +0.191 | 24 | -0.009 |
| `relative_humidity_mean` | 7 | 7 | 2-9 | +0.077 | 24 | -0.118 |
| `rainy_days_frac` | 9 | 8 | 2-10 | +0.079 | 23 | -0.082 |
| `rainfall_daily_mean_mm` | 8 | 8 | 5-10 | +0.045 | 23 | -0.077 |
| `temperature_mean_c` | 19 | 17 | 0-25 | +0.199 | 8 | +0.083 |
| `diurnal_range_c` | 19 | 21 | 13-25 | +0.094 | 5 | -0.064 |
| `wind_speed_mean` | 24 | 24 | 0-25 | +0.048 | 8 | -0.084 |

## Rainfall lag by district

Fold 9, deseasonalised. This is the map the learned kernels
have to reproduce to be believable.

| District | Peak lag | r at peak | Raw peak lag |
| --- | --- | --- | --- |
| Batticaloa | 5 | +0.037 | 9 |
| Colombo | 5 | +0.106 | 5 |
| Jaffna | 5 | +0.008 | 5 |
| Puttalam | 5 | +0.081 | 8 |
| Nuwara Eliya | 6 | +0.072 | 9 |
| Vavuniya | 6 | +0.055 | 6 |
| Mannar | 7 | +0.088 | 7 |
| Gampaha | 7 | +0.040 | 7 |
| Trincomalee | 7 | +0.003 | 11 |
| Kegalle | 7 | +0.051 | 7 |
| Kalutara | 7 | +0.026 | 7 |
| Mullaitivu | 7 | +0.047 | 7 |
| Kandy | 8 | +0.039 | 8 |
| Ampara | 9 | +0.022 | 9 |
| Anuradhapura | 9 | +0.038 | 9 |
| Hambantota | 9 | +0.055 | 9 |
| Ratnapura | 9 | +0.002 | 9 |
| Matara | 9 | +0.065 | 9 |
| Polonnaruwa | 9 | +0.073 | 9 |
| Kurunegala | 9 | +0.022 | 11 |
| Galle | 9 | +0.043 | 9 |
| Moneragala | 9 | +0.045 | 9 |
| Badulla | 10 | +0.044 | 5 |
| Kilinochchi | 10 | +0.049 | 8 |
| Matale | 10 | +0.022 | 6 |

## Stability across folds

Standard deviation of a district's deseasonalised peak lag across
folds, in periods. A feature whose peak moves several periods
between folds is not a delay anyone should quote.

| Feature | Mean sd | Worst district sd |
| --- | --- | --- |
| `relative_humidity_mean` | 0.8 | 3.0 |
| `rainy_days_frac` | 0.8 | 2.8 |
| `dewpoint_mean_c` | 1.2 | 3.2 |
| `rainfall_daily_mean_mm` | 1.4 | 5.3 |
| `diurnal_range_c` | 2.0 | 6.7 |
| `wind_speed_mean` | 2.3 | 8.1 |
| `temperature_mean_c` | 4.3 | 8.5 |

## Output files

- `results/eda/lag_correlation.csv`
- `results/eda/lag_correlation_report.md`

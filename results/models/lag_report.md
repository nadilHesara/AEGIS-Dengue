# Learnable climate lags

4 arms on the same folds, seeds and masks. Only the climate
handling differs; the graph, GRU, head, target and loss are the baseline's.

| Arm | Features | Climate handling |
| --- | --- | --- |
| `hand_lags_v1` | `v1` | trailing 4, 8, 12 period means, hand-chosen, same for all districts |
| `learned_lags` | `v0` | learned per-district kernels over 26 periods |
| `no_lags_v0` | `v0` | none -- climate at the current period only |
| `no_climate` | `v0` minus climate | no climate channels at all -- cases, seasonality and geography only |

## Configuration

| Setting | Value |
| --- | --- |
| `lookback` | 12 |
| `horizon` | 1 |
| `hidden` | 32 |
| `gcn_layers` | 2 |
| `dropout` | 0.2 |
| `learning_rate` | 0.003 |
| `weight_decay` | 0.0001 |
| `batch_size` | 64 |
| `max_epochs` | 150 |
| `patience` | 15 |
| `seeds` | 1 |
| `target` | residual |
| `lag_reach` | 26 |
| `embedding_dim` | 8 |
| `kernel_learning_rate` | 0.02 |

## Results

Mean over the headline folds, averaged over seeds.

| Backbone | Arm | MAE | RMSE | Peak MAE | 2017 MAE | Seed sd |
| --- | --- | --- | --- | --- | --- | --- |
| gcn_gru | `no_lags_v0` | 18.07 | 38.27 | 29.14 | 48.43 | nan |
| gcn_gru | `hand_lags_v1` | 18.36 | 38.90 | 29.65 | 51.06 | nan |
| gcn_gru | `no_climate` | 18.71 | 41.00 | 30.28 | 52.78 | nan |
| gcn_gru | `learned_lags` | 20.67 | 44.45 | 32.82 | 65.18 | nan |

## Verdict

- **gcn_gru**: learned lags 20.67 MAE, worse than hand-coded windows at 18.36 (-12.6%).
- **gcn_gru**: dropping climate entirely costs +3.6% MAE (18.07 to 18.71). Read the delay arms against this: a delay treatment cannot be worth more than climate itself is.

## Cost of the longer window

The 26-period reach needs 37 input periods rather than 12, which costs
training windows in every fold.

| Arm | Max train windows |
| --- | --- |
| `hand_lags_v1` | 877 |
| `learned_lags` | 852 |
| `no_climate` | 877 |
| `no_lags_v0` | 877 |

## Learned delays

Centre of mass of each kernel, in reporting periods, over all folds,
seeds and districts. The spread across districts is the point: a
narrow range would mean the per-district embedding learned nothing.

| Feature | Mean lag | Min | Max |
| --- | --- | --- | --- |
| `dewpoint_mean_c` | 14.8 | 5.7 | 22.8 |
| `diurnal_range_c` | 13.7 | 8.2 | 18.5 |
| `rainfall_daily_mean_mm` | 16.0 | 1.4 | 22.8 |
| `rainy_days_frac` | 10.3 | 1.4 | 19.7 |
| `relative_humidity_mean` | 15.7 | 2.8 | 21.5 |
| `temperature_mean_c` | 14.1 | 2.4 | 22.8 |
| `wind_speed_mean` | 9.4 | 1.5 | 20.9 |

## Output files

- `results/models/lag_metrics.csv`
- `results/models/learned_lags.csv`

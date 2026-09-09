# GCN+GRU baseline

Version: `gcn-gru-v1`

Graph convolution over the 25 districts at each step, then a shared GRU
over the input window, then a linear head. Trained per walk-forward fold
with preprocessing refitted on that fold's history alone.

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
| `seeds` | 3 |
| `target` | residual |
| trainable parameters | 8,193 |

## Results

Mean over the headline folds (1, 2, 3, 6, 7, 8, 9), averaged over seeds.
COVID folds are excluded from the headline and reported separately.

| Model | Features | MAE | RMSE | Peak MAE | 2017 MAE | 2017 peak MAE | COVID MAE | Seed sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gru_only | `v1` | 16.68 | 35.25 | 28.06 | 42.97 | 48.35 | 7.18 | 0.53 |
| gru_only | `v0` | 16.86 | 35.94 | 28.43 | 43.53 | 48.98 | 7.23 | 0.39 |
| gcn_gru | `v0` | 18.60 | 39.68 | 29.81 | 51.87 | 58.56 | 7.29 | 0.30 |
| gcn_gru | `v1` | 18.98 | 40.36 | 30.36 | 54.81 | 62.15 | 7.35 | 0.68 |

### Naive baselines, on the same folds and masks

| Model | MAE | RMSE | Peak MAE | 2017 MAE | 2017 peak MAE |
| --- | --- | --- | --- | --- | --- |
| persistence | 16.42 | 33.61 | 26.61 | 36.08 | 39.42 |
| seasonal_naive | 55.45 | 118.76 | 84.81 | 111.10 | 126.65 |

## Verdict

Best configuration: **gru_only** on `v1`, headline MAE 16.68 against persistence at 16.42 (-1.5%).

Peak MAE 28.06 against persistence at 26.61 (-5.5%).

## Per fold

| Model | Features | Fold | Test year | MAE | RMSE | Peak MAE | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gcn_gru | `v0` | 1 | 2017 | 51.87 | 124.51 | 58.56 | epidemic |
| gcn_gru | `v0` | 2 | 2018 | 10.51 | 16.73 | 16.25 |  |
| gcn_gru | `v0` | 3 | 2019 | 17.71 | 36.55 | 34.43 |  |
| gcn_gru | `v0` | 4 | 2020 | 7.41 | 15.81 | 25.69 | COVID |
| gcn_gru | `v0` | 5 | 2021 | 7.17 | 19.86 | 32.89 | COVID |
| gcn_gru | `v0` | 6 | 2022 | 12.25 | 22.25 | 23.46 |  |
| gcn_gru | `v0` | 7 | 2023 | 18.70 | 42.49 | 33.69 |  |
| gcn_gru | `v0` | 8 | 2024 | 10.19 | 19.53 | 23.98 |  |
| gcn_gru | `v0` | 9 | 2025 | 8.97 | 15.70 | 18.32 |  |
| gcn_gru | `v1` | 1 | 2017 | 54.81 | 129.53 | 62.15 | epidemic |
| gcn_gru | `v1` | 2 | 2018 | 10.10 | 16.13 | 15.98 |  |
| gcn_gru | `v1` | 3 | 2019 | 17.63 | 37.06 | 34.38 |  |
| gcn_gru | `v1` | 4 | 2020 | 7.49 | 16.11 | 27.40 | COVID |
| gcn_gru | `v1` | 5 | 2021 | 7.21 | 19.75 | 33.99 | COVID |
| gcn_gru | `v1` | 6 | 2022 | 12.12 | 22.13 | 23.69 |  |
| gcn_gru | `v1` | 7 | 2023 | 19.16 | 42.77 | 35.50 |  |
| gcn_gru | `v1` | 8 | 2024 | 10.09 | 19.12 | 23.50 |  |
| gcn_gru | `v1` | 9 | 2025 | 8.98 | 15.76 | 17.34 |  |
| gru_only | `v0` | 1 | 2017 | 43.53 | 102.69 | 48.98 | epidemic |
| gru_only | `v0` | 2 | 2018 | 9.90 | 16.53 | 16.99 |  |
| gru_only | `v0` | 3 | 2019 | 16.81 | 34.95 | 32.76 |  |
| gru_only | `v0` | 4 | 2020 | 7.45 | 16.48 | 26.30 | COVID |
| gru_only | `v0` | 5 | 2021 | 7.02 | 18.52 | 38.05 | COVID |
| gru_only | `v0` | 6 | 2022 | 11.58 | 21.06 | 23.83 |  |
| gru_only | `v0` | 7 | 2023 | 18.07 | 40.91 | 34.41 |  |
| gru_only | `v0` | 8 | 2024 | 9.46 | 19.40 | 23.02 |  |
| gru_only | `v0` | 9 | 2025 | 8.67 | 16.08 | 19.04 |  |
| gru_only | `v1` | 1 | 2017 | 42.97 | 99.63 | 48.35 | epidemic |
| gru_only | `v1` | 2 | 2018 | 9.90 | 16.88 | 17.39 |  |
| gru_only | `v1` | 3 | 2019 | 16.82 | 35.65 | 33.14 |  |
| gru_only | `v1` | 4 | 2020 | 7.27 | 16.25 | 24.93 | COVID |
| gru_only | `v1` | 5 | 2021 | 7.09 | 18.14 | 38.63 | COVID |
| gru_only | `v1` | 6 | 2022 | 11.63 | 21.13 | 23.49 |  |
| gru_only | `v1` | 7 | 2023 | 17.95 | 39.87 | 34.26 |  |
| gru_only | `v1` | 8 | 2024 | 9.15 | 18.44 | 21.99 |  |
| gru_only | `v1` | 9 | 2025 | 8.35 | 15.17 | 17.81 |  |

## Reading this

`gcn_gru` uses the contiguity adjacency; `gru_only` is the identical model
with the adjacency replaced by the identity. The difference between them is
what the graph contributes. The difference between `v0` and `v1` is what
hand-specified 4, 8 and 12 period climate lags contribute, and it is the
number a learnable lag module has to beat.

## Output files

- `results/models/baseline_metrics.csv`
- `results/models/baseline_predictions.csv`

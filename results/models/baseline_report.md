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
| gru_only | `v0` | 16.67 | 35.22 | 28.32 | 41.74 | 46.82 | 7.27 | 0.28 |
| gru_only | `v1` | 17.12 | 36.69 | 28.70 | 46.01 | 51.98 | 7.24 | 0.70 |
| gcn_gru | `v1` | 19.01 | 40.63 | 30.47 | 55.01 | 62.40 | 7.39 | 0.62 |
| gcn_gru | `v0` | 19.08 | 41.07 | 30.46 | 55.23 | 62.49 | 7.30 | 0.84 |

### Naive baselines, on the same folds and masks

| Model | MAE | RMSE | Peak MAE | 2017 MAE | 2017 peak MAE |
| --- | --- | --- | --- | --- | --- |
| persistence | 16.42 | 33.61 | 26.61 | 36.08 | 39.42 |
| seasonal_naive | 55.45 | 118.76 | 84.81 | 111.10 | 126.65 |

## Verdict

Best configuration: **gru_only** on `v0`, headline MAE 16.67 against persistence at 16.42 (-1.5%).

Peak MAE 28.32 against persistence at 26.61 (-6.4%).

## Per fold

| Model | Features | Fold | Test year | MAE | RMSE | Peak MAE | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gcn_gru | `v0` | 1 | 2017 | 55.23 | 133.76 | 62.49 | epidemic |
| gcn_gru | `v0` | 2 | 2018 | 10.39 | 16.52 | 16.13 |  |
| gcn_gru | `v0` | 3 | 2019 | 17.90 | 37.01 | 34.79 |  |
| gcn_gru | `v0` | 4 | 2020 | 7.36 | 15.55 | 25.07 | COVID |
| gcn_gru | `v0` | 5 | 2021 | 7.23 | 19.74 | 33.01 | COVID |
| gcn_gru | `v0` | 6 | 2022 | 12.18 | 22.18 | 23.56 |  |
| gcn_gru | `v0` | 7 | 2023 | 18.73 | 42.88 | 34.20 |  |
| gcn_gru | `v0` | 8 | 2024 | 10.09 | 19.30 | 23.78 |  |
| gcn_gru | `v0` | 9 | 2025 | 8.99 | 15.82 | 18.26 |  |
| gcn_gru | `v1` | 1 | 2017 | 55.01 | 131.25 | 62.40 | epidemic |
| gcn_gru | `v1` | 2 | 2018 | 10.08 | 16.09 | 15.96 |  |
| gcn_gru | `v1` | 3 | 2019 | 17.57 | 37.10 | 34.14 |  |
| gcn_gru | `v1` | 4 | 2020 | 7.54 | 16.00 | 26.61 | COVID |
| gcn_gru | `v1` | 5 | 2021 | 7.25 | 19.61 | 35.07 | COVID |
| gcn_gru | `v1` | 6 | 2022 | 12.17 | 22.21 | 23.58 |  |
| gcn_gru | `v1` | 7 | 2023 | 19.07 | 42.57 | 34.97 |  |
| gcn_gru | `v1` | 8 | 2024 | 10.09 | 19.16 | 23.79 |  |
| gcn_gru | `v1` | 9 | 2025 | 9.06 | 16.04 | 18.42 |  |
| gru_only | `v0` | 1 | 2017 | 41.74 | 96.95 | 46.82 | epidemic |
| gru_only | `v0` | 2 | 2018 | 9.93 | 16.68 | 16.97 |  |
| gru_only | `v0` | 3 | 2019 | 17.03 | 35.46 | 33.37 |  |
| gru_only | `v0` | 4 | 2020 | 7.51 | 16.50 | 26.40 | COVID |
| gru_only | `v0` | 5 | 2021 | 7.02 | 18.49 | 39.05 | COVID |
| gru_only | `v0` | 6 | 2022 | 11.61 | 21.13 | 23.65 |  |
| gru_only | `v0` | 7 | 2023 | 18.34 | 41.30 | 35.68 |  |
| gru_only | `v0` | 8 | 2024 | 9.52 | 19.52 | 23.63 |  |
| gru_only | `v0` | 9 | 2025 | 8.48 | 15.49 | 18.12 |  |
| gru_only | `v1` | 1 | 2017 | 46.01 | 109.52 | 51.98 | epidemic |
| gru_only | `v1` | 2 | 2018 | 9.76 | 16.49 | 17.03 |  |
| gru_only | `v1` | 3 | 2019 | 16.82 | 35.79 | 33.09 |  |
| gru_only | `v1` | 4 | 2020 | 7.46 | 16.89 | 26.70 | COVID |
| gru_only | `v1` | 5 | 2021 | 7.03 | 18.21 | 38.36 | COVID |
| gru_only | `v1` | 6 | 2022 | 11.59 | 20.96 | 23.92 |  |
| gru_only | `v1` | 7 | 2023 | 18.17 | 40.36 | 35.36 |  |
| gru_only | `v1` | 8 | 2024 | 9.16 | 18.61 | 22.03 |  |
| gru_only | `v1` | 9 | 2025 | 8.31 | 15.06 | 17.48 |  |

## Reading this

`gcn_gru` uses the contiguity adjacency; `gru_only` is the identical model
with the adjacency replaced by the identity. The difference between them is
what the graph contributes. The difference between `v0` and `v1` is what
hand-specified 4, 8 and 12 period climate lags contribute, and it is the
number a learnable lag module has to beat.

## Output files

- `results/models/baseline_metrics.csv`
- `results/models/baseline_predictions.csv`

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
| gru_only | `v1` | 9.15 | 18.75 | 21.70 | nan | nan | nan | nan |
| gcn_gru | `v1` | 9.98 | 18.82 | 23.44 | nan | nan | nan | nan |

### Naive baselines, on the same folds and masks

| Model | MAE | RMSE | Peak MAE | 2017 MAE | 2017 peak MAE |
| --- | --- | --- | --- | --- | --- |
| persistence | 16.42 | 33.61 | 26.61 | 36.08 | 39.42 |
| seasonal_naive | 55.45 | 118.76 | 84.81 | 111.10 | 126.65 |

## Verdict

Best configuration: **gru_only** on `v1`, headline MAE 9.15 against persistence at 16.42 (+44.3%).

Peak MAE 21.70 against persistence at 26.61 (+18.5%).

## Per fold

| Model | Features | Fold | Test year | MAE | RMSE | Peak MAE | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gcn_gru | `v1` | 8 | 2024 | 9.98 | 18.82 | 23.44 |  |
| gru_only | `v1` | 8 | 2024 | 9.15 | 18.75 | 21.70 |  |

## Reading this

`gcn_gru` uses the contiguity adjacency; `gru_only` is the identical model
with the adjacency replaced by the identity. The difference between them is
what the graph contributes. The difference between `v0` and `v1` is what
hand-specified 4, 8 and 12 period climate lags contribute, and it is the
number a learnable lag module has to beat.

## Output files

- `results/models/baseline_metrics.csv`
- `results/models/baseline_predictions.csv`

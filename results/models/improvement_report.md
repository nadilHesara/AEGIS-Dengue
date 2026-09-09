# Improved GCN+GRU — objective and output-layer changes

Version: `improved-v1`

Every arm shares the baseline's architecture, graph, target
parameterisation, folds, masks, preprocessing, optimiser and early
stopping. Only the training objective changes. The loop is imported from
`scripts/16.train_gcn_gru.py` rather than copied, so a difference in the
numbers cannot come from a difference in the loop.

The `baseline` arm is script 16's own configuration run through this
script's harness. It is the control: if it does not reproduce script 16's
MAE, the harness is not neutral and nothing below is comparable.

## Why these changes

Two measurements from the panel, both computed on the headline test
cells:

- A constant log-space error of 0.20 is **1.3 cases** at a level of 5 and
  **332 cases** at a level of 1500. The baseline's log-MSE spends equal
  effort on both; MAE counts the second roughly 250x more.
- The sd of the log1p week-over-week change is **0.79** for cells under 10
  cases and **0.375** for cells over 200. The cells log-MSE over-weights
  are also the noisiest ones in log space.

14% of headline test cells carry 61% of the case volume. The baseline
allocates most of its capacity to the cells that contribute least to the
reported number, and does so where the signal is worst.

## Arms

| Arm | Loss | Level-weighted | Rationale |
| --- | --- | --- | --- |
| `baseline` | masked MSE | no | script 16, the control |
| `huber` | masked Huber | no | bounds the gradient of heavy-tailed residuals |
| `level_weighted` | masked MSE | yes | matches the training distribution to the metric |
| `huber_weighted` | masked Huber | yes | both |
| `quantile` | pinball at tau=0.5 | yes | MAE is minimised by the median, MSE estimates the mean |

Weights are `log1p(cases)` at the forecast origin, normalised to mean 1
over each batch's observed cells. The origin is already a model input,
so the weighting adds no information the model does not have.
Huber transition point: 0.5 in log1p units.

## Configuration

| Setting | Value |
| --- | --- |
| `variant` | v1 |
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

## Results — single seed models

Mean over the headline folds (1, 2, 3, 6, 7, 8, 9), averaged over seeds.
COVID folds are excluded from the headline and reported separately.

| Arm | MAE | vs baseline | RMSE | Peak MAE | 2017 MAE | 2017 peak MAE | COVID MAE | Seed sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `level_weighted` | 16.59 | +12.6% | 34.30 | 28.14 | 38.06 | 42.32 | 7.14 | 0.57 |
| `quantile` | 17.14 | +9.7% | 35.69 | 27.91 | 42.75 | 47.88 | 7.02 | 0.36 |
| `huber_weighted` | 17.71 | +6.7% | 37.21 | 28.86 | 46.66 | 52.58 | 7.13 | 0.51 |
| `huber` | 18.17 | +4.3% | 38.30 | 29.06 | 49.76 | 56.29 | 7.21 | 0.65 |
| `baseline` | 18.98 | — | 40.36 | 30.36 | 54.81 | 62.15 | 7.35 | 0.68 |

## Results — seed-mean ensembles

The per-seed test predictions averaged **before** the metric rather than
after. Free, since the seeds are already trained. This is a variance
reduction, not a better model, which is why it is reported separately
from the loss comparison above.

| Arm | MAE | vs baseline single | RMSE | Peak MAE | 2017 MAE | 2017 peak MAE | COVID MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `level_weighted` | 16.36 | +13.8% | 33.63 | 27.83 | 36.92 | 41.01 | 7.11 |
| `quantile` | 17.03 | +10.3% | 35.51 | 27.76 | 42.21 | 47.29 | 7.00 |
| `huber_weighted` | 17.61 | +7.2% | 36.99 | 28.72 | 46.22 | 52.11 | 7.11 |
| `huber` | 18.09 | +4.7% | 38.13 | 28.93 | 49.48 | 55.99 | 7.19 |
| `baseline` | 18.88 | +0.5% | 40.19 | 30.21 | 54.45 | 61.77 | 7.32 |

### Naive baselines, on the same folds and masks

| Model | MAE | RMSE | Peak MAE | 2017 MAE | 2017 peak MAE |
| --- | --- | --- | --- | --- | --- |
| persistence | 16.42 | 33.61 | 26.61 | 36.08 | 39.42 |
| seasonal_naive | 55.45 | 118.76 | 84.81 | 111.10 | 126.65 |

## Per fold, MAE (single seed models, averaged over seeds)

| Fold | Year | `baseline` | `huber` | `huber_weighted` | `level_weighted` | `quantile` | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2017 | 54.81 | 49.76 | 46.66 | 38.06 | 42.75 | epidemic |
| 2 | 2018 | 10.10 | 10.06 | 9.88 | 9.95 | 9.87 |  |
| 3 | 2019 | 17.63 | 17.30 | 17.63 | 17.96 | 17.59 |  |
| 4 | 2020 | 7.49 | 7.21 | 7.01 | 7.03 | 6.83 | COVID |
| 5 | 2021 | 7.21 | 7.22 | 7.24 | 7.26 | 7.20 | COVID |
| 6 | 2022 | 12.12 | 12.30 | 12.32 | 12.24 | 12.41 |  |
| 7 | 2023 | 19.16 | 18.80 | 18.49 | 18.73 | 18.35 |  |
| 8 | 2024 | 10.09 | 10.05 | 10.11 | 10.09 | 10.13 |  |
| 9 | 2025 | 8.98 | 8.92 | 8.89 | 9.13 | 8.91 |  |

## Verdict

Best single-seed arm: **`level_weighted`** at headline MAE 16.59, against the baseline's 18.98 (+12.6%) and persistence at 16.42 (-1.0%).

Best ensemble: **`level_weighted`** at headline MAE 16.36 (+13.8% against the baseline single-seed model).

**Read the seed sd column before believing any gap.** A difference
smaller than the seed sd is not established by this run.

## Output files

- `results/models/improvement_metrics.csv`
- `results/models/improvement_predictions.csv`

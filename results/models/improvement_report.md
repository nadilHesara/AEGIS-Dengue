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
| `level_weighted` | 16.69 | +12.2% | 34.38 | 28.30 | 38.52 | 42.88 | 7.29 | 0.66 |
| `quantile` | 17.32 | +8.8% | 36.26 | 28.14 | 44.26 | 49.72 | 7.00 | 0.29 |
| `huber_weighted` | 17.36 | +8.7% | 36.35 | 28.46 | 44.24 | 49.75 | 7.05 | 0.44 |
| `huber` | 17.41 | +8.4% | 36.54 | 28.21 | 44.19 | 49.65 | 7.30 | 0.34 |
| `baseline` | 19.01 | — | 40.63 | 30.47 | 55.01 | 62.40 | 7.39 | 0.62 |

## Results — seed-mean ensembles

The per-seed test predictions averaged **before** the metric rather than
after. Free, since the seeds are already trained. This is a variance
reduction, not a better model, which is why it is reported separately
from the loss comparison above.

| Arm | MAE | vs baseline single | RMSE | Peak MAE | 2017 MAE | 2017 peak MAE | COVID MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `level_weighted` | 16.39 | +13.8% | 33.60 | 27.86 | 37.11 | 41.25 | 7.25 |
| `quantile` | 17.24 | +9.3% | 36.05 | 28.01 | 44.00 | 49.45 | 7.00 |
| `huber_weighted` | 17.26 | +9.2% | 36.13 | 28.29 | 43.93 | 49.41 | 7.04 |
| `huber` | 17.35 | +8.7% | 36.41 | 28.10 | 44.08 | 49.54 | 7.29 |
| `baseline` | 18.92 | +0.4% | 40.45 | 30.29 | 54.82 | 62.21 | 7.35 |

### Naive baselines, on the same folds and masks

| Model | MAE | RMSE | Peak MAE | 2017 MAE | 2017 peak MAE |
| --- | --- | --- | --- | --- | --- |
| persistence | 16.42 | 33.61 | 26.61 | 36.08 | 39.42 |
| seasonal_naive | 55.45 | 118.76 | 84.81 | 111.10 | 126.65 |

## Per fold, MAE (single seed models, averaged over seeds)

| Fold | Year | `baseline` | `huber` | `huber_weighted` | `level_weighted` | `quantile` | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2017 | 55.01 | 44.19 | 44.24 | 38.52 | 44.26 | epidemic |
| 2 | 2018 | 10.08 | 10.17 | 9.90 | 9.97 | 9.87 |  |
| 3 | 2019 | 17.57 | 17.51 | 17.60 | 17.85 | 17.30 |  |
| 4 | 2020 | 7.54 | 7.40 | 6.86 | 7.32 | 6.78 | COVID |
| 5 | 2021 | 7.25 | 7.21 | 7.24 | 7.27 | 7.23 | COVID |
| 6 | 2022 | 12.17 | 12.23 | 12.34 | 12.20 | 12.38 |  |
| 7 | 2023 | 19.07 | 18.76 | 18.38 | 19.13 | 18.42 |  |
| 8 | 2024 | 10.09 | 10.08 | 10.11 | 10.04 | 10.07 |  |
| 9 | 2025 | 9.06 | 8.92 | 8.92 | 9.12 | 8.97 |  |

## Verdict

Best single-seed arm: **`level_weighted`** at headline MAE 16.69, against the baseline's 19.01 (+12.2%) and persistence at 16.42 (-1.6%).

Best ensemble: **`level_weighted`** at headline MAE 16.39 (+13.8% against the baseline single-seed model).

**Read the seed sd column before believing any gap.** A difference
smaller than the seed sd is not established by this run.

## Output files

- `results/models/improvement_metrics.csv`
- `results/models/improvement_predictions.csv`

# Naive baselines

Version: `naive-v1`

Lookback 12, horizon 1. Metrics are on the
original case scale over observed cells only. Peak MAE covers cells at or
above each district's 90th percentile, fitted on the
fold's history.

## Summary

| Model | Headline MAE | Headline RMSE | Headline peak MAE | 2017 MAE | 2017 peak MAE | COVID MAE |
| --- | --- | --- | --- | --- | --- | --- |
| persistence | 16.42 | 33.61 | 26.61 | 36.08 | 39.42 | 7.32 |
| seasonal_naive | 55.45 | 118.76 | 84.81 | 111.10 | 126.65 | 44.48 |

Headline is the mean over folds 1, 2, 3, 6, 7, 8 and 9. The COVID folds
are averaged separately and never folded into the headline.

## Per fold

| Model | Fold | Test year | MAE | RMSE | Peak MAE | Peak cells | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 1 | 2017 | 36.08 | 80.93 | 39.42 | 1114 | epidemic |
| persistence | 2 | 2018 | 10.45 | 16.72 | 16.34 | 273 |  |
| persistence | 3 | 2019 | 17.30 | 35.09 | 32.31 | 476 |  |
| persistence | 4 | 2020 | 7.41 | 16.27 | 26.62 | 117 | COVID |
| persistence | 5 | 2021 | 7.23 | 20.80 | 35.48 | 31 | COVID |
| persistence | 6 | 2022 | 12.57 | 22.82 | 23.66 | 116 |  |
| persistence | 7 | 2023 | 18.83 | 42.68 | 32.58 | 363 |  |
| persistence | 8 | 2024 | 10.78 | 21.57 | 25.77 | 145 |  |
| persistence | 9 | 2025 | 8.95 | 15.50 | 16.19 | 103 |  |
| seasonal_naive | 1 | 2017 | 111.10 | 242.02 | 126.65 | 1114 | epidemic |
| seasonal_naive | 2 | 2018 | 104.20 | 230.90 | 125.26 | 273 |  |
| seasonal_naive | 3 | 2019 | 51.67 | 104.89 | 109.08 | 476 |  |
| seasonal_naive | 4 | 2020 | 65.60 | 136.33 | 64.50 | 117 | COVID |
| seasonal_naive | 5 | 2021 | 23.36 | 51.87 | 60.71 | 31 | COVID |
| seasonal_naive | 6 | 2022 | 21.00 | 36.41 | 41.65 | 116 |  |
| seasonal_naive | 7 | 2023 | 40.12 | 85.19 | 84.98 | 363 |  |
| seasonal_naive | 8 | 2024 | 37.58 | 78.03 | 54.00 | 145 |  |
| seasonal_naive | 9 | 2025 | 22.50 | 53.90 | 52.09 | 103 |  |

## What these numbers are for

A trained model must beat both, on headline MAE **and** on peak MAE. Beating
mean MAE alone is achievable by predicting close to the recent level every
week, which is what persistence already does for free.

## Output files

- `results/models/naive_baseline_metrics.csv`

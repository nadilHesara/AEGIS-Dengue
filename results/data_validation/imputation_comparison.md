# Imputation Comparison

- Panel rows: 25,300
- Train split source year end: 2023
- Validation split source year start: 2024
- Test split source year start: 2025
- Forward fill max gap: 2

## Method Ranking

| method                     |   target_mae |   target_rmse |   target_bias |   target_missing_rate |   climate_mae |   climate_rmse |   climate_bias |   climate_missing_rate |
|:---------------------------|-------------:|--------------:|--------------:|----------------------:|--------------:|---------------:|---------------:|-----------------------:|
| forward_fill               |       0.0000 |        0.0000 |        0.0000 |                0.0000 |        0.0000 |         0.0000 |         0.0000 |                 0.0030 |
| forward_fill_train_only    |       0.0000 |        0.0000 |        0.0000 |                0.0000 |        0.0000 |         0.0000 |         0.0000 |                 0.0030 |
| district_median            |      24.9482 |       75.2471 |      -16.9481 |                0.0000 |        4.7710 |        15.8730 |        -1.5539 |                 0.0030 |
| district_month_climatology |      26.5770 |       69.1531 |       -1.5087 |                0.0000 |        4.0325 |        13.4587 |        -0.0274 |                 0.0030 |
| seasonal_climatology       |      27.0226 |       70.5572 |       -1.4674 |                0.0000 |        4.2316 |        13.7784 |        -0.0155 |                 0.0030 |

## Leakage Rules

- Fit all statistics on the training split only.
- Keep dengue target missingness separate from climate-feature missingness.
- Preserve the frozen master panel unchanged; emit derived outputs only.

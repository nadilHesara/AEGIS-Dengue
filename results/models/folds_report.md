# Walk-forward folds

Version: `folds-v1`

Each fold trains on everything before a calendar year and tests on that
year. Fold years come from `start_date`, never from `source_year`.

## Folds

| Fold | Test year | Train | Val | Test | Note |
| --- | --- | --- | --- | --- | --- |
| 1 | 2017 | 1-471 | 472-524 | 525-576 | epidemic |
| 2 | 2018 | 1-524 | 525-576 | 577-628 | headline |
| 3 | 2019 | 1-576 | 577-628 | 629-680 | headline |
| 4 | 2020 | 1-628 | 629-680 | 681-732 | COVID, report separately |
| 5 | 2021 | 1-680 | 681-732 | 733-784 | COVID, report separately |
| 6 | 2022 | 1-732 | 733-784 | 785-837 | headline |
| 7 | 2023 | 1-784 | 785-837 | 838-889 | headline |
| 8 | 2024 | 1-837 | 838-889 | 890-941 | headline |
| 9 | 2025 | 1-889 | 890-941 | 942-993 | headline |

## Test fold dates

| Fold | Test year | From | To |
| --- | --- | --- | --- |
| 1 | 2017 | 2017-01-07 | 2018-01-05 |
| 2 | 2018 | 2018-01-06 | 2019-01-04 |
| 3 | 2019 | 2019-01-05 | 2020-01-03 |
| 4 | 2020 | 2020-01-04 | 2021-01-01 |
| 5 | 2021 | 2021-01-02 | 2021-12-31 |
| 6 | 2022 | 2022-01-01 | 2023-01-06 |
| 7 | 2023 | 2023-01-07 | 2024-01-05 |
| 8 | 2024 | 2024-01-06 | 2025-01-03 |
| 9 | 2025 | 2025-01-04 | 2026-01-04 |

## Window counts

Lookback 12, horizon 1, after imputation.

| Fold | Test year | Train | Val | Test | Test cells |
| --- | --- | --- | --- | --- | --- |
| 1 | 2017 | 459 | 53 | 52 | 1300 |
| 2 | 2018 | 512 | 52 | 52 | 1300 |
| 3 | 2019 | 564 | 52 | 52 | 1300 |
| 4 | 2020 | 616 | 52 | 52 | 1300 |
| 5 | 2021 | 668 | 52 | 52 | 1300 |
| 6 | 2022 | 720 | 52 | 53 | 1325 |
| 7 | 2023 | 772 | 53 | 52 | 1300 |
| 8 | 2024 | 825 | 52 | 52 | 1300 |
| 9 | 2025 | 877 | 52 | 52 | 1300 |

Headline folds (COVID excluded): 1, 2, 3, 6, 7, 8, 9

COVID folds: 4, 5

## Imputation

District x month climatology, fitted on each fold's history only.
Cells filled per fold:

| Variant | Cells filled (fold 1) | Cells filled (fold 9) |
| --- | --- | --- |
| `v0` | 0 | 0 |
| `v1` | 1575 | 1575 |

Targets are never imputed. A missing case count stays NaN, is flagged
in `y_mask`, and is excluded from the loss and from every metric.

## Output files

- `data/processed/folds.json`
- `data/processed/fold_preprocessing_v0.npz`
- `data/processed/fold_preprocessing_v1.npz`

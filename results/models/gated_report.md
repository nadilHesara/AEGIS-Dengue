# GCN+GRU with a spatial/temporal fusion gate

Version: `gated-v1`

Per district, a learned scalar `g` blends the graph convolution over the
real adjacency against the same convolution over the identity. `g = 1`
for every district is `gcn_gru`; `g = 0` is `gru_only`. Only the blend
moves; the graph, GRU, head, target and loss are the baseline's, imported
from `scripts/16`.

| Arm | Adjacency | Gate |
| --- | --- | --- |
| `gcn_gru` | contiguity | none -- the control |
| `gru_only` | identity | none -- the number to beat |
| `gated` | contiguity | per-district, learned |
| `gated_uniform` | contiguity | frozen at 0.5 |

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
| `variants` | ['v1'] |

## Results

Mean over the headline folds (1, 2, 3, 6, 7, 8, 9), averaged over seeds.
COVID folds are excluded from the headline and reported separately.

| Arm | Features | MAE | RMSE | Peak MAE | 2017 MAE | COVID MAE | Seed sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gru_only | `v1` | 16.88 | 35.98 | 28.49 | 44.17 | 7.23 | 0.29 |
| gated | `v1` | 18.36 | 39.77 | 30.27 | 53.17 | 7.23 | 0.36 |
| gated_uniform | `v1` | 18.53 | 40.14 | 30.44 | 53.79 | 7.46 | 0.33 |
| gcn_gru | `v1` | 19.00 | 40.60 | 30.48 | 55.00 | 7.52 | 0.66 |

## Verdict

`gated` headline MAE 18.36 does not beat `gru_only` at 16.88 (-8.8%); `gcn_gru` is 19.00.

**Read the seed sd column before believing any gap.** A difference
smaller than the seed sd is not established by this run.

## Per fold, MAE

| Arm | Features | Fold | Test year | MAE | RMSE | Peak MAE | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gated | `v1` | 1 | 2017 | 53.17 | 128.49 | 60.38 | epidemic |
| gated | `v1` | 2 | 2018 | 9.99 | 16.30 | 16.32 |  |
| gated | `v1` | 3 | 2019 | 17.28 | 36.99 | 34.21 |  |
| gated | `v1` | 4 | 2020 | 7.44 | 16.56 | 26.15 | COVID |
| gated | `v1` | 5 | 2021 | 7.02 | 18.07 | 33.80 | COVID |
| gated | `v1` | 6 | 2022 | 11.57 | 21.13 | 23.53 |  |
| gated | `v1` | 7 | 2023 | 18.50 | 41.13 | 36.43 |  |
| gated | `v1` | 8 | 2024 | 9.40 | 18.59 | 21.88 |  |
| gated | `v1` | 9 | 2025 | 8.62 | 15.78 | 19.13 |  |
| gated_uniform | `v1` | 1 | 2017 | 53.79 | 129.80 | 61.09 | epidemic |
| gated_uniform | `v1` | 2 | 2018 | 10.00 | 16.29 | 16.28 |  |
| gated_uniform | `v1` | 3 | 2019 | 17.54 | 37.99 | 34.83 |  |
| gated_uniform | `v1` | 4 | 2020 | 7.87 | 17.40 | 27.82 | COVID |
| gated_uniform | `v1` | 5 | 2021 | 7.04 | 18.10 | 33.37 | COVID |
| gated_uniform | `v1` | 6 | 2022 | 11.65 | 21.29 | 23.73 |  |
| gated_uniform | `v1` | 7 | 2023 | 18.56 | 41.37 | 36.47 |  |
| gated_uniform | `v1` | 8 | 2024 | 9.50 | 18.50 | 21.90 |  |
| gated_uniform | `v1` | 9 | 2025 | 8.69 | 15.73 | 18.78 |  |
| gcn_gru | `v1` | 1 | 2017 | 55.00 | 130.98 | 62.37 | epidemic |
| gcn_gru | `v1` | 2 | 2018 | 10.08 | 16.09 | 15.96 |  |
| gcn_gru | `v1` | 3 | 2019 | 17.59 | 37.12 | 34.24 |  |
| gcn_gru | `v1` | 4 | 2020 | 7.78 | 16.83 | 28.91 | COVID |
| gcn_gru | `v1` | 5 | 2021 | 7.26 | 19.62 | 35.12 | COVID |
| gcn_gru | `v1` | 6 | 2022 | 12.17 | 22.21 | 23.59 |  |
| gcn_gru | `v1` | 7 | 2023 | 19.05 | 42.57 | 34.92 |  |
| gcn_gru | `v1` | 8 | 2024 | 10.09 | 19.16 | 23.80 |  |
| gcn_gru | `v1` | 9 | 2025 | 9.06 | 16.06 | 18.44 |  |
| gru_only | `v1` | 1 | 2017 | 44.17 | 103.78 | 49.76 | epidemic |
| gru_only | `v1` | 2 | 2018 | 9.76 | 16.50 | 17.03 |  |
| gru_only | `v1` | 3 | 2019 | 16.92 | 36.22 | 33.36 |  |
| gru_only | `v1` | 4 | 2020 | 7.40 | 16.66 | 26.37 | COVID |
| gru_only | `v1` | 5 | 2021 | 7.07 | 18.17 | 38.46 | COVID |
| gru_only | `v1` | 6 | 2022 | 11.59 | 20.95 | 23.98 |  |
| gru_only | `v1` | 7 | 2023 | 18.14 | 40.37 | 35.44 |  |
| gru_only | `v1` | 8 | 2024 | 9.14 | 18.50 | 21.73 |  |
| gru_only | `v1` | 9 | 2025 | 8.46 | 15.51 | 18.11 |  |

## Learned gate per district

`g` near 1 means the district keeps the graph convolution; near 0
means it routes past it. Mean over folds and seeds, with the spread.
The check is whether the low-degree northern districts (Jaffna,
Mannar, Mullaitivu, Kilinochchi) gate lower than the well-connected
ones.

| District | Mean g | Min | Max |
| --- | --- | --- | --- |
| Puttalam | 0.330 | 0.224 | 0.494 |
| Matale | 0.334 | 0.230 | 0.496 |
| Kegalle | 0.337 | 0.220 | 0.495 |
| Kurunegala | 0.340 | 0.244 | 0.501 |
| Trincomalee | 0.344 | 0.243 | 0.501 |
| Badulla | 0.353 | 0.250 | 0.502 |
| Polonnaruwa | 0.358 | 0.289 | 0.500 |
| Ratnapura | 0.360 | 0.277 | 0.499 |
| Anuradhapura | 0.360 | 0.264 | 0.496 |
| Ampara | 0.363 | 0.280 | 0.497 |
| Kalutara | 0.363 | 0.286 | 0.501 |
| Moneragala | 0.367 | 0.296 | 0.498 |
| Matara | 0.369 | 0.292 | 0.502 |
| Hambantota | 0.374 | 0.296 | 0.497 |
| Vavuniya | 0.379 | 0.262 | 0.499 |
| Kandy | 0.382 | 0.256 | 0.504 |
| Nuwara Eliya | 0.384 | 0.310 | 0.502 |
| Mannar | 0.385 | 0.316 | 0.500 |
| Colombo | 0.395 | 0.289 | 0.500 |
| Kilinochchi | 0.399 | 0.282 | 0.504 |
| Mullaitivu | 0.406 | 0.324 | 0.502 |
| Batticaloa | 0.413 | 0.336 | 0.498 |
| Galle | 0.420 | 0.358 | 0.500 |
| Gampaha | 0.456 | 0.413 | 0.499 |
| Jaffna | 0.530 | 0.494 | 0.557 |

## Output files

- `results/models/gated_metrics.csv`
- `results/models/gated_gates.csv`

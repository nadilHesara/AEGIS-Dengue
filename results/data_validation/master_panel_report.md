# Master weekly modelling panel

Frozen version: `data-v1`
Creation commit: `435864a3fd20e2d4893df4a760f5e93e53c79a5b`

Structural rows: 25300
Reporting periods: 1012
Canonical districts: 25
Observed case rows: 25299
Observed climate rows: 25225
Fully observed rows: 25224
Incomplete rows written to: `results\data_validation\master_panel_missingness.csv`

## Required checks

- Every period contains exactly 25 structural rows: `True`
- 2026 week 7 observed case rows: `24`

## Output files

- `data\processed\panel_weekly.parquet`
- `results\data_validation\master_panel_missingness.csv`

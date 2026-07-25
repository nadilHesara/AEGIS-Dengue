# Master weekly modelling panel

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

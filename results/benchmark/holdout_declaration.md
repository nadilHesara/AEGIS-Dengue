# Hold-out declaration: 2026 (fold 10)

Written 29 September 2026, before any model in the benchmark produced a
forecast for a 2026 target week. Nothing below is changed after the hold-out
is scored; later additions go in a dated section at the end.

## Why a hold-out

The 2017–2025 test years guided successive design choices (level weighting,
the quantile head, feature variants, the Negative Binomial head). A
multiple-comparison adjustment does not undo that. The only weeks no
experiment has scored are the first 19 reporting periods of 2026
(period_id 994–1012, 2026-01-05 to 2026-05-17), so they are reserved as a
frozen-specification check.

## Fold

- Test: periods starting in 2026 (19 weeks). Validation: 2025. Training: up to
  the end of 2024. Built by `scripts/14.build_folds` exactly like folds 1–9.
- Horizons 1, 2, 3, 4, 6, 8, 10, 12. Every horizon scores the same 19 target
  weeks; origins may lie in 2025.
- Climate: the frozen Open-Meteo file recorded in `data_manifest.json`.
- Never used for calibration, stacking, member selection, alarm cutoffs or
  any tuning. Conformal intervals and alarm cutoffs for fold 10 come from
  fold 9 (2025), as for every other fold.

## Primary comparisons (declared)

Each against same-horizon persistence, MAE on the 2026 cells:

| Model | Why it is primary |
| --- | --- |
| `gru_v2_quantile_lw` | best GRU configuration of the 24 September benchmark |
| `nb_shared_v2` | the Negative Binomial model at the benchmark's standard v2 features (v3/v4 are exploratory) |
| `chronos2_joint` | best zero-shot foundation model of the 24 September benchmark |
| `lgbm_v2` | strongest tabular baseline |
| `ens_equal` | the pre-specified equal-weight ensemble |

Secondary (reported, not confirmatory): WIS and 95% coverage of the
probabilistic models above; peak MAE.

## What counts as confirmation

- One year, 19 weeks, 25 districts: there is no paired-over-years test. The
  2026 result is reported as a point estimate with a time-block bootstrap 95%
  interval (4-week blocks resampled jointly across districts).
- A claim from 2017–2025 is described as "consistent with the hold-out" when
  the 2026 difference has the same sign, and "not replicated" otherwise. With
  19 weeks, an interval that includes zero is expected and is reported as such.
- The January–May window contains the first transmission peak of the year in
  most districts but not the second (monsoon) peak; results are not a
  full-year estimate.

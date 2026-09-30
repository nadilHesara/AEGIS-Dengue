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

## Addendum 2026-09-30 (15:18 +0530): climate-horizon extension

Written before any model of this extension has been trained or has produced
a forecast for a 2026 target week.

**Extension.** "Learning climate delays from informative forecast horizons":
a two-branch model (case-history GRU + learnable climate-delay encoder),
Negative Binomial heads for h = 1–4, identity backbone, Open-Meteo climate.
Plan: `docs/climate_horizon_implementation.md`. Outputs go only to
`results/climate_horizon/`.

**Declared arms for fold 10:**
- `hcd_uniform`: climate-encoder gradient weights [1, 1, 1, 1] (baseline).
- `hcd_informed`: weights from the primary utility, the inner-validation
  shuffle gain of a frozen equal-weight pilot, computed only on periods up
  to fold 10's training end (2024). Rule:
  w_h = 0.95·H·max(g_h,0)/Σmax(g,0) + 0.05, or uniform if every g_h ≤ 0.

**Primary 2026 comparison:** `hcd_informed` vs `hcd_uniform`, MAE on the
2026 cells, h = 1–4, reported with a 4-week block-bootstrap 95% interval,
under the confirmation wording above. Secondary: each against same-horizon
persistence; peak MAE; WIS. Ablation arms are not scored on 2026.

**What had been inspected before this addendum.**
- The benchmark's fold-10 results were already scored on 29–30 Sep 2026
  (`results/benchmark/tables/holdout_2026.csv`, `holdout_bootstrap.csv`).
  Their summary was read: "all five declared models beat persistence at
  every horizon; most bootstrap intervals include zero"
  (`docs/revision_2026_09_30.md`).
- The 2026 case counts are in `data/processed` tensors and have been used
  by the benchmark. Their values were not examined for this extension.
- No 2026 forecast or score exists for any arm of this extension.

Because 2026 aggregate outcomes were already known for other models, this
fold is a frozen-specification check for the extension, not a blind test.


## Addendum 2026-09-30 (16:44 +0530): climate-horizon extension, frozen specification

Linked to the declaration of 29 September 2026 and to the addendum of
2026-09-30 15:18 above. Written before any model of this extension has
produced a forecast for a 2026 target week, and before any outer test year
(2017–2025) of this extension has been scored.

**Frozen run manifest.**
`results/climate_horizon/run_manifest_2026-09-30.json`, SHA-256
`f4acdf1c65311fb1116cf64d46803245619f58be7434f415bb6b3f30fd1245cd`
(stored beside it in `.sha256`; both files are read-only). The manifest
fixes:
- the architecture and the NB likelihood;
- the utility definitions and aggregation (no smoothing), the shuffle and
  donor rules, the pilot blocks and seeds;
- the weight rule (ρ = 0.95);
- the arms, seeds (5 for B and D, 3 for the rest) and folds;
- early stopping and metric aggregation;
- the primary comparison, the uncertainty analysis and the decision
  criteria;
- the exact commands and code/data hashes.

`scripts/training/45.run_climate_horizon_sweep.py` refuses to run if the
manifest hash has changed. It refuses fold 10 without `--holdout
--confirm-holdout`.

**Declared arms for 2026 (unchanged from the 15:18 addendum):**
`B_target_equal` (baseline) and `D_target_measured` (proposed), 5 seeds
each. Fold 10 is trained on 2007–2024, early-stopped on 2025, and
preprocessed on training years only. Its utility pilots use inner blocks
scoring 2022, 2023 and 2024. Every fit, scaler, pilot and weight used for
2026 uses only data up to the end of 2025, and only up to 2024 for
training and pilots.

**Order.** The retrospective evaluation (folds 1–9, test years
2017–2025) is run and reported first. 2026 is scored once, afterwards,
with the frozen specification, and reported separately. The
retrospective results are not a hold-out result.

**Prior exposure to 2026, stated plainly.**
- The benchmark's fold-10 results for five other models were scored on
  29–30 September and their summary was read.
- The 2026 case counts sit in the shared tensors. They were not examined
  for this extension, and no extension model has forecast them.

A declaration written now cannot restore an untouched hold-out. For this
extension, 2026 is a frozen-specification check on 19 weeks, not a blind
test. It is reported with the confirmation wording of the 29 September
declaration.

**Approval.** The repository defines no approval step for hold-out scoring.
Its CI only runs tests on pull requests to `main`/`develop`. No supervisor
or reviewer approval has been obtained, and none is claimed.

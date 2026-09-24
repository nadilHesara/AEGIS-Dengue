# Long-horizon benchmark — h = 1–12, strong baselines, foundation models, intervals and early warning

`scripts/data/5c.fetch_open_meteo_climate.py`, `src/evaluation/long_horizon.py`,
`scripts/training/31.long_horizon_neural.py`, `32.long_horizon_tabular.py`,
`33.long_horizon_foundation.py`, `36.onset_classifier.py`,
`scripts/evaluation/34.long_horizon_analysis.py`, `35.long_horizon_figures.py`,
`tests/test_long_horizon.py`. Outputs: `results/benchmark/` (tables, figures,
per-cell predictions). Run on 2026-09-24.

## Why this experiment

The short paper has one positive result (the identity GRU beats same-horizon
persistence at h=3–4, and half of that is climate) and three gaps a reviewer
will point at:

1. **Weak baselines.** Only persistence and seasonal naive. No tree model, no
   linear autoregression, no pretrained forecaster. The two closest Sri Lanka
   papers (Weng et al., IEEE BigData 2024; GulMohamed et al., Sci. Rep. 2026)
   report GNNs beating ARIMA/RF/LSTM **without ever comparing to persistence**.
2. **Horizons stop at 4 weeks**, while the measured rainfall→dengue delay is
   5–10 weeks (scripts/17; Liu et al. 2025 find 7–9 weeks in Sri Lanka). We
   could not say where climate's value peaks.
3. **Point forecasts only, scored only by MAE.** An early-warning paper needs
   calibrated intervals and an outbreak-detection evaluation (the Dengue
   Forecasting Sprint in Brazil and the Vietnam superensemble work score WIS,
   coverage and outbreak detection).

## ⚠ Data caveat — climate was reconstructed

The GEE-extracted ERA5-Land CSVs behind the short paper are not on this machine
or the team Drive. `scripts/data/5c` rebuilds `data/raw/climate_daily_district.csv`
from Open-Meteo `era5_seamless` (ERA5-Land temperature/dew point/humidity, ERA5
rain/wind — the same reanalysis family), **one point per district (GADM
centroid) instead of the polygon mean**, UTC days, same units and columns.
Every row is tagged `data_source=open-meteo:era5_seamless:centroid`.

Checks: persistence reproduces the paper exactly (16.42 / 20.16 / 24.58 / 28.47
at h=1–4 — case data and folds are untouched); the case arrays hash identically
before and after the climate rebuild; the rebuilt GRU lands at h=4 MAE 25.30 vs
the paper's 25.40. One test fails by design on this file:
`test_model_tensors.py::test_known_gaps_are_carried_as_missing` asserts the
original ERA5 extraction ended 2026-04-26 (periods 1010–1012 without weather);
the reconstruction covers them. Those weeks lie after every test year, so no
result depends on them (569 of 570 tests pass). **Neural numbers below are not bit-for-bit reproductions of
the paper's tables** and should be re-run on the original ERA5 file if a teammate
still has it (all scripts pick it up unchanged).

## Protocol

- Folds: the project's 9 walk-forward folds (test years 2017–2025) **plus fold 0
  (test 2016), used only for calibration**. Headline = folds 1, 2, 3, 6, 7, 8, 9.
- Horizons: 1, 2, 3, 4, 6, 8, 10, 12 weeks. Separate model per horizon, windows
  assigned to splits by their own target week.
- **Every method is scored on the identical 208,800 (fold, split, horizon, week,
  district) cells** (script 34 scores the intersection).
- Anything tuned or calibrated for fold *f* uses fold *f−1*'s **test** forecasts
  (a year no model trained on): conformal intervals, stacking weights, ensemble
  member selection, early-warning alarm cutoffs. Chronos fine-tuning steps were
  chosen once on fold 0's validation year (2015).
- Statistics as in the README: paired t over the 7 headline folds, plus Wilcoxon
  and fold win counts. 3 seeds for every trained model.

## Methods (17 forecasters + 4 naive + 3 ensembles)

| Family | Arms |
| --- | --- |
| Naive | persistence, seasonal naive, 5-year climatology, seasonal persistence |
| GRU (identity backbone) | `gru_v1` (the short paper's model), `gru_v2` (+outbreak history), `gru_v2_shuffled` (climate time-shuffled), `gru_v2_quantile` (7-quantile pinball head), `gru_v2_lw` / `gru_v2_quantile_lw` (level-weighted loss, README §8), `gcn_v2` (contiguity graph), `adaptive_v2` (learned graph, LR selected on validation) |
| Tabular | `lgbm_v2`, `lgbm_v2_noclimate`, `ridge_v2` (linear ARX) — case lags, climate lags to 24 weeks, neighbour and national signals, target-week season |
| Foundation (zero-shot) | `chronos_bolt`, `chronos2`, `chronos2_climate` (past climate + known season as covariates), `chronos2_joint` (all 25 districts as one multivariate series) |
| Foundation (fine-tuned) | `chronos2_ft`, `chronos2_joint_ft` — LoRA per fold on data up to train_end only |
| Ensembles | `ens_equal` (pre-specified GRU v2 + LightGBM + Chronos-2), `ens_stacked` (NNLS on prior year), `ens_top3` (3 best on prior year) |

## Results

### 1. Accuracy by horizon (headline MAE; skill vs same-horizon persistence)

| Method | h=1 | h=2 | h=3 | h=4 | h=6 | h=8 | h=10 | h=12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| persistence | 16.42 | 20.16 | 24.58 | 28.47 | 35.01 | 40.07 | 43.76 | 46.47 |
| `gru_v1` (short paper) | 16.80 | 20.41 | 23.02 | 25.95 | 30.19 | 33.99 | 36.18 | 37.77 |
| **`gru_v2_quantile_lw`** | **15.70** (+4.4%) | **19.00** | 22.38 | **24.58** (+13.7%) | 29.94 | 33.10 | 35.83 | 37.94 |
| `lgbm_v2` | 16.99 | 21.05 | 24.30 | 27.16 | 32.34 | 36.29 | 38.14 | 40.59 |
| `ridge_v2` | 17.96 | 21.39 | 24.24 | 26.70 | 30.96 | 34.22 | 36.52 | 38.23 |
| `chronos2` (zero-shot) | 16.03 | 19.34 | 22.86 | 25.89 | 30.46 | 33.98 | 36.37 | 37.64 |
| `chronos2_joint` (zero-shot) | 15.95 | 19.07 | 22.29 | 25.11 | 29.39 | 32.74 | 34.91 | 36.30 |
| `chronos2_climate` (zero-shot) | 16.71 | 19.56 | 22.57 | 24.93 | 29.22 | 32.38 | 34.49 | 35.86 |
| `chronos2_ft` | 18.59 | 21.07 | 23.59 | 25.63 | 29.26 | 31.74 | **33.58** | **34.76** (+25.2%) |
| `chronos2_joint_ft` | 17.32 | 19.80 | 22.38 | 25.07 | **28.82** | **31.70** | 33.86 | 35.17 |
| `ens_equal` | 16.05 | 19.27 | 22.30 | 25.01 | 29.79 | 33.19 | 35.35 | 36.82 |

Figure: `results/benchmark/figures/fig_lh_skill.png`.

1. **Something finally beats persistence at one week.** `gru_v2_quantile_lw`
   (quantile head + level weighting) reaches 15.70 vs 16.42 (+4.4%, paired
   p=0.045, 6/7 folds). Every model already beats persistence on all six
   non-epidemic folds by ~1 case/week; the difference is 2017, where plain
   `gru_v2` loses by +6.9 and this arm by only +0.8. **Caveat:** it is one of ~17
   arms, the Wilcoxon p is 0.11, and it does not survive a multiple-comparison
   correction — report it as promising, not established.
2. **From h=3 on, almost every learned model beats persistence significantly**,
   and the margin grows with horizon: +14% at h=4, +19–25% at h=12, winning 7/7
   folds (Wilcoxon p=0.016, the minimum possible with 7 folds) for the Chronos
   arms, the ensembles and the best GRUs.
3. **A pretrained foundation model, zero-shot, matches the trained GRU.**
   Chronos-2 never saw dengue or Sri Lanka, yet `chronos2_joint` is within
   ±0.6 MAE of the best GRU at every horizon and beats persistence significantly
   from h=2. Fine-tuning shifts it toward long horizons (best single model at
   h=10–12) at the cost of short ones (fine-tuned on pre-2016 data it
   under-predicts the 2017 surge: +15 MAE at h=1 on that fold).
4. **The tabular baselines are weaker than the GRU**: LightGBM never beats
   persistence at h≤2 and trails the GRU by 2–3 MAE at h≥4. This is the
   comparison the Sri Lanka GNN papers made without a persistence anchor.
5. **Ensembles help at h=2–4 but are not a free win.** The pre-specified
   equal-weight ensemble is the best point forecaster at h=3–4 among
   pre-specified methods; learned combination (NNLS stacking, top-3 selection)
   does not beat equal weights — the member ranking changes year to year
   (`tables/top_k_members.csv`), the classic forecast-combination puzzle.

### 2. How much is climate? Three model families agree it matters from ~2–4 weeks on

Error reduction attributable to climate (positive = climate helps):

| h | GRU: shuffled − full | LightGBM: removed − full | Chronos-2: without − with covariates |
| --- | --- | --- | --- |
| 1 | +0.01 | +0.40 | −0.68 |
| 2 | +1.03 | +0.64 | −0.22 |
| 3 | +1.97 | +0.46 | +0.29 |
| 4 | +1.91 | +0.63 | +0.96 |
| 6 | +1.14 | +0.66 | +1.24 |
| 8 | +1.50 | +0.47 | +1.60 |
| 10 | +2.29 | +0.84 | +1.88 |
| 12 | +2.46 | +0.52 | +1.78 |

Figure: `fig_lh_climate.png`. On peak weeks the GRU's shuffle cost is +1.6 to
+4.7 MAE, significant at h=2, 4, 8, 12. Climate helps the GRU on 6/7 folds at
h=2, 3, 12.

- **Replicates the paper's §8h at h=4 (+1.91 vs +1.55)** on a different climate
  extraction and **extends it**: the effect is present in three unrelated model
  families, including a pretrained model that was simply handed climate
  covariates.
- **Refuted hypothesis, stated plainly:** we expected climate's value to peak near
  the 5–10 week biological delay and fade after it. It does not fade by h=12.
  Likely reason: past climate at the origin also encodes the phase and strength
  of the monsoon season, which stays informative beyond the delay. Individual
  per-horizon t-tests are mostly p=0.06–0.2 (7 folds); the consistency across
  horizons and families is the evidence, not any single test.

### 3. Graphs at long horizons — still no

| h | contiguity graph cost | learned (adaptive) graph cost |
| --- | --- | --- |
| 1 | +1.20 | +0.49 |
| 2 | +3.78 | +5.58 |
| 4 | +2.29 | +1.61 |
| 8 | +0.25 | +1.61 |
| 12 | +0.68 | +1.27 |

The README §9 item 1 question is answered: **neither graph helps at any
horizon**. The contiguity penalty shrinks with horizon (from +3.8 to +0.3) but
never turns into a gain. Cross-district information *does* help in a different
form: Chronos-2's joint 25-district mode beats univariate at every horizon
(−0.1 to −1.5 MAE, n.s. individually but monotone). Spatial information is
useful as shared context, not as a fixed or learned message-passing graph.

### 4. Other GRU changes, by horizon (Δ MAE vs `gru_v2`)

- Outbreak-history features (v2 vs v1): help at h=2–4 (−0.6 to −0.9, p=0.017 at
  h=3), neutral/slightly worse at h≥6.
- Pinball (median) loss: −0.3 to −1.4 across horizons (n.s. individually; better
  at 7/8 horizons). Level weighting: similar, and it is what fixes 2017 at h=1.

### 5. Probabilistic forecasts

WIS (lower is better) and 95% coverage, headline folds:

| Method | WIS h=1 | h=4 | h=12 | cov95 h=4 | cov95 h=12 |
| --- | --- | --- | --- | --- | --- |
| persistence + conformal | 12.51 | 19.66 | 33.46 | 0.96 | 0.95 |
| `gru_v2_quantile_lw` native | **9.18** | **14.38** | 25.40 | 0.95 | 0.91 |
| `chronos2_joint` native | 9.08 | 14.74 | 22.74 | 0.95 | 0.93 |
| `chronos2_joint_ft` native | 9.95 | 14.89 | **22.28** | 0.96 | **0.95** |
| `lgbm_v2` + conformal | 10.47 | 18.39 | 39.57 | 0.93 | 0.85 |

Figure: `fig_lh_probabilistic.png`.

- Native quantile forecasts beat conformalised point forecasts at every horizon.
  WIS skill over persistence is ~27% at h=1 and h=4 and ~33% at h=12.
- **Chronos-2's native intervals are calibrated out of the box** (0.93–0.96
  coverage at every horizon). Recalibrating them on the prior year makes WIS
  slightly worse.
- Prior-year conformal intervals under-cover at long horizons for the trained
  point models (0.85–0.89 at h=10–12). That is year-to-year non-stationarity,
  not a bug: the persistence and Chronos intervals hold 0.93–0.95 on the same
  protocol.

### 6. Early warning: outbreak weeks are easy, outbreak onsets are not

- **Outbreak-week discrimination is strong.** Every learned model reaches
  ROC-AUC ≈0.91 at h=1 and ≈0.84–0.85 at h=4, above persistence (0.89, 0.81).
  The gap widens with horizon: 0.69–0.70 vs 0.60 at h=12.
- **Onset prediction is near chance for every method.** An onset is the first
  outbreak week after ≥4 quiet weeks. Onset AUC is 0.60–0.63 at h=1 and
  0.53–0.57 at h=12. Persistence falls below chance (0.44 at h=12). Climatology
  (0.61, flat) is as good as any model, so seasonality is what the models know
  about onsets.
- The rule "warn when forecast ≥ threshold" detects almost no onsets (0–7%),
  because point forecasts are medians and lag turning points. At a matched
  prior-year alarm cutoff, models detect ~20–25% of onsets versus 10–24% for
  persistence. Realised false-alarm rates run at ~0.2 against a 0.1 target,
  again because of non-stationarity.
- **A dedicated onset classifier** (`scripts/training/36`) asks "quiet now →
  outbreak in h weeks?" and helps only at h=1. Scored on those same quiet-origin
  rows, its onset AUC is 0.70 vs 0.67–0.69 for the forecasters. At h=4 it trails
  the GRU (0.66 vs 0.69). Climate still helps it: 0.655 vs 0.620 at h=4 and
  0.647 vs 0.608 at h=6. The ceiling stays around 0.65–0.70.

For the long paper this is an important, honest finding. The literature's high
"outbreak detection" numbers are mostly continuation of outbreaks already under
way. Predicting the week an outbreak starts, the thing an early-warning system
is for, remains largely unsolved with case and weather data alone.

## What the long paper can now claim

1. First configuration to beat persistence at h=1 on the headline mean
   (promising, not yet established).
2. Robust, significant skill from h=3 to h=12, growing with horizon, against
   persistence *and* strong ML baselines.
3. Climate's contribution, replicated across three model families and extended
   to 12 weeks.
4. A zero-shot foundation model rivals a purpose-built GRU. This is the first
   such comparison for dengue in Sri Lanka as far as we found.
5. Graphs do not help at any horizon; joint multivariate context does.
6. Calibrated probabilistic forecasts: 27–33% WIS improvement, with native
   quantiles well calibrated.
7. The onset gap: strong outbreak-week AUC but near-chance onset AUC. This is
   an evaluation lesson the field should adopt.

## Limits

- Reconstructed climate (see caveat); re-run on the original ERA5 file if available.
- 7 headline folds give low statistical power. The smallest possible Wilcoxon p
  is 0.016, and many individual tests sit at p=0.05–0.2. Many arms were
  compared, so the h=1 result in particular needs a correction or replication.
- Chronos fine-tuning was tuned on one validation year only.
- The early-warning threshold is the paper's peak threshold (90th percentile),
  not an operational Ministry of Health epidemic threshold.

## Reproduce

```powershell
.venv\Scripts\python.exe scripts\data\5c.fetch_open_meteo_climate.py   # ~2 h, quota-limited
# then scripts 9, 10, 12, 14 as usual, then:
.venv\Scripts\python.exe scripts\training\31.long_horizon_neural.py      # ~1.5 h GPU
.venv\Scripts\python.exe scripts\training\32.long_horizon_tabular.py     # ~20 min CPU
.venv\Scripts\python.exe scripts\training\33.long_horizon_foundation.py  # zero-shot ~5 min, fine-tune ~45 min/arm
.venv\Scripts\python.exe scripts\evaluation\34.long_horizon_analysis.py
.venv\Scripts\python.exe scripts\training\36.onset_classifier.py
.venv\Scripts\python.exe scripts\evaluation\35.long_horizon_figures.py
```

Run the GPU jobs one or two at a time: on the 8 GB laptop GPU, four concurrent
jobs spilled GPU memory into system RAM and slowed everything tenfold. Extra
packages: `lightgbm statsmodels scikit-learn chronos-forecasting peft pypdf`
(installed with torch pinned at 2.11.0+cu128).

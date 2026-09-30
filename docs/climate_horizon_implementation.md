# Climate delays learned from informative horizons: implementation record


> **Authoritative status (2026-09-30, after the correction).** This file
> is an engineering log in chronological order. Where later sections
> supersede earlier ones, the earlier text is kept and marked.
>
> - **Current routing (v2).** Weighted loss goes only to `encoder.*`
>   (lag kernels) and `climate_gru.*`. Every prediction head, including
>   `climate_delta`, receives the unweighted loss.
> - **Superseded routing (v1).** §9 and §13 describe v1, where
>   `climate_delta` was in the weighted group. That was an implementation
>   error.
> - **Current results:** [`climate_horizon_results.md`](climate_horizon_results.md).
> - **Audit trail:** [`climate_horizon_correction_audit.md`](climate_horizon_correction_audit.md).
>   The v1 results in §15–18 are history.

Status: **frozen protocol fully executed (§17 retrospective, §18 2026 hold-out). Primary: no measurable difference.** New outputs go only to `results/climate_horizon/`. Nothing has been trained for this
extension yet. Every number below comes from existing files on disk or from
the short paper. Each one says which.

Working title: *Learning Climate Delays from Informative Forecast Horizons for
Dengue Forecasting.*

Idea in one paragraph. The model has two branches: a case-history branch and
a climate branch built on the learnable per-district delay encoder. It
predicts h = 1, 2, 3, 4 from one forecast origin. Before training on a fold,
we measure how much climate improves each horizon, using validation blocks
that lie inside that fold's training period. Those measurements set how
strongly each horizon's loss updates the shared climate-delay encoder. The
case branch and the prediction heads still learn from every horizon at full
strength.

---

## 1. Instructions that apply

| Source | What it requires |
| --- | --- |
| `AGENTS.md` / `CLAUDE.md` | **None in the repository** (only one inside `.venv/…/xarray`, which is not ours). |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | Reusable code goes in `src/`. Executable entry points go in `scripts/`. Data stays in `data/`, results in `results/`, models in `models/`. Scripts stay thin. Tests are added or updated with every model or data change. Conventional-commit messages. Run `python -m pytest tests` before a PR. |
| [`docs/workflow.md`](workflow.md) | Numbered script prefixes record chronology. Scripts import from `src/`; they do not import each other's private helpers. |
| [`README.md`](../README.md) §6–§10 | Reporting conventions: headline = folds 1, 2, 3, 6, 7, 8, 9; folds 4 and 5 (COVID) reported separately; 3 seeds; the evidence bar is "≥ 2 seed-sd and p < 0.05 paired over folds"; always decompose by fold 1 (2017), because five earlier "gains" came entirely from that fold. |
| [`docs/revision_2026_09_30.md`](revision_2026_09_30.md) | The current benchmark protocol: common cells; per-cell split purge for shared multi-horizon models; Holm over a declared primary family; 4-week block bootstrap; seed-mean and seed-ensemble estimands kept separate; 2026 hold-out declared in advance. |
| [`results/benchmark/holdout_declaration.md`](../results/benchmark/holdout_declaration.md) | 2026 (fold 10) was frozen on 29 Sep 2026. New models may only be added in **a dated section at the end, written before they are scored on 2026**. |
| CS3631 project brief | GRU/LSTM/Transformer baseline required. No LLM, agentic or large-foundation-model methods in the proposed model. Must train on free or personal hardware. Paper needs ablations, a comparison with existing methods, hyperparameter tuning and a computational analysis. |

---

## 2. Environment and checks run in this step

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.11.0+cu128 True   (RTX 4070 Laptop GPU per results/benchmark/compute/*.csv)

.venv\Scripts\python.exe -m pytest tests/test_lag_encoder.py tests/test_negative_binomial.py `
  tests/test_multi_horizon.py tests/test_climate_ablation.py tests/test_folds.py `
  tests/test_long_horizon.py tests/test_revision.py tests/test_simplex_activations.py -q
# 166 passed, 4 warnings in 24.62s
```

The full suite was not run in this step. The revision notes record one known
failure caused by the data:
`tests/test_model_tensors.py::test_known_gaps_are_carried_as_missing`. It
expects the original GEE ERA5 file to end on 2026-04-26, and the Open-Meteo
reconstruction covers those weeks.

**Climate data in use.** `data/raw/climate_daily_district.csv` is the
Open-Meteo `era5_seamless` reconstruction, sampled at district centroids
(`scripts/data/5c.fetch_open_meteo_climate.py`; hashes are in
`results/benchmark/data_manifest.json`). It is not the GEE polygon-mean
ERA5-Land extraction behind the short paper. Case data and folds are
unchanged: persistence reproduces 16.42 / 20.16 / 24.58 / 28.47. Neural
numbers from the short paper **cannot be compared directly** with anything
trained now.

---

## 3. Existing components

### 3.1 Shared GRU backbone

| Item | Location | Notes |
| --- | --- | --- |
| `GCNGRU` | [`src/models/stgnn.py:11`](../src/models/stgnn.py#L11) | Per step: `relu(A·X·W)` graph convolution (2 layers), then one GRU shared across districts (`batch*nodes` sequences), then dropout, then a linear head on the last state. Defaults: hidden 32, 2 GCN layers, dropout 0.2. |
| `GraphConv` | [`src/models/gnn.py:9`](../src/models/gnn.py#L9) | Dense `einsum("ij,bljf->blif")`. |
| Identity backbone ("GRU-only") | adjacency = `np.eye(25)` | With the identity matrix the graph layers become per-node MLPs. Every positive result so far uses this backbone. The contiguity and learned graphs cost MAE at every horizon (README §8e; `results/benchmark/tables/ablations.csv`: contiguity +1.2 to +3.8, learned +0.5 to +5.6 at h = 1–4). **The new model uses no graph.** |
| `MultiHorizonGCNGRU` | [`src/models/multi_horizon.py:144`](../src/models/multi_horizon.py#L144) | One shared trunk with one `Linear(hidden, 1)` per horizon. The backbone's own head is replaced with `nn.Identity`. |
| `make_multi_horizon_windows` | [`src/models/multi_horizon.py:67`](../src/models/multi_horizon.py#L67) | Every horizon on the same origins. Split assigned by the **longest** horizon (script 27). The benchmark replaced this with the per-cell rule in §3.5. |
| Baseline training loop | [`scripts/training/16.train_gcn_gru.py:293`](../scripts/training/16.train_gcn_gru.py#L293) `train_one` | Adam (lr 3e-3, wd 1e-4), batch 64, max 150 epochs, patience 15 on validation loss, `clip_grad_norm_(…, 1.0)`, best-state restore, seeded `randperm`. Hooks: `build_model`, `make_optimiser`, `make_scheduler`. |
| `parameter_groups` | [`16.train_gcn_gru.py:252`](../scripts/training/16.train_gcn_gru.py#L252) | Detects `model.encoder` and gives its parameters their own learning rate (`kernel_learning_rate`) with no weight decay. |

### 3.2 Learnable climate-lag encoder (short paper §4.2)

| Item | Location | Notes |
| --- | --- | --- |
| `LearnableLagEncoder` | [`src/models/lag_encoder.py:65`](../src/models/lag_encoder.py#L65) | Per district i and climate feature k: kernel `w[i,k,τ] = Σ_b α[i,k,b]·basis_b(τ)` over reach `L = 26` periods, with 6 Gaussian bumps (initial centres 0, 4, 8, 13, 18, 24; width `softplus + 0.5`). `α = simplex(node_embedding[i] · feature_projection[k])`, embedding dimension 8. About 550 parameters. Rows sum to 1 and are non-negative. |
| Causality | `forward`, [`lag_encoder.py:165`](../src/models/lag_encoder.py#L165) | Input `[B, T, N, K]` gives `[B, T−L+1, N, K]` via `unfold` and a flipped kernel (τ = 0 is the most recent step). The origin step only sees τ ≥ 0. |
| Readouts | `kernels()`, `peak_lags()` (centre of mass) | These give the interpretable output ("district i responds to rainfall after about 8 weeks"). |
| Activation | `activation="softmax"` default; [`src/models/simplex_activations.py`](../src/models/simplex_activations.py) | `sparsemax` and bare `entmax15` collapse 58–74% of kernels into a zero-gradient state (README §8b). **Keep softmax** (control). `floored_entmax15` is the only sparse alternative that is safe to use. |
| `LagGCNGRU` | [`lag_encoder.py:185`](../src/models/lag_encoder.py#L185) | Replaces the climate channels with smoothed versions in place, inside **one** backbone. It has no separate branches, so it cannot route gradients per horizon. The new model reuses the encoder, not this wrapper. |
| Training script | [`scripts/training/18.train_lag_gcn_gru.py`](../scripts/training/18.train_lag_gcn_gru.py) | Window = 12 + 26 − 1 = **37** periods. Kernel LR 2e-2. `LAGGED_FEATURES` = the 7 instantaneous climate channels. Writes `learned_lags.csv` (per fold, seed, district, feature and lag). |
| Known result | README §7; short paper §4.2 | Trained end to end at h = 1 it loses: 20.67 MAE vs 18.36 with hand windows and 18.07 with none (GEE climate, single seed). Learned vs measured delay r = −0.16 (another configuration gives +0.36; README §8b). Diagnosis: at h = 1 the gradient carries almost no delay information. **The new method responds directly to this diagnosis.** |
| Measured delays | [`scripts/evaluation/17.lag_correlation_scan.py`](../scripts/evaluation/17.lag_correlation_scan.py) | Per-district cross-correlation, fitted before each test year. Rainfall delay 5–10 weeks, median 8. The script itself warns that the deseasonalised rainfall peak correlation (+0.036) is below its resolvability threshold of 0.10. **Its outputs (`results/eda/lag_correlation.csv`) are not on disk** and must be regenerated on the current climate file. |

### 3.3 Negative Binomial implementation (the newer head)

| Item | Location | Notes |
| --- | --- | --- |
| `negative_binomial_nll` | [`src/models/negative_binomial.py:26`](../src/models/negative_binomial.py#L26) | NB2 with `Var = μ + αμ²`, lgamma form, clamps `α ∈ [1e-6, 1e4]`. |
| `masked_negative_binomial_loss` | [`negative_binomial.py:99`](../src/models/negative_binomial.py#L99) | `Σ(nll·mask)/Σ mask`, normalised by the observed-cell count **pooled over all horizons**. A per-horizon split `L_h = Σ_{cells∈h} nll / N_total` therefore satisfies `Σ_h L_h` = the existing loss exactly. |
| `NegBinHead` | [`negative_binomial.py:130`](../src/models/negative_binomial.py#L130) | `log μ = log1p(y_origin) + Δ` with Δ clamped to [−10, 10]; `α = softplus(·) + 1e-4`. Both linear layers are zero-initialised, so at initialisation μ = y_origin + 1 (not persistence; documented and tested in `tests/test_revision.py`). One `Linear(hidden, H)` for μ and one for α. |
| `NegBinGCNGRU` | [`negative_binomial.py:205`](../src/models/negative_binomial.py#L205) | Backbone with `NegBinHead`; `forward(x, adjacency, anchor)`. |
| NB training | [`scripts/training/31.long_horizon_neural.py:303`](../scripts/training/31.long_horizon_neural.py#L303) `train_nb` | Same loop as `train_one` (Adam through `build_optimiser`, clip 1.0, early stopping on validation NB NLL). |
| Shared-trunk arrays | [`31.long_horizon_neural.py:385`](../scripts/training/31.long_horizon_neural.py#L385) `build_shared_arrays` | One origin, H targets. **Per-cell split purge**: a cell is unmasked only in the split its own target falls in, and an assertion enforces it. **Reuse this rule.** |
| Point forecast | `nb_quantiles` ([`src/evaluation/long_horizon.py:193`](../src/evaluation/long_horizon.py#L193)) | The NB **median** is the point forecast used for MAE; `<arm>_mean` reports μ. Seven quantiles (0.025 … 0.975) feed WIS. |
| Superseded | `scripts/training/31.train_negbin.py`, `32.train_multi_horizon_negbin.py`, `docs/negative_binomial.md` | Split leak (validation origins carried test-year targets) and different cells. **Do not reuse their numbers.** |

### 3.4 Features, scaling, masks and prediction conversion

Tensors: `data/processed/model_tensors_{v0..v4}.npz`, built by
[`scripts/features/12.build_model_tensors.py`](../scripts/features/12.build_model_tensors.py).
Keys are `X [1012, 25, F]`, `y`, `y_mask`, `weather_mask`, `period_id`,
`start_date` and `feature_names`. Values are not scaled and are NaN where
unobserved.

| Variant | F | Channels (verified from the files) |
| --- | --- | --- |
| v0 | 14 | `cases_log1p`, 7 climate (`rainfall_daily_mean_mm`, `rainy_days_frac`, `temperature_mean_c`, `diurnal_range_c`, `dewpoint_mean_c`, `relative_humidity_mean`, `wind_speed_mean`), `doy_sin`, `doy_cos`, `weather_observed`, `case_observed`, `centroid_lat`, `centroid_lon` |
| v1 | 23 | v0 + 9 hand-lagged means (`{rainfall,temperature,humidity}_roll{4,8,12}`) |
| v2 | 25 | v1 + `national_wave_rank`, `trailing_52_cumulative_cases` (outbreak history) |
| v3 | 28 | v2 + 3 neighbour-case channels |
| v4 | 35 | v3 + fixed weather lags and `thermal_suitability` |

- **Climate channels**: `climate_indices(feature_names)`
  ([`src/models/climate_ablation.py:105`](../src/models/climate_ablation.py#L105))
  returns the 7 instantaneous channels, any `_roll*` versions of them, and
  the v4-derived ones. Use `non_climate_indices` for the complement.
- **Scaling and imputation**: `fit_fold_statistics` / `transform`
  ([`scripts/features/14.build_folds.py:247`](../scripts/features/14.build_folds.py#L247), `:360`).
  Imputation uses the district × month climatology, then the district mean,
  then the global mean. Each feature is z-scored (per feature, not per node)
  on periods `≤ fit_end_period`. **`fit_end_period` = end of the validation
  year**, not the end of training. `tests/test_folds.py` checks for leakage
  by overwriting test years with 1e6.
- **Masks**: `y_mask` (target observed) is never imputed. It is excluded
  from the loss and from every metric. `weather_observed` and
  `case_observed` mark imputed inputs.
- **Target and conversion**: the anchor is `log1p(y_origin)`
  (`build_anchor`, [`16.train_gcn_gru.py:209`](../scripts/training/16.train_gcn_gru.py#L209);
  a missing origin cell falls back to the district's mean over its history).
  MSE models predict `log1p(y_{t+h}) − anchor` and convert back with
  `clip(expm1(pred + anchor), 0)` (`predict`, `:438`). NB models output μ and
  α directly on the count scale; the point forecast is the NB median.

### 3.5 Folds and evaluation

| Fold | Test year | Train ≤ | Val | Test | Headline |
| --- | --- | --- | --- | --- | --- |
| 0 | 2016 | — | — | — | calibration only (benchmark) |
| 1 | 2017 | 471 | 472–524 | 525–576 | yes (epidemic) |
| 2 | 2018 | 524 | 525–576 | 577–628 | yes |
| 3 | 2019 | 576 | 577–628 | 629–680 | yes |
| 4 | 2020 | 628 | 629–680 | 681–732 | COVID, separate |
| 5 | 2021 | 680 | 681–732 | 733–784 | COVID, separate |
| 6 | 2022 | 732 | 733–784 | 785–837 | yes |
| 7 | 2023 | 784 | 785–837 | 838–889 | yes |
| 8 | 2024 | 837 | 838–889 | 890–941 | yes |
| 9 | 2025 | 889 | 890–941 | 942–993 | yes |
| 10 | 2026 (19 wk) | 941 | 942–993 | 994–1012 | frozen hold-out |

(Verified from `data/processed/folds.json`. Folds 0 and 10 come from
`build_benchmark_folds`, [`src/evaluation/long_horizon.py:87`](../src/evaluation/long_horizon.py#L87).)

- **Split rule for multi-horizon models**: `split_by_target`
  ([`long_horizon.py:175`](../src/evaluation/long_horizon.py#L175)) assigns
  each (origin, horizon) cell to the split of its own target.
- **Metrics**: `evaluate`, `peak_thresholds`, `peak_mae`
  ([`scripts/evaluation/15.evaluate_naive_baselines.py:111-175`](../scripts/evaluation/15.evaluate_naive_baselines.py#L111)).
  These cover masked MAE and RMSE, and peak MAE on cells at or above each
  district's 90th percentile of pre-test history.
- **Benchmark scoring**: [`scripts/evaluation/34.long_horizon_analysis.py`](../scripts/evaluation/34.long_horizon_analysis.py).
  It reads **every** `results/benchmark/predictions/*.parquet` and scores the
  **intersection of cells across all methods**. It computes the naive
  baselines on those cells, per-fold metrics, paired t / Wilcoxon / win
  counts over the 7 headline folds, Holm over `PRIMARY`
  (`src/evaluation/revision.py:39`), the 4-week block bootstrap (`:56`), WIS
  and coverage, early warning, and compute.
  - ⚠ **Do not write the new arms into `results/benchmark/predictions/`.**
    They cover h = 1–4 only. The global intersection would then drop
    h = 6–12 for every method and overwrite the benchmark tables. Use a
    separate directory (§6).
- **Prediction frame format** (reuse it): `cell_frame`
  ([`long_horizon.py:135`](../src/evaluation/long_horizon.py#L135)). Columns
  are `method, fold_id, split, horizon, seed, target_period_id, node_id,
  prediction`, plus `q0.025…q0.975` and `nb_mu`, `nb_alpha`.

---

## 4. Existing results that can be reused or reproduced

### 4.1 On disk and valid on the current climate file (reusable as references)

`results/benchmark/tables/headline_summary.csv`, 24–30 Sep 2026. Open-Meteo
climate, common cells, 3 seeds, headline folds, mean of per-seed MAE:

| Method | h=1 | h=2 | h=3 | h=4 | Peak h=4 | Seed sd h=4 |
| --- | --- | --- | --- | --- | --- | --- |
| persistence | 16.42 | 20.16 | 24.58 | 28.47 | 47.98 | — |
| `gru_v1` (short-paper model) | 16.80 | 20.41 | 23.02 | 25.95 | 44.20 | 0.83 |
| `gru_v2` (MSE, separate) | 16.64 | 19.51 | 22.33 | 25.30 | 43.59 | 0.80 |
| `gru_v2_nb` (NB, separate) | 15.94 | 19.26 | 22.46 | 25.07 | 42.60 | 0.47 |
| `nb_shared_v2` (NB, one trunk, 8 horizons) | 16.63 | 19.83 | 23.15 | 25.84 | 43.82 | 1.36 |
| `nb_shared_v2_shuffled` | 17.43 | 21.16 | 24.74 | 27.68 | 47.40 | 0.61 |
| `nb_shared_v2_climatology` | 16.92 | 20.41 | 23.88 | 26.84 | 46.05 | 1.03 |
| `nb_shared_v2_noclimate` | 18.08 | 21.59 | 25.10 | 28.15 | 48.11 | 0.68 |
| `gru_v2_quantile_lw` | 15.70 | 19.00 | 22.38 | 24.58 | 42.64 | 0.71 |
| `lgbm_v2` | 16.99 | 21.05 | 24.30 | 27.16 | 46.42 | 0.18 |

Relevant paired ablations (`results/benchmark/tables/ablations.csv`, MAE,
7 folds). Positive Δ means the control is worse:

| Comparison | h=1 | h=2 | h=3 | h=4 |
| --- | --- | --- | --- | --- |
| NB shared: climate shuffled | +0.81 (p .26) | +1.33 (p .18) | +1.59 (p .16) | +1.84 (p .16) |
| GRU MSE: climate shuffled | +0.01 (p .98) | +1.03 (p .08) | +1.97 (p .06) | +1.91 (p .08) |
| NB: shared trunk vs separate | +0.69 (p .11) | +0.58 | +0.69 | +0.77 |
| Head: NB vs MSE (separate) | −0.70 | −0.26 | +0.14 | −0.23 |

What this means for the design:
1. Climate's contribution rises with horizon in both families, with
   near-zero at h = 1 for the MSE GRU. This matches the premise of the
   method.
2. Sharing a trunk costs about 0.6–0.8 MAE for NB. The proposed model is
   shared by design, so the fair comparison is with a shared baseline.
3. No single climate test clears p < 0.05 on 7 folds. The effects we are
   chasing are about 1–2 MAE against seed sd of 0.5–1.4.

Per-cell predictions for all of these are in
`results/benchmark/predictions/*.parquet`. **Reuse them as external
references on common cells. Do not retrain.**

### 4.2 Only in the short paper / README (GEE climate; not on disk)

Tables 1–5 of `docs/short paper.pdf` and README §6–§8h (identity separate
h = 4 skill +10.8%, p = 0.0068; shuffle cost +1.55 at h = 4, p = 0.016;
learnable-lag h = 1 failure). The metric CSVs in `results/models/` are no
longer on disk. Only `adjacency_report.md`, `folds_report.md` and
`model_tensors_report.md` remain. They can be **cited as prior results**,
but any head-to-head comparison with the new model must be re-run on the
current climate file.

---

## 5. Chosen baseline

**Main controlled baseline: `hcd_uniform`.** It is the proposed two-branch
model with **uniform horizon weights** (w_h = 1 for every h). Everything
else is identical: architecture, lag encoder, NB2 head, loss, optimiser,
window, seeds and stopping rule. The only thing that differs between the
baseline and the proposed arm is how strongly each horizon's gradient
reaches the climate encoder. With w = 1, the routing layer is exactly the
identity in both forward and backward passes (to be enforced by a test).

Reasons:

- The research question is about the training signal to the encoder, not
  about adding a branch. Any other baseline would change two things at once.
- **Head and loss fixed = NB2 head** (`NegBinHead` semantics, anchored
  `log μ = log1p(y_origin) + Δ`, `masked_negative_binomial_loss`), NB median
  as the point forecast:
  - It is the repository's current head, with native quantiles for WIS and
    coverage.
  - It is statistically indistinguishable from the MSE and pinball heads
    (`ablations.csv`: |Δ| ≤ 0.7 MAE, p ≥ 0.41).
  - It has the lowest seed sd among separate models at h = 4 (0.47).
  - The loss decomposes exactly per horizon (§3.3).
- **Shared trunk over h = 1–4**, identity (no graph), because the method
  requires one encoder shared across horizons.

**External references, scored on the same cells, not retrained:**
persistence, seasonal naive, `nb_shared_v2` (existing shared NB with hand
lags), `gru_v2_nb` (separate NB), `gru_v2` / `gru_v1` (MSE GRU,
short-paper lineage), `lgbm_v2`. Chronos-2 may appear only as a labelled
external comparator, never as part of the method (course §5).

The CS3631 "obvious baseline" requirement is met by the GRU lineage
(`gru_v1` → `gru_v2` → `nb_shared_v2`), which the short paper already
reported.

---

## 6. Implementation plan

### 6.1 Model: `src/models/climate_horizon.py` (new)

```
input window W = L_look + R − 1 = 12 + 26 − 1 = 37 periods, features from v2

case branch     channels = non_climate_indices(v2) minus nothing climate-derived
                 = cases_log1p, doy_sin, doy_cos, weather_observed, case_observed,
                   centroid_lat, centroid_lon, national_wave_rank,
                   trailing_52_cumulative_cases                       (9 ch)
                last 12 steps → Linear+ReLU (identity "GCN") → GRU(32) → h_case
                (hand-rolled roll4/8/12 climate columns are NOT fed anywhere:
                 the encoder is the only delayed-climate path, or the
                 comparison is confounded)

climate branch  the 7 instantaneous climate channels over all 37 steps
                → LearnableLagEncoder(N=25, K=7, reach 26, 6 bumps, softmax)
                → 12 delayed steps [B,12,N,7] → GRU(16) → h_clim

routing         for each horizon h: z_h = concat(h_case, GradScale(h_clim, w_h))
                GradScale = identity forward, grad × w_h backward
heads           per-horizon NB head on z_h: Δ_h, α_h;  μ_h = exp(log1p(y_o)+Δ_h)
                (zero-initialised, same clamps as NegBinHead)
```

- **Why this is exact.** Head h is the only path from `L_h` to the climate
  parameters, so scaling the gradient on head h's copy of `h_clim` scales
  `∂L_h/∂θ_clim` by `w_h` and leaves nothing else changed. The case branch
  and heads receive `∂(Σ_h L_h)` unscaled. One backward pass is enough.
  Tests compare this against a two-pass `torch.autograd.grad` reference.
- The gated parameter set (`climate_parameters()`) is the encoder plus the
  climate GRU. Ablation: gate only the encoder's kernel parameters.
- Keep `self.encoder` as the attribute name, so script 16's
  `parameter_groups` gives it `kernel_learning_rate` (2e-2, no weight
  decay), as in script 18.
- Reuse `LearnableLagEncoder`, `negative_binomial_nll`, `climate_indices`
  and `non_climate_indices` without modification.

### 6.2 Horizon-weight estimation: `src/training/climate_horizon.py` (new)

**Declared 2026-09-30 15:18 +0530, before any pilot has been run.** Changes
to this rule after pilots run must be dated and reported as deviations.

Per outer fold f, using **only periods ≤ `train_end_period`**:

1. **Inner blocks.** K = 3 consecutive 52-week inner validation blocks
   ending at `train_end_period`. Block k gets a synthetic fold dict:
   pilot train = targets before the year that precedes block k,
   pilot early-stopping year = that preceding year, evaluation = block k,
   `fit_end_period` = end of the early-stopping year. The existing helpers
   (`fit_fold_statistics`, `split_by_target`, the shared-array builder)
   work unchanged, and the scaler never sees block k.
2. **Primary utility: inner-validation shuffle gain from a frozen
   equal-weight pilot.** Train one `hcd_uniform` pilot (w = [1,1,1,1]) per
   block with real climate, using 2 seeds. Freeze it. Score block k twice:
   with real climate inputs, and with the climate channels time-shuffled
   within district (`shuffle_climate`, R = 5 permutations, averaged). There
   is no retraining, so the gain is how much the pilot's forecasts at each
   horizon depend on correctly aligned weather:
   `g_h = mean_{k,seed,perm} (MAE_shuf − MAE_real) / MAE_shuf` on NB-median
   forecasts.
3. **Sensitivity utility: relative improvement over a case-only pilot.**
   Also train a case-only pilot (no climate branch) on the same block and
   seeds: `u_h = mean_{k,seed} (MAE_case − MAE_real) / MAE_case`. This
   drives the sensitivity arm `hcd_informed_caseonly` only.
4. **Weight rule (pre-registered, the same for either utility).**
   `w̃_h = max(g_h, 0)`. If every `g_h ≤ 0`, use w = [1,1,1,1]. Otherwise
   `w_h = 0.95 · H · w̃_h / Σ_h w̃_h + 0.05`. The mean weight is exactly 1.
   *(Qualified 2026-09-30: a mean of one does **not** guarantee equal
   gradient norms. The measured weights raised the raw encoder gradient
   norm by 14.7%; see correction audit §11.)* Every
   horizon keeps at least 0.05.
5. **No reuse of benchmark numbers.** The weights 0.77/1.05/1.07/1.11
   computed from `results/benchmark` shuffle costs in step 1 used test years.
   They are illustration only and are never passed to any model.
6. Save `results/climate_horizon/weights.csv` (fold, utility, h, value per
   block, seed and permutation, final w_h, fallback flag).

Leakage guarantees to test: no inner period ≥ `val_start_period` of the
outer fold; the inner scaler is fitted only up to the pilot's early-stopping
year; the shuffle never changes `y`, the mask or the anchor.

Pilot cost: 10 folds × 3 blocks × 2 seeds × (1 climate pilot + 1 case-only
pilot) = 120 small fits, plus inference-only shuffles. At about 1.2–2.5 s
per fit (compute.csv) this takes a few minutes on GPU.

### 6.3 Training loop

`train_two_branch(arrays, weights, config, seed, device)`, placed in
`src/training/climate_horizon.py`. It follows `train_nb` line for line
(Adam, clip 1.0, patience 15, best state, seeded `randperm`). Early stopping
uses the **unweighted** validation NB NLL for both arms, so stopping cannot
favour either one.

Arrays: a `src/` port of `build_shared_arrays`, with the same per-cell purge
and assertion, parameterised by window length, horizons (1, 2, 3, 4) and
the variant's channel split. A test checks that it reproduces script 31's
masks on one fold at `lookback=12`. Script 31 stays untouched.

Per epoch, log the per-horizon gradient norm reaching the encoder
(`gradients.csv`). This is the mechanism evidence: the h = 1 gradient
should be small and noisy.

### 6.4 Runner: `scripts/training/38.train_climate_horizon.py` (new, thin)

Arms (h = 1–4, identity, v2 channel split, window 37, folds 0–9; fold 10
only after a dated hold-out addendum):

| Arm | Encoder weights w_h | Question it answers |
| --- | --- | --- |
| `hcd_uniform` | 1, 1, 1, 1 | **baseline** |
| `hcd_informed` | measured, primary utility (§6.2) | **proposed** |
| `hcd_informed_caseonly` | measured, case-only sensitivity utility | Is the result robust to the utility definition? |
| `hcd_permuted` | measured, reversed across h | Does the measured *order* matter, or only the inequality? |
| `hcd_ramp` | ∝ h, mean 1 | Does measurement beat the simple prior "further = more climate"? |
| `hcd_mask_h1` | 0, 4/3, 4/3, 4/3 | Is it enough to exclude h = 1? |
| `hcd_frozen` | encoder not trained | Do learned delays matter at all? |
| `hcd_informed_shuffled` | measured, climate shuffled | Content control: does the gain need real weather? |
| `hcd_case_only` | no climate branch | Floor |
| `hcd_measured_init` | measured, kernels initialised at `scripts/17` lags | Does a measured prior plus informed training recover delays better? (optional) |

Seeds: 5 for `hcd_uniform` and `hcd_informed` (primary pair; effects are
about 1 MAE against seed sd of about 0.5–1.4), 3 for the others.

Outputs, all under `results/climate_horizon/` (never `results/benchmark/`):
- `predictions/<arm>.parquet` in `cell_frame` format with NB quantiles and
  μ/α
- `weights.csv`, `kernels.csv` (script 18's `kernel_frame` layout),
  `gradients.csv`
- `compute.csv` (parameters, seconds per fit including pilots, inference
  ms per origin, peak GPU MB)

### 6.5 Analysis: `scripts/evaluation/39.analyse_climate_horizon.py` (new)

- Load the new arms, plus reference arms from
  `results/benchmark/predictions/` filtered to h ≤ 4 and folds 0–9. Build
  the intersection of cells, then compute naive baselines on those cells.
  Reuse `evaluate`, `peak_thresholds`, `revision.holm`,
  `revision.block_bootstrap` and `weighted_interval_score`.
- **Primary family (declared before running):** `hcd_informed` vs
  `hcd_uniform`, headline MAE, h = 1–4. Paired t and Wilcoxon over the
  7 folds, with Holm across the 4 horizons. Expected direction: better at
  h = 3–4. **h = 1 non-inferiority margin: +0.3 MAE** (about one seed sd).
- Secondary: peak MAE, WIS, 50/95% coverage, skill vs same-horizon
  persistence, fold-1 decomposition (ex-2017 Δ), both estimands (seed-mean,
  seed-ensemble), COVID folds reported separately.
- Mechanism: learned peak lag vs `scripts/17` measured lag (Spearman,
  per fold and per seed); kernel stability (sd of peak lag across seeds);
  dead-kernel fraction; per-horizon encoder gradient norms; the fold-by-fold
  horizon-weight table.
- Report: `results/climate_horizon/report.md`.

### 6.6 Tests: `tests/test_climate_horizon.py` (new)

1. The encoder output at step t uses no input after t (perturb future
   steps; output is unchanged).
2. The channel split has no overlap, and every climate-derived channel is
   excluded from the case branch.
3. With w = (1, 1, 1, 1) the gradients equal those of the plain summed loss
   (bit-close).
4. With w_h = 0, `L_h` sends exactly zero gradient to the climate
   parameters and unchanged gradient to the case branch and heads.
5. The single-pass GradScale gradients match a two-pass
   `autograd.grad` reference for random w.
6. The weight rule: mean 1, non-negative, falls back to uniform when every
   gain ≤ 0.
7. Inner blocks never touch outer val or test periods; the inner scaler is
   fitted before the block.
8. `Σ_h L_h` equals `masked_negative_binomial_loss` on the same tensors.
9. The ported array builder matches script 31's per-cell purge.

### 6.7 Order of the following steps

1. ~~Inspect and document~~ (this file).
2. Regenerate `scripts/17` delays on the current climate file.
3. Model and routing, with tests 1–5 and 8.
4. Arrays, training loop, weight estimation, with tests 6, 7 and 9. Smoke
   run: fold 8, 1 seed.
5. Full pilots, then the main pair, then ablations.
6. Analysis script and report. Add a dated hold-out addendum **before**
   scoring 2026.
7. Hyperparameter sensitivity: K ∈ {2, 3, 4}, weight rule
   (clip-normalise vs softmax τ), climate GRU width, lag reach. Select on
   **validation only**.

Commands will be recorded here once they exist. None have been run for this
extension yet.

---

## 7. Risks and open points

- **The effect may be null.** Benchmark shuffle costs (test years;
  illustration only, never used as weights) suggested that the gains might
  be fairly flat. *(Outcome: the measured weights were **not** flat,
  max/min 1.7–4.0, and the result was null anyway; see the results
  document §5.)* That is a
  legitimate result. Report it with the mechanism diagnostics.
- **Early folds.** Fold 1's inner blocks are 2013–2015 with training from
  2007. Probe gains may be ≤ 0, in which case the model falls back to
  uniform. Log how often this happens.
- **Window 37 vs 12.** Only early training cells differ. Test cells are
  identical to the benchmark's, so common-cell scoring stays valid.
- **Weight normalisation choice.** Mean-1 normalisation is required to
  separate "which horizon" from "how much learning rate". The `permuted`
  and `ramp` arms test this.
- **Delay ground truth is weak.** The measured rainfall correlation is below
  script 17's own resolvability threshold. Kernel-recovery results are
  supporting evidence, not proof.
- **2026 hold-out.** Addendum written 2026-09-30 in
  `results/benchmark/holdout_declaration.md` (§ "Addendum 2026-09-30").
- **Climate data.** Everything must be trained on the manifest-recorded
  Open-Meteo file. The paper must describe it as such, not as the GEE
  extraction.

---

## 8. Step 3 record: data interface and evaluator (2026-09-30)

### Files added
- `src/data/climate_horizon.py`:
  - `split_channels` separates case, climate and excluded channels.
  - `inner_blocks` builds the pilot-train / early-stop / utility-score
    blocks.
  - `build_arrays` does per-cell split purging with shared eligible
    origins, NaN-free inputs and masked targets.
  - `build_anchor`, `extension_fold`, `cell_keys`.
  - The module docstring states the data-availability assumptions.
- `src/models/climate_horizon.py`: `nb_loss_by_horizon` (NaN-safe; the sum
  over horizons equals the pooled NB loss), `anchored_mean`,
  `nb_point_forecast` (NB median).
- `src/evaluation/climate_horizon.py`: read-only reference loading,
  provenance checks, persistence on arbitrary cells, scoring.
- `scripts/evaluation/39.evaluate_climate_horizon.py`: the evaluator.
- `scripts/evaluation/40.climate_horizon_delay_scan.py`: the scripts/17
  scan with history cut at `train_end_period`.
- `tests/test_climate_horizon_data.py`: 16 tests covering leakage (split per
  target date, scaler poisoning), masks, district order, window alignment,
  the NaN-safe loss, NB anchoring, median conversion, evaluator provenance,
  and that the benchmark parquet is unchanged by reading.

### Decisions
- **Channels (v2).**
  - Case branch (9): `cases_log1p`, `doy_sin`, `doy_cos`,
    `weather_observed`, `case_observed`, `centroid_lat`, `centroid_lon`,
    `national_wave_rank`, `trailing_52_cumulative_cases`.
  - Climate branch (7 instantaneous channels).
  - Excluded (9 `_roll*` columns).
  - District order is the tensor's `node_id` order throughout.
- **Availability assumptions.** Cases and reconstructed weather for week t
  are known at origin t. Weather may in practice be up to about one week
  late; the benchmark's one-week-late control moved MAE by ≤ 0.27.
  Reporting back-fill is not modelled.
- **Eligible origins.** Every origin with 37 weeks of history (period ≥ 37),
  identical for every fold and arm. Targets after period 993 (2026) are
  dropped in development.
- **Preprocessing.** The extension fits imputation and scaling on the
  **training portion only** (`extension_fold`: `fit_end_period` =
  `train_end_period`; pilots use the pilot-training end). This is stricter
  than the benchmark, which fits up to the end of the validation year.
  Reference arms keep the benchmark's own preprocessing; this is a
  protocol difference, stated in the report.
- **Inner blocks** (`results/climate_horizon/evaluation/inner_blocks.csv`).
  The utility-scoring year is always different from the early-stopping
  year. For example, fold 1 scores 2013/2014/2015, stops early on
  2012/2013/2014, and trains up to 2011/2012/2013.

### Commands and results

```powershell
.venv\Scripts\python.exe scripts\evaluation\39.evaluate_climate_horizon.py
.venv\Scripts\python.exe scripts\evaluation\40.climate_horizon_delay_scan.py
.venv\Scripts\python.exe -m pytest tests/test_climate_horizon_data.py tests/test_long_horizon.py tests/test_negative_binomial.py tests/test_folds.py -q
# 52 passed (then 16/16 in test_climate_horizon_data.py after adding extension_fold)
```

- **Data provenance:** climate, tensors_v2, folds, canonical dengue and the
  calendar all hash-match `data_manifest.json`.
- **Persistence, original protocol** (scripts/15, lookback 12, folds 1–9
  headline): 16.4249 / 20.1579 / 24.5768 / 28.4686 → **matches
  16.42/20.16/24.58/28.47** at 2 dp. (9,125 cells per horizon.)
- **References.** `gru_v2_nb`, `nb_shared_v2` and `lgbm_v2` pass every
  check: written after the manifest, split label = `split_by_target` on
  every row, targets inside the test year and observed, h = 1–4, seeds
  0/1/2. Status: **direct comparison**. Fold 10 rows are ignored.
- **Coverage.** Extension-eligible test cells = 11,725 per horizon
  (folds 1–9). Every reference covers all of them, so the common
  intersection is 100%. Persistence recomputed on this intersection gives
  the same headline 16.42/20.16/24.58/28.47. References on it:
  `gru_v2_nb` 15.94/19.26/22.46/25.07, `nb_shared_v2` 16.63/19.83/23.15/25.84,
  `lgbm_v2` 16.99/21.05/24.30/27.16 (identical to the benchmark table).
- **Delay diagnostics (descriptive only)**, fold 9, training history to
  2023 (`results/climate_horizon/diagnostics/`). Deseasonalised peak lag
  medians:
  - rainfall 9 weeks (range 5–10, median r = +0.041);
  - rainy-day fraction 8 (r = +0.089);
  - humidity 7 (r = +0.072);
  - dew point 5 (r = +0.164);
  - temperature 19 (r = +0.181).

  Rainfall, rainy days, humidity, wind and diurnal range fall at or below
  the scan's 0.10 resolvability threshold (see `alerts.txt`). These are not
  biological delays.
- No file under `results/benchmark/` was written.

---

## 9. Step 4 record: baseline model `hcd_uniform` (2026-09-30)

This supersedes §6.1's concatenation design. For each horizon h, the
corrections combine in NB log-mean space:

    delta_total_h = delta_case_h + sigmoid(gate_logit_h) * delta_climate_h
    mu_h  = exp(log1p(y_origin) + clamp(delta_total_h, ±10))
    alpha_h = softplus(case head) + 1e-4

- **Anchor.** mu is a count mean; the point forecast is the NB median; no
  expm1 is applied to mu.
- **Case branch.** `GCNGRU` (2 per-node layers, GRU 32) with a fixed
  identity adjacency, so there is no spatial message passing.
- **Climate branch.** `LearnableLagEncoder` (reach 26, 6 bumps, softmax)
  → GRU(16) → `climate_delta` Linear(16, 4).
- **Gate.** `gate_state(case state)` + per-district bias + per-horizon
  bias. It is not constrained to increase with horizon.
- **Dispersion.** Case-based in every arm.
- **Initialisation.** All heads start at zero, so μ = y_origin + 1 at
  initialisation and the gate starts at 0.5.
- **Router.** `_GradScale` on `delta_climate` (backward only), weights
  buffer `climate_weights`. Only [1,1,1,1] is used so far.
- **Parameter groups.** *(Superseded: routing v1, an implementation
  error. v2 excludes `climate_delta`; see §19.)*
  - `climate_parameters()`: the encoder, the climate GRU and
    `climate_delta`.
  - `other_parameters()`: the case trunk, case heads (delta and
    dispersion) and all gate parameters.
  - Disjoint and complete (tested).
  - Optimiser: the encoder uses lr 2e-2 with no weight decay; everything
    else uses lr 3e-3 with weight decay 1e-4.
- **Parameters.** 10,028 in total (`nb_shared_v2`: 8,752).

### Files
- `src/models/climate_horizon.py`: `ClimateHorizonNB`, `_GradScale`.
- `src/training/climate_horizon.py`: `train` (the `train_nb` protocol, with
  early stopping on the unweighted validation NLL), `predict`,
  `prediction_frame`.
- `scripts/training/38.train_climate_horizon.py`: `--smoke` only.
- `tests/test_climate_horizon_model.py`: 9 tests covering shapes and finite
  outputs (also for extreme inputs), anchor at initialisation, disjoint and
  complete groups, that `gate_off` removes all climate influence (on μ, α
  and gradients), case-based dispersion, uniform routing equal to the plain
  loss, the router against a two-pass `autograd.grad` reference, and weight
  validation.

### Commands and results

```powershell
.venv\Scripts\python.exe -m pytest tests/test_climate_horizon_model.py tests/test_climate_horizon_data.py tests/test_lag_encoder.py tests/test_negative_binomial.py -q
# 43 passed
.venv\Scripts\python.exe scripts\training\38.train_climate_horizon.py --smoke --fold 8 --seed 0 --epochs 3
```

Smoke check (fold 8, seed 0, 3 epochs, CUDA; **not a result**;
`results/climate_horizon/smoke/`):
- 800 training origins, 5,200 test cells, 1.43 s.
- All outputs finite.
- Mean gate by horizon: 0.496 / 0.520 / 0.555 / 0.561.
- Mean |delta_climate|: 0.188 / 0.252 / 0.293 / 0.282.
- Setting the gate to zero changes μ.
- Test MAE after 3 epochs: 11.87 (untrained; not comparable).

---

## 10. Step 5 record: origin- and target-relative lag modes (2026-09-30)

Notation: origin t, horizon h, climate observed at t − k (k = 0 is the
origin week). Its delay relative to the target week t + h is **d = h + k**.

| Mode | Kernel grid | Horizon h reads | Use |
| --- | --- | --- | --- |
| `target` (default) | d = 0 … `max_delay` (30 points) | x[t + h − d] = x[t − k], only for d ≥ h and k ≤ `history` − 1. Weight renormalised over that support. | **Main model: both `hcd_uniform` and `hcd_informed`.** |
| `origin` | k = 0 … `history` − 1 (26 points), shared by all h | x[t − k], so the implied delay d = h + k shifts with h | Ablation `hcd_origin_lags` |
| `target`, `common_support=True` | as target, restricted to d ∈ [max h, min h + history − 1] = [4, 26] for every h | the same delay support for all horizons | Sensitivity check |

- **Explicit sizes.** `history` = 26 (the observed weeks each kernel may
  read; the input window is 12 + 26 − 1 = 37). `max_delay` defaults to
  `history − 1 + max(h)` = 29, so every horizon can use its whole history.
- **Available support.** Both modes read the same observations (k = 0–25).
  In target mode the kernel's mass on d < h (and on d > h + 25) is
  unusable for horizon h. `available_mass()` [H, N, K] reports the mass on
  usable support before renormalisation.
- **Empty support** gives zero weights, zero output and mass 0, with a safe
  divide and no NaN.
- **Shared parameterisation.** Both modes reuse `LearnableLagEncoder`
  unchanged as the kernel generator (6 Gaussian bumps, learnable centres and
  widths, softmax mixture of a district embedding times a per-feature
  projection). The climate GRU and head run one stream per horizon in both
  modes. **Parameter counts are identical: 10,028** (tested, including
  names and shapes).
- `peak_delays()` [H, N, K] reports the centre of mass as a target-relative
  delay d, comparable across modes and with `diagnostics/`.

### Files
- `src/models/horizon_lag_encoder.py` (new): `delay_weights`,
  `HorizonLagEncoder`.
- `src/models/climate_horizon.py`: the encoder is now
  `HorizonLagEncoder(mode=lag_mode)`, with per-horizon climate streams
  (the diagonal of the shared head), `available_mass()`, and constructor
  arguments `lag_mode`, `common_support`, `max_delay`.
- `src/training/climate_horizon.py`: config keys `lag_mode` (default
  `"target"`), `common_support`, `max_delay`.
- `scripts/training/38.train_climate_horizon.py`: `--lag-mode`,
  `--common-support`. Smoke outputs are tagged by mode.
- `tests/test_horizon_lag_encoder.py` (new, 13 tests):
  - exact selection, d = 8 → h = 1 reads x[t−7], h = 2 reads x[t−6],
    h = 4 reads x[t−4];
  - origin mode reads x[t−8] for all h;
  - d < h unavailable (d = 2 for h = 3, 4);
  - boundaries: d = 29 reachable only for h = 4 (x[t−25]); d = 0 never;
  - explicit `max_delay` truncation and renormalisation;
  - common support identical across h;
  - empty support has no NaN;
  - data after t cannot change a forecast made at t (encoder and full
    model, both modes);
  - identical parameters across modes; target mode is the default.

### Commands and results

```powershell
.venv\Scripts\python.exe -m pytest tests/test_horizon_lag_encoder.py tests/test_climate_horizon_model.py tests/test_climate_horizon_data.py tests/test_lag_encoder.py -q
# 47 passed (after one fix: common support first lacked the upper bound min h + history − 1)
.venv\Scripts\python.exe scripts\training\38.train_climate_horizon.py --smoke --lag-mode target
.venv\Scripts\python.exe scripts\training\38.train_climate_horizon.py --smoke --lag-mode origin
.venv\Scripts\python.exe scripts\training\38.train_climate_horizon.py --smoke --lag-mode target --common-support
```

Smoke (fold 8, seed 0, 3 epochs; **wiring check, not results**): all runs
finite, 10,028 parameters, about 1–1.3 s, and turning the gate off changes
μ.

| Mode | Available mass, h = 1–4 | Rainfall peak delay d, h = 1–4 |
| --- | --- | --- |
| target | 0.92 / 0.84 / 0.76 / 0.67 | 5.2 / 5.9 / 6.7 / 7.5 |
| origin | 1 / 1 / 1 / 1 | 4.9 / 5.9 / 6.9 / 7.9 (= k + h by construction) |
| target, common support | 0.68 (all h) | 7.2 (all h) |

After 3 epochs the kernels are near initialisation, so these values
reflect the initial bump layout, not learned delays. The step-4 files
`smoke/hcd_uniform_smoke.parquet` and `smoke_summary.json` predate the
lag modes (single origin-relative stream) and are kept only as a record.

No full training sweep was started.

---

## 11. Step 6 record: small utility pilots (2026-09-30)

### Protocol as implemented (`src/training/climate_horizon_pilots.py`)
- **Blocks.** Outer fold 9 (training up to the end of 2023), inner blocks
  1 and 2.

  | Block | Pilot training | Early stopping | Utility scoring |
  | --- | --- | --- | --- |
  | 1 | ≤ 2020 | 2021 | 2022 |
  | 2 | ≤ 2021 | 2022 | 2023 |

  Preprocessing is fitted on pilot training only. An assertion checks that
  no 2026 target reaches a pilot.
- **Primary (inference perturbation).**
  - Train the equal-weight two-branch NB model (target-relative lags,
    weights [1,1,1,1], full `DEFAULTS` budget) on real Open-Meteo climate,
    with checkpoint selection on the early-stopping year.
  - Freeze it and score the utility year once with real climate and once
    under each of 5 declared permutation seeds (11, 23, 37, 41, 53) with
    donor climate.
  - No shuffled model is trained.
- **Donors.** For each (recipient origin, district), the whole 37-week ×
  7-variable window is replaced by the same district's window at a donor
  origin that:
  - is ≤ the pilot-training cut-off (hence before the recipient's origin);
  - has a full window;
  - starts within ±2 weeks of the recipient's week of year (circular);
  - lies in a different calendar year.

  The window's internal order and cross-variable structure are preserved.
  Cases, targets, masks and anchors are unchanged. The model uses no
  climate-derived features, so there is nothing to recompute. Mappings are
  saved in `donors_fold9_block{1,2}.parquet`.
- **Sensitivity.** `CaseOnlyNB`, size-matched (hidden 36, 9,980
  parameters vs 10,028), same data, budget and seeds, scored on the same
  cells.
- **Matched subset.** Cells whose (origin, district) had donors under all
  5 permutations. Here that is 100%: 1,400/1,400 and 1,375/1,375
  origin-district pairs, i.e. 1,325 and 1,300 scored cells per horizon.
- **Saved per unit** (`pilots/fold9_block{k}_seed{s}.{parquet,json}`):
  - predictions, NB μ/α, targets, masks, origin and target dates,
    districts, horizons, seed, permutation ID and the matched flag;
  - hashes of the data manifest, block, training arrays, utility cells and
    config;
  - best epochs, parameter counts and runtimes.

  Units are skipped if already present (resumable).

### Commands

```powershell
.venv\Scripts\python.exe -m pytest tests/test_climate_horizon_pilots.py tests/test_climate_horizon_model.py tests/test_horizon_lag_encoder.py tests/test_climate_horizon_data.py -q
# 41 passed
.venv\Scripts\python.exe scripts\training\41.run_climate_horizon_pilots.py --fold 9 --blocks 1 2 --seeds 0 1
# rerun of the same command: 5 s, all units skipped
```

Runtime per unit (CUDA): 8.5–27.1 s. The climate pilot takes 6.3–18.2 s
(best epoch 29–55); the case-only pilot takes 1.5–9.8 s.

### Results (`pilots/utility_summary.csv`; MAE on the matched utility cells)

Primary shuffle gain g = (MAE_shuffled − MAE_real) / MAE_shuffled, where
MAE_shuffled is the mean over 5 permutations:

| Block (year) | Seed | h=1 | h=2 | h=3 | h=4 |
| --- | --- | --- | --- | --- | --- |
| 1 (2022) | 0 | 0.018 | 0.024 | 0.039 | 0.039 |
| 1 (2022) | 1 | 0.011 | 0.013 | 0.017 | 0.029 |
| 2 (2023) | 0 | 0.050 | 0.072 | 0.077 | 0.085 |
| 2 (2023) | 1 | 0.033 | 0.049 | 0.068 | 0.083 |
| **mean** | | **0.028** | **0.039** | **0.050** | **0.059** |

The permutation sd of MAE_shuffled is 0.08–0.52. All 16 gains are
positive and rise with h in every block and seed.

Sensitivity gain over case-only u = (MAE_case − MAE_real) / MAE_case:

| Block | Seed | h=1 | h=2 | h=3 | h=4 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | −0.003 | 0.005 | −0.003 | −0.006 |
| 1 | 1 | −0.010 | −0.023 | −0.043 | −0.058 |
| 2 | 0 | 0.013 | −0.006 | −0.011 | −0.021 |
| 2 | 1 | 0.056 | 0.065 | 0.057 | 0.076 |
| **mean** | | 0.014 | 0.010 | −0.000 | −0.002 |

### Reading (pilot only; not weights and not a result)
- The frozen model depends more on correctly aligned weather as the
  horizon grows. This is consistent with the primary rule producing
  weights that increase with horizon.
- The case-only comparison is dominated by seed noise (its sign flips
  between seeds). It has not established that the climate branch beats an
  equally sized case-only model.
- Two blocks of one fold are far too few to set weights. The full per-fold
  estimation (K = 3 blocks, folds 1–9) is **[PENDING]**. No weights were
  computed or applied in this step.

---

## 12. Step 7 record: fold-specific weights (2026-09-30)

### Rule as implemented (`src/training/climate_horizon_weights.py`)
- **Primary.** g_h = MAE_shuffled_h − MAE_real_h (MAE_shuffled is the mean
  over the 5 declared permutations), u_h = max(g_h, 0). If Σu > 0, then
  q_h = 4·u_h / Σu and w_h = (1 − ρ) + ρ·q_h with ρ = 0.95; otherwise
  w = [1,1,1,1].
- **Sensitivity (separate arm `hcd_informed_caseonly`).**
  u_case_h = max(0, (MAE_case_h − MAE_real_h) / (MAE_case_h + 1e-8)),
  then the same normalisation and mixing.
- **Fixed aggregation.** Per unit and horizon, MAE is taken over the
  matched observed cells. MAEs are then averaged over seeds within a block,
  then over blocks with equal weight. The utility is formed from the
  aggregated MAEs. Per-unit gains are reported only as a measure of
  variability.
- **Missing evidence.** A fold without pilots gets no weight file. The
  weighted arm must not be trained for it.
- **Frozen.** Script 42 never overwrites an existing weight file.
- **Deviation from §6.2 (dated 2026-09-30, before any weighted model was
  trained or any test year was scored).** §6.2 declared a *relative*
  shuffle gain, (MAE_shuf − MAE_real) / MAE_shuf. This step, as
  specified, uses the *absolute* difference. The mixing rule (ρ = 0.95,
  5% uniform, mean one, uniform fallback) is unchanged. §6.2 is kept as
  written, for the record.

### Results (`results/climate_horizon/weights/`)

Fold 9: 2 inner blocks (utility years 2022 and 2023) × 2 seeds, 5,250
matched cells per horizon.

| h | MAE real | MAE shuffled | g_h | unit sd | unit min–max | units > 0 | **w (shuffle)** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 14.71 | 15.18 | +0.47 | 0.37 | 0.13–0.93 | 4/4 | **0.500** |
| 2 | 16.99 | 17.80 | +0.81 | 0.71 | 0.16–1.66 | 4/4 | **0.827** |
| 3 | 19.25 | 20.42 | +1.17 | 0.91 | 0.25–2.07 | 4/4 | **1.169** |
| 4 | 21.05 | 22.57 | +1.51 | 1.17 | 0.44–2.61 | 4/4 | **1.504** |

- Shuffle weights: mean 1.000. Distance from uniform: L1 = 1.34,
  max |w − 1| = 0.50, max/min = 3.00.
- Case-only sensitivity: MAE_case 15.00 / 17.27 / 19.40 / 21.25; relative
  gain +0.019 / +0.016 / +0.008 / +0.009. Per-unit sd is 0.030–0.057, with
  2, 2, 1 and 1 of 4 units positive. Weights: **1.459 / 1.236 / 0.596 /
  0.709** (L1 = 1.39, max/min = 2.45).
- Folds 1–8: **no pilots, so missing evidence [PENDING].**

### Reading
- **Shuffle weights** measure how much a frozen equal-weight model
  *relies* on correctly aligned weather at each horizon. This reliance
  grows with h in every unit. It is not the incremental or causal value of
  climate.
- **Case-only weights** try to measure incremental value, but their
  evidence is within noise: every aggregated gain is smaller than one unit
  sd. Normalising to mean one still turns them into large weight
  differences (max/min 2.45). This happens because the rule is scale-free,
  not because of any tuning. They therefore run in the opposite direction
  to the shuffle weights and should be read as noise-driven. They stay in
  their own arm.
- Mean-one weights fix the average weight. They do not equalise gradient
  norms or learning dynamics, because the per-horizon losses differ in
  size and direction. Gradient norms will be logged when the weighted arm
  is trained.

### Commands

```powershell
.venv\Scripts\python.exe -m pytest tests/test_climate_horizon_weights.py -q
# 10 passed: uniform, all-zero, negative (clipped / rejected), single-positive,
# missing-evidence, mean-one formula, distance, aggregation order
.venv\Scripts\python.exe scripts\training\42.compute_climate_horizon_weights.py
```

Each weight file contains: the rule version, ρ, status, weights, signed
and clipped gains, aggregated and per-unit raw errors, cell and unit
counts, source run IDs, inner-block dates and period ranges, the pilots'
data/split/config fingerprints, a creation timestamp and an
interpretation note.

---

## 13. Step 8 record: climate-only gradient weighting (2026-09-30)

This supersedes the backward-only `_GradScale` router of §9, which stays in
the model fixed at weight 1 and is no longer used for weighting. Weights now
enter through two losses built from one forward pass
(`src/training/climate_weighting.py`).

**Verified baseline reduction.** The pooled masked NB loss is
Σ_h N_h / Σ_h D_h, where `nb_loss_terms` returns N_h (the summed NLL over
horizon h's observed cells) and D_h (their count). It equals
`masked_negative_binomial_loss` (tested). Over the active horizons
(D_h > 0 in the batch):

    L_h = N_h / D_h,   a_h = D_h / Σ D_h
    L_equal   = Σ a_h L_h              (= the pooled baseline loss)
    L_climate = Σ a_h w_h L_h / Σ a_h w_h

- ∇L_climate goes only to `model.climate_parameters()` (lag encoder,
  climate GRU, climate head *[v1; superseded: under v2 the climate head
  receives ∇L_equal]*). ∇L_equal goes to every other parameter
  (case trunk, case heads, gate).
- Both come from the same forward pass (`torch.autograd.grad` with
  `retain_graph`). Each gradient is assigned once. Global-norm clipping
  (1.0) is then applied over all parameters, followed by one Adam step
  with the unchanged optimiser settings. Everything runs in float32 on the
  GPU; the tests use float64.
- The weights are detached constants. If Σ a_h w_h = 0 over the batch's
  active horizons, the climate gradients are set to None (Adam skips those
  parameters) and the rest still updates.
- **Logged per step** (`info["gradient_log"]`):
  - L_equal and L_climate;
  - raw (pre-clip) norms for the climate group, the other group and in
    total;
  - the clip coefficient;
  - the clipped norms per group;
  - every 20 steps, the raw norm of the climate group's *equal-weight*
    gradient and the cosine between its weighted and equal gradients.
- **Pilots.** The step-6 pilots used the old router at weight 1. That is
  numerically the same as the new path with unit weights (test 1), so they
  remain valid.

### Verification (`tests/test_climate_weighting.py`, float64; 7 passed)
1. The loss terms reproduce the pooled reduction, and L_equal = L_climate
   = baseline loss at w = 1.
2. **Unit weights reproduce the baseline** loss (|Δ| < 1e-12), every raw
   gradient (rtol 1e-10) and the parameters after one clipped Adam update.
3. **Changing weights** changes the climate group's raw gradients and
   leaves every other raw gradient bit-identical (same parameters, same
   batch).
4. **Climate gradients equal the explicit combination**
   Σ a_h w_h ∇L_h / Σ a_h w_h, from per-horizon `autograd.grad`
   (atol 1e-12).
5. **Masked targets** set to 1e9 or NaN leave all gradients bit-identical
   and finite.
6. **Zero active climate weight** (w = [0,0,0,4] with horizon 4 masked)
   leaves the climate parameters unchanged while the others update.
7. **Logs:** with clip = 1e-3, the clipped group norms combine to exactly
   the clip norm, the clip coefficient is < 1, and the cosine lies in
   [−1, 1].

### Smoke (fold 9, seed 0, 2 epochs, CUDA; `results/climate_horizon/smoke/weighting_smoke.json`; not results)

| Weights | Raw climate norm | Raw other norm | Raw total | Clip coeff. | Cosine (weighted vs equal, climate) |
| --- | --- | --- | --- | --- | --- |
| uniform | 0.0484 | 0.1327 | 0.1415 | 1.00 | 1.000 |
| fold-9 shuffle (0.50/0.83/1.17/1.50) | 0.0565 | 0.1327 | 0.1447 | 1.00 | 0.947 |

(Means over 28 steps.) Clipping did not fire, so the raw and clipped norms
coincide here. With the fold-9 weights the climate gradient is about 17%
larger in norm and points about 19° away from the equal-weight direction.
This confirms that mean-one weights do not preserve gradient norms.

### Commands

```powershell
.venv\Scripts\python.exe -m pytest tests/test_climate_weighting.py tests/test_climate_horizon_model.py tests/test_climate_horizon_pilots.py tests/test_climate_horizon_weights.py tests/test_horizon_lag_encoder.py tests/test_climate_horizon_data.py tests/test_negative_binomial.py -q
# 67 passed
```

The smoke was run inline: `train(..., weights=uniform | fold9_shuffle.json, max_epochs=2)`
on fold 9, seed 0.

---

## 14. Step 9 record: end-to-end checks (2026-09-30)

### Checkpointing (`src/training/climate_horizon.py`)
- `train(..., checkpoint_path, resume, metadata)` writes a checkpoint
  atomically after every epoch. It contains:
  - config, model dimensions, seed and utility weights;
  - model and optimiser state;
  - early-stopping state (best state, best loss, patience counter);
  - the gradient log;
  - the RNG states: torch, CUDA, numpy and the batch-order generator.
- `checkpoint_metadata` adds the feature schema (case, climate and
  excluded channels), the per-fold scaler and imputer statistics, the
  horizon order, the delay settings, the fold, the weight source and the
  data/split fingerprints.
- `load_checkpoint` rebuilds the model from the saved dimensions and
  config. `resume=True` refuses a checkpoint written for a different seed,
  config or weights.
- The trainer was refactored (`model_dims`, `build_model` from dimensions).
  Behaviour is unchanged; all earlier tests still pass.
- `tests/test_climate_horizon_checkpoint.py` (CPU, synthetic; 3 passed):
  - 2 epochs, then a resume to 4, gives parameters, best epoch, validation
    loss and log length **bit-identical** to an uninterrupted 4-epoch run;
  - reloaded predictions (μ, α, median, gate, δ_climate) are identical;
  - the metadata is complete;
  - a mismatched resume is rejected.

### A. Synthetic checks (`src/data/climate_horizon_synthetic.py`)

The synthetic panel has 6 districts and 420 weeks. The log-mean follows
AR(1) case dynamics plus β × rainfall 8 weeks before the target week, and
counts are NB2 with α = 0.05. Two conditions: β = 0.6 (delayed climate
signal) and β = 0 (none). Equal-weight model, full `DEFAULTS` budget,
seeds 0 and 1. The test period is weeks 351–400. The shuffle permutes
climate windows across origins within each district, averaged over 5
permutations.

| Condition | Seed | Best epoch | MAE real, h=1–4 | Shuffle gain (MAE), h=1–4 | MAE with gate off | Mean gate | Mean \|δ_clim\| | Learned rain delay d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| signal | 0 | 83 | 10.9 / 11.6 / 11.7 / 12.5 | +10.8 / +14.4 / +18.2 / +20.2 | 18.5 / 21.7 / 23.5 / 25.0 | 0.85–0.90 | 0.58–0.86 | 8.2 (all h) |
| signal | 1 | 76 | 9.9 / 10.2 / 11.0 / 11.4 | +12.5 / +18.1 / +22.3 / +24.7 | 18.0 / 21.5 / 24.4 / 25.4 | 0.93–0.97 | 0.59–0.90 | 7.2 (all h) |
| none | 0 | 29 | 4.0 / 4.1 / 4.1 / 4.2 | −0.04 / −0.04 / −0.03 / −0.07 | 4.3 / 4.3 / 4.4 / 4.5 | 0.34–0.40 | 0.29–0.34 | 12.2–14.1 |
| none | 1 | 40 | 3.9 / 3.9 / 4.1 / 4.1 | −0.02 / −0.01 / −0.02 / −0.02 | 3.9 / 3.9 / 4.0 / 4.0 | 0.51–0.58 | 0.11–0.14 | 14.1–15.8 |

- **Health:** all losses and predictions are finite. Both parameter
  groups updated in every run (10/10 climate tensors, 16/16 others).
- **Learnability:**
  - With a planted signal, the model relies heavily on aligned climate,
    the gate opens, and the learned rainfall delay lands at 7.2–8.2 against
    the planted 8.
  - Without a signal, shuffling does not hurt (gains ≤ 0). The gated
    climate correction is small (0.07–0.13), and the kernels sit at
    uninformative delays of about 12–16 weeks.
  - The gate did not close fully in the null condition (mean 0.34–0.58).
    The climate path there is small in size, not switched off.
- These results show that the code can detect and localise a planted
  delay. They are not evidence about dengue biology.

### B. Short development comparison (outer fold 9, **validation year 2024 only**)

`hcd_uniform` vs `hcd_informed` (fold-9 shuffle weights
0.500 / 0.827 / 1.169 / 1.504). Both arms use the same target-relative
architecture, initialisation, seeds (0, 1), data order, full budget, NB
head, loss and checkpoint rule. The checkpoint is chosen on 2024, so these
numbers are **not held-out**. The 2025 test year was **not scored**.

| Arm | Seed | Best / run epochs | Val NLL | Val MAE, h=1–4 | Mean gate | Mean \|δ_clim\| | Mean \|gated clim\| | Raw climate grad norm | Cosine (weighted vs equal) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 0 | 33 / 48 | 3.5774 | 9.20 / 9.64 / 11.20 / 12.08 | 0.41–0.55 | 0.32–0.58 | 0.13–0.32 | 0.091 | 1.000 |
| informed | 0 | 33 / 48 | 3.5758 | 9.16 / 9.63 / 11.17 / 11.95 | 0.41–0.55 | 0.31–0.58 | 0.13–0.32 | 0.105 | 0.973 |
| uniform | 1 | 37 / 52 | 3.5859 | 9.21 / 9.92 / 11.52 / 12.35 | 0.45–0.56 | 0.32–0.61 | 0.15–0.35 | 0.095 | 1.000 |
| informed | 1 | 37 / 52 | 3.5851 | 9.21 / 9.97 / 11.57 / 12.44 | 0.45–0.56 | 0.32–0.62 | 0.15–0.35 | 0.110 | 0.972 |

- The two arms are nearly indistinguishable on the selection year:
  ΔNLL −0.0016 and −0.0008, and the MAE moves in both directions.
- Gates and correction sizes are almost the same. The learned gate rises
  with horizon (about 0.43 → 0.55) in both arms without being constrained
  to.
- Clipping never fired. The climate gradient is about 15% larger under
  weighting, with a cosine of about 0.97 to the equal-weight direction.
- On real data the climate correction (0.3–0.6) is larger than the case
  correction (0.19–0.23). Weather also carries the seasonal cycle, so this
  size is not evidence of a biological delay effect.
- **Reload:** all four checkpoints
  (`checkpoints/dev_fold9_{arm}_seed{s}.pt`) reproduced their predictions
  exactly.
- **Nothing was tuned** after seeing these numbers.

### Commands

```powershell
.venv\Scripts\python.exe -m pytest tests/test_climate_horizon_checkpoint.py tests/test_climate_weighting.py tests/test_climate_horizon_model.py tests/test_climate_horizon_pilots.py -q
# 22 passed
.venv\Scripts\python.exe scripts\evaluation\43.climate_horizon_end_to_end_checks.py
```

Outputs: `results/climate_horizon/end_to_end/{synthetic_checks,dev_comparison_fold9_val2024,checkpoint_reload}.json`
and `results/climate_horizon/checkpoints/`.

---

## 15. Step 10 record: development matrix (2026-09-30)

`scripts/evaluation/44.climate_horizon_dev_matrix.py`. Outer fold 9, seeds
0 and 1. All arms share:
- the full budget, NB head and loss;
- training-only preprocessing and the same origins;
- initialisation (the same seed gives the same initial weights; the lag
  modes have identical parameter shapes) and data order;
- 10,028 parameters.

In every arm the case branch, gate and heads learn from L_equal (all
horizons). Only the climate group's loss changes. **Scored on 2024, the
checkpoint-selection year: these are not held-out estimates. 2025 was not
scored.** Chronos was not used.

| Arm | Lag mode | Climate weights |
| --- | --- | --- |
| A | origin | 1,1,1,1 |
| B | target | 1,1,1,1 (**main baseline**) |
| C | origin | fold-9 shuffle weights 0.500/0.827/1.169/1.504 |
| D | target | same file (**proposed**) |
| E | target | fixed 1,2,3,4 |
| F | target | h=4 only (0,0,0,1) |
| G | target | fold-9 case-only-utility weights 1.459/1.236/0.596/0.709 |
| H | target | GradNorm on the climate group (`src/training/climate_gradnorm.py`) |

`L_climate` is normalised by Σ a_h w_h, so it depends only on the weights'
*ratios*. [1,2,3,4] and its mean-one rescaling give identical training.

**GradNorm (H).** Implemented from the algorithm in Chen et al. 2018:
G_i = ‖∇_W w_i L_i‖, r_i = relative inverse training rate,
L_grad = Σ|G_i − Ḡ·r_i^α|, weights renormalised to sum T. Deviations,
labelled:
- the weights act on the climate group only;
- W is the climate GRU's recurrent weight, not the network's last shared
  layer;
- α = 1.5 and weight lr = 0.025 are fixed a priori, not tuned;
- not resumable.

The implementation was checked against my reading of the paper's
algorithm, not against the paper text in this session: **verify before
citing**. Final GradNorm weights: seed 0 1.67/0.98/0.75/0.60, seed 1
1.42/0.94/0.94/0.70. These *decrease* with h, the opposite of the
measured weights.

### Results (MAE, mean of 2 seeds, 1,300 common 2024 cells per horizon)

| Arm / reference | h=1 | h=2 | h=3 | h=4 | Seed range, h=1–4 | Val NLL |
| --- | --- | --- | --- | --- | --- | --- |
| A origin, equal | 9.13 | 9.71 | 11.27 | 11.98 | 0.11 / 0.17 / 0.22 / 0.18 | 3.5772 |
| **B target, equal** | 9.21 | 9.78 | 11.36 | 12.21 | 0.01 / 0.29 / 0.32 / 0.27 | 3.5816 |
| C origin, measured | 9.11 | 9.77 | 11.33 | 12.10 | 0.14 / 0.10 / 0.07 / 0.00 | 3.5811 |
| **D target, measured** | 9.19 | 9.80 | 11.37 | 12.20 | 0.06 / 0.34 / 0.40 / 0.49 | 3.5804 |
| E target, 1,2,3,4 | 9.21 | 9.77 | 11.35 | 12.18 | 0.02 / 0.38 / 0.36 / 0.48 | 3.5800 |
| F target, h=4 only | 9.18 | 9.86 | 11.42 | **11.50** | 0.10 / 0.05 / 0.20 / 0.20 | 3.5907 |
| G target, case-only utility | 9.21 | 9.80 | 11.34 | 12.18 | 0.03 / 0.06 / 0.15 / 0.00 | 3.5799 |
| H target, GradNorm | 9.18 | 9.71 | 11.20 | 12.00 | 0.03 / 0.11 / 0.09 / 0.04 | 3.5771 |
| `gru_v2_nb` (stored) | 9.26 | 9.62 | 11.06 | 11.47 | | |
| `nb_shared_v2` (stored) | 9.34 | 9.83 | 10.95 | 11.41 | | |
| `lgbm_v2` (stored) | 9.03 | 9.46 | 10.59 | 11.40 | | |
| persistence | 10.78 | 12.37 | 15.65 | 17.53 | | |

Peak MAE (h=1–4): B 21.30/20.90/27.18/27.03; D 21.27/20.96/27.18/26.87;
A 21.39/20.76/26.66/25.72; F 22.55/23.69/32.18/26.21;
`gru_v2_nb` 21.57/19.86/27.20/26.24; `lgbm_v2` 21.57/19.52/26.43/26.76;
persistence 25.77/27.69/40.80/45.23. Full tables:
`results/climate_horizon/dev_matrix/summary.csv` and
`metrics_by_seed.csv`.

### Reading (one fold, one selection year, two seeds: nothing here is established)
- **Proposed vs baseline (D − B): −0.02 / +0.02 / +0.01 / −0.02 MAE.**
  This is null, well inside the seed range (up to 0.49). Val NLL differs
  by −0.001.
- **Other weighting schemes.** E (fixed ramp) and G (case-only utility)
  are equally null. GradNorm (H) is 0.07–0.21 better than B, still within
  about one seed range, and its weights run the other way.
- **Target vs origin lags: a negative result for target-relative on this
  year.** Equal weights: A − B = −0.08 / −0.07 / −0.09 / −0.23. Measured
  weights: C − D = −0.07 / −0.03 / −0.04 / −0.09. The h=4 gap (0.23) is
  near the seed range (0.18–0.27).
- **h=4-only (F).** The climate heads for h=1–3 get no gradient, so they
  stay at their zero initialisation. The model is then effectively
  case-only at h=1–3 (gated climate exactly 0; gate stuck at 0.5). At h=4
  it has the lowest MAE of all extension arms (11.50 vs 12.21), but it is
  worse at h=2–3 and on peak MAE at h=1–3. This is one fold and one year.
  It is recorded as a lead, not a result.
- **Negative against stored references.** At h=3–4, every extension arm
  trails `lgbm_v2`, `nb_shared_v2` and `gru_v2_nb` on this year by
  0.1–0.8 MAE (F only at h=4 is at their level). The references also
  selected checkpoints on 2024, but they use the benchmark's preprocessing
  (scaler fitted up to 2024) and hand-lagged v2 inputs. Every model beats
  persistence by a wide margin.
- **Lag support.**
  - Origin mode: available mass 1 at all h; its rainfall centre of mass,
    as a target delay, shifts by exactly +1 per horizon (4.4 → 7.4).
  - Target mode: usable mass 0.92 → 0.75 from h=1 to 4 (the kernel mass
    below d = h is unusable); rainfall delay 5.7 → 6.7.
- **Gates** rise with h in every arm except F (about 0.40 → 0.55).
  Gated climate corrections are 0.13 → 0.32 and are nearly identical
  across weighted arms.
- No climate step was skipped in any arm. Reference provenance: all three
  stored methods pass the per-target split rule on these cells, with
  seeds 0–2.

### Commands

```powershell
.venv\Scripts\python.exe -m pytest tests/test_climate_gradnorm.py tests/test_climate_weighting.py tests/test_climate_horizon_checkpoint.py tests/test_climate_horizon_model.py -q
# 21 passed
.venv\Scripts\python.exe scripts\evaluation\44.climate_horizon_dev_matrix.py
```

`results/benchmark/` was not written; its only local change is the dated
hold-out addendum from step 2.

---

## 16. Step 11 record: development review and freeze (2026-09-30)

### Review of the development evidence (fold 9; 2024 = selection year; 2 seeds)
- **Utility weighting vs the identical equal-weight baseline (D − B):**
  −0.02 / +0.02 / +0.01 / −0.02 MAE. **No measurable improvement.** This
  is a valid finding at this stage.
- **Weights.** The fold-9 development weights were not near-uniform
  (0.50–1.50, max/min 3.0). Even so, the model barely changed: the gated
  climate corrections and gates were nearly identical to the baseline's.
- **Alignment (reported separately).** Origin-relative lags were 0.03–0.23
  better than target-relative on this year, within or near the seed range.
  Target-relative stays the declared main mode. One fold and two seeds do
  not justify a change, and switching now would be selection on
  development results.
- **Simple controls.** Fixed [1,2,3,4] and case-only-utility weights were
  null. GradNorm was slightly better (up to 0.21) with weights decreasing
  in h. h=4-only improved h=4 but hurt h=2–3.
- **Nothing was selected using outer-test results.** No 2017–2025 test year
  of this extension had been scored when the freeze was made.

### Frozen (`results/climate_horizon/run_manifest_2026-09-30.json`)

SHA-256 `f4acdf1c65311fb1116cf64d46803245619f58be7434f415bb6b3f30fd1245cd`,
written 2026-09-30 16:44 +0530. The file and its `.sha256` are read-only,
and script 46 refuses to overwrite them. The manifest contains:
- the SHA-256 of all 21 extension and pipeline code files (the extension
  code is **not committed** to git; the hashes identify it);
- the data hashes;
- the specification below;
- the exact commands;
- the run count and cost.

| Item | Frozen value |
| --- | --- |
| Architecture / likelihood | `ClimateHorizonNB`, target-relative lags (history 26, max delay 29), NB2, NB-median point forecast, 10,028 parameters |
| Training | `DEFAULTS`: Adam 3e-3 (encoder 2e-2, no weight decay), batch 64, clip 1.0, max 150 epochs, patience 15 on the **unweighted** validation NLL |
| Preprocessing | Fitted on training years only; 37-week origins; targets split by their own date |
| Pilots | K = 3 inner blocks × seeds 0, 1 × 5 permutations (11, 23, 37, 41, 53); donor rule ±2 weeks, same district, different year, pilot-training history |
| Utility | Primary: absolute shuffle gain. Sensitivity: case-only relative gain. Aggregation: seeds within block, then blocks equally. No smoothing across folds. |
| Weights | ρ = 0.95; uniform if all gains ≤ 0; written to `weights_final/` from blocks 0–2 × seeds 0–1. **Fold-9 development weights (2 blocks) are not used.** |
| Arms / seeds | B, D: seeds 0–4. A, C, E, F, G, H: seeds 0–2. |
| Folds | Retrospective 1–9 (headline 1,2,3,6,7,8,9; 4–5 COVID reported separately); hold-out fold 10 (B, D only) |
| Metrics | MAE per fold over observed test cells; seed-mean estimand primary, seed ensemble secondary; peak MAE (p90 thresholds as in the evaluator); NLL; WIS |
| Primary comparison | D vs B, headline MAE, h = 1–4; paired t and Wilcoxon over 7 folds; Holm across 4 horizons; 4-week block bootstrap 95% CI |
| Practical improvement | D − B ≤ −0.30 MAE at h = 3 or 4, with Holm p < 0.05 and bootstrap CI < 0 |
| Non-inferiority | h = 1, 2: bootstrap CI upper bound of D − B ≤ +0.30 |
| Otherwise | "No measurable difference" |

**Still to implement before scoring:** the primary statistics script
(paired tests, Holm, block bootstrap for the extension). Its specification
is frozen above. Its code is **[PENDING]** and will be hashed and added in
a dated manifest supplement, before any outer test year is scored.

### Retrospective vs 2026
- **Retrospective:** folds 1–9 (2017–2025). These years guided nothing in
  this extension so far, but they are not a hold-out.
- **Final 2026 scoring:** fold 10, B and D only, run once after the
  retrospective report. The dated addendum (16:44) in
  `results/benchmark/holdout_declaration.md` records prior exposure:
  benchmark 2026 summaries were read on 29–30 Sep. 2026 is therefore a
  frozen-specification check, not a blind test.
- **Approval:** the repository has no approval step, and none is claimed.

### Exact resumable commands (also in the manifest)

```powershell
.venv\Scripts\python.exe scripts\training\41.run_climate_horizon_pilots.py --folds 1 2 3 4 5 6 7 8 9 10 --blocks 0 1 2 --seeds 0 1
.venv\Scripts\python.exe scripts\training\42.compute_climate_horizon_weights.py --out results\climate_horizon\weights_final --require-blocks 0 1 2 --require-seeds 0 1
.venv\Scripts\python.exe scripts\training\45.run_climate_horizon_sweep.py --folds 1 2 3 4 5 6 7 8 9
.venv\Scripts\python.exe scripts\evaluation\39.evaluate_climate_horizon.py
# primary statistics script [PENDING]
# after the retrospective report only:
.venv\Scripts\python.exe scripts\training\45.run_climate_horizon_sweep.py --folds 10 --holdout --confirm-holdout
```

Every unit is skipped if its outputs exist. Weight files are never
overwritten, and the manifest hash is checked on every sweep run.

### Run count and cost (measured on an RTX 4070 Laptop GPU)

| Stage | Units / fits | Estimate |
| --- | --- | --- |
| Pilots | 60 units (4 already done) = 120 fits + inference-only shuffles | about 18 min (max 25); 18.9 s per unit on average |
| Retrospective | B, D 90 fits + A, C, E, F, G, H 162 fits = 252 | about 41 min (max 56); 9.2 s per fit, 14.0 s with GradNorm |
| 2026 | 10 fits | about 2 min |
| Total | 382 fits | about 1 h (≤ 1.5 h); roughly 2–3× on a Colab T4 |

### Changes in this step
- `src/training/climate_horizon_arms.py`: the frozen arm, seed, fold and
  weight-lookup specification.
- `scripts/training/45.run_climate_horizon_sweep.py`: the resumable sweep,
  with hold-out guards.
- `scripts/evaluation/46.freeze_climate_horizon_manifest.py`: the
  write-once manifest.
- `scripts/training/41`: `--folds` covers 1–10; the default is blocks 0–2.
- `scripts/training/42`: `--out`, `--require-blocks`, `--require-seeds`;
  folds with incomplete evidence get no weights.
- `scripts/evaluation/39`: reads per-arm prediction directories, folds
  1–9 only.
- `tests/test_climate_horizon_arms.py` (3 tests).
- Checks: 112 extension and related tests pass. Nothing was trained or
  scored.

---

## 17. Step 12 record: retrospective evaluation, folds 1–9 (2026-09-30, written before 2026 was scored)

Executed under the frozen manifest (`f4acdf1c…`). Supplements, all dated
and read-only:
- `run_manifest_supplement_2026-09-30T1649_stats.json` (16:49): adds the
  statistics script; script 45 changed only for logging, checkpoints and
  error capture.
- `run_manifest_supplement_2026-09-30T1750_retry.json` (17:50): adds a
  checkpoint-save retry after a Windows PermissionError.

### Execution
- **Pilots.**
  - 60 units: folds 1–10 × inner blocks 0–2 × seeds 0–1 (the 4 fold-9
    units from step 6 were reused; same protocol).
  - Each unit trains 2 fits and runs 5 donor-perturbation inferences.
  - Total 10.9 min; per fold 38–113 s.
  - Matched cells: 100% everywhere.
- **Weights** (`weights_final/`). All 10 folds were "informed" (every
  aggregated shuffle gain was positive).
  - Shuffle weights rise with h in every fold; max/min ratio 1.7–4.0.
    Examples: fold 1 0.64/0.72/0.70/1.94; fold 6 0.46/0.81/1.18/1.54;
    fold 10 0.41/0.75/1.19/1.65.
  - Case-only sensitivity weights are erratic (max/min up to 77).
- **Sweep.**
  - 252 units: B and D with seeds 0–4; the six other arms with seeds 0–2;
    folds 1–9. Total 47.6 min on the RTX 4070 Laptop GPU, peak GPU memory
    274 MB.
  - Mean time per fit: 9.5–11.3 s (GradNorm 17.1 s). Mean best epoch 31–35.
  - Clipping fired on at most 2.8% of steps; no climate step was skipped.
  - **8 units failed first** with `PermissionError` while replacing a
    checkpoint file (records in `logs/failed_attempts/`). Their partial
    checkpoints were deleted, the save retry was added (supplement 17:50),
    and the 8 units were retrained from scratch. **252/252 complete, 0
    outstanding errors.**
- **Scoring.**
  - `scripts/evaluation/39` and `47`, h = 1–4 only. Benchmark files were
    read only: nothing in `results/benchmark/predictions/` was written,
    and script 34 was not invoked.
  - 11,725 common test cells per horizon (folds 1–9); coverage 100% for
    every arm and reference.
  - Persistence was recomputed on those cells and reproduces
    16.42/20.16/24.58/28.47 on the headline folds.
  - Short-paper GEE neural scores are not mixed in.

### Primary result (frozen criteria): D_target_measured vs B_target_equal, headline folds 1,2,3,6,7,8,9, 5 seeds

| h | B | D | D − B | t p | Holm p | Wilcoxon p | D wins | Bootstrap 95% CI | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16.64 | 16.71 | +0.063 | 0.259 | 0.777 | 0.813 | 3/7 | [−0.030, +0.174] | non-inferior (margin 0.30) |
| 2 | 19.67 | 19.76 | +0.082 | 0.151 | 0.602 | 0.375 | 3/7 | [−0.032, +0.213] | non-inferior (margin 0.30) |
| 3 | 22.71 | 22.77 | +0.068 | 0.307 | 0.777 | 0.469 | 3/7 | [−0.071, +0.227] | no measurable difference |
| 4 | 25.22 | 25.27 | +0.049 | 0.522 | 0.777 | 0.688 | 3/7 | [−0.105, +0.205] | no measurable difference |

**Utility-weighted climate gradients did not improve on the identical
equal-weight model at any horizon.** The point estimates are slightly
worse (+0.05 to +0.08 MAE), and the non-inferiority margin holds at
h = 1–2. All 9 folds give the same picture: D − B = +0.05 to +0.07,
Holm p ≥ 0.48, and every CI includes 0.

### Secondary and exploratory (not in the primary family; unadjusted p)

Headline MAE (7 folds):

| Method | h=1 | h=2 | h=3 | h=4 | Peak h=4 | WIS h=4 | 95% cov h=4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A origin, equal | 16.84 | 19.97 | 23.13 | 25.73 | 43.98 | 15.73 | 0.933 |
| B target, equal | 16.64 | 19.67 | 22.71 | 25.22 | 42.99 | 15.24 | 0.938 |
| **C origin, measured** | **16.20** | **19.28** | **22.37** | **24.95** | **42.56** | **14.99** | 0.940 |
| D target, measured | 16.71 | 19.76 | 22.77 | 25.27 | 42.82 | 15.32 | 0.935 |
| E fixed 1,2,3,4 | 16.54 | 19.58 | 22.60 | 25.11 | 42.60 | 15.13 | 0.939 |
| F h=4-only | 16.64 | 20.21 | 23.88 | 25.48 | 43.94 | 15.17 | 0.939 |
| G case-only utility | 17.11 | 20.22 | 23.35 | 25.94 | 43.90 | 15.94 | 0.932 |
| H GradNorm | 16.61 | 19.69 | 22.74 | 25.26 | 43.23 | 15.27 | 0.938 |
| `gru_v2_nb` (stored) | 15.94 | 19.26 | 22.46 | 25.07 | 42.60 | | |
| `nb_shared_v2` (stored) | 16.63 | 19.83 | 23.15 | 25.84 | 43.82 | | |
| `lgbm_v2` (stored) | 16.99 | 21.05 | 24.30 | 27.16 | 46.42 | | |
| persistence | 16.42 | 20.16 | 24.58 | 28.47 | 47.98 | | |

- **Alignment interacts with weighting.**
  - With equal weights, origin-relative lags are worse than
    target-relative: A − B = +0.20 to +0.51 (p 0.27–0.47).
  - With measured weights, origin-relative is better: C − D = −0.50 /
    −0.48 / −0.40 / −0.32 (Wilcoxon p 0.016 / 0.016 / 0.047 / 0.078;
    wins 7/7, 7/7, 6/7, 5/7).
  - **C is the best extension arm on MAE, peak MAE and WIS at every
    horizon.** This is exploratory, has 3 seeds, is not
    multiplicity-adjusted, and was **not** a declared configuration. It
    cannot replace D for 2026, and it must be confirmed on new data before
    being claimed.
- **Controls.**
  - Fixed [1,2,3,4] (E): −0.09 to −0.11 vs B, 5/7 wins, p ≥ 0.11.
  - GradNorm (H): within ±0.04 of B.
  - h=4-only (F): worse at h=2–3 (+1.17 at h=3).
  - Case-only-utility weights (G): worse (+0.46 to +0.72).
- **Against persistence.**
  - B/D lose at h=1 (+0.22/+0.28) and win at h=3 (−1.87/−1.80,
    p ≈ 0.03–0.04) and h=4 (−3.25/−3.20, Wilcoxon p 0.016–0.047).
  - Peak MAE at h=4 is 42.99/42.82 against 47.98.
- **Against stored references.**
  - Better than `nb_shared_v2` at h=3–4 by about 0.4–0.6 (p ≥ 0.35) and
    than `lgbm_v2` by 1.4–1.9 (p ≥ 0.38).
  - Worse than `gru_v2_nb` at every horizon (+0.70 at h=1, +0.15 at h=4).
  - None of these differences is significant.
- **All 9 folds:** the same ordering (C best among extension arms;
  B 14.55/17.27/19.98/22.32).
- **COVID folds 4–5 (reported separately):** extension arms
  7.2/8.8/10.4/12.2 (B) against `nb_shared_v2` 6.70/8.06/9.54/10.79.
  The stored references do better in the COVID years.
- **Empirical 95% coverage** of the extension arms' NB intervals:
  0.93–0.96. That is coverage at one nominal level, not a calibration
  claim.

Outputs: `results/climate_horizon/statistics/` (primary, secondary,
exploratory, per-seed and per-fold metrics, WIS, coverage, seeds per
fold), `results/climate_horizon/evaluation/`, `logs/`.

### Commands actually run

```powershell
.venv\Scripts\python.exe scripts\evaluation\48.write_manifest_supplement.py --tag stats --add scripts/evaluation/47.climate_horizon_statistics.py scripts/evaluation/48.write_manifest_supplement.py --note "..."
.venv\Scripts\python.exe scripts\training\41.run_climate_horizon_pilots.py --folds 1 2 3 4 5 6 7 8 9 10 --blocks 0 1 2 --seeds 0 1
.venv\Scripts\python.exe scripts\training\42.compute_climate_horizon_weights.py --out results\climate_horizon\weights_final --require-blocks 0 1 2 --require-seeds 0 1
.venv\Scripts\python.exe scripts\training\45.run_climate_horizon_sweep.py --folds 1 2 3 4 5 6 7 8 9   # run twice: 8 retries
.venv\Scripts\python.exe scripts\evaluation\48.write_manifest_supplement.py --tag retry --note "..."
.venv\Scripts\python.exe scripts\evaluation\39.evaluate_climate_horizon.py
.venv\Scripts\python.exe scripts\evaluation\47.climate_horizon_statistics.py
```

---

## 18. Step 12b record: 2026 hold-out (fold 10), frozen specification, scored once (2026-09-30)

Scored after §17 was written (snapshot `logs/retrospective_report_snapshot_1751.md`).

**Setup.**
- Only B and D, 5 seeds each, as declared.
- Training uses data to the end of 2024, early stopping uses 2025, and
  preprocessing is fitted on training years.
- Utility pilots scored 2022/2023/2024 inside training history. Fold-10
  weights: 0.410 / 0.752 / 1.189 / 1.649.
- No tuning. 10 fits, 18–58 s each, no errors.

**Scoring.**
- 19 weeks (2026-01-05 to 2026-05-17), 474 observed cells per horizon.
- The stored references carry one additional, unobserved, cell (coverage
  0.998). Unobserved cells are never scored.
- Persistence was recomputed on the same cells.
- References were read only.

| MAE, 2026 | h=1 | h=2 | h=3 | h=4 |
| --- | --- | --- | --- | --- |
| B target, equal | 15.63 | 17.29 | 19.98 | 22.14 |
| D target, measured | 15.64 | 17.51 | 20.08 | 22.45 |
| `gru_v2_nb` (stored) | 15.60 | 16.98 | 19.57 | 21.90 |
| `lgbm_v2` (stored) | 15.77 | 17.94 | 19.65 | 21.39 |
| `nb_shared_v2` (stored) | 16.26 | 17.42 | 19.06 | 20.68 |
| persistence | 17.60 | 18.78 | 20.92 | 22.47 |

| 4-week block bootstrap | h=1 | h=2 | h=3 | h=4 |
| --- | --- | --- | --- | --- |
| D − B | +0.01 [−0.16, +0.20] | +0.22 [−0.08, +0.57] | +0.10 [−0.24, +0.47] | +0.32 [−0.06, +0.64] |
| B − persistence | −1.98 [−5.08, +0.49] | −1.49 [−5.15, +1.75] | −0.94 [−4.92, +2.59] | −0.33 [−5.51, +3.73] |
| D − persistence | −1.97 [−5.25, +0.64] | −1.27 [−5.13, +2.08] | −0.84 [−5.20, +2.91] | −0.01 [−5.13, +4.28] |

- **The primary contrast has the same sign as the retrospective one** (D
  slightly worse than B at every horizon). By the 29 September wording it
  is "consistent with" the retrospective finding of no improvement. Every
  interval includes zero.
- Both models beat 2026 persistence on MAE point estimates at every
  horizon, but every interval includes zero.
- On **peak MAE**, both are worse than persistence at h = 2–4 in 2026:
  - B: 27.24 / 33.46 / 40.35 / 44.82;
  - D: 27.41 / 34.16 / 40.94 / 46.12;
  - persistence: 27.19 / 31.22 / 35.68 / 39.65.
- WIS: B 9.70–12.50 and D 9.59–12.55. 95% interval coverage is
  0.930–0.945.
- **The stored references rank above B and D at h = 3–4 in 2026.**
  `nb_shared_v2` is at 19.06 / 20.68.
- Limits: 19 weeks (January to May, the first transmission peak only);
  no across-year test is possible. The 2026 results of five other models
  were read before this extension was specified, so this is a
  frozen-specification check, not a blind test.

Outputs: `results/climate_horizon/holdout_2026/` (MAE and bootstrap,
per-seed metrics, WIS, coverage) and `logs/*holdout*`.

```powershell
.venv\Scripts\python.exe scripts\training\45.run_climate_horizon_sweep.py --folds 10 --holdout --confirm-holdout
.venv\Scripts\python.exe scripts\evaluation\47.climate_horizon_statistics.py --holdout
```

### Total compute for this step
- Pilots 10.9 min.
- Retrospective 252 fits, 47.6 min.
- Hold-out 10 fits, about 5 min.
- Scoring a few minutes.
- About 65 GPU-minutes on the RTX 4070 Laptop GPU; peak GPU memory
  274 MB.

**Nothing is [PENDING] for the frozen protocol.** Any follow-up on arm C
needs new data or a new declaration.


---

## 19. Correction: routing v2 (2026-09-30)

§9 and §13 describe routing v1, which placed the climate correction head
in the weighted group. That was an implementation error against the
step-4 specification. It is fixed in routing v2: only the lag encoder and
the climate GRU are weighted, and every prediction head is unweighted.

Details, tests and the smoke check are in
[`climate_horizon_correction_audit.md`](climate_horizon_correction_audit.md)
§7. All §15–18 results were produced under v1. Corrected reruns are
**[PENDING]**.


---

## 20. Consolidated final state (2026-09-30)

For the current results and the reproduction commands, see
[`climate_horizon_results.md`](climate_horizon_results.md) and the
correction audit §9–11.

**Documentation corrections made here:**
- §9 and §13 are marked as routing v1.
- The mean-one claim in §6.2 is qualified.
- "Calibration" became "empirical coverage" (§17).
- The "fairly flat weights" expectation in §7 is annotated with the
  measured outcome.

**Earlier versions of this file** are preserved in
`results/climate_horizon/correction_2026-09-30/doc_snapshots/` and in git.

# Climate-horizon extension: results (corrected, authoritative)

**Last revised:** 2026-09-30, after the routing correction.

**What this document covers.** The authoritative corrected results. The
original (pre-correction) results are kept in §9 as history.

**Where the numbers come from.** Every number cites the artefact it comes
from. Paths are relative to `results/climate_horizon/`.

- **Protocol and freeze:** `run_manifest_2026-09-30.json` (SHA-256
  `f4acdf1c…`, 16:44).
- **Correction:** `correction_2026-09-30/correction_manifest.json`
  (SHA-256 `5434845f…`, 18:58).
- **Audit trail:** [`climate_horizon_correction_audit.md`](climate_horizon_correction_audit.md).
- **Corrected tables and figures:** `correction_2026-09-30/analysis/`,
  produced by `scripts/evaluation/49.climate_horizon_analysis.py --correction`.

**Protocol history.**
- **Original specification.** Arms, seeds, folds, utilities and decision
  criteria were frozen at 16:44 (`run_manifest_2026-09-30.json`), before
  any outer test year of this extension was scored.
- **Correction.** After the original results had been inspected, an
  implementation error in gradient routing was found (the climate head was
  in the weighted group). It was corrected, and the affected arms were
  rerun under a correction manifest written at 18:58.
- **Status of the corrected results.** Corrected evaluations on data
  already inspected, both 2017–2025 and 2026. They are **not** a blind or
  newly preregistered test.
- **What stayed fixed.** The decision criteria, seeds and folds are those
  of the original specification.

**Path convention.** Every path below is relative to
`results/climate_horizon/`.
- Corrected artefacts are under `correction_2026-09-30/`.
- Artefacts reused unchanged are marked **(reused)**: A, B, the pilots,
  `weights_final/`.
- Original v1 artefacts are cited only in §9 (history).

---

## 1. What was tested

- **Model.** A two-branch Negative Binomial (NB2) forecaster for h = 1–4
  weeks. A case-history GRU with no spatial message passing sits beside a
  climate branch: learnable per-district, per-variable Gaussian-mixture
  delay kernels (`HorizonLagEncoder`) followed by a climate GRU.
- **Fusion.** For each horizon, the corrections are added in NB log-mean
  space: δ_case + sigmoid(gate)·δ_climate, on top of the anchor
  log(1 + y_origin). The dispersion is case-based. 10,028 parameters.
- **What receives weighted gradients (routing v2, as corrected and run).**
  - **Weighted loss L_climate:** only the climate feature encoder, i.e.
    `encoder.mixture.{centre_raw, width_raw, node_embedding, feature_projection}`
    and `climate_gru.*`.
  - **Unweighted loss L_equal:** everything else. That is the case
    branch, the gate, the dispersion head and **all prediction heads,
    including the climate correction head `climate_delta`**. Every
    horizon's prediction head remains trained using its ordinary,
    unweighted loss.
  - No parameter is shared between the groups.
  - L_climate = Σ a_h w_h L_h / Σ a_h w_h, with a_h the share of observed
    cells at horizon h. L_equal = Σ a_h L_h (the pooled masked NB loss).
- **Weights w_h.** Per fold, computed from inner-validation pilots inside
  the training years, from how much a **frozen equal-weight** model
  degrades when its climate windows are swapped for same-season windows of
  other years ("shuffle reliance"). Rule: w = 0.05 + 0.95·4·u/Σu. The
  weights are frozen in `weights_final/` **(reused)**.
- **Main comparison.** D (target-relative lags, measured weights) against
  B (target-relative lags, equal weights). Everything else is identical.

**Primary, planned exploratory and post-hoc analyses**

| Status | Analyses |
| --- | --- |
| Primary, declared in the 16:44 manifest | D vs B, headline MAE, h = 1–4, 5 seeds |
| Planned exploratory, declared in the manifest | A, C (alignment); E fixed [1,2,3,4]; F h=4-only; G case-only utility; H GradNorm on the climate group; 3 seeds each |
| Post-hoc, designed after the results were seen | the matched-seed reanalysis of exploratory contrasts (§4); the 2×2 factorial contrasts (§4); block-length sensitivity; gradient diagnostics (§6) |

## 2. Primary result (corrected): measured weighting did not improve forecasts

D − B, headline folds (2017–19, 2022–25), seed-mean estimand, 5 seeds.
The frozen criteria are unchanged: improvement needs D − B ≤ −0.30 at
h=3 or 4 with Holm p < 0.05 and a CI below 0; non-inferiority at h=1–2
needs a CI upper bound ≤ +0.30.

Source: `correction_2026-09-30/statistics/primary_D_vs_B_headline7.csv` and
`correction_2026-09-30/analysis/tables/primary_bootstrap_estimands.csv`. B's predictions
are **(reused)**, verified bit-identical under v2.

| h | B MAE | D MAE | D − B | Holm p | 4-week block 95% CI | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 16.64 | 16.70 | +0.061 | 0.72 | [−0.030, +0.170] | non-inferior |
| 2 | 19.67 | 19.75 | +0.079 | 0.44 | [−0.034, +0.208] | non-inferior |
| 3 | 22.71 | 22.78 | +0.071 | 0.72 | [−0.067, +0.231] | no measurable difference |
| 4 | 25.22 | 25.28 | +0.060 | 0.72 | [−0.096, +0.223] | no measurable difference |

- **The same null holds** for the seed-ensemble estimand (+0.057 to
  +0.061) and for all 9 folds (+0.049 to +0.072, Holm p ≥ 0.37).
- **Block length** of 8 or 13 weeks (post-hoc) leaves every CI including
  zero (`correction_2026-09-30/matched_seed/primary_block_sensitivity_v2.csv`, computed on
  the corrected D).
  - The earlier `evaluation_audit/block_sensitivity.csv` used the **v1** D
    predictions and is superseded for the corrected result.
- **2026** (19 weeks, B reused, D rerun): D − B = +0.18 / +0.28 / +0.28
  / +0.50. Source: `correction_2026-09-30/holdout_2026/bootstrap_2026.csv`. The
  interval's properties are in `correction_2026-09-30/matched_seed/diagnostic_facts.json`.
  - At h=4 the interval is [+0.03, +1.00].
  - It is a **pointwise, unadjusted** 95% percentile interval from a
    4-week block bootstrap with only 5 blocks (19 weeks), 2000 resamples.
  - There is no correction for 4 horizons or for the several comparisons
    scored on 2026.
  - Outcomes had been seen before: benchmark 2026 results were read before
    this extension, and the v1 B/D 2026 results before the correction.
  - So it is **not** a confirmatory result. At most it is consistent in
    sign with the retrospective finding that D is not better than B.

**Seed-mean and ensemble estimands**
- **Seed-mean:** the mean over seeds of per-seed MAE, then equal weight
  across folds.
- **Ensemble:** MAE of the per-cell mean of seed forecasts.
- They are reported separately throughout. Tests confirm the definitions
  (target 10, forecasts 0 and 20 gives seed-mean MAE 10 and ensemble MAE
  0).

## 3. Accuracy by horizon (corrected)

Headline folds, seed-mean over each method's own seeds (B and D: 0–4;
stored references: 0–2; persistence and Chronos-2: one deterministic
forecast).

Sources:
- MAE, RMSE, peak MAE: `correction_2026-09-30/analysis/tables/metrics_headline_seed_mean.csv`
  (ensemble version: `metrics_headline_ensemble.csv`);
- WIS and coverage: `correction_2026-09-30/statistics/wis_by_seed.csv`.

These are per-method levels. For **paired** contrasts between arms with
different seed counts, see §4 (matched seeds).

| Method | MAE h1 | h2 | h3 | h4 | RMSE h4 | Peak MAE h4 | WIS h4 | 95% cov. h4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B target, equal | 16.64 | 19.67 | 22.71 | 25.22 | 55.13 | 42.99 | 15.24 | 0.938 |
| D target, measured | 16.70 | 19.75 | 22.78 | 25.28 | 55.17 | 42.81 | 15.31 | 0.935 |
| Persistence (same cells) | 16.42 | 20.16 | 24.58 | 28.47 | 61.96 | 47.98 | — | — |
| GRU-NB separate, stored | 15.94 | 19.26 | 22.46 | 25.07 | 54.60 | 42.60 | — | — |
| GRU-NB shared, stored | 16.63 | 19.83 | 23.15 | 25.85 | 56.89 | 43.82 | — | — |
| LightGBM, stored | 16.99 | 21.05 | 24.30 | 27.16 | 57.65 | 46.42 | — | — |
| Chronos-2 joint, zero-shot (outside comparator) | 15.96 | 19.07 | 22.29 | 25.11 | 54.74 | 43.74 | — | — |

- **B and D against persistence.** Both lose to persistence at h=1 and
  beat it at h=3–4 (h=4: −3.2 MAE). But B does this as much as D, so
  **beating persistence does not show any benefit of weighting.**
- **Uncertainty intervals.** The empirical 95% coverage of the NB
  intervals is 0.93–0.96. That is measured coverage at one nominal level
  only, not a full calibration assessment.
- **Stored baselines** are scored on the identical 11,725 cells per
  horizon and pass the provenance checks. Their training protocols differ:
  - preprocessing fitted through the validation year;
  - hand-lagged climate inputs (`gru_v2_nb`, `nb_shared_v2`) or tabular
    lags (`lgbm_v2`);
  - `nb_shared_v2` trains one trunk for 8 horizons;
  - the files carry no embedded data fingerprint (the data files still
    match the manifest hashes).

  They are direct forecast comparisons, not ablations.
- **Chronos-2** is a permitted outside comparator only: zero-shot, no
  fine-tuned variants, and it contributes nothing to the method.

## 4. Planned exploratory arms and post-hoc factorial (corrected)

**Arms vs B on matched seeds.** Headline MAE, seeds 0, 1 and 2 for **both**
arms of every contrast, on the identical 11,725 cells per horizon (same
fold, district, origin, target, horizon and mask). Unadjusted and
exploratory.

Sources:
- `correction_2026-09-30/matched_seed/matched_seed_contrasts.csv` and `matched_seed_arm_means.csv`;
- regenerated table `correction_2026-09-30/analysis/tables/controls_vs_B.csv` (identical
  values).

The earlier corrected table compared B's seeds 0–4 with the other arms'
0–2. That table and `correction_2026-09-30/statistics/exploratory_headline7.csv` are
superseded for paired contrasts.

| Arm (seeds 0–2) | h1 | h2 | h3 | h4 |
| --- | --- | --- | --- | --- |
| A origin lags, equal | +0.24 | +0.27 | +0.44 | +0.50 |
| C origin lags, measured | −0.40 | −0.42 | −0.31 | −0.27 |
| D target lags, measured (3 of its 5 seeds) | −0.01 | −0.04 | −0.01 | −0.05 |
| E fixed [1,2,3,4] | −0.06 | −0.11 | −0.09 | −0.12 |
| **F climate-encoder gradients from h=4 only** | **−0.65** | **−0.69** | **−0.66** | **−0.64** |
| G case-only-utility weights | +0.62 | +0.58 | +0.70 | +0.80 |
| H GradNorm (climate group; labelled deviations) | +0.12 | +0.11 | +0.18 | +0.11 |

**F vs B, matched seeds 0–2**

| h | F MAE | B MAE (seeds 0–2) | F − B | 95% block CI | F better, folds | Wilcoxon p (unadj.) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 15.95 | 16.60 | −0.65 | [−1.26, −0.20] | 5/7 | 0.16 |
| 2 | 19.01 | 19.70 | −0.69 | [−1.47, −0.08] | 6/7 | 0.08 |
| 3 | 22.04 | 22.70 | −0.66 | [−1.51, +0.13] | 6/7 | 0.08 |
| 4 | 24.59 | 25.23 | −0.64 | [−1.70, +0.41] | 6/7 | 0.03 |

- **All 9 folds:** −0.55 / −0.55 / −0.53 / −0.60, with F better in 6–8
  of 9 folds.
- **F's setup.** In F only the h=4 loss updates the climate encoder.
  Every horizon's prediction head remains trained using its ordinary,
  unweighted loss.
- **F is not established:**
  - it was a planned exploratory arm, but its corrected form was seen only
    after the correction;
  - it has 3 seeds;
  - it is unadjusted;
  - its h=3–4 CIs include zero;
  - it was not scored on 2026.
- **C vs D, matched seeds 0–2:** −0.39 / −0.38 / −0.30 / −0.22
  (Wilcoxon p 0.016 / 0.016 / 0.031 / 0.38, unadjusted).
  - The earlier −0.50 to −0.32 compared C's 3 seeds with D's 5 seeds.

**Post-hoc 2×2 factorial** (alignment × weighting; common seeds 0–2; all
9 folds). Source: `correction_2026-09-30/factorial/factorial_contrasts.csv`.

| Contrast | h1 | h4 | 95% block CI at h4 | Folds negative | Holm-20 p (t) |
| --- | --- | --- | --- | --- | --- |
| C−A weighting, origin lags | −0.55 | −0.64 | [−1.02, −0.29] | 6–7/9 | 1.00 |
| D−B weighting, target lags | −0.02 | −0.06 | [−0.24, +0.11] | 5–6/9 | 1.00 |
| B−A alignment, equal weights | −0.19 | −0.38 | [−0.74, −0.03] | 3–4/9 | 1.00 |
| D−C alignment, measured weights | +0.33 | +0.20 | [−0.03, +0.41] | 1–3/9 | ≥ 0.65 |
| (D−B)−(C−A) interaction | +0.52 | +0.58 | [+0.26, +0.91] | 2–3/9 | 1.00 |

- **Descriptively:** measured weighting helps with origin-relative lags
  and not with target-relative lags.
- **But:** no fold-level test survives Holm, and the effects are
  concentrated in 2017 and 2019. For example, B−A at h=4 is −2.63 in 2017
  and between −0.24 and +0.21 in six other years.
- **Seed sensitivity:** with seeds 0–2, D−B is about −0.04, against
  +0.07 with the declared 5 seeds.

**C's status (audited).** C was **a planned, declared exploratory arm**:
it was in the step-10 development matrix and in the 16:44 manifest's
secondary family. Its favourable results were **discovered afterwards**.
It is not, and must not be presented as, the primary hypothesis.

## 5. What the weights, utilities, kernels and gates show

**Measured weights are not nearly uniform.**
- Shuffle-utility weights rise with horizon in all 10 folds, with a
  max/min ratio of 1.7–4.0. For example, fold 6 is
  0.46 / 0.81 / 1.18 / 1.54 (`weights_final/` **(reused)**,
  `correction_2026-09-30/analysis/figures/weights_vs_uniform`).
- The null result can therefore **not** be explained by weights close to
  1.

**Mean-one weights do not equalise gradient norms.** In the 60 sampled
comparisons at identical states, the measured weights raised the raw
climate-encoder gradient norm by 14.7% on average (§6).

**Utilities measure different things**
(`correction_2026-09-30/analysis/figures/signed_utilities`, `correction_2026-09-30/analysis/tables/weights_by_fold.csv`):
- **Shuffle reliance (primary):** how much a trained model's forecasts
  depend on correctly time-aligned weather. It is positive and rises with
  horizon (0.11–2.44 MAE).
- **Incremental benefit (sensitivity):** the relative MAE gain over a
  size-matched case-only model. It is small and changes sign (−0.03 to
  +0.12). Weights built from it (arm G) did worst.
- **Neither establishes a causal climate effect.** The shuffle breaks
  weather–time alignment. It does not isolate a mechanism.

**Kernels** (`correction_2026-09-30/analysis/tables/rainfall_kernel_summary.csv`,
`correction_2026-09-30/analysis/tables/kernel_vs_scan_spearman.csv`):
- Target-relative rainfall kernels centre at 8.0–8.9 weeks (B) and
  8.7–9.6 weeks (D). Their usable mass falls from 0.96 to 0.80 as h
  grows.
- Origin-relative kernels shift one week per horizon by construction.
- Agreement with the training-only cross-correlation scan is weak
  (Spearman 0.15–0.31), and the scan's own rainfall signal is below its
  resolvability threshold.
- **Learned kernels are not evidence of biological delay recovery.**

**Gates and corrections** (`correction_2026-09-30/analysis/tables/gates_and_corrections.csv`):
- The mean gate rises with horizon (0.43 → 0.56).
- The gated climate correction rises from 0.13 to 0.30 (log-mean scale).
- B and D differ by ≤ 0.02 on each.

## 6. Post-hoc diagnostics

**What they measured.** How measured weights change training at fixed
states. They do **not** establish why forecasts did not improve.

**Sources:**
- `correction_2026-09-30/diagnostics_posthoc/`: 60 paired comparisons at identical model and
  optimiser states (folds 1, 6, 9; seeds 0, 1; B and D final states; 5
  batches). The design was fixed before running.
- `correction_2026-09-30/matched_seed/diagnostic_facts.json`: angle aggregation.
- `correction_2026-09-30/matched_seed/clipping_full_training_logs.csv`: clipping over
  complete runs.

**Direction change.**
- **Raw climate-encoder gradient:** mean cosine 0.9863. That corresponds to
  9.5° (9.6° if the cosine is first rounded to 0.986). The mean of the 60
  per-comparison angles is 8.4° (median 7.4°, range 2.9–20.6°).
- **Adam update of the climate encoder:** mean cosine 0.9912, which
  corresponds to 7.6°. The mean angle is 7.1° (median 6.7°, range
  2.8–13.9°).
- The angle of an average cosine is not the average angle; both are
  reported.

**Update size.**
- The relative update difference ‖u_measured − u_equal‖₂ / ‖u_equal‖₂ is
  computed over all climate-encoder parameters, for one Adam step, per
  comparison.
- Mean 13.1%, median 12.6%, range 5.1–24.5% over the 60 comparisons.
- Update norms differ by +2.6% on average; raw-gradient norms by +14.7%.

**Clipping.**
- **None of the 60 sampled comparisons** clipped.
- Over **complete training logs**, clipping did occur: in 6–13 runs per
  arm, on 0.2% of steps on average and at most 2.0% of a run's steps.

**Other parameters.** In the sampled comparisons (no clipping), the other
parameters' updates were identical between weightings.

**Per-horizon encoder gradients.** Partly aligned: cosine with the h=4
gradient is 0.48–0.68.

**Trained models.** B and D differ less from each other than two seeds of
B do:
- predictions: 1.7–2.9 vs 3.5–5.2 cases;
- kernel L1: 0.41 vs 0.72.

**Hypotheses (not tested).**
- The small per-step changes are swamped by seed-to-seed variation.
- Weights far from uniform (as in F) alter the encoder more.
- The optimiser's per-element normalisation and the rare clipping in full
  runs **may** also play a role. The diagnostics neither establish nor
  exclude these mechanisms.

## 7. Selected forecasts and compute

**Selected forecasts** (`correction_2026-09-30/analysis/figures/selected_forecasts`). The rule
was fixed before plotting: the highest-burden district in fold 1's
training data (Colombo), h=4, in 2017 and 2025.
- B and D under-predict the 2017 epidemic peak by about half.
- They are nearly identical to each other.

**Compute** (RTX 4070 Laptop GPU, float32;
`correction_2026-09-30/diagnostics_posthoc/compute_accounting.csv`,
`correction_2026-09-30/diagnostics_posthoc/latency.csv`)
- **Training:** 583 fits, 93.9 GPU-minutes in total. That breaks down as
  pilots 10.9, original v1 runs 52.9, v2 correction 27.5 and the
  development matrix 2.6. Smoke and check runs, a few minutes more, are
  not itemised.
- **Inference**, forward pass only, 20 warm-up and 200 timed runs,
  synchronised: a complete forecast (25 districts × 4 horizons) takes a
  median **2.38 ms** (p95 2.89 ms) on the GPU and 3.66 ms (p95 5.45 ms) on
  the CPU. NB-median post-processing adds 0.13–0.30 ms.

## 8. Historical context kept separate

- **Short paper.** Its numbers used GEE ERA5-Land polygon-mean weather.
  Everything here uses the Open-Meteo `era5_seamless` centroid
  reconstruction. The short paper's neural scores (identity backbone,
  separate models: 17.20 / 20.22 / 22.89 / 25.40) cannot be paired with
  anything above.
- **Persistence** uses case data only, so it is identical under both
  weather files: 16.42 / 20.16 / 24.58 / 28.47.
- **Prior exposure to 2026.** The benchmark's 2026 results for five other
  models were read on 29–30 September, before this extension was
  specified. The original B/D 2026 results were then seen before the
  correction. The dated hold-out addenda in
  `results/benchmark/holdout_declaration.md` record this.

## 9. History: original (routing v1) results, superseded

Under routing v1, the climate correction head `climate_delta` was
mistakenly in the weighted group. This contradicted the step-4
specification, and the documentation described it as intended. The
arms affected were C, D, E, F, G, H and D-2026. A, B, the pilots and the
weights were unaffected: B was verified bit-identical under v2.
The original outputs are preserved in `statistics/`, `holdout_2026/`,
`analysis/` and `predictions/`.

| Result | v1 (original) | v2 (corrected) |
| --- | --- | --- |
| D − B headline, h=1–4 | +0.063 / +0.082 / +0.068 / +0.049 | +0.061 / +0.079 / +0.071 / +0.060 |
| D − B 2026, h=1–4 | +0.01 / +0.22 / +0.10 / +0.32 | +0.18 / +0.28 / +0.28 / +0.50 |
| F headline MAE, h=1–4 | 16.64 / 20.21 / 23.88 / 25.48 (h=1–3 climate heads frozen at 0) | 15.95 / 19.01 / 22.04 / 24.59 |
| C headline MAE | 16.20 / 19.28 / 22.37 / 24.95 | 16.20 / 19.28 / 22.38 / 24.96 |
| H vs B | −0.04 to +0.04 | +0.07 to +0.17 (unmatched seeds); +0.11 to +0.18 (matched seeds 0–2) |
| F − B (h=1–4) | — | first reported −0.70 / −0.67 / −0.67 / −0.63 (B 5 seeds vs F 3); **matched seeds 0–2: −0.65 / −0.69 / −0.66 / −0.64** |
| C − D (h=1–4) | — | first reported −0.50 / −0.47 / −0.40 / −0.32 (D 5 seeds); **matched: −0.39 / −0.38 / −0.30 / −0.22** |

## 10. Contributions supported by the evidence

1. **Method and negative primary result.**
   - A leakage-controlled method: per-fold utilities from inner-validation
     pilots, with weighted gradients routed only to the climate feature
     encoder.
   - Under the original frozen criteria it did **not** reliably improve an
     otherwise identical equal-weight model, even though the weights were
     far from uniform.
   - It was non-inferior at h=1–2.
2. **Measurements of how the weighting changes training.** On 60 sampled
   identical-state comparisons:
   - update direction changed by about 7–8°;
   - the relative update difference averaged 13%;
   - no clipping occurred.

   Trained B and D differ less than two seeds do. These measurements
   characterise the change. They do not establish the cause of the null
   forecasting result.
3. **Exploratory leads, to confirm.**
   - Restricting climate-encoder learning to the longest horizon (F:
     −0.64 to −0.69 vs B on matched seeds).
   - Measured weighting helped only with origin-relative lags.

   Both are post-hoc or exploratory and unadjusted.

## 11. Unfinished work

- **[PENDING] Confirmation on untouched data before 15 November 2026.**
  - Complete June–December 2026 data cannot arrive in time.
  - **Feasible option 1: new Sri Lankan weeks.** Weekly reports published
    after 2026-05-17 (for example June–September 2026), if
    `denguedatahub` has released them. Those weeks have never been loaded
    or scored.
    1. Write a new dated declaration **before** downloading them, naming
       the hypotheses (F vs B, and C vs A, matched seeds) and margins.
    2. Download the new weeks and matching Open-Meteo weather.
    3. Train on data up to 2025 with the frozen code and weights.
    4. Score once, using a block bootstrap over the roughly 12–18 new
       weeks. The interval will be wide.
  - **Feasible option 2: an external dataset.** Another country's
    district-level weekly dengue plus ERA5 series, run with the frozen
    code and hypotheses.
- **[PENDING] Common-support lag arm.** It exists but was not run.
- **[PENDING] Arms not run.** The gate-only-on-encoder variant and
  measured-lag initialisation.
- **[PENDING] Hyperparameter sensitivity.**
- **[PENDING] Architecture diagram.**

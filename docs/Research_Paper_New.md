# Research brief for the full paper: AEGIS-Dengue climate-horizon extension

This is the single source document for drafting the CS3631 full research
paper. It was rewritten on 2026-09-30 after the routing correction and
supersedes all earlier versions. The previous version is kept in
`results/climate_horizon/correction_2026-09-30/doc_snapshots/`.
`docs/full paper.tex` is **not** to be reused.

The detailed evidence lives in three documents:
- [`climate_horizon_results.md`](climate_horizon_results.md): authoritative
  results, each linked to its artefact.
- [`climate_horizon_correction_audit.md`](climate_horizon_correction_audit.md):
  the audit trail.
- [`climate_horizon_implementation.md`](climate_horizon_implementation.md):
  the engineering history.

> **Rules for whoever (or whatever) writes the paper from this file**
>
> 1. **Never invent a number.** Use only numbers written here or in the
>    results document, and cite the artefact. **[PENDING]** means "not
>    measured". Leave it as a visible placeholder.
> 2. **Keep the weather sources apart.** Short-paper neural numbers used
>    GEE ERA5-Land polygon means. Everything new uses the Open-Meteo
>    `era5_seamless` centroid reconstruction. Never pair the two.
> 3. **Keep the analysis tiers apart.** The **primary** result (D vs B) is
>    a null. **Exploratory** and **post-hoc** findings must be labelled as
>    such and never promoted to the main claim.
> 4. **Wording to use:**
>    - Say "empirical 95% coverage" rather than "calibrated".
>    - Say "shuffle reliance" rather than "climate benefit" or "causal
>      effect".
>    - Say "learned delay kernel" rather than "recovered biological
>      delay".
> 5. **Citations.** Only use references verified against their source
>    (§12).
> 6. ACM primary-article template, unless the target venue requires
>    otherwise.

---

## 1. Title and one-sentence summary

**Working title:** *Does Horizon-Informed Gradient Weighting Help a
Learnable Climate-Delay Encoder? A Leakage-Controlled Study of
District-Level Dengue Forecasting in Sri Lanka*

**Summary.** We measured how much each forecast horizon relies on
climate. We used that measurement to weight the gradients reaching only a
learnable climate-delay encoder. Every horizon's prediction head remains
trained using its ordinary, unweighted loss.

Under a leakage-controlled walk-forward protocol, this gave **no reliable
improvement** over an identical equal-weight model, even though the
weights were far from uniform. The protocol was frozen before scoring; a
routing implementation error was corrected after the results had been
inspected, and the affected arms were rerun.

An exploratory variant that updates the climate encoder only from the
longest horizon was the best configuration. It needs confirmation on
untouched data.

## 2. Course requirements (CS3631) and how they are met

| Requirement | How it is met |
| --- | --- |
| Novel or improved DNN framework | A two-branch GRU with a learnable, horizon-aware climate-delay encoder, NB heads, and **climate-only gradient weighting from out-of-sample utilities** (training-strategy novelty). The result is reported honestly as null. |
| Mandatory obvious baseline (GRU/LSTM) | Identity-graph GRU (short paper) and the equal-weight two-branch model B. |
| Allowed model families only | GRU, learnable convolution-style kernels, NB heads. **Chronos-2 zero-shot** appears only as a labelled outside comparator: no teacher outputs, utilities or features. **Fine-tuned foundation-model variants are omitted** from course-facing tables. |
| Novelty categories | **Learning:** out-of-sample, per-horizon utility routing to one input encoder. **Representation:** target-relative delay kernels shared across horizons. **Evaluation:** frozen manifest, capacity-matched donor shuffles, correction trail. |
| Free compute | 583 training fits, 93.9 GPU-minutes on an RTX 4070 Laptop GPU. Inference 2.38 ms per complete forecast. |
| Paper sections | Abstract; Introduction (background, gap, solution, main results, 2–3 contributions); Related Work; Proposed Framework and implementation; Experimental Setup (data, baselines, ablations, comparisons, hyperparameter tuning, compute, other experiments); Discussion; Conclusion; References. |
| Authorship | First the five students, ordered by contribution: Kulathunge K. A. N. H., Angeesa R. P. T., Gunaweera N., Mallawarachchi H. S., Wijesekara M. S. T. (Department of Computer Science and Engineering, University of Moratuwa; {nadilk.23, thilokyaa.23, nethsithg.23, hesandim.23, senilkam.23}@cse.mrt.ac.lk). Then the assigned TAs **[TA names PENDING]**. Last two: Prof Dulani and Dr Sandareka **[full names PENDING; no titles in the author list]**. |
| Two versions | A clean version for submission, and a colour-highlighted version (one colour per member, with margin comments where the implementer and writer differ). **[Contribution and colour map PENDING from the team; do not guess.]** |
| Deadline | Conference submission proof by 15 November 2026. |

## 3. Background and motivation

- Dengue in Sri Lanka follows two monsoon seasons. District-level
  forecasts several weeks ahead would support vector control by the
  National Dengue Control Unit (NDCU) and Public Health Inspectors (PHIs),
  and hospital preparation.
- Weather affects transmission with a delay. A training-only
  cross-correlation scan on our data puts the rainfall peak at about 5–10
  weeks (median 9 in fold 9). But the deseasonalised correlation is weak
  (+0.04), below the scan's resolvability threshold, so it is descriptive
  only.
- Findings from the short paper (GEE weather), which motivate the design:
  1. At one week ahead nothing reliably beats persistence. A learnable
     delay encoder trained at h=1 learned delays unrelated to the scan.
  2. At 3–4 weeks the model beats persistence, and a capacity-matched
     climate shuffle removes about half that skill.
- **Hypothesis.** A delay encoder shared across horizons is dominated by
  short horizons whose loss says little about weather. Weighting its
  gradients by each horizon's measured climate reliance should help it
  learn useful delays.

## 4. The specific technical gap

1. Dengue deep-learning papers often report one-step accuracy without a
   persistence baseline. Horizon-dependent climate value is rarely
   studied inside the model.
2. Climate lags are usually fixed by hand, or learned with one objective.
   **This study investigates** whether choosing *which horizons' losses
   train* a shared learnable delay encoder, using utilities measured
   inside the training years, improves multi-horizon dengue forecasts.
   We have not verified the literature thoroughly enough to claim this
   question has never been studied. State it as the question this paper
   investigates.
3. Multi-task balancing methods such as GradNorm reweight tasks for all
   shared parameters using in-training statistics. Our design differs: it
   routes weighted gradients to **one input modality's encoder** only,
   using an **out-of-sample measurement** made inside the training
   years.

## 5. Method (as implemented and corrected: routing v2)

### 5.1 Architecture (10,028 parameters)

**Case branch.** Nine channels over 12 weeks: log1p cases, day-of-year
sin/cos, observation flags, centroid coordinates, national wave rank and
trailing 52-week cases. Two per-node layers feed a GRU of 32 units. A
fixed identity adjacency means there is no spatial message passing.

**Climate branch.** Seven weather channels over 37 weeks: rainfall,
rainy-day fraction, temperature, diurnal range, dew point, relative
humidity and wind. They pass through `HorizonLagEncoder`:
- per district i and variable k, a kernel over delay d = 0…29 (target
  mode);
- the kernel is a softmax mixture of 6 Gaussian bumps, with a district
  embedding × feature projection;
- for horizon h, it reads x[t+h−d] = x[t−k] only where d ≥ h and
  k ≤ 25, renormalised over that usable support;
- the usable mass is reported, and empty support gives zero.

Each horizon's smoothed stream then passes through a shared climate GRU
(16 units) and a climate head.

The **origin-relative mode** (a kernel over k, shared by all horizons)
has identical parameters.

**Fusion and likelihood.** For each horizon:
- δ_h = δ_case,h + σ(gate_h)·δ_climate,h;
- log μ_h = log(1 + y_t) + clamp(δ_h, ±10);
- α_h = softplus(case head) + 10⁻⁴.

This is NB2. The point forecast is the NB median, and μ is never passed
through expm1. The gate reads the case state, a district bias and a
horizon bias.

### 5.2 Loss and climate-only gradient routing

The baseline loss is the pooled masked NB negative log-likelihood,
Σ_h N_h / Σ_h D_h. N_h is horizon h's summed NLL over observed cells and
D_h is the count of those cells. With L_h = N_h/D_h and
a_h = D_h / Σ D_h:

- **L_equal** = Σ a_h L_h (the baseline loss);
- **L_climate** = Σ a_h w_h L_h / Σ a_h w_h.

From one forward pass:
- ∇L_climate is assigned to **the climate feature encoder only**: the lag
  kernels `encoder.*` and the climate GRU `climate_gru.*`;
- ∇L_equal is assigned to **every other parameter**: the case branch, the
  gate, **all prediction heads (case, climate, dispersion)**. Every
  horizon's prediction head remains trained using its ordinary,
  unweighted loss.

Then one global-norm clip (1.0) and one Adam step follow. The encoder
uses lr 2×10⁻² with no weight decay; everything else uses 3×10⁻³ and
weight decay 10⁻⁴. Weights are detached constants.

Verified numerically:
- w = 1 reproduces the baseline gradients and update exactly;
- non-uniform w changes only the climate group's raw gradients;
- routed gradients equal the explicit per-horizon combination.

**Mean-one weights do not guarantee equal gradient norms.** In 60 sampled
identical-state comparisons, the measured weights raised the raw encoder
gradient norm by 14.7% on average.

### 5.3 Measuring horizon utility (inside the training years only)

For each outer fold:
1. **Inner blocks.** Three inner blocks, each with its own pilot
   training, early-stopping and utility-scoring year. Preprocessing is
   fitted on pilot training only.
2. **Pilots.** A frozen equal-weight pilot (2 seeds) scores its utility
   year with real climate and with 5 predeclared donor perturbations. Each
   perturbation replaces a district's whole 37 × 7 weather window with the
   same district's window from another year, same season (±2 weeks), from
   pilot-training history.
3. **Primary utility (shuffle reliance).** g_h = MAE_shuffled − MAE_real.
   Aggregate seeds within a block, then blocks equally, then
   u_h = max(g_h, 0).
4. **Weights.** w_h = 0.05 + 0.95 · 4u_h / Σu, or uniform if Σu = 0.
5. **Sensitivity utility.** The relative gain over a size-matched
   case-only pilot, in a separate arm.

**Shuffle reliance is not incremental benefit, and neither is a causal
effect.**

### 5.4 Implementation details

- **Training:** PyTorch 2.11 (CUDA 12.8), batch 64, at most 150 epochs,
  early stopping (patience 15) on the **unweighted** validation NLL.
- **Origins:** every week with 37 weeks of history.
- **Splits:** by each target's own date.
- **Code:**
  - `src/models/{climate_horizon,horizon_lag_encoder}.py`
  - `src/training/{climate_horizon,climate_weighting,climate_horizon_pilots,climate_horizon_weights}.py`
  - `scripts/training/{41,42,45}` and `scripts/evaluation/{47,49,56,57}`

## 6. Dataset

- **Cases.** Weekly confirmed cases from `denguedatahub` (Talagala 2026).
  Kalmunai is merged into Ampara, giving **25 districts** and **1,012
  weeks** (2006-12-23 to 2026-05-17).
- **Weather (current).** Open-Meteo `era5_seamless`: ERA5-Land
  temperature, dew point and humidity; ERA5 rain and wind. Sampled at
  **GADM district centroids**, daily, aggregated to reporting weeks.
  Hashes are in `results/benchmark/data_manifest.json`. The short paper's
  GEE polygon-mean extraction is no longer available.
- **Missing values.** Targets are never imputed. Inputs are imputed from
  training-year district × month climatology and flagged.

## 7. Evaluation protocol

- **Folds.** 9 expanding walk-forward folds (test years 2017–2025).
  - Headline = 7 folds; 2020–21 excluded for COVID reporting disruption
    and reported separately.
  - The validation year before each test year selects checkpoints.
- **2026 hold-out.** January–May, 19 weeks.
  - Benchmark 2026 results for other models were read before this
    extension was specified, and the original B/D 2026 results were seen
    before the correction.
  - So 2026 is a frozen-specification check, **not** a blind test.
- **Metrics.** MAE (primary), RMSE, peak MAE (district 90th percentile of
  pre-test history), WIS, empirical 95% coverage, and skill = 1 − MAE /
  persistence MAE on the same cells.
- **Estimands.** **Seed-mean** (the mean of per-seed metrics) and
  **ensemble** (the metric of the seed-averaged forecast), reported
  separately.
- **Uncertainty.** A 4-week block bootstrap of contiguous origins, all
  districts together, within folds, averaging folds equally. Block
  lengths of 8 and 13 weeks were run as a post-hoc sensitivity. The
  bootstrap does not resolve dependence from overlapping training
  histories.
- **Tests.** Fold-level paired t and Wilcoxon. Holm across the 4 horizons
  for the primary comparison.
- **Frozen decision criteria:**
  - improvement: D − B ≤ −0.30 at h=3 or 4, with Holm p < 0.05 and a CI
    below 0;
  - non-inferiority at h=1–2: CI upper bound ≤ +0.30;
  - otherwise "no measurable difference".
- **Original specification vs correction:**
  - The **original specification** was frozen at 16:44 (manifest SHA-256
    `f4acdf1c…`) before any outer test year was scored. Arms, seeds,
    folds, utilities and criteria were fixed then.
  - **After** those results were inspected, a routing implementation error
    was found and corrected.
  - The affected arms were rerun under a correction manifest (18:58,
    SHA-256 `5434845f…`), with the same criteria, seeds and folds.
  - The corrected results are **corrected evaluations on previously
    observed data**, not a blind or newly preregistered test.

## 8. Baselines and comparators

- **Main controlled baseline.** B, the identical model with equal
  weights.
- **Mandatory course baseline.** The GRU lineage.
- **Stored forecasts**, compared directly on identical cells:
  persistence (recomputed), `gru_v2_nb` (separate NB GRU),
  `nb_shared_v2` (one NB trunk for 8 horizons) and `lgbm_v2`.
  - Their protocols differ: preprocessing fitted through the validation
    year, and hand-lagged climate or tabular inputs.
  - They are **not controlled ablations**.
  - The files carry no embedded data fingerprint. The data files still
    match the manifest hashes.
- **Outside comparator.** Chronos-2 joint, zero-shot only (§2).

## 9. Results (corrected; measured)

Sources are in `climate_horizon_results.md`.

### 9.1 Primary: D (measured weights) vs B (equal weights), headline folds, 5 seeds

| h | B | D | D − B | Holm p | 95% block CI | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 16.64 | 16.70 | +0.061 | 0.72 | [−0.030, +0.170] | non-inferior |
| 2 | 19.67 | 19.75 | +0.079 | 0.44 | [−0.034, +0.208] | non-inferior |
| 3 | 22.71 | 22.78 | +0.071 | 0.72 | [−0.067, +0.231] | no measurable difference |
| 4 | 25.22 | 25.28 | +0.060 | 0.72 | [−0.096, +0.223] | no measurable difference |

- **Ensemble estimand:** +0.057 to +0.061.
- **All 9 folds:** +0.049 to +0.072.
- **2026:** D − B = +0.18 / +0.28 / +0.28 / +0.50.
  - The h=4 interval [+0.03, +1.00] is a **pointwise, unadjusted** 95%
    block-bootstrap interval over 19 weeks (only 5 four-week blocks).
  - 2026 outcomes had been seen before, for other models and for the v1
    runs.
  - It is **not** confirmatory. It is only consistent in sign with D not
    beating B.
- **Reading.** **Measured weighting gave no reliable improvement.**
  - The weights were far from uniform (max/min 1.7–4.0 per fold), so the
    null is **not** because the weights were nearly 1.
  - B and D both beat persistence at h=3–4 (h=4: −3.2 MAE), but that is
    shared by both and shows nothing about weighting.

### 9.2 Accuracy table (headline, seed-mean)

| Method | h=1 | h=2 | h=3 | h=4 | Peak h4 | WIS h4 |
| --- | --- | --- | --- | --- | --- | --- |
| B | 16.64 | 19.67 | 22.71 | 25.22 | 42.99 | 15.24 |
| D | 16.70 | 19.75 | 22.78 | 25.28 | 42.81 | 15.31 |
| F (exploratory) | 15.95 | 19.01 | 22.04 | 24.59 | 41.71 | 14.60 |
| Persistence | 16.42 | 20.16 | 24.58 | 28.47 | 47.98 | — |
| `gru_v2_nb` (stored) | 15.94 | 19.26 | 22.46 | 25.07 | 42.60 | — |
| `nb_shared_v2` (stored) | 16.63 | 19.83 | 23.15 | 25.85 | 43.82 | — |
| `lgbm_v2` (stored) | 16.99 | 21.05 | 24.30 | 27.16 | 46.42 | — |
| Chronos-2 joint, zero-shot (outside comparator) | 15.96 | 19.07 | 22.29 | 25.11 | 43.74 | — |

The empirical 95% coverage of the NB intervals is 0.93–0.96. That is
coverage at one level, not a calibration claim.

### 9.3 Planned exploratory arms (matched seeds 0–2 for both arms; vs B; unadjusted)

**Arms that actually ran:** A, C, E, F, G, H.

Every contrast uses seeds 0, 1 and 2 for **both** arms, on identical cells.
Source: `results/climate_horizon/correction_2026-09-30/matched_seed/`.

| Arm | Δ MAE vs B (seeds 0–2), h=1–4 |
| --- | --- |
| A origin lags, equal | +0.24 / +0.27 / +0.44 / +0.50 |
| C origin lags, measured | −0.40 / −0.42 / −0.31 / −0.27 |
| E fixed [1,2,3,4] | −0.06 / −0.11 / −0.09 / −0.12 |
| F climate-encoder gradients from h=4 only | −0.65 / −0.69 / −0.66 / −0.64 |
| G case-only-utility weights | +0.62 / +0.58 / +0.70 / +0.80 |
| H GradNorm on the climate group | +0.12 / +0.11 / +0.18 / +0.11 |

**Not run:**
- the common-support lag arm;
- gate-only-on-encoder;
- measured-lag initialisation;
- hyperparameter sensitivity.

**F vs B, matched seeds 0–2**

| h | F | B (seeds 0–2) | F − B | 95% block CI | Wilcoxon p (unadj.) | Folds |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 15.95 | 16.60 | −0.65 | [−1.26, −0.20] | 0.16 | 5/7 |
| 2 | 19.01 | 19.70 | −0.69 | [−1.47, −0.08] | 0.08 | 6/7 |
| 3 | 22.04 | 22.70 | −0.66 | [−1.51, +0.13] | 0.08 | 6/7 |
| 4 | 24.59 | 25.23 | −0.64 | [−1.70, +0.41] | 0.03 | 6/7 |

- F is exploratory, and its corrected form was seen only after the
  correction.
- The earlier figures (−0.70 to −0.63) compared F's 3 seeds with B's 5.

**C vs D, matched seeds:** −0.39 / −0.38 / −0.30 / −0.22 (unadjusted).
**C was a planned, declared exploratory arm.** Its advantage was found
afterwards. It is **not** the primary hypothesis.

### 9.4 Post-hoc 2×2 factorial (seeds 0–2, all 9 folds)

| Contrast | h=1 | h=4 |
| --- | --- | --- |
| C−A, weighting with origin lags | −0.55 | −0.64 (CI [−1.02, −0.29]) |
| D−B, weighting with target lags | −0.02 | −0.06 |
| B−A, alignment with equal weights | −0.19 | −0.38 |
| D−C, alignment with measured weights | +0.33 | +0.20 |
| (D−B)−(C−A), interaction | +0.52 | +0.58 (CI [+0.26, +0.91]) |

- **None** of the 20 fold-level tests survives Holm.
- The effects are concentrated in 2017 and 2019.
- The factorial is descriptive only.

### 9.5 What the post-hoc diagnostics measured

The diagnostics characterise how the weighting changes training. They do
not establish why forecasts did not improve. They cover 60 sampled pairs
at identical model and optimiser states.

**Direction change**
- **Raw climate-encoder gradients:** mean cosine 0.986, about 9.5°. The
  mean per-pair angle is 8.4°.
- **Adam updates:** mean cosine 0.991, about 7.6°. The mean per-pair angle
  is 7.1°.

**Update size.** The relative update difference ‖u_measured − u_equal‖ /
‖u_equal‖ averaged 13.1% (range 5–24%) per step.

**Clipping**
- None of the 60 sampled pairs clipped.
- In complete training logs, clipping occurred rarely: at most 2.0% of a
  run's steps.

**Trained models**
- B and D differ less than two seeds of B do.
- Gates rise with horizon (0.43 → 0.56).
- Target-relative rainfall kernels centre at about 8–10 weeks. They agree
  only weakly with the training-only scan (Spearman 0.15–0.31). **That is
  not delay recovery.**

**Hypotheses (not tested)**
- Per-step changes are swamped by seed variation.
- Only weights far from uniform (as in F) matter.
- Optimiser normalisation or rare clipping may contribute. The data
  neither establish nor exclude this.

## 10. Contributions (matching the corrected evidence)

1. **A leakage-controlled climate-only gradient-weighting method.**
   - Out-of-sample per-horizon utilities, routed only to a learnable
     delay encoder.
   - Tested under frozen original criteria. It gave a **null primary
     result**: no reliable improvement, and non-inferior at h=1–2.
2. **Measurements of how the weighting changes training.**
   - About 7–8° of update rotation and a 13% mean relative update
     difference on sampled steps.
   - Trained models differ less than two seeds do.
   - These describe the change. They are not a causal explanation of the
     null.
3. **An exploratory lead** (unadjusted, post-hoc): updating the climate
   encoder only from the longest horizon (F: −0.64 to −0.69 MAE vs B on
   matched seeds), and a weighting × alignment interaction. It needs
   confirmation on untouched data.

## 11. Limitations to state

- **Low power.** 7 headline folds share training histories.
- **Weather data.** Centroid-based reconstructed weather.
- **Unseen data.** All data up to 2026-05-17 have been inspected. The
  corrected results are not blind.
- **Confirmation [PENDING].** Before the 15 November 2026 deadline, the
  only untouched Sri Lankan data would be weekly reports published after
  2026-05-17 (for example June–September 2026, if released). They must be
  scored once, under a new dated declaration written before download.
  The alternative is an external dataset. A complete June–December 2026
  evaluation cannot be ready by the deadline.
- **Scope.** 25 districts in one country; reporting delays are not
  modelled.
- **Robustness.** Exploratory effects depend on a few years and on seed
  subsets.

## 12. Related work and references

**Team bibliography** `docs/references_long.bib` (compiled earlier by the
team; check each entry before final submission): Cho 2014 (GRU); Kipf &
Welling 2017; Wu 2019 (Graph WaveNet); Kingma & Ba 2015; Loshchilov &
Hutter 2019; Muñoz-Sabater 2021 (ERA5-Land); Funk 2015 (CHIRPS); Talagala
2026 (denguedatahub); Weng et al. 2024; GulMohamed et al. 2026; Liu et al.
2025; Johansson et al. 2019; Bracher et al. 2021 (WIS); Colón-González et
al. 2021; Mosqlimate 2026 sprint; the 2025 global ensemble study; Ke 2017
(LightGBM); Ansari 2024/2025 (Chronos); Hu 2022 (LoRA); Vovk 2005.

**Verified in this project:** Z. Chen, V. Badrinarayanan, C.-Y. Lee,
A. Rabinovich, "GradNorm: Gradient Normalization for Adaptive Loss
Balancing in Deep Multitask Networks", ICML 2018, PMLR 80
(https://proceedings.mlr.press/v80/chen18a/chen18a.pdf).

**Do not cite until verified:** Kendall et al. 2018 (uncertainty
weighting); Yu et al. 2020 (PCGrad); Cloud et al. 2024 (gradient routing);
Gasparrini et al. 2010 (DLNM); Salinas et al. 2020 (DeepAR); Lim et al.
2021 (TFT).

## 13. Figures and tables (corrected)

All in `results/climate_horizon/correction_2026-09-30/analysis/`:
- **Figures:**
  - `weights_vs_uniform`, `signed_utilities`
  - `primary_per_fold_differences`
  - `rainfall_delay_kernels`, `gates_and_corrections`
  - `selected_forecasts` (Colombo, chosen by a fixed rule)
- **Tables** (CSV and LaTeX):
  - `primary_D_vs_B`, `primary_bootstrap_estimands`
  - `metrics_{headline,all9,covid}_{seed_mean,ensemble}`
  - `per_fold_B_D`, `holdout_2026_metrics`, `holdout_2026_bootstrap`
  - `controls_vs_B`, `weights_by_fold`
  - `rainfall_kernel_summary`, `kernel_vs_scan_spearman`,
    `gates_and_corrections`
  - `reference_provenance`, `coverage`, `compute`
- **Also:**
  - factorial: `correction_2026-09-30/factorial/`
  - diagnostics and latency: `correction_2026-09-30/diagnostics_posthoc/`
- **[PENDING]** architecture diagram.

## 14. History (for the Discussion; do not hide)

- **Original run (routing v1).** The climate correction head was
  mistakenly in the weighted group. This contradicted the step-4
  specification.
- **Arm F under v1.** It therefore had no climate at h=1–3 under v1.
- **Correction.** Audit, fix, verified-equivalent reuse of A, B, the
  pilots and the weights, and reruns of C–H and D-2026.
- **Effect.** The primary null is unchanged (v1 D − B +0.05 to +0.08).
  Corrected F improved substantially.
- **Later reanalysis.** The first corrected F−B and C−D contrasts compared
  unequal seed sets. They were recomputed on matched seeds 0–2 (§9.3).
- **Where it is recorded:** `climate_horizon_results.md` §9 and the
  correction audit.

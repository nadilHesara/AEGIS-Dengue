# Research brief for the full paper: AEGIS-Dengue extension

Source document for drafting the CS3631 full research paper. It holds
everything the paper needs: problem, gap, method, data, baselines,
experiments, results so far, and constraints. It replaces
`docs/full paper.tex` as the starting point. That draft is **not** to be
reused.

> **Rules for whoever (or whatever) writes the paper from this file**
>
> 1. **Never invent a number.** Use only numbers written in this file, and
>    keep their provenance (short paper vs current benchmark). Anything
>    marked **[PENDING]** has not been measured yet. Leave it as a visible
>    placeholder.
> 2. Numbers from the short paper (GEE ERA5-Land climate) and numbers from
>    the current benchmark (Open-Meteo reconstruction) come from **different
>    climate files**. Never put them in the same table as if comparable.
> 3. Report negative and null results as they are. The project's
>    credibility rests on this (see §9).
> 4. Follow the CS3631 structure in §2 and the ACM primary-article template
>    unless the target conference requires otherwise.
> 5. References marked *(verify)* must be checked against the original
>    source before citation.

---

## 1. Title and one-sentence summary

**Working title:** *Learning Climate Delays from Informative Forecast
Horizons for District-Level Dengue Forecasting in Sri Lanka*

Alternative: *Horizon-Informed Gradient Routing for Learnable Climate-Delay
Encoders in Multi-Horizon Dengue Forecasting*

**Outcome (measured 2026-09-30):** the proposed horizon-informed weighting did
**not** improve on the identical equal-weight model. D − B was +0.05 to +0.08
MAE at h = 1–4, all 95% block-bootstrap CIs include 0, it was non-inferior
at h = 1–2, and 2026 was consistent. The paper should therefore be framed
as a careful negative/diagnostic study (see docs/climate_horizon_results.md).

**Original summary:** A learnable per-district climate-delay encoder cannot learn from
one-week-ahead forecasting, because the most recent case count dominates the
signal. We therefore measure, inside the training data, how much climate
improves each forecast horizon, and use those measurements to route
gradients to the shared climate encoder. Horizons where climate matters
teach the delays; every horizon still trains the case-history branch and
the prediction heads.

---

## 2. Course requirements this paper must satisfy (CS3631)

| Requirement | How this paper meets it |
| --- | --- |
| Novel or improved DNN framework for a real problem | Two-branch GRU with a learnable climate-delay encoder, trained with **horizon-informed gradient routing** (§5). |
| Mandatory obvious baseline (sequential/time series → LSTM/GRU) | GRU (identity-graph GCN-GRU) from the short paper, and its multi-horizon NB version (§7). |
| Allowed model families only (no LLM, agentic, prompt-based or foundation-model fine-tuning) | Proposed model = GRU + small learnable convolution-style encoder + NB head. The benchmark contains Chronos-2 (a pretrained time-series model). If it appears at all, it is an **external comparator only**, never part of the method. Safer: omit its fine-tuned variants. |
| Novelty categories (§8 of the brief) | **Learning:** novel training strategy (measured per-horizon gradient routing). **Architecture:** two-branch case/climate fusion with a shared delay encoder. **Representation:** interpretable per-district climate-delay kernels. |
| Free compute | Every fit is ~1–3 s on an RTX 4070 laptop GPU. The full study is minutes to tens of minutes (§8.6). Runs on Colab/Kaggle. |
| Paper sections | Abstract; Introduction (background, specific gap, solution overview, main results, 2–3 contributions); Related Work; Proposed Framework + implementation details; Experimental Setup (dataset overview, baselines, experiments: ablations, comparison with existing methods, hyperparameter tuning, computational analysis, other supporting experiments); Discussion; Conclusion; References. |
| Authorship | First the five students (order by contribution): Kulathunge K. A. N. H., Angeesa R. P. T., Gunaweera N., Mallawarachchi H. S., Wijesekara M. S. T. (Department of Computer Science and Engineering, University of Moratuwa; {nadilk.23, thilokyaa.23, nethsithg.23, hesandim.23, senilkam.23}@cse.mrt.ac.lk). Then the assigned TAs **[names PENDING]**. Last two: Prof Dulani and Dr Sandareka **[full names PENDING; no titles in the author list]**. |
| Two versions | A clean version for submission, and a colour-highlighted version (one colour per member, with margin comments when the implementer and the writer differ). **[Contribution map PENDING from the team.]** |
| Deadlines in the brief | Complete paper + LaTeX zip + code (Phase 3); conference submission proof by 15 Nov 2026. |

---

## 3. Background and motivation

- Dengue is endemic in Sri Lanka, with two monsoon-driven transmission
  seasons. The National Dengue Control Unit (NDCU) and Public Health
  Inspectors (PHIs) act at district level: vector control, fogging,
  community clean-ups. Hospitals need lead time to prepare staff and beds.
  A useful forecast gives **several weeks** of warning at **district**
  resolution.
- Weather drives transmission with a delay: rain → breeding sites → larval
  development → adult mosquitoes → infection → incubation → reporting.
  Cross-correlation on our data places the rainfall-to-dengue delay at
  **about 5–10 weeks across districts (median 8)**. Caveat: the
  deseasonalised peak correlation is weak, +0.036, below the scan's own 0.10
  resolvability threshold. Liu et al. (2025) report 7–9 weeks for Sri Lanka
  with a different model.
  A re-run of the scan on training data only, on the current climate file
  (fold 9, history to 2023), gives a median deseasonalised rainfall peak of
  9 weeks (district range 5–10, median r = +0.041, not resolvable). Treat it
  as descriptive, not as a known biological delay.
- Two findings from our own prior work set up the gap:
  1. **At one week ahead, nothing reliably beats persistence**, and climate
     carries almost no signal. Removing all climate channels changes MAE by
     about +0.01 on recent normal folds. A learnable delay encoder trained at
     h = 1 learns delays unrelated to the measured ones (r = −0.16) and
     scores worse than having no lag treatment at all.
  2. **At 3–4 weeks ahead the model beats persistence, and about half of
     that skill is climate content.** A capacity-matched climate-shuffle
     ablation shows this (details in §9).
- Together these point to a mechanism: **how much delay information the
  gradient carries depends on the horizon.** A delay encoder shared across
  horizons and trained with equal weight on all of them is dominated by
  horizons whose loss says nothing about climate.

## 4. The specific technical gap

1. Most dengue deep-learning work reports one-step-ahead accuracy and
   compares against ARIMA/RF/LSTM **without a persistence baseline** (e.g.
   Weng et al. 2024; GulMohamed et al. 2026 for Sri Lanka). Horizon-dependent
   climate value is not studied inside the model.
2. Climate lags are usually **fixed by hand** (lag features, DLNM crossbasis)
   or learned end to end with one objective. No method we found **decides
   which training signal is allowed to shape the lag structure**.
3. Multi-task loss balancing (GradNorm, uncertainty weighting, PCGrad)
   balances tasks for **all shared parameters**, using training-time
   gradient or loss statistics. Our setting needs something different:
   weights that are (a) applied **only to one sub-module** (the climate
   encoder), (b) derived from an **out-of-sample measurement of the input
   modality's value** per task, and (c) estimated **without touching
   validation or test years**.

## 5. Proposed method

### 5.1 Problem setup

- Districts i = 1…25, weekly periods t. Target y_{i,t} = confirmed cases
  (0–2631, median 10).
- Forecast origin t. Predict y_{i,t+h} for h ∈ H = {1, 2, 3, 4} from
  information available at t.
- Inputs: case-history channels c_{i,t} and climate channels x_{i,t} ∈ R^7
  (rainfall daily mean, rainy-day fraction, 2 m temperature, diurnal range,
  dew point, relative humidity, wind speed).

### 5.2 Architecture (two branches, shared encoder, per-horizon heads)

**Climate branch.** A learnable delay encoder smooths each district's
climate history with a district- and feature-specific kernel over reach
R = 26 weeks:

  x̃_{i,k,t} = Σ_{τ=0}^{R−1} w_{i,k}(τ) · x_{i,k,t−τ}

  w_{i,k}(τ) = Σ_{b=1}^{6} a_{i,k,b} · φ_b(τ),
  φ_b(τ) ∝ exp(−(τ − c_b)² / 2σ_b²),
  a_{i,k,·} = softmax(e_i^T P_k)

The bumps φ_b have learnable centres c_b (initialised at 0, 4, 8, 13, 18, 24
weeks) and learnable widths σ_b. e_i ∈ R^8 is a district embedding and
P_k ∈ R^{8×6} a per-feature projection. Each kernel is non-negative, sums
to 1, is smooth and strictly causal (τ ≥ 0). The encoder has about 550
parameters. Its centre of mass Σ_τ τ·w_{i,k}(τ) reads directly as "district
i responds to feature k after about N weeks". The last 12 smoothed steps go
into a small GRU (hidden 16), giving h^clim_i.

**Case branch.** Non-climate channels over the last 12 weeks: log1p cases,
day-of-year sin/cos, observation flags, centroid coordinates, national
wave rank, trailing-52-week cumulative cases. A per-node projection and a
GRU (hidden 32), shared across districts, give h^case_i. No graph
convolution is used, because every graph we tested hurt accuracy (§9).
Hand-made rolling climate means are **not** given to either branch, so the
encoder is the only path for delayed climate.

**Heads and gated fusion (Negative Binomial, anchored).** For each
horizon h, the case branch outputs a log-mean correction δ^case_{i,h}. The
climate branch outputs δ^clim_{i,h}. A gate
g_{i,h} = σ(v_h·h^case_i + b_{i,h} + c_h) reads case state, district and
horizon; it is not constrained to increase with h. They combine in NB
log-mean space:

  δ_{i,h} = δ^case_{i,h} + g_{i,h} · δ^clim_{i,h},
  log μ_{i,h} = log(1 + y_{i,t}) + clamp(δ_{i,h}, −10, 10),
  α_{i,h} = softplus(case head) + 10⁻⁴   (dispersion is case-based),

with Y ~ NB2(μ, α), Var = μ + αμ². All heads start at zero, so training
starts at μ = y_t + 1 and gate 0.5. The point forecast is the NB median; μ
is a count mean and is never passed through expm1. Seven NB quantiles
(0.025–0.975) give the predictive distribution. The gate and δ^clim are
saved for every forecast.

**Climate-only gradient weighting.** Let N_h be horizon h's summed NB NLL
over observed cells and D_h their count, so the baseline loss is the pooled
Σ N_h / Σ D_h. With L_h = N_h / D_h and a_h = D_h / Σ D_h:

  L_equal   = Σ_h a_h L_h                 (= the baseline loss)
  L_climate = Σ_h a_h w_h L_h / Σ_h a_h w_h

From one forward pass, ∇L_climate is assigned to the climate parameters
(lag encoder, climate GRU, climate head) and ∇L_equal to all others (case
GRU, case heads, gate). Global-norm clipping and one Adam step follow,
exactly as in the baseline. With w = 1 this reproduces the baseline's
gradients and update exactly (verified numerically). Mean-one weights do
not preserve gradient norms: on a fold-9 smoke run the climate gradient
was about 17% larger and at cosine 0.95 to the equal-weight direction.

**Target-relative lags (used by both the equal-weight and the weighted
model).** For origin t, horizon h and climate observed at t − k, the delay
to the target is d = h + k. The kernel w_{i,f}(d) is defined over
d = 0 … D (D = 29), and horizon h uses
x̃^{(h)}_{i,f,t} = Σ_{d=h}^{h+25} ŵ^{(h)}_{i,f}(d) · x_{i,f,t+h−d}, where
ŵ^{(h)} is w renormalised over the usable delays h ≤ d ≤ min(D, h + 25).
The kernel mass on usable support ("available mass") is reported, and
empty support gives zero, not NaN. The origin-relative kernel (over k,
shared by all h, as in the short paper) is an ablation with identical
parameters (10,028 in total for either mode). A common-support variant
(d ∈ [4, 26] for all h) is a sensitivity check. Each horizon has its own
smoothed climate stream through the same climate GRU.

### 5.3 Loss

Masked NB negative log-likelihood, pooled over horizons and normalised by
the number of observed cells N: L = Σ_h L_h,
L_h = (1/N) Σ_{(i,t) observed at h} −log NB(y_{i,t+h} | μ_{i,h}, α_{i,h}).
Missing targets are masked, never imputed. The loss is identical for the
baseline and the proposed model. Only the routing of ∇L_h into θ_clim
differs.

### 5.4 Measuring each horizon's climate value (inside the training data)

For each outer walk-forward fold, using only periods up to the end of its
training set:

The rule below was declared on 2026-09-30 before any pilot was run.

1. Take K = 3 consecutive 52-week **inner validation blocks** ending at the
   training cut-off. For block k, train on data before the year preceding
   it, early-stop on that preceding year, and fit the scaler only up to
   that year.
2. **Primary utility: shuffle gain from a frozen equal-weight pilot.** Train
   the two-branch model with uniform weights [1,1,1,1] (2 seeds) and freeze
   it. Score block k with real climate and with climate **time-shuffled
   within district** (5 permutations; no retraining):
   g_h = MAE^shuffled_{h} − MAE^real_{h} (absolute, averaged over seeds within
   block, then over blocks; the relative form declared earlier was replaced
   on 2026-09-30, before any weighted model was trained).
3. **Sensitivity utility: relative improvement over a case-only pilot**
   (same blocks and seeds, with no climate branch):
   u_h = mean (MAE^case-only_{h} − MAE^real_{h}) / MAE^case-only_{h}.
   This is used only in a sensitivity arm.
4. **Weights:** w̃_h = max(g_h, 0), then
   **w_h = 0.95 · |H| · w̃_h / Σ_h w̃_h + 0.05**. The mean weight is 1, so
   the encoder's total gradient budget equals the baseline's, and every
   horizon keeps at least 0.05. If every gain is ≤ 0, use w_h = 1.
   Benchmark-derived weights (computed on test years) are never used.

The weights are recomputed for every fold, so they follow the data as the
training history grows. They are reported per fold as a result in their
own right: *how much is climate worth at each horizon, as measured
before the test year?*

### 5.5 Implementation details

PyTorch 2.11 (CUDA 12.8). Adam: lr 3×10⁻³, weight decay 10⁻⁴. The encoder
uses lr 2×10⁻² with no weight decay (its gradient is naturally small).
Batch 64, at most 150 epochs, early stopping (patience 15) on the
**unweighted** validation NB NLL for every arm. Gradient-norm clipping at
1.0. Dropout 0.2. Input window 37 weeks (12 model steps + 26 − 1 for the
kernel reach). Per-fold imputation (district × month climatology) and
z-scoring are fitted only on each fold's **training** periods (stricter than
the benchmark, which also used the validation year). All arms use the same
eligible origins: every week with 37 weeks of history. Seeds: 5 for the
main pair, 3 for ablations. Code:
`src/models/climate_horizon.py`, `src/training/climate_horizon.py`,
`src/models/horizon_lag_encoder.py`, `src/training/climate_weighting.py`,
`scripts/training/41`/`42`/`45` (pilots, weights, sweep), `scripts/evaluation/39`/`47`/`49`
(evaluator, statistics, analysis). Engineering plan: `docs/climate_horizon_implementation.md`.

---

## 6. Dataset (overview)

| Item | Detail |
| --- | --- |
| Cases | Weekly confirmed dengue cases from the `denguedatahub` R package (Talagala 2026). The source has 26 reporting areas; Kalmunai is merged into Ampara, giving **25 districts**. **1,012 reporting periods**, 2006-12-23 to 2026-05-17. The chronological key is the reporting calendar's `period_id`, never the source year/week labels. |
| Climate (current) | ERA5 reanalysis via Open-Meteo `era5_seamless` (ERA5-Land temperature, dew point and humidity; ERA5 rain and wind), **one point per district at the GADM centroid**, daily, aggregated to the reporting periods. The GEE ERA5-Land polygon-mean extraction used for the short paper is no longer available. The reconstruction is recorded with hashes in `results/benchmark/data_manifest.json`. **The paper must describe this source accurately.** |
| Rainfall check | CHIRPS was used in the short paper to validate the ERA5 rainfall extraction. It is not a model input. |
| Features | log1p cases; 7 climate channels; day-of-year sin/cos; weather- and case-observed flags; centroid lat/lon; outbreak history (national wave rank, trailing 52-week cumulative log cases). |
| Missingness | Targets are never imputed; they are masked in the loss and metrics. Inputs are imputed per fold from pre-test climatology, with flags. |
| Graph (not used by the proposed model) | GADM queen contiguity, 57 edges (degree 1–9). |

---

## 7. Baselines and comparison methods

**Main controlled baseline: uniform routing** (w_h = 1 for all h). Same
architecture, encoder, head, loss, data, seeds and stopping rule. This
isolates the single contribution.

**Mandatory course baseline: GRU.** The identity-graph GCN-GRU of the short
paper (MSE on the anchored log residual) and its NB multi-horizon
successors.

**Existing methods, all scored on identical cells** (current benchmark;
reused, not retrained):

| Method | Description |
| --- | --- |
| Persistence | ŷ_{t+h} = y_t, rescored for each horizon |
| Seasonal naive | y from 52 weeks earlier |
| `gru_v1` | short-paper GRU (v1 features, hand-lagged climate means), one model per horizon |
| `gru_v2` / `gru_v2_nb` | + outbreak history; MSE / NB head, one model per horizon |
| `nb_shared_v2` | one GRU trunk, NB head per horizon, hand-lagged climate means |
| `lgbm_v2` | LightGBM with case lags, climate lags to 24 weeks and neighbour signals |
| (optional) `chronos2_joint` | zero-shot pretrained forecaster, outside comparator only (see §2). It supplies no teacher outputs, utilities or features. Fine-tuned Chronos variants are excluded from the course deliverable. |

---

## 8. Experiments

**Protocol.** 9 expanding-window walk-forward folds, test years 2017–2025.
Validation = the year before the test year. **Headline mean = folds 1, 2, 3,
6, 7, 8, 9.** Folds 4–5 (COVID, 2020–21) are reported separately. Fold 1
(2017, the largest epidemic, trained on data with nothing comparable) is
always also decomposed out. Each (origin, horizon) cell belongs to the split
of its own target. Every method is scored on the intersection of cells. The
2026 hold-out (19 weeks) is governed by the addendum dated 2026-09-30 in
`results/benchmark/holdout_declaration.md`. The benchmark's 2026 results
had already been scored and summarised before that date. No forecast from
this extension has been made for 2026.

**Metrics.** MAE (primary), RMSE, peak MAE (cells at or above the district's
90th percentile of pre-test history), WIS and 50/95% interval coverage,
and skill vs same-horizon persistence = 1 − MAE/MAE_persistence.

**Statistics.** Paired t-test and Wilcoxon over the 7 headline folds, fold
win counts, Holm correction over the declared primary family, and a 4-week
time-block bootstrap. The project bar is Δ ≥ 2 seed-sd and p < 0.05.
Seed-mean and seed-ensemble estimands are never mixed.

### 8.0 Utility pilots (small, done 2026-09-30)
Outer fold 9, inner utility years 2022 and 2023, 2 seeds, 5 donor
permutations. All pilot outputs are in results/climate_horizon/pilots/.

- **Frozen equal-weight pilot, primary shuffle gain.** Mean g_h =
  0.028 / 0.039 / 0.050 / 0.059 for h = 1–4. All 16 block × seed values are
  positive and rise with h.
- **Case-only sensitivity gain.** Mean 0.014 / 0.010 / −0.000 / −0.002.
  The sign flips across seeds, so it is inconclusive.
- **Frozen weights, fold 9** (aggregation: seeds within block, then blocks
  equally; primary g_h = MAE_shuffled − MAE_real, absolute):
  - g = +0.47 / +0.81 / +1.17 / +1.51 MAE, positive in 4/4 units at every
    horizon (unit sd 0.37–1.17);
  - **w = 0.500 / 0.827 / 1.169 / 1.504** (mean 1, max/min 3.0).
- **Case-only sensitivity weights, fold 9:** 1.459 / 1.236 / 0.596 / 0.709.
  They come from gains smaller than their own seed/block variability, so
  they are noise-driven. They are used in a separate arm only.
- Interpretation: shuffle weights measure **reliance** of a frozen model on
  correctly aligned weather, not the causal or incremental value of
  climate. Mean-one weights do not equalise gradient norms.
- Superseded: final weights for all 10 folds come from 3 blocks × 2 seeds (see §8.1a).

### 8.0b Pre-sweep checks (done 2026-09-30; not results)
- **Synthetic NB data with a planted 8-week rainfall delay.** Shuffling
  climate raised MAE by 11–25 units, and the learned rainfall delay was
  7.2–8.2 weeks.
- **Synthetic data without climate signal.** Shuffling changed MAE by
  −0.07 to −0.01, and the kernels stayed at uninformative delays. This
  shows the pipeline can find and localise a planted delay. It says
  nothing about dengue biology.
- **Fold-9 development run, validation year 2024 only** (the year used for
  checkpoint selection; 2025 not scored). Equal-weight and weighted models
  were indistinguishable: validation NLL 3.5774 vs 3.5758 (seed 0) and
  3.5859 vs 3.5851 (seed 1), with near-identical gates (rising from about
  0.43 to 0.55 over h) and climate corrections.
- **Checkpoints.** Resuming training reproduces an uninterrupted run
  bit-for-bit, and reloading reproduces predictions exactly.

### 8.0c Development matrix (fold 9, validation year 2024, 2 seeds; done 2026-09-30; not held-out)

All arms: the same architecture (10,028 parameters), NB head, loss,
preprocessing, origins, initialisation and budget. In every arm the case
branch, gate and heads learn from all horizons. MAE, h = 1–4:

| Arm | h=1 | h=2 | h=3 | h=4 |
| --- | --- | --- | --- | --- |
| A origin lags, equal | 9.13 | 9.71 | 11.27 | 11.98 |
| B target lags, equal (baseline) | 9.21 | 9.78 | 11.36 | 12.21 |
| C origin lags, measured | 9.11 | 9.77 | 11.33 | 12.10 |
| D target lags, measured (proposed) | 9.19 | 9.80 | 11.37 | 12.20 |
| E fixed weights 1,2,3,4 | 9.21 | 9.77 | 11.35 | 12.18 |
| F h=4-only climate gradients | 9.18 | 9.86 | 11.42 | 11.50 |
| G case-only-utility weights | 9.21 | 9.80 | 11.34 | 12.18 |
| H GradNorm (climate group; labelled deviations) | 9.18 | 9.71 | 11.20 | 12.00 |
| `gru_v2_nb` / `nb_shared_v2` / `lgbm_v2` (stored) | 9.26 / 9.34 / 9.03 | 9.62 / 9.83 / 9.46 | 11.06 / 10.95 / 10.59 | 11.47 / 11.41 / 11.40 |
| persistence | 10.78 | 12.37 | 15.65 | 17.53 |

- **Proposed vs baseline: null** (|Δ| ≤ 0.02, seed range up to 0.49).
- **Target-relative lags are slightly worse than origin-relative on this
  year** (−0.07 to −0.23 in favour of origin).
- GradNorm's learned weights *decrease* with horizon.
- h=4-only climate gradients leave h=1–3 without climate (their climate
  heads stay at zero). This gives the best extension h=4 MAE but worse
  h=2–3 and worse peak MAE.
- The stored hand-lagged references beat every extension arm at h=3–4 on
  this year.
- One fold, one selection year: nothing is established. The primary test
  comparison over folds 1–9 has since been run (see §8.1a).

### 8.1 Primary comparison (frozen 2026-09-30 16:44, run manifest SHA-256 f4acdf1c…)
D (target lags, measured climate weights) vs B (target lags, equal
weights), 5 seeds each, headline MAE at h = 1–4 over the 7 headline folds.
Paired t-test and Wilcoxon, Holm across the 4 horizons, and 4-week
time-block bootstrap 95% CIs.

- **Practical improvement:** D − B ≤ −0.30 MAE at h = 3 or 4, with Holm
  p < 0.05 and the bootstrap CI below 0.
- **Non-inferiority at h = 1–2:** CI upper bound ≤ +0.30.
- **Otherwise:** "no measurable difference".
- Ablations A, C, E–H (3 seeds) are exploratory.

The development evidence (§8.0c) found no measurable difference on fold 9.
Final weights come from 3 inner blocks × 2 seeds per fold, not from the
2-block development weights. Retrospective folds 1–9 are reported before
2026; 2026 (B and D only) is a frozen-specification check, not a blind
test, because benchmark 2026 summaries were read on 29–30 Sep. No approval
step exists in the repository and none is claimed. Expected cost: about
382 fits, about 1 GPU-hour. **Executed; results in §8.1a.**

### 8.1a RESULTS (executed 2026-09-30 under the frozen manifest; measured, not placeholders)

**Retrospective, headline folds (5 seeds for B and D).** D − B =
+0.063 / +0.082 / +0.068 / +0.049 MAE for h = 1–4. Holm p = 0.78 / 0.60 /
0.78 / 0.78. Bootstrap 95% CIs [−0.03, +0.17], [−0.03, +0.21],
[−0.07, +0.23], [−0.11, +0.21]. D wins 3/7 folds at every h.
- **Verdict: no measurable difference.** Non-inferior at h = 1–2.
  Utility-weighted climate gradients did not improve the forecasts.
- All 9 folds agree (D − B = +0.05 to +0.07).

| Headline MAE | h=1 | h=2 | h=3 | h=4 | Peak h=4 | WIS h=4 |
| --- | --- | --- | --- | --- | --- | --- |
| A origin lags, equal | 16.84 | 19.97 | 23.13 | 25.73 | 43.98 | 15.73 |
| B target lags, equal (baseline) | 16.64 | 19.67 | 22.71 | 25.22 | 42.99 | 15.24 |
| C origin lags, measured | 16.20 | 19.28 | 22.37 | 24.95 | 42.56 | 14.99 |
| D target lags, measured (proposed) | 16.71 | 19.76 | 22.77 | 25.27 | 42.82 | 15.32 |
| E fixed 1,2,3,4 | 16.54 | 19.58 | 22.60 | 25.11 | 42.60 | 15.13 |
| F h=4-only | 16.64 | 20.21 | 23.88 | 25.48 | 43.94 | 15.17 |
| G case-only utility | 17.11 | 20.22 | 23.35 | 25.94 | 43.90 | 15.94 |
| H GradNorm | 16.61 | 19.69 | 22.74 | 25.26 | 43.23 | 15.27 |
| `gru_v2_nb` / `nb_shared_v2` / `lgbm_v2` | 15.94 / 16.63 / 16.99 | 19.26 / 19.83 / 21.05 | 22.46 / 23.15 / 24.30 | 25.07 / 25.84 / 27.16 | 42.60 / 43.82 / 46.42 | |
| persistence | 16.42 | 20.16 | 24.58 | 28.47 | 47.98 | |

**Exploratory (not in the primary family; 3 seeds; unadjusted p).**
- Origin-relative lags with measured weights (C) are the best extension
  arm on every horizon.
  - C − D = −0.50 / −0.48 / −0.40 / −0.32 (Wilcoxon p 0.016 / 0.016 /
    0.047 / 0.078).
  - With equal weights, origin lags are worse (A − B = +0.20 to +0.51).
  - This interaction was not declared and needs confirmation.
- B and D beat persistence at h = 3 (−1.87 / −1.80, p ≈ 0.03–0.04) and
  h = 4 (−3.25 / −3.20). They lose at h = 1.
- They are not significantly different from the stored
  `gru_v2_nb` / `nb_shared_v2` / `lgbm_v2`.
- COVID folds 4–5: the stored references are better.

**2026 hold-out (19 weeks; B and D only; not a blind test).**
- MAE, B 15.63 / 17.29 / 19.98 / 22.14; D 15.64 / 17.51 / 20.08 / 22.45.
- D − B = +0.01 / +0.22 / +0.10 / +0.32; every block-bootstrap CI
  includes 0.
- The sign matches the retrospective result, so it is **"consistent
  with"** no improvement.
- Both beat 2026 persistence on MAE point estimates (17.60 / 18.78 /
  20.92 / 22.47; CIs include 0) but not on peak MAE at h = 2–4.
- In 2026 the stored `nb_shared_v2` has the lowest MAE at h = 3–4
  (19.06 / 20.68).

**Measured weights (all 10 folds, primary utility).** They rise with
horizon in every fold (max/min 1.7–4.0). The case-only weights are
erratic (max/min up to 77).

**Cost.** About 65 GPU-minutes in total (pilots 10.9 min, 252
retrospective fits 47.6 min, 10 hold-out fits); peak GPU memory 274 MB;
RTX 4070 Laptop GPU. 8 fits first failed on a Windows file lock and were
retrained from scratch.

### 8.2 Ablation studies

| Arm | w_h into the climate encoder | Question |
| --- | --- | --- |
| uniform (baseline) | 1, 1, 1, 1 | — |
| **informed (proposed)** | primary utility (frozen-pilot shuffle gain), 5% uniform mix | — |
| informed, case-only utility | relative gain over a case-only pilot | Is the result robust to the utility definition? |
| permuted | measured weights reversed across horizons | Does it matter *which* horizon gets weight? |
| ramp prior | ∝ h, mean 1 | Does measuring beat the prior "longer horizon = more climate"? |
| mask h=1 | 0, 4/3, 4/3, 4/3 | Is excluding h = 1 enough? |
| frozen encoder | encoder not trained | Do learned delays matter at all? |
| informed + shuffled climate | measured, weather time-shuffled | Does the gain require real weather content? |
| case-only | no climate branch | Floor |
| origin-relative lags | 1,1,1,1 and measured | Does defining the delay relative to the target matter? |
| target lags, common support | as main arms | Is any difference due to unequal delay support across h? |
| gate encoder only | routing on the kernels only, not the climate GRU | Where should routing apply? |
| measured-lag initialisation (optional) | kernels start at the cross-correlation delays | Prior vs learned |

**Run on folds 1–9: results in §8.1a** (the arm "gate encoder only" and measured-lag initialisation were not run).

### 8.3 Comparison with existing methods
Done on common cells, h = 1–4, for stored references with verified provenance: see §8.1a and
`results/climate_horizon/analysis/tables/metrics_headline_seed_mean`. No extension arm is
significantly better than the stored separate GRU-NB.

### 8.4 Hyperparameter tuning
Selected on **validation years only**:
- number of inner blocks K ∈ {2, 3, 4};
- weight rule (clip-normalise vs softmax with temperature τ; the measured
  gains may be fairly flat, see §10);
- climate GRU width {8, 16, 32};
- lag reach {13, 26};
- encoder learning rate.

The short paper's 50-trial random search found a flat response surface for
the backbone. Backbone settings are therefore kept fixed. **[PENDING]**

### 8.5 Mechanism and interpretability (measured; see docs/climate_horizon_results.md §5–6)

- **Learned rainfall kernels**, as the centre of mass in target-relative
  delay (folds 1–9):
  - B: 8.0 / 8.3 / 8.6 / 8.9 weeks for h = 1–4;
  - D: 8.7 / 8.9 / 9.2 / 9.6;
  - origin-relative arms shift by one week per horizon by construction;
  - usable kernel mass in target mode falls from 0.96 to 0.80 as h grows.
- **Agreement with the training-only cross-correlation scan** (per
  district): Spearman 0.24–0.31 for B and D, 0.14–0.19 for A and C. The
  scan's own rainfall signal is below its resolvability threshold, so this
  is descriptive agreement, not delay recovery.
- **Gate and corrections.** The mean gate rises with h (about 0.43 →
  0.56). The gated climate correction rises from 0.13 to 0.29 in log-mean
  space and is larger than the case correction only at h = 4. B and D
  differ by ≤ 0.02 on all of these.
- **What the utilities measure.**
  - The primary utility measures **shuffle reliance** of a trained model;
    it rises with h in every fold.
  - The case-only utility measures **incremental benefit** over a
    size-matched case-only model; it is near zero and unstable.
  - Neither identifies a **causal** climate effect.
- Figures: `results/climate_horizon/analysis/figures/rainfall_delay_kernels`,
  `gates_and_corrections`, `weights_vs_uniform`, `signed_utilities`.

### 8.6 Computational analysis
Parameters, training seconds per fit (including the pilot overhead of
weight estimation), inference ms per origin, peak GPU memory, hardware.
Reference values from the current benchmark (RTX 4070 Laptop GPU):

| Model | Parameters | s / fit | ms / origin | Peak GPU MB |
| --- | --- | --- | --- | --- |
| `gru_v1` | 8,193 | 2.51 | 0.037 | 138 |
| `gru_v2` | 8,257 | 1.24 | 0.020 | 141 |
| `gru_v2_nb` | 8,290 | 2.20 | 0.025 | 141 |
| `nb_shared_v2` (8 horizons, one trunk) | 8,752 | 2.41 | 0.023 | 143 |
| Proposed (D) | 10,028 | 13.2 mean over 50 fits (B: 11.8); pilots 10.9 min for all 10 folds | **[PENDING: not measured]** | 275 (max over all extension fits) |

The shared model trains all horizons in about 1/4 of the separate models'
trunk cost (short paper: 5.6 s vs 21.3 s on fold 1). Routing adds no
parameters and no extra backward pass. Weight estimation adds
10 folds × 3 blocks × 2 seeds × (climate pilot + case-only pilot) ≈ 120 small pilot fits, plus inference-only shuffles.

---

## 9. Results already established (for Introduction, Related Work and Discussion)

### 9.1 From the short paper (GEE ERA5-Land climate, 9 folds × 3 seeds)

- **h = 1 headline MAE:** persistence **16.42**; GRU-only v1 16.68; GRU-only
  v0 16.86; GCN-GRU v0 18.60; GCN-GRU v1 18.98. Peak MAE: persistence
  26.61; best model 28.06.
- **The graph hurts:** GRU-only beats GCN-GRU on every fold. At h = 1:
  adaptive 16.61, identity 16.68, contiguity 18.98, Gaussian 19.25. A
  per-district gate reached 18.36 (vs 16.88 without a graph) and barely
  moved from initialisation.
- **Losses** (h = 1): level-weighted 16.59, quantile 17.14, Huber-weighted
  17.71, Huber 18.17, MSE 18.98. The gain of the level-weighted loss comes
  almost entirely from the 2017 fold.
- **Hyperparameter search** (50 trials, 9 axes, validation-only selection):
  18.23 vs 18.98, within two seed-sd, so the response surface is flat.
- **Optimiser:** Adam 18.98, AdamW 18.67, AdamW + ReduceLROnPlateau 18.90.
  In 23 of 27 runs the LR reduction fired only after early stopping had
  already selected the kept checkpoint.
- **Learnable lags at h = 1:** 20.67 MAE vs 18.36 (fixed windows) and 18.07
  (no lags). Learned vs measured delay r = −0.16. **This is the failure the
  new method addresses.**
- **Multi-horizon** (identity backbone, headline MAE h = 1…4):
  shared 16.71/20.04/23.15/25.74; separate 17.20/20.22/22.89/25.40;
  persistence 16.42/20.16/24.58/28.47. Skill (separate) −4.7%, −0.3%,
  +6.9%, +10.8%; peak-MAE skill −11.7%, −5.1%, +5.2%, +9.8%. At h = 4 the
  model wins 7/7 folds (Δ −3.07, t −4.04, p = 0.0068); excluding 2017 it
  still wins 6/6 (Δ −2.71).
- **Climate ablation** (paired, 7 folds; positive = ablation hurts):
  shuffled +0.02/−0.06/+0.43/**+1.55** (p = 0.926/0.854/0.226/**0.016**);
  shuffled peak +0.40/+0.38/+1.65/+3.10; removed −0.25/−0.35/−0.06/+0.76.
  Shuffling removes about half the h = 4 MAE skill (10.8% → 5.3%) and about
  two thirds of the peak skill (9.8% → 3.3%). Removal alone would have
  suggested the opposite conclusion, so the shuffle control is essential.
- **Layer normalisation:** no accuracy gain, but 24–51% lower seed variance.

### 9.2 From the current benchmark (Open-Meteo climate, common cells, 3 seeds, headline MAE)

| Method | h=1 | h=2 | h=3 | h=4 | Peak MAE h=4 |
| --- | --- | --- | --- | --- | --- |
| persistence | 16.42 | 20.16 | 24.58 | 28.47 | 47.98 |
| `gru_v1` | 16.80 | 20.41 | 23.02 | 25.95 | 44.20 |
| `gru_v2` | 16.64 | 19.51 | 22.33 | 25.30 | 43.59 |
| `gru_v2_nb` | 15.94 | 19.26 | 22.46 | 25.07 | 42.60 |
| `nb_shared_v2` | 16.63 | 19.83 | 23.15 | 25.84 | 43.82 |
| `lgbm_v2` | 16.99 | 21.05 | 24.30 | 27.16 | 46.42 |
| uniform routing (baseline, B) | 16.64 | 19.67 | 22.71 | 25.22 | 42.99 |
| **informed routing (proposed, D)** | 16.71 | 19.76 | 22.77 | 25.27 | 42.82 |

Climate controls on the current file (Δ MAE when climate is shuffled,
7 folds):

| Model | h=1 | h=2 | h=3 | h=4 |
| --- | --- | --- | --- | --- |
| GRU (MSE) | +0.01 | +1.03 | +1.97 | +1.91 |
| NB shared | +0.81 | +1.33 | +1.59 | +1.84 |

Individual p values are 0.06–0.98. Wilcoxon p ≈ 0.03–0.08 at h ≥ 2. Climate
still helps compared with each district's training-years week-of-year
climatology in all three model families. A shared NB trunk costs about
+0.6–0.8 MAE against separate NB models. The NB, pinball and MSE heads are
statistically indistinguishable.

---

## 10. Discussion points to prepare (fill after results)

- If informed > uniform at h = 3–4: the delay encoder learns better when it
  is shielded from horizons whose loss carries no climate information.
  Check that kernel recovery and stability improve with it.
- If they are close: the measured weights may be too flat to matter
  (benchmark shuffle costs hint at fairly flat gains at h = 2–4; these are
  never used as weights). The sharpened weight rule (τ) and the `mask h=1`
  arm then become the key evidence. A null result is still a finding about
  where the encoder's learning signal comes from.
- Compare `permuted` and `ramp`. If the ramp does as well as measurement,
  the contribution reduces to "down-weight short horizons", and the paper
  must say so.
- Onset forecasting (the first outbreak week after a quiet period) remained
  hard in earlier work (onset ROC-AUC 0.74–0.77 under a consistent at-risk
  definition). State that the method targets delay learning and accuracy,
  not onset detection.

## 11. Limitations to state

- 7 headline folds give low power; the smallest possible Wilcoxon p is
  0.016.
- Climate is a centroid-based reconstruction, not polygon means.
- Measured "ground-truth" delays are weakly resolved (correlation below
  0.10), so kernel-recovery evidence is supporting, not decisive.
- A single country and 25 nodes. Weekly reporting delays and
  under-reporting are not modelled.
- Earlier design choices were guided by the 2017–2025 test years, which is
  why the 2026 hold-out and a pre-registered primary family are used.

## 12. Candidate contributions (pick 2–3 for the Introduction)

1. **Horizon-informed gradient routing:** a training strategy that measures
   each horizon's climate value on inner validation blocks and uses it to
   control which horizons teach a shared climate-delay encoder, while case
   history and heads learn from all horizons. It adds no parameters and
   needs one backward pass.
2. **An interpretable two-branch forecaster:** per-district, per-variable
   learnable climate-delay kernels fused with a case-history GRU and
   anchored Negative Binomial heads, producing calibrated multi-horizon
   probabilistic forecasts.
3. **A leakage-controlled evaluation:** 9 walk-forward folds, per-horizon
   persistence, capacity-matched climate-shuffle controls, common-cell
   scoring and a pre-registered hold-out. It shows **where** climate adds
   skill (h ≥ 2–3) and that one-week-ahead gains are mostly illusory.

## 13. Related work (topics and references)

Already in `docs/references_long.bib`: Cho 2014 (GRU); Kipf & Welling 2017
(GCN); Wu 2019 (Graph WaveNet); Kingma & Ba 2015; Loshchilov & Hutter 2019;
Muñoz-Sabater 2021 (ERA5-Land); Funk 2015 (CHIRPS); Talagala 2026
(denguedatahub); Weng et al. 2024 (Sri Lanka dengue GNN, IEEE BigData);
GulMohamed et al. 2026 (Sri Lanka dengue GNN, Sci. Rep.); Liu et al. 2025
(explainable dengue–climate lags, Sri Lanka); Johansson et al. 2019 (open
dengue forecasting challenge); Bracher et al. 2021 (WIS); Colón-González
et al. 2021 (probabilistic dengue forecasting); Mosqlimate 2026 sprint;
Ke 2017 (LightGBM); Ansari 2024/2025 (Chronos); Hu 2022 (LoRA); Vovk 2005
(conformal).

To add. **Only GradNorm has been verified in this project so far**: Z. Chen, V. Badrinarayanan, C.-Y. Lee, A. Rabinovich, "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks", ICML 2018, PMLR vol. 80 (https://proceedings.mlr.press/v80/chen18a/chen18a.pdf). The entries below are **unverified and must not be cited until checked** against the source:
- Multi-task loss and gradient balancing: Chen et al. 2018 (GradNorm, ICML);
  Kendall, Gal & Cipolla 2018 (uncertainty weighting, CVPR); Yu et al. 2020
  (PCGrad, NeurIPS).
- Gradient routing / localised learning: Cloud et al. 2024 ("Gradient
  Routing").
- Distributed lag models for climate–health: Gasparrini, Armstrong &
  Kenward 2010 (DLNM, Statistics in Medicine).
- Deep probabilistic forecasting with count likelihoods: Salinas et al. 2020
  (DeepAR, IJF).
- Multi-horizon forecasting with covariates and variable selection: Lim
  et al. 2021 (Temporal Fusion Transformer, IJF).
- Climate-driven dengue early warning: Lowe et al. (Brazil) and similar
  climate–dengue models.

Positioning: the multi-task methods reweight **tasks for all shared
weights** from **in-training statistics**. We route **one input modality's
encoder** by an **out-of-sample, capacity-matched measurement of that
modality's value per task**. DLNMs fix the lag structure statistically; we
learn it and choose which training signal is allowed to shape it.

## 14. Figures and tables (produced; `results/climate_horizon/analysis/`)

Regenerate with `scripts/evaluation/49.climate_horizon_analysis.py`.
Tables are CSV plus a LaTeX `table` environment of the same name.

1. Architecture diagram: **[PENDING: not drawn]**.
2. `figures/weights_vs_uniform`: frozen weights per fold against uniform
   (primary and case-only).
3. `figures/signed_utilities`: signed shuffle gains with unit ranges, and
   case-only gains.
4. `figures/primary_per_fold_differences`: per-fold D − B and the headline
   CI.
5. `figures/rainfall_delay_kernels`: learned rainfall kernels over delay,
   arms A–D.
6. `figures/gates_and_corrections`: gates and correction magnitudes by
   horizon.
7. `figures/selected_forecasts`: Colombo, 2017 and 2025, h = 4. District
   chosen by a fixed rule (highest mean training cases).
8. Tables:
   - `primary_D_vs_B`, `primary_bootstrap_estimands`;
   - `metrics_{headline,all9,covid}_{seed_mean,ensemble}`;
   - `per_fold_B_D`, `holdout_2026_metrics`, `holdout_2026_bootstrap`;
   - `controls_vs_B`, `weights_by_fold`;
   - `rainfall_kernel_summary`, `kernel_vs_scan_spearman`,
     `gates_and_corrections`;
   - `reference_provenance`, `coverage`, `compute`.

**Course-facing tables.** Chronos-2 zero-shot appears only as a labelled
outside comparator. Fine-tuned foundation-model variants are omitted.
Short-paper neural scores (GEE climate) are shown only in a separate
historical table and are never paired with current-climate results.

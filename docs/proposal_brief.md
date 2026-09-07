# AEGIS-Dengue — source brief for the Phase 1 project proposal

Everything needed to write the 2-page ACM-format proposal. All numbers below are
taken from the committed artifacts in this repository and are current as of
2026-08-04. Section 9 lists the claims that are **not** yet proven — do not
promote any of them into results.

---

## 0. Instructions for generating the LaTeX

**Template.** ACM `acmart` class, `sigconf` option, two columns.
`\documentclass[sigconf,nonacm]{acmart}` unless the course specifies otherwise.

**Hard limit: 2 pages excluding references.** This is the binding constraint —
the material below is deliberately more than fits. Budget roughly:

| Section | Target |
|---|---|
| Problem motivation | ~0.35 col |
| Dataset + 1 figure | ~0.9 col |
| Baseline architecture + implementation | ~0.7 col |
| Baseline results + 2 tables | ~1.0 col |
| Proposed contribution | ~0.6 col |
| Experimental plan | ~0.45 col |

**Figures: use two, not seven.** Recommended pair, both single-column:

1. `results/eda/fig3_geography.png` — the 25 districts and the contiguity graph.
   This is the single most useful figure because it introduces the spatial unit
   *and* the graph the model runs on in one image.
2. `results/eda/fig1_national_timeseries.png` — the target variable over the full
   span, with the 2017 epidemic and COVID periods marked.

If a third fits, use `results/eda/fig5_spacetime_heatmap.png` as a full-width
(`figure*`) figure. The remaining figures (2, 4, 6, 7) are available in
`results/eda/` and are better held for the final report.

**Tone.** Report the preliminary baseline results as preliminary. The proposal is
assessed on feasibility and scope, and a baseline that currently *loses* to
persistence on the epidemic fold is a well-defined research problem, not a
weakness to hide. Section 9 is explicit about what may not be claimed.

**Cut first if over length:** the per-fold naive table (keep the summary rows
only), the feature-list detail in §3, and the resource-requirements paragraph.

---

## 1. Problem motivation

**The problem.** Dengue is endemic in Sri Lanka and epidemic in bursts. The
national surveillance system reports confirmed cases per district per weekly
reporting period, but reporting is retrospective: by the time a district's case
count rises in the published series, transmission has already been under way for
roughly one to two incubation-plus-reporting cycles. Vector control — source
reduction, larviciding, targeted public messaging — is only effective if it is
deployed *before* the rise. The operational question is therefore a forecast:
given case history and climate up to now, what will each district report next
period?

**Why it is hard, concretely, in this dataset.**

- **The signal is dominated by interannual variation, not season.** A seasonal
  naive forecast scores a headline MAE of 55.45 against persistence's 16.42
  (§4.1). Knowing the time of year is worth very little; knowing last week is
  worth a great deal.
- **The events that matter most are the ones with least training support.** 2017
  produced 175,389 reported cases against a 2007–2016 median of roughly 23,000.
  A model trained on the preceding decade has seen nothing of that magnitude.
- **The target is zero-inflated and heavy-tailed.** Across 25,299 observed
  district-periods the median is 10 cases and the maximum is 2,631, with 13%
  exact zeros. Mean absolute error over such a distribution is dominated by quiet
  periods, so a model can look good on the headline metric while being useless
  during an outbreak.
- **Burden is spatially concentrated.** Five of the twenty-five districts account
  for 54% of all reported cases, and Colombo alone for 22%.
- **Resources are limited.** This is a public-health setting without GPU
  infrastructure. A deployable model must train and re-train on commodity CPU.
  The baseline here has 8,193 parameters and trains on CPU.

**Who benefits.** The National Dengue Control Unit and district-level public
health inspectorates, who allocate vector-control effort weekly under a fixed
budget; hospital administrators sizing ward capacity ahead of a surge; and,
indirectly, residents of the high-burden western districts. A district-level
one-period-ahead forecast maps directly onto the unit at which control resources
are actually assigned.

**Framing for the proposal.** The accuracy requirement is asymmetric. Being wrong
by 5 cases in a quiet week in Mullaitivu costs nothing; being wrong by 300 cases
at the onset of a Colombo outbreak costs a control campaign. This is why the
evaluation in §6 reports peak-period error separately, and why the central
research target in §5 is the epidemic fold rather than the headline average.

---

## 2. Dataset

### 2.1 At a glance

| Property | Value |
|---|---|
| Spatial units | 25 canonical districts across 9 provinces |
| Time index | 1,012 reporting periods, 2006-12-23 to 2026-05-17 |
| Panel size | 25,300 district-period rows × 33 columns |
| Period length | 6–8 days (2 irregular periods, both in 2009) |
| Total reported cases | 833,559 |
| Missing case records | 1 of 25,300 (0.004%) |
| Missing climate records | 75 of 25,300 (0.30%) — periods 1010–1012 |

### 2.2 Sources

| Source | Contributes | Notes |
|---|---|---|
| denguedatahub weekly Sri Lanka series | confirmed cases per reporting area per week | 26 source reporting areas merged to 25 districts (Kalmune folds into Ampara) |
| ERA5-Land (ECMWF), hourly | rainfall, 2 m temperature (mean/min/max), dewpoint, relative humidity, wind speed | extracted via Google Earth Engine, reduced hourly→daily on the UTC day, area-weighted per district polygon with a land–sea mask, then aggregated over each reporting period's true `start_date`–`end_date` interval |
| CHIRPS daily (UCSB-CHG) | independent rainfall series | used as a cross-check on ERA5-Land precipitation, not as a model feature |
| GADM 4.1 ADM1 | district polygons | source of the contiguity graph and district centroids (EPSG:32644) |

### 2.3 Three data-engineering decisions worth one sentence each in the proposal

These are what make the panel trustworthy and are cheap to state:

1. **A dedicated chronological key.** The source `year`/`week` labels are not
   sortable: the interval labelled "2026 week 53" actually covers 2025-12-20 to
   2025-12-26. Sorting on the label places a December 2025 observation after May
   2026, corrupting every lag across that boundary. The pipeline assigns
   `period_id` after sorting on true dates and uses it as the only sort key.
2. **Climate is joined on exact date intervals, never on week numbers.** Periods
   are not ISO weeks, two are 6 and 8 days long, and the reporting weekday changes
   twice across the series.
3. **Absence is carried as absence.** Puttalam has no record for 2026 week 7. It
   is stored as missing and masked out of the loss and every metric, never
   imputed to zero — a fabricated zero asserts an observation nobody made and
   biases the model toward under-prediction exactly where accuracy matters.

### 2.4 EDA findings to cite in text

- Two transmission peaks per year following the two monsoons, but the
  interquartile range across 2007–2025 spans most of the annual cycle — the
  descriptive counterpart to the seasonal-naive result in §4.1.
- Northern districts (Kilinochchi, Mullaitivu, Mannar) report near-zero before
  roughly 2010, reflecting post-conflict surveillance coverage rather than
  absence of dengue.
- The 2017 epidemic appears as a simultaneous rise across essentially all
  districts, which argues against a purely spatial-diffusion account.
- Rainfall varies far more between districts than temperature does — the
  cross-sectional signal a district-level model must exploit.

### 2.5 Figure captions (ready to use)

**Fig. 1 (`fig3_geography.png`).** *The spatial unit. Left: cumulative reported
dengue cases by district, 2006–2026; burden is concentrated in the western
coastal belt, with Colombo alone accounting for 22% of the national total. Right:
the queen-contiguity graph over the same 25 districts — 25 nodes, 57 edges, mean
degree 4.56. Jaffna has degree 1, so it receives almost no information through
the graph.*

**Fig. 2 (`fig1_national_timeseries.png`).** *National reported cases per weekly
reporting period, December 2006 to May 2026 (top), and the same data as annual
totals (bottom). The 2017 epidemic peaked at 10,590 cases in a single reporting
period, roughly seven times the surrounding baseline; the shaded band marks the
COVID-19 reporting periods, which are held out of the headline evaluation.*

**Fig. 3, optional (`fig5_spacetime_heatmap.png`).** *The panel in full: monthly
cases per district on a log scale, districts ordered by total burden. The 2017
epidemic is the vertical dark band; the pale block in the northern districts
before 2010 is post-conflict surveillance coverage.*

---

## 3. Baseline model

### 3.1 Architecture

Spatial convolution over districts at each time step, then a shared recurrent
pass over the input window, then a linear head:

```
per timestep t:   S = relu(A_norm @ X_t @ W1)
                  S = relu(A_norm @ S   @ W2)
over the window:  H = GRU(S_1 … S_L)      shared weights, one sequence per node
head:             y = Linear(H_L)
```

`A_norm = D^{-1/2}(A + I)D^{-1/2}` over the queen-contiguity adjacency. The GRU
weights are shared across all 25 districts; node identity reaches the model
through the graph and through centroid features, not through per-district
weights — 25 separate GRUs over ~500 training windows would memorise the training
years outright.

**Spatial-then-temporal is deliberate**, not incidental: it is the arrangement
into which the season-gated fusion module of §5 drops without restructuring.

**8,193 trainable parameters** at `hidden=32`. Dense 25×25 matrix multiply, no
PyTorch Geometric — at 25 nodes a dense matmul beats any sparse gather and keeps
the dependency list to `torch` alone.

### 3.2 Target parameterisation

The model does not predict the case count. It predicts the log-scale change from
the forecast origin:

```
predict   d   = log1p(y[t+h]) − log1p(y[t])
report    ŷ   = expm1( log1p(y[t]) + d̂ )
```

An output of exactly zero therefore reproduces the persistence forecast, and the
network only has to learn the correction. The anchor is the case count at the
forecast origin — the last period the model is permitted to see — so it carries
no information the input window does not already contain. This choice was made by
measurement, not preference; the ablation is in §4.3.

### 3.3 Windowing and controls

| Item | Value |
|---|---|
| Lookback `L` | 12 reporting periods |
| Horizon `h` | 1 period ahead |
| Windows available | 1,000 at `L=12, h=1`; v0 retains 987, v1 retains 976 |
| Feature variant `v0` | 14 features: `log1p` cases, 7 climate variables, day-of-year sin/cos, two observation flags, centroid lat/lon |
| Feature variant `v1` | 23 features: v0 plus trailing 4-, 8- and 12-period means of rainfall, temperature and humidity |
| Control model | `gru_only` — identical in every respect with the adjacency replaced by the identity matrix |

`v1 − v0` measures what hand-specified climate lags are worth on this data. That
is precisely the number the learnable lag module of §5 must beat. `gcn_gru −
gru_only` isolates what the graph contributes; without the control, a GCN+GRU
beating persistence would have shown nothing about the graph.

### 3.4 Implementation

- PyTorch 2.13.0, CPU build. No GPU. No PyTorch Geometric.
- Loss: masked MSE. `error = (pred − target) * mask; loss = (error²).sum() / mask.sum()`.
- Optimiser: Adam, gradient-norm clipping at 1.0.
- Missing **targets** are never imputed — they are masked out of the loss and of
  every metric. Missingness is handled per district, not per window: a window with
  one absent district keeps its other 24 observed cells.
- Missing **features** (climate, case history) are imputed by district × month
  climatology, with fallback to district mean then global mean, fitted per fold.
- Reproducibility: 62 tests pass across `test_model_tensors.py`,
  `test_adjacency.py`, `test_folds.py`. The leakage rule is tested as an
  experiment rather than a code review — every period from the test year onward is
  overwritten with `1e6`, the statistics are refitted, and the test asserts that
  not one fitted statistic moves.

### 3.5 Validation protocol

Nine walk-forward folds. Each trains on everything before a calendar year,
validates on the year immediately before the test year, and tests on the test
year.

| Fold | Test year | Train periods | Val | Test | Train windows | Role |
|---|---|---|---|---|---|---|
| 1 | 2017 | 1–471 | 472–524 | 525–576 | 459 | epidemic — the hard fold |
| 2 | 2018 | 1–524 | 525–576 | 577–628 | 512 | headline |
| 3 | 2019 | 1–576 | 577–628 | 629–680 | 564 | headline |
| 4 | 2020 | 1–628 | 629–680 | 681–732 | 616 | COVID — reported separately |
| 5 | 2021 | 1–680 | 681–732 | 733–784 | 668 | COVID — reported separately |
| 6 | 2022 | 1–732 | 733–784 | 785–837 | 720 | headline; first fold with COVID in *train* |
| 7 | 2023 | 1–784 | 785–837 | 838–889 | 772 | headline |
| 8 | 2024 | 1–837 | 838–889 | 890–941 | 825 | headline — the normal case |
| 9 | 2025 | 1–889 | 890–941 | 942–993 | 877 | headline |

Rules enforced by the pipeline:

- Headline = mean over folds 1, 2, 3, 6, 7, 8, 9. COVID folds are reported
  separately and never averaged in.
- Every fitted statistic — climatology, scaler, peak thresholds — comes from
  periods strictly before the fold's test year.
- A window is assigned to a split by its **target** period. Its input history may
  reach into an earlier split, which is correct: at deployment that history is
  available.
- Early stopping is on the inner validation year, never on the test fold.
- `is_covid_window` and `is_2017_outbreak` are **not** features. As inputs they
  would be columns that are 0 in training and 1 in test, set by someone who
  already knew which years were unusual. They are retained solely to stratify
  results.
- 2026 is not a fold: it is a partial year and its last three periods have no
  climate. It is held back as final unseen data.

---

## 4. Baseline experimental results

### 4.1 Naive baselines — complete, all 9 folds

This table is fully computed and can be presented without qualification.

| Model | Headline MAE | Headline RMSE | Headline peak MAE | 2017 MAE | 2017 peak MAE | COVID MAE |
|---|---|---|---|---|---|---|
| persistence | **16.42** | **33.61** | **26.61** | **36.08** | **39.42** | **7.32** |
| seasonal naive | 55.45 | 118.76 | 84.81 | 111.10 | 126.65 | 44.48 |

Per fold (include only if space permits):

| Fold | Year | Persistence MAE | Persistence peak MAE | Seasonal MAE | Peak cells |
|---|---|---|---|---|---|
| 1 | 2017 | 36.08 | 39.42 | 111.10 | 1114 |
| 2 | 2018 | 10.45 | 16.34 | 104.20 | 273 |
| 3 | 2019 | 17.30 | 32.31 | 51.67 | 476 |
| 4 | 2020 | 7.41 | 26.62 | 65.60 | 117 |
| 5 | 2021 | 7.23 | 35.48 | 23.36 | 31 |
| 6 | 2022 | 12.57 | 23.66 | 21.00 | 116 |
| 7 | 2023 | 18.83 | 32.58 | 40.12 | 363 |
| 8 | 2024 | 10.78 | 25.77 | 37.58 | 145 |
| 9 | 2025 | 8.95 | 16.19 | 22.50 | 103 |

**The finding to state in the proposal:** seasonal naive is more than three times
worse than persistence. Sri Lankan dengue's interannual variation dominates its
seasonality, so a model that mostly learns the seasonal cycle will not be
competitive. This sets the bar: persistence is a strong baseline and anything
downstream must beat it on headline MAE *and* on peak MAE, since beating mean MAE
alone is achievable by predicting close to the recent level, which persistence
already does for free.

### 4.2 GCN+GRU — preliminary, two folds, single seed, `v1` only

**Label this table "preliminary" in the paper.** These are the only trained-model
numbers measured so far. The full sweep has not been run (§9).

Fold 8 (2024) — the normal case, twelve years of training data:

| Model | MAE | RMSE | Peak MAE | vs. persistence (MAE) |
|---|---|---|---|---|
| persistence | 10.78 | 21.57 | 25.77 | — |
| `gcn_gru` v1 | 9.98 | 18.82 | 23.44 | −7.4% |
| `gru_only` v1 | **9.15** | **18.75** | **21.70** | **−15.1%** |

Fold 1 (2017) — the epidemic fold:

| Model | MAE | vs. persistence |
|---|---|---|
| persistence | **36.08** | — |
| `gcn_gru` v1, residual target | 52.91 | **+46.6% worse** |

**Both results matter and both belong in the proposal.** On an ordinary year the
model beats persistence by a modest margin. On the epidemic fold it loses badly:
it trains on 2007–2016, which contains no outbreak of that size, and
under-predicts — Colombo's actual mean over the 2017 test fold was 631 cases per
period against 214 predicted, at a correlation of 0.71. The model tracks the
*shape* and misses the *level*. **This is the central open problem the proposed
contribution targets**, and stating it plainly is what makes the proposed work
well-scoped rather than speculative.

A second observation worth one sentence: `gru_only` edges out `gcn_gru` on fold 8.
At one seed this is not decidable either way, but it means the value of the
contiguity graph is currently unproven — which is exactly what §5's dual-graph
contribution is designed to address.

### 4.3 Hyper-parameter configuration and tuning

Committed configuration (`gcn-gru-v1`):

| Setting | Value | Basis |
|---|---|---|
| `lookback` | 12 periods | ≈ one quarter of history; sweep planned over {12, 26} |
| `horizon` | 1 period | the operational forecast unit |
| `hidden` | 32 | gives 8,193 parameters against 459–877 training windows |
| `gcn_layers` | 2 | two hops covers most district pairs at mean degree 4.56 |
| `dropout` | 0.2 | standard, given the small training set |
| `learning_rate` | 0.003 | Adam |
| `weight_decay` | 1e-4 | Adam |
| `batch_size` | 64 | |
| `max_epochs` | 150 | |
| `patience` | 15 | early stopping on the inner validation year |
| `seeds` | 3 | a single-seed result on ~500 windows is noise |
| `target` | `residual` | selected by the ablation below |

**Tuning performed so far — target parameterisation.** This is the one
hyper-parameter that has been selected by measurement rather than by convention,
and it produced the largest single improvement. Measured on folds 1 and 8, single
seed:

| Fold | `direct` (log1p level) | standardised | `residual` (anchored) |
|---|---|---|---|
| 1 (2017) | 87.45 | 76.49 | **52.91** |
| 8 (2024) | 17.58 | 15.07 | **10.11** |

Anchoring cuts fold-1 MAE by 40% and fold-8 MAE by 43% against the unanchored
target, and converges in roughly a third of the epochs. It is a target
parameterisation, not an architecture change — the model remains exactly a graph
convolution followed by a GRU.

*(Note for the writer: the fold-8 residual figure appears as 10.11 in this
ablation and 9.98 in §4.2. They are separate single-seed runs; the discrepancy is
seed noise and is itself the argument for the 3-seed protocol. Quote one or the
other, not both, and do not present the difference as meaningful.)*

**Tuning not yet performed** — this is the content of the first experimental
milestone in §6: the `L ∈ {12, 26}` × `h ∈ {1, 2, 3, 4}` window-geometry sweep,
and the `v0` vs `v1` feature comparison at 3 seeds across all 9 folds.

---

## 5. Proposed contribution

Each item is a single change, measured against the frozen baseline on the same
folds and the same masks. They are ordered by expected impact on the fold-1
failure.

### 5.1 Peak-weighted loss

The most direct attack on the epidemic-fold failure. Weight the masked MSE by
observed case level so that epidemic periods stop being averaged away by the
zero-inflated bulk of the distribution (13% exact zeros, median 10).

*Success criterion:* fold-1 peak MAE below persistence's 39.42.

### 5.2 Learnable lag module

A per-climate-variable, per-district learnable lag, replacing the hand-specified
4/8/12-period trailing means of `v1`. The biological lag between rainfall and
transmission differs between the wet-zone west and the dry-zone north-central
districts, and a single fixed set of windows cannot express that.

*Success criterion:* beats the `v1 − v0` gap, which the full sweep will quantify.

### 5.3 Dual graph with a season-dependent gate

Combine the fixed contiguity adjacency `A_manual` with a learned data-driven
adjacency `A_data`, mixed by a gate conditioned on season. The motivation is in
the data: contiguity is a crude prior in a country this narrow — Colombo and
Galle are 100 km apart, share no border, and are both dense wet-zone coastal
districts, so contiguity scores them as unrelated. A Gaussian-kernel alternative
`A_gaussian = exp(−(d/50 km)²)` is already built and available as a second fixed
candidate.

*Success criterion:* beats both `gcn_gru` (contiguity) and `gru_only` (identity)
by more than the seed standard deviation.

### 5.4 Gated spatial–temporal fusion

A learned gate over the spatial and temporal pathways. The baseline is arranged
spatial-then-temporal precisely so that this drops in without restructuring.

### Expected novelty

The novelty is the **combination applied to an epidemic-year evaluation**, not
any single component. Specifically: a season-gated dual-graph spatio-temporal
network with per-district learnable climate lags, evaluated on a walk-forward
protocol that places a major epidemic in a *test* fold and reports peak-period
error as a first-class metric rather than a mean over predominantly quiet weeks.
The `gru_only` control makes the graph's contribution measurable rather than
assumed — which, on the evidence in §4.2, is not a formality.

### Expected improvement

State these as testable targets, not as expected outcomes:

| Target | Current baseline | Goal |
|---|---|---|
| Fold-1 (epidemic) peak MAE | 39.42 (persistence); model currently worse | below persistence |
| Headline MAE | 16.42 (persistence) | a margin exceeding the seed standard deviation |
| Graph contribution | unproven at 1 seed | `gcn_gru` − `gru_only` significant over 3 seeds |

---

## 6. Experimental plan

### 6.1 Evaluation metrics

All metrics are computed on the original case scale, over observed cells only,
using the target mask.

| Metric | Definition | Why |
|---|---|---|
| MAE | masked mean absolute error | headline comparability |
| RMSE | masked root mean squared error | penalises the large misses |
| **Peak MAE** | MAE restricted to cells at or above each district's 90th percentile, with thresholds fitted on the fold's history only | the operationally decisive number; a model can win on MAE while failing every outbreak |
| Per-district MAE | the above, broken out by district | checks whether low-degree nodes such as Jaffna (degree 1) are systematically worse |
| Seed sd | standard deviation of fold MAE across 3 seeds | the threshold below which a margin is not a result |

Stratified reporting: headline over folds 1, 2, 3, 6, 7, 8, 9; COVID folds 4 and
5 separately as a distribution-shift analysis; fold 1 separately as the epidemic
case.

### 6.2 Baselines

| Baseline | Purpose |
|---|---|
| persistence (`ŷ = y_t`) | the bar that matters; already the strongest naive model |
| seasonal naive (`ŷ = y` at `period_id − 52`) | quantifies how little pure seasonality is worth here |
| `gru_only` (identity adjacency) | isolates the contribution of the graph |
| `gcn_gru` on `v0` vs `v1` | isolates the contribution of hand-specified climate lags |

### 6.3 Milestones

1. **Full baseline sweep** — 2 feature variants × 2 models × 9 folds × 3 seeds =
   108 training runs, approximately 2 hours on CPU. This is the blocking step:
   every claim about the graph, about `v0` vs `v1`, and about beating persistence
   is currently unproven at 9-fold scale.
2. **Window-geometry sweep** — `L ∈ {12, 26}`, `h ∈ {1, 2, 3, 4}`. The cheapest
   possible test of whether lag structure is being missed.
3. **Per-district breakdown**, with attention to Jaffna.
4. Contributions §5.1 → §5.4, each measured as a single change against the frozen
   baseline.

### 6.4 Resource requirements

| Resource | Requirement |
|---|---|
| Compute | CPU only. PyTorch 2.13.0 CPU build; no GPU, no PyTorch Geometric. The full 108-run baseline sweep is ~2 hours on a commodity laptop. |
| Model size | 8,193 trainable parameters at `hidden=32` |
| Storage | < 1 GB. Raw ERA5/CHIRPS daily extracts dominate; the model tensors are a few MB. |
| Data access | Google Earth Engine account (free, for ERA5-Land and CHIRPS extraction); all other sources are public downloads |
| Software | Python 3.13, pandas, numpy, geopandas, torch. Two requirement files: the data pipeline installs without a deep-learning stack. |
| Data debt to clear | ERA5 extraction ends 2026-04-26, leaving periods 1010–1012 without climate. Costs the current folds nothing (fold 9 ends at period 993) but blocks a future 2026 fold. |

The deliberately small footprint is a design position, not a limitation: a
forecasting tool intended for a national dengue control unit has to be
retrainable on the hardware that unit actually has.

---

## 7. Suggested paper structure (2 pages, ACM sigconf)

1. **Introduction / Problem motivation** — §1. Lead with the asymmetric accuracy
   requirement and the operational decision the forecast feeds.
2. **Dataset** — §2, with Fig. 1 (geography + graph) and Fig. 2 (national
   series). Include the "at a glance" table if space allows; otherwise fold the
   numbers into prose.
3. **Baseline model** — §3.1–3.4 condensed. Keep the architecture block, the
   residual-target equation, and the parameter count.
4. **Experimental results** — §4.1 summary table and §4.2 preliminary table.
   State the fold-1 failure explicitly.
5. **Proposed contribution** — §5, four numbered items with success criteria.
6. **Experimental plan** — §6.1–6.4, compressed to a short table plus a
   paragraph.
7. **References.**

---

## 8. References

Verify each against the course's citation requirements before submitting. These
four are the ones the work directly depends on:

- Muñoz-Sabater, J. et al. *ERA5-Land: a state-of-the-art global reanalysis
  dataset for land applications.* Earth System Science Data, 2021.
- Funk, C. et al. *The climate hazards infrared precipitation with stations — a
  new environmental record for monitoring extremes.* Scientific Data, 2015.
  (CHIRPS)
- Kipf, T. N. and Welling, M. *Semi-Supervised Classification with Graph
  Convolutional Networks.* ICLR, 2017.
- Cho, K. et al. *Learning Phrase Representations using RNN Encoder–Decoder for
  Statistical Machine Translation.* EMNLP, 2014. (GRU)

Also cite: the GADM database (v4.1) for administrative boundaries, and the
`denguedatahub` Sri Lanka weekly series for the case data.

**Still to add by the author:** domain references on dengue transmission dynamics
and climate drivers, and any prior work on dengue forecasting in Sri Lanka
specifically. These have not been surveyed in this repository and should not be
invented — a related-work sentence with two or three real citations is worth more
than a fabricated survey.

---

## 9. Claims that may NOT be made

Read this section before writing the results paragraph.

1. **Do not cite a 9-fold GCN+GRU result.** The full sweep has not been run. Only
   fold 8 and fold 1 have trained-model numbers, at a single seed, on `v1` only.
2. **Do not use the "+44.3%" figure** that appears in
   `results/models/baseline_report.md`. That report is a stale artifact of a smoke
   run: it compares a fold-8 model score (9.15) against a *seven-fold*
   persistence average (16.42). On fold 8 persistence scores 10.78, so the true
   margin is 15.1%, not 44%. The correct fold-matched margins are in §4.2.
3. **Do not claim the graph helps.** At one seed `gru_only` is ahead of
   `gcn_gru`. The honest statement is that the graph's contribution is not yet
   decidable.
4. **Do not claim the baseline beats persistence overall.** It beats persistence
   on one ordinary fold and loses heavily on the epidemic fold.
5. **Do not present the §5 targets as expected results.** They are success
   criteria for work not yet done.
6. The COVID folds (4, 5) have no trained-model numbers at all.

Everything in §1, §2, §3, §4.1, §4.3 and §6 is measured or specified and can be
stated without qualification.

---

## Related documents

- [`docs/baseline.md`](baseline.md) — the baseline in full, including the work plan
- [`docs/model_tensors.md`](model_tensors.md) — tensors, folds, adjacency
- [`docs/climate_dataset_schema.md`](climate_dataset_schema.md) — the climate pipeline
- [`docs/reporting_calendar.md`](reporting_calendar.md) — the chronological key
- [`notebooks/2.eda.ipynb`](../notebooks/2.eda.ipynb) — the EDA figures
- [`results/models/naive_baseline_report.md`](../results/models/naive_baseline_report.md)
- [`results/models/folds_report.md`](../results/models/folds_report.md)

### Delay-Encoded Learnable-Prior Hierarchical Interaction GNN for Dengue Early Warning in Sri Lanka

**Baseline paper being extended:** Weng et al., *Graph Representation Learning for Dengue Forecasting*, IEEE BigData 2024
**Target:** conference-style research paper + reproducible codebase
**Planned duration:** 12 working weeks
**Compute envelope:** Google Colab (free tier) / Kaggle / personal laptop — no cluster required

---

## 0. How to read this document

Each phase contains:

| Section | Meaning |
|---|---|
| **Goal** | The single sentence that justifies the phase existing |
| **Tasks** | Numbered, executable steps |
| **Resources** | Data, libraries, papers, tools you must have in hand *before* starting |
| **Deliverables** | Concrete artifacts (files, tables, figures) that must exist at the end |
| **Exit gate** | The condition that must be true before moving on. Do not skip these. |
| **Risks** | What usually goes wrong here, and the fallback |

**Golden rule for the whole project:** every number that will ever appear in the paper is produced by a script that writes to `results/*.csv`, never by a number copied out of a notebook cell. Notebooks are for looking; scripts are for recording.

---

## 1. The contribution in one paragraph (keep this pinned)

> The baseline paper hand-crafted two objects that should have been learned — the **feature delay** (a single global shift applied to all 25 districts) and the **graph** (a binary adjacency matrix drawn by hand by team members) — and produced only point forecasts with no uncertainty. We replace all three: (A) a **learnable distributed-lag encoder** that infers a smooth, per-district, per-feature delay kernel over a 26-week history; (B) a **multiplex domain-informed graph** with contiguity, learned-exponent gravity mobility, climate-similarity, and adaptive relation channels fused by node-wise gates; (C) a **Negative Binomial probabilistic head** trained on incidence per 100k, enabling outbreak-alarm evaluation with lead time and false-alarm rate. We evaluate under a walk-forward protocol that places the 2017 epidemic in the test set, across horizons up to 12 weeks, against naive baselines the original work omitted.

Three components → three of the rubric's contribution categories (representation, architecture/graph, loss/training). Plus efficiency analysis. That is a complete contribution story.

---

## 2. Phase map at a glance

| Phase | Weeks | Name | Blocking? |
|---|---|---|---|
| 0 | 0–1 | Setup, literature lock-in, repo scaffolding | Yes |
| 1 | 1–2 | Data acquisition & the rebuilt dataset | Yes |
| 2 | 2–3 | Evaluation harness + naive/simple baselines | Yes — **hard gate** |
| 3 | 3–4 | Mandatory rubric baseline + reproduction of prior GNNs | Yes |
| 4 | 5 | Component A — Learnable distributed-lag encoder | No |
| 5 | 6 | Component B — Multiplex graph module | No |
| 6 | 7 | Component C — Probabilistic head & loss | No |
| 7 | 8–9 | Ablations, hyperparameter study, computational analysis | Yes |
| 8 | 10 | Interpretability & figure production | No |
| 9 | 10–12 | Paper writing, internal review, submission | Yes |
| 10 | 12 | Artifact release & reproducibility check | No |

Phases 4, 5, 6 are additive and independently testable. If time collapses, ship A + C and demote B to a smaller contribution — do **not** ship B alone, because the adaptive-graph literature will make it look derivative on its own.

---

# PHASE 0 — Setup, Literature Lock-In, Repo Scaffolding
**Weeks 0–1**

### Goal
Remove every possible source of "we didn't know that existed" surprise in week 8, and establish the engineering discipline that makes the later phases cheap.

### Tasks

**0.1 — Read the novelty-threat papers and write a one-page differentiation memo.**
This is not optional and it is not busywork. Your defensible novelty is a *combination*, and a reviewer's first move is to ask "isn't this MepoGNN?" You need a written answer before you write a line of code.

For each paper below, record in a shared table: what graph they use, whether it is learned, what the delay/lag handling is, what the loss is, what disease, and what the evaluation protocol is.

- **Cola-GNN** (Deng et al., CIKM 2020) — cross-location attention for ILI forecasting
- **EpiGNN** (Xie et al., ECML PKDD 2022) — transmission risk encoding
- **MepoGNN** (Cao et al., 2022/2023) — *closest threat*; learns a mobility graph coupled to a metapopulation model, COVID, real mobility data, no climate delay structure
- **Graph WaveNet** (Wu et al., IJCAI 2019) — adaptive adjacency, dilated causal convolutions
- **AGCRN** (Bai et al., NeurIPS 2020) — node-adaptive parameter learning
- **MTGNN** (Wu et al., KDD 2020) — graph learning layer
- **DLinear / NLinear** (Zeng et al., AAAI 2023) — the "are transformers effective" challenge; a one-layer linear model is a genuinely dangerous baseline
- **EpiLearn** (2024) — epidemic modelling toolkit; check whether you can reuse their evaluation code

Write the memo as three sentences: *X does P but not Q. We do P and Q. The reason Q matters for vector-borne climate-driven disease is R.*

**0.2 — Decide and freeze the tech stack.** Recommended: Python 3.11, PyTorch 2.x, PyTorch Geometric + PyTorch Geometric Temporal (for reproducing prior work only), pandas, geopandas, xarray, scikit-learn, statsmodels, matplotlib. Pin versions in `requirements.txt` on day one — PyG/PGT version drift is a well-known time sink.

**0.3 — Create the repository skeleton.**

```
dengue-ews/
├── configs/            # YAML per experiment; every run reads a config, no hard-coded numbers
├── data/
│   ├── raw/            # untouched downloads, never edited
│   ├── interim/        # per-source cleaned
│   └── processed/      # the final panel + graph tensors
├── src/
│   ├── data/           # loaders, splitters, scalers
│   ├── graphs/         # builders for each relation channel
│   ├── models/         # baselines/, delphi/
│   ├── losses/
│   ├── eval/           # metrics, DM test, alarm evaluation
│   └── viz/
├── scripts/            # run_baselines.py, run_experiment.py, make_figures.py
├── results/            # CSVs only — the paper reads from here
├── notebooks/          # exploration only, never a source of paper numbers
├── paper/
└── requirements.txt
```

**0.4 — Set up experiment logging.** A `results/runs.csv` with one row per run: `run_id, config_hash, model, seed, fold, horizon, mae, rmse, mase, crps, train_seconds, params, peak_mem_mb, git_sha`. Everything else in the project reads this file. Weights & Biases free tier is fine as an addition, but the CSV is the source of truth because it survives the account expiring.

**0.5 — Set the global seed policy.** Seeds `{0,1,2,3,4}` minimum for headline results, 10 if runtime allows. Seed torch, numpy, and Python random; set `torch.use_deterministic_algorithms(True)` where possible and record when it isn't.

**0.6 — Agree the division of labour** (assuming a group). A sensible split: one owner for data/graph construction, one for model implementation, one for the evaluation harness and statistics, rotating on writing. The evaluation harness owner should *not* be the model owner — that separation is what keeps you honest.

### Resources
- Semantic Scholar / arXiv for the papers above; Connected Papers to check you haven't missed a 2025 entrant in epidemic GNNs
- Zotero or a shared spreadsheet for the differentiation table
- GitHub private repo + a shared Google Drive folder for the raw data mirror (Colab mounts Drive directly, which is the least painful way to avoid re-downloading)
- Overleaf project created now with the IEEE conference template loaded

### Deliverables
- `paper/related_work_notes.md` with the differentiation memo
- Working repo skeleton with `requirements.txt` and a smoke-test script that imports everything and prints versions
- `results/runs.csv` header committed

### Exit gate
You can state, in one sentence and without hedging, what your paper does that MepoGNN, Graph WaveNet, and EpiGNN do not.

### Risks
- *You discover a 2025/2026 paper that does exactly this.* Better now than in week 10. The fallback is to lean harder on the delay kernel + the outbreak-alarm evaluation protocol, which is under-served in the literature regardless.
- *PyG installation on Colab breaks.* Keep a known-good install cell in a gist; PGT is only needed for reproducing baselines, so your own model should have zero PGT dependency.

---

# PHASE 1 — Data Acquisition & the Rebuilt Dataset
**Weeks 1–2**

### Goal
Produce one clean, versioned, documented panel: **25 districts × ~600+ weeks × K features**, extended beyond the original 2013–2022 window, with population attached and every provenance decision written down.

### Tasks

**1.1 — Epidemiological data.** Sri Lanka's Epidemiology Unit (`epid.gov.lk`) publishes Weekly Epidemiological Reports with district-level notifiable disease counts, and the National Dengue Control Unit publishes monthly/annual district distributions. Target coverage **2010 → latest available**, which gives you more history *and* a post-COVID regime the original paper never saw.

- Write a scraper that downloads the WER PDFs and parses the dengue row per district per week. Expect table-layout drift across years — build the parser to fail loudly rather than silently mis-assign a column.
- Cross-check parsed annual totals against the published annual summaries. Any district-year off by more than a few percent gets manually inspected.
- Record known reporting artifacts explicitly: the 2017 epidemic, COVID-era (2020–21) suppression and reporting disruption, and any week where the report was not published.

**1.2 — Meteorological data.** Use **NASA POWER** rather than EarthData for the core weather variables — it is a plain REST API, no authentication, daily resolution, queried by lat/lon.

- Variables: `T2M`, `T2M_MIN`, `T2M_MAX`, `RH2M`, `PRECTOTCORR`, and optionally `ALLSKY_SFC_SW_DWN`.
- Query strategy: do **not** query only the district centroid. Sample a small grid of points inside each district polygon (e.g. 5–15 points depending on area) and take the areal mean. Centroid-only sampling is a real weakness in this kind of work and costs you nothing to fix.
- Aggregate daily → epidemiological week (align to the same week definition the WER uses; document which).

**1.3 — Vegetation / land surface data.** MODIS NDVI (MOD13Q1/MOD13A2) and LST (MOD11A2) via **NASA AppEEARS** (point or area sampling, no coding needed for extraction) or Google Earth Engine if someone on the team already has an account. Composite periods are 8/16-day — resample to weekly with interpolation and record the method.

**1.4 — Static / socio-economic layers.**
- District polygons and centroids: **GADM** level 2 for Sri Lanka (same source as the original paper, so the comparison is apples-to-apples).
- Population by district and year: Department of Census and Statistics Sri Lanka; interpolate intercensal years. **You need this** — it drives both the gravity graph and the per-100k target.
- District areas from the shapefile, and optionally an urbanisation proxy (population density, or built-up fraction from a global land-cover product).

**1.5 — Assemble the panel.** Output a single tidy parquet: `district_id, iso_week_start, cases, population, [weather features], [ndvi/lst features]`. Add derived columns *only* if mechanistically motivated: degree-week accumulations, consecutive dry/wet week counters, extreme-rain flags. Keep raw variables too — the lag encoder needs unprocessed series.

**1.6 — Missingness policy, written down.** Decide and document: forward-fill limits, interpolation method for satellite gaps, and what happens to weeks with no epidemiological report. Never silently zero-fill case counts — a missing report is not zero cases, and conflating them will corrupt exactly the outbreak periods you care about.

**1.7 — Write the dataset card.** One page: sources, licences, coverage, resolution, known artifacts, missingness rates per variable, and a table of summary statistics. This becomes Section "Dataset Description" of the paper almost verbatim and is what makes the dataset itself a citable contribution.

**1.8 — Exploratory analysis (short).** Per-district case time series, seasonality by climate zone, correlation-vs-lag curves per district for the key drivers. That last plot is important: it is the *empirical motivation* for Component A, and it should visibly show different districts peaking at different lags.

### Resources
| Need | Source |
|---|---|
| Dengue case counts | `epid.gov.lk` Weekly Epidemiological Reports; National Dengue Control Unit |
| Weather (daily, no auth) | NASA POWER REST API |
| NDVI / LST | NASA AppEEARS (MODIS MOD13/MOD11) or Google Earth Engine |
| District polygons | GADM v4 level 2, Sri Lanka |
| Population | Dept. of Census & Statistics, Sri Lanka |
| Libraries | `requests`, `pdfplumber`/`camelot` (PDF tables), `pandas`, `geopandas`, `shapely`, `xarray`, `rioxarray`, `pyarrow` |

### Deliverables
- `data/processed/panel.parquet` + `data/processed/panel_schema.md`
- `paper/dataset_card.md`
- 4–6 EDA figures, including the per-district correlation-vs-lag curves
- Scrapers and builders committed and re-runnable end to end

### Exit gate
`python scripts/build_dataset.py` reproduces the panel from `data/raw/` with no manual steps, and the annual totals match published figures.

### Risks
- *WER PDFs are inconsistent or partially unavailable.* Fallback: reconstruct from monthly NDCU district tables and interpolate to weekly, clearly flagged, or restrict coverage to years that parse cleanly. Losing two years is survivable; a silently wrong parse is not.
- *AppEEARS turnaround is slow.* Submit those requests in week 1, day 1. They queue.
- *Week alignment mismatch* between epi weeks and calendar weeks. Pick one convention, apply everywhere, state it in the paper.

---

# PHASE 2 — Evaluation Harness + Naive Baselines
**Weeks 2–3 · HARD GATE**

### Goal
Build the measuring instrument *before* the thing being measured, and establish the floor that the original paper never reported. This phase is where you most decisively beat the prior work, and it is cheap.

### Tasks

**2.1 — Implement the splitter.** Walk-forward (expanding-window) origins, not one 70/30 split:

| Fold | Train | Test |
|---|---|---|
| 1 | 2010–2015 | 2016 |
| 2 | 2010–2016 | **2017 (epidemic)** |
| 3 | 2010–2017 | 2018 |
| 4 | 2010–2018 | 2019 |
| 5 | 2010–2019 | 2020 (COVID) |
| 6 | 2010–2020 | 2021 (COVID) |
| 7 | 2010–2021 | 2022 |
| 8 | 2010–2022 | 2023+ |

Rules: a validation slice is carved from the **end of train**, never from test. The 2017 epidemic **must** appear as a test year — an EWS that has only ever been tested on quiet years is not an EWS. COVID folds are reported separately as a distribution-shift analysis rather than averaged in silently.

**2.2 — Scaler discipline.** Fit every scaler on training data only, per fold. Prefer per-node normalisation or a per-100k target over a single global MinMax — a global scaler makes the loss almost entirely about Colombo. Write a unit test that asserts no test-fold statistic leaks into the scaler.

**2.3 — Implement the metrics module.**
- **MAE, RMSE** — for comparability with the prior paper
- **MASE** — scale-free; the correct primary metric across 25 districts spanning orders of magnitude. Denominator = in-sample naive forecast error, computed on the training fold.
- **CRPS** — for the probabilistic head in Phase 6. Sample-based estimator is fine and easiest to keep correct.
- **Outbreak-alarm metrics** — define an endemic-channel threshold per district (e.g. rolling multi-year mean + 2 SD by week-of-year), then compute **sensitivity, specificity, false alarm rate, and mean lead time**. These are what a Ministry of Health actually cares about, and their absence in the prior work is a clean gap.
- **Peak error** — MAE restricted to weeks above the district's 90th percentile, reported separately. This is where the prior models visibly fail.

**2.4 — Implement the Diebold–Mariano test.** Regress the loss differential between two models on a constant using OLS with HAC (Newey–West) standard errors; the t-statistic on the constant is the DM statistic. Roughly 30 lines with `statsmodels`. Apply it against the *strongest* baseline, not the weakest.

**2.5 — Run the naive baselines.**
- **Persistence**: ŷ(t+h) = y(t)
- **Seasonal naive**: ŷ(t+h) = y(same week, last year)
- **Historical mean by week-of-year**
- **DLinear / NLinear** — one linear layer over the lookback; brutally strong on short horizons
- **ARIMAX, Random Forest, XGBoost** — re-run on *your* dataset and *your* splits, so the comparison is fair

Run all of them at horizons **h ∈ {1, 2, 4, 8, 12}** weeks.

**2.6 — Write the motivation finding.** Compare your naive baseline numbers against the prior paper's published GNN errors. If persistence is competitive with their best model at h=3, that is a legitimate, publishable observation about the difficulty of the original evaluation setup, and it becomes a paragraph in your introduction. Handle it with care and courtesy in the writing — the point is that the *protocol* was under-specified, not that the authors were careless.

### Resources
- `statsmodels` (ARIMAX, OLS+HAC for DM), `scikit-learn`, `xgboost`
- `properscoring` or `scoringrules` for CRPS (or implement the sample-based estimator)
- `sktime` for MASE reference implementation if you want a cross-check
- WHO endemic-channel / epidemic-threshold methodology for the alarm definition — cite whichever definition you adopt

### Deliverables
- `src/eval/` fully tested, with unit tests for leakage and for each metric on a hand-computed toy example
- `results/baselines.csv` — every naive and classical baseline × 8 folds × 5 horizons × 5 seeds
- Table 1 draft of the paper

### Exit gate
**Nothing in Phase 3+ starts until this table exists.** If you cannot beat seasonal naive at h=12 later, you need to know that in week 3, not week 9.

### Risks
- *The alarm threshold definition is contentious.* Pick one, define it precisely, and show a sensitivity analysis over the multiplier (1.5 SD / 2 SD / 2.5 SD) in the appendix. Defining it clearly is worth more than defining it perfectly.
- *Temptation to skip the naive baselines because they feel beneath a deep-learning paper.* Reviewers at any decent venue check for exactly this. It is your cheapest credibility.

---

# PHASE 3 — Mandatory Rubric Baseline + Reproduction of Prior GNNs
**Weeks 3–4**

### Goal
Satisfy the course's mandatory baseline requirement (graphs → GCN/GAT) and establish a *fair, same-data, same-split* reproduction of the models the original paper used, so your improvement claim is unambiguous.

### Tasks

**3.1 — Build the reference spatio-temporal baseline.** A straightforward **GCN + GRU** and **GAT + GRU** encoder–decoder over the hand-drawn adjacency from the original paper (reproduce their matrix from the shapefile + their description, and record it as `A_manual`). This is the rubric's mandatory baseline and also your "no fancy graph" control.

**3.2 — Reproduce A3TGCN, STGAT, ASTGCN, DCRNN, AAGCN** using PyTorch Geometric Temporal, on your dataset and splits. Expect their numbers to change substantially — that is the point, and it should be reported plainly as a protocol effect rather than framed as a failure of the original work.

**3.3 — Add the adaptive-graph competitors: Graph WaveNet, AGCRN, MTGNN.** These are the strongest published rivals to your Component B. If you skip them, a reviewer will ask, and you will have no answer. Use the authors' reference implementations where possible; log any deviations.

**3.4 — Standardise the training loop.** One trainer, one config schema, early stopping on a validation slice from the end of train, identical optimiser budget across models. Any model-specific deviation goes in a table in the appendix.

**3.5 — Build the model registry.** `src/models/__init__.py` maps a config string to a constructor, so `run_experiment.py --model X --fold f --horizon h --seed s` works for every model including the ones you haven't written yet.

**3.6 — Profile everything now, not later.** While each model is fresh, record parameter count, training seconds/epoch, peak memory, and inference latency per forward pass into `runs.csv`. Collecting this retroactively in week 9 is miserable.

### Resources
- PyTorch Geometric, PyTorch Geometric Temporal (A3TGCN, STGAT, ASTGCN, DCRNN, AAGCN)
- Reference repos for Graph WaveNet, AGCRN, MTGNN (author GitHub releases)
- `torchinfo` or `thop` for parameter/FLOP counts; `torch.cuda.max_memory_allocated` / `tracemalloc` for memory
- Colab free-tier GPU; 25 nodes × ~700 weeks is small enough that CPU is also viable

### Deliverables
- `results/prior_models.csv` — all reproduced models × folds × horizons × seeds, with compute columns filled
- A short reproduction note documenting every deviation from the original papers' settings

### Exit gate
Every prior model runs end-to-end from a config file, and you have a defensible "strongest baseline" identified for DM testing.

### Risks
- *PGT models assume a single-feature or fixed-shape input.* Budget time for adapter code; the original paper's own DCRNN observation (better with case data alone) is a symptom of this.
- *Reference repos are stale.* Fallback: reimplement the graph-learning layer only (it is short) rather than the full architecture, and say so.

---

# PHASE 4 — Component A: Learnable Distributed-Lag Encoder
**Week 5**

### Goal
Replace the original paper's single global feature shift with a smooth, learned, **per-district, per-feature** delay kernel over a long history — the component that most directly turns their own correlation analysis into architecture.

### The mechanism

For district *i*, feature *k*, at time *t*, over a lookback of L = 26 weeks:

```
x̃ⁱₖ,ₜ = Σ_{τ=0}^{L-1} wⁱₖ(τ) · xⁱₖ,ₜ₋τ
```

Do **not** learn L × K × N free weights (that is 26 × 20 × 25 = 13,000 parameters on 6,000 samples — guaranteed overfitting). Instead:

1. Define **B ≈ 4 smooth basis kernels** over τ, each a Gamma or Gaussian shape with learnable centre μ_b and width σ_b, normalised to sum to 1.
2. Give each district a low-dimensional embedding **eᵢ ∈ ℝ^d** (d ≈ 8).
3. Mixture coefficients: `αⁱₖ = softmax(eᵢᵀ W_k)` over the B bases.
4. `wⁱₖ = Σ_b αⁱₖ,b · basis_b`.

Parameter cost: 2B (basis shape) + N·d (embeddings) + K·d·B (projection) — a few hundred parameters. That is the efficiency argument too: 26-week dependence captured without unrolling 26 recurrent steps.

### Tasks

**4.1 — Implement the basis kernels** with a shape constraint that keeps them causal and non-negative. Softplus the width; clamp centres to [0, L−1].

**4.2 — Initialise μ near the empirically observed lags** (≈12 weeks for precipitation, ≈17 for min NDVI, ≈0–2 for autoregressive case features, per the original paper's own analysis). Good initialisation here is the difference between converging in 20 epochs and not converging.

**4.3 — Unit-test on synthetic data.** Generate a series where the target is a known delayed convolution of an input, with different delays per node. Assert the module recovers the delays to within a week or two. **Do not skip this** — it is the single highest-value test in the project, and it is also the figure that sells the paper.

**4.4 — Drop the encoder in front of the Phase 3 GCN+GRU baseline** and measure. This is component A in isolation: same graph, same head, only the delay handling changes. Report the delta at every horizon.

**4.5 — Ablate against:** no shift at all; the original paper's fixed global shift; a plain dilated causal CNN over the same lookback; a free (unconstrained) 26-tap filter. The last one is the important control — it shows the *smooth basis parameterisation* is doing work, not just the longer lookback.

**4.6 — Extract the learned kernels** and save them to `results/learned_lags.csv` for the Phase 8 figures.

### Resources
- PyTorch only; no extra dependencies
- The Phase 1 per-district correlation-vs-lag curves as the empirical prior and the sanity check
- Background reading on distributed-lag non-linear models (DLNM) in epidemiology — useful framing for Related Work and a good citation to show domain awareness

### Deliverables
- `src/models/delphi/lag_encoder.py` + tests
- `results/ablation_lag.csv`
- Learned kernel arrays saved per fold

### Exit gate
The synthetic-data test passes, and Component A shows a measurable improvement over the fixed-shift control at h ≥ 4.

### Risks
- *Kernels collapse to τ=0* (the model ignores the delay). Diagnose with a learning-rate check on the kernel parameters specifically; consider a separate, higher LR for μ, or a mild entropy regulariser on the mixture weights.
- *Improvement only appears at long horizons.* That is fine and expected — make it the story rather than hiding it. "The delay structure matters more as the forecast horizon grows" is a good headline.

---

# PHASE 5 — Component B: Multiplex Domain-Informed Graph
**Week 6**

### Goal
Replace one hand-drawn binary matrix with several *derivable* relation channels, fused by learned per-node gates — and be able to show which relation the model actually relies on.

### The relation channels

| Channel | Definition | What it answers |
|---|---|---|
| A⁽¹⁾ contiguity | Shared borders from GADM | Objective replacement for the hand-drawn matrix |
| A⁽²⁾ gravity mobility | PᵢPⱼ / d_{ij}^θ, **θ learned** (one scalar) | Directly addresses "we had no travel data" |
| A⁽³⁾ climate similarity | From data: correlation of rainfall/temp series, or k-means on climate normals | Recovers Sri Lanka's wet/dry/intermediate zones without external labels |
| A⁽⁴⁾ adaptive | softmax(ReLU(E₁E₂ᵀ)), learned end-to-end | The standard learned-graph competitor, included so you dominate it rather than ignore it |
| A⁽⁵⁾ dynamic coupling *(optional)* | Rolling lagged cross-correlation of case series | Time-varying epidemic linkage |

**Fusion:** node-wise learned gates α_{r,i} (softmax over relations per node), or a small attention over relations. Node-wise matters: Colombo should be allowed to weight mobility while a remote district weights contiguity.

### Tasks

**5.1 — Build each adjacency as a standalone, tested function** in `src/graphs/`, each returning a normalised N×N tensor with a saved provenance record.

**5.2 — Critical leakage check on A⁽³⁾ and A⁽⁵⁾.** Any graph derived from data must be computed using **training-fold data only**, and A⁽⁵⁾ must be computed strictly causally (only information available at time t). Write an explicit test. This is the most likely place for a subtle leak that invalidates your headline result.

**5.3 — Implement the gated multiplex layer.** Message-pass over each relation independently, then combine with the node-wise gates. Keep it thin — the contribution is the relation set and the gating, not a baroque propagation rule.

**5.4 — Ablate:** each channel removed individually; each channel *alone*; identity adjacency (no graph at all — a crucial control, since the original paper never tested whether the graph helps versus 25 independent models); the original hand-drawn matrix; uniform (non-gated) fusion.

**5.5 — Learn and record θ** for the gravity channel. A learned distance exponent with a plausible value (typically ~1–2 in mobility literature) is a nice, concrete, quotable result.

**5.6 — Save the learned gates** per node for the Phase 8 map figure.

### Resources
- `geopandas` / `libpysal` for contiguity (queen vs rook — pick one and state it)
- District population and centroid coordinates from Phase 1
- Gravity-model literature for mobility surrogates (cite a standard reference for the functional form)
- `scipy.spatial` for distance matrices

### Deliverables
- `src/graphs/*.py` with tests including the leakage test
- `results/ablation_graph.csv`
- Learned gate and θ values saved per fold

### Exit gate
The multiplex graph beats both the hand-drawn matrix and the pure adaptive graph, and you can show *which* relation carries the improvement.

### Risks
- *All gates converge to the adaptive channel*, making the domain priors look useless. This is still a reportable finding, but check first whether it is a low-data artifact; try a sparsity or entropy penalty on the gates, and test with fewer training years.
- *Marginal gains over the adaptive graph alone.* Fall back to positioning B as an interpretability and low-data-regime contribution: show it wins when training data is short, which is the realistic setting for a new country deployment. That framing is honest and still valuable.

---

# PHASE 6 — Component C: Probabilistic Head & Peak-Aware Loss
**Week 7**

### Goal
Turn point predictions into an actual early-warning signal: a distribution, an exceedance probability, and a loss that stops systematically under-predicting peaks.

### Tasks

**6.1 — Change the target to incidence per 100k.** Convert back to counts for reporting. This stops the loss being dominated by high-population districts and makes cross-district comparison meaningful.

**6.2 — Implement the Negative Binomial head.** Output (μ, r) per node per horizon; softplus both; train on NB negative log-likelihood. Guard numerically — `torch.lgamma` and a floor on r. Poisson is the degenerate fallback if NB training is unstable, but over-dispersion in dengue counts is real and NB is the right default.

**6.3 — Optional peak-aware term.** A small asymmetric penalty on under-prediction, or a term on the first difference (trajectory slope) to reward getting the *rise* right. Keep the weight λ small and ablate over it — if it doesn't help, drop it and say so. A negative result here is fine and takes one sentence.

**6.4 — Calibration checks.** PIT histograms and reliability diagrams per district group. A well-calibrated interval is a result in its own right and something no baseline in the original paper can offer.

**6.5 — Wire the alarm evaluation.** P(cases > endemic threshold) from the NB distribution → alarm at a chosen probability cut. Produce sensitivity / false-alarm-rate / lead-time curves as the alarm cut varies. This is your most operationally persuasive figure.

**6.6 — Compare against baseline uncertainty.** Quantile-regression or conformal intervals wrapped around the best deterministic baseline, so the probabilistic comparison is fair rather than "we have intervals and they don't."

### Resources
- `torch.distributions.NegativeBinomial` (mind the parameterisation — total_count/probs vs mean/dispersion; write a test against a known likelihood value)
- `properscoring` / `scoringrules` for CRPS
- MAPIE or a short conformal implementation for the baseline intervals
- Background: probabilistic forecasting evaluation (CRPS, PIT, sharpness-subject-to-calibration)

### Deliverables
- `src/losses/nb_nll.py`, `src/models/delphi/heads.py` with tests
- `results/ablation_loss.csv`, calibration figures, alarm curves

### Exit gate
NB head matches or beats MSE training on MAE/RMSE **and** provides calibrated intervals; peak-region MAE improves measurably.

### Risks
- *NB NLL is unstable early in training.* Warm up with MSE for a few epochs, then switch; or initialise r large (near-Poisson) and let it shrink.
- *Intervals are calibrated but very wide.* Report sharpness alongside calibration and discuss it honestly — a wide, honest interval is more useful to a health ministry than a narrow, wrong point forecast, and saying so is a good discussion paragraph.

---

# PHASE 7 — Ablations, Hyperparameter Study, Computational Analysis
**Weeks 8–9**

### Goal
Produce every table the rubric demands and every table a reviewer will ask for, all from scripts.

### Tasks

**7.1 — Full-system ablation.** Additive (baseline → +A → +A+B → +A+B+C) *and* leave-one-out (full → −A, −B, −C). Both, because they answer different questions.

**7.2 — Component-internal ablations.** Aggregate what Phases 4–6 already produced: number of basis kernels B ∈ {1,2,4,8}; lookback L ∈ {8,13,26,52}; each graph relation; loss variants; per-100k vs raw counts; node embedding dimension d.

**7.3 — Hyperparameter study.** Hidden dim, layers, learning rate, weight decay, dropout, batch size. Use a small random or Bayesian search on the **validation slice only**, then report the sensitivity curves — a reviewer wants to see robustness, not just the winning cell. `optuna` is the least painful tool here.

**7.4 — Computational analysis (a full section, with numbers).** For every model: parameter count, FLOPs or MACs per forward pass, seconds/epoch, epochs to convergence, total training wall-clock, inference latency per week per district, peak memory. Run on identical hardware and say what it was. The original paper's compute discussion is qualitative and empty — this is a guaranteed win.

**7.5 — Data-efficiency curve.** Train on 3, 5, 7, all available years and plot error versus training-set size. If the domain priors in Component B help most in the low-data regime, this figure proves it and generalises your claim to "any country starting a new EWS."

**7.6 — Statistical testing.** DM tests, full model versus strongest baseline, per horizon. Report the p-values in the main table. Multi-seed mean ± std everywhere.

**7.7 — Robustness checks.** Performance excluding COVID years; performance on the 2017 epidemic fold alone; per-district breakdown table (all 25); sensitivity to the alarm threshold multiplier.

### Resources
- `optuna` for search; `torchinfo`/`thop` for FLOPs; `psutil`/`tracemalloc`/`torch.cuda` for memory
- A single consistent machine for all timing runs — note the exact spec in the paper
- `results/` CSVs from every prior phase; `make_tables.py` converts CSV → LaTeX so no number is ever transcribed by hand

### Deliverables
- `results/ablation_full.csv`, `results/hparams.csv`, `results/compute.csv`, `results/per_district.csv`
- Auto-generated LaTeX for Tables 1–6

### Exit gate
Every table in the paper outline exists as a `.tex` file generated by a script from a CSV.

### Risks
- *Combinatorial explosion.* Vary one axis at a time from a fixed default configuration; state that default explicitly. A full grid is neither necessary nor expected.
- *Runs die mid-sweep on Colab.* Checkpoint after every run, make the runner resumable by checking `runs.csv` for an existing `run_id` and skipping it.

---

# PHASE 8 — Interpretability & Figure Production
**Week 10**

### Goal
Produce the two or three figures that make the paper memorable. Reviewers remember figures; this phase is disproportionately valuable relative to its cost.

### Tasks

**8.1 — The learned-lag map (headline figure).** Choropleth of Sri Lanka, one panel per key feature, coloured by the learned kernel centroid per district. If wet-zone and dry-zone districts separate visibly, that is the figure. Overlay climate-zone boundaries and let the reader make the connection.

**8.2 — Kernel shape plots.** Learned w(τ) curves for a handful of contrasting districts, with the empirical correlation-vs-lag curve from Phase 1 underneath for validation.

**8.3 — Relation-gate visualisation.** Map or stacked bar per district showing the learned mix over the five relation channels. Expect Colombo to look different from a remote district; if it does, say so.

**8.4 — Error-versus-horizon curves** for your model against the top three baselines, with the gap widening at long horizons.

**8.5 — Alarm-performance figure.** Lead time versus false alarm rate, with the operating point you would recommend to a health ministry marked.

**8.6 — Qualitative forecast plots** for 4–6 districts including the 2017 epidemic fold, with prediction intervals shaded. This is the direct visual answer to the original paper's Figures 4 and 6.

**8.7 — Architecture diagram** with tensor shapes annotated.

**8.8 — Figure hygiene.** Vector PDF, consistent fonts sized to be readable at IEEE column width, colourblind-safe palette, every axis labelled with units, captions that state the conclusion rather than describing the axes.

### Resources
- `matplotlib` + `geopandas` for choropleths; `cmcrameri` or ColorBrewer for colourblind-safe maps
- draw.io / Excalidraw / TikZ for the architecture diagram
- GADM shapefile from Phase 1

### Deliverables
- `paper/figures/*.pdf`, all regenerable via `make_figures.py`

### Exit gate
Someone outside the team can look at the learned-lag map and correctly explain what it shows in one sentence.

### Risks
- *The learned lags look like noise.* Check whether the kernels actually moved from initialisation; if the pattern genuinely isn't there, report it honestly as a negative interpretability result and lean the paper on the accuracy and calibration contributions instead. Do not massage the figure.

---

# PHASE 9 — Paper Writing, Review, Submission
**Weeks 10–12**

### Goal
A complete, correctly structured conference paper with every section the rubric requires.

### Required structure (from the project brief)

Abstract · Introduction · Related Work · Proposed Framework · Implementation Details · Experimental Setup (dataset overview, baselines) · Experiments (ablations, comparative analysis, hyperparameter tuning, computational analysis, additional experiments) · Discussion · Conclusion · References

### Writing order (do not write in reading order)
1. Figures and tables — finalised first, since they determine what you can claim
2. Experimental Setup and Experiments — the factual core, easiest to write
3. Proposed Framework — the mathematical description of A, B, C
4. Related Work — expand the Phase 0 differentiation memo
5. Introduction — write it once you know the results; state contributions as a bulleted list that maps 1:1 to your sections
6. Abstract — last, with the headline number in it
7. Discussion and Limitations — write these *properly*; the original paper's limitations section is its most credible part and yours should be too

### Tasks

**9.1 — Contribution bullets first.** Three or four bullets, each pointing at a section and a table. If a bullet has no table, either run the experiment or delete the bullet.

**9.2 — Number audit.** Every number in the text traced to a `results/*.csv` row. Do this as an explicit checklist pass, by a person who did not write the section.

**9.3 — Limitations section, honest and specific.** Single country; district-level resolution only (25 nodes); gravity mobility is a surrogate, not measured movement; reporting artifacts during COVID; no serotype data; no vector-surveillance data; results have not been prospectively validated.

**9.4 — Reproducibility statement.** Repo link, seeds, hardware, exact package versions, one command to reproduce the main table.

**9.5 — Internal review round.** Each member reviews as a hostile reviewer and writes an actual review with a score. Fix everything that a reasonable reviewer would flag.

**9.6 — Venue selection.** Decide early, because page limits and templates differ. Consider: IEEE BigData (the original venue — a natural and slightly pointed choice), regional IEEE conferences hosted in Sri Lanka, epidemiology/health-informatics venues, or an ML workshop track on AI for social impact. Check the deadline calendar in week 8, not week 12.

**9.7 — Format compliance pass.** Page limit, anonymisation if double-blind, reference formatting, figure resolution, supplementary material rules.

### Resources
- Overleaf + the official template for the chosen venue
- Grammarly or LanguageTool for a mechanical pass
- A reference manager exporting clean BibTeX (the original paper's bibliography has visible author-parsing errors — a good reminder to check yours)
- WikiCFP / conference deadline trackers for venue dates

### Deliverables
- Camera-ready-quality PDF within the page limit
- Internal review documents
- Submission confirmation

### Exit gate
Submitted, or ready to submit with a target date fixed.

---

# PHASE 10 — Artifact Release & Reproducibility
**Week 12**

### Tasks
1. Clean the public repo: no credentials, no large raw binaries, a `README` with a quickstart that actually runs on a fresh Colab instance.
2. Publish the processed dataset with its licence checked (verify the terms for the epidemiological source before redistributing; if in doubt, release the scraper and the derived features rather than the raw counts).
3. Archive with a DOI via Zenodo (free GitHub integration).
4. Do a **clean-machine reproduction test**: a team member who did not build the pipeline runs it from scratch on a new Colab session and reproduces Table 1. Fix whatever breaks.
5. Record model checkpoints for the headline configuration.

### Deliverables
Public repo, Zenodo DOI, dataset card, reproduction confirmed by an independent team member.

---

# Appendix A — Consolidated Resource List

### Data
| Resource | Use | Access |
|---|---|---|
| Sri Lanka Epidemiology Unit (`epid.gov.lk`) Weekly Epidemiological Reports | District weekly dengue counts | Public PDFs — scrape |
| National Dengue Control Unit | Monthly/annual district totals — cross-validation | Public |
| NASA POWER | Daily temperature, humidity, precipitation, radiation | Free REST API, no auth |
| NASA AppEEARS (MODIS MOD13/MOD11) | NDVI, land surface temperature | Free, NASA Earthdata login |
| Google Earth Engine | Alternative satellite extraction | Free for research, account needed |
| GADM v4 | District polygons, centroids, contiguity | Free download |
| Dept. of Census & Statistics Sri Lanka | District population, area | Public |

### Software
`python 3.11` · `pytorch` · `torch-geometric` · `torch-geometric-temporal` · `pandas` · `numpy` · `geopandas` · `shapely` · `libpysal` · `xarray` · `rioxarray` · `scikit-learn` · `statsmodels` · `xgboost` · `optuna` · `matplotlib` · `properscoring` · `pdfplumber` · `pyarrow` · `torchinfo` · `tqdm`

### Compute
Google Colab free tier (T4 when available) · Kaggle notebooks (30 GPU-hours/week, a useful overflow) · any personal machine — the models are small enough that CPU training is viable, which is itself worth stating in the paper.

### Key papers to have read before Phase 4
Graph WaveNet · AGCRN · MTGNN · DLinear · Cola-GNN · EpiGNN · MepoGNN · A3TGCN · STGAT · ASTGCN · DCRNN · plus distributed-lag non-linear model (DLNM) literature from epidemiology.

---

# Appendix B — Weekly Checkpoint Schedule

| Week | Must be true by Friday |
|---|---|
| 1 | Repo live; differentiation memo written; AppEEARS requests submitted |
| 2 | Panel dataset built and validated against published totals |
| 3 | **GATE:** naive + classical baseline table complete across all folds and horizons |
| 4 | GCN/GAT baseline + prior GNNs reproduced with compute metrics logged |
| 5 | Component A implemented, synthetic test passing, ablation run |
| 6 | Component B implemented, leakage test passing, ablation run |
| 7 | Component C implemented, calibration verified, alarm curves produced |
| 8 | Full ablation + hyperparameter sweep launched |
| 9 | All result tables final; DM tests done; compute section complete |
| 10 | All figures final; Experiments and Framework sections drafted |
| 11 | Full draft circulated; internal reviews returned |
| 12 | Submitted; repo public; DOI minted |

---

# Appendix C — Descoping Ladder

If you fall behind, cut in this order. Each rung below still yields a defensible paper.

1. Drop A⁽⁵⁾ dynamic coupling from the graph (least essential relation)
2. Drop the peak-aware loss term; keep the NB head
3. Drop the data-efficiency curve
4. Drop horizons {2, 8}, keep {1, 4, 12}
5. Drop MTGNN from the competitor set (keep Graph WaveNet and AGCRN)
6. Demote Component B to contiguity + gravity + adaptive only

**Never cut:** the naive baselines, the 2017 epidemic test fold, the multi-seed variance, the computational analysis section, or the full ablation of your own components. Those are what separate a paper that gets accepted from one that gets desk-rejected.

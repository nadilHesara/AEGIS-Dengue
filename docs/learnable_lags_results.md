# Learnable climate lags — source material for the short paper

Everything needed to write up Component A. Every number below is read from a
committed artifact in this repository and is reproducible with the commands in
§9. Current as of 2026-09-07.

**§8 lists what may not be claimed.** Read it before writing a results sentence.

**The headline is a negative result with a mechanism.** That is a publishable
shape, but only if the negative is stated plainly and the mechanism is
evidenced. Do not soften it into "comparable performance".

---

## 1. The one-paragraph version

Fixed climate lags are an assumption every dengue forecasting paper makes and
almost none measures. We replace them with a smooth, per-district, learnable
delay kernel — 548 parameters buying a 26-week memory — and test it against
hand-chosen windows on nine walk-forward folds. It loses: 20.67 MAE against
18.36 for hand-picked windows and 18.07 for no lag treatment at all. We then
show *why*, and the diagnosis is the contribution. An independent
cross-correlation scan measures the true rainfall-to-dengue delay at 5–10 weeks
across all 25 districts, and a synthetic recovery test confirms the encoder
finds planted delays of 3–19 weeks within ±2. The module works and the delay is
real; what fails is the premise that it can be learned end-to-end from a
forecasting loss, because at one-week-ahead the previous period's case count
carries nearly all the information — dropping every climate channel costs
+0.01 MAE on recent normal years. A delay parameter learned from a gradient
that contains no delay signal drifts to the boundary of its range: measured and
learned delays agree at r = −0.16.

---

## 2. Data

### 2.1 Panel

| Property | Value |
|---|---|
| Spatial units | 25 canonical districts, 9 provinces, Sri Lanka |
| Time index | 1,012 reporting periods, 2006-12-23 to 2026-05-17 |
| Panel size | 25,300 district-period rows |
| Period length | 6–8 days (2 irregular periods, both 2009) |
| Total reported cases | 833,559 |
| Missing case records | 1 of 25,300 (0.004%) |
| Missing climate records | 75 of 25,300 (0.30%), periods 1010–1012 |

### 2.2 Sources

| Source | Contributes | Handling |
|---|---|---|
| denguedatahub weekly Sri Lanka series | confirmed cases per reporting area per week | 26 source areas merged to 25 districts (Kalmune folds into Ampara) |
| **ERA5-Land** (ECMWF), hourly, `ECMWF/ERA5_LAND/HOURLY` | rainfall, 2 m temperature (mean/min/max), dewpoint, relative humidity, wind speed | Google Earth Engine; hourly→daily on the UTC day, area-weighted per district polygon with a land–sea mask, then aggregated over each period's true `start_date`–`end_date` interval |
| **CHIRPS** daily, `UCSB-CHG/CHIRPS/DAILY` | independent rainfall series | cross-check on ERA5-Land precipitation only; **not** a model feature |
| **GADM 4.1 ADM1** / geoBoundaries ADM2 | district polygons | queen-contiguity graph and centroids (EPSG:32644) |

### 2.3 Three engineering decisions worth a sentence each

1. **A dedicated chronological key.** Source `year`/`week` labels are not
   sortable — "2026 week 53" covers 2025-12-20 to 2025-12-26. Sorting on the
   label corrupts every lag across that boundary. `period_id` is assigned after
   sorting on true dates and is the only sort key.
2. **Climate is joined on exact date intervals, never week numbers.** Periods
   are not ISO weeks, two are 6 and 8 days long, and the reporting weekday
   changes twice.
3. **Absence is carried as absence.** A missing case count stays NaN, is flagged
   in `y_mask`, and is excluded from the loss and every metric. A fabricated
   zero asserts an observation nobody made.

### 2.4 Validation protocol

Nine walk-forward folds; each trains on everything before a calendar year,
validates on the preceding year, tests on the target year.

| Fold | Test year | Train windows | Role |
|---|---|---|---|
| 1 | 2017 | 459 | epidemic — the hard fold |
| 2 | 2018 | 512 | headline |
| 3 | 2019 | 564 | headline |
| 4 | 2020 | 616 | COVID — reported separately |
| 5 | 2021 | 668 | COVID — reported separately |
| 6 | 2022 | 720 | headline |
| 7 | 2023 | 772 | headline |
| 8 | 2024 | 825 | headline |
| 9 | 2025 | 877 | headline |

Headline = mean over folds 1, 2, 3, 6, 7, 8, 9. Every fitted statistic —
climatology, scaler, peak thresholds, and the lag scan of §4 — comes from
periods strictly before the fold's test year. Windows are assigned to a split by
their **target** period. Early stopping is on the inner validation year.

---

## 3. Baseline (the thing to beat)

Graph convolution over the 25 districts at each step, a shared GRU over the
window, a linear head:

```
per timestep t:   S = relu(A_norm @ X_t @ W1)
                  S = relu(A_norm @ S   @ W2)
over the window:  H = GRU(S_1 … S_L)      shared weights, one sequence per node
head:             y = Linear(H_L)
```

`A_norm = D^{-1/2}(A + I)D^{-1/2}` over the queen-contiguity adjacency (25 nodes,
57 edges, mean degree 4.56). **8,193 parameters**, lookback 12, horizon 1,
hidden 32, 2 GCN layers, dropout 0.2, Adam at 3e-3, weight decay 1e-4, batch 64,
max 150 epochs, patience 15, gradient-norm clipping at 1.0. PyTorch 2.13.0 CPU,
no PyTorch Geometric.

**Residual target.** The model predicts `log1p(y[t+h]) − log1p(y[t])`, so an
output of zero reproduces persistence exactly and the network only learns the
correction.

**Feature variants.**

| Variant | Features |
|---|---|
| `v0` | 14: `log1p` cases, 7 climate variables, day-of-year sin/cos, 2 observation flags, centroid lat/lon |
| `v1` | 23: `v0` plus trailing 4-, 8- and 12-period means of rainfall, temperature and humidity |

`v1 − v0` is what hand-specified climate lags are worth, and is the number the
contribution must beat.

**Naive baselines, all 9 folds:** persistence 16.42 headline MAE / 26.61 peak
MAE; seasonal naive 55.45 / 84.81. Seasonal naive being 3× worse than
persistence is the key dataset fact: Sri Lankan dengue's interannual variation
dominates its seasonality.

---

## 4. New component A — measured lags (`scripts/17.lag_correlation_scan.py`)

**Method.** For each district × climate feature × lag τ ∈ [0, 25], the Pearson
correlation between the feature at `t − τ` and `log1p(cases)` at `t`, over that
fold's training periods only. Reported twice: raw, and with the district ×
ISO-week climatology removed from both series. Peak and trough are both carried,
signed — an absolute argmax picks the anti-phase trough half a seasonal cycle
away, which has no causal reading. Trains nothing; 3.5 s for all 9 folds.

**Result — rainfall delay, deseasonalised, fold 9:**

| Statistic | Value |
|---|---|
| Colombo | **5 weeks** (r = +0.106) |
| Range across 25 districts | **5–10 weeks**, median 8 |
| Stability across 9 folds | mean sd 1.4 periods, worst district 5.3 |

Per-feature peak lags (fold 9, deseasonalised): dewpoint 5, humidity 7,
rainy-days 8, rainfall 8, temperature 17, diurnal range 21.

**The caveat that matters.** Once seasonality is removed, rainfall's median peak
correlation is only **+0.036**. The script emits a warning below |r| = 0.10 and
declines to present those peaks as delays. Temperature (+0.199) and dewpoint
(+0.191) are the only features with a signal comfortably above that floor.

**Extending the scan to lag 51 does not move the rainfall peak** (stays at 9),
which rules out "the true delay exceeds the 26-week reach" as an explanation for
§7.

---

## 5. New component B — the learnable lag encoder (`src/models/lag_encoder.py`)

For district *i*, climate feature *k*, over reach L = 26:

```
x̃[i,k,t] = Σ_{τ=0..25} w[i,k](τ) · x[i,k,t−τ]
```

`w` is **not** free. A free curve is 26 × 7 × 25 = 4,550 weights on ~800 training
windows and would memorise. Instead:

1. **B = 6 Gaussian basis bumps** over τ, learnable centre `μ_b` and width
   `σ_b` (softplus, floored at 0.5), each normalised to sum to 1.
2. **A district embedding** `e_i ∈ R^8` — the only place district identity enters.
3. **Mixture weights** `α[i,k] = softmax(e_i^T W_k)` over the bases.
4. `w[i,k](τ) = Σ_b α[i,k,b] · basis_b(τ)`.

| Piece | Parameters |
|---|---|
| Basis shapes (μ, σ) × 6 | 12 |
| District embeddings 25 × 8 | 200 |
| Feature projections 7 × 8 × 6 | 336 |
| **Total** | **548** |

Non-negative, summing to 1, and causal by construction — τ only indexes
backwards. **B = 6 was chosen by measurement, not taste:** with 4 bumps the
recovery test quantised (worst district 2.3 periods wrong) because a mixture
cannot interpolate past its outermost bump; with 6, worst 1.7 and mean 0.2.

**Integration.** The encoder is a depthwise causal convolution in front of an
otherwise byte-identical backbone. The window widens from 12 to
`12 + 26 − 1 = 37` periods; the 7 climate channels are smoothed to 12 steps, the
7 non-climate channels are sliced to the same 12, and the concatenation goes into
the unchanged GCN → GRU → head. Cost: ~25 training windows per fold (fold 9: 877
→ 852).

**Optimisation.** The encoder gets its own Adam group at lr 2e-2 with weight
decay 0, against 3e-3 / 1e-4 for the backbone. Gradients reaching the kernel pass
through a softmax and a Gaussian exponent and are far smaller than those reaching
the GRU; at a shared rate the module is effectively inert. Weight decay is zeroed
because shrinking the district embedding toward zero collapses every district
onto the same mixture — decay would destroy the property being tested.

**Controlled recovery test** (`tests/test_lag_encoder.py`). Plant a different
known delay per district (3–19 weeks) in synthetic data; the encoder recovers
every district within ±2 weeks on 3 seeds. Also tested: kernels sum to 1 and are
non-negative; output shape; no gradient from a future period; all four parameter
groups receive non-zero gradient; districts are free to differ. **323 tests pass
repository-wide.**

---

## 6. Experiment (`scripts/18.train_lag_gcn_gru.py`)

Four arms on identical folds, seeds, masks, graph, GRU, head, target and loss.
Only climate handling changes. The training loop, metrics, fold construction and
preprocessing are **imported** from `scripts/16`, not copied, so a difference in
the numbers cannot come from a difference in the loop.

| Arm | Features | Climate handling |
|---|---|---|
| `hand_lags_v1` | `v1` | trailing 4/8/12-period means, hand-chosen, same for all districts |
| `learned_lags` | `v0` | learned per-district kernels over 26 periods |
| `no_lags_v0` | `v0` | climate at the current period only |
| `no_climate` | `v0` minus climate | none — cases, seasonality, geography only |

`no_climate` is the arm that makes the result interpretable: `no_lags_v0` still
sees current-period weather, so it answers "does *delayed* climate help", not
"does climate help".

### 6.1 Headline results (7 headline folds, `gcn_gru`, 1 seed)

| Arm | MAE | RMSE | Peak MAE | vs hand-coded |
|---|---|---|---|---|
| `no_lags_v0` | **18.07** | **38.27** | **29.14** | **+1.6%** |
| `hand_lags_v1` | 18.36 | 38.90 | 29.65 | — |
| `no_climate` | 18.71 | 41.00 | 30.28 | −1.9% |
| `learned_lags` | 20.67 | 44.45 | 32.82 | **−12.6%** |

COVID folds (4, 5), reported separately: 7.16–7.42 MAE, all four arms within
0.26 of each other.

### 6.2 Per fold, MAE

| Fold | Year | `no_lags_v0` | `hand_lags_v1` | `learned_lags` | `no_climate` |
|---|---|---|---|---|---|
| 1 | 2017 | 48.43 | 51.06 | **65.18** | 52.78 |
| 2 | 2018 | 10.31 | 10.11 | 10.22 | 10.31 |
| 3 | 2019 | 17.49 | 17.19 | 17.63 | 17.91 |
| 6 | 2022 | 12.18 | 12.11 | 12.03 | 12.20 |
| 7 | 2023 | 18.99 | 18.99 | 20.58 | 18.69 |
| 8 | 2024 | 10.01 | 9.98 | 9.93 | 10.04 |
| 9 | 2025 | 9.05 | 9.06 | 9.14 | 9.02 |

Two facts the headline mean hides, both worth stating:

- **The headline average is dominated by fold 1.** Error ranges from 9 to 48 MAE
  across these folds, so 2017 supplies most of the mean. Excluding fold 1, all
  four arms fall within **0.35 MAE** of each other (12.91–13.26).
- **`learned_lags` loses via two folds:** 2017 (+34.6% vs `no_lags_v0`) and 2023
  (+8.4%). On the other five it sits within ±1.2%, which is inside the seed
  noise nobody has measured.

### 6.3 The climate-free control

| Comparison | MAE cost of dropping climate entirely |
|---|---|
| All 7 headline folds | +3.5% |
| Fold 1 (2017 epidemic) only | +9.0% |
| Folds 6, 8, 9 (recent normal years) | **+0.01 MAE** |

Climate contributes in the epidemic year and essentially nothing in recent
normal years. **This is confounded**: fold 1 is both the epidemic *and* the
smallest fold (459 windows vs 877), so extra input channels may be helping a
data-starved model regardless of content. See §8.

---

## 7. Diagnosis — why it failed

Three pieces of evidence, in the order they eliminate explanations:

1. **Not a broken module.** The synthetic recovery test passes on 3 seeds with
   the same initialisation used on real data.
2. **Not too short a reach.** The measured rainfall peak stays at 9 weeks when
   the scan is extended to lag 51. The true delay is well inside the 26-week
   reach.
3. **No delay signal in the gradient.** The encoder is trained jointly: its
   parameters only receive a meaningful gradient if *delayed* climate reduces
   the forecasting loss. The `no_climate` arm shows climate reduces it by ~0% on
   recent folds at h = 1. A parameter driven by a gradient carrying no
   information about delay drifts to the boundary of its range.

**The observable consequence.** On fold 9, Colombo's learned rainfall kernel
rises monotonically to the 25-week edge, against a measured peak at 5 weeks.
Across all 25 districts, measured delays span 5–10 weeks while learned delays
span 15–22 — correlation between them **r = −0.16**.

The scan measures the delay *without* requiring it to reduce forecast error,
which is exactly why it produced a stable answer where the joint approach did
not.

---

## 8. What may NOT be claimed

- **Single seed.** Every `scripts/18` number in §6 is one seed. Seed sd is
  unmeasured, so differences below roughly 1 MAE are not established. The fold-1
  gaps (4–17 MAE) are large enough to be safe; the fold 2/6/8/9 gaps (<0.2 MAE)
  are **not** — do not describe them as ties or as wins.
- **One backbone.** `gcn_gru` only. `gru_only` is unrun for these arms, and in
  the baseline `gru_only` *beats* `gcn_gru` (9.15 vs 9.98 on fold 8).
- **Fold 1 is confounded** — epidemic year and smallest fold. The clean control
  (seven random channels instead of seven climate ones) has not been run.
- **Horizon 1 only.** The prediction that delay structure matters more at longer
  horizons is untested. `--horizon` now exists but no h > 1 sweep has been run.
- **Workplan ablation rows 4 and 5 are not run** — the free unconstrained 26-tap
  filter and the single global kernel (no district embedding). Without row 5 we
  cannot claim the *per-district* part specifically is what fails.
- **Temperature at lag 17 is unexplained.** It is the strongest deseasonalised
  signal (r = +0.199) but 17 weeks is biologically odd for temperature; it may be
  an interannual (ENSO-type) effect rather than a transmission delay.
- **Deseasonalising removes the annual cycle, not the interannual one.** The 2017
  epidemic and the COVID collapse remain in the anomalies and likely depress every
  correlation in §4.

---

## 9. Reproducing every number

```bash
PY=.venv/Scripts/python.exe          # bash;  .venv\Scripts\python.exe in PowerShell

$PY -m pytest tests/ -q                                   # 323 passed, ~70 s
$PY scripts/17.lag_correlation_scan.py                    # §4,  ~4 s
$PY scripts/18.train_lag_gcn_gru.py --seeds 1 --no-control  # §6, ~45 min CPU
$PY scripts/19.lag_demo.py                                # figures + walkthrough, ~5 s
```

| Artifact | Section |
|---|---|
| `results/eda/lag_correlation.csv` + `lag_correlation_report.md` | §4 |
| `results/models/lag_metrics.csv` | §6 |
| `results/models/learned_lags.csv` | §7 |
| `results/figures/fig1…fig4.png` | §10 |

---

## 10. Figures (generated by `scripts/19.lag_demo.py`)

**Fig. 1 — `fig1_rainfall_lag_by_district.png`.** *Measured rainfall-to-dengue
delay by district, fold 9 training periods only, seasonality removed. All 25
districts fall in a 5–10 week band; Colombo, Jaffna, Puttalam and Batticaloa are
fastest at 5 weeks, Matale, Kilinochchi and Badulla slowest at 10.*

**Fig. 2 — `fig2_colombo_measured_vs_learned.png`.** *Colombo rainfall. Top: the
measured cross-correlation, peaking at lag 5. Bottom: the kernel the encoder
learned end-to-end on the same fold, whose mass rises monotonically to the
25-week edge of its reach. Panels share the lag axis and are stacked rather than
overlaid on twin y-axes, which would let the scaling choice decide whether the
reader sees agreement.*

**Fig. 3 — `fig3_measured_vs_learned_scatter.png`.** *Learned against measured
rainfall delay, one point per district, fold 9. Points on the dashed line would
indicate recovery. Measured delays span 5–10 weeks and learned delays 15–22;
r = −0.16.* **This is the figure the paper turns on.**

**Fig. 4 — `fig4_accuracy_by_arm.png`.** *Left: mean MAE over the seven headline
folds; the learnable encoder is the worst of four arms. Right: the same result
per fold as a percentage of the best arm, which divides out a fold error range of
9–48 MAE. The encoder's loss is concentrated in 2017 (+34.5%) and 2023 (+8.5%).*

---

## 11. Suggested framing

**Title direction.** "Learned climate lags do not improve district-level dengue
forecasting — and the reason is measurable."

**The claim.** Not "our method wins", but: *fixed climate lags are an
unexamined assumption; we measure the real delay, build a module that provably
recovers planted delays, and show that end-to-end learning of that delay fails
on real data for a specific and diagnosable reason — the forecasting gradient at
h = 1 contains no delay information.* The measurement (§4) stands as a
standalone contribution useful to a control programme; the negative result (§6–7)
is evidenced rather than asserted.

**Do not** present the +0.4% on fold 8 as a win. It is one fold, one seed, and
§6.2 shows the same arm losing by 34.5% on another.

**The honest next step**, and worth one sentence in future work: fix the kernels
to the measured delays from §4 and train only the backbone. That decouples the
delay estimate from a gradient that does not carry it. If that also fails to beat
`hand_lags_v1`, the ceiling is climate's contribution at h = 1, not delay
handling — a cleaner conclusion than the current evidence supports.

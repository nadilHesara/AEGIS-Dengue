# The  baseline models

What exists, how it is built, what it scores, and what comes next.

---

## 1. How the tensors are built

The master panel (`panel_weekly.parquet`, 1012 periods x 25 districts) becomes
model input in three steps.

**Script 12 — the feature cube.** The flat panel is reshaped into
`X [1012, 25, F (feature types-cases, climates, lags)]`, `y [1012, 25]`, `y_mask [1012, 25]`, ordered `period_id` then
`node_id`. That ordering is the contract: `X[t, i]` and `A[i, j]` refer to the
same district by construction. The builder refuses to run on a panel that is not
a complete, correctly ordered, gap-free grid.

**Script 13 — the graph.** Queen contiguity over the GADM polygons, matched to
districts by `gadm_gid` rather than by name. 25 nodes, 57 edges, mean degree
4.56, no isolated nodes. Normalised as `D^-1/2 (A + I) D^-1/2`.

**Script 14 — folds and preprocessing.** Nine walk-forward folds, plus the
per-fold imputation and scaling statistics.

### Windowing

```
origin t  ->  inputs  X[t-11 : t+1]      12 periods of climate + case history for all 25 districts
              target  y[t+1]             	next period's cases for all 25 districts
```

At `L=12, h=1` there are 1000 possible windows; v0 keeps 987, v1 keeps 976.
L = lookback — how many reporting periods of history the model sees per sample.
h = horizon — how many periods ahead it predicts.
### Missing data

| Kind | Treatment |
| --- | --- |
| Missing climate | imputed, district x month climatology fitted per fold |
| Missing case **feature** | imputed, same method |
| Missing case **target** | never imputed — masked out of loss and metrics |

Target missingness is per district, not per window. Puttalam has no record for
period 999; that one cell is masked and the other 24 districts are kept.

### Leakage rules

- Every fitted statistic (climatology, scaler, peak thresholds) comes from
  periods strictly before the fold's test year.
- Windows are assigned to a split by their **target** period. Input history may
  reach into an earlier split — at deployment it is available.
- `is_covid_window` and `is_2017_outbreak` are **not** features. They are
  retrospective labels used only to stratify results.

---

## 2. What Features used in X


**Feature variants** — what goes in `X`.

| Variant | F | Contents |
| --- | --- | --- |
| `v0` | 14 | cases, climate, seasonality, flags, centroids. No hand lags. |
| `v1` | 23 | v0 + trailing 4, 8, 12 period means of rainfall, temperature, humidity |

`v1 - v0` measures what hand-specified lags are worth. That is the number a
learnable lag module must beat.


v0 and v1 versions are only applied in gcn_gru , gru_only models 

**Models** — what is trained.



| Name | Description |
| --- | --- |
| `persistence` | next period equals this period |
| `seasonal_naive` | next period equals the same period last year |
| `gcn_gru` | 2 graph conv layers over districts, then a shared GRU, then a linear head |
| `gru_only` | identical, adjacency replaced by the identity — the control that isolates what the graph contributes |

**Target parameterisation** — what the network predicts.

One detail about what the models actually outputs: it doesn't predict the case count directly. It predicts the change from last week, on a log scale, which is then converted back to cases. So an output of zero literally reproduces the persistence forecast, and the network only has to learn the correction

| Mode | Predicts | Note |
| --- | --- | --- |
| `residual` | `log1p(y[t+1]) - log1p(y[t])` | default; output 0 reproduces persistence |
| `direct` | `log1p(y[t+1])` | the unanchored version |

Artifact version strings: `tensors-v1`, `adjacency-v1`, `folds-v1`,
`naive-v1`, `gcn-gru-v1`.

### Architecture

```
per timestep:  S = relu(A_norm @ X @ W1);  S = relu(A_norm @ S @ W2)
then:          H = GRU(S_1 .. S_12)        shared weights, one sequence per node
then:          y = Linear(H_12)
```

8,193 trainable parameters at `hidden=32`. Dense 25x25 matmul, no PyTorch
Geometric. Loss is masked MSE. Spatial-then-temporal deliberately, because that
is where the spatial-versus-temporal fusion gate goes later.

---

## 3. Evaluation

Nine walk-forward folds: train on everything before a calendar year, test on
that year. Test years 2017-2025. Headline = mean over folds 1, 2, 3, 6, 7, 8, 9;
COVID folds (2020, 2021) reported separately.

Fold 1 is the hard case. 2017 was Sri Lanka's biggest dengue epidemic. The model trains on ten years that contain nothing like it, then has to forecast it. That's the real test of an early-warning system, and it's where the model currently fails.
Fold 8 is the normal case. 2024 is an ordinary recent year with twelve years of training data behind it. It shows what the model does under favourable conditions.

### Naive baselines — complete, all 9 folds

| Model | Headline MAE | RMSE | Peak MAE | 2017 MAE | 2017 peak MAE |
| --- | --- | --- | --- | --- | --- |
| persistence | **16.42** | 33.61 | **26.61** | 36.08 | 39.42 |
| seasonal_naive | 55.45 | 118.76 | 84.81 | 111.10 | 126.65 |

Seasonal naive is far worse than persistence, which says dengue's interannual
variation dominates its seasonality here.

### GCN+GRU — full sweep, 9 folds, 3 seeds

**Run 2026-09-07.** 2 variants x 2 models x 9 folds x 3 seeds, 44.6 min CPU.
This replaces the fold-8 smoke numbers this section used to carry, and it
overturns three claims that were made from them.

| Model | Features | Headline MAE | RMSE | Peak MAE | 2017 MAE | Seed sd |
| --- | --- | --- | --- | --- | --- | --- |
| persistence | — | **16.42** | **33.61** | **26.61** | **36.08** | — |
| `gru_only` | v0 | 16.67 | 35.22 | 28.32 | 41.74 | 0.28 |
| `gru_only` | v1 | 17.12 | 36.69 | 28.70 | 46.01 | 0.70 |
| `gcn_gru` | v1 | 19.01 | 40.63 | 30.47 | 55.01 | 0.62 |
| `gcn_gru` | v0 | 19.08 | 41.07 | 30.46 | 55.23 | 0.84 |

Per fold, MAE, averaged over 3 seeds:

| Fold | Year | `gcn_gru` v0 | `gcn_gru` v1 | `gru_only` v0 | `gru_only` v1 | persistence |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2017 | 55.23 | 55.01 | 41.74 | 46.01 | **36.08** |
| 2 | 2018 | 10.39 | 10.08 | 9.93 | **9.76** | 10.45 |
| 3 | 2019 | 17.90 | 17.57 | 17.03 | **16.82** | 17.30 |
| 4 | 2020 | 7.36 | 7.54 | 7.51 | 7.46 | **7.41** |
| 5 | 2021 | 7.23 | 7.25 | **7.02** | 7.03 | 7.23 |
| 6 | 2022 | 12.18 | 12.17 | 11.61 | **11.59** | 12.57 |
| 7 | 2023 | 18.73 | 19.07 | 18.34 | **18.17** | 18.83 |
| 8 | 2024 | 10.09 | 10.09 | 9.52 | **9.16** | 10.78 |
| 9 | 2025 | 8.99 | 9.06 | 8.48 | **8.31** | 8.95 |

**Three findings, each contradicting what this document previously claimed.**

1. **No trained model beats persistence on the headline mean.** The best,
   `gru_only` on v0, scores 16.67 against 16.42. The models do beat persistence
   on 5 of the 7 headline folds, and lose the mean on 2017 alone. The headline
   gap *is* the epidemic fold, which is what makes fold 1 the whole problem
   rather than one bad row.

2. **The graph is not neutral, it is harmful.** `gru_only` beats `gcn_gru` on
   **every one of the nine folds**, by 1.9 MAE on the headline mean, against a
   seed sd of 0.28 to 0.84. This is no longer "worth watching" — it is decided.
   Queen contiguity degrades the forecast, and the dual-graph work in step 7
   should not be built on it without first establishing that any graph helps.

3. **Hand-specified climate lags are worth negative value.** On the better
   backbone, v0 (16.67) beats v1 (17.12); the same ordering holds on fold 1 by
   4.3 MAE. So `v1 - v0`, the number `docs/learnable_lags.md` set as the target
   for a learnable lag module, is **negative**. The module was benchmarked
   against a baseline that was itself worse than doing nothing.

> The earlier warning in this section was correct and is now moot: the "+44%"
> claim came from comparing a fold-8 model score against a seven-fold
> persistence average. `baseline_report.md` has been regenerated from the full
> sweep.

### Fold 1 (2017 epidemic) — the hard one

| Model | MAE |
| --- | --- |
| persistence | **36.08** |
| `gcn_gru` v1, residual target | 52.91 |

The model **loses to persistence on the epidemic fold.** It trains on 2007-2016,
which contains no outbreak of that size, and underpredicts: Colombo's actual mean
was 631 cases per period against 214 predicted, at a correlation of 0.71. It
tracks the shape and misses the level. This is the central open problem.

### Why the target is anchored

Measured on folds 1 and 8, single seed:

| Fold | `direct` | standardised | `residual` |
| --- | --- | --- | --- |
| 1 (2017) | 87.45 | 76.49 | **52.91** |
| 8 (2024) | 17.58 | 15.07 | **10.11** |

Anchoring also converges in roughly a third of the epochs. It is a target
parameterisation, not an architecture change.

---

## 4. Work plan

### Finish the baseline

1. ~~**Run the full sweep.**~~ **Done, 2026-09-07.** See section 3. It answered
   all three of its questions, and none of the answers were the expected one:
   no model beats persistence on the headline, the graph hurts on every fold,
   and `v1 - v0` is negative.
2. **Write up the results** — section 3 of this document is updated.
   `docs/model_tensors.md` still lists scripts 15 and 16 as "next" and has not
   been touched.
3. **Sweep the window geometry** — `L` (lookback lags) in {12, 26} and `h`( horizon how many periods ahead it predicts) in {1, 2, 3, 4}.
   Longer lookback is the cheapest possible test of whether lag structure is
   being missed.
4. **Per-district breakdown.** Jaffna has degree 1 under contiguity and receives
   almost no spatial information; check whether it is systematically worse.
   Partly addressed in [`docs/improvements.md`](improvements.md) §4 for fold 1,
   where Jaffna is not an outlier and improves 8.6% under the reweighted
   objective. Note that finding 2 above changes what this question means: if the
   graph hurts everywhere, low degree is not obviously a disadvantage.

### Then the research model

Each item is a single change measured against the frozen baseline on the same
folds and masks.

5. **Peak-weighted loss.** ~~The most direct attack on the fold-1 failure.~~
   **Done, 9 folds x 3 seeds — see [`docs/improvements.md`](improvements.md).**
   `scripts/20.train_improved.py` weights each cell by `log1p(cases)` at the
   forecast origin. The best arm, `level_weighted`, cuts the headline MAE from
   **19.01 to 16.69 (−12.2%)** against a seed sd of 0.62–0.66, and fold-1 MAE
   from **55.01 to 38.52 (−30.0%)**. The seed-mean ensemble reaches 16.39 against
   persistence at 16.42 — a tie, not a win, at that margin. The mechanism is
   confirmed rather than assumed: on fold-1 cells above 200 cases the
   predicted/actual ratio moves from 0.624 to 0.740.

   Two things this did **not** achieve. The success criterion above — fold-1 peak
   MAE below persistence — is **not met**: 41.25 against 39.42, down from 62.40.
   And the entire headline gain is fold 1; averaged over the other six headline
   folds the change is −0.046 MAE. This is an epidemic-conditions fix, which is
   consistent with the diagnosis, since the objective mismatch scales with the
   width of the case distribution.

   Note also that level weighting alone beats every Huber variant; the two
   corrections compete rather than compose.
6. **Learnable lag module.** Per climate variable, per district. Must beat the
   `v1 - v0` gap, which step 1 will quantify.
7. **Dual graph.** `A_manual` (contiguity, already built; `A_gaussian` is the
   alternative candidate) plus a learned `A_data`, mixed by a season-dependent
   gate. **Step 1 has now answered its precondition, and the answer is bad for
   this item:** the margin is not near zero, it is negative on all nine folds —
   contiguity costs 1.9 MAE against no graph at all. Adding a gated mixture on
   top of a component that is measurably harmful risks burying the problem
   rather than fixing it. Establish first that *some* graph beats the identity —
   `A_gaussian` is already built and is the obvious first test, since contiguity
   is a poor prior in a country where Colombo and Galle are 100 km apart, share
   no border, and are both wet-zone coastal.
8. **Gated spatial/temporal fusion.** The current architecture is
   spatial-then-temporal precisely so this gate drops in without restructuring.

### Data debt, worth clearing early

9. **Extend the ERA5 extraction past 2026-04-26.** Periods 1010-1012 have no
   climate at all. Costs the folds nothing today, but blocks a 2026 fold.
10. **The one absent Puttalam record** (period 999) removes 13 windows in v0.
    Small, but it lands inside the test region.

---

## Related documents

- [`docs/model_tensors.md`](model_tensors.md) — tensors, folds, adjacency in detail
- [`docs/climate_dataset_schema.md`](climate_dataset_schema.md) — stages 1 and 2
- [`docs/reporting_calendar.md`](reporting_calendar.md) — why `period_id` is the only safe sort key
- [`results/models/naive_baseline_report.md`](../results/models/naive_baseline_report.md)
- [`results/models/folds_report.md`](../results/models/folds_report.md)

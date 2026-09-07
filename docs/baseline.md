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

### GCN+GRU — **fold 8 only, single seed, v1**

The full 9-fold sweep has **not** been run. These are the only trained-model
numbers measured so far, and they are one fold with one seed:

| Model | Fold 8 (2024) MAE | RMSE | Peak MAE |
| --- | --- | --- | --- |
| persistence | 10.78 | 21.57 | 25.77 |
| `gcn_gru` v1 | 9.98 | 18.82 | 23.44 |
| `gru_only` v1 | **9.15** | 18.75 | **21.70** |

Both beat persistence on that fold, modestly. `gru_only` edging out `gcn_gru` is
a real result worth watching, not noise to dismiss — at one seed it is not yet
decidable either way.

> **Warning.** `results/models/baseline_report.md` currently claims a headline
> MAE of 9.15 against persistence at 16.42, a "+44%" improvement. That compares a
> fold-8 model score against a seven-fold persistence average. On fold 8
> persistence scores 10.78, so the true margin is about 15%, not 44%. The report
> is a stale artifact of a smoke run and is regenerated by step 1 below.

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

1. **Run the full sweep.** `python scripts/16.train_gcn_gru.py` — 2 variants x 2
   models x 9 folds x 3 seeds, about 2 hours on CPU. Regenerates
   `baseline_report.md` and replaces the stale numbers above. This is the
   blocking step: every claim about the graph, about v0 vs v1, and about
   beating persistence is currently unproven at 9-fold scale.
2. **Write up the results** in this document and in `docs/model_tensors.md`,
   which still lists scripts 15 and 16 as "next".
3. **Sweep the window geometry** — `L` (lookback lags) in {12, 26} and `h`( horizon how many periods ahead it predicts) in {1, 2, 3, 4}.
   Longer lookback is the cheapest possible test of whether lag structure is
   being missed.
4. **Per-district breakdown.** Jaffna has degree 1 under contiguity and receives
   almost no spatial information; check whether it is systematically worse.

### Then the research model

Each item is a single change measured against the frozen baseline on the same
folds and masks.

5. **Peak-weighted loss.** The most direct attack on the fold-1 failure. Weight
   the loss by observed case level so the epidemic periods stop being averaged
   away. Success = fold 1 peak MAE below persistence.
6. **Learnable lag module.** Per climate variable, per district. Must beat the
   `v1 - v0` gap, which step 1 will quantify.
7. **Dual graph.** `A_manual` (contiguity, already built; `A_gaussian` is the
   alternative candidate) plus a learned `A_data`, mixed by a season-dependent
   gate. Step 1's `gcn_gru` vs `gru_only` margin tells you how much headroom the
   graph actually has — if it is near zero, this is where to find out why.
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

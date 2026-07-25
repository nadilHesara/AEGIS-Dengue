# Stage 3: model inputs

Everything the baseline needs that is not the model itself: the sequence
tensors, the graph, the walk-forward folds, and the per-fold preprocessing.

Stage 3 is the sequence stage named in
[`docs/climate_dataset_schema.md`](climate_dataset_schema.md). Stage 1 is raw
daily climate, stage 2 aligns it to reporting periods, and stage 3 turns the
flat panel into arrays a graph sequence model can index.

| Script | Builds |
| --- | --- |
| `scripts/12.build_model_tensors.py` | feature tensors, two variants |
| `scripts/13.build_adjacency.py` | district adjacency matrices |
| `scripts/14.build_folds.py` | walk-forward folds and per-fold statistics |

Run order, after the panel exists:

```powershell
python scripts/9.aggregate_climate_to_periods.py
python scripts/10.create_master_panel.py
python scripts/12.build_model_tensors.py
python scripts/13.build_adjacency.py
python scripts/14.build_folds.py
```

Tests: `tests/test_model_tensors.py`, `tests/test_adjacency.py`,
`tests/test_folds.py` — 62 tests, all passing.

---

## 1. The tensors

Each `.npz` holds the same keys:

| Key | Shape | Meaning |
| --- | --- | --- |
| `X` | `[1012, 25, F]` | features, unscaled, NaN where unobserved |
| `y` | `[1012, 25]` | `cases`, raw counts |
| `y_mask` | `[1012, 25]` | 1 where the case count was observed |
| `weather_mask` | `[1012, 25]` | 1 where the period had full weather coverage |
| `period_id` | `[1012]` | 1 to 1012, the chronological key |
| `node_id` | `[25]` | 0 to 24, from `nodes.csv` |
| `start_date` | `[1012]` | period start, for reporting only |
| `feature_names` | `[F]` | column order of `X` |

Axis 0 is `period_id` ascending, axis 1 is `node_id` ascending. That is the
point of the reshape: `X[t, i]` and `A[i, j]` refer to the same district by
construction, so the graph and the features cannot drift apart. The builder
refuses to run if the panel is not in that exact order, is not a complete grid,
or has a gap in `period_id`.

### Features

**v0 — 14 columns, no hand-specified lags.**

| Column | Source | Note |
| --- | --- | --- |
| `cases_log1p` | `log1p(cases)` | counts run 0 to 2631 with a median of 10 |
| `rainfall_daily_mean_mm` | panel | daily mean, not the period sum |
| `rainy_days_frac` | `rainy_days / reporting_days` | |
| `temperature_mean_c` | panel | |
| `diurnal_range_c` | `temperature_max_c - temperature_min_c` | |
| `dewpoint_mean_c` | panel | |
| `relative_humidity_mean` | panel | |
| `wind_speed_mean` | panel | |
| `doy_sin`, `doy_cos` | `start_date` day-of-year | |
| `weather_observed` | `weather_complete` | marks imputed climate cells |
| `case_observed` | panel | marks imputed case-history cells |
| `centroid_lat`, `centroid_lon` | `nodes.csv` | raw; the scaler handles them |

**v1 — v0 plus 9 rolling columns.** Trailing means over 4, 8 and 12 periods for
rainfall, temperature and humidity, which is roughly where the mosquito
development lag is expected to sit.

Decisions worth knowing:

- **Daily mean rainfall, not the period sum.** Periods 122 and 127 are 8 and 6
  days long. A sum is not comparable across them; the mean is. Same reason
  `rainy_days` is divided by the true `reporting_days` rather than an assumed 7.
- **Seasonality from `start_date`, never `source_week`.** The interval labelled
  2026 week 53 falls in December 2025 — see
  [`docs/reporting_calendar.md`](reporting_calendar.md). Day-of-year is divided
  by 365.25, so the angle does not drift across the five leap years.
- **`rainy_days` is cleared where weather is unobserved.** Stage 2 emits it as an
  integer count, so a period with no weather arrives as `rainy_days = 0` — a
  fabricated dry period. Every other weather column was already NaN there.
- **`is_covid_window` and `is_2017_outbreak` are not features.** They are
  retrospective labels. `is_2017_outbreak` covers periods 525-576, which is
  exactly the fold 1 test range: as an input it would be a column that is 0 in
  training and 1 in test, set by someone who already knew 2017 was an epidemic
  year. Both remain in the panel and are used to **stratify results** — see
  section 3.

### Why two variants

The GRU already sees `L` periods of history, so it can in principle learn any
lag structure on its own. The question is whether it does. v0 makes it find the
structure unaided; v1 hands it the 4, 8 and 12 period means directly.

`v1 - v0` is what hand-specified lags are worth on this data, and that is the
number a learnable lag module has to beat. Both files are cheap; keep both.

## 2. Windowing and target missingness

`make_windows(tensors, lookback, horizon)` cuts the samples:

```
origin t  ->  inputs  X[t - lookback + 1 : t + 1]
              target  y[t + horizon]
```

Every input period is at or before the origin; the target is strictly after it.
This is the only place the forecast alignment is defined, and a mistake here is
invisible downstream — a misaligned model trains and scores fine, it just
reports skill it does not have. So `tests/test_model_tensors.py` asserts it
against `period_id` directly: inputs contiguous, increasing, last input equal to
the origin, every input strictly before the target.

Returned keys: `X [B, lookback, 25, F]`, `y [B, 25]`, `y_mask [B, 25]`, and
`origin_period_id`, `target_period_id`, `input_period_id` for auditing.

### The target is never imputed

A missing case count stays NaN, is flagged in `y_mask`, and is excluded from the
loss and from every metric. An imputed target is a number the model is rewarded
for reproducing that no one ever observed.

**Missingness is handled per district, not per window.** Puttalam has no record
for period 999. Dropping the whole window would throw away 24 observed districts
to remove one absent cell, so the window is kept and `y_mask` carries the gap. A
window is dropped only when *no* district in it has an observed target.

Your loss and metrics must therefore respect the mask:

```python
error = (prediction - target) * y_mask
loss = (error ** 2).sum() / y_mask.sum()
```

At `L=12, h=1` there are 1000 possible windows; v0 keeps 987 and v1 keeps 976.

## 3. Walk-forward folds

A single split answers one question about one year. Walk-forward asks it nine
times and puts the 2017 epidemic in a **test** fold, which is the only way to
find out whether an epidemic that size is forecastable from climate and case
history alone.

| Fold | Test year | Train | Val | Test | Train windows | Note |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2017 | 1-471 | 472-524 | 525-576 | 459 | epidemic |
| 2 | 2018 | 1-524 | 525-576 | 577-628 | 512 | |
| 3 | 2019 | 1-576 | 577-628 | 629-680 | 564 | |
| 4 | 2020 | 1-628 | 629-680 | 681-732 | 616 | COVID, report separately |
| 5 | 2021 | 1-680 | 681-732 | 733-784 | 668 | COVID, report separately |
| 6 | 2022 | 1-732 | 733-784 | 785-837 | 720 | first fold with COVID in *train* |
| 7 | 2023 | 1-784 | 785-837 | 838-889 | 772 | |
| 8 | 2024 | 1-837 | 838-889 | 890-941 | 825 | |
| 9 | 2025 | 1-889 | 890-941 | 942-993 | 877 | |

Every fold has a full test year: 52 or 53 windows, 1300 observed district-cells.

**Fold years come from `start_date`, never `source_year`.** A `source_year`
boundary would put the interval labelled 2026 week 53 — which starts
2025-12-20 — a year away from where it belongs.

**The inner validation split is the calendar year before the test year.**
Without it, early stopping would be chosen on the test fold and the reported
score would be a tuned score wearing a held-out costume.

**A window belongs to the split its target falls in.** Its input history may
reach back into an earlier split, which is correct: at deployment that history
is available. What must not cross is the fitted statistics.

**2026 is not a fold.** It is a partial year and its last three periods have no
ERA5 coverage. Hold it back as final unseen data.

### Reporting

- **Headline** = mean over folds 1, 2, 3, 6, 7, 8, 9.
- **Folds 4 and 5 reported separately** as the COVID distribution-shift
  analysis. Do not average them into the headline.
- **Fold 1 is the one that matters.** It trains on 2007-2016, which contains no
  epidemic remotely the size of 2017's. Peak-MAE on fold 1 is the number worth
  putting in a paper.
- **Fold 6 is quietly interesting**: the first fold with COVID inside its
  training data, so it says whether the disruption poisons later forecasts.

`folds.json` carries `covers_covid`, `covers_2017_outbreak` and `headline` per
fold, derived from the panel flags, so the stratification is data-driven rather
than a hardcoded year list.

## 4. Per-fold preprocessing

Scaling and imputation are **fold-dependent**, which is why they are not baked
into the tensors. Both are fitted on periods strictly before the fold's test
year (`fit_end_period`) and stored per fold in
`fold_preprocessing_{variant}.npz`.

| Step | Method |
| --- | --- |
| Imputation | district x month climatology, fallback district mean, then global mean |
| Scaling | per-feature z-score over the fitting periods |

The imputation method is the one selected in
[`configs/imputation/panel_v1.toml`](../configs/imputation/panel_v1.toml).

Two details that matter:

- **Scaling is per feature, not per node.** Colombo really does report more
  cases than Mannar, and scaling that away would remove exactly what the graph
  is supposed to learn.
- **Zero-variance columns are guarded.** `weather_observed` is constant 1 across
  every fold's history, so an unguarded z-score would divide by zero.

Filled cells are already flagged: `weather_observed = 0` marks imputed climate,
`case_observed = 0` marks imputed case history. No extra channel is needed.

`tests/test_folds.py` states the leakage rule as an experiment rather than a
code review: it overwrites every period from the test year onward with `1e6`,
refits, and asserts not one fitted statistic moves.

## 5. Adjacency

Queen contiguity over the GADM level 1 polygons, matched to districts by
`gadm_gid` — never by name, which is what the registry exists to prevent.

| Property | Value |
| --- | --- |
| Nodes | 25 |
| Edges | 57 |
| Degree | min 1, mean 4.56, max 9 |
| Isolated nodes | 0 |
| Contiguity tolerance | 100 m |

`adjacency.npz` keys:

| Key | Meaning |
| --- | --- |
| `A_binary` | queen contiguity, symmetric, zero diagonal — the fixed `A_manual` |
| `A_norm` | `D^-1/2 (A + I) D^-1/2`, ready for a graph conv |
| `A_gaussian` | `exp(-(d/50km)^2)` on centroid distance, thresholded at 0.1 |
| `A_gaussian_norm` | the same, normalised |
| `A_distance_km` | raw centroid distances in EPSG:32644 |

Contiguity is tested on polygons buffered by 100 m. GADM boundaries are
digitised, so districts that share a border in reality can be separated by a
sliver a few metres wide, and a strict `touches()` misses them.

`A_gaussian` is kept beside contiguity because contiguity is a crude prior in a
country this narrow: Colombo and Galle are 100 km apart and share no border, yet
are both dense wet-zone coastal districts. It is a natural second candidate for
`A_manual`.

**Jaffna has degree 1** — its only land neighbour is Kilinochchi. That is
correct geography, not a bug, but it means Jaffna receives almost no spatial
information under `A_binary`. Worth watching in the per-district metrics.

The tests check structure *and* geography: symmetry, zero diagonal, connectivity,
a normalised spectrum inside `[-1, 1]`, and that Colombo borders Gampaha while
Jaffna does not border Colombo. A matrix can satisfy every structural property
and still describe the wrong country.

## 6. Using it

```python
import importlib.util, json
from pathlib import Path
import numpy as np

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

T = load("t", "scripts/12.build_model_tensors.py")
F = load("f", "scripts/14.build_folds.py")

tensors  = F.load_tensors("v1")
calendar = F.load_calendar()
months   = calendar.sort_values("period_id")["month"].to_numpy()
folds    = json.loads(Path("data/processed/folds.json").read_text())["folds"]
A        = np.load("data/processed/adjacency.npz", allow_pickle=True)["A_norm"]

for fold in folds:
    stats = F.fit_fold_statistics(
        tensors, months, tensors["period_id"] <= fold["fit_end_period"]
    )
    scaled = dict(tensors)
    scaled["X"] = F.transform(tensors, months, stats)

    w = T.make_windows(scaled, lookback=12, horizon=1, drop_incomplete=False)
    split = F.assign_windows(w["target_period_id"], fold)

    X_train, y_train = w["X"][split["train"]], w["y"][split["train"]]
    mask_train = w["y_mask"][split["train"]]
    # ... train, early stop on split["val"], score on split["test"]
```

`drop_incomplete=False` is correct here: the features have already been imputed,
so there are no NaN inputs left to drop for.

Target transform: train on `log1p(y)`, invert with `expm1` before computing
metrics. No fitting required, so it stays in the training script.

Verified end to end on fold 1: `X` windows `(1000, 12, 25, 23)`, zero NaN,
459/53/52 train/val/test windows, `A_norm @ X` contracts cleanly, and all 1300
test target cells observed.

## 7. Known gaps

**ERA5 ends 2026-04-26.** Periods 1010-1012 have no weather at all — 75
district-periods, every climate column NaN. This costs the folds nothing, since
fold 9 ends at period 993, but it does mean the panel's last three weeks are
unusable. Extend the extraction before adding a 2026 fold.

**The v1 leading rolling window is imputed.** Rolling means use
`min_periods = window`, so periods 1-11 are NaN and the fold-1 imputer fills
them from climatology. A climatological stand-in for a 12-period trailing mean
is a slightly odd object, and unlike missing climate it carries no flag. It
affects the first 11 training windows of fold 1 only. If it bothers you, drop
targets below period 24 when training on v1.

## 8. Next

The baseline itself:

1. Naive baselines — persistence (`ŷ = y_t`) and seasonal naive
   (`ŷ = y` at `period_id - 52`), scored per fold. Beat both or nothing
   downstream is trustworthy.
2. Metrics module — masked MAE and RMSE on the original case scale, per district
   and aggregate, plus peak-MAE over the top decile of case periods.
3. `scripts/15.train_baseline.py` — GCN+GRU, `L=12`, `h=1`, nine folds.
4. Freeze the per-fold numbers in `results/models/baseline_report.md`.

## Related documents

- [`docs/climate_dataset_schema.md`](climate_dataset_schema.md) — stages 1 and 2
- [`docs/reporting_calendar.md`](reporting_calendar.md) — why `period_id` is the
  only safe sort key
- [`results/models/model_tensors_report.md`](../results/models/model_tensors_report.md)
- [`results/models/adjacency_report.md`](../results/models/adjacency_report.md)
- [`results/models/folds_report.md`](../results/models/folds_report.md)

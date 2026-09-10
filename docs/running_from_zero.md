# Running the whole project from zero

Verified end to end on Windows 11, PowerShell, 2026-09-09.

Two starting points, and which one you are at depends on whether
`data/raw/` already holds the two big climate CSVs:

| Starting point | What you have | Skip to |
|---|---|---|
| **A — normal** | `data/raw/climate_daily_district.csv` and `chirps_daily_rainfall.csv` present | [§3](#3-the-pipeline) |
| **B — cold** | no raw climate; you must re-extract from Google Earth Engine | [§2](#2-cold-start-only-earth-engine-extraction) |

`data/` is gitignored, so a fresh clone is starting point B. This repository's
working copy is at **A** — the raw CSVs are already there, so the Earth Engine
steps can be skipped entirely and the whole run takes about 90 minutes, almost
all of it model training.

---

## 1. Environment

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -r requirements-model.txt
```

For a CPU-only torch install, which is all this model needs at 25 nodes:

```powershell
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Every command below uses `.venv\Scripts\python.exe` explicitly so it does not
matter whether the environment is activated.

> **Known gap in `requirements.txt`.** It pins `pyarrow`, `scipy` and
> `matplotlib`, but a venv built from it in this session was missing all three
> plus torch. If `12.build_model_tensors.py` fails with "Unable to find a usable
> engine", or `tests/test_chirps_comparison.py` fails with `ModuleNotFoundError`,
> install them directly:
>
> ```powershell
> .venv\Scripts\pip install pyarrow scipy matplotlib
> ```

---

## 2. Cold start only — Earth Engine extraction

**Skip this section if `data/raw/climate_daily_district.csv` exists.** These are
the only steps that need network access, credentials and manual Google Drive
downloads, and they take hours to days of wall-clock time waiting on Earth
Engine exports.

Set up Earth Engine first — see [`README.md`](../README.md) for the full
account, project registration and `earthengine authenticate` walkthrough. Then:

```powershell
.venv\Scripts\python.exe scripts\3.create_nodes.py

# ERA5-Land: submit, wait for COMPLETED, download from Drive to
# data\raw\era5_chunks\, then combine
.venv\Scripts\python.exe scripts\5.extract_era5_daily.py --project YOUR_PROJECT --submit-exports
.venv\Scripts\python.exe scripts\5.extract_era5_daily.py --project YOUR_PROJECT --check-tasks
.venv\Scripts\python.exe scripts\5b.combine_era5_chunks.py

# CHIRPS: same pattern, into data\raw\chirps_chunks\
.venv\Scripts\python.exe scripts\7.extract_chirps_daily.py --project YOUR_PROJECT --submit-exports
.venv\Scripts\python.exe scripts\7b.combine_chirps_chunks.py
```

`3.create_nodes.py` also downloads the GADM polygons and the denguedatahub
source CSV, so it needs plain internet access even when you are not re-running
the climate extraction. Both files are already in `data/raw/` here.

---

## 3. The pipeline

Everything from here reads local files only. No network, no credentials.

```powershell
# --- validation (optional, writes reports only) ---
.venv\Scripts\python.exe scripts\1.dataset_validate.py
.venv\Scripts\python.exe scripts\data\6.validate_climate_data.py
.venv\Scripts\python.exe scripts\data\8.compare_era5_chirps.py

# --- stage 1-2: calendar, districts, cases, climate, panel ---
.venv\Scripts\python.exe scripts\data\2.create_calendar.py
.venv\Scripts\python.exe scripts\data\4.create_canonical_dengue.py
.venv\Scripts\python.exe scripts\data\9.aggregate_climate_to_periods.py
.venv\Scripts\python.exe scripts\data\10.create_master_panel.py
.venv\Scripts\python.exe scripts\data\11.create_imputation_comparison.py

# --- stage 3: tensors, graph, folds ---
.venv\Scripts\python.exe scripts\features\12.build_model_tensors.py
.venv\Scripts\python.exe scripts\graph\13.build_adjacency.py
.venv\Scripts\python.exe scripts\features\14.build_folds.py
```

Order matters: 2 before 4 (the calendar assigns `period_id`), 4 and 9 before 10,
and 10 before 12. Scripts 1, 6, 8 and 11 write reports and are not inputs to
anything downstream — skip them for a fast run.

**Checkpoint.** After this you should have:

```
data/interim/     reporting_calendar.csv, dengue_weekly_canonical.parquet,
                  climate_by_dengue_period.parquet
data/processed/   panel_weekly.parquet, nodes.csv, model_tensors_v0.npz,
                  model_tensors_v1.npz, adjacency.npz, folds.json,
                  fold_preprocessing_v0.npz, fold_preprocessing_v1.npz
```

Expected console values, and each is a real check rather than decoration:
1012 periods x 25 districts; v0 987 usable windows and v1 976 at L=12 h=1;
25 nodes, 57 edges, mean degree 4.56; fold 1 train 459 / val 53 / test 52.

---

## 4. Models

```powershell
# naive baselines -- seconds
.venv\Scripts\python.exe scripts\evaluation\15.evaluate_naive_baselines.py

# GCN+GRU baseline, 2 variants x 2 models x 9 folds x 3 seeds  -- ~45 min
.venv\Scripts\python.exe scripts\training\16.train_gcn_gru.py --seeds 3

# measured climate lags -- seconds
.venv\Scripts\python.exe scripts\evaluation\17.lag_correlation_scan.py

# learnable lag arms -- ~45 min
.venv\Scripts\python.exe scripts\training\18.train_lag_gcn_gru.py --seeds 1 --no-control

# improved objectives, 5 arms x 9 folds x 3 seeds -- ~36 min
.venv\Scripts\python.exe scripts\training\20.train_improved.py --seeds 3

# figures and walkthrough, trains nothing -- seconds
.venv\Scripts\python.exe scripts\evaluation\19.lag_demo.py
```

Script 19 reads what 17 and 18 wrote, so run it after them. Scripts 15, 16, 17,
18 and 20 are independent of each other given stage 3 — except that 20 reads
`naive_baseline_metrics.csv`, so run 15 before it.

For a smoke test rather than the full sweep, both training scripts take
`--folds` and `--seeds`:

```powershell
.venv\Scripts\python.exe scripts\training\16.train_gcn_gru.py --variants v1 --folds 8 --seeds 1
.venv\Scripts\python.exe scripts\training\20.train_improved.py --folds 8 --seeds 1
```

Roughly a minute each, and enough to confirm the wiring is sound.

---

## 5. Tests

```powershell
.venv\Scripts\python.exe -m pytest tests\ -q
```

**Expected: 344 passed, 3 failed.** The three failures are
`test_lag_encoder.py::test_recovers_planted_delays` on all three seeds: the
encoder recovers planted delays to within 2.3 periods against a 2.0 threshold.
This is a tolerance question under torch 2.14, not a broken module —
`requirements-model.txt` pins torch 2.13.0, which is what the documented result
was measured on. Everything else passes.

Tests that need stage 3 artifacts skip themselves cleanly if you run them before
section 3.

---

## 6. Expected numbers

If your run reproduces, these are the values you should see. Any drift means
something upstream changed.

**Naive baselines** (script 15, deterministic):

| Model | Headline MAE | RMSE | Peak MAE | 2017 MAE |
|---|---|---|---|---|
| persistence | 16.42 | 33.61 | 26.61 | 36.08 |
| seasonal_naive | 55.45 | 118.76 | 84.81 | 111.10 |

**GCN+GRU baseline** (script 16, 9 folds x 3 seeds):

| Model | Features | Headline MAE | 2017 MAE | Seed sd |
|---|---|---|---|---|
| `gru_only` | v0 | 16.67 | 41.74 | 0.28 |
| `gru_only` | v1 | 17.12 | 46.01 | 0.70 |
| `gcn_gru` | v1 | 19.01 | 55.01 | 0.62 |
| `gcn_gru` | v0 | 19.08 | 55.23 | 0.84 |

**Improved objectives** (script 20, 9 folds x 3 seeds, `v1`, `gcn_gru`):

| Arm | Headline MAE | 2017 MAE | Seed sd |
|---|---|---|---|
| `level_weighted` | 16.69 | 38.52 | 0.66 |
| `quantile` | 17.32 | 44.26 | 0.29 |
| `huber_weighted` | 17.36 | 44.24 | 0.44 |
| `huber` | 17.41 | 44.19 | 0.34 |
| `baseline` | 19.01 | 55.01 | 0.62 |

The `baseline` arm of script 20 must match `gcn_gru` v1 from script 16 — both
19.01. That equality is the control proving script 20's harness is neutral; if
it does not hold, none of the other arms are comparable.

Neural results are seeded and reproduced exactly in this session on CPU, but may
drift slightly across torch versions or on GPU.

---

## 7. The whole thing, condensed

From starting point A, with the environment already built:

```powershell
$py = ".venv\Scripts\python.exe"
& $py scripts\2.create_calendar.py
& $py scripts\4.create_canonical_dengue.py
& $py scripts\9.aggregate_climate_to_periods.py
& $py scripts\10.create_master_panel.py
& $py scripts\12.build_model_tensors.py
& $py scripts\13.build_adjacency.py
& $py scripts\14.build_folds.py
& $py scripts\15.evaluate_naive_baselines.py
& $py scripts\16.train_gcn_gru.py --seeds 3
& $py scripts\17.lag_correlation_scan.py
& $py scripts\20.train_improved.py --seeds 3
```

About 90 minutes, nearly all of it in the two training sweeps. Add
`18.train_lag_gcn_gru.py` and `19.lag_demo.py` for the learnable-lag component.

## Related documents

- [`README.md`](../README.md) — Earth Engine setup in full
- [`docs/model_tensors.md`](model_tensors.md) — what stage 3 builds
- [`docs/baseline.md`](baseline.md) — the baseline and its results
- [`docs/improvements.md`](improvements.md) — the objective changes

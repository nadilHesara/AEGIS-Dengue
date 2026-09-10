# AEGIS-Dengue

District-level dengue forecasting for Sri Lanka: 25 districts, 1,012 weekly
reporting periods (2006-12-23 to 2026-05-17), ERA5-Land/CHIRPS climate plus
case history, a graph-convolution + GRU model over a district contiguity graph.

This file is the single entry point. Section-specific detail lives in
`docs/`; this README states what is true right now, verified against the code
and the committed results as of **2026-09-10** (§8c–§8f and their supporting
runs added this date; §6/§8 GPU numbers re-verified 2026-09-09).

---

## 1. Project status at a glance

| Stage                                                                 | State                                                                                                                                                                                                              |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1. Raw climate extraction (ERA5-Land, CHIRPS)                         | Done. Raw CSVs committed under `data/raw/` (gitignored in a fresh clone — see §4).                                                                                                                                 |
| 2. Canonical panel (25 districts × 1,012 periods)                     | Done. `data/processed/panel_weekly.parquet`.                                                                                                                                                                       |
| 3. Model tensors, graph, walk-forward folds                           | Done. Rebuildable in minutes from the panel.                                                                                                                                                                       |
| 4. Naive baselines                                                    | Done, all 9 folds.                                                                                                                                                                                                 |
| 5. GCN+GRU baseline                                                   | **Done, full 9-fold × 3-seed sweep** (previously only fold 8 had been run — see §6).                                                                                                                               |
| 6. Learnable climate lags (Component A)                               | Done. A documented **negative result with a diagnosed mechanism** — see §7.                                                                                                                                        |
| 7. Objective/loss improvements                                        | Done. Fixes the epidemic-fold failure specifically, not the model generally — see §8.                                                                                                                              |
| 8. Simplex activations for the lag encoder                            | Done. A mechanism result, not an accuracy one — see §8b.                                                                                                                                                           |
| 8c. Hyperparameter search (full GCN+GRU, 9 axes)                      | Done. Random search, validation-fold selection. **Not a real improvement** — the winning config's test gain is inside seed noise — see §8c.                                                                        |
| 8d. Optimiser (Adam vs AdamW) and LR scheduling                       | Done. Neither is established; the useful result is a **mechanism finding** — a plateau scheduler and early stopping on the same metric barely interact — see §8d.                                                  |
| 8e. Graph representation (identity / contiguity / Gaussian / learned) | Done. **No graph beats the identity control.** The Gaussian graph is worse than contiguity; a graph learned end-to-end is a wash. Closes the dual-graph precondition — see §8e.                                    |
| 8f. Multi-horizon forecasting (h = 1–4)                               | Done. **The first real improvement in this repository.** At h=3 and h=4 the model beats same-horizon persistence on MAE _and_ peak MAE, 6–7 of 7 folds, p < 0.01, and the gain is not the 2017 artefact — see §8f. |
| 8g. Layer normalisation before the prediction head                    | Done. **Not established on accuracy** — the headline gain is the 2017 fold and nothing else (±0.31 MAE over the other six). The real result is **24–51% lower seed variance at every horizon** — see §8g.          |
| 8h. Climate ablation at h = 1–4 (three arms, shuffle control)         | Done. **At h=4 climate content is load-bearing**: a shuffle control holding capacity fixed costs +1.55 MAE (p=0.016) and +3.10 peak MAE, removing half of §8f's skill. Monotonic in the horizon — see §8h.          |
| 9. Gated fusion, climate ablation at h=4                              | Dual graph **answered negatively** by §8e. Climate ablation **done** in §8h. See §9.                                                                                                                               |

**The one-paragraph summary of where the model stands:** **at one week ahead, no
configuration in this repository beats persistence** — five separate changes
(§8, §8b, §8c, §8d, §8e) each produced a headline movement inside the seed noise
and concentrated almost entirely in the 2017 fold. The queen-contiguity graph
measurably hurts on every fold, and neither a distance kernel nor a graph learned
end-to-end beats having no graph at all (§8e). **At three and four weeks ahead
the picture changes completely.** §8f finds the model beats same-horizon
persistence by **+6.9% and +10.8% MAE** and, for the first time anywhere in this
project, on **peak MAE** too (+5.2%, +9.8%) — on 6–7 of 7 headline folds, at
p < 0.01, at nearly 4 seed-sd, and with the gain spread across folds rather than
being the usual 2017 artefact. That confirms the explanation §7 measured and §8–§8e
kept running into: at h=1 the previous week's case count carries nearly all the
signal, so there is little left for anything else to add. Remove that crutch and
the model has real skill.

---

## 2. Repository layout

```
scripts/            numbered pipeline, run in order — see §5
                     24.tune_hyperparameters.py — the 9-axis GCN+GRU search (§8c)
                     25.train_optimisers.py — Adam / AdamW / LR schedule (§8d)
                     26.train_graph_variants.py — four adjacencies vs identity (§8e)
                     27.train_multi_horizon.py — h = 1..4, separate and shared (§8f)
src/models/          lag_encoder.py — the learnable-lag module (Component A)
                     simplex_activations.py — alternative simplex maps for its
                     basis mixture (§8b)
                     adaptive_graph.py — the learnable adjacency (§8e)
                     multi_horizon.py — multi-horizon windowing and heads (§8f)
data/raw/             source CSVs (dengue, ERA5, CHIRPS, GADM polygons)
data/interim/         calendar, canonical dengue, climate joined to periods
data/processed/       panel, tensors, adjacency, folds — model-ready arrays
results/              every generated report, metrics CSV and figure
tests/                506 cases collected, all passing under torch 2.11.0+cu128
docs/                 design documents — one topic each, cross-referenced below
```

**`data/` is gitignored.** A fresh clone has none of it; this working copy does,
because the raw extraction has already been run. See §4 for what that means for
setup.

---

## 3. What each doc covers

Read this README first. Go to a doc only for the depth on that topic.

| Doc                                                                | Covers                                                                                                                                                                              | Still accurate?                                                                                                                                                                                                                             |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`docs/running_from_zero.md`](docs/running_from_zero.md)           | Every command to rebuild the project, in order, with expected checkpoint values                                                                                                     | Yes — verified this session, every script run and confirmed                                                                                                                                                                                 |
| [`docs/baseline.md`](docs/baseline.md)                             | The GCN+GRU baseline: architecture, target parameterisation, full 9-fold results, work plan                                                                                         | Findings accurate; exact numbers are the CPU run — README §6 has a newer GPU run, see §10                                                                                                                                                   |
| [`docs/improvements.md`](docs/improvements.md)                     | The objective/loss fix: diagnosis, five arms, full-sweep results, honest limits                                                                                                     | Findings accurate; exact numbers are the CPU run — README §8 has a newer GPU run, see §10                                                                                                                                                   |
| [`docs/model_tensors.md`](docs/model_tensors.md)                   | Tensor shapes, windowing, fold construction, adjacency — the stage-3 reference                                                                                                      | Yes, except the test count ("62 tests") is stale — see §10                                                                                                                                                                                  |
| [`docs/learnable_lags_results.md`](docs/learnable_lags_results.md) | Component A write-up: measured lags, the encoder, why end-to-end learning failed                                                                                                    | Yes — updated this session to flag that `v1 − v0`, its benchmark target, is at best a small negative and on one backbone indistinguishable from zero (§6)                                                                                   |
| [`docs/learnable_lags.md`](docs/learnable_lags.md)                 | The original step-by-step workplan for Component A                                                                                                                                  | **Historical.** Header still says "Status: not started" — the work is done; read `learnable_lags_results.md` for outcomes, this only for the design rationale                                                                               |
| [`docs/climate_dataset_schema.md`](docs/climate_dataset_schema.md) | ERA5 extraction spec: variables, unit conversions, district aggregation, UTC handling                                                                                               | Yes — specification, not results, nothing to date                                                                                                                                                                                           |
| [`docs/reporting_calendar.md`](docs/reporting_calendar.md)         | Why `period_id`, not source year/week, is the only safe sort key                                                                                                                    | Yes — specification                                                                                                                                                                                                                         |
| [`docs/simplex_activations.md`](docs/simplex_activations.md)       | Replacing the lag encoder's softmax: five alternative simplex maps, the sparse-collapse failure mode, full sweep and limits                                                         | Yes — written this session against the run on disk                                                                                                                                                                                          |
| [`docs/hyperparameter_tuning.md`](docs/hyperparameter_tuning.md)   | The 9-axis search over the full GCN+GRU: search space, random-vs-Optuna, validation-only selection, the flat response surface, why the "best" config is not an improvement          | Yes — written this session against the 50-trial run on disk                                                                                                                                                                                 |
| [`docs/optimiser_scheduling.md`](docs/optimiser_scheduling.md)     | Adam vs AdamW vs AdamW+ReduceLROnPlateau: the hooks added to script 16, the three-arm sweep, and why the schedule fires after the kept model is already chosen                      | Yes — written this session against the 9-fold × 3-seed run on disk                                                                                                                                                                          |
| [`docs/graph_representation.md`](docs/graph_representation.md)     | Four adjacencies against the identity control: contiguity, a Gaussian distance kernel, and a learned graph; what the learned one converged to and why it doesn't resemble geography | Yes — written this session against the 9-fold × 3-seed run on disk                                                                                                                                                                          |
| [`docs/multi_horizon.md`](docs/multi_horizon.md)                   | h = 1–4, separate and shared models, persistence rescored per horizon, the significance tests, and why the effect is not the 2017 artefact                                          | Yes — written this session against the 9-fold × 3-seed run on disk                                                                                                                                                                          |
| [`docs/proposal_brief.md`](docs/proposal_brief.md)                 | Source material for a course proposal document, dated 2026-08-04                                                                                                                    | **Superseded.** Written before the full sweep; states fold-8-only numbers as "preliminary" and explicitly forbids citing a 9-fold result. That result now exists — see §6. Keep for the proposal-writing instructions, not for the numbers. |

---

## 4. Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install -r requirements-model.txt
```

**`requirements.txt` is incomplete in practice.** A venv built from it was
missing `pyarrow`, `scipy`, `matplotlib` and torch this session, despite the
first three being pinned in the file. If `scripts/12.build_model_tensors.py`
fails with "Unable to find a usable engine", or tests fail with
`ModuleNotFoundError`, install directly:

```powershell
.venv\Scripts\pip install pyarrow scipy matplotlib
```

**Torch — CPU or GPU.** The model is small (8,193 parameters, 25 nodes) and
was designed to train on commodity CPU:

```powershell
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU, install the matching CUDA build instead — no separate
CUDA Toolkit install needed, the wheel bundles the CUDA runtime:

```powershell
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
```

`scripts/16.train_gcn_gru.py` and `scripts/18.train_lag_gcn_gru.py` already
auto-select `cuda` if available, no flag needed. `scripts/20.train_improved.py`
does the same as of this session.

**Verified GPU speedup, this session, RTX 4070 Laptop GPU, CUDA 12.8:** fold 1
(the largest fold, 46 training epochs) went from roughly 35–70s per model on
CPU to 1.4–2.6s on GPU — a **15–25× speedup**, well beyond what the model's
small parameter count (8,193) would suggest. The full 9-fold × 3-seed sweep
(§6, 44.6 minutes on CPU) should drop to a few minutes on GPU.

One side effect worth knowing: installing the CUDA wheel (`torch==2.11.0+cu128`,
the latest version with prebuilt cu128 wheels at the time) downgrades torch
from the CPU build's `2.14.0`. That downgrade incidentally **fixed** the 3
pre-existing `test_lag_encoder.py` failures mentioned in §5 and §10 — full
suite is 347 passed, 0 failed under `2.11.0+cu128`, where it was 344/3 under
`2.14.0+cpu`. `requirements-model.txt` pins `2.13.0`; none of these three
versions is what's currently installed in every environment, so re-check
`torch.__version__` if numbers drift.

**Two starting points**, depending on whether `data/raw/climate_daily_district.csv`
and `chirps_daily_rainfall.csv` already exist:

- **They exist (this working copy):** skip straight to §5. No network, no
  Google Earth Engine account needed for anything else.
- **They don't (a fresh clone):** you need a Google Earth Engine account and
  must run the extraction first. Full walkthrough in
  [`docs/running_from_zero.md`](docs/running_from_zero.md#2-cold-start-only-earth-engine-extraction).
  This is hours to days of wall-clock time, mostly spent waiting on GEE export
  tasks and manually downloading from Google Drive — there is no way to
  automate that wait.

---

## 5. Running the project

Full detail with checkpoint values and expected numbers at each step is in
[`docs/running_from_zero.md`](docs/running_from_zero.md). Condensed, from a
built environment with the raw CSVs already present:

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
& $py scripts\24.tune_hyperparameters.py --n-trials 50
& $py scripts\25.train_optimisers.py --seeds 3
```

About 90 minutes for the pipeline through script 20, almost all of it in the two
training sweeps; script 24's search adds roughly another 75 minutes for 50
trials on GPU, script 25's three-arm sweep about 2 minutes, and script 26's
four-graph sweep about 6, and script 27's multi-horizon sweep about 7. Add
`18.train_lag_gcn_gru.py --seeds 1 --no-control` then `19.lag_demo.py` for the
learnable-lag component (Component A).

For a smoke test rather than a full sweep, both training scripts accept
`--folds` and `--seeds`:

```powershell
& $py scripts\16.train_gcn_gru.py --variants v1 --folds 8 --seeds 1
& $py scripts\20.train_improved.py --folds 8 --seeds 1
& $py scripts\24.tune_hyperparameters.py --tune-folds 8 --n-trials 5 --trial-seeds 1 --final-seeds 1
& $py scripts\25.train_optimisers.py --folds 8 --seeds 1
```

Roughly a minute each.

**Tests:**

```powershell
.venv\Scripts\python.exe -m pytest tests\ -q
```

**506 tests collected, 506 pass** (some parametrize into multiple cases; count
grew from 293 as `test_improved_losses.py`, `test_tuning.py`,
`test_optimisers.py`, `test_graph_variants.py`, `test_multi_horizon.py` and
others were added) — verified under `torch==2.11.0+cu128`, with `pyarrow`, `scipy` and
`matplotlib` also installed. An earlier state this session was **344 passed, 3
failed** under `torch==2.14.0+cpu`
earlier this session: `test_lag_encoder.py::test_recovers_planted_delays` on
all three seeds, the encoder recovering planted delays to within 2.3 periods
against a 2.0 threshold. That's a tolerance question tied to the exact torch
build, not a broken module — it silently resolved itself when the CUDA install
(§4) pulled in `2.11.0+cu128` instead. `requirements-model.txt` pins `2.13.0`,
which is what the original result was measured on and is untested here. If you
see the 3 failures return, check `torch.__version__` before assuming a
regression.

### Weather data setup (Earth Engine), in full

This project uses **Google Earth Engine** to extract district-level weather
data. Skip this whole section if `data/raw/` already has the combined CSVs.

```text
ERA5-Land:  ECMWF/ERA5_LAND/HOURLY
CHIRPS:     UCSB-CHG/CHIRPS/DAILY
Boundaries: GADM 4.1 ADM1 (used for the contiguity graph and centroids)
```

**Google Earth Engine setup**

1. Create a Google Cloud project.
2. Enable the Google Earth Engine API and register the project for access.
3. Authenticate locally: `earthengine authenticate --force`
4. Set the project: `earthengine set_project YOUR_PROJECT_ID`
5. Test: `python -c "import ee; ee.Initialize(project='YOUR_PROJECT_ID'); print('ok')"`

**ERA5-Land**

```powershell
python scripts\5.extract_era5_daily.py --project YOUR_PROJECT --submit-exports
python scripts\5.extract_era5_daily.py --project YOUR_PROJECT --check-tasks   # wait for COMPLETED
```

Download every yearly CSV from the Drive folder `era5_chunks` into
`data\raw\era5_chunks\`, removing any duplicate or partial files, then:

```powershell
python scripts\5b.combine_era5_chunks.py       # writes data\raw\climate_daily_district.csv
python scripts\5.extract_era5_daily.py --validate-only
```

**CHIRPS** — the same pattern, into `data\raw\chirps_chunks\`:

```powershell
python scripts\7.extract_chirps_daily.py --project YOUR_PROJECT --submit-exports
python scripts\7.extract_chirps_daily.py --project YOUR_PROJECT --check-tasks
python scripts\7b.combine_chirps_chunks.py     # writes data\raw\chirps_daily_rainfall.csv
python scripts\7.extract_chirps_daily.py --validate-only
```

The full specification of variables, unit conversions, UTC handling and
district-polygon aggregation is in
[`docs/climate_dataset_schema.md`](docs/climate_dataset_schema.md).

---

## 6. The baseline — verified 9-fold, 3-seed results

**Two independent full sweeps exist**, both 9 folds × 3 seeds, same code, same
data, same seeds — one on CPU (`torch==2.14.0+cpu`, 2026-09-09, 44.6 min), one
on GPU after the CUDA install (`torch==2.11.0+cu128`, same day, ~2 min). They
agree on every headline conclusion and disagree at the second decimal, which is
expected: GPU matmul/cuDNN kernels are not bit-identical to CPU even at a fixed
seed, and the torch minor version also changed. **The GPU numbers below are the
current ones on disk**; if you rerun this on CPU or under a different torch
build, expect the same story with small (~0.1–0.4 MAE) drift, not different
conclusions.

| Model       | Features | Headline MAE | RMSE      | Peak MAE  | 2017 MAE  | Seed sd |
| ----------- | -------- | ------------ | --------- | --------- | --------- | ------- |
| persistence | —        | **16.42**    | **33.61** | **26.61** | **36.08** | —       |
| `gru_only`  | v1       | 16.68        | 35.25     | 28.06     | 42.97     | 0.53    |
| `gru_only`  | v0       | 16.86        | 35.94     | 28.43     | 43.53     | 0.39    |
| `gcn_gru`   | v0       | 18.60        | 39.68     | 29.81     | 51.87     | 0.30    |
| `gcn_gru`   | v1       | 18.98        | 40.36     | 30.36     | 54.81     | 0.68    |

Headline = mean over folds 1, 2, 3, 6, 7, 8, 9. Folds 4/5 (COVID) reported
separately. `gru_only` is the identical model with the adjacency replaced by
the identity — the control for what the graph contributes.

**Three findings, all correcting the previous state of this repo:**

1. **No trained model beats persistence on the headline mean.** The best,
   `gru_only` on v1 (16.68), still loses to persistence (16.42). Models do beat
   persistence on 5 of the 7 headline folds; they lose the mean entirely on the
   2017 epidemic fold, which is large enough to flip the average.
2. **The graph is not neutral — it measurably hurts.** `gru_only` beats
   `gcn_gru` on **all nine folds**, by roughly 1.9–2.3 MAE on the headline mean
   against a seed sd of 0.30–0.68 — a margin of several sd on every single fold,
   not just on average (fold-by-fold gap ranged 0.12 to 11.84 MAE in the GPU
   run, smallest gap still exceeding seed noise). This had been an open, "worth
   watching" question at one seed; at 9 folds and 3 seeds it is decided.
3. **Hand-specified climate lags (`v1`) are not clearly worth anything, and the
   sign is backbone-dependent.** On `gcn_gru`, `v0` (18.60) beats `v1` (18.98)
   by 0.38 MAE, larger than the 0.30–0.68 seed sd — a real, if small, negative
   result for `v1`. On `gru_only`, the two GPU-run sweeps disagree with each
   other: the CPU run put `v0` ahead of `v1`, the GPU run put `v1` ahead of
   `v0` (16.68 vs 16.86) — a 0.18 MAE gap, _smaller_ than either run's seed sd
   (0.39–0.70). **The honest statement is that `v1 − v0` is at best negative and
   at worst indistinguishable from zero, not confidently negative on both
   backbones.** `v1 − v0` was the target Component A's learnable-lag module was
   built to beat; either way, that target was never solidly positive, which
   still reframes what Component A's result means (§7).

Full per-fold table and discussion: [`docs/baseline.md`](docs/baseline.md) §3
— **note that doc currently reflects the CPU run's exact numbers, not the GPU
rerun above; both support the same three findings.**

---

## 7. Component A — learnable climate lags (done, negative result)

**The question.** The baseline hand-picks fixed climate lag windows (trailing
4/8/12-period means, `v1`), identical for all 25 districts. Does letting the
model learn a per-district, per-feature delay instead do better?

**The answer: no, and the mechanism is diagnosed, not just observed.**

- An independent, non-trained cross-correlation scan (`scripts/17`) measures
  the real rainfall-to-dengue delay directly: **5–10 weeks across the 25
  districts**, median 8.
- The learnable encoder (`src/models/lag_encoder.py`, 548 parameters, 6
  Gaussian basis kernels over a district embedding) **provably recovers
  planted delays** in a synthetic test — this is the part of the module that
  works.
- Trained end-to-end on the real forecasting loss, it loses: **20.67 MAE
  against 18.36 for the hand-picked windows and 18.07 for no lag treatment at
  all** (single seed, headline folds).
- **Why:** at one-week-ahead, the previous period's case count carries nearly
  all the forecasting signal — dropping every climate channel costs only +0.01
  MAE on recent normal folds. A parameter driven by a gradient that carries no
  delay information drifts to the boundary of its range: measured and learned
  delays correlate at **r = −0.16**.
- **Now known additionally:** the `v1 − v0` target this component was
  benchmarked against is, at best, a small negative and on one backbone
  indistinguishable from zero (§6, finding 3) — hand-specified lags were never
  solidly worth anything either. The encoder wasn't just failing to beat a
  clearly good baseline; the baseline it was compared to was itself unproven.

Full write-up, figures and what may not be claimed:
[`docs/learnable_lags_results.md`](docs/learnable_lags_results.md).

---

## 8. Objective/loss improvements — verified 9-fold, 3-seed results

**The diagnosis, before any change was made.** The baseline trains on masked
MSE over the log1p residual but is scored with MAE over raw case counts. A
constant 0.20 log-space error is 1.3 cases at a level of 5 and 332 cases at a
level of 1500 — the loss spends equal effort on both, while MAE counts the
second ~250× more. Worse, the over-weighted small-count cells are also the
_noisiest_ in log space (sd 0.79 vs 0.375 for large-count cells). 14% of test
cells hold 61% of case volume. The model was optimizing the wrong objective.

**The fix** (`scripts/20.train_improved.py`): five arms — Huber, level-weighted
MSE, both combined, and quantile/pinball — that change **only the loss**,
reusing script 16's training loop, folds, masks, graph and architecture
verbatim (the `baseline` arm reproduces script 16's `gcn_gru` v1 number
exactly — 18.98 both, in the current GPU run — the control proving the harness
is neutral).

**As with §6, this is the current GPU run** (`torch==2.11.0+cu128`); the CPU
run from earlier the same day is what `docs/improvements.md` still shows. Both
support the same conclusions with second-decimal drift.

| Arm                  | Headline MAE | vs persistence | 2017 MAE  | Seed sd |
| -------------------- | ------------ | -------------- | --------- | ------- |
| persistence          | **16.42**    | —              | **36.08** | —       |
| **`level_weighted`** | **16.59**    | −1.0%          | **38.06** | 0.57    |
| `quantile`           | 17.14        | −4.4%          | 42.75     | 0.36    |
| `huber_weighted`     | 17.71        | −7.9%          | 46.66     | 0.51    |
| `huber`              | 18.17        | −10.7%         | 49.76     | 0.65    |
| `baseline`           | 18.98        | −15.6%         | 54.81     | 0.68    |

Seed-mean ensemble of `level_weighted`: **16.36 MAE** against persistence's
16.42 — a 0.06 margin, closer to a tie than the CPU run's 0.03.

**What this establishes, stated at the same level of honesty as §6:**

- `level_weighted` cuts headline MAE from 18.98 to 16.59 (**−12.6%**), a gap of
  roughly 4 pooled seed-sd — solidly established.
- **The entire headline gain is one fold.** Fold 1 (2017) contributes 100.1% of
  the improvement; averaged over the other six headline folds the change is
  **−0.004 MAE**, essentially zero. This is an epidemic-conditions fix, not a
  general improvement — the diagnosis predicted exactly this, since the
  objective mismatch scales with how wide the case distribution is. (CPU run:
  101.7% / −0.046 MAE — same story, tighter margin here.)
- **The ensemble ties persistence, it does not beat it.** 16.36 vs 16.42 is a
  0.06 MAE margin against a 0.57 seed sd — not distinguishable from noise.
- **Peak MAE still loses to persistence on every arm.** The workplan's success
  criterion for this item (fold-1 peak MAE below persistence, 39.42) is not
  met by the single-seed mean (42.32) or the ensemble (41.01).
- Mechanism confirmed directly and, in this run, more strongly: on fold-1 cells
  above 200 cases, the predicted/actual ratio moves from 0.594 to 0.844
  (CPU run: 0.624 to 0.740), and 23 of 25 districts improve (CPU run: 22 of 25).

Full results, per-fold tables and limits: [`docs/improvements.md`](docs/improvements.md)
— **reflects the CPU run's exact numbers; both runs support the same findings.**

---

## 8b. Simplex activations for the lag encoder — done, mechanism result

**First, a correction to a common assumption about this model.** The GCN+GRU is
a _regression_ forecaster — it predicts a case count and is scored with MAE. It
has **no classification head and no output softmax**. The only softmax in the
codebase is in `src/models/lag_encoder.py`, normalising a six-Gaussian-bump
mixture into a delay kernel: a structural simplex constraint, not a
classification activation. "Change the activation function" therefore lands
there and nowhere else.

**The question.** That softmax was never chosen, it was the default way onto a
simplex, and it saturates: `scripts/16`'s `parameter_groups` already documents
the symptom ("the gradient reaching them is far smaller than the gradient
reaching the GRU") and patches it with a 10x learning rate. Five alternatives
were tested — `temp_softmax`, `sparsemax`, `entmax15`, `floored_entmax15`,
`gumbel_softmax` — against the unchanged softmax as control, varying nothing
else.

**The mechanism finding, which is the real result.** Sparse simplex maps have an
**absorbing state**. When a mixture collapses onto one bump the map is locally
constant, the gradient is exactly zero, and that district-feature pair can never
recover. Measured on fold 8 over 40 epochs:

| Activation         | Initial grad | Mean support | Dead pairs |
| ------------------ | ------------ | ------------ | ---------- |
| `softmax`          | 7.2e-05      | 5.73         | 2%         |
| `sparsemax`        | **4.1e-04**  | 1.30         | **74%**    |
| `entmax15`         | 1.9e-04      | 1.90         | **58%**    |
| `floored_entmax15` | 1.8e-04      | 6.00         | **0%**     |

Collapse is progressive (begins ~epoch 5, compounds), so a warm-up would not
prevent it, and it replicates on fold 1. **Do not ship bare `sparsemax` or
`entmax15` in this encoder** — three quarters of the delay kernels freeze and
nothing in the training logs would say so. `floored_entmax15` (2% uniform mass
mixed back) removes the failure entirely while keeping the highest gradient.

**Accuracy: no arm improves the model.** Best is `sparsemax` on `gru_only`,
16.67 vs the control's 17.42 — a 0.75 margin against a 0.34 seed sd, which
clears this project's usual bar and still should not be reported as an
improvement. **Fold 1 (2017) contributes 96% of it; the other six headline folds
move by +0.033 MAE, and a paired t-test over folds gives t = +1.05.** Every arm
on both backbones has this shape (fold-1 share 96–117%, ex-fold-1 movement
±0.03). This is the same single-fold artefact §8 records for the loss work, so
`scripts/22` now prints the fold decomposition directly beneath the seed-sd
verdict — that test alone has now been misleading twice here.

**And a flat result was the prediction, not a disappointment.** §7 established
that at horizon 1 the previous period's case count carries nearly all the
signal; dropping every climate channel costs +0.01 MAE. No change to how climate
is _smoothed_ can move a headline that climate barely enters.

**Horizon 4, the predicted place for a signal, shows a trend but not a result.**
The same sweep at `--horizon 4` — where the measured 5–10 week delay should
become load-bearing — has `gumbel_softmax` on `gru_only` at +0.79 MAE over the
control, and this time the epidemic fold is only 58% of it: the other six folds
move **+0.38 MAE** (against +0.03 at h=1). But the paired t over folds is
**t = +1.63**, short of significance at seven folds, and the leading arm is the
_stochastic_ one — `sparsemax` and `entmax15`, whose behaviour is understood,
sit at +0.42 and +0.35. On `gcn_gru` there is nothing (t = +0.64, other folds
−0.25). The honest reading: the activation matters more at longer horizons,
directionally as predicted, but three seeds do not establish an improvement.

**Delay recovery** (learned vs `scripts/17`'s measured per-district rainfall
delay) does not separate at either horizon: at h=1, `floored_entmax15` +0.41,
`sparsemax` +0.40, softmax +0.36 on `gru_only`, and every arm near zero on
`gcn_gru`; at h=4 the sparse arms edge up (+0.20) against softmax's +0.01 but
the per-fold spread (sd ≈ 0.2) still swamps the gap. Note that plain softmax
scoring **+0.36** at h=1 sits against the **−0.16** in
`docs/learnable_lags_results.md` — a different single-seed configuration, so not
a contradiction, but −0.16 should not be quoted as _the_ softmax number without
those conditions. All of this is bounded by `scripts/17` warning that rainfall's
peak correlation (+0.036) is below its own 0.10 resolvability threshold.

**One inversion worth carrying forward:** `floored_entmax15`, the arm built to
be robust, is the _worst_ on `gru_only` at h=4. It removes the dead-gradient
failure but is not a free upgrade in every regime.

Full write-up, tables and limits: [`docs/simplex_activations.md`](docs/simplex_activations.md).

---

## 8c. Hyperparameter search — full GCN+GRU, verified against the frozen test protocol

**The question.** Every training knob in the baseline (`scripts/16`'s `DEFAULTS`)
was set by hand, once, on fold 8, before the full sweep existed. Does a
systematic search over the joint space find a configuration the hand-picked one
missed?

**The setup** (`scripts/24.tune_hyperparameters.py`). Random search (default; an
Optuna TPE backend is also wired in and used if `optuna` is installed) over nine
axes — the full commissioned grid, **27,648 points**:

| Axis            | Values                 | Baseline |
| --------------- | ---------------------- | -------- |
| `lookback`      | 8, 12, 16, 24          | 12       |
| `gcn_hidden`    | 16, 32, 64             | 32       |
| `gcn_layers`    | 1, 2                   | 2        |
| `gru_hidden`    | 32, 64, 128            | 32       |
| `gru_layers`    | 1, 2                   | 1        |
| `dropout`       | 0, 0.1, 0.2, 0.3       | 0.2      |
| `learning_rate` | 1e-4, 3e-4, 1e-3, 3e-3 | 3e-3     |
| `batch_size`    | 16, 32, 64             | 64       |
| `weight_decay`  | 0, 1e-5, 1e-4, 1e-3    | 1e-4     |

`gru_hidden` and `gru_layers` are knobs the baseline's `GCNGRU` does not expose
(it ties the GRU width to the graph-conv width and hardcodes one layer), so the
script defines `TunableGCNGRU`: the same spatial-then-temporal arrangement, with
a linear projection inserted only when `gcn_hidden != gru_hidden`. When the
search lands on matched widths and a single GRU layer it is the baseline
architecture exactly — a test asserts the parameter counts match. Everything
else — training loop, early stopping, optimiser, folds, preprocessing, the
anchored residual target, the masked metric — is imported from `scripts/16`
unchanged.

**Selection is validation-only.** The objective minimised is mean masked MAE on
the **validation split** of the seven headline folds, seeds averaged. The test
split is never read during the search. Only after the configuration was fixed was
it retrained with 3 seeds on all nine folds and scored on the test split by the
same metric the baseline uses.

**Run on disk: 50 trials, random search, seed 0, 2 seeds/trial, ~73 min GPU.**

Best configuration selected on validation: `lookback=16`, `gcn_hidden=32`,
`gcn_layers=1`, `gru_hidden=128`, `gru_layers=2`, `dropout=0.2`,
`learning_rate=3e-4`, `batch_size=32`, `weight_decay=1e-5` — validation MAE
**14.75** against the hand-picked config's validation MAE in the same trial set.

| Model                                      | Headline MAE | Headline peak MAE | 2017 MAE  | Seed sd |
| ------------------------------------------ | ------------ | ----------------- | --------- | ------- |
| persistence                                | **16.42**    | **26.61**         | **36.08** | —       |
| `gru_only` v1 (README headline best)       | 16.68        | 28.06             | 42.97     | —       |
| **tuned `gcn_gru` v1**                     | 18.23        | 29.22             | 49.54     | 0.42    |
| baseline `gcn_gru` v1 (script 16 defaults) | 18.98        | 30.36             | 54.81     | —       |

**Did tuning improve the models? No — not by a margin this run establishes.**

1. **The like-for-like gain is inside the noise.** Against `gcn_gru` v1 with
   script 16's defaults, tuning moves the test headline MAE from 18.98 to 18.23,
   **−0.75 (−4.0%)**. The tuned model's seed sd is 0.42, so −0.75 is under two
   seed-sd — not a result this run can call real. It moves in the right
   direction; it does not clear the bar this project uses for every other
   comparison (§6, §8, §8b all apply the same "read the seed sd first" rule).
2. **It changes nothing about the standing picture.** The tuned `gcn_gru` still
   loses to the no-graph control `gru_only` (16.68), still loses to persistence
   on the headline mean (18.23 vs 16.42) and on peak MAE (29.22 vs 26.61), and
   fold 1 (2017) is still where the headline gap lives (tuned 2017 MAE 49.54).
   Tuning a backbone that §6 showed the graph actively hurts does not recover
   what the graph costs.
3. **The response surface is flat.** Averaged over all 50 trials, mean
   validation MAE moves ~0.1–0.2 across the _entire_ rest of the grid. The only
   axes with a visible signal are "not `learning_rate=3e-3`" (15.07 vs ~14.93
   elsewhere) and "not `dropout=0`" (15.10 vs ~14.90) — and the baseline already
   sits at `dropout=0.2`. There is no rich optimum being missed; the model's
   accuracy is close to insensitive to its hyperparameters on this data, which
   is itself consistent with §7's finding that at horizon 1 the forecast origin
   carries nearly all the signal.

The search is reproducible from `--search-seed` and `--n-trials` alone (random
search) and re-runnable at any budget; `--tune-folds`, `--trial-seeds` and
`--final-seeds` scale it down for a smoke run. Selection touching validation only
is enforced in code and covered by `tests/test_tuning.py`.

Full write-up, the per-fold table, the top-15 trials and the per-axis response:
[`docs/hyperparameter_tuning.md`](docs/hyperparameter_tuning.md).

---

## 8d. Optimiser and learning-rate schedule — done, one accuracy null and one mechanism result

**The two questions.** (1) Adam adds `wd * w` to the gradient, where Adam's own
per-parameter normalisation then rescales it — so the decay a parameter actually
receives is `wd` divided by a running estimate of its gradient magnitude, and
parameters with small gradients get decayed far harder than those with large
ones. AdamW applies the decay directly to the weight instead. At the baseline's
`weight_decay=1e-4` that is a real difference in what the regulariser does.
(2) A fixed learning rate is either too small for the start of training or too
large for the end; `ReduceLROnPlateau` removes the choice. §8c gave a specific
reason to expect this to matter — `learning_rate` is one of only two axes with
measurable signal in the whole 9-axis grid, and the baseline's `3e-3` is the
_worst_ of the four values tried.

**The setup** (`scripts/25.train_optimisers.py`). Three arms — `adam` (control),
`adamw`, `adamw_scheduled` — changing only the optimiser and the schedule.
Script 16 previously hardcoded `torch.optim.Adam` inside `train_one` with no hook
for it, so rather than copy the loop it gained two backward-compatible hooks,
`make_optimiser` and `make_scheduler`, both defaulting to the baseline's
behaviour. The `adam` arm passes Adam back through the hook and **reproduces
script 16's committed fold-1 MAE of 54.81 exactly** — bit-identical predictions,
asserted by test on both synthetic and real fold data.

Early stopping keeps the baseline's patience of 15; the scheduler gets 5, so it
can fire ~3 times before early stopping can trigger. The scheduler is stepped
_after_ early stopping has seen the epoch, so it cannot change which epoch is
selected as best — only what the optimiser does next.

**Run: 9 folds × 3 seeds × 3 arms, 1.9 min GPU.**

| Arm               | Headline MAE | vs `adam` | Peak MAE | 2017 MAE | Mean best epoch | Seed sd |
| ----------------- | ------------ | --------- | -------- | -------- | --------------- | ------- |
| `adamw`           | **18.67**    | −0.31     | 29.89    | 52.81    | 20.0            | 0.97    |
| `adamw_scheduled` | 18.90        | −0.08     | 30.26    | 53.98    | 19.1            | 0.73    |
| `adam` (control)  | 18.98        | —         | 30.36    | 54.81    | 20.6            | 0.68    |

**Neither change is established.** AdamW's −0.31 MAE is **0.38 pooled seed-sd**,
a paired t over the seven headline folds gives **t = −1.11, p = 0.31**, it wins
on only **4 of 7 folds**, and **91% of the gain is fold 1** — the other six folds
move −0.03 MAE, i.e. nothing. `adamw_scheduled` is weaker still (−0.08, t =
−0.58, p = 0.58) and its fold-1 share is **147%**, meaning the non-epidemic folds
moved the _wrong_ way. This is the fourth time a headline movement in this repo
has turned out to be the 2017 epidemic fold and nothing else (§8, §8b, §8c); at
this point that is a prior, not a surprise.

**The mechanism finding, which is the result worth carrying forward.** The
learning-rate traces (`optimiser_lr_traces.csv`, all 27 scheduled runs) show
that **in 23 of 27 runs the rate did not drop until _after_ the best epoch had
already been saved.** The reason is mechanical: `ReduceLROnPlateau` fires after 5
epochs without improvement, and early stopping keeps the best model from _before_
that plateau began — so the first reduction necessarily lands inside the
early-stopping wait, when the kept model is already fixed. On fast-converging
folds it is absolute: fold 2 reaches its best epoch at 2–3 while the first drop
lands at 9–10, and the arms' MAEs are identical to three decimals.

**So a plateau scheduler and early-stopping-on-the-same-metric are close to
mutually exclusive as configured.** Giving a schedule a real chance requires
either a much longer early-stopping patience (which changes the baseline protocol
and breaks comparability with every committed number) or a schedule that does not
wait for a plateau — cosine annealing or a fixed step decay, which reduce the
rate _during_ productive training. The second is the cleaner next test and the
`make_scheduler` hook takes it with no further changes to script 16.

Two secondary observations: AdamW does **not** converge faster (20.0 epochs vs
20.6, the same within noise), and it is **noisier** (seed sd 0.97 vs 0.68) —
part of why its mean advantage fails the sd test.

Full write-up, per-fold tables, the LR-trace analysis and the significance
tests: [`docs/optimiser_scheduling.md`](docs/optimiser_scheduling.md).

---

## 8e. Graph representation — done, and it closes the dual-graph question

**The question.** §6 established that queen contiguity does not help, it hurts.
That is a statement about _one_ graph, and two readings survive it: either the
graph is wrong (contiguity is a poor prior for a country 430 km long where
Colombo and Galle are 100 km apart, share no border, and are both wet-zone
coastal), or there is no spatial signal to find at horizon 1. Separating them
needs more than one alternative, because a fixed alternative that fails could
always be the wrong fixed alternative.

**The setup** (`scripts/26.train_graph_variants.py`, `src/models/adaptive_graph.py`).
Four adjacencies through one GCN+GRU, changing nothing else:

| Arm          | Graph                                                                                                                     |
| ------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `identity`   | `I` — the control. Substituting the identity makes the graph conv a per-node linear layer, so this arm **is** `gru_only`. |
| `contiguity` | `A_norm`, queen contiguity                                                                                                |
| `gaussian`   | `A_gaussian_norm`, `exp(-(d/50km)²)` on centroid distance — built by `scripts/13` and never before trained on             |
| `adaptive`   | learned end-to-end: `softmax(relu(E1 @ E2ᵀ))`, two `[25, 8]` embedding tables, 400 parameters                             |

**The reference is `identity`, not `contiguity`** — contiguity already loses to
doing nothing, so beating it would establish nothing. Both controls reproduce
their committed numbers exactly (`identity` fold 1 = 42.97 = `gru_only` v1;
`contiguity` = 54.81 = `gcn_gru` v1). The adaptive arm's graph learning rate has
a documented failure mode at both ends — at 1× the embeddings never leave their
uniform initialisation and the arm scores _worse_ than the control; at 10× the
graph collapses toward one-hot rows — so it is **selected per fold on the
validation split** from {1, 3, 10, 30}×, never on test.

**Run: 9 folds × 3 seeds × 4 arms, 6.2 min GPU.**

| Arm                  | Headline MAE | vs `identity` | Peak MAE | 2017 MAE  | Folds improved | p    | Seed sd |
| -------------------- | ------------ | ------------- | -------- | --------- | -------------- | ---- | ------- |
| `adaptive`           | **16.61**    | −0.07         | 28.08    | **40.69** | 2/7            | 0.85 | 0.39    |
| `identity` (control) | 16.68        | —             | 28.06    | 42.97     | —              | —    | 0.53    |
| `contiguity`         | 18.98        | +2.30         | 30.36    | 54.81     | 0/7            | 0.20 | 0.68    |
| `gaussian`           | 19.25        | +2.57         | 30.89    | 56.81     | 0/7            | 0.22 | 0.46    |

**No graph beats no graph.** That is the answer.

1. **The Gaussian graph does not rescue the idea — it is worse than contiguity.**
   19.25 vs 18.98, losing on all seven headline folds, and worse than contiguity
   on the epidemic fold (56.81 vs 54.81). This was the specific alternative §9
   named as the thing to test first, and it fails. Distance-based spatial
   smoothing is not the missing ingredient.
2. **The learned graph is a wash.** −0.07 MAE is 0.16 seed-sd, t = −0.19,
   p = 0.85, improving only **2 of 7 folds**. Its headline is the identity's with
   a different fold distribution: a large fold-1 gain (−2.28) paid for by small
   losses on five of the other six. Over the non-epidemic headline folds it is
   **+0.30 MAE — worse than no graph.** Fifth consecutive change with this shape.
3. **The learned graph found structure, but not geography.** Normalised row
   entropy 0.79 with a largest weight 12× uniform — it is not the identity in
   disguise. But its correlation with the contiguity graph is **−0.108**,
   slightly negative, and the collapsed runs pair districts arbitrarily
   (Ampara→Kurunegala, Matale→Mannar). More consistent with a statistical
   shortcut exploited during an unusual year than with a transmission pathway.

**The one genuinely new number.** The adaptive arm's fold-1 result is the largest
single-fold improvement over the identity control anything in this repo has
produced: **42.97 → 40.69** single-seed, **42.57 → 39.26** as a seed-mean
ensemble. And the adaptive ensemble's headline, **16.21**, is the first number
here below persistence's 16.42 — beating it on 6 of 7 folds. But the paired t is
−0.37, **p = 0.72** (fold 1's magnitude dominates the variance), so it ties
persistence rather than beating it, exactly as §8's `level_weighted` ensemble
did — and **peak MAE still loses** (27.47 vs 26.61), which is the criterion an
outbreak warning system is judged on.

**What this closes.** `docs/baseline.md` §4 item 7 required establishing that
_some_ graph beats the identity before building a season-gated dual graph. That
precondition is now tested and **not met**: a border graph, a distance kernel and
a graph learned from the loss itself all fail. Building a gated mixture of two
components that each lose to the identity would be building on a measured
negative. The remaining open version of this idea is the adaptive graph at a
longer horizon (§9 item 1), where the standing explanation for why spatial
structure adds nothing — at h=1 the forecast origin dominates — no longer holds.

Full write-up, per-fold tables, the learned-graph analysis and the LR-selection
table: [`docs/graph_representation.md`](docs/graph_representation.md).

---

## 8f. Multi-horizon forecasting — the first real improvement in this repository

**The question, and why it was the one left.** Every result above is at h=1, and
one measurement explains most of them: §7 found that at one week ahead the
previous period's case count carries nearly all the signal — dropping every
climate channel costs **+0.01 MAE**. That is the standing explanation for §7's
absent gradient, §8's epidemic-only fix, §8b's flat sweep, §8c's flat response
surface and §8e's "no graph beats no graph". It makes a prediction: extend the
horizon, the origin's grip weakens, and everything else the model knows gets room
to matter. The measured rainfall-to-dengue delay is 5–10 weeks, so h=4 is where
it should bite.

**The setup** (`scripts/27.train_multi_horizon.py`, `src/models/multi_horizon.py`).
Two arms — `separate` (one model per horizon, four trunks) and `shared` (one
trunk, four linear heads, summed masked loss) — on both the identity and
contiguity backbones, at h = 1, 2, 3, 4. Windows are cut at the longest horizon
so every horizon is scored on identical forecast origins.

**Persistence is rescored at every horizon**, because it degrades sharply as its
copied value goes stale — comparing an h=4 model to the h=1 baseline would judge
it against a much easier task:

| Horizon              | 1     | 2     | 3     | 4     |
| -------------------- | ----- | ----- | ----- | ----- |
| persistence MAE      | 16.42 | 20.16 | 24.58 | 28.47 |
| persistence peak MAE | 26.61 | 32.51 | 41.48 | 47.98 |

**Run: 9 folds × 3 seeds × 2 arms × 2 backbones × 4 horizons, 6.8 min GPU.**

### Skill over same-horizon persistence

| Backbone     | Arm        | h=1    | h=2    | h=3       | h=4        |
| ------------ | ---------- | ------ | ------ | --------- | ---------- |
| `identity`   | `separate` | −4.7%  | −0.3%  | **+6.9%** | **+10.8%** |
| `identity`   | `shared`   | −1.7%  | +0.6%  | **+5.8%** | **+9.6%**  |
| `contiguity` | `shared`   | −10.8% | −10.7% | −5.1%     | −0.7%      |
| `contiguity` | `separate` | −18.6% | −16.2% | −10.2%    | −5.1%      |

**Monotonic in the horizon, on both backbones and both arms** — the first curve
in this project to move consistently in a predicted direction.

### Peak MAE — the criterion nothing had ever beaten

| Backbone   | Arm        | h=1    | h=2   | h=3       | h=4       |
| ---------- | ---------- | ------ | ----- | --------- | --------- |
| `identity` | `separate` | −11.7% | −5.1% | **+5.2%** | **+9.8%** |
| `identity` | `shared`   | −7.7%  | −3.0% | **+4.7%** | **+8.0%** |

`docs/baseline.md` names peak MAE as the criterion that matters for an outbreak
warning system, and **no previous change in this repository ever beat persistence
on it** — not §8's loss fix, not §8e's graph work. At h=3 and h=4 it is beaten.

### Is it real? Every test the project applies

`identity` backbone against same-horizon persistence, paired over the seven
headline folds:

| Arm        | h   | Δ MAE     | Folds won | t     | p          | Δ in seed-sd |
| ---------- | --- | --------- | --------- | ----- | ---------- | ------------ |
| `separate` | 3   | −1.69     | 6/7       | −2.56 | **0.043**  | **3.69**     |
| `separate` | 4   | **−3.07** | **7/7**   | −4.04 | **0.0068** | **3.80**     |
| `shared`   | 4   | −2.72     | **7/7**   | −3.71 | **0.0099** | **4.40**     |

This project's bar has been "two seed-sd and p < 0.05" throughout. h=4 clears it
on both arms; h=3 clears it on `separate`.

### And it is not the 2017 artefact

Five consecutive changes produced movements that were fold 1 and nothing else.
This one is the opposite — excluding fold 1 entirely, `identity`/`separate`:

| Horizon | All folds Δ | Ex-fold-1 Δ | Ex-fold-1 folds won |
| ------- | ----------- | ----------- | ------------------- |
| 1       | +0.78       | −0.82       | 5/6                 |
| 2       | +0.07       | −1.34       | **6/6**             |
| 3       | −1.69       | −2.15       | **6/6**             |
| 4       | −3.07       | −2.71       | **6/6**             |

**The model beats persistence on every non-epidemic headline fold at h=2, 3 and 4.** Fold 1 itself flips sign (+10.36 at h=1 → −5.18 at h=4). The seed-mean
ensemble pushes h=4 skill to **+11.9%**.

### Shared versus separate

On the identity backbone they are a wash — `shared` wins h=1 and h=2, `separate`
wins h=3 and h=4, all within or near a seed-sd. On contiguity `shared` wins by
1.1–1.3 MAE at every horizon, but that is a regularisation effect (a quarter the
trunk parameters, constrained to serve four targets) limiting how badly it
overfits a graph §8e showed to be harmful — a fix for a problem better solved by
not using the graph. **The argument for `shared` is cost, not accuracy:** ~¼ the
training time (5.6s vs 21.3s on fold 1) and ¼ the trunk parameters, for
statistically indistinguishable accuracy on the backbone that matters.

### Limits, stated plainly

- **h=1 and h=2 still lose or tie.** Nothing here changes §6 for a one-week
  forecast.
- **This does not by itself prove climate is the mechanism.** The skill curve is
  equally consistent with the model exploiting longer-range autocorrelation or
  seasonality that persistence cannot represent. Distinguishing them needs the
  climate-ablation experiment rerun at h=4 — §9 item 1.
- **The graph still hurts at every horizon**, so §8e is unaffected.
- **Raw error still grows.** +10.8% skill at h=4 sits on an absolute MAE of
  25.40 against 17.20 at h=1: better relative to the alternative, not better
  absolutely.

Full write-up, per-fold tables and the significance tests:
[`docs/multi_horizon.md`](docs/multi_horizon.md).

---

## 8g. Layer normalisation before the prediction head — an accuracy null and a variance result

`GRU -> LayerNorm(32) -> dropout -> head`, against the identical model without
the norm. The norm sits on the clean final hidden state, before the dropout the
baseline already applies, with statistics over the hidden axis per (window,
district) row. `baseline` is the same class with `normalise=False` and
reproduces the committed 18.98 exactly, which is the check that the harness is
neutral. 64 extra parameters, on 8,193.

| h | `baseline` | `layer_norm` | Δ | Folds improved | p | **Δ ex-fold-1** |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 18.98 | 17.41 | −1.57 | 3/7 | 0.35 | **−0.02** |
| 2 | 23.79 | 21.73 | −2.05 | 4/7 | 0.32 | **−0.16** |
| 3 | 27.18 | 25.69 | −1.49 | 5/7 | 0.36 | **−0.01** |
| 4 | 29.93 | 27.65 | −2.29 | 5/7 | 0.29 | **−0.31** |

**The headline movement is the 2017 fold and nothing else.** Fold 1 moves −10.9
to −14.2 MAE; no other fold moves by as much as 1.5, the sign is not consistent
fold to fold, and no paired test clears p = 0.05 at any horizon. This is the
fifth change in this project to produce a headline gain that is entirely the
epidemic fold (§8, §8b, §8c, §8d).

**What survives is seed stability.** Mean per-fold seed sd falls at every
horizon — 0.795→0.605 (h=1), 1.294→0.639 (h=2), 1.300→0.824 (h=3), 1.671→1.008
(h=4): 24–51%, same direction throughout, with the absolute gap widening as the
horizon grows. In a project whose effect sizes are routinely under 1 MAE, a
tighter seed spread makes every subsequent comparison cheaper to resolve. That,
not the headline, is the reason to keep the norm. Convergence speed did not move
consistently, so the usual "trains faster" claim is not supported here.

The learned gain stays at 0.905–0.931 with a near-zero bias at every horizon, so
the module did plain normalisation rather than learning to switch itself off —
the flat headline is "normalising did not help the score", which is a different
finding and is why the affine parameters are logged.

Consistent with the standing explanation (§6–§8e): normalisation improved
conditioning measurably and the score still did not move, because the ceiling is
set by the information in the inputs, not by optimisation.

Full write-up: [`docs/layer_norm.md`](docs/layer_norm.md).

---

## 8h. Climate ablation — what actually causes the h=3–4 skill

**The gap §8f left.** §8f proved longer-horizon forecasting produces real skill
and attributed it to the measured 5–10 week rainfall-to-dengue delay. That
attribution was never measured. Skill at h=4 is equally consistent with
persistence going stale faster than the model degrades — the relative margin
grows while climate contributes nothing. §9 named this the key next experiment.

**The setup** (`scripts/29.train_climate_ablation.py`,
`src/models/climate_ablation.py`). Three arms on the identity backbone, v1,
h = 1–4, 9 folds × 3 seeds: `full` (all 23 channels), `no_climate` (the 16
climate-derived channels sliced out — including the rolling 4/8/12-period lags,
which are the channels the delay story actually runs through), and `shuffled`
(all 23 channels, climate time-shuffled within each district).

**The `shuffled` arm is the point.** §6.3 ran the two-arm ablation at h=1 and
flagged it as confounded: dropping 16 of 23 channels changes input width,
parameter count and effective regularisation at once, so a penalty need not be
about weather. The shuffle holds all of that fixed — identical width, identical
parameter count, identical scaler statistics, identical per-district marginals —
and destroys only the alignment between weather and time. It permutes whole
windows, within district, within split, independently per channel, and never
touches `y`, the mask or the anchor.

### The cost of removing climate, by horizon

| h | `no_climate` Δ MAE | `shuffled` Δ MAE | `shuffled` Δ peak MAE | `shuffled` p |
| - | ------------------ | ---------------- | --------------------- | ------------ |
| 1 | −0.25              | +0.02            | +0.40                 | 0.926        |
| 2 | −0.35              | −0.06            | +0.38                 | 0.854        |
| 3 | −0.06              | +0.43            | +1.65                 | 0.226        |
| 4 | **+0.76**          | **+1.55**        | **+3.10**             | **0.016**    |

Positive means the ablation hurt. **Both metrics rise monotonically with the
horizon** — the shape the 5–10 week delay predicts, and not the shape
persistence-decay-only would produce (flat near zero everywhere).

### How much of §8f's skill is climate?

| h   | persistence | `full`     | `no_climate` | `shuffled` |
| --- | ----------- | ---------- | ------------ | ---------- |
| 3   | 24.58       | **+5.0%**  | +5.2%        | +3.2%      |
| 4   | 28.47       | **+10.8%** | +8.1%        | +5.3%      |

At h=4, scrambling climate removes **half** the MAE skill (10.8% → 5.3%) and
about two thirds of the peak-MAE skill (9.8% → 3.3%). At 2.4 seed-sd and
p = 0.016 it clears this project's standing bar, and it is **not** the 2017
artefact: `shuffled` hurts on 6/7 folds and excluding fold 1 the cost is
unchanged (+1.55 → +1.27).

**The h=1 column reproduces §6.3's known anchor** (+0.17 vs the reported ≈+0.01
MAE on recent normal years), so the harness is checked against a known value
before the h=4 number is read.

### What this does and does not establish

- **Established:** at h=4 climate's *temporal content* is load-bearing, on a
  control that holds capacity fixed. This is the first result here to attribute
  an effect to climate content rather than channel count.
- **Not established:** climate is not the whole story — `shuffled` still beats
  persistence by 5.3%, so persistence decay is a real component. And h=3 does
  not clear the bar on its own (p = 0.226).
- **A warning:** `no_climate` alone is weak (+0.76, p = 0.267) and negative at
  h = 1–3. Read without the shuffle control, the plain ablation would have
  supported the opposite conclusion — which is exactly the §6.3 confound.

Full write-up: [`docs/climate_ablation.md`](docs/climate_ablation.md).

---

## 9. What the evidence says to do next

Ordered by what §6–8f actually established. Two directions are closed and one has
just opened. §8c and §8d closed the _training-procedure_ direction: the
hyperparameter response surface is flat and neither optimiser nor schedule moves
the headline out of the noise. §8e closed the _fixed-graph_ direction: three
graphs including one learned end-to-end all fail to beat no graph. What is left
is mostly the horizon.

1. **Rerun §8e's adaptive graph at h=4.** Multi-horizon training is now done
   (§8f) and §8h has shown climate content is load-bearing there, so the
   condition several negative results were waiting on is established. The graph
   question is the one that has not yet been retested under it: `scripts/26`
   takes `--horizon` with no further change. **Superseded in part** — the
   climate half of this item is answered by §8h.

   _Original text:_ `--horizon` exists and is unrun. At h=1 the forecast origin carries
   nearly all the signal, which is _why_ §7's learnable lags found no gradient,
   why §8's fix only bites in epidemic conditions, and the standing explanation
   for why no graph helps in §8e. This is now the single highest-value item,
   because it is the one condition under which several separate negative results
   might change sign at once. §8b already found a directional hint for the lag
   encoder at h=4, and §8e's adaptive graph is the natural companion test —
   `scripts/26` takes `--horizon` with no further change.
2. **Fix lag kernels to the §7 measured delays instead of learning them
   end-to-end**, paired with (1). Decouples the delay estimate from a gradient
   that provably doesn't carry it.
3. **A count likelihood** (negative binomial or Tweedie) instead of Gaussian-
   in-log-space. §8's reweighting is a partial, hand-built approximation to
   what a proper count model would do natively.
4. **Quantile forecasts** at τ ∈ {0.1, 0.5, 0.9} for operational use — the
   pinball loss from §8 already exists; extending it to a real interval is a
   small step. §8f raises the priority: a horizon at which the model actually
   has skill is a horizon at which a calibrated interval is worth publishing.
5. **A count likelihood** (negative binomial or Tweedie) instead of Gaussian-
   in-log-space. §8's reweighting is a partial, hand-built approximation to
   what a proper count model would do natively.
6. **Gated spatial/temporal fusion** (`src/models/gated_fusion.py` exists and is
   unrun at full sweep). §8e weakens the case: the gate chooses per district
   between a graph branch and an identity branch, and §8e found no graph worth
   choosing. Its remaining value is diagnostic — the learned gates would say
   _which_ districts, if any, ever want a graph — rather than an expected
   accuracy gain.
7. **A non-plateau learning-rate schedule** (cosine annealing or a fixed step
   decay), if the schedule question is worth revisiting at all. §8d showed
   `ReduceLROnPlateau` and early-stopping-on-the-same-metric barely interact —
   in 23 of 27 runs the rate never dropped until after the kept model was
   already chosen — so a schedule that reduces the rate _during_ productive
   training is the only version of this idea that has a mechanism to work
   through. `scripts/25`'s `make_scheduler` hook takes one with no further
   change to script 16. Low priority: §8d's accuracy result gives no reason to
   expect much.

---

## 10. Known inconsistencies in the docs

Kept visible rather than silently fixed, so nothing is trusted past what was
actually checked this session:

- `docs/model_tensors.md` states "62 tests, all passing" for the stage-3
  tensor/adjacency/fold tests specifically — still true for that subset, but
  the repository-wide count has grown to **506 collected, 506 passing** under
  `torch==2.11.0+cu128` as more components were added (most recently
  `tests/test_tuning.py`, 14 tests, §8c; `tests/test_optimisers.py`, 16 tests,
  §8d; `tests/test_graph_variants.py`, 25 tests, §8e; and
  `tests/test_multi_horizon.py`, 34 tests, §8f).
- `docs/learnable_lags_results.md` states "323 tests pass repository-wide" —
  that was true when written; it's **554 collected / 554 passed** now, after
  `test_improved_losses.py` (24 tests, §8), `test_tuning.py` (14 tests, §8c),
  `test_optimisers.py` (16 tests, §8d), `test_graph_variants.py` (25 tests, §8e),
  `test_multi_horizon.py` (34 tests, §8f), `test_layer_norm.py` (21 tests,
  §8g) and `test_climate_ablation.py` (27 tests, §8h) were added, one dependency-driven set of
  failures was resolved by installing `scipy` and `matplotlib`, and the 3
  `test_lag_encoder.py` failures stopped once the CUDA torch install (§4) landed
  on `2.11.0+cu128` instead of `2.14.0+cpu`.
- **`scripts/16.train_gcn_gru.py` was modified for §8d.** `train_one` gained two
  optional hooks, `make_optimiser` and `make_scheduler`, and the hardcoded
  `torch.optim.Adam` moved into a `build_optimiser` function that is still the
  default. `train_one`'s return dict gained `learning_rates` and `final_lr`.
  **Behaviour is unchanged when neither hook is passed** — the `adam` control arm
  reproduces script 16's committed fold-1 MAE of 54.81 with bit-identical
  predictions, asserted by test on both synthetic and real fold data, and the
  full baseline sweep is unaffected. Any doc quoting script 16's optimiser as
  "hardcoded Adam" is describing the pre-§8d state.
- `docs/learnable_lags.md` header still reads "Status: not started" — it's a
  workplan document frozen at the point work began; `learnable_lags_results.md`
  is the outcome doc and is current.
- `docs/proposal_brief.md` is dated 2026-08-04 and explicitly forbids citing a
  9-fold result "not yet run" — that result exists now (§6). Treat its numbers
  as historical; its LaTeX/writing instructions are still usable.
- **`docs/baseline.md` §3 finding 1 ("no trained model beats persistence") is
  now horizon-scoped.** It remains true at h=1, which is the only horizon that
  document covers, but §8f shows it is false at h=3 and h=4. Read it as a
  statement about one-week-ahead forecasting, not about the model in general.
  The same applies to every "does not beat persistence" claim in §6–§8e.
- **`docs/baseline.md` and `docs/improvements.md` show the CPU-run numbers
  (`torch==2.14.0+cpu`); §6 and §8 of this README show a second, independent
  full sweep run afterward on GPU (`torch==2.11.0+cu128`).** The two runs agree
  on every headline conclusion and differ at the second decimal — expected,
  since neither GPU kernels nor the torch minor version are bit-identical to
  CPU at a fixed seed. One exception worth knowing: the `gru_only` v0-vs-v1
  ordering flips between the two runs (CPU: v0 ahead; GPU: v1 ahead), and the
  gap is smaller than either run's seed sd either way — this is why §6 states
  `v1 − v0` as "at best negative, at worst noise" rather than a confident
  negative. The `docs/` files were not rewritten to match the GPU run; treat
  this README's §6/§8 tables as current, `docs/baseline.md`/`docs/improvements.md`
  as a fully consistent but numerically superseded snapshot.

None of these affect correctness of the pipeline or the findings in §6–8, which
were independently re-verified this session by rerunning the scripts twice
(CPU and GPU), not by reading the docs.

---

## Related documents

- [`docs/running_from_zero.md`](docs/running_from_zero.md) — full rebuild instructions
- [`docs/baseline.md`](docs/baseline.md) — baseline architecture and full results
- [`docs/improvements.md`](docs/improvements.md) — the objective-fix work
- [`docs/hyperparameter_tuning.md`](docs/hyperparameter_tuning.md) — the 9-axis search and why its best config is not an improvement
- [`docs/optimiser_scheduling.md`](docs/optimiser_scheduling.md) — Adam vs AdamW and the plateau-scheduler mechanism finding
- [`docs/graph_representation.md`](docs/graph_representation.md) — four adjacencies against the identity control
- [`docs/multi_horizon.md`](docs/multi_horizon.md) — h = 1–4, and the first result that beats persistence
- [`docs/layer_norm.md`](docs/layer_norm.md) — LayerNorm before the head: an accuracy null and a seed-variance result
- [`docs/climate_ablation.md`](docs/climate_ablation.md) — what causes the h=3–4 skill: the shuffle control that attributes it to climate
- [`docs/model_tensors.md`](docs/model_tensors.md) — tensors, folds, adjacency
- [`docs/learnable_lags_results.md`](docs/learnable_lags_results.md) — Component A
- [`docs/climate_dataset_schema.md`](docs/climate_dataset_schema.md) — ERA5 extraction spec
- [`docs/reporting_calendar.md`](docs/reporting_calendar.md) — the chronological key

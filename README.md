# AEGIS-Dengue

District-level dengue forecasting for Sri Lanka: 25 districts, 1,012 weekly
reporting periods (2006-12-23 to 2026-05-17), ERA5-Land/CHIRPS climate plus
case history, a graph-convolution + GRU model over a district contiguity graph.

This file is the single entry point. Section-specific detail lives in
`docs/`; this README states what is true right now, verified against the code
and the committed results as of **2026-09-10** (§8c and its supporting run
added this date; §6/§8 GPU numbers re-verified 2026-09-09).

---

## 1. Project status at a glance

| Stage | State |
|---|---|
| 1. Raw climate extraction (ERA5-Land, CHIRPS) | Done. Raw CSVs committed under `data/raw/` (gitignored in a fresh clone — see §4). |
| 2. Canonical panel (25 districts × 1,012 periods) | Done. `data/processed/panel_weekly.parquet`. |
| 3. Model tensors, graph, walk-forward folds | Done. Rebuildable in minutes from the panel. |
| 4. Naive baselines | Done, all 9 folds. |
| 5. GCN+GRU baseline | **Done, full 9-fold × 3-seed sweep** (previously only fold 8 had been run — see §6). |
| 6. Learnable climate lags (Component A) | Done. A documented **negative result with a diagnosed mechanism** — see §7. |
| 7. Objective/loss improvements | Done. Fixes the epidemic-fold failure specifically, not the model generally — see §8. |
| 8. Simplex activations for the lag encoder | Done. A mechanism result, not an accuracy one — see §8b. |
| 8c. Hyperparameter search (full GCN+GRU, 9 axes) | Done. Random search, validation-fold selection. **Not a real improvement** — the winning config's test gain is inside seed noise — see §8c. |
| 9. Dual graph, gated fusion, multi-horizon | Not started. See §9 for what the evidence says to do next. |

**The one-paragraph summary of where the model stands:** no configuration in
this repository beats persistence on the headline mean by a margin that isn't
also achievable by a tie. The queen-contiguity graph is not neutral — it
measurably hurts, on every fold. The entire headline gap between the baseline
and persistence lives in a single fold, the 2017 epidemic; the reweighted
objective closes most of that gap without touching the other eight folds. A
9-axis hyperparameter search over the full GCN+GRU (§8c) does not change this
picture: its validation-selected best config improves the like-for-like baseline
by 4% on the test headline, which is inside the seed noise, and still loses to
both the no-graph control and persistence. None of this is a setback dressed up —
it's what nine years of data and three seeds actually show, and it points at a
specific next step (§9) rather than a vague one.

---

## 2. Repository layout

```
scripts/            numbered pipeline, run in order — see §5
                     24.tune_hyperparameters.py — the 9-axis GCN+GRU search (§8c)
src/models/          lag_encoder.py — the learnable-lag module (Component A)
                     simplex_activations.py — alternative simplex maps for its
                     basis mixture (§8b)
data/raw/             source CSVs (dengue, ERA5, CHIRPS, GADM polygons)
data/interim/         calendar, canonical dengue, climate joined to periods
data/processed/       panel, tensors, adjacency, folds — model-ready arrays
results/              every generated report, metrics CSV and figure
tests/                431 cases collected, all passing under torch 2.11.0+cu128
docs/                 design documents — one topic each, cross-referenced below
```

**`data/` is gitignored.** A fresh clone has none of it; this working copy does,
because the raw extraction has already been run. See §4 for what that means for
setup.

---

## 3. What each doc covers

Read this README first. Go to a doc only for the depth on that topic.

| Doc | Covers | Still accurate? |
|---|---|---|
| [`docs/running_from_zero.md`](docs/running_from_zero.md) | Every command to rebuild the project, in order, with expected checkpoint values | Yes — verified this session, every script run and confirmed |
| [`docs/baseline.md`](docs/baseline.md) | The GCN+GRU baseline: architecture, target parameterisation, full 9-fold results, work plan | Findings accurate; exact numbers are the CPU run — README §6 has a newer GPU run, see §10 |
| [`docs/improvements.md`](docs/improvements.md) | The objective/loss fix: diagnosis, five arms, full-sweep results, honest limits | Findings accurate; exact numbers are the CPU run — README §8 has a newer GPU run, see §10 |
| [`docs/model_tensors.md`](docs/model_tensors.md) | Tensor shapes, windowing, fold construction, adjacency — the stage-3 reference | Yes, except the test count ("62 tests") is stale — see §10 |
| [`docs/learnable_lags_results.md`](docs/learnable_lags_results.md) | Component A write-up: measured lags, the encoder, why end-to-end learning failed | Yes — updated this session to flag that `v1 − v0`, its benchmark target, is at best a small negative and on one backbone indistinguishable from zero (§6) |
| [`docs/learnable_lags.md`](docs/learnable_lags.md) | The original step-by-step workplan for Component A | **Historical.** Header still says "Status: not started" — the work is done; read `learnable_lags_results.md` for outcomes, this only for the design rationale |
| [`docs/climate_dataset_schema.md`](docs/climate_dataset_schema.md) | ERA5 extraction spec: variables, unit conversions, district aggregation, UTC handling | Yes — specification, not results, nothing to date |
| [`docs/reporting_calendar.md`](docs/reporting_calendar.md) | Why `period_id`, not source year/week, is the only safe sort key | Yes — specification |
| [`docs/simplex_activations.md`](docs/simplex_activations.md) | Replacing the lag encoder's softmax: five alternative simplex maps, the sparse-collapse failure mode, full sweep and limits | Yes — written this session against the run on disk |
| [`docs/hyperparameter_tuning.md`](docs/hyperparameter_tuning.md) | The 9-axis search over the full GCN+GRU: search space, random-vs-Optuna, validation-only selection, the flat response surface, why the "best" config is not an improvement | Yes — written this session against the 50-trial run on disk |
| [`docs/proposal_brief.md`](docs/proposal_brief.md) | Source material for a course proposal document, dated 2026-08-04 | **Superseded.** Written before the full sweep; states fold-8-only numbers as "preliminary" and explicitly forbids citing a 9-fold result. That result now exists — see §6. Keep for the proposal-writing instructions, not for the numbers. |

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
```

About 90 minutes for the pipeline through script 20, almost all of it in the two
training sweeps; script 24's search adds roughly another 75 minutes for 50
trials on GPU. Add
`18.train_lag_gcn_gru.py --seeds 1 --no-control` then `19.lag_demo.py` for the
learnable-lag component (Component A).

For a smoke test rather than a full sweep, both training scripts accept
`--folds` and `--seeds`:

```powershell
& $py scripts\16.train_gcn_gru.py --variants v1 --folds 8 --seeds 1
& $py scripts\20.train_improved.py --folds 8 --seeds 1
& $py scripts\24.tune_hyperparameters.py --tune-folds 8 --n-trials 5 --trial-seeds 1 --final-seeds 1
```

Roughly a minute each.

**Tests:**

```powershell
.venv\Scripts\python.exe -m pytest tests\ -q
```

**431 tests collected, 431 pass** (some parametrize into multiple cases; count
grew from 293 as `test_improved_losses.py`, `test_tuning.py` and others were
added) — verified under `torch==2.11.0+cu128`, with `pyarrow`, `scipy` and
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

| Model | Features | Headline MAE | RMSE | Peak MAE | 2017 MAE | Seed sd |
|---|---|---|---|---|---|---|
| persistence | — | **16.42** | **33.61** | **26.61** | **36.08** | — |
| `gru_only` | v1 | 16.68 | 35.25 | 28.06 | 42.97 | 0.53 |
| `gru_only` | v0 | 16.86 | 35.94 | 28.43 | 43.53 | 0.39 |
| `gcn_gru` | v0 | 18.60 | 39.68 | 29.81 | 51.87 | 0.30 |
| `gcn_gru` | v1 | 18.98 | 40.36 | 30.36 | 54.81 | 0.68 |

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
   `v0` (16.68 vs 16.86) — a 0.18 MAE gap, *smaller* than either run's seed sd
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
*noisiest* in log space (sd 0.79 vs 0.375 for large-count cells). 14% of test
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

| Arm | Headline MAE | vs persistence | 2017 MAE | Seed sd |
|---|---|---|---|---|
| persistence | **16.42** | — | **36.08** | — |
| **`level_weighted`** | **16.59** | −1.0% | **38.06** | 0.57 |
| `quantile` | 17.14 | −4.4% | 42.75 | 0.36 |
| `huber_weighted` | 17.71 | −7.9% | 46.66 | 0.51 |
| `huber` | 18.17 | −10.7% | 49.76 | 0.65 |
| `baseline` | 18.98 | −15.6% | 54.81 | 0.68 |

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
a *regression* forecaster — it predicts a case count and is scored with MAE. It
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

| Activation | Initial grad | Mean support | Dead pairs |
|---|---|---|---|
| `softmax` | 7.2e-05 | 5.73 | 2% |
| `sparsemax` | **4.1e-04** | 1.30 | **74%** |
| `entmax15` | 1.9e-04 | 1.90 | **58%** |
| `floored_entmax15` | 1.8e-04 | 6.00 | **0%** |

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
is *smoothed* can move a headline that climate barely enters.

**Horizon 4, the predicted place for a signal, shows a trend but not a result.**
The same sweep at `--horizon 4` — where the measured 5–10 week delay should
become load-bearing — has `gumbel_softmax` on `gru_only` at +0.79 MAE over the
control, and this time the epidemic fold is only 58% of it: the other six folds
move **+0.38 MAE** (against +0.03 at h=1). But the paired t over folds is
**t = +1.63**, short of significance at seven folds, and the leading arm is the
*stochastic* one — `sparsemax` and `entmax15`, whose behaviour is understood,
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
a contradiction, but −0.16 should not be quoted as *the* softmax number without
those conditions. All of this is bounded by `scripts/17` warning that rainfall's
peak correlation (+0.036) is below its own 0.10 resolvability threshold.

**One inversion worth carrying forward:** `floored_entmax15`, the arm built to
be robust, is the *worst* on `gru_only` at h=4. It removes the dead-gradient
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

| Axis | Values | Baseline |
|---|---|---|
| `lookback` | 8, 12, 16, 24 | 12 |
| `gcn_hidden` | 16, 32, 64 | 32 |
| `gcn_layers` | 1, 2 | 2 |
| `gru_hidden` | 32, 64, 128 | 32 |
| `gru_layers` | 1, 2 | 1 |
| `dropout` | 0, 0.1, 0.2, 0.3 | 0.2 |
| `learning_rate` | 1e-4, 3e-4, 1e-3, 3e-3 | 3e-3 |
| `batch_size` | 16, 32, 64 | 64 |
| `weight_decay` | 0, 1e-5, 1e-4, 1e-3 | 1e-4 |

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

| Model | Headline MAE | Headline peak MAE | 2017 MAE | Seed sd |
|---|---|---|---|---|
| persistence | **16.42** | **26.61** | **36.08** | — |
| `gru_only` v1 (README headline best) | 16.68 | 28.06 | 42.97 | — |
| **tuned `gcn_gru` v1** | 18.23 | 29.22 | 49.54 | 0.42 |
| baseline `gcn_gru` v1 (script 16 defaults) | 18.98 | 30.36 | 54.81 | — |

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
   validation MAE moves ~0.1–0.2 across the *entire* rest of the grid. The only
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

## 9. What the evidence says to do next

Ordered by what §6–8 actually established, not by the original work plan (which
predates the full sweep and assumed the graph was neutral):

1. **Test whether *any* graph beats the identity, before building on top of
   contiguity.** §6 answered the dual-graph work plan's own stated
   precondition, and the answer is negative — contiguity costs 1.9 MAE on every
   fold. `adjacency.npz` already has `A_gaussian`
   (`exp(-(d/50km)²)`, a better prior for a country this narrow — Colombo and
   Galle are 100 km apart, share no border, and are both wet-zone coastal) —
   test it against the identity before adding a season-gated mixture.
2. **Multi-horizon training (h = 2, 3, 4).** `--horizon` exists and is unrun.
   At h=1 the forecast origin carries nearly all the signal, which is *why*
   §7's learnable lags found no gradient and why §8's fix only bites in
   epidemic conditions. Longer horizons should make climate — and the measured
   5–10 week delay — actually load-bearing.
3. **Fix lag kernels to the §7 measured delays instead of learning them
   end-to-end**, paired with (2). Decouples the delay estimate from a gradient
   that provably doesn't carry it.
4. **A count likelihood** (negative binomial or Tweedie) instead of Gaussian-
   in-log-space. §8's reweighting is a partial, hand-built approximation to
   what a proper count model would do natively.
5. **Quantile forecasts** at τ ∈ {0.1, 0.5, 0.9} for operational use — the
   pinball loss from §8 already exists; extending it to a real interval is a
   small step.

---

## 10. Known inconsistencies in the docs

Kept visible rather than silently fixed, so nothing is trusted past what was
actually checked this session:

- `docs/model_tensors.md` states "62 tests, all passing" for the stage-3
  tensor/adjacency/fold tests specifically — still true for that subset, but
  the repository-wide count has grown to **431 collected, 431 passing** under
  `torch==2.11.0+cu128` as more components were added (most recently
  `tests/test_tuning.py`, 14 tests, §8c).
- `docs/learnable_lags_results.md` states "323 tests pass repository-wide" —
  that was true when written; it's **431 collected / 431 passed** now, after
  `test_improved_losses.py` (24 tests, §8) and `test_tuning.py` (14 tests, §8c)
  were added, one dependency-driven set of failures was resolved by installing
  `scipy` and `matplotlib`, and the 3 `test_lag_encoder.py` failures stopped
  once the CUDA torch install (§4) landed on `2.11.0+cu128` instead of
  `2.14.0+cpu`.
- `docs/learnable_lags.md` header still reads "Status: not started" — it's a
  workplan document frozen at the point work began; `learnable_lags_results.md`
  is the outcome doc and is current.
- `docs/proposal_brief.md` is dated 2026-08-04 and explicitly forbids citing a
  9-fold result "not yet run" — that result exists now (§6). Treat its numbers
  as historical; its LaTeX/writing instructions are still usable.
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
- [`docs/model_tensors.md`](docs/model_tensors.md) — tensors, folds, adjacency
- [`docs/learnable_lags_results.md`](docs/learnable_lags_results.md) — Component A
- [`docs/climate_dataset_schema.md`](docs/climate_dataset_schema.md) — ERA5 extraction spec
- [`docs/reporting_calendar.md`](docs/reporting_calendar.md) — the chronological key

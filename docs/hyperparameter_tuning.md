# Hyperparameter search — full GCN+GRU

What was searched, how the winner was chosen, what it scored on the frozen test
protocol, and why "the best configuration the search found" is not the same
statement as "tuning improved the model".

Reproduce with `python scripts/24.tune_hyperparameters.py`. Tests:
`tests/test_tuning.py`. The run described here is on disk as
`results/models/tuning_report.md`, `tuning_trials.csv`,
`tuning_best_config.json`, `tuning_test_metrics.csv`.

---

## 1. The question

The baseline (`scripts/16.train_gcn_gru.py`) fixes every training knob in a
`DEFAULTS` dict. Those values were set once, by hand, on fold 8, before the full
9-fold sweep in [`baseline.md`](baseline.md) existed. Nothing since has revisited
them jointly. This is the standard reason to run a search: a hand-picked point in
a 9-dimensional space is unlikely to be the best point, and the only way to know
how much that costs is to look.

The search is designed so its answer is falsifiable in the same way every other
result in this repo is: **selection uses the validation split only, and the
winning configuration is run through the frozen test protocol exactly once,
after it is fixed.** A search that selected on the test folds would be
guaranteed to "find an improvement" and it would mean nothing.

---

## 2. The search space

Nine axes, exactly as commissioned — `SEARCH_SPACE` in `scripts/24`:

| Axis | Values | Baseline default |
|---|---|---|
| `lookback` | 8, 12, 16, 24 | 12 |
| `gcn_hidden` | 16, 32, 64 | 32 |
| `gcn_layers` | 1, 2 | 2 |
| `gru_hidden` | 32, 64, 128 | 32 |
| `gru_layers` | 1, 2 | 1 |
| `dropout` | 0.0, 0.1, 0.2, 0.3 | 0.2 |
| `learning_rate` | 1e-4, 3e-4, 1e-3, 3e-3 | 3e-3 |
| `batch_size` | 16, 32, 64 | 64 |
| `weight_decay` | 0.0, 1e-5, 1e-4, 1e-3 | 1e-4 |

4·3·2·3·2·4·4·3·4 = **27,648 points**.

Held at the baseline's value throughout, because they are the problem definition
rather than knobs being tuned: `horizon=1`, `target=residual`, `max_epochs=150`,
`patience=15`. The early-stopping budget in particular is shared so every trial
gets the same one.

### The model the two new axes need

The baseline's `GCNGRU` ties the GRU hidden size to the graph-conv hidden size
and hardcodes `num_layers=1`, so `gru_hidden` and `gru_layers` cannot be varied
against it. `scripts/24` defines `TunableGCNGRU`:

- the same spatial-then-temporal arrangement — graph convolution over the 25
  districts at each step (reusing the baseline's `GraphConv` verbatim), then one
  shared GRU over the window, then a linear head;
- a linear projection `gcn_hidden → gru_hidden` inserted **only** when those two
  differ; `nn.Identity` otherwise;
- `gru_layers` passed straight to `nn.GRU(num_layers=...)`, with inter-layer
  dropout applied by `nn.GRU` itself only when `gru_layers > 1`, matching
  torch's convention.

When the search lands on `gcn_hidden == gru_hidden` and `gru_layers == 1` the
module is the baseline architecture with no extra parameters — asserted by
`test_reduces_to_the_baseline_when_widths_match_and_gru_is_one_layer`.

Everything downstream of the model — the training loop, early stopping, Adam,
gradient clipping, seeding, fold construction, the per-fold preprocessing, the
windowing, the anchored residual target, and the masked MAE — is **imported**
from `scripts/16` through its existing `build_model` hook. A search that
reimplemented any of those would be tuning a different model than the one it
reports a test number for.

---

## 3. Method

Two backends, same objective and same space:

- **`--method random`** (default). Uniform draws from the discrete grid with a
  seeded `random.Random`, so the exact ordered trial list is reproducible from
  `--search-seed` and `--n-trials` alone, with no third-party dependency.
  Duplicate points are skipped.
- **`--method optuna`**. TPE sampler over `suggest_categorical` on the identical
  lists. Used only if `optuna` imports; otherwise the script prints a notice and
  falls back to random. A TPE run and a random run explore the same 27,648
  points — TPE just concentrates its later draws.

**Objective (minimised):** mean masked MAE on the **validation** split of each
tuning fold, raw case scale, seeds averaged. `--tune-folds` selects which folds
contribute; the default is the seven headline folds (1, 2, 3, 6, 7, 8, 9), the
COVID folds left out for the same reason the headline leaves them out. The test
split is never read during the search — `score_config` is called with
`split="val"`, and `test_score_config_reads_the_split_it_is_told_to_and_no_other`
pins that.

**After the search:** the single best configuration by validation MAE is
retrained with `--final-seeds` seeds (3 here) on all nine folds and scored on the
**test** split by `scripts/15.evaluate_naive_baselines.evaluate` — the same
function the baseline uses — so the tuned number and the baseline number are the
same measurement.

---

## 4. The run on disk

`python scripts/24.tune_hyperparameters.py --method random --n-trials 50
--search-seed 0 --trial-seeds 2 --final-seeds 3 --variant v1`

50 trials, ~73 min on an RTX 4070 Laptop GPU. Each trial is 7 folds × 2 seeds =
14 full training runs; per-trial wall time ranged 14 s to 224 s depending on how
many epochs the sampled learning rate and batch size needed before early
stopping.

50 of 27,648 points is a 0.18% sample. That is a low coverage number in the
abstract; §6 below is the reason it is nonetheless enough to answer the question
that was asked.

### Best configuration (selected on validation folds only)

| Axis | Value | Baseline | Moved? |
|---|---|---|---|
| `lookback` | 16 | 12 | yes |
| `gcn_hidden` | 32 | 32 | — |
| `gcn_layers` | 1 | 2 | yes |
| `gru_hidden` | 128 | 32 | yes |
| `gru_layers` | 2 | 1 | yes |
| `dropout` | 0.2 | 0.2 | — |
| `learning_rate` | 3e-4 | 3e-3 | yes |
| `batch_size` | 32 | 64 | yes |
| `weight_decay` | 1e-5 | 1e-4 | yes |

Validation MAE **14.754**, mean over the seven headline folds' validation
splits. (Its per-fold validation MAE is uneven — fold 2's val split scores 34.86,
fold 6's 7.22 — because each fold's validation year is a different calendar year;
the mean is what selection used.)

---

## 5. The verdict on the test protocol

Retrained with 3 seeds on all nine folds, scored on the test split.

| Model | Headline MAE | Headline peak MAE | 2017 MAE | Seed sd |
|---|---|---|---|---|
| persistence | **16.42** | **26.61** | **36.08** | — |
| `gru_only` v1 (README headline best) | 16.68 | 28.06 | 42.97 | — |
| **tuned `gcn_gru` v1** | 18.23 | 29.22 | 49.54 | 0.42 |
| baseline `gcn_gru` v1 (script 16 defaults) | 18.98 | 30.36 | 54.81 | 0.68 |
| baseline `gcn_gru` v0 | 18.60 | 29.81 | 51.87 | 0.30 |

### Per fold, test MAE (tuned, seeds averaged), against the like-for-like baseline

| Fold | Year | baseline `gcn_gru` v1 | tuned | Δ | Note |
|---|---|---|---|---|---|
| 1 | 2017 | 54.81 | 49.54 | **−5.27** | epidemic |
| 2 | 2018 | 10.10 | 10.08 | −0.02 | |
| 3 | 2019 | 17.63 | 17.89 | +0.26 | |
| 4 | 2020 | 7.49 | 7.22 | −0.27 | COVID |
| 5 | 2021 | 7.21 | 7.20 | −0.01 | COVID |
| 6 | 2022 | 12.12 | 12.25 | +0.13 | |
| 7 | 2023 | 19.16 | 18.79 | −0.37 | |
| 8 | 2024 | 10.09 | 10.09 | +0.01 | |
| 9 | 2025 | 8.98 | 8.96 | −0.02 | |

---

## 6. Did tuning improve the models? No — not by a margin this run establishes.

**1. The like-for-like gain is inside the noise.** Headline MAE moves from 18.98
to 18.23, **−0.75 (−4.0%)**, against the tuned model's seed sd of 0.42. That is
under two seed-sd. This project applies the same rule to §6, §8 and §8b of the
README — *read the seed sd before believing a gap* — and by that rule −0.75 is
not a result. It points the right way; it does not clear the bar.

**2. The entire gain is one fold.** Fold 1 (2017) contributes −5.27 of the
−0.75·7 = −5.25 total headline movement. Averaged over the other six headline
folds the change is **−0.02 MAE**, with individual folds moving between −0.37 and
+0.26 — i.e. nothing. This is the identical single-fold shape that
[`improvements.md`](improvements.md) and the README's §8b record for the loss and
activation work: whatever helps on this data helps on the 2017 epidemic and is
invisible everywhere else, because the epidemic fold is the only one where the
case distribution is wide enough for the model's weaknesses to show.

**3. It changes nothing about the standing picture.** The tuned `gcn_gru` still
loses to the no-graph control `gru_only` (16.68 vs 18.23), still loses to
persistence on the headline mean (16.42) and on peak MAE (26.61 vs 29.22), and
2017 is still where its headline gap lives (49.54 MAE, against persistence's
36.08). Tuning a backbone that [`baseline.md`](baseline.md) §3 finding 2 shows
the queen-contiguity graph *actively hurts* does not recover what the graph
costs — the search cannot tune away a structural handicap.

**4. The response surface is flat.** Mean validation MAE by the value taken on
each axis, over all 50 trials:

| Axis | Response (mean val MAE by value) | Spread |
|---|---|---|
| `lookback` | 8: 14.97  12: 14.98  16: 14.96  24: 14.94 | 0.04 |
| `gcn_hidden` | 16: 15.00  32: 14.96  64: 14.93 | 0.07 |
| `gcn_layers` | 1: 14.92  2: 15.04 | 0.12 |
| `gru_hidden` | 32: 14.92  64: 14.96  128: 14.99 | 0.06 |
| `gru_layers` | 1: 14.98  2: 14.94 | 0.05 |
| `dropout` | 0.0: 15.10  0.1: 14.93  0.2: 14.89  0.3: 14.91 | **0.21** |
| `learning_rate` | 1e-4: 14.93  3e-4: 14.93  1e-3: 14.94  3e-3: 15.07 | **0.14** |
| `batch_size` | 16: 14.93  32: 14.92  64: 15.02 | 0.10 |
| `weight_decay` | 0: 14.95  1e-5: 14.99  1e-4: 14.93  1e-3: 14.97 | 0.06 |

The only axes with a signal above ~0.1 MAE are "avoid `learning_rate=3e-3`" and
"avoid `dropout=0`" — and the baseline is already at `dropout=0.2`. Six of the
nine axes move the validation MAE by less than 0.07 across their entire range.
There is no rich optimum hiding in the 99.8% of the grid this run did not
sample, because the surface it did sample is nearly level. That is itself
consistent with [`learnable_lags_results.md`](learnable_lags_results.md): at
horizon 1 the previous period's case count carries almost all the forecasting
signal, so the model's capacity and regularisation settings have little to bite
on.

Note one apparent tension the flatness explains: the winning config sits at
`learning_rate=3e-4` and `gru_hidden=128`, while the per-axis table has
`gru_hidden=128` as the *worst* of its three values on average. Both are true —
the winner is a single lucky joint draw 0.13 MAE below the field, and the axis
means say that draw is not reproducible by fixing any one of its coordinates.
That is what a flat surface with seed noise on top looks like.

---

## 7. What would change this answer

- **A longer horizon.** The README's §9 item 2 and §8b both argue climate — and
  with it lookback, GRU depth, and the lag machinery — should become
  load-bearing at h = 2–4, where the forecast origin no longer dominates. The
  search is `--horizon`-agnostic in principle but pinned to h=1 here; rerunning
  it at h=4 is the obvious next use of the script.
- **A backbone that isn't handicapped.** Tuning `gru_only`, or a `gcn_gru` on
  `A_gaussian` instead of contiguity (README §9 item 1), would at least be
  tuning something that isn't starting from behind. The script takes `--variant`
  but always uses the real adjacency for the `gcn_gru` it scores; a `gru_only`
  or alternate-graph mode would be a small extension.
- **More seeds per trial.** At `--trial-seeds 2` the selection objective itself
  carries seed noise comparable to the between-config differences on most axes.
  `--trial-seeds 3` or more would sharpen selection, at linear cost.

None of these are expected to overturn finding 3 — the graph result in
`baseline.md` is not a tuning artefact — but they are the conditions under which
a search could plausibly find something finding 1 would call real.

---

## 8. Files

| File | Contents |
|---|---|
| `scripts/24.tune_hyperparameters.py` | the search |
| `tests/test_tuning.py` | 14 tests: search-space fidelity, validation-only selection, baseline-equivalence of `TunableGCNGRU`, random-search reproducibility, CSV round-trip |
| `results/models/tuning_report.md` | generated report — best config, verdict table, per-fold, top-15 trials, per-axis response |
| `results/models/tuning_trials.csv` | every trial: all nine axes, mean val MAE, per-fold val MAE, wall time |
| `results/models/tuning_best_config.json` | the selected configuration, its validation MAE, and the run's provenance |
| `results/models/tuning_test_metrics.csv` | per-fold per-seed test metrics for the retrained best config |

---

## Related documents

- [`baseline.md`](baseline.md) — the GCN+GRU baseline and the 9-fold sweep this is measured against
- [`improvements.md`](improvements.md) — the objective fix; same single-fold shape as finding 2 here
- [`learnable_lags_results.md`](learnable_lags_results.md) — why horizon 1 leaves so little for hyperparameters to move
- [`model_tensors.md`](model_tensors.md) — folds, windowing, the validation/test split the selection respects

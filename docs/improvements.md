# Objective improvements to the GCN+GRU baseline

What was wrong with the baseline's training objective, how it was measured, what
was changed, and what it bought.

Reproduce with `python scripts/20.train_improved.py`. Tests:
`tests/test_improved_losses.py`.

---

## 1. The diagnosis

The baseline trains on masked MSE over the log1p residual and is scored with MAE
over raw case counts. Those are different objectives, and the panel makes the
gap large rather than academic.

**Measurement 1 — the same log error is a different case error at every level.**

| Case level | Raw-case error of a constant 0.20 log-space error |
| --- | --- |
| 5 | 1.3 |
| 25 | 5.8 |
| 100 | 22.4 |
| 500 | 110.9 |
| 1500 | 332.3 |

Log-MSE spends equal effort on the first row and the last. MAE counts the last
roughly 250x more.

**Measurement 2 — the over-weighted cells are also the noisiest.**

| Case level at origin | n | sd of the log1p week-over-week change |
| --- | --- | --- |
| 0–10 | 12,365 | 0.789 |
| 10–50 | 8,821 | 0.630 |
| 50–200 | 3,349 | 0.513 |
| 200+ | 738 | 0.375 |

Small districts are the *least* predictable in log space. So log-MSE puts most
of the model's capacity where the signal-to-noise ratio is worst.

**Measurement 3 — the metric is concentrated.** Over the headline test cells,
14% of cells hold 61% of the case volume. Across the headline folds, ten
districts account for 70% of persistence's total absolute error, with Colombo
and Gampaha alone at 25%.

**Consequence, measured on fold 1.** The baseline underpredicts high-count cells
by a ratio of 0.624 — mean actual 490.5 against mean predicted 306.0. This is the
failure `docs/baseline.md` records as the central open problem ("tracks the
shape and misses the level"), and the objective mismatch is a sufficient
explanation for it.

---

## 2. What changed

Five arms. Each changes the training objective and **nothing else**: the model,
graph, target parameterisation, folds, masks, preprocessing, optimiser, batching,
gradient clipping, seeding and early stopping are all imported from
`scripts/16.train_gcn_gru.py` rather than copied. A difference in the numbers
cannot come from a difference in the loop.

| Arm | Loss | Level-weighted |
| --- | --- | --- |
| `baseline` | masked MSE | no |
| `huber` | masked Huber, delta 0.5 | no |
| `level_weighted` | masked MSE | yes |
| `huber_weighted` | masked Huber | yes |
| `quantile` | pinball at tau = 0.5 | yes |

### Huber

The residual distribution is heavy-tailed — p99 of the raw week-over-week change
is 105 cases against a median of 4. Squared error lets those tails set the
gradient direction, and MAE gives no reward for chasing them. Huber is quadratic
below delta and linear above, so a single epidemic week cannot dominate a batch.
delta = 0.5 in log1p units, which is roughly a 65% change in the case count.

### Level weighting

Each cell is weighted by `log1p(cases)` at the **forecast origin**, normalised to
mean 1 over the batch's observed cells.

Three properties, each asserted in the tests:

- **It leaks nothing.** The anchor is the case count at the last period the model
  is allowed to see, and is already one of its inputs. This is a reweighting of
  the training distribution, not new information.
- **`log1p`, not the raw count.** Weighting by raw counts would hand a 1500-case
  district 300x the gradient of a 5-case one, replacing one imbalance with a
  worse one. `log1p` keeps that ratio near 4x.
- **Normalised to mean 1.** The loss stays on the baseline's scale, so one
  patience and one learning rate remain valid across arms. Otherwise the loss
  comparison would be confounded with a tuning comparison.

### Quantile

MAE is minimised by the conditional *median*; MSE estimates the conditional
*mean*. On a right-skewed count distribution these differ and the mean sits
above the median. Pinball at tau = 0.5 targets the median directly.

### Seed-mean ensemble

The per-seed test predictions are averaged **before** the metric rather than
after. This is free — the seeds are already trained — and it is not the same
operation as averaging the per-seed scores. It is reported separately from the
loss comparison because it is a variance reduction, not a better model.

---

## 3. The control

The `baseline` arm is script 16's own configuration run through script 20's
harness. On fold 8 it reproduces script 16's MAE to the second decimal
(9.97 both). `tests/test_improved_losses.py` additionally asserts the numerical
identity: `masked_weighted_loss` with unit weights and `kind="mse"` equals
`scripts/16.masked_mse` exactly.

Without that control, none of the comparisons below mean anything.

---

## 4. Results

### Context from the full baseline sweep

The 9-fold, 3-seed baseline sweep (run 2026-09-07, `scripts/16`) puts these
numbers in their proper setting, and it is not the setting the older docs
assumed:

| Model | Features | Headline MAE | 2017 MAE | Seed sd |
| --- | --- | --- | --- | --- |
| persistence | — | **16.42** | **36.08** | — |
| `gru_only` | v0 | 16.67 | 41.74 | 0.28 |
| `gru_only` | v1 | 17.12 | 46.01 | 0.70 |
| `gcn_gru` | v1 | 19.01 | 55.01 | 0.62 |
| `gcn_gru` | v0 | 19.08 | 55.23 | 0.84 |

**No trained model beats persistence on the headline mean.** The models win on 5
of the 7 headline folds and lose the mean entirely on 2017. That is the whole
argument for attacking the objective: the headline gap *is* the epidemic fold.

### Full sweep — 9 folds, 3 seeds, `v1`, `gcn_gru`

**Run 2026-09-07, 36.3 min CPU.** These are the numbers to quote.

Single-seed models, mean over 3 seeds:

| Arm | Headline MAE | vs persistence | Peak MAE | 2017 MAE | 2017 peak MAE | Seed sd |
| --- | --- | --- | --- | --- | --- | --- |
| persistence | **16.42** | — | **26.61** | **36.08** | **39.42** | — |
| `level_weighted` | 16.69 | −1.6% | 28.30 | 38.52 | 42.88 | 0.66 |
| `quantile` | 17.32 | −5.5% | 28.14 | 44.26 | 49.72 | 0.29 |
| `huber_weighted` | 17.36 | −5.7% | 28.46 | 44.24 | 49.75 | 0.44 |
| `huber` | 17.41 | −6.0% | 28.21 | 44.19 | 49.65 | 0.34 |
| `baseline` | 19.01 | −15.7% | 30.47 | 55.01 | 62.40 | 0.62 |

Seed-mean ensembles:

| Arm | Headline MAE | vs persistence | Peak MAE | 2017 MAE | 2017 peak MAE |
| --- | --- | --- | --- | --- | --- |
| **`level_weighted`** | **16.39** | **+0.2%** | 27.86 | **37.11** | 41.25 |
| `huber_weighted` | 17.26 | −5.1% | 28.29 | 43.93 | 49.41 |
| `quantile` | 17.24 | −5.0% | 28.01 | 44.00 | 49.45 |
| `huber` | 17.35 | −5.6% | 28.10 | 44.08 | 49.54 |
| `baseline` | 18.92 | −15.2% | 30.29 | 54.82 | 62.21 |

Per fold, MAE, single-seed models:

| Fold | Year | `baseline` | `huber` | `quantile` | `huber_weighted` | `level_weighted` | persistence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2017 | 55.01 | 44.19 | 44.26 | 44.24 | **38.52** | 36.08 |
| 2 | 2018 | 10.08 | 10.17 | **9.87** | 9.90 | 9.97 | 10.45 |
| 3 | 2019 | 17.57 | 17.51 | **17.30** | 17.60 | 17.85 | 17.30 |
| 4 | 2020 | 7.54 | 7.40 | **6.78** | 6.86 | 7.32 | 7.41 |
| 5 | 2021 | 7.25 | **7.21** | 7.23 | 7.24 | 7.27 | 7.23 |
| 6 | 2022 | **12.17** | 12.23 | 12.38 | 12.34 | 12.20 | 12.57 |
| 7 | 2023 | 19.07 | 18.76 | 18.42 | **18.38** | 19.13 | 18.83 |
| 8 | 2024 | 10.09 | 10.08 | 10.07 | 10.11 | **10.04** | 10.78 |
| 9 | 2025 | 9.06 | **8.92** | 8.97 | **8.92** | 9.12 | 8.95 |

**What this establishes.**

- **`level_weighted` cuts the headline MAE from 19.01 to 16.69, a 12.2%
  improvement over the baseline**, against a seed sd of 0.62 and 0.66. That gap
  is roughly 3.5 pooled sd and is comfortably established.
- **On fold 1 it cuts MAE from 55.01 to 38.52 (−30.0%)** and peak MAE from 62.40
  to 42.88 (−31.3%). This is the fold that was carrying the entire headline gap.
- **The seed-mean ensemble reaches 16.39, which finally edges persistence at
  16.42.** Treat this as a tie, not a win: the margin is 0.03 MAE against a
  single-model seed sd of 0.66. What can be said is that the reweighted model is
  the first configuration in this repository that is not clearly *worse* than
  persistence on the headline.
- **Peak MAE still loses to persistence** (27.86 vs 26.61 at best). The
  workplan's success criterion for this item — fold-1 peak MAE below persistence
  — is **not met**: 41.25 against 39.42. It went from missing by 23 to missing by
  1.8.
- **Level weighting is the active ingredient, not Huber.** `level_weighted`
  (plain MSE, weighted) beats every Huber arm by 0.6–0.7 MAE. Huber alone buys
  1.6 MAE over the baseline; adding Huber *on top of* weighting costs 0.67. The
  bounded gradient and the reweighting are competing for the same correction,
  and the reweighting does it better. `huber_weighted` looked best at one seed on
  fold 1 and is not the best configuration at 9 folds and 3 seeds — a clean
  illustration of why the single-seed numbers below are kept only as history.

### Fold 1 (2017 epidemic) — the single-seed exploratory run

Kept for the mechanism it exposes, not for its numbers. Single seed, `v1`,
`gcn_gru`; the 3-seed baseline mean for this configuration is **55.01**, so the
51.19 below is a favourable draw for the baseline.

| Arm | MAE | Peak MAE | High-count pred/actual ratio |
| --- | --- | --- | --- |
| `baseline` | 51.19 | 58.02 | 0.624 |
| `huber` | 45.45 | 51.23 | 0.692 |
| `quantile` | 44.41 | 49.89 | 0.695 |
| `level_weighted` | 43.88 | 49.38 | 0.720 |
| **`huber_weighted`** | **41.82** | **46.94** | **0.740** |

`huber_weighted` cuts fold-1 MAE by **18.3%** and peak MAE by **19.1%**.

The ratio column is the mechanism, not a side effect: on cells above 200 cases
the mean prediction rises from 306.0 to 363.1 against an actual of 490.5, and
their MAE falls from 194.3 to 150.8. The arms are ordered the same way by MAE and
by underprediction, which is what a real fix looks like.

### Fold 8 (2024, quiet year)

| Arm | MAE |
| --- | --- |
| `baseline` | 9.97 |
| `huber` | 10.01 |
| `quantile` | 10.04 |
| `level_weighted` | 10.05 |
| `huber_weighted` | 10.07 |

All five arms sit inside 0.1 MAE. **This is the expected result, not a
disappointment.** The objective mismatch is a function of how wide the case
distribution is; in a quiet year persistence is already near-optimal and there is
nothing for a reweighting to recover. It is also a useful negative control: the
weighting does not damage the ordinary case.

### Per district, fold 1

22 of 25 districts improve under `huber_weighted`; mean improvement 12.9%.

| District | Baseline MAE | Improved MAE | Mean cases | Change |
| --- | --- | --- | --- | --- |
| Colombo | 220.97 | 169.93 | 630.8 | +23.1% |
| Gampaha | 209.24 | 163.01 | 554.2 | +22.1% |
| Kandy | 97.21 | 76.12 | 273.2 | +21.7% |
| Kurunegala | 86.96 | 69.50 | 210.5 | +20.1% |
| Ratnapura | 71.65 | 56.20 | 144.4 | +21.6% |
| Ampara | 54.60 | 41.55 | 112.0 | +23.9% |
| Jaffna | 29.02 | 26.54 | 112.4 | +8.6% |

The three districts that get worse are Kilinochchi, Mannar and Mullaitivu, at
−1.8%, −2.6% and −5.0%. All three average under 10 cases per period, and all
three lose about 0.2 cases of MAE. That is the trade the weighting is *designed*
to make: 0.2 cases in a district reporting 8, against 51 cases in Colombo.
Stating it as a regression without the magnitudes would misrepresent it.

**Jaffna improves by 8.6%**, which is worth recording against the open question
in `docs/model_tensors.md` about its degree-1 isolation under contiguity.

---

## 5. What may not be claimed

- **The headline gain is entirely one fold.** `level_weighted` beats the baseline
  by 2.316 MAE on the headline mean. Fold 1 alone contributes 2.356 of that —
  **101.7%**. Averaged over the other six headline folds the change is
  **−0.046 MAE**, a marginal loss. Per fold the reduction is +16.49 on fold 1 and
  then +0.11, −0.28, −0.03, −0.06, +0.05, −0.06. This is a fix for epidemic
  conditions and nothing else; describing it as a general improvement would be
  false.
- **The ensemble does not beat persistence.** 16.39 against 16.42 is a 0.03 MAE
  margin against a 0.66 seed sd. It is a tie. No configuration in this repository
  beats persistence on the headline mean.
- **Peak MAE still loses to persistence on every arm**, including on fold 1. The
  workplan criterion for this item is not met.
- **Fold-level differences below ~0.7 MAE are not established** by 3 seeds at
  these spreads. That covers every arm-to-arm comparison on folds 2–9.
- **`gcn_gru` only.** `gru_only` was not run for these arms, and in the baseline
  `gru_only` beats `gcn_gru` on fold 8.
- **`v1` only.** The arms were not crossed with `v0`.
- **Horizon 1 only.** The objective mismatch should widen at longer horizons,
  because the anchor carries less of the answer, but that is untested.
- **delta and the weighting exponent were not tuned.** delta = 0.5 and `log1p`
  were chosen by argument, not by search. A sweep would likely improve them, and
  would then need its own validation-only selection to stay honest.
- **The gain is concentrated in epidemic conditions.** On quiet folds these arms
  are neutral. The headline mean will therefore move much less than fold 1 does,
  and quoting the fold-1 improvement as the headline would repeat exactly the
  error `docs/baseline.md` flags in the stale baseline report.

---

## 6. Files

| Path | Contents |
| --- | --- |
| `scripts/20.train_improved.py` | the five arms and the runner |
| `tests/test_improved_losses.py` | 24 tests, including the baseline identity |
| `results/models/improvement_report.md` | generated report |
| `results/models/improvement_metrics.csv` | per arm, fold and seed |
| `results/models/improvement_predictions.csv` | per district-period predictions |

## Related documents

- [`docs/baseline.md`](baseline.md) — the baseline and its open problems
- [`docs/model_tensors.md`](model_tensors.md) — tensors, folds, adjacency
- [`docs/learnable_lags_results.md`](learnable_lags_results.md) — component A

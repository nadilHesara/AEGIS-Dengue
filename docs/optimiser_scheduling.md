# Optimiser and learning-rate schedule

Whether AdamW beats Adam on this model, whether a plateau schedule helps, and
why the schedule mostly cannot help given how the baseline stops training.

Reproduce with `python scripts/25.train_optimisers.py`. Tests:
`tests/test_optimisers.py`. The run described here is on disk as
`results/models/optimiser_report.md`, `optimiser_metrics.csv`,
`optimiser_lr_traces.csv`.

---

## 1. The two questions

**AdamW.** Adam implements weight decay by adding `wd * w` to the gradient. That
sum then passes through Adam's own per-parameter normalisation, so the decay a
parameter actually receives is `wd` divided by a running estimate of its gradient
magnitude — parameters with small gradients get decayed far harder than
parameters with large ones. AdamW applies the decay directly to the weight,
outside the normalisation. At the baseline's `weight_decay=1e-4` this is a
genuine difference in what the regulariser does, not a reparameterisation, and
nothing in this repository had tested which one suits the model.

**Scheduling.** A fixed learning rate is either too small for the start of
training or too large for the end. `ReduceLROnPlateau` removes the choice: train
at the base rate until validation stops improving, then halve. There was a
documented reason to expect this to matter here —
[`hyperparameter_tuning.md`](hyperparameter_tuning.md) found `learning_rate` is
one of only two axes with any measurable signal in the whole 9-axis grid, and
that the baseline's `3e-3` is the *worst* of the four values tried. If the
baseline trains too hot, a schedule that cools it should show a benefit.

---

## 2. Design

Three arms, changing only the optimiser and the schedule:

| Arm | Optimiser | Schedule |
|---|---|---|
| `adam` | Adam | fixed rate — the control |
| `adamw` | AdamW | fixed rate |
| `adamw_scheduled` | AdamW | `ReduceLROnPlateau` on validation loss |

### The loop is not copied

`scripts/16.train_gcn_gru.py` previously hardcoded `torch.optim.Adam` inside
`train_one`, with no hook for it — unlike `build_model`, which it already had.
Rather than copy the loop (which script 20's own docstring argues against:
"an experiment that reimplements early stopping or the optimiser is no longer
comparable to this baseline"), script 16 gained two backward-compatible hooks:

```python
train_one(..., make_optimiser=None, make_scheduler=None)
```

Both default to the baseline's behaviour — Adam, no scheduler — so a caller that
passes neither gets the committed configuration bit-for-bit. Script 25 passes
its own. `test_adam_through_the_hook_matches_script_16_exactly` and
`test_the_control_reproduces_the_baseline_on_a_real_fold` assert the identity on
synthetic and real fold data respectively, and the `adam` arm reproduces script
16's committed fold-1 MAE of **54.81** to the second decimal in the run below.

### The two patiences

Early stopping and `ReduceLROnPlateau` both watch the validation loss, so their
patiences interact. Early stopping keeps the baseline's **15** — changing it
would break comparability with every committed result. The scheduler gets **5**,
so it can fire roughly three times before early stopping can trigger.

The scheduler is stepped **after** early stopping has already seen the epoch.
That ordering is deliberate and is what makes the arm an isolation of the
schedule: adding a scheduler cannot change which epoch is selected as best, only
what the optimiser does next.

### What "same folds and seeds" means

Every arm trains on the same nine folds with the same seed range. Within a fold,
each arm sees identical data, identical batch order and an identically
initialised model, because `train_one` reseeds from `seed` before building
anything. The arms differ in the optimiser and nothing else.

---

## 3. Results — 9 folds, 3 seeds, ~1.9 min GPU

| Arm | Headline MAE | vs `adam` | Peak MAE | 2017 MAE | Mean best epoch | Seed sd |
|---|---|---|---|---|---|---|
| `adamw` | **18.67** | −0.31 | 29.89 | 52.81 | 20.0 | 0.97 |
| `adamw_scheduled` | 18.90 | −0.08 | 30.26 | 53.98 | 19.1 | 0.73 |
| `adam` (control) | 18.98 | — | 30.36 | 54.81 | 20.6 | 0.68 |

Seed-mean ensembles: `adamw` 18.53, `adamw_scheduled` 18.82, `adam` 18.88.

For context on the same folds: persistence 16.42, `gru_only` v1 16.68.

### Per fold, headline folds only, seeds averaged

| Fold | Year | `adam` | `adamw` | `adamw_scheduled` | Δ AdamW |
|---|---|---|---|---|---|
| 1 | 2017 | 54.81 | 52.81 | 53.98 | **−2.00** |
| 2 | 2018 | 10.10 | 10.10 | 10.10 | −0.00 |
| 3 | 2019 | 17.63 | 17.35 | 17.53 | −0.28 |
| 6 | 2022 | 12.12 | 12.14 | 12.14 | +0.02 |
| 7 | 2023 | 19.16 | 19.26 | 19.54 | +0.09 |
| 8 | 2024 | 10.09 | 10.04 | 10.04 | −0.04 |
| 9 | 2025 | 8.98 | 8.99 | 8.99 | +0.01 |

---

## 4. Verdict — neither change is established by this run

**AdamW moves the headline by −0.31 MAE (−1.7%), and that is not a result.**

| Test | Value | Reading |
|---|---|---|
| Gap vs pooled seed sd | 0.31 vs 0.82 = **0.38 sd** | Far short of the two-sd bar |
| Paired t over the 7 headline folds | **t = −1.11, p = 0.31** | Not significant |
| Folds improved | **4 of 7** | Barely better than a coin |
| Fold 1's share of the gain | **91%** | Single-fold, as always |
| Δ over the other six folds | **−0.03 MAE** | Nothing |

`adamw_scheduled` is weaker still: −0.08 MAE, t = −0.58, p = 0.58, and its
fold-1 share is 147% — meaning the non-epidemic folds moved *the wrong way*
(+0.045) and fold 1 alone carried it.

This is the fourth time in this repository that a change has produced a headline
movement which turns out to be the 2017 epidemic fold and nothing else (README
§8, §8b, §8c). The pattern is now well enough established to be a prior: on this
data at horizon 1, a training-procedure change that moves the headline is moving
fold 1.

---

## 5. The mechanism finding — why the schedule mostly cannot help

This is the part worth carrying forward, and it is a structural interaction
rather than a tuning accident.

`optimiser_lr_traces.csv` records the learning rate at every epoch of all 27
scheduled runs. Comparing the epoch of the first LR reduction against the epoch
whose model was actually kept:

**In 23 of 27 runs the learning rate did not drop until *after* the best epoch
had already been saved.**

| Fold | Runs where the first LR drop preceded the best epoch |
|---|---|
| 1 (2017) | 1 of 3 |
| 3 (2019) | 1 of 3 |
| 7 (2023) | 2 of 3 |
| 2, 4, 5, 6, 8, 9 | **0 of 3** each |

The reason is mechanical. `ReduceLROnPlateau` fires after 5 epochs without
improvement; early stopping keeps the best model from before that plateau began.
So by construction the first reduction lands *inside the early-stopping wait*, at
a point where the kept model is already fixed. The reduced rate then trains for
at most another 10 epochs before early stopping ends the run — and if it does not
find a new best within those, none of that training is kept.

On folds that converge fast this is absolute. Fold 2 reaches its best epoch at 2
or 3 and the first LR drop lands at 9 or 10; the schedule is pure overhead there,
which is exactly what the identical MAEs (10.095 vs 10.097) show.

**The implication for anyone extending this:** a plateau scheduler and
early-stopping-on-the-same-metric are close to mutually exclusive as configured.
To give a schedule a real chance you would have to either

- give early stopping a much longer patience than the scheduler (so a reduced
  rate has room to find a new best), which changes the baseline protocol and
  breaks comparability with every committed number; or
- use a schedule that does not wait for a plateau — cosine annealing or a step
  decay on a fixed epoch count, which reduces the rate *during* productive
  training rather than after it.

The second is the cleaner next test, and this script's `make_scheduler` hook
takes it without further changes to script 16.

---

## 6. Secondary observations

- **AdamW does not train faster.** Mean best epoch 20.0 against Adam's 20.6 —
  the same, within noise. Whatever AdamW does here, it is not accelerating
  convergence.
- **AdamW is noisier.** Seed sd 0.97 against Adam's 0.68. Its −0.31 mean
  advantage comes with more run-to-run variance than the control, which is part
  of why the gap fails the sd test.
- **The scheduled arm is *more* stable** (0.73) than plain AdamW, consistent with
  a lower terminal learning rate — but it buys that stability at a worse mean.
- **Fold 7 (2023) is where scheduling actively hurts**: 19.54 against the
  control's 19.16. It is also the fold with the most LR reductions (3.3 per run)
  and the most runs where a drop preceded the best epoch — i.e. the fold where
  the schedule had the most influence, and it was negative.

---

## 7. Files

| File | Contents |
|---|---|
| `scripts/25.train_optimisers.py` | the three arms |
| `scripts/16.train_gcn_gru.py` | gained `make_optimiser` / `make_scheduler` hooks and a `build_optimiser` default; behaviour unchanged when neither is passed |
| `tests/test_optimisers.py` | 16 tests: control bit-identity, Adam/AdamW divergence at `wd>0` and equivalence at `wd=0`, scheduler type/floor/firing, the early-stopping ordering guarantee, LR-trace recording |
| `results/models/optimiser_report.md` | generated report |
| `results/models/optimiser_metrics.csv` | per-arm per-fold per-seed metrics, plus ensemble rows |
| `results/models/optimiser_lr_traces.csv` | per-epoch learning rate for every scheduled run — the evidence for §5 |

---

## Related documents

- [`baseline.md`](baseline.md) — the model, folds and the 18.98 the control reproduces
- [`hyperparameter_tuning.md`](hyperparameter_tuning.md) — where the "3e-3 is too hot" expectation came from, and the same flat-surface conclusion
- [`improvements.md`](improvements.md) — the objective work; same single-fold shape

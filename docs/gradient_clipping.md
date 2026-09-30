# Gradient clipping on the current model

Whether clipping the global gradient norm before every `optimiser.step()` makes
the multi-horizon Negative Binomial model (README §8i) train more stably or
forecast better, and whether it has exploding gradients to prevent in the first
place.

Reproduce with `python scripts/training/37.train_grad_clipping.py` (28.7 min on
a 12-thread CPU; `--quick` for a ~3 min smoke test). Tests:
`tests/test_grad_clipping.py`. The run described here is on disk as
`results/models/grad_clipping_{report.md,metrics,runs,epochs,steps}.csv` and
`results/figures/grad_clipping_norms.png`.

---

## 1. The question

Script 32 has always called `nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
before every optimiser step. Clipping was never tested: no run existed without
it, and nothing logged how large the gradients actually were. Three things
could be true, with different consequences:

1. Gradients explode now and then, and the clip at 1.0 is what keeps training
   stable. Removing it should hurt.
2. Gradients are routinely above 1.0, and the clip is quietly acting as a
   per-step learning-rate cut. Changing the threshold should change results.
3. Gradients never come near 1.0, and the clip is inert.

Logging the pre-clip norm tells these apart before any accuracy is compared.

---

## 2. Design

### The hook, not a copy

`train_shared_negbin` in script 32 now calls:

```python
epoch_norms.append(clip_gradients(model, config["clip_max_norm"]))
```

`clip_gradients` returns the **pre-clip** global L2 norm. For a number it calls
`clip_grad_norm_`. For `None` it calls `get_total_norm` and leaves the gradients
untouched, so every arm logs the same quantity. `DEFAULTS["clip_max_norm"] = 1.0`
keeps the committed behaviour, and `--clip-max-norm none` disables clipping.
The function returns `info["grad_norms"]` (every step) and `info["history"]`
(per epoch: train and validation loss, norm mean and max, clipped fraction,
non-finite steps).

### Arms

| Arm | `max_norm` | Why |
|---|---|---|
| `no_clip` | None | The control |
| `clip_0.25` | 0.25 | Near the median step norm, so it binds on about half of all steps |
| `clip_1` | 1.0 | Script 32's committed value |
| `clip_5` | 5.0 | A looser clip |

`clip_0.25` was added after the smoke test showed that 1.0 and 5.0 almost never
fire (§3). Without it, the ablation could only answer "no difference".

Current model configuration: tensor `v3`, identity backbone, parallel heads,
h = 1–4, early stopping patience 15. The run covers the 7 headline folds × 3
seeds. Within a (fold, seed), every arm gets the same data, batch order and
initialisation.

### Keeping it short

- **Provably identical runs are copied, not retrained.** `clip_grad_norm_`
  scales gradients by `min(1, max_norm / (norm + 1e-6))`. If every step of the
  control had `norm + 1e-6 <= max_norm`, that factor is exactly 1 at every step,
  so the clipped run is the same computation. Script 37 trains each (fold, seed)
  control first, and when it finishes, either queues the clipped arms or copies
  the control's result for them (`inferred=True`).
  `test_a_threshold_above_every_norm_is_bit_identical_to_no_clip` checks that
  premise on the real training loop. 33 of 84 runs were copied, which saved
  about 18 minutes. `--no-skip` turns this off.
- **Parallel workers with few threads each.** At ~8k parameters, per-batch
  overhead dominates. Measured on a 12-thread CPU, one process at 4 threads was
  1.7× faster than at PyTorch's default of 10, and 6 workers × 2 threads gave
  the best throughput. Those are the defaults (`cpu_count/2` workers × 2).

---

## 3. Are there exploding gradients? No.

Pre-clip norm of every step of the 21 control runs:

| Fold | p50 | p90 | p99 | max | share > 0.25 | share > 1 |
|---|---|---|---|---|---|---|
| 1 (2017) | 0.21 | 0.37 | 0.59 | 1.03 | 34% | 0.2% |
| 2 | 0.18 | 0.30 | 0.42 | 0.54 | 20% | 0% |
| 3 | 0.27 | 0.42 | 0.62 | 1.08 | 59% | 0.1% |
| 6 | 0.26 | 0.43 | 0.72 | 1.72 | 55% | 0.3% |
| 7 | 0.27 | 0.51 | 1.15 | 2.87 | 60% | 1.8% |
| 8 | 0.26 | 0.40 | 0.62 | 1.04 | 54% | 0.1% |
| 9 | 0.28 | 0.45 | 0.67 | 1.08 | 62% | 0% |

- **No step in any run exceeded 10× its run's median.** The largest step is on
  average 4.1× the median. The single largest norm in the sweep is 2.87
  (fold 7).
- **No non-finite loss or gradient** in any of the 84 runs.
- **Norms rise gently and do not spike** — from about 0.12 in epoch 1 to about
  0.35 by epoch 50 (figure, left). This is the zero-initialised Δμ head moving
  away from persistence, not instability.

That settles the three possibilities in §1: this is case 3. The clip at 1.0
fires on 0.3% of steps, and at 5.0 it never fires.

Why the gradients are this tame: the anchored head starts exactly at
persistence (Δ = 0), so the initial loss is already reasonable. The NLL is
averaged over up to 64 × 25 × 4 observed cells. The head clamps Δ to [−10, 10]
before the `exp`, which bounds how far μ can move in one step.

---

## 4. Results

Headline folds, single-seed mean MAE, with Δ taken against `no_clip` on
per-fold seed-mean MAE:

| h | `no_clip` | `clip_0.25` | Δ | p | `clip_1` | Δ | p | `clip_5` |
|---|---|---|---|---|---|---|---|---|
| 1 | 16.00 | 16.04 | +0.04 | 0.75 | 16.00 | +0.01 | 0.73 | 16.00 |
| 2 | 19.35 | 19.31 | −0.04 | 0.74 | 19.36 | +0.01 | 0.53 | 19.35 |
| 3 | 22.84 | 22.77 | −0.07 | 0.62 | 22.86 | +0.02 | 0.48 | 22.84 |
| 4 | 25.78 | 25.74 | −0.05 | 0.81 | 25.81 | +0.03 | 0.38 | 25.78 |

Seed-mean ensembles tell the same story: h=4 is 25.41 / 25.38 / 25.44 / 25.41.
Every arm beats same-horizon persistence at every horizon (16.42 / 20.23 /
24.81 / 28.78), as §8i reports.

**`clip_5` is identical to `no_clip`**, with all 21 runs copied under the proven
skip rule.

**`clip_1` versus `no_clip` is noise.** The clip fired in 9 of 21 runs, on
fewer than 2% of steps even in the worst fold. The largest per-fold movement
is 0.17 MAE, and no horizon comes close to p = 0.05. Removing the committed
clip neither helps nor hurts.

**`clip_0.25` binds on 51% of steps and still does not move the headline**
(|Δ| ≤ 0.07, p ≥ 0.62). The split by fold runs the opposite way to the usual
pattern in this repository:

| h | Δ fold 1 (2017) | Δ mean of the other six folds |
|---|---|---|
| 1 | +0.76 | −0.08 |
| 2 | +0.48 | −0.12 |
| 3 | +0.46 | −0.16 |
| 4 | +0.67 | −0.17 |

Tight clipping costs a little on the epidemic fold, where large corrective
gradients are legitimate, and gains a little elsewhere. Neither half is large
enough to claim.

### Training stability

| | `no_clip` | `clip_0.25` | `clip_1` | `clip_5` |
|---|---|---|---|---|
| Steps clipped | 0% | 51% | 0.3% | 0% |
| Epochs with a validation-loss rise | 43.4% | 42.7% | 42.7% | 43.4% |
| Median \|Δ val loss\| per epoch | 0.016 | 0.016 | 0.016 | 0.016 |
| Mean best epoch | 37.1 | 34.8 | 36.4 | 37.1 |
| **Mean per-fold seed sd, h=1 / 2 / 3 / 4** | 0.36 / 0.59 / 0.85 / 1.04 | **0.27 / 0.42 / 0.60 / 0.91** | 0.37 / 0.62 / 0.87 / 1.03 | = `no_clip` |

Epoch-to-epoch smoothness does not change under any threshold. The one
consistent difference is **seed variance under `clip_0.25`: 12–30% lower at
every horizon.** This is the same kind of result as §8g's LayerNorm: better
conditioning that shows up as reproducibility rather than accuracy. It rests on
3 seeds per fold, so treat it as a lead, not a finding.

---

## 5. Verdict

- **There are no exploding gradients to prevent.** The largest norm in 84 runs
  is 2.87, spikes never exceed 10× the median, and nothing goes non-finite.
- **The committed clip at 1.0 is inert insurance.** It fires on 0.3% of steps,
  and removing it changes no horizon by more than 0.03 MAE. Keep it as the
  default: it costs nothing and would matter if a future change (a higher
  learning rate, an unanchored head, a new loss) did produce spikes.
- **5.0 never fires** on this model.
- **A binding clip (0.25) does not improve accuracy.** It lowers seed variance
  by 12–30%, and it is slightly worse on 2017 and slightly better elsewhere,
  none of it significant. It is not adopted as the default. If seed variance
  becomes the bottleneck, a 5–10 seed run of `no_clip` versus `clip_0.25`
  (`--max-norms none 0.25 --seeds 10`, ~75 min) would settle it.

This agrees with the standing explanation in README §6–§8h: the optimiser is
not what limits this model. The ceiling is set by the information in the
inputs.

# Simplex activations for the lag encoder

**Status: done, 2026-09-10. A mechanism result, not an accuracy result.**
Swept at horizon 1 and horizon 4; h = 4 shows a directional trend (§7) that
does not reach significance.

What was changed, what it did, and what may not be claimed from it.

---

## 1. What this is about, and what it is not

The GCN+GRU is a **regression** forecaster. It predicts a weekly case count and
is scored with MAE. It has no classification head and **no output softmax**.

The one softmax in the model sits in `src/models/lag_encoder.py`, in
`mixture_weights()`. The encoder builds each district's climate delay curve as
a mixture of six Gaussian bumps, and the softmax is what forces those six
mixture weights to be non-negative and sum to one — a *structural simplex
constraint*, not a classification activation.

So "try a better softmax" here means: **is the simplex map the right one for a
delay mixture?** That is a real question with a real answer, but it is not, and
cannot be, a question about the model's output layer.

---

## 2. Why softmax was suspected

Three reasons, each with a symptom already recorded in this repository before
this work started.

**Saturation.** Softmax's Jacobian is `diag(p) − ppᵀ`. As one bump comes to
dominate, every gradient reaching the basis logits goes to zero.
`scripts/16.train_gcn_gru.py`'s `parameter_groups` says so outright — *"The
delay kernels sit behind a softmax and a Gaussian shape, so the gradient
reaching them is far smaller than the gradient reaching the GRU"* — and works
around it by giving the encoder a 10× learning rate rather than by changing the
activation.

Measured, in `tests/test_simplex_activations.py`: scaling the logits by 20 drops
the gradient norm from `2.2e-01` to `4.1e-09`; by 50, to `3.9e-22`. That is not
a slow decay, it is death.

**Unconstrained scale.** The logits are `einsum` of two `0.1·randn` tensors.
Nothing fixes their scale, so the effective temperature is an artefact of
initialisation and drifts during training.

**Full support.** Softmax cannot return zero, so every learned kernel is a blend
of all six bumps. A biological delay is one peak. Blending six of them widens
the curve and drags its centre of mass toward the middle of the reach — and the
centre of mass is exactly what `docs/learnable_lags_results.md` compares against
the measured delay when it reports **r = −0.16**.

---

## 3. What was built

`src/models/simplex_activations.py`. Every scheme maps `R⁶` onto the same
simplex, so each is a drop-in replacement and nothing else in the encoder
changes.

| Arm | What it does | Which problem it targets |
|---|---|---|
| `softmax` | the control — unchanged | — |
| `temp_softmax` | `softmax(z/τ)`, τ learned behind a softplus | saturation, scale |
| `sparsemax` | Euclidean projection onto the simplex | density |
| `entmax15` | α-entmax at α = 1.5 | density, less abruptly |
| `floored_entmax15` | `entmax15` with 2% uniform mass mixed back | density, without the dead end in §4 |
| `gumbel_softmax` | annealed stochastic sampling | boundary drift |

`sparsemax` and `entmax15` are implemented against torch autograd rather than
taken from the `entmax` package — two functions did not justify a dependency,
and the numerics are worth having explicit. Both are verified against a
definition rather than against their own output: `sparsemax` against an
independently written projection (agreement to `1e-12`), `entmax15` against the
KKT stationarity condition its solution must satisfy.

**The encoder's default is still `softmax`,** and a test asserts
`mixture_weights()` is bit-identical to `torch.softmax` when no activation is
named. Every result recorded before this work remains reproducible.

---

## 4. The finding that matters: sparse maps have an absorbing state

This is the part worth carrying forward, and it was found by measuring rather
than by reasoning.

Sparse simplex maps look strictly better on paper — exact zeros, and 5.7× the
initial gradient of softmax in the real encoder. They have a failure mode that
only shows up during training.

**When a mixture collapses onto a single bump, the output is one-hot and
locally constant, so the gradient is exactly zero — and nothing can ever move
it again.** It is not a bug: at support size one that *is* the true derivative.
It is an absorbing state.

`scripts/23.activation_gradient_probe.py`, fold 8, 40 epochs:

| Activation | Initial grad | Mean support | **Dead pairs** |
|---|---|---|---|
| `softmax` | 7.2e-05 | 5.73 | 2% |
| `temp_softmax` | 6.5e-05 | 5.34 | 2% |
| `sparsemax` | **4.1e-04** | 1.30 | **74%** |
| `entmax15` | 1.9e-04 | 1.90 | **58%** |
| `floored_entmax15` | 1.8e-04 | 6.00 | **0%** |
| `gumbel_softmax` | 5.0e-05 | 5.05 | 5% |

Collapse is **progressive** — it begins around epoch 5 and compounds, so it is
not an initialisation artefact and a warm-up would not prevent it. It replicates
on fold 1 (42% / 21%), less severely, which is consistent with more training
data.

`floored_entmax15` — 2% uniform mass mixed back in — eliminates it completely
while keeping the highest end-of-training gradient of any arm.

**Practical consequence: do not ship bare `sparsemax` or `entmax15` in this
encoder.** Three quarters of the delay kernels freeze, and nothing in the
training logs would tell you. If sparsity is wanted, use `floored_entmax15`.

---

## 5. Forecast accuracy at horizon 1 — no arm improves the model

Full sweep, 9 folds × 3 seeds × 2 backbones, 20.6 min on an RTX 4070 Laptop GPU.
Headline = mean over folds 1, 2, 3, 6, 7, 8, 9.

| Backbone | Arm | MAE | Peak MAE | Seed sd |
|---|---|---|---|---|
| `gru_only` | `sparsemax` | 16.67 | 28.45 | 0.50 |
| `gru_only` | `gumbel_softmax` | 16.84 | 28.68 | 0.32 |
| `gru_only` | `entmax15` | 16.99 | 28.98 | 0.24 |
| `gru_only` | `floored_entmax15` | 17.00 | 28.86 | 0.20 |
| `gru_only` | `temp_softmax` | 17.09 | 28.96 | 0.17 |
| `gru_only` | `softmax` | 17.42 | 29.32 | 0.34 |
| `gcn_gru` | `entmax15` | 19.51 | 31.14 | 0.22 |
| `gcn_gru` | `softmax` | 20.00 | 31.70 | 0.55 |
| `gcn_gru` | `gumbel_softmax` | 21.42 | 33.38 | 1.43 |

`sparsemax` on `gru_only` beats the control by 0.75 MAE against a seed sd of
0.34, which clears the usual bar in this project. **It should not be reported as
an improvement, for two independent reasons.**

**It is one fold.** Decomposed per fold, fold 1 (the 2017 epidemic) contributes
**96%** of that margin; the mean over the other six headline folds moves by
**+0.033 MAE**. The same holds for every arm on both backbones — fold-1 share
ranged 96–117%, ex-fold-1 movement ±0.03 MAE. A paired t-test over folds gives
**t = +1.05**, nowhere near significance at seven folds.

This is precisely the shape README §8 records for the loss work, where
`level_weighted` also cleared the seed sd and also turned out to be 100.1% one
fold. `scripts/22` now performs this decomposition itself and prints it directly
beneath the seed-sd verdict, because the seed-sd test on its own has now been
misleading twice in this repository.

**And it is the expected result.** README §7 established that at horizon 1 the
previous period's case count carries nearly all the forecasting signal —
dropping every climate channel costs about +0.01 MAE. No change to how the
climate channels are *smoothed* can move a headline that climate barely enters.
A flat MAE column here is not evidence against the activation; it is evidence
about the forecasting problem, and it was predicted before the sweep was run.

---

## 6. Delay recovery

The column that can carry information at h = 1. `r` is the Pearson correlation
between the learned per-district rainfall delay and the independent
cross-correlation delay from `scripts/17`.

| Arm | r (`gru_only`) | r (`gcn_gru`) |
|---|---|---|
| `floored_entmax15` | **+0.41** | +0.01 |
| `sparsemax` | +0.40 | −0.09 |
| `temp_softmax` | +0.37 | −0.01 |
| `softmax` | +0.36 | −0.03 |
| `entmax15` | +0.36 | −0.02 |
| `gumbel_softmax` | +0.29 | +0.13 |

**No arm meaningfully separates from the control**, and the spread across arms
on `gru_only` (+0.29 to +0.41) is not large enough at three seeds to rank them.

Two things are worth noting rather than claiming:

- On `gru_only` every arm — **including plain softmax, at +0.36** — recovers the
  measured delay far better than the **−0.16** in `docs/learnable_lags_results.md`.
  That figure came from a single seed on a different configuration, so this is
  not a contradiction, but it does mean **−0.16 should not be quoted as the
  softmax encoder's delay-recovery number** without stating those conditions.
- The `gcn_gru` column collapses to near zero for every arm. The graph, which
  §6 of the README shows measurably hurts, appears to destroy delay recovery as
  well — consistent with the contiguity graph mixing districts whose delays
  differ.

**These numbers are pooled over folds, and the per-fold spread is larger than
the between-arm spread.** Computing `r` fold by fold on `gru_only` gives softmax
+0.28 (sd 0.17, range +0.00 to +0.49), `sparsemax` +0.27 (sd 0.14), and
`floored_entmax15` +0.34 (sd 0.12). A ranking whose gaps are 0.05 cannot be read
off quantities that move by 0.5 between folds — which is the arithmetic reason
no arm can be said to separate here, independent of the caveat below.

**A caveat that limits all of §6.** `scripts/17` itself warns that rainfall's
median deseasonalised peak correlation is only **+0.036**, below its own stated
0.10 resolvability threshold: *"the peak lag is not resolvable and should not be
quoted as a delay."* The target these correlations are measured against is
therefore weak. A number computed against an unresolvable target is a weak
number, whichever direction it points.

---

## 7. Horizon 4 — the follow-up, and where a signal would live

README §9 (and §5's list here) singled out horizon 4 as the one condition that
could give the activation something to bite on: at h = 1 the forecast origin
carries nearly all the signal, but at h = 4 the measured 5–10 week rainfall
delay should become load-bearing. The same 9-fold × 3-seed × 2-backbone sweep
was run at `--horizon 4`.

**Accuracy.** Headline MAE, mean over the seven headline folds:

| Backbone | Arm | MAE | Seed sd |
|---|---|---|---|
| `gru_only` | `gumbel_softmax` | 26.03 | 0.19 |
| `gru_only` | `sparsemax` | 26.39 | 0.25 |
| `gru_only` | `entmax15` | 26.47 | 0.29 |
| `gru_only` | `temp_softmax` | 26.66 | 0.29 |
| `gru_only` | `softmax` | 26.82 | 0.25 |
| `gru_only` | `floored_entmax15` | 26.90 | 0.46 |
| `gcn_gru` | `softmax` | 29.60 | 0.61 |

**The signal is stronger than at h = 1 but still not established.** Best on
`gru_only` is `gumbel_softmax`, +0.79 MAE over the control against a 0.25 seed
sd. Two things separate this from the h = 1 false positive:

- **It is not one fold.** Fold 1 contributes 58% of the margin (against 96–117%
  at h = 1), and the mean over the other six headline folds moves by **+0.38
  MAE** — a real shift, where at h = 1 it was +0.03. `sparsemax` and `entmax15`
  show the same pattern, smaller (ex-fold-1 +0.18, +0.14).
- **But the paired t-test over folds is t = +1.63** (need ≈ 2.4 at seven
  folds), so this is a trend, not a result. And the leading arm is
  `gumbel_softmax`, the *stochastic* one — some of its edge is sampling
  variance, not the sparsity mechanism the experiment is about. `sparsemax` and
  `entmax15`, the arms whose behaviour is understood, sit at +0.42 and +0.35
  (t = +1.58, +1.45).

On `gcn_gru` there is nothing: `gumbel_softmax`'s apparent +0.44 is 148% fold 1,
the other folds move **−0.25**, and t = +0.64. The graph erases the effect here
exactly as it erased delay recovery at h = 1.

**Delay recovery at h = 4** does move in the expected direction on `gru_only` —
`sparsemax` +0.20 and `gumbel_softmax` +0.21 against softmax's +0.01 — but the
per-fold spread (sd ≈ 0.20, ranges crossing zero) is still as wide as the
between-arm gap, so this cannot be ranked either, and §6's resolvability caveat
still applies.

**One inversion worth noting.** `floored_entmax15`, the arm built to be robust,
is the *worst* on `gru_only` at h = 4 (−0.08 vs the control). The dead-gradient
guard costs something here: at h = 4 a concentrated single-bump kernel appears
to be closer to right, and the arms that freeze onto one bump (`sparsemax`,
`entmax15`) do slightly better for it. That does not change the §4 advice —
freezing 58–74% of *all* kernels is still a failure mode, not a feature — but it
means `floored_entmax15` is not a free upgrade in every regime.

**h = 4 verdict:** the activation matters more at longer horizons than at h = 1,
directionally as predicted, but the sweep does not establish an accuracy
improvement at three seeds. It would need more seeds, or a horizon further out,
to move from trend to result — and the arm to test is `sparsemax` or
`entmax15`, not the stochastic `gumbel_softmax`.

Numbers on disk: `results/models/activation_*_h4.*` (the `_h1.*` copies are the
horizon-1 run).

---

## 8. What may and may not be claimed

**May be claimed:**

- Softmax saturation in this encoder is real and measured, and the 10× learning
  rate in `scripts/16` is treating a genuine symptom.
- Sparse simplex maps freeze 58–74% of delay kernels into an unrecoverable
  state; `floored_entmax15` prevents this entirely.
- No activation change improves forecast accuracy at horizon 1, and the reason
  is understood and was predicted in advance.
- At horizon 4 the activation has a directional effect that survives removing
  the epidemic fold (`gumbel_softmax` +0.38 MAE ex-fold-1 on `gru_only`), which
  it did not at h = 1.

**May not be claimed:**

- That any arm improves the model. None does at h = 1 once the epidemic fold is
  separated out; at h = 4 the best trend is t = +1.63, short of significance,
  and led by the stochastic arm.
- That any arm improves delay recovery. The per-fold spread does not separate
  from the control at either horizon, and the target is below its own
  resolvability threshold.
- That `−0.16` has been "fixed". Plain softmax scores +0.36 in this
  configuration, so the change is a configuration difference, not an
  activation one.
- That `floored_entmax15` is a free upgrade — it is the worst arm on `gru_only`
  at h = 4.

---

## 9. Reproducing

```powershell
.venv\Scripts\python scripts\17.lag_correlation_scan.py                 # measured delays
.venv\Scripts\python scripts\22.train_simplex_activations.py            # h=1 sweep, ~21 min GPU
.venv\Scripts\python scripts\22.train_simplex_activations.py --horizon 4  # h=4 sweep, ~45 min GPU
.venv\Scripts\python scripts\23.activation_gradient_probe.py --fold 8   # mechanism probe, ~1 min
```

The two sweeps write the same filenames, so copy the first run's
`results/models/activation_{metrics,kernels}.csv` and `activation_report.md`
aside (the repo keeps them as `*_h1.*` / `*_h4.*`) before the second.

Useful flags: `--arms softmax entmax15`, `--folds 8`, `--seeds 1`,
`--no-control`.

Tests: `tests/test_simplex_activations.py` (43) and
`tests/test_activation_experiment.py` (16).

---

## 10. What to do next

1. **Do not pursue the activation further at h = 1.** The null result is well
   explained — a property of the forecasting problem that no simplex map can
   change.
2. **h = 4 is a trend, not a result (t = +1.63).** If it is worth chasing, run
   `sparsemax` and `entmax15` — not `gumbel_softmax` — at more seeds, or at a
   longer horizon still. The mechanism arms sit at t ≈ +1.5; the stochastic arm
   leading the table is partly sampling noise.
3. **`floored_entmax15` for a concentrated kernel where the regime is unknown** —
   it removes the §4 dead end. But it is *not* a universal upgrade: it is the
   worst arm on `gru_only` at h = 4, where hard concentration happens to help.
4. **Whatever the graph work in README §9 concludes, re-check delay recovery on
   it.** Every activation's delay signal collapses to zero on `gcn_gru` at both
   horizons; a better graph might restore it, or might not, and that is cheap to
   measure once a candidate graph exists.

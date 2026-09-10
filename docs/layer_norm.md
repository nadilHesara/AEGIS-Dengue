# Layer normalisation before the prediction head

`scripts/28.train_layer_norm.py`, `src/models/layer_norm.py`,
`tests/test_layer_norm.py`. Report: `results/models/layer_norm_report.md`.

## The question

The GCN+GRU trunk hands the GRU's final hidden state straight to a linear head,
with nothing controlling that state's scale. The recurrent cell can drift its
activation magnitude across the 12-period window, and the graph convolution
feeding it is an unnormalised linear map on features whose per-fold
standardisation was fitted on the training years only. Layer normalisation at
that point is the standard treatment for recurrent sequence models
(Ba et al., 2016).

Two arms, one code path:

| Arm | Forward pass |
| --- | --- |
| `baseline` | `GRU -> dropout -> head` — the committed architecture |
| `layer_norm` | `GRU -> LayerNorm(32) -> dropout -> head` |

## Design decisions

**The norm goes before the dropout, not after.** Normalising after dropout would
compute statistics on a vector whose moments dropout has distorted at training
time but not at evaluation time, widening the train/eval gap dropout already
creates. Before-dropout is also the ordering the recurrent and transformer
blocks this borrows from use.

**Statistics are over the hidden axis, per (window, district) row.** Each
district's representation is normalised on its own terms. Normalising across the
node axis would rescale a district in an outbreak by what its neighbours are
doing, which is a materially different model.

**The affine gain and bias are left on** (torch's default). Turning them off
would impose unit scale as a constraint rather than offer normalisation as a
reparameterisation; the affine version is what "add LayerNorm" conventionally
means and is strictly the more general of the two. It also makes the module
falsifiable: the learned gain says whether the model used the normalisation or
learned to undo it.

**The control is the same class with `normalise=False`.** It reproduces `GCNGRU`
bit-for-bit — `tests/test_layer_norm.py` asserts identical outputs, an identical
parameter count and identical `state_dict` keys. Both arms build the backbone
under the same seed in the same order and the wrapper adopts its submodules by
reference, so the shared weights are drawn identically and the paired comparison
is genuinely paired. The `baseline` arm reproduces the committed 18.98 from
`docs/baseline.md` exactly, which is the check that the harness is neutral.

**The head is always one wide.** `horizon` is overloaded in `scripts/16`: the
forecast lead in `build_fold_arrays`, the head width in the `GCNGRU`
constructor. They coincide at h=1, the only value script 16 runs. This
experiment trains one model per lead, as in script 27's `separate` arm, so the
lead belongs to the windower and the head stays one wide.

## Results

Nine folds, three seeds, `gcn_gru` on `v1`. Headline MAE is the mean over the
seven headline folds.

| h | `baseline` | `layer_norm` | Δ | Folds improved | p | Δ ex-fold-1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 18.98 | 17.41 | −1.57 | 3/7 | 0.35 | **−0.02** |
| 2 | 23.79 | 21.73 | −2.05 | 4/7 | 0.32 | **−0.16** |
| 3 | 27.18 | 25.69 | −1.49 | 5/7 | 0.36 | **−0.01** |
| 4 | 29.93 | 27.65 | −2.29 | 5/7 | 0.29 | **−0.31** |

### The headline movement is fold 1 and nothing else

Per-fold Δ (negative = normalisation better):

| Fold | Year | h=1 | h=2 | h=3 | h=4 |
| --- | --- | --- | --- | --- | --- |
| **1** | **2017** | **−10.86** | **−13.41** | **−10.33** | **−14.17** |
| 2 | 2018 | +0.04 | −0.02 | −0.50 | −0.27 |
| 3 | 2019 | −0.20 | +0.33 | −0.03 | −0.83 |
| 6 | 2022 | +0.01 | −0.21 | −0.13 | −0.14 |
| 7 | 2023 | −0.16 | −1.49 | +0.88 | +0.29 |
| 8 | 2024 | +0.13 | +0.31 | −0.49 | −0.98 |
| 9 | 2025 | +0.04 | +0.12 | +0.20 | +0.10 |

This is the fifth change in this project to produce a headline movement that is
the 2017 epidemic fold and nothing else (README §8, §8b, §8c, §8d, and now this).
Averaged over the other six headline folds the change is between −0.02 and −0.31
MAE — smaller than the seed spread. No paired test clears p = 0.05 at any
horizon, and the sign is not even consistent fold to fold.

**On the headline forecasting score, layer normalisation is not established as
an improvement.** The one place it moves the number substantially is the
epidemic year, where the baseline's own errors are largest.

### What does survive: seed stability

Mean per-fold seed standard deviation:

| h | `baseline` | `layer_norm` | Change |
| --- | --- | --- | --- |
| 1 | 0.795 | 0.605 | −24% |
| 2 | 1.294 | 0.639 | −51% |
| 3 | 1.300 | 0.824 | −37% |
| 4 | 1.671 | 1.008 | −40% |

This holds at **every** horizon, in the same direction, and the absolute gap
widens as the horizon grows — the pattern expected if normalisation is doing
what it is advertised to do. Run-to-run variance is a real cost in a project
whose effect sizes are routinely under 1 MAE: a tighter seed spread makes every
subsequent comparison cheaper to resolve. That, rather than the headline MAE, is
the defensible reason to keep the norm.

Convergence speed did not move consistently (Δ epochs to best: +4.6, −1.7, +2.6,
+0.1), so the usual "trains faster" claim is not supported here.

### What the norm learned

| h | Gain mean | Gain sd | Gain min | Gain max | Mean abs bias |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.905 | 0.065 | 0.658 | 0.970 | 0.038 |
| 2 | 0.924 | 0.054 | 0.717 | 0.988 | 0.034 |
| 3 | 0.925 | 0.051 | 0.732 | 0.988 | 0.034 |
| 4 | 0.931 | 0.052 | 0.731 | 0.994 | 0.032 |

The gain stays near its initialisation of 1.0 and the bias near 0 at every
horizon. The module is doing plain normalisation: it did not collapse toward
zero (the head ignoring the representation) and did not grow large (capacity
spent undoing the rescaling). So the flat headline result is "normalising did
not help the score", not "the model switched the normalisation off" — those are
different findings and the affine parameters are logged to
`results/models/layer_norm_affine.csv` to tell them apart.

## Interpretation

The result is consistent with the standing explanation for every negative result
in this project (`docs/learnable_lags_results.md`): the previous period's case
count carries nearly all the signal, so changes that improve optimisation
conditioning cannot move a ceiling set by the information in the inputs. Layer
normalisation addresses conditioning. It measurably improved conditioning — the
seed spread fell by 24–51% — and the forecasting score still did not move outside
the epidemic fold. That is evidence about where the ceiling comes from, and it is
the reason the experiment was worth running despite the prior.

The 2017 behaviour is worth a note rather than a claim. 2017 is the epidemic year
and the fold where the un-normalised model does worst (54.81 MAE at h=1); a
normalised representation bounding the head's input scale plausibly helps most
where the activations are furthest from their training-year range. On a single
fold that is a hypothesis, not a finding.

## Recommendation

Keep the norm configurable and default it **off** for headline comparability
with §6–§8e, which is how `scripts/16` is left. Turn it on when seed variance is
the binding constraint — hyperparameter selection, or any comparison with an
expected effect under 1 MAE — where a 24–51% reduction in run-to-run spread is
worth having and costs 64 parameters.

## Reproducing

```bash
python scripts/28.train_layer_norm.py                    # h=1, both arms, 9 folds, 3 seeds
python scripts/28.train_layer_norm.py --horizons 1 2 3 4 # the tables above
python scripts/28.train_layer_norm.py --model gru_only   # identity adjacency
python -m pytest tests/test_layer_norm.py
```

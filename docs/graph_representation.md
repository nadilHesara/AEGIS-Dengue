# Graph representation — does any adjacency beat no adjacency?

Four graphs through one GCN+GRU, anchored on the identity control. What each one
scores, why the fixed alternative failed, and what the learned one actually
changes.

Reproduce with `python scripts/26.train_graph_variants.py`. Tests:
`tests/test_graph_variants.py`. Run on disk:
`results/models/graph_report.md`, `graph_metrics.csv`,
`graph_learned_adjacency.csv`.

---

## 1. The question, and why it is not "does contiguity help"

[`baseline.md`](baseline.md) §3 finding 2 settled that queen contiguity does not
help: `gru_only` — the identical model with the identity substituted for the
adjacency — beats `gcn_gru` on all nine folds. That result is about *one* graph.
Two readings survive it:

    the graph is wrong        contiguity is a poor prior for a country 430 km
                              long. Colombo and Galle are 100 km apart, share no
                              border, and are both wet-zone coastal; contiguity
                              scores that pair zero. On this reading a better
                              graph should help.

    there is no spatial       whatever cross-district signal exists at one week
    signal to find            ahead is already inside each district's own case
                              history. On this reading no graph helps.

Separating them needs more than one alternative graph, because a fixed
alternative that fails could always be the wrong fixed alternative. So the sweep
includes a graph with no prior at all, learned end-to-end from the forecasting
loss.

**The reference is `identity`, not `contiguity`.** Contiguity already loses to
doing nothing, so beating it establishes nothing. Every comparison below is
against the identity control.

---

## 2. The four arms

| Arm | Graph | Edges | Symmetric | Source |
|---|---|---|---|---|
| `identity` | `I` | 0 | yes | the control — graph conv becomes a per-node linear layer, i.e. `gru_only` |
| `contiguity` | `A_norm` | 114 | yes | queen contiguity on GADM polygons, `scripts/13` |
| `gaussian` | `A_gaussian_norm` | 106 | yes | `exp(-(d/50km)²)` on centroid distance, thresholded at 0.1, `scripts/13` |
| `adaptive` | learned | dense | **no** | `softmax(relu(E1 @ E2ᵀ))`, `src/models/adaptive_graph.py` |

Only the adjacency changes. Architecture, training loop, early stopping,
optimiser, folds, masks, preprocessing, windowing, the anchored residual target,
loss and metrics are imported from `scripts/16.train_gcn_gru.py`.

**Two controls, both verified exact.** The `identity` arm reproduces the
committed `gru_only` v1 fold-1 MAE of **42.97**, and `contiguity` reproduces
`gcn_gru` v1's **54.81**. If either had drifted, nothing else would be
comparable.

### The adaptive graph

Graph WaveNet's adaptive adjacency (Wu et al., 2019): two learned `[25, 8]`
embedding tables, `A = softmax(relu(E1 @ E2ᵀ))`. **400 parameters** against
~8,200 for the backbone. Three properties matter: it is *asymmetric* (both fixed
graphs are symmetric by construction; disease spread need not be),
*row-stochastic* (so the convolution averages and cannot rescale the features),
and *small* — a free 25×25 matrix would be 625 parameters learned from ~500
training windows, more than the backbone.

### The graph learning rate is not a free choice

The embeddings sit behind a relu and a softmax, so their gradient is far smaller
than the GRU's — the asymmetry `scripts/16.parameter_groups` documents for the
lag encoder. Measured on fold 8:

| Multiplier | Normalised row entropy | Largest weight | Fold-8 MAE |
|---|---|---|---|
| 1× | **1.000** (uniform) | 0.050 (1.3× uniform) | 10.30 |
| 3× | 0.061 | 0.9999 (25× uniform) | 9.21 |
| 10× | 0.085 | 1.0000 (25× uniform) | 9.45 |

At 1× the graph never leaves its initialisation and the arm scores *worse* than
the identity — it tests whether the embeddings can move, not whether a learned
graph helps. At 3–10× it moves and collapses toward one-hot rows.

Neither extreme is obviously right, and choosing by test score is the mistake
[`hyperparameter_tuning.md`](hyperparameter_tuning.md) exists to avoid. So the
multiplier is **selected per fold on the validation split** from
{1, 3, 10, 30}×. The selected values were 30× (folds 1–2), 10× (folds 3–6),
3× (folds 7–9) — note the trend, which tracks how much training data each fold
has.

---

## 3. Results — 9 folds, 3 seeds, 6.2 min GPU

### Single-seed models, headline folds

| Arm | MAE | vs `identity` | Peak MAE | 2017 MAE | Seed sd |
|---|---|---|---|---|---|
| `adaptive` | **16.61** | −0.07 | 28.08 | **40.69** | 0.39 |
| `identity` (control) | 16.68 | — | 28.06 | 42.97 | 0.53 |
| `contiguity` | 18.98 | +2.30 | 30.36 | 54.81 | 0.68 |
| `gaussian` | 19.25 | +2.57 | 30.89 | 56.81 | 0.46 |

### Per fold

| Fold | Year | `identity` | `adaptive` | `contiguity` | `gaussian` | adaptive − identity |
|---|---|---|---|---|---|---|
| 1 | 2017 | 42.97 | **40.69** | 54.81 | 56.81 | **−2.28** |
| 2 | 2018 | **9.90** | 10.00 | 10.10 | 10.03 | +0.11 |
| 3 | 2019 | **16.82** | 17.11 | 17.63 | 17.61 | +0.29 |
| 6 | 2022 | 11.63 | **11.61** | 12.12 | 11.98 | −0.01 |
| 7 | 2023 | **17.95** | 18.57 | 19.16 | 19.30 | +0.62 |
| 8 | 2024 | **9.15** | 9.66 | 10.09 | 9.97 | +0.51 |
| 9 | 2025 | **8.35** | 8.61 | 8.98 | 9.06 | +0.26 |

### Against the identity control

| Arm | Δ | Folds improved | t | p | Δ in seed-sd | Verdict |
|---|---|---|---|---|---|---|
| `adaptive` | −0.07 | 2/7 | −0.19 | 0.853 | 0.16 | not established |
| `contiguity` | +2.30 | 0/7 | +1.45 | 0.199 | 3.81 | **worse** |
| `gaussian` | +2.57 | 0/7 | +1.37 | 0.221 | 5.19 | **worse** |

---

## 4. Verdict: no graph beats no graph

**No arm improves on the identity control by an established margin.** That is
the answer to the question the brief asked.

**The Gaussian graph does not rescue the idea — it is worse than contiguity.**
19.25 against 18.98, losing on all seven headline folds and *worse than
contiguity on the epidemic fold* (56.81 vs 54.81). This was the obvious
competing prior, explicitly named in the work plan as the thing to test before
building a dual graph, and it fails. Distance-based spatial smoothing is not the
missing ingredient.

**The adaptive graph is a wash, not a win.** −0.07 MAE is 0.16 seed-sd,
t = −0.19, p = 0.85, and it improves only **2 of 7 folds**. Its headline number
is essentially the identity's with a different distribution across folds: a large
gain on fold 1 (−2.28) paid for by small losses on five of the other six
(+0.11 to +0.62). Averaged over the non-epidemic headline folds it is **+0.30
MAE — worse than no graph at all.**

This is the fifth consecutive change in this project whose headline movement is
the 2017 epidemic fold and nothing else (README §8, §8b, §8c, §8d). At this point
the pattern is a prior, not a coincidence.

### The one genuinely new thing: fold 1

The adaptive graph's fold-1 result is the largest single-fold improvement over
the identity control anything in this repository has produced — **42.97 → 40.69
single-seed, 42.57 → 39.26 as a seed-mean ensemble.** For context, the entire
headline gap between the baseline and persistence lives in this fold, and no
previous change moved it while starting from the *identity* backbone rather than
from contiguity.

The seed-mean ensemble is worth stating separately, because it is free (the seeds
are already trained) and it is where the effect is largest:

| Arm | Ensemble headline MAE | 2017 |
|---|---|---|
| `adaptive` | **16.21** | **39.26** |
| `identity` | 16.55 | 42.57 |
| persistence | 16.42 | 36.08 |

The adaptive ensemble at 16.21 is the **first number in this repository below
persistence's 16.42**, and it beats persistence on 6 of 7 headline folds. But
the paired t is **−0.37, p = 0.72** — fold 1's magnitude dominates the variance,
and 6-of-7 with a 0.22 mean margin is not a win at this sample size. It ties
persistence, as §8's `level_weighted` ensemble did. **Peak MAE still loses**
(27.47 vs 26.61), which is the criterion that matters for an outbreak warning
system.

---

## 5. What the learned graph converged to

A learned adjacency that stayed uniform would reproduce the identity's answer for
an uninteresting reason, so this was measured rather than assumed.

| Quantity | Value | Reading |
|---|---|---|
| Normalised row entropy | **0.794** | 1.0 = uniform, 0.0 = one-hot |
| Largest weight | 0.480 | 12× the uniform 0.04 |
| Mass in each row's top 3 | — | vs 0.12 if uniform |
| Correlation with contiguity | **−0.108** | did it rediscover the border graph? |

Two things follow.

**It learned real structure.** Entropy 0.79 with a largest weight 12× uniform is
a graph that concentrated substantially, averaged over folds and seeds. It is not
the identity in disguise and not a uniform blur. (At the per-fold level the
higher multipliers push much further — fold 8 at 3× reached entropy 0.06 with
near-one-hot rows — so 0.79 is the average over a range of selected rates, not a
uniformly mild graph.)

**It did not rediscover geography.** The correlation with the contiguity graph is
**−0.108** — slightly *negative*. Whatever the model found useful, it is not the
border structure, and it is not distance either, since the Gaussian graph
performs worst of all. Inspection of the one-hot-collapsed runs showed
geographically arbitrary pairings (Ampara→Kurunegala, Matale→Mannar) with only
4 of 25 districts selecting themselves.

The honest reading: a graph given 400 free parameters and the training objective
itself finds *something*, that something helps materially in epidemic conditions
and hurts slightly otherwise, and it bears no resemblance to physical adjacency.
That is more consistent with the model exploiting a statistical shortcut across
districts during an unusual year than with it discovering a spatial transmission
pathway.

---

## 6. What this closes and what it opens

**Closed: the "contiguity is just the wrong graph" hypothesis.** Three graphs —
a border graph, a distance kernel, and one learned end-to-end from the loss — all
fail to beat having no graph. A fixed alternative failing could be the wrong
fixed alternative; a learned one failing too is evidence about the data. At
horizon 1 the previous period's case count carries nearly all the signal
([`learnable_lags_results.md`](learnable_lags_results.md) measured dropping every
climate channel at +0.01 MAE), and cross-district structure has little left to
add.

**Also closed: the dual-graph work plan's precondition.** `docs/baseline.md` §4
item 7 said to establish that *some* graph beats the identity before building a
season-gated mixture of contiguity and a learned graph. That precondition is now
tested and not met. Building a gated mixture of two components that each lose to
the identity would be building on a measured negative.

**Open, and worth one more test: the adaptive graph at longer horizons.** The
fold-1 result is the only real signal any graph has produced here, and the
standing explanation for why spatial structure has nothing to add is
horizon-specific — at h=1 the forecast origin dominates. §8b found the same
directional hint for the lag encoder at h=4. Running this same sweep at
`--horizon 4` is a small change and is the one condition under which the adaptive
graph might separate from the control on more than one fold.

**Not worth pursuing:** more fixed graphs. Two priors have now failed in the same
direction and the learned graph found something uncorrelated with both.

---

## 7. Files

| File | Contents |
|---|---|
| `src/models/adaptive_graph.py` | `AdaptiveAdjacency` (the learnable graph) and `AdaptiveGraphGCNGRU` (the wrapper) |
| `scripts/26.train_graph_variants.py` | the four-arm sweep |
| `tests/test_graph_variants.py` | 25 tests: row-stochasticity, differentiability, asymmetry, the identity-control identity, optimiser routing, validation-only LR selection |
| `results/models/graph_report.md` | generated report |
| `results/models/graph_metrics.csv` | per-arm per-fold per-seed metrics plus ensemble rows |
| `results/models/graph_learned_adjacency.csv` | the learned adjacency, per fold and seed — the evidence for §5 |

---

## Related documents

- [`baseline.md`](baseline.md) — the graph result this extends, and the numbers both controls reproduce
- [`hyperparameter_tuning.md`](hyperparameter_tuning.md) — the validation-only selection discipline reused here
- [`learnable_lags_results.md`](learnable_lags_results.md) — why horizon 1 leaves so little for any structure to add
- [`improvements.md`](improvements.md) — the single-fold pattern this makes five of

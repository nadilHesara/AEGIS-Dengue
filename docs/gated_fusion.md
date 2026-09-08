# Gated spatial/temporal fusion

**Item #8 of the AEGIS-Dengue contribution.** Give each district a learned
scalar that blends the graph convolution over the contiguity adjacency against
the same convolution over the identity, so the model can keep the graph where it
helps and route past it where it does not — instead of the whole country being
forced to one choice.

What the failure was, the scope decision, how it was measured, what it bought.

Reproduce with `python scripts/21.train_gated_gcn_gru.py --variants v1 --seeds 3`.
Tests: `tests/test_gated_fusion.py`.

**The headline is a negative result with a mechanism, and it is the second
result in the project with the same mechanism.** The learnable lag module
(item #6) failed because the forecasting gradient at horizon 1 carries almost no
information about the structure it was learning. This gate fails the same way.
That the same diagnosis explains two independent architecture changes is the
finding; do not present either in isolation.

---

## 1. The failure this targets

In the committed baseline sweep (9 folds x 3 seeds, `results/models/baseline_report.md`,
[`docs/baseline.md`](baseline.md) §3), **`gru_only` beats `gcn_gru` on every
headline fold.** Replacing the contiguity adjacency with the identity — cutting
the graph out entirely — lowers the error: 17.12 to 16.67 on `v1` headline MAE,
and on every individual headline fold. The contiguity graph is not earning its
place.

`docs/baseline.md` §4 records the consequence for the dual-graph item (#7):
*"the margin is not near zero, it is negative on all nine folds — contiguity
costs 1.9 MAE against no graph at all. Adding a gated mixture on top of a
component that is measurably harmful risks burying the problem rather than
fixing it."*

**What this item asks.** `gru_only > gcn_gru` is a statement about the *average*
district. Colombo borders six districts and sits on the main road network;
Jaffna borders one. There is no reason the graph should help or hurt all 25
equally. So instead of choosing between `gcn_gru` and `gru_only` for the whole
country, let each district choose.

**The number to beat:** `gru_only`, the specific failure this targets. Not
`gcn_gru`, and not persistence — a gate that can route each district to whichever
fixed choice is better should at least weakly dominate the better of the two.

---

## 2. Scope — the small gate, not the full Phase 5

Item #8 can be read two ways. The plan weighed them before any code:

| | Workplan Phase 5 gate | This item's spatial-vs-temporal gate |
|---|---|---|
| Gates over | 5 relation channels (contiguity, gravity, climate-similarity, adaptive, dynamic) | GCN branch vs. GRU-only branch |
| Needs first | `src/graphs/` — 4 new adjacency builders, a leakage test on the climate-similarity channel, a learned gravity exponent θ | nothing new upstream; `adjacency.npz` and the identity are both already in `scripts/16` |
| Fixes | "which relation matters" — an open question nobody has asked yet | **`gru_only` beats `gcn_gru` on every headline fold** — a measured, documented failure |
| Size | large — a full week; 4 adjacencies + fusion + ablations | small — one gate module, 25 parameters |
| Diagnosability | gate mix per district over 5 channels | gate value per district: does it go low for isolated nodes like Jaffna? |

**Decision: the small gate.** Reasons, in order:

1. **It targets the failure that is actually documented and blocking.** Phase
   5's "which relation matters" is speculative until the graph earns its place
   at all.
2. **It is the mechanism `scripts/16` was structured for.** Its docstring:
   *"Spatial first, then temporal, because that is where a spatial-versus-temporal
   fusion gate goes later. Swapping the order or fusing the two into a single
   recurrent graph cell would make this baseline structurally unlike the model
   it is supposed to be the baseline for."*
3. **It is thin,** per Phase 5's own instruction (5.3: *"Keep it thin — the
   contribution is the relation set and the gating, not a baroque propagation
   rule."*).
4. **It avoids the `src/graphs/` leakage risk.** The full multiplex needs a
   climate-similarity adjacency computed on training-fold data only; the
   workplan flags this as *"the most likely place for a subtle leak that
   invalidates your headline result."* Not a first step.
5. **It degrades gracefully either way.** If the gate learns to route around the
   graph (g → 0 everywhere), that is a clean, quantified statement of "the
   contiguity graph does not help at this resolution" — which is exactly the
   `gru_only > gcn_gru` observation, now with a mechanism.

The full Phase 5 multiplex remains a legitimate reading of item #8, and item #7
(dual graph) is the same territory. Both need the `src/graphs/` builders as a
prerequisite phase and are out of scope here.

---

## 3. What changed

`src/models/gated_fusion.py` — `GatedGCNGRU(nn.Module)`, following the
`LagGCNGRU` pattern: the constructor takes a plain `GCNGRU` as `backbone` and
reuses its `graph_layers`, `dropout`, `gru` and `head` directly; its `forward`
is never called. For district *i*, at each step of the input window:

```
spatial[i]  = GCN_stack(x, A_norm)[i]
temporal[i] = GCN_stack(x, I)[i]           the SAME layers, identity in place of A
g[i]        = sigmoid(gate_logit[i])       one learned scalar per district, init 0
fused[i]    = g[i] * spatial[i] + (1 - g[i]) * temporal[i]
```

then the baseline's unchanged tail: one GRU sequence per district, the shared
GRU, the linear head.

**The shared weights are the whole point.** `temporal` is not a separate branch
with its own parameters — it is the identical `GraphConv` layers applied with
the identity matrix, which collapses each to a per-node linear map. So every
`g[i] = 1` reproduces `gcn_gru` exactly, every `g[i] = 0` reproduces `gru_only`
exactly, and nothing sits between the two except 25 numbers deciding, per
district, which one to be. A separate temporal branch with its own weights would
add capacity and blur what the gate measures.

**Parameter budget.** `gate_logit`, one scalar per node: **25**. The model goes
8,193 → 8,218, up 0.3%. Too few to plausibly be a capacity change, which the
`gated_uniform` ablation confirms (§6.2).

**Initialisation.** `gate_logit = 0` → `sigmoid(0) = 0.5`, an even blend, no
bias toward either branch. Whether the gate then moves off 0.5, and toward which
end, is the measurement that decided the outcome (§6.1).

**What is deliberately not touched.** The graph. This is the *fusion* half of
workplan Phase 5, not the multiplex-relation half. Only the spatial/temporal
blend moves; the adjacency, the GCN weights, the GRU, the head, the anchored
residual target, the masked loss, the folds and the per-fold refitted
preprocessing are all the baseline's, imported from `scripts/16`.

### The arms

`scripts/21.train_gated_gcn_gru.py` imports the training loop, fold
construction, preprocessing, windowing, anchored target, masked loss and metrics
from `scripts/16` rather than copying them — the same discipline `scripts/18`
and `scripts/20` follow. Four arms, all on the same folds, seeds and masks:

| Arm | Adjacency | Model | Role |
|---|---|---|---|
| `gcn_gru` | contiguity | plain `GCNGRU` | control — must reproduce `scripts/16` |
| `gru_only` | identity | plain `GCNGRU` | **the number to beat** |
| `gated` | contiguity | `GatedGCNGRU`, learned | the contribution |
| `gated_uniform` | contiguity | `GatedGCNGRU`, `freeze_gate=0.5` | ablation: does *learning* the gate matter, or is a fixed 50/50 blend already this? |

`gcn_gru` and `gru_only` are re-run through this harness rather than read from
the committed report — the harness-neutrality control (§5). `gate_frame()` dumps
the per-district gate value **per fold and per seed**, not averaged, because a
gate that is stable across seeds is a finding and one that is not is a different
finding; same reasoning as `scripts/18.kernel_frame`.

---

## 4. Steps as followed

| Step | What | Exit gate | Outcome |
|---|---|---|---|
| 1 | `src/models/gated_fusion.py` + `tests/test_gated_fusion.py` — 11 self-contained tests | `11 passed`, including both boundary conditions | Met, `11 passed in 17.95s` |
| 2 | `scripts/21.train_gated_gcn_gru.py` with the four arms, importing `scripts/16`'s loop | parses, `--help` renders, import chain resolves without pipeline data | Met |
| 3 | Cheap pass, `--folds 1 8 --variants v1 --seeds 1` | `gated` lands between the two controls; check fold-8 gate values | `gated` landed between the controls, closer to `gcn_gru`; gate values had a spread but did not commit |
| 4 | Full run, `--variants v1 --seeds 3`, all 9 folds, 4 arms | writes `gated_report.md`, `gated_metrics.csv`, `gated_gates.csv` | Met — §6 |

The pipeline artifacts (`folds.json`, `adjacency.npz`, `model_tensors_v1.npz`)
are rebuilt fresh on Kaggle each session; steps 3–4 ran there.

The two boundary conditions in step 1, which the whole design turns on:

- **`test_gate_one_reproduces_gcn_gru`** — `gate_logit` forced to +30, both
  models in `eval()`: `GatedGCNGRU(x, A_real)` matches plain `GCNGRU(x, A_real)`
  to `rtol=atol=1e-5`.
- **`test_gate_zero_reproduces_gru_only`** — `gate_logit` forced to −30:
  `GatedGCNGRU(x, A_real)` matches plain `GCNGRU(x, I)` — exactly `gru_only` — to
  the same tolerance.

Plus: a hand-rolled 50/50 blend matches `gated_uniform`; the learnable gate
receives non-zero gradient after one backward; the frozen gate (a buffer) does
not; output shape; `g` stays in (0, 1) across a wide logit sweep; parameter
count is exactly backbone + 25; no submodule named `encoder` (so `scripts/16`'s
`parameter_groups` leaves the gate at the network's learning rate); wrong node
count raises.

---

## 5. Harness neutrality

If `gcn_gru` and `gru_only` do not reproduce the committed baseline through this
script's harness, nothing else is comparable — the control `scripts/20` insists
on.

| Arm | Metric | Committed `baseline_report.md` (v1) | `scripts/21` this run |
|---|---|---|---|
| `gcn_gru` | headline MAE | 19.01 | **19.00** |
| `gcn_gru` | 2017 MAE | 55.01 | **55.00** |
| `gru_only` | headline MAE | 17.12 | **16.88** |
| `gru_only` | 2017 MAE | 46.01 | **44.17** |

`gcn_gru` reproduces to 0.01. `gru_only` lands ~0.25 lower on the headline and
~1.8 lower on fold 1 — within a plausible seed draw (the committed `gru_only`
seed sd is 0.70, the widest in that table), but `gru_only` is the arm that
wobbles most and this is noted rather than waved away. The imported loop is
neutral.

---

## 6. Results

**Full run: 9 folds, 3 seeds, `v1`, all four arms.** Mean over the headline
folds (1, 2, 3, 6, 7, 8, 9), averaged over seeds. COVID folds (4, 5) excluded
from the headline.

| Arm | MAE | RMSE | Peak MAE | 2017 MAE | COVID MAE | Seed sd |
|---|---|---|---|---|---|---|
| `gru_only` | **16.88** | 35.98 | 28.49 | 44.17 | 7.23 | 0.29 |
| `gated` | 18.36 | 39.77 | 30.27 | 53.17 | 7.23 | 0.36 |
| `gated_uniform` | 18.53 | 40.14 | 30.44 | 53.79 | 7.46 | 0.33 |
| `gcn_gru` | 19.00 | 40.60 | 30.48 | 55.00 | 7.52 | 0.66 |

Per fold, MAE:

| Fold | Year | `gru_only` | `gated` | `gated_uniform` | `gcn_gru` | Note |
|---|---|---|---|---|---|---|
| 1 | 2017 | 44.17 | 53.17 | 53.79 | 55.00 | epidemic |
| 2 | 2018 | 9.76 | 9.99 | 10.00 | 10.08 | |
| 3 | 2019 | 16.92 | 17.28 | 17.54 | 17.59 | |
| 4 | 2020 | 7.40 | 7.44 | 7.87 | 7.78 | COVID |
| 5 | 2021 | 7.07 | 7.02 | 7.04 | 7.26 | COVID |
| 6 | 2022 | 11.59 | 11.57 | 11.65 | 12.17 | |
| 7 | 2023 | 18.14 | 18.50 | 18.56 | 19.05 | |
| 8 | 2024 | 9.14 | 9.40 | 9.50 | 10.09 | |
| 9 | 2025 | 8.46 | 8.62 | 8.69 | 9.06 | |

`gated` sits between the two fixed choices on every headline fold, closer to
`gcn_gru`. It improves on plain `gcn_gru` consistently (2017 53.17 vs 55.00;
fold 8 9.40 vs 10.09) but never reaches `gru_only`, which is already below it on
every fold.

### Verdict against the success criteria

The criteria, as stated in the plan before the run:

**Primary — `gated` headline MAE < `gru_only` by more than the pooled seed
standard deviation.** A gate that can route each district to whichever fixed
choice is better should weakly dominate both. *Stricter than workplan Phase 5's
exit gate* (*"beats both the hand-drawn matrix and the pure adaptive graph"*),
because that is for the full multiplex; for a 2-branch gate, beating `gru_only` —
the specific documented failure — is the correct specialisation.

> **Not met.** `gated` 18.36 against `gru_only` 16.88 is a 1.48 MAE loss, roughly
> 4× the ~0.3 seed sd. An established loss, not noise.

**Secondary — legible per-district gate spread.** The low-degree northern
districts (Jaffna, degree 1; Mannar, Mullaitivu, Kilinochchi) should gate toward
the temporal branch if the contiguity graph is what is hurting them. If gates
spread but Jaffna is high, that is still a finding, just a different one.

> **Disconfirmed** — §6.3. Jaffna gates *highest*.

**Honest-negative fallback (explicitly allowed, like the lag module's).** If
`gated` ≈ `gru_only` and every gate collapses toward 0, that is a quantified
confirmation that the contiguity adjacency contributes nothing at district
resolution.

> **Closest to this, but weaker in form.** The gate did not collapse to 0, it
> stayed near 0.5, and `gated` landed above `gru_only` rather than at it. The
> contiguity graph is confirmed as not salvageable by per-district weighting,
> but the gate could not even route cleanly to the better of its two endpoints.

---

## 6.1 Mechanism — the gate barely moved from its initialisation

`gate_logit` starts at 0, so every gate starts at `sigmoid(0) = 0.5`. After
training, over all 675 district × fold × seed cells (`gated_gates.csv`):

| Statistic | Value |
|---|---|
| Mean `g` | **0.380** |
| sd of `g` | 0.069 |
| Signed mean distance from the 0.5 init | **−0.120** |
| Fraction of cells below 0.5 | 0.945 |
| Range of `g` | 0.224 – 0.557 |

The gate nudged toward the temporal branch — 94.5% of cells ended below 0.5 —
but stopped a long way short of the `g ≈ 0` that would recover `gru_only`'s
number. It moved about a quarter of the way from its init toward the useful end
and stalled.

---

## 6.2 Mechanism — a fixed 0.5 blend does as well as the learned gate

`gated` 18.36 against `gated_uniform` 18.53 — a 0.17 MAE gap against seed sds of
0.36 and 0.33. **Not established.** Freezing the gate at 0.5 and never training
it produces statistically the same headline number as learning it. The 25 gate
parameters demonstrably changed nothing, which the `gated_uniform` ablation was
included to test — the same role `no_lags_v0` played for the lag module.

---

## 6.3 Mechanism — the per-fold gate is unstable across seeds

Mean `g` per (fold, seed):

| Fold | seed 0 | seed 1 | seed 2 | spread |
|---|---|---|---|---|
| 1 | 0.367 | 0.377 | 0.386 | 0.019 |
| 2 | 0.495 | 0.458 | 0.499 | 0.041 |
| 3 | 0.383 | 0.371 | 0.396 | 0.025 |
| 4 | 0.408 | 0.369 | 0.341 | 0.067 |
| 5 | 0.364 | 0.363 | 0.447 | 0.084 |
| 6 | 0.422 | 0.428 | 0.334 | 0.094 |
| 7 | 0.380 | 0.361 | 0.341 | 0.039 |
| 8 | 0.313 | 0.348 | 0.354 | 0.041 |
| 9 | 0.296 | 0.334 | 0.328 | 0.038 |

On folds 5 and 6 the three seeds disagree by ~0.09 on the country-mean gate —
different random initialisations of the *backbone* settle the gate in different
places. A parameter driven by a weak, noisy gradient does not converge.

---

## 6.4 The diagnosis, and its connection to the lag module

The three observations above are the lag module's failure signature, from
[`docs/learnable_lags_results.md`](learnable_lags_results.md) §7:

> The encoder is trained jointly: its parameters only receive a meaningful
> gradient if *delayed* climate reduces the forecasting loss. [...] A parameter
> driven by a gradient carrying no information about delay drifts to the
> boundary of its range.

Here the parameter is the gate, and the gradient carries information about it
only if the spatial and temporal branches produce meaningfully different
forecasting losses. At horizon 1 the anchored residual target is dominated by
the previous period's case count — the same fact that made climate nearly
redundant for the lag module (dropping every climate channel cost +0.01 MAE on
recent normal years). So the loss barely distinguishes `spatial` from
`temporal`, the gate gradient is small and noisy, and the gate sits near its
init.

The one difference from the lag case: `gru_only` *does* visibly beat `gcn_gru`
at h = 1, so a signal distinguishing the branches provably exists in the test
metric. The gate still failed to exploit it — which says the signal is in the
*outcome* but not in the *per-batch training gradient* the gate learns from. A
sharper version of the same diagnosis, not a different one.

---

## 6.5 Per-district gate — the Jaffna finding

Mean `g` per district over all folds and seeds, sorted low to high
(`gated_gates.csv`):

| District | Mean g | Min | Max |
|---|---|---|---|
| Puttalam | 0.330 | 0.224 | 0.494 |
| Matale | 0.334 | 0.230 | 0.496 |
| Kegalle | 0.337 | 0.220 | 0.495 |
| Kurunegala | 0.340 | 0.244 | 0.501 |
| Trincomalee | 0.344 | 0.243 | 0.501 |
| Badulla | 0.353 | 0.250 | 0.502 |
| Polonnaruwa | 0.358 | 0.289 | 0.500 |
| Ratnapura | 0.360 | 0.277 | 0.499 |
| Anuradhapura | 0.360 | 0.264 | 0.496 |
| Ampara | 0.363 | 0.280 | 0.497 |
| Kalutara | 0.363 | 0.286 | 0.501 |
| Moneragala | 0.367 | 0.296 | 0.498 |
| Matara | 0.369 | 0.292 | 0.502 |
| Hambantota | 0.374 | 0.296 | 0.497 |
| Vavuniya | 0.379 | 0.262 | 0.499 |
| Kandy | 0.382 | 0.256 | 0.504 |
| Nuwara Eliya | 0.384 | 0.310 | 0.502 |
| Mannar | 0.385 | 0.316 | 0.500 |
| Colombo | 0.395 | 0.289 | 0.500 |
| Kilinochchi | 0.399 | 0.282 | 0.504 |
| Mullaitivu | 0.406 | 0.324 | 0.502 |
| Batticaloa | 0.413 | 0.336 | 0.498 |
| Galle | 0.420 | 0.358 | 0.500 |
| Gampaha | 0.456 | 0.413 | 0.499 |
| Jaffna | 0.530 | 0.494 | 0.557 |

The hypothesis was that the isolated northern districts would gate low. **The
pattern is the opposite.** Jaffna, the degree-1 node, has the **highest** mean
gate (0.530) and is the **only** district with a mean above 0.5 — it keeps the
*most* graph. Kilinochchi (0.399), Mullaitivu (0.406) and Mannar (0.385) all sit
above the 0.38 country mean. The districts gating lowest — Puttalam, Matale,
Kegalle, Kurunegala — are ordinary mid-degree districts. Gampaha (0.456) and
Colombo (0.395), the two highest-degree, highest-volume districts, also gate
above average.

**Reported as disconfirming the specific hypothesis, not as a confirmed
alternative** — see §7.

---

## 7. What this does NOT establish

- **One horizon.** Every number here is horizon 1. The diagnosis in §6.4
  predicts the gate has more to learn from at longer horizons, where the anchor
  carries less of the answer. `--horizon` exists on `scripts/21`; no h > 1 run
  has been done.
- **One feature variant.** `v1` only. The arms were not crossed with `v0`.
- **`gated` improving on `gcn_gru` is not a result.** It is 18.36 vs 19.00, and
  §6.2 shows a frozen 0.5 gate gets 18.53 — so the "improvement" over `gcn_gru`
  is a fixed half-and-half blend of two representations, not the learned gate
  doing work. Do not report `gated < gcn_gru` as a win.
- **The per-district gate ordering.** The gate moved too little to interpret its
  ordering: country-mean distance from init is only −0.12, and §6.3 shows the
  per-fold gate is seed-unstable. The 0.33–0.53 spread across districts may be as
  much initialisation noise as learned signal. Jaffna's position is the most
  stable in the table (min 0.494, always the highest), so *something* systematic
  is happening there, but a single weak run is not enough to build an
  interpretation on. A degree explanation in either direction — "Jaffna keeps
  its one link because it is worth retaining" vs. "the gate is near 0.5
  everywhere and Jaffna is a mild outlier for unrelated reasons" — is not
  supported at this evidence level.
- **District-level `gcn_gru` vs `gru_only`.** The prediction files
  (`gated_predictions.csv`, `baseline_predictions.csv`) have the per-district
  errors to check whether the graph specifically hurts the low-degree districts,
  but that analysis has not been run.
- **The entropy/sparsity penalty was not used.** Phase 5's risk note suggests
  one if gates collapse; they did not collapse, they failed to move, which the
  penalty would not fix. A penalty *pushing* the gate toward 0 or 1 is a
  different experiment.
- **`gcn_gru` / `gru_only` reproduce the committed baseline to within a seed
  draw, not exactly** — §5. `gru_only` is ~0.25 MAE low.
- **This does not prove no graph can help** — only that the contiguity graph,
  weighted per district by a jointly-trained gate at h = 1, does not. A better
  relation (Phase 5's `A_gaussian` is already built) or a longer horizon are
  both untested.

---

## 8. Risks and what confirmed them

Written before the run. The first risk is the one that occurred, and this table
reads as a prediction that came true rather than a formality — the mechanism was
anticipated from the lag module's post-mortem.

| Risk | Why it is the same trap as the lag module | What would confirm it | Outcome |
|---|---|---|---|
| **The gate has nothing to learn from.** At h = 1 the anchored target is dominated by the previous period's case count, so neither branch matters much to the loss. The gate gradient is near-zero and `gate_logit` barely leaves its init at 0.5. | The lag encoder failed because it was trained end-to-end from a forecasting loss whose gradient at h = 1 carried almost no information about the delay it was learning; the parameter drifted to its range boundary. A gate fed by a loss that does not distinguish the branches is the identical failure. | The learned gate ≈ `gated_uniform`; the gate's mean distance from its 0.5 init is small; per-fold gate values are unstable across seeds. | **Occurred.** Mean g 0.38, distance from init −0.12. `gated` 18.36 ≈ `gated_uniform` 18.53 (inside seed sd). Per-fold gate means vary across seeds by up to 0.09. |
| **Gate collapses to one extreme for all districts.** All-0 (→ `gru_only`) or all-1 (→ `gcn_gru`), with no per-district spread. | Same "drifts to boundary" behaviour as the lag kernels collapsing to their range edge. | Per-district gate values clustered at 0 or 1 with negligible variation. | **Did not occur.** The gate stayed near the middle — spread 0.33–0.53, never approaching either boundary. It did not collapse; it failed to move. |
| **Apparent win is really just added capacity or noise.** | The lag work needed `no_lags_v0` and `no_climate` to prove its result was not the longer input window. | Any gap between `gated` and `gated_uniform` larger than the seed sd, with the same 8,218-parameter architecture. | **Moot** — there was no win to explain. `gated` did not beat `gru_only`, and `gated` ≈ `gated_uniform`. |

The one reason to have expected this gate to work where the lag module did not:
`gru_only` *already visibly beats* `gcn_gru` at h = 1, so there *is* a measured
signal distinguishing the two branches. That the gate still failed to exploit it
is the informative part of the result.

---

## 9. Bottom line and next steps

**Item #8 is done as a negative result.** The per-district spatial/temporal gate
does not beat `gru_only`; learning the gate does statistically nothing over
freezing it at 0.5; and the `gru_only > gcn_gru` finding stands, now with a
mechanism — the contiguity graph is not salvageable by per-district weighting.
This also removes the remaining motivation for item #7 (the dual graph) in its
current framing: `docs/baseline.md` §4 already flagged that adding a gated
mixture on top of a harmful contiguity component "risks burying the problem",
and a gate that cannot even route around that component confirms it.

**The priority is now item #3 — the horizon sweep — and it is higher than
before.** Two independent architecture changes (item #6, the learnable lag
module; item #8, this gate) have now failed for the *same* reason: at horizon 1
the forecasting gradient carries almost no information about the structure being
learned, because the anchored target is dominated by the previous period's case
count. Before investing in any further architecture idea at h = 1 — a better
graph relation, a learned adjacency, a deeper fusion — the horizon sweep
(`lookback ∈ {12, 26}`, `horizon ∈ {1, 2, 3, 4}`) would establish whether this
failure mode is a property of the h = 1 task specifically. If the lag module and
the gate both start earning their keep at h = 3–4, that reframes every item in
the "research model" half of the work plan. If they do not, the conclusion is
that the ceiling at this resolution is the objective, not the architecture —
consistent with the only real traction so far coming from item #5's reweighted
loss on the epidemic fold.

Both `scripts/18` and `scripts/21` already take `--horizon`, so the sweep is a
runner over existing code, not new modelling.

---

## 10. Explaining this to my supervisor

*(Plain English, roughly one paragraph.)*

> The graph version of our model is slightly worse than the no-graph version —
> on every test fold, drawing district borders in as a graph and averaging
> neighbours together makes the forecast worse, not better. But that is an
> average over 25 very different districts. Colombo has six neighbours and sits
> on the main roads; Jaffna has one. So instead of the whole country using the
> graph or not, I gave each district a single dial the model learns for itself:
> 1 means "keep the graph", 0 means "ignore it", and it can sit anywhere
> between. If the model is right that the graph hurts, the isolated northern
> districts should turn their dial down. What happened is that the dials barely
> moved from the halfway point — the model could not find a strong reason to
> prefer either setting — and a fixed halfway blend does just as well as the
> learned dial, so the dial is not doing anything. This is the *same* failure we
> saw with the learnable climate-lag module: at one week ahead, the forecast is
> carried almost entirely by last week's case count, so there is very little
> signal left for either the lag shape or the graph dial to latch onto. That
> points at the next thing to check — whether this changes when we forecast
> further ahead — before we spend more time on architecture changes at one week.

**If asked "so was it a waste?"** — no. It confirms, with a mechanism, that the
contiguity graph is not salvageable by weighting it per district, which rules
out item #7 (the dual graph) in its current form and says the graph work needs
either a genuinely better relation than borders or a longer horizon. And it is
the second independent result pointing at the same root cause, which is what
makes the horizon sweep (item #3) the priority rather than a nice-to-have.

---

## 11. Files

| Path | Contents |
|---|---|
| `src/models/gated_fusion.py` | `GatedGCNGRU` — the per-district gate |
| `scripts/21.train_gated_gcn_gru.py` | the four arms and the runner |
| `tests/test_gated_fusion.py` | 11 tests, including both boundary conditions |
| `results/models/gated_report.md` | generated report |
| `results/models/gated_metrics.csv` | per arm, fold and seed |
| `results/models/gated_gates.csv` | per fold, seed and district gate value |
| `results/models/gated_predictions.csv` | per district-period predictions (unused analysis) |

## Related documents

- [`docs/baseline.md`](baseline.md) — the baseline, its open problems, and item #7's precondition
- [`docs/learnable_lags_results.md`](learnable_lags_results.md) — item #6, the first result with this mechanism
- [`docs/improvements.md`](improvements.md) — item #5, the reweighted objective
- [`docs/model_tensors.md`](model_tensors.md) — tensors, folds, adjacency

# Learnable Climate Lags — Step-by-Step Workplan

**Component A of the AEGIS-Dengue contribution.** Replace the fixed climate
aggregation windows (`v1`'s trailing 4, 8 and 12 period means) with a smooth,
learnable, **per-district, per-feature** delay kernel.

Owner: Hesandi · Target: 1 working week · Status: not started

---

## 0. Where we are starting from

| Thing | Current state | File |
|---|---|---|
| Features | `v0` = raw per-period. `v1` = `v0` + trailing 4/8/12-period means of rainfall, temperature, humidity | [12.build_model_tensors.py](scripts/12.build_model_tensors.py) |
| Model | GCN over 25 districts per step to a shared GRU to a linear head, lookback 12, residual (anchored) target | [16.train_gcn_gru.py](scripts/16.train_gcn_gru.py) |
| Best so far | `gru_only` on `v1`, headline MAE **9.15**; `gcn_gru` on `v1` MAE **9.98**; persistence **16.42** | [baseline_report.md](results/models/baseline_report.md) |

**Two facts that shape this phase.**

1. The windows `4, 8, 12` were chosen by hand. Nobody measured whether Colombo's
   rainfall-to-cases delay is the same as Jaffna's. That is the gap.
2. `gru_only` currently beats `gcn_gru`. The graph is not earning its place yet,
   so **do not touch the graph in this phase.** Change only the delay handling,
   keep everything else frozen, and the comparison stays clean.

**The number to beat:** the `v1` result above, and specifically `v1 − v0`, which
is what hand-picked lags are worth today.

---

## 1. The mechanism (one page, keep this pinned)

For district *i*, climate feature *k*, at time *t*, over a lag reach of `L = 26` weeks:

```
x̃[i,k,t]  =  sum over tau = 0..25 of  w[i,k](tau) * x[i,k,t-tau]
```

We do **not** learn `w` freely. That would be 26 × 7 × 25 = 4,550 free weights on
roughly 500 training windows — it would memorise the training years. Instead:

1. **B = 6 smooth basis kernels** over tau, each a Gaussian shape with a learnable
   centre `mu_b` and width `sigma_b`. Softplus the width, clamp the centre to
   `[0, 25]`, normalise each basis to sum to 1. Non-negative and causal by construction.

   *B = 6, not 4, and the count was settled by the Step 4 test rather than by
   taste.* A mixture can only interpolate between its bumps, never past the
   outermost one, and the centres contract towards wherever the data puts its
   mass instead of spreading to cover the reach. With 4 bumps the recovery test
   quantised — every district wanting a delay beyond the last centre was pinned
   to it, worst district 2.3 periods wrong. With 6: worst 1.7, mean 0.2.
2. **A district embedding** `e_i` in R^8, one row per node — this is what makes the
   lag district-specific.
3. **Mixture weights** `alpha[i,k] = softmax(e_i^T W_k)` over the B bases, so each
   (district, feature) pair picks its own blend.
4. **The kernel** `w[i,k](tau) = sum over b of alpha[i,k,b] * basis_b(tau)`.

**Parameter budget**

| Piece | Count |
|---|---|
| Basis shapes (mu, sigma) × 6 | 12 |
| District embeddings 25 × 8 | 200 |
| Feature projections 7 × 8 × 6 | 336 |
| **Total** | **548** |

548 parameters buys a 26-week memory without unrolling 26 recurrent steps. That
is the efficiency argument for the paper, and it is also why this will not overfit.

**Which features get a kernel**

| Lagged (7) | Passed through untouched (7) |
|---|---|
| `rainfall_daily_mean_mm` | `cases_log1p` |
| `rainy_days_frac` | `doy_sin`, `doy_cos` |
| `temperature_mean_c` | `weather_observed`, `case_observed` |
| `diurnal_range_c` | `centroid_lat`, `centroid_lon` |
| `dewpoint_mean_c` | |
| `relative_humidity_mean` | |
| `wind_speed_mean` | |

Cases stay raw — the GRU already handles autoregression, and smoothing the target's
own history would blur the signal the anchored target depends on.

**How it bolts onto the existing model (this is the important bit)**

The encoder sits *in front of* the unchanged GCN+GRU as a depthwise causal
convolution. Feed the model a longer window:

```
input window = 12 (model lookback) + 26 (lag reach) - 1 = 37 periods
        |
        +-- climate channels -> causal conv with learned kernel -> 12 smoothed steps
        +-- non-climate       -> just slice the last 12 steps
        |
        v
   concatenate -> GCN -> GRU -> head      <- completely unchanged
```

So the architecture diff is one module, and the ablation is honest: same graph,
same GRU, same head, same loss, same folds. Only the delay handling moves.

**Cost of the longer window:** each fold loses about 25 extra early windows. Say so
in the report and confirm the fold window counts before and after.

---

## 2. Steps

### Step 1 — Measure the real lags first
**`scripts/17.lag_correlation_scan.py` produces `results/eda/lag_correlation.csv`**

For every district × climate feature, compute the cross-correlation between the
feature and `cases_log1p` at lags 0 to 25, **using training periods only** (respect
`fit_end_period` — no test year touches this).

Output one row per `node_id, feature, lag, correlation`, plus a summary of the peak
lag per district and feature.

This does two jobs: it gives sensible initial values for `mu_b`, and it is the
independent yardstick you check the learned kernels against at Step 7.

**Exit gate:** the CSV exists and the peak lags are physically plausible (rainfall
roughly 8–16 weeks, temperature shorter). If rainfall peaks at lag 0 everywhere,
stop and check the panel alignment before going further.

---

### Step 2 — Build the encoder module
**`src/models/lag_encoder.py`** (new package; add `src/__init__.py` and
`src/models/__init__.py`, and `sys.path.insert` from the training script the same
way the numbered scripts already load each other)

```python
class LearnableLagEncoder(nn.Module):
    """Smooth, per-district, per-feature causal delay kernels."""

    def __init__(self, n_nodes=25, n_features=7, n_basis=4,
                 lag_reach=26, embedding_dim=8, init_centres=None):
        ...
        self.centre_raw = nn.Parameter(...)   # clamped to [0, lag_reach - 1]
        self.width_raw  = nn.Parameter(...)   # softplus -> positive
        self.node_embedding = nn.Embedding(n_nodes, embedding_dim)
        self.feature_projection = nn.Parameter(   # [n_features, emb, n_basis]
            torch.randn(n_features, embedding_dim, n_basis) * 0.1)

    def kernels(self):
        """Return [n_nodes, n_features, lag_reach], each row summing to 1."""
        # basis  : gaussian over tau, normalised
        # alpha  : softmax(e_i^T W_k) over the basis dimension
        # w      : alpha @ basis

    def forward(self, x):
        """x [batch, 37, nodes, n_features] -> [batch, 12, nodes, n_features]."""
        # depthwise causal conv via F.conv1d(groups=nodes * n_features)
```

Rules to hold yourself to:

- kernels non-negative and summing to 1, always — assert it in the test
- causal: index only tau >= 0, never a future period
- expose `.kernels()` so Step 7 can dump them without re-deriving anything

**Exit gate:** shapes are right, kernels sum to 1, and gradients reach
`centre_raw`, `width_raw`, `node_embedding` and `feature_projection` (assert all
four `.grad` are non-`None` and non-zero after one backward pass).

---

### Step 3 — Initialise from Step 1, not from noise

Spread the six `mu_b` across the reach, `[0, 4, 8, 13, 18, 24]`, and nudge them
towards the measured peaks. Widths start around 3 weeks. Do not cluster them:
the mixture cannot reach past its outermost bump, so a clustered init makes part
of the lag range unrepresentable.

Good initialisation here is the difference between converging in 20 epochs and not
converging at all. Write the chosen values into the config, not into the code.

---

### Step 4 — The synthetic-data test (do not skip this)
**`tests/test_lag_encoder.py`**

Build a fake series where the answer is known:

1. Draw random climate `x[t, i, k]`.
2. Give each district a *different* true delay, e.g. node 0 to 4 weeks, node 12 to
   11 weeks, node 24 to 18 weeks.
3. Set `y[t, i] = x[t - delay_i, i, 0] + small noise`.
4. Train the encoder plus one linear layer for a few hundred steps.
5. **Assert the recovered peak lag is within ±2 weeks of the true delay for every node.**

Also test: kernels sum to 1; output shape is `[batch, 12, nodes, features]`; a
zero-lag input reproduces itself when the kernel collapses to tau=0; no gradient
flows from a future period (shift the input forward, the output must not change).

This is the single highest-value test in the whole component. It is also the figure
that sells the paper: *we plant known delays, the module finds them.*

**Exit gate:** the recovery test passes on three different random seeds.

---

### Step 5 — Wire it into training and measure
**`scripts/18.train_lag_gcn_gru.py`** — copy `16.train_gcn_gru.py`, change three things:

1. `lookback = 37` when the encoder is on (`model_lookback + lag_reach - 1`)
2. instantiate `LearnableLagEncoder` and call it before the graph layers
3. optionally a separate, higher learning rate for the kernel parameters
   (`mu`, `sigma`, embeddings) than for the rest of the network

Everything else — folds, masking, anchored target, seeds, early stopping, per-fold
refitted preprocessing — stays byte-for-byte the same. Reuse `build_fold_arrays`,
`masked_mse`, `predict` and `naive.evaluate` by importing script 16, exactly the way
16 imports 15.

Run on the same folds, the same 3 seeds, and report the delta per fold.

**Exit gate:** the run completes on all folds and writes
`results/models/lag_metrics.csv` and `results/models/lag_report.md`.

---

### Step 6 — The ablation table
**`results/models/ablation_lag.csv`**

Five rows, all on the same folds, seeds and masks:

| # | Variant | What it controls for |
|---|---|---|
| 1 | `v0`, no lag features at all | the floor |
| 2 | `v1`, hand-coded 4/8/12 means | **the number to beat** |
| 3 | `v0` + learnable lag encoder | **the contribution** |
| 4 | `v0` + free unconstrained 26-tap filter | proves the *smooth basis* is doing the work, not just the longer window |
| 5 | `v0` + single global kernel (no district embedding) | proves *per-district* is doing the work |

Rows 4 and 5 are the ones a reviewer will ask for. Row 4 should be *worse* than row 3
(it overfits); row 5 should sit between rows 2 and 3.

**Exit gate:** row 3 beats row 2 on headline MAE, and beats it by more at longer
horizons. If it only wins at h >= 4, that is fine — make it the headline story:
*the delay structure matters more as the forecast horizon grows.*

---

### Step 7 — Extract and sanity-check the learned kernels
**`results/models/learned_lags.csv`** — one row per `fold_id, node_id, feature, lag,
weight`, plus a derived `peak_lag` column.

Then the check that makes this interpretable rather than just accurate:

- plot learned peak lag against Step 1's measured peak lag, one point per district ×
  feature — they should correlate
- map the learned rainfall lag across the 25 districts — wet-zone and dry-zone
  districts should not look identical
- confirm no kernel has collapsed to tau=0 (see Risks)

These are the Phase 8 figures. Produce the CSV now so the figures are cheap later.

---

## 3. Config

Add to `configs/` — no hard-coded numbers in the model file.

```yaml
lag_encoder:
  enabled: true
  lag_reach: 26
  n_basis: 6
  embedding_dim: 8
  init_centres: [0, 4, 8, 13, 18, 24]   # spread across the reach
  init_width: 3.0
  kernel_learning_rate: 1e-2       # higher than the network's 3e-3
  entropy_penalty: 0.0             # raise if kernels collapse
  lagged_features:
    - rainfall_daily_mean_mm
    - rainy_days_frac
    - temperature_mean_c
    - diurnal_range_c
    - dewpoint_mean_c
    - relative_humidity_mean
    - wind_speed_mean
```

---

## 4. Deliverables checklist

- [ ] `scripts/17.lag_correlation_scan.py` and `results/eda/lag_correlation.csv`
- [ ] `src/models/lag_encoder.py`
- [ ] `tests/test_lag_encoder.py` (synthetic recovery, causality, normalisation)
- [ ] `scripts/18.train_lag_gcn_gru.py`
- [ ] `configs/lag_encoder.yaml`
- [ ] `results/models/lag_metrics.csv` and `lag_report.md`
- [ ] `results/models/ablation_lag.csv`
- [ ] `results/models/learned_lags.csv`
- [ ] `docs/learnable_lags_results.md` — what happened, in prose

---

## 5. Risks and what to do about them

| Risk | Symptom | Fix |
|---|---|---|
| **Kernels collapse to tau=0** | every learned peak lag is 0–1 weeks; the model ignores the delay | separate higher LR for mu; add a small entropy penalty on alpha; check mu is not stuck at a clamp boundary |
| **No improvement over `v1`** | headline MAE flat | check it at h = 2, 4, 8 before concluding — the win is expected to grow with horizon |
| **Longer window starves the folds** | far fewer training windows, unstable seeds | report window counts; if severe, cut `lag_reach` to 20 |
| **Improvement is really just "more history"** | row 4 of the ablation matches row 3 | that is the answer to the question, not a failure — report it honestly and reframe the contribution around the *interpretable* kernels |
| **`gcn_gru` still loses to `gru_only`** | expected at this stage | out of scope here; it is Phase 5's problem. Report the encoder delta on *both* backbones. |

---

## 6. Order of work

```
Day 1     Step 1   lag correlation scan
Day 2     Step 2   encoder module
Day 2     Step 3   initialise from the scan
Day 3     Step 4   synthetic recovery test          <- hard gate
Day 4     Step 5   wire into training, run folds
Day 5     Step 6   ablations
Day 5     Step 7   extract kernels and write up
```

---

## 7. Script for explaining this to my supervisor

*(Plain English, roughly two minutes. Say it in this order.)*

> **The problem.**
> Dengue doesn't follow the rain immediately. It rains, water collects, mosquitoes
> breed, they bite people, people get sick, and only then do they show up at a
> hospital. That whole chain takes something like two to three months. So when we
> predict dengue cases, we can't just look at this week's weather — we have to look
> back at the weather from a couple of months ago.
>
> **What everybody does, including our baseline.**
> The usual approach is to pick a delay by hand. The original paper shifted all the
> weather data by one fixed amount, for the whole country. In our baseline I did
> something similar: I added the average rainfall over the last 4 weeks, the last 8
> weeks, and the last 12 weeks, and let the model choose between them.
>
> **Why that's not good enough.**
> Those numbers — 4, 8, 12 — I chose them. Nobody measured them. And more
> importantly, they're the same for all 25 districts. But Colombo is a wet, dense,
> urban district and Jaffna is dry and spread out. There's no reason the delay
> between rain and dengue should be identical in both places. We're forcing a single
> assumption onto 25 very different districts.
>
> **What I'm building.**
> Instead of me picking the delay, the model learns it — separately for each district
> and each weather variable. Concretely, each district gets its own weighting curve
> over the last 26 weeks that says how much each past week matters. If Colombo's
> rainfall curve peaks at 10 weeks and Anuradhapura's peaks at 15, the model
> discovers that from the data rather than being told.
>
> **The trick that stops it overfitting.**
> If I let the model freely choose 26 weights per district per variable, that's over
> 4,000 numbers to learn from only about 500 training examples. It would just
> memorise the training years. So instead I build each curve out of four smooth
> bumps, and the model only learns where each bump sits, how wide it is, and how much
> of each bump each district uses. That's about 430 numbers instead of 4,000 — and it
> forces the curves to be smooth, which is what a real biological delay looks like
> anyway.
>
> **How I'll know it works.**
> Three checks, in order.
> First, a controlled test: I generate fake data where I *plant* a known delay —
> district A at 4 weeks, district B at 11 weeks — and check the model finds them
> back. If it can't recover a delay I put there myself, nothing else matters.
> Second, accuracy: same graph, same everything else, only the delay handling
> changes, run on the same walk-forward folds — does the error go down compared to my
> hand-picked 4/8/12 windows?
> Third, and this is the one I care about most: I pull the learned curves out and
> check them against the correlations I measured directly from the data, and I map
> them across the country. If the model has learned something real, the wet-zone
> districts should look different from the dry-zone ones.
>
> **What I expect.**
> Probably a modest accuracy gain at one week ahead, and a bigger gain further out —
> because the further ahead you predict, the more you're relying on the delayed
> weather signal rather than on last week's case count. And if the accuracy gain is
> small, the interpretability result still stands on its own: a per-district map of
> climate-to-dengue delay is useful to a public health team regardless.
>
> **What I'm deliberately not touching.**
> The graph. Right now the graph version of the model is actually slightly worse than
> the no-graph version, which is its own problem — but if I change two things at once
> I won't know which one did what. So the graph stays frozen for this phase.

**If asked "why 26 weeks?"** — it covers roughly six months, which comfortably
contains the full rain to breeding to infection to reporting chain, and it matches
the lookback the original paper's own correlation analysis used.

**If asked "why not just a CNN over the history?"** — that's ablation row 4. A free
filter can represent the same thing but has no reason to stay smooth or non-negative,
so it overfits and the learned weights aren't readable. We test it precisely so we
can say the smooth version is doing the work.

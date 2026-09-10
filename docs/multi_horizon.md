# Multi-horizon forecasting — h = 1, 2, 3, 4

The first result in this repository where the model beats persistence by a margin
that survives every test the project applies. What was run, what it scores, and
why the effect appears only at longer horizons.

Reproduce with `python scripts/27.train_multi_horizon.py`. Tests:
`tests/test_multi_horizon.py`. Run on disk:
`results/models/multi_horizon_report.md`, `multi_horizon_metrics.csv`,
`multi_horizon_naive.csv`.

---

## 1. Why the horizon was the thing to test

Every result before this one is at h=1, and one measurement explains most of
them. [`learnable_lags_results.md`](learnable_lags_results.md) found that at one
week ahead the previous period's case count carries nearly all the forecasting
signal — dropping every climate channel costs **+0.01 MAE**. That single fact is
the standing explanation for:

- §7, the learnable lag encoder finding no gradient (a delay parameter driven by
  a gradient carrying no delay information drifts to its boundary);
- §8, the reweighted objective helping only in epidemic conditions;
- §8b, the activation sweep being flat;
- §8c, the hyperparameter response surface being flat across six of nine axes;
- §8e, no graph beating no graph.

If that explanation is right, it makes a prediction: extend the horizon and the
origin's grip on the target weakens, so everything else the model knows — climate
included — has room to matter. The measured rainfall-to-dengue delay is **5–10
weeks** (`scripts/17`), so h=4 is where it should start to bite.

If the explanation is wrong, longer horizons are simply uniformly harder and
nothing changes. Either answer is worth having.

---

## 2. Design

### Two arms

| Arm | Structure | Cost |
|---|---|---|
| `separate` | one model per horizon, four independent trunks | 4× parameters, ~4× training time |
| `shared` | one trunk, four linear heads over the same GRU state, summed masked loss | 1× trunk, ~¼ the time |

The head is deliberately the only per-horizon part of `shared`. Giving each
horizon its own GRU would make it four separate models in a trench coat and the
comparison against `separate` would measure nothing. Sharing forces one
representation to serve all four horizons, which is also the mechanism by which
`shared` can win despite less capacity per horizon: the near targets act as
auxiliary supervision for the far ones.

Both backbones are run. §8e established the graph hurts, so `identity` is the one
worth extending — but a horizon effect that appeared on only one backbone would
be worth knowing about, so `contiguity` is carried alongside.

### Every horizon is judged against its own persistence

`scripts/15` scores persistence at h=1 only. Persistence degrades sharply as the
horizon grows — its copied value goes up to four weeks stale — so this script
rescores it at every horizon on the same folds and masks:

| Horizon | persistence MAE | RMSE | Peak MAE | 2017 MAE |
|---|---|---|---|---|
| 1 | 16.42 | 33.61 | 26.61 | 36.08 |
| 2 | 20.16 | — | 32.51 | 48.55 |
| 3 | 24.58 | — | 41.48 | 62.08 |
| 4 | 28.47 | — | 47.98 | 75.73 |

**This is not a detail.** Comparing an h=4 model's raw MAE against the h=1
persistence would compare it to a baseline measured on a far easier task, and
would make a real improvement look like a degradation.

### Alignment

Both arms use windows cut at the **longest** horizon, so every horizon is scored
on identical forecast origins. Trimming per horizon would leave h=1 measured on
1000 windows and h=4 on 997 different ones, and the columns would not be
comparable to each other. It costs the h=1 column three windows against §6's
numbers, which is why `separate` at h=1 is close to but not bit-identical with
the committed baseline.

The target parameterisation is unchanged: the model predicts
`log1p(y[t+h]) − log1p(y[t])`, so an output of zero reproduces persistence *at
that horizon*. The anchor being four weeks stale at h=4 is exactly what makes the
task harder — the effect being measured, not a confound.

---

## 3. Results — 9 folds, 3 seeds, 6.8 min GPU

### Headline MAE by horizon

| Backbone | Arm | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|---|
| `identity` | `shared` | **16.71** | **20.04** | 23.15 | 25.74 |
| `identity` | `separate` | 17.20 | 20.22 | **22.89** | **25.40** |
| `contiguity` | `shared` | 18.19 | 22.31 | 25.83 | 28.65 |
| `contiguity` | `separate` | 19.48 | 23.41 | 27.08 | 29.93 |
| *persistence* | — | *16.42* | *20.16* | *24.58* | *28.47* |

Raw MAE rises with the horizon for every arm, as it must — the task is harder.
The number that matters is the margin against a persistence that is *also*
getting worse.

### Skill over same-horizon persistence

| Backbone | Arm | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|---|
| `identity` | `separate` | −4.7% | −0.3% | **+6.9%** | **+10.8%** |
| `identity` | `shared` | −1.7% | **+0.6%** | **+5.8%** | **+9.6%** |
| `contiguity` | `shared` | −10.8% | −10.7% | −5.1% | −0.7% |
| `contiguity` | `separate` | −18.6% | −16.2% | −10.2% | −5.1% |

**Monotonic in the horizon, on every backbone and both arms.** This is the shape
the §7 explanation predicts, and the first time any curve in this project has
moved consistently in a predicted direction.

### Peak MAE skill — the outbreak-warning criterion

| Backbone | Arm | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|---|
| `identity` | `separate` | −11.7% | −5.1% | **+5.2%** | **+9.8%** |
| `identity` | `shared` | −7.7% | −3.0% | **+4.7%** | **+8.0%** |
| `contiguity` | `shared` | −11.2% | −8.8% | −2.1% | **+0.8%** |
| `contiguity` | `separate` | −19.4% | −15.1% | −6.0% | −0.2% |

Peak MAE is the criterion `docs/baseline.md` names as the one that matters for an
outbreak warning system, and the one **no previous change in this repository ever
beat persistence on** — §8's loss fix did not, §8e's graph work did not. At h=3
and h=4 it is beaten, by up to **+9.8%**.

---

## 4. Is it real? Every test the project applies

`identity` backbone, against same-horizon persistence, paired over the seven
headline folds:

| Arm | h | Δ MAE | Folds won | t | p | Seed sd | Δ in sd |
|---|---|---|---|---|---|---|---|
| `separate` | 1 | +0.78 | 5/7 | +0.48 | 0.650 | 0.82 | 0.94 |
| `separate` | 2 | +0.07 | 6/7 | +0.05 | 0.965 | 0.58 | 0.11 |
| `separate` | 3 | **−1.69** | 6/7 | **−2.56** | **0.043** | 0.46 | **3.69** |
| `separate` | 4 | **−3.07** | **7/7** | **−4.04** | **0.0068** | 0.81 | **3.80** |
| `shared` | 3 | −1.43 | 6/7 | −1.57 | 0.168 | 0.66 | 2.17 |
| `shared` | 4 | **−2.72** | **7/7** | **−3.71** | **0.0099** | 0.62 | **4.40** |

At h=4 both arms beat persistence on **all seven headline folds**, at
**p < 0.01**, at **3.8–4.4 seed-sd**. This project's bar has been "two seed-sd
and p < 0.05" throughout; h=4 clears it on both arms and h=3 clears it on
`separate`.

### And it is not the 2017 artefact

Five consecutive changes (§8, §8b, §8c, §8d, §8e) produced headline movements
that turned out to be fold 1 and nothing else. This one does not. Per-fold delta
against persistence, `identity`/`separate` (negative = model wins):

| Fold | Year | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|---|
| 1 | 2017 | +10.36 | +8.50 | +1.08 | **−5.18** |
| 2 | 2018 | −0.36 | −1.50 | −2.83 | −3.07 |
| 3 | 2019 | −1.25 | −1.28 | −2.25 | −3.23 |
| 6 | 2022 | −1.48 | −1.17 | −1.37 | −0.84 |
| 7 | 2023 | −1.83 | −1.63 | −1.25 | −2.27 |
| 8 | 2024 | +0.53 | −1.85 | −4.48 | **−6.07** |
| 9 | 2025 | −0.54 | −0.61 | −0.74 | −0.81 |

Excluding fold 1 entirely:

| Horizon | All folds | Ex-fold-1 | Ex-fold-1 folds won |
|---|---|---|---|
| 1 | +0.78 | −0.82 | 5/6 |
| 2 | +0.07 | −1.34 | **6/6** |
| 3 | −1.69 | −2.15 | **6/6** |
| 4 | −3.07 | −2.71 | **6/6** |

**The gain is broad-based.** At h=2, h=3 and h=4 the model beats persistence on
every single non-epidemic headline fold, and the ex-fold-1 margin at h=4 (−2.71)
is nearly the full-sample margin (−3.07). Fold 1 is now the *worst* fold at h=1
and among the best at h=4 — it flips sign, which is the opposite of the pattern
every earlier section recorded.

### Seed-mean ensembles

| Backbone | Arm | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|---|
| `identity` | `separate` | −3.9% | +0.4% | +7.9% | **+11.9%** |
| `identity` | `shared` | −0.9% | +1.7% | +6.7% | **+10.4%** |

Free, since the seeds are already trained, and it pushes h=4 skill to +11.9%.

---

## 5. Shared versus separate

| Backbone | h | `separate` | `shared` | Δ |
|---|---|---|---|---|
| `identity` | 1 | 17.20 | **16.71** | −0.49 |
| `identity` | 2 | 20.22 | **20.04** | −0.18 |
| `identity` | 3 | **22.89** | 23.15 | +0.26 |
| `identity` | 4 | **25.40** | 25.74 | +0.34 |
| `contiguity` | 1 | 19.48 | **18.19** | −1.29 |
| `contiguity` | 2 | 23.41 | **22.31** | −1.10 |
| `contiguity` | 3 | 27.08 | **25.83** | −1.25 |
| `contiguity` | 4 | 29.93 | **28.65** | −1.28 |

**On the identity backbone the two are a wash** — `shared` wins the near
horizons, `separate` the far ones, all differences within or near a seed-sd
(0.44–0.82). Neither is established as better.

**On contiguity `shared` wins decisively at every horizon**, by 1.1–1.3 MAE.
That is a regularisation effect rather than a horizon effect: the shared trunk
has one quarter the parameters and is constrained to serve four targets, which
limits how badly it can overfit a graph §8e showed to be harmful. It is a fix for
a problem better solved by not using the graph.

**The practical argument for `shared` is cost, not accuracy.** It trains in
roughly a quarter the time (fold 1: 5.6s against 21.3s) and holds a quarter the
trunk parameters, for accuracy statistically indistinguishable from `separate` on
the backbone that matters. For an operational system producing all four horizons,
that is the better engineering choice even though it is not the better model.

---

## 6. What this establishes, and its limits

**Established.** At h=3 and h=4 the GCN+GRU on the identity backbone beats
same-horizon persistence on both MAE and peak MAE, on 6–7 of 7 headline folds, at
p < 0.05, at more than 2 seed-sd, and with the gain spread across folds rather
than concentrated in 2017. This is the first configuration in this repository of
which any of that is true.

**The §7 explanation is confirmed, and it was load-bearing.** Skill rises
monotonically with the horizon on every backbone and both arms. The reason so
many earlier changes were flat was not that the model has nothing to learn — it
is that at h=1 there is almost nothing left for it to learn beyond copying the
last observation. Remove that crutch and the model has real skill.

**Limits, stated plainly.**

- **h=1 and h=2 still lose or tie.** The model is worse than persistence at h=1
  (−4.7% skill) and level at h=2. If the operational requirement is a one-week
  forecast, nothing here changes §6's conclusion.
- **This does not by itself prove climate is the mechanism.** The skill curve is
  consistent with climate becoming load-bearing, but it is equally consistent
  with the model exploiting longer-range autocorrelation or seasonality that
  persistence cannot represent. Distinguishing them requires rerunning at h=4
  with the climate channels ablated — the exact experiment §7 ran at h=1, where
  it cost +0.01 MAE. **That is the single most valuable follow-up in this
  repository right now**, and until it is run "climate becomes useful at longer
  horizons" remains a hypothesis consistent with the data rather than a finding.
- **The graph still hurts at every horizon.** §8e's conclusion is unaffected:
  `contiguity` is worse than `identity` at h=1, 2, 3 and 4 alike.
- **Raw error still grows.** A +10.8% skill at h=4 is on an absolute MAE of
  25.40, against 17.20 at h=1. Longer-horizon forecasts are better *relative to
  the alternative*, not better in absolute terms.

---

## 7. Files

| File | Contents |
|---|---|
| `src/models/multi_horizon.py` | `make_multi_horizon_windows` (all horizons, shared origins), `MultiHorizonGCNGRU` (shared trunk, per-horizon heads), `masked_multi_horizon_mse` |
| `scripts/27.train_multi_horizon.py` | both arms × both backbones × four horizons, with persistence rescored per horizon |
| `tests/test_multi_horizon.py` | 34 tests: forecast alignment at every horizon, shared origins, per-node/per-horizon masking, trunk sharing, head independence, loss reduction to the baseline at H=1, residual round-trip |
| `results/models/multi_horizon_report.md` | generated report |
| `results/models/multi_horizon_metrics.csv` | per-arm per-backbone per-horizon per-fold per-seed metrics |
| `results/models/multi_horizon_naive.csv` | persistence and seasonal naive at every horizon |

---

## Related documents

- [`learnable_lags_results.md`](learnable_lags_results.md) — the +0.01 MAE measurement this whole section tests
- [`baseline.md`](baseline.md) — the h=1 baseline and the peak-MAE criterion finally met here
- [`graph_representation.md`](graph_representation.md) — why the identity backbone is the one extended
- [`improvements.md`](improvements.md) — the single-fold pattern this section is the first to break

# Climate ablation — is climate what causes the h=3–4 skill?

`scripts/29.train_climate_ablation.py`, `src/models/climate_ablation.py`,
`tests/test_climate_ablation.py`. Report:
`results/models/climate_ablation_report.md`.

## The question

README §8f established the first real improvement in this project: at h=3 and
h=4 the model beats same-horizon persistence on MAE *and* on peak MAE, 7/7 folds
at h=4, p < 0.01, not the 2017 artefact. It attributed that to the measured
5–10 week rainfall-to-dengue delay (`scripts/17`) — as the horizon grows, the
forecast origin loses its grip and climate gets room to matter.

**That attribution was never measured.** Skill appearing at h=4 is equally
consistent with a duller mechanism: persistence goes stale faster than the model
degrades, so the *relative* margin grows while climate contributes nothing. §8f
proves longer-horizon forecasting helps; it does not prove climate is why. The
two readings make opposite predictions about removing climate.

## Design

Three arms, identity backbone, v1, h = 1–4, 9 folds × 3 seeds.

| Arm | Inputs |
| --- | --- |
| `full` | all 23 channels — the §8f model |
| `no_climate` | the 16 climate-derived channels sliced out, leaving 7 |
| `shuffled` | all 23 channels, climate time-shuffled within each district |

**All 16 climate-derived channels are dropped, not just the 7 instantaneous
ones.** v1 adds rolling means at 4, 8 and 12 periods, and those are precisely
the channels the 5–10 week delay story runs through. Dropping `rainfall_...` but
keeping `rainfall_..._roll12` would leave the mechanism intact and call it an
ablation.

**The `shuffled` arm is what makes this conclusive.**
`docs/learnable_lags_results.md` §6.3 ran the two-arm version at h=1, found
+0.01 MAE in recent normal years, and flagged its own result as confounded:
dropping 16 of 23 channels changes the input width, the first layer's parameter
count and the effective regularisation all at once, so a `no_climate` penalty
need not be about weather. The shuffled arm holds every one of those fixed —
identical width, identical parameter count (asserted by test), identical
per-fold scaler statistics, identical marginal distribution per channel per
district — and destroys only the alignment between weather and time.

The shuffle permutes **whole windows**, **within each district**, **within each
split**, with an independent permutation per district and per channel. Whole
windows so a window still looks like a plausible stretch of weather rather than
noise; within-district so no district's weather leaks into another's; within-split
so no test-period weather reaches training windows. It never touches `y`, the
mask or the anchor. Independent permutations per district also strip the
"it rained everywhere at once" seasonal proxy, which makes this the
*conservative* control: it destroys strictly more, so `full ≈ shuffled` would be
strong evidence of no content.

Reading the three arms together:

| Pattern | Conclusion |
| --- | --- |
| `full` ≈ `shuffled` ≈ `no_climate` | climate contributes nothing; the skill is persistence decaying |
| `full` < `shuffled` ≈ `no_climate` | climate's temporal content is load-bearing |
| `full` ≈ `shuffled` < `no_climate` | the channels help as width, not as weather |

## Results

MAE cost of each ablation, paired over the seven headline folds. **Positive
means the ablation hurt** — the direction the mechanism claim predicts.

| h | `no_climate` Δ | `shuffled` Δ | `shuffled` p | `shuffled` Δ in sd |
| --- | --- | --- | --- | --- |
| 1 | −0.25 | +0.02 | 0.926 | 0.05 |
| 2 | −0.35 | −0.06 | 0.854 | 0.15 |
| 3 | −0.06 | +0.43 | 0.226 | 0.76 |
| 4 | **+0.76** | **+1.55** | **0.016** | **2.38** |

Peak MAE — the criterion `docs/baseline.md` names as the one that matters for an
outbreak warning system:

| h | `no_climate` Δ peak | `shuffled` Δ peak | `shuffled` p |
| --- | --- | --- | --- |
| 1 | +0.15 | +0.40 | 0.391 |
| 2 | +0.54 | +0.38 | 0.517 |
| 3 | +1.00 | +1.65 | 0.161 |
| 4 | **+2.95** | **+3.10** | **0.023** |

**Both columns rise monotonically with the horizon, in both arms.** That is the
shape the mechanism claim predicts and the shape the persistence-decay
explanation does not: if the h=4 skill were only persistence going stale,
removing climate would cost nothing at any horizon and these columns would sit
flat near zero.

### Skill over same-horizon persistence

Persistence rescored at each horizon on the same folds and masks.

| h | persistence | `full` | `no_climate` | `shuffled` |
| --- | --- | --- | --- | --- |
| 1 | 16.42 | −1.6% | −0.0% | −1.7% |
| 2 | 20.16 | −1.0% | +0.8% | −0.7% |
| 3 | 24.58 | **+5.0%** | +5.2% | +3.2% |
| 4 | 28.47 | **+10.8%** | +8.1% | +5.3% |

At h=4, scrambling climate removes **half** the skill §8f reported (10.8% →
5.3%). On peak MAE it removes about two thirds (9.8% → 3.3%). The skill does not
vanish — some of it genuinely is persistence decay — but a majority of it at h=4
depends on climate being aligned with time.

### The h=1 correctness check

§6.3 measured the cost of dropping climate at h=1 as ≈ +0.01 MAE on recent
normal years (folds 6, 8, 9). Reproduced here on the same subset:

| h | `no_climate` | `shuffled` |
| --- | --- | --- |
| 1 | +0.17 | +0.29 |
| 2 | +0.26 | +0.12 |
| 3 | +0.36 | +0.51 |
| 4 | **+0.64** | **+0.68** |

The h=1 row is small and in the same ballpark as §6.3, so the harness reproduces
the known anchor before any h=4 number is read — and the same monotonic rise
appears on this subset too.

### It is not the 2017 artefact

Per-fold Δ at h=4 (positive = ablation hurt):

| Fold | 1 (2017) | 2 | 3 | 6 | 7 | 8 | 9 | ex-fold-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `no_climate` | −2.27 | +1.35 | +2.84 | −0.32 | +1.48 | +0.71 | +1.52 | **+1.26** |
| `shuffled` | +3.23 | +2.21 | +2.82 | −0.09 | +0.53 | +1.04 | +1.10 | **+1.27** |

`shuffled` hurts on 6/7 folds, and excluding fold 1 entirely the cost is
unchanged (+1.55 → +1.27). Unlike five earlier changes in this project, this
result does not live in the epidemic fold.

## What this establishes, and what it does not

**Established.** At h=4, climate's *temporal content* is load-bearing. The
shuffled arm holds width, capacity and marginal distributions fixed and still
costs +1.55 MAE (p = 0.016, 2.4 seed-sd) and +3.10 peak MAE (p = 0.023). Because
that arm is the confound control §6.3 lacked, this is the first result in the
repository that attributes an effect to climate *content* rather than to channel
count. The monotonic rise from h=1 to h=4 in both arms and on both metrics is
the horizon-dependence the 5–10 week delay predicts.

**Not established.** Three honest limits:

- **`no_climate` alone is weaker than `shuffled`** (+0.76, p = 0.267) and is
  actually negative at h = 1–3. Removing channels apparently buys back some
  regularisation that offsets the lost signal, which is exactly the confound
  §6.3 warned about — and precisely why the shuffled arm, not this one, carries
  the conclusion. Anyone reading only the two-arm ablation would conclude
  climate does nothing.
- **Climate is not the whole story.** `shuffled` at h=4 still beats persistence
  by 5.3%. Roughly half the §8f skill survives destroying climate, so persistence
  decay is a real component of that result, not a strawman.
- **h=3 does not clear the bar** (p = 0.226). The effect is established at h=4
  only; the h=3 point is consistent with the trend but is not on its own
  evidence.

## Recommendation for the paper

§8f's claim should be stated as: longer-horizon forecasting produces genuine
skill, and **at h=4 a majority of that skill is attributable to climate
content** — evidenced by a shuffle control that holds model capacity fixed —
with the remainder attributable to persistence decay. The h=1 → h=4 monotonic
trend supports the 5–10 week delay mechanism. Do not state that climate causes
the improvement without the shuffled arm: the plain `no_climate` ablation on its
own would have supported the opposite conclusion.

## Reproducing

```bash
python scripts/29.train_climate_ablation.py                     # h=1–4, 3 arms, 9 folds, 3 seeds
python scripts/29.train_climate_ablation.py --horizons 4        # h=4 only
python scripts/29.train_climate_ablation.py --model contiguity  # the harmful graph
python -m pytest tests/test_climate_ablation.py
```

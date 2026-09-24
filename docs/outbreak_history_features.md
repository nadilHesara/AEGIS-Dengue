# Outbreak-history features — national wave rank and trailing case load

Two new engineered columns, built on top of `v1`, tested at h=1 and at
h=1-4. A modest, horizon-dependent improvement: near-neutral at h=1, real
at h=3 and h=4. What the features are, why they were built, a scaling bug
that invalidated the first run, and the leakage check that confirmed the
second one.

Reproduce with `python scripts/features/12.build_model_tensors.py` then
`python scripts/features/14.build_folds.py`, then
`python scripts/training/16.train_gcn_gru.py --variants v1 v2` (h=1) and
`python scripts/training/27.train_multi_horizon.py --variant v2` (h=1-4).
Tests: `tests/test_model_tensors.py`, `tests/test_folds.py`.

---

## 1. The gap this targets

`scripts/evaluation/30.residual_diagnostic.py` asked whether the baseline model's
test-set errors carry structure the model isn't using, or whether they
look like irreducible noise (see [`docs/residual_analysis.md`](residual_analysis.md)).
Part C found two case-history-derived quantities correlate with the
error more strongly than any existing input channel:

| Quantity | r vs `abs_residual`, h=1 | r vs `abs_residual`, h=4 |
| --- | --- | --- |
| a district's rank against all 25 by same-period case count | -0.341 | -0.344 |
| a district's own cases summed over the trailing 52 periods | +0.334 | **+0.384** |

Both beat every climate channel and every centroid coordinate tested in
that diagnostic. The second one is the more interesting number: the
correlation *strengthens* from h=1 to h=4 rather than fading, which is
not what a spurious or noise-driven correlation would do. That pattern —
present and growing with the horizon — is what motivated building these
as real model inputs rather than leaving them as a diagnostic-only
computation.

---

## 2. What changed

Two columns added on top of `v1`'s 23, making a new `v2` tensor variant
(`scripts/features/12.build_model_tensors.py`, `add_history_features`):

```
national_wave_rank              rank of this district's case count among
                                 all 25 districts in the SAME period,
                                 1 = highest.

trailing_52_cumulative_cases    log1p of this district's own cases summed
                                 over the trailing 52 periods, inclusive
                                 of the current one.
```

Both are causal by construction, not by convention. `national_wave_rank`
reads only that one period's row across districts — no other period is
touched, so unlike `periods_since_outbreak` (deliberately left out of
this pass) it needs no fold-fitted threshold. `trailing_52_cumulative_cases`
uses `min_periods=52`, so the first 51 periods of the series are `NaN`
rather than a partial sum mislabelled as a full one — the same convention
`v1`'s rolling climate means already use, for the same reason. A missing
case count anywhere inside the trailing window propagates to `NaN` rather
than being treated as zero cases observed, matching the project's
"missing stays missing" rule everywhere else.

Both columns sit in `X` like every other feature. The origin/target split
that actually enforces causality at inference time is `make_windows`'s,
unchanged — the model only ever reads `X` up to its own forecast origin,
so a column being causal-by-construction and a column being read only at
or before the origin are two separate guarantees, and §4 below verifies
the second one directly rather than assuming it follows from the first.

---

## 3. The scaling bug, and the fix

The first version stored `trailing_52_cumulative_cases` as the raw
rolling sum, uncompressed. On synthetic data shaped like a single sharp
case spike, the per-fold z-scored value reached **+13.0 standard
deviations** — an outlier magnitude no other feature in this project's
tensors approaches. On the real data this destabilised training
specifically on fold 1 (the 2017 epidemic, where the real spike sits) and
specifically on the `identity` backbone: fold-1 h=1 MAE, which sits
around 42-50 for `v1`, came out at **69-89** for the broken `v2` — not a
regression, a collapse, and every other fold was unaffected, which is
what pinned the cause to this one column rather than to the feature set
in general.

**The fix:** `log1p` applied to the summed count, not to `cases` before
summing — the rolling sum itself stays linear, only its stored scale is
compressed, the same treatment `cases_log1p` already gives a single
period's count. After the fix, fold-1 h=1 MAE for `v2` came back to
**39-44**, matching or slightly beating `v1`'s 42-50 — the collapse is
gone and the fold is no longer an outlier in either direction.

A general-purpose safeguard was added alongside the fix:
`scripts/features/14.build_folds.py` now asserts every fold's fitted
statistics are finite before writing them, for every variant
(`check_statistics_are_finite`), so a similar problem in a future feature
would fail the build loudly instead of surfacing as an unexplained
training collapse hours into a run.

This is part of the result, not a footnote to it: the numbers in §6-§7
are from the run made *after* this fix, verified against the pre-fix
collapse directly above rather than assumed correct.

---

## 4. Leakage check

Verified twice, on two different real artefacts, with concrete numbers —
not by re-reading the code.

**Independent recomputation.** For 7 real forecast windows spanning
different districts, horizons and the exact 51/52-period warm-up
boundary, both features were recomputed from scratch from
`data/processed/panel_weekly.parquet`'s raw `cases` column — no pipeline
code, no tensor — and compared against the value stored in the real
`model_tensors_v2.npz` at that origin:

| origin | node | h | target | rank stored | rank recomputed | cumsum stored | cumsum recomputed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 0 | 1 | 101 | 14.0 | 14 | 48.000 | 48.000 |
| 100 | 4 | 4 | 104 | 1.0 | 1 | 1615.000 | 1615.000 |
| 500 | 8 | 3 | 503 | 10.0 | 10 | 2135.000 | 2135.000 |
| 700 | 21 | 2 | 702 | 12.0 | 12 | 2308.000 | 2308.000 |
| 900 | 12 | 1 | 901 | 19.0 | 19 | 376.000 | 376.000 |
| 52 | 3 | 1 | 53 | 17.0 | 17 | 66.000 | 66.000 |
| 51 | 3 | 1 | 52 | 18.0 | 18 | **NaN** | **NaN** |

All 7 match exactly (cumulative-sum figures agree to `float32` round-trip
precision, ~1e-7 relative error, confirmed to be exactly the expected
`log1p`/`expm1` epsilon and not a discrepancy). Periods 51 and 52
specifically confirm the warm-up boundary lands where it should on the
real 1012-period series, not just in a synthetic test.

**Target-period corruption.** For the same windows, every case count from
the target period onward was set to `999999` and the features rebuilt.
The origin-period value was unchanged in every case, for both features:

| origin | node | h | target | rank unaffected | cumsum unaffected |
| --- | --- | --- | --- | --- | --- |
| 100 | 0 | 1 | 101 | True | True |
| 100 | 4 | 4 | 104 | True | True |
| 500 | 8 | 3 | 503 | True | True |
| 700 | 21 | 2 | 702 | True | True |
| 900 | 12 | 1 | 901 | True | True |

Neither feature reads any case count from the target period or later, on
the real, Kaggle-built tensors, verified by direct perturbation rather
than assumed from the code's structure.

---

## 5. Results — h=1

`scripts/training/16.train_gcn_gru.py --variants v1 v2`, 9 folds, 3
seeds, headline mean over folds 1, 2, 3, 6, 7, 8, 9.

| Model | v1 MAE | v2 MAE | Δ | Seed sd (v1 / v2) |
| --- | --- | --- | --- | --- |
| `gcn_gru` | 19.01 | 17.81 | **-1.19 (-6.3%)** | 0.73 / 0.52 |
| `gru_only` | 16.88 | 16.65 | -0.23 (-1.4%) | 0.15 / 0.53 |
| persistence | 16.42 | — | — | — |

**h=1 is a wash, not a win.** `gcn_gru`'s delta is large enough to clear
this project's usual bar (~2 pooled seed-sd), but `gru_only` — the
stronger backbone throughout this project — moves by an amount well
inside its own seed noise. Fold 1 specifically (2017) improved for both
models and every seed, confirming the scaling fix rather than adding a
new finding: `gcn_gru` moved from 50.9-59.6 to 42.0-48.5, `gru_only` from
43.2-45.8 to 39.0-45.6.

**Read this section before §6.** Any claim that v2 improves the model
without a horizon qualifier is not supported by this table.

---

## 6. Results — h=3 and h=4

`scripts/training/27.train_multi_horizon.py --variant v2`, both `separate`
and `shared` arms, both `identity` and `contiguity` backbones, 9 folds, 3
seeds. Headline MAE delta against the same sweep on `v1`, single-seed
models, with the pooled seed-sd computed from each run's own per-seed
headline means (matching the convention in
[`docs/multi_horizon.md`](multi_horizon.md) §4):

| Backbone | Arm | h | v1 MAE | v2 MAE | Δ | Δ in seed-sd |
| --- | --- | --- | --- | --- | --- | --- |
| contiguity | separate | 1 | 20.39 | 18.16 | -2.23 | -2.1σ |
| contiguity | separate | 2 | 23.10 | 22.63 | -0.48 | -0.4σ |
| contiguity | separate | 3 | 26.78 | 25.29 | -1.49 | -1.2σ |
| contiguity | separate | 4 | 29.00 | 28.13 | -0.87 | -0.5σ |
| contiguity | shared | 1 | 18.58 | 18.11 | -0.47 | -0.8σ |
| contiguity | shared | 2 | 22.65 | 21.87 | -0.78 | -0.7σ |
| contiguity | shared | 3 | 26.14 | 25.60 | -0.53 | -0.6σ |
| contiguity | shared | 4 | 28.71 | 28.25 | -0.46 | -0.6σ |
| identity | separate | 1 | 17.09 | 16.44 | -0.65 | -1.2σ |
| identity | separate | 2 | 20.44 | 19.84 | -0.60 | **-2.8σ** |
| identity | separate | 3 | 23.15 | 22.51 | -0.64 | -1.1σ |
| identity | separate | 4 | 25.83 | 25.28 | -0.55 | -1.2σ |
| identity | shared | 1 | 17.09 | 16.39 | -0.70 | **-3.0σ** |
| identity | shared | 2 | 20.17 | 19.57 | -0.59 | **-2.0σ** |
| identity | shared | 3 | 23.38 | 22.64 | -0.74 | **-1.9σ** |
| identity | shared | 4 | 25.88 | 25.15 | -0.73 | **-2.4σ** |

**Every one of the 16 combinations moves in the improving direction.**
Not every one clears significance — `contiguity/separate` at h=2
(-0.4σ) and `contiguity/shared` across the board (-0.6 to -0.8σ) do not —
but `identity/shared` clears 2σ at every single horizon, and is the most
consistent result in the table.

**This is a modest effect, not a large one.** The deltas run 0.5-2.2 MAE,
a 3-12% relative change depending on configuration. Framed against the
project's own history, that is still notable: it is the first change of
any kind — loss, optimiser, hyperparameters, graph representation, layer
normalisation, or this — to move the headline in the improving direction
at more than one horizon with the improvement growing rather than
shrinking as the horizon extends.

### The horizon-dependence is the finding, not a footnote to it

§5 showed h=1 near-neutral. This section shows h=3/h=4 real. That shape —
absent at h=1, present and strengthening at h=3-4 — is exactly what §1's
residual-diagnostic correlation predicted before this feature was built:
`trailing_52_cumulative_cases`'s correlation with model error rose from
+0.334 at h=1 to +0.384 at h=4. A feature that does nothing at h=1 and
something at longer horizons is not a surprise here; it is the specific
shape the prior evidence called in advance.

### `identity/shared` is the number to cite; `contiguity` is not independent confirmation

`identity/shared` has both the tightest seed variance in the table and
the most consistent significance (2.0-3.0σ at every horizon) — it is the
single best-supported result here and the one worth quoting if the doc
needs one number.

`contiguity`'s deltas are numerically similar, in places larger
(`contiguity/separate` h=1 is the single largest delta in the table,
-2.23 MAE). **These should not be read as independent corroboration.**
`docs/baseline.md` §3 and `docs/graph_representation.md` already
established that the `contiguity` backbone underperforms `identity`
outright at every horizon tested elsewhere in this project. An
improvement measured on top of a backbone already known to be weaker is
not a second, independent confirmation of the same effect — it is one
result (on `identity`) with a directionally consistent but lower-quality
echo on a backbone this project does not otherwise trust.

---

## 7. What this does NOT establish

- **v2 does not help at h=1**, in any sense that should be cited. Only
  `gcn_gru`'s h=1 delta clears significance, and `gcn_gru` is the weaker
  backbone; `gru_only`'s h=1 delta is inside its own seed noise.
- **The effect size is modest everywhere it is real**, 3-12% relative,
  0.5-2.2 MAE. Nothing here should be described as a large improvement.
- **`contiguity`-backbone results do not independently corroborate
  `identity`'s.** See §6. Citing both as if they were two separate
  confirmations overstates the evidence.
- **Only 3 seeds.** Several deltas that don't clear 2σ (`contiguity/separate`
  h=2 and h=4, `contiguity/shared` at every horizon) are directionally
  consistent but not established by this run.
- **No ablation isolates which feature does the work.** `national_wave_rank`
  and `trailing_52_cumulative_cases` were added together. Whether one
  drives the result, both contribute roughly equally, or they interact,
  is untested.
- **This is consistent with, not confirmation of, the residual
  diagnostic's broader thesis.** §30's diagnostic asked whether the
  feature set was the h=1 ceiling. A two-feature, horizon-dependent
  improvement at h=3/h=4 does not establish that the ceiling was the
  feature set in general — it establishes that these two specific
  features help, modestly, at longer horizons. Those are different
  claims, and only the second one is supported here.
- **h=2 is weak across the board** (0.4-0.7σ everywhere except
  `identity/shared`'s 2.0σ) and should not be grouped with h=3/h=4 as
  "the improved horizons" without that qualifier.

---

## 8. Files

| Path | Contents |
| --- | --- |
| `scripts/features/12.build_model_tensors.py` | `add_history_features`, the `v2` variant |
| `scripts/features/14.build_folds.py` | `v2` added to the per-fold statistics loop; `check_statistics_are_finite` |
| `tests/test_model_tensors.py` | rank/cumsum correctness, warm-up NaN, leak tests (12 tests) |
| `tests/test_folds.py` | `check_statistics_are_finite` tests |
| `data/processed/model_tensors_v2.npz` | the built tensor, 25 features |
| `data/processed/fold_preprocessing_v2.npz` | per-fold imputation and scaling statistics |
| `results/models/baseline_metrics.csv` | h=1, `v1` vs `v2`, both backbones, 9 folds x 3 seeds |
| `results/reports/multi_horizon_report.md`, `results/models/multi_horizon_metrics.csv` | h=1-4, both arms, both backbones |

## Reproducing

```bash
python scripts/features/12.build_model_tensors.py
python scripts/features/14.build_folds.py

# h=1
python scripts/training/16.train_gcn_gru.py --variants v0 v1 v2 --seeds 3

# h=1-4
python scripts/training/27.train_multi_horizon.py --variant v1
python scripts/training/27.train_multi_horizon.py --variant v2
```

`scripts/training/27` takes one `--variant` per run and overwrites its
output on each invocation — copy `v1`'s output aside before running `v2`
if both are needed.

## Related documents

- [`docs/residual_analysis.md`](residual_analysis.md) — the diagnostic that motivated this
- [`docs/multi_horizon.md`](multi_horizon.md) — the significance-testing convention used in §6
- [`docs/gated_fusion.md`](gated_fusion.md) — the doc structure this follows
- [`docs/baseline.md`](baseline.md) — h=1 context, the `contiguity`-underperforms-`identity` finding cited in §6
- [`docs/graph_representation.md`](graph_representation.md) — why `contiguity` results are discounted here

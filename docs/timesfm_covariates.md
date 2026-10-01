# Experiment B — TimesFM + future-known covariates

Experiment B compares Google's TimesFM 2.5 zero-shot model with the same model augmented by its built-in in-context XReg component. It is separate from Experiment A and does not alter, train, or fine-tune any AEGIS-Dengue model.

## Frozen, leakage-safe design

- **Arm:** `timesfm2p5_calendar_xreg`.
- **Target and history:** the same district-level canonical dengue case history and walk-forward benchmark cells as Experiment A.
- **Covariates:** `season_sin` and `season_cos`, derived from reporting-period start dates. They are calendar facts known for every future target week.
- **Not included:** observed or future weather, climate forecasts, district identifiers, province, neighbour cases, spatial graph inputs, or future dengue values.
- **XReg mode:** `xreg + timesfm`. The TimesFM API fits an in-context regression to the origin-available case history and passes its residual series to the frozen foundation model. This is not fine-tuning or a cross-fold fitted model.
- **Missing case values:** forward-filled within each context only; a missing value is replaced by the most recent earlier observation (or zero at the start). No future value is used.
- **Intervals:** native q10–q90 output is deterministically mapped to the benchmark quantile columns using Experiment A's documented interpolation/extrapolation rule.

This design deliberately avoids climate because the local Open-Meteo cache is incomplete and future observed weather would leak information. A later climate experiment should use a complete, versioned climate panel and a prospectively available weather forecast or a history-only, causal climate transformation—not actual target-week weather.

## Run

Install the optional XReg dependencies once. This explicitly installs the Windows-compatible CPU JAX runtime and scikit-learn; XReg is kept on CPU while TimesFM can still use the GPU.

```powershell
.venv\Scripts\python.exe -m pip install --upgrade -r requirements-timesfm.txt
```

Validate the local dengue inputs without loading a model:

```powershell
.venv\Scripts\python.exe scripts\training\47.timesfm_covariates.py --dry-run
```

Run a small pilot before the full experiment:

```powershell
.venv\Scripts\python.exe scripts\training\47.timesfm_covariates.py --folds 0 --horizons 1 4 12 --suffix _pilot --no-holdout
```

Then run the complete arm:

```powershell
.venv\Scripts\python.exe scripts\training\47.timesfm_covariates.py
```

The resulting prediction file is `results/benchmark/predictions/timesfm2p5_calendar_xreg.parquet`. The existing full scorer can include it after the complete climate-based benchmark tensors are available.

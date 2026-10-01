# Experiment A — Google TimesFM 2.5 zero-shot forecasting

## Purpose

This experiment adds Google's pretrained TimesFM 2.5 (`google/timesfm-2.5-200m-pytorch`) as an **outside, univariate zero-shot comparator** for the AEGIS-Dengue long-horizon benchmark. It answers whether a generic foundation model, without dengue-specific training, can forecast district case trajectories from their own past values.

It is not a replacement for the project's current Negative Binomial model and it does not alter that model, its data preparation, or its benchmark results.

## Frozen protocol

- **Arm:** `timesfm2p5_zero_shot`.
- **Target:** weekly district dengue cases from `data/interim/dengue_weekly_canonical.parquet`. TimesFM deliberately does not load the climate-based `v2` tensors.
- **Input at each forecast origin:** only that district's observed case history through that origin. Missing case values remain missing; TimesFM handles them internally.
- **Transform:** `log1p(cases)` before inference and `expm1` afterwards. This matches the existing zero-shot foundation-model comparison and makes the count scale more stable without fitting any data-dependent transform.
- **No fitting or tuning:** no fine-tuning, validation-selected settings, climate variables, calendar variables, district labels, neighbour values, or future covariates.
- **Context:** the most recent 512 reporting weeks, with TimesFM's per-series input normalization enabled. This limit is fixed in `configs/timesfm_zero_shot.toml` and is available at every origin.
- **Horizons and cells:** the benchmark horizons (1, 2, 3, 4, 6, 8, 10, 12 weeks), walk-forward folds, validation/test assignment by target week, and `cell_frame` output contract are exactly those in `src/evaluation/long_horizon.py`.
- **Intervals:** TimesFM 3.0.2 exposes a fixed native grid of 10% through 90%. The median is used directly; 25% and 75% are linearly interpolated and 2.5%/97.5% are linearly extrapolated from the nearest native grid segment, all on the log1p scale. This deterministic mapping is necessary to meet the benchmark's fixed quantile-column contract and is recorded here rather than presented as native TimesFM tail coverage.

The only information a prediction for target `t` at horizon `h` reads is the case history through `t - h`; the script builds a separate input slice for every origin to enforce this.

## Install and run

From the repository root, install the optional dependency once. The first non-dry run downloads the roughly 800 MB model weights to the Hugging Face cache. Experiment A does **not** require the Open-Meteo climate reconstruction.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-timesfm.txt
```

First confirm the prepared dengue data and requested cells without downloading or loading the model:

```powershell
.venv\Scripts\python.exe scripts\training\46.timesfm_zero_shot.py --dry-run
```

If this checkout has no ignored data directory, prepare only the dengue inputs first (these commands do not make climate requests):

```powershell
.venv\Scripts\python.exe scripts\data\0.load_dataset.py
.venv\Scripts\python.exe scripts\data\2.create_calendar.py
.venv\Scripts\python.exe scripts\data\4.create_canonical_dengue.py
```

If `data/processed/nodes.csv` is absent, run `scripts/data/3.create_nodes.py` before the canonical-dengue command.

Run the full declared zero-shot experiment:

```powershell
.venv\Scripts\python.exe scripts\training\46.timesfm_zero_shot.py
```

For a quick non-headline smoke run, keep outputs separate with a suffix:

```powershell
.venv\Scripts\python.exe scripts\training\46.timesfm_zero_shot.py --folds 0 --horizons 1 4 12 --suffix _pilot --no-holdout
```

The full run writes `results/benchmark/predictions/timesfm2p5_zero_shot.parquet` and `results/benchmark/compute/timesfm2p5_zero_shot.csv`. The existing complete scorer requires the climate-based `v2` tensors because it also scores climate-aware models and rebuilds naive baselines. Run it only when those prepared benchmark tensors already exist; it will then include TimesFM automatically:

```powershell
.venv\Scripts\python.exe scripts\evaluation\34.long_horizon_analysis.py
```

Do not add this arm to a pre-specified ensemble or to the declared primary family after seeing results. Report it as an exploratory external comparator unless the benchmark declaration is prospectively revised before its results are inspected.

## Reproducibility and compatibility

The run settings live in `configs/timesfm_zero_shot.toml`; the executable is `scripts/training/46.timesfm_zero_shot.py`. TimesFM is deliberately listed in `requirements-timesfm.txt`, rather than the base requirements, so users who do not run Experiment A retain the existing environment. The dependency is pinned to the API verified here: package `timesfm` 3.0.2 with the TimesFM 2.5 200M PyTorch checkpoint. See the [official TimesFM API reference](https://github.com/google-research/timesfm/blob/master/timesfm-forecasting/references/api_reference.md).

# Experiment B-weather — TimesFM with weather and calendar covariates

This extension of Experiment B adds district rainfall, temperature, relative humidity and wind speed to TimesFM XReg, alongside future-known calendar seasonality.

For every forecast origin, the model receives observed local weather only through that origin. The future climate portion required by XReg is filled using a district-specific, week-of-year climatology calculated from weather observed through that origin. Actual weather from the target weeks is never supplied. This makes the result a causal weather-information experiment rather than a retrospective forecast with leaked target weather.

The run requires a complete `data/interim/climate_by_dengue_period.parquet` panel for all 25 districts. It deliberately fails on partial data and never starts Open-Meteo requests itself.

The quickest route is restoring that panel from a prior project workspace. If it is unavailable, the existing `data/raw/open_meteo_cache/` is resumable: the climate fetch only requests districts not already cached, then the aggregation writes the required panel.

```powershell
.venv\Scripts\python.exe scripts\data\5c.fetch_open_meteo_climate.py
.venv\Scripts\python.exe scripts\data\9.aggregate_climate_to_periods.py
```

```powershell
.venv\Scripts\python.exe scripts\training\48.timesfm_weather_covariates.py --dry-run
.venv\Scripts\python.exe scripts\training\48.timesfm_weather_covariates.py --folds 0 --horizons 1 4 12 --suffix _pilot --no-holdout
.venv\Scripts\python.exe scripts\training\48.timesfm_weather_covariates.py
```

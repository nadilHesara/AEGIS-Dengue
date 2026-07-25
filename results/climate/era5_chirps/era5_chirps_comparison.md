# ERA5-Land and CHIRPS rainfall comparison

Neither source is treated as truth. ERA5-Land is a reanalysis: physically consistent and complete by construction, but modelled. CHIRPS is satellite infrared calibrated against rain gauges: closer to observation and finer at 0.05 degrees, but with retrieval gaps and an uneven gauge network.

**This comparison changes nothing on its own.** ERA5 rainfall remains the extracted series; no substitution is applied.

## Recommendation

**Additional climate channel**

- Median daily Pearson correlation 0.520, Spearman 0.616.
- Median absolute relative bias in mean rainfall 2.9%.
- 1 of 5 districts correlate below 0.5 on daily values: Ratnapura.
- The products agree on the broad signal but differ enough on individual days that each carries information the other does not, and the difference is not a simple rescaling. Carrying both as separate channels lets the model use that difference rather than forcing a choice.

### What each outcome would mean

| Outcome | Meaning |
| --- | --- |
| Quality cross-check | CHIRPS stays outside the model and is used to audit ERA5 rainfall. |
| Additional channel | Both series feed the model as separate features; the model weighs them. |
| Replacement candidate | CHIRPS *may* be the better rainfall source, to be decided by held-out dengue forecast skill, never by rainfall statistics alone. |

## Districts compared

Chosen to span the rainfall regimes, so that agreement in one regime is not mistaken for agreement everywhere.

| District | Regime | Why included |
| --- | --- | --- |
| Colombo | Wet zone, coastal | Highest dengue burden; smaller than one native ERA5 cell |
| Gampaha | Wet zone, coastal | Adjacent to Colombo; checks local consistency |
| Ratnapura | Wet zone, inland | Orographic rainfall, which reanalysis and satellite treat differently |
| Jaffna | Dry zone, northern | Low totals, where relative differences are largest |
| Anuradhapura | Dry zone, interior | Largest district, so grid resolution is least likely to confound |

## Comparison statistics

| canonical_name   |   days_both_present |   pearson_r |   spearman_r |   era5_mean_mm |   chirps_mean_mm |   relative_bias |   era5_wet_days |   chirps_wet_days |
|:-----------------|--------------------:|------------:|-------------:|---------------:|-----------------:|----------------:|----------------:|------------------:|
| Colombo          |                7429 |      0.51   |       0.6025 |          9.067 |            9.068 |         -0.0001 |            6021 |              4332 |
| Gampaha          |                7429 |      0.5198 |       0.6164 |          7.515 |            7.637 |         -0.016  |            5772 |              4232 |
| Ratnapura        |                7429 |      0.4906 |       0.6431 |          8.921 |            8.322 |          0.072  |            6545 |              4735 |
| Jaffna           |                7429 |      0.6601 |       0.5804 |          3.273 |            3.37  |         -0.029  |            2899 |              1920 |
| Anuradhapura     |                7429 |      0.6453 |       0.7084 |          3.696 |            4.374 |         -0.155  |            3788 |              3096 |

Pearson measures agreement on magnitude and Spearman on ordering. Rainfall is heavily skewed, so a large gap between the two means a few extreme days dominate the Pearson value.

## Missingness

| canonical_name   |   total_days |   days_both_present |   era5_missing_days |   chirps_missing_days |
|:-----------------|-------------:|--------------------:|--------------------:|----------------------:|
| Colombo          |         7429 |                7429 |                   0 |                     0 |
| Gampaha          |         7429 |                7429 |                   0 |                     0 |
| Ratnapura        |         7429 |                7429 |                   0 |                     0 |
| Jaffna           |         7429 |                7429 |                   0 |                     0 |
| Anuradhapura     |         7429 |                7429 |                   0 |                     0 |

Correlations use only days where both products report a value. These counts show how many days that excluded, so a high correlation on a small overlap is not mistaken for broad agreement.

## Heavy rainfall events

Days where either product reported at least 50.0 mm. Whether the two agree on the *same* heavy days matters more than whether they report the same number of them.

| Agreement | Days |
| --- | --- |
| both | 63 |
| era5 only | 293 |
| chirps only | 308 |

Full list: `heavy_rainfall_events.csv` (664 rows).

## Seasonal cycle

Sri Lanka has two monsoons. A product that reproduces the annual total but misplaces the seasonal peak would mislead a dengue model driven by lagged rainfall, so the monthly means are compared separately in `monthly_rainfall_comparison.csv`.

Mean daily difference by month (ERA5 minus CHIRPS, mm/day):

|   month |   Anuradhapura |   Colombo |   Gampaha |   Jaffna |   Ratnapura |
|--------:|---------------:|----------:|----------:|---------:|------------:|
|       1 |          -0.2  |      0.22 |      0.03 |    -0.05 |        3.31 |
|       2 |           0.21 |     -0.07 |     -0.69 |     0.04 |        1.39 |
|       3 |          -0.55 |     -1.13 |     -1.49 |     0.13 |       -0.57 |
|       4 |          -1.39 |     -2.83 |     -3.12 |    -0.25 |        0.48 |
|       5 |          -1.68 |     -1.62 |     -1.66 |     0.7  |       -2.8  |
|       6 |           0.97 |      0.26 |     -0.19 |     0.9  |       -2.56 |
|       7 |           0.31 |      3.4  |      3.19 |     1.19 |        0.13 |
|       8 |           0.2  |      2.9  |      2.94 |     1.26 |       -0.53 |
|       9 |          -1.15 |     -1.47 |      0.14 |     1    |       -1.52 |
|      10 |          -1.13 |     -1.09 |     -0.97 |    -0.74 |        1.1  |
|      11 |          -1.27 |      0.23 |     -0.18 |    -4.01 |        4.06 |
|      12 |          -2.4  |      1.21 |      0.58 |    -1.39 |        4.57 |

## Plots

One four-panel figure per district under `plots/`: monthly totals, mean seasonal cycle, daily scatter against the 1:1 line, and the wet-day distribution. Together these separate a constant bias from a seasonal misplacement from a disagreement confined to extremes.

## Method notes

- A day counts as wet at 1.0 mm. Both products produce many near-zero values that are numerical drizzle rather than rain; counting those would compare rounding behaviour.
- Heavy rainfall is 50.0 mm and extreme is 100.0 mm.
- Both series are aggregated over identical GADM polygons at identical scale, so a difference is a property of the rainfall products, not of the extraction.
- The pairing is an outer join. An inner join would hide the missingness this comparison is meant to measure.

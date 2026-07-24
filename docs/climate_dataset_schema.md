# Climate dataset specification

ERA5 daily weather for the 25 canonical Sri Lankan districts.

This document specifies the climate side of AEGIS-Dengue. It defines the raw
daily schema, the ERA5 variables behind each column, the unit conversions,
and the district aggregation method. It is a specification: no lags, no
dengue merge, no model features.

## The three stages, and why they are separate

Climate data passes through three distinct stages. Confusing them is the
easiest way to leak future information into a forecast, so they are named
and separated here before anything else.

| Stage | Grain | Key | Built by | This document |
| --- | --- | --- | --- | --- |
| **1. Raw daily climate** | one row per date × district | `date` + `node_id` | ERA5 extraction | **specified here** |
| **2. Aligned reporting-period climate** | one row per period × district | `period_id` + `node_id` | calendar join | interface only |
| **3. Model sequence windows** | one tensor per forecast origin | window index | feature builder | out of scope |

**Stage 1 — raw daily climate.** Pure ERA5. One row per calendar date per
district, on the UTC day boundary. It knows nothing about dengue: not the
reporting periods, not the case counts, not the train/test split. It is
built once and is independent of any modelling decision. Because it is
independent, it can be rebuilt, extended or audited without touching the
dengue data. **This is the only stage specified in this document.**

**Stage 2 — aligned reporting-period climate.** Daily rows aggregated onto
the reporting calendar by joining on the closed date interval
`start_date` to `end_date` of each period, then keyed by `period_id`.
This is where the calendar's irregularities matter: the 8-day and 6-day 2009
periods must aggregate over their true lengths, and the two uncovered days
(2025-12-27, 2025-12-28) belong to no period and must be dropped rather than
attributed to a neighbour. Sums scale with period length and means do not —
see *Aggregating to reporting periods* below.

**Stage 3 — model sequence windows.** Lags, rolling statistics and sequence
tensors built from stage 2, aligned to a forecast origin. Every value in a
window must be observable at that origin. Nothing in stages 1 or 2 assumes a
lag structure, so this stage is free to change without a re-extraction.

> **Scope rules for stage 1.** Do not create weekly lags. Do not merge dengue
> cases. Do not aggregate to reporting periods. Those belong to stages 2
> and 3. Keeping stage 1 dengue-independent is what makes the climate data
> reusable and auditable.

---

## Raw daily schema

`data/interim/weather_daily.parquet` — one row per `date` × `node_id`.

### Required columns

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `date` | date | — | Calendar date, `YYYY-MM-DD`, UTC day boundary |
| `node_id` | int | — | District key, 0–24, from `nodes.csv` |
| `canonical_name` | string | — | District name, from `nodes.csv` |
| `rainfall_mm` | float | mm/day | Total precipitation over the UTC day |
| `temperature_mean_c` | float | °C | Daily mean 2 m air temperature |
| `temperature_min_c` | float | °C | Daily minimum 2 m air temperature |
| `temperature_max_c` | float | °C | Daily maximum 2 m air temperature |
| `dewpoint_mean_c` | float | °C | Daily mean 2 m dewpoint temperature |
| `relative_humidity_mean` | float | % (0–100) | Derived from temperature and dewpoint |
| `wind_speed_mean` | float | m/s | Derived from the u and v components |
| `weather_observed` | int | 0 or 1 | 1 when derived from real ERA5 pixels |

### Optional columns

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `u_wind_mean` | float | m/s | Daily mean eastward 10 m wind |
| `v_wind_mean` | float | m/s | Daily mean northward 10 m wind |
| `surface_pressure_mean` | float | hPa | Daily mean surface pressure |
| `data_source` | string | — | e.g. `ERA5-Land` or `ERA5` |
| `extraction_version` | string | — | Extraction run identifier |

Retaining `u_wind_mean` and `v_wind_mean` is recommended: wind **direction**
carries monsoon information that `wind_speed_mean` alone discards, and the
components cannot be recovered from the scalar speed.

### Keys and identity

`node_id` and `canonical_name` must come from
[`data/processed/nodes.csv`](../data/processed/nodes.csv) by lookup — never
by re-deriving names from a geospatial file or by string matching. `node_id`
values 0–24 are a permanent part of the model definition.

The climate data uses the **canonical 25 districts**, not the 26 source
reporting areas. Kalmune has no separate climate row: it is part of the
Ampara polygon. The 26→25 merge is a dengue-source concern that never
reaches the climate side.

### `weather_observed`

Mirrors `case_observed` in the dengue dataset and carries the same meaning:
`1` means the value came from real ERA5 pixels.

A row must never be emitted with `weather_observed = 0` and fabricated
values in the weather columns. If a value cannot be derived, the weather
columns are null and the flag records why. As with the missing Puttalam
dengue record, **absence is carried as absence** — a filled zero for
rainfall would assert a dry day that was never observed.

### Uniqueness

`(date, node_id)` is unique. 25 rows per date, every date in range, with no
gaps — see *Missing dates* below.

---

## ERA5 variables

**Recommended product: ERA5-Land** (~9 km native grid), falling back to ERA5
(~31 km) only where ERA5-Land has no coverage. The reason is in
*District aggregation* below: several districts are smaller than one ERA5
grid cell.

| Column | ERA5 variable | CDS short name | Source unit |
| --- | --- | --- | --- |
| `rainfall_mm` | Total precipitation | `tp` | m (metres) |
| `temperature_*_c` | 2 m temperature | `t2m` | K |
| `dewpoint_mean_c` | 2 m dewpoint temperature | `d2m` | K |
| `u_wind_mean` | 10 m u-component of wind | `u10` | m/s |
| `v_wind_mean` | 10 m v-component of wind | `v10` | m/s |
| `surface_pressure_mean` | Surface pressure | `sp` | Pa |

All are extracted at **hourly** resolution and reduced to daily values.
Hourly is required, not optional: daily minimum and maximum temperature
cannot be recovered from daily means, and ERA5-Land precipitation is an
accumulation that must be differenced correctly (below).

Record the exact dataset name, product version and extraction date in
`data_source` and `extraction_version`. ERA5 is periodically reprocessed;
without a version stamp a rebuilt dataset cannot be compared against an
earlier model run.

### Unit conversions

| Quantity | Conversion |
| --- | --- |
| Temperature | `°C = K − 273.15` |
| Dewpoint | `°C = K − 273.15` |
| Precipitation | `mm = m × 1000` |
| Pressure | `hPa = Pa ÷ 100` |
| Wind | m/s in the source; no conversion |

Apply conversions **once**, at extraction, and never again downstream. A
double conversion of temperature produces values near −273 that are obvious;
a double conversion of precipitation produces plausible-looking numbers that
are wrong by 1000×. Assert that daily temperatures fall in 0–50 °C and
rainfall in 0–1000 mm before writing.

---

## Daily aggregation from hourly data

All reductions are over the 24 hourly steps of one **UTC** day.

### Rainfall

```
rainfall_mm = sum(hourly total precipitation over the UTC day) × 1000
```

ERA5-Land `tp` is a **running accumulation from 00 UTC**, not a per-hour
value. The hourly increment is the difference between consecutive steps, and
the 00 UTC step of the following day carries the final hour. Summing the raw
accumulation values without differencing overestimates rainfall by roughly an
order of magnitude. This is the single most common ERA5 extraction error —
validate against a known wet day before trusting a full extraction.

Clip small negative values arising from numerical noise to zero. Investigate
large negatives rather than clipping them; they indicate the accumulation was
differenced across a reset boundary.

### Temperature

```
temperature_mean_c = mean(hourly t2m) − 273.15
temperature_min_c  = min(hourly t2m)  − 273.15
temperature_max_c  = max(hourly t2m)  − 273.15
```

The mean is the mean of 24 hourly values, not `(min + max) / 2`. The two
differ systematically in the tropics, where the diurnal cycle is asymmetric.

Minimum and maximum are the extremes **within the UTC day**. This is not the
same as a station's midnight-to-midnight local minimum — see *Time zone*.

### Dewpoint

```
dewpoint_mean_c = mean(hourly d2m) − 273.15
```

### Relative humidity

Derive from temperature and dewpoint using the Magnus formula, **hourly, then
average**:

```
e(T)  = 6.112 × exp((17.67 × T) / (T + 243.5))      # saturation vapour pressure, hPa
RH_h  = 100 × e(Td_h) / e(T_h)                       # per hour, T in °C
relative_humidity_mean = mean(RH_h over the day)
```

Compute RH per hour and then average. Applying the formula to daily-mean
temperature and daily-mean dewpoint gives a different, biased answer, because
the relation is non-linear. Clip the result to 0–100; values marginally above
100 occur in saturated conditions and are physically meaningful but should be
capped for consistency.

**Do not** take `dewpoint_mean_c` and `temperature_mean_c` from this table
and re-derive RH downstream — that reintroduces exactly the bias this step
avoids.

### Wind speed

Derive hourly, then average:

```
speed_h = sqrt(u10_h² + v10_h²)
wind_speed_mean = mean(speed_h over the day)
```

Again the order matters. `sqrt(mean(u)² + mean(v)²)` is the speed of the mean
**vector**, which is smaller than the mean speed whenever direction varies
within the day — near zero for a sea breeze that reverses. The mean scalar
speed is what is wanted.

If `u_wind_mean` and `v_wind_mean` are retained, they are the means of the
components (the vector mean) and are **not** consistent with
`wind_speed_mean` by construction. This is expected, not an error, and is
worth a comment in the extraction code.

### Surface pressure

```
surface_pressure_mean = mean(hourly sp) ÷ 100
```

---

## District aggregation

### Why area weighting is not sufficient on its own

Sri Lanka is small relative to the ERA5 grid. At 7.5° N a native ERA5 cell
(0.25°) is about **27.8 × 27.6 km ≈ 768 km²**. Measured against the GADM
district polygons:

| District | Area (km²) | ≈ ERA5 0.25° cells |
| --- | --- | --- |
| Colombo | 686 | **0.9** |
| Jaffna | 1 064 | 1.4 |
| Kilinochchi | 1 306 | 1.7 |
| Matara | 1 307 | 1.7 |
| Gampaha | 1 388 | 1.8 |
| … | … | … |
| Anuradhapura | 7 237 | 9.4 |

**Colombo is smaller than a single ERA5 cell**, and roughly half the
districts cover fewer than three. Colombo's bounding box touches only four
native cells, and because it is a narrow coastal strip, several of those are
predominantly ocean or lie mostly in Gampaha and Kalutara.

At native ERA5 resolution, area weighting therefore cannot separate Colombo
from its neighbours: the answer is dominated by whichever cell centre happens
to fall inside the polygon. Colombo is also the highest-burden dengue
district, so this is precisely where precision matters most.

### Required method

1. **Use ERA5-Land (~9 km) as the primary product.** At 9 km, Colombo spans
   roughly 8–9 cells instead of 1, which makes area weighting meaningful.
   This is the main reason to prefer ERA5-Land over ERA5 here.

2. **Downsample before aggregating.** Resample the native grid to a finer
   mesh (0.01°–0.05°, bilinear for continuous fields, nearest for
   precipitation) so that even the smallest district contains many sample
   points. This does not create information; it makes the area weighting
   numerically stable for small polygons.

3. **Apply a land–sea mask.** Use the ERA5 `lsm` field, keeping cells above a
   documented threshold (0.5 is conventional). Ocean pixels must not enter a
   district mean: sea surface conditions differ systematically from land, and
   including them biases coastal districts — which in Sri Lanka is most of
   them.

4. **Area-weight the aggregation.** Each contributing cell is weighted by the
   area of its intersection with the district polygon, computed in an
   equal-area projection, not in degrees:

   ```
   value_district = Σ(value_cell × area_overlap_cell) / Σ(area_overlap_cell)
   ```

5. **Compute weights once and store them.** The polygons and the grid are
   both fixed, so the weight matrix (25 districts × N cells) is constant
   across all dates. Compute it once, store it beside the extraction, and
   reuse it. This makes the aggregation reproducible and turns a re-run into
   a matrix multiply.

Use the same projection as the node registry uses for centroids
(**EPSG:32644**, UTM zone 44N) for all area calculations.

### Grid cells intersecting multiple districts

A cell straddling a boundary contributes to **every** district it intersects,
weighted by each intersection area. It is not assigned exclusively to one.

Weights are normalised **per district** (each district's weights sum to 1),
not per cell. A cell therefore legitimately contributes to two districts, and
the district values are not independent near boundaries. This is a real
property of coarse gridded reanalysis, not an artefact to be removed — but it
should be documented, because it means neighbouring districts share
measurement error in a way a spatial model may need to account for.

### Coastal districts and polygons containing ocean

Every Sri Lankan district except a small interior set touches the coast, and
GADM polygons follow the coastline, so the bounding box of a coastal district
contains substantial ocean.

The land–sea mask handles this: ocean cells are excluded before weighting.
The residual risk is a district retaining **too few** land cells after
masking. The required fallback ladder, applied in order and recorded:

1. Area-weighted mean over masked land cells at the downsampled resolution
   (the normal path).
2. If no land cell survives, relax the land–sea threshold to include partial
   land cells (`lsm > 0.1`) and record the relaxation.
3. If still empty, fall back to the value at the district **centroid**
   (`centroid_lat`, `centroid_lon` from `nodes.csv`) and record it.
4. If that fails, emit nulls with `weather_observed = 0`.

Steps 2–4 must be logged per district and per date, not silently applied. In
practice, with ERA5-Land and a downsampled grid, step 1 should succeed for
all 25 districts on all dates; anything else is a signal that the extraction
is misconfigured.

### Missing pixels

ERA5-Land is undefined over open water, so masked-out pixels are expected and
are not "missing". Genuine missing values within land areas are rare.

Handle by exclusion, never by imputation: drop the pixel from the weighted
mean and renormalise the remaining weights. Track the fraction of a
district's usual weight that survived; if it falls below a documented
threshold (0.5 is reasonable), treat the value as unobserved rather than
reporting a mean built from a fragment of the district.

Never interpolate across space to fill a pixel, and never carry a value
forward from the previous day. Both invent data.

---

## Time and calendar handling

### Time zone

ERA5 is published in **UTC**. Sri Lanka is **UTC+05:30**.

Daily aggregation in this specification uses the **UTC day boundary**. This
is a deliberate choice, and it must be applied consistently:

- A "day" here runs 00:00–23:59 UTC, which is 05:30–05:29 Sri Lanka time.
- Daily minimum and maximum temperature are extremes within that UTC window,
  which is offset from a local-midnight day.
- Daily rainfall totals are UTC-day totals.

UTC is chosen because it is what ERA5 accumulations are natively defined on:
`tp` resets at 00 UTC, so a UTC-day total requires no re-accumulation and
introduces no boundary error. A local-day total would require shifting the
accumulation, and getting that wrong is a silent, systematic bias.

The 5.5-hour offset is small relative to a 6–8 day reporting period and is
consistent across the entire series, so it cannot introduce a spurious trend.
**Whichever convention is used, it must be stated in `data_source` and never
changed mid-series.** If a later decision favours local days, the entire
history must be re-extracted, not patched.

### UTC daily boundaries and the reporting calendar

Reporting periods are defined by calendar dates, not timestamps. A period
from `start_date` to `end_date` covers the **closed interval** of UTC days
from `start_date` through `end_date` inclusive.

The 2009 irregular periods are 8 and 6 days long, not 7. Aggregation must use
each period's true `reporting_days`, never an assumed 7.

### Leap years

The span contains five leap years: **2008, 2012, 2016, 2020, 2024**.

29 February is an ordinary date. Extract it, aggregate it, and include it.
Specifically:

- Do **not** build the date range from a fixed 365-day year.
- Do **not** use day-of-year as a feature without accounting for the shift
  after 29 February in leap years.
- Reporting periods spanning 29 February are still 7 days; the leap day
  occupies one of them.

A missing 29 February shows up as a one-day gap in an otherwise complete
series and should be caught by the completeness assertion below.

### Missing dates

The daily grid must be **complete**: every date in the extraction range
present for all 25 districts, with no gaps.

This is a stronger requirement than the dengue side, and deliberately so.
Dengue has genuine reporting gaps, because reporting is a human process.
ERA5 is a reanalysis: it produces a value for every cell on every date by
construction, so a missing date means the **extraction** failed, not that
weather did not occur.

A missing date must therefore be treated as an extraction error and re-run.
If it genuinely cannot be recovered, emit the row with null weather columns
and `weather_observed = 0` so the gap is explicit and countable. Never omit
the row: a silently absent row breaks the "25 rows per date" invariant and
will misalign a sequence window without raising anything.

### The two uncovered dengue dates

**2025-12-27** and **2025-12-28** are covered by no dengue reporting period
(the seam between the Saturday and Monday conventions).

Weather **is** extracted for both dates — they are ordinary calendar dates
and stage 1 knows nothing about reporting periods. They simply do not join to
any `period_id` at stage 2 and are dropped there.

They must not be attributed to either neighbouring period. Doing so would
silently lengthen that period's weather window relative to its dengue window,
mismatching exposure. The stage 2 join is on the closed interval
`start_date`–`end_date`, which excludes them naturally; an assertion should
confirm the total days consumed by all periods equals
`sum(reporting_days)`, not the raw span.

---

## Coverage window

The climate series must extend **before** the dengue series so that the first
usable forecast has a complete history behind it.

| Bound | Date | Basis |
| --- | --- | --- |
| Dengue first period start | 2006-12-23 | reporting calendar |
| Dengue last period end | 2026-05-17 | reporting calendar |
| **Minimum** start (26 weeks prior) | **2006-06-24** | 26-week history |
| **Preferred** start (52 weeks prior) | **2005-12-24** | 52-week history |

**Use the 52-week window (from 2005-12-24).** Choosing it over the 26-week
minimum costs only 182 extra days per district — about 4 550 rows in total —
and a 52-week window is the more
likely choice for capturing annual seasonality and the monsoon cycle. Being
forced to re-extract because a 26-week history cannot support a 52-week
lookback is a far larger cost than extracting it now.

Extend the end date to the last available ERA5 date at or after 2026-05-17.
ERA5 has a release latency of about five days (ERA5T preliminary) to two to
three months (final). Record which was used: preliminary data is subject to
revision, and a model trained on ERA5T and scored against final ERA5 is
comparing two slightly different datasets.

Rows before 2006-12-23 have no dengue counterpart and will not join to any
`period_id`. That is expected — they exist to fill lag windows for the
earliest forecastable periods.

---

## Aggregating to reporting periods (stage 2 interface)

Specified here only to fix the interface. **Not part of stage 1.**

Join daily rows to the calendar on the closed interval
`start_date ≤ date ≤ end_date`, group by `period_id` and `node_id`, and
reduce with the correct statistic per variable:

| Variable | Reduction | Note |
| --- | --- | --- |
| `rainfall_mm` | **sum** | scales with period length |
| `temperature_mean_c` | mean | |
| `temperature_min_c` | **min** | not the mean of daily minima |
| `temperature_max_c` | **max** | not the mean of daily maxima |
| `dewpoint_mean_c` | mean | |
| `relative_humidity_mean` | mean | |
| `wind_speed_mean` | mean | |

Because the 2009 periods are 8 and 6 days long, **summed** rainfall is not
comparable across periods without normalising. Carry `reporting_days` from
the calendar and provide a mean daily rainfall alongside the total so the
model can use whichever is appropriate. Means are unaffected.

Assert that each period × district group contains exactly `reporting_days`
daily rows. A count below that means missing weather days; above means the
join overlapped periods.

---

## Validation assertions

The extraction should assert, before writing:

**Structure**
- Exactly 25 distinct `node_id` values, equal to `nodes.csv` node 0–24.
- `canonical_name` matches `nodes.csv` for every `node_id`.
- `(date, node_id)` is unique.
- Exactly 25 rows for every date in range.
- No date gaps between the first and last date, leap days included.

**Range**
- `temperature_min_c ≤ temperature_mean_c ≤ temperature_max_c` per row.
- Temperatures within 0–50 °C.
- `rainfall_mm ≥ 0` and below a documented daily maximum (1000 mm).
- `relative_humidity_mean` within 0–100.
- `wind_speed_mean ≥ 0`, below 60 m/s.
- `dewpoint_mean_c ≤ temperature_mean_c` per row — dewpoint above air
  temperature is physically impossible and indicates a variable mix-up.

**Coverage**
- First date on or before the chosen history start (2005-12-24 preferred).
- Last date on or after 2026-05-17.
- Every date in the dengue span 2006-12-23 to 2026-05-17 present.

**Integrity**
- `weather_observed ∈ {0, 1}`.
- Where `weather_observed = 1`, no weather column is null.
- Where `weather_observed = 0`, **all** weather columns are null — never a
  zero, and never a carried-forward value.
- The count of `weather_observed = 0` rows is reported, not silently
  tolerated.

**Aggregation**
- Stored district weights sum to 1 per district.
- Every district retains at least one land cell after masking.
- Any district using a fallback path is listed by name and date.

---

## What this stage must not do

- Merge dengue cases. Stage 1 is dengue-independent.
- Create weekly lags, rolling means or sequence windows.
- Aggregate to reporting periods or attach `period_id`.
- Fill missing weather with zeros, means or carried-forward values.
- Interpolate across space to fill masked or missing pixels.
- Re-derive `node_id` or district names from anything but `nodes.csv`.
- Use the 26 source reporting areas. Climate uses the canonical 25.
- Mix UTC and local day boundaries within one series.
- Silently switch ERA5 product or version between extractions.

## Related documents

- [`docs/reporting_calendar.md`](reporting_calendar.md) — the chronological
  key, the reporting conventions, and the two uncovered dates.
- [`data/processed/nodes.csv`](../data/processed/nodes.csv) — the permanent
  district registry: `node_id`, `canonical_name`, centroids.
- [`data/interim/reporting_calendar.csv`](../data/interim/reporting_calendar.csv)
  — `period_id`, period bounds and `reporting_days` for the stage 2 join.

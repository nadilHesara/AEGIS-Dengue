# AEGIS Dengue


## Weather Data Setup

This project uses **Google Earth Engine** to extract district-level weather data for Sri Lanka.

The datasets used are:

```text
ERA5-Land:  ECMWF/ERA5_LAND/HOURLY
CHIRPS:     UCSB-CHG/CHIRPS/DAILY
Boundaries: WM/geoLab/geoBoundaries/600/ADM2
```

### 1. Google Earth Engine Setup

1. Create a Google Cloud project.
2. Enable the **Google Earth Engine API**.
3. Register the project for Earth Engine access.
4. Authenticate Earth Engine locally:

```powershell
earthengine authenticate --force
```

5. Set the Google Cloud project:

```powershell
earthengine set_project YOUR_PROJECT_ID
```

Example:

```powershell
earthengine set_project aegis-dengue
```

6. Test the connection:

```powershell
python -c "import ee; ee.Initialize(project='aegis-dengue'); print('Earth Engine initialized successfully')"
```

Expected output:

```text
Earth Engine initialized successfully
```

---

## ERA5-Land Weather Data

### Submit ERA5-Land Exports

Run:

```powershell
python scripts/5.extract_era5_daily.py `
  --project aegis-dengue `
  --submit-exports
```

The exported CSV files will be created in Google Drive inside:

```text
era5_chunks
```

### Check Export Status

Run:

```powershell
python scripts/5.extract_era5_daily.py `
  --project aegis-dengue `
  --check-tasks
```

Wait until all required tasks show:

```text
COMPLETED
```

### Download ERA5-Land Files

1. Open Google Drive.
2. Open the `era5_chunks` folder.
3. Download all yearly CSV files.
4. Place them inside:

```text
data/raw/era5_chunks/
```

Remove any duplicate, partial, or test files before combining.

### Combine ERA5-Land Files

Run:

```powershell
python scripts/5b.combine_era5_chunks.py
```

This creates:

```text
data/raw/climate_daily_district.csv
```

### Validate ERA5-Land Data

Run:

```powershell
python scripts/5.extract_era5_daily.py --validate-only
```

A successful validation should report:

```text
Canonical districts:      25
Duplicate district-date:  0
No findings. The extraction is complete and plausible.
```

---

## CHIRPS Rainfall Data

### Submit CHIRPS Exports

Run:

```powershell
python scripts/7.extract_chirps_daily.py `
  --project aegis-dengue `
  --submit-exports
```

The exported CSV files will be created in Google Drive inside:

```text
chirps_chunks
```

### Check Export Status

Run:

```powershell
python scripts/7.extract_chirps_daily.py `
  --project aegis-dengue `
  --check-tasks
```

Wait until all required tasks show:

```text
COMPLETED
```

### Download CHIRPS Files

1. Open Google Drive.
2. Open the `chirps_chunks` folder.
3. Download all yearly CSV files.
4. Place them inside:

```text
data/raw/chirps_chunks/
```

Remove any duplicate, partial, or test files before combining.

### Combine CHIRPS Files

Run:

```powershell
python scripts/7b.combine_chirps_chunks.py
```

This creates:

```text
data/raw/chirps_daily_rainfall.csv
```

### Validate CHIRPS Data

Run:

```powershell
python scripts/7.extract_chirps_daily.py --validate-only
```

A successful validation should report:

```text
Canonical districts:      25
Duplicate district-date:  0
Null rainfall values:     0
```

---

## Generated Weather Data

The downloaded and combined weather files should normally not be committed to Git.

Add these lines to `.gitignore`:

```gitignore
data/raw/era5_chunks/
data/raw/chirps_chunks/
data/raw/climate_daily_district.csv
data/raw/chirps_daily_rainfall.csv
```

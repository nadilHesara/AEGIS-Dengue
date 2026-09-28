# Data directory

Data moves in one direction through this directory:

```text
raw -> validation and cleaning -> interim -> feature engineering -> processed
```

- `raw/` contains immutable source downloads: dengue notifications, ERA5-Land
  and CHIRPS climate observations, and administrative boundaries. Never edit a
  raw file in place; record its URL, retrieval date, checksum, licence, and
  source version in the experiment notes or extraction manifest.
- `interim/` contains reproducible cleaned and joined products that are useful
  between pipeline stages, such as the reporting calendar and canonical dengue
  series.
- `processed/` contains model-ready panels, tensors, graph matrices, and
  walk-forward fold definitions. Every file here must be rebuildable from
  `raw/`, `external/`, configuration, and code.
- `external/` contains third-party supporting data that is not part of the
  primary dengue, climate, or geographic downloads. Preserve its provenance in
  the same way as raw data.

Generated data is ignored by Git because it may be large or sensitive. The
small boundary and district-registry files already tracked by this repository
are exceptions retained for reproducible graph construction. Never commit
patient-level or otherwise identifying health records.

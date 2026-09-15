# Dataset guide

Place source datasets under `data/raw/` by type: dengue data in
`data/raw/dengue/`, climate products in `data/raw/climate/`, and district
geometries in `data/raw/geographic/`. Intermediate pipeline artefacts belong in
`data/interim/`; model-ready arrays and panels belong in `data/processed/`.

Data is ignored by Git. Record provenance, retrieval date, schema changes, and
licensing alongside any new external source.

# Generated results

All contents except this directory contract are generated and ignored by Git.

- `figures/`: publication-ready plots; include experiment IDs in filenames.
- `tables/`: machine-readable and publication-ready summary tables.
- `predictions/`: out-of-fold and future forecasts with timestamps, districts,
  horizons, and model versions.
- `reports/`: evaluation, validation, error-analysis, and run-summary documents.
- `experiments/`: one directory per run for configuration snapshots, logs, raw
  metrics, and environment metadata.

Some established numbered scripts currently write compatibility outputs to
`results/models/`, `results/data_validation/`, `results/climate/`, and
`results/eda/`. Keep those paths stable while those entry points are in use;
new shared result-writing code should use the categories above. Curated results
needed by a paper should be copied to a release/archive with a manifest rather
than silently committed from an ad hoc run.

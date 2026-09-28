# Development workflow

The pipeline is intentionally split between reusable implementation in `src/`
and executable, versioned research entry points in `scripts/`.

```text
Data Collection
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Climate Lag Creation
      ↓
Graph Construction
      ↓
Model Training
      ↓
Evaluation
      ↓
Prediction
```

1. Run collection and validation commands from `scripts/data/` to create raw,
   interim, and processed datasets.
2. Run `scripts/features/12.build_model_tensors.py` and
   `scripts/features/14.build_folds.py` to create leakage-safe model inputs.
3. Run `scripts/graph/13.build_adjacency.py` to generate spatial matrices.
4. Run a selected entry point from `scripts/training/`; configuration defaults
   are documented in `configs/`.
5. Use `scripts/evaluation/` to score baselines and produce diagnostics.
6. Store learned checkpoints in `models/checkpoints/` and publication artefacts
   in `results/figures/`, `results/tables/`, `results/predictions/`, or
   `results/reports/`.

The numbered filenames preserve the established research chronology. They are
not imports: scripts that need shared behaviour should import `src/` modules.

# Repository migration plan

## Scope and safety

This migration reorganises the repository without altering its research
questions, data transformations, model architectures, or experiment settings.
Tracked files are moved with `git mv` so Git can retain rename history. The
numbered prefixes on pipeline scripts are retained because they document the
established execution order and are referenced by the experiment records.

## Inventory before migration

| Area | Previous location | Finding |
| --- | --- | --- |
| Data workflow | `scripts/0` through `scripts/14` | A flat numbered pipeline mixed ingestion, cleaning, features, and graph construction. |
| Training and analysis | `scripts/15` through `scripts/29` | Baseline evaluation, experiments, and visual analysis were also flat. |
| Reusable models | `src/models/` | Research model variants were already correctly separated from executable scripts. |
| Artefacts | `data/`, `models/`, `results/` | Directories already existed and are intentionally ignored by Git. |
| Documentation | `docs/` | Strong experiment documentation exists, but no repository workflow or contributor guide. |

## File migration map

| Previous location | New location | Responsibility |
| --- | --- | --- |
| `scripts/0.load_dataset.py` | `scripts/data/0.load_dataset.py` | Download data |
| `scripts/1.dataset_validate.py` | `scripts/data/1.dataset_validate.py` | Validate raw data |
| `scripts/2.create_calendar.py` | `scripts/data/2.create_calendar.py` | Build reporting calendar |
| `scripts/3.create_nodes.py` | `scripts/data/3.create_nodes.py` | Build district registry |
| `scripts/4.create_canonical_dengue.py` | `scripts/data/4.create_canonical_dengue.py` | Canonicalise dengue data |
| `scripts/5*` through `scripts/11*` | `scripts/data/` | Climate extraction, validation, aggregation, panel building, imputation study |
| `scripts/12.build_model_tensors.py` | `scripts/features/12.build_model_tensors.py` | Build model-ready features/tensors |
| `scripts/14.build_folds.py` | `scripts/features/14.build_folds.py` | Build leakage-safe folds and transformations |
| `scripts/13.build_adjacency.py` | `scripts/graph/13.build_adjacency.py` | Construct spatial graph |
| `scripts/15`, `17`, `19`, `23`, `analysis/` | `scripts/evaluation/` | Baselines, diagnostics, figures, and evaluation analyses |
| `scripts/16`, `18`, `20` through `22`, `24` through `29` | `scripts/training/` | Baseline training and controlled experiment runners |

## Compatibility work

1. Update each moved script to locate the project root from its new depth.
2. Update dynamic script loaders and test fixtures to the new locations.
3. Update user-facing commands in documentation to the new structure.
4. Add package markers to script subdirectories so their purpose is explicit;
   scripts remain directly executable with `python path/to/script.py`.
5. Create standard config, source-package, artifact, test, and documentation
   directories without putting generated data or weights under source control.

## Validation plan

* Compile all Python source and scripts.
* Run the test suite where data-dependent fixtures are available.
* Search for stale `scripts/<number>` paths and stale project-root calculations.

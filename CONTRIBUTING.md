# Contributing to AEGIS-Dengue

## Repository rules

1. All reusable code goes into `src/`.
2. All executable commands go into `scripts/`.
3. Data must stay inside `data/`.
4. Results must stay inside `results/`.
5. Models must stay inside `models/`.

Keep experiment scripts thin: parse arguments, load configuration, call reusable
code, and write declared artefacts. Do not add datasets, model weights, or
generated reports to source directories. Add or update tests with every change
to data transformations, graph construction, models, or inference.

## Branches and commits

Use `main` for reviewed releases and `develop` for integration. Recommended
working branches are `feature/data-processing`, `feature/model-development`,
and `feature/evaluation` (or a scoped child of one of them).

Use Conventional Commit-style messages:

* `feat: add climate feature extraction`
* `fix: correct graph generation bug`
* `docs: update workflow documentation`
* `test: cover missing fold boundary`
* `refactor: separate training entry points`

Before opening a pull request, run `python -m pytest tests` and state any
data-dependent checks that could not run locally.

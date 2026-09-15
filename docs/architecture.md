# Architecture

AEGIS-Dengue uses a district graph and a shared temporal sequence model. Model
inputs are generated from the processed weekly panel, transformed per
walk-forward fold, and passed through a dense graph convolution followed by a
GRU. The baseline implementation is `src/models/stgnn.py`; experiment-specific
variants remain in `src/models/`.

`scripts/` coordinates experiments and writes artefacts. `src/` contains the
reusable components those scripts call. This separation keeps architecture
changes reviewable independently of experiment orchestration.

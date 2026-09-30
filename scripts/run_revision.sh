#!/usr/bin/env bash
# Revision of 29 September 2026: rebuild the data from the frozen climate file
# and re-run the full long-horizon benchmark (scripts 31-37).
# Usage: bash scripts/run_revision.sh <stage>   stage = data | smoke | gpu | cpu | analysis
set -euo pipefail
cd "$(dirname "$0")/.."
P=.venv/Scripts/python.exe
LOG=results/benchmark/logs
mkdir -p "$LOG"

case "${1:-}" in
  data)
    for s in data/9.aggregate_climate_to_periods.py data/10.create_master_panel.py \
             features/12.build_model_tensors.py graph/13.build_adjacency.py \
             features/14.build_folds.py data/5d.write_data_manifest.py; do
      echo "== $s"; $P scripts/$s 2>&1 | tail -4
    done ;;
  smoke)
    # One fold, one seed, two horizons, every new arm; outputs go to *_smoke files.
    $P scripts/training/31.long_horizon_neural.py --folds 9 --seeds 1 --horizons 1 4 --suffix _smoke \
      --arms gru_v2 gru_v2_nb nb_shared_v2 nb_shared_v2_climatology nb_shared_v2_noclimate \
             nb_shared_v2_wxlag1 nb_shared_v2_shuffled nb_shared_v4_t26 gru_v2_climatology ;;
  gpu)
    # Neural arms (GPU), then Chronos (GPU).
    $P scripts/training/31.long_horizon_neural.py \
      --arms gru_v1 gru_v2 gru_v2_shuffled gru_v2_quantile gcn_v2 gru_v2_lw gru_v2_quantile_lw \
             adaptive_v2 gru_v2_climatology gru_v2_noclimate gru_v2_nb > "$LOG/neural.log" 2>&1
    $P scripts/training/33.long_horizon_foundation.py > "$LOG/foundation.log" 2>&1 ;;
  cpu)
    # Shared-trunk NB arms on CPU in parallel, and the tabular arms.
    $P scripts/training/32.long_horizon_tabular.py > "$LOG/tabular.log" 2>&1 &
    for arm in nb_shared_v2 nb_shared_v3 nb_shared_v4 nb_shared_v2_shuffled nb_shared_v2_climatology \
               nb_shared_v2_noclimate nb_shared_v2_wxlag1 nb_shared_v4_t26 nb_shared_v4_t27 \
               nb_shared_v4_t29 nb_shared_v4_t30; do
      $P scripts/training/31.long_horizon_neural.py --device cpu --threads 2 --arms $arm \
        > "$LOG/$arm.log" 2>&1 &
    done
    wait ;;
  analysis)
    $P scripts/evaluation/34.long_horizon_analysis.py > "$LOG/analysis.log" 2>&1
    $P scripts/training/37.onset_task.py > "$LOG/onset_task.log" 2>&1
    $P scripts/evaluation/35.long_horizon_figures.py > "$LOG/figures.log" 2>&1 ;;
  *) echo "stage: data | smoke | gpu | cpu | analysis"; exit 1 ;;
esac

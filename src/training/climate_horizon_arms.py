"""
Frozen experiment specification for the climate-horizon extension (2026-09-30).

Frozen before any outer-test year of this extension was scored, and recorded
in results/climate_horizon/run_manifest_2026-09-30.json. Changing anything
here after that point is a protocol deviation and must be dated.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
FINAL_WEIGHTS_DIR = PROJECT_DIR / "results" / "climate_horizon" / "weights_final"

RETROSPECTIVE_FOLDS = tuple(range(1, 10))     # test years 2017-2025
HEADLINE_FOLDS = (1, 2, 3, 6, 7, 8, 9)
HOLDOUT_FOLD = 10                             # 2026, Jan-May, frozen-specification check

PRIMARY_SEEDS = (0, 1, 2, 3, 4)               # B and D
ABLATION_SEEDS = (0, 1, 2)                    # every other arm
PILOT_SEEDS = (0, 1)
PILOT_BLOCKS = (0, 1, 2)                      # K = 3 inner blocks per fold

# arm: (lag mode, weight source, gradnorm, scored on 2026)
#   weight source: "equal", "measured" (primary shuffle utility), "case_only",
#   or a literal tuple.
ARMS = {
    "B_target_equal": ("target", "equal", False, True),          # main baseline
    "D_target_measured": ("target", "measured", False, True),    # proposed
    "A_origin_equal": ("origin", "equal", False, False),
    "C_origin_measured": ("origin", "measured", False, False),
    "E_target_fixed_1234": ("target", (1.0, 2.0, 3.0, 4.0), False, False),
    "F_target_h4_only": ("target", (0.0, 0.0, 0.0, 1.0), False, False),
    "G_target_caseonly_utility": ("target", "case_only", False, False),
    "H_target_gradnorm": ("target", "equal", True, False),
}
PRIMARY_ARMS = ("B_target_equal", "D_target_measured")


def seeds_for(arm: str) -> tuple[int, ...]:
    return PRIMARY_SEEDS if arm in PRIMARY_ARMS else ABLATION_SEEDS


def weights_for(arm: str, fold_id: int, weights_dir: Path = FINAL_WEIGHTS_DIR) -> tuple[tuple[float, ...], str]:
    """(weights, source). Measured weights must exist and be informed or an explicit fallback."""

    _, source, _, _ = ARMS[arm]
    if source == "equal":
        return (1.0, 1.0, 1.0, 1.0), "equal"
    if isinstance(source, tuple):
        return source, "fixed"
    utility = "shuffle" if source == "measured" else "case_only"
    path = weights_dir / f"fold{fold_id}_{utility}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing frozen weights {path}: run the pilots and script 42 first")
    record = json.loads(path.read_text())
    if record["status"] not in ("informed", "all_nonpositive_uniform"):
        raise ValueError(f"{path.name}: status {record['status']}")
    return tuple(record["weights"]), str(path.relative_to(PROJECT_DIR))


def config_for(arm: str, defaults: dict) -> dict:
    mode, _, gradnorm, _ = ARMS[arm]
    return {**defaults, "lag_mode": mode, "gradnorm": gradnorm}

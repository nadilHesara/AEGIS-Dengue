"""
Write the immutable run manifest for the climate-horizon extension.

    python scripts/evaluation/46.freeze_climate_horizon_manifest.py

Writes results/climate_horizon/run_manifest_2026-09-30.json once, its SHA-256
to run_manifest_2026-09-30.sha256, and marks both read-only. It refuses to
overwrite an existing manifest. Script 45 checks the hash before every run.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.evaluation.climate_horizon import data_provenance  # noqa: E402
from src.training import climate_horizon as tr  # noqa: E402
from src.training import climate_horizon_arms as spec  # noqa: E402
from src.training import climate_horizon_pilots as pilots  # noqa: E402
from src.training import climate_horizon_weights as cw  # noqa: E402

OUT = PROJECT_DIR / "results" / "climate_horizon"
MANIFEST = OUT / "run_manifest_2026-09-30.json"
HASH = OUT / "run_manifest_2026-09-30.sha256"

CODE = [
    "src/data/climate_horizon.py", "src/models/climate_horizon.py", "src/models/horizon_lag_encoder.py",
    "src/models/lag_encoder.py", "src/models/negative_binomial.py", "src/models/climate_ablation.py",
    "src/training/climate_horizon.py", "src/training/climate_weighting.py",
    "src/training/climate_gradnorm.py", "src/training/climate_horizon_pilots.py",
    "src/training/climate_horizon_weights.py", "src/training/climate_horizon_arms.py",
    "src/evaluation/climate_horizon.py", "src/evaluation/long_horizon.py", "src/evaluation/revision.py",
    "scripts/features/14.build_folds.py", "scripts/evaluation/15.evaluate_naive_baselines.py",
    "scripts/evaluation/39.evaluate_climate_horizon.py", "scripts/training/41.run_climate_horizon_pilots.py",
    "scripts/training/42.compute_climate_horizon_weights.py", "scripts/training/45.run_climate_horizon_sweep.py",
]

PY = r".venv\Scripts\python.exe"
COMMANDS = [
    f"{PY} scripts\\training\\41.run_climate_horizon_pilots.py --folds 1 2 3 4 5 6 7 8 9 10 --blocks 0 1 2 --seeds 0 1",
    f"{PY} scripts\\training\\42.compute_climate_horizon_weights.py --out results\\climate_horizon\\weights_final "
    "--require-blocks 0 1 2 --require-seeds 0 1",
    f"{PY} scripts\\training\\45.run_climate_horizon_sweep.py --folds 1 2 3 4 5 6 7 8 9",
    f"{PY} scripts\\evaluation\\39.evaluate_climate_horizon.py",
    "# primary statistics (paired tests, Holm, block bootstrap): analysis script [PENDING], spec below",
    "# only after the retrospective report is written:",
    f"{PY} scripts\\training\\45.run_climate_horizon_sweep.py --folds 10 --holdout --confirm-holdout",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_DIR, capture_output=True, text=True).stdout.strip()


def main() -> int:
    if MANIFEST.exists():
        raise SystemExit(f"{MANIFEST.name} already exists and is immutable")
    data = data_provenance()
    if not data["matches_manifest"].all():
        raise SystemExit("data do not match the benchmark data manifest")

    retro = 9
    fits_primary = 2 * retro * len(spec.PRIMARY_SEEDS)
    fits_ablation = (len(spec.ARMS) - 2) * retro * len(spec.ABLATION_SEEDS)
    pilot_units = 10 * len(spec.PILOT_BLOCKS) * len(spec.PILOT_SEEDS)
    manifest = {
        "title": "Climate-horizon extension: frozen run manifest",
        "written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "linked_declaration": "results/benchmark/holdout_declaration.md (29 Sep 2026) and its addenda of 2026-09-30",
        "git": {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "extension_code_committed": False,
                "note": "the extension code is not committed; the SHA-256 hashes below identify it exactly"},
        "code_sha256": {c: sha(PROJECT_DIR / c) for c in CODE},
        "data": data.to_dict("records"),
        "frozen": {
            "architecture": "ClimateHorizonNB: case GCNGRU (identity adjacency, hidden 32, 2 layers) + "
                            "HorizonLagEncoder (target-relative, history 26, max_delay 29, 6 bumps, softmax) + "
                            "climate GRU 16; gated log-mean fusion; NB2 heads; case-based dispersion; 10,028 params",
            "likelihood": "NB2, log mu = log1p(y_origin) + clamp(delta, ±10); point forecast = NB median",
            "training": {k: v for k, v in tr.DEFAULTS.items()},
            "loss": "pooled masked NB NLL; climate group gets grad of L_climate = sum a_h w_h L_h / sum a_h w_h, "
                    "all other parameters grad of L_equal = sum a_h L_h; global clip 1.0; one Adam step",
            "early_stopping": "unweighted validation NB NLL, patience 15, max 150 epochs, best-state restore; "
                              "validation = the year before the test year",
            "preprocessing": "per-fold imputation + z-scoring fitted on training years only (fit_end = train_end)",
            "features": "v2 tensors; case channels 9; climate channels 7 instantaneous; rolling means excluded",
            "origins": "every origin with 37 weeks of history; targets per split by their own date",
            "pilots": {"blocks": list(spec.PILOT_BLOCKS), "seeds": list(spec.PILOT_SEEDS),
                       "permutation_seeds": list(pilots.PERMUTATION_SEEDS), "season_weeks": pilots.SEASON_WEEKS,
                       "shuffle": "whole 37x7 window from the same district, donor origin in pilot-training "
                                  "history, +-2 weeks of year, different calendar year; frozen-model inference"},
            "utility": {"primary": "g_h = MAE_shuffled - MAE_real (absolute), u_h = max(g_h, 0)",
                        "sensitivity": "u_case_h = max(0, (MAE_case - MAE_real) / (MAE_case + 1e-8))",
                        "aggregation": "MAE per unit on matched cells; mean over seeds within block; "
                                       "mean over blocks equally; no smoothing across folds",
                        "weights": f"q = 4 u / sum u; w = (1 - rho) + rho q, rho = {cw.RHO}; uniform if sum u = 0",
                        "weights_dir": "results/climate_horizon/weights_final (blocks 0,1,2 x seeds 0,1 required)",
                        "development_weights_not_used": "results/climate_horizon/weights/fold9_* (2 blocks)"},
            "arms": {a: {"lag_mode": v[0], "weights": v[1] if isinstance(v[1], str) else list(v[1]),
                         "gradnorm": v[2], "scored_on_2026": v[3], "seeds": list(spec.seeds_for(a))}
                     for a, v in spec.ARMS.items()},
            "folds": {"retrospective": list(spec.RETROSPECTIVE_FOLDS), "headline": list(spec.HEADLINE_FOLDS),
                      "covid_reported_separately": [4, 5], "holdout": spec.HOLDOUT_FOLD},
            "metrics": "MAE (primary), peak MAE (district p90 of history up to the benchmark fit_end), NB NLL, "
                       "WIS/coverage from NB quantiles; per fold = mean over observed test cells; seed-mean "
                       "estimand primary (mean of per-seed MAE), seed-ensemble secondary; headline = mean of 7 folds",
            "primary_comparison": "D_target_measured vs B_target_equal, headline MAE, each h = 1..4; paired t-test "
                                  "and Wilcoxon over the 7 headline folds; Holm across the 4 horizons; "
                                  "4-week time-block bootstrap 95% CI (all districts jointly, 2000 resamples)",
            "criteria": {"practical_improvement": "D - B <= -0.30 MAE at h = 3 or 4 with Holm-adjusted p < 0.05 "
                                                  "and bootstrap CI below 0",
                         "non_inferiority": "h = 1, 2: upper bound of the bootstrap 95% CI of D - B <= +0.30 MAE",
                         "otherwise": "report as no measurable difference (a valid finding)"},
            "secondary": "alignment (A vs B, C vs D), controls E/F/G/H vs B, all exploratory; references on "
                         "common cells: persistence, gru_v2_nb, nb_shared_v2, lgbm_v2 (read-only)",
            "holdout_2026": "B and D only, 5 seeds; point estimate + 4-week block bootstrap CI; wording per the "
                            "29 Sep declaration; not a blind test (see addendum)",
            "chronos": "outside comparator only; no teacher outputs, utilities or features; fine-tuned variants "
                       "excluded from the course deliverable",
        },
        "run_count": {
            "pilot_units": pilot_units, "pilot_units_already_done": 4, "pilot_fits": 2 * pilot_units,
            "retrospective_fits": fits_primary + fits_ablation,
            "holdout_fits": 2 * len(spec.PRIMARY_SEEDS),
            "measured_seconds": {"pilot_unit_mean": 18.9, "pilot_unit_max": 27.1, "fit_mean": 9.2,
                                 "fit_max": 13.0, "gradnorm_fit_mean": 14.0, "hardware": "RTX 4070 Laptop GPU"},
            "estimate_minutes": {"pilots": "about 18 (max 25)", "retrospective": "about 41 (max 56)",
                                 "holdout": "about 2", "total": "about 1 hour, at most 1.5 hours on this GPU; "
                                 "roughly 2-3x on a Colab T4"},
        },
        "commands": COMMANDS,
        "not_yet_implemented": ["primary statistics script (paired tests, Holm, bootstrap) for the extension"],
        "approval": "The repository defines no approval step for hold-out scoring (CI runs tests on PRs to "
                    "main/develop only). No supervisor or reviewer approval has been obtained or is claimed.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    digest = sha(MANIFEST)
    HASH.write_text(f"{digest}  {MANIFEST.name}\n", encoding="utf-8")
    for path in (MANIFEST, HASH):
        os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

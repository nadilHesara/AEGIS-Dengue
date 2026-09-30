"""
Write the dated, write-once correction manifest for routing v2.

    python scripts/evaluation/54.freeze_correction_manifest.py

results/climate_horizon/correction_2026-09-30/correction_manifest.json (+ .sha256,
both read-only). The parent manifest is not modified.
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
from src.models.climate_horizon import ROUTING_VERSION  # noqa: E402
from src.training import climate_horizon as tr  # noqa: E402
from src.training import climate_horizon_arms as spec  # noqa: E402

ROOT = PROJECT_DIR / "results" / "climate_horizon"
CORR = ROOT / "correction_2026-09-30"
OUT, HASH = CORR / "correction_manifest.json", CORR / "correction_manifest.sha256"
PARENT = ROOT / "run_manifest_2026-09-30.json"

RERUNS = ["C_origin_measured", "D_target_measured", "E_target_fixed_1234", "F_target_h4_only",
          "G_target_caseonly_utility", "H_target_gradnorm"]
REUSED_ARMS = ["A_origin_equal", "B_target_equal"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT.name} exists and is immutable")
    dev = json.loads((CORR / "dev_check" / "dev_check.json").read_text())
    if not dev["passed"]:
        raise SystemExit("development check did not pass")
    parent = json.loads(PARENT.read_text())
    code = sorted(set(parent["code_sha256"]) | {
        "src/training/climate_horizon.py", "scripts/evaluation/47.climate_horizon_statistics.py",
        "scripts/evaluation/53.climate_horizon_correction_dev_check.py",
        "scripts/evaluation/54.freeze_correction_manifest.py"})
    model = tr.build_model({"n_nodes": 25, "n_case": 9, "n_climate": 7, "n_horizons": 4, "lookback": 12,
                            "lag_reach": 26}, dict(tr.DEFAULTS))
    names = {id(p): n for n, p in model.named_parameters()}
    record = {
        "title": "Climate-horizon correction manifest (routing v2)",
        "written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": ("Correction of a confirmed implementation error, made AFTER inspection of the original "
                   "retrospective and 2026 results. Corrected runs are corrected evaluations on previously "
                   "observed data, not a new blind test."),
        "parent_manifest": {"file": PARENT.name, "sha256": sha(PARENT),
                            "supplements": {p.name: sha(p) for p in sorted(ROOT.glob("run_manifest_supplement_*.json"))}},
        "confirmed_corrections": [
            "routing v2: climate correction head (climate_delta) moved from the weighted to the unweighted group "
            "(docs/climate_horizon_correction_audit.md §1-7)",
            "empty batches (no observed target) skip the optimiser step; shown never to occur in any training split",
            "results-doc wording on the block bootstrap (analysis only)",
        ],
        "routing_version": ROUTING_VERSION,
        "parameter_groups": {
            "weighted (L_climate)": [names[id(p)] for p in model.climate_parameters()],
            "unweighted (L_equal)": [names[id(p)] for p in model.other_parameters()],
            "shared_between_groups": [],
            "learning_rates": "encoder.* lr 2e-2 wd 0; all others lr 3e-3 wd 1e-4 (unchanged from v1)",
        },
        "unchanged": ["NB head and loss", "architecture", "preprocessing (training years only)",
                      "checkpoint selection (unweighted validation NLL, patience 15)", "budget (150 epochs)",
                      "folds", "origins", "seeds", "weights_final files", "criteria and statistics"],
        "code_sha256": {c: sha(PROJECT_DIR / c) for c in code},
        "data": data_provenance().to_dict("records"),
        "training_defaults": dict(tr.DEFAULTS),
        "reused": {
            "arms": {a: "results/climate_horizon/predictions/" + a for a in REUSED_ARMS},
            "B_holdout_2026": "results/climate_horizon/predictions/B_target_equal/fold10_seed*",
            "pilots": "results/climate_horizon/pilots/ (60 units)",
            "weights": "results/climate_horizon/weights_final/ (unchanged; computed from uniform-weight pilots)",
            "justification": {
                "equal weights make L_climate identical to L_equal, so routing is irrelevant": True,
                "v2 B fold 9 seed 0 reproduced the v1 model exactly": dev["B_v1_vs_v2"],
                "zero-loss steps in 77 reused A/B checkpoints": 0,
                "training origins without an observed target (40 outer/pilot splits)": 0,
            },
        },
        "reruns": RERUNS,
        "rerun_plan": {
            "retrospective": {a: {"folds": list(spec.RETROSPECTIVE_FOLDS), "seeds": list(spec.seeds_for(a))} for a in RERUNS},
            "holdout_2026": {"D_target_measured": {"fold": spec.HOLDOUT_FOLD, "seeds": list(spec.PRIMARY_SEEDS)}},
            "fits": sum(len(spec.seeds_for(a)) * 9 for a in RERUNS) + len(spec.PRIMARY_SEEDS),
        },
        "outputs": {"predictions": "results/climate_horizon/correction_2026-09-30/predictions/<arm>/",
                    "checkpoints": "results/climate_horizon/correction_2026-09-30/checkpoints/sweep/<arm>/",
                    "statistics": "results/climate_horizon/correction_2026-09-30/statistics/",
                    "holdout": "results/climate_horizon/correction_2026-09-30/holdout_2026/",
                    "status": "results/climate_horizon/correction_2026-09-30/run_status.csv"},
        "commands": [
            r".venv\Scripts\python.exe scripts\training\45.run_climate_horizon_sweep.py --correction --folds 1 2 3 4 5 6 7 8 9 "
            "--arms C_origin_measured D_target_measured E_target_fixed_1234 F_target_h4_only G_target_caseonly_utility H_target_gradnorm",
            r".venv\Scripts\python.exe scripts\evaluation\47.climate_horizon_statistics.py --correction",
            r".venv\Scripts\python.exe scripts\training\45.run_climate_horizon_sweep.py --correction --folds 10 --holdout --confirm-holdout --arms D_target_measured",
            r".venv\Scripts\python.exe scripts\evaluation\47.climate_horizon_statistics.py --correction --holdout",
        ],
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, capture_output=True, text=True).stdout.strip(),
    }
    OUT.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    digest = sha(OUT)
    HASH.write_text(f"{digest}  {OUT.name}\n", encoding="utf-8")
    for p in (OUT, HASH):
        os.chmod(p, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

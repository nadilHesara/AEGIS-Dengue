"""
Write a dated, write-once supplement to the frozen run manifest.

    python scripts/evaluation/48.write_manifest_supplement.py --tag stats

It records the current SHA-256 of every file hashed in the manifest,
flagging which ones changed since the freeze, plus any new files passed with
--add. Nothing in the manifest itself is modified.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import stat
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
ROOT = PROJECT_DIR / "results" / "climate_horizon"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--add", nargs="*", default=[])
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    manifest_path = ROOT / "run_manifest_2026-09-30.json"
    manifest = json.loads(manifest_path.read_text())
    stamp = dt.datetime.now().astimezone()
    out = ROOT / f"run_manifest_supplement_{stamp:%Y-%m-%dT%H%M}_{args.tag}.json"
    if out.exists():
        raise SystemExit(f"{out.name} exists")
    changed = {f: {"frozen": h, "now": sha(PROJECT_DIR / f)} for f, h in manifest["code_sha256"].items()
               if sha(PROJECT_DIR / f) != h}
    record = {"written": stamp.isoformat(timespec="seconds"), "supplements": manifest_path.name,
              "manifest_sha256": sha(manifest_path), "note": args.note,
              "changed_since_freeze": changed, "added": {f: sha(PROJECT_DIR / f) for f in args.add}}
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.chmod(out, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    print(out.name)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

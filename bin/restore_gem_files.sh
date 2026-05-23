#!/usr/bin/env bash
# restore_gem_files.sh — Download gem_*.json + run_summary.json for one model from HF paper_n250.
#
# Resolves the model_id → directory name mapping via run_summary.json,
# downloads only the gem JSON files (~few MB), and rsyncs to models/.
#
# Usage:
#   bash bin/restore_gem_files.sh "Qwen/Qwen2.5-7B"
#   bash bin/restore_gem_files.sh "openai-community/gpt2-xl"
#
# Written: 2026-05-23

set -euo pipefail

MODEL_ID="${1:?Usage: $0 MODEL_ID}"
LOCAL_DIR="${ROSETTA_DATA:-$HOME/rosetta_data}"

echo "=== Restoring gem files for ${MODEL_ID} ==="

python3 - <<PYEOF
import json, shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download

model_id = """${MODEL_ID}"""
local_dir = Path("""${LOCAL_DIR}""")

# Try to find the HF dir name from already-downloaded run_summary.json files
slug_hf = None
for p in sorted((local_dir / "paper_n250").glob("*/run_summary.json")):
    try:
        with open(p) as f:
            s = json.load(f)
        if s.get("model_id") == model_id:
            slug_hf = p.parent.name
            break
    except Exception:
        continue

if slug_hf is None:
    print(f"  Dir mapping not cached — downloading all run_summary.json files...")
    snapshot_download(
        "james-ra-henry/Rosetta-Activations",
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=["paper_n250/*/run_summary.json"],
    )
    for p in sorted((local_dir / "paper_n250").glob("*/run_summary.json")):
        try:
            with open(p) as f:
                s = json.load(f)
            if s.get("model_id") == model_id:
                slug_hf = p.parent.name
                break
        except Exception:
            continue

if slug_hf is None:
    print(f"ERROR: {model_id} not found in paper_n250 on HF", file=sys.stderr)
    sys.exit(1)

print(f"  Found dir: {slug_hf}")
print(f"  Downloading gem_*.json + run_summary.json...")

snapshot_download(
    "james-ra-henry/Rosetta-Activations",
    repo_type="dataset",
    local_dir=str(local_dir),
    allow_patterns=[
        f"paper_n250/{slug_hf}/gem_*.json",
        f"paper_n250/{slug_hf}/run_summary.json",
    ],
)

src = local_dir / "paper_n250" / slug_hf
dst = local_dir / "models" / slug_hf
dst.mkdir(parents=True, exist_ok=True)
n = 0
for f in src.glob("*.json"):
    shutil.copy2(f, dst / f.name)
    n += 1
print(f"  Synced {n} JSON files → {dst}")
PYEOF

echo "=== Restore complete: ${MODEL_ID} ==="

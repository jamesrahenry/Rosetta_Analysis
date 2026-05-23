#!/usr/bin/env bash
# restore_paper_n250_align.sh — Selective HF restore for PRH alignment jobs.
#
# Downloads only what alignment/align.py needs (~1-3 GB vs ~20+ GB full restore):
#   - paper_n250/*/caz_*.json        (dom vectors + peak layer per model/concept)
#   - paper_n250/*/calibration_{concept}.npy  (peak-layer activations for Procrustes)
#
# Uses Python snapshot_download (not hf CLI) to ensure glob patterns match
# across directory levels correctly.
#
# Written: 2026-05-23 UTC

set -euo pipefail

LOCAL_DIR="${ROSETTA_DATA:-$HOME/rosetta_data}"

echo "=== Restoring paper_n250 (alignment files only) from HF ==="
echo "    Patterns: caz_*.json + calibration_{17 concepts}.npy"

python3 - <<'PYEOF'
import os, sys
from pathlib import Path
from huggingface_hub import snapshot_download

local_dir = os.environ.get("ROSETTA_DATA", str(Path.home() / "rosetta_data"))

concepts = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

allow_patterns = ["paper_n250/*/caz_*.json"] + [
    f"paper_n250/*/calibration_{c}.npy" for c in concepts
]

print(f"  Downloading to {local_dir}")
print(f"  Patterns: {len(allow_patterns)} total")

snapshot_download(
    "james-ra-henry/Rosetta-Activations",
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=allow_patterns,
)
print("  HF download complete.")
PYEOF

echo "=== Syncing paper_n250/ → models/ ==="
mkdir -p "$LOCAL_DIR/models"
rsync -a "$LOCAL_DIR/paper_n250/" "$LOCAL_DIR/models/"

echo "=== Restore complete ==="
du -sh "$LOCAL_DIR/models/" 2>/dev/null || true

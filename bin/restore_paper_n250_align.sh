#!/usr/bin/env bash
# restore_paper_n250_align.sh — Selective HF restore for PRH alignment jobs.
#
# Downloads only what alignment/align.py needs (~1-3 GB vs ~20+ GB full restore):
#   - paper_n250/*/caz_*.json        (dom vectors + peak layer per model/concept)
#   - paper_n250/*/calibration_{concept}.npy  (peak-layer activations for Procrustes)
#
# Intentionally skips calibration_alllayer_*.npy (all-layer activations, very large).
# After download, rsyncs paper_n250/ → models/ for canonical script path.
#
# Written: 2026-05-23 UTC

set -euo pipefail

REPO="james-ra-henry/Rosetta-Activations"
LOCAL_DIR="${ROSETTA_DATA:-$HOME/rosetta_data}"

CONCEPTS=(
    agency authorization causation certainty credibility
    deception exfiltration formality moral_valence negation
    plurality sarcasm sentiment specificity temporal_order
    threat_severity urgency
)

# Build include patterns
INCLUDES=("--include=paper_n250/*/caz_*.json")
for concept in "${CONCEPTS[@]}"; do
    INCLUDES+=("--include=paper_n250/*/calibration_${concept}.npy")
done

echo "=== Restoring paper_n250 (alignment files only) from HF ==="
echo "    Patterns: caz_*.json + calibration_{17 concepts}.npy"
hf download "$REPO" \
    --repo-type dataset \
    --local-dir "$LOCAL_DIR" \
    "${INCLUDES[@]}" \
    -q

echo "=== Syncing paper_n250/ → models/ ==="
mkdir -p "$LOCAL_DIR/models"
rsync -a --include='*/' \
         --include='caz_*.json' \
         --include='calibration_*.npy' \
         --exclude='calibration_alllayer_*' \
         --exclude='*' \
         "$LOCAL_DIR/paper_n250/" "$LOCAL_DIR/models/"

echo "=== Restore complete ==="
du -sh "$LOCAL_DIR/models/" 2>/dev/null || true

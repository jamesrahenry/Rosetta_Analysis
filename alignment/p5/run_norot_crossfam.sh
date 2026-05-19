#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="$HOME/rosetta_data/results/p5_permutation"
LOG="$HOME/rosetta_data/results/p5_norot_crossfam_v3.log"
mkdir -p "$OUT_DIR"
df -h "$OUT_DIR"
cd "$(dirname "$0")/../.."
python alignment/p5/p5_propdepth.py \
    --data-root ~/rosetta_data/models \
    --out-dir "$OUT_DIR" \
    --cross-family \
    --exclude-models EleutherAI_pythia_1b \
    --no-rotation \
    --out-name p5_propdepth_crossfamily_norot_results.json \
    2>&1 | tee "$LOG"

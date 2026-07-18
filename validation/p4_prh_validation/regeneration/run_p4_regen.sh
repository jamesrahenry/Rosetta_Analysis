#!/usr/bin/env bash
# run_p4_regen.sh — regenerate P4's CPU-reproducible data from the published HF
# artifacts (primary A–F alignment + headline nulls). See README.md.
#
# Prereqs: python3 with numpy, scipy, huggingface_hub. ~1GB scratch on
# $P4_REGEN_STAGE for cluster-F all-layer streaming.
#
# Usage: bash run_p4_regen.sh [--out-dir DIR]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

OUT="./p4_regen_output"
for a in "$@"; do case $a in --out-dir) shift; OUT="$1";; esac; done
export P4_REGEN_STAGE="${P4_REGEN_STAGE:-./_p4_stage}"

echo "== P4 regen -> ${OUT} (stage: ${P4_REGEN_STAGE}) =="
echo "-- step 1: primary A–F alignment (validates first) --"
python3 step1_primary_alignment.py --out-dir "${OUT}"
echo "-- step 2: headline nulls (permuted / universality / peak-depth) --"
python3 step2_headline_nulls.py --out-dir "${OUT}"
echo "== done. outputs in ${OUT}/ =="

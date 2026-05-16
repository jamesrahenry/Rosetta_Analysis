#!/usr/bin/env bash
# p3_reporting.sh — P3 CPU reporting (runs after GPU extraction is complete).
#
# Runs scored CAZ analysis and cross-architecture concept ordering.
# All data reads from ~/rosetta_data/paper_n250/ (frozen HF snapshot).
#
# Usage:
#   ./scripts/p3_reporting.sh
#
# Written: 2026-05-15 UTC

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

if command -v uv &>/dev/null; then
    UV="uv run python"
else
    UV="python"
fi

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

DATA_ROOT=~/rosetta_data/paper_n250
PAPER_OUT=~/rosetta_data/results/CAZ_Validation

log "P3 reporting start (data: ${DATA_ROOT})"

# 1. Scored CAZ analysis across all models
log "analyze_scored..."
$UV caz/analyze_scored.py \
    --results-dir "${DATA_ROOT}" \
    --output "${PAPER_OUT}/scored_analysis.md" \
    --csv

# 2. Cross-architecture concept ordering
log "analyze (cross-arch ordering)..."
$UV caz/analyze.py --results "${DATA_ROOT}"/*/

log "P3 reporting complete"

#!/usr/bin/env bash
# p1_reporting.sh — P1 CPU reporting (runs after GPU extraction is complete).
#
# Runs scored CAZ analysis, family heatmaps, P5 depth-matched alignment,
# P5 null-test battery, and the full P1 claim validation suite.
# Data source is ~/rosetta_data/model_snapshots/ (P1 extraction dir).
#
# Usage:
#   ./scripts/p1_reporting.sh
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

DATA_ROOT=~/rosetta_data/model_snapshots
PAPER_OUT=~/rosetta_data/results/CAZ_Framework

log "P1 reporting start"

# 1. Scored CAZ analysis across all models
log "analyze_scored..."
$UV caz/analyze_scored.py \
    --results-dir "${DATA_ROOT}" \
    --output "${PAPER_OUT}/scored_analysis.md" \
    --csv

# 2. Family analysis — peak heatmaps + concept overlays
log "analyze (family heatmaps)..."
$UV caz/analyze.py --all

# 3. P5 depth-matched alignment
log "p5_propdepth..."
$UV alignment/p5/p5_propdepth.py \
    --data-root "${DATA_ROOT}" \
    --out-dir "${PAPER_OUT}/p5"

# 4. P5 validation battery (null tests)
log "p5_validation_battery..."
$UV alignment/p5/p5_validation_battery.py \
    --data-root "${DATA_ROOT}" \
    --out-dir "${PAPER_OUT}/p5"

# 5. Validation suite — all Paper 1 claims
log "pytest p1_caz_framework..."
$UV -m pytest validation/p1_caz_framework/ -v --tb=short

log "P1 reporting complete"

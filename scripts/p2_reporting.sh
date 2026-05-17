#!/usr/bin/env bash
# p2_reporting.sh — P2 CPU reporting (runs after GPU extraction is complete).
#
# Aggregates GEM ablation results, regenerates figures, and runs all CPU-side
# analysis steps. Data source is paper_n250/ (frozen HF snapshot); scripts
# default to ROSETTA_PAPER_N250 when --models-dir is not passed.
#
# Usage:
#   ./scripts/p2_reporting.sh        # standard — reads from paper_n250/
#
# Written: 2026-05-15 UTC
# Updated: 2026-05-17 UTC — reads from paper_n250 (not live models dir)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

if command -v uv &>/dev/null; then
    UV="uv run python"
else
    UV="python"
fi

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "P2 reporting start"

# 1. Aggregate GEM ablation results → gem_sweep_aggregate.md
log "aggregate_gem_results..."
$UV gem/aggregate_gem_results.py \
    --width 1 \
    --out-dir ~/rosetta_data/results/CAZ_GEM/

# 2. EEC figure
log "viz_gem_eec_fig1..."
$UV viz/viz_gem_eec_fig1.py \
    --out-dir ~/rosetta_data/results/gem_figures/

# 3. EEC table 2 data
log "extract_eec_table2..."
$UV gem/extract_eec_table2.py

# 4. Failure analysis
log "analyze_gem_failures..."
$UV gem/analyze_gem_failures.py

# 5. Routing survival
log "analyze_routing_survival..."
$UV gem/analyze_routing_survival.py

# 6. Post-chain decay (CPU part only)
log "test_p4_post_chain --part a..."
$UV gem/test_p4_post_chain.py --part a

log "P2 reporting complete"

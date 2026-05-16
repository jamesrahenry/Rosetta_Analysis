#!/usr/bin/env bash
# p2_reporting.sh — P2 CPU reporting (runs after GPU extraction is complete).
#
# Aggregates GEM ablation results, regenerates figures, and runs all CPU-side
# analysis steps. Data source is ~/rosetta_data/models/ (live extraction dir
# on GPU host); paper_n250/ is the frozen snapshot used by reproduce_p2.sh.
#
# Usage:
#   ./scripts/p2_reporting.sh        # standard — reads from ~/rosetta_data/models/
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

log "P2 reporting start"

# 1. Aggregate GEM ablation results → gem_sweep_aggregate.md
log "aggregate_gem_results..."
$UV gem/aggregate_gem_results.py \
    --width 1 \
    --models-dir ~/rosetta_data/models/ \
    --out-dir ~/rosetta_data/results/CAZ_GEM/

# 2. EEC figure
log "viz_gem_eec_fig1..."
$UV viz/viz_gem_eec_fig1.py \
    --models-dir ~/rosetta_data/models/ \
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

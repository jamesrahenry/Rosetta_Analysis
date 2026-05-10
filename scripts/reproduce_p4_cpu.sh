#!/usr/bin/env bash
# reproduce_p4_cpu.sh — Paper 4 (PRH / Concept-Selective Convergence) CPU-only analysis
#
# Companion to reproduce_p4.sh --gpu-only.  Run this on a machine with a good CPU
# (dev machine, CPU instance) AFTER GPU extraction is complete and results have been
# synced from the H200 via rosetta_tools/bin/sync_results.sh.
#
# What this covers (all pure CPU, no GPU required):
#   Step 5  — Procrustes alignment: primary result (same-dim-only)
#   Step 6a — Permuted-label null (100 trials per pair)
#   Step 6b — Cross-concept rotation transfer (universality test)
#   Step 6c — Split-calibration artifact test (20 splits)
#   Step 7  — P5 proportional-depth alignment analysis
#   Step 8  — P5 CKA analysis (uses CKA scores extracted on GPU)
#   Step 9  — P5 validation battery (random-input, structure-scramble, procedure-off)
#   Step 10 — Frontier alignment (optional, --with-frontier)
#
# What this does NOT cover (GPU-only, stays on H200):
#   Steps 1-4 of reproduce_p4.sh — CAZ extraction, random calib null, P5 CKA extract
#
# Prerequisites:
#   - GPU extraction complete: ~/rosetta_data/models/<model_slug>/ populated
#   - P5 CKA scores synced: ~/rosetta_data/results/PRH/p5/p5_cka_*.json present
#   - uv available (or python in PATH with required packages)
#   - Run from ~/rosetta_analysis/ (GPU host) or ~/Source/rosetta_analysis/ (dev machine)
#
# Usage:
#   ./scripts/reproduce_p4_cpu.sh                    # full CPU analysis
#   ./scripts/reproduce_p4_cpu.sh --with-frontier    # include frontier alignment (step 10)
#   ./scripts/reproduce_p4_cpu.sh --skip-p5          # alignment nulls only, skip P5 steps

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WITH_FRONTIER=false
SKIP_P5=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case $arg in
        --with-frontier) WITH_FRONTIER=true ;;
        --skip-p5)       SKIP_P5=true ;;
        --help|-h)
            sed -n '2,30p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
step() { echo; echo "══════════════════════════════════════════"; echo "  $*"; echo "══════════════════════════════════════════"; }
info() { echo "  [INFO] $*"; }
elapsed() { echo "  [TIME] $(date -u +"%H:%M:%S UTC") — $(( ($(date +%s) - START_TS) / 60 ))m elapsed"; }

START_TS=$(date +%s)
START_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
PAPER_OUT="${HOME}/rosetta_data/results/PRH"
mkdir -p "${PAPER_OUT}"

cd "${REPO_ROOT}"

RA_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
RT_SHA=$(git -C rosetta_tools rev-parse HEAD 2>/dev/null || echo "unknown")

# ---------------------------------------------------------------------------
# Resolve Python — prefer uv, fall back to whatever is in PATH
# ---------------------------------------------------------------------------
if command -v uv &>/dev/null; then
    info "uv found — syncing environment"
    uv sync --quiet
    PY="uv run python"
else
    info "uv not found — using python from PATH"
    PY="python"
fi

# ---------------------------------------------------------------------------
# Step 0 — Sanity checks (no GPU check — CPU only)
# ---------------------------------------------------------------------------
step "0 / Checking environment (CPU mode)"

$PY - <<'PYCHECK'
import sys, importlib

missing = [pkg for pkg in ["scipy", "numpy", "sklearn"]
           if importlib.util.find_spec(pkg) is None]
if importlib.util.find_spec("rosetta_tools") is None:
    missing.append("rosetta_tools")
if missing:
    print(f"[ERROR] Missing packages: {', '.join(missing)}", file=sys.stderr)
    sys.exit(1)
print("  CPU environment: OK")
PYCHECK

# Check that extracted activations exist
$PY - <<PYCHECK
import sys
from pathlib import Path

models_dir = Path.home() / "rosetta_data" / "models"
if not models_dir.exists():
    print(f"[ERROR] {models_dir} not found — run reproduce_p4.sh --gpu-only first", file=sys.stderr)
    sys.exit(1)

n_models = len([d for d in models_dir.iterdir() if d.is_dir()])
if n_models == 0:
    print(f"[ERROR] {models_dir} is empty — sync results from H200 first", file=sys.stderr)
    sys.exit(1)

print(f"  Extracted models found: {n_models} in {models_dir}")
PYCHECK

info "Started: ${START_UTC}"
info "rosetta_analysis: ${RA_SHA}"
info "rosetta_tools:    ${RT_SHA}"

# ---------------------------------------------------------------------------
# Step 5 — Procrustes alignment: primary result (CPU)
# ---------------------------------------------------------------------------
step "5 / Procrustes alignment — primary result (same-dim-only)"
info "Expected: mean aligned cosine ~0.98, raw ~0.001 (~300× SNR)."

$PY alignment/align.py --all --same-dim-only \
    --out "${PAPER_OUT}/prh_main.csv"

elapsed

# ---------------------------------------------------------------------------
# Step 6 — Alignment nulls (CPU)
# ---------------------------------------------------------------------------
step "6a / Permuted-label null (100 trials per pair)"

$PY alignment/align.py --all --same-dim-only \
    --permute-labels 100

elapsed

step "6b / Cross-concept rotation transfer (universality test)"
info "Expected: universality ratio ~0.194 (concept-selective, not universal)."

$PY alignment/align.py --all --same-dim-only \
    --cross-concept-transfer

elapsed

step "6c / Split-calibration artifact test (20 splits)"
info "Confirms R generalises to held-out DOM vectors."

$PY alignment/align.py --all --same-dim-only \
    --split-calibration --n-splits 20

elapsed

# ---------------------------------------------------------------------------
# Steps 7-9 — P5 analysis (CPU, long-running — skip with --skip-p5)
# ---------------------------------------------------------------------------
if [ "${SKIP_P5}" = true ]; then
    info "Skipping P5 steps (--skip-p5 set)."
else
    step "7 / P5 proportional-depth alignment analysis"
    info "Expected: matched 0.331 vs mismatched 0.198, Δ=+0.134, 98/98, p=1.2×10⁻³⁰."
    info "Note: numbers will update as the 17-concept / N=250 corpus is used."

    $PY alignment/p5/p5_propdepth.py --out-dir "${PAPER_OUT}/p5"

    elapsed

    step "8 / P5 CKA analysis"
    info "Requires CKA scores from p5_cka_extract.py (GPU step 4) to be synced."

    $PY alignment/p5/p5_cka_analyze.py --out-dir "${PAPER_OUT}/p5"

    elapsed

    step "9 / P5 validation battery (random-input, structure-scramble, procedure-off nulls)"

    $PY alignment/p5/p5_validation_battery.py --out-dir "${PAPER_OUT}/p5"

    elapsed
fi

# ---------------------------------------------------------------------------
# Optional: Procrustes alignment including frontier models (CPU)
# Requires frontier extraction from reproduce_p4.sh --with-frontier already done.
# ---------------------------------------------------------------------------
if [ "${WITH_FRONTIER}" = true ]; then
    step "10 / Procrustes alignment — including frontier models"

    $PY alignment/align.py --all --same-dim-only \
        --out "${PAPER_OUT}/prh_with_frontier.csv"

    elapsed
fi

END_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
TOTAL_MIN=$(( ($(date +%s) - START_TS) / 60 ))

# Append CPU completion metadata to existing provenance (written by GPU run)
$PY - <<PYPROV
import json, sys
from pathlib import Path

prov_path = Path("${PAPER_OUT}") / "provenance.json"
if prov_path.exists():
    prov = json.loads(prov_path.read_text())
    prov["cpu_analysis_completed_utc"] = "${END_UTC}"
    prov["cpu_analysis_minutes"] = ${TOTAL_MIN}
    prov["cpu_rosetta_analysis_sha"] = "${RA_SHA}"
    prov["cpu_rosetta_tools_sha"] = "${RT_SHA}"
    prov_path.write_text(json.dumps(prov, indent=2))
    print(f"  Provenance updated: {prov_path}")
else:
    print(f"  [WARN] No provenance.json found at {prov_path} — GPU run may not have completed")
PYPROV

echo
echo "  Paper 4 CPU analysis complete. ${END_UTC} (${TOTAL_MIN}m)"
echo "  Primary result:       ${PAPER_OUT}/prh_main.csv"
echo "  Permuted-label null:  rosetta_data/results/null_permuted_<concept>.csv"
echo "  Cross-concept:        rosetta_data/results/cross_concept_transfer.csv"
[ "${SKIP_P5}" = false ] && echo "  P5 depth result:      ${PAPER_OUT}/p5/p5_propdepth_samedim_results.json"
echo "  Provenance:           ${PAPER_OUT}/provenance.json"

#!/usr/bin/env bash
# reproduce_p1.sh — Paper 1 (CAZ Framework) end-to-end reproduction
#
# Extracts model activations, computes per-concept CAZ metrics, runs the
# depth-matched alignment analysis (P5), then verifies all paper claims
# with the validation suite.
#
# Usage:
#   ./scripts/reproduce_p1.sh                       # full corpus (~6h on L4 24GB)
#   ./scripts/reproduce_p1.sh --quick               # GPT-2-XL only, skips P5 (~20 min)
#   ./scripts/reproduce_p1.sh --no-clean-cache      # keep HF cache between models (use on H200)
#
# Requirements:
#   - NVIDIA GPU (≥16GB VRAM; full corpus needs 24GB for the larger models)
#   - HF_TOKEN env var (or huggingface-cli login) for gated models
#   - uv (https://docs.astral.sh/uv/) — used to manage the Python environment
#   - Rosetta_Concept_Pairs available (see concept pairs section in README)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
N_PAIRS=100
QUICK=false
NO_CLEAN_CACHE=false
GPU_ONLY=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case $arg in
        --quick)          QUICK=true ;;
        --no-clean-cache) NO_CLEAN_CACHE=true ;;
        --gpu-only)       GPU_ONLY=true ;;
        --help|-h)
            sed -n '2,20p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

CACHE_FLAG=""
[ "$NO_CLEAN_CACHE" = true ] && CACHE_FLAG="--no-clean-cache"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
step() { echo; echo "══════════════════════════════════════════"; echo "  $*"; echo "══════════════════════════════════════════"; }
info() { echo "  [INFO] $*"; }
elapsed() { echo "  [TIME] $(date -u +"%H:%M:%S UTC") — $(( ($(date +%s) - START_TS) / 60 ))m elapsed"; }

START_TS=$(date +%s)
START_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
P1_CONCEPTS="credibility certainty causation temporal_order negation sentiment moral_valence"
PAPER_OUT="${HOME}/rosetta_data/results/CAZ_Framework"
mkdir -p "${PAPER_OUT}"

cd "${REPO_ROOT}"

RA_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
RT_SHA=$(git -C rosetta_tools rev-parse HEAD 2>/dev/null || echo "unknown")
RCP_SHA=$(git -C "${HOME}/Source/Rosetta_Concept_Pairs" rev-parse HEAD 2>/dev/null || \
          git -C "${HOME}/Rosetta_Concept_Pairs" rev-parse HEAD 2>/dev/null || echo "unknown")

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
# Step 0 — Environment checks
# ---------------------------------------------------------------------------
step "0 / Checking environment"

$PY - <<'PYCHECK'
import sys, importlib

missing = [pkg for pkg in ["torch", "transformers", "scipy", "numpy", "sklearn"]
           if importlib.util.find_spec(pkg) is None]
if importlib.util.find_spec("rosetta_tools") is None:
    missing.append("rosetta_tools")
if missing:
    print(f"[ERROR] Missing packages: {', '.join(missing)}", file=sys.stderr)
    print("  Install uv (https://docs.astral.sh/uv/) and re-run — it handles everything.", file=sys.stderr)
    sys.exit(1)

import torch
if not torch.cuda.is_available():
    print("[ERROR] No CUDA GPU detected. CAZ extraction requires a GPU.", file=sys.stderr)
    sys.exit(1)

vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.0f}GB VRAM)")
if vram_gb < 15:
    print(f"  [WARNING] Only {vram_gb:.0f}GB VRAM — some models may OOM. --quick recommended.")
PYCHECK

$PY - <<'PYCHECK'
import sys
try:
    from rosetta_tools.dataset import load_concept_pairs
    pairs = load_concept_pairs("credibility", n=1)
    print(f"  Concept pairs: OK")
except FileNotFoundError as e:
    print(f"[ERROR] {e}", file=sys.stderr)
    print("  Set ROSETTA_CONCEPTS_ROOT to the *_consensus_pairs.jsonl directory, or:", file=sys.stderr)
    print("    git clone https://github.com/jamesrahenry/Rosetta_Concept_Pairs ../Rosetta_Concept_Pairs", file=sys.stderr)
    sys.exit(1)
PYCHECK

info "Environment OK"
elapsed

# ---------------------------------------------------------------------------
# Provenance snapshot — git SHAs + run metadata (written before any extraction)
# ---------------------------------------------------------------------------
info "Started: ${START_UTC}"
info "rosetta_analysis: ${RA_SHA}"
info "rosetta_tools:    ${RT_SHA}"
info "concept_pairs:    ${RCP_SHA}"

$PY - <<PYPROV
import json
from pathlib import Path
prov = {
    "started_utc": "${START_UTC}",
    "rosetta_analysis_sha": "${RA_SHA}",
    "rosetta_tools_sha": "${RT_SHA}",
    "concept_pairs_sha": "${RCP_SHA}",
    "n_pairs": ${N_PAIRS},
    "hf_model_revisions": {},
}
out = Path("${PAPER_OUT}") / "provenance.json"
out.write_text(json.dumps(prov, indent=2))
print(f"  Provenance initialized: {out}")
PYPROV

# ---------------------------------------------------------------------------
# Step 1 — CAZ extraction: GPT-2-XL (proof-of-concept model)
# ---------------------------------------------------------------------------
step "1 / CAZ extraction — GPT-2-XL (N=${N_PAIRS} pairs)"

$PY extraction/extract.py \
    --model openai-community/gpt2-xl \
    --n-pairs "${N_PAIRS}" \
    --concepts ${P1_CONCEPTS} \
    ${CACHE_FLAG}

elapsed

if [ "${QUICK}" = true ]; then
    step "Validation (quick — GPT-2-XL claims only)"
    $PY -m pytest validation/p1_caz_framework/ -m "not slow" -v \
        --tb=short \
        -k "GPT2XL or ScoredDetection"
    echo
    info "Quick validation done. Run without --quick to reproduce cross-arch and P5 claims."
    elapsed
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 2 — CAZ extraction: full paper corpus
# ---------------------------------------------------------------------------
step "2 / CAZ extraction — full corpus (~5h)"
info "Skips models already extracted."

$PY extraction/extract.py \
    --p1-corpus \
    --n-pairs "${N_PAIRS}" \
    --concepts ${P1_CONCEPTS} \
    ${CACHE_FLAG}

elapsed

if [ "${GPU_ONLY}" = true ]; then
    END_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    TOTAL_MIN=$(( ($(date +%s) - START_TS) / 60 ))
    $PY - <<PYPROV
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from extraction.extract import P1_MODELS
from rosetta_tools.paths import ROSETTA_MODELS
def _slug(mid): return mid.replace("/", "_").replace("-", "_")
revisions = {}
for model_id in P1_MODELS:
    meta = ROSETTA_MODELS / _slug(model_id) / "metadata.json"
    if meta.exists():
        try:
            sha = json.loads(meta.read_text()).get("hf_revision_sha")
            if sha and sha != "unknown":
                revisions[model_id] = sha
        except Exception:
            pass
prov_path = Path("${PAPER_OUT}") / "provenance.json"
if prov_path.exists():
    prov = json.loads(prov_path.read_text())
    prov["completed_utc"] = "${END_UTC}"
    prov["total_minutes"] = ${TOTAL_MIN}
    prov["gpu_only"] = True
    prov["hf_model_revisions"] = revisions
    prov_path.write_text(json.dumps(prov, indent=2))
    print(f"  Provenance: {len(revisions)}/{len(P1_MODELS)} revision hashes — {prov_path}")
PYPROV
    echo
    echo "  --gpu-only: extraction complete. ${END_UTC} (${TOTAL_MIN}m)"
    echo "  Run without --gpu-only for P5 + validation suite."
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 3 — P5: depth-matched alignment
# ---------------------------------------------------------------------------
step "3 / P5 — depth-matched alignment analysis"
$PY alignment/p5/p5_propdepth.py --out-dir "${PAPER_OUT}/p5"
elapsed

# ---------------------------------------------------------------------------
# Step 4 — P5: validation battery (null tests)
# ---------------------------------------------------------------------------
step "4 / P5 — validation battery"
$PY alignment/p5/p5_validation_battery.py --out-dir "${PAPER_OUT}/p5"
elapsed

# ---------------------------------------------------------------------------
# Step 5 — Verify all paper claims
# ---------------------------------------------------------------------------
step "5 / Validation suite — all Paper 1 claims"

$PY -m pytest validation/p1_caz_framework/ -v --tb=short
RESULT=$?
elapsed

END_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
TOTAL_MIN=$(( ($(date +%s) - START_TS) / 60 ))

$PY - <<PYPROV
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from extraction.extract import P1_MODELS
from rosetta_tools.paths import ROSETTA_MODELS

def _slug(mid): return mid.replace("/", "_").replace("-", "_")

revisions = {}
for model_id in P1_MODELS:
    meta = ROSETTA_MODELS / _slug(model_id) / "metadata.json"
    if meta.exists():
        try:
            sha = json.loads(meta.read_text()).get("hf_revision_sha")
            if sha and sha != "unknown":
                revisions[model_id] = sha
        except Exception:
            pass

prov_path = Path("${PAPER_OUT}") / "provenance.json"
if prov_path.exists():
    prov = json.loads(prov_path.read_text())
    prov["completed_utc"] = "${END_UTC}"
    prov["total_minutes"] = ${TOTAL_MIN}
    prov["hf_model_revisions"] = revisions
    prov_path.write_text(json.dumps(prov, indent=2))
    missing = [m for m in P1_MODELS if m not in revisions]
    print(f"  Provenance: {len(revisions)}/{len(P1_MODELS)} revision hashes")
    if missing:
        print(f"  [WARN] missing: {', '.join(m.split('/')[-1] for m in missing)}", file=sys.stderr)
PYPROV

echo
if [ $RESULT -eq 0 ]; then
    echo "  All Paper 1 claims verified. Completed: ${END_UTC} (${TOTAL_MIN}m)"
else
    echo "  Some claims failed — see output above. Completed: ${END_UTC} (${TOTAL_MIN}m)"
fi
exit $RESULT

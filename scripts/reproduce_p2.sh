#!/usr/bin/env bash
# reproduce_p2.sh — Paper 2 (GEM) end-to-end reproduction
#
# Runs CAZ extraction for the P2 corpus, builds GEMs, runs handoff-vs-peak
# ablation across all model × concept pairs, then aggregates results.
#
# P2 corpus: 16 base models (Appendix A), 17 concepts, N=250 pairs.
# Paper stats are derived from whatever data this script produces — run with
# more models or pairs and the numbers update accordingly.
#
# Usage:
#   ./scripts/reproduce_p2.sh                       # full corpus
#   ./scripts/reproduce_p2.sh --quick               # GPT-2-XL only (~30 min)
#   ./scripts/reproduce_p2.sh --no-clean-cache      # keep HF cache between models (use on H200)
#   ./scripts/reproduce_p2.sh --gpu-only            # extraction + ablation, skip aggregate
#
# Requirements:
#   - NVIDIA GPU (≥16GB VRAM; 140GB recommended for full corpus without reloads)
#   - HF_TOKEN env var (or huggingface-cli login) for gated models
#   - uv (https://docs.astral.sh/uv/) — used to manage the Python environment
#   - Rosetta_Concept_Pairs available (see concept pairs section in README)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
N_PAIRS=250
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
PAPER_OUT="${HOME}/rosetta_data/results/CAZ_GEM"
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
    print("[ERROR] No CUDA GPU detected. GEM ablation requires a GPU.", file=sys.stderr)
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
# Provenance snapshot — git SHAs + run metadata
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
# Step 1 — CAZ extraction: GPT-2-XL (required for GEM ablation)
# ---------------------------------------------------------------------------
step "1 / CAZ extraction — GPT-2-XL (N=${N_PAIRS} pairs)"
info "Skips if already extracted."

$PY extraction/extract.py \
    --model openai-community/gpt2-xl \
    --n-pairs "${N_PAIRS}" \
    ${CACHE_FLAG}

elapsed

# ---------------------------------------------------------------------------
# Step 1b — Build GEMs: GPT-2-XL (CPU — reads caz_*.json, writes gem_*.json)
# ---------------------------------------------------------------------------
step "1b / Build GEMs — GPT-2-XL"
$PY gem/build_gems.py --model openai-community/gpt2-xl
elapsed

# ---------------------------------------------------------------------------
# Step 2 — GEM ablation: GPT-2-XL (proof-of-concept)
# ---------------------------------------------------------------------------
step "2 / GEM ablation — GPT-2-XL (handoff vs peak, N=${N_PAIRS} pairs)"

$PY gem/ablate_gem.py \
    --model openai-community/gpt2-xl \
    --n-pairs "${N_PAIRS}" \
    --compare-peak \
    ${CACHE_FLAG}

elapsed

if [ "${QUICK}" = true ]; then
    step "Quick run complete — GPT-2-XL ablation only"
    info "Run without --quick to run the full P2 corpus."
    elapsed
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 3 — CAZ extraction: full P2 corpus
# ---------------------------------------------------------------------------
step "3 / CAZ extraction — full P2 corpus (16 models, N=${N_PAIRS} pairs)"
info "Skips models already extracted."

$PY extraction/extract.py \
    --p2-corpus \
    --n-pairs "${N_PAIRS}" \
    ${CACHE_FLAG}

elapsed

# ---------------------------------------------------------------------------
# Step 3b — Build GEMs: full P2 corpus (CPU)
# ---------------------------------------------------------------------------
step "3b / Build GEMs — full P2 corpus"
$PY gem/build_gems.py --all
elapsed

# ---------------------------------------------------------------------------
# Step 4 — GEM ablation: full P2 corpus
# ---------------------------------------------------------------------------
step "4 / GEM ablation — full P2 corpus (16 models × 17 concepts, N=${N_PAIRS} pairs)"

$PY gem/ablate_gem.py \
    --p2-corpus \
    --n-pairs "${N_PAIRS}" \
    --compare-peak \
    ${CACHE_FLAG}

elapsed

if [ "${GPU_ONLY}" = true ]; then
    END_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    TOTAL_MIN=$(( ($(date +%s) - START_TS) / 60 ))
    $PY - <<PYPROV
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from gem.ablate_gem import P2_MODELS
from rosetta_tools.paths import ROSETTA_MODELS
def _slug(mid): return mid.replace("/", "_").replace("-", "_")
revisions = {}
for model_id in P2_MODELS:
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
    print(f"  Provenance: {len(revisions)}/{len(P2_MODELS)} revision hashes — {prov_path}")
PYPROV
    echo
    echo "  --gpu-only: extraction + ablation complete. ${END_UTC} (${TOTAL_MIN}m)"
    echo "  Run without --gpu-only for aggregate."
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 5 — Aggregate results
# ---------------------------------------------------------------------------
step "5 / Aggregate GEM results"

$PY gem/aggregate_gem_results.py --p2-corpus --out-dir "${PAPER_OUT}"

elapsed

END_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
TOTAL_MIN=$(( ($(date +%s) - START_TS) / 60 ))

$PY - <<PYPROV
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from gem.ablate_gem import P2_MODELS
from rosetta_tools.paths import ROSETTA_MODELS

def _slug(mid): return mid.replace("/", "_").replace("-", "_")

revisions = {}
for model_id in P2_MODELS:
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
    missing = [m for m in P2_MODELS if m not in revisions]
    print(f"  Provenance: {len(revisions)}/{len(P2_MODELS)} revision hashes")
    if missing:
        print(f"  [WARN] missing: {', '.join(m.split('/')[-1] for m in missing)}", file=sys.stderr)
PYPROV

echo
echo "  Paper 2 reproduction complete. Completed: ${END_UTC} (${TOTAL_MIN}m)"
echo "  Results: ${PAPER_OUT}/gem_sweep_aggregate.md"
echo "  Provenance: ${PAPER_OUT}/provenance.json"

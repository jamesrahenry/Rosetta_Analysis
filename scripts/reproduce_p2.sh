#!/usr/bin/env bash
# reproduce_p2.sh — Paper 2 (GEM) end-to-end reproduction
#
# Extracts model activations (Paper 1 corpus is a prerequisite), runs GEM
# ablation across 16 models × 17 concepts with handoff/peak comparison,
# then aggregates results.
#
# Usage:
#   ./scripts/reproduce_p2.sh                       # full corpus (16 models)
#   ./scripts/reproduce_p2.sh --quick               # GPT-2-XL only (~30 min)
#   ./scripts/reproduce_p2.sh --no-clean-cache      # keep HF cache between models (use on H200)
#
# Requirements:
#   - NVIDIA GPU (≥16GB VRAM; 140GB recommended for full corpus without reloads)
#   - HF_TOKEN env var (or huggingface-cli login) for gated models
#   - uv (https://docs.astral.sh/uv/) — used to manage the Python environment
#   - Rosetta_Concept_Pairs available (see concept pairs section in README)
#   - CAZ extraction already run for the P2 corpus (or run reproduce_p1.sh first)

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
elapsed() { echo "  [TIME] Elapsed: $(( ($(date +%s) - START_TS) / 60 ))m"; }

START_TS=$(date +%s)
PAPER_OUT="${HOME}/rosetta_data/results/CAZ_GEM"
mkdir -p "${PAPER_OUT}"

cd "${REPO_ROOT}"

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
    info "Run without --quick to reproduce full 16-model corpus."
    elapsed
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 3 — CAZ extraction: full P2 corpus
# ---------------------------------------------------------------------------
step "3 / CAZ extraction — full P2 corpus (16 models, N=${N_PAIRS} pairs)"
info "Skips models already extracted."

$PY extraction/extract.py \
    --p1-corpus \
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
    echo
    echo "  --gpu-only: GPU extraction and ablation complete. Run without --gpu-only for aggregate."
    echo "  Total: $(( ($(date +%s) - START_TS) / 60 ))m"
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 5 — Aggregate results
# ---------------------------------------------------------------------------
step "5 / Aggregate GEM results"

$PY gem/aggregate_gem_results.py --out-dir "${PAPER_OUT}"

elapsed

echo
echo "  Paper 2 reproduction complete. Total: $(( ($(date +%s) - START_TS) / 60 ))m"
echo "  Results: ${PAPER_OUT}/gem_sweep_aggregate.md"

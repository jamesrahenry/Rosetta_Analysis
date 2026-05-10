#!/usr/bin/env bash
# reproduce_p3.sh — Paper 3 (CAZ Validation) end-to-end reproduction
#
# Extracts model activations for 26 base models, runs single-layer ablation,
# activation patching, direction-specificity null, and scored CAZ analysis.
# Supplementary instruct variants (9 models) are extracted separately and
# require --with-instruct.
#
# Usage:
#   ./scripts/reproduce_p3.sh                       # full 26-model corpus
#   ./scripts/reproduce_p3.sh --quick               # GPT-2-XL only (~45 min)
#   ./scripts/reproduce_p3.sh --no-clean-cache      # keep HF cache (use on H200)
#   ./scripts/reproduce_p3.sh --with-instruct       # also run 9 instruct variants
#
# Requirements:
#   - NVIDIA GPU (≥16GB VRAM; 140GB recommended for full corpus without reloads)
#   - HF_TOKEN env var for gated models (Llama-3.2, Gemma-2, Mistral)
#   - uv (https://docs.astral.sh/uv/) — used to manage the Python environment
#   - Rosetta_Concept_Pairs available (see concept pairs section in README)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
N_PAIRS=250
QUICK=false
NO_CLEAN_CACHE=false
GPU_ONLY=false
WITH_INSTRUCT=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case $arg in
        --quick)          QUICK=true ;;
        --no-clean-cache) NO_CLEAN_CACHE=true ;;
        --gpu-only)       GPU_ONLY=true ;;
        --with-instruct)  WITH_INSTRUCT=true ;;
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

# ablate.py defaults to keeping cache (--clean-cache opts in); no flag needed.
# ablate_random_direction.py and patch.py default to purging; pass --no-clean-cache.
RD_CACHE_FLAG=""
[ "$NO_CLEAN_CACHE" = true ] && RD_CACHE_FLAG="--no-clean-cache"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
step() { echo; echo "══════════════════════════════════════════"; echo "  $*"; echo "══════════════════════════════════════════"; }
info() { echo "  [INFO] $*"; }
elapsed() { echo "  [TIME] Elapsed: $(( ($(date +%s) - START_TS) / 60 ))m"; }

START_TS=$(date +%s)
PAPER_OUT="${HOME}/rosetta_data/results/CAZ_Validation"
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
# Step 1 — CAZ extraction: GPT-2-XL (proof-of-concept model)
# ---------------------------------------------------------------------------
step "1 / CAZ extraction — GPT-2-XL (N=${N_PAIRS} pairs)"
info "Skips if already extracted."

$PY extraction/extract.py \
    --model openai-community/gpt2-xl \
    --n-pairs "${N_PAIRS}" \
    ${CACHE_FLAG}

elapsed

if [ "${QUICK}" = true ]; then
    step "Quick: ablation + patching — GPT-2-XL only"

    $PY gem/ablate.py \
        --model openai-community/gpt2-xl \
        --n-pairs "${N_PAIRS}"

    $PY gem/patch.py \
        --model openai-community/gpt2-xl \
        --n-pairs "${N_PAIRS}" \
        ${RD_CACHE_FLAG}

    echo
    info "Quick run complete — GPT-2-XL only. Run without --quick for full 26-model corpus."
    elapsed
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 2 — CAZ extraction: full P3 base corpus (26 models)
# ---------------------------------------------------------------------------
step "2 / CAZ extraction — full P3 corpus (26 models, N=${N_PAIRS} pairs)"
info "Skips models already extracted."

$PY extraction/extract.py \
    --p3-corpus \
    --n-pairs "${N_PAIRS}" \
    ${CACHE_FLAG}

elapsed

# ---------------------------------------------------------------------------
# Step 3 — Single-layer ablation sweep (CAZ Prediction 1)
# ---------------------------------------------------------------------------
step "3 / Single-layer ablation sweep — P3 corpus (26 models)"
info "Tests whether CAZ peak is the functionally active layer."

$PY gem/ablate.py \
    --p3-corpus \
    --n-pairs "${N_PAIRS}"

# ablate.py defaults to keeping cache; no flag needed

elapsed

# ---------------------------------------------------------------------------
# Step 4 — Activation patching (causal validation)
# ---------------------------------------------------------------------------
step "4 / Activation patching — P3 corpus (26 models)"
info "Triangulates CAZ causal load-bearing via mean-field shift patching."

$PY gem/patch.py \
    --p3-corpus \
    --n-pairs "${N_PAIRS}" \
    ${RD_CACHE_FLAG}

elapsed

# ---------------------------------------------------------------------------
# Step 5 — Direction-specificity null (§6.8)
# ---------------------------------------------------------------------------
step "5 / Random-direction null — P3 corpus (26 models)"
info "Confirms suppression is direction-specific, not a layer-level artifact."

$PY gem/ablate_random_direction.py \
    --p3-corpus \
    --n-pairs "${N_PAIRS}" \
    ${RD_CACHE_FLAG}

elapsed

if [ "${GPU_ONLY}" = true ]; then
    echo
    echo "  --gpu-only: GPU extraction, ablation, and patching complete. Run without --gpu-only for CPU analysis."
    echo "  Total: $(( ($(date +%s) - START_TS) / 60 ))m"
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 6 — Scored CAZ analysis (CPU — no GPU required)
# ---------------------------------------------------------------------------
step "6 / Scored CAZ analysis — all extracted models"
info "Runs scored detection (0.5% prominence floor), produces per-model profiles."

$PY caz/analyze_scored.py --output "${PAPER_OUT}/scored_analysis.md" --csv

elapsed

# ---------------------------------------------------------------------------
# Step 7 — Cross-architecture analysis (CPU — no GPU required)
# ---------------------------------------------------------------------------
step "7 / Cross-architecture concept ordering analysis"

$PY caz/analyze.py --all

elapsed

# ---------------------------------------------------------------------------
# Optional: instruct variants (Supplementary §B)
# ---------------------------------------------------------------------------
if [ "${WITH_INSTRUCT}" = true ]; then
    step "8 / CAZ extraction — instruct variants (9 models, Supplementary §B)"

    $PY extraction/extract.py \
        --p3-corpus-instruct \
        --n-pairs "${N_PAIRS}" \
        ${CACHE_FLAG}

    elapsed
fi

echo
echo "  Paper 3 reproduction complete. Total: $(( ($(date +%s) - START_TS) / 60 ))m"
echo "  Scored analysis:     ${PAPER_OUT}/scored_analysis.md"
echo "  Ablation results:    rosetta_data/models/<model>/ablation_<concept>.json"
echo "  Patching results:    rosetta_data/models/<model>/patch_<concept>.json"

#!/usr/bin/env bash
# reproduce_p1.sh — Paper 1 (CAZ Framework) end-to-end reproduction
#
# Extracts model activations, computes per-concept CAZ metrics, runs the
# depth-matched alignment analysis (P5), then verifies all paper claims
# with the validation suite.
#
# Usage:
#   ./scripts/reproduce_p1.sh            # full corpus (~6h on L4 24GB)
#   ./scripts/reproduce_p1.sh --quick    # GPT-2-XL only, skips P5 (~20 min)
#
# Requirements:
#   - NVIDIA GPU (≥16GB VRAM; full corpus needs 24GB for the larger models)
#   - HF_TOKEN env var (or huggingface-cli login) for gated models
#   - rosetta-tools installed: pip install rosetta-tools
#   - Rosetta_Concept_Pairs available (see below)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
N_PAIRS=250
QUICK=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case $arg in
        --quick) QUICK=true ;;
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
step() { echo; echo "══════════════════════════════════════════"; echo "  $*"; echo "══════════════════════════════════════════"; }
info() { echo "  [INFO] $*"; }
fail() { echo "  [ERROR] $*" >&2; exit 1; }
elapsed() { echo "  [TIME] Elapsed: $(( ($(date +%s) - START_TS) / 60 ))m"; }

START_TS=$(date +%s)
P1_CONCEPTS="credibility certainty causation temporal_order negation sentiment moral_valence"

# ---------------------------------------------------------------------------
# Step 0 — Environment checks
# ---------------------------------------------------------------------------
step "0 / Checking environment"

cd "${REPO_ROOT}"

python - <<'PYCHECK'
import sys, importlib
missing = [pkg for pkg in ["torch", "transformers", "scipy", "numpy", "sklearn"] if importlib.util.find_spec(pkg) is None]
try:
    import rosetta_tools  # noqa: F401
except ImportError:
    missing.append("rosetta_tools (pip install rosetta-tools)")
if missing:
    print(f"[ERROR] Missing packages: {', '.join(missing)}", file=sys.stderr)
    sys.exit(1)

import torch
if not torch.cuda.is_available():
    print("[ERROR] No CUDA GPU detected. CAZ extraction requires a GPU.", file=sys.stderr)
    sys.exit(1)

vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.0f}GB VRAM)")
if vram_gb < 15:
    print(f"[WARNING] Only {vram_gb:.0f}GB VRAM — some models may OOM. Use --quick for GPT-2-XL only.")
PYCHECK

# Verify concept pairs are reachable
python - <<'PYCHECK'
import sys
try:
    from rosetta_tools.dataset import load_concept_pairs
    pairs = load_concept_pairs("credibility", n=1)
    print(f"  Concept pairs: found ({len(pairs)} loaded for smoke test)")
except FileNotFoundError as e:
    print(f"[ERROR] {e}", file=sys.stderr)
    print("  Set ROSETTA_CONCEPTS_ROOT to the directory containing *_consensus_pairs.jsonl files,", file=sys.stderr)
    print("  or clone Rosetta_Concept_Pairs alongside this repo:", file=sys.stderr)
    print("    git clone https://github.com/jamesrahenry/Rosetta_Concept_Pairs ../Rosetta_Concept_Pairs", file=sys.stderr)
    sys.exit(1)
PYCHECK

info "Environment OK"
elapsed

# ---------------------------------------------------------------------------
# Step 1 — CAZ extraction: GPT-2-XL (proof-of-concept model)
# ---------------------------------------------------------------------------
step "1 / CAZ extraction — GPT-2-XL (N=${N_PAIRS} pairs)"

python extraction/extract.py \
    --model openai-community/gpt2-xl \
    --n-pairs "${N_PAIRS}" \
    --concepts ${P1_CONCEPTS}

elapsed

if [ "${QUICK}" = true ]; then
    # ---------------------------------------------------------------------------
    # Quick mode: verify GPT-2-XL claims only (P5 and cross-arch tests skip)
    # ---------------------------------------------------------------------------
    step "Validation (quick — GPT-2-XL claims only)"
    python -m pytest validation/p1_caz_framework/ -m "not slow" -v \
        --tb=short \
        -k "GPT2XL or ScoredDetection"
    echo
    info "Quick validation done. Run without --quick to reproduce cross-arch and P5 claims."
    elapsed
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 2 — CAZ extraction: full paper corpus (all same-dim model families)
# ---------------------------------------------------------------------------
step "2 / CAZ extraction — full corpus (~5h)"
info "Extracting all L4-runnable models. Skips models already extracted."

python extraction/extract.py \
    --all \
    --n-pairs "${N_PAIRS}" \
    --concepts ${P1_CONCEPTS}

elapsed

# ---------------------------------------------------------------------------
# Step 3 — P5: depth-matched alignment (needs ≥2 same-dim models extracted)
# ---------------------------------------------------------------------------
step "3 / P5 — depth-matched alignment analysis"

python alignment/p5/p5_propdepth.py

elapsed

# ---------------------------------------------------------------------------
# Step 4 — P5: validation battery (null tests)
# ---------------------------------------------------------------------------
step "4 / P5 — validation battery (null tests)"

python alignment/p5/p5_validation_battery.py

elapsed

# ---------------------------------------------------------------------------
# Step 5 — Verify all paper claims
# ---------------------------------------------------------------------------
step "5 / Validation suite — all Paper 1 claims"

python -m pytest validation/p1_caz_framework/ -v --tb=short

RESULT=$?
elapsed

echo
if [ $RESULT -eq 0 ]; then
    echo "  All Paper 1 claims verified. Total: $(( ($(date +%s) - START_TS) / 60 ))m"
else
    echo "  Some claims failed — see output above. Total: $(( ($(date +%s) - START_TS) / 60 ))m"
fi
exit $RESULT

#!/usr/bin/env bash
# reproduce_p3.sh — Paper 3 (CAZ Validation) end-to-end reproduction
#
# Runs CAZ extraction for the P3 corpus (26 base models, 8 families, 7 concepts,
# N=250 pairs), single-layer ablation, activation patching, direction-specificity
# null, and scored CAZ analysis.  Optional --with-instruct adds 9 instruct variants
# (Supplementary §B).
#
# Paper stats are derived from whatever data this script produces — run with
# more models or pairs and the numbers update accordingly.
#
# Usage:
#   ./scripts/reproduce_p3.sh                       # full 26-model corpus
#   ./scripts/reproduce_p3.sh --quick               # GPT-2-XL only (~45 min)
#   ./scripts/reproduce_p3.sh --no-clean-cache      # keep HF cache (use on H200)
#   ./scripts/reproduce_p3.sh --with-instruct       # also run 9 instruct variants
#   ./scripts/reproduce_p3.sh --gpu-only            # extraction + ablation + patching, skip scored analysis
#
# Requirements:
#   - NVIDIA GPU (≥16GB VRAM; 140GB recommended for full corpus without reloads)
#   - HF_TOKEN env var for gated models (Llama-3.2, Gemma-2, Mistral)
#   - uv (https://docs.astral.sh/uv/) — used to manage the Python environment
#   - Rosetta_Concept_Pairs available (see concept pairs section in README)
#   - Rosetta_Feature_Library at ~/Rosetta_Feature_Library (for Gemma Scope xval step)

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
elapsed() { echo "  [TIME] $(date -u +"%H:%M:%S UTC") — $(( ($(date +%s) - START_TS) / 60 ))m elapsed"; }

START_TS=$(date +%s)
START_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
PAPER_OUT="${HOME}/rosetta_data/results/CAZ_Validation"
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

# ---------------------------------------------------------------------------
# Step 5b — Gemma Scope SAE cross-validation (GPU — loads gemma-2-2b)
# ---------------------------------------------------------------------------
step "5b / Gemma Scope SAE cross-validation"
info "Validates CAZ layer assignments against independent SAE feature activations."
info "Requires Rosetta_Feature_Library at ~/Rosetta_Feature_Library or repo root."

if [ -f "${PAPER_OUT}/gemma_scope_xval/summary.json" ]; then
    info "Gemma Scope xval already complete — skipping (delete ${PAPER_OUT}/gemma_scope_xval/ to re-run)."
else
    $PY caz/gemma_scope_xval.py \
        --out "${PAPER_OUT}/gemma_scope_xval"
fi

elapsed

if [ "${GPU_ONLY}" = true ]; then
    END_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    TOTAL_MIN=$(( ($(date +%s) - START_TS) / 60 ))
    $PY - <<PYPROV
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from extraction.extract import P3_MODELS, P3_INSTRUCT_MODELS
from rosetta_tools.paths import ROSETTA_MODELS
all_models = P3_MODELS + (P3_INSTRUCT_MODELS if "${WITH_INSTRUCT}" == "true" else [])
def _slug(mid): return mid.replace("/", "_").replace("-", "_")
revisions = {}
for model_id in all_models:
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
    print(f"  Provenance: {len(revisions)}/{len(all_models)} revision hashes — {prov_path}")
PYPROV
    echo
    echo "  --gpu-only: extraction + ablation + patching complete. ${END_UTC} (${TOTAL_MIN}m)"
    echo "  Run without --gpu-only for scored analysis + cross-arch ordering."
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

END_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
TOTAL_MIN=$(( ($(date +%s) - START_TS) / 60 ))

$PY - <<PYPROV
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from extraction.extract import P3_MODELS, P3_INSTRUCT_MODELS
from rosetta_tools.paths import ROSETTA_MODELS

all_models = P3_MODELS + (P3_INSTRUCT_MODELS if "${WITH_INSTRUCT}" == "true" else [])

def _slug(mid): return mid.replace("/", "_").replace("-", "_")

revisions = {}
for model_id in all_models:
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
    missing = [m for m in all_models if m not in revisions]
    print(f"  Provenance: {len(revisions)}/{len(all_models)} revision hashes")
    if missing:
        print(f"  [WARN] missing: {', '.join(m.split('/')[-1] for m in missing)}", file=sys.stderr)
PYPROV

echo
echo "  Paper 3 reproduction complete. Completed: ${END_UTC} (${TOTAL_MIN}m)"
echo "  Scored analysis:     ${PAPER_OUT}/scored_analysis.md"
echo "  Ablation results:    rosetta_data/models/<model>/ablation_<concept>.json"
echo "  Patching results:    rosetta_data/models/<model>/patch_<concept>.json"
echo "  Gemma Scope xval:    ${PAPER_OUT}/gemma_scope_xval/summary.json"
echo "  Provenance:          ${PAPER_OUT}/provenance.json"

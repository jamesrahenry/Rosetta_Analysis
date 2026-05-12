#!/usr/bin/env bash
# ablate_gem_p2.sh — Targeted GEM handoff-vs-peak ablation for the P2 corpus.
#
# Designed for isolated H200 runs. Unlike reproduce_p2.sh (which is end-to-end
# from scratch), this script assumes CAZ extraction and GEM files already exist
# and focuses only on running the ablation comparisons and aggregating results.
#
# Per-model isolation: each model is run in its own subprocess. One model
# crashing does not kill the run. Status is printed per model at the end.
#
# Usage:
#   bash scripts/ablate_gem_p2.sh
#   bash scripts/ablate_gem_p2.sh --dry-run        # show what would run, no execution
#   bash scripts/ablate_gem_p2.sh --skip-done      # default: skip models where all 17 concepts have comparison data
#   bash scripts/ablate_gem_p2.sh --force          # re-run all models even if complete
#   bash scripts/ablate_gem_p2.sh --model Qwen/Qwen2.5-14B  # single model
#   bash scripts/ablate_gem_p2.sh --no-clean-cache # keep HF cache between models (faster on H200)
#
# Requirements:
#   - NVIDIA GPU (>=22GB VRAM; H200 140GB recommended)
#   - CAZ extraction already done: ~/rosetta_data/models/<model>/caz_*.json
#   - GEM files already built: ~/rosetta_data/models/<model>/gem_*.json
#   - rosetta_tools and rosetta_analysis on PYTHONPATH

set -uo pipefail  # no -e: we handle per-model failures ourselves

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

N_PAIRS=250
DRY_RUN=false
FORCE=false
SINGLE_MODEL=""
NO_CLEAN_CACHE=false

for arg in "$@"; do
    case $arg in
        --dry-run)        DRY_RUN=true ;;
        --force)          FORCE=true ;;
        --skip-done)      true ;;  # default, no-op
        --no-clean-cache) NO_CLEAN_CACHE=true ;;
        --model)          shift; SINGLE_MODEL="$1" ;;
        --model=*)        SINGLE_MODEL="${arg#*=}" ;;
        --help|-h)
            sed -n '2,20p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

CACHE_FLAG=""
[ "$NO_CLEAN_CACHE" = true ] && CACHE_FLAG="--no-clean-cache"

PAPER_OUT="${HOME}/rosetta_data/results/CAZ_GEM"
LOG_DIR="${HOME}/rosetta_data/results/CAZ_GEM/ablation_logs"
mkdir -p "${PAPER_OUT}" "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
step() { echo; echo "══════════════════════════════════════════"; echo "  $*"; echo "══════════════════════════════════════════"; }
info() { echo "  [$(date -u +"%H:%M:%S")] $*"; }
pass() { echo "  [PASS] $*"; }
fail() { echo "  [FAIL] $*"; }

if command -v uv &>/dev/null; then
    PY="uv run python"
else
    PY="python"
fi

# ---------------------------------------------------------------------------
# Check completion status for a model (Python helper)
# ---------------------------------------------------------------------------
is_complete() {
    local model_id="$1"
    $PY - <<PYEOF
import json
from pathlib import Path

model_id = "${model_id}"
slug = model_id.replace("/", "_").replace("-", "_")
models_dir = Path.home() / "rosetta_data" / "models" / slug
CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]
n_pairs_required = ${N_PAIRS}
complete = 0
missing = []
for c in CONCEPTS:
    p = models_dir / f"ablation_gem_{c}.json"
    if p.exists():
        try:
            d = json.loads(p.read_text())
            if "comparison" in d and d.get("n_pairs", 0) >= n_pairs_required:
                complete += 1
            else:
                missing.append(c)
        except Exception:
            missing.append(c)
    else:
        missing.append(c)

if complete == len(CONCEPTS):
    print("COMPLETE")
elif complete == 0:
    print("MISSING")
else:
    print(f"PARTIAL:{complete}/{len(CONCEPTS)}:{'|'.join(missing)}")
PYEOF
}

# ---------------------------------------------------------------------------
# P2 corpus model list
# ---------------------------------------------------------------------------
if [ -n "${SINGLE_MODEL}" ]; then
    MODELS=("${SINGLE_MODEL}")
else
    MODELS=(
        "EleutherAI/pythia-70m"
        "EleutherAI/pythia-160m"
        "EleutherAI/pythia-410m"
        "EleutherAI/pythia-1b"
        "EleutherAI/pythia-2.8b"
        "EleutherAI/pythia-6.9b"
        "EleutherAI/pythia-12b"
        "openai-community/gpt2"
        "facebook/opt-6.7b"
        "Qwen/Qwen2.5-0.5B"
        "Qwen/Qwen2.5-1.5B"
        "Qwen/Qwen2.5-3B"
        "Qwen/Qwen2.5-7B"
        "Qwen/Qwen2.5-14B"
        "mistralai/Mistral-7B-v0.3"
        "google/gemma-2-9b"
    )
fi

# ---------------------------------------------------------------------------
# Step 0 — Status check
# ---------------------------------------------------------------------------
step "0 / Status check (${#MODELS[@]} models, N=${N_PAIRS} pairs)"

declare -a TO_RUN=()
declare -a ALREADY_DONE=()
declare -A STATUS_MAP

for mid in "${MODELS[@]}"; do
    status=$(is_complete "${mid}")
    STATUS_MAP["${mid}"]="${status}"
    short="${mid##*/}"
    if [ "${status}" = "COMPLETE" ] && [ "${FORCE}" = false ]; then
        info "${short}: COMPLETE — skipping"
        ALREADY_DONE+=("${mid}")
    else
        if [ "${status}" = "COMPLETE" ]; then
            info "${short}: COMPLETE but --force set — will re-run"
        else
            info "${short}: ${status} — will run"
        fi
        TO_RUN+=("${mid}")
    fi
done

echo
info "Already complete: ${#ALREADY_DONE[@]}"
info "To run: ${#TO_RUN[@]}"

if [ "${DRY_RUN}" = true ]; then
    echo
    echo "  [DRY RUN] Would run: ${TO_RUN[*]:-none}"
    exit 0
fi

if [ ${#TO_RUN[@]} -eq 0 ]; then
    info "All models complete — skipping to aggregate."
fi

# ---------------------------------------------------------------------------
# Step 1 — Per-model ablation (crash-isolated)
# ---------------------------------------------------------------------------
if [ ${#TO_RUN[@]} -gt 0 ]; then
    step "1 / GEM ablation (${#TO_RUN[@]} models)"
fi

declare -a FAILED_MODELS=()
declare -a PASSED_MODELS=()

for mid in "${TO_RUN[@]}"; do
    short="${mid##*/}"
    log_file="${LOG_DIR}/${short//\//_}.log"
    info "Running: ${short} → ${log_file}"

    if $PY gem/ablate_gem.py \
            --model "${mid}" \
            --n-pairs "${N_PAIRS}" \
            --compare-peak \
            ${CACHE_FLAG} \
        >"${log_file}" 2>&1; then
        pass "${short}: success"
        PASSED_MODELS+=("${mid}")
    else
        fail "${short}: FAILED — see ${log_file}"
        FAILED_MODELS+=("${mid}")
        # Show last 20 lines of the log for immediate diagnosis
        tail -20 "${log_file}" | sed 's/^/    /'
    fi
done

# ---------------------------------------------------------------------------
# Step 2 — Aggregate (always run, even with partial failures)
# ---------------------------------------------------------------------------
step "2 / Aggregate P2 results"

$PY gem/aggregate_gem_results.py --p2-corpus --width 0 --out-dir "${PAPER_OUT}" 2>&1 | tee "${PAPER_OUT}/aggregate.log"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
step "Summary"

echo "  Models complete at start : ${#ALREADY_DONE[@]}"
echo "  Models run this session  : ${#TO_RUN[@]}"
echo "  Passed                   : ${#PASSED_MODELS[@]}"
echo "  Failed                   : ${#FAILED_MODELS[@]}"
echo "  Results                  : ${PAPER_OUT}/gem_sweep_aggregate.md"

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo
    echo "  FAILED MODELS:"
    for mid in "${FAILED_MODELS[@]}"; do
        short="${mid##*/}"
        echo "    ${mid} — log: ${LOG_DIR}/${short//\//_}.log"
    done
    echo
    echo "  Re-run failed models with:"
    for mid in "${FAILED_MODELS[@]}"; do
        echo "    bash scripts/ablate_gem_p2.sh --model '${mid}'"
    done
    exit 1
fi

echo
echo "  All models complete. $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

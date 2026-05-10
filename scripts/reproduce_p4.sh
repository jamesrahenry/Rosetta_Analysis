#!/usr/bin/env bash
# reproduce_p4.sh — Paper 4 (PRH / Concept-Selective Convergence) end-to-end reproduction
#
# Extracts concept vectors for the PRH proxy corpus (19 models across 5 same-dim
# clusters), runs Procrustes alignment and all nulls, then runs the P5
# proportional-depth analysis and validation battery.
#
# Usage:
#   ./scripts/reproduce_p4.sh                       # full PRH proxy corpus
#   ./scripts/reproduce_p4.sh --quick               # Cluster A only (4 models, ~30 min)
#   ./scripts/reproduce_p4.sh --no-clean-cache      # keep HF cache (use on H200)
#   ./scripts/reproduce_p4.sh --with-frontier       # also extract Cluster F (H200 only)
#
# Requirements:
#   - NVIDIA GPU (≥16GB VRAM; H200 recommended for Cluster E and frontier)
#   - HF_TOKEN env var for gated models (Llama-3.1-8B, Mistral, Gemma-2)
#   - uv (https://docs.astral.sh/uv/) — used to manage the Python environment
#   - Rosetta_Concept_Pairs available (see concept pairs section in README)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
N_PAIRS=250
QUICK=false
NO_CLEAN_CACHE=false
GPU_ONLY=false
WITH_FRONTIER=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case $arg in
        --quick)          QUICK=true ;;
        --no-clean-cache) NO_CLEAN_CACHE=true ;;
        --gpu-only)       GPU_ONLY=true ;;
        --with-frontier)  WITH_FRONTIER=true ;;
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
PAPER_OUT="${HOME}/rosetta_data/results/PRH"
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
    print(f"  [WARNING] Only {vram_gb:.0f}GB VRAM — some models may OOM.")
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
# Step 1 — CAZ extraction: Cluster A proof-of-concept (smallest same-dim group)
# ---------------------------------------------------------------------------
step "1 / CAZ extraction — Cluster A (4 models, 768-dim, N=${N_PAIRS} pairs)"
info "Skips models already extracted."

$PY extraction/extract.py \
    --prh-cluster A \
    --n-pairs "${N_PAIRS}" \
    ${CACHE_FLAG}

elapsed

if [ "${QUICK}" = true ]; then
    step "Quick: Procrustes alignment — Cluster A only"

    $PY alignment/align.py --all --same-dim-only \
        --out results/prh_main_clusterA.csv

    echo
    info "Quick run complete — Cluster A only."
    info "Run without --quick for full 19-model PRH proxy corpus."
    elapsed
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 2 — CAZ extraction: full PRH proxy corpus (19 models, all clusters)
# ---------------------------------------------------------------------------
step "2 / CAZ extraction — full PRH proxy corpus (19 models, N=${N_PAIRS} pairs)"
info "Skips models already extracted."

$PY extraction/extract.py \
    --prh-proxy \
    --n-pairs "${N_PAIRS}" \
    ${CACHE_FLAG}

elapsed

# ---------------------------------------------------------------------------
# Step 3 — Random calibration null (GPU — re-extracts random activations)
# ---------------------------------------------------------------------------
step "3 / Random calibration null — same-dim pairs"
info "Tests whether generic rotation explains PRH alignment (~8× SNR expected)."

$PY alignment/align_random_calib.py \
    --out "${PAPER_OUT}/prh_random_calib_null.json" \
    ${CACHE_FLAG}

elapsed

# ---------------------------------------------------------------------------
# Step 4 — P5 CKA extraction (GPU — requires model load)
# ---------------------------------------------------------------------------
step "4 / P5 CKA extraction — PRH proxy corpus"
info "Extracts adjacent-layer CKA at proportional depths {0.3, 0.5, 0.7}."

$PY alignment/p5/p5_cka_extract.py ${CACHE_FLAG}

elapsed

if [ "${GPU_ONLY}" = true ]; then
    END_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    TOTAL_MIN=$(( ($(date +%s) - START_TS) / 60 ))
    $PY - <<PYPROV
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from extraction.extract import PRH_PROXY_MODELS, PRH_FRONTIER_MODELS
from rosetta_tools.paths import ROSETTA_MODELS
corpus = PRH_PROXY_MODELS + (PRH_FRONTIER_MODELS if "${WITH_FRONTIER}" == "true" else [])
def _slug(mid): return mid.replace("/", "_").replace("-", "_")
revisions = {}
for model_id in corpus:
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
    print(f"  Provenance: {len(revisions)}/{len(corpus)} revision hashes — {prov_path}")
PYPROV
    echo
    echo "  --gpu-only: extraction + random calib + P5 CKA complete. ${END_UTC} (${TOTAL_MIN}m)"
    echo "  Run without --gpu-only for Procrustes, alignment nulls, and P5 analysis."
    exit 0
fi

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
# Step 7 — P5 proportional-depth analysis (CPU)
# ---------------------------------------------------------------------------
step "7 / P5 proportional-depth alignment analysis"
info "Expected: matched 0.331 vs mismatched 0.198, Δ=+0.134, 98/98, p=1.2×10⁻³⁰."

$PY alignment/p5/p5_propdepth.py --out-dir "${PAPER_OUT}/p5"

elapsed

# ---------------------------------------------------------------------------
# Step 8 — P5 CKA analysis (CPU)
# ---------------------------------------------------------------------------
step "8 / P5 CKA analysis"

$PY alignment/p5/p5_cka_analyze.py --out-dir "${PAPER_OUT}/p5"

elapsed

# ---------------------------------------------------------------------------
# Step 9 — P5 validation battery (CPU — nulls for the P5 result)
# ---------------------------------------------------------------------------
step "9 / P5 validation battery (random-input, structure-scramble, procedure-off nulls)"

$PY alignment/p5/p5_validation_battery.py --out-dir "${PAPER_OUT}/p5"

elapsed

# ---------------------------------------------------------------------------
# Optional: Cluster F frontier models (H200 only)
# ---------------------------------------------------------------------------
if [ "${WITH_FRONTIER}" = true ]; then
    step "10 / CAZ extraction — Cluster F frontier (falcon-40b, Llama-70B, Qwen-72B)"
    info "H200 only — requires ~140GB VRAM for 70B models."

    $PY extraction/extract.py \
        --prh-frontier \
        --n-pairs "${N_PAIRS}" \
        ${CACHE_FLAG}

    elapsed

    step "11 / Procrustes alignment — including frontier models"

    $PY alignment/align.py --all --same-dim-only \
        --out "${PAPER_OUT}/prh_with_frontier.csv"

    elapsed
fi

END_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
TOTAL_MIN=$(( ($(date +%s) - START_TS) / 60 ))

$PY - <<PYPROV
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from extraction.extract import PRH_PROXY_MODELS, PRH_FRONTIER_MODELS
from rosetta_tools.paths import ROSETTA_MODELS

corpus = PRH_PROXY_MODELS + (PRH_FRONTIER_MODELS if "${WITH_FRONTIER}" == "true" else [])

def _slug(mid): return mid.replace("/", "_").replace("-", "_")

revisions = {}
for model_id in corpus:
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
    missing = [m for m in corpus if m not in revisions]
    print(f"  Provenance: {len(revisions)}/{len(corpus)} revision hashes")
    if missing:
        print(f"  [WARN] missing: {', '.join(m.split('/')[-1] for m in missing)}", file=sys.stderr)
PYPROV

echo
echo "  Paper 4 reproduction complete. Completed: ${END_UTC} (${TOTAL_MIN}m)"
echo "  Primary result:       ${PAPER_OUT}/prh_main.csv"
echo "  Permuted-label null:  rosetta_data/results/null_permuted_<concept>.csv"
echo "  Cross-concept:        rosetta_data/results/cross_concept_transfer.csv"
echo "  Random calib null:    ${PAPER_OUT}/prh_random_calib_null.json"
echo "  P5 depth result:      ${PAPER_OUT}/p5/p5_propdepth_samedim_results.json"
echo "  Provenance:           ${PAPER_OUT}/provenance.json"

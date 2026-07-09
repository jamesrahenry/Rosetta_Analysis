#!/usr/bin/env python3
"""
prh_permuted_null_percluster_gpu.py — GPU-accelerated per-dimension-cluster
permuted-label null for P4 §4.3 (width-confounder rebuttal).

Same protocol and CLI as the CPU version (`prh_permuted_null_percluster.py`
in this directory): for each same-hidden-dimension cluster, enumerate
cross-family ordered model pairs, fit the real Procrustes rotation R per
(pair, concept) on peak-layer calibration activations, then run k_perm=25
label-permutation trials per (pair, concept) and report the pooled null
mean/sd for that dimension.

Why this script exists: the CPU version's Procrustes fit is a dense SVD of a
(hidden_dim x hidden_dim) matrix — cheap at the primary corpus's smaller
dimensions, but O(d^3) and genuinely expensive at Cluster E's 5120-dim
(observed: 68 (pair,concept) fits still not complete after 1h+ of wall time
on a CPU-only dev machine at ~1.8 cores of BLAS parallelism). SVD is one of
the operations that GPUs accelerate dramatically; this script does the exact
same fit via `torch.linalg.svd` on CUDA instead of `scipy.linalg.svd` on CPU.

Do not use this as a replacement for the CPU script in general — it exists
specifically for expensive high-dimension clusters (Cluster E, and Cluster F
if that's ever revisited). For the cheaper clusters already in
`p4_permuted_null_percluster.json` (768–4096-dim), the CPU version is fine.

USAGE (see bottom of this file / the paper's shared/GPU_NEEDS_P3_P4.md for
the full runbook):

    # Sanity check first — reproduces an existing, already-published cluster
    # (Cluster A, 768-dim) so you can trust the GPU fit before spending time
    # on Cluster E. Should complete in well under a minute on any GPU.
    python alignment/prh_permuted_null_percluster_gpu.py --validate A

    # The actual job: Cluster E (5120-dim), the one item this script exists for.
    python alignment/prh_permuted_null_percluster_gpu.py --cluster E \
        --models EleutherAI_pythia_12b,Qwen_Qwen2.5_14B,Qwen_Qwen2.5_32B \
        --dim 5120 \
        --out /path/to/papers/prh-validation/scripts/results/p4_null_clusterE.json

Requires: torch (CUDA build), huggingface_hub, rosetta_tools. Downloads
calibration_<concept>.npy (falling back to slicing calibration_alllayer_<concept>.npy
at the concept's peak layer) from the james-ra-henry/Rosetta-Activations HF
dataset on demand for any file not already local.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from rosetta_tools.gpu_utils import get_device

DATA_ROOT = Path.home() / "rosetta_data" / "paper_n250"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

HF_REPO = "james-ra-henry/Rosetta-Activations"
PAPER_PREFIX = "paper_n250"

CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

K_PERM = 25
SEED = 42

# Reference: existing 6-cluster table this script's output should sit alongside
# (papers/prh-validation/scripts/results/p4_permuted_null_percluster.json).
VALIDATION_CLUSTERS = {
    "A": {"models": ["openai_community_gpt2", "EleutherAI_pythia_160m", "facebook_opt_125m"],
          "dim": 768, "expected_n": 102, "expected_mean": -0.0024195066036761443,
          "expected_sd": 0.22661633237026654},
}


def get_family(name: str) -> str:
    n = name.lower()
    if "pythia" in n: return "pythia"
    if "gpt_neo" in n: return "gpt_neo"
    if "opt_" in n: return "opt"
    if "gpt2" in n: return "gpt2"
    if "gemma_2" in n: return "gemma2"
    if "llama_3.1" in n: return "llama31"
    if "llama_3.2" in n: return "llama32"
    if "phi" in n: return "phi"
    if "mistral" in n: return "mistral"
    if "qwen" in n: return "qwen25"
    if "falcon" in n: return "falcon"
    return "unknown"


def _ensure_file(model_dir: str, fname: str) -> Path:
    local = DATA_ROOT / model_dir / fname
    if local.exists():
        return local
    log.info("  downloading %s/%s from HF...", model_dir, fname)
    p = hf_hub_download(HF_REPO, f"{PAPER_PREFIX}/{model_dir}/{fname}", repo_type="dataset",
                         local_dir=str(DATA_ROOT.parent))
    return Path(p)


def load_peak_calibration(model_dir: str, concept: str) -> np.ndarray | None:
    """Load (500, hidden_dim) peak-layer calibration activations, downloading if needed."""
    try:
        plain = _ensure_file(model_dir, f"calibration_{concept}.npy")
        return np.load(plain).astype(np.float32)
    except Exception:
        pass
    try:
        caz_path = _ensure_file(model_dir, f"caz_{concept}.json")
        caz = json.load(open(caz_path))
        peak_layer = caz["layer_data"]["peak_layer"]
        alllayer_path = _ensure_file(model_dir, f"calibration_alllayer_{concept}.npy")
        arr = np.load(alllayer_path).astype(np.float32)
        return arr[peak_layer]
    except Exception as e:
        log.warning("  no calibration data for %s/%s: %s", model_dir, concept, e)
        return None


def compute_procrustes_rotation_gpu(source_acts: np.ndarray, target_acts: np.ndarray, device: str) -> torch.Tensor:
    """Same-dim orthogonal Procrustes: R minimising ||target @ R - source||_F, via GPU SVD.

    Mirrors rosetta_tools.alignment.compute_procrustes_rotation exactly for the
    same-hidden-dimension case (no PCA branch needed — Cluster E is same-dim by
    construction), just computed on CUDA instead of CPU/scipy.
    """
    source = torch.as_tensor(source_acts, dtype=torch.float32, device=device)
    target = torch.as_tensor(target_acts, dtype=torch.float32, device=device)

    n = min(source.shape[0], target.shape[0])
    source = source[:n] - source[:n].mean(dim=0, keepdim=True)
    target = target[:n] - target[:n].mean(dim=0, keepdim=True)

    # orthogonal_procrustes(A, B) finds R minimising ||A @ R - B||; A=target, B=source
    M = target.T @ source
    U, _, Vt = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    return R


def permuted_dom_vector(cal_acts: torch.Tensor, rng: np.random.Generator, device: str) -> torch.Tensor:
    """Same convention as prh_permuted_null_streaming.py's _permuted_dom_vector."""
    n = cal_acts.shape[0]
    n_pos = n // 2
    perm = rng.permutation(n)
    perm_t = torch.as_tensor(perm, device=device)
    pos = cal_acts[perm_t[:n_pos]]
    neg = cal_acts[perm_t[n_pos:]]
    diff = pos.mean(dim=0) - neg.mean(dim=0)
    norm = torch.linalg.norm(diff)
    return diff / (norm + 1e-10)


def cosine_similarity_t(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(torch.dot(a, b) / denom)


def run_cluster_null(models: list[str], dim: int, rng: np.random.Generator, device: str) -> dict:
    families = {m: get_family(m) for m in models}
    ordered_pairs = [(a, b) for a in models for b in models if a != b and families[a] != families[b]]

    all_trials: list[float] = []
    n_pair_concepts = 0
    n_failed = 0
    failed = []

    for src, tgt in ordered_pairs:
        for concept in CONCEPTS:
            sa = load_peak_calibration(src, concept)
            ta = load_peak_calibration(tgt, concept)
            if sa is None or ta is None:
                n_failed += 1
                failed.append({"source": src, "target": tgt, "concept": concept, "reason": "missing calibration"})
                continue
            try:
                R = compute_procrustes_rotation_gpu(sa, ta, device)
            except Exception as e:
                n_failed += 1
                failed.append({"source": src, "target": tgt, "concept": concept, "reason": str(e)})
                continue

            sa_t = torch.as_tensor(sa, dtype=torch.float32, device=device)
            ta_t = torch.as_tensor(ta, dtype=torch.float32, device=device)

            n_pair_concepts += 1
            log.info("  [%d/%d] %s -> %s / %s", n_pair_concepts, len(ordered_pairs) * len(CONCEPTS),
                      src, tgt, concept)
            for _ in range(K_PERM):
                perm_src = permuted_dom_vector(sa_t, rng, device)
                perm_tgt = permuted_dom_vector(ta_t, rng, device)
                rotated = perm_tgt @ R
                all_trials.append(cosine_similarity_t(perm_src, rotated))

    arr = np.array(all_trials)
    return {
        "dim": dim,
        "models": models,
        "ordered_pairs": [f"{a}->{b}" for a, b in ordered_pairs],
        "dn_ratio": round(dim / 500, 3),
        "n_pair_concepts": n_pair_concepts,
        "n_trials": len(all_trials),
        "n_failed": n_failed,
        "failed": failed,
        "mean": float(arr.mean()) if len(arr) else None,
        "sd": float(arr.std()) if len(arr) else None,
        "device": device,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", type=str, default=None,
                     help="Reproduce a known cluster (e.g. 'A') and compare to the stored 6-cluster table")
    ap.add_argument("--cluster", type=str, default=None, help="Cluster label, e.g. 'E'")
    ap.add_argument("--models", type=str, default=None, help="Comma-separated model dir names")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", type=str, default="auto", help="'cuda', 'cpu', or 'auto'")
    args = ap.parse_args()

    device = get_device(args.device) if args.device != "auto" else get_device("auto")
    if device == "cpu":
        log.warning("No GPU detected — falling back to CPU. This script exists specifically "
                    "to avoid the slow CPU path; if you're seeing this on the intended GPU "
                    "host, check the torch/CUDA install before running the real job.")
    log.info("Using device: %s", device)

    rng = np.random.default_rng(SEED)

    if args.validate:
        spec = VALIDATION_CLUSTERS[args.validate]
        log.info("Validating against known Cluster %s (dim=%d, expected n=%d)...",
                  args.validate, spec["dim"], spec["expected_n"])
        result = run_cluster_null(spec["models"], spec["dim"], rng, device)
        log.info("Got n_pair_concepts=%d (expected %d)", result["n_pair_concepts"], spec["expected_n"])
        log.info("Got mean=%.6f sd=%.6f (expected mean=%.6f sd=%.6f — exact match not expected, "
                  "this is a stochastic permutation test with a different draw order; near-zero "
                  "mean and similar-magnitude sd is the bar to clear)",
                  result["mean"], result["sd"], spec["expected_mean"], spec["expected_sd"])
        print(json.dumps(result, indent=2))
        return

    if not (args.cluster and args.models and args.dim):
        ap.error("--cluster, --models, and --dim are required unless --validate is given")

    models = [m.strip() for m in args.models.split(",")]
    result = run_cluster_null(models, args.dim, rng, device)
    result["cluster"] = args.cluster
    print(json.dumps(result, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)
        log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()

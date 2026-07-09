#!/usr/bin/env python3
"""
prh_permuted_null_percluster.py — Per-dimension-cluster permuted-label null
for P4 §4.3 (width-confounder rebuttal).

Reproduces the protocol described in the paper: for each same-hidden-dimension
cluster, enumerate cross-family ordered model pairs, fit the real Procrustes
rotation R per (pair, concept) on peak-layer calibration activations, then
run k_perm=25 label-permutation trials per (pair, concept) and report the
pooled null mean/sd for that dimension.

Reuses rosetta_tools.alignment.compute_procrustes_rotation / apply_rotation /
cosine_similarity — the same primary-pipeline helpers used by
prh_permuted_null_streaming.py (the pooled cross-cluster null) — so results
are directly comparable to the existing `p4_permuted_null_percluster.json`
6-cluster table. The exact original script that produced that file could not
be located in the repo; this is a from-scratch reimplementation validated by
reproducing an existing cluster's stored stats before trusting it on new data
(see --validate).

Peak-layer calibration activations: tries `calibration_<concept>.npy` first
(500 x hidden_dim), falling back to slicing `calibration_alllayer_<concept>.npy`
(n_layers x 500 x hidden_dim) at the concept's peak_layer from caz_<concept>.json
when the plain file isn't present locally (downloads from HF on demand either way).

Usage:
    python alignment/prh_permuted_null_percluster.py --validate A
    python alignment/prh_permuted_null_percluster.py --cluster E \
        --models EleutherAI_pythia_12b,Qwen_Qwen2.5_14B,Qwen_Qwen2.5_32B \
        --dim 5120 --out results/p4_null_clusterE.json
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import permutations
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from rosetta_tools.alignment import compute_procrustes_rotation, apply_rotation, cosine_similarity
from rosetta_tools.paths import ROSETTA_RESULTS

try:
    DATA_ROOT = Path.home() / "rosetta_data" / "paper_n250"
except Exception:  # pragma: no cover
    DATA_ROOT = None

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
        return np.load(plain).astype(np.float64)
    except Exception:
        pass
    # Fall back: slice the all-layer array at the concept's peak layer
    try:
        caz_path = _ensure_file(model_dir, f"caz_{concept}.json")
        caz = json.load(open(caz_path))
        peak_layer = caz["layer_data"]["peak_layer"]
        alllayer_path = _ensure_file(model_dir, f"calibration_alllayer_{concept}.npy")
        arr = np.load(alllayer_path).astype(np.float64)
        return arr[peak_layer]
    except Exception as e:
        log.warning("  no calibration data for %s/%s: %s", model_dir, concept, e)
        return None


def permuted_dom_vector(cal_acts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same convention as prh_permuted_null_streaming.py's _permuted_dom_vector."""
    n = cal_acts.shape[0]
    n_pos = n // 2
    perm = rng.permutation(n)
    pos = cal_acts[perm[:n_pos]]
    neg = cal_acts[perm[n_pos:]]
    diff = pos.mean(axis=0) - neg.mean(axis=0)
    norm = np.linalg.norm(diff)
    return diff / (norm + 1e-10)


def run_cluster_null(models: list[str], dim: int, rng: np.random.Generator) -> dict:
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
                R = compute_procrustes_rotation(sa, ta)
            except Exception as e:
                n_failed += 1
                failed.append({"source": src, "target": tgt, "concept": concept, "reason": str(e)})
                continue

            n_pair_concepts += 1
            for _ in range(K_PERM):
                perm_src = permuted_dom_vector(sa, rng)
                perm_tgt = permuted_dom_vector(ta, rng)
                rotated = apply_rotation(perm_tgt, R)
                all_trials.append(cosine_similarity(perm_src, rotated))

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
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", type=str, default=None,
                     help="Reproduce a known cluster (e.g. 'A') and compare to the stored 6-cluster table")
    ap.add_argument("--cluster", type=str, default=None, help="Cluster label, e.g. 'E'")
    ap.add_argument("--models", type=str, default=None, help="Comma-separated model dir names")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)

    if args.validate:
        spec = VALIDATION_CLUSTERS[args.validate]
        log.info("Validating against known Cluster %s (dim=%d, expected n=%d)...",
                  args.validate, spec["dim"], spec["expected_n"])
        result = run_cluster_null(spec["models"], spec["dim"], rng)
        log.info("Got n_pair_concepts=%d (expected %d)", result["n_pair_concepts"], spec["expected_n"])
        log.info("Got mean=%.6f sd=%.6f (expected mean=%.6f sd=%.6f)",
                  result["mean"], result["sd"], spec["expected_mean"], spec["expected_sd"])
        print(json.dumps(result, indent=2))
        return

    if not (args.cluster and args.models and args.dim):
        ap.error("--cluster, --models, and --dim are required unless --validate is given")

    models = [m.strip() for m in args.models.split(",")]
    result = run_cluster_null(models, args.dim, rng)
    result["cluster"] = args.cluster
    print(json.dumps(result, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)
        log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()

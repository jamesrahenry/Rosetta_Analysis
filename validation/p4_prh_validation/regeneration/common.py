"""
common.py — shared core for the P4 (PRH / Concept-Selective Convergence) data
regeneration package.

This package regenerates every headline P4 quantity from the *published* HF
activation artifacts, with no dependence on the older top-level `align.py`
pipeline: it carries its own authoritative 33-model A–F roster, its own
Procrustes implementation (validated identical to the full-SVD reference to
machine precision), and its own memory-safe HF loaders. It is self-validating
— `step1` reproduces a known concept from stored artifacts to <1e-8 before it
trusts any number.

Data source of record: HuggingFace dataset `james-ra-henry/Rosetta-Activations`,
tree `paper_n250/<model_slug>/`:
  - caz_<concept>.json          — peak layer + DOM (difference-of-means) vector
  - calibration_<concept>.npy   — peak-layer calibration activations [n, d]
                                  (present for clusters A–E; for cluster F only
                                   the all-layer array below is stored)
  - calibration_alllayer_<concept>.npy — [n_layers, n, d] all-layer calibration

Corpus state reproduced here (as of 2026-07-18):
  - exfiltration labels CORRECTED (recorded-draw reconstruction, N=249; the
    label-inversion fix, ~72% of pos/neg swapped, now applied on HF)
  - cluster F (falcon-40b, Llama-3.1-70B, Qwen2.5-72B) FOLDED INTO the primary
    corpus → A–F, 33 models (F's 70–72B calibration is 8-bit-derived; disclosed)

Written 2026-07-18 for the P4 correction + A–F fold.
"""
from __future__ import annotations
import json
import gc
import os
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes, svd as _scipy_svd

HF_REPO = "james-ra-henry/Rosetta-Activations"
HF_ROOT = "paper_n250"

CONCEPTS_17 = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

# ---------------------------------------------------------------------------
# Authoritative A–F roster: slug -> (family, hidden_dim).
# Derived directly from the published primary CSV (its fam_s/fam_t columns are
# the family labels the cross-family filter actually used). The same-dim
# clusters are: 768(A) 1024(G) 2048(B) 2560(H) 3584(D) 4096(C) 5120(E) 8192(F).
# ---------------------------------------------------------------------------
ROSTER: dict[str, tuple[str, int]] = {
    "openai_community_gpt2": ("gpt2", 768),
    "EleutherAI_gpt_neo_125m": ("gpt_neo", 768),
    "EleutherAI_pythia_160m": ("pythia", 768),
    "facebook_opt_125m": ("opt_", 768),
    "openai_community_gpt2_medium": ("gpt2", 1024),
    "facebook_opt_350m": ("opt_", 1024),
    "EleutherAI_pythia_410m": ("pythia", 1024),
    "EleutherAI_pythia_1b": ("pythia", 2048),
    "EleutherAI_pythia_1.4b": ("pythia", 2048),
    "facebook_opt_1.3b": ("opt_", 2048),
    "meta_llama_Llama_3.2_1B": ("llama_3.2", 2048),
    "meta_llama_Llama_3.2_1B_Instruct": ("llama_3.2", 2048),
    "Qwen_Qwen2.5_3B": ("qwen", 2048),
    "Qwen_Qwen2.5_3B_Instruct": ("qwen", 2048),
    "EleutherAI_pythia_2.8b": ("pythia", 2560),
    "facebook_opt_2.7b": ("opt_", 2560),
    "microsoft_phi_2": ("phi", 2560),
    "Qwen_Qwen2.5_7B": ("qwen", 3584),
    "Qwen_Qwen2.5_7B_Instruct": ("qwen", 3584),
    "google_gemma_2_9b": ("gemma_2", 3584),
    "google_gemma_2_9b_it": ("gemma_2", 3584),
    "EleutherAI_pythia_6.9b": ("pythia", 4096),
    "facebook_opt_6.7b": ("opt_", 4096),
    "meta_llama_Llama_3.1_8B": ("llama_3.1", 4096),
    "meta_llama_Llama_3.1_8B_Instruct": ("llama_3.1", 4096),
    "mistralai_Mistral_7B_v0.3": ("mistral", 4096),
    "mistralai_Mistral_7B_Instruct_v0.3": ("mistral", 4096),
    "EleutherAI_pythia_12b": ("pythia", 5120),
    "Qwen_Qwen2.5_14B": ("qwen", 5120),
    "Qwen_Qwen2.5_32B": ("qwen", 5120),
    # Cluster F (8192) — 8-bit calibration on the two 70–72B models (see module docstring)
    "tiiuae_falcon_40b": ("falcon", 8192),
    "meta_llama_Llama_3.1_70B": ("llama3", 8192),
    "Qwen_Qwen2.5_72B": ("qwen2", 8192),
}
CLUSTER_F = {"tiiuae_falcon_40b", "meta_llama_Llama_3.1_70B", "Qwen_Qwen2.5_72B"}


def cross_family_same_dim_pairs(slugs=None) -> list[tuple[str, str]]:
    """Ordered (source, target) pairs: same hidden_dim, different family."""
    slugs = list(ROSTER) if slugs is None else slugs
    out = []
    for s in slugs:
        for t in slugs:
            if s == t:
                continue
            if ROSTER[s][1] != ROSTER[t][1]:
                continue
            if ROSTER[s][0] == ROSTER[t][0]:
                continue
            out.append((s, t))
    return out


# ---------------------------------------------------------------------------
# HF loaders (memory- and disk-safe). Peak-layer calibration is used directly
# where stored (A–E); for cluster F it is sliced out of the all-layer array via
# mmap so the ~900MB file never fully enters RAM, and the download is removed
# after slicing so disk stays bounded.
# ---------------------------------------------------------------------------
def _hf(path, **kw):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, path, repo_type="dataset", **kw)


def load_dom_and_peak(slug: str, concept: str):
    """Returns (dom_vector float64 [d], peak_layer int). None dom if degenerate."""
    caz = json.load(open(_hf(f"{HF_ROOT}/{slug}/caz_{concept}.json")))
    ld = caz["layer_data"]
    peak = ld["peak_layer"]
    pm = next(m for m in ld["metrics"] if m["layer"] == peak)
    dom = np.asarray(pm["dom_vector"], np.float64) if "dom_vector" in pm else None
    return dom, peak


def load_calibration(slug: str, concept: str, peak: int, stage_dir: str | None = None):
    """Peak-layer calibration activations [n, d], float64.

    Uses the stored peak-layer file when present (A–E); otherwise slices the
    peak layer out of the all-layer array (cluster F) via mmap and deletes the
    downloaded all-layer file afterwards. `stage_dir` (a real dir on a disk
    with room) is where big all-layer files are staged for F; defaults to
    $P4_REGEN_STAGE or ./_p4_stage.
    """
    peak_path = f"{HF_ROOT}/{slug}/calibration_{concept}.npy"
    try:
        return np.load(_hf(peak_path)).astype(np.float64)
    except Exception:
        pass  # not stored at peak-layer granularity (cluster F) — slice all-layer
    stage = Path(stage_dir or os.environ.get("P4_REGEN_STAGE", "./_p4_stage"))
    stage.mkdir(parents=True, exist_ok=True)
    big = _hf(f"{HF_ROOT}/{slug}/calibration_alllayer_{concept}.npy", local_dir=str(stage))
    arr = np.load(big, mmap_mode="r")
    cal = np.array(arr[peak], dtype=np.float64)
    del arr
    gc.collect()
    try:
        os.remove(big)
    except OSError:
        pass
    return cal


# ---------------------------------------------------------------------------
# Procrustes. `aligned_cosine` uses a rank-reduced projection that is provably
# identical to the full-dimension orthogonal Procrustes for this problem: the
# DOM vectors lie in the ≤n-dimensional row-space of the (mean-centred)
# calibration matrix, so the arbitrary null-space part of R never touches them.
# For an 8192-dim model this turns a full 8192×8192 SVD (~30s) into a ~500×500
# one (<1s) with no loss of precision. `aligned_cosine_full` is the reference
# implementation used to validate this (see step1 --validate).
# ---------------------------------------------------------------------------
def cosine(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if d < 1e-12 else float(a @ b / d)


def _procrustes_R(t, s):
    """R such that t @ R ≈ s (both already mean-centred), with the gesvd
    fallback the round-3 GPU battery standardised on for rank-deficient SVDs."""
    try:
        R, _ = orthogonal_procrustes(t, s)
    except np.linalg.LinAlgError:
        u, _, vt = _scipy_svd(t.T @ s, lapack_driver="gesvd")
        R = u @ vt
    return R


def aligned_cosine_full(dom_s, dom_t, cal_s, cal_t) -> float:
    """Reference: full-dimension orthogonal Procrustes."""
    sc = np.asarray(cal_s, np.float64); sc = sc - sc.mean(0)
    tc = np.asarray(cal_t, np.float64); tc = tc - tc.mean(0)
    R = _procrustes_R(tc, sc)
    return cosine(dom_s, np.asarray(dom_t, np.float64).ravel() @ R)


def aligned_cosine(dom_s, dom_t, cal_s, cal_t) -> float:
    """Rank-reduced Procrustes — identical result, ~1000× cheaper at high d."""
    sc = np.asarray(cal_s, np.float64); sc = sc - sc.mean(0)
    tc = np.asarray(cal_t, np.float64); tc = tc - tc.mean(0)
    Q, _ = np.linalg.qr(np.hstack([tc.T, sc.T]))     # d × ≤2n orthonormal, spans both row-spaces
    Rq = _procrustes_R(tc @ Q, sc @ Q)
    return cosine(dom_s @ Q, (np.asarray(dom_t, np.float64).ravel() @ Q) @ Rq)


def raw_cosine(dom_s, dom_t) -> float:
    return cosine(dom_s, dom_t)

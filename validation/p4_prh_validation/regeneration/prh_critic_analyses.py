#!/usr/bin/env python3
"""P4 critic-engagement analyses — OLS-universality + null-calibrated CKA.

Two reviewer-rebuttal analyses run under the SAME protocol as the primary
zero-PCA Procrustes alignment (cross-family, same-hidden-dim ordered pairs;
peak-layer calibration activations; n=250 pairs -> 500 class-balanced rows).

1. OLS-UNIVERSALITY (adjudicates Huang et al. 2025).
   Huang report OLS transforms are ~concept-independent (near-universal).
   P4 reports orthogonal-Procrustes universality ratio 0.194 (concept-specific).
   We fit BOTH an orthogonal Procrustes R and an unconstrained OLS map W on each
   concept's peak-layer calibration activations, then build the cross-concept
   transfer matrix for each and report both universality ratios on the same data.
   Resolution predicted: OLS ratio >> Procrustes ratio because OLS in the n<d
   regime is underdetermined and can map almost anything onto anything — i.e.
   the "near-universality" Huang see is an artifact of the unconstrained map's
   expressivity, not of shared geometry. The constrained (isometry) test is the
   honest one.

2. NULL-CALIBRATED CKA (adjudicates Groger et al. 2026).
   Groger argue global spectral similarity (CKA/SVCCA) is confounded and that
   after permutation null-calibration cross-architecture convergence largely
   vanishes (models share local topology, not global metric structure).
   We compute linear CKA between the two models' peak-layer activation matrices
   (same input sentences -> rows correspond) for each concept, plus a row-
   permutation null distribution (shuffle the example correspondence). We report
   real CKA, null mean/95th-pct, and a z-style separation, per pair and grand.

Both analyses iterate the full 17 concepts (data is complete for caz/DOM and
calibration activations on HF), so they also advance the N=250 / C=17 goal.

Usage:
  python prh_critic_analyses.py --cluster A            # 768-dim 4-family testbed
  python prh_critic_analyses.py --dims 768,2048        # specific dim clusters
  python prh_critic_analyses.py --models a,b,c         # explicit model dir names
Outputs JSON to rosetta_data/results/PRH/critic_analyses_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "rosetta_data" / "paper_n250"
OUT_DIR = REPO_ROOT / "rosetta_data" / "results" / "PRH"
HF_REPO = "james-ra-henry/Rosetta-Activations"
HF_REV = "paper-n250"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("prh_critic")

CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

# Cluster A = 768-dim, 4 distinct families (the cleanest cross-family testbed).
CLUSTERS = {
    "A": ["openai_community_gpt2", "EleutherAI_pythia_160m",
          "facebook_opt_125m", "EleutherAI_gpt_neo_125m"],
}
RNG_SEED = 42
N_PERM = 200  # CKA null-calibration permutations (ample for null mean/p95/sd)
N_ROWS_FULL = 500  # 250 pos + 250 neg; concepts with fewer dropped examples (skip)


def get_family(name: str) -> str:
    n = name.lower()
    if "pythia" in n:     return "pythia"
    if "gpt_neo" in n:    return "gpt_neo"
    if "opt_" in n:       return "opt"
    if "gpt2" in n:       return "gpt2"
    if "gemma_2" in n:    return "gemma2"
    if "llama_3.1" in n:  return "llama31"
    if "llama_3.2" in n:  return "llama32"
    if "phi" in n:        return "phi"
    if "mistral" in n:    return "mistral"
    if "qwen" in n:       return "qwen25"
    if "falcon" in n:     return "falcon"
    return "unknown"


ALLOW_DOWNLOAD = True


def _ensure_file(model: str, fname: str) -> Path:
    """Return local path to a per-model artifact, downloading from HF if absent."""
    local = DATA_ROOT / model / fname
    if local.exists():
        return local
    if not ALLOW_DOWNLOAD:
        raise FileNotFoundError(f"{local} (HF download disabled with --data-root)")
    from huggingface_hub import hf_hub_download
    log.info("  downloading %s/%s from HF ...", model, fname)
    got = hf_hub_download(HF_REPO, filename=f"paper_n250/{model}/{fname}",
                          repo_type="dataset", revision=HF_REV,
                          local_dir=str(DATA_ROOT.parent))
    return Path(got)


def load_model(model: str) -> dict | None:
    """Load peak-layer DOM vectors + peak-layer calibration activations (17 concepts)."""
    out = {"model": model, "family": get_family(model), "concepts": {}}
    hidden_dim = None
    for c in CONCEPTS:
        try:
            caz = json.loads(_ensure_file(model, f"caz_{c}.json").read_text())
        except Exception as e:
            log.warning("  %s/%s caz missing: %s", model, c, e)
            continue
        ld = caz["layer_data"]
        peak = ld["peak_layer"]
        dom = np.asarray(ld["metrics"][peak]["dom_vector"], dtype=np.float64)
        dom /= max(np.linalg.norm(dom), 1e-12)
        try:
            acts_all = np.load(_ensure_file(model, f"calibration_alllayer_{c}.npy"))
        except Exception as e:
            log.warning("  %s/%s alllayer npy missing: %s", model, c, e)
            continue
        acts = np.asarray(acts_all[peak], dtype=np.float64)  # [n_rows, d]
        hidden_dim = acts.shape[1]
        # Require the full 500 rows (250 pos + 250 neg). A few concepts dropped
        # examples during extraction, and different models dropped DIFFERENT ones,
        # which would silently misalign the paired rows that Procrustes/OLS/CKA
        # depend on. Restricting to fully-extracted concepts guarantees the same
        # corpus in the same order -> row correspondence holds across models.
        if acts.shape[0] != N_ROWS_FULL:
            out.setdefault("dropped_incomplete", []).append((c, int(acts.shape[0])))
            continue
        out["concepts"][c] = {"peak": peak, "dom": dom, "acts": acts}
    out["hidden_dim"] = hidden_dim
    if not out["concepts"]:
        return None
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def fit_procrustes(src_acts: np.ndarray, tgt_acts: np.ndarray) -> np.ndarray:
    """Orthogonal R minimizing ||src_acts @ R - tgt_acts||  (source -> target)."""
    R, _ = orthogonal_procrustes(src_acts, tgt_acts)
    return R


def fit_ols(src_acts: np.ndarray, tgt_acts: np.ndarray) -> np.ndarray:
    """Unconstrained least-squares W s.t. src_acts @ W ~= tgt_acts (source -> target)."""
    W, *_ = np.linalg.lstsq(src_acts, tgt_acts, rcond=None)
    return W


def _gram(X: np.ndarray) -> np.ndarray:
    """Feature-centered linear Gram matrix K = Xc @ Xc.T  ([n, n])."""
    Xc = X - X.mean(0, keepdims=True)
    return Xc @ Xc.T


def linear_cka_from_grams(K: np.ndarray, L: np.ndarray) -> float:
    """Linear CKA via Gram matrices: <K,L>_F / (||K||_F ||L||_F).

    Identity used: ||Xc^T Yc||_F^2 = <K, L>_F and ||Xc^T Xc||_F = ||K||_F,
    so this equals the feature-space CKA exactly but lets the row-permutation
    null reuse K and L (||L||_F is permutation-invariant)."""
    nk = np.linalg.norm(K)
    nl = np.linalg.norm(L)
    if nk < 1e-12 or nl < 1e-12:
        return float("nan")
    return float(np.sum(K * L) / (nk * nl))


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    return linear_cka_from_grams(_gram(X), _gram(Y))


def transfer_matrix(A: dict, B: dict, fitter) -> np.ndarray:
    """concepts x concepts aligned-cosine matrix.

    M[i, j] = cos( dom_A[c_j] @ T_{c_i},  dom_B[c_j] ), where T_{c_i} is the
    map (Procrustes or OLS) fit on concept c_i's peak-layer activations (A->B).
    Diagonal = same-concept (fit and eval on the same concept). Off-diagonal =
    cross-concept transfer (map fit on one concept, applied to another's DOM).
    """
    cs = [c for c in CONCEPTS if c in A["concepts"] and c in B["concepts"]]
    M = np.full((len(cs), len(cs)), np.nan)
    maps = {ci: fitter(A["concepts"][ci]["acts"], B["concepts"][ci]["acts"]) for ci in cs}
    for i, ci in enumerate(cs):
        T = maps[ci]
        for j, cj in enumerate(cs):
            dom_a = A["concepts"][cj]["dom"]
            dom_b = B["concepts"][cj]["dom"]
            M[i, j] = cosine(dom_a @ T, dom_b)
    return M, cs


def single_map_alignment(A: dict, B: dict, fitter) -> dict:
    """Huang-style test: fit ONE map on ALL concepts' peak activations stacked
    together, then measure mean per-concept DOM aligned cosine. Tests whether a
    single (orthogonal vs unconstrained-OLS) transform aligns every concept --
    i.e. 'is one map universal?' Stacking gives n=concepts*500 >> d, so OLS here
    is a properly-determined regression (unlike the per-concept fit where n<d)."""
    cs = [c for c in CONCEPTS if c in A["concepts"] and c in B["concepts"]]
    Xs = np.vstack([A["concepts"][c]["acts"] for c in cs])
    Xt = np.vstack([B["concepts"][c]["acts"] for c in cs])
    T = fitter(Xs, Xt)
    cosines = [cosine(A["concepts"][c]["dom"] @ T, B["concepts"][c]["dom"]) for c in cs]
    return {"mean_aligned_cosine": float(np.nanmean(cosines)), "n_concepts": len(cs)}


def universality_ratio(M: np.ndarray) -> dict:
    diag = np.diagonal(M)
    off = M[~np.eye(M.shape[0], dtype=bool)]
    same = float(np.nanmean(diag))
    cross = float(np.nanmean(off))
    return {"same_concept_mean": same, "cross_concept_mean": cross,
            "universality_ratio": (cross / same) if abs(same) > 1e-9 else float("nan")}


def cka_with_null(A: dict, B: dict, rng: np.random.Generator) -> dict:
    cs = [c for c in CONCEPTS if c in A["concepts"] and c in B["concepts"]]
    per_concept = []
    for c in cs:
        K = _gram(A["concepts"][c]["acts"])
        L = _gram(B["concepts"][c]["acts"])
        nk, nl = np.linalg.norm(K), np.linalg.norm(L)
        denom = nk * nl
        real = float(np.sum(K * L) / denom) if denom > 1e-12 else float("nan")
        # Row-permutation null: permuting the example correspondence of model B
        # permutes rows AND columns of its Gram matrix; ||L|| is unchanged.
        n = L.shape[0]
        null = np.empty(N_PERM)
        for k in range(N_PERM):
            p = rng.permutation(n)
            null[k] = np.sum(K * L[p][:, p]) / denom
        nm, ns = float(null.mean()), float(null.std())
        per_concept.append({
            "concept": c, "cka_real": real,
            "cka_null_mean": nm, "cka_null_p95": float(np.percentile(null, 95)),
            "cka_null_sd": ns,
            "z_over_null": (real - nm) / ns if ns > 1e-12 else float("nan"),
            "real_exceeds_all_null": bool(real > null.max()),
        })
    return {"per_concept": per_concept,
            "cka_real_mean": float(np.nanmean([p["cka_real"] for p in per_concept])),
            "cka_null_mean": float(np.nanmean([p["cka_null_mean"] for p in per_concept])),
            "frac_real_exceeds_all_null":
                float(np.mean([p["real_exceeds_all_null"] for p in per_concept]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", type=str, default="A", help="named cluster (A)")
    ap.add_argument("--models", type=str, default="", help="explicit comma-sep model dir names")
    ap.add_argument("--out-name", type=str, default="")
    ap.add_argument("--skip-cka", action="store_true",
                    help="skip the (slow) CKA null; compute only the universality metrics")
    ap.add_argument("--data-root", type=str, default="",
                    help="read activations from this root instead of paper_n250 (disables HF download)")
    args = ap.parse_args()

    global DATA_ROOT, ALLOW_DOWNLOAD
    if args.data_root:
        DATA_ROOT = Path(args.data_root).expanduser()
        ALLOW_DOWNLOAD = False

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        tag = "custom"
    else:
        models = CLUSTERS[args.cluster]
        tag = f"cluster{args.cluster}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    log.info("Loading %d models: %s", len(models), models)
    loaded = {}
    for m in models:
        d = load_model(m)
        if d is not None:
            loaded[m] = d
            log.info("  %s [%s, dim=%s] %d/17 concepts", m, d["family"],
                     d["hidden_dim"], len(d["concepts"]))

    pairs_out = []
    proc_ratios, ols_ratios = [], []
    proc_same, ols_same = [], []
    single_proc_all, single_ols_all = [], []
    cka_real_all, cka_null_all = [], []

    names = list(loaded)
    for a, b in permutations(names, 2):
        A, B = loaded[a], loaded[b]
        if A["hidden_dim"] != B["hidden_dim"]:
            continue
        if A["family"] == B["family"]:
            continue  # cross-family only, per primary protocol
        Mp, cs = transfer_matrix(A, B, fit_procrustes)
        Mo, _ = transfer_matrix(A, B, fit_ols)
        up = universality_ratio(Mp)
        uo = universality_ratio(Mo)
        single_proc = single_map_alignment(A, B, fit_procrustes)
        single_ols = single_map_alignment(A, B, fit_ols)
        cka = cka_with_null(A, B, rng) if not args.skip_cka else None
        proc_ratios.append(up["universality_ratio"]); proc_same.append(up["same_concept_mean"])
        ols_ratios.append(uo["universality_ratio"]); ols_same.append(uo["same_concept_mean"])
        single_proc_all.append(single_proc["mean_aligned_cosine"])
        single_ols_all.append(single_ols["mean_aligned_cosine"])
        if cka is not None:
            cka_real_all.append(cka["cka_real_mean"]); cka_null_all.append(cka["cka_null_mean"])
        pairs_out.append({
            "source": a, "target": b,
            "family_pair": f"{A['family']}->{B['family']}",
            "n_concepts": len(cs),
            "procrustes": up, "ols": uo,
            "single_map_procrustes": single_proc, "single_map_ols": single_ols,
            "cka": cka,
        })
        log.info("%s->%s | per-concept ratio P=%.3f O=%.3f | single-map P=%.3f O=%.3f | CKA real=%s null=%s",
                 A["family"], B["family"], up["universality_ratio"], uo["universality_ratio"],
                 single_proc["mean_aligned_cosine"], single_ols["mean_aligned_cosine"],
                 f"{cka['cka_real_mean']:.3f}" if cka else "skip",
                 f"{cka['cka_null_mean']:.3f}" if cka else "skip")

    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "tag": tag, "models": names, "n_pairs": len(pairs_out), "concepts": CONCEPTS,
        "n_perm_cka_null": N_PERM,
        "procrustes_universality_ratio_mean": float(np.nanmean(proc_ratios)),
        "procrustes_same_concept_mean": float(np.nanmean(proc_same)),
        "ols_universality_ratio_mean": float(np.nanmean(ols_ratios)),
        "ols_same_concept_mean": float(np.nanmean(ols_same)),
        "single_map_procrustes_mean": float(np.nanmean(single_proc_all)),
        "single_map_ols_mean": float(np.nanmean(single_ols_all)),
        "cka_real_mean": float(np.nanmean(cka_real_all)) if cka_real_all else None,
        "cka_null_mean": float(np.nanmean(cka_null_all)) if cka_null_all else None,
    }
    out = {"summary": summary, "pairs": pairs_out}
    name = args.out_name or f"critic_analyses_{tag}.json"
    (OUT_DIR / name).write_text(json.dumps(out, indent=2))
    log.info("=" * 70)
    log.info("WROTE %s", OUT_DIR / name)
    log.info("Procrustes universality ratio: %.3f (same-concept aligned cosine %.3f)",
             summary["procrustes_universality_ratio_mean"], summary["procrustes_same_concept_mean"])
    log.info("OLS        universality ratio: %.3f (same-concept aligned cosine %.3f)",
             summary["ols_universality_ratio_mean"], summary["ols_same_concept_mean"])
    log.info("CKA real mean %.3f vs null mean %.3f",
             summary["cka_real_mean"], summary["cka_null_mean"])


if __name__ == "__main__":
    main()

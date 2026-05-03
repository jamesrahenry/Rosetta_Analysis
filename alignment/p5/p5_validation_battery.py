#!/usr/bin/env python3
"""P5 validation battery — five orthogonal tests of the path-3 result.

Path 3 (scratch/p5_proportional_depth_samedim.py) reported:
  matched 0.331 / mismatched 0.198 / Δ=+0.134 / p=1.2e-30 / 95% CI [0.117, 0.151]
across 98 ordered-pair × held-out-concept observations on 14 same-dim ordered
pairs in 5 dim clusters, using zero-PCA Procrustes alignment and proportional
depths {0.3, 0.5, 0.7}.

This script runs five orthogonal validations to test what's driving Δ:

  1. Windowed-CKA (no rotation) — replaces Procrustes cosine with linear CKA on
     K-layer windows of dom_vectors. If matched > mismatched holds under CKA,
     the depth-stratification is metric-independent — not a Procrustes artifact.

  2. Random-vector control — replaces dom_vectors with seeded random unit
     vectors of correct dim. Same pipeline. Expected Δ ≈ 0; if not, methodology
     is leaking.

  3. Concept-label shuffle — within each model, randomly permute which concept
     each dom_vector cube is assigned to. Same pipeline. Expected Δ ≈ 0 if the
     effect requires concept-specific correspondence between A and B.

  4. No-rotation raw cosine — same matched/mismatched comparison as path 3 but
     skipping Procrustes (R = I). Tests whether rotation is doing the work or
     the depth structure is already there.

  5. Depth-label permutation null — randomly reassigns layer indices to the
     {0.3, 0.5, 0.7} labels per (pair, concept). Tests whether the actual
     proportional alignment matters vs any 3-layer slicing.

Output: rosetta_data/results/p5_permutation/p5_validation_battery.json
        rosetta_data/results/p5_permutation/p5_validation_per_test.json (split)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes, svd as _robust_svd
from scipy.stats import mannwhitneyu

REPO_ROOT = Path.home()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONCEPTS = [
    "credibility", "certainty", "causation",
    "temporal_order", "negation", "sentiment", "moral_valence",
]
DEPTHS = [0.3, 0.5, 0.7]
CKA_WINDOW = 5  # layers around each proportional depth
N_BOOTSTRAP = 10000
N_DEPTH_PERMS = 1000
RNG_SEED = 42

DEFAULT_DATA_ROOT = REPO_ROOT / "rosetta_data" / "models"
DEFAULT_OUT_DIR = REPO_ROOT / "rosetta_data" / "results" / "p5_permutation"
DATA_ROOT = DEFAULT_DATA_ROOT
OUT_DIR = DEFAULT_OUT_DIR

# Forensic log of every Procrustes call that needed the gesvd fallback or
# fully failed. Populated by safe_procrustes() and dumped to JSON in main().
SVD_FAILURES: list[dict] = []
SVD_FAILURES_DIR: Path | None = None  # set in main() if --save-svd-failures


def safe_procrustes(first: np.ndarray, second: np.ndarray, *,
                    context: dict | None = None) -> np.ndarray | None:
    """orthogonal_procrustes(first, second) with a gesvd fallback.

    Mirrors scipy's closed-form solution: SVD of first.T @ second, then U @ Vt.
    The default LAPACK driver (gesdd) occasionally fails to converge on
    near-rank-deficient cross-covariance matrices. When that happens we retry
    with the slower-but-robust gesvd driver. The fallback computes the *same*
    decomposition — no math is changed. Each fallback (or hard failure) is
    appended to SVD_FAILURES with full forensic context.

    Returns the rotation R, or None if both drivers fail.
    """
    try:
        R, _ = orthogonal_procrustes(first, second)
        return R
    except Exception as e_fast:
        # Only catch numerical convergence failures — re-raise unrelated bugs.
        msg = str(e_fast)
        if "converge" not in msg.lower() and "SVD" not in msg:
            raise
        info = {
            "context": context or {},
            "shape_first": list(first.shape),
            "shape_second": list(second.shape),
            "fro_first": float(np.linalg.norm(first)),
            "fro_second": float(np.linalg.norm(second)),
            "any_nan": bool(np.isnan(first).any() or np.isnan(second).any()),
            "any_inf": bool(np.isinf(first).any() or np.isinf(second).any()),
            "gesdd_error": str(e_fast),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        R: np.ndarray | None = None
        try:
            M = first.T @ second
            U, S, Vt = _robust_svd(M, lapack_driver="gesvd")
            R = U @ Vt
            info["fallback"] = "gesvd_succeeded"
            info["singular_value_max"] = float(S[0])
            info["singular_value_min"] = float(S[-1])
            denom = max(float(S[-1]), float(np.finfo(S.dtype).tiny))
            info["condition_number"] = float(S[0] / denom)
            log.warning(
                "Procrustes gesdd failed (%s); recovered with gesvd. cond=%.2e ctx=%s",
                e_fast, info["condition_number"], context,
            )
        except Exception as e_robust:
            info["fallback"] = f"gesvd_also_failed: {e_robust}"
            log.error("Procrustes BOTH gesdd and gesvd failed. ctx=%s", context)

        if SVD_FAILURES_DIR is not None:
            try:
                stamp = info["timestamp"].replace(":", "-").replace("+00:00", "Z")
                ctx_vals = "_".join(str(v) for v in (context or {}).values())
                ctx_tag = "".join(c if c.isalnum() or c in "-_" else "_"
                                  for c in ctx_vals)[:80]
                fname = SVD_FAILURES_DIR / f"{stamp}_{ctx_tag}.npz"
                np.savez_compressed(fname, first=first, second=second)
                info["matrix_path"] = str(fname)
            except Exception as e_save:
                info["matrix_save_error"] = str(e_save)

        SVD_FAILURES.append(info)
        return R


# ============================================================================
# Helpers
# ============================================================================

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def interp_rows(M: np.ndarray, n_target: int) -> np.ndarray:
    n = len(M)
    if n == n_target:
        return M
    idx = np.round(np.linspace(0, n - 1, n_target)).astype(int)
    return M[idx]


def depth_layer(n_layers: int, frac: float) -> int:
    return int(np.clip(round(frac * (n_layers - 1)), 0, n_layers - 1))


def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    """Linear CKA between two (n × dim) matrices.

    CKA(A, B) = ||A^T B||_F^2 / (||A^T A||_F * ||B^T B||_F).
    Centered features.
    """
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(A.T @ B, "fro") ** 2
    self_a = np.linalg.norm(A.T @ A, "fro")
    self_b = np.linalg.norm(B.T @ B, "fro")
    if self_a < 1e-10 or self_b < 1e-10:
        return float("nan")
    return float(cross / (self_a * self_b))


def load_dom_vecs(model_dir: Path, concept: str) -> dict | None:
    p = model_dir / f"caz_{concept}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    raw = data["layer_data"]["metrics"]
    dom_vecs = np.array([m["dom_vector"] for m in raw], dtype=np.float64)
    norms = np.linalg.norm(dom_vecs, axis=1, keepdims=True)
    dom_vecs = dom_vecs / np.where(norms > 1e-10, norms, 1.0)
    return {
        "model_id": data["model_id"],
        "hidden_dim": data["hidden_dim"],
        "n_layers": data["layer_data"]["n_layers"],
        "dom_vecs": dom_vecs,
    }


def collect_store() -> tuple[dict, dict]:
    """Return (store, by_dim). store[name][concept] = dom_vec dict."""
    from collections import defaultdict
    store = {}
    for md in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):
        store[md.name] = {}
        for c in CONCEPTS:
            d = load_dom_vecs(md, c)
            if d is not None:
                store[md.name][c] = d
    by_dim = defaultdict(list)
    for name, cm in store.items():
        if not cm:
            continue
        any_c = next(iter(cm.values()))
        by_dim[any_c["hidden_dim"]].append(name)
    return store, by_dim


def fit_loco_rotation(cm_a: dict, cm_b: dict, held: str, n_common: int) -> np.ndarray | None:
    fit_concepts = [c for c in CONCEPTS
                    if c != held and c in cm_a and c in cm_b]
    if len(fit_concepts) < 2:
        return None
    blocks_a, blocks_b = [], []
    for c in fit_concepts:
        blocks_a.append(interp_rows(cm_a[c]["dom_vecs"], n_common))
        blocks_b.append(interp_rows(cm_b[c]["dom_vecs"], n_common))
    A = np.vstack(blocks_a)
    B = np.vstack(blocks_b)
    return safe_procrustes(B, A, context={
        "call": "fit_loco_rotation",
        "held_concept": held,
        "n_common": n_common,
    })


# ============================================================================
# Pipeline core: produce 9-cell cosine matrix per (pair × concept)
# ============================================================================

def build_cos_matrix(dvA: np.ndarray, dvB: np.ndarray, R: np.ndarray | None,
                     n_a: int, n_b: int) -> np.ndarray:
    """Compute 3×3 cosine matrix at DEPTHS, optionally rotating B by R."""
    cm = np.zeros((len(DEPTHS), len(DEPTHS)))
    for i, di in enumerate(DEPTHS):
        li = depth_layer(n_a, di)
        vA = dvA[li]
        for j, dj in enumerate(DEPTHS):
            lj = depth_layer(n_b, dj)
            vB = dvB[lj]
            vB_use = vB @ R if R is not None else vB
            cm[i, j] = cosine(vA, vB_use)
    return cm


def stats_from_cos_matrices(rows: list[dict]) -> dict:
    """Aggregate matched/mismatched stats from a list of pair_results."""
    matched, mismatched, deltas = [], [], []
    for r in rows:
        cm = np.array(r["cos_matrix"])
        if np.any(np.isnan(cm)):
            continue
        m_vals = [cm[k, k] for k in range(len(DEPTHS))]
        n_vals = [cm[i, j] for i in range(len(DEPTHS))
                  for j in range(len(DEPTHS)) if i != j]
        matched.extend(m_vals)
        mismatched.extend(n_vals)
        deltas.append(float(np.mean(m_vals) - np.mean(n_vals)))
    if not deltas:
        return {"n_observations": 0}
    s = {
        "n_observations": len(deltas),
        "n_positive_delta": int(sum(1 for d in deltas if d > 0)),
        "mean_matched": float(np.mean(matched)),
        "mean_mismatched": float(np.mean(mismatched)),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
    }
    try:
        stat, p = mannwhitneyu(matched, mismatched, alternative="greater")
        s["mannwhitney_p"] = float(p)
        s["mannwhitney_stat"] = float(stat)
    except ValueError:
        s["mannwhitney_p"] = float("nan")
    rng = np.random.default_rng(RNG_SEED)
    arr = np.array(deltas)
    n = len(arr)
    boot = np.array([float(np.mean(arr[rng.integers(0, n, size=n)]))
                     for _ in range(N_BOOTSTRAP)])
    s["bootstrap_ci_95"] = [float(np.percentile(boot, 2.5)),
                             float(np.percentile(boot, 97.5))]
    s["bootstrap_p_gt_zero"] = float(np.mean(boot > 0))
    return s


# ============================================================================
# Main pipeline used by tests 2, 3, 4, baseline
# ============================================================================

def run_propdepth_pipeline(
    store: dict, by_dim: dict, label: str,
    rotate: bool = True,
    dom_vec_override: callable | None = None,
    concept_perm_per_model: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    """Re-runs path-3 with hooks for nulls.

    rotate=False → no Procrustes (R=identity).
    dom_vec_override(name, concept) → returns dom_vecs to use instead of caz JSON.
    concept_perm_per_model[name][c] → real concept whose dom_vecs go under label c.
    """
    pair_results = []
    for dim, names in sorted(by_dim.items()):
        if len(names) < 2:
            continue
        log.info("[%s] dim %d: %d models", label, dim, len(names))
        t0 = time.time()
        for name_a, name_b in permutations(names, 2):
            cm_a, cm_b = store[name_a], store[name_b]

            # Apply concept permutation if provided.
            def get(name, cm, c):
                if dom_vec_override is not None:
                    out = dom_vec_override(name, c)
                    if out is not None:
                        return out
                if concept_perm_per_model is not None:
                    real_c = concept_perm_per_model[name].get(c, c)
                    return cm[real_c]
                return cm[c]

            for held in CONCEPTS:
                if held not in cm_a or held not in cm_b:
                    continue

                a_held = get(name_a, cm_a, held)
                b_held = get(name_b, cm_b, held)
                n_a = a_held["n_layers"]
                n_b = b_held["n_layers"]
                n_common = min(n_a, n_b)

                # Build LOCO basis using overridden cm.
                if rotate:
                    fit_concepts = [c for c in CONCEPTS
                                    if c != held and c in cm_a and c in cm_b]
                    if len(fit_concepts) < 2:
                        continue
                    blocks_a, blocks_b = [], []
                    for c in fit_concepts:
                        a_c = get(name_a, cm_a, c)
                        b_c = get(name_b, cm_b, c)
                        blocks_a.append(interp_rows(a_c["dom_vecs"], n_common))
                        blocks_b.append(interp_rows(b_c["dom_vecs"], n_common))
                    A = np.vstack(blocks_a)
                    B = np.vstack(blocks_b)
                    R = safe_procrustes(B, A, context={
                        "call": "run_propdepth_pipeline",
                        "label": label,
                        "model_a": name_a,
                        "model_b": name_b,
                        "dim": dim,
                        "held_concept": held,
                    })
                    if R is None:
                        continue
                else:
                    R = None

                cm_mat = build_cos_matrix(a_held["dom_vecs"], b_held["dom_vecs"],
                                          R, n_a, n_b)
                if np.any(np.isnan(cm_mat)):
                    continue
                pair_results.append({
                    "test_concept": held,
                    "model_a": name_a,
                    "model_b": name_b,
                    "dim": dim,
                    "cos_matrix": cm_mat.tolist(),
                })
        log.info("[%s] dim %d done in %.1fs (%d obs so far)",
                 label, dim, time.time() - t0, len(pair_results))
    return pair_results


# ============================================================================
# Test 5: depth-label permutation null
# ============================================================================

def run_depth_perm_null(store: dict, by_dim: dict, n_perms: int = N_DEPTH_PERMS) -> dict:
    """For each pair × concept, fit R once, then for each of N permutations
    randomly relabel which 3 layer indices in A (and in B) play the roles of
    {0.3, 0.5, 0.7}. Compute Δ_null distribution.

    The null preserves all data and rotation, only randomizes which layers
    are assigned to depth labels.
    """
    rng = np.random.default_rng(RNG_SEED)
    null_deltas = []

    for dim, names in sorted(by_dim.items()):
        if len(names) < 2:
            continue
        log.info("[depth-perm-null] dim %d: %d models", dim, len(names))
        for name_a, name_b in permutations(names, 2):
            cm_a, cm_b = store[name_a], store[name_b]
            for held in CONCEPTS:
                if held not in cm_a or held not in cm_b:
                    continue
                n_a = cm_a[held]["n_layers"]
                n_b = cm_b[held]["n_layers"]
                n_common = min(n_a, n_b)
                R = fit_loco_rotation(cm_a, cm_b, held, n_common)
                if R is None:
                    continue
                dvA = cm_a[held]["dom_vecs"]
                dvB = cm_b[held]["dom_vecs"]
                # Pre-rotate B once.
                dvB_rot = dvB @ R

                # For each permutation: pick K=3 random layer indices in each
                # model, treat them as labels [0,1,2], compute Δ.
                K = len(DEPTHS)
                for _ in range(n_perms):
                    li_a = rng.choice(n_a, size=K, replace=False)
                    li_b = rng.choice(n_b, size=K, replace=False)
                    cm_mat = np.zeros((K, K))
                    for i in range(K):
                        for j in range(K):
                            cm_mat[i, j] = cosine(dvA[li_a[i]], dvB_rot[li_b[j]])
                    if np.any(np.isnan(cm_mat)):
                        continue
                    matched = np.array([cm_mat[k, k] for k in range(K)])
                    mismatched = np.array([cm_mat[i, j] for i in range(K)
                                           for j in range(K) if i != j])
                    null_deltas.append(float(np.mean(matched) - np.mean(mismatched)))

    arr = np.array(null_deltas)
    return {
        "n_null_observations": len(null_deltas),
        "null_mean_delta": float(np.mean(arr)) if len(arr) else float("nan"),
        "null_std_delta": float(np.std(arr)) if len(arr) else float("nan"),
        "null_p99": float(np.percentile(arr, 99)) if len(arr) else float("nan"),
        "null_p95": float(np.percentile(arr, 95)) if len(arr) else float("nan"),
        "null_max": float(arr.max()) if len(arr) else float("nan"),
    }


# ============================================================================
# Test 1: Windowed CKA
# ============================================================================

def run_windowed_cka(store: dict, by_dim: dict) -> list[dict]:
    """At each proportional depth, take a K-layer window of dom_vectors as the
    representation. Compute linear CKA between A_window(d_i) and B_window(d_j)
    for all (i, j). Build 3×3 matrix per pair × concept.

    No rotation — CKA is rotation-invariant by construction.
    """
    pair_results = []
    half = CKA_WINDOW // 2
    for dim, names in sorted(by_dim.items()):
        if len(names) < 2:
            continue
        log.info("[CKA] dim %d: %d models", dim, len(names))
        t0 = time.time()
        for name_a, name_b in permutations(names, 2):
            cm_a, cm_b = store[name_a], store[name_b]
            for held in CONCEPTS:
                if held not in cm_a or held not in cm_b:
                    continue
                dvA = cm_a[held]["dom_vecs"]
                dvB = cm_b[held]["dom_vecs"]
                n_a = cm_a[held]["n_layers"]
                n_b = cm_b[held]["n_layers"]

                cm_mat = np.zeros((len(DEPTHS), len(DEPTHS)))
                for i, di in enumerate(DEPTHS):
                    li = depth_layer(n_a, di)
                    a_lo = max(0, li - half)
                    a_hi = min(n_a, li + half + 1)
                    A_win = dvA[a_lo:a_hi]
                    for j, dj in enumerate(DEPTHS):
                        lj = depth_layer(n_b, dj)
                        b_lo = max(0, lj - half)
                        b_hi = min(n_b, lj + half + 1)
                        B_win = dvB[b_lo:b_hi]
                        # Trim to common length.
                        n_common = min(len(A_win), len(B_win))
                        cm_mat[i, j] = linear_cka(A_win[:n_common], B_win[:n_common])
                if np.any(np.isnan(cm_mat)):
                    continue
                pair_results.append({
                    "test_concept": held,
                    "model_a": name_a,
                    "model_b": name_b,
                    "dim": dim,
                    "cos_matrix": cm_mat.tolist(),
                })
        log.info("[CKA] dim %d done in %.1fs", dim, time.time() - t0)
    return pair_results


# ============================================================================
# Random-vector & concept-shuffle helpers
# ============================================================================

def make_random_store(store: dict, seed: int) -> dict:
    """Replace every dom_vec matrix with random unit vectors of the same shape."""
    rng = np.random.default_rng(seed)
    new_store = {}
    for name, cm in store.items():
        new_store[name] = {}
        for c, info in cm.items():
            shape = info["dom_vecs"].shape
            v = rng.standard_normal(shape)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            new_info = dict(info)
            new_info["dom_vecs"] = v
            new_store[name][c] = new_info
    return new_store


def make_concept_perm(store: dict, seed: int) -> dict[str, dict[str, str]]:
    """Per-model random permutation of concept labels.

    Returns concept_perm_per_model[name][label] = real_concept.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for name in store:
        concepts_present = [c for c in CONCEPTS if c in store[name]]
        if len(concepts_present) < 2:
            out[name] = {c: c for c in CONCEPTS}
            continue
        shuffled = list(concepts_present)
        rng.shuffle(shuffled)
        mapping = {label: real for label, real in zip(concepts_present, shuffled)}
        # Concepts not in this model stay as identity.
        for c in CONCEPTS:
            mapping.setdefault(c, c)
        out[name] = mapping
    return out


# ============================================================================
# Driver
# ============================================================================

def write_partial(name: str, payload):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"p5_validation_{name}.json"
    p.write_text(json.dumps(payload, indent=2))
    log.info("Saved partial: %s", p)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="Number of random seeds for tests 2 and 3 (>1 builds null distribution)")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="Suffix for output filenames, e.g. '_v2'")
    ap.add_argument("--save-svd-failures", action="store_true",
                    help="Save raw (A, B) matrices for any Procrustes call that "
                         "needed the gesvd fallback or fully failed. Stored as "
                         "compressed .npz under <out-dir>/svd_failures/.")
    args = ap.parse_args()
    global DATA_ROOT, OUT_DIR, SVD_FAILURES_DIR
    DATA_ROOT = args.data_root
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.save_svd_failures:
        SVD_FAILURES_DIR = OUT_DIR / "svd_failures"
        SVD_FAILURES_DIR.mkdir(parents=True, exist_ok=True)
        log.info("SVD failure forensics enabled: %s", SVD_FAILURES_DIR)
    log.info("data_root=%s  out_dir=%s  n_seeds=%d", DATA_ROOT, OUT_DIR, args.n_seeds)

    overall_t0 = time.time()
    store, by_dim = collect_store()
    log.info("Loaded %d models, %d dim clusters",
             len(store), sum(1 for n in by_dim.values() if len(n) >= 2))

    results = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "depths": DEPTHS,
        "cka_window": CKA_WINDOW,
        "n_bootstrap": N_BOOTSTRAP,
        "n_depth_perms": N_DEPTH_PERMS,
        "rng_seed": RNG_SEED,
    }

    # --- Test 4: no-rotation raw cosine (fast) ---
    log.info("\n========== TEST 4: no-rotation raw cosine ==========")
    t0 = time.time()
    rows = run_propdepth_pipeline(store, by_dim, "no-rot", rotate=False)
    s = stats_from_cos_matrices(rows)
    s["elapsed_seconds"] = time.time() - t0
    s["pair_results"] = rows
    results["test_4_no_rotation"] = s
    write_partial("test4_no_rotation", s)
    log.info("[no-rot] n=%d Δ=%+.4f matched=%.4f mismatched=%.4f p=%.2e",
             s.get("n_observations", 0), s.get("mean_delta", float("nan")),
             s.get("mean_matched", float("nan")), s.get("mean_mismatched", float("nan")),
             s.get("mannwhitney_p", float("nan")))

    # --- Test 5: depth-label permutation null (medium) ---
    log.info("\n========== TEST 5: depth-label permutation null ==========")
    t0 = time.time()
    s = run_depth_perm_null(store, by_dim)
    s["elapsed_seconds"] = time.time() - t0
    results["test_5_depth_perm_null"] = s
    write_partial("test5_depth_perm_null", s)
    log.info("[depth-perm] null mean Δ=%.4f std=%.4f p99=%.4f max=%.4f",
             s.get("null_mean_delta", float("nan")),
             s.get("null_std_delta", float("nan")),
             s.get("null_p99", float("nan")),
             s.get("null_max", float("nan")))

    # --- Test 1: windowed CKA (slow) ---
    log.info("\n========== TEST 1: windowed CKA ==========")
    t0 = time.time()
    rows = run_windowed_cka(store, by_dim)
    s = stats_from_cos_matrices(rows)
    s["elapsed_seconds"] = time.time() - t0
    s["pair_results"] = rows
    results["test_1_windowed_cka"] = s
    write_partial("test1_windowed_cka", s)
    log.info("[CKA] n=%d Δ=%+.4f matched=%.4f mismatched=%.4f p=%.2e",
             s.get("n_observations", 0), s.get("mean_delta", float("nan")),
             s.get("mean_matched", float("nan")), s.get("mean_mismatched", float("nan")),
             s.get("mannwhitney_p", float("nan")))

    # --- Test 2: random-vector control (slow, multi-seed) ---
    log.info("\n========== TEST 2: random-vector control (n_seeds=%d) ==========", args.n_seeds)
    test2_seed_results = []
    for seed_i in range(args.n_seeds):
        seed = RNG_SEED + seed_i
        log.info("[rand-vec] seed %d/%d (RNG=%d)", seed_i + 1, args.n_seeds, seed)
        t0 = time.time()
        rand_store = make_random_store(store, seed)
        rows = run_propdepth_pipeline(rand_store, by_dim, f"rand-vec-s{seed}",
                                      rotate=True)
        s = stats_from_cos_matrices(rows)
        s["seed"] = seed
        s["elapsed_seconds"] = time.time() - t0
        if seed_i == 0:
            s["pair_results"] = rows  # only first to keep output size manageable
        test2_seed_results.append(s)
        log.info("[rand-vec seed %d] n=%d Δ=%+.4f matched=%.4f mismatched=%.4f p=%.2e",
                 seed, s.get("n_observations", 0), s.get("mean_delta", float("nan")),
                 s.get("mean_matched", float("nan")), s.get("mean_mismatched", float("nan")),
                 s.get("mannwhitney_p", float("nan")))
    results["test_2_random_vector"] = {
        "n_seeds": args.n_seeds,
        "per_seed": test2_seed_results,
        "delta_across_seeds_mean": float(np.mean([r["mean_delta"] for r in test2_seed_results])),
        "delta_across_seeds_std": float(np.std([r["mean_delta"] for r in test2_seed_results])) if args.n_seeds > 1 else 0.0,
    }
    write_partial(f"test2_random_vector{args.out_suffix}", results["test_2_random_vector"])

    # --- Test 3: concept-label shuffle (slow, multi-seed) ---
    log.info("\n========== TEST 3: concept-label shuffle (n_seeds=%d) ==========", args.n_seeds)
    test3_seed_results = []
    for seed_i in range(args.n_seeds):
        seed = RNG_SEED + seed_i
        log.info("[concept-shuf] seed %d/%d (RNG=%d)", seed_i + 1, args.n_seeds, seed)
        t0 = time.time()
        cp = make_concept_perm(store, seed)
        rows = run_propdepth_pipeline(store, by_dim, f"concept-shuf-s{seed}",
                                      rotate=True,
                                      concept_perm_per_model=cp)
        s = stats_from_cos_matrices(rows)
        s["seed"] = seed
        s["elapsed_seconds"] = time.time() - t0
        s["concept_perm_per_model"] = cp
        if seed_i == 0:
            s["pair_results"] = rows
        test3_seed_results.append(s)
        log.info("[concept-shuf seed %d] n=%d Δ=%+.4f matched=%.4f mismatched=%.4f p=%.2e",
                 seed, s.get("n_observations", 0), s.get("mean_delta", float("nan")),
                 s.get("mean_matched", float("nan")), s.get("mean_mismatched", float("nan")),
                 s.get("mannwhitney_p", float("nan")))
    results["test_3_concept_shuffle"] = {
        "n_seeds": args.n_seeds,
        "per_seed": test3_seed_results,
        "delta_across_seeds_mean": float(np.mean([r["mean_delta"] for r in test3_seed_results])),
        "delta_across_seeds_std": float(np.std([r["mean_delta"] for r in test3_seed_results])) if args.n_seeds > 1 else 0.0,
    }
    write_partial(f"test3_concept_shuffle{args.out_suffix}", results["test_3_concept_shuffle"])

    results["total_elapsed_seconds"] = time.time() - overall_t0

    # SVD failure summary — counts both recovered (gesvd succeeded) and hard
    # failures. Detailed entries dumped to a sidecar JSON for paper auditability.
    n_recovered = sum(1 for f in SVD_FAILURES if f.get("fallback") == "gesvd_succeeded")
    n_hard_fail = len(SVD_FAILURES) - n_recovered
    results["svd_failures"] = {
        "n_total": len(SVD_FAILURES),
        "n_recovered_with_gesvd": n_recovered,
        "n_hard_fail": n_hard_fail,
    }
    if SVD_FAILURES:
        sidecar = OUT_DIR / f"p5_svd_failures{args.out_suffix}.json"
        sidecar.write_text(json.dumps(SVD_FAILURES, indent=2))
        log.info("SVD failure log: %s (recovered=%d hard=%d)",
                 sidecar, n_recovered, n_hard_fail)

    out_path = OUT_DIR / f"p5_validation_battery{args.out_suffix}.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("\n=== ALL TESTS DONE in %.1fs ===", results["total_elapsed_seconds"])
    log.info("Output: %s", out_path)


if __name__ == "__main__":
    main()

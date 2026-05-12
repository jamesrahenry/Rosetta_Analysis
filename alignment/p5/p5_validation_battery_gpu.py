#!/usr/bin/env python3
"""P5 validation battery — GPU-accelerated (PyTorch) version.

Drop-in replacement for p5_validation_battery.py. Produces identical output
format so downstream code and skip-checks are unaffected.

Speedups over the CPU version:
  - torch.linalg.svd runs the Procrustes decomposition in CUDA kernels
    (~10-100× over scipy for large dims)
  - Test 5 depth-permutation null is fully vectorised: all n_perms random
    subsets are generated and evaluated in a single batched GPU operation
    instead of a Python loop (~n_perms× further speedup)
  - Tests 2/3/4 pairs are still iterated in Python but each SVD runs on GPU

Usage (identical flags to CPU version):
    python p5_validation_battery_gpu.py --out-dir <dir> [--n-seeds N]
    python p5_validation_battery_gpu.py --out-dir <dir> --out-suffix _gpu

Written: 2026-05-12 UTC
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
import torch
import torch.nn.functional as F
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
CKA_WINDOW = 5
N_BOOTSTRAP = 10000
N_DEPTH_PERMS = 1000
RNG_SEED = 42

DEFAULT_DATA_ROOT = REPO_ROOT / "rosetta_data" / "models"
DEFAULT_OUT_DIR = REPO_ROOT / "rosetta_data" / "results" / "p5_permutation"
DATA_ROOT = DEFAULT_DATA_ROOT
OUT_DIR = DEFAULT_OUT_DIR

DEVICE: torch.device = torch.device("cpu")   # set in main()
DTYPE: torch.dtype = torch.float64           # set in main()


# ============================================================================
# GPU helpers
# ============================================================================

def to_gpu(arr: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=DTYPE, device=DEVICE)


def gpu_procrustes(A: torch.Tensor, B: torch.Tensor):
    """Thin-SVD Procrustes on GPU. Returns factored rotation (Vh_a, R_mid, Vh_b).

    A, B: [m, n]. Finds R minimising ||A - B @ R||_F.
    Result applied via apply_rotation(vecs, R_fac).
    Returns None on numerical failure.
    """
    try:
        U_a, S_a, Vh_a = torch.linalg.svd(A, full_matrices=False)
        U_b, S_b, Vh_b = torch.linalg.svd(B, full_matrices=False)
        M = (S_a.unsqueeze(-1) * U_a.T) @ U_b * S_b.unsqueeze(0)
        U_m, _, Vh_m = torch.linalg.svd(M, full_matrices=False)
        R_mid = U_m @ Vh_m
        return (Vh_a, R_mid, Vh_b)
    except Exception as e:
        log.warning("gpu_procrustes failed: %s", e)
        return None


def apply_rotation(vecs: torch.Tensor, R_fac) -> torch.Tensor:
    """Apply factored rotation to [k, n] or [n] tensor.
    vecs @ R = (vecs @ Vh_a.T) @ R_mid @ Vh_b
    """
    Vh_a, R_mid, Vh_b = R_fac
    return (vecs @ Vh_a.T) @ R_mid @ Vh_b


def gpu_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return float("nan")
    return float((a @ b) / (na * nb))


def gpu_linear_cka(A: torch.Tensor, B: torch.Tensor) -> float:
    """Linear CKA between [n, d] tensors. Centred features."""
    A = A - A.mean(dim=0, keepdim=True)
    B = B - B.mean(dim=0, keepdim=True)
    cross = torch.linalg.matrix_norm(A.T @ B) ** 2
    self_a = torch.linalg.matrix_norm(A.T @ A)
    self_b = torch.linalg.matrix_norm(B.T @ B)
    if self_a < 1e-10 or self_b < 1e-10:
        return float("nan")
    return float(cross / (self_a * self_b))


def depth_layer(n_layers: int, frac: float) -> int:
    return int(np.clip(round(frac * (n_layers - 1)), 0, n_layers - 1))


def interp_rows_gpu(M: torch.Tensor, n_target: int) -> torch.Tensor:
    n = len(M)
    if n == n_target:
        return M
    idx = np.round(np.linspace(0, n - 1, n_target)).astype(int)
    return M[idx]


# ============================================================================
# Data loading
# ============================================================================

def load_dom_vecs(model_dir: Path, concept: str) -> dict | None:
    p = model_dir / f"caz_{concept}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    raw = data["layer_data"]["metrics"]
    arr = np.array([m["dom_vector"] for m in raw], dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr /= np.where(norms > 1e-10, norms, 1.0)
    return {
        "model_id": data["model_id"],
        "hidden_dim": data["hidden_dim"],
        "n_layers": data["layer_data"]["n_layers"],
        "dom_vecs": to_gpu(arr),          # [n_layers, hidden_dim] on GPU
        "dom_vecs_np": arr,               # keep numpy copy for CPU stats
    }


def collect_store() -> tuple[dict, dict]:
    from collections import defaultdict
    store = {}
    for md in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):
        store[md.name] = {}
        for c in CONCEPTS:
            d = load_dom_vecs(md, c)
            if d is not None:
                store[md.name][c] = d
    by_dim: dict[int, list[str]] = defaultdict(list)
    for name, cm in store.items():
        if not cm:
            continue
        any_c = next(iter(cm.values()))
        by_dim[any_c["hidden_dim"]].append(name)
    return store, by_dim


# ============================================================================
# LOCO rotation
# ============================================================================

def fit_loco_rotation_gpu(cm_a: dict, cm_b: dict, held: str, n_common: int):
    """LOCO Procrustes: fit on all concepts except `held`."""
    fit_concepts = [c for c in CONCEPTS if c != held and c in cm_a and c in cm_b]
    if len(fit_concepts) < 2:
        return None
    blocks_a = [interp_rows_gpu(cm_a[c]["dom_vecs"], n_common) for c in fit_concepts]
    blocks_b = [interp_rows_gpu(cm_b[c]["dom_vecs"], n_common) for c in fit_concepts]
    A = torch.cat(blocks_a, dim=0)
    B = torch.cat(blocks_b, dim=0)
    return gpu_procrustes(B, A)


# ============================================================================
# Build 3×3 cosine matrix (one pair × one concept)
# ============================================================================

def build_cos_matrix_gpu(dvA: torch.Tensor, dvB: torch.Tensor, R_fac,
                          n_a: int, n_b: int) -> np.ndarray:
    cm = np.zeros((len(DEPTHS), len(DEPTHS)))
    for i, di in enumerate(DEPTHS):
        vA = dvA[depth_layer(n_a, di)]
        for j, dj in enumerate(DEPTHS):
            vB = dvB[depth_layer(n_b, dj)]
            vB_use = apply_rotation(vB, R_fac) if R_fac is not None else vB
            cm[i, j] = gpu_cosine(vA, vB_use)
    return cm


# ============================================================================
# Stats (identical to CPU version — operates on list of row dicts)
# ============================================================================

def stats_from_cos_matrices(rows: list[dict]) -> dict:
    matched, mismatched, deltas = [], [], []
    for r in rows:
        cm = np.array(r["cos_matrix"])
        if np.any(np.isnan(cm)):
            continue
        K = len(DEPTHS)
        m_vals = [cm[k, k] for k in range(K)]
        n_vals = [cm[i, j] for i in range(K) for j in range(K) if i != j]
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
# Checkpoint helpers
# ============================================================================

def write_partial(name: str, payload):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"p5_validation_{name}.json"
    p.write_text(json.dumps(payload, indent=2))
    log.info("Saved partial: %s", p)


def load_partial(name: str) -> dict | None:
    p = OUT_DIR / f"p5_validation_{name}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        log.warning("Could not load partial %s: %s", p, e)
        return None


# ============================================================================
# Test 4: no-rotation raw cosine (fast baseline)
# ============================================================================

def run_propdepth_gpu(store: dict, by_dim: dict, label: str,
                      rotate: bool = True,
                      concept_perm_per_model: dict | None = None) -> list[dict]:
    pair_results = []
    for dim, names in sorted(by_dim.items()):
        if len(names) < 2:
            continue
        t0 = time.time()
        for name_a, name_b in permutations(names, 2):
            cm_a, cm_b = store[name_a], store[name_b]

            def get_vecs(name, cm, c):
                real = concept_perm_per_model[name].get(c, c) if concept_perm_per_model else c
                return cm[real]

            for held in CONCEPTS:
                if held not in cm_a or held not in cm_b:
                    continue
                a_held = get_vecs(name_a, cm_a, held)
                b_held = get_vecs(name_b, cm_b, held)
                n_a = a_held["n_layers"]
                n_b = b_held["n_layers"]
                n_common = min(n_a, n_b)

                if rotate:
                    R_fac = fit_loco_rotation_gpu(
                        {c: get_vecs(name_a, cm_a, c) for c in CONCEPTS if c in cm_a},
                        {c: get_vecs(name_b, cm_b, c) for c in CONCEPTS if c in cm_b},
                        held, n_common,
                    )
                    if R_fac is None:
                        continue
                else:
                    R_fac = None

                cm_mat = build_cos_matrix_gpu(
                    a_held["dom_vecs"], b_held["dom_vecs"], R_fac, n_a, n_b)
                if np.any(np.isnan(cm_mat)):
                    continue
                pair_results.append({
                    "test_concept": held, "model_a": name_a, "model_b": name_b,
                    "dim": dim, "cos_matrix": cm_mat.tolist(),
                })
        log.info("[%s] dim %d done in %.1fs", label, dim, time.time() - t0)
    return pair_results


# ============================================================================
# Test 5: depth-permutation null — vectorised over n_perms on GPU
# ============================================================================

def run_depth_perm_null_gpu(store: dict, by_dim: dict,
                             n_perms: int = N_DEPTH_PERMS) -> dict:
    """Vectorised depth-label permutation null.

    For each pair × concept: fit R once, then generate all n_perms random
    depth-label assignments as a batched GPU operation (no Python inner loop).
    """
    K = len(DEPTHS)
    eye_mask = ~torch.eye(K, dtype=torch.bool, device=DEVICE)
    null_deltas: list[float] = []

    for dim, names in sorted(by_dim.items()):
        if len(names) < 2:
            continue
        log.info("[depth-perm-null] dim %d: %d models", dim, len(names))
        for i_pair, (name_a, name_b) in enumerate(permutations(names, 2)):
            cm_a, cm_b = store[name_a], store[name_b]
            for held in CONCEPTS:
                if held not in cm_a or held not in cm_b:
                    continue
                n_a = cm_a[held]["n_layers"]
                n_b = cm_b[held]["n_layers"]
                n_common = min(n_a, n_b)
                R_fac = fit_loco_rotation_gpu(cm_a, cm_b, held, n_common)
                if R_fac is None:
                    continue
                dvA = cm_a[held]["dom_vecs"]          # [n_a, d]
                dvB_rot = apply_rotation(             # [n_b, d]
                    cm_b[held]["dom_vecs"], R_fac)

                # Deterministic per-pair seed (reproducible regardless of order)
                pair_seed = RNG_SEED + i_pair * len(CONCEPTS) + CONCEPTS.index(held)
                g = torch.Generator(device=DEVICE)
                g.manual_seed(pair_seed)

                # Sample [n_perms, K] layer indices without replacement via argsort trick
                li_a = torch.rand(n_perms, n_a, generator=g,
                                  device=DEVICE, dtype=DTYPE).argsort(dim=-1)[:, :K]
                li_b = torch.rand(n_perms, n_b, generator=g,
                                  device=DEVICE, dtype=DTYPE).argsort(dim=-1)[:, :K]

                vA = dvA[li_a]       # [n_perms, K, d]
                vB = dvB_rot[li_b]   # [n_perms, K, d]

                vA_n = F.normalize(vA.float(), dim=-1).to(DTYPE)
                vB_n = F.normalize(vB.float(), dim=-1).to(DTYPE)

                cosines = torch.bmm(vA_n, vB_n.transpose(-1, -2))  # [n_perms, K, K]

                matched = cosines.diagonal(dim1=-2, dim2=-1)        # [n_perms, K]
                mismatched = cosines[:, eye_mask]                    # [n_perms, K*(K-1)]

                deltas = matched.mean(dim=-1) - mismatched.mean(dim=-1)  # [n_perms]
                null_deltas.extend(deltas.cpu().tolist())

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
# Test 1: windowed CKA (GPU)
# ============================================================================

def run_windowed_cka_gpu(store: dict, by_dim: dict) -> list[dict]:
    pair_results = []
    half = CKA_WINDOW // 2
    for dim, names in sorted(by_dim.items()):
        if len(names) < 2:
            continue
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
                    A_win = dvA[max(0, li - half):min(n_a, li + half + 1)]
                    for j, dj in enumerate(DEPTHS):
                        lj = depth_layer(n_b, dj)
                        B_win = dvB[max(0, lj - half):min(n_b, lj + half + 1)]
                        n_c = min(len(A_win), len(B_win))
                        cm_mat[i, j] = gpu_linear_cka(A_win[:n_c], B_win[:n_c])
                if np.any(np.isnan(cm_mat)):
                    continue
                pair_results.append({
                    "test_concept": held, "model_a": name_a, "model_b": name_b,
                    "dim": dim, "cos_matrix": cm_mat.tolist(),
                })
        log.info("[CKA] dim %d done in %.1fs", dim, time.time() - t0)
    return pair_results


# ============================================================================
# Random-store and concept-perm helpers (identical logic to CPU version)
# ============================================================================

def make_random_store(store: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    new_store = {}
    for name, cm in store.items():
        new_store[name] = {}
        for c, info in cm.items():
            shape = info["dom_vecs"].shape
            v = rng.standard_normal((shape[0], shape[1])).astype(np.float64)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            new_info = dict(info)
            new_info["dom_vecs"] = to_gpu(v)
            new_store[name][c] = new_info
    return new_store


def make_concept_perm(store: dict, seed: int) -> dict[str, dict[str, str]]:
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
        for c in CONCEPTS:
            mapping.setdefault(c, c)
        out[name] = mapping
    return out


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="Number of random seeds for tests 2 and 3")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="Suffix for output filenames, e.g. '_gpu'")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float64",
                    help="Tensor dtype (float64 matches CPU version)")
    args = ap.parse_args()

    global DATA_ROOT, OUT_DIR, DEVICE, DTYPE
    DATA_ROOT = args.data_root
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DTYPE = torch.float64 if args.dtype == "float64" else torch.float32

    if not torch.cuda.is_available():
        log.error("No CUDA GPU found. Use p5_validation_battery.py for CPU execution.")
        sys.exit(1)
    DEVICE = torch.device("cuda")
    log.info("GPU: %s  dtype=%s  data_root=%s  out_dir=%s  n_seeds=%d",
             torch.cuda.get_device_name(0), args.dtype, DATA_ROOT, OUT_DIR, args.n_seeds)

    overall_t0 = time.time()
    store, by_dim = collect_store()
    log.info("Loaded %d models, %d dim clusters",
             len(store), sum(1 for n in by_dim.values() if len(n) >= 2))

    results = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "accelerator": "gpu",
        "gpu_name": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "depths": DEPTHS,
        "cka_window": CKA_WINDOW,
        "n_bootstrap": N_BOOTSTRAP,
        "n_depth_perms": N_DEPTH_PERMS,
        "rng_seed": RNG_SEED,
    }

    # --- Test 4: no-rotation raw cosine (fast) ---
    log.info("\n========== TEST 4: no-rotation raw cosine ==========")
    if (cached := load_partial("test4_no_rotation")) is not None:
        log.info("[no-rot] loaded from checkpoint")
        results["test_4_no_rotation"] = cached
    else:
        t0 = time.time()
        rows = run_propdepth_gpu(store, by_dim, "no-rot", rotate=False)
        s = stats_from_cos_matrices(rows)
        s["elapsed_seconds"] = time.time() - t0
        s["pair_results"] = rows
        results["test_4_no_rotation"] = s
        write_partial("test4_no_rotation", s)
        log.info("[no-rot] n=%d Δ=%+.4f p=%.2e",
                 s.get("n_observations", 0), s.get("mean_delta", float("nan")),
                 s.get("mannwhitney_p", float("nan")))

    # --- Test 5: depth-label permutation null ---
    log.info("\n========== TEST 5: depth-label permutation null ==========")
    if (cached := load_partial("test5_depth_perm_null")) is not None:
        log.info("[depth-perm] loaded from checkpoint")
        results["test_5_depth_perm_null"] = cached
    else:
        t0 = time.time()
        s = run_depth_perm_null_gpu(store, by_dim)
        s["elapsed_seconds"] = time.time() - t0
        results["test_5_depth_perm_null"] = s
        write_partial("test5_depth_perm_null", s)
        log.info("[depth-perm] null mean Δ=%.4f std=%.4f p99=%.4f (%.1fs)",
                 s.get("null_mean_delta", float("nan")), s.get("null_std_delta", float("nan")),
                 s.get("null_p99", float("nan")), s["elapsed_seconds"])

    # --- Test 1: windowed CKA ---
    log.info("\n========== TEST 1: windowed CKA ==========")
    if (cached := load_partial("test1_windowed_cka")) is not None:
        log.info("[CKA] loaded from checkpoint")
        results["test_1_windowed_cka"] = cached
    else:
        t0 = time.time()
        rows = run_windowed_cka_gpu(store, by_dim)
        s = stats_from_cos_matrices(rows)
        s["elapsed_seconds"] = time.time() - t0
        s["pair_results"] = rows
        results["test_1_windowed_cka"] = s
        write_partial("test1_windowed_cka", s)
        log.info("[CKA] n=%d Δ=%+.4f p=%.2e (%.1fs)",
                 s.get("n_observations", 0), s.get("mean_delta", float("nan")),
                 s.get("mannwhitney_p", float("nan")), s["elapsed_seconds"])

    # --- Test 2: random-vector control ---
    log.info("\n========== TEST 2: random-vector control (n_seeds=%d) ==========",
             args.n_seeds)
    test2_final = f"test2_random_vector{args.out_suffix}"
    test2_ckpt  = f"test2_random_vector{args.out_suffix}_ckpt"
    if (cached := load_partial(test2_final)) is not None:
        log.info("[rand-vec] loaded from checkpoint")
        results["test_2_random_vector"] = cached
    else:
        ckpt = load_partial(test2_ckpt)
        start = ckpt["n_seeds_done"] if ckpt else 0
        seed_results = ckpt["per_seed"] if ckpt else []
        for seed_i in range(start, args.n_seeds):
            seed = RNG_SEED + seed_i
            log.info("[rand-vec] seed %d/%d", seed_i + 1, args.n_seeds)
            t0 = time.time()
            rows = run_propdepth_gpu(make_random_store(store, seed), by_dim,
                                     f"rand-vec-s{seed}", rotate=True)
            s = stats_from_cos_matrices(rows)
            s["seed"] = seed
            s["elapsed_seconds"] = time.time() - t0
            if seed_i == 0:
                s["pair_results"] = rows
            seed_results.append(s)
            write_partial(test2_ckpt, {"n_seeds_done": seed_i + 1, "per_seed": seed_results})
        results["test_2_random_vector"] = {
            "n_seeds": args.n_seeds,
            "per_seed": seed_results,
            "delta_across_seeds_mean": float(np.mean([r["mean_delta"] for r in seed_results])),
            "delta_across_seeds_std": float(np.std([r["mean_delta"] for r in seed_results]))
                                      if args.n_seeds > 1 else 0.0,
        }
        write_partial(test2_final, results["test_2_random_vector"])

    # --- Test 3: concept-label shuffle ---
    log.info("\n========== TEST 3: concept-label shuffle (n_seeds=%d) ==========",
             args.n_seeds)
    test3_final = f"test3_concept_shuffle{args.out_suffix}"
    test3_ckpt  = f"test3_concept_shuffle{args.out_suffix}_ckpt"
    if (cached := load_partial(test3_final)) is not None:
        log.info("[concept-shuf] loaded from checkpoint")
        results["test_3_concept_shuffle"] = cached
    else:
        ckpt = load_partial(test3_ckpt)
        start = ckpt["n_seeds_done"] if ckpt else 0
        seed_results = ckpt["per_seed"] if ckpt else []
        for seed_i in range(start, args.n_seeds):
            seed = RNG_SEED + seed_i
            log.info("[concept-shuf] seed %d/%d", seed_i + 1, args.n_seeds)
            t0 = time.time()
            cp = make_concept_perm(store, seed)
            rows = run_propdepth_gpu(store, by_dim, f"concept-shuf-s{seed}",
                                     rotate=True, concept_perm_per_model=cp)
            s = stats_from_cos_matrices(rows)
            s["seed"] = seed
            s["elapsed_seconds"] = time.time() - t0
            s["concept_perm_per_model"] = cp
            if seed_i == 0:
                s["pair_results"] = rows
            seed_results.append(s)
            write_partial(test3_ckpt, {"n_seeds_done": seed_i + 1, "per_seed": seed_results})
        results["test_3_concept_shuffle"] = {
            "n_seeds": args.n_seeds,
            "per_seed": seed_results,
            "delta_across_seeds_mean": float(np.mean([r["mean_delta"] for r in seed_results])),
            "delta_across_seeds_std": float(np.std([r["mean_delta"] for r in seed_results]))
                                      if args.n_seeds > 1 else 0.0,
        }
        write_partial(test3_final, results["test_3_concept_shuffle"])

    results["total_elapsed_seconds"] = time.time() - overall_t0

    out_path = OUT_DIR / f"p5_validation_battery{args.out_suffix}.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("\n=== ALL TESTS DONE in %.1fs ===", results["total_elapsed_seconds"])
    log.info("Output: %s", out_path)


if __name__ == "__main__":
    main()

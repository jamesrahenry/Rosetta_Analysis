#!/usr/bin/env python3
"""P5 proportional-depth same-dim test (zero-PCA, no multimodality requirement).

Drops the multimodality filter that gutted scratch/p5_loco_samedim.py to n=6.
Instead, evaluates depth-stratified alignment at fixed proportional depths
{0.3, 0.5, 0.7} of model depth — which exists in every model regardless of
CAZ profile. Tests whether matched-depth cross-architecture cosine exceeds
mismatched-depth cosine after orthogonal Procrustes alignment.

Uses all 14 same-dim ordered pairs × 7 held-out concepts = 98 trials.

Methodology, per ordered pair (A, B) at same hidden dim, per held-out concept c:
  1. Fit basis: stack all-layer dom_vectors from the 6 non-held concepts
     (per-concept rows, B's rows interpolated to match A's layer count).
     A = stack of (6 concepts × n_A layers) rows in dim_A.
     B = stack of (6 concepts × n_A layers) rows in dim_B = dim_A.
  2. R, _ = orthogonal_procrustes(B_basis, A_basis).
  3. For depths d ∈ {0.3, 0.5, 0.7}:
       v_A(d) = dom_vec_A(c, round(d * n_A_layers))
       v_B(d) = dom_vec_B(c, round(d * n_B_layers))
       v_B_rot(d) = v_B(d) @ R
  4. Compute 3×3 cosine matrix M[i,j] = cos(v_A(d_i), v_B_rot(d_j)).
  5. matched = diag(M), mismatched = off-diag(M).
     delta = mean(matched) - mean(mismatched).
  6. Aggregate per concept and grand. Mann-Whitney + bootstrap CI.

Output: rosetta_data/results/p5_permutation/p5_propdepth_samedim_results.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
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
DEFAULT_DEPTHS = [0.3, 0.5, 0.7]

DEFAULT_DATA_ROOT = REPO_ROOT / "rosetta_data" / "models"
DEFAULT_OUT_DIR = REPO_ROOT / "rosetta_data" / "results" / "p5_permutation"
N_BOOTSTRAP = 10000
RNG_SEED = 42


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


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


def interp_rows(M: np.ndarray, n_target: int) -> np.ndarray:
    n = len(M)
    if n == n_target:
        return M
    idx = np.round(np.linspace(0, n - 1, n_target)).astype(int)
    return M[idx]


def depth_layer(n_layers: int, frac: float) -> int:
    """Round proportional depth to a layer index in [0, n_layers-1]."""
    return int(np.clip(round(frac * (n_layers - 1)), 0, n_layers - 1))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--depths", type=str, default=",".join(str(d) for d in DEFAULT_DEPTHS),
                    help="Comma-separated proportional depths, e.g. '0.3,0.5,0.7'")
    ap.add_argument("--out-name", type=str, default="p5_propdepth_samedim_results.json",
                    help="Output filename (allows running multiple depth sets without clobbering)")
    args = ap.parse_args()

    global DEPTHS, DATA_ROOT, OUT_DIR
    DEPTHS = [float(x) for x in args.depths.split(",")]
    DATA_ROOT = args.data_root
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("data_root=%s  out_dir=%s  depths=%s", DATA_ROOT, OUT_DIR, DEPTHS)

    model_dirs = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir())
    log.info("Found %d model dirs", len(model_dirs))

    # store[name][concept] = dom_vec dict
    store: dict[str, dict[str, dict]] = {}
    for md in model_dirs:
        store[md.name] = {}
        for c in CONCEPTS:
            d = load_dom_vecs(md, c)
            if d is not None:
                store[md.name][c] = d

    # Same-dim cluster grouping
    from collections import defaultdict
    by_dim = defaultdict(list)
    for name, cm in store.items():
        if not cm:
            continue
        any_c = next(iter(cm.values()))
        by_dim[any_c["hidden_dim"]].append(name)

    pair_results = []

    for dim, names in sorted(by_dim.items()):
        if len(names) < 2:
            continue
        log.info("dim %d: %d models — %s", dim, len(names), names)
        for name_a, name_b in permutations(names, 2):
            cm_a, cm_b = store[name_a], store[name_b]

            for held in CONCEPTS:
                if held not in cm_a or held not in cm_b:
                    continue

                fit_concepts = [c for c in CONCEPTS
                                if c != held and c in cm_a and c in cm_b]
                if len(fit_concepts) < 2:
                    continue

                blocks_a = []
                blocks_b = []
                n_a = cm_a[held]["n_layers"]
                n_b = cm_b[held]["n_layers"]
                n_common = min(n_a, n_b)
                for c in fit_concepts:
                    Ma = cm_a[c]["dom_vecs"]
                    Mb = cm_b[c]["dom_vecs"]
                    blocks_a.append(interp_rows(Ma, n_common))
                    blocks_b.append(interp_rows(Mb, n_common))
                A = np.vstack(blocks_a)
                B = np.vstack(blocks_b)

                try:
                    R, _ = orthogonal_procrustes(B, A)
                except Exception as e:
                    log.warning("Procrustes fail %s × %s on %s: %s",
                                name_a, name_b, held, e)
                    continue

                dvA = cm_a[held]["dom_vecs"]
                dvB = cm_b[held]["dom_vecs"]

                # Build 3x3 cosine matrix
                cos_matrix = np.zeros((len(DEPTHS), len(DEPTHS)))
                for i, di in enumerate(DEPTHS):
                    li = depth_layer(n_a, di)
                    vA = dvA[li]
                    for j, dj in enumerate(DEPTHS):
                        lj = depth_layer(n_b, dj)
                        vB = dvB[lj]
                        vB_rot = vB @ R
                        cos_matrix[i, j] = cosine(vA, vB_rot)

                matched = np.array([cos_matrix[k, k] for k in range(len(DEPTHS))])
                mismatched = np.array([cos_matrix[i, j]
                                       for i in range(len(DEPTHS))
                                       for j in range(len(DEPTHS)) if i != j])
                if np.any(np.isnan(matched)) or np.any(np.isnan(mismatched)):
                    continue
                delta = float(np.mean(matched) - np.mean(mismatched))

                pair_results.append({
                    "test_concept": held,
                    "model_a": name_a,
                    "model_b": name_b,
                    "dim": dim,
                    "n_layers_a": n_a,
                    "n_layers_b": n_b,
                    "depths": DEPTHS,
                    "n_fit_concepts": len(fit_concepts),
                    "fit_concepts": fit_concepts,
                    "cos_matrix": cos_matrix.tolist(),
                    "matched_mean": float(np.mean(matched)),
                    "mismatched_mean": float(np.mean(mismatched)),
                    "obs_delta": delta,
                })

    log.info("Total pair × concept observations: %d", len(pair_results))

    if not pair_results:
        log.error("No pairs.")
        return

    # Grand stats
    all_matched = []
    all_mismatched = []
    deltas = []
    for r in pair_results:
        cm = np.array(r["cos_matrix"])
        for k in range(len(DEPTHS)):
            v = cm[k, k]
            if not np.isnan(v):
                all_matched.append(float(v))
        for i in range(len(DEPTHS)):
            for j in range(len(DEPTHS)):
                if i == j:
                    continue
                v = cm[i, j]
                if not np.isnan(v):
                    all_mismatched.append(float(v))
        deltas.append(r["obs_delta"])

    grand = {
        "n_observations": len(pair_results),
        "n_positive_delta": int(sum(1 for d in deltas if d > 0)),
        "n_matched_cos": len(all_matched),
        "n_mismatched_cos": len(all_mismatched),
        "mean_matched": float(np.mean(all_matched)),
        "mean_mismatched": float(np.mean(all_mismatched)),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
    }
    stat, p = mannwhitneyu(all_matched, all_mismatched, alternative="greater")
    grand["mannwhitney_p"] = float(p)
    grand["mannwhitney_stat"] = float(stat)

    rng = np.random.default_rng(RNG_SEED)
    arr = np.array(deltas)
    n = len(arr)
    boot = np.array([float(np.mean(arr[rng.integers(0, n, size=n)]))
                     for _ in range(N_BOOTSTRAP)])
    grand["bootstrap_ci_95"] = [float(np.percentile(boot, 2.5)),
                                 float(np.percentile(boot, 97.5))]
    grand["bootstrap_p_gt_zero"] = float(np.mean(boot > 0))

    by_concept = {}
    for c in CONCEPTS:
        rows = [r for r in pair_results if r["test_concept"] == c]
        if not rows:
            continue
        ds = [r["obs_delta"] for r in rows]
        ms = []
        ns = []
        for r in rows:
            cm = np.array(r["cos_matrix"])
            for k in range(len(DEPTHS)):
                if not np.isnan(cm[k, k]):
                    ms.append(float(cm[k, k]))
            for i in range(len(DEPTHS)):
                for j in range(len(DEPTHS)):
                    if i != j and not np.isnan(cm[i, j]):
                        ns.append(float(cm[i, j]))
        try:
            stat_c, p_c = mannwhitneyu(ms, ns, alternative="greater")
            mw_p = float(p_c)
        except ValueError:
            mw_p = float("nan")
        by_concept[c] = {
            "n_observations": len(rows),
            "n_positive_delta": int(sum(1 for d in ds if d > 0)),
            "mean_delta": float(np.mean(ds)),
            "mean_matched": float(np.mean(ms)) if ms else float("nan"),
            "mean_mismatched": float(np.mean(ns)) if ns else float("nan"),
            "mannwhitney_p": mw_p,
        }

    by_dim_summary = {}
    for dim_value in sorted(set(r["dim"] for r in pair_results)):
        rows = [r for r in pair_results if r["dim"] == dim_value]
        ds = [r["obs_delta"] for r in rows]
        by_dim_summary[str(dim_value)] = {
            "n_observations": len(rows),
            "n_positive_delta": int(sum(1 for d in ds if d > 0)),
            "mean_delta": float(np.mean(ds)),
            "median_delta": float(np.median(ds)),
        }

    output = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "method": ("proportional-depth depth-matched alignment, same-dim "
                   "only, zero-PCA; LOCO fit on non-held all-layer dom_vectors"),
        "depths": DEPTHS,
        "n_bootstrap": N_BOOTSTRAP,
        "rng_seed": RNG_SEED,
        "summary": {
            "grand": grand,
            "by_concept": by_concept,
            "by_dim": by_dim_summary,
        },
        "pair_results": pair_results,
    }

    out_path = OUT_DIR / args.out_name
    out_path.write_text(json.dumps(output, indent=2))

    log.info("\n=== GRAND SUMMARY (proportional-depth, same-dim, zero-PCA) ===")
    log.info("  Observations: %d   pos delta: %d/%d",
             grand["n_observations"], grand["n_positive_delta"],
             grand["n_observations"])
    log.info("  matched mean: %.4f  (n=%d)", grand["mean_matched"], grand["n_matched_cos"])
    log.info("  mismatched mean: %.4f  (n=%d)", grand["mean_mismatched"], grand["n_mismatched_cos"])
    log.info("  mean delta: %.4f   median delta: %.4f",
             grand["mean_delta"], grand["median_delta"])
    log.info("  Mann-Whitney p (matched > mismatched): %.4e", grand["mannwhitney_p"])
    log.info("  Bootstrap 95%% CI on delta: [%.4f, %.4f]",
             *grand["bootstrap_ci_95"])
    log.info("  Output: %s", out_path)


if __name__ == "__main__":
    main()

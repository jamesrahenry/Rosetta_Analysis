#!/usr/bin/env python3
"""P5 CKA analysis — companion to p5_cka_extract.py.

Loads cka_acts_<concept>.npz files saved by the GPU extraction job, computes
linear CKA at proportional depths {0.3, 0.5, 0.7} for each same-dim ordered
pair × held-out concept, and produces a 3×3 CKA matrix per (pair, concept).

CKA is rotation-invariant by construction — no Procrustes, no LOCO basis fit.
This is the standard-form CKA test that test 1 in the validation battery
approximated with a windowed-on-dom-vectors hack.

Output:
  rosetta_data/results/p5_permutation/p5_cka_real_results.json

The matched (diag) vs mismatched (off-diag) test mirrors path 3 / path 4
exactly so the result is directly comparable to Δ=+0.134 from path 3.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import numpy as np
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
N_BOOTSTRAP = 10000
RNG_SEED = 42

DEFAULT_DATA_ROOT = REPO_ROOT / "rosetta_data" / "models"
DEFAULT_OUT_DIR = REPO_ROOT / "rosetta_data" / "results" / "p5_permutation"


def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    """Linear CKA between (n × dim_A) and (n × dim_B) feature matrices.

    CKA(A, B) = ||A^T B||_F^2 / (||A^T A||_F * ||B^T B||_F).
    Centered features along the n-axis. Rotation-invariant within each model;
    handles different dim_A and dim_B.
    """
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(A.T @ B, "fro") ** 2
    self_a = np.linalg.norm(A.T @ A, "fro")
    self_b = np.linalg.norm(B.T @ B, "fro")
    if self_a < 1e-10 or self_b < 1e-10:
        return float("nan")
    return float(cross / (self_a * self_b))


def load_acts(model_dir: Path, concept: str) -> dict | None:
    p = model_dir / f"cka_acts_{concept}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {
        "acts": z["acts"],  # [n, 3, dim]
        "labels": z["labels"],
        "depth_layers": z["depth_layers"],
        "n_layers": int(z["n_layers"][0]),
        "hidden_dim": int(z["acts"].shape[-1]),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                    help=f"Path to rosetta_data/models (default: {DEFAULT_DATA_ROOT})")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"Output dir (default: {DEFAULT_OUT_DIR})")
    args = ap.parse_args()
    data_root: Path = args.data_root
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("data_root=%s  out_dir=%s", data_root, out_dir)

    # Build (model, concept) → acts map; skip models with incomplete extraction.
    store: dict[str, dict[str, dict]] = {}
    for md in sorted(p for p in data_root.iterdir() if p.is_dir()):
        store[md.name] = {}
        for c in CONCEPTS:
            d = load_acts(md, c)
            if d is not None:
                store[md.name][c] = d
    coverage = {n: len(cm) for n, cm in store.items()}
    log.info("Coverage: %s", coverage)

    # Same-dim grouping.
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
            common = sorted(set(cm_a) & set(cm_b))
            if len(common) < 1:
                continue
            for held in common:
                a = cm_a[held]
                b = cm_b[held]
                # Sanity: both should have used the same calibration set size.
                # If not, trim to common length.
                n = min(len(a["acts"]), len(b["acts"]))
                if n < 20:
                    continue
                cka_matrix = np.zeros((len(DEPTHS), len(DEPTHS)))
                for i in range(len(DEPTHS)):
                    A = a["acts"][:n, i, :].astype(np.float64)
                    for j in range(len(DEPTHS)):
                        B = b["acts"][:n, j, :].astype(np.float64)
                        cka_matrix[i, j] = linear_cka(A, B)
                if np.any(np.isnan(cka_matrix)):
                    continue
                pair_results.append({
                    "test_concept": held,
                    "model_a": name_a,
                    "model_b": name_b,
                    "dim": dim,
                    "n_examples": int(n),
                    "cos_matrix": cka_matrix.tolist(),
                })

    log.info("Total pair × concept observations: %d", len(pair_results))
    if not pair_results:
        log.error("No observations — check that extractions completed.")
        return

    # Aggregate.
    matched, mismatched, deltas = [], [], []
    for r in pair_results:
        cm = np.array(r["cos_matrix"])
        m_vals = [float(cm[k, k]) for k in range(len(DEPTHS))]
        n_vals = [float(cm[i, j]) for i in range(len(DEPTHS))
                  for j in range(len(DEPTHS)) if i != j]
        matched.extend(m_vals)
        mismatched.extend(n_vals)
        deltas.append(float(np.mean(m_vals) - np.mean(n_vals)))

    grand = {
        "n_observations": len(pair_results),
        "n_positive_delta": int(sum(1 for d in deltas if d > 0)),
        "n_matched": len(matched),
        "n_mismatched": len(mismatched),
        "mean_matched": float(np.mean(matched)),
        "mean_mismatched": float(np.mean(mismatched)),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
    }
    stat, p = mannwhitneyu(matched, mismatched, alternative="greater")
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
        ms, ns = [], []
        ds = []
        for r in rows:
            cm = np.array(r["cos_matrix"])
            ms.extend([float(cm[k, k]) for k in range(len(DEPTHS))])
            ns.extend([float(cm[i, j]) for i in range(len(DEPTHS))
                       for j in range(len(DEPTHS)) if i != j])
            ds.append(float(np.mean([cm[k, k] for k in range(len(DEPTHS))])
                            - np.mean([cm[i, j] for i in range(len(DEPTHS))
                                        for j in range(len(DEPTHS)) if i != j])))
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

    output = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "method": "real linear CKA at proportional depths, same-dim only",
        "depths": DEPTHS,
        "n_bootstrap": N_BOOTSTRAP,
        "rng_seed": RNG_SEED,
        "summary": {
            "grand": grand,
            "by_concept": by_concept,
        },
        "pair_results": pair_results,
        "model_coverage": coverage,
    }
    out_path = out_dir / "p5_cka_real_results.json"
    out_path.write_text(json.dumps(output, indent=2))

    log.info("\n=== GRAND SUMMARY (real CKA, same-dim, proportional depth) ===")
    log.info("  Observations: %d   pos delta: %d/%d",
             grand["n_observations"], grand["n_positive_delta"],
             grand["n_observations"])
    log.info("  matched: %.4f   mismatched: %.4f   delta: %.4f",
             grand["mean_matched"], grand["mean_mismatched"],
             grand["mean_delta"])
    log.info("  Mann-Whitney p (matched > mismatched): %.4e", grand["mannwhitney_p"])
    log.info("  Bootstrap 95%% CI on delta: [%.4f, %.4f]",
             *grand["bootstrap_ci_95"])
    log.info("  Output: %s", out_path)


if __name__ == "__main__":
    main()

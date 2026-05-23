#!/usr/bin/env python3
"""prh_cluster_bootstrap.py — Cluster-level bootstrap CI for P4 §3.7.

Written: 2026-05-23 UTC

The trial-level bootstrap CI in the paper (N=10,000 bootstrap draws over 563
individual trials) may understate variance if model-pair observations within
the same dimension cluster are correlated.

This script resamples at the model-pair level: unique unordered (model_a,
model_b) pairs are the resample unit, and all concept-level observations for a
sampled pair are included together. This gives a CI that accounts for
within-cluster correlation.

Reports both trial-level and cluster-level CIs for direct comparison.

Input:  ~/rosetta_data/results/PRH/p5/p5_propdepth_samedim_results.json
Output: ~/rosetta_data/results/PRH/prh_cluster_bootstrap_n250.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_CANDIDATES = [
    Path.home() / "rosetta_data" / "results" / "PRH" / "p5",
]
DATA_DIR = next((p for p in _DATA_CANDIDATES if p.exists()), None)
if DATA_DIR is None:
    raise SystemExit("PRH/p5 results not found. Run HF restore first.")

INPUT_FILE = DATA_DIR / "p5_propdepth_samedim_results.json"
OUTPUT_FILE = DATA_DIR.parent / "prh_cluster_bootstrap_n250.json"

N_BOOTSTRAP = 10_000
RNG_SEED = 42

# ---------------------------------------------------------------------------
# Family detection — maps model slug → family string
# ---------------------------------------------------------------------------
SLUG_FAMILY: list[tuple[str, str]] = [
    ("EleutherAI_pythia", "pythia"),
    ("facebook_opt", "opt"),
    ("openai_community_gpt2", "gpt2"),
    ("google_gemma", "gemma"),
    ("meta_llama_Llama", "llama"),
    ("Qwen_Qwen", "qwen"),
    ("mistralai_Mistral", "mistral"),
    ("tiiuae_falcon", "falcon"),
    ("microsoft_phi", "phi"),
]


def get_family(slug: str) -> str:
    for prefix, fam in SLUG_FAMILY:
        if slug.startswith(prefix):
            return fam
    return "unknown"


def is_cross_family(model_a: str, model_b: str) -> bool:
    return get_family(model_a) != get_family(model_b)


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def bootstrap_ci(
    deltas: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Trial-level bootstrap: resample individual delta observations."""
    boot_means = np.array([
        rng.choice(deltas, size=len(deltas), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(deltas.mean()), lo, hi


def cluster_bootstrap_ci(
    pair_to_deltas: dict[frozenset, list[float]],
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Cluster-level bootstrap: resample model pairs, include all concepts."""
    pairs = list(pair_to_deltas.keys())
    all_deltas = [d for ds in pair_to_deltas.values() for d in ds]
    grand_mean = float(np.mean(all_deltas))

    boot_means = []
    for _ in range(n_bootstrap):
        sampled_pairs = rng.choice(len(pairs), size=len(pairs), replace=True)
        sample_deltas = [
            d for idx in sampled_pairs for d in pair_to_deltas[pairs[idx]]
        ]
        boot_means.append(np.mean(sample_deltas))

    boot_means = np.array(boot_means)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return grand_mean, lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Loading %s", INPUT_FILE)
    data = json.loads(INPUT_FILE.read_text())
    pair_results = data["pair_results"]
    log.info("Loaded %d pair-concept observations", len(pair_results))

    # Split into cross-family and within-family
    cross_family = [r for r in pair_results if is_cross_family(r["model_a"], r["model_b"])]
    within_family = [r for r in pair_results if not is_cross_family(r["model_a"], r["model_b"])]
    log.info("Cross-family: %d | Within-family: %d", len(cross_family), len(within_family))

    rng = np.random.default_rng(RNG_SEED)
    results: dict = {
        "written": datetime.now(timezone.utc).isoformat(),
        "input_file": str(INPUT_FILE),
        "n_bootstrap": N_BOOTSTRAP,
        "rng_seed": RNG_SEED,
        "cross_family": {},
        "within_family": {},
    }

    for label, subset in [("cross_family", cross_family), ("within_family", within_family)]:
        if not subset:
            continue

        deltas = np.array([r["obs_delta"] for r in subset])
        n_positive = int((deltas > 0).sum())
        log.info("[%s] N=%d obs, %d positive (%.1f%%)",
                 label, len(deltas), n_positive, 100 * n_positive / len(deltas))

        # Build cluster map: frozenset({model_a, model_b}) → list of deltas
        pair_to_deltas: dict[frozenset, list[float]] = {}
        for r in subset:
            key = frozenset({r["model_a"], r["model_b"]})
            pair_to_deltas.setdefault(key, []).append(r["obs_delta"])
        n_clusters = len(pair_to_deltas)
        log.info("[%s] %d unique model pairs (clusters)", label, n_clusters)

        # Trial-level bootstrap
        trial_mean, trial_lo, trial_hi = bootstrap_ci(deltas, N_BOOTSTRAP, rng)
        log.info("[%s] Trial-level CI: %.4f [%.4f, %.4f]",
                 label, trial_mean, trial_lo, trial_hi)

        # Cluster-level bootstrap
        cluster_mean, cluster_lo, cluster_hi = cluster_bootstrap_ci(
            pair_to_deltas, N_BOOTSTRAP, rng
        )
        log.info("[%s] Cluster-level CI: %.4f [%.4f, %.4f]",
                 label, cluster_mean, cluster_lo, cluster_hi)

        results[label] = {
            "n_observations": len(deltas),
            "n_positive": n_positive,
            "pct_positive": round(100 * n_positive / len(deltas), 1),
            "n_clusters": n_clusters,
            "grand_mean_delta": round(float(deltas.mean()), 6),
            "trial_bootstrap": {
                "mean": round(trial_mean, 6),
                "ci_lo": round(trial_lo, 6),
                "ci_hi": round(trial_hi, 6),
                "ci_width": round(trial_hi - trial_lo, 6),
            },
            "cluster_bootstrap": {
                "mean": round(cluster_mean, 6),
                "ci_lo": round(cluster_lo, 6),
                "ci_hi": round(cluster_hi, 6),
                "ci_width": round(cluster_hi - cluster_lo, 6),
            },
        }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2))
    log.info("Results → %s", OUTPUT_FILE)

    # Print comparison table
    for label in ("cross_family", "within_family"):
        if label not in results or not results[label]:
            continue
        r = results[label]
        print(f"\n=== {label} ===")
        print(f"  N obs={r['n_observations']}, N clusters={r['n_clusters']}, "
              f"Δ={r['grand_mean_delta']:.4f}")
        t = r["trial_bootstrap"]
        c = r["cluster_bootstrap"]
        print(f"  Trial-level   CI: [{t['ci_lo']:.4f}, {t['ci_hi']:.4f}]  "
              f"width={t['ci_width']:.4f}")
        print(f"  Cluster-level CI: [{c['ci_lo']:.4f}, {c['ci_hi']:.4f}]  "
              f"width={c['ci_width']:.4f}")
        ratio = c["ci_width"] / t["ci_width"] if t["ci_width"] > 0 else float("nan")
        print(f"  Width ratio (cluster/trial): {ratio:.2f}x")


if __name__ == "__main__":
    main()

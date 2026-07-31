#!/usr/bin/env python3
"""P5 Test 6 — shuffled labels + no rotation (missing 2×2 cell).

Fills the pure-null corner of the decomposition table:

                    | Correct labels | Shuffled labels
--------------------|----------------|------------------
  With rotation     |  Δ=+0.1515     |  Δ≈+0.012  (Test 3)
  No rotation       |  Δ=+0.0995     |  Δ=???      (THIS)

Expected Δ ≈ 0 — no rotation + wrong labels = pure ARH floor.
If close to zero, confirms the signal requires BOTH coordinate alignment
AND concept-specific correspondence.

Runs in ~seconds (CPU-only; no Procrustes SVD, no CKA).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# --- import battery helpers ---
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from p5_validation_battery import (
    DEFAULT_DATA_ROOT, DEFAULT_OUT_DIR,
    RNG_SEED, N_BOOTSTRAP,
    collect_store, run_propdepth_pipeline,
    stats_from_cos_matrices, make_concept_perm,
)

import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-seeds", type=int, default=10,
                    help="Seeds for label shuffle (default 10, matches battery v2)")
    args = ap.parse_args()

    # patch globals used by collect_store / run_propdepth_pipeline
    import p5_validation_battery as bat
    bat.DATA_ROOT = args.data_root
    bat.OUT_DIR = args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.time()
    store, by_dim = collect_store()
    log.info("Loaded %d models, %d dim clusters",
             len(store), sum(1 for n in by_dim.values() if len(n) >= 2))

    seed_results = []
    for seed_i in range(args.n_seeds):
        seed = RNG_SEED + seed_i
        log.info("[shuf-no-rot] seed %d/%d (RNG=%d)", seed_i + 1, args.n_seeds, seed)
        t0 = time.time()
        cp = make_concept_perm(store, seed)
        rows = run_propdepth_pipeline(
            store, by_dim, f"shuf-no-rot-s{seed}",
            rotate=False,
            concept_perm_per_model=cp,
        )
        s = stats_from_cos_matrices(rows)
        s["seed"] = seed
        s["elapsed_seconds"] = time.time() - t0
        s["concept_perm_per_model"] = cp
        if seed_i == 0:
            s["pair_results"] = rows
        seed_results.append(s)
        log.info("[shuf-no-rot seed %d] n=%d Δ=%+.4f matched=%.4f mismatched=%.4f p=%.2e",
                 seed, s.get("n_observations", 0), s.get("mean_delta", float("nan")),
                 s.get("mean_matched", float("nan")), s.get("mean_mismatched", float("nan")),
                 s.get("mannwhitney_p", float("nan")))

    deltas = [s["mean_delta"] for s in seed_results]
    result = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "test": "6_shuf_no_rot",
        "description": "Shuffled concept labels + no Procrustes rotation (pure ARH floor)",
        "n_seeds": args.n_seeds,
        "per_seed": seed_results,
        "delta_across_seeds_mean": float(np.mean(deltas)),
        "delta_across_seeds_std": float(np.std(deltas)) if args.n_seeds > 1 else 0.0,
        "total_elapsed_seconds": time.time() - overall_t0,
    }

    out_path = args.out_dir / "p5_validation_test6_shuf_no_rot.json"
    out_path.write_text(json.dumps(result, indent=2))
    log.info("\n=== TEST 6 DONE in %.1fs ===", result["total_elapsed_seconds"])
    log.info("mean Δ across %d seeds = %+.4f ± %.4f",
             args.n_seeds, result["delta_across_seeds_mean"], result["delta_across_seeds_std"])
    log.info("Output: %s", out_path)


if __name__ == "__main__":
    main()

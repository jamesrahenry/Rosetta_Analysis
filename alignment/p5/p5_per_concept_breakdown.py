#!/usr/bin/env python3
"""Per-concept breakdown of the path-3 result.

Path 3's grand Δ=+0.134 hides per-concept variance: sentiment Δ=+0.251 was
~5× bigger than credibility Δ=+0.054. This script unpacks where that
variance comes from:
  - Concept-level summary (already in path-3 output)
  - Within-concept × cluster decomposition (which dim clusters drive
    each concept's Δ)
  - Per-pair contributions (which model pairs are doing the work)
  - Cross-concept correlation in pair-level Δ (do the same pairs win
    across all concepts, or do different pairs win on different concepts?)

Reads: rosetta_data/results/p5_permutation/p5_propdepth_samedim_results.json
Writes: rosetta_data/results/p5_permutation/p5_per_concept_breakdown.json

CPU-only, ~30 seconds runtime.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_INPUT = REPO_ROOT / "rosetta_data" / "results" / "p5_permutation" / "p5_propdepth_samedim_results.json"
DEFAULT_OUTPUT = REPO_ROOT / "rosetta_data" / "results" / "p5_permutation" / "p5_per_concept_breakdown.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    data = json.loads(args.input.read_text())
    pair_results = data["pair_results"]
    log.info("Loaded %d pair × concept observations from %s",
             len(pair_results), args.input.name)

    # ---- 1. Per-concept × cluster decomposition ----
    by_concept_cluster = defaultdict(lambda: defaultdict(list))
    for r in pair_results:
        by_concept_cluster[r["test_concept"]][r["dim"]].append(r["obs_delta"])

    concept_cluster_table = {}
    for c in sorted(by_concept_cluster):
        concept_cluster_table[c] = {}
        for dim in sorted(by_concept_cluster[c]):
            ds = by_concept_cluster[c][dim]
            concept_cluster_table[c][str(dim)] = {
                "n": len(ds),
                "mean_delta": float(np.mean(ds)),
                "median_delta": float(np.median(ds)),
                "min": float(np.min(ds)),
                "max": float(np.max(ds)),
                "n_positive": int(sum(1 for d in ds if d > 0)),
            }

    # ---- 2. Per-pair (across concepts) — which pairs drive the most signal? ----
    by_pair = defaultdict(list)
    for r in pair_results:
        key = f"{r['model_a']} → {r['model_b']}"
        by_pair[key].append({
            "concept": r["test_concept"],
            "delta": r["obs_delta"],
        })
    pair_table = {}
    for p, items in by_pair.items():
        ds = [it["delta"] for it in items]
        pair_table[p] = {
            "n_concepts": len(items),
            "mean_delta": float(np.mean(ds)),
            "median_delta": float(np.median(ds)),
            "n_positive": int(sum(1 for d in ds if d > 0)),
            "per_concept": {it["concept"]: it["delta"] for it in items},
        }

    # ---- 3. Cross-concept correlation in pair-level Δ ----
    # For each pair, build a vector of [delta_concept_1, ..., delta_concept_7].
    # Then compute concept × concept Pearson correlation across pairs.
    concepts_seen = sorted({r["test_concept"] for r in pair_results})
    pair_keys = sorted(by_pair)
    n_concepts = len(concepts_seen)
    n_pairs = len(pair_keys)
    matrix = np.full((n_pairs, n_concepts), np.nan)
    for i, pk in enumerate(pair_keys):
        per_c = pair_table[pk]["per_concept"]
        for j, c in enumerate(concepts_seen):
            if c in per_c:
                matrix[i, j] = per_c[c]

    # Pairwise correlation between concepts (drop NaNs per-pair).
    concept_corr = {}
    for j1, c1 in enumerate(concepts_seen):
        concept_corr[c1] = {}
        for j2, c2 in enumerate(concepts_seen):
            mask = ~(np.isnan(matrix[:, j1]) | np.isnan(matrix[:, j2]))
            if mask.sum() < 3:
                concept_corr[c1][c2] = float("nan")
                continue
            x = matrix[mask, j1]
            y = matrix[mask, j2]
            if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                concept_corr[c1][c2] = float("nan")
                continue
            concept_corr[c1][c2] = float(np.corrcoef(x, y)[0, 1])

    # ---- 4. Spread headline ----
    grand = data["summary"]["grand"]
    by_concept = data["summary"]["by_concept"]
    spread = {}
    for c, s in by_concept.items():
        spread[c] = {
            "mean_delta": s["mean_delta"],
            "ratio_to_grand": s["mean_delta"] / grand["mean_delta"] if grand["mean_delta"] else float("nan"),
        }
    sorted_spread = sorted(spread.items(), key=lambda kv: -kv[1]["mean_delta"])

    output = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source_file": str(args.input.name),
        "grand_mean_delta": grand["mean_delta"],
        "concept_spread_ranked": [
            {"concept": c, "mean_delta": v["mean_delta"], "ratio": v["ratio_to_grand"]}
            for c, v in sorted_spread
        ],
        "concept_x_cluster_table": concept_cluster_table,
        "per_pair_table": pair_table,
        "concept_correlation_matrix": concept_corr,
    }
    args.output.write_text(json.dumps(output, indent=2))

    log.info("\n=== CONCEPT SPREAD (ranked by Δ) ===")
    for c, v in sorted_spread:
        log.info("  %-16s Δ=%+.3f  (%.2f× grand)",
                 c, v["mean_delta"], v["ratio_to_grand"])

    log.info("\n=== TOP-5 PAIRS by mean Δ across concepts ===")
    pairs_by_delta = sorted(pair_table.items(), key=lambda kv: -kv[1]["mean_delta"])
    for pk, info in pairs_by_delta[:5]:
        log.info("  %-50s Δ=%+.3f  (%d/%d positive)",
                 pk, info["mean_delta"], info["n_positive"], info["n_concepts"])

    log.info("\n=== BOTTOM-5 PAIRS ===")
    for pk, info in pairs_by_delta[-5:]:
        log.info("  %-50s Δ=%+.3f  (%d/%d positive)",
                 pk, info["mean_delta"], info["n_positive"], info["n_concepts"])

    log.info("\n=== CONCEPT CORRELATIONS (off-diagonal) ===")
    for c1 in concepts_seen:
        for c2 in concepts_seen:
            if c1 < c2:
                r = concept_corr[c1][c2]
                if not np.isnan(r):
                    flag = " <==" if abs(r) > 0.5 else ""
                    log.info("  %-16s × %-16s r=%+.3f%s", c1, c2, r, flag)

    log.info("\nOutput: %s", args.output)


if __name__ == "__main__":
    main()

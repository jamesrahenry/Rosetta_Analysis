#!/usr/bin/env python3
"""Aggregate multimodal ablation interaction matrices into a dependency partition.

Reads all ablation_multimodal_*.json files produced by gem/ablate_multimodal.py
from <data-root>/<model-slug>/ablation_multimodal_<concept>.json, classifies each
directed off-diagonal pair as forward-dependent, backward-dependent, or independent,
and rolls up per-model, per-concept, and grand summary statistics.

Dependency classification (per directed pair i → j):
  - i ablated → j retains ≤ dep_threshold (default 60%) of baseline → j depends on i
  - Pair direction: i is "shallow" if cazs[i].depth_pct < cazs[j].depth_pct
    - shallow→deep: "forward_dep"
    - deep→shallow: "backward_dep"
  - retained > dep_threshold: "independent" (for that directed edge)

Only (model, concept) pairs with n_cazs ≥ 2 contribute observations.

Usage
-----
    uv run python gem/aggregate_multimodal_ablation.py \\
        --data-root ~/rosetta_data/paper_n250 \\
        --n-concepts 17 \\
        --dep-threshold 0.60 \\
        --out ~/rosetta_data/results/ablation_multimodal_summary_C17.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def classify_pair(
    caz_i: dict, caz_j: dict, retained_pct: float, threshold_pct: float
) -> str:
    """Classify directed edge i→j as forward_dep, backward_dep, or independent."""
    dependent = retained_pct <= threshold_pct
    if not dependent:
        return "independent"
    # shallow = lower depth_pct
    if caz_i["depth_pct"] < caz_j["depth_pct"]:
        return "forward_dep"   # ablating shallow reduces deep
    else:
        return "backward_dep"  # ablating deep reduces shallow


def process_file(path: Path, threshold_pct: float) -> list[dict] | None:
    """Return list of directed-pair observations from one ablation_multimodal file.

    Returns None if n_cazs < 2 (single-CAZ concepts are excluded by design).
    """
    data = json.loads(path.read_text())
    n_cazs = int(data.get("n_cazs", 0))
    if n_cazs < 2:
        return None

    model_id = data["model_id"]
    concept = data["concept"]
    cazs = data["cazs"]
    matrix = data["interaction_matrix"]  # matrix[i][j] = % of j retained when i ablated

    observations = []
    for i in range(n_cazs):
        for j in range(n_cazs):
            if i == j:
                continue
            retained_pct = float(matrix[i][j])
            label = classify_pair(cazs[i], cazs[j], retained_pct, threshold_pct)
            observations.append({
                "model_id": model_id,
                "concept": concept,
                "caz_i": i,
                "caz_j": j,
                "depth_pct_i": cazs[i]["depth_pct"],
                "depth_pct_j": cazs[j]["depth_pct"],
                "retained_pct": retained_pct,
                "label": label,
            })
    return observations


def count_labels(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"forward_dep": 0, "backward_dep": 0, "independent": 0}
    for r in rows:
        counts[r["label"]] += 1
    return counts


def fraction_summary(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: round(v / total, 4) for k, v in counts.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate multimodal ablation interaction matrices — C17 dependency partition"
    )
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Root directory with model subdirs (paper_n250/ layout)")
    ap.add_argument("--n-concepts", type=int, default=17,
                    help="Expected number of concepts (informational only; no filtering)")
    ap.add_argument("--dep-threshold", type=float, default=0.60,
                    help="Retained-pct threshold for dependency (default 0.60 = 60%%)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSON path")
    args = ap.parse_args()

    threshold_pct = args.dep_threshold * 100  # convert fraction → %

    # Glob all ablation_multimodal_*.json files
    files = sorted(args.data_root.glob("*/ablation_multimodal_*.json"))
    log.info("Found %d ablation_multimodal_*.json files under %s", len(files), args.data_root)

    all_observations: list[dict] = []
    skipped_single_caz = 0
    skipped_error = 0

    for path in files:
        try:
            obs = process_file(path, threshold_pct)
        except Exception as e:
            log.warning("Error reading %s: %s", path, e)
            skipped_error += 1
            continue
        if obs is None:
            skipped_single_caz += 1
            continue
        all_observations.extend(obs)

    log.info("Total directed-pair observations: %d", len(all_observations))
    log.info("Skipped (single-CAZ): %d  Skipped (error): %d",
             skipped_single_caz, skipped_error)

    if not all_observations:
        log.error("No observations — check data-root path.")
        return

    # --- Grand summary ---
    grand_counts = count_labels(all_observations)
    grand_fracs = fraction_summary(grand_counts)
    log.info("Grand:  forward_dep=%d (%.1f%%)  backward_dep=%d (%.1f%%)  independent=%d (%.1f%%)",
             grand_counts["forward_dep"], 100 * grand_fracs["forward_dep"],
             grand_counts["backward_dep"], 100 * grand_fracs["backward_dep"],
             grand_counts["independent"], 100 * grand_fracs["independent"])

    # --- Per-concept summary ---
    by_concept: dict[str, list[dict]] = defaultdict(list)
    for o in all_observations:
        by_concept[o["concept"]].append(o)

    per_concept = {}
    for concept, rows in sorted(by_concept.items()):
        counts = count_labels(rows)
        n_pairs = len(rows)
        n_model_concept = len({r["model_id"] for r in rows})
        per_concept[concept] = {
            "n_directed_pairs": n_pairs,
            "n_models_contributing": n_model_concept,
            "counts": counts,
            "fractions": fraction_summary(counts),
        }

    # --- Per-model summary ---
    by_model: dict[str, list[dict]] = defaultdict(list)
    for o in all_observations:
        by_model[o["model_id"]].append(o)

    per_model = {}
    for model_id, rows in sorted(by_model.items()):
        counts = count_labels(rows)
        per_model[model_id] = {
            "n_directed_pairs": len(rows),
            "n_concepts_contributing": len({r["concept"] for r in rows}),
            "counts": counts,
            "fractions": fraction_summary(counts),
        }

    output = {
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "method": "Multimodal ablation dependency partition — directed pair classification",
        "data_root": str(args.data_root),
        "dep_threshold_pct": threshold_pct,
        "n_concepts_expected": args.n_concepts,
        "n_files_found": len(files),
        "n_skipped_single_caz": skipped_single_caz,
        "n_skipped_error": skipped_error,
        "n_directed_pairs_total": len(all_observations),
        "note_single_caz": (
            "Single-CAZ concepts per model are excluded by design — "
            "multimodal interaction requires ≥2 CAZs."
        ),
        "grand": {
            "counts": grand_counts,
            "fractions": grand_fracs,
        },
        "per_concept": per_concept,
        "per_model": per_model,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    log.info("Output: %s", args.out)


if __name__ == "__main__":
    main()

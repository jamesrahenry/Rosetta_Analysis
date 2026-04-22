#!/usr/bin/env python3
"""
label_features.py — Per-layer concept alignment for deep dive features.

For each feature in a deep dive, checks alignment against concept dom_vectors
at EVERY layer the feature exists (not just the peak). Saves feature_labels.json
alongside the feature_map.json.

Usage:
    python src/label_features.py --model EleutherAI/pythia-1.4b
    python src/label_features.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CONCEPTS = ["credibility", "certainty", "sentiment", "moral_valence", "causation", "temporal_order", "negation"]

THRESHOLD = 0.5


def find_extraction_dir(model_id: str) -> Path | None:
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        sf = d / "run_summary.json"
        if not sf.exists():
            continue
        try:
            if json.load(open(sf)).get("model_id") == model_id:
                return d
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def find_deepdive_dir(model_id: str) -> Path | None:
    """Find the most recent deepdive dir whose feature_map.json has model_id == model_id.
    Matches on the actual model_id field to avoid slug collisions (e.g. base vs instruct)."""
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        fm = d / "feature_map.json"
        if not fm.exists():
            continue
        try:
            if json.load(open(fm)).get("model_id") == model_id:
                return d
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def label_model(model_id: str) -> dict | None:
    extraction_dir = find_extraction_dir(model_id)
    if not extraction_dir:
        log.warning("No extraction results for %s, skipping", model_id)
        return None

    deepdive_dir = find_deepdive_dir(model_id)
    if not deepdive_dir:
        log.warning("No deep dive results for %s, skipping", model_id)
        return None

    feature_map = json.load(open(deepdive_dir / "feature_map.json"))
    n_features = feature_map["n_features"]
    hidden_dim = feature_map["hidden_dim"]

    # Load concept dom_vectors per layer
    concept_dirs = {}
    for concept in CONCEPTS:
        caz_file = extraction_dir / f"caz_{concept}.json"
        if not caz_file.exists():
            continue
        cdata = json.load(open(caz_file))
        for m in cdata["layer_data"]["metrics"]:
            vec = np.array(m["dom_vector"], dtype=np.float64)
            if len(vec) == hidden_dim:
                vec = vec / (np.linalg.norm(vec) + 1e-12)
                concept_dirs[(concept, m["layer"])] = vec

    if not concept_dirs:
        log.warning("No concept directions for %s, skipping", model_id)
        return None

    # Label every feature at every layer
    feature_labels = {}
    n_any_labeled = 0

    for f in feature_map["features"]:
        fid = f["feature_id"]
        layer_alignments = []

        for li, layer in enumerate(f["layer_indices"]):
            pc_idx = f["pc_indices"][li]
            npy_file = deepdive_dir / f"directions_L{layer:03d}.npy"
            if not npy_file.exists():
                layer_alignments.append({"layer": layer, "best_concept": None, "best_cos": 0, "eigenvalue": f["eigenvalues"][li]})
                continue

            dirs = np.load(npy_file)
            if pc_idx >= len(dirs):
                layer_alignments.append({"layer": layer, "best_concept": None, "best_cos": 0, "eigenvalue": f["eigenvalues"][li]})
                continue

            fvec = dirs[pc_idx]
            fvec = fvec / (np.linalg.norm(fvec) + 1e-12)

            best_cos = 0
            best_concept = None
            for concept in CONCEPTS:
                key = (concept, layer)
                if key not in concept_dirs:
                    continue
                cos = abs(float(np.dot(fvec, concept_dirs[key])))
                if cos > best_cos:
                    best_cos = cos
                    best_concept = concept

            layer_alignments.append({
                "layer": layer,
                "best_concept": best_concept if best_cos >= THRESHOLD else None,
                "best_cos": round(best_cos, 4),
                "eigenvalue": f["eigenvalues"][li],
            })

        feature_labels[fid] = layer_alignments
        if any(la["best_concept"] is not None for la in layer_alignments):
            n_any_labeled += 1

    # Save
    out = {
        "model_id": model_id,
        "threshold": THRESHOLD,
        "n_features": n_features,
        "n_labeled": n_any_labeled,
        "n_unlabeled": n_features - n_any_labeled,
        "features": {str(k): v for k, v in feature_labels.items()},
    }

    out_path = deepdive_dir / "feature_labels.json"
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2)

    # Count handoffs
    n_handoff = sum(1 for fid, layers in feature_labels.items()
                    if len(set(la["best_concept"] for la in layers if la["best_concept"])) > 1)

    log.info("  %s: %d/%d labeled (%d handoffs), saved to %s",
             model_id.split("/")[-1], n_any_labeled, n_features, n_handoff, out_path.name)

    return out


def get_all_models() -> list[str]:
    models = set()
    for d in RESULTS_DIR.iterdir():
        if d.name.startswith("deepdive_") and (d / "feature_map.json").exists():
            fm = json.load(open(d / "feature_map.json"))
            models.add(fm["model_id"])
    return sorted(models)


def main():
    parser = argparse.ArgumentParser(description="Label deep dive features with per-layer concept alignment")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Single model ID")
    group.add_argument("--all", action="store_true", help="All models with deep dive results")
    args = parser.parse_args()

    if args.all:
        models = get_all_models()
        log.info("Labeling %d models...", len(models))
    else:
        models = [args.model]

    results = []
    for model_id in models:
        result = label_model(model_id)
        if result:
            results.append(result)

    if results:
        log.info("")
        log.info("=== SUMMARY ===")
        total_features = sum(r["n_features"] for r in results)
        total_labeled = sum(r["n_labeled"] for r in results)
        log.info("  Models: %d", len(results))
        log.info("  Total features: %d", total_features)
        log.info("  Labeled: %d (%.0f%%)", total_labeled, 100 * total_labeled / total_features)
        log.info("  Unlabeled: %d (%.0f%%)", total_features - total_labeled,
                 100 * (total_features - total_labeled) / total_features)


if __name__ == "__main__":
    main()

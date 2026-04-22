#!/usr/bin/env python3
"""
compute_shared_pca.py — Compute and save a shared PCA coordinate frame.

All visualization scripts should load coordinates from this file rather
than recomputing PCA independently. This ensures the same spatial layout
regardless of which features are shown/hidden.

Outputs:
    results/<deepdive_dir>/shared_coords.npz
        - coords_2d: [n_vectors, 2] PCA coordinates
        - labels: structured array with (type, id, layer) per vector
        - pca_components: [2, hidden_dim] for projecting new vectors
        - pca_mean: [hidden_dim] centering vector
        - axis_ranges: [x_min, x_max, y_min, y_max] with padding

Usage:
    python src/compute_shared_pca.py --model EleutherAI/pythia-1.4b
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CONCEPTS = ["credibility", "certainty", "sentiment", "moral_valence", "causation", "temporal_order", "negation"]


def load_model_data(model_id: str):
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        sf = d / "run_summary.json"
        if not sf.exists():
            continue
        try:
            if json.load(open(sf)).get("model_id") == model_id:
                data = {"model_id": model_id, "concepts": {}}
                for concept in CONCEPTS:
                    caz_file = d / f"caz_{concept}.json"
                    if not caz_file.exists():
                        continue
                    data["concepts"][concept] = json.load(open(caz_file))
                    data["n_layers"] = int(data["concepts"][concept]["n_layers"])
                    data["hidden_dim"] = int(data["concepts"][concept]["hidden_dim"])
                return data
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def load_deep_dive(model_id: str):
    model_slug = model_id.replace("/", "_").replace("-", "_")
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if not d.name.startswith(f"deepdive_{model_slug}"):
            continue
        fm_file = d / "feature_map.json"
        if not fm_file.exists():
            continue
        feature_map = json.load(open(fm_file))
        directions = {}
        for layer_idx in range(feature_map["n_layers"]):
            npy_file = d / f"directions_L{layer_idx:03d}.npy"
            if npy_file.exists():
                directions[layer_idx] = np.load(npy_file)
        return feature_map, directions, d
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Compute shared PCA coordinate frame")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-1.4b")
    args = parser.parse_args()

    log.info("Loading data for %s...", args.model)
    model_data = load_model_data(args.model)
    if not model_data:
        log.error("No CAZ extraction data for %s", args.model)
        return

    feature_map, directions, deepdive_dir = load_deep_dive(args.model)
    if not feature_map:
        log.error("No deep dive data for %s", args.model)
        return

    n_layers = model_data["n_layers"]
    hidden_dim = model_data["hidden_dim"]

    # ── Collect ALL vectors: concepts + persistent + transient ──
    all_vectors = []
    label_types = []
    label_ids = []
    label_layers = []

    # Concept dom_vectors
    for concept, cdata in model_data["concepts"].items():
        for m in cdata["layer_data"]["metrics"]:
            vec = np.array(m["dom_vector"], dtype=np.float64)
            if len(vec) == hidden_dim:
                all_vectors.append(vec)
                label_types.append("concept")
                label_ids.append(concept)
                label_layers.append(m["layer"])

    # All features from deep dive (persistent and transient)
    for f in feature_map["features"]:
        fid = str(f["feature_id"])

        # Peak direction
        peak_layer = f["peak_layer"]
        if peak_layer in f["layer_indices"]:
            pi = f["pc_indices"][f["layer_indices"].index(peak_layer)]
        else:
            pi = f["pc_indices"][0]
        if peak_layer in directions and pi < len(directions[peak_layer]):
            vec = directions[peak_layer][pi]
            if len(vec) == hidden_dim:
                all_vectors.append(vec)
                label_types.append("dark_peak" if f["lifespan"] >= 3 else "transient")
                label_ids.append(fid)
                label_layers.append(peak_layer)

        # Trajectory directions (persistent features only)
        if f["lifespan"] >= 3:
            for li, layer in enumerate(f["layer_indices"]):
                pi = f["pc_indices"][li]
                if layer in directions and pi < len(directions[layer]):
                    vec = directions[layer][pi]
                    if len(vec) == hidden_dim:
                        all_vectors.append(vec)
                        label_types.append("dark_traj")
                        label_ids.append(fid)
                        label_layers.append(layer)

    all_vectors = np.array(all_vectors)
    n_concept = sum(1 for t in label_types if t == "concept")
    n_dark = sum(1 for t in label_types if t.startswith("dark"))
    n_trans = sum(1 for t in label_types if t == "transient")
    log.info("PCA on %d vectors (%d concept + %d dark + %d transient)",
             len(all_vectors), n_concept, n_dark, n_trans)

    # ── PCA ──
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(all_vectors)

    # ── Compute fixed axis ranges with 10% padding ──
    x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
    y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()
    x_pad = 0.1 * (x_max - x_min)
    y_pad = 0.1 * (y_max - y_min)
    axis_ranges = np.array([x_min - x_pad, x_max + x_pad,
                            y_min - y_pad, y_max + y_pad])

    # ── Save ──
    out_path = deepdive_dir / "shared_coords.npz"
    np.savez(
        out_path,
        coords_2d=coords_2d,
        label_types=np.array(label_types, dtype="U20"),
        label_ids=np.array(label_ids, dtype="U40"),
        label_layers=np.array(label_layers, dtype=np.int32),
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        axis_ranges=axis_ranges,
        n_layers=np.array([n_layers]),
        explained_variance_ratio=pca.explained_variance_ratio_,
    )

    log.info("Saved shared coordinates to %s", out_path)
    log.info("  %d vectors, axis ranges: x=[%.2f, %.2f] y=[%.2f, %.2f]",
             len(all_vectors), *axis_ranges)
    log.info("  PCA explained variance: %.1f%% + %.1f%%",
             100 * pca.explained_variance_ratio_[0],
             100 * pca.explained_variance_ratio_[1])
    log.info("Done.")


if __name__ == "__main__":
    main()

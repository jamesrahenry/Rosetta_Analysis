"""
align_depth_matched.py — Depth-matched Procrustes alignment across architectures.

The original align.py tests PRH by comparing concept vectors at each model's
peak layer. This script tests the REFINED PRH: do sub-representations at
matched processing depths align better than at mismatched depths?

For each multimodal concept × model pair:
  1. Detect assembly regions (shallow peak, deep peak) via find_caz_regions
  2. Load all-layer calibration activations (calibration_alllayer_{concept}.npy)
  3. Fit Procrustes rotation at each pair of region peaks
  4. Compare: shallow↔shallow vs shallow↔deep alignment

Requires re-extraction with the updated extract.py that saves all-layer
calibration files.

Usage
-----
    # All concepts, all models
    python src/align_depth_matched.py --all

    # Single concept
    python src/align_depth_matched.py --concept credibility
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from rosetta_tools.alignment import align_and_score, cosine_similarity
from rosetta_tools.caz import find_caz_regions, LayerMetrics
from rosetta_tools.viz import CONCEPT_META
from rosetta_tools.paths import ROSETTA_RESULTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_ROOT = ROSETTA_RESULTS
OUT_DIR = Path("visualizations/depth_matched")
CONCEPTS = list(CONCEPT_META.keys())


def load_model_data(concept: str) -> dict:
    """Load dom_vectors and all-layer calibrations for multimodal models.

    Returns dict: model_id → {dom_vecs, all_layer_cal, n_layers, hidden_dim,
                               shallow_peak, deep_peak, regions}
    """
    models = {}

    for d in sorted(RESULTS_ROOT.glob("xarch_*")):
        checkpoint = d / f"caz_{concept}.json"
        alllayer_path = d / f"calibration_alllayer_{concept}.npy"

        if not checkpoint.exists():
            continue

        with checkpoint.open() as f:
            data = json.load(f)

        if data.get("concept") != concept:
            continue

        model_id = data["model_id"]
        layer_data = data["layer_data"]
        n_layers = layer_data["n_layers"]
        hidden_dim = data["hidden_dim"]
        layer_metrics_raw = layer_data["metrics"]

        # Check for consistent dom_vector dims
        dims = set(len(m["dom_vector"]) for m in layer_metrics_raw)
        if len(dims) > 1:
            log.warning("  Skipping %s: inconsistent dom_vector dims", model_id)
            continue

        # Detect regions
        metrics = [
            LayerMetrics(m["layer"], m["separation_fisher"],
                         m["coherence"], m["velocity"])
            for m in layer_metrics_raw
        ]
        profile = find_caz_regions(metrics)

        # Load all-layer calibration if available
        if alllayer_path.exists():
            all_layer_cal = np.load(alllayer_path)  # [n_layers, n_texts, hidden_dim]
        else:
            all_layer_cal = None

        # Dom vectors at every layer
        dom_vecs = np.array([m["dom_vector"] for m in layer_metrics_raw],
                            dtype=np.float64)
        norms = np.linalg.norm(dom_vecs, axis=1, keepdims=True)
        dom_vecs = dom_vecs / (norms + 1e-10)

        sorted_regions = sorted(profile.regions, key=lambda r: r.peak)

        models[model_id] = {
            "dom_vecs": dom_vecs,
            "all_layer_cal": all_layer_cal,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "shallow_peak": sorted_regions[0].peak,
            "deep_peak": sorted_regions[-1].peak if len(sorted_regions) > 1 else sorted_regions[0].peak,
            "is_multimodal": profile.is_multimodal,
            "profile": profile,
        }

    return models


def align_pair_at_layers(
    src_data: dict,
    tgt_data: dict,
    src_layer: int,
    tgt_layer: int,
) -> dict:
    """Align two models at specific layers and return cosine similarity.

    Uses all-layer calibration activations if available, otherwise falls
    back to dom_vector-based regression alignment.
    """
    src_vec = src_data["dom_vecs"][src_layer]
    tgt_vec = tgt_data["dom_vecs"][tgt_layer]

    # Try full Procrustes with calibration activations
    if (src_data["all_layer_cal"] is not None and
            tgt_data["all_layer_cal"] is not None):
        src_acts = src_data["all_layer_cal"][src_layer].astype(np.float64)
        tgt_acts = tgt_data["all_layer_cal"][tgt_layer].astype(np.float64)

        result = align_and_score(src_vec, tgt_vec, src_acts, tgt_acts)
        return result

    # Fallback: dom_vector regression alignment (less precise but works
    # without calibration activations)
    n1, n2 = src_data["n_layers"], tgt_data["n_layers"]
    dim1, dim2 = src_data["hidden_dim"], tgt_data["hidden_dim"]

    n_common = min(n1, n2)
    if n_common < 4:
        return {"aligned_cosine": float("nan"), "raw_cosine": float("nan"),
                "method": "insufficient_layers"}

    grid = np.linspace(0, 1, n_common)
    idx1 = np.clip(np.round(grid * (n1 - 1)).astype(int), 0, n1 - 1)
    idx2 = np.clip(np.round(grid * (n2 - 1)).astype(int), 0, n2 - 1)
    V1 = src_data["dom_vecs"][idx1]
    V2 = tgt_data["dom_vecs"][idx2]

    if dim1 == dim2:
        from scipy.linalg import orthogonal_procrustes
        R, _ = orthogonal_procrustes(V2, V1)
        tgt_rot = tgt_vec @ R
        aligned = cosine_similarity(src_vec, tgt_rot)
    else:
        W, _, _, _ = np.linalg.lstsq(V2, V1, rcond=None)
        tgt_proj = tgt_vec @ W
        tgt_proj = tgt_proj / (np.linalg.norm(tgt_proj) + 1e-10)
        aligned = cosine_similarity(src_vec, tgt_proj)

    same_dim = dim1 == dim2
    raw = cosine_similarity(src_vec, tgt_vec) if same_dim else float("nan")

    return {"aligned_cosine": aligned, "raw_cosine": raw,
            "method": "procrustes" if same_dim else "regression"}


def analyze_concept(concept: str) -> pd.DataFrame | None:
    """Run depth-matched alignment analysis for one concept."""
    log.info("=== %s ===", concept)
    models = load_model_data(concept)

    # Filter to multimodal models
    multi_models = {k: v for k, v in models.items() if v["is_multimodal"]}
    log.info("  %d models total, %d multimodal", len(models), len(multi_models))

    if len(multi_models) < 2:
        log.warning("  <2 multimodal models — skipping")
        return None

    rows = []
    for (src_id, src), (tgt_id, tgt) in combinations(multi_models.items(), 2):
        # Four comparisons: SS, DD, SD, DS
        for src_depth, tgt_depth, label in [
            ("shallow_peak", "shallow_peak", "S-S"),
            ("deep_peak", "deep_peak", "D-D"),
            ("shallow_peak", "deep_peak", "S-D"),
            ("deep_peak", "shallow_peak", "D-S"),
        ]:
            src_layer = src[src_depth]
            tgt_layer = tgt[tgt_depth]

            result = align_pair_at_layers(src, tgt, src_layer, tgt_layer)

            rows.append({
                "concept": concept,
                "source_model": src_id,
                "target_model": tgt_id,
                "comparison": label,
                "src_layer": src_layer,
                "tgt_layer": tgt_layer,
                "src_depth_pct": 100.0 * src_layer / src["n_layers"],
                "tgt_depth_pct": 100.0 * tgt_layer / tgt["n_layers"],
                "aligned_cosine": result.get("aligned_cosine", float("nan")),
                "raw_cosine": result.get("raw_cosine", float("nan")),
                "method": result.get("method", "full_procrustes"),
            })

    df = pd.DataFrame(rows)

    # Summary
    for label in ["S-S", "D-D", "S-D", "D-S"]:
        sub = df[df["comparison"] == label]["aligned_cosine"].dropna()
        if not sub.empty:
            log.info("  %s: mean=%.3f  std=%.3f  n=%d",
                     label, sub.mean(), sub.std(), len(sub))

    matched = df[df["comparison"].isin(["S-S", "D-D"])]["aligned_cosine"].dropna()
    mismatched = df[df["comparison"].isin(["S-D", "D-S"])]["aligned_cosine"].dropna()

    if len(matched) >= 4 and len(mismatched) >= 4:
        stat, p = mannwhitneyu(np.abs(matched), np.abs(mismatched),
                               alternative="greater")
        log.info("  matched=%.3f  mismatched=%.3f  p=%.4f",
                 matched.mean(), mismatched.mean(), p)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Depth-matched Procrustes alignment — refined PRH")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--concept", type=str, choices=CONCEPTS)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    concepts = CONCEPTS if args.all else [args.concept]
    all_dfs = []

    for concept in concepts:
        df = analyze_concept(concept)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        log.error("No results produced.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = OUT_DIR / "depth_matched_alignment.csv"
    combined.to_csv(out_path, index=False)
    log.info("Results → %s", out_path)

    # Grand summary
    print()
    print("=" * 70)
    print("GRAND SUMMARY — Depth-matched PRH alignment")
    print("=" * 70)

    for concept in combined["concept"].unique():
        csub = combined[combined["concept"] == concept]
        matched = csub[csub["comparison"].isin(["S-S", "D-D"])]["aligned_cosine"].dropna()
        mismatched = csub[csub["comparison"].isin(["S-D", "D-S"])]["aligned_cosine"].dropna()

        if len(matched) < 2 or len(mismatched) < 2:
            continue

        if len(matched) >= 4 and len(mismatched) >= 4:
            _, p = mannwhitneyu(np.abs(matched), np.abs(mismatched),
                                alternative="greater")
        else:
            p = float("nan")

        sig = "**" if p < 0.01 else "* " if p < 0.05 else "  "
        print(f"  {sig} {concept:>20s}: matched={matched.mean():+.3f}  "
              f"mis={mismatched.mean():+.3f}  "
              f"delta={matched.mean()-mismatched.mean():+.3f}  p={p:.4f}")

    # Overall
    all_matched = combined[combined["comparison"].isin(["S-S", "D-D"])]["aligned_cosine"].dropna()
    all_mismatched = combined[combined["comparison"].isin(["S-D", "D-S"])]["aligned_cosine"].dropna()
    print(f"\n  Grand matched:    {all_matched.mean():+.3f}")
    print(f"  Grand mismatched: {all_mismatched.mean():+.3f}")


if __name__ == "__main__":
    main()

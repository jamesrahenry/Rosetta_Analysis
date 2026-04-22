#!/usr/bin/env python3
"""
analyze_coasting.py — CKA-derived CAZ boundaries and relay feature switch locations.

Two analyses:

1. CKA BOUNDARY REFINEMENT
   Fisher detection finds where concept assembly peaks (velocity maximum).
   Adjacent-layer CKA measures representational stability: high CKA = coasting,
   low CKA = active transformation. The CKA dip around each Fisher peak defines
   the CAZ extent more precisely than velocity thresholds alone.

   Algorithm:
   - Detrend the CKA curve (subtract smoothed baseline) to isolate local dips
   - Find local minima of the detrended curve near each Fisher peak
   - Define CKA-bounded CAZ: contiguous layers where detrended CKA < -threshold
   - Compare CKA-derived extents to Fisher-derived regions

2. RELAY FEATURE SWITCH LOCATIONS
   Relay features carry different concepts at different layers (concept handoffs).
   Where do the switches happen — inside active CAZ regions or in coasting
   intermediate space?

   If switches cluster in coasting regions, intermediate space is doing semantic
   work (concept re-labeling while the stream is stable). If they cluster at
   CAZ peaks, switches are part of the assembly event itself.

Usage
-----
    python src/analyze_coasting.py
    python src/analyze_coasting.py --threshold 0.003
    python src/analyze_coasting.py --output-dir results/coasting_analysis

Results written to:
    results/coasting_analysis/boundary_comparison.json
    results/coasting_analysis/relay_switch_locations.json
    results/coasting_analysis/summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_ROOT = Path("results")
CONCEPTS = ["credibility", "negation", "sentiment", "causation",
            "certainty", "moral_valence", "temporal_order"]


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def find_cka_dir(model_id: str) -> Path | None:
    """Find the most recent result dir with CKA files for this model."""
    candidates = []
    for d in sorted(RESULTS_ROOT.iterdir(), reverse=True):
        summary = d / "run_summary.json"
        if d.is_dir() and summary.exists():
            try:
                if json.loads(summary.read_text()).get("model_id") == model_id:
                    if any(d.glob("cka_*.json")):
                        candidates.append(d)
            except Exception:
                continue
    return candidates[0] if candidates else None


def find_deepdive_dir(model_id: str) -> Path | None:
    """Find the most recent deepdive dir with feature_labels.json for this model."""
    candidates = []
    for d in sorted(RESULTS_ROOT.iterdir(), reverse=True):
        fl = d / "feature_labels.json"
        if fl.exists():
            try:
                mid = json.loads(fl.read_text()).get("model_id")
                if mid == model_id:
                    candidates.append(d)
            except Exception:
                continue
    return candidates[0] if candidates else None


def discover_models() -> list[str]:
    """Find all models with both CKA and feature_labels data."""
    cka_models = set()
    dd_models = set()

    for d in RESULTS_ROOT.iterdir():
        if not d.is_dir():
            continue
        summary = d / "run_summary.json"
        if summary.exists() and any(d.glob("cka_*.json")):
            try:
                mid = json.loads(summary.read_text()).get("model_id")
                if mid:
                    cka_models.add(mid)
            except Exception:
                pass
        fl = d / "feature_labels.json"
        if fl.exists():
            try:
                mid = json.loads(fl.read_text()).get("model_id")
                if mid:
                    dd_models.add(mid)
            except Exception:
                pass

    both = sorted(cka_models & dd_models)
    log.info("Models with both CKA and feature_labels: %d", len(both))
    return both


# ---------------------------------------------------------------------------
# CKA boundary derivation
# ---------------------------------------------------------------------------

def detrend_cka(cka: list[float], smooth_window: int = 5) -> np.ndarray:
    """
    Detrend CKA curve to isolate local dips.

    CKA rises monotonically with depth (shallow layers transform more).
    Subtracting a smoothed local maximum reveals where CKA drops below
    its local context — the active transformation windows (CAZ events).

    Returns detrended curve: negative values = CKA below local baseline.
    """
    arr = np.array(cka, dtype=np.float64)
    # Smooth with a running maximum to get the local ceiling
    baseline = uniform_filter1d(arr, size=smooth_window, mode="nearest")
    # Use the smoothed curve as baseline (running mean approximates local ceiling
    # better than running max for these very-high-CKA curves)
    return arr - baseline


def find_cka_caz_extents(
    cka_adjacent: list[float],
    fisher_peaks: list[int],
    threshold: float = 0.003,
    smooth_window: int = 5,
) -> list[dict]:
    """
    For each Fisher peak, find the CKA-derived CAZ extent.

    The extent is the contiguous run of layers around the peak where the
    detrended CKA is below -threshold (i.e., CKA is suppressed relative
    to its local context). If no layers qualify, fall back to the single
    peak layer.

    Returns list of dicts: {peak, cka_start, cka_end, min_detrended, min_layer}
    """
    detrended = detrend_cka(cka_adjacent, smooth_window)
    n = len(cka_adjacent)  # n = n_layers - 1 (adjacent pairs)
    results = []

    for peak in fisher_peaks:
        # Convert Fisher peak (layer index) to CKA pair index
        # Pair i connects layer i and layer i+1; peak at layer P
        # corresponds most directly to pair P-1 (before peak) and P (after peak)
        # We look in a window around the peak
        p_pair = max(0, min(peak - 1, n - 1))

        # Find the local minimum of detrended CKA nearest the Fisher peak
        window = 3
        lo = max(0, p_pair - window)
        hi = min(n, p_pair + window + 1)
        local_detrended = detrended[lo:hi]
        local_min_idx = lo + int(np.argmin(local_detrended))
        min_val = float(detrended[local_min_idx])

        # Walk outward from local_min_idx while detrended < -threshold
        if min_val < -threshold:
            start = local_min_idx
            while start > 0 and detrended[start - 1] < -threshold:
                start -= 1
            end = local_min_idx
            while end < n - 1 and detrended[end + 1] < -threshold:
                end += 1
        else:
            # No clear dip — record the single pair as the extent
            start = end = local_min_idx

        results.append({
            "fisher_peak": peak,
            "cka_pair_start": start,
            "cka_pair_end": end,
            # Convert pair indices back to layer indices
            # Pair i = transition between layer i and i+1
            # Layer extent: start of pair_start to end of pair_end
            "cka_layer_start": start,
            "cka_layer_end": end + 1,
            "min_detrended": round(min_val, 6),
            "min_pair_idx": local_min_idx,
            "dip_detected": min_val < -threshold,
        })

    return results


def label_layers(n_layers: int, caz_extents: list[dict]) -> list[str]:
    """
    Label each layer as 'caz', 'coasting', or 'transition'.

    caz:        within a CKA-derived CAZ extent
    coasting:   outside all CAZ extents (stable, high-CKA plateau)
    transition: within 1 layer of a CAZ boundary (optional, for future use)
    """
    labels = ["coasting"] * n_layers
    for ext in caz_extents:
        for l in range(ext["cka_layer_start"], ext["cka_layer_end"] + 1):
            if 0 <= l < n_layers:
                labels[l] = "caz"
    return labels


# ---------------------------------------------------------------------------
# Relay feature analysis
# ---------------------------------------------------------------------------

def extract_relay_switches(feature_labels: dict) -> list[dict]:
    """
    Find all concept switches in relay features.

    A relay feature changes its best_concept at least once (above threshold).
    Returns list of switch events: {feature_id, switch_layer, from_concept, to_concept}.
    """
    switches = []
    features = feature_labels.get("features", {})

    for fid, layers in features.items():
        if not isinstance(layers, list):
            continue

        # Build sequence of (layer, concept) where concept is not None
        labeled = [(e["layer"], e["best_concept"])
                   for e in layers if e.get("best_concept") is not None]

        if len(labeled) < 2:
            continue

        # Find concept changes
        prev_concept = labeled[0][1]
        for layer, concept in labeled[1:]:
            if concept != prev_concept:
                switches.append({
                    "feature_id": fid,
                    "switch_layer": layer,
                    "from_concept": prev_concept,
                    "to_concept": concept,
                })
            prev_concept = concept

    return switches


def classify_switch(switch_layer: int, fisher_regions: list[dict],
                    layer_labels: list[str]) -> dict:
    """
    Classify a relay switch layer using both Fisher regions and CKA labels.

    fisher_classification: 'caz' if inside a Fisher region, 'inter_caz' otherwise
    cka_classification: 'caz' or 'coasting' from CKA-derived labels
    """
    # Fisher classification
    fisher_class = "inter_caz"
    fisher_region_idx = None
    for i, region in enumerate(fisher_regions):
        if region["start"] <= switch_layer <= region["end"]:
            fisher_class = "caz"
            fisher_region_idx = i
            break

    # CKA classification
    cka_class = "unknown"
    if 0 <= switch_layer < len(layer_labels):
        cka_class = layer_labels[switch_layer]

    return {
        "fisher": fisher_class,
        "cka": cka_class,
        "fisher_region_idx": fisher_region_idx,
    }


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def analyze_model(model_id: str, threshold: float) -> dict | None:
    cka_dir = find_cka_dir(model_id)
    dd_dir = find_deepdive_dir(model_id)

    if cka_dir is None or dd_dir is None:
        log.warning("Missing data for %s — skipping", model_id)
        return None

    log.info("=== %s ===", model_id)

    # Load feature labels
    try:
        feature_labels = json.loads((dd_dir / "feature_labels.json").read_text())
    except Exception as e:
        log.error("Failed to load feature_labels for %s: %s", model_id, e)
        return None

    relay_switches = extract_relay_switches(feature_labels)
    log.info("  Relay switches found: %d", len(relay_switches))

    # Per-concept CKA analysis
    concept_results = {}
    all_layer_labels = {}  # concept -> layer_labels list

    for concept in CONCEPTS:
        cka_path = cka_dir / f"cka_{concept}.json"
        if not cka_path.exists():
            continue

        try:
            cka_data = json.loads(cka_path.read_text())
        except Exception:
            continue

        cka_adj = cka_data.get("cka_adjacent", [])
        fisher_regions = cka_data.get("caz_regions", [])
        n_layers = cka_data.get("n_layers", len(cka_adj) + 1)

        if not cka_adj or not fisher_regions:
            continue

        # Fisher peak layers
        fisher_peaks = [r["peak"] for r in fisher_regions]

        # CKA-derived extents
        cka_extents = find_cka_caz_extents(cka_adj, fisher_peaks, threshold=threshold)

        # Layer labels
        layer_labels = label_layers(n_layers, cka_extents)
        all_layer_labels[concept] = layer_labels

        # Compare Fisher vs CKA boundaries
        boundary_comparisons = []
        for fisher_region, cka_extent in zip(fisher_regions, cka_extents):
            fisher_width = fisher_region["end"] - fisher_region["start"]
            cka_width = cka_extent["cka_layer_end"] - cka_extent["cka_layer_start"]
            boundary_comparisons.append({
                "peak": fisher_region["peak"],
                "fisher_start": fisher_region["start"],
                "fisher_end": fisher_region["end"],
                "fisher_width": fisher_width,
                "cka_start": cka_extent["cka_layer_start"],
                "cka_end": cka_extent["cka_layer_end"],
                "cka_width": cka_width,
                "dip_detected": cka_extent["dip_detected"],
                "min_detrended": cka_extent["min_detrended"],
            })

        concept_results[concept] = {
            "n_fisher_regions": len(fisher_regions),
            "n_cka_dips_detected": sum(1 for e in cka_extents if e["dip_detected"]),
            "boundary_comparisons": boundary_comparisons,
            "layer_labels": layer_labels,
        }

    if not concept_results:
        log.warning("  No concept results for %s", model_id)
        return None

    # Relay switch classification
    # Use majority-vote concept for layer label (most concepts agree)
    switch_results = []
    for sw in relay_switches:
        sl = sw["switch_layer"]

        # Classify by Fisher (use credibility as representative — it has the most data)
        # Actually use all available concepts and take the most common classification
        fisher_votes = []
        cka_votes = []

        for concept, cres in concept_results.items():
            fisher_regions = []
            cka_dir_concept = cka_dir / f"cka_{concept}.json"
            if cka_dir_concept.exists():
                try:
                    cd = json.loads(cka_dir_concept.read_text())
                    fisher_regions = cd.get("caz_regions", [])
                except Exception:
                    pass

            layer_labels = cres["layer_labels"]
            classification = classify_switch(sl, fisher_regions, layer_labels)
            fisher_votes.append(classification["fisher"])
            cka_votes.append(classification["cka"])

        # Majority vote
        fisher_class = max(set(fisher_votes), key=fisher_votes.count) if fisher_votes else "unknown"
        cka_class = max(set(cka_votes), key=cka_votes.count) if cka_votes else "unknown"

        switch_results.append({
            **sw,
            "fisher_classification": fisher_class,
            "cka_classification": cka_class,
            "fisher_votes": fisher_votes,
            "cka_votes": cka_votes,
        })

    log.info("  Relay switches classified: %d", len(switch_results))

    return {
        "model_id": model_id,
        "n_relay_switches": len(relay_switches),
        "concept_results": concept_results,
        "relay_switches": switch_results,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(model_results: list[dict]) -> dict:
    all_switches = []
    boundary_stats = defaultdict(list)

    for mr in model_results:
        all_switches.extend(mr["relay_switches"])
        for concept, cres in mr["concept_results"].items():
            for bc in cres["boundary_comparisons"]:
                if bc["dip_detected"]:
                    boundary_stats["fisher_width"].append(bc["fisher_width"])
                    boundary_stats["cka_width"].append(bc["cka_width"])
                    boundary_stats["width_ratio"].append(
                        bc["cka_width"] / bc["fisher_width"]
                        if bc["fisher_width"] > 0 else None
                    )

    # Relay switch summary
    n_switches = len(all_switches)
    if n_switches > 0:
        fisher_in_caz = sum(1 for s in all_switches if s["fisher_classification"] == "caz")
        fisher_inter = sum(1 for s in all_switches if s["fisher_classification"] == "inter_caz")
        cka_in_caz = sum(1 for s in all_switches if s["cka_classification"] == "caz")
        cka_coasting = sum(1 for s in all_switches if s["cka_classification"] == "coasting")

        # Direction pattern
        directions = defaultdict(int)
        for s in all_switches:
            directions[f"{s['from_concept']} → {s['to_concept']}"] += 1
        top_directions = sorted(directions.items(), key=lambda x: -x[1])[:10]
    else:
        fisher_in_caz = fisher_inter = cka_in_caz = cka_coasting = 0
        top_directions = []

    # Boundary width stats
    width_ratios = [r for r in boundary_stats["width_ratio"] if r is not None]

    return {
        "timestamp": datetime.now().isoformat(),
        "n_models": len(model_results),
        "n_relay_switches_total": n_switches,
        "relay_switch_locations": {
            "fisher_in_caz": fisher_in_caz,
            "fisher_in_caz_pct": round(100 * fisher_in_caz / n_switches, 1) if n_switches else None,
            "fisher_inter_caz": fisher_inter,
            "fisher_inter_caz_pct": round(100 * fisher_inter / n_switches, 1) if n_switches else None,
            "cka_in_caz": cka_in_caz,
            "cka_in_caz_pct": round(100 * cka_in_caz / n_switches, 1) if n_switches else None,
            "cka_coasting": cka_coasting,
            "cka_coasting_pct": round(100 * cka_coasting / n_switches, 1) if n_switches else None,
        },
        "top_switch_directions": top_directions,
        "boundary_width_comparison": {
            "n_dips_detected": len(width_ratios),
            "mean_fisher_width": round(float(np.mean(boundary_stats["fisher_width"])), 2) if boundary_stats["fisher_width"] else None,
            "mean_cka_width": round(float(np.mean(boundary_stats["cka_width"])), 2) if boundary_stats["cka_width"] else None,
            "mean_width_ratio": round(float(np.mean(width_ratios)), 3) if width_ratios else None,
            "note": "cka_width / fisher_width; <1 means CKA boundary is tighter than Fisher"
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CKA-derived CAZ boundaries and relay feature switch locations"
    )
    parser.add_argument("--threshold", type=float, default=0.003,
                        help="Detrended CKA deficit threshold for CAZ boundary (default: 0.003)")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Smoothing window for CKA detrending (default: 5)")
    parser.add_argument("--output-dir", type=str, default="results/coasting_analysis",
                        help="Output directory")
    parser.add_argument("--model", type=str, default=None,
                        help="Single model ID (default: all)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else discover_models()
    log.info("Running on %d models", len(models))

    model_results = []
    for model_id in models:
        result = analyze_model(model_id, threshold=args.threshold)
        if result is not None:
            model_results.append(result)

    if not model_results:
        log.error("No results produced")
        return

    # Write per-model results
    per_model_path = out_dir / "per_model.json"
    per_model_path.write_text(json.dumps(model_results, indent=2))
    log.info("Per-model results: %s", per_model_path)

    # Write summary
    summary = aggregate(model_results)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Summary: %s", summary_path)

    # Print key results
    rs = summary["relay_switch_locations"]
    bw = summary["boundary_width_comparison"]
    log.info("")
    log.info("=== RELAY SWITCH LOCATIONS (%d total) ===", summary["n_relay_switches_total"])
    log.info("  Fisher classification: %d in CAZ (%.1f%%), %d inter-CAZ (%.1f%%)",
             rs["fisher_in_caz"], rs["fisher_in_caz_pct"] or 0,
             rs["fisher_inter_caz"], rs["fisher_inter_caz_pct"] or 0)
    log.info("  CKA classification:   %d in CAZ (%.1f%%), %d coasting (%.1f%%)",
             rs["cka_in_caz"], rs["cka_in_caz_pct"] or 0,
             rs["cka_coasting"], rs["cka_coasting_pct"] or 0)
    log.info("")
    log.info("=== BOUNDARY WIDTH COMPARISON ===")
    log.info("  Dips detected: %d / mean Fisher width: %.1f / mean CKA width: %.1f / ratio: %.3f",
             bw["n_dips_detected"],
             bw["mean_fisher_width"] or 0,
             bw["mean_cka_width"] or 0,
             bw["mean_width_ratio"] or 0)
    log.info("")
    log.info("Top switch directions:")
    for direction, count in summary["top_switch_directions"]:
        log.info("  %3d  %s", count, direction)


if __name__ == "__main__":
    main()

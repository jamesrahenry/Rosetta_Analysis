#!/usr/bin/env python3
"""
caz_ablation_control.py — Control analysis for CAZ ablation claims.

Tests whether features outside CAZ regions produce measurable concept suppression
when ablated, addressing the question: is the causal impact specific to CAZ-
located features, or do arbitrary features cause similar suppression?

Two classification methods:
  1. ALIGNMENT: features with high concept_alignment score (aligned) vs. low
     (dark matter w.r.t. that concept). No CAZ data needed.
  2. CAZ OVERLAP: features whose layer range overlaps a CAZ for that concept
     (in-CAZ) vs. features with no layer overlap with any CAZ (non-CAZ).

For each classification, compares distributions of retained_pct (fraction of
concept separation remaining after ablation). Lower = more suppression = more
causal impact.

Usage:
    python src/caz_ablation_control.py
    python src/caz_ablation_control.py --alignment-threshold 0.2
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR  = Path(__file__).resolve().parents[1] / "results"
FEATURE_LIB  = Path(__file__).resolve().parents[2] / "Rosetta_Feature_Library"

CONCEPTS = [
    "credibility", "certainty", "sentiment", "moral_valence",
    "causation", "temporal_order", "negation",
]

# Default threshold: concept_alignment >= this → "aligned" feature for that concept
DEFAULT_ALIGNMENT_THRESHOLD = 0.15


# ── data loading ──────────────────────────────────────────────────────────────

def load_ablation_data() -> list[dict]:
    """Load all ablation results across all models."""
    records = []
    for abl_dir in sorted(RESULTS_DIR.glob("dark_ablation_*")):
        abl_file = abl_dir / "dark_matter_ablation.json"
        if not abl_file.exists():
            continue
        data = json.loads(abl_file.read_text())
        model_id = data["model_id"]
        for result in data.get("results", []):
            result["model_id"] = model_id
            records.append(result)
    log.info("Loaded %d ablated features across %d models",
             len(records),
             len({r["model_id"] for r in records}))
    return records


def load_caz_regions(model_slug: str) -> dict[str, list[dict]]:
    """
    Load CAZ regions for a model from Rosetta_Feature_Library.
    Returns {concept: [region_dict, ...]} where each region has
    start_layer, peak_layer, end_layer, caz_type, caz_score.
    """
    regions: dict[str, list[dict]] = {}
    for concept in CONCEPTS:
        path = FEATURE_LIB / "cazs" / concept / f"{model_slug}.json"
        if path.exists():
            data = json.loads(path.read_text())
            regions[concept] = data.get("regions", [])
    return regions


def model_id_to_slug(model_id: str) -> str:
    """Convert model_id to the slug used in feature library filenames."""
    # e.g. EleutherAI/pythia-1.4b → pythia-1.4b
    return model_id.split("/")[-1]


def overlaps_any_caz(
    birth_layer: int,
    death_layer: int,
    caz_regions: list[dict],
) -> tuple[bool, str | None, float | None]:
    """
    Check if a feature's layer range overlaps any CAZ region.
    Returns (overlaps, caz_type, caz_score) for the highest-scoring overlapping CAZ.
    """
    best_type  = None
    best_score = None
    for r in caz_regions:
        # Feature overlaps CAZ if [birth, death] intersects [start, end]
        if birth_layer <= r["end_layer"] and death_layer >= r["start_layer"]:
            if best_score is None or r["caz_score"] > best_score:
                best_score = r["caz_score"]
                best_type  = r["caz_type"]
    return (best_type is not None), best_type, best_score


# ── analysis ──────────────────────────────────────────────────────────────────

def run_alignment_analysis(
    records: list[dict],
    threshold: float,
) -> dict:
    """
    Method 1: classify by concept_alignment score.
    For each (feature, concept) pair: aligned if alignment >= threshold, else dark matter.
    Compare retained_pct distributions.
    """
    aligned_retained   = []   # retained_pct for aligned features
    dark_retained      = []   # retained_pct for dark matter features

    by_concept: dict[str, dict[str, list]] = {
        c: {"aligned": [], "dark": []} for c in CONCEPTS
    }

    for rec in records:
        concept_alignment = rec.get("concept_alignment", {})
        concept_impact    = rec.get("concept_impact", {})

        for concept in CONCEPTS:
            alignment = concept_alignment.get(concept, 0.0)
            impact    = concept_impact.get(concept, {})
            retained  = impact.get("retained_pct")
            if retained is None:
                continue

            if alignment >= threshold:
                aligned_retained.append(retained)
                by_concept[concept]["aligned"].append(retained)
            else:
                dark_retained.append(retained)
                by_concept[concept]["dark"].append(retained)

    def stats(vals):
        if not vals:
            return {"n": 0, "mean": None, "median": None, "std": None}
        arr = np.array(vals)
        return {
            "n":      len(arr),
            "mean":   float(arr.mean()),
            "median": float(np.median(arr)),
            "std":    float(arr.std()),
            "pct_below_90": float((arr < 90).mean() * 100),  # % with >10% suppression
        }

    result = {
        "method": "alignment_threshold",
        "threshold": threshold,
        "overall": {
            "aligned":   stats(aligned_retained),
            "dark_matter": stats(dark_retained),
        },
        "by_concept": {
            c: {
                "aligned":    stats(by_concept[c]["aligned"]),
                "dark_matter": stats(by_concept[c]["dark"]),
            }
            for c in CONCEPTS
        },
    }
    return result


def run_caz_overlap_analysis(records: list[dict]) -> dict:
    """
    Method 2: classify by whether the feature's layer range overlaps a CAZ
    for each concept.
    in_caz = overlaps, further split by caz_type (gentle / standard / black_hole).
    non_caz = no overlap with any CAZ for that concept.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    # Groups: "non_caz", "gentle", "standard", "black_hole", "embedding"

    per_concept: dict[str, dict[str, list]] = {
        c: defaultdict(list) for c in CONCEPTS
    }

    # Cache CAZ regions per model
    caz_cache: dict[str, dict] = {}

    for rec in records:
        model_id = rec["model_id"]
        slug     = model_id_to_slug(model_id)

        if slug not in caz_cache:
            caz_cache[slug] = load_caz_regions(slug)
        caz_regions_by_concept = caz_cache[slug]

        birth = rec["birth_layer"]
        death = rec["death_layer"]
        concept_impact = rec.get("concept_impact", {})

        for concept in CONCEPTS:
            impact   = concept_impact.get(concept, {})
            retained = impact.get("retained_pct")
            if retained is None:
                continue

            caz_for_concept = caz_regions_by_concept.get(concept, [])
            overlaps, caz_type, _ = overlaps_any_caz(birth, death, caz_for_concept)

            group = caz_type if overlaps else "non_caz"
            groups[group].append(retained)
            per_concept[concept][group].append(retained)

    def stats(vals):
        if not vals:
            return {"n": 0, "mean": None, "median": None, "std": None,
                    "pct_below_90": None}
        arr = np.array(vals)
        return {
            "n":            len(arr),
            "mean":         float(arr.mean()),
            "median":       float(np.median(arr)),
            "std":          float(arr.std()),
            "pct_below_90": float((arr < 90).mean() * 100),
        }

    result = {
        "method": "caz_layer_overlap",
        "overall": {g: stats(v) for g, v in sorted(groups.items())},
        "by_concept": {
            c: {g: stats(v) for g, v in sorted(per_concept[c].items())}
            for c in CONCEPTS
        },
    }
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Control analysis: do non-CAZ features affect concept separation?"
    )
    parser.add_argument(
        "--alignment-threshold", type=float, default=DEFAULT_ALIGNMENT_THRESHOLD,
        help=f"concept_alignment threshold for 'aligned' classification "
             f"(default: {DEFAULT_ALIGNMENT_THRESHOLD})"
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parents[1] / "CAZ_ABLATION_CONTROL.json"
    )
    args = parser.parse_args()

    records = load_ablation_data()

    log.info("Running alignment analysis (threshold=%.2f)...", args.alignment_threshold)
    alignment_result = run_alignment_analysis(records, args.alignment_threshold)

    log.info("Running CAZ overlap analysis...")
    caz_result = run_caz_overlap_analysis(records)

    output = {
        "n_features":   len(records),
        "n_models":     len({r["model_id"] for r in records}),
        "alignment":    alignment_result,
        "caz_overlap":  caz_result,
    }

    args.out.write_text(json.dumps(output, indent=2))
    log.info("Results written to %s", args.out)

    # ── human-readable summary ────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("ALIGNMENT ANALYSIS (threshold=%.2f)", args.alignment_threshold)
    log.info("=" * 60)
    ov = alignment_result["overall"]
    al = ov["aligned"]
    dk = ov["dark_matter"]
    log.info("  ALIGNED features  (n=%d): mean retained=%.1f%%  >10%% suppression: %.1f%%",
             al["n"], al["mean"] or 0, al["pct_below_90"] or 0)
    log.info("  DARK MATTER feat  (n=%d): mean retained=%.1f%%  >10%% suppression: %.1f%%",
             dk["n"], dk["mean"] or 0, dk["pct_below_90"] or 0)
    delta = (al["mean"] or 0) - (dk["mean"] or 0)
    log.info("  Δ mean retained (aligned − dark): %.1f pp", delta)
    log.info("  Interpretation: %s",
             "aligned features cause MORE suppression (as expected)" if delta < 0
             else "dark matter features cause COMPARABLE or MORE suppression (unexpected)")

    log.info("")
    log.info("=" * 60)
    log.info("CAZ OVERLAP ANALYSIS")
    log.info("=" * 60)
    ov2 = caz_result["overall"]
    for group in ["non_caz", "embedding", "gentle", "standard", "black_hole"]:
        s = ov2.get(group)
        if s and s["n"] > 0:
            log.info("  %-12s (n=%4d): mean retained=%.1f%%  >10%% suppression: %.1f%%",
                     group, s["n"], s["mean"], s["pct_below_90"])

    non_caz_mean = (ov2.get("non_caz") or {}).get("mean") or 0
    gentle_mean  = (ov2.get("gentle")  or {}).get("mean") or 0
    bh_mean      = (ov2.get("black_hole") or {}).get("mean") or 0
    log.info("")
    log.info("  Δ retained: non_caz − gentle    = %.1f pp", non_caz_mean - gentle_mean)
    log.info("  Δ retained: non_caz − black_hole = %.1f pp", non_caz_mean - bh_mean)
    log.info("  Interpretation: %s",
             "non-CAZ features cause LESS suppression (CAZ claim supported)"
             if non_caz_mean > gentle_mean
             else "non-CAZ features cause COMPARABLE suppression (CAZ claim needs revision)")

    log.info("")
    log.info("By concept (CAZ overlap — mean retained %%):")
    for concept in CONCEPTS:
        c_data = caz_result["by_concept"][concept]
        nc = c_data.get("non_caz", {})
        ge = c_data.get("gentle", {})
        st = c_data.get("standard", {})
        bh = c_data.get("black_hole", {})
        log.info("  %-16s  non_caz=%s  gentle=%s  standard=%s  black_hole=%s",
                 concept,
                 f"{nc['mean']:.1f}%(n={nc['n']})" if nc.get("mean") else "—",
                 f"{ge['mean']:.1f}%(n={ge['n']})" if ge.get("mean") else "—",
                 f"{st['mean']:.1f}%(n={st['n']})" if st.get("mean") else "—",
                 f"{bh['mean']:.1f}%(n={bh['n']})" if bh.get("mean") else "—",
                 )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
build_library.py — Build the feature_library/ directory from run results.

Aggregates deepdive, label, extraction, and ablation data into a single
structured feature library with universal feature discovery and an atlas index.

Prerequisites (run in order):
    python src/analyze_scored.py --csv      # regenerates SCORED_CAZ_ANALYSIS.csv
    python src/label_features.py --all      # adds feature_labels.json to deepdive dirs
    python src/build_library.py             # builds feature_library/

Output structure:
    feature_library/
        models/{slug}/features.json         # per-model feature summary
        cazs/{concept}/{slug}.json          # CAZ regions per concept per model
        universal/{UF_ID}/provenance.json   # contributing models per UF
        atlas.json                          # master index

Usage:
    python src/build_library.py
    python src/build_library.py --out /path/to/feature_library
    python src/build_library.py --min-families 2  # UF threshold (default: 2)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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
DEFAULT_OUT  = Path(__file__).resolve().parents[2] / "Rosetta_Feature_Library"
SCORED_CSV   = Path(__file__).resolve().parents[1] / "SCORED_CAZ_ANALYSIS.csv"
CONCEPTS     = ["credibility", "negation", "sentiment", "causation",
                "certainty", "moral_valence", "temporal_order"]
N_DEPTH_BINS = 20
LABEL_COS_THRESHOLD = 0.5   # from label_features.py
UF_SIM_THRESHOLD    = 0.35  # Jaccard similarity for UF matching
CAZ_TYPE_THRESHOLDS = {
    "embedding": (0.0, 0.15),   # first 15% of depth
    "black_hole": (0.5, 1.0),   # caz_score >= 0.5
    "gentle":     (0.0, 0.05),  # caz_score < 0.05
}

FAMILY_MAP = {
    "pythia": ["EleutherAI/pythia"],
    "gpt2":   ["openai-community/gpt2"],
    "opt":    ["facebook/opt"],
    "qwen2":  ["Qwen/Qwen2.5"],
    "gemma2": ["google/gemma-2"],
    "llama3": ["meta-llama/Llama-3.2"],
    "mistral":["mistralai/Mistral"],
    "phi":    ["microsoft/phi"],
}

# ── helpers ──────────────────────────────────────────────────────────────────

def model_family(model_id: str) -> str:
    for fam, prefixes in FAMILY_MAP.items():
        if any(model_id.startswith(p) for p in prefixes):
            return fam
    return "unknown"

def model_slug(model_id: str) -> str:
    return model_id.split("/")[-1]

def find_latest_dir(pattern: str) -> Path | None:
    dirs = sorted(RESULTS_DIR.glob(pattern))
    return dirs[-1] if dirs else None

def find_deepdive_dir(model_id: str) -> Path | None:
    slug = model_id.replace("/", "_").replace("-", "_")
    return find_latest_dir(f"deepdive_{slug}_*")

def find_ablation_dir(model_id: str) -> Path | None:
    slug = model_id.replace("/", "_").replace("-", "_").replace(".", "_")
    candidates = list(RESULTS_DIR.glob(f"dark_ablation_{slug}"))
    if not candidates:
        # try partial slug match
        short = model_slug(model_id).replace("-", "_").replace(".", "_")
        candidates = list(RESULTS_DIR.glob(f"dark_ablation_*{short}*"))
    return candidates[0] if candidates else None

def compute_handoff_sequence(feature_labels: list[dict], n_layers: int) -> list[str | None]:
    """Map per-layer concept labels to N_DEPTH_BINS depth bins (majority vote)."""
    bins: list[list[str]] = [[] for _ in range(N_DEPTH_BINS)]
    for entry in feature_labels:
        layer = entry["layer"]
        concept = entry.get("best_concept")
        cos = entry.get("best_cos", 0.0)
        if concept and cos >= LABEL_COS_THRESHOLD:
            bin_idx = min(int(layer / n_layers * N_DEPTH_BINS), N_DEPTH_BINS - 1)
            bins[bin_idx].append(concept)
    result = []
    for b in bins:
        if b:
            # majority vote
            counts = defaultdict(int)
            for c in b:
                counts[c] += 1
            result.append(max(counts, key=counts.get))
        else:
            result.append(None)
    return result

def handoff_label_from_sequence(seq: list[str | None]) -> str:
    """Compact description: unique concepts in order, e.g. 'certainty -> credibility'."""
    seen = []
    for c in seq:
        if c and (not seen or seen[-1] != c):
            seen.append(c)
    return " -> ".join(seen) if seen else "unlabeled"

def jaccard_similarity(seq_a: list, seq_b: list) -> float:
    """Jaccard on non-None positions."""
    a_set = {i for i, v in enumerate(seq_a) if v is not None}
    b_set = {i for i, v in enumerate(seq_b) if v is not None}
    # Also require concept agreement where both are non-None
    both = a_set & b_set
    agree = sum(1 for i in both if seq_a[i] == seq_b[i])
    union = len(a_set | b_set)
    return agree / union if union > 0 else 0.0

def caz_type(peak_depth_pct: float, caz_score: float) -> str:
    if peak_depth_pct <= 15.0:
        return "embedding"
    if caz_score >= 0.5:
        return "black_hole"
    if caz_score < 0.05:
        return "gentle"
    return "standard"

# ── data loading ─────────────────────────────────────────────────────────────

def load_all_models() -> dict[str, dict]:
    """Load deepdive + labels + ablation for every model that has a deepdive result."""
    models = {}

    for dd_dir in sorted(RESULTS_DIR.glob("deepdive_*")):
        fm_path = dd_dir / "feature_map.json"
        if not fm_path.exists():
            continue
        fm = json.loads(fm_path.read_text())
        mid = fm["model_id"]
        # keep only the most recent run per model
        if mid in models and dd_dir.name < models[mid]["deepdive_dir"].name:
            continue

        # feature labels
        labels_path = dd_dir / "feature_labels.json"
        labels_by_fid = {}
        if labels_path.exists():
            lab = json.loads(labels_path.read_text())
            labels_by_fid = lab.get("features", {})
        else:
            log.warning("%s: no feature_labels.json — run label_features.py --all first", mid)

        # ablation
        abl_dir = find_ablation_dir(mid)
        ablation_by_fid = {}
        if abl_dir:
            abl_path = abl_dir / "dark_matter_ablation.json"
            if abl_path.exists():
                abl_data = json.loads(abl_path.read_text())
                for r in abl_data.get("results", []):
                    ablation_by_fid[r["feature_id"]] = r

        models[mid] = {
            "model_id":     mid,
            "n_layers":     fm["n_layers"],
            "hidden_dim":   fm["hidden_dim"],
            "deepdive_dir": dd_dir,
            "features":     fm["features"],
            "labels":       labels_by_fid,
            "ablation":     ablation_by_fid,
            "family":       model_family(mid),
        }

    log.info("Loaded %d models", len(models))
    return models

def load_scored_csv() -> dict[tuple, list]:
    """Load SCORED_CAZ_ANALYSIS.csv → {(model_id, concept): [region_dicts]}."""
    if not SCORED_CSV.exists():
        log.error("SCORED_CAZ_ANALYSIS.csv not found — run: python src/analyze_scored.py --csv")
        return {}
    import csv
    regions: dict[tuple, list] = defaultdict(list)
    with SCORED_CSV.open() as f:
        for row in csv.DictReader(f):
            key = (row["model_id"], row["concept"])
            regions[key].append({
                "start_layer":     int(row["start"]),
                "peak_layer":      int(row["peak"]),
                "end_layer":       int(row["end"]),
                "n_layers_total":  int(row["n_layers"]),
                "peak_depth_pct":  float(row["depth_pct"]),
                "start_depth_pct": float(row["start"]) / int(row["n_layers"]) * 100,
                "end_depth_pct":   float(row["end"])   / int(row["n_layers"]) * 100,
                "width_depth_pct": float(row["width_pct"]),
                "caz_score":       float(row["caz_score"]),
                "peak_separation": float(row["peak_separation"]),
                "caz_type":        caz_type(float(row["depth_pct"]), float(row["caz_score"])),
                "overlapping_ufs": [],         # filled in after UF assignment
                "overlapping_feature_ids": [], # filled in after UF assignment
            })
    log.info("Loaded CAZ regions for %d (model, concept) pairs", len(regions))
    return regions

# ── feature enrichment ───────────────────────────────────────────────────────

def enrich_features(model: dict) -> list[dict]:
    """Merge deepdive feature data with labels and ablation."""
    n_layers = model["n_layers"]
    enriched = []

    for raw in model["features"]:
        fid = raw["feature_id"]
        lab_layers = model["labels"].get(str(fid), [])
        abl = model["ablation"].get(fid, {})

        handoff_seq = compute_handoff_sequence(lab_layers, n_layers)
        label = handoff_label_from_sequence(handoff_seq)

        # ablation impact: retained_pct per concept
        ablation_impact = None
        ablation_verdict = None
        if abl:
            ablation_impact = {
                c: round(v["retained_pct"], 1)
                for c, v in abl.get("concept_impact", {}).items()
            }
            ablation_verdict = abl.get("verdict")

        enriched.append({
            "feature_id":     fid,
            "birth_layer":    raw["birth_layer"],
            "death_layer":    raw["death_layer"],
            "lifespan":       raw["lifespan"],
            "layer_indices":  raw["layer_indices"],
            "eigenvalues":    raw["eigenvalues"],
            "handoff_label":  label,
            "handoff_sequence": handoff_seq,
            "concept_alignment": raw.get("concept_alignment", {}),
            "ablation_impact":   ablation_impact,
            "ablation_verdict":  ablation_verdict,
            "uf_id":          None,  # assigned later
        })

    return enriched

# ── universal feature discovery ──────────────────────────────────────────────

def discover_universal_features(
    all_enriched: dict[str, list[dict]],
    all_models: dict[str, dict],
    min_families: int,
) -> tuple[dict[str, str], list[dict]]:
    """
    Cluster features across models by handoff_sequence similarity.
    Returns:
        uf_assignments: {(model_id, feature_id): uf_id}
        universal_features: list of UF dicts for atlas
    """
    # Build flat list of (model_id, feature_id, family, sequence)
    all_feats = []
    for mid, feats in all_enriched.items():
        fam = all_models[mid]["family"]
        for f in feats:
            seq = f["handoff_sequence"]
            # Only consider features with at least one labeled bin
            if any(v is not None for v in seq):
                all_feats.append((mid, f["feature_id"], fam, seq))

    log.info("Clustering %d labeled features for UF discovery", len(all_feats))

    # Single-linkage clustering by Jaccard similarity
    clusters: list[list[int]] = []  # list of indices into all_feats
    assigned = [False] * len(all_feats)

    for i in range(len(all_feats)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, len(all_feats)):
            if assigned[j]:
                continue
            # Don't cluster same-model features together
            if all_feats[i][0] == all_feats[j][0]:
                continue
            sim = jaccard_similarity(all_feats[i][3], all_feats[j][3])
            if sim >= UF_SIM_THRESHOLD:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    # Filter to clusters spanning >= min_families
    uf_counter = 0
    uf_assignments: dict[tuple, str] = {}
    universal_features = []

    for cluster in clusters:
        families = {all_feats[i][2] for i in cluster}
        if len(families) < min_families:
            continue

        uf_counter += 1
        uf_id = f"UF{uf_counter:03d}"

        # Compute consensus handoff template (majority vote per bin)
        seqs = [all_feats[i][3] for i in cluster]
        template = []
        for bin_idx in range(N_DEPTH_BINS):
            vals = [s[bin_idx] for s in seqs if s[bin_idx] is not None]
            if vals:
                counts = defaultdict(int)
                for v in vals:
                    counts[v] += 1
                template.append(max(counts, key=counts.get))
            else:
                template.append(None)

        description = handoff_label_from_sequence(template)

        # Assign UF ID to all cluster members
        models_in_uf = set()
        for i in cluster:
            mid, fid, fam, _ = all_feats[i]
            uf_assignments[(mid, fid)] = uf_id
            models_in_uf.add(mid)

        universal_features.append({
            "uf_id":            uf_id,
            "description":      description,
            "n_models":         len(models_in_uf),
            "n_families":       len(families),
            "handoff_template": template,
        })

    log.info(
        "Discovered %d universal features spanning >= %d families",
        len(universal_features), min_families,
    )
    return uf_assignments, universal_features

# ── CAZ overlap annotation ────────────────────────────────────────────────────

def annotate_caz_overlaps(
    caz_regions: dict[tuple, list],
    all_enriched: dict[str, list[dict]],
    all_models: dict[str, dict],
) -> None:
    """Fill overlapping_ufs and overlapping_feature_ids in CAZ regions in-place."""
    for (mid, concept), regions in caz_regions.items():
        if mid not in all_enriched:
            continue
        n_layers = all_models[mid]["n_layers"]
        feats = all_enriched[mid]
        for region in regions:
            start, end = region["start_layer"], region["end_layer"]
            overlapping_ufs = []
            overlapping_fids = []
            for f in feats:
                # Feature overlaps with region if their layer ranges intersect
                if f["birth_layer"] <= end and f["death_layer"] >= start:
                    overlapping_fids.append(f["feature_id"])
                    if f["uf_id"]:
                        overlapping_ufs.append(f["uf_id"])
            region["overlapping_ufs"]        = sorted(set(overlapping_ufs))
            region["overlapping_feature_ids"] = sorted(overlapping_fids)

# ── writers ──────────────────────────────────────────────────────────────────

def write_model_features(out_dir: Path, mid: str, model: dict, enriched: list[dict]) -> None:
    slug_dir = out_dir / "models" / model_slug(mid)
    slug_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "model_id":    mid,
        "n_layers":    model["n_layers"],
        "hidden_dim":  model["hidden_dim"],
        "deepdive_dir": str(model["deepdive_dir"].relative_to(
            Path(__file__).resolve().parents[1])),
        "n_features":  len(enriched),
        "n_persistent": sum(1 for f in enriched if f["lifespan"] >= 5),
        "features":    enriched,
    }
    (slug_dir / "features.json").write_text(json.dumps(out, indent=2))

def write_caz_regions(
    out_dir: Path,
    caz_regions: dict[tuple, list],
    all_model_ids: set[str],
) -> None:
    for (mid, concept), regions in caz_regions.items():
        if mid not in all_model_ids:
            continue
        concept_dir = out_dir / "cazs" / concept
        concept_dir.mkdir(parents=True, exist_ok=True)
        slug = model_slug(mid)
        out = {
            "model_id":  mid,
            "concept":   concept,
            "n_regions": len(regions),
            "regions":   regions,
        }
        (concept_dir / f"{slug}.json").write_text(json.dumps(out, indent=2))

def write_universal_features(
    out_dir: Path,
    universal_features: list[dict],
    uf_assignments: dict[tuple, str],
    all_enriched: dict[str, list[dict]],
    all_models: dict[str, dict],
) -> None:
    # Build reverse map: uf_id → [(model_id, feature_id)]
    uf_members: dict[str, list] = defaultdict(list)
    for (mid, fid), uid in uf_assignments.items():
        uf_members[uid].append((mid, fid))

    for uf in universal_features:
        uf_id = uf["uf_id"]
        uf_dir = out_dir / "universal" / uf_id
        uf_dir.mkdir(parents=True, exist_ok=True)

        provenance = []
        for (mid, fid) in uf_members[uf_id]:
            feat = next((f for f in all_enriched[mid] if f["feature_id"] == fid), None)
            if not feat:
                continue
            provenance.append({
                "model_id":     mid,
                "feature_id":   fid,
                "handoff_label": feat["handoff_label"],
                "alignment_cos": 0.0,  # placeholder — Procrustes alignment not recomputed here
                "peak_eigenvalue": max(feat["eigenvalues"]) if feat["eigenvalues"] else 0.0,
                "lifespan":     feat["lifespan"],
            })

        (uf_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

def write_atlas(
    out_dir: Path,
    universal_features: list[dict],
    all_models: dict[str, dict],
) -> None:
    # Sort by n_families DESC, n_models DESC
    sorted_ufs = sorted(
        universal_features,
        key=lambda u: (-u["n_families"], -u["n_models"]),
    )
    # Re-number UF IDs in sorted order
    for idx, uf in enumerate(sorted_ufs, 1):
        old_id = uf["uf_id"]
        uf["uf_id"] = f"UF{idx:03d}"

    atlas = {
        "reference_model":       "openai-community/gpt2",
        "aligned_dim":           768,
        "n_depth_bins":          N_DEPTH_BINS,
        "n_universal_features":  len(sorted_ufs),
        "n_models":              len(all_models),
        "build_date":            __import__("datetime").date.today().isoformat(),
        "universal_features":    sorted_ufs,
    }
    (out_dir / "atlas.json").write_text(json.dumps(atlas, indent=2))
    log.info("Atlas written: %d UFs, %d models", len(sorted_ufs), len(all_models))

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build feature_library/ from run results")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output directory (default: feature_library/)")
    parser.add_argument("--min-families", type=int, default=2,
                        help="Minimum architectural families for a universal feature (default: 2)")
    parser.add_argument("--base-only", action="store_true",
                        help="Exclude instruct/IT models from UF discovery (include in cazs/models)")
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load all data ──────────────────────────────────────────────────────
    all_models = load_all_models()
    caz_regions = load_scored_csv()

    # ── 2. Enrich features ───────────────────────────────────────────────────
    log.info("Enriching features...")
    all_enriched: dict[str, list[dict]] = {}
    for mid, model in all_models.items():
        all_enriched[mid] = enrich_features(model)
    total_feats = sum(len(v) for v in all_enriched.values())
    log.info("  %d total features across %d models", total_feats, len(all_models))

    # ── 3. Universal feature discovery ───────────────────────────────────────
    if args.base_only:
        uf_models = {mid: m for mid, m in all_models.items()
                     if not any(x in mid.lower() for x in ["instruct", "-it"])}
        uf_enriched = {mid: all_enriched[mid] for mid in uf_models}
        log.info("UF discovery on %d base models only (--base-only)", len(uf_models))
    else:
        uf_models, uf_enriched = all_models, all_enriched

    uf_assignments, universal_features = discover_universal_features(
        uf_enriched, uf_models, args.min_families,
    )

    # Apply UF IDs back to enriched features
    for (mid, fid), uid in uf_assignments.items():
        for f in all_enriched[mid]:
            if f["feature_id"] == fid:
                f["uf_id"] = uid
                break

    # ── 4. Annotate CAZ overlaps ─────────────────────────────────────────────
    log.info("Annotating CAZ region overlaps...")
    annotate_caz_overlaps(caz_regions, all_enriched, all_models)

    # ── 5. Write model features ───────────────────────────────────────────────
    log.info("Writing model features...")
    for mid, model in all_models.items():
        write_model_features(out_dir, mid, model, all_enriched[mid])

    # ── 6. Write CAZ regions ──────────────────────────────────────────────────
    log.info("Writing CAZ regions...")
    write_caz_regions(out_dir, caz_regions, set(all_models.keys()))

    # ── 7. Write universal feature provenance ─────────────────────────────────
    log.info("Writing universal feature provenance...")
    write_universal_features(out_dir, universal_features, uf_assignments,
                             all_enriched, all_models)

    # ── 8. Write atlas ────────────────────────────────────────────────────────
    write_atlas(out_dir, universal_features, all_models)

    log.info("")
    log.info("=== Build complete ===")
    log.info("  Output: %s", out_dir)
    log.info("  Models: %d", len(all_models))
    log.info("  Total features: %d", total_feats)
    labeled = sum(
        1 for feats in all_enriched.values()
        for f in feats if f["handoff_label"] != "unlabeled"
    )
    log.info("  Labeled features: %d (%.0f%%)", labeled, 100 * labeled / max(total_feats, 1))
    log.info("  Universal features: %d (>= %d families)", len(universal_features), args.min_families)


if __name__ == "__main__":
    main()

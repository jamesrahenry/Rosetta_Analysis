#!/usr/bin/env python3
"""
saddle_vs_rotation.py — Compare CAZ boundary definitions:
  (1) Separation saddle points (current method, implied by velocity zero-crossings)
  (2) Adjacent-layer direction rotation (cosine between consecutive dominant vectors)

For each model, for each concept:
  - Compute adjacent-layer cosine profile from direction vectors
  - Load CAZ separation profile from scored CAZ JSON
  - Find saddle points (local minima in S(l) between two peaks)
  - Find rotation peaks (local minima in adjacent-layer cosine)
  - Measure coincidence: do they land at the same layer?

No GPU needed — all data already on disk.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import argrelmin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT   = Path(__file__).resolve().parents[2]
CAZ_ROOT    = Path(__file__).resolve().parents[1]
RESULTS_DIR = CAZ_ROOT / "results"
FEATURE_LIB = REPO_ROOT / "Rosetta_Feature_Library"

CONCEPTS = ["credibility", "certainty", "sentiment", "moral_valence",
            "causation", "temporal_order", "negation"]

# How close (in layers) do a saddle and a rotation peak need to be to count as coincident?
COINCIDENCE_WINDOW = 2

# Rotation threshold below which we call it a "sharp rotation event"
ROTATION_THRESHOLD = 0.7


def get_latest_deepdive(model_slug: str) -> Path | None:
    candidates = sorted(RESULTS_DIR.glob(f"deepdive_{model_slug}_2*"))
    return candidates[-1] if candidates else None


def load_direction_vectors(dd_dir: Path, n_layers: int) -> np.ndarray | None:
    """Load top eigenvector (first column) at each layer. Returns (n_layers, hidden_dim)."""
    vecs = []
    expected_dim = None
    for l in range(n_layers):
        path = dd_dir / f"directions_L{l:03d}.npy"
        if not path.exists():
            return None
        arr = np.load(path)  # (n_dirs, hidden_dim)
        vec = arr[0]  # top direction, shape (hidden_dim,)
        if expected_dim is None:
            expected_dim = vec.shape[0]
        if vec.shape[0] != expected_dim:
            log.warning("  %s L%03d: shape mismatch (%d vs %d), truncating at %d layers",
                        dd_dir.name, l, vec.shape[0], expected_dim, l)
            break
        vecs.append(vec)
    if len(vecs) < 4:
        return None
    return np.stack(vecs, axis=0)  # (n_layers, hidden_dim)


def adjacent_cosine(vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity between consecutive layer direction vectors. Length = n_layers - 1."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    normed = vecs / norms
    return np.einsum("ld,ld->l", normed[:-1], normed[1:])  # (n_layers-1,)


def find_caz_result_dir(slug: str) -> Path | None:
    """Find the latest per-model CAZ results directory (has caz_*.json files)."""
    SKIP = ("deepdive_", "manifold_", "dark_ablation_", "custom_")
    candidates = sorted(
        d for d in RESULTS_DIR.glob(f"*_{slug}_2*")
        if d.is_dir() and not any(d.name.startswith(p) for p in SKIP)
        and any(d.glob("caz_*.json"))
    )
    return candidates[-1] if candidates else None


def load_separation_profile(slug: str, concept: str) -> np.ndarray | None:
    """Load layer-wise Fisher separation from per-model caz_{concept}.json."""
    caz_dir = find_caz_result_dir(slug)
    if caz_dir is None:
        return None
    path = caz_dir / f"caz_{concept}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    metrics = data.get("layer_data", {}).get("metrics")
    if not metrics:
        return None
    sep = np.array([m["separation_fisher"] for m in metrics], dtype=np.float32)
    return sep


def find_saddle_points(sep: np.ndarray, order: int = 2) -> list[int]:
    """Local minima in separation profile (between peaks)."""
    minima = argrelmin(sep, order=order)[0].tolist()
    return minima


def find_rotation_peaks(cos_sim: np.ndarray, threshold: float = ROTATION_THRESHOLD,
                        order: int = 2) -> list[int]:
    """Local minima in |adjacent-layer cosine| (= direction rotation or flip events).
    Uses |cos| so that both rotations (cos≈0) and sign flips (cos≈-1) are detected.
    Threshold is on |cos|: events with |cos| < threshold are sharp rotations.
    """
    abs_cos = np.abs(cos_sim)
    minima = argrelmin(abs_cos, order=order)[0].tolist()
    # Also include any point that is globally below threshold even if not a local min
    sharp = set(minima) | set(int(i) for i in np.where(abs_cos < threshold)[0])
    # Filter: must be local minimum of |cos| OR globally below threshold
    return sorted(i for i in sharp if abs_cos[i] < threshold)


def coincidence(saddles: list[int], rotations: list[int], window: int) -> dict:
    """
    For each saddle point, check if a rotation event is within ±window layers.
    For each rotation event, check if a saddle is within ±window layers.
    """
    saddle_hits = sum(
        1 for s in saddles if any(abs(s - r) <= window for r in rotations)
    )
    rotation_hits = sum(
        1 for r in rotations if any(abs(r - s) <= window for s in saddles)
    )
    return {
        "n_saddles": len(saddles),
        "n_rotations": len(rotations),
        "saddles_with_rotation_nearby": saddle_hits,
        "rotations_with_saddle_nearby": rotation_hits,
        "saddle_coverage": saddle_hits / len(saddles) if saddles else None,
        "rotation_coverage": rotation_hits / len(rotations) if rotations else None,
    }


def model_slug_to_id(slug: str) -> str:
    """Convert deepdive directory slug back to feature library model ID."""
    # The feature lib uses the HF model name's last component
    replacements = {
        "EleutherAI_pythia_": "pythia-",
        "facebook_opt_": "opt-",
        "openai_community_gpt2": "gpt2",
        "google_gemma_2_": "gemma-2-",
        "meta_llama_Llama_3.2_": "Llama-3.2-",
        "mistralai_Mistral_7B_v0.3": "Mistral-7B-v0.3",
        "mistralai_Mistral_7B_Instruct_v0.3": "Mistral-7B-Instruct-v0.3",
        "microsoft_phi_2": "phi-2",
        "Qwen_Qwen2.5_": "Qwen2.5-",
    }
    result = slug
    for k, v in replacements.items():
        result = result.replace(k, v)
    # Clean up trailing underscores and fix common patterns
    result = result.replace("_", "-").replace("--", "-").rstrip("-")
    return result


def run():
    # Find all unique base model slugs (skip instruct)
    all_slugs = set()
    for d in RESULTS_DIR.glob("deepdive_*_2026*"):
        slug = "_".join(d.name.split("_")[1:-2])  # strip 'deepdive_' and timestamp
        if "Instruct" not in slug and "it" not in slug.split("_")[-1]:
            all_slugs.add(slug)

    log.info("Found %d base model deep dives", len(all_slugs))

    all_results = []
    concept_summary: dict[str, list] = defaultdict(list)
    model_summary: dict[str, list] = defaultdict(list)

    for slug in sorted(all_slugs):
        dd_dir = get_latest_deepdive(slug)
        if dd_dir is None:
            continue

        # Count layers from available direction files
        n_layers = len(list(dd_dir.glob("directions_L*.npy")))
        if n_layers < 4:
            log.warning("  %s: only %d layers, skipping", slug, n_layers)
            continue

        # Load direction vectors
        vecs = load_direction_vectors(dd_dir, n_layers)
        if vecs is None:
            log.warning("  %s: missing direction files", slug)
            continue

        cos_sim = adjacent_cosine(vecs)  # (n_layers-1,)
        model_id = model_slug_to_id(slug)

        log.info("%s (%d layers, model_id=%s)", slug, n_layers, model_id)

        for concept in CONCEPTS:
            sep = load_separation_profile(slug, concept)
            if sep is None or len(sep) < 4:
                continue

            # Align sep to cos_sim length (n_layers-1).
            # Direction files include the embedding layer (L000), so sep typically
            # has n_layers-1 entries (transformer layers only).
            n_cos = len(cos_sim)  # = n_layers - 1
            if len(sep) == n_cos + 1:
                sep = sep[1:]  # drop first entry if sep includes embedding layer
            if len(sep) != n_cos:
                log.debug("  %s/%s: sep length %d != cos length %d, skipping",
                          slug, concept, len(sep), n_cos)
                continue

            saddles   = find_saddle_points(sep)
            rotations = find_rotation_peaks(cos_sim)
            stats     = coincidence(saddles, rotations, COINCIDENCE_WINDOW)

            result = {
                "model": slug,
                "model_id": model_id,
                "concept": concept,
                "n_layers": n_layers,
                "saddle_layers": saddles,
                "rotation_layers": rotations,
                "mean_cos_at_saddles": float(np.mean([cos_sim[min(s, len(cos_sim)-1)] for s in saddles])) if saddles else None,
                "mean_cos_at_rotations": float(np.mean([cos_sim[r] for r in rotations])) if rotations else None,
                "global_min_cos": float(cos_sim.min()),
                "global_mean_cos": float(cos_sim.mean()),
                **stats,
            }
            all_results.append(result)

            if stats["n_saddles"] > 0:
                concept_summary[concept].append(stats["saddle_coverage"])
                model_summary[slug].append(stats["saddle_coverage"])

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SADDLE POINT ↔ DIRECTION ROTATION COINCIDENCE")
    print(f"Window: ±{COINCIDENCE_WINDOW} layers | Rotation threshold: cos < {ROTATION_THRESHOLD}")
    print("="*70)

    total_saddles   = sum(r["n_saddles"]   for r in all_results)
    total_rotations = sum(r["n_rotations"] for r in all_results)
    covered_saddles = sum(r["saddles_with_rotation_nearby"] for r in all_results)
    covered_rots    = sum(r["rotations_with_saddle_nearby"] for r in all_results)

    print(f"\nOverall:")
    print(f"  Total saddle points:       {total_saddles}")
    print(f"  Total rotation events:     {total_rotations}")
    print(f"  Saddles with rotation nearby: {covered_saddles}/{total_saddles} "
          f"({100*covered_saddles/total_saddles:.1f}%)" if total_saddles else "  (no saddles)")
    print(f"  Rotations with saddle nearby: {covered_rots}/{total_rotations} "
          f"({100*covered_rots/total_rotations:.1f}%)" if total_rotations else "  (no rotations)")

    print("\nBy concept (% of saddles with nearby rotation):")
    for concept in CONCEPTS:
        vals = [v for v in concept_summary[concept] if v is not None]
        if vals:
            print(f"  {concept:16s}  mean={np.mean(vals):.2f}  n={len(vals)}")

    # Find divergence cases — saddles without nearby rotations
    divergent = [r for r in all_results
                 if r["n_saddles"] > 0 and r["saddle_coverage"] is not None
                 and r["saddle_coverage"] < 0.5]
    print(f"\nDivergent cases (saddle_coverage < 50%): {len(divergent)}")
    for r in sorted(divergent, key=lambda x: x["saddle_coverage"] or 1)[:10]:
        print(f"  {r['model']:40s} {r['concept']:16s} "
              f"saddles={r['saddle_layers']} rotations={r['rotation_layers']} "
              f"coverage={r['saddle_coverage']:.2f}")

    # Rotation events with no nearby saddle
    orphan_rotations = [r for r in all_results
                        if r["n_rotations"] > 0 and r["rotation_coverage"] is not None
                        and r["rotation_coverage"] < 0.5]
    print(f"\nOrphan rotation events (rotation_coverage < 50%): {len(orphan_rotations)}")
    for r in sorted(orphan_rotations, key=lambda x: x["rotation_coverage"] or 1)[:10]:
        print(f"  {r['model']:40s} {r['concept']:16s} "
              f"saddles={r['saddle_layers']} rotations={r['rotation_layers']} "
              f"coverage={r['rotation_coverage']:.2f}")

    # opt-6.7b spotlight (mentioned in paper)
    print("\nopt-6.7b spotlight:")
    opt_results = [r for r in all_results if "opt_6.7b" in r["model"] or "opt-6.7b" in r["model_id"]]
    for r in opt_results:
        print(f"  {r['concept']:16s}  saddles={r['saddle_layers']}  "
              f"rotations={r['rotation_layers']}  "
              f"min_cos={r['global_min_cos']:.3f}  "
              f"cos@saddles={r['mean_cos_at_saddles']}")

    # Save full results
    out_path = RESULTS_DIR / "saddle_vs_rotation.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    log.info("\nFull results written to %s", out_path)


if __name__ == "__main__":
    run()

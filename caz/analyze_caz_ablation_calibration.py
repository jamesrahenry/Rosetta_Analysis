#!/usr/bin/env python3
"""
analyze_caz_ablation_calibration.py — CAZ scoring recalibration via ablation impact.

Tests whether CAZ prominence score predicts ablation impact at peak, or whether
it's an architectural artifact (MHA resonance amplification vs genuine semantic weight).

Key hypothesis: Black hole CAZes (high prominence) and gentle CAZes (low prominence)
should have similar ablation impact if prominence is purely architectural. Gemma-2's
structurally-gentle CAZes are the critical test case.

Reads existing ablation_<concept>.json files from caz_scaling/results/ and
CAZ prominence data from semantic_convergence/results/.

Usage:
    python src/analyze_caz_ablation_calibration.py
    python src/analyze_caz_ablation_calibration.py --min-models 3
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from rosetta_tools.models import encoding_strategy_of
except ImportError:
    def encoding_strategy_of(mid): return "unknown"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CAZ_RESULTS = Path(__file__).parent.parent / "results"

CONCEPTS = ["credibility", "certainty", "sentiment", "causation",
            "temporal_order", "moral_valence", "negation"]

# CAZ prominence thresholds (from framework paper)
BLACK_HOLE_THRESHOLD = 0.15   # separation_fisher at peak
GENTLE_THRESHOLD     = 0.05


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ablation_data(results_root: Path) -> dict:
    """Load ablation_<concept>.json for each model × concept.

    Returns: {model_id: {concept: {caz_peak, peak_separation_reduction,
                                    peak_kl, suppression_damage_ratio, ...}}}
    """
    data = defaultdict(dict)
    seen = set()

    for d in sorted(results_root.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        for abl_file in sorted(d.glob("ablation_*.json")):
            try:
                abl = json.load(open(abl_file))
            except Exception:
                continue
            mid = abl.get("model_id", "")
            concept = abl.get("concept", "")
            if not mid or not concept or (mid, concept) in seen:
                continue
            seen.add((mid, concept))

            peak = abl.get("caz_peak")
            if peak is None:
                continue

            # Find ablation impact AT the CAZ peak layer
            layers = {r["layer"]: r for r in abl.get("layers", [])}
            if peak not in layers:
                # Use closest layer
                if not layers:
                    continue
                peak = min(layers.keys(), key=lambda l: abs(l - peak))

            r = layers[peak]
            data[mid][concept] = {
                "caz_peak": peak,
                "caz_start": abl.get("caz_start"),
                "caz_end": abl.get("caz_end"),
                "n_layers": abl.get("n_layers"),
                "peak_separation_reduction": r.get("separation_reduction", float("nan")),
                "peak_kl": r.get("kl_divergence", float("nan")),
                "suppression_damage_ratio": r.get("suppression_damage_ratio", float("nan")),
                "baseline_separation": r.get("baseline_separation", float("nan")),
            }

    return dict(data)


def load_caz_prominence(results_root: Path) -> dict:
    """Load CAZ prominence scores (separation_fisher at peak) from caz_scaling results.

    caz_<concept>.json files live in the same result dirs as ablation_<concept>.json.

    Returns: {model_id: {concept: {peak_layer, peak_score, score_category}}}
    """
    data = defaultdict(dict)
    seen = set()

    for d in sorted(results_root.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        for caz_file in sorted(d.glob("caz_*.json")):
            if "ablation" in caz_file.name or "multimodal" in caz_file.name:
                continue
            try:
                caz = json.load(open(caz_file))
            except Exception:
                continue
            mid = caz.get("model_id", "")
            concept = caz.get("concept", "")
            if not mid or not concept or (mid, concept) in seen:
                continue
            seen.add((mid, concept))

            ld = caz.get("layer_data", {})
            peak_layer = ld.get("peak_layer")
            metrics = {m["layer"]: m for m in ld.get("metrics", [])}

            if peak_layer is None or peak_layer not in metrics:
                continue

            peak_score = metrics[peak_layer].get("separation_fisher", float("nan"))
            if np.isnan(peak_score):
                peak_score = metrics[peak_layer].get("separation", float("nan"))

            if peak_score >= BLACK_HOLE_THRESHOLD:
                category = "black_hole"
            elif peak_score >= GENTLE_THRESHOLD:
                category = "gentle"
            else:
                category = "subtle"

            data[mid][concept] = {
                "peak_layer": peak_layer,
                "peak_score": peak_score,
                "category": category,
            }

    return dict(data)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CAZ ablation calibration analysis")
    parser.add_argument("--min-models", type=int, default=2,
                        help="Min models required per concept for comparison (default: 2)")
    args = parser.parse_args()

    log.info("Loading ablation data from %s", CAZ_RESULTS)
    ablation = load_ablation_data(CAZ_RESULTS)
    log.info("  %d models with ablation data", len(ablation))

    log.info("Loading CAZ prominence scores from %s", CAZ_RESULTS)
    prominence = load_caz_prominence(CAZ_RESULTS)
    log.info("  %d models with CAZ prominence data", len(prominence))

    # Join: models with both ablation and prominence data
    joint_models = sorted(set(ablation) & set(prominence))
    log.info("  %d models with both", len(joint_models))

    # Collect (prominence_score, separation_reduction, model_id, concept, category)
    rows = []
    for mid in joint_models:
        strat = encoding_strategy_of(mid)
        for concept in CONCEPTS:
            if concept not in ablation.get(mid, {}) or concept not in prominence.get(mid, {}):
                continue
            abl = ablation[mid][concept]
            prom = prominence[mid][concept]
            sep_reduction = abl["peak_separation_reduction"]
            score = prom["peak_score"]
            if np.isnan(sep_reduction) or np.isnan(score):
                continue
            rows.append({
                "model_id": mid,
                "model_slug": mid.split("/")[-1],
                "concept": concept,
                "encoding_strategy": strat,
                "prominence_score": score,
                "caz_category": prom["category"],
                "separation_reduction": sep_reduction,
                "suppression_damage_ratio": abl["suppression_damage_ratio"],
                "peak_kl": abl["peak_kl"],
                "baseline_separation": abl["baseline_separation"],
            })

    if not rows:
        log.error("No data after joining. Check paths.")
        return

    print(f"\n{'='*70}")
    print(f"CAZ ABLATION CALIBRATION ANALYSIS")
    print(f"{'='*70}")
    print(f"  {len(rows)} measurements from {len(joint_models)} models\n")

    # ── 1. Overall correlation: prominence score vs ablation impact ──
    scores  = np.array([r["prominence_score"] for r in rows])
    impacts = np.array([r["separation_reduction"] for r in rows])
    corr = np.corrcoef(scores, impacts)[0, 1]
    print(f"Prominence score vs separation_reduction correlation: r={corr:.3f}")
    print(f"  (r≈0 → prominence is NOT predictive of ablation impact)")
    print(f"  (r≈1 → prominence predicts ablation impact — it's real)\n")

    # ── 2. By CAZ category: mean ablation impact ──
    print("Mean ablation impact by CAZ category:")
    print(f"  {'Category':12s}  {'N':>5s}  {'Sep Reduction':>14s}  {'Supp/Damage':>12s}")
    print(f"  {'-'*55}")
    for cat in ["black_hole", "gentle", "subtle"]:
        cat_rows = [r for r in rows if r["caz_category"] == cat]
        if not cat_rows:
            continue
        sr = np.array([r["separation_reduction"] for r in cat_rows])
        sd = np.array([r["suppression_damage_ratio"] for r in cat_rows
                       if not np.isnan(r["suppression_damage_ratio"])])
        print(f"  {cat:12s}  {len(cat_rows):>5d}  "
              f"{np.mean(sr):>12.3f} ± {np.std(sr):.3f}  "
              f"{np.mean(sd) if len(sd) else float('nan'):>10.3f}")

    # ── 3. By encoding strategy ──
    print("\nMean ablation impact by encoding strategy:")
    print(f"  {'Strategy':12s}  {'N':>5s}  {'Score':>8s}  {'Sep Reduction':>14s}  {'Models'}")
    print(f"  {'-'*65}")
    for strat in ["redundant", "sparse", "unknown"]:
        strat_rows = [r for r in rows if r["encoding_strategy"] == strat]
        if not strat_rows:
            continue
        sr    = np.array([r["separation_reduction"] for r in strat_rows])
        score = np.array([r["prominence_score"] for r in strat_rows])
        models = sorted({r["model_slug"] for r in strat_rows})
        print(f"  {strat:12s}  {len(strat_rows):>5d}  "
              f"{np.mean(score):>7.3f}  "
              f"{np.mean(sr):>12.3f} ± {np.std(sr):.3f}  "
              f"{', '.join(models[:4])}{'...' if len(models)>4 else ''}")

    # ── 4. Gemma spotlight: gentle CAZ ablation impact ──
    gemma_rows = [r for r in rows if "gemma" in r["model_id"].lower()]
    if gemma_rows:
        print(f"\nGemma spotlight ({len(gemma_rows)} measurements):")
        for r in sorted(gemma_rows, key=lambda x: x["concept"]):
            print(f"  {r['model_slug']:20s}  {r['concept']:15s}  "
                  f"score={r['prominence_score']:.3f} [{r['caz_category']:10s}]  "
                  f"sep_reduction={r['separation_reduction']:.3f}")

    # ── 5. Per-concept comparison: redundant vs sparse impact ──
    print(f"\nPer-concept: mean separation_reduction (redundant vs sparse):")
    print(f"  {'Concept':15s}  {'Redundant':>10s}  {'Sparse':>10s}  {'Delta':>8s}")
    print(f"  {'-'*50}")
    for concept in CONCEPTS:
        r_rows = [r for r in rows if r["concept"] == concept and r["encoding_strategy"] == "redundant"]
        s_rows = [r for r in rows if r["concept"] == concept and r["encoding_strategy"] == "sparse"]
        if not r_rows or not s_rows:
            continue
        r_mean = np.mean([r["separation_reduction"] for r in r_rows])
        s_mean = np.mean([r["separation_reduction"] for r in s_rows])
        print(f"  {concept:15s}  {r_mean:>10.3f}  {s_mean:>10.3f}  {s_mean-r_mean:>+8.3f}")

    # ── 6. Key finding summary ──
    bh  = [r["separation_reduction"] for r in rows if r["caz_category"] == "black_hole"]
    gen = [r["separation_reduction"] for r in rows if r["caz_category"] == "gentle"]

    print(f"\n{'='*70}")
    print(f"KEY FINDING")
    print(f"{'='*70}")
    if bh and gen:
        print(f"  Black hole mean impact: {np.mean(bh):.3f}")
        print(f"  Gentle CAZ mean impact: {np.mean(gen):.3f}")
        ratio = np.mean(gen) / np.mean(bh) if np.mean(bh) != 0 else float("nan")
        print(f"  Ratio (gentle/black_hole): {ratio:.2f}")
        if ratio > 0.7:
            print("  → Gentle CAZes have comparable ablation impact to black holes.")
            print("    Prominence score is likely architectural, not semantic.")
        elif ratio < 0.4:
            print("  → Black holes have substantially higher ablation impact.")
            print("    Prominence score may carry genuine semantic weight.")
        else:
            print("  → Mixed result — partial architectural effect.")
    print(f"  Prominence↔impact correlation: r={corr:.3f}")


if __name__ == "__main__":
    main()

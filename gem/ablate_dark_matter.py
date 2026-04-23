#!/usr/bin/env python3
"""
ablate_dark_matter.py — Ablate the titan dark matter features.

For each persistent unlabeled feature from the deep dive, ablate its
direction at EVERY layer it's alive (using the per-layer eigenvectors)
and measure the impact on:
  1. All 7 concept separations (does it affect known concepts?)
  2. Output logit divergence (does the model break?)

This tells us whether dark matter features are:
  - Load-bearing infrastructure (model collapses)
  - Unnamed concepts (specific capabilities break)
  - Epiphenomenal (nothing happens)

Usage:
    python src/ablate_dark_matter.py --model EleutherAI/pythia-1.4b
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from rosetta_tools.ablation import DirectionalAblator, get_transformer_layers
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.caz import compute_separation
from rosetta_tools.gpu_utils import (
    get_device, get_dtype, log_device_info, log_vram, release_model, purge_hf_cache,
    load_model_with_retry, NumpyJSONEncoder,
)
from rosetta_tools.dataset import load_concept_pairs, texts_by_label
from rosetta_tools.paths import ROSETTA_RESULTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RESULTS_DIR = ROSETTA_RESULTS

CONCEPTS: list[str] = [
    "credibility", "negation", "sentiment", "causation",
    "certainty", "moral_valence", "temporal_order",
]


def load_deep_dive(model_id: str):
    """Load deep dive feature map + per-layer eigenvectors."""
    slug = model_id.replace("/", "_").replace("-", "_")
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if not d.name.startswith(f"deepdive_{slug}"):
            continue
        fm = d / "feature_map.json"
        if not fm.exists():
            continue
        feature_map = json.load(open(fm))
        directions = {}
        for li in range(feature_map["n_layers"]):
            npy = d / f"directions_L{li:03d}.npy"
            if npy.exists():
                directions[li] = np.load(npy)
        return feature_map, directions
    return None, None


def find_models_with_deep_dive() -> list[str]:
    """Find enabled models that have deep dive results.

    Disabled models (e.g. gemma-2-9b, which needs --load-in-8bit) are excluded
    from --all runs; invoke them explicitly with --model instead.
    """
    try:
        from rosetta_tools.models import get_model
        def _enabled(mid: str) -> bool:
            m = get_model(mid)
            return m is None or m.enabled
    except ImportError:
        def _enabled(mid: str) -> bool:
            return True

    found = {}
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.name.startswith("deepdive_") and (d / "feature_map.json").exists():
            fm = json.load(open(d / "feature_map.json"))
            mid = fm["model_id"]
            if mid not in found and _enabled(mid):
                found[mid] = d.name
    return sorted(found.keys())


def run_model(model_id: str, args):
    """Run dark matter ablation for one model."""
    device = get_device(args.device)
    dtype = get_dtype(device)
    log_device_info(device, dtype)

    # Load deep dive
    feature_map, directions = load_deep_dive(model_id)
    if not feature_map:
        log.error("No deep dive data for %s", model_id)
        return

    # Get persistent features sorted by eigenvalue
    persistent = sorted(
        [f for f in feature_map["features"] if f["lifespan"] >= 5],
        key=lambda f: -f["peak_eigenvalue"],
    )[:args.top_n]

    log.info("Ablating top %d persistent dark matter features from %s",
             len(persistent), model_id)

    # Force VRAM cleanup before loading
    if device.startswith("cuda"):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model_with_retry(AutoModel, model_id, dtype=dtype, device=device)
    model.eval()
    log_vram("after model load")

    layers = get_transformer_layers(model)
    n_layers = len(layers)

    # Load concept pairs for measuring concept impact
    concept_texts = {}
    for concept in CONCEPTS:
        pairs = load_concept_pairs(concept, n=50)  # 50 pairs for speed
        pos, neg = texts_by_label(pairs)
        concept_texts[concept] = (pos, neg)

    # Baseline: concept separations without ablation
    log.info("Measuring baseline concept separations...")
    baseline_acts = {}
    for concept, (pos, neg) in concept_texts.items():
        pos_acts = extract_layer_activations(model, tokenizer, pos, device=device, batch_size=args.batch_size)
        neg_acts = extract_layer_activations(model, tokenizer, neg, device=device, batch_size=args.batch_size)
        baseline_acts[concept] = (pos_acts, neg_acts)

    baseline_seps = {}
    for concept, (pos_acts, neg_acts) in baseline_acts.items():
        # Measure at the peak layer (highest separation)
        seps = [compute_separation(pos_acts[l+1], neg_acts[l+1]) for l in range(n_layers)]
        peak_layer = int(np.argmax(seps))
        baseline_seps[concept] = {
            "peak_layer": peak_layer,
            "peak_sep": seps[peak_layer],
            "mean_sep": float(np.mean(seps)),
        }

    log.info("Baselines:")
    for c, s in baseline_seps.items():
        log.info("  %s: peak=%.3f at L%d, mean=%.3f", c, s["peak_sep"], s["peak_layer"], s["mean_sep"])

    # Ablate each dark matter feature
    results = []
    for fi, feature in enumerate(persistent):
        fid = feature["feature_id"]
        birth = feature["birth_layer"]
        death = feature["death_layer"]
        lifespan = feature["lifespan"]
        peak_eig = feature["peak_eigenvalue"]

        log.info("")
        log.info("=== Ablating F%03d (L%d-L%d, %d layers, eig=%.1f) ===",
                 fid, birth, death, lifespan, peak_eig)

        # Build per-layer ablation directions
        ablation_layers = []
        ablation_dirs = []
        for li, layer_idx in enumerate(feature["layer_indices"]):
            pc_idx = feature["pc_indices"][li]
            if layer_idx in directions and pc_idx < len(directions[layer_idx]):
                if layer_idx < n_layers:  # skip embedding layer
                    ablation_layers.append(layer_idx)
                    ablation_dirs.append(directions[layer_idx][pc_idx])

        if not ablation_layers:
            log.warning("  No valid ablation layers, skipping")
            continue

        log.info("  Ablating at %d layers: L%d-L%d",
                 len(ablation_layers), ablation_layers[0], ablation_layers[-1])

        # Apply ablation at all layers simultaneously
        model_dtype = torch.bfloat16 if next(model.parameters()).dtype == torch.bfloat16 else torch.float32

        concept_impact = {}
        with ExitStack() as stack:
            for layer_idx, direction in zip(ablation_layers, ablation_dirs):
                stack.enter_context(
                    DirectionalAblator(layers[layer_idx], direction, dtype=model_dtype)
                )

            # Measure concept separations under ablation
            for concept, (pos, neg) in concept_texts.items():
                pos_acts = extract_layer_activations(
                    model, tokenizer, pos, device=device, batch_size=args.batch_size)
                neg_acts = extract_layer_activations(
                    model, tokenizer, neg, device=device, batch_size=args.batch_size)

                bl = baseline_seps[concept]
                act_idx = bl["peak_layer"] + 1
                if act_idx >= len(pos_acts):
                    act_idx = len(pos_acts) - 1
                ablated_sep = compute_separation(pos_acts[act_idx], neg_acts[act_idx])
                retained_pct = 100 * ablated_sep / bl["peak_sep"] if bl["peak_sep"] > 0 else 100

                concept_impact[concept] = {
                    "baseline": round(bl["peak_sep"], 4),
                    "ablated": round(ablated_sep, 4),
                    "retained_pct": round(retained_pct, 1),
                }

        # Log results
        log.info("  Impact on concepts:")
        max_damage = 0
        for concept, impact in sorted(concept_impact.items(), key=lambda x: x[1]["retained_pct"]):
            damage = 100 - impact["retained_pct"]
            max_damage = max(max_damage, damage)
            marker = "!!!" if damage > 20 else "!" if damage > 10 else ""
            log.info("    %s: %.3f → %.3f (%.1f%% retained) %s",
                     concept, impact["baseline"], impact["ablated"],
                     impact["retained_pct"], marker)

        if max_damage < 5:
            verdict = "EPIPHENOMENAL (no concept impact)"
        elif max_damage < 20:
            verdict = "MILD IMPACT (subtle concept effects)"
        elif max(impact["retained_pct"] for impact in concept_impact.values()) > 80:
            verdict = "SELECTIVE (damages some concepts, spares others)"
        else:
            verdict = "INFRASTRUCTURE (broad damage across concepts)"

        log.info("  Verdict: %s", verdict)

        results.append({
            "feature_id": fid,
            "birth_layer": birth,
            "death_layer": death,
            "lifespan": lifespan,
            "peak_eigenvalue": peak_eig,
            "n_ablation_layers": len(ablation_layers),
            "concept_impact": concept_impact,
            "max_damage_pct": round(max_damage, 1),
            "verdict": verdict,
            "concept_alignment": feature.get("concept_alignment", {}),
        })

    # Save results
    release_model(model)
    if not args.no_clean_cache:
        purge_hf_cache(model_id)

    slug = model_id.replace("/", "_").replace("-", "_")
    out_dir = RESULTS_DIR / f"dark_ablation_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "dark_matter_ablation.json", "w") as f:
        json.dump({
            "model_id": model_id,
            "n_features_tested": len(results),
            "baseline_separations": baseline_seps,
            "results": results,
        }, f, indent=2, cls=NumpyJSONEncoder)

    log.info("")
    log.info("=== SUMMARY ===")
    for r in results:
        log.info("  F%03d (eig=%.0f, %dL): %s (max damage=%.1f%%)",
                 r["feature_id"], r["peak_eigenvalue"], r["lifespan"],
                 r["verdict"], r["max_damage_pct"])

    log.info("")
    log.info("Results → %s", out_dir)


def main():
    parser = argparse.ArgumentParser(description="Ablate dark matter features")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Single model ID")
    group.add_argument("--all", action="store_true",
                       help="Run all models with deep dive results")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Ablate the top N persistent features (default: 10)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--no-clean-cache", action="store_true")
    parser.add_argument("--skip-model", action="append", default=[],
                        metavar="MODEL_ID", help="Skip this model (may be repeated)")
    args = parser.parse_args()

    if args.all:
        models = find_models_with_deep_dive()
        log.info("Found %d models with deep dive results: %s",
                 len(models), ", ".join(m.split("/")[-1] for m in models))
    else:
        models = [args.model]

    if args.skip_model:
        skip = set(args.skip_model)
        models = [m for m in models if m not in skip]
        log.info("Skipping %d models: %s", len(skip), sorted(skip))

    for model_id in models:
        run_model(model_id, args)


if __name__ == "__main__":
    main()

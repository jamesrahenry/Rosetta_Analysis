#!/usr/bin/env python3
"""
deep_dive.py — Single-model unsupervised feature discovery.

Runs a complete ACM-style analysis on one model:
  1. Extract activations at every layer from diverse text
  2. Compute manifold census with eigenvector storage
  3. Track features across layers (cross-layer cosine matching)
  4. Compare discovered features against known concept directions
  5. Report: total features, persistent vs transient, labeled vs dark matter

This is the core of the Activation Manifold Cartography methodology —
finding every organized direction in a model's activation space and
tracking how features are born, evolve, and die across depth.

Usage:
    python src/deep_dive.py --model EleutherAI/pythia-1.4b
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.gpu_utils import (
    get_device,
    get_dtype,
    log_device_info,
    log_vram,
    release_model,
    purge_hf_cache,
)
from rosetta_tools.manifold_detector import _layer_census
from rosetta_tools.feature_tracker import track_features
from rosetta_tools.dataset import load_concept_pairs
from rosetta_tools.paths import ROSETTA_RESULTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_ROOT = ROSETTA_RESULTS

CONCEPTS: list[str] = [
    "credibility", "negation", "sentiment", "causation",
    "certainty", "moral_valence", "temporal_order",
]


def load_all_texts(n_pairs: int = 200) -> list[str]:
    """Load all contrastive pair texts, labels stripped."""
    all_texts = []
    for concept in CONCEPTS:
        pairs = load_concept_pairs(concept, n=n_pairs)
        all_texts.extend([p.pos_text for p in pairs])
        all_texts.extend([p.neg_text for p in pairs])
        log.info("  %s: %d texts", concept, len(pairs) * 2)

    rng = np.random.default_rng(42)
    rng.shuffle(all_texts)
    log.info("  Total texts: %d", len(all_texts))
    return all_texts


def load_concept_directions_at_layers(model_id: str) -> dict[str, dict[int, np.ndarray]]:
    """Load per-layer concept directions from prior CAZ extraction."""
    for d in sorted(RESULTS_ROOT.iterdir(), reverse=True):
        sf = d / "run_summary.json"
        if not sf.exists():
            continue
        try:
            if json.load(open(sf)).get("model_id") == model_id:
                concept_dirs = {}
                for concept in CONCEPTS:
                    caz_file = d / f"caz_{concept}.json"
                    if not caz_file.exists():
                        continue
                    data = json.load(open(caz_file))
                    layer_dirs = {}
                    for m in data["layer_data"]["metrics"]:
                        layer_dirs[m["layer"]] = np.array(m["dom_vector"], dtype=np.float64)
                    concept_dirs[concept] = layer_dirs
                log.info("Loaded concept directions from %s", d)
                return concept_dirs
        except (json.JSONDecodeError, KeyError):
            continue
    return {}


def run_deep_dive(model_id: str, args) -> None:
    """Run full unsupervised feature discovery on one model."""
    log.info("=" * 60)
    log.info("DEEP DIVE: %s", model_id)
    log.info("=" * 60)

    device = get_device(args.device)
    dtype = get_dtype(device)

    if device.startswith("cuda"):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    log_device_info(device, dtype)

    # Load texts
    log.info("Loading texts...")
    texts = load_all_texts(n_pairs=getattr(args, "n_pairs", 200))

    # Load known concept directions
    all_concept_dirs = load_concept_directions_at_layers(model_id)

    # Load model
    log.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if args.load_in_8bit:
        # 8-bit quantization via bitsandbytes — requires accelerate + bitsandbytes
        try:
            import accelerate  # noqa: F401
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise SystemExit(
                f"--load-in-8bit requires accelerate and bitsandbytes: {e}\n"
                "  pip install accelerate bitsandbytes"
            ) from e
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModel.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto"
        )
    else:
        try:
            model = AutoModel.from_pretrained(model_id, dtype=dtype, device_map=device)
        except (ValueError, ImportError):
            model = AutoModel.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    log_vram("after model load")

    # Extract activations
    log.info("Extracting activations from %d texts...", len(texts))
    t0 = time.time()
    layer_acts = extract_layer_activations(
        model, tokenizer, texts,
        device=device, batch_size=args.batch_size,
    )
    n_layers = len(layer_acts)
    hidden_dim = layer_acts[0].shape[1]
    log.info("Extraction: %.1fs (%d layers × %d samples × %d dims)",
             time.time() - t0, n_layers, layer_acts[0].shape[0], hidden_dim)

    release_model(model)
    if not args.no_clean_cache:
        purge_hf_cache(model_id)

    # ── Step 1: Manifold census with eigenvector storage ──
    log.info("Computing manifold census with direction storage...")
    t0 = time.time()

    n_top = min(50, layer_acts[0].shape[0] - 1)  # can't have more PCs than samples
    layer_results = []

    for layer_idx in range(n_layers):
        # Get concept directions for this layer, filtering for dimension match
        layer_concept_dirs = {}
        for concept, layer_dirs in all_concept_dirs.items():
            if layer_idx in layer_dirs:
                vec = layer_dirs[layer_idx]
                if len(vec) == hidden_dim:
                    layer_concept_dirs[concept] = vec

        result = _layer_census(
            layer_acts[layer_idx], layer_idx, layer_concept_dirs,
            n_top_eigenvalues=n_top, store_directions=True,
        )
        layer_results.append(result)

        if layer_idx % 5 == 0 or layer_idx == n_layers - 1:
            log.info("  L%d/%d: eff_dim=%.1f  sig=%d  concept_cov=%.1f%%",
                     layer_idx, n_layers - 1,
                     result.effective_dim, result.significant_dims,
                     100 * result.concept_coverage)

    log.info("Census: %.1fs", time.time() - t0)

    # Free raw activations
    del layer_acts

    # ── Step 2: Cross-layer feature tracking ──
    log.info("Tracking features across layers...")
    t0 = time.time()

    layer_directions = [r.top_directions for r in layer_results]
    layer_eigenvalues = [r.top_eigenvalues for r in layer_results]

    feature_map = track_features(
        layer_directions=layer_directions,
        layer_eigenvalues=layer_eigenvalues,
        n_layers_total=n_layers,
        cos_threshold=args.cos_threshold,
        concept_directions=all_concept_dirs,
        model_id=model_id,
    )

    log.info("Tracking: %.1fs", time.time() - t0)

    # ── Step 3: Report ──
    log.info("")
    log.info("=" * 60)
    log.info("FEATURE MAP: %s", model_id.split("/")[-1])
    log.info("=" * 60)
    log.info("  Layers: %d  Hidden dim: %d", n_layers, hidden_dim)
    log.info("  Total features discovered: %d", feature_map.n_features)
    log.info("  Persistent (5+ layers):    %d", feature_map.n_persistent)
    log.info("  Transient (1-2 layers):    %d", feature_map.n_transient)
    log.info("  Max concurrent at one layer: %d", feature_map.max_concurrent)
    log.info("  Matching known concepts:   %d", feature_map.n_labeled)
    log.info("  UNLABELED (dark matter):   %d", feature_map.n_unlabeled)
    log.info("")

    # Top features by strength
    log.info("── TOP 20 FEATURES (by peak eigenvalue) ──")
    for f in feature_map.features[:20]:
        # Best concept match
        if f.concept_alignment:
            best_concept = max(f.concept_alignment, key=f.concept_alignment.get)
            best_cos2 = f.concept_alignment[best_concept]
            label = f"{best_concept}({best_cos2:.2f})" if best_cos2 > 0.1 else "UNLABELED"
        else:
            label = "UNLABELED"

        log.info("  F%03d  L%d-L%d (%d layers)  peak=L%d (%.0f%%)  "
                 "eig=%.1f  [%s]",
                 f.feature_id, f.birth_layer, f.death_layer, f.lifespan,
                 f.peak_layer, f.peak_depth_pct,
                 f.peak_eigenvalue, label)

    # Persistent unlabeled features — the dark matter
    unlabeled_persistent = [f for f in feature_map.features
                           if f.is_persistent and not any(v > 0.3 for v in f.concept_alignment.values())]
    if unlabeled_persistent:
        log.info("")
        log.info("── PERSISTENT UNLABELED FEATURES (dark matter) ──")
        for f in unlabeled_persistent[:15]:
            best = max(f.concept_alignment, key=f.concept_alignment.get) if f.concept_alignment else "?"
            best_val = f.concept_alignment.get(best, 0)
            log.info("  F%03d  L%d-L%d (%d layers)  peak_eig=%.1f  "
                     "nearest_concept=%s(%.2f)",
                     f.feature_id, f.birth_layer, f.death_layer, f.lifespan,
                     f.peak_eigenvalue, best, best_val)

    # Layer occupancy
    log.info("")
    log.info("── LAYER OCCUPANCY ──")
    for l in range(n_layers):
        alive = feature_map.features_at_layer(l)
        n_alive = len(alive)
        bar = "#" * min(n_alive, 60)
        labeled = sum(1 for f in alive if any(v > 0.3 for v in f.concept_alignment.values()))
        log.info("  L%2d: %3d features (%2d labeled, %3d dark)  %s",
                 l, n_alive, labeled, n_alive - labeled, bar)

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_id.replace("/", "_").replace("-", "_")
    out_dir = Path("results") / f"deepdive_{model_slug}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save feature map as JSON
    features_json = []
    for f in feature_map.features:
        features_json.append({
            "feature_id": f.feature_id,
            "birth_layer": f.birth_layer,
            "death_layer": f.death_layer,
            "lifespan": f.lifespan,
            "layer_indices": f.layer_indices,
            "pc_indices": f.pc_indices,
            "eigenvalues": f.eigenvalues,
            "cos_chain": f.cos_chain,
            "peak_layer": f.peak_layer,
            "peak_eigenvalue": f.peak_eigenvalue,
            "peak_depth_pct": f.peak_depth_pct,
            "mean_eigenvalue": f.mean_eigenvalue,
            "concept_alignment": f.concept_alignment,
            "concept_alignment_trajectory": {
                c: {str(layer): val for layer, val in layers.items()}
                for c, layers in f.concept_alignment_trajectory.items()
            },
            "is_persistent": f.is_persistent,
            "is_transient": f.is_transient,
        })

    summary = {
        "model_id": model_id,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "n_features": feature_map.n_features,
        "n_persistent": feature_map.n_persistent,
        "n_transient": feature_map.n_transient,
        "n_labeled": feature_map.n_labeled,
        "n_unlabeled": feature_map.n_unlabeled,
        "max_concurrent": feature_map.max_concurrent,
        "cos_threshold": args.cos_threshold,
        "load_in_8bit": args.load_in_8bit,
        "timestamp": timestamp,
        "features": features_json,
    }

    with open(out_dir / "feature_map.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save eigenvectors for later analysis (numpy)
    for layer_idx, result in enumerate(layer_results):
        if result.top_directions is not None:
            np.save(out_dir / f"directions_L{layer_idx:03d}.npy", result.top_directions)

    log.info("")
    log.info("Results saved to %s", out_dir)


from rosetta_tools.models import all_models, models_by_tag


def main():
    parser = argparse.ArgumentParser(description="Single-model deep feature dive")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="HuggingFace model ID")
    group.add_argument("--all", action="store_true",
                       help="Run all enabled base models")
    group.add_argument("--tag", type=str,
                       help="Run all models with this tag (e.g. 'instruct')")
    group.add_argument("--everything", action="store_true",
                       help="Run all enabled base models + all tagged models")
    parser.add_argument("--n-pairs", type=int, default=200,
                        help="Pairs per concept to load (default: 200)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--cos-threshold", type=float, default=0.5,
                        help="Cosine similarity threshold for feature tracking (default: 0.5)")
    parser.add_argument("--no-clean-cache", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit quantization via bitsandbytes")
    args = parser.parse_args()

    if args.everything:
        models = all_models(include_disabled=False) + models_by_tag("instruct")
    elif args.tag:
        from rosetta_tools.models import get_model
        all_tagged = models_by_tag(args.tag, include_disabled=True)
        if args.load_in_8bit:
            models = all_tagged
        else:
            # Skip models flagged as OOM in bfloat16 — run those separately with --load-in-8bit
            oom, models = [], []
            for mid in all_tagged:
                m = get_model(mid)
                if m and any("OOM" in q for q in m.quirks):
                    oom.append(mid)
                else:
                    models.append(mid)
            if oom:
                log.warning("Skipping OOM models (add --load-in-8bit to include): %s",
                            ", ".join(m.split("/")[-1] for m in oom))
    elif args.all:
        models = all_models()
    else:
        models = [args.model]

    log.info("Queued %d models", len(models))
    for model_id in models:
        run_deep_dive(model_id, args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
detect_manifolds.py — Unsupervised manifold census for transformer models.

At each layer, measures how many distinct organized directions exist in
the activation space, how much is explained by known concept directions
(from prior CAZ extraction), and how much remains unexplained.

This is the foundation for discovering unlabeled concept assembly zones —
features the model computes that don't map to any human-labeled concept.

Usage:
    # Single model
    python src/detect_manifolds.py --model EleutherAI/pythia-1.4b

    # All cross-arch models
    python src/detect_manifolds.py --all

    # With custom text corpus
    python src/detect_manifolds.py --model EleutherAI/pythia-1.4b --corpus data/diverse.txt
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
    safe_batch_size,
)
from rosetta_tools.manifold_detector import layer_manifold_census
from rosetta_tools.dataset import load_concept_pairs
from rosetta_tools.paths import ROSETTA_RESULTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Concept directions live in the semantic_convergence extraction results
EXTRACTION_ROOT = Path(__file__).parent.parent.parent / "semantic_convergence" / "results"

CONCEPTS: list[str] = [
    "credibility", "negation", "sentiment", "causation",
    "certainty", "moral_valence", "temporal_order",
]

from rosetta_tools.models import all_models

CROSS_ARCH_MODELS = all_models()


# ---------------------------------------------------------------------------
# Text corpus loading
# ---------------------------------------------------------------------------


def load_diverse_texts(corpus_path: Path | None, n_texts: int = 200) -> list[str]:
    """Load diverse texts for unsupervised manifold detection.

    If a corpus file is provided, reads lines from it.
    Otherwise, mixes ALL contrastive pair texts (ignoring labels)
    to create a diverse sample from existing data.
    """
    if corpus_path and corpus_path.exists():
        log.info("Loading corpus from %s", corpus_path)
        texts = [line.strip() for line in corpus_path.read_text().splitlines() if line.strip()]
        if n_texts and len(texts) > n_texts:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(texts), size=n_texts, replace=False)
            texts = [texts[i] for i in idx]
        log.info("  Loaded %d texts from corpus", len(texts))
        return texts

    # Mix all contrastive pairs (ignoring labels) — use everything we have
    log.info("No corpus provided — mixing all contrastive pair texts (labels stripped)")
    all_texts = []
    for concept in CONCEPTS:
        pairs = load_concept_pairs(concept)
        all_texts.extend([p.pos_text for p in pairs])
        all_texts.extend([p.neg_text for p in pairs])
        log.info("  %s: %d texts", concept, len(pairs))

    # Shuffle deterministically
    rng = np.random.default_rng(42)
    rng.shuffle(all_texts)
    if n_texts and len(all_texts) > n_texts:
        all_texts = all_texts[:n_texts]

    log.info("  Total diverse texts: %d", len(all_texts))
    return all_texts


# ---------------------------------------------------------------------------
# Load known concept directions from prior CAZ results
# ---------------------------------------------------------------------------


def find_extraction_dir(model_id: str, search_root: Path | None = None) -> Path | None:
    """Find the most recent CAZ extraction directory for a model."""
    root = search_root or EXTRACTION_ROOT
    if not root.exists():
        log.warning("Extraction results dir does not exist: %s", root)
        return None
    candidates = []
    for d in root.iterdir():
        summary_f = d / "run_summary.json"
        if not summary_f.exists():
            continue
        try:
            with open(summary_f) as f:
                summary = json.load(f)
            if summary.get("model_id") == model_id:
                candidates.append(d)
        except (json.JSONDecodeError, KeyError):
            continue
    if not candidates:
        return None
    # Return the most recent (sorted by name, which contains timestamp)
    return sorted(candidates)[-1]


def load_concept_directions(model_id: str, search_root: Path | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Load per-layer concept directions from prior CAZ extraction.

    Returns
    -------
    dict mapping concept_name → dict mapping layer_index → direction vector
    """
    ext_dir = find_extraction_dir(model_id, search_root)
    if ext_dir is None:
        log.warning("No prior CAZ extraction found for %s", model_id)
        return {}

    log.info("Loading concept directions from %s", ext_dir)
    concept_dirs = {}

    for concept in CONCEPTS:
        caz_file = ext_dir / f"caz_{concept}.json"
        if not caz_file.exists():
            continue
        with open(caz_file) as f:
            data = json.load(f)

        layer_dirs = {}
        for m in data["layer_data"]["metrics"]:
            layer_dirs[m["layer"]] = np.array(m["dom_vector"], dtype=np.float64)

        concept_dirs[concept] = layer_dirs
        log.info("  %s: %d layers with directions", concept, len(layer_dirs))

    return concept_dirs


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_model(model_id: str, args) -> None:
    """Run manifold census on one model."""
    log.info("=" * 60)
    log.info("Manifold census: %s", model_id)
    log.info("=" * 60)

    device = get_device(args.device)
    dtype = get_dtype(device)

    if device.startswith("cuda"):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    log_device_info(device, dtype)

    # Load diverse texts
    corpus_path = Path(args.corpus) if args.corpus else None
    texts = load_diverse_texts(corpus_path, n_texts=args.n_texts)

    # Load concept directions from prior extraction
    ext_root = Path(args.extraction_dir) if getattr(args, "extraction_dir", None) else None
    all_concept_dirs = load_concept_directions(model_id, ext_root)

    # Load model
    log.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModel.from_pretrained(model_id, dtype=dtype, device_map=device)
    except (ValueError, ImportError):
        model = AutoModel.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    log_vram("after model load")

    # Extract activations at all layers
    batch = safe_batch_size(args.batch_size, device)
    log.info("Extracting activations from %d texts (batch_size=%d)...", len(texts), batch)
    t0 = time.time()
    layer_acts = extract_layer_activations(
        model, tokenizer, texts,
        device=device, batch_size=batch,
    )
    log.info("Extraction: %.1fs  (%d layers × %d samples × %d dims)",
             time.time() - t0, len(layer_acts), layer_acts[0].shape[0], layer_acts[0].shape[1])

    # Build per-layer concept direction dict
    # For the census, we need concept directions at each specific layer
    n_layers = len(layer_acts)
    census_results = []

    log.info("Computing manifold census...")
    t0 = time.time()
    for layer_idx in range(n_layers):
        # Gather concept directions for THIS layer
        hidden_dim = layer_acts[layer_idx].shape[1]
        layer_concept_dirs = {}
        for concept, layer_dirs in all_concept_dirs.items():
            if layer_idx in layer_dirs:
                vec = layer_dirs[layer_idx]
                if vec.shape[0] != hidden_dim:
                    if layer_idx == 0:
                        log.warning(
                            "Concept direction dim mismatch for %s: "
                            "direction=%d, activations=%d — skipping",
                            concept, vec.shape[0], hidden_dim)
                    continue
                layer_concept_dirs[concept] = vec

        # Run single-layer census
        from rosetta_tools.manifold_detector import _layer_census
        result = _layer_census(
            layer_acts[layer_idx], layer_idx, layer_concept_dirs,
        )
        census_results.append(result)

        # Progress
        if layer_idx % 5 == 0 or layer_idx == n_layers - 1:
            log.info(
                "  L%d/%d: eff_dim=%.1f  sig_dims=%d  concept_cov=%.1f%%  "
                "residual_sig=%d  residual_dim=%.1f",
                layer_idx, n_layers - 1,
                result.effective_dim, result.significant_dims,
                100 * result.concept_coverage,
                result.residual_significant, result.residual_dim,
            )

    log.info("Census: %.1fs", time.time() - t0)

    # Release model
    release_model(model)
    if not getattr(args, "no_clean_cache", False):
        purge_hf_cache(model_id)

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_id.replace("/", "_").replace("-", "_")
    out_dir = ROSETTA_RESULTS / f"manifold_{model_slug}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    summary = {
        "model_id": model_id,
        "n_layers": n_layers,
        "hidden_dim": int(layer_acts[0].shape[1]),
        "n_samples": int(layer_acts[0].shape[0]),
        "concept_names": sorted(all_concept_dirs.keys()),
        "timestamp": timestamp,
        "layers": [],
    }

    for r in census_results:
        layer_data = {
            "layer": r.layer,
            "effective_dim": round(r.effective_dim, 2),
            "significant_dims": r.significant_dims,
            "total_variance": round(r.total_variance, 4),
            "concept_coverage": round(r.concept_coverage, 6),
            "concept_dims": r.concept_dims,
            "residual_dim": round(r.residual_dim, 2),
            "residual_significant": r.residual_significant,
            "residual_variance": round(r.residual_variance, 4),
            "per_concept_variance": {
                k: round(v, 6) for k, v in r.per_concept_variance.items()
            },
            "top_eigenvalues": [round(v, 6) for v in r.top_eigenvalues[:20]],
        }
        summary["layers"].append(layer_data)

    with open(out_dir / "manifold_census.json", "w") as f:
        json.dump(summary, f, indent=2)

    # CSV for easy analysis
    import csv
    csv_path = out_dir / "manifold_census.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer", "depth_pct", "effective_dim", "significant_dims",
            "concept_coverage", "residual_dim", "residual_significant",
            "total_variance", "residual_variance",
        ])
        for r in census_results:
            writer.writerow([
                r.layer,
                round(100 * r.layer / (n_layers - 1), 1) if n_layers > 1 else 0,
                round(r.effective_dim, 2),
                r.significant_dims,
                round(r.concept_coverage, 6),
                round(r.residual_dim, 2),
                r.residual_significant,
                round(r.total_variance, 4),
                round(r.residual_variance, 4),
            ])

    log.info("Results saved to %s", out_dir)

    # Print summary
    log.info("")
    log.info("── MANIFOLD CENSUS SUMMARY: %s ──", model_id.split("/")[-1])
    log.info("  Layers: %d  Hidden dim: %d  Samples: %d",
             n_layers, layer_acts[0].shape[1], layer_acts[0].shape[0])
    log.info("  Known concepts: %d (%s)",
             len(all_concept_dirs), ", ".join(sorted(all_concept_dirs.keys())))
    log.info("")

    eff_dims = [r.effective_dim for r in census_results]
    sig_dims = [r.significant_dims for r in census_results]
    coverages = [r.concept_coverage for r in census_results]
    res_sigs = [r.residual_significant for r in census_results]

    log.info("  Effective dim:     min=%.1f  max=%.1f  mean=%.1f",
             min(eff_dims), max(eff_dims), np.mean(eff_dims))
    log.info("  Significant dims:  min=%d  max=%d  mean=%.1f",
             min(sig_dims), max(sig_dims), np.mean(sig_dims))
    log.info("  Concept coverage:  min=%.2f%%  max=%.2f%%  mean=%.2f%%",
             100 * min(coverages), 100 * max(coverages), 100 * np.mean(coverages))
    log.info("  Residual sig dims: min=%d  max=%d  mean=%.1f",
             min(res_sigs), max(res_sigs), np.mean(res_sigs))
    log.info("")
    log.info("  → %d known concept directions explain %.1f%% of variance on average",
             len(all_concept_dirs), 100 * np.mean(coverages))
    log.info("  → %.0f significant unexplained directions remain on average",
             np.mean(res_sigs))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Unsupervised manifold census")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Single HuggingFace model ID")
    group.add_argument("--all", action="store_true", help="Run all cross-arch models")
    parser.add_argument("--corpus", type=str, default=None,
                        help="Path to text file (one text per line) for diverse activations")
    parser.add_argument("--n-texts", type=int, default=0,
                        help="Max texts to sample (default: 0 = use all available)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--no-clean-cache", action="store_true")
    parser.add_argument("--extraction-dir", type=str, default=None,
                        help="Path to semantic_convergence results dir "
                             "(default: ../semantic_convergence/results)")
    return parser.parse_args()


def main():
    args = parse_args()
    models = CROSS_ARCH_MODELS if args.all else [args.model]
    for model_id in models:
        run_model(model_id, args)


if __name__ == "__main__":
    main()

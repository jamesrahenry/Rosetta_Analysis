"""
ablate_random_direction.py — Random-direction ablation null at CAZ peak.

Answers the reviewer's question: is the separation reduction we observe at CAZ
peak layers *specific to the concept direction*, or would ablating any random
unit vector at that layer produce a comparable drop?

For each (model, concept):
  1. Load baseline + concept-direction reduction at CAZ peak from existing
     `ablation_global_sweep_<concept>.json`.
  2. Generate N random unit vectors (same dimensionality as residual stream),
     ablate each at the CAZ peak layer, measure separation reduction at the
     final layer.
  3. Report concept reduction vs random distribution: mean, std, max, p-value,
     and a specificity ratio (concept_red / random_mean_red).

This is the direct null that §8.4 acknowledged was missing.

Usage
-----
    python src/ablate_random_direction.py --model EleutherAI/pythia-70m
    python src/ablate_random_direction.py --all
    python src/ablate_random_direction.py --all --n-seeds 20

Output per (model, concept):
    results/<extraction_dir>/ablation_random_<concept>.json

Written: 2026-04-18 UTC
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta_tools.gpu_utils import (
    get_device, get_dtype, log_device_info, log_vram, release_model, purge_hf_cache,
    vram_stats,
)
from rosetta_tools.models import vram_gb as _registry_vram
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.caz import compute_separation
from rosetta_tools.ablation import DirectionalAblator, get_transformer_layers
from rosetta_tools.dataset import load_concept_pairs, texts_by_label
from rosetta_tools.gem import find_extraction_dir, discover_all_models, load_gem
from rosetta_tools.paths import ROSETTA_RESULTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONCEPTS: list[str] = [
    "credibility", "negation", "sentiment", "causation",
    "certainty", "moral_valence", "temporal_order",
]


def random_unit_vector(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Draw an isotropic random unit vector of given dimensionality."""
    v = rng.standard_normal(dim).astype(np.float64)
    return v / np.linalg.norm(v)


def measure_final_sep_with_ablation(
    model, tokenizer, layers, ablate_layer_idx,
    direction, pos_texts, neg_texts, device, batch_size,
) -> float:
    dtype = torch.bfloat16 if next(model.parameters()).dtype == torch.bfloat16 else torch.float32
    direction_t = torch.tensor(direction, dtype=dtype, device=device)
    direction_t = direction_t / direction_t.norm()

    with DirectionalAblator(layers[ablate_layer_idx], direction_t, dtype=dtype):
        pos_acts = extract_layer_activations(
            model, tokenizer, pos_texts, device=device, batch_size=batch_size, pool="last"
        )
        neg_acts = extract_layer_activations(
            model, tokenizer, neg_texts, device=device, batch_size=batch_size, pool="last"
        )
    return float(compute_separation(pos_acts[-1], neg_acts[-1]))


def measure_baseline_final_sep(
    model, tokenizer, pos_texts, neg_texts, device, batch_size,
) -> float:
    pos_acts = extract_layer_activations(
        model, tokenizer, pos_texts, device=device, batch_size=batch_size, pool="last"
    )
    neg_acts = extract_layer_activations(
        model, tokenizer, neg_texts, device=device, batch_size=batch_size, pool="last"
    )
    return float(compute_separation(pos_acts[-1], neg_acts[-1]))


def run_concept(
    model, tokenizer, concept, extraction_dir,
    n_seeds, device, n_pairs, batch_size,
) -> dict | None:
    """Run random-direction ablation at CAZ peak for one concept."""
    abl_path = extraction_dir / f"ablation_gem_{concept}.json"
    gem_path = extraction_dir / f"gem_{concept}.json"
    if not abl_path.exists() or not gem_path.exists():
        log.warning("  No ablation_gem/gem for %s — skipping", concept)
        return None

    abl = json.loads(abl_path.read_text())
    gem = load_gem(gem_path)
    if gem.n_nodes == 0:
        log.warning("  No GEM nodes for %s — skipping", concept)
        return None

    # Dominant node (highest caz_score) sets the target peak layer
    dominant_node = max(gem.nodes, key=lambda n: n.caz_score)
    caz_peak = dominant_node.caz_peak
    n_layers = dominant_node.n_layers_total
    model_id = abl["model_id"]

    # Concept reduction from peak ablation section
    peak = abl.get("peak", {})
    concept_red = float(peak.get("final_sep_reduction", 0.0))

    # Baseline final-layer separation: check peak then handoff per_layer
    final_layer = n_layers - 1
    baseline_final_sep = 0.0
    for pl in (peak.get("per_layer", {}), abl.get("handoff", {}).get("per_layer", {})):
        for k, v in pl.items():
            if int(k) == final_layer and isinstance(v, dict):
                baseline_final_sep = float(v.get("baseline_sep", 0.0))
                break
        if baseline_final_sep > 0:
            break

    if concept_red <= 0.0:
        log.warning("  Concept reduction zero for %s — skipping", concept)
        return None

    # Load pairs
    pairs = load_concept_pairs(concept, n=n_pairs or 200)
    pos_texts, neg_texts = texts_by_label(pairs)

    layers = get_transformer_layers(model)

    # Infer residual stream dim from model config
    try:
        hidden_dim = int(model.config.hidden_size)
    except AttributeError:
        hidden_dim = int(model.config.n_embd)

    log.info("  Concept=%s caz_peak=L%d dim=%d concept_red=%.4f n_seeds=%d",
             concept, caz_peak, hidden_dim, concept_red, n_seeds)

    # Re-measure baseline for consistency (since we're running fresh forward passes)
    baseline_check = measure_baseline_final_sep(
        model, tokenizer, pos_texts, neg_texts, device, batch_size
    )
    log.info("    baseline check: %.4f (sweep reported: %.4f)",
             baseline_check, baseline_final_sep)

    # Use the freshly measured baseline as the denominator
    baseline_final_sep = baseline_check

    # Random-direction ablations
    rng = np.random.default_rng(seed=42)
    random_results = []
    for seed in range(n_seeds):
        t0 = time.time()
        direction = random_unit_vector(hidden_dim, rng)
        ablated_sep = measure_final_sep_with_ablation(
            model, tokenizer, layers, caz_peak, direction,
            pos_texts, neg_texts, device, batch_size,
        )
        reduction = max(0.0, (baseline_final_sep - ablated_sep) / baseline_final_sep) \
            if baseline_final_sep > 0 else 0.0
        random_results.append({
            "seed": seed,
            "ablated_final_sep": round(ablated_sep, 4),
            "global_sep_reduction": round(reduction, 4),
        })
        log.info("    seed=%d ablated_sep=%.4f reduction=%.3f (%.1fs)",
                 seed, ablated_sep, reduction, time.time() - t0)

    random_reds = np.array([r["global_sep_reduction"] for r in random_results])
    random_mean = float(random_reds.mean())
    random_std  = float(random_reds.std(ddof=1)) if len(random_reds) > 1 else 0.0
    random_max  = float(random_reds.max())

    # Specificity ratio: how much bigger is concept-direction ablation vs random
    ratio = concept_red / max(random_mean, 1e-6)

    # One-sample z-test: is concept_red significantly above the random distribution?
    if random_std > 0:
        z_score = (concept_red - random_mean) / random_std
    else:
        z_score = float("inf") if concept_red > random_mean else 0.0

    # Count of random seeds that met or exceeded concept_red
    n_random_ge_concept = int((random_reds >= concept_red).sum())
    # One-sided empirical p: (n_random >= concept + 1) / (n_random + 1)
    empirical_p = (n_random_ge_concept + 1) / (len(random_reds) + 1)

    log.info("    concept_red=%.4f random_mean=%.4f±%.4f ratio=%.2fx z=%.2f p=%.3f",
             concept_red, random_mean, random_std, ratio, z_score, empirical_p)

    return {
        "model_id":                  model_id,
        "concept":                   concept,
        "caz_peak":                  caz_peak,
        "n_layers":                  n_layers,
        "hidden_dim":                hidden_dim,
        "n_random_seeds":            n_seeds,
        "baseline_final_sep":        round(baseline_final_sep, 4),
        "concept_direction_reduction":   round(concept_red, 4),
        "random_directions":         random_results,
        "random_mean_reduction":     round(random_mean, 4),
        "random_std_reduction":      round(random_std, 4),
        "random_max_reduction":      round(random_max, 4),
        "specificity_ratio":         round(ratio, 2),
        "z_score":                   round(z_score, 2),
        "empirical_p_one_sided":     round(empirical_p, 4),
        "n_random_ge_concept":       n_random_ge_concept,
    }


def run_model(model_id: str, concepts: list[str], args) -> None:
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.error("No extraction results for %s — skipping", model_id)
        return

    pending = [c for c in concepts
               if not (extraction_dir / f"ablation_random_{c}.json").exists()
               or args.overwrite]
    if not pending:
        log.info("Already done: %s (use --overwrite to rerun)", model_id)
        return
    concepts = pending

    log.info("=== Random-direction control: %s ===", model_id)
    device = get_device(args.device)
    dtype  = get_dtype(args.dtype, device)
    log_device_info(device, dtype)

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    model_vram = _registry_vram(model_id)
    device_map = "auto" if (model_vram > 20.0 and n_gpus > 1) else None
    if device_map:
        log.info("Large model (%.0f GB bf16): device_map='auto' across %d GPUs",
                 model_vram, n_gpus)
    try:
        from rosetta_tools.gpu_utils import load_causal_lm
        model, tokenizer = load_causal_lm(model_id, device, dtype, device_map=device_map)
    except Exception as e:
        log.error("Failed to load %s: %s", model_id, e)
        return

    if device == "cuda":
        stats = vram_stats(device)
        if stats:
            log_vram(device)

    t_model_start = time.time()

    for concept in concepts:
        out_path = extraction_dir / f"ablation_random_{concept}.json"
        if out_path.exists() and not args.overwrite:
            log.info("  Skipping %s (already done)", concept)
            continue
        try:
            result = run_concept(
                model, tokenizer, concept, extraction_dir,
                n_seeds=args.n_seeds, device=device,
                n_pairs=args.n_pairs, batch_size=args.batch_size,
            )
        except Exception as e:
            log.error("  %s %s failed: %s", model_id, concept, e)
            continue
        if result is None:
            continue
        out_path.write_text(json.dumps(result, indent=2))
        log.info("  Wrote %s", out_path)

    release_model(model)
    if not args.no_clean_cache:
        purge_hf_cache(model_id)

    log.info("Done: %s (%.1fs)", model_id, time.time() - t_model_start)


def main():
    parser = argparse.ArgumentParser(
        description="Random-direction ablation null at CAZ peak layer — the reviewer's missing control."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all", action="store_true")

    parser.add_argument("--concepts", nargs="+", default=None)
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Number of random unit vectors to ablate per (model, concept). Default: 10")
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-clean-cache", action="store_true")

    args = parser.parse_args()
    concepts = args.concepts or CONCEPTS

    if args.all:
        models = discover_all_models()
        log.info("Found %d models", len(models))
    else:
        models = [args.model]

    for model_id in models:
        run_model(model_id, concepts, args)


if __name__ == "__main__":
    main()

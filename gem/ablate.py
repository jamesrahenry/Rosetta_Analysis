"""
ablate.py — Layer-wise ablation sweep for CAZ Prediction 1.

Tests the Mid-Stream Ablation Hypothesis: does orthogonal projection
ablation at the CAZ peak produce maximum behavioral suppression with
minimum collateral capability damage?

For each model × concept, this script:
  1. Loads the concept direction (dom_vector) from prior extraction results
  2. Sweeps every layer, applying DirectionalAblator at each
  3. Measures separation reduction (concept suppression) and KL divergence
     (capability damage) at each layer
  4. Computes the suppression-to-damage ratio

The prediction: this ratio peaks within the CAZ window [l_start, l_end]
and degrades sharply post-CAZ.

Usage
-----
    # Single model — uses latest extraction results
    python src/ablate.py --model EleutherAI/pythia-410m

    # Specific concept
    python src/ablate.py --model EleutherAI/pythia-410m --concepts credibility negation

    # All models that have extraction results
    python src/ablate.py --all

    # With cache cleanup
    python src/ablate.py --all --clean-cache

Results are written to results/<matching_extraction_dir>/ablation_<concept>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from rosetta_tools.gpu_utils import (
    get_device, get_dtype, log_device_info, log_vram, release_model, purge_hf_cache,
)
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.caz import compute_separation, LayerMetrics, find_caz_boundary
from rosetta_tools.ablation import (
    DirectionalAblator, get_transformer_layers,
    compute_dominant_direction, kl_divergence_from_logits,
)
from rosetta_tools.dataset import load_concept_pairs, texts_by_label
from rosetta_tools.gem import find_extraction_dir, discover_all_models
from rosetta_tools.paths import ROSETTA_RESULTS
from rosetta_tools.gpu_utils import load_causal_lm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONCEPTS: list[str] = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

P3_MODELS: list[str] = [
    # Pythia
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    # GPT-2
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
    # OPT
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    # Qwen2.5
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    # Gemma-2
    "google/gemma-2-2b",
    "google/gemma-2-9b",
    # Llama-3.2
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    # Mistral / Phi
    "mistralai/Mistral-7B-v0.3",
    "microsoft/phi-2",
]

# General capability prompts for KL divergence measurement.
# These should be unrelated to any tested concept — we want to measure
# collateral damage, not concept suppression.
CAPABILITY_PROMPTS = [
    "The capital of France is",
    "In the year 1969, humans first",
    "The chemical formula for water is",
    "To solve a quadratic equation, you can use the",
    "The speed of light in a vacuum is approximately",
    "Machine learning models are trained by",
    "The mitochondria is often called the",
    "According to Newton's third law,",
    "The largest planet in our solar system is",
    "DNA stands for deoxyribonucleic",
    "Photosynthesis is the process by which",
    "The Pythagorean theorem states that",
]


# ---------------------------------------------------------------------------
# Find extraction results for a model
# ---------------------------------------------------------------------------

def load_concept_directions(extraction_dir: Path, concept: str) -> dict | None:
    """Load the per-layer concept directions from an extraction result."""
    caz_path = extraction_dir / f"caz_{concept}.json"
    if not caz_path.exists():
        return None
    with open(caz_path) as f:
        data = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Ablation measurement
# ---------------------------------------------------------------------------

def measure_separation_with_ablation(
    model,
    tokenizer,
    layers: list,
    layer_idx: int,
    direction: np.ndarray,
    pos_texts: list[str],
    neg_texts: list[str],
    device: str,
    batch_size: int,
) -> float:
    """Run extraction with ablation at a specific layer, return peak separation."""
    dtype = torch.bfloat16 if next(model.parameters()).dtype == torch.bfloat16 else torch.float32

    with DirectionalAblator(layers[layer_idx], direction, dtype=dtype):
        pos_acts = extract_layer_activations(
            model, tokenizer, pos_texts, device=device, batch_size=batch_size, pool="last"
        )
        neg_acts = extract_layer_activations(
            model, tokenizer, neg_texts, device=device, batch_size=batch_size, pool="last"
        )

    # Measure separation at the same layer where we ablated
    # (Skip embedding layer — index 0 in the extraction output is embedding)
    act_idx = layer_idx + 1  # +1 because extraction includes embedding at [0]
    if act_idx >= len(pos_acts):
        act_idx = len(pos_acts) - 1

    return compute_separation(pos_acts[act_idx], neg_acts[act_idx])


def measure_kl_with_ablation(
    model,
    tokenizer,
    layers: list,
    layer_idx: int,
    direction: np.ndarray,
    baseline_logits: list[torch.Tensor],
    prompts: list[str],
    device: str,
) -> float:
    """Measure mean KL divergence with ablation at a specific layer."""
    dtype = torch.bfloat16 if next(model.parameters()).dtype == torch.bfloat16 else torch.float32

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kl_values = []
    with DirectionalAblator(layers[layer_idx], direction, dtype=dtype):
        for i, prompt in enumerate(prompts):
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc)
            ablated_logits = out.logits[0, -1, :].cpu()
            kl = kl_divergence_from_logits(baseline_logits[i], ablated_logits)
            kl_values.append(kl)

    return float(np.mean(kl_values))


# ---------------------------------------------------------------------------
# Per-concept ablation sweep
# ---------------------------------------------------------------------------

def ablation_sweep(
    model,
    tokenizer,
    concept: str,
    extraction_data: dict,
    device: str,
    n_pairs: int,
    batch_size: int,
) -> dict:
    """Sweep ablation across all layers for one concept."""
    # Load concept pairs
    pairs = load_concept_pairs(concept, n=n_pairs or 200)
    pos_texts, neg_texts = texts_by_label(pairs)

    # Get transformer layers
    layers = get_transformer_layers(model)
    n_layers = len(layers)

    # Get per-layer concept directions from extraction
    layer_data = extraction_data["layer_data"]
    metrics_raw = layer_data["metrics"]

    # Baseline separation (no ablation) from extraction
    baseline_seps = [m["separation_fisher"] for m in metrics_raw]
    peak_layer = layer_data["peak_layer"]
    baseline_peak_sep = layer_data["peak_separation"]

    # Compute CAZ boundaries
    layer_metrics = [
        LayerMetrics(m["layer"], m["separation_fisher"], m["coherence"], m["velocity"])
        for m in metrics_raw
    ]
    try:
        boundary = find_caz_boundary(layer_metrics)
    except Exception:
        boundary = None

    # Compute baseline KL (no ablation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    baseline_logits = []
    for prompt in CAPABILITY_PROMPTS:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        baseline_logits.append(out.logits[0, -1, :].cpu())

    # Sweep all layers
    results_per_layer = []
    for layer_idx in range(n_layers):
        torch.cuda.empty_cache() if device == "cuda" else None
        # Get concept direction at this layer
        dom_vector = np.array(metrics_raw[layer_idx]["dom_vector"], dtype=np.float64)

        t0 = time.time()

        # Measure separation reduction
        ablated_sep = measure_separation_with_ablation(
            model, tokenizer, layers, layer_idx, dom_vector,
            pos_texts, neg_texts, device, batch_size,
        )
        sep_reduction = max(0.0, 1.0 - ablated_sep / baseline_seps[layer_idx]) if baseline_seps[layer_idx] > 0 else 0.0

        # Measure KL divergence
        kl_div = measure_kl_with_ablation(
            model, tokenizer, layers, layer_idx, dom_vector,
            baseline_logits, CAPABILITY_PROMPTS, device,
        )

        # Suppression-to-damage ratio
        ratio = sep_reduction / kl_div if kl_div > 1e-6 else 0.0

        elapsed = time.time() - t0

        results_per_layer.append({
            "layer": layer_idx,
            "depth_pct": round(100.0 * layer_idx / n_layers, 1),
            "baseline_separation": baseline_seps[layer_idx],
            "ablated_separation": ablated_sep,
            "separation_reduction": round(sep_reduction, 4),
            "kl_divergence": round(kl_div, 6),
            "suppression_damage_ratio": round(ratio, 4),
            "seconds": round(elapsed, 1),
        })

        log.info(
            "  L%d (%.0f%%) sep_red=%.3f kl=%.4f ratio=%.3f (%.1fs)",
            layer_idx, 100.0 * layer_idx / n_layers,
            sep_reduction, kl_div, ratio, elapsed,
        )

    # Find optimal layer (max ratio)
    ratios = [r["suppression_damage_ratio"] for r in results_per_layer]
    optimal_layer = int(np.argmax(ratios))

    return {
        "concept": concept,
        "model_id": extraction_data["model_id"],
        "n_layers": n_layers,
        "n_pairs": len(pairs),
        "caz_start": boundary.caz_start if boundary else None,
        "caz_peak": boundary.caz_peak if boundary else peak_layer,
        "caz_end": boundary.caz_end if boundary else None,
        "caz_width": boundary.caz_width if boundary else None,
        "optimal_ablation_layer": optimal_layer,
        "optimal_in_caz": (
            boundary.caz_start <= optimal_layer <= boundary.caz_end
            if boundary else None
        ),
        "layers": results_per_layer,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_model(model_id: str, concepts: list[str], args) -> None:
    dir_suffix = getattr(args, "model_dir_suffix", None) or ""
    if dir_suffix:
        from pathlib import Path
        model_slug = model_id.replace("/", "_").replace("-", "_")
        candidate = Path.home() / "rosetta_data" / "model_snapshots" / (model_slug + dir_suffix)
        extraction_dir = candidate if candidate.is_dir() else None
    else:
        extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.error("No extraction results found for %s — run extract.py first", model_id)
        return

    # Skip model load entirely when every concept already has a valid result
    if not getattr(args, "force", False):
        def _result_valid(c: str) -> bool:
            p = extraction_dir / f"ablation_{c}.json"
            if not p.exists():
                return False
            try:
                d = json.loads(p.read_text())
                return (d.get("model_id") == model_id
                        and d.get("concept") == c
                        and d.get("n_pairs", 0) >= args.n_pairs
                        and d.get("n_layers", 0) > 0)
            except (json.JSONDecodeError, OSError):
                return False

        todo = [c for c in concepts if not _result_valid(c)]
        if not todo:
            log.info("=== Ablation sweep: %s — all %d concepts done (n_pairs≥%d), skipping ===",
                     model_id, len(concepts), args.n_pairs)
            return
        if len(todo) < len(concepts):
            log.info("=== Ablation sweep: %s — %d/%d concepts remaining ===",
                     model_id, len(todo), len(concepts))
            concepts = todo

    log.info("=== Ablation sweep: %s ===", model_id)
    log.info("Using extraction results from: %s", extraction_dir)

    device = get_device(args.device)
    dtype = get_dtype(device)
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float32":
        dtype = torch.float32
    log_device_info(device, dtype)

    # Need CausalLM for logits (KL divergence measurement)
    model, tokenizer = load_causal_lm(model_id, device, dtype)
    log_vram("after model load")

    # Auto-reduce batch size if VRAM headroom is tight
    if device == "cuda":
        free_gb = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated(0)) / 2**30
        if free_gb < 4.0 and args.batch_size > 2:
            old_bs = args.batch_size
            args.batch_size = 2
            log.warning("Low VRAM headroom (%.1f GiB free) — reducing batch_size %d → %d",
                        free_gb, old_bs, args.batch_size)
        if free_gb < 2.0 and args.batch_size > 1:
            args.batch_size = 1
            log.warning("Very low VRAM headroom (%.1f GiB free) — reducing batch_size → 1", free_gb)

    t_start = time.time()

    for i, concept in enumerate(concepts):
        log.info("--- Concept %d/%d: %s ---", i + 1, len(concepts), concept)

        # Concept-level skip — validate model, concept, and pair counts
        out_file = extraction_dir / f"ablation_{concept}.json"
        if out_file.exists() and not getattr(args, "force", False):
            try:
                with open(out_file) as _f:
                    _existing = json.load(_f)
                _ok = (
                    _existing.get("model_id") == model_id
                    and _existing.get("concept") == concept
                    and _existing.get("n_pairs", 0) >= args.n_pairs
                    and _existing.get("n_layers", 0) > 0
                )
                if _ok:
                    log.info(
                        "  Already done (model=%s concept=%s n_pairs=%d n_layers=%d) — skipping",
                        model_id.split("/")[-1], concept,
                        _existing["n_pairs"], _existing["n_layers"],
                    )
                    continue
                else:
                    log.info(
                        "  Existing result invalid or stale (n_pairs=%s, n_layers=%s) — re-running",
                        _existing.get("n_pairs"), _existing.get("n_layers"),
                    )
            except (json.JSONDecodeError, OSError):
                log.warning("  Existing %s unreadable — re-running", out_file.name)

        extraction_data = load_concept_directions(extraction_dir, concept)
        if extraction_data is None:
            log.warning("No extraction data for %s, skipping", concept)
            continue

        # Validate n_pairs against what was extracted
        extracted_n = extraction_data.get("layer_data", {}).get("n_pairs",
                      extraction_data.get("n_pairs", None))
        if extracted_n is not None and args.n_pairs > extracted_n:
            log.warning(
                "  Requested n_pairs=%d > extracted n_pairs=%d — clamping to %d",
                args.n_pairs, extracted_n, extracted_n,
            )
            args.n_pairs = extracted_n

        result = ablation_sweep(
            model, tokenizer, concept, extraction_data,
            device=device,
            n_pairs=args.n_pairs,
            batch_size=args.batch_size,
        )

        # Save alongside extraction results (timestamped to preserve history)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = extraction_dir / f"ablation_{concept}_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        # Also write a latest symlink for easy access
        latest = extraction_dir / f"ablation_{concept}.json"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_path.name)

        opt = result["optimal_ablation_layer"]
        in_caz = result["optimal_in_caz"]
        caz_peak = result["caz_peak"]
        log.info(
            "  [%s] %s → optimal ablation L%d, CAZ peak L%d, in_caz=%s",
            model_id.split("/")[-1], concept, opt, caz_peak,
            in_caz if in_caz is not None else "?",
        )

    total_elapsed = time.time() - t_start
    release_model(model)

    if getattr(args, "clean_cache", False):
        purge_hf_cache(model_id)

    log.info("Done: %s  (%.1fs total)", model_id, total_elapsed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer-wise ablation sweep for CAZ Prediction 1",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Single HuggingFace model ID")
    group.add_argument("--all", action="store_true", help="Run all models with extraction results")
    group.add_argument("--p3-corpus", action="store_true", help="Paper 3 CAZ Validation: 26 base models")
    parser.add_argument("--concepts", nargs="+", default=None,
                        help="Concepts to ablate (default: all with extraction results)")
    parser.add_argument("--model-dir-suffix", type=str, default="",
                        help="Read extraction results from model directory with this suffix appended")
    parser.add_argument("--n-pairs", type=int, default=50,
                        help="Number of contrastive pairs for separation measurement (default: 50)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="auto")
    parser.add_argument("--clean-cache", action="store_true",
                        help="Delete model from HF cache after ablation")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if ablation_{concept}.json already exists")
    return parser.parse_args()


def main():
    args = parse_args()
    concepts = args.concepts or CONCEPTS

    if args.all:
        models = discover_all_models()
        log.info("Found %d models with extraction results", len(models))
    elif args.p3_corpus:
        models = P3_MODELS
    else:
        models = [args.model]

    for model_id in models:
        run_model(model_id, concepts, args)


if __name__ == "__main__":
    main()

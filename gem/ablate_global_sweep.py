"""
ablate_global_sweep.py — Fixed-direction global ablation sweep.

Answers the question: which layer, when ablated, maximally suppresses concept
separation at the FINAL output — and does that match the CAZ peak?

The existing ablate.py sweeps use a layer-specific dom_vector and measure
separation locally (one step downstream of the ablated layer). This confounds
two things: whether the direction is present at that layer, and whether removing
it has downstream consequences.

This script uses a FIXED reference direction (extracted at the CAZ peak layer,
where the concept is most strongly expressed) and ablates that direction at each
layer individually, measuring the final-layer separation after each ablation.

This lets us ask cleanly:
  - In MHA models: does ablating the reference direction have equal impact
    at every layer (confirming architecture-wide redundancy)?
  - In GQA models: does ablating early truly do nothing (downstream recovery),
    or was the existing null result an artifact of using the wrong direction?
  - In Gemma: same as GQA.

Usage
-----
    python src/ablate_global_sweep.py --model EleutherAI/pythia-410m
    python src/ablate_global_sweep.py --model Qwen/Qwen2.5-7B
    python src/ablate_global_sweep.py --all
    python src/ablate_global_sweep.py --all --ref-layer final  # use last layer direction
    python src/ablate_global_sweep.py --all --ref-layer caz    # use CAZ peak direction (default)

Results written to:
    results/<extraction_dir>/ablation_global_sweep_<concept>.json

Each file contains:
    {
      "model_id": ...,
      "concept": ...,
      "ref_layer": <layer index used for reference direction>,
      "ref_layer_mode": "caz" | "final",
      "caz_peak": ...,
      "n_layers": ...,
      "layers": [
        {
          "layer": L,
          "depth_pct": ...,
          "baseline_final_sep": ...,   # separation at final layer, no ablation
          "ablated_final_sep": ...,    # separation at final layer, ablating at L
          "global_sep_reduction": ..., # (baseline - ablated) / baseline, clamped 0
        }, ...
      ],
      "optimal_ablation_layer": ...,
      "optimal_global_sep_reduction": ...,
      "caz_global_sep_reduction": ...,   # reduction when ablating at CAZ peak
      "caz_is_near_optimal": ...,        # True if CAZ peak within 2 layers of optimal
    }
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
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.caz import compute_separation
from rosetta_tools.ablation import DirectionalAblator, get_transformer_layers
from rosetta_tools.dataset import load_pairs, texts_by_label

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_ROOT = Path("results")
DATA_ROOT = Path(__file__).parent.parent / "data"

CONCEPT_DATASETS: dict[str, str] = {
    "credibility":    "credibility_pairs.jsonl",
    "negation":       "negation_pairs.jsonl",
    "sentiment":      "sentiment_pairs.jsonl",
    "causation":      "causation_pairs.jsonl",
    "certainty":      "certainty_pairs.jsonl",
    "moral_valence":  "moral_valence_pairs.jsonl",
    "temporal_order": "temporal_order_pairs.jsonl",
}


# ---------------------------------------------------------------------------
# Model / extraction discovery
# ---------------------------------------------------------------------------

def find_extraction_dir(model_id: str) -> Path | None:
    candidates = []
    for d in sorted(RESULTS_ROOT.iterdir(), reverse=True):
        summary = d / "run_summary.json"
        if d.is_dir() and summary.exists():
            try:
                if json.loads(summary.read_text()).get("model_id") == model_id:
                    candidates.append(d)
            except (json.JSONDecodeError, KeyError):
                continue
    return candidates[0] if candidates else None


def load_concept_directions(extraction_dir: Path, concept: str) -> dict | None:
    caz_path = extraction_dir / f"caz_{concept}.json"
    if not caz_path.exists():
        return None
    return json.loads(caz_path.read_text())


def discover_models() -> list[str]:
    models = set()
    for d in RESULTS_ROOT.iterdir():
        s = d / "run_summary.json"
        if s.exists():
            try:
                models.add(json.loads(s.read_text()).get("model_id", ""))
            except Exception:
                pass
    return sorted(m for m in models if m)


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def measure_final_sep_with_ablation(
    model,
    tokenizer,
    layers: list,
    ablate_layer_idx: int,
    direction: np.ndarray,
    pos_texts: list[str],
    neg_texts: list[str],
    device: str,
    batch_size: int,
) -> float:
    """Ablate `direction` at `ablate_layer_idx`, return separation at the FINAL layer."""
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

    # Final layer is last index in extraction output
    return float(compute_separation(pos_acts[-1], neg_acts[-1]))


def measure_baseline_final_sep(
    model,
    tokenizer,
    pos_texts: list[str],
    neg_texts: list[str],
    device: str,
    batch_size: int,
) -> float:
    """Measure concept separation at the final layer with no ablation."""
    pos_acts = extract_layer_activations(
        model, tokenizer, pos_texts, device=device, batch_size=batch_size, pool="last"
    )
    neg_acts = extract_layer_activations(
        model, tokenizer, neg_texts, device=device, batch_size=batch_size, pool="last"
    )
    return float(compute_separation(pos_acts[-1], neg_acts[-1]))


# ---------------------------------------------------------------------------
# Per-concept sweep
# ---------------------------------------------------------------------------

def global_sweep(
    model,
    tokenizer,
    concept: str,
    extraction_data: dict,
    ref_layer_mode: str,
    device: str,
    n_pairs: int,
    batch_size: int,
) -> dict:
    """Run a fixed-direction global sweep for one concept."""
    dataset_path = DATA_ROOT / CONCEPT_DATASETS[concept]
    pairs = load_pairs(dataset_path)
    if n_pairs and len(pairs) > n_pairs:
        pairs = pairs[:n_pairs]
    pos_texts, neg_texts = texts_by_label(pairs)

    layers = get_transformer_layers(model)
    n_layers = len(layers)

    layer_data = extraction_data["layer_data"]
    metrics_raw = layer_data["metrics"]
    caz_peak = layer_data["peak_layer"]

    # Choose reference layer for the fixed direction
    if ref_layer_mode == "final":
        ref_layer = n_layers - 1
    else:  # "caz" (default)
        ref_layer = caz_peak

    # Clamp to valid range
    ref_layer = max(0, min(ref_layer, len(metrics_raw) - 1))
    ref_direction = np.array(metrics_raw[ref_layer]["dom_vector"], dtype=np.float64)
    ref_direction /= np.linalg.norm(ref_direction)

    log.info("  Concept: %s | n_layers=%d | caz_peak=L%d | ref_layer=L%d (%s)",
             concept, n_layers, caz_peak, ref_layer, ref_layer_mode)

    # Baseline (no ablation)
    baseline_sep = measure_baseline_final_sep(
        model, tokenizer, pos_texts, neg_texts, device, batch_size
    )
    log.info("  Baseline final-layer separation: %.4f", baseline_sep)

    # Sweep
    results_per_layer = []
    for layer_idx in range(n_layers):
        if device == "cuda":
            torch.cuda.empty_cache()
        t0 = time.time()
        ablated_sep = measure_final_sep_with_ablation(
            model, tokenizer, layers, layer_idx, ref_direction,
            pos_texts, neg_texts, device, batch_size,
        )
        reduction = max(0.0, (baseline_sep - ablated_sep) / baseline_sep) if baseline_sep > 0 else 0.0
        elapsed = time.time() - t0

        depth_pct = round(100 * layer_idx / max(n_layers - 1, 1), 1)
        results_per_layer.append({
            "layer":                 layer_idx,
            "depth_pct":             depth_pct,
            "baseline_final_sep":    round(baseline_sep, 4),
            "ablated_final_sep":     round(ablated_sep, 4),
            "global_sep_reduction":  round(reduction, 4),
        })
        log.info("    L%2d (%4.1f%%)  ablated_sep=%.4f  reduction=%.3f  (%.1fs)",
                 layer_idx, depth_pct, ablated_sep, reduction, elapsed)

    # Summary
    reductions = [r["global_sep_reduction"] for r in results_per_layer]
    opt_idx = int(np.argmax(reductions))
    opt_layer = results_per_layer[opt_idx]["layer"]
    opt_red = reductions[opt_idx]
    caz_red = next(r["global_sep_reduction"] for r in results_per_layer if r["layer"] == caz_peak)
    caz_is_near_opt = abs(opt_layer - caz_peak) <= 2

    log.info("  CAZ peak L%d: global_reduction=%.3f", caz_peak, caz_red)
    log.info("  Optimal L%d: global_reduction=%.3f  (caz_is_near_opt=%s)",
             opt_layer, opt_red, caz_is_near_opt)

    return {
        "model_id":                    extraction_data.get("model_id", ""),
        "concept":                     concept,
        "ref_layer":                   ref_layer,
        "ref_layer_mode":              ref_layer_mode,
        "caz_peak":                    caz_peak,
        "n_layers":                    n_layers,
        "baseline_final_sep":          round(baseline_sep, 4),
        "layers":                      results_per_layer,
        "optimal_ablation_layer":      opt_layer,
        "optimal_global_sep_reduction": round(opt_red, 4),
        "caz_global_sep_reduction":    round(caz_red, 4),
        "caz_is_near_optimal":         caz_is_near_opt,
    }


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(model_id: str, concepts: list[str], args) -> None:
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.error("No extraction results for %s — skipping", model_id)
        return

    pending = [c for c in concepts
               if not (extraction_dir / f"ablation_global_sweep_{c}.json").exists()
               or args.overwrite]
    if not pending:
        log.info("Already done: %s (use --overwrite to rerun)", model_id)
        return
    concepts = pending

    log.info("=== Global sweep: %s ===", model_id)
    device = get_device(args.device)
    dtype  = get_dtype(args.dtype, device)
    log_device_info(device, dtype)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
        model.eval()
    except Exception as e:
        log.error("Failed to load %s: %s", model_id, e)
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stats = vram_stats(device)
    if stats:
        log_vram(device)

    t_model_start = time.time()

    for concept in concepts:
        out_path = extraction_dir / f"ablation_global_sweep_{concept}.json"
        if out_path.exists() and not args.overwrite:
            log.info("  Skipping %s (already done)", concept)
            continue

        extraction_data = load_concept_directions(extraction_dir, concept)
        if extraction_data is None:
            log.warning("  No extraction data for %s, skipping", concept)
            continue

        extraction_data["model_id"] = model_id

        try:
            result = global_sweep(
                model, tokenizer, concept, extraction_data,
                ref_layer_mode=args.ref_layer,
                device=device,
                n_pairs=args.n_pairs,
                batch_size=args.batch_size,
            )
        except Exception as e:
            log.error("  Sweep failed for %s %s: %s", model_id, concept, e)
            continue

        out_path.write_text(json.dumps(result, indent=2))
        log.info("  Wrote %s", out_path)

    release_model(model)
    if not args.no_clean_cache:
        purge_hf_cache(model_id)

    log.info("Done: %s (%.1fs)", model_id, time.time() - t_model_start)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fixed-direction global ablation sweep: which layer matters for final output?"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all", action="store_true")

    parser.add_argument("--concepts", nargs="+", default=None)
    parser.add_argument(
        "--ref-layer", choices=["caz", "final"], default="caz",
        help="Layer to extract the reference direction from: 'caz' (CAZ peak, default) or 'final' (last layer)",
    )
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="auto")
    parser.add_argument("--overwrite", action="store_true", help="Rerun even if output already exists")
    parser.add_argument("--no-clean-cache", action="store_true")

    args = parser.parse_args()
    concepts = args.concepts or list(CONCEPT_DATASETS.keys())

    if args.all:
        models = discover_models()
        log.info("Found %d models", len(models))
    else:
        models = [args.model]

    for model_id in models:
        run_model(model_id, concepts, args)


if __name__ == "__main__":
    main()

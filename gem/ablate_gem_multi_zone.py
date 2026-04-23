"""
ablate_gem_multi_zone.py
========================
Multi-zone simultaneous ablation: do all CAZ zones together approach
complete concept removal?

The standard GEM protocol ablates only upstream (ablation_target) nodes,
leaving downstream nodes intact (superposition avoidance).  For multimodal
concepts with multiple CAZ nodes, this leaves part of the concept in place.

This script tests: what happens when you ablate ALL nodes simultaneously?

Three conditions:
  A. upstream_only  : current GEM default (ablation_targets only)
  B. all_zones      : all nodes ablated simultaneously
  C. cascade_block  : ablate upstream node(s), then check if downstream
                      nodes' signal is suppressed passively (no direct
                      ablation of downstream nodes)

For single-node concepts, A == B.  The interesting cases are multi-node
concepts (most multimodal concepts, typically 2-3 nodes per model).

Prediction: B > A in final separation reduction.  The gap between B and A
quantifies how much residual concept signal propagates through downstream
nodes not targeted by the upstream-only protocol.

Usage
-----
    cd ~/caz_scaling
    python src/ablate_gem_multi_zone.py --model EleutherAI/pythia-1.4b
    python src/ablate_gem_multi_zone.py --all

Written: 2026-04-21 UTC
"""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from rosetta_tools.ablation import DirectionalAblator, get_transformer_layers
from rosetta_tools.caz import compute_separation
from rosetta_tools.dataset import load_concept_pairs, texts_by_label
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.gem import discover_concepts, discover_base_models, find_extraction_dir
from rosetta_tools.gpu_utils import (
    get_device, get_dtype, log_device_info,
    release_model, purge_hf_cache, NumpyJSONEncoder, load_causal_lm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUT_DIR = Path.home() / "rosetta_data" / "results" / "gem_multi_zone"

N_PAIRS = 50
BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def measure_final_sep(model, tokenizer, pos_texts, neg_texts, device,
                       ablate_spec: list[tuple[int, np.ndarray]]) -> float:
    """
    ablate_spec: list of (layer_idx, direction) to ablate simultaneously.
    Empty list = baseline.
    """
    dtype = next(model.parameters()).dtype

    with ExitStack() as stack:
        for layer_idx, direction in ablate_spec:
            layers = get_transformer_layers(model)
            dir_t = torch.tensor(direction, dtype=dtype,
                                 device=str(next(model.parameters()).device))
            dir_t = dir_t / dir_t.norm()
            stack.enter_context(DirectionalAblator(layers[layer_idx], dir_t, dtype=dtype))
        pos_acts = extract_layer_activations(
            model, tokenizer, pos_texts, device=device,
            batch_size=BATCH_SIZE, pool="last")
        neg_acts = extract_layer_activations(
            model, tokenizer, neg_texts, device=device,
            batch_size=BATCH_SIZE, pool="last")
    return float(compute_separation(pos_acts[-1], neg_acts[-1]))


# ---------------------------------------------------------------------------
# Per-concept
# ---------------------------------------------------------------------------

def run_concept(model, tokenizer, concept: str, extraction_dir: Path,
                device: str) -> dict | None:
    gem_path = extraction_dir / f"gem_{concept}.json"
    abl_path = extraction_dir / f"ablation_gem_{concept}.json"
    if not gem_path.exists():
        return None

    gem = json.loads(gem_path.read_text())
    n_nodes = gem.get("n_nodes", 0)
    if n_nodes == 0:
        return None

    pairs = load_concept_pairs(concept, n=N_PAIRS)
    pos_texts, neg_texts = texts_by_label(pairs)

    # Load existing upstream-only result if available
    existing_upstream_reduction = None
    if abl_path.exists():
        abl = json.loads(abl_path.read_text())
        existing_upstream_reduction = abl.get("handoff", {}).get(
            "final_sep_reduction")

    # Build ablation specs for each node (using handoff_layer + settled_direction)
    node_specs = []
    abl_data = json.loads(abl_path.read_text()) if abl_path.exists() else {}
    abl_layers = abl_data.get("handoff", {}).get("ablation_layers", [])
    handoff_width = len(abl_layers) if abl_layers else 3

    for i, node in enumerate(gem["nodes"]):
        handoff_layer = node["handoff_layer"]
        settled = np.array(node["settled_direction"], dtype=np.float64)
        settled /= np.linalg.norm(settled)
        # window layers around handoff
        window = list(range(
            max(0, handoff_layer - handoff_width // 2),
            handoff_layer + (handoff_width - handoff_width // 2),
        ))
        node_specs.append({
            "node_idx": i,
            "is_target": i in gem.get("ablation_targets", []),
            "handoff_layer": handoff_layer,
            "window": window,
            "direction": settled,
        })

    layers_obj = get_transformer_layers(model)
    n_layers = len(layers_obj)

    # Baseline
    baseline_sep = measure_final_sep(model, tokenizer, pos_texts, neg_texts,
                                      device, [])

    def reduction(ablated_sep: float) -> float:
        return max(0.0, (baseline_sep - ablated_sep) / baseline_sep) if baseline_sep > 0 else 0.0

    # Condition A: upstream_only (current default)
    upstream_spec = []
    for ns in node_specs:
        if ns["is_target"]:
            for li in ns["window"]:
                if 0 <= li < n_layers:
                    upstream_spec.append((li, ns["direction"]))
    sep_upstream = measure_final_sep(
        model, tokenizer, pos_texts, neg_texts, device, upstream_spec)
    reduction_upstream = reduction(sep_upstream)

    # Condition B: all_zones
    all_spec = []
    for ns in node_specs:
        for li in ns["window"]:
            if 0 <= li < n_layers:
                all_spec.append((li, ns["direction"]))
    sep_all = measure_final_sep(
        model, tokenizer, pos_texts, neg_texts, device, all_spec)
    reduction_all = reduction(sep_all)

    # Condition C: cascade_block (upstream only, measure downstream node seps)
    # Measure Fisher sep at each downstream node's peak after upstream ablation
    downstream_seps = {}
    downstream_nodes = [ns for ns in node_specs if not ns["is_target"]]
    if downstream_nodes:
        from rosetta_tools.caz import compute_separation as cs
        for ns in downstream_nodes:
            # Extract at node's own caz_peak after upstream ablation
            node_peak = gem["nodes"][ns["node_idx"]]["caz_peak"]
            with ExitStack() as stack:
                dtype = next(model.parameters()).dtype
                dev_str = str(next(model.parameters()).device)
                for li, d in upstream_spec:
                    layers_list = get_transformer_layers(model)
                    dir_t = torch.tensor(d, dtype=dtype, device=dev_str)
                    dir_t = dir_t / dir_t.norm()
                    stack.enter_context(
                        DirectionalAblator(layers_list[li], dir_t, dtype=dtype))
                pos_acts = extract_layer_activations(
                    model, tokenizer, pos_texts, device=device,
                    batch_size=BATCH_SIZE, pool="last")
                neg_acts = extract_layer_activations(
                    model, tokenizer, neg_texts, device=device,
                    batch_size=BATCH_SIZE, pool="last")
            downstream_seps[f"node_{ns['node_idx']}_peak_{node_peak}"] = round(
                float(cs(pos_acts[node_peak], neg_acts[node_peak])), 4)

    log.info("  %s: baseline=%.3f  upstream=%.3f (%.3f red)  all=%.3f (%.3f red)  delta=%.3fpp",
             concept, baseline_sep, sep_upstream, reduction_upstream,
             sep_all, reduction_all,
             100 * (reduction_all - reduction_upstream))

    return {
        "concept": concept,
        "n_nodes": n_nodes,
        "n_target_nodes": len([ns for ns in node_specs if ns["is_target"]]),
        "baseline_sep": round(baseline_sep, 4),
        "upstream_only": {
            "sep": round(sep_upstream, 4),
            "reduction": round(reduction_upstream, 4),
            "existing_reduction": existing_upstream_reduction,
        },
        "all_zones": {
            "sep": round(sep_all, 4),
            "reduction": round(reduction_all, 4),
        },
        "all_zones_delta_pp": round(100 * (reduction_all - reduction_upstream), 2),
        "all_zones_better": bool(reduction_all > reduction_upstream),
        "downstream_sep_after_upstream_ablation": downstream_seps,
    }


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(model_id: str, args) -> None:
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.warning("No extraction dir for %s", model_id)
        return

    out_path = OUT_DIR / f"{extraction_dir.name}_multi_zone.json"
    if out_path.exists() and not args.overwrite:
        log.info("Already done: %s", model_id)
        return

    log.info("=== %s ===", model_id)
    device = get_device(args.device)
    dtype = get_dtype(args.dtype, device)
    log_device_info(device, dtype)

    try:
        model, tokenizer = load_causal_lm(model_id, device, dtype)
    except Exception as e:
        log.error("Failed to load %s: %s", model_id, e)
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for concept in discover_concepts(extraction_dir):
        r = run_concept(model, tokenizer, concept, extraction_dir, device)
        if r:
            r["model_id"] = model_id
            results.append(r)

    if results:
        multi_node = [r for r in results if r["n_nodes"] > 1]
        n_better = sum(1 for r in results if r["all_zones_better"])
        delta_pp = [r["all_zones_delta_pp"] for r in results]
        log.info("  %s: all_better=%d/%d  mean_delta=%.2fpp  "
                 "multi-node concepts=%d/%d",
                 model_id.split("/")[-1], n_better, len(results),
                 np.mean(delta_pp), len(multi_node), len(results))

    out_path.write_text(json.dumps(
        {"model_id": model_id, "results": results},
        cls=NumpyJSONEncoder, indent=2))
    log.info("Wrote %s", out_path)

    release_model(model)
    if not args.no_clean_cache:
        purge_hf_cache(model_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-clean-cache", action="store_true")
    args = parser.parse_args()

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    models = discover_base_models() if args.all else [args.model]
    log.info("Running on %d models", len(models))
    for model_id in models:
        run_model(model_id, args)


if __name__ == "__main__":
    main()

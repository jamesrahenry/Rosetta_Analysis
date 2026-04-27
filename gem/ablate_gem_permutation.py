"""
ablate_gem_permutation.py — Full permutation ablation across all GEM node subsets.

Tests every non-empty subset of GEM nodes per concept, measuring separation at
each node's own handoff layer AND the final layer.

Motivation
----------
The standard ablate_gem.py picks ablation_targets[0], which for models without
multimodal interaction data is the shallowest/weakest node.  This produces near-
zero final-layer reduction because the strong deep node is untouched.

Instead of choosing a "best" node, enumerate ALL 2^N - 1 non-empty subsets.
This answers three questions simultaneously:

  1. Which single node drives most of the final-layer representation?
     → Compare final_reduction across single-node subsets.

  2. Are nodes redundant or synergistic?
     → synergy = reduction(all) - max(reduction(any_single))
     Positive = combining nodes hits harder than any single node.
     Negative = nodes are redundant (the strong one is sufficient).

  3. Does ablating an upstream (shallow) node disrupt downstream assembly?
     → cross_reduction: reduction at node[j]'s handoff layer when
        only upstream nodes are ablated (j not in subset).
     If >0, the shallow CAZ feeds the deep CAZ — real cascade, not independence.

Design
------
- Width=1 at each node's handoff_layer (settled_direction).
  Maximally surgical. Width sensitivity can be added later.
- Measure at ALL node handoff layers + final layer for each subset.
  Captures both local effects (at ablated node's layer) and cross-effects.
- N_PAIRS=50, batch_size=4 — consistent with other ablation experiments.

Usage
-----
    python rosetta_analysis/gem/ablate_gem_permutation.py \\
        --model EleutherAI/pythia-6.9b
    python rosetta_analysis/gem/ablate_gem_permutation.py --all

Output
------
    ~/rosetta_data/results/gem_permutation/<model_slug>_permutation.json
    ~/rosetta_data/results/gem_permutation/gem_permutation_summary.txt  (aggregate)

Written: 2026-04-27 UTC
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
from rosetta_tools.models import vram_gb as _registry_vram
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

OUT_DIR = Path.home() / "rosetta_data" / "results" / "gem_permutation"

N_PAIRS = 50
BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def measure_layers_sep(
    model,
    tokenizer,
    layers,
    ablate_layer_list: list[int],
    ablate_dir_list: list[np.ndarray],
    pos_texts: list[str],
    neg_texts: list[str],
    measure_layer_list: list[int],
    device: str,
) -> dict[int, float]:
    """Ablate at ablate_layer_list, measure separation at measure_layer_list.

    Returns {layer_idx: separation_value}.
    Embedding is at index 0 of extract_layer_activations output; layer i is at i+1.
    """
    model_dtype = next(model.parameters()).dtype

    with ExitStack() as stack:
        for li, direction in zip(ablate_layer_list, ablate_dir_list):
            # DirectionalAblator normalizes internally and moves to device in hook
            stack.enter_context(DirectionalAblator(layers[li], direction, dtype=model_dtype))

        pos_acts = extract_layer_activations(
            model, tokenizer, pos_texts,
            device=device, batch_size=BATCH_SIZE, pool="last",
        )
        neg_acts = extract_layer_activations(
            model, tokenizer, neg_texts,
            device=device, batch_size=BATCH_SIZE, pool="last",
        )

    result = {}
    for li in measure_layer_list:
        act_idx = li + 1  # +1 because index 0 is the embedding layer
        if act_idx >= len(pos_acts):
            act_idx = len(pos_acts) - 1
        result[li] = float(compute_separation(pos_acts[act_idx], neg_acts[act_idx]))
    return result


# ---------------------------------------------------------------------------
# Per-concept runner
# ---------------------------------------------------------------------------

def run_concept(
    model,
    tokenizer,
    concept: str,
    extraction_dir: Path,
    device: str,
) -> dict | None:
    gem_path = extraction_dir / f"gem_{concept}.json"
    if not gem_path.exists():
        return None

    gem = json.loads(gem_path.read_text())
    nodes = gem.get("nodes", [])
    n_nodes = len(nodes)
    if n_nodes == 0:
        log.warning("  %s: empty GEM, skipping", concept)
        return None

    layers = get_transformer_layers(model)
    n_total = nodes[0]["n_layers_total"]
    final_layer = n_total - 1

    # Measurement points: every node's handoff layer + final layer
    all_hl = [node["handoff_layer"] for node in nodes]
    measure_layers = sorted(set(all_hl + [final_layer]))

    # Load texts
    pairs = load_concept_pairs(concept, n=N_PAIRS)
    pos_texts, neg_texts = texts_by_label(pairs)
    pos_texts = [t for t in pos_texts if t and t.strip()]
    neg_texts = [t for t in neg_texts if t and t.strip()]
    if not pos_texts or not neg_texts:
        log.warning("  %s: no usable texts", concept)
        return None

    # Baseline (no ablation)
    baseline = measure_layers_sep(
        model, tokenizer, layers,
        ablate_layer_list=[], ablate_dir_list=[],
        pos_texts=pos_texts, neg_texts=neg_texts,
        measure_layer_list=measure_layers, device=device,
    )

    # Enumerate all non-empty subsets via bitmask
    subset_results = {}
    n_subsets = (1 << n_nodes) - 1

    for mask in range(1, n_subsets + 1):
        subset_indices = [i for i in range(n_nodes) if (mask >> i) & 1]
        key = ",".join(str(i) for i in subset_indices)

        ablate_layers = [nodes[i]["handoff_layer"] for i in subset_indices]
        ablate_dirs = []
        for i in subset_indices:
            d = np.array(nodes[i]["settled_direction"], dtype=np.float64)
            norm = np.linalg.norm(d)
            if norm > 1e-12:
                d = d / norm
            ablate_dirs.append(d)

        ablated = measure_layers_sep(
            model, tokenizer, layers,
            ablate_layer_list=ablate_layers, ablate_dir_list=ablate_dirs,
            pos_texts=pos_texts, neg_texts=neg_texts,
            measure_layer_list=measure_layers, device=device,
        )

        # Reduction = (baseline - ablated) / baseline, clamped to [0, 1]
        reduction = {}
        for li in measure_layers:
            bl = baseline[li]
            ab = ablated[li]
            reduction[li] = round(max(0.0, (bl - ab) / bl) if bl > 0 else 0.0, 4)

        subset_results[key] = {
            "nodes": subset_indices,
            "ablation_layers": ablate_layers,
            "sep": {str(li): round(ablated[li], 4) for li in measure_layers},
            "reduction": {str(li): reduction[li] for li in measure_layers},
            "final_reduction": reduction[final_layer],
        }

        # Progress log for multi-subset concepts
        if n_nodes > 1:
            node_labels = "+".join(
                f"n{i}(HL={nodes[i]['handoff_layer']})" for i in subset_indices
            )
            log.info("    subset {%s}: final_red=%.3f", node_labels,
                     reduction[final_layer])

    # Derive synergy metrics
    single_finals = {}
    for i in range(n_nodes):
        key = str(i)
        if key in subset_results:
            single_finals[i] = subset_results[key]["final_reduction"]

    max_single = max(single_finals.values()) if single_finals else 0.0
    best_single_node = max(single_finals, key=single_finals.get) if single_finals else -1

    full_key = ",".join(str(i) for i in range(n_nodes))
    full_final = subset_results[full_key]["final_reduction"] if full_key in subset_results else None
    synergy = round(full_final - max_single, 4) if full_final is not None else None

    # Cross-disruption: does ablating shallow nodes reduce sep at deep HL?
    # Compute for the subset of all shallow nodes (all except the deepest)
    cross_disruption = None
    if n_nodes >= 2:
        # Ablate all nodes EXCEPT the deepest; measure at deepest node's HL
        deepest_i = max(range(n_nodes), key=lambda i: nodes[i]["handoff_layer"])
        shallow_indices = [i for i in range(n_nodes) if i != deepest_i]
        shallow_key = ",".join(str(i) for i in shallow_indices)
        deepest_hl = nodes[deepest_i]["handoff_layer"]
        if shallow_key in subset_results:
            cross_disruption = subset_results[shallow_key]["reduction"].get(str(deepest_hl), 0.0)

    node_summary = [
        {
            "idx": i,
            "handoff_layer": nodes[i]["handoff_layer"],
            "depth_pct": round(100 * nodes[i]["handoff_layer"] / n_total, 1),
            "caz_score": round(nodes[i]["caz_score"], 4),
            "single_final_reduction": single_finals.get(i, None),
        }
        for i in range(n_nodes)
    ]

    log.info(
        "  %s | %d nodes | best_single=%s(%.3f) | synergy=%s | cross_disrupt=%s",
        concept, n_nodes,
        f"n{best_single_node}" if best_single_node >= 0 else "-",
        max_single,
        f"{synergy:+.3f}" if synergy is not None else "n/a",
        f"{cross_disruption:.3f}" if cross_disruption is not None else "n/a",
    )

    return {
        "concept": concept,
        "n_nodes": n_nodes,
        "n_layers": n_total,
        "final_layer": final_layer,
        "measure_layers": measure_layers,
        "nodes": node_summary,
        "baseline": {str(li): round(baseline[li], 4) for li in measure_layers},
        "subsets": subset_results,
        "derived": {
            "best_single_node": best_single_node,
            "best_single_final_reduction": round(max_single, 4),
            "full_subset_final_reduction": round(full_final, 4) if full_final is not None else None,
            "synergy_over_best_single": synergy,
            "cross_disruption_at_deepest_hl": cross_disruption,
        },
    }


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(model_id: str, args) -> None:
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.warning("No extraction dir for %s", model_id)
        return

    out_path = OUT_DIR / f"{extraction_dir.name}_permutation.json"
    if out_path.exists() and not args.overwrite:
        log.info("Already done: %s (use --overwrite to redo)", model_id)
        return

    log.info("=== Permutation ablation: %s ===", model_id)
    device = get_device(args.device)
    dtype = get_dtype(args.dtype, device)
    log_device_info(device, dtype)

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    model_vram = _registry_vram(model_id)
    device_map = "auto" if (model_vram > 20.0 and n_gpus > 1) else None
    if device_map:
        log.info("Large model (%.0f GB bf16): device_map='auto' across %d GPUs",
                 model_vram, n_gpus)

    try:
        model, tokenizer = load_causal_lm(model_id, device, dtype, device_map=device_map)
    except Exception as e:
        log.error("Failed to load %s: %s", model_id, e)
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    concepts = args.concepts or discover_concepts(extraction_dir, source="gem")
    if not concepts:
        log.warning("No GEM files found for %s", model_id)
        release_model(model)
        return

    log.info("Running %d concepts", len(concepts))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for concept in concepts:
        log.info("--- %s ---", concept)
        try:
            r = run_concept(model, tokenizer, concept, extraction_dir, device)
            if r:
                r["model_id"] = model_id
                results.append(r)
        except Exception as e:
            log.error("  %s: FAILED — %s", concept, e, exc_info=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path.write_text(json.dumps(
        {"model_id": model_id, "results": results},
        cls=NumpyJSONEncoder, indent=2,
    ))
    log.info("Wrote %s (%d concepts)", out_path, len(results))

    release_model(model)
    if not args.no_clean_cache:
        purge_hf_cache(model_id)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate(out_dir: Path) -> None:
    records = []
    for f in sorted(out_dir.glob("*_permutation.json")):
        if f.name == "gem_permutation_summary.json":
            continue
        raw = json.loads(f.read_text())
        batch = raw if isinstance(raw, list) else raw.get("results", [])
        for r in batch:
            if isinstance(r, dict) and r.get("concept"):
                records.append(r)

    # Deduplicate
    seen, deduped = set(), []
    for r in records:
        key = (r.get("model_id", ""), r["concept"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    records = deduped

    if not records:
        log.warning("No results to aggregate")
        return

    # Compute aggregate stats
    n_total = len(records)
    best_singles = [r["derived"]["best_single_final_reduction"] for r in records
                    if r["derived"].get("best_single_final_reduction") is not None]
    synergies = [r["derived"]["synergy_over_best_single"] for r in records
                 if r["derived"].get("synergy_over_best_single") is not None]
    cross_vals = [r["derived"]["cross_disruption_at_deepest_hl"] for r in records
                  if r["derived"].get("cross_disruption_at_deepest_hl") is not None]

    # Node depth vs effectiveness analysis
    # For each concept: is the deepest node the most effective?
    deepest_wins = 0
    n_multi = 0
    for r in records:
        if r["n_nodes"] < 2:
            continue
        n_multi += 1
        nodes = r["nodes"]
        deepest = max(range(len(nodes)), key=lambda i: nodes[i]["handoff_layer"])
        best_single = r["derived"]["best_single_node"]
        if best_single == deepest:
            deepest_wins += 1

    lines = [
        "GEM Permutation Ablation — Node Subset Analysis",
        f"Written: 2026-04-27 UTC",
        f"N records: {n_total}",
        "",
        "Single-node effectiveness (final layer):",
        f"  Mean best-single reduction: {np.mean(best_singles):.3f}"
        f"  (std={np.std(best_singles):.3f})" if best_singles else "  N/A",
        f"  Deepest node = most effective: {deepest_wins}/{n_multi}"
        f" ({100*deepest_wins/n_multi:.0f}%)" if n_multi > 0 else "",
        "",
        "Synergy (full subset vs best single, final layer):",
        f"  N with synergy data: {len(synergies)}",
        f"  Mean synergy: {np.mean(synergies):+.3f}  (>0 = additive benefit)" if synergies else "  N/A",
        f"  Synergy > 0.01: {sum(1 for s in synergies if s > 0.01)}/{len(synergies)}"
        if synergies else "",
        f"  Synergy < -0.01: {sum(1 for s in synergies if s < -0.01)}/{len(synergies)}"
        if synergies else "",
        "",
        "Cross-node disruption (ablate shallow, measure at deepest HL):",
        f"  N with disruption data: {len(cross_vals)}",
        f"  Mean cross-reduction: {np.mean(cross_vals):.3f}"
        f"  (>0 = shallow nodes feed deep assembly)" if cross_vals else "  N/A",
        f"  Cross-disruption > 0.05: {sum(1 for c in cross_vals if c > 0.05)}/{len(cross_vals)}"
        if cross_vals else "",
        "",
        f"{'Model':<28} {'Concept':<18} {'N':>2} {'BestSingle':>10} {'BestNode':>9}"
        f" {'Synergy':>8} {'CrossDisrp':>11}",
        "-" * 94,
    ]
    for r in sorted(records, key=lambda x: (x.get("model_id", ""), x["concept"])):
        d = r["derived"]
        model_short = r.get("model_id", "").split("/")[-1]
        lines.append(
            f"{model_short:<28} {r['concept']:<18} {r['n_nodes']:>2}"
            f" {d.get('best_single_final_reduction', 0):>10.3f}"
            f" n{d.get('best_single_node', -1):>7}"
            f" {d.get('synergy_over_best_single') or 0:>+8.3f}"
            f" {d.get('cross_disruption_at_deepest_hl') or 0:>11.3f}"
        )

    text = "\n".join(lines)
    print(text)
    (out_dir / "gem_permutation_summary.txt").write_text(text)
    (out_dir / "gem_permutation_summary.json").write_text(
        json.dumps(records, indent=2, cls=NumpyJSONEncoder)
    )
    log.info("Aggregate written to %s", out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global BATCH_SIZE
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Single model HF ID")
    group.add_argument("--all", action="store_true",
                       help="Run all base models with GEM data")
    group.add_argument("--aggregate-only", action="store_true",
                       help="Skip model runs, just re-aggregate existing results")
    parser.add_argument("--concepts", nargs="+", default=None,
                        help="Subset of concepts (default: all from GEM files)")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="auto")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-clean-cache", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_SIZE = args.batch_size

    if not args.aggregate_only:
        models = discover_base_models() if args.all else [args.model]
        log.info("Running permutation ablation on %d model(s)", len(models))
        for model_id in models:
            run_model(model_id, args)

    aggregate(OUT_DIR)


if __name__ == "__main__":
    main()

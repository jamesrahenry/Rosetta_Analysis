"""
routing_held_sweep.py — Per-input routing survival sweep across all models.

GPU required.  Implements CIA's DeepModule._compute_routing logic adapted
for Rosetta's GEM framework.

For each (model, concept, node): load the model, extract per-input
activations at the CAZ peak layer and the final layer, compute the cosine
similarity against the settled probe direction, and check whether the
final-layer cosine is within 95% of the peak-layer cosine.

This is the per-input complement to analyze_routing_survival.py's
population-level baseline_routing_ratio.  It answers a sharper question:
on how many individual inputs does the assembled concept signal actually
survive to the output?

Key research questions
----------------------
1. Which concepts route reliably (>90% of inputs) vs. unreliably (<50%)?
2. Do shallow nodes route less reliably than deep nodes?
3. [Future] Do instruct/RLHF models route differently from base models?
   ("RLHF preserves geometry, destroys routing" hypothesis.)

Usage
-----
    python rosetta_analysis/gem/routing_held_sweep.py \\
        --model EleutherAI/pythia-6.9b
    python rosetta_analysis/gem/routing_held_sweep.py --all

Output
------
    ~/rosetta_data/results/routing_held_sweep/<model_slug>_routing_held.json
    ~/rosetta_data/results/routing_held_sweep/routing_held_summary.txt  (aggregate)

Written: 2026-04-28 UTC
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rosetta_tools.dataset import load_concept_pairs, texts_by_label
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.gem import (
    discover_base_models, find_extraction_dir, load_gem,
    routing_held_ratio, stage_detail,
)
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

OUT_DIR = Path.home() / "rosetta_data" / "results" / "routing_held_sweep"

N_PAIRS = 50
BATCH_SIZE = 4
DEGRADATION_FACTOR = 0.95


def compute_cosine_per_layer(
    model,
    tokenizer,
    texts: list[str],
    direction: np.ndarray,
    measure_layers: list[int],
    device: str,
) -> dict[int, list[float]]:
    """Extract per-input cosine similarity vs direction at each measure layer.

    Returns {layer_idx: [cosine_per_input]}.
    """
    dir_t = torch.tensor(direction, dtype=torch.float32)
    dir_t = F.normalize(dir_t.unsqueeze(0), dim=-1).squeeze(0)

    all_acts = extract_layer_activations(
        model, tokenizer, texts,
        device=device, batch_size=BATCH_SIZE, pool="last",
    )

    results: dict[int, list[float]] = {li: [] for li in measure_layers}
    for li in measure_layers:
        act_idx = li + 1  # embedding at [0]
        if act_idx >= len(all_acts):
            act_idx = len(all_acts) - 1
        acts = torch.tensor(all_acts[act_idx], dtype=torch.float32)
        # acts: [n_texts, hidden_dim]
        acts_norm = F.normalize(acts, dim=-1)
        dir_dev = dir_t.to(acts.device)
        cosines = torch.mv(acts_norm, dir_dev).tolist()
        results[li] = [float(c) for c in cosines]
    return results


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
    gem = load_gem(gem_path)
    if gem.n_nodes == 0:
        return None

    n_layers = gem.nodes[0].n_layers_total
    final_layer = n_layers - 1

    pairs = load_concept_pairs(concept, n=N_PAIRS)
    pos_texts, neg_texts = texts_by_label(pairs)
    pos_texts = [t for t in pos_texts if t and t.strip()]
    neg_texts = [t for t in neg_texts if t and t.strip()]
    if not pos_texts:
        return None

    node_results = []
    routing_held_per_node: dict[int, bool] = {}

    for i, node in enumerate(gem.nodes):
        direction = np.array(node.settled_direction, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction = direction / norm

        measure_layers = sorted(set([node.caz_peak, final_layer]))

        pos_cosines = compute_cosine_per_layer(
            model, tokenizer, pos_texts, direction, measure_layers, device,
        )
        # Use positive class only for routing check (concept should be present)
        peak_cosines = pos_cosines[node.caz_peak]
        final_cosines = pos_cosines[final_layer]

        # Per-input routing_held check (CIA formula)
        held_per_input = [
            f >= p * DEGRADATION_FACTOR
            for p, f in zip(peak_cosines, final_cosines)
        ]
        routing_rate = float(np.mean(held_per_input))
        held = routing_rate >= 0.5  # majority of inputs route successfully

        # Population-level ratio (peak vs final mean cosine)
        mean_peak = float(np.mean([abs(c) for c in peak_cosines]))
        mean_final = float(np.mean([abs(c) for c in final_cosines]))
        ratio = mean_final / mean_peak if mean_peak > 0 else 0.0

        routing_held_per_node[i] = held

        node_results.append({
            "node_idx": i,
            "caz_peak": node.caz_peak,
            "handoff_layer": node.handoff_layer,
            "depth_pct": node.depth_pct,
            "caz_score": round(node.caz_score, 4),
            "mean_peak_cosine": round(mean_peak, 4),
            "mean_final_cosine": round(mean_final, 4),
            "routing_survival_ratio": round(ratio, 4),
            "routing_held_rate": round(routing_rate, 4),
            "routing_held": held,
        })

        log.info(
            "    node %d (HL=%d, %.0f%%, score=%.3f): "
            "peak_cos=%.3f final_cos=%.3f ratio=%.3f held_rate=%.2f %s",
            i, node.handoff_layer, node.depth_pct, node.caz_score,
            mean_peak, mean_final, ratio, routing_rate,
            "ROUTED" if held else "LOST",
        )

    detail = stage_detail(gem.nodes, n_layers, routing_held_per_node)
    log.info("  %s: %s", concept, detail)

    return {
        "concept": concept,
        "n_nodes": gem.n_nodes,
        "n_layers": n_layers,
        "nodes": node_results,
        "stage_detail": detail,
        "n_routed": sum(1 for n in node_results if n["routing_held"]),
        "n_lost": sum(1 for n in node_results if not n["routing_held"]),
    }


def run_model(model_id: str, args) -> None:
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.warning("No extraction dir for %s", model_id)
        return

    out_path = OUT_DIR / f"{extraction_dir.name}_routing_held.json"
    if out_path.exists() and not args.overwrite:
        log.info("Already done: %s", model_id)
        return

    log.info("=== Routing held sweep: %s ===", model_id)
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gem_paths = sorted(extraction_dir.glob("gem_*.json"))
    concepts = args.concepts or [p.stem[4:] for p in gem_paths]

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

    n_routed = sum(r["n_routed"] for r in results)
    n_lost = sum(r["n_lost"] for r in results)
    total_nodes = n_routed + n_lost
    log.info(
        "  %s: %d/%d nodes route to final layer (%.0f%%)",
        model_id.split("/")[-1], n_routed, total_nodes,
        100 * n_routed / max(total_nodes, 1),
    )

    out_path.write_text(json.dumps(
        {"model_id": model_id, "results": results},
        cls=NumpyJSONEncoder, indent=2,
    ))
    log.info("Wrote %s", out_path)

    release_model(model)
    if not args.no_clean_cache:
        purge_hf_cache(model_id)


def aggregate(out_dir: Path) -> None:
    records = []
    for f in sorted(out_dir.glob("*_routing_held.json")):
        if "summary" in f.name:
            continue
        raw = json.loads(f.read_text())
        for r in raw.get("results", []):
            for node in r.get("nodes", []):
                records.append({
                    "model_id": r["model_id"],
                    "concept": r["concept"],
                    "n_nodes": r["n_nodes"],
                    "stage_detail": r.get("stage_detail", ""),
                    **node,
                })

    if not records:
        log.warning("No records to aggregate")
        return

    ratios = [r["routing_survival_ratio"] for r in records]
    rates = [r["routing_held_rate"] for r in records]
    held = [r for r in records if r["routing_held"]]
    shallow = [r for r in records if r["depth_pct"] < 50]
    deep = [r for r in records if r["depth_pct"] >= 50]

    lines = [
        "GEM Routing Held Sweep — Per-Input Cosine Analysis",
        "Written: 2026-04-28 UTC",
        f"N node records: {len(records)}",
        "",
        f"Overall routing_held rate: {len(held)}/{len(records)}"
        f" ({100*len(held)/len(records):.1f}%)",
        f"Mean survival ratio (final/peak cosine): {np.mean(ratios):.3f}"
        f"  std={np.std(ratios):.3f}",
        f"Mean per-input routing_held rate: {np.mean(rates):.3f}",
        "",
        "By depth:",
        (f"  Shallow (<50%): N={len(shallow)}"
         f"  held={sum(1 for r in shallow if r['routing_held'])}/{len(shallow)}"
         f"  mean_ratio={np.mean([r['routing_survival_ratio'] for r in shallow]):.3f}")
        if shallow else "  Shallow: none",
        (f"  Deep    (>=50%): N={len(deep)}"
         f"  held={sum(1 for r in deep if r['routing_held'])}/{len(deep)}"
         f"  mean_ratio={np.mean([r['routing_survival_ratio'] for r in deep]):.3f}")
        if deep else "  Deep: none",
        "",
        f"{'Model':<28} {'Concept':<18} {'N':>2} {'Node':>4}"
        f" {'Depth':>6} {'Score':>6} {'PkCos':>6} {'FnCos':>6}"
        f" {'Ratio':>6} {'Rate':>6} {'Held':>5}",
        "-" * 103,
    ]
    for r in sorted(records, key=lambda x: (x["model_id"], x["concept"], x["node_idx"])):
        lines.append(
            f"{r['model_id'].split('/')[-1]:<28} {r['concept']:<18}"
            f" {r['n_nodes']:>2} {r['node_idx']:>4}"
            f" {r['depth_pct']:>5.1f}%"
            f" {r['caz_score']:>6.3f}"
            f" {r['mean_peak_cosine']:>6.3f}"
            f" {r['mean_final_cosine']:>6.3f}"
            f" {r['routing_survival_ratio']:>6.3f}"
            f" {r['routing_held_rate']:>6.3f}"
            f" {'Y' if r['routing_held'] else 'N':>5}"
        )

    text = "\n".join(lines)
    print(text)
    (out_dir / "routing_held_summary.txt").write_text(text)
    (out_dir / "routing_held.json").write_text(
        json.dumps(records, indent=2, cls=NumpyJSONEncoder)
    )
    log.info("Aggregate written to %s", out_dir)


def main() -> None:
    global BATCH_SIZE
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all", action="store_true")
    group.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--concepts", nargs="+", default=None)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="auto")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-clean-cache", action="store_true")
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        models = discover_base_models() if args.all else [args.model]
        log.info("Running routing_held sweep on %d model(s)", len(models))
        for model_id in models:
            run_model(model_id, args)

    aggregate(OUT_DIR)


if __name__ == "__main__":
    main()

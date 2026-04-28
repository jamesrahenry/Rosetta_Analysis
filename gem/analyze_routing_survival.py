"""
analyze_routing_survival.py — Compute routing survival from existing data.

No GPU needed.  Mines ablation_gem_*.json baseline separations and
permutation ablation results to answer:

  For each (model, concept, node): does the assembled signal at the CAZ
  peak layer survive passively to the final layer?

Two complementary metrics computed per node:

  1. baseline_routing_ratio  (from ablation_gem_*.json)
     = baseline_sep[final] / baseline_sep[node.caz_peak]
     Population-level: ratio of clean separation at final vs. peak layer.
     Values < 0.95 indicate signal attenuation.

  2. permutation_routing_held  (from gem_permutation/*.json, where available)
     = final_reduction[single node ablation] >= contribution_threshold (0.05)
     Intervention-based: does ablating this node move the final-layer needle?
     False = the node's signal doesn't reach the output regardless.

These two metrics are complementary:
  - baseline_routing_ratio measures overall signal preservation.
  - permutation_routing_held tests whether this node's SPECIFIC direction
    contributes to final-layer representation (not masked by other nodes).

For most models we have ablation_gem data; permutation data is currently
available for pythia-6.9b only.

Usage
-----
    python rosetta_analysis/gem/analyze_routing_survival.py
    python rosetta_analysis/gem/analyze_routing_survival.py --model EleutherAI/pythia-6.9b

Output
------
    ~/rosetta_data/results/routing_survival/routing_survival_summary.txt
    ~/rosetta_data/results/routing_survival/routing_survival.json

Written: 2026-04-28 UTC
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from rosetta_tools.gem import (
    discover_all_models, find_extraction_dir, load_gem,
    routing_held_ratio, routing_held_from_permutation, stage_detail,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUT_DIR = Path.home() / "rosetta_data" / "results" / "routing_survival"
PERM_DIR = Path.home() / "rosetta_data" / "results" / "gem_permutation"
CONTRIBUTION_THRESHOLD = 0.05


def process_model(model_id: str) -> list[dict]:
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        return []

    # Load permutation data if available
    perm_data: dict[str, dict] = {}
    perm_slug = extraction_dir.name + "_permutation.json"
    perm_path = PERM_DIR / perm_slug
    if perm_path.exists():
        raw = json.loads(perm_path.read_text())
        for r in raw.get("results", []):
            perm_data[r["concept"]] = r
        log.info("  Loaded permutation data for %d concepts", len(perm_data))

    records = []
    for gem_path in sorted(extraction_dir.glob("gem_*.json")):
        concept = gem_path.stem[4:]  # strip "gem_"

        # Load ablation_gem for baseline separations
        abl_path = extraction_dir / f"ablation_gem_{concept}.json"
        if not abl_path.exists():
            continue
        abl = json.loads(abl_path.read_text())
        handoff = abl.get("handoff", {})
        per_layer_raw = handoff.get("per_layer", {})
        # Keys are strings in JSON
        baseline_per_layer = {
            int(k): v["baseline_sep"]
            for k, v in per_layer_raw.items()
            if isinstance(v, dict) and "baseline_sep" in v
        }

        gem = load_gem(gem_path)
        if gem.n_nodes == 0:
            continue

        n_layers = gem.nodes[0].n_layers_total
        final_layer = n_layers - 1

        # Permutation subset results for this concept
        perm_concept = perm_data.get(concept, {})
        perm_subsets = perm_concept.get("subsets", {})
        perm_baseline_final = float(
            perm_concept.get("baseline", {}).get(str(final_layer), 0.0)
        )

        routing_held_per_node: dict[int, bool] = {}

        for i, node in enumerate(gem.nodes):
            # Metric 1: baseline routing ratio
            ratio, held_ratio = routing_held_ratio(
                baseline_per_layer,
                peak_layer=node.caz_peak,
                final_layer=final_layer,
            )

            # Metric 2: permutation routing (intervention-based)
            held_perm = None
            if perm_subsets:
                held_perm = routing_held_from_permutation(
                    perm_subsets, i, final_layer,
                    perm_baseline_final, CONTRIBUTION_THRESHOLD,
                )
                routing_held_per_node[i] = held_perm
            else:
                routing_held_per_node[i] = held_ratio

            records.append({
                "model_id": model_id,
                "concept": concept,
                "node_idx": i,
                "n_nodes": gem.n_nodes,
                "caz_peak": node.caz_peak,
                "handoff_layer": node.handoff_layer,
                "depth_pct": node.depth_pct,
                "caz_score": round(node.caz_score, 4),
                "n_layers": n_layers,
                "baseline_peak_sep": round(
                    baseline_per_layer.get(node.caz_peak, 0.0), 4),
                "baseline_final_sep": round(
                    baseline_per_layer.get(final_layer, 0.0), 4),
                "baseline_routing_ratio": ratio,
                "routing_held_ratio": held_ratio,
                "permutation_routing_held": held_perm,
                "routing_held": (
                    held_perm if held_perm is not None else held_ratio
                ),
            })

        # Stage detail string
        detail_str = stage_detail(
            gem.nodes, n_layers, routing_held_per_node
        )
        log.info(
            "  %s: %d nodes | %s",
            concept, gem.n_nodes, detail_str
        )

    return records


def aggregate_and_report(records: list[dict]) -> None:
    if not records:
        log.warning("No records to aggregate")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Overall stats
    n_total = len(records)
    n_multi = [r for r in records if r["n_nodes"] > 1]
    n_single = [r for r in records if r["n_nodes"] == 1]

    ratios = [r["baseline_routing_ratio"] for r in records]
    held_ratio = [r for r in records if r["routing_held_ratio"]]
    held_perm = [r for r in records if r.get("permutation_routing_held") is True]
    lost_perm = [r for r in records if r.get("permutation_routing_held") is False]

    # Shallow vs deep breakdown
    shallow = [r for r in records if r["depth_pct"] < 50]
    deep = [r for r in records if r["depth_pct"] >= 50]

    shallow_ratios = [r["baseline_routing_ratio"] for r in shallow]
    deep_ratios = [r["baseline_routing_ratio"] for r in deep]

    lines = [
        "GEM Routing Survival Analysis",
        "Written: 2026-04-28 UTC",
        f"N node records: {n_total}  ({len(n_multi)} multi-node, {len(n_single)} single-node)",
        "",
        "Baseline routing ratio (final_sep / peak_sep):",
        f"  All nodes:     mean={np.mean(ratios):.3f}  median={np.median(ratios):.3f}"
        f"  std={np.std(ratios):.3f}",
        f"  Routing held (>=0.95): {len(held_ratio)}/{n_total}"
        f" ({100*len(held_ratio)/n_total:.1f}%)",
        "",
        "By depth:",
        f"  Shallow (<50%): N={len(shallow)}"
        + (f"  mean_ratio={np.mean(shallow_ratios):.3f}" if shallow_ratios else ""),
        f"  Deep    (>=50%): N={len(deep)}"
        + (f"  mean_ratio={np.mean(deep_ratios):.3f}" if deep_ratios else ""),
        "",
    ]
    if held_perm or lost_perm:
        n_perm = len(held_perm) + len(lost_perm)
        lines += [
            f"Permutation routing_held (pythia-6.9b, N={n_perm}):",
            f"  Routed (contributes to final layer): {len(held_perm)}/{n_perm}"
            f" ({100*len(held_perm)/n_perm:.1f}%)",
            f"  Lost   (ablation has no final effect): {len(lost_perm)}/{n_perm}"
            f" ({100*len(lost_perm)/n_perm:.1f}%)",
            "",
        ]

    lines += [
        f"{'Model':<28} {'Concept':<18} {'N':>2} {'Node':>4} "
        f"{'Depth':>6} {'PeakSep':>8} {'FinalSep':>8} "
        f"{'Ratio':>6} {'Held?':>6} {'Perm?':>6}",
        "-" * 98,
    ]
    for r in sorted(records, key=lambda x: (x["model_id"], x["concept"], x["node_idx"])):
        perm_str = (
            "Y" if r["permutation_routing_held"] is True
            else "N" if r["permutation_routing_held"] is False
            else "-"
        )
        lines.append(
            f"{r['model_id'].split('/')[-1]:<28} {r['concept']:<18}"
            f" {r['n_nodes']:>2} {r['node_idx']:>4}"
            f" {r['depth_pct']:>5.1f}%"
            f" {r['baseline_peak_sep']:>8.3f}"
            f" {r['baseline_final_sep']:>8.3f}"
            f" {r['baseline_routing_ratio']:>6.3f}"
            f" {'Y' if r['routing_held_ratio'] else 'N':>6}"
            f" {perm_str:>6}"
        )

    text = "\n".join(lines)
    print(text)
    (OUT_DIR / "routing_survival_summary.txt").write_text(text)
    (OUT_DIR / "routing_survival.json").write_text(
        json.dumps(records, indent=2)
    )
    log.info("Wrote results to %s", OUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument("--model", type=str, default=None,
                        help="Single model ID (default: all)")
    args = parser.parse_args()

    models = [args.model] if args.model else discover_all_models()
    log.info("Processing %d models", len(models))

    all_records = []
    for model_id in models:
        log.info("=== %s ===", model_id)
        records = process_model(model_id)
        all_records.extend(records)
        log.info("  %d node records", len(records))

    aggregate_and_report(all_records)


if __name__ == "__main__":
    main()

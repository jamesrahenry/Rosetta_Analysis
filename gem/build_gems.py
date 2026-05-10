"""
build_gems.py — Batch Geometric Evolution Map (GEM) builder for all models × concepts.

No GPU required.  Reads existing caz_*.json and ablation_multimodal_*.json
files to construct ConceptGEM objects, then writes gem_*.json files
alongside the existing extraction data.

Usage
-----
    python src/build_gems.py --model EleutherAI/pythia-1.4b
    python src/build_gems.py --all
    python src/build_gems.py --all --phase 2 --k 3   # (Phase 2, not yet implemented)

Outputs: ~/rosetta_data/models/<model_slug>/gem_<concept>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from rosetta_tools.gem import (
    discover_concepts, discover_all_models, find_extraction_dir,
)

# ---------------------------------------------------------------------------
# Model / extraction discovery
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_model_gems(
    model_id: str,
    concepts: list[str] | None = None,
    k: int = 1,
    force: bool = False,
) -> dict:
    """Build GEMs for one model across all concepts.

    Returns dict of {concept: diagnostics_dict}.
    """
    from rosetta_tools.gem import (
        build_concept_gem, save_gem, validate_gem_node,
        gem_diagnostics,
    )
    from rosetta_tools.models import attention_paradigm_of

    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.error("No extraction results for %s", model_id)
        return None

    if concepts is None:
        concepts = discover_concepts(extraction_dir, source="caz")

    paradigm = attention_paradigm_of(model_id)
    log.info("=== Building GEMs: %s [%s] ===", model_id, paradigm)

    results = {}
    for concept in concepts:
        caz_path = extraction_dir / f"caz_{concept}.json"
        if not caz_path.exists():
            log.warning("  No caz_%s.json, skipping", concept)
            continue

        out_path = extraction_dir / f"gem_{concept}.json"
        if out_path.exists() and not force:
            # Check freshness: GEM should be newer than caz data
            if out_path.stat().st_mtime > caz_path.stat().st_mtime:
                log.info("  %s: GEM up-to-date, skipping (use --force to rebuild)",
                         concept)
                continue

        caz_data = json.loads(caz_path.read_text())

        # Load multimodal interaction data if available
        mm_path = extraction_dir / f"ablation_multimodal_{concept}.json"
        multimodal_data = None
        if mm_path.exists():
            multimodal_data = json.loads(mm_path.read_text())

        try:
            # Build the GEM
            gem = build_concept_gem(
                caz_data,
                multimodal_data=multimodal_data,
                attention_paradigm=paradigm,
                k=k,
            )
        except (ValueError, KeyError, IndexError) as exc:
            log.error("  %s: FAILED — %s", concept, exc)
            continue

        # Validate every node
        all_warnings = []
        for node in gem.nodes:
            warnings = validate_gem_node(node)
            if warnings:
                for w in warnings:
                    log.warning("  %s node %d: %s", concept, node.caz_index, w)
                all_warnings.extend(warnings)

        # Save
        save_gem(gem, out_path)

        # Diagnostics
        diag = gem_diagnostics(gem)
        results[concept] = diag

        # Summary line
        n_targets = len(gem.ablation_targets or [])
        types_str = ""
        if gem.node_types:
            from collections import Counter
            counts = Counter(gem.node_types)
            types_str = " | ".join(f"{t}={c}" for t, c in sorted(counts.items()))

        log.info(
            "  %s: %d nodes, %d targets | handoff_cos=%.3f | entry_exit=%.3f | %s%s",
            concept,
            gem.n_nodes,
            n_targets,
            diag.get("handoff_cosine_mean", 0),
            diag.get("entry_exit_cosine_mean", 0),
            types_str,
            f" | {len(all_warnings)} warnings" if all_warnings else "",
        )

    return results


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def print_summary(all_results: dict[str, dict[str, dict]]) -> None:
    """Print aggregate summary across all models."""
    import numpy as np

    print("\n" + "=" * 80)
    print("GEM BUILD SUMMARY")
    print("=" * 80)

    total_nodes = 0
    total_targets = 0
    all_handoff = []
    all_entry_exit = []
    all_rotation = []

    for model_id, concept_diags in sorted(all_results.items()):
        model_nodes = 0
        model_targets = 0
        for concept, diag in concept_diags.items():
            n = diag.get("n_nodes", 0)
            t = diag.get("n_ablation_targets", 0)
            model_nodes += n
            model_targets += t
            total_nodes += n
            total_targets += t
            if n > 0:
                all_handoff.append(diag["handoff_cosine_mean"])
                all_entry_exit.append(diag["entry_exit_cosine_mean"])
                all_rotation.append(diag["max_rotation_mean"])

        short = model_id.split("/")[-1]
        print(f"  {short:<30} {model_nodes:>3} nodes, {model_targets:>3} targets "
              f"({len(concept_diags)} concepts)")

    print("-" * 80)
    print(f"  {'TOTAL':<30} {total_nodes:>3} nodes, {total_targets:>3} targets")

    if all_handoff:
        print(f"\n  Handoff cosine:    mean={np.mean(all_handoff):.3f}  "
              f"std={np.std(all_handoff):.3f}  "
              f"range=[{np.min(all_handoff):.3f}, {np.max(all_handoff):.3f}]")
        print(f"  Entry-exit cosine: mean={np.mean(all_entry_exit):.3f}  "
              f"std={np.std(all_entry_exit):.3f}  "
              f"range=[{np.min(all_entry_exit):.3f}, {np.max(all_entry_exit):.3f}]")
        print(f"  Max rotation/layer: mean={np.mean(all_rotation):.3f}  "
              f"range=[{np.min(all_rotation):.3f}, {np.max(all_rotation):.3f}]")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build CAZ GEMs (Geometric Evolution Maps) from existing extraction data (no GPU needed)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Single model ID")
    group.add_argument("--all", action="store_true",
                       help="Build for all models with extraction results")
    parser.add_argument("--concepts", nargs="+", default=None,
                        help="Subset of concepts (default: all 7)")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Phase 1 (k=1, dom_vector) or Phase 2 (k>1, deep dive)")
    parser.add_argument("--k", type=int, default=1,
                        help="Number of eigenvector threads to track (Phase 2)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if GEM files are up-to-date")
    parser.add_argument("--skip-model", action="append", default=[],
                        metavar="MODEL_ID", help="Skip this model (may be repeated)")
    args = parser.parse_args()

    concepts = args.concepts or None  # None → auto-discover per model from caz_*.json

    if args.phase == 2 and args.k == 1:
        args.k = 3
        log.info("Phase 2 requested; defaulting to k=3")

    if args.all:
        models = discover_all_models()
        log.info("Found %d models with extraction results", len(models))
    else:
        models = [args.model]

    if args.skip_model:
        skip = set(args.skip_model)
        models = [m for m in models if m not in skip]

    all_results = {}
    any_extraction_found = False
    for model_id in models:
        results = build_model_gems(
            model_id, concepts, k=args.k, force=args.force,
        )
        if results is None:
            continue  # no extraction dir — already logged as error
        any_extraction_found = True
        if results:
            all_results[model_id] = results

    if all_results:
        print_summary(all_results)
    elif any_extraction_found:
        log.info("All GEMs already up-to-date.")
    else:
        log.warning("No GEMs built. Check that extraction data exists.")


if __name__ == "__main__":
    main()

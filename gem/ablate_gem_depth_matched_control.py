"""
ablate_gem_depth_matched_control.py
====================================
Reviewer null: does the GEM handoff probe outperform a depth-matched control?

The GEM comparison (handoff vs. peak) is confounded by depth: L_H >= L_peak
by construction, so any benefit from probing at a later, more-processed layer
would favour the handoff even without directional settling contributing.

This script answers the depth-confound directly: for each (model, concept) pair,
we select a *control layer* at the same relative depth as L_H but chosen without
the GEM settling criterion — specifically, a random post-CAZ layer at matched
relative depth. We then ablate the concept direction (centroid difference) at
that control layer and compare suppression to the stored GEM handoff result.

If GEM's advantage is purely from depth, control ablation at matched depth
should achieve similar suppression. If the settling criterion adds value, GEM
should outperform the depth-matched control.

Method
------
For each (model, concept) pair with existing ablation_gem data:
  1. Load the handoff layer L_H and CAZ end layer from stored GEM JSON.
  2. Compute target relative depth r = L_H / N.
  3. Enumerate post-CAZ candidate layers (L_CAZ_end+1 .. N-1), excluding L_H.
  4. Select the candidate whose relative depth is closest to r.
     If no post-CAZ candidates exist (L_H is the only post-CAZ layer), skip.
  5. Extract the concept direction at the control layer (centroid difference).
  6. Ablate the concept direction at the control layer (width=1); measure final
     separation reduction.
  7. Compare to stored GEM handoff result.

Output: ~/rosetta_data/results/gem_depth_matched_control/
  - Per-model JSON with comparison table
  - Aggregate summary

Written: 2026-05-23 UTC
"""

from __future__ import annotations

import argparse
import json
import logging
import time
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
    get_device, get_dtype, log_device_info, log_vram,
    release_model, purge_hf_cache, NumpyJSONEncoder, load_causal_lm,
)
from rosetta_tools.paths import ROSETTA_RESULTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUT_DIR = ROSETTA_RESULTS / "gem_depth_matched_control"
N_PAIRS = 250
BATCH_SIZE = 4



# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gem_json(extraction_dir: Path, concept: str) -> dict | None:
    path = extraction_dir / f"gem_{concept}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_ablation_gem(extraction_dir: Path, concept: str) -> dict | None:
    path = extraction_dir / f"ablation_gem_{concept}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def measure_ablation(
    model,
    tokenizer,
    layers: list,
    ablation_layer: int,
    direction: np.ndarray,
    pos_texts: list[str],
    neg_texts: list[str],
    device: str,
) -> tuple[float, float]:
    """Ablate direction at ablation_layer; return (baseline_sep, final_sep)."""
    dtype = next(model.parameters()).dtype
    dir_t = torch.tensor(direction, dtype=dtype, device=device)
    dir_t = dir_t / dir_t.norm()

    # Baseline
    pos_acts = extract_layer_activations(
        model, tokenizer, pos_texts, device=device,
        batch_size=BATCH_SIZE, pool="last",
    )
    neg_acts = extract_layer_activations(
        model, tokenizer, neg_texts, device=device,
        batch_size=BATCH_SIZE, pool="last",
    )
    baseline = float(compute_separation(pos_acts[-1], neg_acts[-1]))

    # Ablated
    with DirectionalAblator(layers[ablation_layer], dir_t, dtype=dtype):
        pos_acts = extract_layer_activations(
            model, tokenizer, pos_texts, device=device,
            batch_size=BATCH_SIZE, pool="last",
        )
        neg_acts = extract_layer_activations(
            model, tokenizer, neg_texts, device=device,
            batch_size=BATCH_SIZE, pool="last",
        )
    ablated = float(compute_separation(pos_acts[-1], neg_acts[-1]))

    return baseline, ablated


# ---------------------------------------------------------------------------
# Per-concept run
# ---------------------------------------------------------------------------

def run_concept(
    model,
    tokenizer,
    concept: str,
    extraction_dir: Path,
    device: str,
) -> dict | None:
    gem = load_gem_json(extraction_dir, concept)
    abl = load_ablation_gem(extraction_dir, concept)
    if gem is None or abl is None:
        log.info("  Skipping %s — missing gem or ablation_gem data", concept)
        return None

    layers = get_transformer_layers(model)
    n_layers = len(layers)

    # Primary GEM node
    targets = gem.get("ablation_targets", [0])
    node = gem["nodes"][targets[0]]
    handoff_layer = int(node["handoff_layer"])
    caz_end = int(node.get("caz_end", handoff_layer - 1))

    # Stored handoff ablation result
    comp = abl.get("comparison", {})
    handoff_retained_pct = comp.get("handoff_retained_pct")
    if handoff_retained_pct is None:
        log.info("  Skipping %s — no handoff_retained_pct in stored ablation", concept)
        return None

    # Find control layer: post-CAZ, closest relative depth to L_H/N, excluding L_H
    post_caz_candidates = [l for l in range(caz_end + 1, n_layers) if l != handoff_layer]
    if not post_caz_candidates:
        log.info("  Skipping %s — no post-CAZ candidates distinct from handoff (L_H=%d, N=%d)",
                 concept, handoff_layer, n_layers)
        return {"concept": concept, "skipped": True, "reason": "no_distinct_post_caz_layer",
                "handoff_layer": handoff_layer, "n_layers": n_layers}

    target_depth = handoff_layer / n_layers
    control_layer = min(post_caz_candidates, key=lambda l: abs(l / n_layers - target_depth))
    control_rel_depth = control_layer / n_layers

    # Concept direction at control layer: centroid difference (same as DOM vector computation)
    pairs = load_concept_pairs(concept, n=N_PAIRS)
    pos_texts, neg_texts = texts_by_label(pairs)

    log.info("  %s: L_H=%d (%.3f), control=%d (%.3f), stored_handoff_ret=%.1f%%",
             concept, handoff_layer, target_depth, control_layer, control_rel_depth,
             handoff_retained_pct)

    # Extract activations at all layers to get control direction
    pos_acts = extract_layer_activations(
        model, tokenizer, pos_texts, device=device,
        batch_size=BATCH_SIZE, pool="last",
    )
    neg_acts = extract_layer_activations(
        model, tokenizer, neg_texts, device=device,
        batch_size=BATCH_SIZE, pool="last",
    )

    pos_ctrl = pos_acts[control_layer]
    neg_ctrl = neg_acts[control_layer]
    direction = (pos_ctrl.mean(0) - neg_ctrl.mean(0)).astype(np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        log.warning("  Zero-norm direction at control layer for %s, skipping", concept)
        return None
    direction /= norm

    # Ablate at control layer, measure final-layer separation
    baseline_sep = float(compute_separation(pos_acts[-1], neg_acts[-1]))
    if baseline_sep <= 0:
        log.warning("  Zero baseline for %s, skipping", concept)
        return None

    dtype = next(model.parameters()).dtype
    dir_t = torch.tensor(direction, dtype=dtype, device=device)
    with DirectionalAblator(get_transformer_layers(model)[control_layer], dir_t, dtype=dtype):
        ctrl_pos = extract_layer_activations(
            model, tokenizer, pos_texts, device=device,
            batch_size=BATCH_SIZE, pool="last",
        )
        ctrl_neg = extract_layer_activations(
            model, tokenizer, neg_texts, device=device,
            batch_size=BATCH_SIZE, pool="last",
        )
    control_sep = float(compute_separation(ctrl_pos[-1], ctrl_neg[-1]))
    control_retained_pct = 100.0 * control_sep / baseline_sep if baseline_sep > 0 else float("nan")

    handoff_better = handoff_retained_pct < control_retained_pct
    delta_pp = control_retained_pct - handoff_retained_pct

    log.info("    handoff_ret=%.1f%% control_ret=%.1f%% delta=%.1fpp handoff_better=%s",
             handoff_retained_pct, control_retained_pct, delta_pp, handoff_better)

    return {
        "concept": concept,
        "skipped": False,
        "handoff_layer": handoff_layer,
        "handoff_rel_depth": target_depth,
        "control_layer": control_layer,
        "control_rel_depth": control_rel_depth,
        "n_layers": n_layers,
        "baseline_sep": baseline_sep,
        "handoff_retained_pct": handoff_retained_pct,
        "control_retained_pct": control_retained_pct,
        "delta_pp": delta_pp,
        "handoff_better": handoff_better,
    }


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------

def run_model(model_id: str) -> None:
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.warning("No extraction dir for %s", model_id)
        return

    out_path = OUT_DIR / f"{extraction_dir.name}_depth_matched_control.json"
    if out_path.exists():
        log.info("Already done: %s", model_id)
        return

    device = get_device()
    dtype = get_dtype(device)
    model, tokenizer = load_causal_lm(model_id, device, dtype)
    log_device_info(device, dtype)

    results = []
    for concept in discover_concepts(extraction_dir):
        r = run_concept(model, tokenizer, concept, extraction_dir, device)
        if r is not None:
            r["model_id"] = model_id
            results.append(r)

    release_model(model)
    purge_hf_cache()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model_id": model_id, "results": results}, cls=NumpyJSONEncoder, indent=2))
    log.info("Wrote %s", out_path)
    aggregate()


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate() -> None:
    files = sorted(OUT_DIR.glob("*_depth_matched_control.json"))
    all_pairs = []
    for f in files:
        d = json.loads(f.read_text())
        for r in d.get("results", []):
            if not r.get("skipped", False):
                all_pairs.append(r)

    if not all_pairs:
        return

    n = len(all_pairs)
    wins = sum(1 for r in all_pairs if r["handoff_better"])
    ties = sum(1 for r in all_pairs if r["delta_pp"] == 0)
    mean_delta = float(np.mean([r["delta_pp"] for r in all_pairs]))

    summary = {
        "n_pairs": n,
        "handoff_beats_control": wins,
        "handoff_beats_control_pct": 100.0 * wins / n if n else 0,
        "ties": ties,
        "mean_delta_pp": mean_delta,
    }
    summary_path = OUT_DIR / "aggregate.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Aggregate: %d pairs, handoff beats control %d/%d (%.1f%%), mean delta %.2fpp",
             n, wins, n, summary["handoff_beats_control_pct"], mean_delta)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Single model ID")
    group.add_argument("--all", action="store_true", help="All base models")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        models = list(discover_base_models())
    else:
        models = [args.model]

    for model_id in models:
        log.info("=== %s ===", model_id)
        run_model(model_id)


if __name__ == "__main__":
    main()

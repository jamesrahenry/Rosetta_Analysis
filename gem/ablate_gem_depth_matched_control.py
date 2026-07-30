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
from rosetta_tools.models import vram_gb as _registry_vram
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


def pick_site_matched_controls(
    gem: dict, targets: list[int], n_layers: int
) -> tuple[list[int], list[int], list[str]] | None:
    """One depth-matched control layer per target GEM, for EVERY atlas.

    Why this is not simply "a post-CAZ layer per GEM". ``ablate_gem.py`` ablates every
    target GEM's handoff layer at once, so a control that ablates one layer differs from
    it in both depth and site count. Matching site count is the point — but a strictly
    post-CAZ control cannot be built for a **terminal** GEM, one whose segment ends at
    ``N-1``, because ``range(caz_end+1, n_layers)`` is empty there. Every atlas has a
    terminal GEM, and a **single-GEM atlas is nothing but one**, so a post-CAZ-only rule
    silently drops the entire single-GEM class.

    That class is the crux: it is the only case where the handoff arm ablates one site,
    making it the one directly comparable to a one-site control. Measured on the stored
    run: **111 of 493 pairs skipped, every one of them single-GEM**, leaving n=2 — the
    two single-GEM atlases that happen not to reach ``N-1``. The comparison that decides
    the section was being excluded by the control's own layer selection.

    So terminal GEMs fall back to a **within-segment** control: the layer inside
    ``[caz_start, caz_end]`` closest in relative depth to ``L_H``. This is a weaker
    comparator — not post-CAZ, so it tests "this settled layer vs. a neighbour inside the
    same segment" rather than "vs. an equally deep layer outside it" — and the mode is
    recorded per site so the two can be reported separately rather than pooled.

    Returns ``(handoff_layers, control_layers, modes)``, equal length, or ``None`` if no
    target admits any control layer at all.
    """
    nodes = gem["nodes"]
    forbidden = {int(nodes[i]["handoff_layer"]) for i in targets}
    hs: list[int] = []
    cs: list[int] = []
    modes: list[str] = []
    for i in targets:
        node = nodes[i]
        h = int(node["handoff_layer"])
        caz_end = int(node.get("caz_end", h - 1))
        caz_start = int(node.get("caz_start", 0))
        target_depth = h / n_layers

        post = [l for l in range(caz_end + 1, n_layers)
                if l not in forbidden and l not in cs]
        if post:
            cs.append(min(post, key=lambda l: abs(l / n_layers - target_depth)))
            modes.append("post_caz")
            hs.append(h)
            continue

        within = [l for l in range(caz_start, caz_end + 1)
                  if l not in forbidden and l not in cs]
        if not within:
            continue                      # nothing usable for this GEM; drop the site
        cs.append(min(within, key=lambda l: abs(l / n_layers - target_depth)))
        modes.append("within_segment")
        hs.append(h)

    if not cs:
        return None
    return hs, cs, modes


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

    # Stored handoff ablation result — support both new (comparison dict) and old (handoff dict) formats
    comp = abl.get("comparison", {})
    handoff_retained_pct = comp.get("handoff_retained_pct")
    if handoff_retained_pct is None:
        handoff_retained_pct = abl.get("handoff", {}).get("final_retained_pct")
    if handoff_retained_pct is None:
        log.info("  Skipping %s — no handoff_retained_pct in stored ablation", concept)
        return None

    # Find control layer: post-CAZ, closest relative depth to L_H/N, excluding L_H
    target_depth = handoff_layer / n_layers
    post_caz_candidates = [l for l in range(caz_end + 1, n_layers) if l != handoff_layer]
    if post_caz_candidates:
        control_layer = min(post_caz_candidates, key=lambda l: abs(l / n_layers - target_depth))
        control_rel_depth = control_layer / n_layers
    else:
        # Terminal/single-GEM atlas: no post-CAZ layer exists, so the LEGACY
        # single-layer arm is undefined — but the site-matched block below
        # (e336572's within-segment fallback) exists precisely for this class.
        # Returning early here is what skipped the 111-pair single-GEM crux.
        log.info("  %s: no post-CAZ layer distinct from handoff (L_H=%d, N=%d) — "
                 "legacy arm skipped, site-matched arms still run", concept,
                 handoff_layer, n_layers)
        control_layer = None
        control_rel_depth = None

    # Concept direction at control layer: centroid difference (same as DOM vector computation)
    pairs = load_concept_pairs(concept, n=N_PAIRS)
    pos_texts, neg_texts = texts_by_label(pairs)

    log.info("  %s: L_H=%d (%.3f), control=%s, stored_handoff_ret=%.1f%%",
             concept, handoff_layer, target_depth,
             ("%d (%.3f)" % (control_layer, control_rel_depth)
              if control_layer is not None else "none/terminal"),
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

    # Baseline + dtype are shared by both the legacy and site-matched arms
    baseline_sep = float(compute_separation(pos_acts[-1], neg_acts[-1]))
    if baseline_sep <= 0:
        log.warning("  Zero baseline for %s, skipping", concept)
        return None
    dtype = next(model.parameters()).dtype

    # ---- Legacy single-layer control arm (undefined for terminal atlases) --
    control_retained_pct = handoff_better = delta_pp = None
    if control_layer is not None:
        direction = (pos_acts[control_layer].mean(0)
                     - neg_acts[control_layer].mean(0)).astype(np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            log.warning("  Zero-norm direction at control layer for %s — "
                        "legacy arm skipped, site-matched arms still run", concept)
        else:
            direction /= norm
            dir_t = torch.tensor(direction, dtype=dtype, device=device)
            with DirectionalAblator(get_transformer_layers(model)[control_layer],
                                    dir_t, dtype=dtype):
                ctrl_pos = extract_layer_activations(
                    model, tokenizer, pos_texts, device=device,
                    batch_size=BATCH_SIZE, pool="last",
                )
                ctrl_neg = extract_layer_activations(
                    model, tokenizer, neg_texts, device=device,
                    batch_size=BATCH_SIZE, pool="last",
                )
            control_sep = float(compute_separation(ctrl_pos[-1], ctrl_neg[-1]))
            control_retained_pct = 100.0 * control_sep / baseline_sep
            handoff_better = handoff_retained_pct < control_retained_pct
            delta_pp = control_retained_pct - handoff_retained_pct

    # ---- Site-matched arms ------------------------------------------------
    # BOTH arms are recomputed here over the same sub-atlas. The stored
    # handoff_retained_pct above is a WHOLE-atlas ablation; comparing it against a
    # sub-atlas control would reintroduce the site-count mismatch this exists to
    # remove. Only depth differs between the two arms below.
    matched = pick_site_matched_controls(gem, targets, n_layers)
    sm: dict = {"site_matched": False, "reason": "no_matchable_gem"}
    if matched is not None:
        h_layers, c_layers, c_modes = matched
        all_layers = get_transformer_layers(model)

        def _ablate_at(layer_idxs, dirs):
            with ExitStack() as stack:
                for li, d in zip(layer_idxs, dirs):
                    stack.enter_context(
                        DirectionalAblator(all_layers[li],
                                           torch.tensor(d, dtype=dtype, device=device),
                                           dtype=dtype))
                ap = extract_layer_activations(model, tokenizer, pos_texts, device=device,
                                               batch_size=BATCH_SIZE, pool="last")
                an = extract_layer_activations(model, tokenizer, neg_texts, device=device,
                                               batch_size=BATCH_SIZE, pool="last")
            return float(compute_separation(ap[-1], an[-1]))

        def _unit(v):
            v = np.asarray(v, dtype=np.float64)
            n = np.linalg.norm(v)
            return None if n < 1e-8 else v / n

        # handoff arm: each matched GEM's settled direction at its own handoff layer
        h_dirs, c_dirs = [], []
        nodes = gem["nodes"]
        by_handoff = {int(nodes[i]["handoff_layer"]): nodes[i] for i in targets}
        for hl in h_layers:
            h_dirs.append(_unit(by_handoff[hl]["settled_direction"]))
        # control arm: centroid difference measured at each control layer
        for cl in c_layers:
            c_dirs.append(_unit(pos_acts[cl].mean(0) - neg_acts[cl].mean(0)))

        if any(d is None for d in h_dirs + c_dirs):
            sm = {"site_matched": False, "reason": "zero_norm_direction"}
        else:
            h_ret = 100.0 * _ablate_at(h_layers, h_dirs) / baseline_sep
            c_ret = 100.0 * _ablate_at(c_layers, c_dirs) / baseline_sep
            sm = {
                "site_matched": True,
                "n_sites": len(c_layers),
                "n_targets_total": len(targets),
                "atlas_coverage": round(len(c_layers) / len(targets), 3),
                "handoff_layers": h_layers,
                "control_layers": c_layers,
                "control_modes": c_modes,
                "all_post_caz": all(m == "post_caz" for m in c_modes),
                "handoff_retained_pct": h_ret,
                "control_retained_pct": c_ret,
                "delta_pp": c_ret - h_ret,
                "handoff_better": h_ret < c_ret,
            }
            log.info("    site-matched %d/%d GEMs: handoff%s=%.1f%% control%s=%.1f%% delta=%.1fpp",
                     len(c_layers), len(targets), h_layers, h_ret, c_layers, c_ret, c_ret - h_ret)
    else:
        log.info("    site-matched: no GEM in this atlas has a distinct post-CAZ layer")

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
        "n_targets": len(targets),
        "site_matched_control": sm,
    }


# ---------------------------------------------------------------------------
# Per-model run
# ---------------------------------------------------------------------------

def run_model(model_id: str, overwrite: bool = False) -> None:
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.warning("No extraction dir for %s", model_id)
        return

    out_path = OUT_DIR / f"{extraction_dir.name}_depth_matched_control.json"
    if out_path.exists() and not overwrite:
        log.info("Already done: %s", model_id)
        return

    device = get_device()
    dtype = get_dtype(device)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    model_vram = _registry_vram(model_id)
    device_map = None  # may be overridden below once per-GPU VRAM is known

    # _init_weights in transformers calls .float() on weights, creating a transient
    # float32 copy (~2× bf16 size). Skip models where this would OOM.
    if n_gpus > 0:
        total_vram_gb = sum(
            torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            for i in range(n_gpus)
        )
        per_gpu_vram = total_vram_gb / n_gpus
        # Use device_map="auto" when model exceeds 80% of a single GPU — avoids
        # OOM during inference (e.g. gemma-2-9b logit softcapping at 18/22 GB).
        device_map = "auto" if (model_vram > per_gpu_vram * 0.8 and n_gpus > 1) else None
        if model_vram * 2.1 > total_vram_gb:
            log.warning(
                "Skipping %s — %.0f GB bf16 requires ~%.0f GB peak (float32 init); "
                "only %.0f GB available across %d GPU(s)",
                model_id, model_vram, model_vram * 2.1, total_vram_gb, n_gpus,
            )
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(
                {"model_id": model_id, "results": [], "skipped_model": True,
                 "reason": "vram_insufficient", "model_vram_gb": model_vram,
                 "total_vram_gb": total_vram_gb},
                indent=2,
            ))
            return

    if device_map:
        log.info("Large model (%.0f GB bf16): device_map='auto' across %d GPUs", model_vram, n_gpus)
    model, tokenizer = load_causal_lm(model_id, device, dtype, device_map=device_map)
    log_device_info(device, dtype)

    results = []
    for concept in discover_concepts(extraction_dir):
        r = run_concept(model, tokenizer, concept, extraction_dir, device)
        if r is not None:
            r["model_id"] = model_id
            results.append(r)

    release_model(model)
    purge_hf_cache(model_id, min_free_gb=0.0)

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

    # Legacy single-layer arm: defined only for non-terminal atlases (delta_pp
    # is None where the post-CAZ control does not exist).
    legacy = [r for r in all_pairs if r.get("delta_pp") is not None]
    summary = {"n_pairs_total": len(all_pairs), "legacy": None, "site_matched": None}
    if legacy:
        n = len(legacy)
        wins = sum(1 for r in legacy if r["handoff_better"])
        summary["legacy"] = {
            "n_pairs": n,
            "handoff_beats_control": wins,
            "handoff_beats_control_pct": 100.0 * wins / n,
            "ties": sum(1 for r in legacy if r["delta_pp"] == 0),
            "mean_delta_pp": float(np.mean([r["delta_pp"] for r in legacy])),
        }

    # Site-matched arms (the P2 §5.5 / P3 F8 comparison), split by control mode
    # per the pick_site_matched_controls docstring: pure post-CAZ vs any
    # within-segment fallback are different comparators — never pooled silently.
    sm_rows = [r for r in all_pairs
               if r.get("site_matched_control", {}).get("site_matched")]
    for label, rows in (
        ("site_matched_post_caz",
         [r for r in sm_rows if r["site_matched_control"]["all_post_caz"]]),
        ("site_matched_with_fallback",
         [r for r in sm_rows if not r["site_matched_control"]["all_post_caz"]]),
    ):
        if not rows:
            continue
        n = len(rows)
        sm = [r["site_matched_control"] for r in rows]
        summary[label] = {
            "n_pairs": n,
            "handoff_beats_control": sum(1 for x in sm if x["handoff_better"]),
            "handoff_beats_control_pct":
                100.0 * sum(1 for x in sm if x["handoff_better"]) / n,
            "mean_delta_pp": float(np.mean([x["delta_pp"] for x in sm])),
        }
    summary["site_matched"] = {"n_pairs": len(sm_rows)}

    summary_path = OUT_DIR / "aggregate.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Aggregate: %s", json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Single model ID")
    group.add_argument("--all", action="store_true", help="All base models")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute models that already have output (needed for the "
                             "site-matched re-run, which adds a field to existing files).")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        models = list(discover_base_models())
    else:
        models = [args.model]

    for model_id in models:
        log.info("=== %s ===", model_id)
        run_model(model_id, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

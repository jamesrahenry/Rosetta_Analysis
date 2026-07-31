#!/usr/bin/env python3
"""TNCONF — terminal-node confound control for P3 §6.5 (handoff-vs-peak).

Self-sufficient (round3 path): depends only on rosetta_tools + this dir's
common.py/forward_utils — NO ablate_gem import. The three ablation functions
below (measure_separation, run_handoff_ablation, run_peak_ablation) are copied
VERBATIM from Rosetta_Analysis/gem/ablate_gem.py so the measurement is
byte-identical to the one that produced the stored ablation_gem artifacts; the
per-pair validation gate re-verifies that against the store.

MOTIVATION (GEM_DEFINITION.md, claude:p3-corpus-review 2026-07-27): every GEM
has one node whose handoff is pinned to N-1 (caz.py:946), and P3 measures
separation at the FINAL layer — so the handoff target set includes a
readout-adjacent ablation in 100% of cells vs 8.2% for peaks. Part of the
304/442 (68.8%) handoff-better result (§6.5) may be that pinned-boundary
artifact, not the "settled product is more causally complete" thesis.

TEST: rerun handoff-vs-peak EXCLUDING the terminal node from BOTH target sets
(interior handoffs vs interior peaks) and report whether the advantage survives.
Only multi-node pairs (n_nodes >= 2) have an interior set.

Standalone:  python tnconf_terminal_node_confound.py --all [--smoke]
Outputs one shard per model (checkpointed) + tnconf_terminal_node_confound_results.json
uploaded to HF paper_n250/_round3_gpu/tnconf/ at finalize.
Written: 2026-07-27 UTC — claude:p3-corpus-review.
"""
from __future__ import annotations

import argparse
import json
from contextlib import ExitStack

import numpy as np

from rosetta_tools.ablation import DirectionalAblator, get_transformer_layers
from rosetta_tools.caz import compute_separation
from rosetta_tools.dataset import load_concept_pairs, texts_by_label
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.gem import load_gem

from common import (
    BASE_28, CONCEPTS_17, MODELS_ROOT, _fetch_artifact_from_hf,
    hf_upload, hf_verify, log, shard_done, shard_write, slugify,
)
from forward_utils import load_model, release

JOB = "tnconf"
N_PAIRS = 250
N_PAIRS_OVERRIDE = {"exfiltration": 249}


def _ensure(slug, filename):
    """Self-heal a paper_n250 artifact from HF if absent (fresh host).
    Returns True if present/fetched, False if not fetchable."""
    if (MODELS_ROOT / slug / filename).exists():
        return True
    try:
        _fetch_artifact_from_hf(slug, filename)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Ablation — copied VERBATIM from Rosetta_Analysis/gem/ablate_gem.py so the
# measurement is identical to the stored artifacts (validated per-pair below).
# ---------------------------------------------------------------------------
def measure_separation(model, tokenizer, layers, ablate_layers, ablate_directions,
                       pos_texts, neg_texts, measure_layers, device, batch_size):
    model_dtype = next(model.parameters()).dtype
    with ExitStack() as stack:
        for layer_idx, direction in zip(ablate_layers, ablate_directions):
            stack.enter_context(
                DirectionalAblator(layers[layer_idx], direction, dtype=model_dtype)
            )
        pos_acts = extract_layer_activations(
            model, tokenizer, pos_texts, device=device, batch_size=batch_size, pool="last")
        neg_acts = extract_layer_activations(
            model, tokenizer, neg_texts, device=device, batch_size=batch_size, pool="last")
    results = {}
    for layer_idx in measure_layers:
        act_idx = layer_idx + 1  # extraction includes embedding at [0]
        if act_idx >= len(pos_acts):
            act_idx = len(pos_acts) - 1
        results[layer_idx] = float(compute_separation(pos_acts[act_idx], neg_acts[act_idx]))
    return results


def _per_layer(baseline, ablated, measure_at, n_layers):
    per_layer = {}
    for li in measure_at:
        bl = baseline.get(li, 0)
        ab = ablated.get(li, 0)
        retained = (100 * ab / bl) if bl > 0 else 100.0
        per_layer[li] = {
            "baseline_sep": round(bl, 4), "ablated_sep": round(ab, 4),
            "retained_pct": round(retained, 1),
            "sep_reduction": round(max(0, 1 - ab / bl) if bl > 0 else 0, 4),
        }
    return per_layer, per_layer.get(n_layers - 1, {})


def run_handoff_ablation(model, tokenizer, layers, gem, pos_texts, neg_texts,
                         device, batch_size, width=1):
    n_layers = len(layers)
    measure_at = sorted(set([node.caz_peak for node in gem.nodes] + [n_layers - 1]))
    baseline = measure_separation(model, tokenizer, layers, [], [], pos_texts, neg_texts,
                                  measure_at, device, batch_size)
    ablate_layers, ablate_dirs = [], []
    for node in gem.target_nodes:
        direction = np.array(node.settled_direction, dtype=np.float64)
        for offset in range(width):
            layer_idx = node.handoff_layer + offset
            if layer_idx < n_layers:
                ablate_layers.append(layer_idx)
                ablate_dirs.append(direction)
    if not ablate_layers:
        return {"error": "no_valid_layers"}
    ablated = measure_separation(model, tokenizer, layers, ablate_layers, ablate_dirs,
                                 pos_texts, neg_texts, measure_at, device, batch_size)
    per_layer, final = _per_layer(baseline, ablated, measure_at, n_layers)
    return {"mode": "handoff", "n_targets": len(gem.target_nodes),
            "ablation_layers": ablate_layers,
            "final_retained_pct": final.get("retained_pct", 100.0),
            "final_sep_reduction": final.get("sep_reduction", 0.0)}


def run_peak_ablation(model, tokenizer, layers, gem, pos_texts, neg_texts,
                      device, batch_size, width=1):
    n_layers = len(layers)
    measure_at = sorted(set([node.caz_peak for node in gem.nodes] + [n_layers - 1]))
    baseline = measure_separation(model, tokenizer, layers, [], [], pos_texts, neg_texts,
                                  measure_at, device, batch_size)
    ablate_layers, ablate_dirs = [], []
    for node in gem.target_nodes:
        peak_idx = node.caz_peak - node.caz_start
        if peak_idx < 0 or peak_idx >= len(node.concept_thread.directions):
            peak_idx = 0
        peak_dir = np.array(node.concept_thread.directions[peak_idx], dtype=np.float64)
        for offset in range(width):
            layer_idx = node.caz_peak + offset
            if layer_idx < n_layers:
                ablate_layers.append(layer_idx)
                ablate_dirs.append(peak_dir)
    if not ablate_layers:
        return {"error": "no_valid_layers"}
    ablated = measure_separation(model, tokenizer, layers, ablate_layers, ablate_dirs,
                                 pos_texts, neg_texts, measure_at, device, batch_size)
    per_layer, final = _per_layer(baseline, ablated, measure_at, n_layers)
    return {"mode": "peak", "n_targets": len(gem.target_nodes),
            "ablation_layers": ablate_layers,
            "final_retained_pct": final.get("retained_pct", 100.0),
            "final_sep_reduction": final.get("sep_reduction", 0.0)}


# ---------------------------------------------------------------------------
# Terminal-node control
# ---------------------------------------------------------------------------
class _GemView:
    """Delegates to the real GEM except target_nodes (overridden). Leaving
    `nodes` delegated keeps measure_at identical between full and interior —
    only the ablation target set changes."""
    def __init__(self, gem, target_nodes):
        object.__setattr__(self, "_gem", gem)
        object.__setattr__(self, "target_nodes", list(target_nodes))

    def __getattr__(self, name):
        return getattr(self._gem, name)


def _handoff_better(h, p):
    return h.get("final_retained_pct", 100.0) < p.get("final_retained_pct", 100.0)


def _stored_handoff_better(slug, concept):
    p = MODELS_ROOT / slug / f"ablation_gem_{concept}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("comparison", {}).get("handoff_better")
    except Exception:
        return None


def run_for_model(model_id, model, tok, device, batch_size, concepts=CONCEPTS_17, smoke=False):
    slug = slugify(model_id)
    if shard_done(JOB, slug) and not smoke:
        log.info("[tnconf] %s already done, skipping", slug)
        return
    layers = get_transformer_layers(model)
    n_layers = len(layers)
    rows = []
    for concept in concepts:
        if not _ensure(slug, f"gem_{concept}.json"):
            continue
        _ensure(slug, f"ablation_gem_{concept}.json")  # for the validation gate
        gp = MODELS_ROOT / slug / f"gem_{concept}.json"
        gem = load_gem(gp)
        if gem.n_nodes < 2:
            continue
        interior = [nd for nd in gem.target_nodes if nd.handoff_layer != n_layers - 1]
        if not interior or len(interior) == len(gem.target_nodes):
            continue
        pairs = load_concept_pairs(concept, n=N_PAIRS_OVERRIDE.get(concept, N_PAIRS))
        pos, neg = texts_by_label(pairs)
        pos = [t for t in pos if t and t.strip()]
        neg = [t for t in neg if t and t.strip()]

        hf = run_handoff_ablation(model, tok, layers, gem, pos, neg, device, batch_size)
        pf = run_peak_ablation(model, tok, layers, gem, pos, neg, device, batch_size)
        hb_full = _handoff_better(hf, pf)
        stored = _stored_handoff_better(slug, concept)

        view = _GemView(gem, interior)
        hi = run_handoff_ablation(model, tok, layers, view, pos, neg, device, batch_size)
        pi = run_peak_ablation(model, tok, layers, view, pos, neg, device, batch_size)
        hb_int = _handoff_better(hi, pi)

        rows.append({
            "model_id": model_id, "concept": concept, "n_nodes": gem.n_nodes,
            "n_layers": n_layers,
            "full": {"handoff_retained_pct": hf.get("final_retained_pct"),
                     "peak_retained_pct": pf.get("final_retained_pct"),
                     "handoff_sep_reduction": hf.get("final_sep_reduction"),
                     "peak_sep_reduction": pf.get("final_sep_reduction"),
                     "handoff_better": hb_full},
            "interior": {"n_targets": len(interior),
                         "handoff_retained_pct": hi.get("final_retained_pct"),
                         "peak_retained_pct": pi.get("final_retained_pct"),
                         "handoff_sep_reduction": hi.get("final_sep_reduction"),
                         "peak_sep_reduction": pi.get("final_sep_reduction"),
                         "handoff_better": hb_int},
            "validation": {"stored_handoff_better": stored,
                           "full_reproduces_stored": (stored is None or hb_full == stored)},
        })
        log.info("[tnconf] %s/%s n=%d full_hb=%s(stored=%s) int_hb=%s",
                 slug, concept, gem.n_nodes, hb_full, stored, hb_int)
        if smoke:
            break
    shard_write(JOB, slug, {"model_id": model_id, "rows": rows})


def finalize(smoke=False):
    from common import CKPT_ROOT, OUT_ROOT
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    d = CKPT_ROOT / JOB
    for shard in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            rows.extend(json.loads(shard.read_text()).get("rows", []))
        except Exception:
            continue
    n = len(rows)
    full_hb = sum(1 for r in rows if r["full"]["handoff_better"])
    int_hb = sum(1 for r in rows if r["interior"]["handoff_better"])
    gate = sum(1 for r in rows if r["validation"]["full_reproduces_stored"])
    flipped = sum(1 for r in rows if r["full"]["handoff_better"] and not r["interior"]["handoff_better"])
    summary = {
        "job": JOB, "n_multinode_pairs": n,
        "full_handoff_better": full_hb,
        "full_handoff_better_rate": round(100 * full_hb / n, 1) if n else None,
        "interior_handoff_better": int_hb,
        "interior_handoff_better_rate": round(100 * int_hb / n, 1) if n else None,
        "flipped_full_to_interior": flipped,
        "validation_gate_match": gate,
        "validation_gate_rate": round(100 * gate / n, 1) if n else None,
        "interpretation": ("advantage survives terminal-node exclusion — not a "
                           "pinned-boundary artifact" if n and int_hb / n > 0.6
                           else "advantage weakens without the terminal node — "
                           "confound confirmed; state it at §6.5"),
        "rows": rows,
    }
    fpath = OUT_ROOT / "tnconf_terminal_node_confound_results.json"
    fpath.write_text(json.dumps(summary, indent=2))
    log.info("[tnconf] full HB %s%% vs interior HB %s%% over %d pairs (gate %s%%, flipped %d)",
             summary["full_handoff_better_rate"], summary["interior_handoff_better_rate"],
             n, summary["validation_gate_rate"], flipped)
    if not smoke:
        hf_upload(JOB, fpath)
        hf_verify(JOB, [fpath.name])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    models = args.models or BASE_28
    if args.smoke:
        models = models[:1]
    for mid in models:
        model = None
        try:
            model, tok, device = load_model(mid)
            run_for_model(mid, model, tok, device, args.batch_size, smoke=args.smoke)
        finally:
            if model is not None:
                try:
                    release(model)
                except Exception:
                    pass
    finalize(smoke=args.smoke)


if __name__ == "__main__":
    main()

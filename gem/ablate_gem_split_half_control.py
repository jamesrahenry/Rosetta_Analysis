#!/usr/bin/env python3
"""
ablate_gem_split_half_control.py
================================
Out-of-sample layer-selection control for GEM (P2 job G-P2-2 / triage item M9).

The objection
------------
GEM selects the handoff layer L_H where the difference-of-means direction has
stopped changing between adjacent layers — mechanically, where ``u`` is
low-variance and therefore well estimated. It then *evaluates* how effectively
that same ``u`` ablates. A better-estimated direction ablates better, so the
headline result is open to a deflationary reading: GEM may be a
variance-reduction heuristic ("extract where your estimate is least noisy")
rather than a claim about where concepts are assembled. §5.5's depth-matched
control does not resolve this — its comparator layer is one where ``u`` is
*less* settled, so "+32.5pp for the settled layer" restates the same thing.

The disambiguating test
-----------------------
Split each concept's pairs into disjoint halves A and B.

* Detect on **A** → L_H^A and peak^A. The layer choice never sees B.
* Extract the direction on **B**, ablate on **B**, measure on **B**.

Both the in-sample and out-of-sample arms estimate ``u`` from B and are scored
on B, so estimator quality and sample size are held constant. The *only* thing
that differs is whether the layer index was chosen on the same data it is
evaluated on. If the handoff-over-peak advantage survives out-of-sample
selection, the circularity reading is refuted; if it collapses, the advantage
was selection on noise and the paper must say so.

Four arms per (model, concept), all measured on half B::

    in_sample_handoff       L_H^B    (the standard pipeline, restricted to B)
    out_of_sample_handoff   L_H^A    (layer chosen on held-out data)
    in_sample_peak          peak^B
    out_of_sample_peak      peak^A

Usage
-----
    python ablate_gem_split_half_control.py --model EleutherAI/pythia-410m
    python ablate_gem_split_half_control.py --all --seed 42

Written: 2026-07-27 UTC — claude:p2-review, for P2 triage M9.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from rosetta_tools.ablation import (
    DirectionalAblator, get_transformer_layers, compute_dominant_direction,
)
from rosetta_tools.caz import (
    compute_separation, compute_layer_metrics, find_caz_regions_scored,
)
from rosetta_tools.dataset import load_concept_pairs, texts_by_label
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.gem import discover_concepts, discover_base_models, find_extraction_dir
from rosetta_tools.gpu_utils import (
    get_device, get_dtype, log_device_info, release_model,
    NumpyJSONEncoder, load_causal_lm,
)
from rosetta_tools.paths import ROSETTA_RESULTS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUT_DIR = ROSETTA_RESULTS / "gem_split_half_control"
N_PAIRS = 250
BATCH_SIZE = 4
MIN_HALF = 40          # below this a half is too small for a stable DoM estimate


# ---------------------------------------------------------------------------
# Detection on an arbitrary subset
# ---------------------------------------------------------------------------

def detect_layers(pos_acts: list[np.ndarray], neg_acts: list[np.ndarray],
                  n_layers: int) -> tuple[int, int] | None:
    """Run the CAZ/GEM detector on one half. Returns (handoff_layer, peak_layer).

    Mirrors the production path: separation curve -> scored saddle-to-saddle
    segmentation -> the largest-separation segment -> handoff = end + 1. The
    largest-separation segment is a *locator* label with no causal privilege
    (P2 §4.1); it is used here only because it is what the paper reports.
    """
    seps = np.array([float(compute_separation(p, n))
                     for p, n in zip(pos_acts, neg_acts)])
    if not np.isfinite(seps).all() or seps.max() <= 0:
        return None
    # compute_layer_metrics takes the per-layer (pos, neg) activation pairs and
    # derives S/C/v itself — it does not take a precomputed separation curve.
    metrics = compute_layer_metrics(list(zip(pos_acts, neg_acts)))
    profile = find_caz_regions_scored(metrics)   # returns a CAZProfile
    regions = getattr(profile, "regions", None)
    if not regions:
        return None
    region = max(regions, key=lambda r: r.caz_score)
    handoff = min(region.end + 1, n_layers - 1)
    peak = int(region.peak)
    return handoff, peak


def measure(model, tokenizer, layers, ablate_at: int, direction: np.ndarray,
            pos_texts: list[str], neg_texts: list[str], device: str,
            baseline_final: float) -> float:
    """Ablate `direction` at `ablate_at`; return final-layer retained %."""
    dtype = next(model.parameters()).dtype
    d = torch.tensor(direction, dtype=dtype, device=device)
    d = d / d.norm()
    with DirectionalAblator(layers[ablate_at], d, dtype=dtype):
        p = extract_layer_activations(model, tokenizer, pos_texts, device=device,
                                      batch_size=BATCH_SIZE, pool="last")
        n = extract_layer_activations(model, tokenizer, neg_texts, device=device,
                                      batch_size=BATCH_SIZE, pool="last")
    ablated = float(compute_separation(p[-1], n[-1]))
    if baseline_final <= 0:
        return float("nan")
    return round(100.0 * ablated / baseline_final, 2)


# ---------------------------------------------------------------------------
# Per-concept
# ---------------------------------------------------------------------------

def run_concept(model, tokenizer, layers, model_id: str, concept: str,
                device: str, seed: int) -> dict | None:
    pairs = load_concept_pairs(concept, n=N_PAIRS)
    pos, neg = texts_by_label(pairs)
    n = min(len(pos), len(neg))
    if n < 2 * MIN_HALF:
        log.warning("  %s: only %d pairs, need %d — skipped", concept, n, 2 * MIN_HALF)
        return None

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a_idx, b_idx = idx[: n // 2], idx[n // 2:]
    A = ([pos[i] for i in a_idx], [neg[i] for i in a_idx])
    B = ([pos[i] for i in b_idx], [neg[i] for i in b_idx])

    # One clean forward pass per half — detection needs all layers.
    a_pos = extract_layer_activations(model, tokenizer, A[0], device=device,
                                      batch_size=BATCH_SIZE, pool="last")
    a_neg = extract_layer_activations(model, tokenizer, A[1], device=device,
                                      batch_size=BATCH_SIZE, pool="last")
    b_pos = extract_layer_activations(model, tokenizer, B[0], device=device,
                                      batch_size=BATCH_SIZE, pool="last")
    b_neg = extract_layer_activations(model, tokenizer, B[1], device=device,
                                      batch_size=BATCH_SIZE, pool="last")

    n_layers = len(layers)
    det_a = detect_layers(a_pos, a_neg, n_layers)
    det_b = detect_layers(b_pos, b_neg, n_layers)
    if det_a is None or det_b is None:
        log.warning("  %s: detection failed on a half — skipped", concept)
        return None
    h_a, pk_a = det_a
    h_b, pk_b = det_b

    baseline_final = float(compute_separation(b_pos[-1], b_neg[-1]))
    if baseline_final <= 0:
        log.warning("  %s: degenerate baseline on half B — skipped", concept)
        return None

    # Every direction is estimated on B, so estimator quality is held constant
    # across arms; only the provenance of the layer index differs.
    arms = {}
    for name, layer in (("in_sample_handoff", h_b), ("out_of_sample_handoff", h_a),
                        ("in_sample_peak", pk_b), ("out_of_sample_peak", pk_a)):
        direction = compute_dominant_direction(b_pos[layer], b_neg[layer])
        arms[name] = {"layer": int(layer),
                      "retained_pct": measure(model, tokenizer, layers, layer,
                                              direction, B[0], B[1], device,
                                              baseline_final)}

    rec = {
        "model_id": model_id, "concept": concept, "seed": seed,
        "n_pairs_total": n, "n_half": len(b_idx), "n_layers": n_layers,
        "baseline_final_sep": round(baseline_final, 6),
        "layers": {"handoff_A": h_a, "handoff_B": h_b, "peak_A": pk_a, "peak_B": pk_b,
                   "handoff_agrees": h_a == h_b, "peak_agrees": pk_a == pk_b},
        "arms": arms,
    }
    # The headline contrast: does handoff beat peak, in-sample and out-of-sample?
    rec["in_sample_handoff_better"] = (
        arms["in_sample_handoff"]["retained_pct"] < arms["in_sample_peak"]["retained_pct"])
    rec["out_of_sample_handoff_better"] = (
        arms["out_of_sample_handoff"]["retained_pct"] < arms["out_of_sample_peak"]["retained_pct"])
    log.info("  %-16s L_H A/B=%2d/%2d  peak A/B=%2d/%2d  in-samp %s  out-samp %s",
             concept, h_a, h_b, pk_a, pk_b,
             "handoff" if rec["in_sample_handoff_better"] else "peak   ",
             "handoff" if rec["out_of_sample_handoff_better"] else "peak")
    return rec


def run_model(model_id: str, seed: int, overwrite: bool) -> None:
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.warning("no extraction dir for %s — skipped", model_id)
        return
    out_path = OUT_DIR / f"{extraction_dir.name}_split_half_control.json"
    if out_path.exists() and not overwrite:
        log.info("%s: exists, skipping (use --overwrite)", out_path.name)
        return

    device = get_device("auto")
    dtype = get_dtype("auto")
    log_device_info(device)
    log.info("Loading %s", model_id)
    model, tokenizer = load_causal_lm(model_id, device=device, dtype=dtype)
    layers = get_transformer_layers(model)

    records, t0 = [], time.time()
    try:
        for concept in discover_concepts(extraction_dir):
            try:
                r = run_concept(model, tokenizer, layers, model_id, concept, device, seed)
                if r:
                    records.append(r)
            except Exception as e:                      # one concept must not sink the model
                log.error("  %s failed: %s", concept, e)
    finally:
        release_model(model)

    if not records:
        log.warning("%s: no usable records", model_id)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": model_id, "seed": seed, "n_concepts": len(records),
        "elapsed_s": round(time.time() - t0, 1),
        "summary": summarise(records),
        "records": records,
    }
    out_path.write_text(json.dumps(payload, indent=2, cls=NumpyJSONEncoder))
    log.info("Wrote %s", out_path)


def summarise(records: list[dict]) -> dict:
    ins = sum(r["in_sample_handoff_better"] for r in records)
    outs = sum(r["out_of_sample_handoff_better"] for r in records)
    agree = sum(r["layers"]["handoff_agrees"] for r in records)
    n = len(records)
    return {
        "n": n,
        "in_sample_handoff_better": ins,
        "out_of_sample_handoff_better": outs,
        "handoff_layer_agreement": agree,
        "delta_pp": round(100.0 * (outs - ins) / n, 1) if n else None,
    }


def aggregate() -> None:
    files = sorted(OUT_DIR.glob("*_split_half_control.json"))
    if not files:
        log.warning("nothing to aggregate")
        return
    ins = outs = agree = n = 0
    for f in files:
        s = json.loads(f.read_text())["summary"]
        ins += s["in_sample_handoff_better"]; outs += s["out_of_sample_handoff_better"]
        agree += s["handoff_layer_agreement"]; n += s["n"]
    out = {"models": len(files), "pairs": n,
           "in_sample_handoff_better": ins, "in_sample_pct": round(100 * ins / n, 1),
           "out_of_sample_handoff_better": outs, "out_of_sample_pct": round(100 * outs / n, 1),
           "delta_pp": round(100 * (outs - ins) / n, 1),
           "handoff_layer_agreement_pct": round(100 * agree / n, 1)}
    (OUT_DIR / "aggregate.json").write_text(json.dumps(out, indent=2))
    log.info("AGGREGATE %s", json.dumps(out))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", type=str)
    g.add_argument("--all", action="store_true")
    g.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    p.add_argument("--no-clean-cache", action="store_true")
    a = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.aggregate_only:
        aggregate(); return
    models = discover_base_models() if a.all else [a.model]
    for m in models:
        run_model(m, seed=a.seed, overwrite=a.overwrite)
    aggregate()


if __name__ == "__main__":
    main()

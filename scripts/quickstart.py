#!/usr/bin/env python3
"""Quickstart: live concept detection on a single HuggingFace model.

Loads a model, runs the full CAZ/GEM pipeline on a small number of concept
pairs, and prints an ASCII layer profile with CAZ peaks and GEM summary —
no files saved.

Usage:
  python scripts/quickstart.py
  python scripts/quickstart.py --concept sentiment --n-pairs 15
  python scripts/quickstart.py --model openai-community/gpt2-large --concept negation

Default: GPT-2-XL, credibility concept, 10 pairs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — works on GPU hosts (~/rosetta_tools/) and dev machine
# ---------------------------------------------------------------------------
for _p in [
    Path.home() / "rosetta_tools",
    Path.home() / "Source" / "Rosetta_Program" / "rosetta_tools",
]:
    if (_p / "rosetta_tools").exists():
        sys.path.insert(0, str(_p))
        break

DEFAULT_MODEL = "openai-community/gpt2-xl"
DEFAULT_CONCEPT = "credibility"
DEFAULT_N_PAIRS = 10
BAR_WIDTH = 40


def _bar(value: float, max_val: float) -> str:
    filled = int(round(BAR_WIDTH * value / max_val)) if max_val > 0 else 0
    filled = min(filled, BAR_WIDTH)
    return "█" * filled + "░" * (BAR_WIDTH - filled)


def print_profile(
    separations: list[float],
    caz_profile,
    model_id: str,
    concept: str,
    n_pairs: int,
    elapsed_s: float,
) -> None:
    peak_layer = int(np.argmax(separations))
    max_sep = max(separations) if separations else 1.0

    # Mark layers with CAZ events
    caz_layers: dict[int, str] = {}
    if caz_profile.dominant:
        r = caz_profile.dominant
        for l in range(r.start, r.end + 1):
            caz_layers[l] = "·"
        caz_layers[r.peak] = "▲"  # dominant peak
    if caz_profile.secondary:
        r = caz_profile.secondary
        for l in range(r.start, r.end + 1):
            if l not in caz_layers:
                caz_layers[l] = "·"
        if r.peak not in caz_layers or caz_layers[r.peak] == "·":
            caz_layers[r.peak] = "△"  # secondary peak

    n = len(separations)
    print()
    print(f"  Model : {model_id}")
    print(f"  Concept: {concept}  ({n_pairs} pairs × 2 — {elapsed_s:.0f}s)")
    print(f"  Layers : {n}   Peak: L{peak_layer} ({100*peak_layer/n:.0f}% depth)   "
          f"Max sep: {max_sep:.3f}")
    print()
    print(f"  {'L':>3}  {'Separation':>{BAR_WIDTH}}   Score  CAZ")
    print(f"  {'─'*3}  {'─'*BAR_WIDTH}   {'─'*5}  {'─'*3}")

    for i, s in enumerate(separations):
        marker = caz_layers.get(i, " ")
        bar = _bar(s, max_sep)
        peak_flag = " ◀ peak" if i == peak_layer else ""
        print(f"  {i:>3}  {bar}   {s:5.3f}  {marker}{peak_flag}")

    print()
    if caz_profile.dominant:
        r = caz_profile.dominant
        print(f"  CAZ dominant:  L{r.start}–L{r.end} (peak L{r.peak}, "
              f"depth {r.depth_pct:.0f}%, score {r.caz_score:.3f})")
    else:
        print("  CAZ dominant:  none detected")
    if caz_profile.secondary:
        r = caz_profile.secondary
        print(f"  CAZ secondary: L{r.start}–L{r.end} (peak L{r.peak}, "
              f"depth {r.depth_pct:.0f}%, score {r.caz_score:.3f})")


def print_gem_summary(node) -> None:
    print()
    print("  ── GEM node ──────────────────────────────")
    print(f"  Entry layer  : L{node.caz_start} ({node.caz_start/node.n_layers_total*100:.0f}% depth)")
    print(f"  Peak layer   : L{node.caz_peak} ({node.caz_peak/node.n_layers_total*100:.0f}% depth)")
    print(f"  Handoff layer: L{node.handoff_layer} ({node.handoff_layer/node.n_layers_total*100:.0f}% depth)")
    print(f"  Entry→exit cos: {node.entry_exit_cosine:.3f}  "
          f"(>0.9 = stable direction, <0.5 = rotation in-transit)")
    print(f"  Max rot/layer: {node.max_rotation_per_layer:.3f}  "
          f"CAZ score: {node.caz_score:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")
    ap.add_argument("--concept", default=DEFAULT_CONCEPT,
                    help=f"Concept name (default: {DEFAULT_CONCEPT})")
    ap.add_argument("--n-pairs", type=int, default=DEFAULT_N_PAIRS,
                    help=f"Number of contrastive pairs to use (default: {DEFAULT_N_PAIRS})")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Batch size for forward pass (default: 4)")
    ap.add_argument("--max-length", type=int, default=256,
                    help="Token truncation length (default: 256)")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU even if CUDA is available")
    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # Import checks
    # -----------------------------------------------------------------------
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        from rosetta_tools.dataset import load_concept_pairs
        from rosetta_tools.extraction import extract_contrastive_activations
        from rosetta_tools.caz import compute_separation, compute_coherence, compute_velocity
        from rosetta_tools.gem import (
            LayerMetrics, find_caz_regions_scored,
            build_gem_node_k1, CAZRegion,
        )
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}", file=sys.stderr)
        print("  Install with: uv sync  (or pip install rosetta_tools torch transformers)",
              file=sys.stderr)
        sys.exit(1)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------------
    # Load concept pairs
    # -----------------------------------------------------------------------
    print(f"\n  Loading {args.n_pairs} '{args.concept}' pairs …", end="", flush=True)
    try:
        pairs = load_concept_pairs(args.concept, n=args.n_pairs)
    except Exception as e:
        print(f"\n[ERROR] Could not load concept pairs: {e}", file=sys.stderr)
        print("  Clone https://github.com/jamesrahenry/Rosetta_Concept_Pairs to "
              "~/Rosetta_Concept_Pairs", file=sys.stderr)
        sys.exit(1)
    pos_texts = [p.pos_text for p in pairs]
    neg_texts = [p.neg_text for p in pairs]
    print(f" {len(pairs)} pairs loaded.")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"  Loading {args.model} …", end="", flush=True)
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(
            args.model,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if device == "cpu":
            model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"\n[ERROR] Could not load {args.model}: {e}", file=sys.stderr)
        sys.exit(1)
    load_s = time.time() - t0
    vram = ""
    if device == "cuda":
        vram = f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB"
    print(f" {load_s:.1f}s.{vram}")

    # -----------------------------------------------------------------------
    # Extract layer-wise activations
    # -----------------------------------------------------------------------
    print(f"  Extracting activations ({device}) …", end="", flush=True)
    t0 = time.time()
    layer_acts = extract_contrastive_activations(
        model, tokenizer, pos_texts, neg_texts,
        device=device if device == "cpu" else str(next(model.parameters()).device),
        batch_size=args.batch_size,
        pool="last",
        max_length=args.max_length,
    )
    extract_s = time.time() - t0
    print(f" {len(layer_acts)} layers × {len(pos_texts)} pairs — {extract_s:.1f}s.")

    # -----------------------------------------------------------------------
    # Compute per-layer metrics
    # -----------------------------------------------------------------------
    separations, coherences, dom_vectors = [], [], []
    for pos, neg in layer_acts:
        pos_f = pos.astype(np.float32)
        neg_f = neg.astype(np.float32)
        S = compute_separation(pos_f, neg_f)
        C = compute_coherence(pos_f, neg_f)
        diff = pos_f.mean(axis=0) - neg_f.mean(axis=0)
        norm = np.linalg.norm(diff)
        dom = (diff / norm).tolist() if norm > 0 else diff.tolist()
        separations.append(S)
        coherences.append(C)
        dom_vectors.append(dom)

    vel_array = compute_velocity(separations, window=3)
    n_layers = len(separations)
    peak_layer = int(np.argmax(separations))

    # -----------------------------------------------------------------------
    # CAZ detection
    # -----------------------------------------------------------------------
    layer_metrics = [
        LayerMetrics(layer=i, separation=separations[i],
                     coherence=coherences[i], velocity=float(vel_array[i]))
        for i in range(n_layers)
    ]
    caz_profile = find_caz_regions_scored(layer_metrics, attention_paradigm="mha")

    total_s = load_s + extract_s
    print_profile(separations, caz_profile, args.model, args.concept,
                  len(pairs), total_s)

    # -----------------------------------------------------------------------
    # GEM node (requires dominant CAZ)
    # -----------------------------------------------------------------------
    if caz_profile.dominant:
        caz_data = {
            "layer_data": {
                "n_layers": n_layers,
                "metrics": [
                    {
                        "layer": i,
                        "separation_fisher": separations[i],
                        "coherence": coherences[i],
                        "dom_vector": dom_vectors[i],
                        "velocity": float(vel_array[i]),
                    }
                    for i in range(n_layers)
                ],
            },
            "peak_layer": peak_layer,
            "peak_separation": separations[peak_layer],
            "n_pairs": len(pairs),
        }
        try:
            node = build_gem_node_k1(caz_data, caz_profile.dominant,
                                      caz_index=0, attention_paradigm="mha")
            print_gem_summary(node)
        except Exception as e:
            print(f"  [WARN] GEM node build failed: {e}")

    print()


if __name__ == "__main__":
    main()

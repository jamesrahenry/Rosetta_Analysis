#!/usr/bin/env python3
"""Extract calibration activations at proportional depths for P5 CKA test.

Companion to scratch/p5_validation_battery.py test 1 — but with REAL CKA
on activation matrices instead of windowed CKA on dom_vector trajectories.

For each model in rosetta_data/models/, for each of 7 concepts:
  1. Load contrastive pairs from Rosetta_Concept_Pairs/pairs/raw/v1/.
  2. Tokenize and forward through model with output_hidden_states.
  3. At proportional depths {0.3, 0.5, 0.7} of the model's layer count,
     last-token-pool activations.
  4. Save as rosetta_data/models/<model>/cka_acts_<concept>.npz with:
       acts: [n_examples, 3, hidden_dim] float32
       labels: [n_examples] int (0 = neg, 1 = pos)
       depth_layers: [3] int — actual layer indices used
       depth_fractions: [3] float — {0.3, 0.5, 0.7}

Run as a GPU job. Skips models/concepts already extracted (idempotent).

Usage:
  python scratch/p5_cka_extract.py [--model <model_dir_name>] [--concept <c>]
                                   [--limit <n_examples>] [--batch-size <b>]

If --model omitted, processes all model dirs in rosetta_data/models/.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path.home()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONCEPTS = [
    "credibility", "certainty", "causation",
    "temporal_order", "negation", "sentiment", "moral_valence",
]
DEPTHS = [0.3, 0.5, 0.7]
DEFAULT_LIMIT = 200  # n positives = n negatives = limit
DEFAULT_MAX_LEN = 512
DEFAULT_BATCH = 8

DEFAULT_DATA_ROOT = REPO_ROOT / "rosetta_data" / "models"
DEFAULT_PAIRS_ROOT = REPO_ROOT / "Rosetta_Concept_Pairs" / "pairs" / "raw" / "v1"


def load_concept_pairs(concept: str, limit: int, pairs_root: Path) -> tuple[list[str], list[int]]:
    """Load up to `limit` pos + `limit` neg texts for a concept.

    Pairs are deduplicated by (text, label) — multiple model authorings of the
    same pair may include semantic duplicates; we keep all unique strings
    until reaching `limit` per class.
    """
    p = pairs_root / f"{concept}_consensus_pairs.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"No concept pair file: {p}")
    pos, neg = [], []
    seen_pos, seen_neg = set(), set()
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("concept") != concept:
            continue
        text = rec.get("text", "")
        label = rec.get("label")
        if label == 1 and text not in seen_pos and len(pos) < limit:
            seen_pos.add(text)
            pos.append(text)
        elif label == 0 and text not in seen_neg and len(neg) < limit:
            seen_neg.add(text)
            neg.append(text)
        if len(pos) >= limit and len(neg) >= limit:
            break
    n = min(len(pos), len(neg))
    log.info("  loaded %d pos + %d neg (using %d each = %d total)",
             len(pos), len(neg), n, n * 2)
    texts = pos[:n] + neg[:n]
    labels = [1] * n + [0] * n
    return texts, labels


def proportional_depth_layers(n_layers: int, fractions: list[float]) -> list[int]:
    return [int(np.clip(round(f * (n_layers - 1)), 0, n_layers - 1))
            for f in fractions]


def already_extracted(model_dir: Path, concept: str) -> bool:
    return (model_dir / f"cka_acts_{concept}.npz").exists()


def model_id_from_dir(model_dir: Path) -> str:
    """Convert e.g. 'EleutherAI_pythia_6.9b' → 'EleutherAI/pythia-6.9b'.

    Handles the '_' → '/' (first occurrence) and '_' → '-' (rest) convention
    used by extraction scripts in this project.
    """
    name = model_dir.name
    if "_" not in name:
        return name
    org, rest = name.split("_", 1)
    rest = rest.replace("_", "-")
    # Special-case the openai_community → openai-community-style.
    return f"{org}/{rest}"


def extract_one_model(model_dir: Path, concepts: list[str], limit: int,
                       batch_size: int, max_length: int, pairs_root: Path):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from rosetta_tools.extraction import extract_layer_activations

    model_id = model_id_from_dir(model_dir)
    log.info("=== %s (id=%s) ===", model_dir.name, model_id)

    # Determine which concepts still need extraction.
    todo = [c for c in concepts if not already_extracted(model_dir, c)]
    if not todo:
        log.info("  all concepts already extracted, skipping load")
        return

    log.info("  loading model …")
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()
    except Exception as e:
        log.error("  FAILED to load %s: %s", model_id, e)
        return
    log.info("  loaded in %.1fs", time.time() - t0)

    # Determine n_layers from a tiny probe.
    with torch.no_grad():
        probe = tokenizer("hello", return_tensors="pt").to(
            next(model.parameters()).device)
        out = model(**probe, output_hidden_states=True)
    n_layers = len(out.hidden_states)
    depth_layers = proportional_depth_layers(n_layers, DEPTHS)
    log.info("  n_layers=%d → depth layers %s for %s",
             n_layers, depth_layers, DEPTHS)

    for concept in todo:
        out_path = model_dir / f"cka_acts_{concept}.npz"
        log.info("  --- %s ---", concept)
        try:
            texts, labels = load_concept_pairs(concept, limit, pairs_root)
        except FileNotFoundError as e:
            log.warning("    skip: %s", e)
            continue

        device = str(next(model.parameters()).device)
        t0 = time.time()
        all_layers = extract_layer_activations(
            model, tokenizer, texts,
            device=device, batch_size=batch_size,
            pool="last", max_length=max_length,
        )
        log.info("    forward done in %.1fs (n=%d)", time.time() - t0, len(texts))

        # Pick out only the proportional-depth layers; stack into [n, 3, dim].
        slices = [all_layers[li] for li in depth_layers]
        acts = np.stack(slices, axis=1).astype(np.float32)  # [n, 3, dim]

        np.savez_compressed(
            out_path,
            acts=acts,
            labels=np.asarray(labels, dtype=np.int8),
            depth_layers=np.asarray(depth_layers, dtype=np.int32),
            depth_fractions=np.asarray(DEPTHS, dtype=np.float32),
            n_layers=np.asarray([n_layers], dtype=np.int32),
            model_id=np.asarray([model_id]),
        )
        log.info("    saved %s (acts shape %s)", out_path.name, acts.shape)
        del all_layers, slices, acts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="Model dir name (default: all)")
    ap.add_argument("--concept", default=None,
                    help="Concept (default: all 7)")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help=f"N pos + N neg per concept (default: {DEFAULT_LIMIT})")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LEN)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                    help=f"Path to rosetta_data/models (default: {DEFAULT_DATA_ROOT})")
    ap.add_argument("--pairs-root", type=Path, default=DEFAULT_PAIRS_ROOT,
                    help=f"Path to concept pair jsonl files (default: {DEFAULT_PAIRS_ROOT})")
    args = ap.parse_args()

    data_root: Path = args.data_root
    pairs_root: Path = args.pairs_root
    log.info("data_root=%s  pairs_root=%s", data_root, pairs_root)

    concepts = [args.concept] if args.concept else CONCEPTS

    if args.model:
        model_dirs = [data_root / args.model]
    else:
        model_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    log.info("Targets: %d model dir(s) × %d concept(s)",
             len(model_dirs), len(concepts))

    # Smoke-test CUDA before looping 34 models — catches missing libs early.
    try:
        import torch
        if torch.cuda.is_available():
            torch.zeros(1).cuda()
    except Exception as e:
        log.error("CUDA init failed — aborting: %s", e)
        sys.exit(1)

    succeeded = 0
    for md in model_dirs:
        if not md.exists():
            log.warning("Skip missing model dir: %s", md)
            continue
        try:
            extract_one_model(md, concepts, args.limit,
                              args.batch_size, args.max_length, pairs_root)
            succeeded += 1
        except Exception as e:
            log.error("FAILED on %s: %s", md.name, e)
            continue

    log.info("All done. %d/%d model(s) processed.", succeeded, len(model_dirs))
    if succeeded == 0:
        log.error("Zero models extracted — check CUDA and model dirs.")
        sys.exit(1)


if __name__ == "__main__":
    main()

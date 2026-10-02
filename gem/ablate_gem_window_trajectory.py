"""
ablate_gem_window_trajectory.py
===============================
Does following a GEM's rotation matter?  Whole-window trajectory ablation
vs. static-direction window ablation.

GEM ablation normally projects out one static ``settled_direction`` (at a
single site or across a layer window).  A GEM, however, records a
layer-indexed trajectory u^(l) of the concept direction across its CAZ
(the concept thread's ``directions``).  Mean entry-exit cosine is ~0.29, so
the direction rotates substantially during assembly.  Hypothesis under test:
projecting out the *layer-matched* u^(l) at each layer of the CAZ window
suppresses the concept more than projecting out the static u_settled at the
same layers.

Arms (all measured as final-layer separation reduction, same pairs):
  site_settled   : u_settled at the handoff layer only (width 1)
  window_static  : u_settled at every layer in [caz_start, caz_end]
  window_traj    : u^(l) at each layer l in [caz_start, caz_end]   <-- key arm
  window_shuffled: u^(l) assigned to the wrong layers (permuted)   <-- control:
                   same set of directions, trajectory order destroyed
  window_random  : random unit direction per layer (n seeds)       <-- floor

Key contrast: window_traj vs window_static (identical layer set; only the
direction schedule differs).  window_shuffled isolates ordering from
direction content.  Nodes with a CAZ window of < 2 layers are skipped (the
trajectory degenerates to the static direction).

Pilot scope: small models that fit a 4 GB GPU.  This generates a hypothesis;
it is not a paper-grade result at this model coverage.

Usage
-----
    python gem/ablate_gem_window_trajectory.py --model EleutherAI/pythia-160m
    python gem/ablate_gem_window_trajectory.py --model openai-community/gpt2 \
        --concepts sentiment negation --n-pairs 100

Written: 2026-10-02 UTC
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablate_gem_random_window_null import baseline_separation, load_gem  # noqa: E402
from rosetta_tools.ablation import DirectionalAblator, get_transformer_layers
from rosetta_tools.caz import compute_separation
from rosetta_tools.dataset import load_concept_pairs, texts_by_label
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.gem import discover_concepts, find_extraction_dir
from rosetta_tools.gpu_utils import (
    NumpyJSONEncoder, get_device, get_dtype, load_causal_lm, log_device_info,
    purge_hf_cache, release_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUT_DIR = Path.home() / "rosetta_data" / "results" / "gem_window_trajectory"
BATCH_SIZE = 4


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def ablate_schedule_and_measure(
    model, tokenizer, layers: list, schedule: dict[int, np.ndarray],
    pos_texts: list[str], neg_texts: list[str], device: str,
) -> float:
    """Ablate schedule[layer] at each layer simultaneously; return final-layer sep."""
    dtype = next(model.parameters()).dtype
    with ExitStack() as stack:
        for li, d in schedule.items():
            stack.enter_context(DirectionalAblator(layers[li], d, dtype=dtype))
        pos = extract_layer_activations(
            model, tokenizer, pos_texts, device=device,
            batch_size=BATCH_SIZE, pool="last",
        )
        neg = extract_layer_activations(
            model, tokenizer, neg_texts, device=device,
            batch_size=BATCH_SIZE, pool="last",
        )
    return float(compute_separation(pos[-1], neg[-1]))


def run_node(
    model, tokenizer, layers, node: dict, baseline: float,
    pos_texts, neg_texts, device: str, rng: np.random.Generator,
    n_seeds: int, n_shuffles: int,
) -> dict | None:
    start, end = node["caz_start"], node["caz_end"]
    window = list(range(start, end + 1))
    if len(window) < 2:
        return None

    threads = node.get("threads") or []
    tid = node.get("concept_thread_id")
    thread = next((t for t in threads if t.get("thread_id") == tid), None)
    if thread is None:
        return None
    by_layer = {int(l): _unit(d) for l, d in zip(thread["layer_indices"], thread["directions"])}
    if any(l not in by_layer for l in window) or max(window) >= len(layers):
        return None

    settled = _unit(node["settled_direction"])
    traj = {l: by_layer[l] for l in window}
    dim = settled.shape[0]

    def reduction(schedule: dict[int, np.ndarray]) -> float:
        sep = ablate_schedule_and_measure(
            model, tokenizer, layers, schedule, pos_texts, neg_texts, device)
        return max(0.0, (baseline - sep) / baseline)

    handoff = node["handoff_layer"]
    arms = {
        "site_settled": reduction({handoff: settled}) if handoff < len(layers) else None,
        "window_static": reduction({l: settled for l in window}),
        "window_traj": reduction(traj),
    }

    shuffled = []
    for _ in range(n_shuffles):
        perm = rng.permutation(len(window))
        if len(window) > 1 and np.array_equal(perm, np.arange(len(window))):
            perm = np.roll(perm, 1)
        shuffled.append(reduction({l: traj[window[p]] for l, p in zip(window, perm)}))
    arms["window_shuffled"] = float(np.mean(shuffled))

    rand = [reduction({l: _unit(rng.standard_normal(dim)) for l in window})
            for _ in range(n_seeds)]
    arms["window_random"] = float(np.mean(rand))

    path_cos = [float(by_layer[a] @ by_layer[b]) for a, b in zip(window[:-1], window[1:])]
    return {
        "caz_index": node.get("caz_index"),
        "window": window,
        "handoff_layer": handoff,
        "entry_exit_cosine": node.get("entry_exit_cosine"),
        "mean_adjacent_cos": float(np.mean(path_cos)),
        "traj_vs_settled_cos": [float(by_layer[l] @ settled) for l in window],
        "arms": arms,
        "shuffled_all": shuffled,
        "random_all": rand,
        "traj_minus_static": arms["window_traj"] - arms["window_static"],
        "traj_beats_static": bool(arms["window_traj"] > arms["window_static"]),
    }


def run_model(model_id: str, args, rng: np.random.Generator) -> None:
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.warning("No extraction dir for %s", model_id)
        return
    out_path = OUT_DIR / f"{extraction_dir.name}_window_trajectory.json"
    if out_path.exists() and not args.overwrite:
        log.info("Already done: %s", model_id)
        return

    device = get_device(args.device)
    dtype = get_dtype(args.dtype, device)
    log_device_info(device, dtype)
    model, tokenizer = load_causal_lm(model_id, device, dtype)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    layers = get_transformer_layers(model)

    results = []
    concepts = args.concepts or discover_concepts(extraction_dir)
    for concept in concepts:
        gem = load_gem(extraction_dir, concept)
        if gem is None:
            continue
        pairs = load_concept_pairs(concept, n=args.n_pairs)
        pos_texts, neg_texts = texts_by_label(pairs)
        baseline = baseline_separation(model, tokenizer, pos_texts, neg_texts, device)
        if baseline <= 0:
            log.warning("  zero baseline for %s, skipping", concept)
            continue
        for node in gem["nodes"]:
            r = run_node(model, tokenizer, layers, node, baseline, pos_texts,
                         neg_texts, device, rng, args.n_seeds, args.n_shuffles)
            if r is None:
                continue
            r.update(concept=concept, model_id=model_id, baseline_sep=baseline)
            results.append(r)
            a = r["arms"]
            log.info("  %-18s node%s L%d-%d  traj=%.3f static=%.3f shuf=%.3f rand=%.3f site=%s",
                     concept, r["caz_index"], r["window"][0], r["window"][-1],
                     a["window_traj"], a["window_static"], a["window_shuffled"],
                     a["window_random"],
                     "n/a" if a["site_settled"] is None else f"{a['site_settled']:.3f}")

    if results:
        wins = sum(r["traj_beats_static"] for r in results)
        log.info("%s: traj > static in %d/%d nodes, mean diff %.4f",
                 model_id, wins, len(results),
                 float(np.mean([r["traj_minus_static"] for r in results])))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"model_id": model_id, "n_pairs": args.n_pairs, "n_seeds": args.n_seeds,
         "n_shuffles": args.n_shuffles, "results": results},
        cls=NumpyJSONEncoder, indent=2))
    log.info("Wrote %s", out_path)

    release_model(model)
    if not args.no_clean_cache:
        purge_hf_cache(model_id)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", action="append", required=True, help="HF model id (repeatable)")
    ap.add_argument("--concepts", nargs="*", default=None)
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-clean-cache", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    for m in args.model:
        run_model(m, args, rng)


if __name__ == "__main__":
    main()

"""
ablate_gem_random_window_null.py
================================
Reviewer null: does GEM handoff ablation outperform a random width-matched window?

The current GEM comparison (handoff vs. peak) shows that a width=3 window
at the handoff layer beats a width=3 window at the CAZ peak.  A reviewer
will ask: "is the spatial targeting doing work, or does any width=3 window
produce the same effect?"

This script answers by ablating N random contiguous windows of the same
width as the GEM handoff window and comparing the resulting suppression
distribution to the observed GEM handoff result.

Method
------
For each (model, concept) pair with existing GEM + ablation_gem data:
  1. Load the handoff ablation result (final_sep_reduction) from ablation_gem.
  2. Load the GEM node's settled_direction and window width.
  3. Sample N_WINDOWS random contiguous windows of the same width from the
     model's layer range.
  4. For each random window, ablate the settled_direction across all window
     layers, measure final-layer separation.
  5. Report: observed GEM reduction vs. null distribution (mean, p95, p-value).

For efficiency, runs on the base-model subset only (instruct variants share
the same architecture signal and add runtime without adding information).

Usage
-----
    cd ~/caz_scaling
    python src/ablate_gem_random_window_null.py
    python src/ablate_gem_random_window_null.py --n-windows 200
    python src/ablate_gem_random_window_null.py --model EleutherAI/pythia-1.4b

Written: 2026-04-21 UTC
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta_tools.ablation import DirectionalAblator, get_transformer_layers
from rosetta_tools.caz import compute_separation
from rosetta_tools.dataset import load_concept_pairs, texts_by_label
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.gem import discover_concepts, discover_base_models, find_extraction_dir
from rosetta_tools.gpu_utils import (
    get_device, get_dtype, log_device_info, log_vram,
    release_model, purge_hf_cache, NumpyJSONEncoder,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUT_DIR = Path.home() / "rosetta_data" / "results" / "gem_random_window_null"

N_WINDOWS_DEFAULT = 100
N_PAIRS = 50
BATCH_SIZE = 4


def load_gem_ablation(extraction_dir: Path, concept: str) -> dict | None:
    path = extraction_dir / f"ablation_gem_{concept}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_gem(extraction_dir: Path, concept: str) -> dict | None:
    path = extraction_dir / f"gem_{concept}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def ablate_window_and_measure(
    model,
    tokenizer,
    layers: list,
    window_layers: list[int],
    direction: np.ndarray,
    pos_texts: list[str],
    neg_texts: list[str],
    device: str,
) -> float:
    """Ablate direction at all window_layers simultaneously, return final-layer sep."""
    dtype = next(model.parameters()).dtype
    dir_t = torch.tensor(direction, dtype=dtype, device=device)
    dir_t = dir_t / dir_t.norm()

    with ExitStack() as stack:
        for li in window_layers:
            stack.enter_context(DirectionalAblator(layers[li], dir_t, dtype=dtype))
        pos_acts = extract_layer_activations(
            model, tokenizer, pos_texts, device=device,
            batch_size=BATCH_SIZE, pool="last",
        )
        neg_acts = extract_layer_activations(
            model, tokenizer, neg_texts, device=device,
            batch_size=BATCH_SIZE, pool="last",
        )
    return float(compute_separation(pos_acts[-1], neg_acts[-1]))


def baseline_separation(model, tokenizer, pos_texts, neg_texts, device) -> float:
    pos_acts = extract_layer_activations(
        model, tokenizer, pos_texts, device=device,
        batch_size=BATCH_SIZE, pool="last",
    )
    neg_acts = extract_layer_activations(
        model, tokenizer, neg_texts, device=device,
        batch_size=BATCH_SIZE, pool="last",
    )
    return float(compute_separation(pos_acts[-1], neg_acts[-1]))


# ---------------------------------------------------------------------------
# Per-concept run
# ---------------------------------------------------------------------------

def run_concept(
    model,
    tokenizer,
    concept: str,
    extraction_dir: Path,
    n_windows: int,
    device: str,
    rng: np.random.Generator,
) -> dict | None:
    gem = load_gem(extraction_dir, concept)
    if gem is None:
        log.warning("  No GEM data for %s, skipping", concept)
        return None

    layers = get_transformer_layers(model)
    n_layers = len(layers)

    # Get the settled direction and handoff window from GEM
    targets = gem.get("ablation_targets", [0])
    node = gem["nodes"][targets[0]]
    handoff_layer = node["handoff_layer"]
    direction = np.array(node["settled_direction"], dtype=np.float64)
    direction /= np.linalg.norm(direction)

    # Width: prefer ablation_gem if available, else default 3
    abl = load_gem_ablation(extraction_dir, concept)
    if abl is not None:
        ablation_layers = abl.get("handoff", {}).get("ablation_layers", [])
        width = len(ablation_layers) if ablation_layers else 3
    else:
        width = 3
    half = width // 2
    handoff_window = list(range(
        max(0, handoff_layer - half),
        min(n_layers, handoff_layer + (width - half)),
    ))

    pairs = load_concept_pairs(concept, n=N_PAIRS)
    pos_texts, neg_texts = texts_by_label(pairs)

    baseline_sep = baseline_separation(model, tokenizer, pos_texts, neg_texts, device)
    if baseline_sep <= 0:
        log.warning("  Zero baseline for %s, skipping", concept)
        return None

    # Compute GEM handoff reduction directly (don't require pre-run ablation_gem)
    handoff_sep = ablate_window_and_measure(
        model, tokenizer, layers, handoff_window, direction,
        pos_texts, neg_texts, device,
    )
    handoff_reduction = max(0.0, (baseline_sep - handoff_sep) / baseline_sep)

    log.info("  %s | GEM handoff reduction=%.4f | width=%d | n_windows=%d",
             concept, handoff_reduction, width, n_windows)

    # Sample random windows and measure suppression
    null_reductions = []
    for i in range(n_windows):
        # Random contiguous window of same width, excluding the actual handoff window
        max_start = n_layers - width
        if max_start <= 0:
            break
        window_start = int(rng.integers(0, max_start + 1))
        window = list(range(window_start, window_start + width))

        ablated_sep = ablate_window_and_measure(
            model, tokenizer, layers, window, direction,
            pos_texts, neg_texts, device,
        )
        reduction = max(0.0, (baseline_sep - ablated_sep) / baseline_sep)
        null_reductions.append(float(reduction))

        if (i + 1) % 20 == 0:
            log.info("    window %d/%d  mean_null=%.4f", i + 1, n_windows, np.mean(null_reductions))

    null_arr = np.array(null_reductions)
    p_value = float(np.mean(null_arr >= handoff_reduction))
    ratio = handoff_reduction / max(float(null_arr.mean()), 1e-6)

    return {
        "concept": concept,
        "handoff_reduction": round(handoff_reduction, 4),
        "null_mean": round(float(null_arr.mean()), 4),
        "null_p95": round(float(np.percentile(null_arr, 95)), 4),
        "p_value": round(p_value, 6),
        "ratio_vs_null": round(ratio, 3),
        "handoff_better": bool(handoff_reduction > null_arr.mean()),
        "width": width,
        "n_windows": len(null_reductions),
        "null_reductions": null_reductions,
    }


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(model_id: str, args, rng: np.random.Generator) -> None:
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.warning("No extraction dir for %s", model_id)
        return

    out_path = OUT_DIR / f"{extraction_dir.name}_random_window_null.json"
    if out_path.exists() and not args.overwrite:
        log.info("Already done: %s", model_id)
        return

    log.info("=== %s ===", model_id)
    device = get_device(args.device)
    dtype = get_dtype(args.dtype, device)
    log_device_info(device, dtype)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, device_map=device)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype).to(device)
        model.eval()
    except Exception as e:
        log.error("Failed to load %s: %s", model_id, e)
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for concept in discover_concepts(extraction_dir):
        r = run_concept(model, tokenizer, concept, extraction_dir,
                        args.n_windows, device, rng)
        if r:
            r["model_id"] = model_id
            results.append(r)

    # Summary line
    if results:
        wins = sum(1 for r in results if r["handoff_better"])
        mean_ratio = np.mean([r["ratio_vs_null"] for r in results])
        sig = sum(1 for r in results if r["p_value"] < 0.05)
        log.info("  %s: %d/%d wins  mean_ratio=%.2f×  sig=%d/%d",
                 model_id.split("/")[-1], wins, len(results), mean_ratio, sig, len(results))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"model_id": model_id, "results": results}, cls=NumpyJSONEncoder, indent=2))
    log.info("Wrote %s", out_path)

    release_model(model)
    if not args.no_clean_cache:
        purge_hf_cache(model_id)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate(out_dir: Path) -> None:
    records = []
    for f in sorted(out_dir.glob("*_random_window_null.json")):
        d = json.loads(f.read_text())
        records.extend(d.get("results", []))

    if not records:
        log.warning("No results to aggregate")
        return

    wins = sum(1 for r in records if r["handoff_better"])
    ratios = [r["ratio_vs_null"] for r in records]
    sig = sum(1 for r in records if r["p_value"] < 0.05)
    nulls = [r["null_mean"] for r in records]
    obs = [r["handoff_reduction"] for r in records]

    lines = [
        "GEM handoff vs. matched random-window null",
        f"N records: {len(records)}",
        f"Handoff > null: {wins}/{len(records)} ({100*wins/len(records):.1f}%)",
        f"Significant (p<0.05): {sig}/{len(records)}",
        f"Grand mean observed: {np.mean(obs):.4f}",
        f"Grand mean null: {np.mean(nulls):.4f}",
        f"Grand mean ratio: {np.mean(ratios):.2f}×",
        "",
        f"{'Model':<30} {'Concept':<16} {'GEM':>6} {'Null':>6} {'Ratio':>6} {'p':>8} {'Win'}",
        "-" * 80,
    ]
    for r in sorted(records, key=lambda x: (x["model_id"], x["concept"])):
        lines.append(
            f"{r['model_id'].split('/')[-1]:<30} {r['concept']:<16} "
            f"{r['handoff_reduction']:>6.3f} {r['null_mean']:>6.3f} "
            f"{r['ratio_vs_null']:>5.2f}× {r['p_value']:>8.4f} "
            f"{'✓' if r['handoff_better'] else '✗'}"
        )

    text = "\n".join(lines)
    print(text)
    (out_dir / "gem_random_window_null_summary.txt").write_text(text)
    slim = [{k: v for k, v in r.items() if k != "null_reductions"} for r in records]
    (out_dir / "gem_random_window_null.json").write_text(json.dumps(slim, indent=2))
    log.info("Wrote aggregate to %s", out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--n-windows", type=int, default=N_WINDOWS_DEFAULT)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-clean-cache", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    models = discover_base_models() if args.all else [args.model]
    log.info("Running on %d models", len(models))

    for model_id in models:
        run_model(model_id, args, rng)

    aggregate(OUT_DIR)


if __name__ == "__main__":
    main()

"""
ablate_gem_adaptive_width.py
============================
Test architecture-aware width selection for GEM ablation.

Motivation
----------
Width sensitivity analysis revealed one robust failure mode for fixed w=3:

  Near-final handoffs (HL/N > 0.85): window dilutes concentrated terminal-layer
  signal with earlier, less-concentrated layers — w=1 wins by 2–5pp.

  A Gemma-2 alternating-attention parity rule (w=1 on global layers) was
  hypothesized but falsified by direct experiment: mid-model global-layer
  handoffs in gemma-2-2b prefer w=3 over w=1 by +3.72pp on average.  The SWA
  neighbors contribute signal rather than diluting it.

Adaptive width rule (updated after Gemma-2-2b validation)
----------------------------------------------------------
  if HL/N > 0.85  → w=1  (near-final, no downstream runway)
  else            → w=3  (default)

For each concept this script measures:
  - adaptive_width ablation
  - fixed w=3 ablation (direct intra-run comparison, not cross-run)
  - N_WINDOWS random windows at adaptive_width (null distribution)

Output keys per record include adaptive_width, delta_vs_fixed3, ratio_vs_null,
enabling direct comparison with the random-window-null sweep results.

Usage
-----
    cd ~/caz_scaling
    python src/ablate_gem_adaptive_width.py --model google/gemma-2-2b
    python src/ablate_gem_adaptive_width.py --model facebook/opt-6.7b
    python src/ablate_gem_adaptive_width.py --all
    python src/ablate_gem_adaptive_width.py --all --n-windows 50

Written: 2026-04-22 UTC
"""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta_tools.ablation import DirectionalAblator, get_transformer_layers
from rosetta_tools.caz import compute_separation
from rosetta_tools.dataset import load_pairs, texts_by_label
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.gpu_utils import (
    get_device, get_dtype, log_device_info,
    release_model, purge_hf_cache, NumpyJSONEncoder,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_ROOT = Path("results")
DATA_ROOT = Path(__file__).parent.parent / "data"
OUT_DIR = RESULTS_ROOT / "gem_adaptive_width"

N_WINDOWS_DEFAULT = 100
N_PAIRS = 50
BATCH_SIZE = 4

CONCEPTS = [
    "causation", "certainty", "credibility", "moral_valence",
    "negation", "sentiment", "temporal_order",
]
CONCEPT_DATASETS = {
    "causation": "causation_pairs.jsonl",
    "certainty": "certainty_pairs.jsonl",
    "credibility": "credibility_pairs.jsonl",
    "moral_valence": "moral_valence_pairs.jsonl",
    "negation": "negation_pairs.jsonl",
    "sentiment": "sentiment_pairs.jsonl",
    "temporal_order": "temporal_order_pairs.jsonl",
}

# ---------------------------------------------------------------------------
# Width selection
# ---------------------------------------------------------------------------

def adaptive_width(handoff_layer: int, n_layers: int, model_id: str) -> int:
    """
    Near-final handoffs concentrate signal at the terminal layer; a wider
    window averages in earlier less-concentrated layers and hurts.
    Gemma-2 parity rule (w=1 on global layers) was tested and falsified —
    mid-model SWA neighbors contribute signal rather than diluting it.
    """
    if handoff_layer / max(n_layers, 1) > 0.85:
        return 1
    return 3


def width_to_window(handoff_layer: int, width: int, n_layers: int) -> list[int]:
    half = width // 2
    start = max(0, handoff_layer - half)
    end = min(n_layers, handoff_layer + (width - half))
    return list(range(start, end))


# ---------------------------------------------------------------------------
# Discovery (same as other GEM scripts)
# ---------------------------------------------------------------------------

def find_extraction_dir(model_id: str) -> Path | None:
    candidates = []
    for d in RESULTS_ROOT.iterdir():
        s = d / "run_summary.json"
        if d.is_dir() and s.exists() and list(d.glob("gem_*.json")):
            try:
                if json.loads(s.read_text()).get("model_id") == model_id:
                    candidates.append((d.stat().st_mtime, d))
            except Exception:
                continue
    return max(candidates, key=lambda x: x[0])[1] if candidates else None


def discover_base_models() -> list[str]:
    models = set()
    for d in RESULTS_ROOT.iterdir():
        s = d / "run_summary.json"
        if s.exists():
            try:
                mid = json.loads(s.read_text()).get("model_id", "")
                if mid and not any(t in mid for t in ["Instruct", "instruct", "-it"]):
                    models.add(mid)
            except Exception:
                pass
    return sorted(models)


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def ablate_and_measure(model, tokenizer, layers, window_layers, direction,
                       pos_texts, neg_texts, device) -> float:
    dtype = next(model.parameters()).dtype
    dir_t = torch.tensor(direction, dtype=dtype, device=device)
    dir_t = dir_t / dir_t.norm()
    with ExitStack() as stack:
        for li in window_layers:
            stack.enter_context(DirectionalAblator(layers[li], dir_t, dtype=dtype))
        pos_acts = extract_layer_activations(
            model, tokenizer, pos_texts, device=device,
            batch_size=BATCH_SIZE, pool="last")
        neg_acts = extract_layer_activations(
            model, tokenizer, neg_texts, device=device,
            batch_size=BATCH_SIZE, pool="last")
    return float(compute_separation(pos_acts[-1], neg_acts[-1]))


def baseline_sep(model, tokenizer, pos_texts, neg_texts, device) -> float:
    pos_acts = extract_layer_activations(
        model, tokenizer, pos_texts, device=device,
        batch_size=BATCH_SIZE, pool="last")
    neg_acts = extract_layer_activations(
        model, tokenizer, neg_texts, device=device,
        batch_size=BATCH_SIZE, pool="last")
    return float(compute_separation(pos_acts[-1], neg_acts[-1]))


# ---------------------------------------------------------------------------
# Per-concept run
# ---------------------------------------------------------------------------

def run_concept(model, tokenizer, concept: str, extraction_dir: Path,
                model_id: str, n_windows: int, device: str,
                rng: np.random.Generator) -> dict | None:
    gem_path = extraction_dir / f"gem_{concept}.json"
    if not gem_path.exists():
        return None
    gem = json.loads(gem_path.read_text())
    targets = gem.get("ablation_targets", [0])
    node = gem["nodes"][targets[0]]
    handoff_layer = node["handoff_layer"]
    direction = np.array(node["settled_direction"], dtype=np.float64)
    direction /= np.linalg.norm(direction)

    layers = get_transformer_layers(model)
    n_layers = len(layers)

    dataset_path = DATA_ROOT / CONCEPT_DATASETS[concept]
    pairs = load_pairs(dataset_path)[:N_PAIRS]
    pos_texts, neg_texts = texts_by_label(pairs)

    b_sep = baseline_sep(model, tokenizer, pos_texts, neg_texts, device)
    if b_sep <= 0:
        log.warning("  Zero baseline for %s, skipping", concept)
        return None

    def reduction(s: float) -> float:
        return max(0.0, (b_sep - s) / b_sep)

    # Adaptive width
    w_adapt = adaptive_width(handoff_layer, n_layers, model_id)
    window_adapt = width_to_window(handoff_layer, w_adapt, n_layers)
    sep_adapt = ablate_and_measure(model, tokenizer, layers, window_adapt,
                                   direction, pos_texts, neg_texts, device)
    red_adapt = reduction(sep_adapt)

    # Fixed w=3 (intra-run baseline)
    window_w3 = width_to_window(handoff_layer, 3, n_layers)
    sep_w3 = ablate_and_measure(model, tokenizer, layers, window_w3,
                                direction, pos_texts, neg_texts, device)
    red_w3 = reduction(sep_w3)

    # Null distribution at adaptive width
    null_reductions = []
    max_start = n_layers - w_adapt
    if max_start > 0:
        for _ in range(n_windows):
            start = int(rng.integers(0, max_start + 1))
            window = list(range(start, start + w_adapt))
            s = ablate_and_measure(model, tokenizer, layers, window,
                                   direction, pos_texts, neg_texts, device)
            null_reductions.append(reduction(s))

    null_arr = np.array(null_reductions) if null_reductions else np.array([0.0])
    p_value = float(np.mean(null_arr >= red_adapt))
    ratio = red_adapt / max(float(null_arr.mean()), 1e-6)
    delta_vs_w3 = round(100 * (red_adapt - red_w3), 2)

    rel_depth = round(handoff_layer / n_layers, 3)
    rule = "near-final" if w_adapt == 1 else "default"

    log.info("  %s | HL=%d (%.0f%%, %s) w_adapt=%d | "
             "adapt=%.3f w3=%.3f delta=%+.1fpp ratio=%.2fx p=%.3f",
             concept, handoff_layer, 100 * rel_depth, rule,
             w_adapt, red_adapt, red_w3, delta_vs_w3, ratio, p_value)

    return {
        "concept": concept,
        "handoff_layer": handoff_layer,
        "n_layers": n_layers,
        "rel_depth": rel_depth,
        "rule": rule,
        "adaptive_width": w_adapt,
        "baseline_sep": round(b_sep, 4),
        "adaptive_reduction": round(red_adapt, 4),
        "fixed_w3_reduction": round(red_w3, 4),
        "delta_vs_fixed3_pp": delta_vs_w3,
        "adaptive_better": bool(red_adapt > red_w3),
        "null_mean": round(float(null_arr.mean()), 4),
        "null_p95": round(float(np.percentile(null_arr, 95)), 4),
        "p_value": round(p_value, 6),
        "ratio_vs_null": round(ratio, 3),
        "n_windows": len(null_reductions),
    }


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(model_id: str, args, rng: np.random.Generator) -> None:
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.warning("No extraction dir for %s", model_id)
        return

    out_path = OUT_DIR / f"{extraction_dir.name}_adaptive_width.json"
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for concept in CONCEPTS:
        r = run_concept(model, tokenizer, concept, extraction_dir,
                        model_id, args.n_windows, device, rng)
        if r:
            r["model_id"] = model_id
            results.append(r)

    if results:
        better = sum(1 for r in results if r["adaptive_better"])
        mean_delta = np.mean([r["delta_vs_fixed3_pp"] for r in results])
        mean_ratio = np.mean([r["ratio_vs_null"] for r in results])
        log.info("  %s: adaptive better %d/%d  mean_delta=%+.2fpp  mean_ratio=%.2fx",
                 model_id.split("/")[-1], better, len(results), mean_delta, mean_ratio)

    out_path.write_text(json.dumps(
        {"model_id": model_id, "results": results},
        cls=NumpyJSONEncoder, indent=2))
    log.info("Wrote %s", out_path)

    release_model(model)
    if not args.no_clean_cache:
        purge_hf_cache(model_id)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate(out_dir: Path) -> None:
    records = []
    for f in sorted(out_dir.glob("*_adaptive_width.json")):
        if f.name == "gem_adaptive_width.json":
            continue
        raw = json.loads(f.read_text())
        batch = raw if isinstance(raw, list) else raw.get("results", [])
        for r in batch:
            if isinstance(r, dict) and "/" in r.get("model_id", ""):
                records.append(r)
    # Deduplicate by (model_id, concept)
    seen, deduped = set(), []
    for r in records:
        key = (r["model_id"], r["concept"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    records = deduped

    if not records:
        log.warning("No results to aggregate")
        return

    better = sum(1 for r in records if r["adaptive_better"])
    deltas = [r["delta_vs_fixed3_pp"] for r in records]
    ratios = [r["ratio_vs_null"] for r in records]
    sig = sum(1 for r in records if r["p_value"] < 0.05)

    # Migrate old records written before the rule field existed
    for r in records:
        if "rule" not in r:
            r["rule"] = "near-final" if r.get("rel_depth", 0) > 0.85 else "default"

    # Split by rule triggered
    near_final = [r for r in records if r["rule"] == "near-final"]
    default = [r for r in records if r["rule"] == "default"]

    lines = [
        "GEM adaptive width — comparison with fixed w=3",
        f"N records: {len(records)}",
        f"Adaptive > fixed w=3: {better}/{len(records)} ({100*better/len(records):.1f}%)",
        f"Mean delta vs w=3: {np.mean(deltas):+.2f}pp",
        f"Mean ratio vs null: {np.mean(ratios):.2f}x",
        f"Significant (p<0.05): {sig}/{len(records)}",
        "",
        "By rule triggered:",
        (f"  near-final (HL/N>0.85, w=1): N={len(near_final)}  "
         f"mean_delta={np.mean([r['delta_vs_fixed3_pp'] for r in near_final]):+.2f}pp")
        if near_final else "  near-final: none",
        (f"  default (w=3, no change):     N={len(default)}  "
         f"mean_delta={np.mean([r['delta_vs_fixed3_pp'] for r in default]):+.2f}pp")
        if default else "  default: none",
        "",
        f"{'Model':<28} {'Concept':<16} {'HL':>3} {'Rel':>5} {'Rule':<10} "
        f"{'Wad':>3} {'Adapt':>6} {'W=3':>6} {'Delta':>7} {'Ratio':>6} {'p':>8}",
        "-" * 97,
    ]
    for r in sorted(records, key=lambda x: (x["model_id"], x["concept"])):
        lines.append(
            f"{r['model_id'].split('/')[-1]:<28} {r['concept']:<16} "
            f"{r['handoff_layer']:>3} {r['rel_depth']:>5.2f} {r['rule']:<10} "
            f"{r['adaptive_width']:>3} {r['adaptive_reduction']:>6.3f} "
            f"{r['fixed_w3_reduction']:>6.3f} {r['delta_vs_fixed3_pp']:>+7.2f}pp "
            f"{r['ratio_vs_null']:>5.2f}x {r['p_value']:>8.4f}"
        )

    text = "\n".join(lines)
    print(text)
    (out_dir / "gem_adaptive_width_summary.txt").write_text(text)
    (out_dir / "gem_adaptive_width.json").write_text(json.dumps(records, indent=2))
    log.info("Wrote aggregate to %s", out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all", action="store_true")
    group.add_argument("--aggregate-only", action="store_true",
                       help="Skip model runs, just re-aggregate existing results")
    parser.add_argument("--n-windows", type=int, default=N_WINDOWS_DEFAULT)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-clean-cache", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.aggregate_only:
        rng = np.random.default_rng(args.seed)
        models = discover_base_models() if args.all else [args.model]
        log.info("Running on %d models", len(models))
        for model_id in models:
            run_model(model_id, args, rng)

    aggregate(OUT_DIR)


if __name__ == "__main__":
    main()

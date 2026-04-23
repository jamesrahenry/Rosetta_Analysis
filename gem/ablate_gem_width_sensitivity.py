"""
ablate_gem_width_sensitivity.py
================================
Window-width sensitivity analysis for GEM handoff ablation.

The GEM currently ablates a width=3 window centred on the handoff layer.
This choice is principled (CAZ boundary detection) but not empirically
validated.  This script asks: is width=3 optimal, and does performance
plateau at some width?

For each (model, concept) pair, ablates the settled_direction across
windows of increasing width centred on the handoff layer:
  width = 1 (single point), 3 (current default), 5, 7, 10, 15, 20

Reports:
  - Final-layer separation reduction at each width
  - Peak (Fisher-optimal) as reference
  - Plots: suppression vs. width per concept family

Run on a representative subset of 6 models (one per architecture family)
to keep GPU time manageable.  Full sweep can be triggered with --all.

Usage
-----
    cd ~/caz_scaling
    python src/ablate_gem_width_sensitivity.py
    python src/ablate_gem_width_sensitivity.py --all
    python src/ablate_gem_width_sensitivity.py --model EleutherAI/pythia-1.4b

Written: 2026-04-21 UTC
"""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import ExitStack
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from rosetta_tools.ablation import DirectionalAblator, get_transformer_layers
from rosetta_tools.caz import compute_separation
from rosetta_tools.dataset import load_concept_pairs, texts_by_label
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.gem import discover_concepts, discover_base_models, find_extraction_dir
from rosetta_tools.gpu_utils import (
    get_device, get_dtype, log_device_info,
    release_model, purge_hf_cache, NumpyJSONEncoder, load_causal_lm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUT_DIR = Path.home() / "rosetta_data" / "results" / "gem_width_sensitivity"

# Representative subset — one per architectural family
DEFAULT_MODELS = [
    "EleutherAI/pythia-1.4b",       # MHA — Pythia
    "openai-community/gpt2-xl",      # MHA — GPT-2
    "facebook/opt-1.3b",             # MHA — OPT
    "meta-llama/Llama-3.2-3B",       # GQA — Llama
    "Qwen/Qwen2.5-3B",               # GQA — Qwen
    "google/gemma-2-9b",             # Alternating
]

WIDTHS = [1, 3, 5, 7, 10, 15, 20]
N_PAIRS = 50
BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def ablate_window(model, tokenizer, layers, handoff_layer: int, width: int,
                  direction: np.ndarray, pos_texts, neg_texts, device) -> float:
    n_layers = len(layers)
    half = width // 2
    window = list(range(
        max(0, handoff_layer - half),
        min(n_layers, handoff_layer + (width - half)),
    ))
    dtype = next(model.parameters()).dtype
    dev_str = str(next(model.parameters()).device)
    dir_t = torch.tensor(direction, dtype=dtype, device=dev_str)
    dir_t = dir_t / dir_t.norm()

    with ExitStack() as stack:
        for li in window:
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
# Per-concept
# ---------------------------------------------------------------------------

def run_concept(model, tokenizer, concept: str, extraction_dir: Path,
                device: str, widths: list[int]) -> dict | None:
    gem_path = extraction_dir / f"gem_{concept}.json"
    if not gem_path.exists():
        return None
    gem = json.loads(gem_path.read_text())

    targets = gem.get("ablation_targets", [0])
    node = gem["nodes"][targets[0]]
    handoff_layer = node["handoff_layer"]
    settled = np.array(node["settled_direction"], dtype=np.float64)
    settled /= np.linalg.norm(settled)

    # CAZ peak layer (reference single-point)
    caz_peak = node.get("caz_peak", handoff_layer)

    pairs = load_concept_pairs(concept, n=N_PAIRS)
    pos_texts, neg_texts = texts_by_label(pairs)

    layers = get_transformer_layers(model)
    n_layers = len(layers)

    base = baseline_sep(model, tokenizer, pos_texts, neg_texts, device)
    if base <= 0:
        return None

    def reduction(sep: float) -> float:
        return max(0.0, (base - sep) / base)

    width_results = {}
    for w in widths:
        sep = ablate_window(
            model, tokenizer, layers, handoff_layer, w,
            settled, pos_texts, neg_texts, device)
        width_results[w] = {
            "sep": round(sep, 4),
            "reduction": round(reduction(sep), 4),
        }
        log.info("  %s w=%2d  sep=%.4f  reduction=%.4f", concept, w, sep, reduction(sep))

    # CAZ peak reference (width=1 at peak layer, using peak direction from CAZ)
    caz_path = extraction_dir / f"caz_{concept}.json"
    peak_reduction = None
    if caz_path.exists():
        caz_d = json.loads(caz_path.read_text())
        metrics = caz_d.get("layer_data", caz_d).get("metrics", [])
        if caz_peak < len(metrics):
            peak_vec = np.array(metrics[caz_peak].get("dom_vector", []), dtype=np.float64)
            if peak_vec.size > 0:
                peak_vec /= np.linalg.norm(peak_vec)
                sep_peak = ablate_window(
                    model, tokenizer, layers, caz_peak, 1,
                    peak_vec, pos_texts, neg_texts, device)
                peak_reduction = round(reduction(sep_peak), 4)

    return {
        "concept": concept,
        "handoff_layer": handoff_layer,
        "caz_peak": caz_peak,
        "baseline_sep": round(base, 4),
        "n_layers": n_layers,
        "natural_caz_width": node.get("caz_end", 0) - node.get("caz_start", 0),
        "width_results": width_results,
        "caz_peak_reference_reduction": peak_reduction,
    }


# ---------------------------------------------------------------------------
# Per-model runner + plotting
# ---------------------------------------------------------------------------

def plot_width_curves(all_results: list[dict], out_dir: Path) -> None:
    from viz_style import THEME
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    # Left: per-model mean reduction vs width
    model_curves: dict[str, dict[int, list[float]]] = {}
    for r in all_results:
        mid = r["model_id"].split("/")[-1]
        if mid not in model_curves:
            model_curves[mid] = {w: [] for w in WIDTHS}
        for cr in r.get("concepts", []):
            for w, wr in cr.get("width_results", {}).items():
                model_curves[mid][int(w)].append(wr["reduction"])

    ax = axes[0]
    ax.set_facecolor("white")
    for mid, curves in model_curves.items():
        xs = sorted(curves.keys())
        ys = [np.mean(curves[w]) if curves[w] else 0 for w in xs]
        ax.plot(xs, ys, marker="o", label=mid, linewidth=1.5)
    ax.axvline(3, color="#9E9E9E", linestyle="--", linewidth=1, label="current default (3)")
    ax.set_xlabel("Window width (layers)", color=THEME["text"])
    ax.set_ylabel("Mean separation reduction", color=THEME["text"])
    ax.set_title("GEM suppression vs. window width (per model)",
                 color=THEME["text"], fontweight="bold", loc="left", fontsize=10)
    ax.legend(fontsize=7, facecolor="white")
    ax.grid(axis="y", linewidth=0.4, color="#ECEFF1")

    # Right: grand mean ± std
    grand: dict[int, list[float]] = {w: [] for w in WIDTHS}
    for curves in model_curves.values():
        for w in WIDTHS:
            grand[w].extend(curves[w])
    xs = sorted(grand.keys())
    ys = [np.mean(grand[w]) if grand[w] else 0 for w in xs]
    errs = [np.std(grand[w]) if grand[w] else 0 for w in xs]

    ax2 = axes[1]
    ax2.set_facecolor("white")
    ax2.errorbar(xs, ys, yerr=errs, marker="o", color="#C62828",
                 linewidth=2, capsize=4, label="Grand mean ± 1 SD")
    ax2.axvline(3, color="#9E9E9E", linestyle="--", linewidth=1,
                label="current default (3)")
    ax2.set_xlabel("Window width (layers)", color=THEME["text"])
    ax2.set_ylabel("Mean separation reduction", color=THEME["text"])
    ax2.set_title("GEM suppression vs. window width (grand mean)",
                  color=THEME["text"], fontweight="bold", loc="left", fontsize=10)
    ax2.legend(fontsize=8, facecolor="white")
    ax2.grid(axis="y", linewidth=0.4, color="#ECEFF1")

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor(THEME["spine"])
        ax.tick_params(colors=THEME["dim"])

    fig.tight_layout()
    out_path = out_dir / "gem_width_sensitivity.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")
    log.info("Saved %s", out_path)


def run_model(model_id: str, args, widths: list[int]) -> dict | None:
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.warning("No extraction dir for %s", model_id)
        return None

    out_path = OUT_DIR / f"{extraction_dir.name}_width_sensitivity.json"
    if out_path.exists() and not args.overwrite:
        log.info("Already done: %s", model_id)
        return json.loads(out_path.read_text())

    log.info("=== %s ===", model_id)
    device = get_device(args.device)
    dtype = get_dtype(args.dtype, device)
    log_device_info(device, dtype)

    try:
        model, tokenizer = load_causal_lm(model_id, device, dtype)
    except Exception as e:
        log.error("Failed to load %s: %s", model_id, e)
        return None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    concept_results = []
    for concept in discover_concepts(extraction_dir):
        r = run_concept(model, tokenizer, concept, extraction_dir,
                        device, widths)
        if r:
            concept_results.append(r)

    result = {"model_id": model_id, "widths": widths, "concepts": concept_results}
    out_path.write_text(json.dumps(result, cls=NumpyJSONEncoder, indent=2))
    log.info("Wrote %s", out_path)

    release_model(model)
    if not args.no_clean_cache:
        purge_hf_cache(model_id)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all", action="store_true")
    group.add_argument("--default-subset", action="store_true",
                       help="Run on the 6-model representative subset (default)")
    parser.add_argument("--widths", nargs="+", type=int, default=WIDTHS)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-clean-cache", action="store_true")
    args = parser.parse_args()

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        models = discover_base_models()
    elif args.model:
        models = [args.model]
    else:
        models = DEFAULT_MODELS

    log.info("Running width sensitivity on %d models, widths=%s",
             len(models), args.widths)

    all_results = []
    for model_id in models:
        r = run_model(model_id, args, args.widths)
        if r:
            all_results.append(r)

    if len(all_results) > 1:
        plot_width_curves(all_results, OUT_DIR)

    # Summary table
    lines = ["Width sensitivity summary — mean reduction by width",
             f"{'Width':>6}  {'Mean reduction':>15}  {'N':>5}"]
    all_by_width: dict[int, list[float]] = {w: [] for w in args.widths}
    for r in all_results:
        for cr in r.get("concepts", []):
            for w, wr in cr.get("width_results", {}).items():
                all_by_width[int(w)].append(wr["reduction"])
    for w in sorted(all_by_width.keys()):
        vals = all_by_width[w]
        lines.append(f"  {w:>4}   {np.mean(vals):>14.4f}   {len(vals):>5}")
    summary = "\n".join(lines)
    print(summary)
    (OUT_DIR / "gem_width_sensitivity_summary.txt").write_text(summary)


if __name__ == "__main__":
    main()

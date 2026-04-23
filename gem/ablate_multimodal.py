"""
ablate_multimodal.py — Test functional independence of multimodal assembly peaks.

For each multimodal concept × model:
  1. Detect the shallow and deep assembly peaks via find_caz_regions
  2. Ablate at the shallow peak only → measure separation at both peaks
  3. Ablate at the deep peak only → measure separation at both peaks
  4. Ablate at both peaks → measure separation at both peaks

If the two peaks are functionally independent, ablating one should leave
the other's separation largely intact. If they're redundant, ablating
either should suppress both.

Usage
-----
    python src/ablate_multimodal.py --all
    python src/ablate_multimodal.py --all
    python src/ablate_multimodal.py --model EleutherAI/pythia-410m
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta_tools.gpu_utils import (
    get_device, get_dtype, log_device_info, log_vram, release_model, purge_hf_cache,
    vram_stats, load_model_with_retry, NumpyJSONEncoder,
)
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.caz import compute_separation, find_caz_regions, find_caz_regions_scored, LayerMetrics
from rosetta_tools.ablation import DirectionalAblator, get_transformer_layers
from rosetta_tools.dataset import load_concept_pairs, texts_by_label
from rosetta_tools.gem import find_extraction_dir, discover_all_models
from rosetta_tools.paths import ROSETTA_RESULTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONCEPTS = [
    "credibility", "negation", "sentiment", "causation",
    "certainty", "moral_valence", "temporal_order",
]


def measure_separation_at_layers(
    model, tokenizer, layers, ablate_indices, directions,
    pos_texts, neg_texts, measure_indices, device, batch_size,
) -> dict[int, float]:
    """Apply ablation at specified layers, measure separation at other layers.

    Parameters
    ----------
    ablate_indices : list of int
        Layer indices to ablate simultaneously.
    directions : list of ndarray
        Concept direction at each ablation layer.
    measure_indices : list of int
        Layer indices where separation should be measured.

    Returns
    -------
    dict mapping layer_index → separation after ablation.
    """
    dtype = torch.bfloat16 if next(model.parameters()).dtype == torch.bfloat16 else torch.float32

    # Stack ablators as context managers
    from contextlib import ExitStack
    with ExitStack() as stack:
        for idx, direction in zip(ablate_indices, directions):
            stack.enter_context(DirectionalAblator(layers[idx], direction, dtype=dtype))

        pos_acts = extract_layer_activations(
            model, tokenizer, pos_texts, device=device, batch_size=batch_size, pool="last"
        )
        neg_acts = extract_layer_activations(
            model, tokenizer, neg_texts, device=device, batch_size=batch_size, pool="last"
        )

    results = {}
    for layer_idx in measure_indices:
        act_idx = layer_idx + 1  # extraction includes embedding at [0]
        if act_idx >= len(pos_acts):
            act_idx = len(pos_acts) - 1
        results[layer_idx] = compute_separation(pos_acts[act_idx], neg_acts[act_idx])

    return results


def ablate_concept(
    model, tokenizer, concept, extraction_data, device, n_pairs, batch_size,
) -> dict | None:
    """Run N-CAZ ablation experiment for one concept × model.

    Finds ALL CAZes with the scored detector, then for each CAZ:
    ablates it individually and measures the effect at every other CAZ.
    Builds an N×N interaction matrix showing the full dependency graph.
    """
    layer_data = extraction_data["layer_data"]
    metrics_raw = layer_data["metrics"]
    n_layers = int(layer_data["n_layers"])
    model_id = extraction_data["model_id"]

    # Check for inconsistent dom_vector dims
    dims = set(len(m["dom_vector"]) for m in metrics_raw)
    if len(dims) > 1:
        log.warning("  Skipping %s: inconsistent dom_vector dims", model_id)
        return None

    # Detect ALL regions with scored detector
    layer_metrics = [
        LayerMetrics(m["layer"], m["separation_fisher"], m["coherence"], m["velocity"])
        for m in metrics_raw
    ]
    profile = find_caz_regions_scored(layer_metrics)

    if profile.n_regions < 2:
        log.info("  %s has only %d CAZ — skipping", concept, profile.n_regions)
        return None

    # Sort regions by layer (shallow to deep)
    regions = sorted(profile.regions, key=lambda r: r.peak)
    caz_peaks = [int(r.peak) for r in regions]
    caz_scores = [r.caz_score for r in regions]

    caz_summary = "  ".join(
        "L%d(%.0f%%, score=%.3f)" % (r.peak, r.depth_pct, r.caz_score)
        for r in regions
    )
    log.info("  %d CAZes: %s", len(regions), caz_summary)

    # Load contrastive pairs
    pairs = load_concept_pairs(concept, n=n_pairs or 200)
    pos_texts, neg_texts = texts_by_label(pairs)

    # Get transformer layers and directions
    layers = get_transformer_layers(model)
    caz_directions = [
        np.array(metrics_raw[peak]["dom_vector"], dtype=np.float64)
        for peak in caz_peaks
    ]

    # Measure one layer downstream of each peak (forward hook offset fix)
    measure_at = [min(peak + 1, n_layers - 1) for peak in caz_peaks]

    # Baseline: no ablation
    log.info("  Measuring baseline...")
    baseline = measure_separation_at_layers(
        model, tokenizer, layers,
        ablate_indices=[], directions=[],
        pos_texts=pos_texts, neg_texts=neg_texts,
        measure_indices=measure_at, device=device, batch_size=batch_size,
    )

    # N×N interaction matrix: ablate CAZ_i, measure at CAZ_j
    # interaction[i][j] = retained_pct when ablating CAZ_i, measured at CAZ_j
    n_cazs = len(caz_peaks)
    interaction = np.zeros((n_cazs, n_cazs))

    for i in range(n_cazs):
        log.info("  Ablating CAZ %d/%d (L%d, score=%.3f)...",
                 i + 1, n_cazs, caz_peaks[i], caz_scores[i])
        ablated = measure_separation_at_layers(
            model, tokenizer, layers,
            ablate_indices=[caz_peaks[i]], directions=[caz_directions[i]],
            pos_texts=pos_texts, neg_texts=neg_texts,
            measure_indices=measure_at, device=device, batch_size=batch_size,
        )
        for j in range(n_cazs):
            m_key = measure_at[j]
            b_val = baseline.get(m_key, 0)
            a_val = ablated.get(m_key, 0)
            interaction[i][j] = round(100 * a_val / b_val, 1) if b_val > 0 else 100.0

    # Log interaction matrix
    log.info("  Interaction matrix (rows=ablated, cols=measured, values=%%retained):")
    header = "          " + "".join("  L%-5d" % p for p in caz_peaks)
    log.info("  %s", header)
    for i in range(n_cazs):
        row = "  L%-5d " % caz_peaks[i]
        for j in range(n_cazs):
            val = interaction[i][j]
            marker = "*" if i == j else " "
            row += "%6.1f%s" % (val, marker)
        log.info("  %s", row)

    def _pct(ablated_val, baseline_val):
        return round(100 * ablated_val / baseline_val, 1) if baseline_val > 0 else 0.0

    # Build result
    caz_list = []
    for i, r in enumerate(regions):
        caz_list.append({
            "peak": caz_peaks[i],
            "depth_pct": round(r.depth_pct, 1),
            "caz_score": r.caz_score,
            "separation": round(r.peak_separation, 4),
            "coherence": round(r.peak_coherence, 4),
            "width": r.width,
            "baseline_sep": round(baseline.get(measure_at[i], 0), 4),
            "self_retained_pct": interaction[i][i],
        })

    result = {
        "model_id": model_id,
        "concept": concept,
        "n_layers": n_layers,
        "n_cazs": n_cazs,
        "n_pairs": len(pairs),
        "cazs": caz_list,
        "interaction_matrix": interaction.tolist(),
        "interaction_peaks": caz_peaks,
        # Legacy compat: shallow/deep for the first and last CAZ
        "shallow_peak": caz_peaks[0],
        "deep_peak": caz_peaks[-1],
        "shallow_depth_pct": round(100 * caz_peaks[0] / n_layers, 1),
        "deep_depth_pct": round(100 * caz_peaks[-1] / n_layers, 1),
    }

    log.info("  Summary:")
    for c in caz_list:
        log.info("    L%d (score=%.3f): self_retained=%.1f%%",
                 c["peak"], c["caz_score"], c["self_retained_pct"])

    return result


def is_ablation_current(extraction_dir: Path, concepts: list[str]) -> bool:
    """Return True if every available concept already has a fresh ablation result.

    'Fresh' means ablation_multimodal_{concept}.json exists AND is newer than
    caz_{concept}.json (the extraction it was computed from).
    """
    available = [c for c in concepts if (extraction_dir / f"caz_{c}.json").exists()]
    if not available:
        return False
    for concept in available:
        src = extraction_dir / f"caz_{concept}.json"
        out = extraction_dir / f"ablation_multimodal_{concept}.json"
        if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
            return False
    return True


def run_model(model_id: str, concepts: list[str], args) -> None:
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.error("No extraction results for %s", model_id)
        return

    if getattr(args, "skip_done", False) and is_ablation_current(extraction_dir, concepts):
        log.info("=== Skipping %s — ablation results current ===", model_id)
        return

    log.info("=== Multimodal ablation: %s ===", model_id)

    device = get_device(args.device)
    dtype = get_dtype(device)
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float32":
        dtype = torch.float32
    if device.startswith("cuda"):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    log_device_info(device, dtype)

    from rosetta_tools.gpu_utils import load_causal_lm
    model, tokenizer = load_causal_lm(model_id, device, dtype)
    log_vram("after model load")

    # Skip if the model barely fits — ablation needs headroom for forward passes
    if device.startswith("cuda"):
        stats = vram_stats()
        if stats and stats["free_gib"] < 0.20 * stats["total_gib"]:
            log.warning(
                "Skipping %s: only %.1f GiB free of %.1f GiB — not enough for ablation",
                model_id, stats["free_gib"], stats["total_gib"])
            release_model(model)
            if not getattr(args, "no_clean_cache", False):
                purge_hf_cache(model_id)
            return

    all_results = []
    t_start = time.time()

    for i, concept in enumerate(concepts):
        log.info("--- Concept %d/%d: %s ---", i + 1, len(concepts), concept)

        caz_path = extraction_dir / f"caz_{concept}.json"
        if not caz_path.exists():
            log.warning("  No extraction data for %s, skipping", concept)
            continue

        with open(caz_path) as f:
            extraction_data = json.load(f)

        torch.cuda.empty_cache() if device.startswith("cuda") else None

        result = ablate_concept(
            model, tokenizer, concept, extraction_data,
            device=device, n_pairs=args.n_pairs, batch_size=args.batch_size,
        )
        if result:
            all_results.append(result)

            # Save per-concept
            out_path = extraction_dir / f"ablation_multimodal_{concept}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, cls=NumpyJSONEncoder)

    total_elapsed = time.time() - t_start
    release_model(model)

    if not getattr(args, "no_clean_cache", False):
        purge_hf_cache(model_id)

    # Save combined results — per-CAZ rows with interaction data
    if all_results:
        import csv
        out_csv = Path("visualizations/structure/multimodal_ablation_v2.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "model_id", "concept", "n_layers", "n_cazs",
            "caz_idx", "peak", "depth_pct", "caz_score",
            "separation", "coherence", "width",
            "baseline_sep", "self_retained_pct",
        ]

        write_header = not out_csv.exists()
        with open(out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            for result in all_results:
                for idx, caz in enumerate(result["cazs"]):
                    w.writerow({
                        "model_id": result["model_id"],
                        "concept": result["concept"],
                        "n_layers": result["n_layers"],
                        "n_cazs": result["n_cazs"],
                        "caz_idx": idx,
                        **caz,
                    })
        log.info("Results appended to %s", out_csv)

    log.info("Done: %s  (%.1fs total)", model_id, total_elapsed)




def main():
    parser = argparse.ArgumentParser(
        description="Multimodal ablation — test functional independence of assembly peaks",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--concepts", nargs="+", default=None)
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="auto")
    parser.add_argument("--no-clean-cache", action="store_true",
                        help="Keep models in HF cache (default: auto-purge)")
    parser.add_argument("--clean-cache", action="store_true",
                        help="(deprecated, now default) Kept for backward compat")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip models whose ablation results are already current "
                             "(output newer than extraction source). Safe to re-run.")
    parser.add_argument("--skip-model", action="append", default=[],
                        metavar="MODEL_ID",
                        help="Skip this model (may be repeated). Useful for gated "
                             "or already-completed models.")
    args = parser.parse_args()

    concepts = args.concepts or CONCEPTS

    if args.all:
        models = discover_all_models()
        log.info("Found %d models with extraction results", len(models))
    else:
        models = [args.model]

    if args.skip_model:
        skip = set(args.skip_model)
        models = [m for m in models if m not in skip]
        log.info("Skipping %d models: %s", len(skip), sorted(skip))

    for model_id in models:
        run_model(model_id, concepts, args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
cka_validation.py — CKA validation of CAZ boundaries.

Computes linear CKA (Centered Kernel Alignment; Kornblith et al. 2019) between
adjacent transformer layers and tests whether within-CAZ layer pairs show higher
representational similarity than cross-CAZ (boundary/saddle-point) pairs.

This provides independent validation of CAZ boundaries using a metric that
shares no assumptions with the Fisher-based separation used for detection.

Hypothesis: if CAZ boundaries mark genuine transitions in representational
structure, CKA should drop at boundaries and remain high within regions.

Usage
-----
    python src/cka_validation.py --model EleutherAI/pythia-410m
    python src/cka_validation.py --all
    python src/cka_validation.py --all --full-matrix   # also save NxN CKA matrix

Results written to:
    results/<extraction_dir>/cka_<concept>.json

Aggregate written to:
    results/cka_aggregate_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import torch
from transformers import AutoModel, AutoTokenizer

from rosetta_tools.gpu_utils import (
    get_device, get_dtype, log_device_info, log_vram,
    release_model, purge_hf_cache, vram_stats,
    load_model_with_retry, NumpyJSONEncoder,
)
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.caz import find_caz_regions_scored, LayerMetrics
from rosetta_tools.dataset import load_pairs, texts_by_label
from rosetta_tools.models import attention_paradigm_of

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_ROOT = Path("results")
DATA_ROOT = Path(__file__).parent.parent / "data"

CONCEPT_DATASETS: dict[str, str] = {
    "credibility":    "credibility_pairs.jsonl",
    "negation":       "negation_pairs.jsonl",
    "sentiment":      "sentiment_pairs.jsonl",
    "causation":      "causation_pairs.jsonl",
    "certainty":      "certainty_pairs.jsonl",
    "moral_valence":  "moral_valence_pairs.jsonl",
    "temporal_order": "temporal_order_pairs.jsonl",
}


# ---------------------------------------------------------------------------
# Linear CKA (Kornblith et al. 2019)
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two activation matrices.

    Parameters
    ----------
    X : (n, p) — activations at layer i for n examples
    Y : (n, q) — activations at layer j for n examples

    Returns
    -------
    CKA score in [0, 1]. Higher = more similar representations.
    """
    # Center columns (features)
    X = (X - X.mean(axis=0)).astype(np.float64)
    Y = (Y - Y.mean(axis=0)).astype(np.float64)

    # Gram matrices (n x n) — cheap when n << p
    K = X @ X.T
    L = Y @ Y.T

    # HSIC with linear kernels: trace(KHLH) / (n-1)^2
    # Simplification: for centered Gram matrices, HSIC = trace(KL) / (n-1)^2
    # after centering K and L with the centering matrix H = I - 11^T/n
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    KH = K @ H
    LH = L @ H

    hsic_kl = np.trace(KH @ LH) / (n - 1) ** 2
    hsic_kk = np.trace(KH @ KH) / (n - 1) ** 2
    hsic_ll = np.trace(LH @ LH) / (n - 1) ** 2

    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-12:
        return 0.0
    return float(np.clip(hsic_kl / denom, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Model / extraction discovery (same pattern as ablate_global_sweep.py)
# ---------------------------------------------------------------------------

def find_extraction_dir(model_id: str) -> Path | None:
    candidates = []
    for d in sorted(RESULTS_ROOT.iterdir(), reverse=True):
        summary = d / "run_summary.json"
        if d.is_dir() and summary.exists():
            try:
                if json.loads(summary.read_text()).get("model_id") == model_id:
                    candidates.append(d)
            except (json.JSONDecodeError, KeyError):
                continue
    return candidates[0] if candidates else None


def discover_models() -> list[str]:
    try:
        from rosetta_tools.models import get_model
        def _enabled(mid: str) -> bool:
            m = get_model(mid)
            return m is None or m.enabled
    except ImportError:
        def _enabled(mid: str) -> bool:
            return True

    models = set()
    for d in RESULTS_ROOT.iterdir():
        s = d / "run_summary.json"
        if s.exists():
            try:
                mid = json.loads(s.read_text()).get("model_id", "")
                if mid and _enabled(mid):
                    models.add(mid)
            except Exception:
                pass
    return sorted(models)


def load_concept_data(extraction_dir: Path, concept: str) -> dict | None:
    path = extraction_dir / f"caz_{concept}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# CAZ region loading & pair classification
# ---------------------------------------------------------------------------

def load_caz_regions(extraction_data: dict, model_id: str) -> list[dict]:
    """Reconstruct CAZ regions from extraction data using find_caz_regions_scored."""
    layer_data = extraction_data["layer_data"]
    metrics_raw = layer_data["metrics"]

    layer_metrics = [
        LayerMetrics(
            layer=m["layer"],
            separation=m["separation_fisher"],
            coherence=m.get("coherence", 0.0),
            velocity=m.get("velocity", 0.0),
        )
        for m in metrics_raw
    ]

    paradigm = attention_paradigm_of(model_id)
    n_layers = len(layer_metrics)

    # For alternating models, compute functional peak layer
    functional_peak = None
    if paradigm == "alternating":
        from rosetta_tools.caz import final_global_attention_layer
        functional_peak = final_global_attention_layer(n_layers)

    profile = find_caz_regions_scored(
        layer_metrics,
        attention_paradigm=paradigm,
        functional_peak_layer=functional_peak,
    )

    regions = []
    for r in profile.regions:
        regions.append({"start": r.start, "peak": r.peak, "end": r.end})
    return regions


def classify_pairs(n_layers: int, regions: list[dict]) -> list[str]:
    """Classify each adjacent layer pair as 'within' or 'cross' CAZ.

    Every layer belongs to exactly one region (find_caz_regions_scored
    partitions the full layer range). A pair (i, i+1) is 'within' if
    both layers fall in the same region, 'cross' if they straddle a
    boundary.
    """
    # Map each layer to its region index
    layer_region: dict[int, int] = {}
    for r_idx, region in enumerate(regions):
        for l in range(region["start"], region["end"] + 1):
            layer_region[l] = r_idx

    labels = []
    for i in range(n_layers - 1):
        r_i = layer_region.get(i)
        r_next = layer_region.get(i + 1)
        if r_i is not None and r_next is not None and r_i == r_next:
            labels.append("within")
        else:
            labels.append("cross")
    return labels


# ---------------------------------------------------------------------------
# Per-concept analysis
# ---------------------------------------------------------------------------

def analyze_concept(
    model,
    tokenizer,
    concept: str,
    extraction_data: dict,
    model_id: str,
    device: str,
    n_pairs: int,
    batch_size: int,
    full_matrix: bool,
) -> dict:
    """Compute adjacent CKA and test within-CAZ vs cross-CAZ hypothesis."""
    dataset_path = DATA_ROOT / CONCEPT_DATASETS[concept]
    pairs = load_pairs(dataset_path)
    if n_pairs and len(pairs) > n_pairs:
        pairs = pairs[:n_pairs]
    pos_texts, neg_texts = texts_by_label(pairs)
    all_texts = pos_texts + neg_texts

    log.info("  %s: extracting activations (%d texts)...", concept, len(all_texts))
    t0 = time.time()

    acts_by_layer = extract_layer_activations(
        model, tokenizer, all_texts, device=device, batch_size=batch_size, pool="last",
    )

    # Drop embedding layer (index 0), consistent with all other scripts
    acts_by_layer = acts_by_layer[1:]
    n_layers = len(acts_by_layer)

    t_extract = time.time() - t0
    log.info("  %s: extracted %d layers in %.1fs", concept, n_layers, t_extract)

    # Compute adjacent CKA
    cka_adjacent = []
    for i in range(n_layers - 1):
        score = linear_cka(acts_by_layer[i], acts_by_layer[i + 1])
        cka_adjacent.append(round(score, 6))
    log.info("  %s: adjacent CKA range [%.4f, %.4f]",
             concept, min(cka_adjacent), max(cka_adjacent))

    # Full matrix (optional)
    cka_matrix = None
    if full_matrix:
        log.info("  %s: computing full %dx%d CKA matrix...", concept, n_layers, n_layers)
        mat = np.zeros((n_layers, n_layers))
        for i in range(n_layers):
            mat[i, i] = 1.0
            for j in range(i + 1, n_layers):
                score = linear_cka(acts_by_layer[i], acts_by_layer[j])
                mat[i, j] = score
                mat[j, i] = score
        cka_matrix = mat

    # Load CAZ regions and classify pairs
    regions = load_caz_regions(extraction_data, model_id)
    pair_labels = classify_pairs(n_layers, regions)

    within_scores = [cka_adjacent[i] for i, l in enumerate(pair_labels) if l == "within"]
    cross_scores = [cka_adjacent[i] for i, l in enumerate(pair_labels) if l == "cross"]

    # Statistical test
    within_mean = float(np.mean(within_scores)) if within_scores else None
    within_std = float(np.std(within_scores)) if within_scores else None
    cross_mean = float(np.mean(cross_scores)) if cross_scores else None
    cross_std = float(np.std(cross_scores)) if cross_scores else None

    u_stat, p_value, effect_d, supported = None, None, None, None

    if len(within_scores) >= 3 and len(cross_scores) >= 3:
        # One-sided: within > cross (alternative='greater')
        u_stat, p_value = mannwhitneyu(within_scores, cross_scores, alternative="greater")
        u_stat = float(u_stat)
        p_value = float(p_value)

        # Cohen's d
        pooled_std = np.sqrt(
            ((len(within_scores) - 1) * np.var(within_scores, ddof=1)
             + (len(cross_scores) - 1) * np.var(cross_scores, ddof=1))
            / (len(within_scores) + len(cross_scores) - 2)
        )
        effect_d = float((within_mean - cross_mean) / pooled_std) if pooled_std > 0 else 0.0
        supported = p_value < 0.05 and within_mean > cross_mean

        log.info("  %s: within=%.4f±%.4f (%d), cross=%.4f±%.4f (%d), U=%.1f, p=%.4f, d=%.3f %s",
                 concept, within_mean, within_std, len(within_scores),
                 cross_mean, cross_std, len(cross_scores),
                 u_stat, p_value, effect_d,
                 "✓" if supported else "✗")
    else:
        log.warning("  %s: too few pairs for test (within=%d, cross=%d)",
                    concept, len(within_scores), len(cross_scores))

    t_total = time.time() - t0

    result = {
        "model_id":             model_id,
        "concept":              concept,
        "n_layers":             n_layers,
        "n_texts_used":         len(all_texts),
        "cka_adjacent":         cka_adjacent,
        "within_caz_indices":   [i for i, l in enumerate(pair_labels) if l == "within"],
        "cross_caz_indices":    [i for i, l in enumerate(pair_labels) if l == "cross"],
        "within_caz_mean":      round(within_mean, 6) if within_mean is not None else None,
        "within_caz_std":       round(within_std, 6) if within_std is not None else None,
        "cross_caz_mean":       round(cross_mean, 6) if cross_mean is not None else None,
        "cross_caz_std":        round(cross_std, 6) if cross_std is not None else None,
        "mann_whitney_U":       round(u_stat, 4) if u_stat is not None else None,
        "mann_whitney_p":       round(p_value, 6) if p_value is not None else None,
        "effect_size_d":        round(effect_d, 4) if effect_d is not None else None,
        "hypothesis_supported": supported,
        "caz_regions":          regions,
        "pair_labels":          pair_labels,
        "extraction_seconds":   round(t_total, 1),
    }

    return result, cka_matrix


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(model_id: str, concepts: list[str], args) -> list[dict]:
    """Run CKA validation for one model across all concepts."""
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.error("No extraction results for %s — skipping", model_id)
        return []

    # Check if already done
    first_concept = concepts[0]
    out_path = extraction_dir / f"cka_{first_concept}.json"
    if out_path.exists() and not args.overwrite:
        log.info("Already done: %s (use --overwrite to rerun)", model_id)
        return []

    log.info("=== CKA validation: %s ===", model_id)
    device = get_device(args.device)
    dtype = get_dtype(device, args.dtype)
    log_device_info(device, dtype)

    # Force VRAM cleanup before loading
    if device.startswith("cuda"):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = load_model_with_retry(AutoModel, model_id, dtype=dtype, device=device)
        model.eval()
    except Exception as e:
        log.error("Failed to load %s: %s", model_id, e)
        return []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    stats = vram_stats(device)
    if stats:
        log_vram(device)

    t_model_start = time.time()
    results = []

    for concept in concepts:
        out_path = extraction_dir / f"cka_{concept}.json"
        if out_path.exists() and not args.overwrite:
            log.info("  Skipping %s (already done)", concept)
            continue

        extraction_data = load_concept_data(extraction_dir, concept)
        if extraction_data is None:
            log.warning("  No extraction data for %s, skipping", concept)
            continue

        try:
            if device == "cuda":
                torch.cuda.empty_cache()
            result, cka_matrix = analyze_concept(
                model, tokenizer, concept, extraction_data, model_id,
                device=device, n_pairs=args.n_pairs, batch_size=args.batch_size,
                full_matrix=args.full_matrix,
            )
        except Exception as e:
            log.error("  CKA failed for %s %s: %s", model_id, concept, e)
            continue

        out_path.write_text(json.dumps(result, indent=2, cls=NumpyJSONEncoder))
        log.info("  Wrote %s", out_path)

        if cka_matrix is not None:
            npy_path = extraction_dir / f"cka_matrix_{concept}.npy"
            np.save(npy_path, cka_matrix)
            log.info("  Wrote %s", npy_path)

        results.append(result)

    release_model(model)
    if not args.no_clean_cache:
        purge_hf_cache(model_id)

    log.info("Done: %s (%.1fs)", model_id, time.time() - t_model_start)
    return results


# ---------------------------------------------------------------------------
# Cross-model aggregation
# ---------------------------------------------------------------------------

def aggregate_results(all_results: list[dict]) -> dict:
    """Aggregate CKA results across all model/concept pairs."""
    all_within = []
    all_cross = []
    n_significant = 0
    n_testable = 0

    by_concept: dict[str, list[dict]] = {}
    by_model: dict[str, list[dict]] = {}

    for r in all_results:
        concept = r["concept"]
        model_id = r["model_id"]

        by_concept.setdefault(concept, []).append(r)
        by_model.setdefault(model_id, []).append(r)

        within_idx = r["within_caz_indices"]
        cross_idx = r["cross_caz_indices"]
        cka = r["cka_adjacent"]

        all_within.extend(cka[i] for i in within_idx)
        all_cross.extend(cka[i] for i in cross_idx)

        if r["hypothesis_supported"] is not None:
            n_testable += 1
            if r["hypothesis_supported"]:
                n_significant += 1

    # Grand test
    grand_u, grand_p, grand_d = None, None, None
    if len(all_within) >= 3 and len(all_cross) >= 3:
        grand_u, grand_p = mannwhitneyu(all_within, all_cross, alternative="greater")
        grand_u = float(grand_u)
        grand_p = float(grand_p)
        w_var = np.var(all_within, ddof=1)
        c_var = np.var(all_cross, ddof=1)
        pooled = np.sqrt(
            ((len(all_within) - 1) * w_var + (len(all_cross) - 1) * c_var)
            / (len(all_within) + len(all_cross) - 2)
        )
        grand_d = float((np.mean(all_within) - np.mean(all_cross)) / pooled) if pooled > 0 else 0.0

    # Per-concept summaries
    concept_summaries = {}
    for c, rs in sorted(by_concept.items()):
        testable = [r for r in rs if r["hypothesis_supported"] is not None]
        concept_summaries[c] = {
            "n_models": len(rs),
            "n_significant": sum(1 for r in testable if r["hypothesis_supported"]),
            "n_testable": len(testable),
            "mean_within": round(float(np.mean([r["within_caz_mean"] for r in rs if r["within_caz_mean"] is not None])), 6),
            "mean_cross": round(float(np.mean([r["cross_caz_mean"] for r in rs if r["cross_caz_mean"] is not None])), 6),
        }

    # Per-model summaries
    model_summaries = {}
    for m, rs in sorted(by_model.items()):
        testable = [r for r in rs if r["hypothesis_supported"] is not None]
        model_summaries[m] = {
            "n_concepts": len(rs),
            "n_significant": sum(1 for r in testable if r["hypothesis_supported"]),
            "n_testable": len(testable),
        }

    return {
        "timestamp":                     datetime.now().isoformat(),
        "n_models":                      len(by_model),
        "n_concepts":                    len(by_concept),
        "n_tests":                       len(all_results),
        "n_testable":                    n_testable,
        "n_significant_correct_direction": n_significant,
        "overall_within_mean":           round(float(np.mean(all_within)), 6) if all_within else None,
        "overall_cross_mean":            round(float(np.mean(all_cross)), 6) if all_cross else None,
        "overall_mann_whitney_U":        round(grand_u, 4) if grand_u is not None else None,
        "overall_mann_whitney_p":        round(grand_p, 6) if grand_p is not None else None,
        "overall_effect_size_d":         round(grand_d, 4) if grand_d is not None else None,
        "by_concept":                    concept_summaries,
        "by_model":                      model_summaries,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CKA validation of CAZ boundaries: do within-CAZ layers "
                    "show higher representational similarity than cross-CAZ pairs?"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Single model ID")
    group.add_argument("--all", action="store_true", help="Run all models with extraction results")

    parser.add_argument("--concepts", nargs="+", default=None,
                        help="Specific concepts (default: all 7)")
    parser.add_argument("--n-pairs", type=int, default=100,
                        help="Max pairs per concept (default: 100, yields ~200 texts for CKA)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="auto")
    parser.add_argument("--overwrite", action="store_true", help="Rerun even if output exists")
    parser.add_argument("--no-clean-cache", action="store_true")
    parser.add_argument("--full-matrix", action="store_true",
                        help="Also compute full NxN CKA matrix (for visualization)")
    parser.add_argument("--skip-model", action="append", default=[],
                        metavar="MODEL_ID", help="Skip this model (may be repeated)")

    args = parser.parse_args()
    concepts = args.concepts or list(CONCEPT_DATASETS.keys())

    if args.all:
        models = discover_models()
        log.info("Found %d models with extraction results", len(models))
    else:
        models = [args.model]

    if args.skip_model:
        skip = set(args.skip_model)
        models = [m for m in models if m not in skip]
        log.info("Skipping %d models: %s", len(skip), sorted(skip))

    all_results = []
    for model_id in models:
        results = run_model(model_id, concepts, args)
        all_results.extend(results)

    # Aggregate if we ran multiple models
    if len(all_results) > 1:
        agg = aggregate_results(all_results)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        agg_path = RESULTS_ROOT / f"cka_aggregate_{ts}.json"
        agg_path.write_text(json.dumps(agg, indent=2, cls=NumpyJSONEncoder))
        log.info("Aggregate: %s", agg_path)
        log.info("  %d/%d tests support hypothesis (within-CAZ CKA > cross-CAZ CKA)",
                 agg["n_significant_correct_direction"], agg["n_testable"])
        if agg["overall_mann_whitney_p"] is not None:
            log.info("  Overall: within=%.4f, cross=%.4f, p=%.6f, d=%.3f",
                     agg["overall_within_mean"], agg["overall_cross_mean"],
                     agg["overall_mann_whitney_p"], agg["overall_effect_size_d"])


if __name__ == "__main__":
    main()

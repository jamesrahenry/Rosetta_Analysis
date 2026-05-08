#!/usr/bin/env python3
"""
patch.py — Activation patching for CAZ causal validation.

Companion to ablate.py. Where ablation tests:
  "Does removing the concept direction at layer L damage downstream encoding?"

Patching tests:
  "Does injecting the concept direction at layer L restore downstream encoding?"

Together they triangulate whether a layer is causally load-bearing for concept
propagation, or whether projection ablation's near-zero impact on sparse models
(GQA+SwiGLU) reflects a genuine architectural difference vs a measurement gap.

Method — mean-field shift patching:
  For each layer L:
    1. Cache baseline activations for pos (concept present) and neg (concept absent)
       texts via a clean forward pass.
    2. Compute the mean shift at layer L: Δ_L = μ_pos_L - μ_neg_L
    3. Run neg texts through the model with a hook at layer L that adds Δ_L to
       every hidden state — shifting the neg distribution's centroid to where
       the pos distribution was at that layer.
    4. At the final layer, measure how much the patched neg representations moved
       along the concept direction toward the pos class:

         concept_score_recovery(L) = (mean_patched_neg_score - mean_neg_score)
                                   / (mean_pos_score - mean_neg_score)

       where scores are projections onto the concept direction at the final layer.

  Recovery ≈ 0 → patching at L has no downstream effect (layer causally inert)
  Recovery ≈ 1 → patching at L fully restores concept encoding at the output
  Recovery > 1 → overshoot (patch caused overcorrection downstream)

Why concept_score_recovery instead of Fisher separation:
  Fisher separation between pos and patched-neg collapses to near-zero after a
  mean-shift patch — both distributions are now centered at ~μ_pos, so the
  between-class distance is ~0. The concept score approach avoids this by
  projecting onto the final-layer concept direction and measuring how much the
  patched neg distribution moved toward the pos scores.

Results are written to results/<extraction_dir>/patch_<concept>.json

Usage:
    python src/patch.py --model EleutherAI/pythia-410m
    python src/patch.py --model google/gemma-2-9b --load-in-8bit
    python src/patch.py --all
    python src/patch.py --all --no-clean-cache
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.ablation import get_transformer_layers
from rosetta_tools.caz import find_caz_boundary, LayerMetrics
from rosetta_tools.gpu_utils import (
    get_device, get_dtype, log_device_info, log_vram, release_model, purge_hf_cache,
    load_model_with_retry,
)
from rosetta_tools.dataset import load_concept_pairs, texts_by_label
from rosetta_tools.gem import find_extraction_dir, discover_all_models
from rosetta_tools.paths import ROSETTA_RESULTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONCEPTS: list[str] = [
    "credibility", "negation", "sentiment", "causation",
    "certainty", "moral_valence", "temporal_order",
]


# ---------------------------------------------------------------------------
# Infrastructure (mirrors ablate.py)
# ---------------------------------------------------------------------------

def load_concept_directions(extraction_dir: Path, concept: str) -> dict | None:
    caz_path = extraction_dir / f"caz_{concept}.json"
    if not caz_path.exists():
        return None
    with open(caz_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Patching hook
# ---------------------------------------------------------------------------

class MeanShiftPatcher:
    """Context manager that hooks a transformer layer to inject a concept shift.

    Adds Δ = (μ_pos - μ_neg) at layer L to every hidden state in the batch,
    shifting the neg distribution's centroid toward the pos distribution's
    centroid at that layer. This is mean-field activation patching.

    Parameters
    ----------
    layer_module:
        The transformer block to hook (from get_transformer_layers(model)[L]).
    delta:
        Shift vector [hidden_dim] = μ_pos_L - μ_neg_L. Will be cast to model dtype.
    dtype:
        Torch dtype matching the model's activations.
    """

    def __init__(self, layer_module, delta: np.ndarray, dtype: torch.dtype = torch.float32):
        self._layer = layer_module
        self._handle = None
        self._delta = torch.tensor(delta, dtype=dtype)

    def _hook(self, module, input, output):
        if isinstance(output, tuple):
            hidden = None
            hidden_idx = 0
            target_dim = self._delta.shape[0]
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor) and o.dim() == 3 and o.shape[-1] == target_dim:
                    hidden = o
                    hidden_idx = i
                    break
            if hidden is None:
                hidden = output[0]
                hidden_idx = 0
        else:
            hidden = output
            hidden_idx = None

        dev = hidden.device
        dt  = hidden.dtype
        delta = self._delta.to(device=dev, dtype=dt)

        if hidden.shape[-1] != delta.shape[0]:
            return output  # dimension mismatch — skip rather than crash

        # Add delta to all batch × sequence positions: h' = h + Δ
        shifted = hidden + delta.unsqueeze(0).unsqueeze(0)  # [1, 1, dim] broadcasts

        if isinstance(output, tuple):
            return output[:hidden_idx] + (shifted,) + output[hidden_idx + 1:]
        return shifted

    def __enter__(self):
        self._handle = self._layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# Per-layer patching measurement
# ---------------------------------------------------------------------------

def concept_score_recovery(
    pos_acts_final: np.ndarray,
    neg_acts_final: np.ndarray,
    patched_neg_acts_final: np.ndarray,
) -> float:
    """Fraction of final-layer concept gap recovered by patching.

    Projects all representations onto the final-layer concept direction
    (unit vector from neg mean to pos mean) and measures how far the
    patched neg distribution moved toward the pos distribution.

    Returns
    -------
    float
        0.0 = no recovery (patching had no downstream effect)
        1.0 = full recovery (patched neg looks like pos at final layer)
        Negative = patch moved neg away from pos (anti-recovery)
        >1.0  = overshoot (patched neg overshot pos)
    """
    d = pos_acts_final.mean(0) - neg_acts_final.mean(0)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return 0.0
    d = d / norm  # unit concept direction at final layer

    score_pos    = pos_acts_final @ d            # [n_pos]
    score_neg    = neg_acts_final @ d            # [n_neg]
    score_patch  = patched_neg_acts_final @ d    # [n_neg]

    gap_baseline = score_pos.mean() - score_neg.mean()
    if abs(gap_baseline) < 1e-9:
        return 0.0

    gap_recovered = score_patch.mean() - score_neg.mean()
    return float(gap_recovered / gap_baseline)


def patch_sweep(
    model,
    tokenizer,
    concept: str,
    extraction_data: dict,
    device: str,
    n_pairs: int,
    batch_size: int,
) -> dict:
    """Sweep activation patching across all layers for one concept.

    For each layer L, shifts neg hidden states at L by (μ_pos_L - μ_neg_L)
    and measures how much concept signal is recovered at the final layer.
    """
    pairs = load_concept_pairs(concept, n=n_pairs or 200)
    pos_texts, neg_texts = texts_by_label(pairs)

    layers    = get_transformer_layers(model)
    n_layers  = len(layers)
    dtype     = next(model.parameters()).dtype

    layer_data  = extraction_data["layer_data"]
    metrics_raw = layer_data["metrics"]
    peak_layer  = layer_data["peak_layer"]

    # CAZ boundary from extraction
    layer_metrics = [
        LayerMetrics(m["layer"], m["separation_fisher"], m["coherence"], m["velocity"])
        for m in metrics_raw
    ]
    try:
        boundary = find_caz_boundary(layer_metrics)
    except Exception:
        boundary = None

    # ── Step 1: cache clean baseline activations (2 forward passes total) ──
    log.info("  Caching baseline activations (%d pos, %d neg)...", len(pos_texts), len(neg_texts))
    pos_acts_all = extract_layer_activations(
        model, tokenizer, pos_texts, device=device, batch_size=batch_size, pool="last"
    )
    neg_acts_all = extract_layer_activations(
        model, tokenizer, neg_texts, device=device, batch_size=batch_size, pool="last"
    )
    # pos/neg_acts_all: list of [n_texts, hidden_dim], length = n_layers + 1 (embedding + blocks)

    # Final-layer baseline concept score gap (no patching)
    pos_final = pos_acts_all[-1]   # [n_pos, dim]
    neg_final = neg_acts_all[-1]   # [n_neg, dim]
    d_final   = pos_final.mean(0) - neg_final.mean(0)
    gap_norm  = np.linalg.norm(d_final)
    d_final_unit = d_final / gap_norm if gap_norm > 1e-12 else d_final

    baseline_pos_score = float((pos_final @ d_final_unit).mean())
    baseline_neg_score = float((neg_final @ d_final_unit).mean())
    baseline_gap       = baseline_pos_score - baseline_neg_score

    log.info("  Baseline final-layer gap: %.4f  (pos=%.3f  neg=%.3f)",
             baseline_gap, baseline_pos_score, baseline_neg_score)

    # ── Step 2: sweep layers ──
    results_per_layer = []

    for layer_idx in range(n_layers):
        if device == "cuda":
            torch.cuda.empty_cache()

        # act index = layer_idx + 1 because index 0 is the embedding layer
        act_idx = min(layer_idx + 1, len(pos_acts_all) - 1)

        # Mean-shift delta at this layer
        mu_pos = pos_acts_all[act_idx].mean(axis=0)
        mu_neg = neg_acts_all[act_idx].mean(axis=0)
        delta  = mu_pos - mu_neg  # [hidden_dim]

        t0 = time.time()

        # Run neg texts with patch at layer_idx
        with MeanShiftPatcher(layers[layer_idx], delta, dtype=dtype):
            patched_neg_acts = extract_layer_activations(
                model, tokenizer, neg_texts, device=device, batch_size=batch_size, pool="last"
            )

        # Concept score recovery at the final layer
        recovery = concept_score_recovery(pos_final, neg_final, patched_neg_acts[-1])

        # Absolute final-layer concept score of patched neg
        patched_neg_score = float((patched_neg_acts[-1] @ d_final_unit).mean())

        elapsed = time.time() - t0

        results_per_layer.append({
            "layer":                 layer_idx,
            "depth_pct":             round(100.0 * layer_idx / n_layers, 1),
            "baseline_separation":   float(metrics_raw[layer_idx]["separation_fisher"]),
            "delta_magnitude":       round(float(np.linalg.norm(delta)), 4),
            "patched_neg_score":     round(patched_neg_score, 4),
            "concept_score_recovery": round(recovery, 4),
            "seconds":               round(elapsed, 1),
        })

        log.info("  L%d (%.0f%%)  recovery=%.3f  (%.1fs)",
                 layer_idx, 100.0 * layer_idx / n_layers, recovery, elapsed)

    # Peak recovery layer
    recoveries = [r["concept_score_recovery"] for r in results_per_layer]
    peak_recovery_layer = int(np.argmax(recoveries))

    return {
        "concept":                 concept,
        "model_id":                extraction_data["model_id"],
        "n_layers":                n_layers,
        "n_pairs":                 len(pairs),
        "caz_start":               boundary.caz_start if boundary else None,
        "caz_peak":                boundary.caz_peak if boundary else peak_layer,
        "caz_end":                 boundary.caz_end if boundary else None,
        "baseline_pos_score":      round(baseline_pos_score, 4),
        "baseline_neg_score":      round(baseline_neg_score, 4),
        "baseline_gap":            round(baseline_gap, 4),
        "peak_recovery_layer":     peak_recovery_layer,
        "peak_recovery_in_caz":    (
            boundary.caz_start <= peak_recovery_layer <= boundary.caz_end
            if boundary else None
        ),
        "layers":                  results_per_layer,
    }


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------

def run_model(model_id: str, concepts: list[str], args) -> None:
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.error("No extraction results for %s — run extract.py first", model_id)
        return

    log.info("=== Patch sweep: %s ===", model_id)
    log.info("Using extraction results from: %s", extraction_dir)

    device = get_device(args.device)
    dtype  = get_dtype(device)
    log_device_info(device, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    from rosetta_tools.models import vram_gb as _registry_vram
    model_vram = _registry_vram(model_id)
    single_gpu_vram = 22.0
    explicit_8bit = args.load_in_8bit
    auto_8bit = (not explicit_8bit) and model_vram > single_gpu_vram
    use_8bit = explicit_8bit or auto_8bit

    if use_8bit:
        try:
            import accelerate  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                f"8-bit loading requires accelerate and bitsandbytes: {e}\n"
                "  pip install accelerate bitsandbytes"
            ) from e
        if auto_8bit:
            log.info("Large model (%.0f GB bf16 > %.0f GB single GPU): auto 8-bit",
                     model_vram, single_gpu_vram)
        model = load_model_with_retry(AutoModel, model_id, dtype=dtype,
                                      device=device, device_map="auto",
                                      load_in_8bit=True)
    else:
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        use_multi_gpu = model_vram > 12.0 and n_gpus > 1
        effective_device_map = "auto" if use_multi_gpu else device
        if use_multi_gpu:
            log.info("Large model (%.0f GB bf16): device_map='auto' across %d GPUs",
                     model_vram, n_gpus)
        model = load_model_with_retry(AutoModel, model_id, dtype=dtype,
                                      device=device, device_map=effective_device_map)

    model.eval()
    log_vram("after model load")

    # Auto-reduce batch size under memory pressure
    if device == "cuda":
        free_gb = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated(0)) / 2**30
        if free_gb < 4.0 and args.batch_size > 2:
            args.batch_size = 2
            log.warning("Low VRAM (%.1f GiB free) — batch_size → 2", free_gb)
        if free_gb < 2.0:
            args.batch_size = 1
            log.warning("Very low VRAM (%.1f GiB free) — batch_size → 1", free_gb)

    t_start = time.time()

    for i, concept in enumerate(concepts):
        log.info("--- Concept %d/%d: %s ---", i + 1, len(concepts), concept)

        extraction_data = load_concept_directions(extraction_dir, concept)
        if extraction_data is None:
            log.warning("No extraction data for %s/%s, skipping", model_id, concept)
            continue

        # Skip if patch result already exists (idempotent re-runs)
        existing = extraction_dir / f"patch_{concept}.json"
        if existing.exists() and not args.force:
            log.info("  Already done — skipping (use --force to re-run)")
            continue

        result = patch_sweep(
            model, tokenizer, concept, extraction_data,
            device=device, n_pairs=args.n_pairs, batch_size=args.batch_size,
        )

        # Write timestamped file + latest symlink (mirrors ablate.py convention)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = extraction_dir / f"patch_{concept}_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        latest = extraction_dir / f"patch_{concept}.json"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_path.name)

        caz_peak      = result["caz_peak"]
        peak_rec      = result["peak_recovery_layer"]
        peak_rec_val  = result["layers"][peak_rec]["concept_score_recovery"]
        in_caz        = result["peak_recovery_in_caz"]
        log.info(
            "  [%s] %s → peak recovery L%d (%.3f), CAZ peak L%d, in_caz=%s",
            model_id.split("/")[-1], concept, peak_rec, peak_rec_val, caz_peak,
            in_caz if in_caz is not None else "?",
        )

    release_model(model)
    if not args.no_clean_cache:
        purge_hf_cache(model_id)

    log.info("Done: %s  (%.1fs)", model_id, time.time() - t_start)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Activation patching sweep — causal validation of CAZ layers"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Single HuggingFace model ID")
    group.add_argument("--all",   action="store_true",
                       help="Run all models with extraction results")
    parser.add_argument("--concepts",    nargs="+", default=None,
                        help="Concepts to patch (default: all with extraction data)")
    parser.add_argument("--n-pairs",     type=int, default=50,
                        help="Contrastive pairs per concept (default: 50)")
    parser.add_argument("--batch-size",  type=int, default=8)
    parser.add_argument("--device",      choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit quantization (requires bitsandbytes)")
    parser.add_argument("--no-clean-cache", action="store_true",
                        help="Keep model in HF cache after patching (default: purge)")
    parser.add_argument("--force",       action="store_true",
                        help="Re-run even if patch_<concept>.json already exists")
    args = parser.parse_args()

    models = [args.model] if args.model else discover_all_models()
    log.info("Queued %d models", len(models))

    for model_id in models:
        extraction_dir = find_extraction_dir(model_id)
        if extraction_dir is None:
            log.warning("No extraction results for %s — skipping", model_id)
            continue

        # Determine concepts
        if args.concepts:
            concepts = args.concepts
        else:
            concepts = [
                c for c in CONCEPTS
                if (extraction_dir / f"caz_{c}.json").exists()
            ]

        if not concepts:
            log.warning("No concept data found for %s — skipping", model_id)
            continue

        run_model(model_id, concepts, args)


if __name__ == "__main__":
    main()

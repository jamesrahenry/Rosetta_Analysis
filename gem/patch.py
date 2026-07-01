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
from datetime import datetime, timezone
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
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

P3_MODELS: list[str] = [
    # Pythia
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    # GPT-2
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
    # OPT
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    # Qwen2.5
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    # Gemma-2
    "google/gemma-2-2b",
    "google/gemma-2-9b",
    # Llama-3.2
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    # Mistral / Phi
    "mistralai/Mistral-7B-v0.3",
    "microsoft/phi-2",
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
    heldout_n: int = 0,
) -> dict:
    """Sweep activation patching across all layers for one concept.

    For each layer L, shifts neg hidden states at L by (μ_pos_L - μ_neg_L)
    and measures how much concept signal is recovered at the final layer.

    If heldout_n > 0, also evaluates recovery on validation-split pairs whose
    activations were not used to derive Δ_L or the concept direction, quantifying
    endogeneity inflation in the calibration-set figures.
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

    # ── Held-out baseline (validation split, never used in calibration) ──
    heldout_neg_acts_all: list | None = None
    heldout_neg_final: np.ndarray | None = None
    heldout_neg_texts: list[str] = []
    if heldout_n > 0:
        heldout_pairs = load_concept_pairs(concept, split="validation", n=heldout_n)
        _, heldout_neg_texts = texts_by_label(heldout_pairs)
        if heldout_neg_texts:
            log.info("  Caching held-out baseline (%d neg, validation split)...",
                     len(heldout_neg_texts))
            heldout_neg_acts_all = extract_layer_activations(
                model, tokenizer, heldout_neg_texts,
                device=device, batch_size=batch_size, pool="last"
            )
            heldout_neg_final = heldout_neg_acts_all[-1]
        else:
            log.warning("  No held-out pairs available for %s — skipping heldout eval", concept)
            heldout_n = 0

    # ── Step 2: sweep layers ──
    # n_tracked may be < n_layers for models with heterogeneous hidden dims
    # (e.g. OPT-350m word_embed_proj_dim=512 — boundary states filtered by extraction).
    n_tracked = len(metrics_raw)
    if n_tracked < n_layers:
        log.warning(
            "  metrics_raw has %d entries but model has %d layers — "
            "skipping last %d (heterogeneous hidden dims, e.g. OPT-350m)",
            n_tracked, n_layers, n_layers - n_tracked,
        )
    results_per_layer = []

    for layer_idx in range(n_tracked):
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

        # Held-out recovery: apply same calibration Δ_L to held-out neg pairs
        heldout_recovery: float | None = None
        if heldout_n > 0 and heldout_neg_texts and heldout_neg_final is not None:
            with MeanShiftPatcher(layers[layer_idx], delta, dtype=dtype):
                patched_heldout_acts = extract_layer_activations(
                    model, tokenizer, heldout_neg_texts,
                    device=device, batch_size=batch_size, pool="last"
                )
            heldout_recovery = round(
                concept_score_recovery(pos_final, heldout_neg_final, patched_heldout_acts[-1]),
                4,
            )

        elapsed = time.time() - t0

        row: dict = {
            "layer":                  layer_idx,
            "depth_pct":              round(100.0 * layer_idx / n_layers, 1),
            "baseline_separation":    float(metrics_raw[layer_idx]["separation_fisher"]),
            "delta_magnitude":        round(float(np.linalg.norm(delta)), 4),
            "patched_neg_score":      round(patched_neg_score, 4),
            "concept_score_recovery": round(recovery, 4),
            "seconds":                round(elapsed, 1),
        }
        if heldout_recovery is not None:
            row["heldout_concept_score_recovery"] = heldout_recovery

        results_per_layer.append(row)

        log.info("  L%d (%.0f%%)  recovery=%.3f  heldout=%s  (%.1fs)",
                 layer_idx, 100.0 * layer_idx / n_layers, recovery,
                 f"{heldout_recovery:.3f}" if heldout_recovery is not None else "n/a",
                 elapsed)

    # Peak recovery layer (calibration)
    recoveries = [r["concept_score_recovery"] for r in results_per_layer]
    peak_recovery_layer = int(np.argmax(recoveries))
    peak_calib_recovery = round(recoveries[peak_recovery_layer], 4)

    # Held-out peak recovery (at same peak layer used by calibration)
    heldout_recoveries = [r.get("heldout_concept_score_recovery") for r in results_per_layer]
    peak_heldout_recovery: float | None = None
    if any(v is not None for v in heldout_recoveries):
        peak_heldout_recovery = heldout_recoveries[peak_recovery_layer]

    result: dict = {
        "concept":                 concept,
        "model_id":                extraction_data["model_id"],
        "n_layers":                n_layers,
        "n_pairs_calibration":     len(pairs),
        "n_pairs_heldout":         len(heldout_neg_texts) if heldout_n > 0 else 0,
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
        "peak_calibration_recovery": peak_calib_recovery,
        "layers":                  results_per_layer,
    }
    if peak_heldout_recovery is not None:
        result["peak_heldout_recovery"] = peak_heldout_recovery
        result["endogeneity_inflation_pp"] = round(
            (peak_calib_recovery - peak_heldout_recovery) * 100, 2
        )
    return result


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------

def get_cohort(model_id: str) -> str:
    """Map model_id to architecture cohort label for aggregation."""
    m = model_id.lower()
    if "gemma" in m:
        return "Gemma"
    if any(x in m for x in ("qwen", "llama", "mistral")):
        return "GQA"
    return "MHA"


def run_model(model_id: str, concepts: list[str], args,
              heldout_results: list | None = None) -> None:
    models_root = Path(args.models_dir) if getattr(args, "models_dir", None) else None
    extraction_dir = find_extraction_dir(model_id, models_root)
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
    # Compare against AGGREGATE VRAM across all visible GPUs (minus an 8 GB
    # activation/overhead reserve), not just GPU 0. A model that fits in bf16
    # across multiple cards via device_map should shard, not auto-quantize —
    # 8-bit degrades the activations a patch analysis measures. (multi-GPU host fix)
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        total_vram = sum(torch.cuda.get_device_properties(i).total_memory
                         for i in range(n_gpus)) / 1e9 - 8.0
    else:
        total_vram = 22.0
    explicit_8bit = args.load_in_8bit
    auto_8bit = (not explicit_8bit) and model_vram > total_vram
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
            log.info("Large model (%.0f GB bf16 > %.0f GB total across GPUs): auto 8-bit",
                     model_vram, total_vram)
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

        heldout_n = getattr(args, "heldout_n", 0) or 0
        result = patch_sweep(
            model, tokenizer, concept, extraction_data,
            device=device, n_pairs=args.n_pairs, batch_size=args.batch_size,
            heldout_n=heldout_n,
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
        peak_rec_val  = result["peak_calibration_recovery"]
        in_caz        = result["peak_recovery_in_caz"]
        heldout_val   = result.get("peak_heldout_recovery")
        inflation     = result.get("endogeneity_inflation_pp")
        log.info(
            "  [%s] %s → calib=%.3f  heldout=%s  inflation=%s pp  L%d  in_caz=%s",
            model_id.split("/")[-1], concept,
            peak_rec_val,
            f"{heldout_val:.3f}" if heldout_val is not None else "n/a",
            f"{inflation:.1f}" if inflation is not None else "n/a",
            peak_rec,
            in_caz if in_caz is not None else "?",
        )

        if heldout_results is not None and heldout_val is not None:
            heldout_results.append({
                "model_id":                  model_id,
                "cohort":                    get_cohort(model_id),
                "concept":                   concept,
                "peak_recovery_layer":       peak_rec,
                "peak_calibration_recovery": peak_rec_val,
                "peak_heldout_recovery":     heldout_val,
                "endogeneity_inflation_pp":  inflation,
                "n_pairs_calibration":       result["n_pairs_calibration"],
                "n_pairs_heldout":           result["n_pairs_heldout"],
            })

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
    group.add_argument("--p3-corpus", action="store_true", help="Paper 3 CAZ Validation: 26 base models")
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
    parser.add_argument("--models-dir", type=str, default=None,
                        help="Root directory containing per-model extraction dirs "
                             "(default: ~/rosetta_data/paper_n250). "
                             "Pass ~/rosetta_data/models/ on GPU hosts.")
    parser.add_argument("--heldout-n", type=int, default=0,
                        help="Held-out pairs per concept from the validation split "
                             "(never used in extraction). 0 = disable. Default: 0.")
    parser.add_argument("--heldout-out", type=str, default=None,
                        help="Path for aggregated held-out recovery JSON "
                             "(required when --heldout-n > 0).")
    args = parser.parse_args()

    # --heldout-out is optional; omitting it skips the per-run aggregate write
    # (the Prefect flow does its own aggregation from per-model JSONs)

    models_root = Path(args.models_dir) if args.models_dir else None
    if args.model:
        models = [args.model]
    elif args.p3_corpus:
        models = P3_MODELS
    else:
        models = discover_all_models(models_root)
    log.info("Queued %d models", len(models))

    heldout_results: list[dict] = [] if args.heldout_n > 0 else []

    for model_id in models:
        extraction_dir = find_extraction_dir(model_id, models_root)
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

        # Skip model load entirely when every concept already has a patch result
        # (only applies when not running heldout — heldout always needs a fresh pass)
        if not args.force and args.heldout_n == 0:
            already_done = [c for c in concepts
                            if (extraction_dir / f"patch_{c}.json").exists()]
            if len(already_done) == len(concepts):
                log.info("=== Patch sweep: %s — all %d concepts done, skipping ===",
                         model_id, len(concepts))
                continue

        run_model(model_id, concepts, args,
                  heldout_results=heldout_results if args.heldout_n > 0 else None)

    # Write aggregated held-out summary
    if args.heldout_n > 0 and args.heldout_out and heldout_results:
        from collections import defaultdict

        cohort_calib: dict[str, list[float]] = defaultdict(list)
        cohort_heldout: dict[str, list[float]] = defaultdict(list)
        for r in heldout_results:
            cohort_calib[r["cohort"]].append(r["peak_calibration_recovery"])
            cohort_heldout[r["cohort"]].append(r["peak_heldout_recovery"])

        def _mean(vals: list[float]) -> float:
            return round(float(np.mean(vals)), 4) if vals else float("nan")

        cohorts = sorted(set(r["cohort"] for r in heldout_results))
        calib_by_cohort  = {c: _mean(cohort_calib[c])  for c in cohorts}
        heldout_by_cohort = {c: _mean(cohort_heldout[c]) for c in cohorts}
        inflation_by_cohort = {
            c: round((calib_by_cohort[c] - heldout_by_cohort[c]) * 100, 2)
            for c in cohorts
        }

        agg = {
            "written": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "n_pairs_heldout": args.heldout_n,
            "heldout_split": "validation",
            "calibration_recovery":  calib_by_cohort,
            "heldout_recovery":      heldout_by_cohort,
            "endogeneity_inflation_pp": inflation_by_cohort,
            "per_model_per_concept": heldout_results,
        }
        out_path = Path(args.heldout_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(agg, indent=2))
        log.info("Held-out summary → %s", out_path)


if __name__ == "__main__":
    main()

"""
ablate_gem.py — Zone-aware ablation using CAZ Geometric Evolution Maps (GEMs).

Instead of ablating a concept direction at the geometric peak (single-layer),
this script ablates the GEM's settled product at the handoff layer — the
first post-CAZ layer where the assembly product has settled.

For multimodal concepts with multiple CAZ nodes, ablation targets are chosen
by the GEM's dependency classifier: independent nodes and upstream nodes
are ablated; downstream nodes are left alone (superposition avoidance).

Three modes:
  1. Handoff ablation (default): Ablate settled direction at handoff layer
     for each target node.  Measure separation at all CAZ peaks and final layer.
  2. Compare-peak (--compare-peak): Run BOTH handoff and legacy peak ablation.
     Output side-by-side comparison.  This is the core validation test.
  3. Cascade (--cascade, Phase 3): Ablate only upstream handoffs.  Measure
     downstream propagation without intervening at downstream CAZes.

Usage
-----
    python src/ablate_gem.py --model EleutherAI/pythia-1.4b
    python src/ablate_gem.py --model EleutherAI/pythia-1.4b --compare-peak
    python src/ablate_gem.py --model EleutherAI/pythia-1.4b --cascade
    python src/ablate_gem.py --all --compare-peak

Outputs: results/<extraction_dir>/ablation_gem_<concept>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta_tools.ablation import DirectionalAblator, get_transformer_layers
from rosetta_tools.caz import compute_separation
from rosetta_tools.dataset import texts_by_label
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.models import vram_gb as _registry_vram
from rosetta_tools.gpu_utils import (
    get_device, get_dtype, log_device_info, log_vram,
    release_model, purge_hf_cache, vram_stats,
    load_model_with_retry, NumpyJSONEncoder,
)
from rosetta_tools.dataset import load_concept_pairs
from rosetta_tools.gem import (
    load_gem, ConceptGEM, GEMNode,
    find_extraction_dir, discover_all_models,
)
from rosetta_tools.paths import ROSETTA_RESULTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Local model paths (modelscope mirrors for gated/large models)
MODELSCOPE_ROOT = Path.home() / ".cache" / "modelscope" / "hub" / "models"
LOCAL_MODEL_PATHS: dict[str, Path] = {
    "google/gemma-2-9b":         MODELSCOPE_ROOT / "google"    / "gemma-2-9b",
    "mistralai/Mistral-7B-v0.3": MODELSCOPE_ROOT / "mistralai" / "Mistral-7B-v0.3",
}

CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _find_tokenizer_dir(path_str: str) -> str | None:
    """Return the directory that contains tokenizer_config.json, searching up to 2 levels."""
    p = Path(path_str)
    if not p.is_dir():
        return None
    if (p / "tokenizer_config.json").exists():
        return path_str
    for child in sorted(p.iterdir()):
        if not child.is_dir():
            continue
        if (child / "tokenizer_config.json").exists():
            return str(child)
        for grandchild in sorted(child.iterdir()):
            if grandchild.is_dir() and (grandchild / "tokenizer_config.json").exists():
                return str(grandchild)
    return None


def _populate_modelscope_configs(load_path: str, model_id: str) -> None:
    """Ensure config files are present alongside modelscope weight shards.

    Modelscope caches only the safetensors weights; config/tokenizer files must
    come from HF.  huggingface_hub ≥0.36 validates repo IDs in
    try_to_load_from_cache, rejecting absolute paths — so from_pretrained(local_path)
    crashes if config.json is missing from that directory.  We fix this by
    downloading each config file individually using the valid model_id string,
    then copying it into the weights directory so subsequent loads are fully local.
    """
    import shutil
    from huggingface_hub import hf_hub_download
    p = Path(load_path)
    if not p.is_dir():
        log.error("Modelscope path is not a directory: %s", load_path)
        return
    if (p / "config.json").exists():
        return
    config_files = [
        "config.json", "generation_config.json",
        "tokenizer_config.json", "tokenizer.json", "tokenizer.model",
        "special_tokens_map.json", "added_tokens.json",
    ]
    for fname in config_files:
        dst = p / fname
        if dst.exists():
            continue
        try:
            src = hf_hub_download(model_id, fname)
            shutil.copy(src, dst)
            log.info("Seeded %s into %s", fname, load_path)
        except Exception:
            pass  # file may not exist for this model; that's fine


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def measure_separation(
    model, tokenizer, layers,
    ablate_layers: list[int],
    ablate_directions: list[np.ndarray],
    pos_texts: list[str],
    neg_texts: list[str],
    measure_layers: list[int],
    device: str,
    batch_size: int,
) -> dict[int, float]:
    """Apply ablation at specified layers, measure separation at others.

    Parameters
    ----------
    ablate_layers : layer indices to ablate simultaneously.
    ablate_directions : concept direction at each ablation layer (numpy).
    measure_layers : layer indices where separation is measured.

    Returns
    -------
    dict mapping layer_index -> separation value.
    """
    model_dtype = next(model.parameters()).dtype

    with ExitStack() as stack:
        for layer_idx, direction in zip(ablate_layers, ablate_directions):
            stack.enter_context(
                DirectionalAblator(layers[layer_idx], direction, dtype=model_dtype)
            )

        pos_acts = extract_layer_activations(
            model, tokenizer, pos_texts, device=device,
            batch_size=batch_size, pool="last",
        )
        neg_acts = extract_layer_activations(
            model, tokenizer, neg_texts, device=device,
            batch_size=batch_size, pool="last",
        )

    results = {}
    for layer_idx in measure_layers:
        act_idx = layer_idx + 1  # extraction includes embedding at [0]
        if act_idx >= len(pos_acts):
            act_idx = len(pos_acts) - 1
        results[layer_idx] = float(compute_separation(
            pos_acts[act_idx], neg_acts[act_idx],
        ))

    return results


def measure_kl_divergence(
    model, tokenizer, layers,
    ablate_layers: list[int],
    ablate_directions: list[np.ndarray],
    texts: list[str],
    device: str,
    batch_size: int,
) -> float:
    """Measure KL divergence of output logits under ablation vs clean.

    Returns mean KL(clean || ablated) across texts.
    """
    model_dtype = next(model.parameters()).dtype

    # Clean forward pass — collect logits
    clean_logits = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**enc)
        # Last token logits
        logits = out.logits[:, -1, :]  # [batch, vocab]
        clean_logits.append(logits.float())
    clean_logits = torch.cat(clean_logits, dim=0)

    # Ablated forward pass
    ablated_logits = []
    with ExitStack() as stack:
        for layer_idx, direction in zip(ablate_layers, ablate_directions):
            stack.enter_context(
                DirectionalAblator(layers[layer_idx], direction, dtype=model_dtype)
            )

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = model(**enc)
            logits = out.logits[:, -1, :]
            ablated_logits.append(logits.float())
    ablated_logits = torch.cat(ablated_logits, dim=0)

    # KL(clean || ablated)
    clean_probs = torch.softmax(clean_logits, dim=-1)
    ablated_log_probs = torch.log_softmax(ablated_logits, dim=-1)
    kl = torch.sum(clean_probs * (torch.log(clean_probs + 1e-10) - ablated_log_probs), dim=-1)

    return float(kl.mean().item())


# ---------------------------------------------------------------------------
# Ablation modes
# ---------------------------------------------------------------------------

def run_handoff_ablation(
    model, tokenizer, layers, gem: ConceptGEM,
    pos_texts: list[str], neg_texts: list[str],
    device: str, batch_size: int, measure_kl: bool = True,
    width: int = 1,
) -> dict:
    """Ablate settled direction at handoff layer for each target node.

    Parameters
    ----------
    width : int
        Number of consecutive layers to ablate starting at the handoff.
        width=1 is the original single-layer handoff ablation.
        width=3 creates a 3-layer wall that tests zone-level intervention.

    Returns result dict with per-node and aggregate metrics.
    """
    n_layers = len(layers)

    # Measurement points: all CAZ peaks + final layer
    measure_at = sorted(set(
        [node.caz_peak for node in gem.nodes]
        + [n_layers - 1]
    ))

    # Baseline: no ablation
    baseline = measure_separation(
        model, tokenizer, layers,
        ablate_layers=[], ablate_directions=[],
        pos_texts=pos_texts, neg_texts=neg_texts,
        measure_layers=measure_at, device=device, batch_size=batch_size,
    )

    # Handoff ablation: ablate all target nodes simultaneously
    # With width>1, ablate at handoff through handoff+width-1
    target_nodes = gem.target_nodes
    ablate_layers = []
    ablate_dirs = []
    for node in target_nodes:
        direction = np.array(node.settled_direction, dtype=np.float64)
        for offset in range(width):
            layer_idx = node.handoff_layer + offset
            if layer_idx < n_layers:
                ablate_layers.append(layer_idx)
                ablate_dirs.append(direction)

    if not ablate_layers:
        log.warning("  No valid ablation layers for handoff mode")
        return {"error": "no_valid_layers"}

    ablated = measure_separation(
        model, tokenizer, layers,
        ablate_layers=ablate_layers, ablate_directions=ablate_dirs,
        pos_texts=pos_texts, neg_texts=neg_texts,
        measure_layers=measure_at, device=device, batch_size=batch_size,
    )

    # KL divergence
    kl_div = None
    if measure_kl:
        all_texts = pos_texts[:25] + neg_texts[:25]
        kl_div = measure_kl_divergence(
            model, tokenizer, layers,
            ablate_layers=ablate_layers, ablate_directions=ablate_dirs,
            texts=all_texts, device=device, batch_size=batch_size,
        )

    # Compute per-measurement-point retained percentages
    per_layer = {}
    for li in measure_at:
        bl = baseline.get(li, 0)
        ab = ablated.get(li, 0)
        retained = (100 * ab / bl) if bl > 0 else 100.0
        per_layer[li] = {
            "baseline_sep": round(bl, 4),
            "ablated_sep": round(ab, 4),
            "retained_pct": round(retained, 1),
            "sep_reduction": round(max(0, 1 - ab / bl) if bl > 0 else 0, 4),
        }

    # Final layer is the key metric
    final = per_layer.get(n_layers - 1, {})

    return {
        "mode": "handoff",
        "width": width,
        "n_targets": len(target_nodes),
        "ablation_layers": ablate_layers,
        "per_layer": {str(k): v for k, v in per_layer.items()},
        "final_retained_pct": final.get("retained_pct", 100.0),
        "final_sep_reduction": final.get("sep_reduction", 0.0),
        "kl_divergence": round(kl_div, 6) if kl_div is not None else None,
    }


def run_peak_ablation(
    model, tokenizer, layers, gem: ConceptGEM,
    pos_texts: list[str], neg_texts: list[str],
    device: str, batch_size: int, measure_kl: bool = True,
    width: int = 1,
) -> dict:
    """Legacy peak ablation: ablate dom_vector at CAZ peak for each target.

    This is the baseline for comparison with handoff ablation.
    With width>1, ablates at peak through peak+width-1.
    """
    n_layers = len(layers)

    measure_at = sorted(set(
        [node.caz_peak for node in gem.nodes]
        + [n_layers - 1]
    ))

    baseline = measure_separation(
        model, tokenizer, layers,
        ablate_layers=[], ablate_directions=[],
        pos_texts=pos_texts, neg_texts=neg_texts,
        measure_layers=measure_at, device=device, batch_size=batch_size,
    )

    # Peak ablation: use dom_vector at the CAZ peak for each target
    target_nodes = gem.target_nodes
    ablate_layers = []
    ablate_dirs = []
    for node in target_nodes:
        # Use the direction at the peak, not the settled direction
        peak_idx = node.caz_peak - node.caz_start
        if peak_idx < 0 or peak_idx >= len(node.concept_thread.directions):
            peak_idx = 0
        peak_dir = np.array(
            node.concept_thread.directions[peak_idx], dtype=np.float64,
        )
        for offset in range(width):
            layer_idx = node.caz_peak + offset
            if layer_idx < n_layers:
                ablate_layers.append(layer_idx)
                ablate_dirs.append(peak_dir)

    if not ablate_layers:
        return {"error": "no_valid_layers"}

    ablated = measure_separation(
        model, tokenizer, layers,
        ablate_layers=ablate_layers, ablate_directions=ablate_dirs,
        pos_texts=pos_texts, neg_texts=neg_texts,
        measure_layers=measure_at, device=device, batch_size=batch_size,
    )

    kl_div = None
    if measure_kl:
        all_texts = pos_texts[:25] + neg_texts[:25]
        kl_div = measure_kl_divergence(
            model, tokenizer, layers,
            ablate_layers=ablate_layers, ablate_directions=ablate_dirs,
            texts=all_texts, device=device, batch_size=batch_size,
        )

    per_layer = {}
    for li in measure_at:
        bl = baseline.get(li, 0)
        ab = ablated.get(li, 0)
        retained = (100 * ab / bl) if bl > 0 else 100.0
        per_layer[li] = {
            "baseline_sep": round(bl, 4),
            "ablated_sep": round(ab, 4),
            "retained_pct": round(retained, 1),
            "sep_reduction": round(max(0, 1 - ab / bl) if bl > 0 else 0, 4),
        }

    final = per_layer.get(n_layers - 1, {})

    return {
        "mode": "peak",
        "n_targets": len(target_nodes),
        "ablation_layers": ablate_layers,
        "per_layer": {str(k): v for k, v in per_layer.items()},
        "final_retained_pct": final.get("retained_pct", 100.0),
        "final_sep_reduction": final.get("sep_reduction", 0.0),
        "kl_divergence": round(kl_div, 6) if kl_div is not None else None,
    }


def run_cascade_ablation(
    model, tokenizer, layers, gem: ConceptGEM,
    pos_texts: list[str], neg_texts: list[str],
    device: str, batch_size: int,
) -> dict:
    """Cascade mode: ablate only upstream nodes, measure downstream propagation.

    Tests whether ablating upstream CAZes suppresses downstream assembly
    without directly intervening at the downstream site.
    """
    n_layers = len(layers)

    # Identify upstream and downstream nodes
    upstream_nodes = gem.upstream_nodes
    downstream_nodes = [
        n for n, t in zip(gem.nodes, gem.node_types or [])
        if t == "downstream"
    ]

    if not upstream_nodes:
        return {"error": "no_upstream_nodes", "note": "All nodes are independent"}

    measure_at = sorted(set(
        [node.caz_peak for node in gem.nodes]
        + [n_layers - 1]
    ))

    # Baseline
    baseline = measure_separation(
        model, tokenizer, layers,
        ablate_layers=[], ablate_directions=[],
        pos_texts=pos_texts, neg_texts=neg_texts,
        measure_layers=measure_at, device=device, batch_size=batch_size,
    )

    # Ablate only upstream handoffs
    ablate_layers = []
    ablate_dirs = []
    for node in upstream_nodes:
        if node.handoff_layer < n_layers:
            ablate_layers.append(node.handoff_layer)
            ablate_dirs.append(np.array(node.settled_direction, dtype=np.float64))

    ablated = measure_separation(
        model, tokenizer, layers,
        ablate_layers=ablate_layers, ablate_directions=ablate_dirs,
        pos_texts=pos_texts, neg_texts=neg_texts,
        measure_layers=measure_at, device=device, batch_size=batch_size,
    )

    # Downstream propagation: how much did downstream CAZ peaks lose?
    downstream_results = []
    for node in downstream_nodes:
        bl = baseline.get(node.caz_peak, 0)
        ab = ablated.get(node.caz_peak, 0)
        retained = (100 * ab / bl) if bl > 0 else 100.0
        downstream_results.append({
            "caz_index": node.caz_index,
            "caz_peak": node.caz_peak,
            "depth_pct": node.depth_pct,
            "baseline_sep": round(bl, 4),
            "ablated_sep": round(ab, 4),
            "retained_pct": round(retained, 1),
            "propagation_suppressed": retained < 80.0,
        })

    final_bl = baseline.get(n_layers - 1, 0)
    final_ab = ablated.get(n_layers - 1, 0)
    final_retained = (100 * final_ab / final_bl) if final_bl > 0 else 100.0

    return {
        "mode": "cascade",
        "n_upstream_ablated": len(upstream_nodes),
        "n_downstream_measured": len(downstream_nodes),
        "ablation_layers": ablate_layers,
        "downstream_propagation": downstream_results,
        "final_retained_pct": round(final_retained, 1),
    }


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(
    model_id: str,
    concepts: list[str],
    args,
) -> bool:
    """Run GEM ablation for one model. Returns True on success."""
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.error("No extraction results for %s", model_id)
        return False

    # Check that GEMs exist
    available = []
    for concept in concepts:
        gem_path = extraction_dir / f"gem_{concept}.json"
        if gem_path.exists():
            available.append(concept)
    if not available:
        log.error("No GEMs for %s. Run build_gems.py first.", model_id)
        return False

    log.info("=== GEM ablation: %s (%d concepts) ===", model_id, len(available))

    device = get_device(args.device)
    dtype = get_dtype(device)

    if device.startswith("cuda"):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    log_device_info(device, dtype)

    # Resolve local model path (modelscope mirrors, --local-path override)
    _override = getattr(args, "local_path", "")
    if _override:
        LOCAL_MODEL_PATHS[model_id] = Path(_override)
    load_path = str(LOCAL_MODEL_PATHS.get(model_id, model_id))
    if load_path != model_id:
        log.info("Loading from local path: %s", load_path)

    _is_local = load_path != model_id
    if _is_local and not Path(load_path).is_dir():
        log.warning("Local path %s does not exist; falling back to HF loading", load_path)
        _is_local = False
        load_path = model_id
    if _is_local:
        # Model weights are in modelscope cache; tokenizer may be in HF cache (downloaded
        # separately on first run) or in the modelscope tree.  Try both in order.
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            log.info("Tokenizer loaded from HF cache")
        except Exception:
            pass
        if tokenizer is None:
            tok_dir = _find_tokenizer_dir(load_path)
            if tok_dir:
                log.info("Tokenizer loaded from modelscope: %s", tok_dir)
                tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        if tokenizer is None:
            log.info("Tokenizer not in HF cache or modelscope tree; downloading from HF…")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(load_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    model_vram = _registry_vram(model_id)
    single_gpu_vram = 22.0  # L4 capacity; ablation hooks need headroom beyond model weights

    # Models that don't fit on a single GPU in bf16 are loaded in 8-bit instead of
    # being spread via device_map="auto". Activation patching across GPUs causes OOM
    # even when the weights technically fit — ablation forward passes need 4-8 GB
    # headroom on top of the model. 8-bit halves the footprint to a single GPU.
    explicit_8bit = getattr(args, "load_8bit", False)
    auto_8bit = (not explicit_8bit) and model_vram > single_gpu_vram
    use_8bit = explicit_8bit or auto_8bit

    use_multi_gpu = (not use_8bit) and model_vram > 12.0 and n_gpus > 1
    effective_device_map = "auto" if use_multi_gpu else device

    if auto_8bit:
        log.info("Large model (%.0f GB bf16 > %.0f GB single GPU): auto 8-bit on %s",
                 model_vram, single_gpu_vram, device)
    elif use_multi_gpu:
        log.info("Large model (%.0f GB bf16): device_map='auto' across %d GPUs",
                 model_vram, n_gpus)

    load_kwargs = dict(torch_dtype=dtype, device_map=effective_device_map)
    if use_8bit:
        load_kwargs["load_in_8bit"] = True
        load_kwargs.pop("torch_dtype", None)  # let bitsandbytes handle dtype
        if explicit_8bit:
            log.info("Loading in 8-bit quantization (explicit flag)")
        else:
            log.info("Loading in 8-bit quantization (auto: model too large for single GPU)")
    if _is_local:
        _populate_modelscope_configs(load_path, model_id)
        model = AutoModelForCausalLM.from_pretrained(load_path, **load_kwargs)
    else:
        model = load_model_with_retry(AutoModelForCausalLM, model_id, dtype=dtype,
                                      device=device, device_map=effective_device_map,
                                      load_in_8bit=use_8bit)
    model.eval()
    log_vram("after model load")

    transformer_layers = get_transformer_layers(model)

    t_start = time.time()
    all_results = []

    for ci, concept in enumerate(available):
        log.info("--- Concept %d/%d: %s ---", ci + 1, len(available), concept)

        gem_path = extraction_dir / f"gem_{concept}.json"
        gem = load_gem(gem_path)

        if gem.n_nodes == 0:
            log.warning("  Empty GEM, skipping")
            continue

        # Load contrastive pairs
        pairs = load_concept_pairs(concept, n=args.n_pairs)
        pos_texts, neg_texts = texts_by_label(pairs)
        pos_texts = [t for t in pos_texts if t and t.strip()]
        neg_texts = [t for t in neg_texts if t and t.strip()]

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        result = {
            "model_id": model_id,
            "concept": concept,
            "attention_paradigm": gem.nodes[0].attention_paradigm,
            "n_nodes": gem.n_nodes,
            "n_targets": len(gem.ablation_targets or []),
            "node_types": gem.node_types,
            "n_pairs": len(pairs),
        }

        # Handoff ablation (always run)
        width = getattr(args, "width", 1)
        width_label = f" (width={width})" if width > 1 else ""
        log.info("  Running handoff ablation%s...", width_label)
        handoff = run_handoff_ablation(
            model, tokenizer, transformer_layers, gem,
            pos_texts, neg_texts, device, args.batch_size,
            width=width,
        )
        handoff["timestamp"] = datetime.now(timezone.utc).isoformat()
        result["handoff"] = handoff
        log.info("  Handoff%s: final_retained=%.1f%%, sep_reduction=%.3f, KL=%.4f",
                 width_label,
                 handoff.get("final_retained_pct", -1),
                 handoff.get("final_sep_reduction", -1),
                 handoff.get("kl_divergence", -1) or -1)

        # Compare-peak mode
        if args.compare_peak:
            log.info("  Running peak ablation%s (comparison)...", width_label)
            peak = run_peak_ablation(
                model, tokenizer, transformer_layers, gem,
                pos_texts, neg_texts, device, args.batch_size,
                width=width,
            )
            peak["timestamp"] = datetime.now(timezone.utc).isoformat()
            result["peak"] = peak

            # Compute comparison
            h_ret = handoff.get("final_retained_pct", 100)
            p_ret = peak.get("final_retained_pct", 100)
            diff = p_ret - h_ret  # positive = handoff better
            result["comparison"] = {
                "handoff_retained_pct": h_ret,
                "peak_retained_pct": p_ret,
                "retained_diff_pp": round(diff, 1),
                "handoff_better": diff > 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "handoff_kl": handoff.get("kl_divergence"),
                "peak_kl": peak.get("kl_divergence"),
                "width": width,
            }

            log.info("  Peak%s: final_retained=%.1f%%, sep_reduction=%.3f",
                     width_label,
                     peak.get("final_retained_pct", -1),
                     peak.get("final_sep_reduction", -1))
            log.info("  Δ retained: %.1f pp (%s)",
                     diff, "HANDOFF WINS" if diff > 0 else "PEAK WINS" if diff < 0 else "TIE")

        # Cascade mode
        if args.cascade:
            log.info("  Running cascade ablation...")
            cascade = run_cascade_ablation(
                model, tokenizer, transformer_layers, gem,
                pos_texts, neg_texts, device, args.batch_size,
            )
            cascade["timestamp"] = datetime.now(timezone.utc).isoformat()
            result["cascade"] = cascade
            if "error" not in cascade:
                n_suppressed = sum(
                    1 for d in cascade.get("downstream_propagation", [])
                    if d.get("propagation_suppressed")
                )
                log.info("  Cascade: %d/%d downstream suppressed, final_retained=%.1f%%",
                         n_suppressed, cascade.get("n_downstream_measured", 0),
                         cascade.get("final_retained_pct", -1))
            else:
                log.info("  Cascade: %s", cascade.get("note", cascade.get("error")))

        # Save per-concept result — merge into existing file to preserve
        # results from prior runs (e.g. cascade + compare-peak coexist).
        # Width-aware: keep the higher-width handoff/peak/comparison.
        out_path = extraction_dir / f"ablation_gem_{concept}.json"
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text())
                # For mode-specific keys, keep existing if it has higher width
                for key in ("handoff", "peak"):
                    if key in existing and key in result:
                        old_w = existing[key].get("width", 1)
                        new_w = result[key].get("width", 1)
                        if old_w > new_w:
                            result[key] = existing[key]
                if "comparison" in existing and "comparison" in result:
                    old_w = existing["comparison"].get("width", 1)
                    new_w = result["comparison"].get("width", 1)
                    if old_w > new_w:
                        result["comparison"] = existing["comparison"]
                # Merge: new keys overwrite, but preserve old keys not in result
                existing.update(result)
                result = existing
            except (json.JSONDecodeError, OSError):
                pass  # corrupt file — overwrite entirely
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, cls=NumpyJSONEncoder)

        all_results.append(result)

    total_elapsed = time.time() - t_start

    release_model(model)
    if not getattr(args, "no_clean_cache", False) and model_id not in LOCAL_MODEL_PATHS:
        purge_hf_cache(model_id)

    # Print summary
    if all_results and args.compare_peak:
        print(f"\n{'='*72}")
        print(f"COMPARISON SUMMARY: {model_id}")
        print(f"{'='*72}")
        print(f"{'Concept':<18} {'Handoff':>10} {'Peak':>10} {'Δ (pp)':>10} {'Winner':>12}")
        print(f"{'-'*72}")
        for r in all_results:
            comp = r.get("comparison", {})
            h = comp.get("handoff_retained_pct", -1)
            p = comp.get("peak_retained_pct", -1)
            d = comp.get("retained_diff_pp", 0)
            winner = "HANDOFF" if d > 0 else "PEAK" if d < 0 else "TIE"
            print(f"  {r['concept']:<16} {h:>8.1f}%  {p:>8.1f}%  {d:>+8.1f}  {winner:>10}")
        print()

    log.info("Done: %s  (%.1fs total)", model_id, total_elapsed)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zone-aware ablation using CAZ GEMs (Geometric Evolution Maps)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Single model ID")
    group.add_argument("--all", action="store_true",
                       help="Run all models with GEM data")
    parser.add_argument("--concepts", nargs="+", default=None,
                        help="Subset of concepts (default: all 7)")
    parser.add_argument("--n-pairs", type=int, default=50,
                        help="Number of contrastive pairs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--compare-peak", action="store_true",
                        help="Also run legacy peak ablation for A/B comparison")
    parser.add_argument("--cascade", action="store_true",
                        help="Run cascade ablation (upstream-only, Phase 3)")
    parser.add_argument("--width", type=int, default=1,
                        help="Ablation width: number of consecutive layers to ablate "
                             "at each handoff/peak point (default: 1, try 3 for zone-level)")
    parser.add_argument("--8bit", action="store_true", dest="load_8bit",
                        help="Load model in 8-bit quantization (requires bitsandbytes)")
    parser.add_argument("--no-clean-cache", action="store_true",
                        help="Keep models in HF cache")
    parser.add_argument("--local-path", default="",
                        help="Override local weight path for --model "
                             "(e.g. /tmp/Mistral-7B-v0.3)")
    parser.add_argument("--skip-model", action="append", default=[],
                        metavar="MODEL_ID", help="Skip this model")
    args = parser.parse_args()

    concepts = args.concepts or CONCEPTS

    if args.all:
        models = discover_all_models()
        log.info("Found %d models", len(models))
    else:
        models = [args.model]

    if args.skip_model:
        skip = set(args.skip_model)
        models = [m for m in models if m not in skip]

    any_failed = False
    for model_id in models:
        if not run_model(model_id, concepts, args):
            any_failed = True

    if any_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ablate_propagation.py — Causal ablation PROPAGATION sweep.

The existing ablate.py answers "ablate the concept direction at layer L, how much
separation is left AT L?" — a same-layer, single-readout measurement. It cannot
distinguish a transient hit (the model re-derives the concept downstream) from a
permanent one.

This script answers the missing question: ablate at L, then measure the concept's
Fisher separation at EVERY layer L' — and the collateral output KL. The result is an
L x L' propagation matrix:

    propagation_matrix[L][L'] = separation at layer L' after ablating the concept
                                direction at layer L

The diagonal-ish band reproduces ablate.py's same-layer number; the off-diagonal
(L' > L) is new: it shows whether the concept recovers downstream (transient) or
stays suppressed (permanent). Compare against the IlyaGusev abliterated model, where
the direction is removed from the *weights* permanently at all layers.

COST: one ablated forward pass already produces hidden states for every layer —
ablate.py just discards all but one. So the whole matrix costs n_layers ablated
forward-sets (+ n_layers KL passes), the SAME order as the existing per-layer sweep.
We add instrumentation, not compute.

Methodology notes:
  - Direction ablated at L is the layer-L dom_vector (centroid difference), matching
    ablate.py. This is a UNIMODAL intervention (one direction); a follow-up multimodal
    variant (top-k directions) is a separate experiment.
  - Fisher separation (compute_separation) is basis-free per layer, so measuring it at
    L' is a fair readout regardless of the layer-L' concept direction.

Output: ablation_propagation_{concept}_{ts}.json (+ ablation_propagation_{concept}.json
symlink), written alongside the extraction results. Never collides with ablate.py's
ablation_{concept}.json.

Usage:
    python gem/ablate_propagation.py --model EleutherAI/pythia-1.4b --n-pairs 250
    python gem/ablate_propagation.py --models EleutherAI/pythia-6.9b openai-community/gpt2-xl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Reuse the GPU-tested machinery from ablate.py rather than duplicating it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablate import (  # noqa: E402
    CONCEPTS, CAPABILITY_PROMPTS,
    load_concept_directions, measure_kl_with_ablation,
)

from rosetta_tools.gpu_utils import (  # noqa: E402
    get_device, get_dtype, log_device_info, log_vram, release_model,
    purge_hf_cache, load_causal_lm,
)
from rosetta_tools.extraction import extract_layer_activations  # noqa: E402
from rosetta_tools.caz import compute_separation, LayerMetrics, find_caz_boundary  # noqa: E402
from rosetta_tools.ablation import (  # noqa: E402
    DirectionalAblator, DirectionalShifter, get_transformer_layers,
    kl_divergence_from_logits,
)
from rosetta_tools.dataset import load_concept_pairs, texts_by_label  # noqa: E402
from rosetta_tools.gem import find_extraction_dir, discover_all_models, _model_slug  # noqa: E402
from rosetta_tools.models import vram_gb as _registry_vram  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# All-layer separation under one ablation
# ---------------------------------------------------------------------------

def measure_all_layer_separation(
    model, tokenizer, layers: list, layer_idx: int, make_ctx,
    pos_texts: list[str], neg_texts: list[str], device: str, batch_size: int,
    n_tracked: int,
) -> list[float]:
    """Intervene at ``layer_idx`` (via the ``make_ctx`` context-manager factory);
    return Fisher separation at every tracked layer L' (0..n_tracked-1). One forward
    pass per class yields all downstream readouts for free. ``make_ctx`` builds the
    intervention (DirectionalAblator for subtractive, DirectionalShifter for additive)
    given the layer module."""
    with make_ctx(layers[layer_idx]):
        pos_acts = extract_layer_activations(
            model, tokenizer, pos_texts, device=device, batch_size=batch_size, pool="last"
        )
        neg_acts = extract_layer_activations(
            model, tokenizer, neg_texts, device=device, batch_size=batch_size, pool="last"
        )

    # extraction output index 0 is the embedding layer; layer L' lives at index L'+1.
    seps: list[float] = []
    n_avail = min(len(pos_acts), len(neg_acts))
    for lp in range(n_tracked):
        act_idx = lp + 1
        if act_idx >= n_avail:
            seps.append(float("nan"))
            continue
        seps.append(round(float(compute_separation(pos_acts[act_idx], neg_acts[act_idx])), 4))
    return seps


def measure_kl_with_ctx(
    model, tokenizer, layers: list, layer_idx: int, make_ctx,
    baseline_logits: list, prompts: list[str], device: str,
) -> float:
    """Mean output-KL under the intervention built by ``make_ctx`` at ``layer_idx``.
    Generalises ablate.measure_kl_with_ablation to any intervention context manager."""
    kl_values = []
    with make_ctx(layers[layer_idx]):
        for i, prompt in enumerate(prompts):
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**enc)
            kl_values.append(kl_divergence_from_logits(baseline_logits[i], out.logits[0, -1, :].cpu()))
    return float(np.mean(kl_values))


# ---------------------------------------------------------------------------
# Per-concept propagation sweep
# ---------------------------------------------------------------------------

def _sweep_one_intervention(
    model, tokenizer, layers, metrics_raw, n_tracked, n_layers,
    pos_texts, neg_texts, baseline_logits, base_last,
    make_ctx_for, label, device, batch_size,
) -> dict:
    """Run the L×L′ sweep for ONE intervention. ``make_ctx_for(L, dom, raw_distance)``
    returns a context-manager factory that takes a layer module."""
    matrix, output_kl, same_layer, recovery_ratio = [], [], [], []
    for L in range(n_tracked):
        if device == "cuda":
            torch.cuda.empty_cache()
        dom = np.array(metrics_raw[L]["dom_vector"], dtype=np.float64)
        raw_distance = float(metrics_raw[L].get("raw_distance", 0.0))
        make_ctx = make_ctx_for(L, dom, raw_distance)
        t0 = time.time()

        row = measure_all_layer_separation(
            model, tokenizer, layers, L, make_ctx,
            pos_texts, neg_texts, device, batch_size, n_tracked,
        )
        kl = measure_kl_with_ctx(
            model, tokenizer, layers, L, make_ctx, baseline_logits, CAPABILITY_PROMPTS, device,
        )
        matrix.append(row)
        output_kl.append(round(float(kl), 6))
        same_layer.append(row[L] if L < len(row) else float("nan"))
        last = row[-1] if row and not np.isnan(row[-1]) else float("nan")
        recovery_ratio.append(round(last / base_last, 4) if not np.isnan(last) else float("nan"))
        log.info(
            "  [%s] @L%d (%.0f%%): same-layer S=%.3f  final S=%.3f (recov %.2f)  kl=%.4f  (%.1fs)",
            label, L, 100.0 * L / n_layers, same_layer[L],
            row[-1] if row else float("nan"), recovery_ratio[L], kl, time.time() - t0,
        )
    return {"propagation_matrix": matrix, "same_layer_separation": same_layer,
            "recovery_ratio": recovery_ratio, "output_kl": output_kl}


def propagation_sweep(
    model, tokenizer, concept: str, extraction_data: dict,
    device: str, n_pairs: int, batch_size: int, intervention: str = "ablate",
    split: str = "train",
) -> dict:
    pairs = load_concept_pairs(concept, n=n_pairs or 250, split=split)
    pos_texts, neg_texts = texts_by_label(pairs)

    layers = get_transformer_layers(model)
    n_layers = len(layers)
    metrics_raw = extraction_data["layer_data"]["metrics"]
    n_tracked = len(metrics_raw)

    baseline_seps = [round(float(m["separation_fisher"]), 4) for m in metrics_raw]
    peak_layer = extraction_data["layer_data"]["peak_layer"]

    layer_metrics = [
        LayerMetrics(m["layer"], m["separation_fisher"], m["coherence"], m.get("velocity", 0.0))
        for m in metrics_raw
    ]
    try:
        boundary = find_caz_boundary(layer_metrics)
    except Exception:
        boundary = None

    # baseline output logits for KL (no intervention)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    baseline_logits = []
    for prompt in CAPABILITY_PROMPTS:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        baseline_logits.append(out.logits[0, -1, :].cpu())

    base_last = baseline_seps[-1] if baseline_seps[-1] > 0 else 1.0
    mdtype = torch.bfloat16 if next(model.parameters()).dtype == torch.bfloat16 else torch.float32

    result = {
        "concept": concept,
        "model_id": extraction_data["model_id"],
        "n_layers": n_layers,
        "n_tracked": n_tracked,
        "n_pairs": len(pairs),
        "peak_layer": peak_layer,
        "caz_start": boundary.caz_start if boundary else None,
        "caz_peak": boundary.caz_peak if boundary else peak_layer,
        "caz_end": boundary.caz_end if boundary else None,
        "baseline_separation": baseline_seps,
        "ablation_layers": list(range(n_tracked)),
        "direction": "per_layer_dom_vector",
    }

    if intervention == "add":
        # Additive "trench": shift activations by ±1 class-gap along the concept axis.
        # dom_vector is unit-norm; raw_distance is the centroid gap, so shift = ±raw·dom.
        result["intervention"] = "additive_centroid_gap"
        result["signs"] = ["pos", "neg"]
        for sign, tag in ((+1.0, "pos"), (-1.0, "neg")):
            log.info("  --- additive sweep (%s, coeff=%+d × gap) ---", tag, int(sign))
            def make_ctx_for(L, dom, raw, _sign=sign):
                shift = _sign * raw * dom                 # raw centroid difference, signed
                return lambda layer: DirectionalShifter(layer, shift, coefficient=1.0, dtype=mdtype)
            sub = _sweep_one_intervention(
                model, tokenizer, layers, metrics_raw, n_tracked, n_layers,
                pos_texts, neg_texts, baseline_logits, base_last,
                make_ctx_for, f"add{'+' if sign > 0 else '-'}", device, batch_size,
            )
            result[f"propagation_matrix_{tag}"] = sub["propagation_matrix"]
            result[f"same_layer_separation_{tag}"] = sub["same_layer_separation"]
            result[f"recovery_ratio_{tag}"] = sub["recovery_ratio"]
            result[f"output_kl_{tag}"] = sub["output_kl"]
    else:
        result["intervention"] = "unimodal_orthogonal_projection"
        def make_ctx_for(L, dom, raw):
            return lambda layer: DirectionalAblator(layer, dom, dtype=mdtype)
        sub = _sweep_one_intervention(
            model, tokenizer, layers, metrics_raw, n_tracked, n_layers,
            pos_texts, neg_texts, baseline_logits, base_last,
            make_ctx_for, "ablate", device, batch_size,
        )
        result.update(sub)   # propagation_matrix, same_layer_separation, recovery_ratio, output_kl

    return result


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _result_valid(extraction_dir: Path, model_id: str, concept: str, n_pairs: int,
                  prefix: str = "ablation_propagation") -> bool:
    p = extraction_dir / f"{prefix}_{concept}.json"
    if not p.exists():
        return False
    try:
        d = json.loads(p.read_text())
        # additive results carry propagation_matrix_pos instead of propagation_matrix
        has_matrix = bool(d.get("propagation_matrix") or d.get("propagation_matrix_pos"))
        return (d.get("model_id") == model_id and d.get("concept") == concept
                and d.get("n_pairs", 0) >= n_pairs and has_matrix)
    except (json.JSONDecodeError, OSError):
        return False


def run_model(model_id: str, concepts: list[str], args) -> None:
    models_root = Path(args.models_dir) if getattr(args, "models_dir", None) else None
    extraction_dir = find_extraction_dir(model_id, models_root)
    if extraction_dir is None:
        # Fallback for caz-only downloads: the propagation job fetches just caz_*.json,
        # which has no run_summary.json (the gate find_extraction_dir requires). We only
        # need the per-layer caz files (dom_vectors), so accept a slug dir that has them.
        root = models_root or Path("~/rosetta_data/paper_n250").expanduser()
        cand = root / _model_slug(model_id)
        if cand.is_dir() and any(cand.glob("caz_*.json")):
            extraction_dir = cand
            log.info("Using caz-only extraction dir (no run_summary.json): %s", cand)
    if extraction_dir is None:
        log.error("No extraction results for %s — run extract.py first", model_id)
        return

    # Additive results live in a separate file family so they never collide with the
    # subtractive matrices (and the HF upload glob picks both up).
    prefix = "ablation_propagation_add" if args.intervention == "add" else "ablation_propagation"

    if not args.force:
        todo = [c for c in concepts if not _result_valid(extraction_dir, model_id, c, args.n_pairs, prefix)]
        if not todo:
            log.info("=== Propagation: %s — all %d concepts done, skipping ===", model_id, len(concepts))
            return
        concepts = todo

    log.info("=== Propagation sweep: %s (%d concepts) ===", model_id, len(concepts))
    log.info("Extraction results: %s", extraction_dir)

    device = get_device(args.device)
    dtype = get_dtype(device)
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float32":
        dtype = torch.float32
    log_device_info(device, dtype)

    # Shard large models across GPUs, or they OOM at load on a single card (pythia-12b
    # is 24 GB bf16 > one 20 GB GPU). load_causal_lm upgrades device_map='auto' to
    # 'balanced' internally (even split + activation headroom, no disk offload).
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    single_gpu_limit = (torch.cuda.get_device_properties(0).total_memory / 1e9 - 4.0) if n_gpus > 0 else 0
    device_map = "auto" if (_registry_vram(model_id) > single_gpu_limit and n_gpus > 1) else None
    if device_map:
        log.info("Large model (%.0f GB bf16 > %.0f GB single-GPU limit): sharding across %d GPUs",
                 _registry_vram(model_id), single_gpu_limit, n_gpus)

    model, tokenizer = load_causal_lm(model_id, device, dtype, device_map=device_map)
    log_vram("after model load")

    if device == "cuda":
        free_gb = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated(0)) / 2**30
        if free_gb < 4.0 and args.batch_size > 2:
            log.warning("Low VRAM (%.1f GiB) — batch_size %d → 2", free_gb, args.batch_size)
            args.batch_size = 2
        if free_gb < 2.0 and args.batch_size > 1:
            args.batch_size = 1

    t_start = time.time()
    for i, concept in enumerate(concepts):
        log.info("--- Concept %d/%d: %s ---", i + 1, len(concepts), concept)
        extraction_data = load_concept_directions(extraction_dir, concept)
        if extraction_data is None:
            log.warning("No extraction data for %s, skipping", concept)
            continue
        extracted_n = (extraction_data.get("layer_data", {}).get("n_pairs")
                       or extraction_data.get("n_pairs"))
        n_pairs = args.n_pairs
        if extracted_n and n_pairs > extracted_n:
            log.warning("  n_pairs %d > extracted %d — clamping", n_pairs, extracted_n)
            n_pairs = extracted_n

        result = propagation_sweep(
            model, tokenizer, concept, extraction_data,
            device=device, n_pairs=n_pairs, batch_size=args.batch_size,
            intervention=args.intervention, split=args.split,
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = extraction_dir / f"{prefix}_{concept}_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        latest = extraction_dir / f"{prefix}_{concept}.json"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_path.name)
        log.info("  wrote %s", out_path.name)

    release_model(model)
    if getattr(args, "clean_cache", False):
        purge_hf_cache(model_id)
    log.info("Done: %s  (%.1fs total)", model_id, time.time() - t_start)


# Fine-grained-depth pilot set (MHA = clean Fisher↔ablation mapping; base models).
PILOT_MODELS = [
    "EleutherAI/pythia-1.4b",   # 24L — dev/validate
    "EleutherAI/pythia-6.9b",   # 32L — primary in-depth
    "EleutherAI/pythia-12b",    # 36L — bigger MHA
    "openai-community/gpt2-xl", # 48L — finest depth grid
]


def parse_args():
    p = argparse.ArgumentParser(description="Causal ablation propagation sweep (ablate@L → measure all L′)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", type=str, help="Single HuggingFace model ID")
    g.add_argument("--models", nargs="+", help="Explicit list of model IDs")
    g.add_argument("--pilot", action="store_true", help="Fine-grained-depth pilot set (4 MHA base models)")
    g.add_argument("--all", action="store_true", help="All models with extraction results")
    p.add_argument("--concepts", nargs="+", default=None, help="Concepts (default: all 17)")
    p.add_argument("--models-dir", type=str, default=None,
                   help="Root of per-model extraction dirs (default ~/rosetta_data/paper_n250; "
                        "pass ~/rosetta_data/models/ on GPU hosts)")
    p.add_argument("--intervention", choices=["ablate", "add"], default="ablate",
                   help="ablate = subtractive orthogonal projection (default); "
                        "add = additive ±1-gap shift (both signs) — the 'trench'")
    p.add_argument("--split", choices=["train", "validation", "all"], default="train",
                   help="Pair split; use 'all' for concepts outside the train/val split map (e.g. refusal)")
    p.add_argument("--n-pairs", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    p.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="auto")
    p.add_argument("--clean-cache", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    concepts = args.concepts or CONCEPTS
    models_root = Path(args.models_dir) if args.models_dir else None

    if args.all:
        models = discover_all_models(models_root)
        log.info("Found %d models with extraction results", len(models))
    elif args.pilot:
        models = PILOT_MODELS
    elif args.models:
        models = args.models
    else:
        models = [args.model]

    for m in models:
        try:
            run_model(m, concepts, args)
        except Exception:
            log.exception("Model %s failed — continuing", m)


if __name__ == "__main__":
    main()

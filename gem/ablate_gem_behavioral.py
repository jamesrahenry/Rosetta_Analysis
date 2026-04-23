"""
ablate_gem_behavioral.py
========================
Zone-level behavioral validation of GEM handoff ablation.

The single-layer behavioral pilot (ablate_behavioral_pilot.py) showed 45%
peak wins vs. chance of 50% — a null result explained by concept directions
being distributed across many layers.  This script tests the zone-level
prediction: ablating across the full GEM handoff window (width=3 by default)
should produce stronger behavioral suppression than the single-layer peak.

For each (model, concept) pair:
  1. Load GEM data → settled_direction, handoff_layer, window width.
  2. For each probe sentence, measure logit_diff under four conditions:
       - baseline  : no ablation
       - gem_zone  : settled_direction ablated across handoff window (width=3)
       - peak_single: concept direction ablated at single CAZ peak layer
       - random    : random unit vector ablated across handoff window (zone null)
  3. Suppression = logit_diff(baseline) − logit_diff(condition)
     Prediction: gem_zone >> peak_single; random ≈ 0

Uses the same probe sentences as ablate_behavioral_pilot.py for direct
comparison with the single-layer results.

Usage
-----
    cd ~/caz_scaling
    python src/ablate_gem_behavioral.py --model EleutherAI/pythia-1.4b
    python src/ablate_gem_behavioral.py --all

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
from rosetta_tools.gem import discover_base_models, find_extraction_dir
from rosetta_tools.gpu_utils import (
    get_device, get_dtype, log_device_info,
    release_model, purge_hf_cache,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUT_DIR = Path.home() / "rosetta_data" / "results" / "behavioral_gem"

# Same probe sentences as ablate_behavioral_pilot.py for direct comparison
BEHAVIORAL_PROBES = [
    {"concept": "sentiment",
     "prefix": "The film deeply moved audiences and earned universal praise. Critics called it",
     "pos_token": " masterful", "neg_token": " disappointing"},
    {"concept": "sentiment",
     "prefix": "The experience was deeply unpleasant and left everyone feeling",
     "pos_token": " terrible", "neg_token": " wonderful"},
    {"concept": "sentiment",
     "prefix": "The new restaurant received glowing reviews and long wait lists. Diners said the food was",
     "pos_token": " exquisite", "neg_token": " awful"},
    {"concept": "credibility",
     "prefix": "The witness had been caught lying three times under oath. The jury found her testimony",
     "pos_token": " unreliable", "neg_token": " credible"},
    {"concept": "credibility",
     "prefix": "The peer-reviewed study was replicated across twelve independent labs. Scientists considered the findings",
     "pos_token": " reliable", "neg_token": " dubious"},
    {"concept": "credibility",
     "prefix": "The anonymous blog post made extraordinary claims without any citations. Readers treated the content as",
     "pos_token": " suspect", "neg_token": " authoritative"},
    {"concept": "negation",
     "prefix": "She never ate meat and had been a vegetarian for twenty years. At dinner she ordered",
     "pos_token": " salad", "neg_token": " steak"},
    {"concept": "negation",
     "prefix": "He didn't finish the project and missed every deadline. His manager described his work as",
     "pos_token": " incomplete", "neg_token": " finished"},
    {"concept": "negation",
     "prefix": "The medicine did not reduce the patient's fever at all. After treatment the patient felt",
     "pos_token": " worse", "neg_token": " better"},
    {"concept": "causation",
     "prefix": "The bridge collapsed because the support beams had been corroded for decades. Engineers blamed",
     "pos_token": " corrosion", "neg_token": " weather"},
    {"concept": "causation",
     "prefix": "Increased rainfall led to flooding throughout the valley. The disaster was caused by",
     "pos_token": " rain", "neg_token": " drought"},
    {"concept": "causation",
     "prefix": "The fire spread because the wind shifted direction suddenly. Investigators cited the",
     "pos_token": " wind", "neg_token": " arson"},
    {"concept": "certainty",
     "prefix": "The experiment has been replicated hundreds of times with identical results. Scientists are",
     "pos_token": " certain", "neg_token": " uncertain"},
    {"concept": "certainty",
     "prefix": "No one knows what will happen in the election tomorrow. Analysts remain",
     "pos_token": " uncertain", "neg_token": " confident"},
    {"concept": "certainty",
     "prefix": "The mathematical proof has been verified by every major institution. There is",
     "pos_token": " certainty", "neg_token": " doubt"},
    {"concept": "moral_valence",
     "prefix": "He donated his entire savings anonymously to feed homeless children. Everyone agreed it was",
     "pos_token": " admirable", "neg_token": " wrong"},
    {"concept": "moral_valence",
     "prefix": "She stole medicine to save her dying child when no other option existed. People called her action",
     "pos_token": " understandable", "neg_token": " criminal"},
    {"concept": "moral_valence",
     "prefix": "The factory knowingly dumped toxic chemicals into the drinking water supply. The company's behavior was",
     "pos_token": " reprehensible", "neg_token": " acceptable"},
    {"concept": "temporal_order",
     "prefix": "She submitted the application before the deadline and then waited for a response. The submission came",
     "pos_token": " first", "neg_token": " after"},
    {"concept": "temporal_order",
     "prefix": "The earthquake struck and shortly afterward the tsunami warning was issued. The warning came",
     "pos_token": " second", "neg_token": " before"},
    {"concept": "temporal_order",
     "prefix": "He finished his degree and then started his first job. The job came",
     "pos_token": " later", "neg_token": " earlier"},
]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def load_caz_peak(extraction_dir: Path, concept: str) -> int | None:
    p = extraction_dir / f"caz_{concept}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        return d.get("layer_data", d).get("peak_layer")
    except Exception:
        return None


def load_peak_direction(extraction_dir: Path, concept: str, peak_layer: int) -> np.ndarray | None:
    p = extraction_dir / f"caz_{concept}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        metrics = d.get("layer_data", d).get("metrics", [])
        v = metrics[peak_layer].get("dom_vector") if peak_layer < len(metrics) else None
        if v is None:
            return None
        arr = np.array(v, dtype=np.float64)
        return arr / np.linalg.norm(arr)
    except Exception:
        return None


def load_gem_info(extraction_dir: Path, concept: str) -> dict | None:
    p = extraction_dir / f"gem_{concept}.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        targets = d.get("ablation_targets", [0])
        node = d["nodes"][targets[0]]
        abl_p = extraction_dir / f"ablation_gem_{concept}.json"
        width = 3
        if abl_p.exists():
            abl = json.loads(abl_p.read_text())
            width = len(abl.get("handoff", {}).get("ablation_layers", [3]))
        return {
            "handoff_layer": node["handoff_layer"],
            "settled_direction": np.array(node["settled_direction"], dtype=np.float64),
            "width": width,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure_logit_diff(model, tokenizer, prefix: str, pos_token: str, neg_token: str,
                       device: str) -> float:
    enc_pre = tokenizer(prefix, return_tensors="pt").to(device)
    pos_id = tokenizer(pos_token, add_special_tokens=False).input_ids[0]
    neg_id = tokenizer(neg_token, add_special_tokens=False).input_ids[0]
    with torch.no_grad():
        out = model(**enc_pre)
    logits = out.logits[0, -1]
    return float(logits[pos_id] - logits[neg_id])


def measure_logit_diff_ablated(model, tokenizer, layers, ablate_layers, direction_t,
                                prefix, pos_token, neg_token, device) -> float:
    enc_pre = tokenizer(prefix, return_tensors="pt").to(device)
    pos_id = tokenizer(pos_token, add_special_tokens=False).input_ids[0]
    neg_id = tokenizer(neg_token, add_special_tokens=False).input_ids[0]
    dtype = direction_t.dtype
    with ExitStack() as stack:
        for li in ablate_layers:
            stack.enter_context(DirectionalAblator(layers[li], direction_t, dtype=dtype))
        with torch.no_grad():
            out = model(**enc_pre)
    logits = out.logits[0, -1]
    return float(logits[pos_id] - logits[neg_id])


# ---------------------------------------------------------------------------
# Per-concept
# ---------------------------------------------------------------------------

def run_concept(model, tokenizer, concept: str, extraction_dir: Path,
                device: str) -> dict | None:
    peak_layer = load_caz_peak(extraction_dir, concept)
    if peak_layer is None:
        return None
    peak_dir = load_peak_direction(extraction_dir, concept, peak_layer)
    if peak_dir is None:
        return None
    gem_info = load_gem_info(extraction_dir, concept)
    if gem_info is None:
        return None

    layers = get_transformer_layers(model)
    n_layers = len(layers)
    width = gem_info["width"]
    handoff_layer = gem_info["handoff_layer"]
    settled_dir = gem_info["settled_direction"] / np.linalg.norm(gem_info["settled_direction"])

    dtype = next(model.parameters()).dtype
    device_str = str(next(model.parameters()).device)

    peak_dir_t = torch.tensor(peak_dir, dtype=dtype, device=device)
    settled_dir_t = torch.tensor(settled_dir, dtype=dtype, device=device)
    random_dir_t = torch.randn(settled_dir.shape, dtype=dtype, device=device)
    random_dir_t = random_dir_t / random_dir_t.norm()

    # Build windows
    gem_window = list(range(
        max(0, handoff_layer - width // 2),
        min(n_layers, handoff_layer + (width - width // 2))
    ))
    peak_window = [peak_layer]  # single-layer for direct comparison

    probes = [p for p in BEHAVIORAL_PROBES if p["concept"] == concept]
    if not probes:
        return None

    rows = []
    for p in probes:
        baseline = measure_logit_diff(model, tokenizer, p["prefix"],
                                      p["pos_token"], p["neg_token"], device)
        gem_zone = measure_logit_diff_ablated(
            model, tokenizer, layers, gem_window, settled_dir_t,
            p["prefix"], p["pos_token"], p["neg_token"], device)
        peak_single = measure_logit_diff_ablated(
            model, tokenizer, layers, peak_window, peak_dir_t,
            p["prefix"], p["pos_token"], p["neg_token"], device)
        random_zone = measure_logit_diff_ablated(
            model, tokenizer, layers, gem_window, random_dir_t,
            p["prefix"], p["pos_token"], p["neg_token"], device)

        rows.append({
            "prefix": p["prefix"][:60],
            "baseline": round(baseline, 4),
            "suppression_gem_zone": round(baseline - gem_zone, 4),
            "suppression_peak_single": round(baseline - peak_single, 4),
            "suppression_random_zone": round(baseline - random_zone, 4),
            "gem_better_than_peak": bool((baseline - gem_zone) > (baseline - peak_single)),
        })

    supp_gem = [r["suppression_gem_zone"] for r in rows]
    supp_peak = [r["suppression_peak_single"] for r in rows]
    supp_rand = [r["suppression_random_zone"] for r in rows]

    return {
        "concept": concept,
        "n_probes": len(rows),
        "mean_suppression_gem_zone": round(float(np.mean(supp_gem)), 4),
        "mean_suppression_peak_single": round(float(np.mean(supp_peak)), 4),
        "mean_suppression_random_zone": round(float(np.mean(supp_rand)), 4),
        "gem_win_rate": round(sum(r["gem_better_than_peak"] for r in rows) / len(rows), 3),
        "gem_window": gem_window,
        "peak_layer": peak_layer,
        "width": width,
        "probes": rows,
    }


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model(model_id: str, args) -> None:
    extraction_dir = find_extraction_dir(model_id)
    if extraction_dir is None:
        log.warning("No extraction dir for %s", model_id)
        return

    model_slug = extraction_dir.name
    out_dir = OUT_DIR / model_slug
    summary_path = out_dir / "behavioral_gem_summary.json"
    if summary_path.exists() and not args.overwrite:
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

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    concepts = ["causation", "certainty", "credibility", "moral_valence",
                "negation", "sentiment", "temporal_order"]

    for concept in concepts:
        r = run_concept(model, tokenizer, concept, extraction_dir, device)
        if r:
            r["model_id"] = model_id
            results.append(r)
            log.info("  %s: gem=%.3f  peak=%.3f  rand=%.3f  gem_wins=%.0f%%",
                     concept, r["mean_suppression_gem_zone"],
                     r["mean_suppression_peak_single"],
                     r["mean_suppression_random_zone"],
                     100 * r["gem_win_rate"])
            (out_dir / f"behavioral_gem_{concept}.json").write_text(
                json.dumps(r, indent=2))

    if results:
        summary = {
            "model_id": model_id,
            "n_concepts": len(results),
            "mean_gem_zone": round(float(np.mean([r["mean_suppression_gem_zone"] for r in results])), 4),
            "mean_peak_single": round(float(np.mean([r["mean_suppression_peak_single"] for r in results])), 4),
            "mean_random_zone": round(float(np.mean([r["mean_suppression_random_zone"] for r in results])), 4),
            "gem_win_rate": round(float(np.mean([r["gem_win_rate"] for r in results])), 3),
        }
        log.info("  SUMMARY: gem=%.3f  peak=%.3f  rand=%.3f  gem_wins=%.0f%%",
                 summary["mean_gem_zone"], summary["mean_peak_single"],
                 summary["mean_random_zone"], 100 * summary["gem_win_rate"])
        summary_path.write_text(json.dumps(summary, indent=2))

    release_model(model)
    if not args.no_clean_cache:
        purge_hf_cache(model_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-clean-cache", action="store_true")
    args = parser.parse_args()

    models = discover_base_models() if args.all else [args.model]
    log.info("Running on %d models", len(models))
    for model_id in models:
        run_model(model_id, args)


if __name__ == "__main__":
    main()

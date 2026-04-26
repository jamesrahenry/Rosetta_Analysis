"""
cia_trajectory_scoring.py — Multi-layer trajectory vs single-peak probe comparison.

Research question: Does scoring against the full GEM trajectory improve
concept detection robustness over the single dominant-peak approach used
by CIA 1.0?

Three scoring approaches compared per concept per model:

  A. Single-peak   : cosine similarity at dominant CAZ endpoint (current CIA)
  B. Multi-probe   : cosine at each detected CAZ peak; aggregate via max/mean/quorum
  C. Trajectory    : cosine of activation path against mean training trajectory
                     across the full dominant CAZ span (start → end)

Evaluated on:
  - Clean separation: AUROC on held-out CIA concept pairs
  - Adversarial pairs: the low-class pairs (suppressed) vs high-class pairs,
    testing whether each approach maintains discrimination under natural
    suppression conditions in the dataset

CIA concept set: authorization, negation, causation, threat_severity, urgency,
source_credibility, deceptive_intent, exfiltration, obfuscation

Data source: CIA repo at ~/Concept_Integrity_Auditor/datasets/
GEM data:    ~/rosetta_analysis results on the GPU host

Usage
-----
    python gem/cia_trajectory_scoring.py --model Qwen/Qwen2.5-7B-Instruct
    python gem/cia_trajectory_scoring.py --model EleutherAI/pythia-6.9b
    python gem/cia_trajectory_scoring.py --all-cia-models

Written: 2026-04-25 UTC
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta_tools.caz import compute_layer_metrics, find_caz_regions_scored
from rosetta_tools.extraction import extract_contrastive_activations
from rosetta_tools.gpu_utils import (
    NumpyJSONEncoder,
    get_device, get_dtype, load_model_with_retry,
    log_device_info, log_vram, release_model,
)
from rosetta_tools.gem import load_gem, find_extraction_dir
from rosetta_tools.probes import extract_gem_probe

log = logging.getLogger(__name__)

# CIA concept set and dataset name overrides
CIA_CONCEPTS = [
    "authorization",
    "negation",
    "causation",
    "threat_severity",
    "urgency",
    "source_credibility",
    "deceptive_intent",
    "exfiltration",
    "obfuscation",
]
DATASET_NAME_MAP = {
    "source_credibility": "credibility",
    "deceptive_intent":   "deception",
}
# Concepts where high score = alert (probe points toward threat)
ADVERSE_CONCEPTS = {"obfuscation"}

# Models to compare when --all-cia-models is used
CIA_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "EleutherAI/pythia-6.9b",
]

CIA_DATASET_DIR = Path("~/Concept_Integrity_Auditor/datasets").expanduser()
RESULTS_DIR = Path("~/rosetta_data/cia_trajectory").expanduser()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cia_pairs(concept: str) -> tuple[list[str], list[str]]:
    stem = DATASET_NAME_MAP.get(concept, concept)
    path = CIA_DATASET_DIR / f"{stem}_consensus_pairs.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"CIA dataset not found: {path}")
    pos, neg = [], []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            (pos if int(rec["label"]) == 1 else neg).append(rec["text"])
    return pos, neg


def train_eval_split(pos, neg, eval_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    def split(lst):
        idx = rng.permutation(len(lst))
        n_eval = max(1, int(len(lst) * eval_frac))
        return [lst[i] for i in idx[n_eval:]], [lst[i] for i in idx[:n_eval]]
    ptr, pev = split(pos)
    ntr, nev = split(neg)
    return ptr, ntr, pev, nev


# ---------------------------------------------------------------------------
# Probe builders
# ---------------------------------------------------------------------------

def build_probe_A(train_acts):
    """Single-peak: dominant CAZ endpoint (current CIA approach)."""
    probe = extract_gem_probe(train_acts, eval_frac=0.0)
    return probe.layer, probe.direction, "single_peak"


def build_probe_B(train_acts):
    """Multi-probe: direction at each detected CAZ peak."""
    metrics = compute_layer_metrics(train_acts)
    profile = find_caz_regions_scored(metrics)
    probes = []
    if profile.n_regions == 0:
        # Fall back to single peak
        probe = extract_gem_probe(train_acts, eval_frac=0.0)
        probes.append((probe.layer, probe.direction))
    else:
        for region in profile.regions:
            layer = region.end
            pos_acts, neg_acts = train_acts[layer]  # each shape (n_samples, hidden_dim)
            direction = (pos_acts.mean(0) - neg_acts.mean(0))
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
                probes.append((layer, direction))
    return probes, profile.n_regions


def build_probe_C(train_acts, caz_start, caz_end):
    """Trajectory probe: mean direction across full dominant CAZ span."""
    directions = []
    for layer in range(caz_start, caz_end + 1):
        pos_acts, neg_acts = train_acts[layer]
        diff = pos_acts.mean(0) - neg_acts.mean(0)
        norm = np.linalg.norm(diff)
        if norm > 1e-8:
            directions.append(diff / norm)
    if not directions:
        return None, None
    traj = np.stack(directions).mean(0)
    traj = traj / (np.linalg.norm(traj) + 1e-8)
    # Score at the midpoint of the span
    score_layer = (caz_start + caz_end) // 2
    return score_layer, traj


def build_probe_D(train_acts, caz_start, caz_end):
    """Layer-matched: per-layer direction across the full dominant CAZ span.

    Unlike C (mean trajectory), D keeps each layer's own direction and scores
    eval examples against the direction extracted at *that* layer. More faithful
    to GEM's rotation-aware picture than C's single mean vector.
    """
    probes = []
    for layer in range(caz_start, caz_end + 1):
        pos_acts, neg_acts = train_acts[layer]
        diff = pos_acts.mean(0) - neg_acts.mean(0)
        norm = np.linalg.norm(diff)
        if norm > 1e-8:
            probes.append((layer, diff / norm))
    return probes


def build_probe_E(train_acts, profile, n_layers):
    """Handoff-layer probe: direction at region.end+1 (GEM handoff) per region.

    Tests whether scoring at the settled-product layer (handoff, not peak) is
    better than scoring at the peak. Directly operationalises the GEM claim that
    the probe should come from the settled product, not the mid-assembly peak.
    """
    if profile.n_regions > 0:
        regions = profile.regions
    else:
        # No detected regions: use last layer as handoff
        last = n_layers - 1
        pos_acts, neg_acts = train_acts[last]
        diff = pos_acts.mean(0) - neg_acts.mean(0)
        norm = np.linalg.norm(diff)
        return [(last, diff / norm)] if norm > 1e-8 else []
    probes = []
    for region in regions:
        handoff = min(region.end + 1, n_layers - 1)
        pos_acts, neg_acts = train_acts[handoff]
        diff = pos_acts.mean(0) - neg_acts.mean(0)
        norm = np.linalg.norm(diff)
        if norm > 1e-8:
            probes.append((handoff, diff / norm))
    return probes


def build_probe_F(train_acts, caz_start, caz_end):
    """Velocity-weighted probe: direction emphasising highest-rate-of-change layers.

    Computes V(l) = |S(l) - S(l-1)| (separation velocity) across the CAZ span.
    Returns:
      - peak-velocity probe: direction at the layer with highest |V(l)|
      - velocity-weighted mean direction: layers with high dV contribute more
    Peak velocity marks active assembly, not just completed assembly.
    """
    sep_layers, dirs = [], []
    for layer in range(caz_start, caz_end + 1):
        pos_acts, neg_acts = train_acts[layer]
        diff = pos_acts.mean(0) - neg_acts.mean(0)
        norm = np.linalg.norm(diff)
        if norm < 1e-8:
            continue
        d = diff / norm
        s = float(
            np.mean([cosine_score(a, d) for a in pos_acts])
            - np.mean([cosine_score(a, d) for a in neg_acts])
        )
        sep_layers.append((layer, s))
        dirs.append((layer, d))
    if len(sep_layers) < 2:
        return None
    seps = np.array([s for _, s in sep_layers])
    velocities = np.abs(np.diff(seps, prepend=seps[0]))
    peak_idx = int(np.argmax(velocities))
    peak_layer, peak_dir = dirs[peak_idx]
    weights = velocities / (velocities.sum() + 1e-8)
    weighted = sum(w * d for (_, d), w in zip(dirs, weights))
    weighted = weighted / (np.linalg.norm(weighted) + 1e-8)
    score_layer = (caz_start + caz_end) // 2
    return peak_layer, peak_dir, score_layer, weighted


# ---------------------------------------------------------------------------
# Multi-probe scoring (shared by D and E)
# ---------------------------------------------------------------------------

def score_multiprobe_max(eval_acts, probes) -> tuple[np.ndarray, np.ndarray]:
    """Max score across a list of (layer, direction) probes."""
    all_pos, all_neg = [], []
    for layer, direction in probes:
        pos_acts, neg_acts = eval_acts[layer]
        all_pos.append([cosine_score(a, direction) for a in pos_acts])
        all_neg.append([cosine_score(a, direction) for a in neg_acts])
    return (
        np.max(np.stack(all_pos), axis=0),
        np.max(np.stack(all_neg), axis=0),
    )


def score_multiprobe_mean(eval_acts, probes) -> tuple[np.ndarray, np.ndarray]:
    """Mean score across a list of (layer, direction) probes."""
    all_pos, all_neg = [], []
    for layer, direction in probes:
        pos_acts, neg_acts = eval_acts[layer]
        all_pos.append([cosine_score(a, direction) for a in pos_acts])
        all_neg.append([cosine_score(a, direction) for a in neg_acts])
    return (
        np.mean(np.stack(all_pos), axis=0),
        np.mean(np.stack(all_neg), axis=0),
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def cosine_score(act: np.ndarray, direction: np.ndarray) -> float:
    c = float(np.dot(act, direction) / (np.linalg.norm(act) * np.linalg.norm(direction) + 1e-8))
    return (c + 1) / 2


def score_probe_A(eval_acts, layer, direction) -> np.ndarray:
    pos_acts, neg_acts = eval_acts[layer]
    pos_scores = np.array([cosine_score(a, direction) for a in pos_acts])
    neg_scores = np.array([cosine_score(a, direction) for a in neg_acts])
    return pos_scores, neg_scores


def score_probe_B_max(eval_acts, probes) -> tuple[np.ndarray, np.ndarray]:
    """Max score across all detected peaks."""
    all_pos, all_neg = [], []
    for layer, direction in probes:
        pos_acts, neg_acts = eval_acts[layer]
        all_pos.append([cosine_score(a, direction) for a in pos_acts])
        all_neg.append([cosine_score(a, direction) for a in neg_acts])
    pos_scores = np.max(np.stack(all_pos), axis=0)
    neg_scores = np.max(np.stack(all_neg), axis=0)
    return pos_scores, neg_scores


def score_probe_B_mean(eval_acts, probes) -> tuple[np.ndarray, np.ndarray]:
    """Mean score across all detected peaks."""
    all_pos, all_neg = [], []
    for layer, direction in probes:
        pos_acts, neg_acts = eval_acts[layer]
        all_pos.append([cosine_score(a, direction) for a in pos_acts])
        all_neg.append([cosine_score(a, direction) for a in neg_acts])
    pos_scores = np.mean(np.stack(all_pos), axis=0)
    neg_scores = np.mean(np.stack(all_neg), axis=0)
    return pos_scores, neg_scores


def auroc(pos_scores, neg_scores) -> float:
    scores = np.concatenate([pos_scores, neg_scores])
    labels = [1] * len(pos_scores) + [0] * len(neg_scores)
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def separation(pos_scores, neg_scores) -> float:
    return float(pos_scores.mean() - neg_scores.mean())


# ---------------------------------------------------------------------------
# Per-concept experiment
# ---------------------------------------------------------------------------

def run_concept(concept, model, tokenizer, device, batch_size=4):
    log.info("=== %s ===", concept)
    pos, neg = load_cia_pairs(concept)
    pos_tr, neg_tr, pos_ev, neg_ev = train_eval_split(pos, neg)
    log.info("  %d train pairs, %d eval pairs", len(pos_tr), len(neg_tr))

    log.info("  Extracting train activations...")
    train_acts = extract_contrastive_activations(
        model, tokenizer, pos_tr, neg_tr, device=device, batch_size=batch_size
    )
    train_acts = train_acts[1:]  # skip embedding

    log.info("  Extracting eval activations...")
    eval_acts = extract_contrastive_activations(
        model, tokenizer, pos_ev, neg_ev, device=device, batch_size=batch_size
    )
    eval_acts = eval_acts[1:]

    # --- CAZ profile ---
    metrics = compute_layer_metrics(train_acts)
    profile = find_caz_regions_scored(metrics)
    dom = profile.dominant if profile.n_regions > 0 else None
    caz_start = dom.start if dom else 0
    caz_end   = dom.end   if dom else len(train_acts) - 1

    log.info("  CAZ: %d region(s), dominant L%d–L%d", profile.n_regions, caz_start, caz_end)

    results = {
        "concept":    concept,
        "n_regions":  profile.n_regions,
        "is_multimodal": profile.is_multimodal,
        "caz_start":  caz_start,
        "caz_end":    caz_end,
    }

    # --- Approach A: single peak ---
    layer_A, dir_A, _ = build_probe_A(train_acts)
    pos_A, neg_A = score_probe_A(eval_acts, layer_A, dir_A)
    results["A_single_peak"] = {
        "layer":      layer_A,
        "auroc":      auroc(pos_A, neg_A),
        "separation": separation(pos_A, neg_A),
    }
    log.info("  A (single-peak)  layer=%d  AUROC=%.3f  sep=%.3f",
             layer_A, results["A_single_peak"]["auroc"], results["A_single_peak"]["separation"])

    # --- Approach B: multi-probe ---
    probes_B, n_peaks = build_probe_B(train_acts)
    pos_Bmax, neg_Bmax = score_probe_B_max(eval_acts, probes_B)
    pos_Bmean, neg_Bmean = score_probe_B_mean(eval_acts, probes_B)
    results["B_multi_probe"] = {
        "n_peaks":         n_peaks,
        "auroc_max":       auroc(pos_Bmax, neg_Bmax),
        "auroc_mean":      auroc(pos_Bmean, neg_Bmean),
        "separation_max":  separation(pos_Bmax, neg_Bmax),
        "separation_mean": separation(pos_Bmean, neg_Bmean),
    }
    log.info("  B (multi-probe)  peaks=%d  AUROC max=%.3f mean=%.3f",
             n_peaks, results["B_multi_probe"]["auroc_max"], results["B_multi_probe"]["auroc_mean"])

    n_layers = len(train_acts)

    # --- Approach C: trajectory ---
    layer_C, dir_C = build_probe_C(train_acts, caz_start, caz_end)
    if dir_C is not None:
        pos_C, neg_C = score_probe_A(eval_acts, layer_C, dir_C)
        results["C_trajectory"] = {
            "score_layer": layer_C,
            "caz_span":    caz_end - caz_start + 1,
            "auroc":       auroc(pos_C, neg_C),
            "separation":  separation(pos_C, neg_C),
        }
        log.info("  C (trajectory)   span=L%d–L%d  AUROC=%.3f  sep=%.3f",
                 caz_start, caz_end,
                 results["C_trajectory"]["auroc"],
                 results["C_trajectory"]["separation"])
    else:
        results["C_trajectory"] = {"error": "insufficient span"}

    # --- Approach D: layer-matched ---
    probes_D = build_probe_D(train_acts, caz_start, caz_end)
    if probes_D:
        pos_Dmax, neg_Dmax   = score_multiprobe_max(eval_acts, probes_D)
        pos_Dmean, neg_Dmean = score_multiprobe_mean(eval_acts, probes_D)
        results["D_layer_matched"] = {
            "n_layers_in_span": len(probes_D),
            "auroc_max":        auroc(pos_Dmax, neg_Dmax),
            "auroc_mean":       auroc(pos_Dmean, neg_Dmean),
            "separation_max":   separation(pos_Dmax, neg_Dmax),
            "separation_mean":  separation(pos_Dmean, neg_Dmean),
        }
        log.info("  D (layer-matched) span=%d  AUROC max=%.3f mean=%.3f",
                 len(probes_D),
                 results["D_layer_matched"]["auroc_max"],
                 results["D_layer_matched"]["auroc_mean"])
    else:
        results["D_layer_matched"] = {"error": "insufficient span"}

    # --- Approach E: handoff-layer ---
    probes_E = build_probe_E(train_acts, profile, n_layers)
    if probes_E:
        pos_Emax, neg_Emax   = score_multiprobe_max(eval_acts, probes_E)
        pos_Emean, neg_Emean = score_multiprobe_mean(eval_acts, probes_E)
        results["E_handoff"] = {
            "n_handoffs":      len(probes_E),
            "auroc_max":       auroc(pos_Emax, neg_Emax),
            "auroc_mean":      auroc(pos_Emean, neg_Emean),
            "separation_max":  separation(pos_Emax, neg_Emax),
            "separation_mean": separation(pos_Emean, neg_Emean),
        }
        log.info("  E (handoff)       n=%d  AUROC max=%.3f mean=%.3f",
                 len(probes_E),
                 results["E_handoff"]["auroc_max"],
                 results["E_handoff"]["auroc_mean"])
    else:
        results["E_handoff"] = {"error": "no handoff layers"}

    # --- Approach F: velocity-weighted ---
    probe_F = build_probe_F(train_acts, caz_start, caz_end)
    if probe_F is not None:
        layer_Fpeak, dir_Fpeak, layer_Fwt, dir_Fwt = probe_F
        pos_Fpeak, neg_Fpeak = score_probe_A(eval_acts, layer_Fpeak, dir_Fpeak)
        pos_Fwt,   neg_Fwt   = score_probe_A(eval_acts, layer_Fwt,   dir_Fwt)
        results["F_velocity"] = {
            "peak_vel_layer":   layer_Fpeak,
            "auroc_peak_vel":   auroc(pos_Fpeak, neg_Fpeak),
            "auroc_weighted":   auroc(pos_Fwt,   neg_Fwt),
            "sep_peak_vel":     separation(pos_Fpeak, neg_Fpeak),
            "sep_weighted":     separation(pos_Fwt,   neg_Fwt),
        }
        log.info("  F (velocity)      peak_layer=%d  AUROC peak=%.3f weighted=%.3f",
                 layer_Fpeak,
                 results["F_velocity"]["auroc_peak_vel"],
                 results["F_velocity"]["auroc_weighted"])
    else:
        results["F_velocity"] = {"error": "insufficient span"}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--all-cia-models", action="store_true")
    parser.add_argument("--concepts", nargs="+", default=CIA_CONCEPTS)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    models = CIA_MODELS if args.all_cia_models else [args.model]

    for model_name in models:
        log.info("Loading model: %s", model_name)
        device = get_device(prefer="auto")
        dtype  = get_dtype(device)
        log_device_info(device, dtype)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = load_model_with_retry(AutoModelForCausalLM, model_name, dtype=dtype, device="cuda", device_map="auto")
        model.eval()
        log_vram("after model load")

        all_results = []
        for concept in args.concepts:
            try:
                r = run_concept(concept, model, tokenizer, device, args.batch_size)
                all_results.append(r)
            except Exception as e:
                log.error("  FAILED %s: %s", concept, e)
                all_results.append({"concept": concept, "error": str(e)})

        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        model_slug = model_name.replace("/", "_")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"trajectory_scoring_{model_slug}_{ts}.json"
        out_path.write_text(json.dumps({
            "model":     model_name,
            "timestamp": ts,
            "results":   all_results,
        }, indent=2, cls=NumpyJSONEncoder))
        log.info("Results → %s", out_path)

        # Print summary table
        log.info("\n%s", "=" * 100)
        log.info("%-22s %8s %8s %8s %8s %8s %8s %8s %8s",
                 "Concept", "A", "B_max", "B_mean", "C_traj", "D_max", "D_mean", "E_max", "F_peak")
        log.info("-" * 100)
        for r in all_results:
            if "error" in r:
                log.info("%-22s  ERROR: %s", r["concept"], r["error"])
                continue
            a   = r.get("A_single_peak",  {}).get("auroc",       float("nan"))
            bx  = r.get("B_multi_probe",  {}).get("auroc_max",   float("nan"))
            bm  = r.get("B_multi_probe",  {}).get("auroc_mean",  float("nan"))
            c   = r.get("C_trajectory",   {}).get("auroc",       float("nan"))
            dx  = r.get("D_layer_matched",{}).get("auroc_max",   float("nan"))
            dm  = r.get("D_layer_matched",{}).get("auroc_mean",  float("nan"))
            ex  = r.get("E_handoff",      {}).get("auroc_max",   float("nan"))
            fp  = r.get("F_velocity",     {}).get("auroc_peak_vel", float("nan"))
            log.info("%-22s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f",
                     r["concept"], a, bx, bm, c, dx, dm, ex, fp)
        log.info("=" * 100)

        release_model(model)
        del model


if __name__ == "__main__":
    main()

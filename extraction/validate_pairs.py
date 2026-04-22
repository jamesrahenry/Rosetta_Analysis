#!/usr/bin/env python3
"""
validate_pairs.py — Cross-model pair validation pipeline (task t0dc5f3c)

Scores each contrastive pair against N diverse target architectures and reports
the survival rate: fraction of pairs that show consistent positive separation
across all models tested.

A pair "survives" a model if the model's peak Fisher-normalized separation
across layers exceeds a threshold (default 0.3). A pair "survives the gauntlet"
if it survives every model in the validation set.

Survival rate = dataset quality metric. Low survival = pairs are model-specific
or topic-confounded. High survival = pairs are robustly concept-diagnostic.

Usage
-----
    # Full validation gauntlet (default 6-model diverse set)
    python src/validate_pairs.py

    # Single concept
    python src/validate_pairs.py --concept credibility

    # Custom model set
    python src/validate_pairs.py --models EleutherAI/pythia-1.4b Qwen/Qwen2.5-1.5B

    # Adjust survival threshold
    python src/validate_pairs.py --threshold 0.5

    # Limit pairs per concept (faster dev run)
    python src/validate_pairs.py --n-pairs 50

Output
------
    results/pair_validation_<timestamp>/
        summary.json          — per-concept survival rates + overall
        pair_scores.jsonl     — per-pair scores across all models
        failed_pairs.jsonl    — pairs that failed ≥1 model (for inspection)
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.caz import compute_separation
from rosetta_tools.dataset import load_pairs, texts_by_label
from rosetta_tools.gpu_utils import (
    get_device, get_dtype, log_device_info, log_vram,
    release_model, purge_hf_cache,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT   = Path(__file__).resolve().parents[2]
PAIRS_DIR   = REPO_ROOT / "Rosetta_Concept_Pairs" / "pairs" / "raw" / "v1"
RESULTS_ROOT = Path("results")

CONCEPTS = [
    "credibility", "certainty", "sentiment", "moral_valence",
    "causation", "temporal_order", "negation",
]

# 6-model diverse gauntlet: 2 MHA families, 3 GQA families, varying scale
DEFAULT_MODELS = [
    "EleutherAI/pythia-1.4b",       # RoPE + MHA, mid-scale
    "openai-community/gpt2-medium",  # learned pos + MHA, mid-scale
    "facebook/opt-1.3b",             # learned pos + MHA, mid-scale
    "Qwen/Qwen2.5-1.5B",            # GQA + SwiGLU, small
    "meta-llama/Llama-3.2-3B",      # GQA + SwiGLU, small
    "microsoft/phi-2",              # MHA, synthetic training
]

PAIR_FILE_TEMPLATE = "{concept}_consensus_pairs.jsonl"

# Default separation threshold for a pair to "pass" a model
DEFAULT_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Core: score one pair against one model's layer activations
# ---------------------------------------------------------------------------

def score_pair(
    pos_text: str,
    neg_text: str,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str,
    dtype: torch.dtype,
    batch_size: int = 8,
) -> float:
    """Return peak Fisher separation for a single pos/neg pair across all layers."""
    # Extract as single-item batches, then stack
    pos_acts = extract_layer_activations(
        model, tokenizer, [pos_text], device=device, dtype=dtype,
        batch_size=batch_size,
    )  # (1, n_layers, hidden_dim)
    neg_acts = extract_layer_activations(
        model, tokenizer, [neg_text], device=device, dtype=dtype,
        batch_size=batch_size,
    )  # (1, n_layers, hidden_dim)

    n_layers = pos_acts.shape[1]
    peak_sep = 0.0
    for l in range(n_layers):
        # compute_separation expects (n_samples, hidden_dim) — single sample each
        pos_l = pos_acts[:, l, :]  # (1, hidden_dim)
        neg_l = neg_acts[:, l, :]
        # With n=1 per class, Fisher separation degenerates; use raw distance
        # normalized by the model's typical within-class std at this layer.
        # We collect all pairs first and use batch scoring for accuracy.
        sep = float(np.linalg.norm(pos_l - neg_l))
        peak_sep = max(peak_sep, sep)

    return peak_sep


def score_pairs_batch(
    pos_texts: list[str],
    neg_texts: list[str],
    model,
    tokenizer,
    device: str,
    batch_size: int = 8,
) -> tuple[np.ndarray, float]:
    """
    Score all pairs for a concept against one model.
    Returns (peak_sep_per_pair, population_sep):
      - peak_sep_per_pair: shape (n_pairs,), per-pair peak across layers
      - population_sep: float, Fisher sep across full pos/neg populations at peak layer

    extract_layer_activations returns list[ndarray] — one (n_texts, hidden) per layer.
    """
    n = len(pos_texts)
    all_texts = pos_texts + neg_texts

    # list of (2n, hidden) arrays, one per layer
    layer_acts = extract_layer_activations(
        model, tokenizer, all_texts, device=device, batch_size=batch_size,
    )

    n_layers = len(layer_acts)
    peak_seps = np.zeros(n)
    best_pop_sep = 0.0

    for acts in layer_acts:          # acts: (2n, hidden)
        pos_l = acts[:n]             # (n, hidden)
        neg_l = acts[n:]             # (n, hidden)

        pop_sep = compute_separation(pos_l, neg_l)
        if pop_sep > best_pop_sep:
            best_pop_sep = pop_sep

        pooled_std = float(np.sqrt(
            (np.var(pos_l, axis=0).mean() + np.var(neg_l, axis=0).mean()) / 2
        )) + 1e-8
        hidden_dim = pos_l.shape[1]

        for i in range(n):
            pair_dist = float(np.linalg.norm(pos_l[i] - neg_l[i])) / (pooled_std * np.sqrt(hidden_dim))
            if pair_dist > peak_seps[i]:
                peak_seps[i] = pair_dist

    return peak_seps, best_pop_sep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    device = get_device()
    dtype = get_dtype(device)
    log_device_info(device, dtype)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_ROOT / f"pair_validation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = args.models
    concepts = args.concepts or CONCEPTS
    threshold = args.threshold
    n_pairs = args.n_pairs

    log.info("Validation gauntlet: %d models × %d concepts × up to %d pairs",
             len(models_to_run), len(concepts), n_pairs or 9999)
    log.info("Survival threshold: peak_sep > %.2f", threshold)

    # results[concept][pair_id] = {model_id: peak_sep, ...}
    pair_results: dict[str, dict[str, dict[str, float]]] = {c: {} for c in concepts}
    # pop_sep[concept][model_id] = float
    pop_seps: dict[str, dict[str, float]] = {c: {} for c in concepts}

    for model_id in models_to_run:
        log.info("=" * 60)
        log.info("Loading model: %s", model_id)

        load_8bit = args.load_in_8bit or getattr(
            _get_registry_entry(model_id), "load_in_8bit", False
        )

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if load_8bit:
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, quantization_config=bnb_cfg, device_map="auto"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, torch_dtype=dtype, device_map="auto"
                )
            model.eval()
            log_vram("after model load")
        except Exception as e:
            log.error("Failed to load %s: %s", model_id, e)
            continue

        dtype = next(model.parameters()).dtype

        for concept in concepts:
            pair_file = PAIRS_DIR / PAIR_FILE_TEMPLATE.format(concept=concept)
            if not pair_file.exists():
                log.warning("  No pair file for concept %s, skipping", concept)
                continue

            pairs = load_pairs(pair_file)
            if n_pairs:
                pairs = pairs[:n_pairs]

            pos_texts, neg_texts = texts_by_label(pairs)
            pair_ids = [p.pair_id for p in pairs]

            log.info("  %s: %d pairs", concept, len(pairs))

            try:
                peak_seps, pop_sep = score_pairs_batch(
                    pos_texts, neg_texts, model, tokenizer,
                    device=device, batch_size=args.batch_size,
                )
            except Exception as e:
                log.error("  Failed scoring %s / %s: %s", model_id, concept, e)
                continue

            pop_seps[concept][model_id] = float(pop_sep)

            for pid, sep in zip(pair_ids, peak_seps):
                if pid not in pair_results[concept]:
                    pair_results[concept][pid] = {}
                pair_results[concept][pid][model_id] = float(sep)

            n_pass = int((peak_seps > threshold).sum())
            log.info("    pop_sep=%.3f  pair survival: %d/%d (%.1f%%)",
                     pop_sep, n_pass, len(pairs), 100 * n_pass / len(pairs))

        release_model(model)
        if not args.no_clean_cache:
            purge_hf_cache(model_id)

    # ── Compute survival rates ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Computing survival rates...")

    summary = {
        "models": models_to_run,
        "concepts": concepts,
        "threshold": threshold,
        "n_pairs_limit": n_pairs,
        "timestamp": timestamp,
        "by_concept": {},
        "overall": {},
    }

    all_survival_rates = []
    pair_scores_lines = []
    failed_lines = []

    for concept in concepts:
        prs = pair_results[concept]
        if not prs:
            continue

        n_total = len(prs)
        n_survived = 0
        per_model_pass = {m: 0 for m in models_to_run}

        for pid, scores in prs.items():
            passed_all = all(
                scores.get(m, 0.0) > threshold for m in models_to_run
                if m in scores
            )
            if passed_all:
                n_survived += 1
            else:
                failed_lines.append(json.dumps({
                    "concept": concept, "pair_id": pid, "scores": scores
                }))
            for m, s in scores.items():
                if s > threshold:
                    per_model_pass[m] = per_model_pass.get(m, 0) + 1

            pair_scores_lines.append(json.dumps({
                "concept": concept, "pair_id": pid, "scores": scores,
                "survived": passed_all,
            }))

        survival_rate = n_survived / n_total if n_total else 0.0
        all_survival_rates.append(survival_rate)

        summary["by_concept"][concept] = {
            "n_pairs": n_total,
            "n_survived": n_survived,
            "survival_rate": survival_rate,
            "per_model_pass_rate": {
                m: per_model_pass.get(m, 0) / n_total for m in models_to_run
            },
            "pop_sep_by_model": pop_seps[concept],
        }

        log.info("  %-16s  survival=%d/%d (%.1f%%)",
                 concept, n_survived, n_total, 100 * survival_rate)

    overall_survival = float(np.mean(all_survival_rates)) if all_survival_rates else 0.0
    summary["overall"]["mean_survival_rate"] = overall_survival
    log.info("Overall mean survival: %.1f%%", 100 * overall_survival)

    # ── Write outputs ─────────────────────────────────────────────────────────
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "pair_scores.jsonl").write_text("\n".join(pair_scores_lines))
    (out_dir / "failed_pairs.jsonl").write_text("\n".join(failed_lines))

    log.info("Results written to %s", out_dir)
    return out_dir


def _get_registry_entry(model_id: str):
    try:
        from rosetta_tools.models import get_model
        return get_model(model_id)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-model pair validation — dataset quality via survival rate"
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="HuggingFace model IDs to validate against (default: 6-model gauntlet)",
    )
    parser.add_argument(
        "--concepts", nargs="+", default=None,
        help="Concepts to validate (default: all 7)",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="Peak separation threshold for a pair to 'pass' a model (default: 0.3)",
    )
    parser.add_argument(
        "--n-pairs", type=int, default=None,
        help="Limit pairs per concept (default: all)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Tokenizer/extraction batch size (default: 8)",
    )
    parser.add_argument(
        "--load-in-8bit", action="store_true",
        help="Force 8-bit quantization for all models",
    )
    parser.add_argument(
        "--no-clean-cache", action="store_true",
        help="Keep HF cache between models (default: purge after each)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

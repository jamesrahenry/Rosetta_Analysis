"""
permutation_null.py — Null model for CAZ detection specificity.

For each model × concept:
  1. Load contrastive pairs, extract all-text activations (one forward pass)
  2. Run N permutations: randomly swap pos/neg within each pair
  3. Recompute Fisher separation + coherence per layer, run CAZ detection
  4. Compare observed CAZ count to null distribution

The null hypothesis: concept labels carry no signal.  Under random label
assignment, the pipeline should detect fewer (or zero) CAZ peaks.  If the
null produces a comparable peak count, the detection is not specific to
concept structure.

Usage
-----
    # Representative subset (default: 5 models × 7 concepts × 100 perms)
    python src/permutation_null.py

    # Single model
    python src/permutation_null.py --model EleutherAI/pythia-70m

    # All 26 base models
    python src/permutation_null.py --all

    # Custom permutation count
    python src/permutation_null.py --n-perms 200

    # Skip models already computed
    python src/permutation_null.py --all --skip-done
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
    get_device, get_dtype, log_device_info, log_vram, release_model,
    purge_hf_cache, vram_stats, load_model_with_retry, NumpyJSONEncoder,
)
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.caz import (
    compute_separation, compute_coherence, compute_velocity,
    LayerMetrics, find_caz_regions_scored,
)
from rosetta_tools.dataset import load_pairs, texts_by_label
from rosetta_tools.paths import ROSETTA_RESULTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_ROOT = ROSETTA_RESULTS
DATA_ROOT = Path(__file__).parent.parent / "data"

CONCEPT_DATASETS = {
    "credibility": "credibility_pairs.jsonl",
    "negation": "negation_pairs.jsonl",
    "sentiment": "sentiment_pairs.jsonl",
    "causation": "causation_pairs.jsonl",
    "certainty": "certainty_pairs.jsonl",
    "moral_valence": "moral_valence_pairs.jsonl",
    "temporal_order": "temporal_order_pairs.jsonl",
    "sham": "sham_pairs.jsonl",
}

# Representative subset spanning architecture families and scales
DEFAULT_MODELS = [
    "EleutherAI/pythia-70m",      # smallest MHA
    "EleutherAI/pythia-1.4b",     # mid MHA
    "openai-community/gpt2-xl",   # largest GPT-2
    "Qwen/Qwen2.5-3B",           # GQA
    "google/gemma-2-2b",          # alternating attention
]

# Full base model list (matches gpu_jobs_per_model.txt)
ALL_MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "microsoft/phi-2",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "mistralai/Mistral-7B-v0.3",
    "google/gemma-2-2b",
    "google/gemma-2-2b-it",
]


def run_permutation_null(
    all_acts: list[np.ndarray],
    pos_indices: np.ndarray,
    neg_indices: np.ndarray,
    n_perms: int,
    rng: np.random.Generator,
) -> dict:
    """Run permutation null on pre-extracted activations.

    Parameters
    ----------
    all_acts : list of arrays, one per layer, shape [n_texts, hidden_dim]
    pos_indices, neg_indices : arrays of ints indexing into all_acts
    n_perms : number of permutations
    rng : numpy random generator

    Returns
    -------
    dict with observed and null statistics
    """
    n_layers = len(all_acts)
    n_pairs = len(pos_indices)

    # --- Observed (real labels) ---
    real_seps = []
    real_cohs = []
    for layer_idx in range(n_layers):
        pos_acts = all_acts[layer_idx][pos_indices]
        neg_acts = all_acts[layer_idx][neg_indices]
        real_seps.append(compute_separation(pos_acts, neg_acts))
        real_cohs.append(compute_coherence(pos_acts, neg_acts))

    real_vel = compute_velocity(real_seps)
    real_metrics = [
        LayerMetrics(layer=i, separation=real_seps[i],
                     coherence=real_cohs[i], velocity=float(real_vel[i]))
        for i in range(n_layers)
    ]
    real_profile = find_caz_regions_scored(real_metrics)
    real_n_peaks = real_profile.n_regions
    real_peak_sep = real_profile.global_peak_separation
    real_mean_sep = float(np.mean(real_seps))

    # --- Null distribution ---
    null_n_peaks = []
    null_peak_seps = []
    null_mean_seps = []

    for perm_id in range(n_perms):
        # For each pair, swap pos/neg with 50% probability
        swap_mask = rng.integers(0, 2, size=n_pairs).astype(bool)
        perm_pos = np.where(swap_mask, neg_indices, pos_indices)
        perm_neg = np.where(swap_mask, pos_indices, neg_indices)

        perm_seps = []
        perm_cohs = []
        for layer_idx in range(n_layers):
            p = all_acts[layer_idx][perm_pos]
            n = all_acts[layer_idx][perm_neg]
            perm_seps.append(compute_separation(p, n))
            perm_cohs.append(compute_coherence(p, n))

        perm_vel = compute_velocity(perm_seps)
        perm_metrics = [
            LayerMetrics(layer=i, separation=perm_seps[i],
                         coherence=perm_cohs[i], velocity=float(perm_vel[i]))
            for i in range(n_layers)
        ]
        perm_profile = find_caz_regions_scored(perm_metrics)
        null_n_peaks.append(perm_profile.n_regions)
        null_peak_seps.append(perm_profile.global_peak_separation)
        null_mean_seps.append(float(np.mean(perm_seps)))

    null_n_peaks = np.array(null_n_peaks)
    null_peak_seps = np.array(null_peak_seps)

    # p-value: fraction of null perms with >= observed peak count
    p_count = float(np.mean(null_n_peaks >= real_n_peaks))
    # p-value: fraction of null perms with >= observed peak separation
    p_sep = float(np.mean(null_peak_seps >= real_peak_sep))

    return {
        "observed": {
            "n_peaks": real_n_peaks,
            "peak_separation": float(real_peak_sep),
            "mean_separation": real_mean_sep,
        },
        "null": {
            "n_peaks_mean": float(null_n_peaks.mean()),
            "n_peaks_std": float(null_n_peaks.std()),
            "n_peaks_max": int(null_n_peaks.max()),
            "n_peaks_distribution": null_n_peaks.tolist(),
            "peak_sep_mean": float(null_peak_seps.mean()),
            "peak_sep_std": float(null_peak_seps.std()),
            "mean_sep_mean": float(np.mean(null_mean_seps)),
            "mean_sep_std": float(np.std(null_mean_seps)),
        },
        "p_value_n_peaks": p_count,
        "p_value_peak_sep": p_sep,
        "n_permutations": n_perms,
    }


def process_model(
    model_id: str,
    concepts: list[str],
    n_perms: int,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
    output_dir: Path,
    skip_done: bool,
):
    """Run permutation null for all concepts on one model."""
    model_slug = model_id.replace("/", "_")
    model_dir = output_dir / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check skip-done
    if skip_done:
        existing = list(model_dir.glob("null_*.json"))
        done_concepts = {p.stem.replace("null_", "") for p in existing}
        remaining = [c for c in concepts if c not in done_concepts]
        if not remaining:
            log.info("All concepts done for %s, skipping", model_id)
            return
        concepts = remaining

    log.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = load_model_with_retry(
        AutoModelForCausalLM, model_id, dtype=dtype, device=device,
    )
    model.eval()
    log_vram("after model load")

    rng = np.random.default_rng(seed=42)

    for ci, concept in enumerate(concepts):
        log.info("--- [%s] Concept %d/%d: %s ---",
                 model_id.split("/")[-1], ci + 1, len(concepts), concept)

        dataset_path = DATA_ROOT / CONCEPT_DATASETS[concept]
        pairs = load_pairs(dataset_path)
        n_pairs = len(pairs)

        # Build text list: [pos_0, neg_0, pos_1, neg_1, ...]
        all_texts = []
        pos_indices = []
        neg_indices = []
        for pair in pairs:
            pos_indices.append(len(all_texts))
            all_texts.append(pair.pos_text)
            neg_indices.append(len(all_texts))
            all_texts.append(pair.neg_text)
        pos_indices = np.array(pos_indices)
        neg_indices = np.array(neg_indices)

        # Single forward pass — extract all layers
        t0 = time.time()
        log.info("  Extracting activations for %d texts...", len(all_texts))
        all_acts = extract_layer_activations(
            model, tokenizer, all_texts,
            device=device, batch_size=batch_size, pool="last",
        )
        # Drop embedding layer
        all_acts = all_acts[1:]
        t_extract = time.time() - t0
        n_layers = len(all_acts)
        log.info("  %d layers extracted in %.1fs", n_layers, t_extract)

        # Run permutations
        t0 = time.time()
        log.info("  Running %d permutations...", n_perms)
        result = run_permutation_null(
            all_acts, pos_indices, neg_indices, n_perms, rng,
        )
        t_perm = time.time() - t0

        result["model_id"] = model_id
        result["concept"] = concept
        result["n_layers"] = n_layers
        result["n_pairs"] = n_pairs
        result["extraction_time_s"] = round(t_extract, 1)
        result["permutation_time_s"] = round(t_perm, 1)

        # Save
        out_path = model_dir / f"null_{concept}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, cls=NumpyJSONEncoder)

        obs = result["observed"]
        null = result["null"]
        log.info(
            "  %s: observed %d peaks (sep=%.3f) vs null %.1f±%.1f peaks "
            "(sep=%.3f±%.3f) | p(count)=%.3f p(sep)=%.3f | "
            "%.1fs extract + %.1fs perms",
            concept, obs["n_peaks"], obs["peak_separation"],
            null["n_peaks_mean"], null["n_peaks_std"],
            null["peak_sep_mean"], null["peak_sep_std"],
            result["p_value_n_peaks"], result["p_value_peak_sep"],
            t_extract, t_perm,
        )

        # Free layer activations
        del all_acts
        torch.cuda.empty_cache() if device.startswith("cuda") else None

    release_model(model)
    log.info("Done with %s", model_id)


def aggregate_results(output_dir: Path):
    """Aggregate all per-model results into a summary."""
    summary = []
    for model_dir in sorted(output_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for null_file in sorted(model_dir.glob("null_*.json")):
            with open(null_file) as f:
                data = json.load(f)
            summary.append({
                "model": data["model_id"],
                "concept": data["concept"],
                "observed_peaks": data["observed"]["n_peaks"],
                "null_peaks_mean": round(data["null"]["n_peaks_mean"], 2),
                "null_peaks_std": round(data["null"]["n_peaks_std"], 2),
                "observed_sep": round(data["observed"]["peak_separation"], 4),
                "null_sep_mean": round(data["null"]["peak_sep_mean"], 4),
                "p_count": round(data["p_value_n_peaks"], 4),
                "p_sep": round(data["p_value_peak_sep"], 4),
            })

    if not summary:
        log.warning("No results to aggregate")
        return

    out_path = output_dir / "null_model_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Model':<35} {'Concept':<16} {'Obs':>4} {'Null':>8} "
          f"{'Obs Sep':>8} {'Null Sep':>10} {'p(cnt)':>7} {'p(sep)':>7}")
    print("-" * 100)
    for row in summary:
        model_short = row["model"].split("/")[-1]
        null_str = f"{row['null_peaks_mean']:.1f}±{row['null_peaks_std']:.1f}"
        null_sep_str = f"{row['null_sep_mean']:.3f}"
        print(f"{model_short:<35} {row['concept']:<16} "
              f"{row['observed_peaks']:>4} {null_str:>8} "
              f"{row['observed_sep']:>8.3f} {null_sep_str:>10} "
              f"{row['p_count']:>7.3f} {row['p_sep']:>7.3f}")
    print("=" * 100)

    # Grand summary
    obs_peaks = [r["observed_peaks"] for r in summary]
    null_peaks = [r["null_peaks_mean"] for r in summary]
    p_counts = [r["p_count"] for r in summary]
    p_seps = [r["p_sep"] for r in summary]
    n_sig = sum(1 for p in p_counts if p < 0.05)

    print(f"\nGrand summary ({len(summary)} model×concept combinations):")
    print(f"  Observed peaks: mean {np.mean(obs_peaks):.1f}, "
          f"range [{min(obs_peaks)}, {max(obs_peaks)}]")
    print(f"  Null peaks:     mean {np.mean(null_peaks):.1f}")
    print(f"  Significant (p<0.05 on count): {n_sig}/{len(summary)} "
          f"({100*n_sig/len(summary):.0f}%)")
    print(f"  Significant (p<0.05 on sep):   "
          f"{sum(1 for p in p_seps if p < 0.05)}/{len(summary)}")
    print()

    log.info("Summary written to %s", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Permutation null model for CAZ detection specificity",
    )
    parser.add_argument("--model", type=str, help="Single model to test")
    parser.add_argument("--all", action="store_true",
                        help="Run all 26 base models")
    parser.add_argument("--n-perms", type=int, default=100,
                        help="Number of permutations per model×concept")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--concepts", nargs="+",
                        help="Subset of concepts to test")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip model×concept pairs with existing results")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only aggregate existing results, don't run")
    parser.add_argument("--no-clean-cache", action="store_true",
                        help="Don't purge HF cache between models")
    args = parser.parse_args()

    output_dir = RESULTS_ROOT / "permutation_null"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate_results(output_dir)
        return

    device = get_device()
    if args.dtype == "auto":
        dtype = get_dtype(device)
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    concepts = args.concepts or list(CONCEPT_DATASETS.keys())

    if args.all:
        models = ALL_MODELS
    elif args.model:
        models = [args.model]
    else:
        models = DEFAULT_MODELS

    log.info("Permutation null model: %d models × %d concepts × %d perms",
             len(models), len(concepts), args.n_perms)

    for mi, model_id in enumerate(models):
        log.info("=== Model %d/%d: %s ===", mi + 1, len(models), model_id)
        try:
            process_model(
                model_id, concepts, args.n_perms,
                device, dtype, args.batch_size, output_dir, args.skip_done,
            )
        except Exception as e:
            log.error("Failed on %s: %s", model_id, e, exc_info=True)
            continue

        if not args.no_clean_cache and not args.model:
            purge_hf_cache(model_id)

    aggregate_results(output_dir)


if __name__ == "__main__":
    main()

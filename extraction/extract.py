"""
extract.py — Multi-model activation extraction for PRH alignment experiments.

Extracts CAZ metrics and dominant concept vectors for cross-architecture
comparison. The extraction schema is identical to caz_scaling/src/extract.py —
results from both experiments can be loaded with the same rosetta_tools.reporting
functions.

The key difference from the CAZ scaling experiment: here we run architecturally
diverse models, not a controlled scale ladder. The question is not *where* the
CAZ peak falls, but whether the *dominant vectors* at those peaks align after
Procrustes rotation.

Target architectures
--------------------
These represent the major decoder-only families present in the literature:

    GPT-2 family         gpt2, gpt2-xl             (OpenAI, learned positions)
    Pythia family        pythia-160m, pythia-410m   (EleutherAI, RoPE, The Pile)
    GPT-Neo family       gpt-neo-125m               (EleutherAI, ALiBi)
    OPT family           facebook/opt-125m           (Meta, learned positions)
    Llama family         meta-llama/Llama-3.1-8B    (Meta, RoPE, GQA, SwiGLU)
    Mistral family       mistralai/Mistral-7B-v0.3  (Mistral AI, sliding window)

Frontier models (require --frontier flag and sufficient VRAM):

    Llama family         meta-llama/Llama-3.1-70B   (Meta, RoPE, GQA, SwiGLU, 80 layers)
    Qwen family          Qwen/Qwen2.5-72B           (Alibaba, RoPE, GQA, SwiGLU, 80 layers)
    Falcon family        tiiuae/falcon-40b           (TII, RoPE, multi-query attention, 60 layers)

Usage
-----
    # Single model
    python src/extract.py --model gpt2

    # All cross-architecture models (proxy scale)
    python src/extract.py --all

    # All cross-architecture models including frontier (requires H100/H200)
    python src/extract.py --all --include-frontier --load-8bit

    # Frontier models only
    python src/extract.py --frontier --load-8bit

    # Single frontier model
    python src/extract.py --model meta-llama/Llama-3.1-70B --load-8bit

    # Specific concepts and model
    python src/extract.py --model meta-llama/Llama-3.1-8B --concepts credibility negation

    # Force bfloat16 (recommended for L4/A10 class GPUs)
    python src/extract.py --all --dtype bfloat16

    # Parallel mode: run 2 models on 2 GPUs simultaneously
    python src/extract.py --all --parallel

    # Validate only
    python src/extract.py --model gpt2 --validate-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from rosetta_tools.gpu_utils import (
    get_device, get_dtype, log_device_info, log_vram,
    release_model, purge_hf_cache, safe_batch_size,
    load_model_with_retry, disk_free_gib,
)
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.caz import compute_separation, compute_coherence, compute_velocity
from rosetta_tools.dataset import load_pairs, texts_by_label, validate_dataset
from rosetta_tools.tracking import start_run, log_concept, end_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PRH paper model sets — organised by zero-PCA dimension cluster
#
# Each cluster contains models that share hidden_dim, allowing Procrustes
# alignment without any PCA compression (no inflation artifacts).
# Run one model at a time; purge weights after each extraction.
# ---------------------------------------------------------------------------

# Cluster A — 768-dim  (single L4, bf16, ~0.3–0.5 GB each)
PRH_CLUSTER_A = [
    "openai-community/gpt2",         # GPT-2, MHA, learned pos
    "EleutherAI/gpt-neo-125m",        # GPT-Neo, MHA, ALiBi
    "EleutherAI/pythia-160m",         # NeoX, parallel attn, RoPE
    "facebook/opt-125m",              # OPT, MHA, learned pos
]

# Cluster B — 2048-dim  (single L4, bf16, ~2–6 GB each)
PRH_CLUSTER_B = [
    "EleutherAI/pythia-1b",           # NeoX, RoPE, 1B
    "facebook/opt-1.3b",              # OPT, MHA, 1.3B
    "meta-llama/Llama-3.2-1B",        # Llama3, GQA+SwiGLU, 1B  [gated]
    "Qwen/Qwen2.5-3B",                # Qwen2, GQA+SwiGLU, 3B
]

# Cluster C — 4096-dim  (single L4, bf16, ~13–16 GB each)
PRH_CLUSTER_C = [
    "EleutherAI/pythia-6.9b",         # NeoX, RoPE, 6.9B — redundant encoding
    "facebook/opt-6.7b",              # OPT, MHA, 6.7B  — redundant encoding
    "meta-llama/Llama-3.1-8B",        # Llama3, GQA+SwiGLU, 8B — sparse  [gated]
    "mistralai/Mistral-7B-v0.3",      # Mistral, GQA+SWA, 7B — sparse
]

# Cluster D — 3584-dim  (single L4, bf16, ~14–18 GB each)
PRH_CLUSTER_D = [
    "Qwen/Qwen2.5-7B",                # Qwen2, GQA+SwiGLU, 7B
    "google/gemma-2-9b",              # Gemma2, sliding/global+GQA, 9B
]

# Cluster E — 5120-dim  (both L4s via device_map="auto", bf16, ~24–28 GB each)
PRH_CLUSTER_E = [
    "EleutherAI/pythia-12b",          # NeoX, RoPE, 12B — redundant encoding
    "Qwen/Qwen2.5-14B",               # Qwen2, GQA+SwiGLU, 14B — sparse
]

# Scale ladder — cross-dim, within-family only (for convergence-vs-scale plots)
# These are NOT zero-PCA pairs. Reported separately with PCA caveats.
PRH_SCALE_LADDER = [
    "EleutherAI/pythia-70m",          # 512-dim
    "EleutherAI/pythia-410m",         # 1024-dim
    "EleutherAI/pythia-2.8b",         # 2560-dim
    "Qwen/Qwen2.5-0.5B",              # 896-dim
    "Qwen/Qwen2.5-1.5B",              # 1536-dim
]

# All proxy models (L4-runnable): clusters A-E + scale ladder
PRH_PROXY_MODELS = (
    PRH_CLUSTER_A + PRH_CLUSTER_B + PRH_CLUSTER_C
    + PRH_CLUSTER_D + PRH_CLUSTER_E + PRH_SCALE_LADDER
)

# Cluster F — 8192-dim  (H200 only, 8-bit for 70B; Falcon fits bf16)
PRH_FRONTIER_MODELS = [
    "tiiuae/falcon-40b",              # Falcon, MQA, 40B — bf16 on H200
    "Qwen/Qwen2.5-72B",               # Qwen2, GQA+SwiGLU, 72B — 8-bit on H200
    "meta-llama/Llama-3.1-70B",       # Llama3, GQA+SwiGLU, 70B — 8-bit on H200  [gated]
]

# Legacy list — kept for backward compatibility with existing scripts
CROSS_ARCH_MODELS = [
    "openai-community/gpt2",
    "openai-community/gpt2-xl",
    "EleutherAI/gpt-neo-125m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "facebook/opt-125m",
    "meta-llama/Llama-3.1-8B",
    "mistralai/Mistral-7B-v0.3",
]

FRONTIER_MODELS = PRH_FRONTIER_MODELS

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

DEFAULT_CONCEPTS = list(CONCEPT_DATASETS.keys())


# ---------------------------------------------------------------------------
# Extraction (identical to caz_scaling for result format compatibility)
# ---------------------------------------------------------------------------


def extract_layer_wise_metrics(model, tokenizer, pos_texts, neg_texts, device, batch_size):
    pos_by_layer = extract_layer_activations(
        model, tokenizer, pos_texts, device=device, batch_size=batch_size, pool="last"
    )
    neg_by_layer = extract_layer_activations(
        model, tokenizer, neg_texts, device=device, batch_size=batch_size, pool="last"
    )
    pos_by_layer = pos_by_layer[1:]
    neg_by_layer = neg_by_layer[1:]

    n_layers = len(pos_by_layer)
    separations, coherences, dom_vectors, raw_distances = [], [], [], []

    for pos, neg in zip(pos_by_layer, neg_by_layer):
        S = compute_separation(pos, neg)
        C = compute_coherence(pos, neg)
        pos64, neg64 = pos.astype(np.float64), neg.astype(np.float64)
        diff = pos64.mean(axis=0) - neg64.mean(axis=0)
        raw_dist = float(np.linalg.norm(diff))
        norm = np.linalg.norm(diff)
        dom = (diff / norm).tolist() if norm > 0 else diff.tolist()
        separations.append(S)
        coherences.append(C)
        dom_vectors.append(dom)
        raw_distances.append(raw_dist)

    vel_array = compute_velocity(separations, window=3)
    peak_layer = int(np.argmax(separations))
    peak_depth_pct = 100.0 * peak_layer / n_layers if n_layers > 0 else 0.0

    if peak_depth_pct < 5.0:
        log.warning(
            "  ⚠ Peak at L%d (%.1f%% depth) — likely embedding leakage, "
            "not genuine concept assembly. Interpret with caution.",
            peak_layer, peak_depth_pct,
        )

    # Calibration activations at peak layer — pos and neg combined.
    # Saved alongside the metrics JSON for Procrustes alignment in align.py.
    cal_acts = np.concatenate(
        [pos_by_layer[peak_layer].astype(np.float32),
         neg_by_layer[peak_layer].astype(np.float32)],
        axis=0,
    )

    # All-layer calibration activations for depth-matched alignment.
    # Shape: [n_layers, n_pos+n_neg, hidden_dim].  Stored as a separate file
    # to keep backward compat with align.py (which reads calibration_{concept}.npy).
    all_layer_cal = np.stack(
        [np.concatenate([pos_by_layer[i].astype(np.float32),
                         neg_by_layer[i].astype(np.float32)], axis=0)
         for i in range(n_layers)],
        axis=0,
    )

    metrics_dict = {
        "n_layers": n_layers,
        "metrics": [
            {
                "layer": i,
                "separation_fisher": separations[i],
                "coherence": coherences[i],
                "raw_distance": raw_distances[i],
                "dom_vector": dom_vectors[i],
                "velocity": float(vel_array[i]),
            }
            for i in range(n_layers)
        ],
        "peak_layer": peak_layer,
        "peak_separation": separations[peak_layer],
        "peak_depth_pct": round(100.0 * peak_layer / n_layers, 1),
    }
    return metrics_dict, cal_acts, all_layer_cal


def extract_concept(concept, model, tokenizer, device, n_pairs, batch_size, out_dir):
    dataset_path = DATA_ROOT / CONCEPT_DATASETS[concept]
    pairs = load_pairs(dataset_path)
    if n_pairs and len(pairs) > n_pairs:
        pairs = pairs[:n_pairs]

    pos_texts, neg_texts = texts_by_label(pairs)
    t0 = time.time()
    layer_data, cal_acts, all_layer_cal = extract_layer_wise_metrics(
        model, tokenizer, pos_texts, neg_texts, device=device, batch_size=batch_size
    )
    elapsed = time.time() - t0

    model_id = getattr(model, "name_or_path", "unknown")
    results = {
        "model_id": model_id,
        "concept": concept,
        "n_pairs": len(pairs),
        "hidden_dim": model.config.hidden_size,
        "n_layers": model.config.num_hidden_layers,
        "token_pos": -1,
        "extraction_seconds": round(elapsed, 1),
        "layer_data": layer_data,
    }

    out_path = out_dir / f"caz_{concept}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Save peak-layer calibration activations for Procrustes alignment (legacy compat)
    cal_path = out_dir / f"calibration_{concept}.npy"
    np.save(cal_path, cal_acts)

    # Save all-layer calibration activations for depth-matched alignment
    all_cal_path = out_dir / f"calibration_alllayer_{concept}.npy"
    np.save(all_cal_path, all_layer_cal)

    log.info(
        "  [%s] %s → peak L%d (%.1f%%) S=%.4f  (%.1fs)  cal=%s",
        model_id.split("/")[-1], concept,
        layer_data["peak_layer"], layer_data["peak_depth_pct"],
        layer_data["peak_separation"], elapsed,
        cal_acts.shape,
    )
    return {
        "concept": concept,
        "n_pairs": len(pairs),
        "peak_layer": layer_data["peak_layer"],
        "peak_separation": layer_data["peak_separation"],
        "peak_depth_pct": layer_data["peak_depth_pct"],
        "extraction_seconds": round(elapsed, 1),
        "output_file": str(out_path),
    }


def _resolve_dtype(args_dtype: str | None, device: str) -> torch.dtype:
    """Resolve dtype from CLI flag or auto-detect."""
    if args_dtype == "bfloat16":
        return torch.bfloat16
    elif args_dtype == "float32":
        return torch.float32
    else:
        return get_dtype(device)


# Approximate bf16 VRAM (GB). Used to decide single-GPU vs device_map="auto".
# 8-bit is roughly half these values.
MODEL_VRAM_GB = {
    # Cluster A (768-dim)
    "openai-community/gpt2":          0.5,
    "EleutherAI/gpt-neo-125m":        0.3,
    "EleutherAI/pythia-160m":         0.4,
    "facebook/opt-125m":              0.3,
    # Cluster B (2048-dim)
    "EleutherAI/pythia-1b":           2.1,
    "facebook/opt-1.3b":              2.6,
    "meta-llama/Llama-3.2-1B":        2.4,
    "Qwen/Qwen2.5-3B":                6.2,
    # Cluster C (4096-dim)
    "EleutherAI/pythia-6.9b":        14.0,
    "facebook/opt-6.7b":             13.4,
    "meta-llama/Llama-3.1-8B":       16.0,
    "mistralai/Mistral-7B-v0.3":     14.5,
    # Cluster D (3584-dim)
    "Qwen/Qwen2.5-7B":               14.5,
    "google/gemma-2-9b":             18.5,
    # Cluster E (5120-dim) — need device_map="auto" across 2x L4
    "EleutherAI/pythia-12b":         24.0,
    "Qwen/Qwen2.5-14B":              28.0,
    # Scale ladder
    "EleutherAI/pythia-70m":          0.2,
    "EleutherAI/pythia-410m":         1.0,
    "EleutherAI/pythia-2.8b":         5.7,
    "Qwen/Qwen2.5-0.5B":              1.0,
    "Qwen/Qwen2.5-1.5B":              3.1,
    # Cluster F (8192-dim, H200 only)
    "tiiuae/falcon-40b":             80.0,
    "Qwen/Qwen2.5-72B":             144.0,
    "meta-llama/Llama-3.1-70B":     140.0,
    # Legacy keys (kept for backward compat)
    "gpt2":                           0.5,
    "gpt2-xl":                        6.3,
}


def run_model(model_id: str, concepts: list[str], args, device_override: str | None = None) -> None:
    log.info("=== %s ===", model_id)

    for concept in concepts:
        issues = validate_dataset(DATA_ROOT / CONCEPT_DATASETS[concept])
        if issues:
            log.warning("Dataset issues for %s: %s", concept, issues)

    if args.validate_only:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_id.replace("/", "_").replace("-", "_")
    out_dir = Path("results") / f"xarch_{model_slug}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device_override or get_device(args.device)
    dtype = _resolve_dtype(getattr(args, "dtype", None), device)
    load_8bit = getattr(args, "load_8bit", False)

    # Flush any stale GPU state (e.g. from a crashed prior run or loop iteration)
    if device.startswith("cuda"):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    log_device_info(device, dtype)

    mlflow_run = start_run("semantic_convergence", model_id, {
        "device": device,
        "dtype": "int8 (bitsandbytes)" if load_8bit else str(dtype),
        "quantization": "8-bit" if load_8bit else "none",
        "n_pairs": args.n_pairs or 100,
        "batch_size": args.batch_size,
    })

    log.info("  Disk free: %.1f GiB", disk_free_gib())
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Models >20 GB bf16 span both L4s via pipeline parallelism (device_map="auto").
    # Input tensors go to cuda:0 (where the embedding layer always lands).
    model_vram = MODEL_VRAM_GB.get(model_id, 0)
    n_gpus = torch.cuda.device_count() if device.startswith("cuda") else 1
    use_multi_gpu = (not load_8bit) and model_vram > 20.0 and n_gpus > 1

    if use_multi_gpu:
        log.info(
            "Large model (%.0f GB bf16): device_map='auto' across %d GPUs",
            model_vram, n_gpus,
        )

    model = load_model_with_retry(
        AutoModel,
        model_id,
        dtype=dtype,
        device=device,
        device_map="auto" if (load_8bit or use_multi_gpu) else device,
        load_in_8bit=load_8bit,
    )

    # Resolve input device: first parameter lives on embedding layer's GPU.
    if load_8bit or use_multi_gpu:
        device = str(next(model.parameters()).device)
        log.info("  Model input device: %s", device)

    model.eval()
    log_vram("after model load")

    batch = safe_batch_size(args.batch_size, device)
    run_summary = []
    t_start = time.time()

    for i, concept in enumerate(concepts):
        log.info("--- Concept %d/%d: %s ---", i + 1, len(concepts), concept)
        summary = extract_concept(
            concept=concept,
            model=model,
            tokenizer=tokenizer,
            device=device,
            n_pairs=args.n_pairs or 100,
            batch_size=batch,
            out_dir=out_dir,
        )
        run_summary.append(summary)
        log_concept(concept, summary)

    total_elapsed = time.time() - t_start
    release_model(model)

    # Always purge HF cache after each model to prevent disk exhaustion
    # on multi-model runs. Use --no-clean-cache to disable.
    if not getattr(args, "no_clean_cache", False):
        purge_hf_cache(model_id)

    index = {
        "model_id": model_id,
        "concepts": concepts,
        "n_pairs_cap": args.n_pairs,
        "device": device,
        "dtype": "int8 (bitsandbytes)" if load_8bit else str(dtype),
        "quantization": "8-bit" if load_8bit else "none",
        "total_seconds": round(total_elapsed, 1),
        "timestamp": timestamp,
        "results": run_summary,
    }
    with (out_dir / "run_summary.json").open("w") as f:
        json.dump(index, f, indent=2)

    end_run(mlflow_run, out_dir)
    log.info("Done: %s  (%.1fs)  → %s", model_id, total_elapsed, out_dir)


# ---------------------------------------------------------------------------
# Parallel two-GPU runner
# ---------------------------------------------------------------------------

def _run_model_on_gpu(model_id: str, concepts: list[str], args_dict: dict, gpu_id: int) -> str:
    """Worker function for parallel GPU execution (runs in subprocess)."""
    import torch
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    class Args:
        pass
    args = Args()
    for k, v in args_dict.items():
        setattr(args, k, v)
    args.validate_only = False

    run_model(model_id, concepts, args, device_override=device)
    return model_id


def run_parallel(models: list[str], concepts: list[str], args) -> None:
    """Run models across 2 GPUs, two at a time.

    Pairs large models with small ones to balance VRAM across GPUs.
    The two biggest models (Llama 8B, Mistral 7B) each get a full GPU.
    """
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        log.warning("Parallel mode requested but only %d GPU(s) found. Falling back to sequential.", n_gpus)
        for model_id in models:
            run_model(model_id, concepts, args)
        return

    log.info("Parallel mode: %d GPUs detected, running 2 models concurrently", n_gpus)

    # Sort by VRAM: pair largest with smallest for balanced GPU load
    sorted_models = sorted(models, key=lambda m: MODEL_VRAM_GB.get(m, 0), reverse=True)
    pairs = []
    remaining = list(sorted_models)
    while len(remaining) >= 2:
        large = remaining.pop(0)
        small = remaining.pop(-1)
        pairs.append((large, small))
    if remaining:
        pairs.append((remaining[0], None))

    args_dict = vars(args)
    for pair in pairs:
        futures = {}
        with ProcessPoolExecutor(max_workers=2, mp_context=__import__("multiprocessing").get_context("spawn")) as pool:
            for i, model_id in enumerate(pair):
                if model_id is None:
                    continue
                gpu_id = i
                log.info("Launching %s on cuda:%d", model_id, gpu_id)
                fut = pool.submit(_run_model_on_gpu, model_id, concepts, args_dict, gpu_id)
                futures[fut] = model_id

            for fut in as_completed(futures):
                model_id = futures[fut]
                try:
                    fut.result()
                    log.info("Completed: %s", model_id)
                except Exception as e:
                    log.error("FAILED %s: %s", model_id, e)


_PRH_CLUSTER_MAP = {
    "A": PRH_CLUSTER_A,
    "B": PRH_CLUSTER_B,
    "C": PRH_CLUSTER_C,
    "D": PRH_CLUSTER_D,
    "E": PRH_CLUSTER_E,
    "F": PRH_FRONTIER_MODELS,
    "scale": PRH_SCALE_LADDER,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-architecture CAZ extraction")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all", action="store_true", help="Run legacy cross-arch model set")
    group.add_argument("--frontier", action="store_true",
                       help="Run frontier-scale models only (8192-dim, H200)")
    group.add_argument("--prh-proxy", action="store_true",
                       help="PRH paper: all L4-runnable clusters (A–E + scale ladder)")
    group.add_argument("--prh-frontier", action="store_true",
                       help="PRH paper: 8192-dim frontier cluster (H200 only)")
    group.add_argument("--prh-cluster", choices=list(_PRH_CLUSTER_MAP.keys()),
                       help="PRH paper: single named cluster (A/B/C/D/E/F/scale)")
    parser.add_argument("--include-frontier", action="store_true",
                        help="With --all, also include frontier models")
    parser.add_argument("--concepts", nargs="+", default=None)
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="auto",
                        help="Override model dtype (default: auto-detect from hardware)")
    parser.add_argument("--load-8bit", action="store_true",
                        help="Load model in 8-bit quantization via bitsandbytes (halves VRAM)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run 2 small models concurrently on separate GPUs. "
                             "Do NOT use with Cluster E (those span both GPUs).")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-clean-cache", action="store_true",
                        help="Keep model weights in HF cache after extraction. "
                             "Default is to purge after each model (disk-space safety).")
    parser.add_argument("--clean-cache", action="store_true",
                        help="(deprecated, now default) Kept for backward compat")
    return parser.parse_args()


def main():
    args = parse_args()
    concepts = args.concepts or DEFAULT_CONCEPTS
    unknown = [c for c in concepts if c not in CONCEPT_DATASETS]
    if unknown:
        log.error("Unknown concepts: %s", unknown)
        sys.exit(1)

    if args.prh_proxy:
        models = PRH_PROXY_MODELS
    elif args.prh_frontier:
        models = PRH_FRONTIER_MODELS
    elif args.prh_cluster:
        models = _PRH_CLUSTER_MAP[args.prh_cluster]
    elif args.frontier:
        models = FRONTIER_MODELS
    elif args.all and args.include_frontier:
        models = CROSS_ARCH_MODELS + FRONTIER_MODELS
    elif args.all:
        models = CROSS_ARCH_MODELS
    else:
        models = [args.model]

    if args.parallel and (args.all or args.frontier):
        run_parallel(models, concepts, args)
    else:
        for model_id in models:
            run_model(model_id, concepts, args)


if __name__ == "__main__":
    main()

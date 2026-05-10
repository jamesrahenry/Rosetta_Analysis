"""
extract.py — Multi-model activation extraction for Rosetta Papers 1–3.

Extracts CAZ metrics and dominant concept vectors for cross-architecture
comparison. Results land in ~/rosetta_data/models/{model_slug}/ (no timestamps;
one canonical directory per model).

Data: Rosetta_Concept_Pairs/pairs/raw/v1/ via load_concept_pairs().
      N=200 pairs per concept (clamped to available), train split only.

Clusters (zero-PCA dim groups for Procrustes alignment):
    A — 768-dim     GPT-2, GPT-Neo, Pythia-160m, OPT-125m
    B — 2048-dim    Pythia-1b, OPT-1.3b, Llama-3.2-1B, Qwen2.5-3B
    C — 4096-dim    Pythia-6.9b, OPT-6.7b, Llama-3.1-8B (+Instruct), Mistral-7B, Mixtral-8x7B
    D — 3584-dim    Qwen2.5-7B, Gemma-2-9b
    E — 5120-dim    Pythia-12b, Qwen2.5-14B, Qwen2.5-32B
    G — 5376-dim    Gemma4-27B MoE, Gemma4-31B dense  [gated, needs --load-4bit]
    F — 8192-dim    Falcon-40b, Qwen2.5-72B, Llama-3.1-70B  [H200 only]

Usage
-----
    # Single model
    python extraction/extract.py --model gpt2

    # All L4-runnable clusters (A–E+G + scale ladder)
    python extraction/extract.py --prh-proxy

    # Single named cluster
    python extraction/extract.py --prh-cluster C

    # Models requiring 4-bit (auto-detected; flag also available manually)
    python extraction/extract.py --prh-cluster G
    python extraction/extract.py --model mistralai/Mixtral-8x7B-v0.1

    # Frontier models (H200)
    python extraction/extract.py --prh-frontier --load-8bit

    # Parallel mode (2 small models on 2 GPUs simultaneously)
    python extraction/extract.py --prh-cluster A --parallel
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
from rosetta_tools.dataset import (
    load_concept_pairs, texts_by_label, CAZ_PRH_CONCEPTS,
)
from rosetta_tools.tracking import start_run, log_concept, end_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model sets — sourced from rosetta_tools.models registry (models.yaml)
# ---------------------------------------------------------------------------
from rosetta_tools.models import (
    models_by_cluster, all_models as _registry_all_models,
    requires_quantization, vram_gb as _registry_vram,
)

PRH_CLUSTER_A      = models_by_cluster("A")
PRH_CLUSTER_B      = models_by_cluster("B")
PRH_CLUSTER_C      = models_by_cluster("C")
PRH_CLUSTER_D      = models_by_cluster("D")
PRH_CLUSTER_E      = models_by_cluster("E")
PRH_CLUSTER_G      = models_by_cluster("G")
PRH_FRONTIER_MODELS = models_by_cluster("F", include_disabled=True)
PRH_SCALE_LADDER   = models_by_cluster("scale")

PRH_PROXY_MODELS = (
    PRH_CLUSTER_A + PRH_CLUSTER_B + PRH_CLUSTER_C
    + PRH_CLUSTER_D + PRH_CLUSTER_E + PRH_CLUSTER_G + PRH_SCALE_LADDER
)

FRONTIER_MODELS = PRH_FRONTIER_MODELS
CROSS_ARCH_MODELS = PRH_PROXY_MODELS  # legacy alias

# Paper 1 (CAZ Framework) canonical model set — 19 L4-runnable models across
# 8 architectural families.  gpt2-large and gpt2-xl are not in the PRH registry
# clusters but are included here as the GPT-2 scale ladder.
P1_MODELS: list[str] = [
    # Pythia scale ladder (EleutherAI)
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    # GPT-2 family (OpenAI)
    "openai-community/gpt2",
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
    # OPT (Meta)
    "facebook/opt-6.7b",
    # Qwen2.5 (Alibaba)
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    # Llama (Meta)
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    # Mistral
    "mistralai/Mistral-7B-v0.3",
    # Gemma (Google)
    "google/gemma-2-9b",
]

# Paper 3 (CAZ Validation) canonical model set — 26 base models across
# 8 architecture families, 7 concepts (credibility, certainty, causation,
# temporal_order, negation, sentiment, moral_valence).
P3_MODELS: list[str] = [
    # Pythia scale ladder (EleutherAI) — 70M–6.9B, excludes pythia-12b
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    # GPT-2 family (OpenAI)
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
    # OPT (Meta) — 125M–6.7B
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    # Qwen2.5 (Alibaba)
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    # Gemma 2 (Google)
    "google/gemma-2-2b",
    "google/gemma-2-9b",
    # Llama 3.2 (Meta)
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    # Mistral
    "mistralai/Mistral-7B-v0.3",
    # Phi (Microsoft)
    "microsoft/phi-2",
]

# Paper 3 supplementary instruct variants — 9 models
P3_INSTRUCT_MODELS: list[str] = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
]

DEFAULT_CONCEPTS = CAZ_PRH_CONCEPTS  # 17 concepts from canonical dataset

# Canonical results root — all extractions land here, no timestamps
ROSETTA_DATA_ROOT = Path.home() / "rosetta_data"


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

    # Some architectures (e.g. OPT-350m: word_embed_proj_dim=512, hidden_size=1024)
    # have layers with mismatched hidden dims that survive the [1:] skip.
    # Keep only layers at the modal dimension so np.stack doesn't crash.
    if pos_by_layer:
        from collections import Counter
        dims = [p.shape[1] for p in pos_by_layer]
        modal_dim = Counter(dims).most_common(1)[0][0]
        if len(set(dims)) > 1:
            log.warning(
                "Heterogeneous hidden dims across layers %s — keeping %d-dim layers only",
                dict(Counter(dims)), modal_dim,
            )
            keep = [p.shape[1] == modal_dim for p in pos_by_layer]
            pos_by_layer = [p for p, ok in zip(pos_by_layer, keep) if ok]
            neg_by_layer = [n for n, ok in zip(neg_by_layer, keep) if ok]

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
    out_path = out_dir / f"caz_{concept}.json"
    if out_path.exists():
        try:
            existing_n = json.loads(out_path.read_text()).get("n_pairs", 0)
        except (json.JSONDecodeError, OSError):
            existing_n = 0
        requested = n_pairs or 200
        if existing_n >= requested:
            log.info("  [%s] %s already extracted (n=%d) — skipping", out_dir.name, concept, existing_n)
            return {"concept": concept, "skipped": True}
        log.info("  [%s] %s exists at n=%d < requested %d — re-extracting", out_dir.name, concept, existing_n, requested)

    pairs = load_concept_pairs(concept, n=n_pairs or 200)

    pos_texts, neg_texts = texts_by_label(pairs)

    # Qwen2.5 (and any tokenizer without BOS) can produce 0-length input_ids
    # for empty/whitespace-only strings — filter before they reach the model.
    def _nonempty(texts):
        filtered = [t for t in texts if t and t.strip()]
        if len(filtered) < len(texts):
            log.warning("  Dropped %d empty texts for concept %s",
                        len(texts) - len(filtered), concept)
        return filtered
    pos_texts = _nonempty(pos_texts)
    neg_texts = _nonempty(neg_texts)

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




def run_model(model_id: str, concepts: list[str], args, device_override: str | None = None) -> None:
    log.info("=== %s ===", model_id)

    if args.validate_only:
        log.info("  validate_only — skipping extraction")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_id.replace("/", "_").replace("-", "_")
    dir_suffix = getattr(args, "model_dir_suffix", None) or ""
    if dir_suffix:
        snapshots_root = Path.home() / "rosetta_data" / "model_snapshots"
        out_dir = snapshots_root / (model_slug + dir_suffix)
    else:
        out_dir = ROSETTA_DATA_ROOT / "models" / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip model entirely if every concept file already exists at >= requested n_pairs.
    requested_n = args.n_pairs or 200
    def _needs_extraction(concept):
        p = out_dir / f"caz_{concept}.json"
        if not p.exists():
            return True
        try:
            return json.loads(p.read_text()).get("n_pairs", 0) < requested_n
        except (json.JSONDecodeError, OSError):
            return True
    remaining = [c for c in concepts if _needs_extraction(c)]
    if not remaining:
        log.info("  All %d concepts already extracted — skipping model load", len(concepts))
        return
    if len(remaining) < len(concepts):
        log.info("  Resuming: %d/%d concepts remaining", len(remaining), len(concepts))
    concepts = remaining

    device = device_override or get_device(args.device)
    dtype = _resolve_dtype(getattr(args, "dtype", None), device)
    load_8bit = getattr(args, "load_8bit", False)
    load_4bit = getattr(args, "load_4bit", False) or (requires_quantization(model_id) == "4bit")

    # Flush any stale GPU state (e.g. from a crashed prior run or loop iteration)
    if device.startswith("cuda"):
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    log_device_info(device, dtype)

    quant = "4-bit" if load_4bit else ("8-bit" if load_8bit else "none")
    mlflow_run = start_run("rosetta_analysis", model_id, {
        "device": device,
        "dtype": "nf4 (bitsandbytes)" if load_4bit else ("int8 (bitsandbytes)" if load_8bit else str(dtype)),
        "quantization": quant,
        "n_pairs": args.n_pairs or 200,
        "batch_size": args.batch_size,
    })

    log.info("  Disk free: %.1f GiB", disk_free_gib())
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Models >12 GB bf16 span both L4s via pipeline parallelism (device_map="auto").
    # Threshold lowered from 20 → 12: Cluster C models (~14 GB) OOM on a single
    # 22 GB L4 once forward-pass activation tensors are added; spreading across
    # both GPUs gives each ~7 GB model weight + ~15 GB headroom for activations.
    # Input tensors go to cuda:0 (where the embedding layer always lands).
    model_vram = _registry_vram(model_id)
    n_gpus = torch.cuda.device_count() if device.startswith("cuda") else 1
    use_multi_gpu = (not load_8bit and not load_4bit) and model_vram > 12.0 and n_gpus > 1

    if load_4bit:
        log.info("4-bit nf4 quantization (model_id=%s)", model_id)
    elif use_multi_gpu:
        log.info(
            "Large model (%.0f GB bf16): device_map='auto' across %d GPUs",
            model_vram, n_gpus,
        )

    load_kwargs: dict = {}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = load_model_with_retry(
        AutoModel,
        model_id,
        dtype=dtype,
        device=device,
        device_map="auto" if (load_8bit or load_4bit or use_multi_gpu) else device,
        load_in_8bit=load_8bit,
        **load_kwargs,
    )

    # Resolve input device: first parameter lives on embedding layer's GPU.
    if load_8bit or load_4bit or use_multi_gpu:
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
            n_pairs=args.n_pairs or 200,
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
        "n_pairs_cap": args.n_pairs or 200,
        "device": device,
        "dtype": "nf4 (bitsandbytes)" if load_4bit else ("int8 (bitsandbytes)" if load_8bit else str(dtype)),
        "quantization": quant,
        "total_seconds": round(total_elapsed, 1),
        "timestamp": timestamp,
        "results": run_summary,
    }
    with (out_dir / "run_summary.json").open("w") as f:
        json.dump(index, f, indent=2)

    # Write provenance metadata alongside results
    hf_revision_sha = getattr(model.config, "_commit_hash", None) or "unknown"
    metadata = {
        "model_id": model_id,
        "hf_revision_sha": hf_revision_sha,
        "extracted_at": timestamp,
        "quantization": quant,
        "n_pairs_cap": args.n_pairs or 200,
        "dataset": "Rosetta_Concept_Pairs/pairs/raw/v1",
        "split": "train",
        "concepts": concepts,
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

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
    sorted_models = sorted(models, key=lambda m: _registry_vram(m), reverse=True)
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
    "G": PRH_CLUSTER_G,
    "scale": PRH_SCALE_LADDER,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-architecture CAZ extraction")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--p1-corpus", action="store_true",
                       help="Paper 1 CAZ corpus: 19 L4-runnable models across 8 architectural families")
    group.add_argument("--p3-corpus", action="store_true",
                       help="Paper 3 CAZ Validation: 26 base models across 8 architecture families")
    group.add_argument("--p3-corpus-instruct", action="store_true",
                       help="Paper 3 supplementary: 9 instruct-tuned variants")
    group.add_argument("--all", action="store_true", help="Run legacy cross-arch model set")
    group.add_argument("--frontier", action="store_true",
                       help="Run frontier-scale models only (8192-dim, H200)")
    group.add_argument("--prh-proxy", action="store_true",
                       help="PRH paper: all L4-runnable clusters (A–E + scale ladder)")
    group.add_argument("--prh-frontier", action="store_true",
                       help="PRH paper: 8192-dim frontier cluster (H200 only)")
    group.add_argument("--prh-cluster", choices=list(_PRH_CLUSTER_MAP.keys()),
                       help="PRH paper: single named cluster (A/B/C/D/E/F/G/scale)")
    parser.add_argument("--include-frontier", action="store_true",
                        help="With --all, also include frontier models")
    parser.add_argument("--concepts", nargs="+", default=None)
    parser.add_argument("--n-pairs", type=int, default=200)
    parser.add_argument("--model-dir-suffix", type=str, default="",
                        help="Append suffix to model directory name (e.g. _p1n100 for isolated P1 storage)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="auto",
                        help="Override model dtype (default: auto-detect from hardware)")
    parser.add_argument("--load-8bit", action="store_true",
                        help="Load model in 8-bit quantization via bitsandbytes (halves VRAM)")
    parser.add_argument("--load-4bit", action="store_true",
                        help="Load model in 4-bit nf4 quantization (bitsandbytes). "
                             "Auto-enabled for models with quantization=4bit in models.yaml.")
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
    from rosetta_tools.dataset import ALL_CONCEPTS
    concepts = args.concepts or DEFAULT_CONCEPTS
    unknown = [c for c in concepts if c not in ALL_CONCEPTS]
    if unknown:
        log.error("Unknown concepts: %s", unknown)
        sys.exit(1)

    if args.p1_corpus:
        models = P1_MODELS
    elif args.p3_corpus:
        models = P3_MODELS
    elif args.p3_corpus_instruct:
        models = P3_INSTRUCT_MODELS
    elif args.prh_proxy:
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

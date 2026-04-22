#!/usr/bin/env python3
"""
test_p4_post_chain.py — Test revised Prediction 4: Post-Chain Degradation.

Measures separation decay after the FINAL CAZ in each concept's chain,
and correlates with:
  (a) remaining depth fraction (final peak to last layer)
  (b) concept-token clustering in unembedding space

The original P4 assumed single-peak assembly. Multimodal assembly (mean 3.4
CAZes/concept/model) means inter-peak dips are reallocation, not degradation.
Only the post-chain region represents genuine decay.

Requires: extraction results (caz_*.json) in results/ directory.
GPU required only for part (b) — unembedding extraction.
Part (a) runs entirely from saved data.

Usage:
    # Part (a) only — no GPU needed:
    python test_p4_post_chain.py --part a

    # Both parts — needs GPU for unembedding extraction:
    python test_p4_post_chain.py --part both

    # Specific model families:
    python test_p4_post_chain.py --families pythia gpt2 qwen2
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from scipy import stats

# Add rosetta_tools to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "rosetta_tools"))
from rosetta_tools.caz import find_caz_regions_scored, LayerMetrics


RESULTS_DIR = Path(__file__).parent.parent / "results"
CONCEPTS = [
    "credibility", "negation", "causation", "temporal_order",
    "sentiment", "certainty", "moral_valence",
]

# Concept-relevant tokens for unembedding analysis.
# These are high-frequency tokens associated with each concept,
# used to measure clustering in unembedding space.
CONCEPT_TOKENS = {
    "credibility": [
        "reliable", "trustworthy", "credible", "dubious", "unreliable",
        "questionable", "legitimate", "authentic", "suspect", "verified",
    ],
    "negation": [
        "not", "never", "no", "neither", "none", "nothing", "nowhere",
        "without", "hardly", "barely",
    ],
    "causation": [
        "because", "caused", "therefore", "hence", "since", "resulted",
        "consequence", "due", "led", "reason",
    ],
    "temporal_order": [
        "before", "after", "then", "first", "next", "previously",
        "subsequently", "later", "earlier", "following",
    ],
    "sentiment": [
        "good", "bad", "great", "terrible", "wonderful", "awful",
        "excellent", "horrible", "amazing", "disgusting",
    ],
    "certainty": [
        "definitely", "certainly", "probably", "maybe", "perhaps",
        "surely", "possibly", "undoubtedly", "likely", "unlikely",
    ],
    "moral_valence": [
        "right", "wrong", "ethical", "immoral", "virtuous", "evil",
        "just", "unjust", "moral", "corrupt",
    ],
}


@dataclass
class PostChainDecay:
    """Decay metrics for one concept in one model."""
    model: str
    concept: str
    n_layers: int
    final_peak_layer: int
    final_peak_depth_pct: float
    final_peak_separation: float
    final_layer_separation: float
    remaining_depth_frac: float  # (n_layers - final_peak) / n_layers
    decay_ratio: float  # final_layer_sep / final_peak_sep
    decay_slope: float  # linear slope from peak to end (normalized)
    n_post_chain_layers: int


def load_extraction_results():
    """Find all extraction result directories."""
    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        sys.exit(1)

    result_dirs = sorted([
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir() and any((d / f"caz_{c}.json").exists() for c in CONCEPTS)
    ])

    # Deduplicate: keep latest per model (tag_model_date format)
    model_dirs = {}
    for d in result_dirs:
        parts = d.name.split("_")
        # Extract model identifier (between tag and date)
        # Format: tag_org_model_date
        # We want to group by org_model
        date_str = "_".join(parts[-2:])  # last two parts are date_time
        tag = parts[0]
        model_key = "_".join(parts[1:-2])

        # Skip instruct models for this analysis
        if "instruct" in d.name.lower() or d.name.startswith("custom_"):
            continue

        full_key = f"{tag}_{model_key}"
        if full_key not in model_dirs or d.name > model_dirs[full_key].name:
            model_dirs[full_key] = d

    return list(model_dirs.values())


def get_post_chain_decay(result_dir: Path, concept: str) -> PostChainDecay | None:
    """Compute post-chain decay for one concept in one model."""
    caz_file = result_dir / f"caz_{concept}.json"
    if not caz_file.exists():
        return None

    data = json.loads(caz_file.read_text())
    layer_data = data["layer_data"]
    metrics_list = layer_data["metrics"]
    n_layers = layer_data["n_layers"]

    # Build LayerMetrics for scored detection
    layer_metrics = []
    separations = []
    for m in metrics_list:
        lm = LayerMetrics(
            layer=m["layer"],
            separation=m["separation_fisher"],
            coherence=m["coherence"],
            velocity=0.0,  # placeholder — scored detector recomputes internally
        )
        layer_metrics.append(lm)
        separations.append(m["separation_fisher"])

    # Run scored detection to find all CAZes
    profile = find_caz_regions_scored(layer_metrics)

    if not profile.regions:
        return None

    # Find the final CAZ (deepest peak)
    final_region = max(profile.regions, key=lambda r: r.peak)
    final_peak = final_region.peak
    final_peak_sep = separations[final_peak]

    # Compute post-chain metrics
    final_layer_sep = separations[-1]
    remaining_layers = n_layers - final_peak - 1
    remaining_depth_frac = remaining_layers / n_layers

    if remaining_layers < 1:
        # Final CAZ is at the last layer — no post-chain region
        decay_ratio = 1.0
        decay_slope = 0.0
    else:
        decay_ratio = final_layer_sep / final_peak_sep if final_peak_sep > 0 else 1.0
        # Normalized slope: separation change per % depth
        post_chain_seps = separations[final_peak:]
        if len(post_chain_seps) > 1:
            x = np.arange(len(post_chain_seps)) / n_layers * 100  # as % depth
            slope, _, _, _, _ = stats.linregress(x, post_chain_seps)
            decay_slope = slope
        else:
            decay_slope = 0.0

    model_name = data.get("model_id", result_dir.name)

    return PostChainDecay(
        model=model_name,
        concept=concept,
        n_layers=n_layers,
        final_peak_layer=final_peak,
        final_peak_depth_pct=final_peak / (n_layers - 1) * 100,
        final_peak_separation=final_peak_sep,
        final_layer_separation=final_layer_sep,
        remaining_depth_frac=remaining_depth_frac,
        decay_ratio=decay_ratio,
        decay_slope=decay_slope,
        n_post_chain_layers=max(0, n_layers - final_peak - 1),
    )


def run_part_a(result_dirs):
    """Part (a): Correlate post-chain decay with remaining depth fraction."""
    print("=" * 70)
    print("P4 REVISED — Post-Chain Degradation Analysis")
    print("=" * 70)
    print(f"\nAnalyzing {len(result_dirs)} extraction directories...")
    print()

    all_decays = []
    for rd in result_dirs:
        for concept in CONCEPTS:
            decay = get_post_chain_decay(rd, concept)
            if decay is not None:
                all_decays.append(decay)

    if not all_decays:
        print("No data found!")
        return

    print(f"Total concept × model combinations: {len(all_decays)}")
    print()

    # --- Summary statistics ---
    remaining_fracs = [d.remaining_depth_frac for d in all_decays]
    decay_ratios = [d.decay_ratio for d in all_decays]
    decay_slopes = [d.decay_slope for d in all_decays]

    print("--- Post-Chain Summary ---")
    print(f"Mean remaining depth after final CAZ: {np.mean(remaining_fracs)*100:.1f}%")
    print(f"Mean decay ratio (final_sep / peak_sep): {np.mean(decay_ratios):.3f}")
    print(f"Mean decay slope (sep change per %depth): {np.mean(decay_slopes):.4f}")
    print()

    # --- Correlation: remaining depth vs decay ---
    r_ratio, p_ratio = stats.pearsonr(remaining_fracs, decay_ratios)
    r_slope, p_slope = stats.pearsonr(remaining_fracs, decay_slopes)
    print("--- Correlation: Remaining Depth vs Decay ---")
    print(f"  remaining_depth vs decay_ratio:  r={r_ratio:.3f}, p={p_ratio:.4f}")
    print(f"  remaining_depth vs decay_slope:  r={r_slope:.3f}, p={p_slope:.4f}")
    print()

    # --- Per-concept breakdown ---
    print("--- Per-Concept Post-Chain Decay ---")
    print(f"{'Concept':<18} {'Mean Final Peak%':>16} {'Mean Decay Ratio':>16} "
          f"{'Mean Remaining%':>16} {'N':>4}")
    print("-" * 74)

    concept_data = {}
    for concept in CONCEPTS:
        cd = [d for d in all_decays if d.concept == concept]
        if cd:
            concept_data[concept] = cd
            mean_peak = np.mean([d.final_peak_depth_pct for d in cd])
            mean_decay = np.mean([d.decay_ratio for d in cd])
            mean_remaining = np.mean([d.remaining_depth_frac for d in cd]) * 100
            print(f"{concept:<18} {mean_peak:>15.1f}% {mean_decay:>16.3f} "
                  f"{mean_remaining:>15.1f}% {len(cd):>4}")

    print()

    # --- Concept type analysis ---
    type_map = {
        "epistemic": ["credibility", "certainty"],
        "syntactic": ["negation"],
        "relational": ["causation", "temporal_order"],
        "affective": ["sentiment", "moral_valence"],
    }
    print("--- By Concept Type ---")
    print(f"{'Type':<14} {'Mean Decay Ratio':>16} {'Mean Remaining%':>16}")
    print("-" * 50)
    for ctype, concepts in type_map.items():
        cd = [d for d in all_decays if d.concept in concepts]
        if cd:
            print(f"{ctype:<14} {np.mean([d.decay_ratio for d in cd]):>16.3f} "
                  f"{np.mean([d.remaining_depth_frac for d in cd])*100:>15.1f}%")

    print()

    # --- Does concept type predict decay after controlling for remaining depth? ---
    print("--- Partial Correlation: Concept Type vs Decay (controlling for depth) ---")
    # Encode concept type as abstraction level
    abstraction = {"negation": 1, "causation": 2, "temporal_order": 2,
                   "sentiment": 3, "moral_valence": 3, "certainty": 4, "credibility": 4}
    abs_levels = [abstraction[d.concept] for d in all_decays]

    # Partial correlation: abstraction vs decay_ratio, controlling for remaining_depth
    from numpy.linalg import lstsq
    X = np.column_stack([remaining_fracs, abs_levels])
    # Residualize both on remaining_depth
    r_abs = np.array(abs_levels) - np.mean(abs_levels)
    r_decay = np.array(decay_ratios) - np.mean(decay_ratios)
    # Simple partial: regress both on remaining_depth, correlate residuals
    rf_arr = np.array(remaining_fracs)
    abs_resid = np.array(abs_levels) - (
        np.polyval(np.polyfit(rf_arr, abs_levels, 1), rf_arr))
    decay_resid = np.array(decay_ratios) - (
        np.polyval(np.polyfit(rf_arr, decay_ratios, 1), rf_arr))
    r_partial, p_partial = stats.pearsonr(abs_resid, decay_resid)
    print(f"  Partial r (abstraction vs decay | remaining_depth): {r_partial:.3f}, p={p_partial:.4f}")
    print()

    # --- Save results ---
    output = {
        "n_models": len(result_dirs),
        "n_measurements": len(all_decays),
        "summary": {
            "mean_remaining_depth_pct": float(np.mean(remaining_fracs) * 100),
            "mean_decay_ratio": float(np.mean(decay_ratios)),
            "mean_decay_slope": float(np.mean(decay_slopes)),
        },
        "correlations": {
            "remaining_depth_vs_decay_ratio": {"r": float(r_ratio), "p": float(p_ratio)},
            "remaining_depth_vs_decay_slope": {"r": float(r_slope), "p": float(p_slope)},
            "abstraction_vs_decay_partial": {"r": float(r_partial), "p": float(p_partial)},
        },
        "per_concept": {},
        "raw_data": [],
    }

    for concept in CONCEPTS:
        cd = [d for d in all_decays if d.concept == concept]
        if cd:
            output["per_concept"][concept] = {
                "n": len(cd),
                "mean_final_peak_depth_pct": float(np.mean([d.final_peak_depth_pct for d in cd])),
                "mean_decay_ratio": float(np.mean([d.decay_ratio for d in cd])),
                "mean_remaining_depth_pct": float(np.mean([d.remaining_depth_frac for d in cd]) * 100),
                "mean_decay_slope": float(np.mean([d.decay_slope for d in cd])),
            }

    for d in all_decays:
        output["raw_data"].append({
            "model": d.model,
            "concept": d.concept,
            "final_peak_layer": d.final_peak_layer,
            "final_peak_depth_pct": d.final_peak_depth_pct,
            "final_peak_separation": d.final_peak_separation,
            "final_layer_separation": d.final_layer_separation,
            "remaining_depth_frac": d.remaining_depth_frac,
            "decay_ratio": d.decay_ratio,
            "decay_slope": d.decay_slope,
            "n_post_chain_layers": d.n_post_chain_layers,
        })

    out_path = Path(__file__).parent.parent / "P4_POST_CHAIN_RESULTS.json"

    def to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    out_path.write_text(json.dumps(output, indent=2, default=to_serializable))
    print(f"Results saved to {out_path}")


def run_part_b(result_dirs):
    """Part (b): Correlate decay with concept-token clustering in unembedding space."""
    print()
    print("=" * 70)
    print("P4 Part (b) — Unembedding Token Clustering")
    print("=" * 70)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("torch/transformers not available. Skipping part (b).")
        return

    # Load part (a) results
    results_path = Path(__file__).parent.parent / "P4_POST_CHAIN_RESULTS.json"
    if not results_path.exists():
        print("Run part (a) first!")
        return

    results = json.loads(results_path.read_text())

    # Get unique models from results
    model_ids = list(set(d["model"] for d in results["raw_data"]))
    print(f"Models to analyze: {len(model_ids)}")

    unembed_results = []

    for model_id in sorted(model_ids):
        print(f"\nLoading {model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        # Get unembedding matrix (lm_head weight)
        # Handle meta tensors from CPU offload — materialize first
        unembed_tensor = None
        for attr in ("lm_head", "embed_out"):
            head = getattr(model, attr, None)
            if head is not None and hasattr(head, "weight"):
                w = head.weight
                if w.device.type == "meta":
                    print(f"  lm_head on meta device — skipping {model_id}")
                    break
                unembed_tensor = w.data.float().cpu()
                break

        if unembed_tensor is None:
            print(f"  Cannot find unembedding matrix — skipping")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        unembed = unembed_tensor.numpy()

        # vocab_size × hidden_dim
        print(f"  Unembedding shape: {unembed.shape}")

        for concept in CONCEPTS:
            tokens = CONCEPT_TOKENS[concept]
            token_ids = []
            for t in tokens:
                ids = tokenizer.encode(t, add_special_tokens=False)
                if ids:
                    token_ids.append(ids[0])

            if len(token_ids) < 3:
                continue

            # Get unembedding vectors for concept tokens
            vecs = unembed[token_ids]

            # Compute mean pairwise cosine similarity (clustering measure)
            from sklearn.metrics.pairwise import cosine_similarity
            cos_sim = cosine_similarity(vecs)
            # Upper triangle only (exclude diagonal)
            n = len(token_ids)
            triu_idx = np.triu_indices(n, k=1)
            mean_cos = cos_sim[triu_idx].mean()

            unembed_results.append({
                "model": model_id,
                "concept": concept,
                "n_tokens": len(token_ids),
                "unembed_clustering": float(mean_cos),
            })

        del model, unembed_tensor, unembed, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clean HF cache for this model to avoid filling disk
        import gc
        gc.collect()
        try:
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if model_id.replace("/", "--") in str(repo.repo_path):
                    for rev in repo.revisions:
                        cache_info = scan_cache_dir()
                        strategy = cache_info.delete_revisions(rev.commit_hash)
                        strategy.execute()
                        print(f"  Cleared HF cache for {model_id}")
                        break
                    break
        except Exception as e:
            print(f"  Cache cleanup failed (non-fatal): {e}")

    if not unembed_results:
        print("No unembedding results computed.")
        return

    # Merge with decay data
    decay_lookup = {}
    for d in results["raw_data"]:
        decay_lookup[(d["model"], d["concept"])] = d

    merged = []
    for ur in unembed_results:
        key = (ur["model"], ur["concept"])
        if key in decay_lookup:
            merged.append({**decay_lookup[key], **ur})

    if not merged:
        print("No matched pairs between decay and unembedding data.")
        return

    print(f"\n--- Unembedding Clustering vs Post-Chain Decay ({len(merged)} pairs) ---")
    clusterings = [m["unembed_clustering"] for m in merged]
    decay_ratios = [m["decay_ratio"] for m in merged]
    r, p = stats.pearsonr(clusterings, decay_ratios)
    print(f"  unembed_clustering vs decay_ratio: r={r:.3f}, p={p:.4f}")

    # Per-concept
    print(f"\n{'Concept':<18} {'Mean Clustering':>16} {'Mean Decay':>12} {'N':>4}")
    print("-" * 54)
    for concept in CONCEPTS:
        cm = [m for m in merged if m["concept"] == concept]
        if cm:
            print(f"{concept:<18} {np.mean([m['unembed_clustering'] for m in cm]):>16.3f} "
                  f"{np.mean([m['decay_ratio'] for m in cm]):>12.3f} {len(cm):>4}")

    # Save
    results["part_b"] = {
        "n_pairs": len(merged),
        "unembed_vs_decay": {"r": float(r), "p": float(p)},
        "data": merged,
    }
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nUpdated results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Test revised P4: Post-Chain Degradation")
    parser.add_argument("--part", choices=["a", "b", "both"], default="both",
                        help="Which part to run (a=depth correlation, b=unembedding, both)")
    parser.add_argument("--families", nargs="*",
                        help="Only analyze these families (e.g., pythia gpt2)")
    args = parser.parse_args()

    result_dirs = load_extraction_results()
    if args.families:
        result_dirs = [d for d in result_dirs
                       if any(f in d.name.lower() for f in args.families)]

    print(f"Found {len(result_dirs)} extraction result directories")

    if args.part in ("a", "both"):
        run_part_a(result_dirs)

    if args.part in ("b", "both"):
        run_part_b(result_dirs)


if __name__ == "__main__":
    main()

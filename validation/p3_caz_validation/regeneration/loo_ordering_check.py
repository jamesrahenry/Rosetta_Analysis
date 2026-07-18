#!/usr/bin/env python3
"""Extends the leave-one-generator-out pilot: for each model, compute the
17-concept depth *ordering* under each leave-one-out subset and compare to
the baseline ordering via Kendall's tau. Memory-efficient: processes one
concept's array at a time and discards it, rather than holding all 17
concepts' arrays in memory simultaneously (needed for larger models like
gemma-2-9b, which OOM'd the original all-at-once version)."""
import gc
import json
import numpy as np
from pathlib import Path
from scipy.stats import kendalltau

DATA_ROOT = Path.home() / "rosetta_data" / "paper_n250"
MODELS = [
    "EleutherAI_pythia_70m", "EleutherAI_pythia_160m", "EleutherAI_pythia_410m",
    "EleutherAI_pythia_1b", "EleutherAI_pythia_1.4b", "EleutherAI_pythia_2.8b",
    "EleutherAI_pythia_6.9b",
    "openai_community_gpt2", "openai_community_gpt2_medium",
    "openai_community_gpt2_large", "openai_community_gpt2_xl",
    "facebook_opt_125m", "facebook_opt_350m", "facebook_opt_1.3b",
    "facebook_opt_2.7b", "facebook_opt_6.7b",
    "Qwen_Qwen2.5_0.5B", "Qwen_Qwen2.5_1.5B", "Qwen_Qwen2.5_3B", "Qwen_Qwen2.5_7B",
    "google_gemma_2_2b", "google_gemma_2_9b",
    "meta_llama_Llama_3.2_1B", "meta_llama_Llama_3.2_3B",
    "mistralai_Mistral_7B_v0.3", "microsoft_phi_2",
]
CONCEPTS = [
    "credibility", "negation", "causation", "temporal_order", "sentiment",
    "certainty", "moral_valence", "specificity", "plurality", "agency",
    "formality", "threat_severity", "authorization", "urgency", "sarcasm",
    "deception", "exfiltration",
]


def fisher_separation(pos, neg):
    mu_pos, mu_neg = pos.mean(0), neg.mean(0)
    dist = np.linalg.norm(mu_pos - mu_neg)
    var_pos = pos.var(0, ddof=1).sum() if len(pos) > 1 else 0.0
    var_neg = neg.var(0, ddof=1).sum() if len(neg) > 1 else 0.0
    denom = np.sqrt(0.5 * (var_pos + var_neg))
    return dist / denom if denom > 0 else 0.0


def peak_depth(arr, n_pairs, n_layers, keep_mask):
    idx = np.where(keep_mask)[0]
    if len(idx) < 5:
        return None
    scores = np.empty(n_layers)
    for l in range(n_layers):
        pos = arr[l, idx, :]
        neg = arr[l, n_pairs + idx, :]
        scores[l] = fisher_separation(pos, neg)
    peak_l = int(np.argmax(scores))
    return peak_l / max(n_layers - 1, 1)


def process_concept(model_dir, concept):
    """Returns (baseline_depth, {generator: loo_depth}) or None."""
    meta_path = model_dir / f"calibration_{concept}_meta.json"
    alllayer_path = model_dir / f"calibration_alllayer_{concept}.npy"
    if not meta_path.exists() or not alllayer_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    pair_ids = meta["corpus"]["pair_ids"]
    n_pairs = len(pair_ids)
    generators = np.array([pid.split("__")[-1] for pid in pair_ids])
    arr = np.load(alllayer_path, mmap_mode="r")
    n_layers = arr.shape[0]

    full_mask = np.ones(n_pairs, dtype=bool)
    base = peak_depth(arr, n_pairs, n_layers, full_mask)
    if base is None:
        del arr
        return None

    loo = {}
    for g in sorted(set(generators.tolist())):
        mask = generators != g
        d = peak_depth(arr, n_pairs, n_layers, mask)
        if d is not None:
            loo[g] = d

    del arr
    gc.collect()
    return base, loo


def main():
    summary_lines = []
    for model in MODELS:
        model_dir = DATA_ROOT / model
        baseline_depths = {}
        loo_depths_by_gen = {}  # generator -> {concept: depth}

        for concept in CONCEPTS:
            result = process_concept(model_dir, concept)
            if result is None:
                continue
            base, loo = result
            baseline_depths[concept] = base
            for g, d in loo.items():
                loo_depths_by_gen.setdefault(g, {})[concept] = d

        if len(baseline_depths) < 5:
            line = f"{model}: insufficient concepts ({len(baseline_depths)}), skipping"
            print(line)
            summary_lines.append(line)
            continue

        concepts_present = list(baseline_depths.keys())
        baseline_vec = [baseline_depths[c] for c in concepts_present]

        taus = []
        for g, depths in loo_depths_by_gen.items():
            if not all(c in depths for c in concepts_present):
                continue
            loo_vec = [depths[c] for c in concepts_present]
            tau, p = kendalltau(baseline_vec, loo_vec)
            taus.append(tau)

        taus = np.array(taus)
        line = (f"{model:35s} n_concepts={len(concepts_present):2d}  "
                f"n_generators_tested={len(taus):2d}  "
                f"tau: mean={taus.mean():.3f} min={taus.min():.3f} max={taus.max():.3f} "
                f"std={taus.std():.3f}")
        print(line, flush=True)
        summary_lines.append(line)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "full_ordering_summary.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

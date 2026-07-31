#!/usr/bin/env python3
"""Leave-one-generator-out robustness check for P4's Procrustes alignment
result. For each cross-family same-dimension model pair and each concept,
fits R via orthogonal Procrustes on 200-example calibration activations
(matching P4 §2.4), applies it, and measures aligned cosine. Repeats 14
times (once per excluded generator) and compares to the full-250 baseline.
No new GPU extraction — reuses existing calibration_alllayer_*.npy."""
import json
import numpy as np
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from scipy.stats import kendalltau

DATA_ROOT = Path.home() / "rosetta_data" / "paper_n250"

# Cross-family same-dimension pairs (2048-dim cluster, matching P4's Cluster B)
DIM_2048 = ["EleutherAI_pythia_1.4b", "facebook_opt_1.3b",
            "meta_llama_Llama_3.2_1B", "Qwen_Qwen2.5_3B"]
PAIRS_2048 = [
    ("EleutherAI_pythia_1.4b", "facebook_opt_1.3b"),
    ("EleutherAI_pythia_1.4b", "meta_llama_Llama_3.2_1B"),
    ("EleutherAI_pythia_1.4b", "Qwen_Qwen2.5_3B"),
    ("facebook_opt_1.3b", "meta_llama_Llama_3.2_1B"),
    ("facebook_opt_1.3b", "Qwen_Qwen2.5_3B"),
    ("meta_llama_Llama_3.2_1B", "Qwen_Qwen2.5_3B"),
]
# 2560-dim cluster
PAIRS_2560 = [("EleutherAI_pythia_2.8b", "facebook_opt_2.7b")]

CONCEPTS = [
    "credibility", "negation", "causation", "temporal_order", "sentiment",
    "certainty", "moral_valence", "specificity", "plurality", "agency",
    "formality", "threat_severity", "authorization", "urgency", "sarcasm",
    "deception", "exfiltration",
]


def load_concept(model_dir, concept):
    meta_path = model_dir / f"calibration_{concept}_meta.json"
    alllayer_path = model_dir / f"calibration_alllayer_{concept}.npy"
    if not meta_path.exists() or not alllayer_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    pair_ids = meta["corpus"]["pair_ids"]
    n_pairs = len(pair_ids)
    generators = np.array([pid.split("__")[-1] for pid in pair_ids])
    arr = np.load(alllayer_path)  # (n_layers, 2*n_pairs, hidden_dim)
    peak_layer = meta["files"][f"calibration_{concept}.npy"]["peak_layer"]
    return arr, generators, n_pairs, arr.shape[0], peak_layer


def fisher_separation(pos, neg):
    mu_pos, mu_neg = pos.mean(0), neg.mean(0)
    dist = np.linalg.norm(mu_pos - mu_neg)
    var_pos = pos.var(0, ddof=1).sum() if len(pos) > 1 else 0.0
    var_neg = neg.var(0, ddof=1).sum() if len(neg) > 1 else 0.0
    denom = np.sqrt(0.5 * (var_pos + var_neg))
    return dist / denom if denom > 0 else 0.0


def best_layer_for_subset(arr, n_pairs, n_layers, idx):
    """Re-select peak layer within the kept subset (matches the paper's
    per-concept peak-layer selection being data-dependent)."""
    scores = np.empty(n_layers)
    for l in range(n_layers):
        pos = arr[l, idx, :]
        neg = arr[l, n_pairs + idx, :]
        scores[l] = fisher_separation(pos, neg)
    return int(np.argmax(scores))


def dom_and_calib(arr, n_pairs, idx, layer):
    pos = arr[layer, idx, :]
    neg = arr[layer, n_pairs + idx, :]
    dom = pos.mean(0) - neg.mean(0)
    dom = dom / (np.linalg.norm(dom) + 1e-12)
    calib = np.vstack([pos, neg])  # (2*len(idx), d)
    return dom, calib


def aligned_cosine(dom_s, calib_s, dom_t, calib_t):
    """Fit R minimizing ||calib_t @ R - calib_s||_F (source=s, target=t,
    matching P4's ||A_s - A_t R||_F), rotate target DOM into source frame,
    return cosine(dom_s, dom_t_rotated)."""
    n = min(len(calib_s), len(calib_t))
    R, _ = orthogonal_procrustes(calib_t[:n], calib_s[:n])
    dom_t_rot = dom_t @ R
    dom_t_rot = dom_t_rot / (np.linalg.norm(dom_t_rot) + 1e-12)
    return float(np.dot(dom_s, dom_t_rot))


def main():
    all_pairs = PAIRS_2048 + PAIRS_2560
    results = {}

    for model_a, model_b in all_pairs:
        dir_a = DATA_ROOT / model_a
        dir_b = DATA_ROOT / model_b
        pair_key = f"{model_a} x {model_b}"
        baseline_cosines = []
        loo_cosines_by_gen = {}

        for concept in CONCEPTS:
            la = load_concept(dir_a, concept)
            lb = load_concept(dir_b, concept)
            if la is None or lb is None:
                continue
            arr_a, gens_a, np_a, nl_a, _ = la
            arr_b, gens_b, np_b, nl_b, _ = lb

            full_idx_a = np.arange(np_a)
            full_idx_b = np.arange(np_b)
            layer_a = best_layer_for_subset(arr_a, np_a, nl_a, full_idx_a)
            layer_b = best_layer_for_subset(arr_b, np_b, nl_b, full_idx_b)
            dom_a, calib_a = dom_and_calib(arr_a, np_a, full_idx_a, layer_a)
            dom_b, calib_b = dom_and_calib(arr_b, np_b, full_idx_b, layer_b)
            base_cos = aligned_cosine(dom_a, calib_a, dom_b, calib_b)
            baseline_cosines.append(base_cos)

            common_gens = sorted(set(gens_a.tolist()) & set(gens_b.tolist()))
            for g in common_gens:
                idx_a = np.where(gens_a != g)[0]
                idx_b = np.where(gens_b != g)[0]
                if len(idx_a) < 10 or len(idx_b) < 10:
                    continue
                la_layer = best_layer_for_subset(arr_a, np_a, nl_a, idx_a)
                lb_layer = best_layer_for_subset(arr_b, np_b, nl_b, idx_b)
                d_a, c_a = dom_and_calib(arr_a, np_a, idx_a, la_layer)
                d_b, c_b = dom_and_calib(arr_b, np_b, idx_b, lb_layer)
                cos = aligned_cosine(d_a, c_a, d_b, c_b)
                loo_cosines_by_gen.setdefault(g, []).append(cos)

        if not baseline_cosines:
            continue
        baseline_mean = np.mean(baseline_cosines)
        loo_means = {g: np.mean(v) for g, v in loo_cosines_by_gen.items() if v}
        loo_vals = np.array(list(loo_means.values()))
        results[pair_key] = {
            "n_concepts": len(baseline_cosines),
            "baseline_mean_cosine": round(float(baseline_mean), 4),
            "n_generators_tested": len(loo_vals),
            "loo_mean_cosine_mean": round(float(loo_vals.mean()), 4) if len(loo_vals) else None,
            "loo_mean_cosine_std": round(float(loo_vals.std()), 4) if len(loo_vals) else None,
            "loo_mean_cosine_min": round(float(loo_vals.min()), 4) if len(loo_vals) else None,
            "loo_mean_cosine_max": round(float(loo_vals.max()), 4) if len(loo_vals) else None,
            "max_abs_shift": round(float(np.max(np.abs(loo_vals - baseline_mean))), 4) if len(loo_vals) else None,
        }
        r = results[pair_key]
        print(f"{pair_key:55s} n_concepts={r['n_concepts']:2d}  baseline={r['baseline_mean_cosine']:.4f}  "
              f"LOO mean={r['loo_mean_cosine_mean']:.4f} std={r['loo_mean_cosine_std']:.4f} "
              f"range=[{r['loo_mean_cosine_min']:.4f},{r['loo_mean_cosine_max']:.4f}] "
              f"max_shift={r['max_abs_shift']:.4f}")

    out_path = Path("/tmp/claude-1000/-home-eigan-James-Rosetta-Program/"
                     "0f2f4484-d225-4d1e-a515-f3cc50a8192c/scratchpad/loo_procrustes_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

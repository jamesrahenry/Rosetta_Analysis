#!/usr/bin/env python3
"""Authoritative §3.1/§3.2 recompute: per-model dominant-peak depth for all
17 concepts x 28 base models, per-model Kendall tau against the grand-mean
ordering, Wilcoxon signed-rank test, and LOO (self-inclusion-corrected) tau.
Also recomputes the §3.3 credibility 3-way sub-population split.

No original script for this statistic could be located in the repo (see
papers/caz-validation/P3_REVIEW_LOOSE_ENDS.md section "NOT RESOLVED"). This
script is the from-scratch reimplementation adopted as authoritative,
using load_scored_region_df (0.5% prominence floor, composite scoring) --
the detector Table 3's caption and Section 3.3's credibility discussion both
point to for "dominant = tallest Fisher peak" and gentle-CAZ visibility.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wilcoxon

sys.path.insert(0, str(Path.home() / "rosetta_tools"))
from rosetta_tools.reporting import load_results_dir, load_scored_region_df

DATA_ROOT = Path.home() / "rosetta_data" / "paper_n250"

MODELS_28 = [
    "EleutherAI_pythia_70m", "EleutherAI_pythia_160m", "EleutherAI_pythia_410m",
    "EleutherAI_pythia_1b", "EleutherAI_pythia_1.4b", "EleutherAI_pythia_2.8b",
    "EleutherAI_pythia_6.9b", "EleutherAI_pythia_12b",
    "openai_community_gpt2", "openai_community_gpt2_medium",
    "openai_community_gpt2_large", "openai_community_gpt2_xl",
    "facebook_opt_125m", "facebook_opt_350m", "facebook_opt_1.3b",
    "facebook_opt_2.7b", "facebook_opt_6.7b",
    "Qwen_Qwen2.5_0.5B", "Qwen_Qwen2.5_1.5B", "Qwen_Qwen2.5_3B",
    "Qwen_Qwen2.5_7B", "Qwen_Qwen2.5_14B",
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

assert len(MODELS_28) == 28, len(MODELS_28)


def main():
    layer_df = load_results_dir([DATA_ROOT / m for m in MODELS_28])
    layer_df = layer_df[layer_df["concept"].isin(CONCEPTS)]
    region_df = load_scored_region_df(layer_df, min_prominence_frac=0.005)
    dom = region_df[region_df["is_dominant"]].copy()

    # pivot: model x concept -> depth_pct
    pivot = dom.pivot_table(index="model_id", columns="concept", values="depth_pct")
    pivot = pivot.reindex(columns=CONCEPTS)
    print("Coverage (models x concepts present):", pivot.shape,
          "missing cells:", pivot.isna().sum().sum())

    grand_mean = pivot.mean(axis=0)
    grand_order = grand_mean.sort_values()
    print("\n=== Grand mean ordering (Table 3 recompute) ===")
    for rank, (concept, val) in enumerate(grand_order.items(), 1):
        print(f"{rank:2d}. {concept:16s} {val:5.1f}%  std={pivot[concept].std():.1f}")

    taus = {}
    for model in pivot.index:
        row = pivot.loc[model].dropna()
        common = [c for c in CONCEPTS if c in row.index]
        if len(common) < 5:
            continue
        tau, p = kendalltau(grand_mean[common], row[common])
        taus[model] = (tau, p)

    tau_series = pd.Series({m: t for m, (t, p) in taus.items()}).sort_values()
    print("\n=== Per-model tau vs grand-mean ordering ===")
    for model, (tau, p) in sorted(taus.items(), key=lambda kv: kv[1][0]):
        print(f"{model:35s} tau={tau:+.3f}  p={p:.4f}")

    print(f"\nn models = {len(tau_series)}")
    print(f"median tau = {tau_series.median():.3f}")
    print(f"min tau = {tau_series.min():.3f} ({tau_series.idxmin()})")
    print(f"max tau = {tau_series.max():.3f} ({tau_series.idxmax()})")
    print(f"n positive = {(tau_series > 0).sum()} / {len(tau_series)}")
    print(f"n significant p<0.05 = {sum(1 for t,p in taus.values() if p < 0.05)} / {len(taus)}")

    w_stat, w_p = wilcoxon(tau_series.values)
    print(f"Wilcoxon signed-rank vs 0: W={w_stat:.1f} p={w_p:.3e}")

    # LOO (self-exclusion): grand mean of the OTHER 27 models
    loo_taus = {}
    for model in pivot.index:
        others = pivot.drop(index=model)
        gm_loo = others.mean(axis=0)
        row = pivot.loc[model].dropna()
        common = [c for c in CONCEPTS if c in row.index and c in gm_loo.index]
        tau, p = kendalltau(gm_loo[common], row[common])
        loo_taus[model] = tau
    loo_series = pd.Series(loo_taus)
    print(f"\nLOO median tau = {loo_series.median():.3f}")
    w2, p2 = wilcoxon(loo_series.values)
    print(f"LOO Wilcoxon: W={w2:.1f} p={p2:.3e}")

    # Cohort medians (model_id format as produced by load_result_df: HF-style with /)
    mha = ["EleutherAI/pythia-70m","EleutherAI/pythia-160m","EleutherAI/pythia-410m",
           "EleutherAI/pythia-1b","EleutherAI/pythia-1.4b","EleutherAI/pythia-2.8b",
           "EleutherAI/pythia-6.9b","EleutherAI/pythia-12b",
           "openai-community/gpt2","openai-community/gpt2-medium",
           "openai-community/gpt2-large","openai-community/gpt2-xl",
           "facebook/opt-125m","facebook/opt-350m","facebook/opt-1.3b",
           "facebook/opt-2.7b","facebook/opt-6.7b","microsoft/phi-2"]
    gqa = ["Qwen/Qwen2.5-0.5B","Qwen/Qwen2.5-1.5B","Qwen/Qwen2.5-3B",
           "Qwen/Qwen2.5-7B","Qwen/Qwen2.5-14B",
           "meta-llama/Llama-3.2-1B","meta-llama/Llama-3.2-3B",
           "mistralai/Mistral-7B-v0.3"]
    gemma = ["google/gemma-2-2b","google/gemma-2-9b"]
    for name, group in [("MHA", mha), ("GQA+SwiGLU", gqa), ("Gemma", gemma)]:
        vals = tau_series[[m for m in group if m in tau_series.index]]
        print(f"{name}: n={len(vals)} median tau={vals.median():.3f}")

    # §3.3 credibility sub-population split
    print("\n=== §3.3 credibility sub-population split ===")
    cred = dom[dom["concept"] == "credibility"][["model_id", "depth_pct"]].dropna()
    leakage = cred[cred["depth_pct"] < 25]
    late = cred[cred["depth_pct"] > 75]
    mid = cred[(cred["depth_pct"] >= 25) & (cred["depth_pct"] <= 75)]
    print(f"embedding leakage (<25%): {len(leakage)}/{len(cred)}")
    print(f"late assembly (>75%): {len(late)}/{len(cred)}")
    print(f"mid-stream (25-75%): {len(mid)}/{len(cred)}")
    print("leakage models:", sorted(leakage['model_id'].tolist()))
    print("late models:", sorted(late['model_id'].tolist()))

    outdir = Path(__file__).parent / "results"
    tau_series.to_csv(outdir / "tau_per_model.csv")
    pivot.to_csv(outdir / "depth_pivot.csv")


if __name__ == "__main__":
    main()

"""
align_gem_handoff.py — primary zero-PCA Procrustes alignment using GEM
handoff-layer DOM vectors, replacing the CAZ peak-layer extraction used in
preprint.md sec 3.1. Same cross-family, same-hidden-dimension pair design.

Reads papers/prh-validation/scripts/handoff_cache/<model>__<concept>.npz
(produced by extract_handoff_dom.py). Reuses rosetta_tools.alignment for
the actual Procrustes math (identical to the CAZ-peak primary analysis).

Written: 2026-07-09 UTC
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "rosetta_tools"))
from rosetta_tools.alignment import align_and_score, cosine_similarity  # noqa: E402

CACHE_DIR = Path(__file__).parent / "handoff_cache"
CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

# model directory slug -> (family, dimension cluster label)
FAMILY = {
    "openai_community_gpt2": "GPT-2",
    "EleutherAI_gpt_neo_125m": "GPT-Neo",
    "EleutherAI_pythia_160m": "Pythia", "EleutherAI_pythia_410m": "Pythia",
    "EleutherAI_pythia_1b": "Pythia", "EleutherAI_pythia_1.4b": "Pythia",
    "EleutherAI_pythia_2.8b": "Pythia", "EleutherAI_pythia_6.9b": "Pythia",
    "EleutherAI_pythia_12b": "Pythia",
    "facebook_opt_125m": "OPT", "facebook_opt_350m": "OPT",
    "facebook_opt_1.3b": "OPT", "facebook_opt_2.7b": "OPT",
    "facebook_opt_6.7b": "OPT",
    "openai_community_gpt2_medium": "GPT-2",
    "meta_llama_Llama_3.2_1B": "Llama 3", "meta_llama_Llama_3.2_1B_Instruct": "Llama 3",
    "meta_llama_Llama_3.1_8B": "Llama 3", "meta_llama_Llama_3.1_8B_Instruct": "Llama 3",
    "Qwen_Qwen2.5_3B": "Qwen 2.5", "Qwen_Qwen2.5_3B_Instruct": "Qwen 2.5",
    "Qwen_Qwen2.5_7B": "Qwen 2.5", "Qwen_Qwen2.5_7B_Instruct": "Qwen 2.5",
    "Qwen_Qwen2.5_14B": "Qwen 2.5", "Qwen_Qwen2.5_32B": "Qwen 2.5",
    "EleutherAI_pythia_2_8b": "Pythia",
    "microsoft_phi_2": "Phi-2",
    "mistralai_Mistral_7B_v0.3": "Mistral", "mistralai_Mistral_7B_Instruct_v0.3": "Mistral",
    "google_gemma_2_9b": "Gemma 2", "google_gemma_2_9b_it": "Gemma 2",
}

CLUSTER = {
    "openai_community_gpt2": "A", "EleutherAI_gpt_neo_125m": "A",
    "EleutherAI_pythia_160m": "A", "facebook_opt_125m": "A",
    "openai_community_gpt2_medium": "G", "facebook_opt_350m": "G",
    "EleutherAI_pythia_410m": "G",
    "EleutherAI_pythia_1b": "B", "EleutherAI_pythia_1.4b": "B",
    "facebook_opt_1.3b": "B", "meta_llama_Llama_3.2_1B": "B",
    "meta_llama_Llama_3.2_1B_Instruct": "B", "Qwen_Qwen2.5_3B": "B",
    "Qwen_Qwen2.5_3B_Instruct": "B",
    "EleutherAI_pythia_2.8b": "H", "facebook_opt_2.7b": "H", "microsoft_phi_2": "H",
    "EleutherAI_pythia_6.9b": "C", "facebook_opt_6.7b": "C",
    "meta_llama_Llama_3.1_8B": "C", "meta_llama_Llama_3.1_8B_Instruct": "C",
    "mistralai_Mistral_7B_v0.3": "C", "mistralai_Mistral_7B_Instruct_v0.3": "C",
    "Qwen_Qwen2.5_7B": "D", "Qwen_Qwen2.5_7B_Instruct": "D",
    "google_gemma_2_9b": "D", "google_gemma_2_9b_it": "D",
    "EleutherAI_pythia_12b": "E", "Qwen_Qwen2.5_14B": "E", "Qwen_Qwen2.5_32B": "E",
}

MODELS = sorted(FAMILY.keys())


def load(model: str, concept: str):
    d = np.load(CACHE_DIR / f"{model}__{concept}.npz")
    return d["dom_vector"], d["acts"], int(d["handoff_layer"])


def main():
    rows = []
    import time
    for concept in CONCEPTS:
        t0 = time.time()
        vectors, activations, handoffs = {}, {}, {}
        for m in MODELS:
            try:
                v, a, h = load(m, concept)
            except FileNotFoundError:
                continue
            vectors[m] = v
            activations[m] = a
            handoffs[m] = h

        for src in MODELS:
            for tgt in MODELS:
                if src == tgt or src not in vectors or tgt not in vectors:
                    continue
                if vectors[src].shape[0] != vectors[tgt].shape[0]:
                    continue  # not same-dim
                if FAMILY[src] == FAMILY[tgt]:
                    continue  # cross-family only, matching primary analysis
                try:
                    result = align_and_score(
                        vectors[src], vectors[tgt], activations[src], activations[tgt]
                    )
                except Exception as e:
                    print(f"SVD fail {src}x{tgt} {concept}: {e}")
                    continue
                rows.append({
                    "concept": concept,
                    "source_model": src, "target_model": tgt,
                    "source_family": FAMILY[src], "target_family": FAMILY[tgt],
                    "hidden_dim": vectors[src].shape[0],
                    "cluster": CLUSTER.get(src, "?"),
                    "source_handoff": handoffs[src], "target_handoff": handoffs[tgt],
                    **result,
                })
        n_done = sum(1 for r in rows if r["concept"] == concept)
        print(f"{concept}: {n_done} directional pairs ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "prh_gem_handoff_primary.csv", index=False)

    print(f"\nTotal directional pairs: {len(df)}")
    print(f"Mean raw cosine:     {df['raw_cosine'].mean():.4f}")
    print(f"Mean aligned cosine: {df['aligned_cosine'].mean():.4f}")
    print(f"Std aligned cosine:  {df['aligned_cosine'].std():.4f}")
    print("\nPer-concept means:")
    print(df.groupby("concept")["aligned_cosine"].mean().sort_values(ascending=False).to_string())
    print("\nPer-cluster means:")
    print(df.groupby("cluster")["aligned_cosine"].agg(["mean", "std", "count"]).to_string())


if __name__ == "__main__":
    main()

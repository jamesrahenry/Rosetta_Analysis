"""
extract_handoff_dom.py — extract GEM handoff-layer DOM vectors + calibration
activations for the PRH primary corpus, replacing the CAZ peak-layer extraction.

For each (model, concept): reads gem_<concept>.json for the deepest node's
handoff_layer, slices calibration_alllayer_<concept>.npy at that layer, and
saves a compact cache (500 x hidden_dim activations + DOM vector + handoff
layer index) to handoff_cache/<model_slug>_<concept>.npz.

Downloads calibration_alllayer_*.npy from HF on demand (paper-n250 revision)
and deletes it immediately after slicing — the full per-layer cache is
70-300MB/model-concept and disk is tight; the HF copy is the source of
truth and safe to re-fetch.

Written: 2026-07-09 UTC
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

ROSETTA_DATA = Path.home() / "rosetta_data"
PAPER_N250 = ROSETTA_DATA / "paper_n250"
CACHE_DIR = Path(__file__).parent / "handoff_cache"
CACHE_DIR.mkdir(exist_ok=True)

CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

# The 30 primary-corpus models (clusters A-E + instruct variants).
# Cluster F (Llama-3.1-70B, Qwen2.5-72B, falcon-40b) excluded: no all-layer
# cache exists anywhere (same gap as the current CAZ-peak analysis).
PRIMARY_MODELS = [
    # Cluster A - 768
    "openai_community_gpt2", "EleutherAI_gpt_neo_125m",
    "EleutherAI_pythia_160m", "facebook_opt_125m",
    # Cluster G - 1024
    "openai_community_gpt2_medium", "facebook_opt_350m", "EleutherAI_pythia_410m",
    # Cluster B - 2048
    "EleutherAI_pythia_1b", "EleutherAI_pythia_1.4b", "facebook_opt_1.3b",
    "meta_llama_Llama_3.2_1B", "Qwen_Qwen2.5_3B",
    # Cluster H - 2560
    "EleutherAI_pythia_2.8b", "facebook_opt_2.7b", "microsoft_phi_2",
    # Cluster C - 4096
    "EleutherAI_pythia_6.9b", "facebook_opt_6.7b",
    "meta_llama_Llama_3.1_8B", "mistralai_Mistral_7B_v0.3",
    # Cluster D - 3584
    "Qwen_Qwen2.5_7B", "google_gemma_2_9b",
    # Cluster E - 5120
    "EleutherAI_pythia_12b", "Qwen_Qwen2.5_14B", "Qwen_Qwen2.5_32B",
    # Instruct variants (matched dims, structurally independent points)
    "Qwen_Qwen2.5_3B_Instruct", "Qwen_Qwen2.5_7B_Instruct",
    "meta_llama_Llama_3.2_1B_Instruct", "meta_llama_Llama_3.1_8B_Instruct",
    "mistralai_Mistral_7B_Instruct_v0.3", "google_gemma_2_9b_it",
]


def ensure_alllayer(model: str, concept: str) -> Path:
    p = PAPER_N250 / model / f"calibration_alllayer_{concept}.npy"
    if not p.exists():
        hf_hub_download(
            "james-ra-henry/Rosetta-Activations",
            f"paper_n250/{model}/calibration_alllayer_{concept}.npy",
            repo_type="dataset", revision="paper-n250",
            local_dir=str(ROSETTA_DATA),
        )
    return p


def deepest_handoff_layer(model: str, concept: str) -> int:
    p = PAPER_N250 / model / f"gem_{concept}.json"
    d = json.loads(p.read_text())
    nodes = d["nodes"]
    deepest = max(nodes, key=lambda n: n["caz_end"])
    return int(deepest["handoff_layer"])


def process_one(model: str, concept: str, downloaded_this_call: list[Path]) -> dict | None:
    out_path = CACHE_DIR / f"{model}__{concept}.npz"
    if out_path.exists():
        return {"model": model, "concept": concept, "status": "cached"}

    handoff = deepest_handoff_layer(model, concept)
    alllayer_path = ensure_alllayer(model, concept)
    # Always slated for deletion after slicing, whether pre-existing or just
    # downloaded -- the compact handoff cache is all we need going forward,
    # and the full per-layer cache is fully HF-backed (disk is tight).
    downloaded_this_call.append(alllayer_path)

    acts = np.load(alllayer_path)  # (n_layers, 500, hidden_dim)
    n_layers = acts.shape[0]
    handoff = min(handoff, n_layers - 1)
    slice_ = acts[handoff]  # (500, hidden_dim)
    n = slice_.shape[0]
    half = n // 2
    pos_centroid = slice_[:half].mean(axis=0)
    neg_centroid = slice_[half:].mean(axis=0)
    dom = pos_centroid - neg_centroid
    norm = np.linalg.norm(dom)
    dom = dom / norm if norm > 0 else dom

    np.savez_compressed(
        out_path,
        acts=slice_.astype(np.float32),
        dom_vector=dom.astype(np.float32),
        handoff_layer=handoff,
        n_layers_total=n_layers,
    )
    return {"model": model, "concept": concept, "status": "extracted", "handoff_layer": handoff}


def main():
    results = []
    for model in PRIMARY_MODELS:
        downloaded_this_call: list[Path] = []
        for concept in CONCEPTS:
            try:
                r = process_one(model, concept, downloaded_this_call)
                results.append(r)
            except Exception as e:
                print(f"FAIL {model} {concept}: {e}")
                results.append({"model": model, "concept": concept, "status": "fail", "error": str(e)})
        # free disk: delete the raw alllayer files we downloaded for this model,
        # now that the compact handoff slices are saved. Fully HF-backed, safe.
        for p in downloaded_this_call:
            try:
                os.remove(p)
            except OSError:
                pass
        n_ok = sum(1 for r in results if r["model"] == model and r["status"] in ("extracted", "cached"))
        print(f"{model}: {n_ok}/{len(CONCEPTS)} ok")

    n_fail = sum(1 for r in results if r["status"] == "fail")
    print(f"\nTotal: {len(results)} attempted, {n_fail} failed")
    if n_fail:
        for r in results:
            if r["status"] == "fail":
                print(" ", r["model"], r["concept"], r.get("error"))


if __name__ == "__main__":
    main()

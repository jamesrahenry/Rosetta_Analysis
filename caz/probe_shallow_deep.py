"""
probe_shallow_deep.py — Test whether shallow peaks are lexical and deep peaks are compositional.

For each multimodal concept × model, compares the dom_vector at each peak
against two reference frames:

  1. TOKEN EMBEDDINGS — cosine similarity between the peak dom_vector and
     the embedding vectors of concept-relevant tokens (e.g. "reliable",
     "dubious" for credibility). High similarity at the shallow peak would
     confirm lexical encoding.

  2. CONTEXTUAL DIVERGENCE — for each contrastive pair, compute the
     difference between positive and negative activations at each peak layer.
     Project onto the dom_vector. Measure how much of the separation comes
     from individual token positions vs. distributed across the sequence.
     High concentration at concept-keyword positions = lexical; distributed
     = compositional.

Usage
-----
    # Run on GPU machine
    python src/probe_shallow_deep.py --all

    # Single family
    python src/probe_shallow_deep.py --family pythia

Output: results in visualizations/structure/probing/
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from rosetta_tools.caz import find_caz_regions, LayerMetrics
from rosetta_tools.paths import ROSETTA_RESULTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_ROOT = ROSETTA_RESULTS
DATA_ROOT = Path("data")
OUT_DIR = Path("visualizations/structure/probing")

KNOWN_FAMILIES = ["pythia", "gpt2", "opt", "qwen2", "gemma2"]

# Concept-specific probe tokens — words strongly associated with each pole
CONCEPT_TOKENS = {
    "credibility": {
        "positive": ["reliable", "trustworthy", "credible", "authoritative", "reputable",
                      "verified", "established", "legitimate", "expert", "proven"],
        "negative": ["dubious", "unreliable", "questionable", "unverified", "suspicious",
                      "fraudulent", "discredited", "untrustworthy", "fake", "bogus"],
    },
    "sentiment": {
        "positive": ["wonderful", "excellent", "amazing", "fantastic", "great",
                      "brilliant", "outstanding", "superb", "delightful", "perfect"],
        "negative": ["terrible", "awful", "horrible", "dreadful", "disgusting",
                      "miserable", "atrocious", "appalling", "abysmal", "wretched"],
    },
    "negation": {
        "positive": ["not", "never", "neither", "nowhere", "nothing",
                      "nobody", "none", "nor", "cannot", "without"],
        "negative": ["always", "every", "all", "certainly", "definitely",
                      "indeed", "absolutely", "completely", "everywhere", "everyone"],
    },
    "certainty": {
        "positive": ["certainly", "definitely", "undoubtedly", "clearly", "obviously",
                      "surely", "absolutely", "unquestionably", "evidently", "indisputably"],
        "negative": ["maybe", "perhaps", "possibly", "uncertain", "doubtful",
                      "questionable", "unclear", "debatable", "arguably", "conceivably"],
    },
    "causation": {
        "positive": ["because", "therefore", "caused", "resulted", "consequently",
                      "hence", "thus", "since", "due", "led"],
        "negative": ["despite", "although", "coincidentally", "regardless", "nevertheless",
                      "however", "unrelated", "independently", "randomly", "accidentally"],
    },
    "moral_valence": {
        "positive": ["virtuous", "ethical", "righteous", "honorable", "noble",
                      "moral", "just", "principled", "compassionate", "benevolent"],
        "negative": ["evil", "immoral", "corrupt", "wicked", "cruel",
                      "malicious", "depraved", "sinful", "vile", "reprehensible"],
    },
    "temporal_order": {
        "positive": ["before", "previously", "earlier", "prior", "preceding",
                      "formerly", "already", "initially", "first", "began"],
        "negative": ["after", "subsequently", "later", "following", "next",
                      "eventually", "finally", "then", "afterward", "concluded"],
    },
}


def get_all_embedding_vectors(model_id: str) -> dict[str, dict]:
    """Get token embedding vectors for ALL concepts from one model load.

    Loads the model once, extracts embeddings for every concept's probe
    tokens, then deletes the model and purges the HF cache.

    Returns dict: concept → {positive: [(word, vec), ...], negative: [...]}
    """
    from rosetta_tools.gpu_utils import purge_hf_cache

    log.info("  Loading model for embedding extraction...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True,
                                       dtype=torch.float32)

    # Get embedding matrix
    if hasattr(model, "get_input_embeddings"):
        embed_layer = model.get_input_embeddings()
    else:
        raise ValueError(f"Cannot find embedding layer for {model_id}")

    embed_matrix = embed_layer.weight.detach().cpu().numpy().astype(np.float64)

    # Extract for all concepts at once
    all_concepts = {}
    for concept, poles in CONCEPT_TOKENS.items():
        result = {}
        for pole in ["positive", "negative"]:
            words = poles.get(pole, [])
            vecs = []
            for word in words:
                ids = tokenizer.encode(word, add_special_tokens=False)
                if ids:
                    vec = embed_matrix[ids[0]]
                    vec = vec / (np.linalg.norm(vec) + 1e-10)
                    vecs.append((word, vec))
            result[pole] = vecs
        all_concepts[concept] = result

    # Clean up: delete model, free GPU, purge HF cache
    del model, embed_matrix, embed_layer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    purge_hf_cache(model_id)
    log.info("  Model deleted and HF cache purged.")

    return all_concepts


def probe_model_concept(
    model_dir: Path,
    concept: str,
    embed_vecs: dict,
) -> dict | None:
    """Compare peak dom_vectors against embedding vectors.

    Returns dict with probe results, or None if not multimodal.
    """
    f = model_dir / f"caz_{concept}.json"
    if not f.exists():
        return None

    with open(f) as fh:
        data = json.load(fh)

    model_id = data["model_id"]
    model_short = model_id.split("/")[-1]
    layer_metrics_raw = data["layer_data"]["metrics"]

    # Skip models with inconsistent dims
    dims = set(len(m["dom_vector"]) for m in layer_metrics_raw)
    if len(dims) > 1:
        return None

    metrics = [
        LayerMetrics(m["layer"], m["separation_fisher"], m["coherence"], m["velocity"])
        for m in layer_metrics_raw
    ]
    profile = find_caz_regions(metrics)
    if not profile.is_multimodal:
        return None

    sorted_regions = sorted(profile.regions, key=lambda r: r.peak)
    shallow_peak = sorted_regions[0].peak
    deep_peak = sorted_regions[-1].peak

    shallow_vec = np.array(layer_metrics_raw[shallow_peak]["dom_vector"], dtype=np.float64)
    deep_vec = np.array(layer_metrics_raw[deep_peak]["dom_vector"], dtype=np.float64)
    shallow_vec = shallow_vec / (np.linalg.norm(shallow_vec) + 1e-10)
    deep_vec = deep_vec / (np.linalg.norm(deep_vec) + 1e-10)

    # Compute cosine similarity between each peak dom_vector and each embedding vector
    results = {"model": model_short, "concept": concept,
               "shallow_peak": shallow_peak, "deep_peak": deep_peak}

    for pole in ["positive", "negative"]:
        for word, emb_vec in embed_vecs[pole]:
            # Need to match dimensions — embedding may differ from hidden_dim
            # (e.g. OPT embedding dim != hidden dim)
            if len(emb_vec) != len(shallow_vec):
                continue

            s_cos = float(np.dot(shallow_vec, emb_vec))
            d_cos = float(np.dot(deep_vec, emb_vec))
            results.setdefault("shallow_embed_cos", []).append(abs(s_cos))
            results.setdefault("deep_embed_cos", []).append(abs(d_cos))

    if "shallow_embed_cos" in results:
        results["shallow_embed_mean"] = np.mean(results["shallow_embed_cos"])
        results["deep_embed_mean"] = np.mean(results["deep_embed_cos"])
        results["ratio"] = results["shallow_embed_mean"] / (results["deep_embed_mean"] + 1e-10)

    return results


def discover_families() -> dict[str, list[Path]]:
    families = {}
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        for fam in KNOWN_FAMILIES:
            if d.name.startswith(f"{fam}_"):
                families.setdefault(fam, []).append(d)
                break
    return families


def main():
    parser = argparse.ArgumentParser(description="Probe shallow vs deep sub-representations")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--family", nargs="+", help="Family name(s)")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    family_dirs = discover_families()

    if args.all:
        targets = list(family_dirs.keys())
    else:
        targets = args.family

    all_dirs = []
    for fam in targets:
        all_dirs.extend(family_dirs.get(fam, []))

    concepts = list(CONCEPT_TOKENS.keys())

    # Group directories by model_id so we load each model only once
    from collections import defaultdict
    dirs_by_model = defaultdict(list)
    for model_dir in sorted(all_dirs):
        sample_f = next(model_dir.glob("caz_*.json"), None)
        if not sample_f or sample_f.name == "run_summary.json":
            continue
        with open(sample_f) as fh:
            model_id = json.load(fh)["model_id"]
        dirs_by_model[model_id].append(model_dir)

    all_results = []

    for model_id, model_dirs in dirs_by_model.items():
        model_short = model_id.split("/")[-1]
        log.info("Processing %s...", model_short)

        # Load model ONCE, extract embeddings for ALL concepts, then purge
        try:
            embed_by_concept = get_all_embedding_vectors(model_id)
        except Exception as e:
            log.warning("  Failed to load embeddings for %s: %s", model_id, e)
            continue

        for model_dir in model_dirs:
            for concept in concepts:
                embed_vecs = embed_by_concept.get(concept)
                if embed_vecs is None:
                    continue

                result = probe_model_concept(model_dir, concept, embed_vecs)
                if result:
                    all_results.append(result)
                    log.info("  %s × %s: shallow_embed=%.3f  deep_embed=%.3f  ratio=%.2f",
                             model_short, concept,
                             result.get("shallow_embed_mean", 0),
                             result.get("deep_embed_mean", 0),
                             result.get("ratio", 0))

    # Save results
    if all_results:
        import csv
        out_path = OUT_DIR / "shallow_deep_probing.csv"
        keys = ["model", "concept", "shallow_peak", "deep_peak",
                "shallow_embed_mean", "deep_embed_mean", "ratio"]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_results)
        log.info("Saved %d results to %s", len(all_results), out_path)

        # Summary
        import pandas as pd
        df = pd.DataFrame(all_results)
        print("\n" + "=" * 70)
        print("SUMMARY: Shallow vs Deep embedding similarity")
        print("=" * 70)
        for concept in df["concept"].unique():
            csub = df[df["concept"] == concept]
            if "shallow_embed_mean" not in csub.columns:
                continue
            csub = csub.dropna(subset=["shallow_embed_mean", "deep_embed_mean"])
            if csub.empty:
                continue
            print(f"  {concept:>20s}: n={len(csub)}  "
                  f"shallow={csub['shallow_embed_mean'].mean():.3f}  "
                  f"deep={csub['deep_embed_mean'].mean():.3f}  "
                  f"ratio={csub['ratio'].mean():.2f}")

        overall_shallow = df["shallow_embed_mean"].dropna().mean()
        overall_deep = df["deep_embed_mean"].dropna().mean()
        print(f"\n  Overall: shallow={overall_shallow:.3f}  deep={overall_deep:.3f}  "
              f"ratio={overall_shallow/overall_deep:.2f}")
        if overall_shallow > overall_deep:
            print("  → Shallow peaks are MORE aligned with token embeddings (lexical)")
        else:
            print("  → Deep peaks are MORE aligned with token embeddings (?)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gemma_scope_xval.py — Cross-validate CAZ peaks against Gemma Scope SAE features.

Loads Gemma-2-2b + Gemma Scope residual stream SAEs (one layer at a time),
runs concept-contrastive pairs through the model, and measures three things:

  1. LAYER AGREEMENT — Do SAE features that best discriminate concept+/concept-
     pairs cluster at CAZ peak layers?

  2. DIRECTION AGREEMENT — At CAZ peak layers, does the top differential SAE
     feature's decoder direction align with our eigenvector for that concept?

  3. SHARED CAZ PEAKS — At layers where multiple concept CAZs peak simultaneously,
     do any SAE features activate for more than one concept (polysemanticity)?

Usage:
    python src/gemma_scope_xval.py
    python src/gemma_scope_xval.py --concept credibility
    python src/gemma_scope_xval.py --pairs-per-concept 30 --top-k 10
    python src/gemma_scope_xval.py --out results/my_xval_run
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
CAZ_ROOT  = Path(__file__).resolve().parents[1]

def _find_feature_lib() -> Path:
    """Locate the feature library — submodule layout or standalone build."""
    candidates = [
        REPO_ROOT / "Rosetta_Feature_Library",   # submodule: Rosetta_Program/Rosetta_Feature_Library
        Path.home() / "Rosetta_Feature_Library",  # standalone clone alongside caz_scaling
        CAZ_ROOT / "feature_library",             # legacy: built inside caz_scaling/
    ]
    for p in candidates:
        if (p / "cazs").exists():
            return p
    return candidates[0]  # return primary path so error message is useful

def _find_pairs_dir() -> Path:
    """Locate the concept pairs directory."""
    candidates = [
        REPO_ROOT / "Rosetta_Concept_Pairs" / "pairs" / "raw" / "v1",
        Path.home() / "Rosetta_Concept_Pairs" / "pairs" / "raw" / "v1",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

FEATURE_LIB  = _find_feature_lib()
PAIRS_DIR    = _find_pairs_dir()
RESULTS_DIR  = CAZ_ROOT / "results"

MODEL_ID      = "google/gemma-2-2b"
SAE_RELEASE   = "gemma-scope-2b-pt-res"
SAE_WIDTH     = "16k"
# Target l0 sparsity — we pick the available variant closest to this per layer.
# ~75 is the medium-sparsity variant across most layers (5 variants exist per layer).
SAE_TARGET_L0 = 75

CONCEPTS = [
    "credibility", "certainty", "sentiment", "moral_valence",
    "causation", "temporal_order", "negation",
]

# ── data loading ──────────────────────────────────────────────────────────────

def load_pairs(concept: str, n: int) -> tuple[list[str], list[str]]:
    """Load up to n positive and n negative texts for a concept."""
    path = PAIRS_DIR / f"{concept}_consensus_pairs.jsonl"
    if not path.exists():
        log.warning("No pairs file for %s", concept)
        return [], []

    pos, neg = [], []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if item["label"] == 1 and len(pos) < n:
                pos.append(item["text"])
            elif item["label"] == 0 and len(neg) < n:
                neg.append(item["text"])
            if len(pos) >= n and len(neg) >= n:
                break

    log.info("  %s: %d positive, %d negative pairs", concept, len(pos), len(neg))
    return pos, neg


def load_caz_peaks(concept: str) -> list[dict]:
    """Load CAZ regions for Gemma-2-2b for a given concept."""
    path = FEATURE_LIB / "cazs" / concept / "gemma-2-2b.json"
    if not path.exists():
        log.warning("No CAZ data for %s/gemma-2-2b", concept)
        return []
    data = json.loads(path.read_text())
    return data.get("regions", [])


def concept_eigenvectors(
    pos_acts: np.ndarray,
    neg_acts: np.ndarray,
    n_components: int = 10,
) -> np.ndarray:
    """
    Compute concept-specific eigenvectors from paired activation differences.

    pos_acts / neg_acts: (n_pairs, hidden_dim) — activations at one layer.
    Returns (n_components, hidden_dim) unit eigenvectors (top SVD components of
    the paired-difference matrix), concept-specific rather than global.
    """
    delta = (pos_acts - neg_acts).astype(np.float32)   # (n_pairs, hidden_dim)
    delta -= delta.mean(axis=0, keepdims=True)
    k = min(n_components, delta.shape[0] - 1, delta.shape[1])
    if k < 1:
        return delta / (np.linalg.norm(delta, axis=1, keepdims=True) + 1e-8)
    # SVD: U (n_pairs×k), s, Vt (k×hidden_dim)
    _, _, Vt = np.linalg.svd(delta, full_matrices=False)
    return Vt[:k]   # (k, hidden_dim) — already unit norm from SVD


# ── model + SAE loading ───────────────────────────────────────────────────────

def load_model(device: str) -> tuple:
    log.info("Loading %s...", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    log.info("Model loaded (%d layers, hidden_dim=%d)",
             model.config.num_hidden_layers,
             model.config.hidden_size)
    return model, tokenizer


def build_sae_index() -> dict[int, str]:
    """
    Build a {layer: sae_id} map by reading the SAELens YAML registry directly.
    For each layer, picks the 16k-width variant whose average_l0 is closest
    to SAE_TARGET_L0.
    """
    import yaml, importlib, os
    sae_lens_pkg = os.path.dirname(importlib.import_module("sae_lens").__file__)
    yaml_path = os.path.join(sae_lens_pkg, "pretrained_saes.yaml")
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    saes = data.get(SAE_RELEASE, {}).get("saes", [])
    # Collect all layer_N/width_16k ids
    by_layer: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for entry in saes:
        sae_id = entry["id"]
        if not sae_id.startswith("layer_") or f"width_{SAE_WIDTH}" not in sae_id:
            continue
        if "checkpoint" in sae_id:
            continue
        parts = sae_id.split("/")    # ['layer_N', 'width_16k', 'average_l0_X']
        layer = int(parts[0].split("_")[1])
        l0    = int(parts[2].split("_")[-1])
        by_layer[layer].append((l0, sae_id))

    index = {}
    for layer, variants in by_layer.items():
        # Pick variant with l0 closest to target
        best = min(variants, key=lambda x: abs(x[0] - SAE_TARGET_L0))
        index[layer] = best[1]

    log.info("SAE index built: %d layers, target_l0=%d", len(index), SAE_TARGET_L0)
    for L in sorted(index):
        log.debug("  layer %2d → %s", L, index[L])
    return index


def load_sae(layer: int, sae_index: dict[int, str], device: str):
    """Load Gemma Scope SAE for a given layer. Returns (sae, W_dec) or (None, None)."""
    sae_id = sae_index.get(layer)
    if sae_id is None:
        log.warning("No SAE available for layer %d", layer)
        return None, None
    try:
        from sae_lens import SAE
        sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id)
        sae = sae.to(device)
        sae.eval()
        # W_dec: (n_features, hidden_dim) — decoder directions
        W_dec = sae.W_dec.detach().float().cpu().numpy()
        return sae, W_dec
    except Exception as e:
        log.warning("Could not load SAE for layer %d (%s): %s", layer, sae_id, e)
        return None, None


# ── forward pass + activation capture ────────────────────────────────────────

@torch.no_grad()
def get_residual_streams(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    batch_size: int = 4,
    max_length: int = 256,
) -> np.ndarray:
    """
    Run texts through the model, capturing the mean-pooled residual stream
    at every layer for each text.

    Returns: (n_texts, n_layers, hidden_dim) float32
    """
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    all_acts = np.zeros((len(texts), n_layers, hidden_dim), dtype=np.float32)

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start : batch_start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        # Collect hidden states from every layer
        layer_acts = []

        def make_hook(layer_idx):
            def hook(module, input, output):
                # Gemma DecoderLayer output: (hidden_states, ...) or just hidden_states
                hs = output[0] if isinstance(output, tuple) else output
                # Mean pool over token dim, ignoring padding
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hs.float() * mask).sum(dim=1) / mask.sum(dim=1)
                layer_acts.append(pooled.cpu().numpy())
            return hook

        handles = []
        for i, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(make_hook(i)))

        try:
            model(**enc)
        finally:
            for h in handles:
                h.remove()

        # layer_acts is a list of length n_layers, each (batch, hidden_dim)
        for layer_idx, acts in enumerate(layer_acts):
            all_acts[batch_start : batch_start + len(batch_texts), layer_idx] = acts

        log.info("    batch %d/%d done", batch_start // batch_size + 1,
                 (len(texts) + batch_size - 1) // batch_size)

    return all_acts


# ── per-layer SAE analysis ────────────────────────────────────────────────────

def compute_differential_acts(
    sae,
    pos_resid: np.ndarray,
    neg_resid: np.ndarray,
    device: str,
) -> np.ndarray:
    """
    Encode pos and neg activations through the SAE, return
    mean_pos_acts - mean_neg_acts per SAE feature.

    pos_resid: (n_pos, hidden_dim)
    neg_resid: (n_neg, hidden_dim)
    Returns: (n_sae_features,)
    """
    def encode_batch(arr):
        t = torch.tensor(arr, dtype=torch.float32, device=device)
        with torch.no_grad():
            acts = sae.encode(t)   # (n, n_features)
        return acts.cpu().float().numpy()

    pos_acts = encode_batch(pos_resid)
    neg_acts = encode_batch(neg_resid)
    return pos_acts.mean(axis=0) - neg_acts.mean(axis=0)


def layer_agreement_score(
    differential: np.ndarray,
    top_k: int,
) -> float:
    """Sum of top-K absolute differential activations — how strongly does any
    SAE feature discriminate at this layer?"""
    top_vals = np.sort(np.abs(differential))[-top_k:]
    return float(top_vals.mean())


def direction_agreement(
    differential: np.ndarray,
    W_dec: np.ndarray,
    eigenvectors: np.ndarray,
    top_k: int,
) -> list[dict]:
    """
    For the top-K differential SAE features, compute cosine similarity
    between their decoder direction and each of our top eigenvectors.

    Returns list of {feature_idx, differential, cos_sim_to_top_eigvec, top_eigvec_idx}
    """
    top_idxs = np.argsort(np.abs(differential))[-top_k:][::-1]
    results = []
    for feat_idx in top_idxs:
        dec_dir = W_dec[feat_idx]                          # (hidden_dim,)
        dec_dir = dec_dir / (np.linalg.norm(dec_dir) + 1e-8)
        # cosine with each eigenvector
        eigvec_norms = eigenvectors / (
            np.linalg.norm(eigenvectors, axis=1, keepdims=True) + 1e-8
        )                                                  # (n_dirs, hidden_dim)
        cos_sims = eigvec_norms @ dec_dir                  # (n_dirs,)
        best_eigvec = int(np.argmax(np.abs(cos_sims)))
        results.append({
            "feature_idx": int(feat_idx),
            "differential": float(differential[feat_idx]),
            "cos_sim_to_top_eigvec": float(cos_sims[best_eigvec]),
            "top_eigvec_idx": best_eigvec,
            "all_cos_sims": cos_sims[:10].tolist(),        # top-10 eigvecs
        })
    return results


# ── shared_caz_peaks polysemanticity ────────────────────────────────────────────

def find_shared_caz_peaks(all_caz_peaks: dict[str, list[dict]]) -> dict[int, list[str]]:
    """Return {layer: [concept, ...]} for layers where ≥2 concepts peak."""
    layer_concepts: dict[int, list[str]] = defaultdict(list)
    for concept, regions in all_caz_peaks.items():
        for r in regions:
            layer_concepts[r["peak_layer"]].append(concept)
    return {L: cs for L, cs in layer_concepts.items() if len(cs) >= 2}


# ── main analysis ─────────────────────────────────────────────────────────────

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 0. Validate required data paths ──────────────────────────────────────
    missing = []
    if not PAIRS_DIR.exists():
        missing.append(f"Concept pairs: {PAIRS_DIR}")
    if not FEATURE_LIB.exists():
        missing.append(f"Feature library: {FEATURE_LIB}")
    if missing:
        for m in missing:
            log.error("MISSING: %s", m)
        log.error(
            "Required data repos not found. Clone them alongside caz_scaling:\n"
            "  git clone git@github.com:jamesrahenry/Rosetta_Concept_Pairs.git ~/Rosetta_Concept_Pairs\n"
            "  git clone git@github.com:jamesrahenry/Rosetta_Feature_Library.git ~/Rosetta_Feature_Library\n"
            "Or rsync from dev: rsync -a <dev>:~/Source/Rosetta_Program/Rosetta_Concept_Pairs ~/Rosetta_Concept_Pairs"
        )
        raise SystemExit(1)

    concepts = [args.concept] if args.concept else CONCEPTS

    # ── 1. Load model + SAE index ─────────────────────────────────────────────
    model, tokenizer = load_model(device)
    n_layers = model.config.num_hidden_layers   # 26 for Gemma-2-2b
    sae_index = build_sae_index()

    # ── 2. Load CAZ peaks for all concepts ───────────────────────────────────
    log.info("Loading CAZ peaks...")
    all_caz_peaks = {c: load_caz_peaks(c) for c in concepts}
    shared_caz_peaks = find_shared_caz_peaks(all_caz_peaks)
    log.info("Shared CAZ peak layers (≥2 concepts): %s",
             sorted(shared_caz_peaks.keys()))

    # ── 3. Per-concept analysis ───────────────────────────────────────────────
    layer_agreement    : dict[str, list[float]]      = {}
    direction_results  : dict[str, dict[int, list]]  = {}
    shared_caz_peaks_poly: dict[int, dict]             = defaultdict(dict)

    for concept in concepts:
        log.info("=== %s ===", concept)
        t0 = time.time()

        pos_texts, neg_texts = load_pairs(concept, args.pairs_per_concept)
        if not pos_texts or not neg_texts:
            continue

        # Run model once — capture residual stream at every layer
        log.info("  Forward pass: %d pos + %d neg texts",
                 len(pos_texts), len(neg_texts))
        pos_resid = get_residual_streams(
            model, tokenizer, pos_texts, device, args.batch_size)
        neg_resid = get_residual_streams(
            model, tokenizer, neg_texts, device, args.batch_size)
        # shapes: (n_texts, n_layers, hidden_dim)

        caz_peak_layers = {r["peak_layer"] for r in all_caz_peaks.get(concept, [])}

        agreement_by_layer = []
        dir_by_layer = {}

        for layer in range(n_layers):
            # Load SAE for this layer (and immediately discard after use)
            sae, W_dec = load_sae(layer, sae_index, device)
            if sae is None:
                agreement_by_layer.append(0.0)
                continue

            # Encode activations
            differential = compute_differential_acts(
                sae,
                pos_resid[:, layer, :],
                neg_resid[:, layer, :],
                device,
            )

            agreement_score = layer_agreement_score(differential, args.top_k)
            agreement_by_layer.append(agreement_score)

            # Direction agreement only at CAZ peak layers (and shared_caz_peaks)
            if layer in caz_peak_layers or layer in shared_caz_peaks:
                eigvecs = concept_eigenvectors(
                    pos_resid[:, layer, :].cpu().numpy() if hasattr(pos_resid, 'cpu') else pos_resid[:, layer, :],
                    neg_resid[:, layer, :].cpu().numpy() if hasattr(neg_resid, 'cpu') else neg_resid[:, layer, :],
                )
                dir_by_layer[layer] = direction_agreement(
                    differential, W_dec, eigvecs, args.top_k
                )

            # Shared CAZ peak polysemanticity
            if layer in shared_caz_peaks and concept in shared_caz_peaks[layer]:
                top_feat_idxs = np.argsort(np.abs(differential))[-args.top_k:].tolist()
                shared_caz_peaks_poly[layer][concept] = {
                    "top_features": top_feat_idxs,
                    "top_differentials": [float(differential[i]) for i in top_feat_idxs],
                }

            # Free SAE from GPU
            del sae
            if device == "cuda":
                torch.cuda.empty_cache()

        layer_agreement[concept] = agreement_by_layer
        direction_results[concept] = dir_by_layer
        log.info("  Done in %.1fs", time.time() - t0)

    # ── 4. Compute shared_caz_peaks polysemanticity score ───────────────────────
    # For each shared_caz_peaks layer: which SAE features appear in top-K for
    # multiple concepts? Those are polysemantic features.
    shared_caz_peaks_summary = {}
    for layer, concept_data in shared_caz_peaks_poly.items():
        concepts_at_layer = list(concept_data.keys())
        if len(concepts_at_layer) < 2:
            continue
        # Count feature overlap
        feature_concept_map: dict[int, list[str]] = defaultdict(list)
        for concept, data in concept_data.items():
            for feat in data["top_features"]:
                feature_concept_map[feat].append(concept)
        polysemantic = {
            feat: cs
            for feat, cs in feature_concept_map.items()
            if len(cs) >= 2
        }
        shared_caz_peaks_summary[layer] = {
            "concepts": concepts_at_layer,
            "caz_concept_names": shared_caz_peaks.get(layer, []),
            "polysemantic_features": {str(k): v for k, v in polysemantic.items()},
            "n_polysemantic": len(polysemantic),
            "total_top_features": sum(
                len(d["top_features"]) for d in concept_data.values()
            ),
        }

    # ── 5. Summarise layer agreement vs. CAZ peaks ────────────────────────────
    agreement_summary = {}
    for concept, scores in layer_agreement.items():
        caz_regions = all_caz_peaks.get(concept, [])
        caz_peak_layers = [r["peak_layer"] for r in caz_regions]
        if not caz_peak_layers or not scores:
            continue

        scores_arr = np.array(scores)
        # Rank of each CAZ peak layer in the agreement profile
        sorted_layers = np.argsort(scores_arr)[::-1].tolist()
        caz_ranks = [sorted_layers.index(L) for L in caz_peak_layers if L < len(scores)]
        sae_top_layer = int(np.argmax(scores_arr))
        sae_peak_score = float(scores_arr[sae_top_layer])

        agreement_summary[concept] = {
            "caz_peak_layers": caz_peak_layers,
            "sae_top_layer": sae_top_layer,
            "sae_peak_score": sae_peak_score,
            "caz_peak_ranks_in_sae": caz_ranks,   # lower = better agreement
            "mean_caz_peak_rank": float(np.mean(caz_ranks)) if caz_ranks else None,
            "n_layers": len(scores),
        }

    # ── 6. Write output ───────────────────────────────────────────────────────
    log.info("Writing results to %s", out_dir)

    (out_dir / "layer_agreement.json").write_text(
        json.dumps({
            "description": "Per-layer SAE discrimination score (mean top-K |differential|). "
                           "Higher = SAE features better separate concept+/concept- at this layer.",
            "by_concept": layer_agreement,
            "summary": agreement_summary,
        }, indent=2)
    )

    # Flatten direction_results for serialization
    dir_out = {}
    for concept, by_layer in direction_results.items():
        dir_out[concept] = {str(L): v for L, v in by_layer.items()}
    (out_dir / "direction_agreement.json").write_text(
        json.dumps({
            "description": "At CAZ peak layers: cosine similarity between top differential "
                           "SAE decoder directions and our eigenvectors.",
            "by_concept": dir_out,
        }, indent=2)
    )

    (out_dir / "shared_caz_peaks.json").write_text(
        json.dumps({
            "description": "At multi-concept CAZ peak layers: which SAE features appear "
                           "in top-K for multiple concepts (polysemanticity signal).",
            "shared_caz_peaks_layers": {str(L): v for L, v in shared_caz_peaks_summary.items()},
            "all_shared_caz_peaks": {str(L): cs for L, cs in shared_caz_peaks.items()},
        }, indent=2)
    )

    # High-level summary
    summary = {
        "model": MODEL_ID,
        "sae_release": SAE_RELEASE,
        "n_concepts": len(concepts),
        "pairs_per_concept": args.pairs_per_concept,
        "top_k": args.top_k,
        "layer_agreement": {
            c: {
                "sae_top_layer": v["sae_top_layer"],
                "caz_peak_layers": v["caz_peak_layers"],
                "mean_caz_peak_rank": v["mean_caz_peak_rank"],
            }
            for c, v in agreement_summary.items()
        },
        "shared_caz_peaks": {
            str(L): {
                "concepts": v["concepts"],
                "n_polysemantic_features": v["n_polysemantic"],
            }
            for L, v in shared_caz_peaks_summary.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Print readable summary
    log.info("")
    log.info("=== LAYER AGREEMENT ===")
    for concept, v in agreement_summary.items():
        ranks = v["caz_peak_ranks_in_sae"]
        log.info(
            "  %-16s  CAZ peaks=%s  SAE top layer=%d  CAZ ranks in SAE=%s",
            concept, v["caz_peak_layers"], v["sae_top_layer"],
            ranks,
        )

    log.info("")
    log.info("=== SHARED CAZ PEAK POLYSEMANTICITY ===")
    if shared_caz_peaks_summary:
        for layer, v in sorted(shared_caz_peaks_summary.items()):
            log.info(
                "  Layer %2d: concepts=%s  polysemantic features=%d",
                layer, v["concepts"], v["n_polysemantic"],
            )
    else:
        log.info("  (no shared_caz_peaks layers in analyzed concepts)")

    log.info("Done. Results at %s", out_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-validate CAZ peaks against Gemma Scope SAE features"
    )
    parser.add_argument(
        "--concept", default=None,
        help="Single concept to analyze (default: all 7)"
    )
    parser.add_argument(
        "--pairs-per-concept", type=int, default=50,
        help="Max positive/negative pairs per concept (default: 50)"
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Top-K SAE features to consider per layer (default: 20)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Tokenization batch size for forward passes (default: 4)"
    )
    parser.add_argument(
        "--out", type=str,
        default=str(CAZ_ROOT / "results" / "gemma_scope_xval"),
        help="Output directory"
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

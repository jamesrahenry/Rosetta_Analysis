#!/usr/bin/env python3
"""C14 — calibration-draw + shared-topic sensitivity of DOM estimates.

Round-3 corpus-quality gate for G7 (ROUND3_COMPUTE_PLAN.md C12-C14, Hopper
tb98f327); covers P4 §4.5's calibration-composition caveat. Two legs:

  (a) CALIBRATION-DRAW STABILITY. Bootstrap-resample the 250-pair calibration
      set (draw pairs with replacement, B times); recompute the DOM direction
      at every layer and the Fisher-peak layer per draw. Report how much the
      peak depth and the DOM direction wander with calibration composition.
      NOTE: this is the bootstrap-WITHIN-250 proxy. The full "re-draw from the
      ~1,300/concept pool" version needs fresh activations (a GPU extraction
      job) — flagged as a follow-up, mirroring C5's leg split.

  (b) SHARED-TOPIC INFLATION. All generators share ~100 topics, so a DOM
      direction could be riding topic vocabulary rather than the concept. Test:
      split the pairs into two disjoint TOPIC groups (no topic in both),
      estimate DOM from each, and compare the cross-half cosine against the
      ordinary RANDOM split-half cosine. If topic-disjoint agreement is much
      below random-split agreement, the direction is topic-inflated.

INDEPENDENCE (no Eigan / no CIA leakage — hard requirement 2026-07-16):
  * NO CIA data: 17 semantic concepts only; the CIA-only `obfuscation` concept
    is untouched.
  * NO Eigan-pipeline leakage: DOM (centroid difference) and Fisher separation
    are recomputed here with self-contained numpy from the stored per-text
    calibration activations — no rosetta_tools extraction/CAZ/GEM code path.
    `dataset` text loader is not even needed (labels come from the stored
    activation ordering: first half pos, second half neg, the pipeline's own
    convention). Inputs are the calibration_alllayer_<concept>.npy on HF
    @ paper-n250 — the same files the DOM pipeline consumes, read directly.

Representative 8-model subset (families x scales) keeps the activation download
bounded; stability/topic-inflation are model-level properties well estimated
from a spanning sample. Output: c14_calibration_topic_sensitivity_results.json,
uploaded to HF _round3_corpus_quality/.

Written: 2026-07-16 UTC
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]
assert "obfuscation" not in CONCEPTS, "obfuscation is CIA-only — must not appear"

# Representative subset: one+ per family, dims 768 -> 5120.
MODELS = [
    "openai_community_gpt2",         # GPT-2    768
    "EleutherAI_pythia_1.4b",        # Pythia   2048
    "microsoft_phi_2",               # Phi-2    2560
    "facebook_opt_2.7b",             # OPT      2560
    "Qwen_Qwen2.5_7B",               # Qwen 2.5 3584
    "meta_llama_Llama_3.1_8B",       # Llama 3  4096
    "mistralai_Mistral_7B_v0.3",     # Mistral  4096
    "EleutherAI_pythia_12b",         # Pythia   5120
]

N_BOOT = 50
SEED = 42
INPUT_REVISION = "paper-n250"
ROSETTA_DATA = Path(os.environ.get("ROSETTA_DATA", Path.home() / "rosetta_data"))
LOCAL = ROSETTA_DATA / "paper_n250"
OUT_JSON = Path(__file__).parent / "c14_calibration_topic_sensitivity_results.json"
HF_SUBTREE = "paper_n250/_round3_corpus_quality"
HF_DATASET = "james-ra-henry/Rosetta-Activations"


def ensure(model: str, fname: str) -> Path:
    p = LOCAL / model / fname
    if not p.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download(HF_DATASET, f"paper_n250/{model}/{fname}",
                        repo_type="dataset", revision=INPUT_REVISION,
                        local_dir=str(ROSETTA_DATA))
    return p


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def dom(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Normalised centroid difference — the pipeline's DOM, self-contained."""
    return unit(pos.mean(0) - neg.mean(0))


def fisher_nd(pos: np.ndarray, neg: np.ndarray) -> float:
    num = float(np.linalg.norm(pos.mean(0) - neg.mean(0)))
    den = float(np.sqrt(0.5 * (pos.var(0, ddof=1).sum() + neg.var(0, ddof=1).sum())))
    return num / den if den > 1e-12 else 0.0


def peak_layer(acts_pos: np.ndarray, acts_neg: np.ndarray) -> int:
    """argmax over layers of Fisher separation (self-contained)."""
    return int(np.argmax([fisher_nd(acts_pos[l], acts_neg[l])
                          for l in range(acts_pos.shape[0])]))


def load_pos_neg(model: str, concept: str):
    """Returns (acts_pos, acts_neg) each [n_layers, n_pairs, d] from the stored
    calibration file, using the pipeline's ordering (first half pos, 2nd neg)."""
    p = ensure(model, f"calibration_alllayer_{concept}.npy")
    acts = np.load(p)                       # [n_layers, n_texts, d]
    n = acts.shape[1]
    half = n // 2
    pos = acts[:, :half, :].astype(np.float64)
    neg = acts[:, half:2 * half, :].astype(np.float64)
    return pos, neg, p


# Per-pair topics are model-independent (same n=250 'train' draw for every
# model), so they can be precomputed once and shipped as c14_topics.json —
# this lets the job run on a host that doesn't carry the corpus (e.g. the
# rented CPU box). TEXT metadata only; no pipeline features.
_TOPICS_CACHE: dict | None = None


def topic_groups(model: str, concept: str, n_pairs: int) -> np.ndarray | None:
    """Per-pair topic labels for this concept's n=250 'train' draw. Prefers the
    shipped c14_topics.json; falls back to the text loader if the corpus is
    present; else None (topic leg skipped)."""
    global _TOPICS_CACHE
    if _TOPICS_CACHE is None:
        tf = Path(__file__).parent / "c14_topics.json"
        _TOPICS_CACHE = json.loads(tf.read_text()) if tf.exists() else {}
    if concept in _TOPICS_CACHE and len(_TOPICS_CACHE[concept]) == n_pairs:
        return np.array(_TOPICS_CACHE[concept])
    try:
        RP = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(RP / "rosetta_tools"))
        from rosetta_tools.dataset import load_concept_pairs
        pairs = load_concept_pairs(concept, n=n_pairs, split="train")
        if len(pairs) != n_pairs:
            return None
        return np.array([p.topic for p in pairs])
    except Exception:  # noqa: BLE001
        return None


def calibration_stability(pos: np.ndarray, neg: np.ndarray, pk: int,
                          rng: np.random.Generator) -> dict:
    n_pairs = pos.shape[1]
    n_layers = pos.shape[0]
    ref = unit(pos[pk].mean(0) - neg[pk].mean(0))
    peaks, cosines = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n_pairs, size=n_pairs)      # bootstrap w/ replacement
        bp, bn = pos[:, idx, :], neg[:, idx, :]
        peaks.append(peak_layer(bp, bn))
        cosines.append(float(np.dot(ref, unit(bp[pk].mean(0) - bn[pk].mean(0)))))
    peaks = np.array(peaks)
    return {
        "peak_layer_full": pk, "n_layers": n_layers,
        "peak_layer_boot_sd": float(peaks.std()),
        "peak_depth_pct_boot_sd": float(peaks.std() / max(n_layers - 1, 1)),
        "peak_layer_boot_iqr": float(np.subtract(*np.percentile(peaks, [75, 25]))),
        "dom_cosine_to_full_mean": float(np.mean(cosines)),
        "dom_cosine_to_full_min": float(np.min(cosines)),
    }


def split_half_cos(pos: np.ndarray, neg: np.ndarray, pk: int,
                   idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    da = unit(pos[pk][idx_a].mean(0) - neg[pk][idx_a].mean(0))
    db = unit(pos[pk][idx_b].mean(0) - neg[pk][idx_b].mean(0))
    return float(np.dot(da, db))


def topic_inflation(pos: np.ndarray, neg: np.ndarray, pk: int,
                    topics: np.ndarray, rng: np.random.Generator) -> dict | None:
    if topics is None:
        return None
    n_pairs = pos.shape[1]
    uniq = list(dict.fromkeys(topics.tolist()))
    if len(uniq) < 4:
        return None
    # topic-disjoint split: partition topics into two groups, average over a few
    # random topic partitions for stability
    tdisj = []
    for _ in range(20):
        perm = rng.permutation(uniq)
        ga = set(perm[: len(perm) // 2])
        mask_a = np.array([t in ga for t in topics])
        if mask_a.sum() < 5 or (~mask_a).sum() < 5:
            continue
        tdisj.append(split_half_cos(pos, neg, pk, np.where(mask_a)[0],
                                    np.where(~mask_a)[0]))
    # random split (ignores topic) at matched sizes
    rnd = []
    for _ in range(20):
        perm = rng.permutation(n_pairs)
        h = n_pairs // 2
        rnd.append(split_half_cos(pos, neg, pk, perm[:h], perm[h:]))
    if not tdisj:
        return None
    td, rd = float(np.mean(tdisj)), float(np.mean(rnd))
    return {
        "n_topics": len(uniq),
        "topic_disjoint_splithalf_cos": td,
        "random_splithalf_cos": rd,
        "inflation_gap_random_minus_topic": rd - td,
        "note": "gap ~ 0 => direction is topic-independent; large positive gap "
                "=> topic vocabulary inflates within-concept agreement",
    }


def main() -> None:
    upload = "--no-upload" not in sys.argv
    only = None
    if "--concepts" in sys.argv:
        i = sys.argv.index("--concepts")
        only = sys.argv[i + 1].split(",")
    concepts = only or CONCEPTS
    rng = np.random.default_rng(SEED)
    per = {}
    for model in MODELS:
        per[model] = {}
        for concept in concepts:
            npy_p = None
            try:
                pos, neg, npy_p = load_pos_neg(model, concept)
                pk = peak_layer(pos, neg)
                topics = topic_groups(model, concept, pos.shape[1])
                per[model][concept] = {
                    "calibration_stability":
                        calibration_stability(pos, neg, pk, rng),
                    "topic_inflation":
                        topic_inflation(pos, neg, pk, topics, rng),
                }
                cs = per[model][concept]["calibration_stability"]
                ti = per[model][concept]["topic_inflation"]
                gap = "n/a" if ti is None else f"{ti['inflation_gap_random_minus_topic']:.3f}"
                print(f"[c14] {model:28s} {concept:15s} "
                      f"peakSD={cs['peak_layer_boot_sd']:.2f}L "
                      f"domCos={cs['dom_cosine_to_full_mean']:.3f} "
                      f"topicGap={gap}")
            except Exception as e:  # noqa: BLE001
                print(f"[c14] SKIP {model} {concept}: {e}")
            finally:
                if npy_p is not None:
                    try:
                        os.remove(npy_p)     # bound disk: calibration npys are large
                    except OSError:
                        pass

    # aggregates across the subset
    def collect(path):
        out = []
        for m in per:
            for c in per[m]:
                v = per[m][c]
                for k in path:
                    v = v.get(k) if isinstance(v, dict) else None
                    if v is None:
                        break
                if v is not None:
                    out.append(v)
        return out

    peak_sds = collect(["calibration_stability", "peak_depth_pct_boot_sd"])
    dom_cos = collect(["calibration_stability", "dom_cosine_to_full_mean"])
    gaps = collect(["topic_inflation", "inflation_gap_random_minus_topic"])
    td = collect(["topic_inflation", "topic_disjoint_splithalf_cos"])
    rd = collect(["topic_inflation", "random_splithalf_cos"])
    summary = {
        "n_model_concept_cells": len(dom_cos),
        "calibration_draw": {
            "mean_peak_depth_pct_boot_sd": float(np.mean(peak_sds)) if peak_sds else None,
            "mean_dom_cosine_to_full": float(np.mean(dom_cos)) if dom_cos else None,
            "min_dom_cosine_to_full": float(np.min(dom_cos)) if dom_cos else None,
        },
        "topic_inflation": {
            "mean_topic_disjoint_splithalf_cos": float(np.mean(td)) if td else None,
            "mean_random_splithalf_cos": float(np.mean(rd)) if rd else None,
            "mean_inflation_gap": float(np.mean(gaps)) if gaps else None,
            "note": "compare mean topic-disjoint vs random split-half cosine; a "
                    "small gap means DOM directions are not topic-driven",
        },
    }
    out = {
        "job": "c14_calibration_topic_sensitivity",
        "scope": f"{len(MODELS)}-model representative subset; 17 semantic "
                 f"concepts (obfuscation/CIA excluded); bootstrap-within-250 "
                 f"(full ~1300-pool re-draw is a GPU follow-up)",
        "independence": "self-contained numpy DOM/Fisher from stored calibration "
                        "activations; no rosetta_tools extraction/CAZ/GEM; "
                        "topics via text loader only",
        "n_boot": N_BOOT, "seed": SEED, "models": MODELS,
        "summary": summary, "per_model_concept": per,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print("\n[c14] SUMMARY:", json.dumps(summary, indent=2))
    print(f"[c14] wrote {OUT_JSON.name}")

    if upload:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=str(OUT_JSON),
                            path_in_repo=f"{HF_SUBTREE}/{OUT_JSON.name}",
                            repo_id=HF_DATASET, repo_type="dataset",
                            commit_message="C14 calibration-draw + topic "
                                            "sensitivity (G7 gate, tb98f327)")
        print(f"[c14] uploaded to {HF_SUBTREE}/")


if __name__ == "__main__":
    main()

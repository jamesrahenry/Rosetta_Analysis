#!/usr/bin/env python3
"""gemma_softcap_ablation.py — is gemma-2's DOM-vector instability caused by
attention logit softcapping?

Background
----------
GEMMA_INSTABILITY_NOTE.md (claude:h200-gpu-run) established that gemma-2's
per-concept DOM vectors don't converge across independent contrastive-pair
draws, at any sample size tested (125 -> ~2000 pairs), while every other
architecture in the corpus reproduces at >=0.96. ROBUST_DOM_ESTIMATOR_TEST.md
(claude:p4-review) then ruled out "a few outlier dimensions dominate the
mean-difference" as the mechanism: switching to a median- or trimmed-mean
DOM estimator made stability *worse* for gemma-2-2b on 17/17 concepts, not
better -- the opposite of what outlier-contamination would predict.

That leaves open question #1's second half: an architectural cause. gemma-2's
one genuinely distinctive feature vs. every control model in this corpus
(gpt2, Pythia, OPT, Qwen, Llama, Mistral -- none of which softcap) is
attention logit softcapping:

    attn_weights = tanh(attn_weights / cap) * cap        (cap = 50.0 for gemma-2-2b)

confirmed directly in google/gemma-2-2b's config (attn_logit_softcapping =
50.0) and in transformers' modeling_gemma2.py (eager_attention_forward).
This is a saturating nonlinearity applied to every attention layer's raw
scores before the softmax -- exactly the kind of thing that could make
hidden states more sensitive to *which* specific texts land in a batch
(small differences in raw attention scores get squashed non-linearly,
rather than just rescaled) without needing any "outlier dimension" at all.
(final_logit_softcapping, cap=30.0, is NOT tested here -- it only applies to
the LM head's vocab logits, which this pipeline's AutoModel extraction never
computes; irrelevant to hidden-state DOM vectors.)

Method
------
Load google/gemma-2-2b TWICE: once with attn_logit_softcapping as shipped
(50.0), once with it forced to None (disabled) via the config passed to
`from_pretrained` -- softcap is read once per attention layer at __init__,
so patching model.config *after* construction would silently do nothing;
the config must be modified before the model is built. Both loads force
`attn_implementation="eager"` so the softcap codepath is guaranteed to run
(sdpa/flash-attention historically had inconsistent softcap support) --
holding the attention backend constant isolates softcapping as the only
variable. A same-layer assertion after each load confirms the intended
state actually reached the constructed attention modules, not just the
top-level config object.

For each of a handful of concepts (the worst performers from
ROBUST_DOM_ESTIMATOR_TEST.md's split-half results, plus two of the best as
an internal control that the intervention doesn't secretly break things):
draw N=250 pairs (rosetta_tools' current deterministic sha256-seeded
sampler), run 5 random half/half splits, compute the mean-difference DOM
vector per half at every layer, and record split-half cosine agreement.
Compare the softcap-on vs softcap-off curves per concept and per layer.

If disabling softcap raises split-half agreement toward the ~0.96+ the
non-gemma corpus achieves: softcapping is (at least a major part of) the
mechanism. If agreement is unchanged or still low: softcapping isn't it,
and the cause is something else architectural (256k-vocab tied embeddings,
the pre+post RMSNorm pattern, or something not yet hypothesized) -- rules
out rather than confirms, same as the median-estimator test did.

Cost: cheap. google/gemma-2-2b's own run_summary.json (paper_n250) shows
~15s per concept for a full 500-text extraction: this job runs roughly the
same amount of forward-pass work per (concept, split, config) as one
concept's worth of the original corpus extraction, twice over (two model
loads) -- back-of-envelope ~15-20 GPU-minutes total for 8 concepts, no
timeout/fallback machinery needed (contrast with the Cluster F job, which
budgets hours for 40-70B models; this is a single 2B model, fp16/bf16, that
already loads and extracts fast in the existing corpus timing data).

Written: 2026-07-16 UTC
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from rosetta_tools.dataset import load_concept_pairs, texts_by_label
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.gpu_utils import get_device, get_dtype

from common import dom_direction, hf_upload, hf_verify, log

MODEL_ID = "google/gemma-2-2b"

# Worst 6 from ROBUST_DOM_ESTIMATOR_TEST.md's split-half mean-diff results
# (mean agreement, paper_n250 N=250): exfiltration 0.603, authorization 0.500,
# threat_severity 0.542, urgency 0.607, deception 0.572, agency 0.635.
# Plus 2 of the best (formality 0.907, credibility 0.876) as an internal
# control -- if disabling softcap breaks these, the intervention itself is
# suspect, not just "different from before."
CONCEPTS = [
    "exfiltration", "authorization", "threat_severity", "urgency",
    "deception", "agency", "formality", "credibility",
]
N_PAIRS = 250
N_SPLITS = 5
BATCH_SIZE = 16
RESULTS_FILE = "gemma_softcap_ablation_results.json"


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else 0.0


def load_model(softcap_enabled: bool, device: str, dtype):
    """Load gemma-2-2b with attn_logit_softcapping as-shipped or disabled.

    Softcap is copied from config into each Gemma2Attention.__init__ as
    self.attn_logit_softcapping -- setting model.config.attn_logit_softcapping
    after construction does NOT change already-built layers. The config must
    be modified before from_pretrained builds the model.
    """
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    expected = cfg.attn_logit_softcapping  # 50.0 as shipped
    if not softcap_enabled:
        cfg.attn_logit_softcapping = None
        expected = None

    model = AutoModel.from_pretrained(
        MODEL_ID, config=cfg, dtype=dtype, attn_implementation="eager",
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Verify the intended state actually reached the constructed attention
    # layers, not just the top-level config object we passed in.
    actual = model.layers[0].self_attn.attn_logit_softcapping
    if actual != expected:
        raise RuntimeError(
            f"softcap verification failed: expected {expected!r} on layer 0's "
            f"self_attn.attn_logit_softcapping, got {actual!r}. The config-"
            f"before-construction approach may not apply to this transformers "
            f"version's Gemma2 implementation -- check modeling_gemma2.py "
            f"before trusting any result from this script."
        )
    log.info(
        "loaded %s: attn_logit_softcapping=%s (softcap_enabled=%s, verified on layer 0)",
        MODEL_ID, actual, softcap_enabled,
    )
    return model, tokenizer


def split_half_curves(model, tokenizer, device, pos_texts, neg_texts, n_splits=N_SPLITS):
    """Returns list[list[float]]: per_layer_cos[layer] = [cosine per split]."""
    n = min(len(pos_texts), len(neg_texts))
    per_layer_cos: list[list[float]] | None = None

    for s in range(n_splits):
        rng = np.random.RandomState(s)
        perm_pos = rng.permutation(n)
        perm_neg = rng.permutation(n)
        h = n // 2
        pos_a = [pos_texts[i] for i in perm_pos[:h]]
        pos_b = [pos_texts[i] for i in perm_pos[h : 2 * h]]
        neg_a = [neg_texts[i] for i in perm_neg[:h]]
        neg_b = [neg_texts[i] for i in perm_neg[h : 2 * h]]

        acts_pos_a = extract_layer_activations(model, tokenizer, pos_a, device=device, batch_size=BATCH_SIZE, pool="last")
        acts_neg_a = extract_layer_activations(model, tokenizer, neg_a, device=device, batch_size=BATCH_SIZE, pool="last")
        acts_pos_b = extract_layer_activations(model, tokenizer, pos_b, device=device, batch_size=BATCH_SIZE, pool="last")
        acts_neg_b = extract_layer_activations(model, tokenizer, neg_b, device=device, batch_size=BATCH_SIZE, pool="last")

        n_layers = len(acts_pos_a)
        if per_layer_cos is None:
            per_layer_cos = [[] for _ in range(n_layers)]
        for layer in range(n_layers):
            dom_a = dom_direction(acts_pos_a[layer], acts_neg_a[layer])
            dom_b = dom_direction(acts_pos_b[layer], acts_neg_b[layer])
            per_layer_cos[layer].append(cosine(dom_a, dom_b))

    assert per_layer_cos is not None
    return per_layer_cos


def run_one_config(softcap_enabled: bool, concepts: list[str], device: str, dtype,
                    n_splits: int = N_SPLITS) -> dict:
    model, tokenizer = load_model(softcap_enabled, device, dtype)
    label = "softcap_on" if softcap_enabled else "softcap_off"
    out = {}
    try:
        for concept in concepts:
            pairs = load_concept_pairs(concept, n=N_PAIRS, split="train")
            pos_texts, neg_texts = texts_by_label(pairs)
            per_layer_cos = split_half_curves(model, tokenizer, device, pos_texts, neg_texts, n_splits=n_splits)
            means = [float(np.mean(c)) for c in per_layer_cos]
            mins = [float(np.min(c)) for c in per_layer_cos]
            best = int(np.argmax(means))
            out[concept] = {
                "per_layer_mean": means,
                "per_layer_min": mins,
                "overall_mean": float(np.mean(means)),
                "best_layer": best,
                "best_layer_mean": means[best],
            }
            log.info("[%s] %-16s overall_mean=%.4f best_layer=%d (%.4f)",
                      label, concept, out[concept]["overall_mean"], best, means[best])
    finally:
        del model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", nargs="*", default=CONCEPTS)
    ap.add_argument("--smoke", action="store_true",
                     help="1 concept, 2 splits -- fast correctness check before the real run")
    args = ap.parse_args()

    concepts = args.concepts
    n_splits = N_SPLITS
    if args.smoke:
        concepts = concepts[:1]
        n_splits = 2

    device = get_device("auto")
    dtype = get_dtype(device)
    log.info("device=%s dtype=%s concepts=%s n_splits=%d", device, dtype, concepts, n_splits)

    results = {
        "model_id": MODEL_ID,
        "n_pairs": N_PAIRS,
        "n_splits": n_splits,
        "softcap_on": run_one_config(True, concepts, device, dtype, n_splits=n_splits),
        "softcap_off": run_one_config(False, concepts, device, dtype, n_splits=n_splits),
    }

    out_path = Path(RESULTS_FILE)
    out_path.write_text(json.dumps(results, indent=2))
    log.info("wrote %s", out_path)

    if not args.smoke:
        hf_upload("gemma_softcap_ablation", out_path)
        hf_verify("gemma_softcap_ablation", [out_path.name])

    log.info("=== SUMMARY (overall_mean split-half cosine, softcap on vs off) ===")
    for concept in concepts:
        on = results["softcap_on"][concept]["overall_mean"]
        off = results["softcap_off"][concept]["overall_mean"]
        log.info("%-16s  on=%.4f  off=%.4f  delta=%+.4f", concept, on, off, off - on)


if __name__ == "__main__":
    main()

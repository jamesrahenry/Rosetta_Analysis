#!/usr/bin/env python3
"""gemma_rmsnorm_sandwich_ablation.py — does gemma-2's pre+post RMSNorm
sandwich explain its DOM-vector instability?

Background
----------
Third mechanism test in the gemma-2 DOM-instability chain (see
GEMMA_INSTABILITY_NOTE.md's "Mechanism candidates" section):
1. Outlier dimensions / estimator geometry -- RULED OUT
   (ROBUST_DOM_ESTIMATOR_TEST.md: median/trimmed-mean made things worse).
2. Environment (transformers version) -- RULED OUT (this repo's own
   verdict section: instability reproduces in the canonical env).
3. Attention logit softcapping -- RULED OUT
   (gemma_softcap_ablation.py: on/off delta <=0.0008 on every concept;
   softcap is a de-facto no-op at this corpus's activation scale).

While pulling gemma_softcap_ablation.py's results together, a side
observation sharpened the remaining question: `extract_layer_activations`
includes layer 0 (the raw embedding output, before any transformer block --
the corpus's own `extract.py` normally discards this layer, but this repo's
split-half harness doesn't). Layer 0 turns out to already be gemma-2-2b's
*worst* layer on every one of the 8 tested concepts (0.05-0.53, one
concept negative) -- compare gpt2 control, where 6/8 concepts are already
0.85-0.98 at layer 0. That's a strong, separate piece of evidence for the
"256k-vocab tied embeddings" hypothesis -- a large chunk of gemma's
instability is present before any transformer computation happens at all,
which this script does not re-litigate (see GEMMA_EMBEDDING_LAYER_CHECK.md).

What layer-0 instability does NOT explain is the rest of the curve: gemma's
split-half agreement climbs substantially from layer 0 toward a mid/late-
depth peak (still only 0.59-0.91, never gpt2's 0.95+), then in several
concepts erodes again toward the final layers. That climb-then-erode
dynamic through the transformer stack is where the remaining architectural
suspect lives: gemma-2 wraps each sublayer in FOUR RMSNorms instead of the
usual two --

    residual = hidden_states
    hidden_states = input_layernorm(hidden_states)              # pre-attn (standard)
    hidden_states = self_attn(hidden_states)
    hidden_states = post_attention_layernorm(hidden_states)      # gemma-2 ONLY
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = pre_feedforward_layernorm(hidden_states)     # pre-mlp (standard)
    hidden_states = mlp(hidden_states)
    hidden_states = post_feedforward_layernorm(hidden_states)    # gemma-2 ONLY
    hidden_states = residual + hidden_states

(confirmed directly against transformers' modeling_gemma2.py,
Gemma2DecoderLayer.forward). The two "post" norms -- renormalizing each
sublayer's OUTPUT before it's added into the residual stream -- are
gemma-2's distinguishing addition vs. the pre-norm-only design every other
model in this corpus uses (Llama, Mistral, Qwen, GPT-2, Pythia, OPT). Each
carries a learned per-dimension scale (`output * (1 + weight)`), applied
freshly per example. If that per-example renormalization is sensitive to
exactly which texts land in a batch -- amplifying small differences instead
of averaging them out -- it's a plausible mechanism for depth-accumulating,
sample-dependent instability of exactly the shape observed.

Method
------
Same split-half harness as gemma_softcap_ablation.py. Two arms:
- **sandwich_on**: gemma-2-2b as shipped (all 4 RMSNorms per layer).
- **sandwich_off**: after loading, replace every layer's
  `post_attention_layernorm` and `post_feedforward_layernorm` with
  `nn.Identity()` -- collapsing the block to the standard pre-norm-only
  design (residual + raw sublayer output, no renormalization before the
  add). `input_layernorm` and `pre_feedforward_layernorm` (the two
  standard, non-gemma-specific norms) are left untouched in both arms, so
  only the gemma-specific "post" half of the sandwich is the variable.

This is a much more invasive intervention than the softcap toggle -- it's
architectural surgery on a pretrained model, not a documented config knob,
and will very likely degrade the model's actual output quality/coherence.
That's an acceptable cost here: the question is narrowly mechanistic (does
this specific design choice casually contribute to split-half DOM-vector
instability), not "does the model still work well." A same-layer type
check after the swap confirms the intended module is actually in place
(`nn.Identity` vs. `Gemma2RMSNorm`) before trusting any result.

If sandwich_off's climb-to-peak reaches materially higher agreement than
sandwich_on (closer to gpt2's 0.95+, not just noise-level movement like the
softcap test showed): the post-norms are a real contributor, and the paper
now has a mechanism, not just a phenomenon. If the delta is noise-level like
softcap's was: this is ruled out too, and instability's remaining depth-wise
component (on top of the embedding-layer floor) is still unexplained --
worth documenting as a genuinely open question rather than continuing to
guess architecturally.

Cost: same ballpark as gemma_softcap_ablation.py (~15-20 GPU-minutes, single
2B model, two loads, 8 concepts, 5 splits each).

Written: 2026-07-16 UTC
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from rosetta_tools.dataset import load_concept_pairs, texts_by_label
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.gpu_utils import get_device, get_dtype

from common import dom_direction, hf_upload, hf_verify, log

MODEL_ID = "google/gemma-2-2b"

# Same 8 concepts as gemma_softcap_ablation.py, for direct comparability
# across the mechanism tests (worst 6 from ROBUST_DOM_ESTIMATOR_TEST.md's
# split-half results + best 2 as an internal control).
CONCEPTS = [
    "exfiltration", "authorization", "threat_severity", "urgency",
    "deception", "agency", "formality", "credibility",
]
N_PAIRS = 250
N_SPLITS = 5
BATCH_SIZE = 16
RESULTS_FILE = "gemma_rmsnorm_sandwich_ablation_results.json"


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else 0.0


def load_model(sandwich_ablated: bool, device: str, dtype):
    """Load gemma-2-2b as shipped, optionally stripping the gemma-specific
    'post' RMSNorm half (post_attention_layernorm, post_feedforward_layernorm)
    on every decoder layer -- collapsing to a standard pre-norm-only block.

    Unlike softcap (cached per-attention-layer at __init__ from config, so it
    had to be set before construction), the post-norm modules are ordinary
    submodules that can be swapped directly on an already-built model.
    """
    model = AutoModel.from_pretrained(
        MODEL_ID, dtype=dtype, attn_implementation="eager",
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if sandwich_ablated:
        for layer in model.layers:
            layer.post_attention_layernorm = nn.Identity()
            layer.post_feedforward_layernorm = nn.Identity()

    # Verify the intended module is actually in place on every layer, not
    # just layer 0 -- a partial swap would silently produce a meaningless
    # mixed-architecture result.
    expect_identity = sandwich_ablated
    for i, layer in enumerate(model.layers):
        for name in ("post_attention_layernorm", "post_feedforward_layernorm"):
            mod = getattr(layer, name)
            is_identity = isinstance(mod, nn.Identity)
            if is_identity != expect_identity:
                raise RuntimeError(
                    f"sandwich ablation verification failed at layer {i}.{name}: "
                    f"expected {'Identity' if expect_identity else 'Gemma2RMSNorm'}, "
                    f"got {type(mod).__name__}"
                )
    log.info(
        "loaded %s: post-norm sandwich %s (verified on all %d layers)",
        MODEL_ID, "ABLATED (Identity)" if sandwich_ablated else "as shipped (Gemma2RMSNorm)",
        len(model.layers),
    )
    return model, tokenizer


def split_half_curves(model, tokenizer, device, pos_texts, neg_texts, n_splits=N_SPLITS):
    """Returns list[list[float]]: per_layer_cos[layer] = [cosine per split].
    layer 0 = raw embedding output (extract_layer_activations includes it;
    the corpus's own extract.py discards it -- kept here deliberately, see
    module docstring)."""
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


def run_one_config(sandwich_ablated: bool, concepts: list[str], device: str, dtype,
                    n_splits: int = N_SPLITS) -> dict:
    model, tokenizer = load_model(sandwich_ablated, device, dtype)
    label = "sandwich_off" if sandwich_ablated else "sandwich_on"
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
                "layer0_embed_mean": means[0],
                "overall_mean": float(np.mean(means)),
                "best_layer": best,
                "best_layer_mean": means[best],
            }
            log.info("[%s] %-16s layer0=%.4f overall_mean=%.4f best_layer=%d (%.4f)",
                      label, concept, means[0], out[concept]["overall_mean"], best, means[best])
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
        "sandwich_on": run_one_config(False, concepts, device, dtype, n_splits=n_splits),
        "sandwich_off": run_one_config(True, concepts, device, dtype, n_splits=n_splits),
    }

    out_path = Path(RESULTS_FILE)
    out_path.write_text(json.dumps(results, indent=2))
    log.info("wrote %s", out_path)

    if not args.smoke:
        hf_upload("gemma_rmsnorm_sandwich_ablation", out_path)
        hf_verify("gemma_rmsnorm_sandwich_ablation", [out_path.name])

    log.info("=== SUMMARY (overall_mean split-half cosine, sandwich on vs off) ===")
    for concept in concepts:
        on = results["sandwich_on"][concept]["overall_mean"]
        off = results["sandwich_off"][concept]["overall_mean"]
        log.info("%-16s  on=%.4f  off=%.4f  delta=%+.4f", concept, on, off, off - on)


if __name__ == "__main__":
    main()

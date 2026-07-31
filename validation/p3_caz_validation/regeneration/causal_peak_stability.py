#!/usr/bin/env python3
"""Split-half stability of the CAUSAL-peak assignment (§6.4 divergence control).

Round-4 adversarial review (antagonistic-review-2026-07-19-round4.md) raised
the deepest objection to the paper's headline: the "dominant peak" (argmax
CAZ score) and the "causal peak" (argmin self-retained separation from
single-direction ablation) both derive from the same difference-of-means
estimates on the same 250 pairs, and §6.9 shows those estimates are unstable.
So the 50.3% divergence could be estimation noise in the causal-peak
ASSIGNMENT, not a real legibility/causality dissociation. Decisive control:
is the argmin-self-retained region reproducible across independent halves of
the 250 pairs?

Protocol faithful to gem/ablate_multimodal.py: region set fixed from the
full-draw scored detector; per half, DOM direction recomputed at each region
peak, ablated during a forward pass (DirectionalAblator at the peak layer),
self-retained = separation at peak+1 after ablation / baseline, per half.
causal_peak = argmin self_retained. Compare across halves.

CPU; small/mid models only (the GPU host is retired). Reports, per model and
pooled: (1) causal-peak assignment agreement A vs B; (2) divergence-verdict
agreement (dominant fixed from full draw); (3) a chance baseline (1/k).
"""
import json
import sys
import numpy as np
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.caz import compute_separation, find_caz_regions_scored, LayerMetrics
from rosetta_tools.ablation import DirectionalAblator, get_transformer_layers
from rosetta_tools.dataset import load_concept_pairs, texts_by_label

ROOT = Path.home() / "rosetta_data" / "paper_n250"
CONCEPTS = ["credibility", "negation", "causation", "temporal_order", "sentiment",
            "certainty", "moral_valence", "specificity", "plurality", "agency",
            "formality", "threat_severity", "authorization", "urgency", "sarcasm", "deception"]
MODELS = [  # (hf_id, local_slug) — CPU-runnable spread: small MHA + GQA
    ("EleutherAI/pythia-70m", "EleutherAI_pythia_70m"),
    ("EleutherAI/pythia-160m", "EleutherAI_pythia_160m"),
    ("EleutherAI/pythia-410m", "EleutherAI_pythia_410m"),
    ("openai-community/gpt2", "openai_community_gpt2"),
    ("openai-community/gpt2-medium", "openai_community_gpt2_medium"),
    ("Qwen/Qwen2.5-0.5B", "Qwen_Qwen2.5_0.5B"),
    ("Qwen/Qwen2.5-1.5B", "Qwen_Qwen2.5_1.5B"),
]
rng = np.random.default_rng(0)


def regions_for(slug, concept):
    f = ROOT / slug / f"caz_{concept}.json"
    if not f.exists():
        return None, None
    ld = json.load(open(f))["layer_data"]
    lm = [LayerMetrics(m["layer"], m["separation_fisher"], m["coherence"], m["velocity"]) for m in ld["metrics"]]
    prof = find_caz_regions_scored(lm)
    if prof.n_regions < 2:
        return None, None
    regs = sorted(prof.regions, key=lambda r: r.peak)
    return [(int(r.peak), float(r.caz_score)) for r in regs], int(ld["n_layers"])


def self_retained_for_half(model, tok, layers, peaks, pos_t, neg_t, n_layers, device, bs):
    measure_at = [min(p + 1, n_layers - 1) for p in peaks]
    # baseline (one extraction, no ablation)
    pos = extract_layer_activations(model, tok, pos_t, device=device, batch_size=bs, pool="last")
    neg = extract_layer_activations(model, tok, neg_t, device=device, batch_size=bs, pool="last")
    def sep_at(P, N, layer):
        ai = min(layer + 1, len(P) - 1)
        return compute_separation(P[ai], N[ai])
    base = {p: sep_at(pos, neg, p) for p in measure_at}
    # DOM per region at its peak (from this half's baseline acts)
    doms = []
    for p in peaks:
        ai = min(p + 1, len(pos) - 1)
        d = pos[ai].mean(0) - neg[ai].mean(0)
        doms.append(d.astype(np.float64))
    # per-region ablation
    sr = []
    for i, p in enumerate(peaks):
        with DirectionalAblator(layers[p], doms[i], dtype=next(model.parameters()).dtype):
            posa = extract_layer_activations(model, tok, pos_t, device=device, batch_size=bs, pool="last")
            nega = extract_layer_activations(model, tok, neg_t, device=device, batch_size=bs, pool="last")
        a = sep_at(posa, nega, measure_at[i]); b = base[measure_at[i]]
        sr.append(100 * a / b if b > 0 else 100.0)
    return sr


def main():
    device = "cpu"; bs = 16
    out = {"job": "causal-peak split-half assignment stability", "models": {}, "cells": []}
    for hf_id, slug in MODELS:
        print(f"=== {hf_id}", flush=True)
        tok = AutoTokenizer.from_pretrained(hf_id)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32)
        model.eval()
        layers = get_transformer_layers(model)
        m_agree_causal = []; m_agree_verdict = []; m_k = []
        for c in CONCEPTS:
            regs, nL = regions_for(slug, c)
            if regs is None:
                continue
            peaks = [p for p, _ in regs]; scores = [s for _, s in regs]
            dominant = int(np.argmax(scores))
            pairs = load_concept_pairs(c, n=250)
            pos_all, neg_all = texts_by_label(pairs)
            n = len(pos_all); idx = rng.permutation(n); A, B = idx[: n // 2], idx[n // 2:]
            srA = self_retained_for_half(model, tok, layers, peaks,
                                         [pos_all[i] for i in A], [neg_all[i] for i in A], nL, device, bs)
            srB = self_retained_for_half(model, tok, layers, peaks,
                                         [pos_all[i] for i in B], [neg_all[i] for i in B], nL, device, bs)
            cA, cB = int(np.argmin(srA)), int(np.argmin(srB))
            agree_causal = int(cA == cB)
            verdictA = int(dominant != cA); verdictB = int(dominant != cB)
            agree_verdict = int(verdictA == verdictB)
            m_agree_causal.append(agree_causal); m_agree_verdict.append(agree_verdict); m_k.append(len(peaks))
            out["cells"].append({"model": slug, "concept": c, "k": len(peaks),
                                 "causalA": cA, "causalB": cB, "dominant": dominant,
                                 "agree_causal": agree_causal, "agree_verdict": agree_verdict})
            print(f"  {c:14s} k={len(peaks)} causal A={cA} B={cB} {'OK' if agree_causal else 'DIFF'} "
                  f"verdict {'OK' if agree_verdict else 'DIFF'}", flush=True)
        if m_agree_causal:
            out["models"][slug] = {
                "n_cells": len(m_agree_causal),
                "causal_agree_rate": float(np.mean(m_agree_causal)),
                "verdict_agree_rate": float(np.mean(m_agree_verdict)),
                "chance_causal_agree": float(np.mean([1.0 / k for k in m_k])),
                "mean_k": float(np.mean(m_k))}
            print(f"  MODEL {slug}: causal-agree {np.mean(m_agree_causal):.2f} "
                  f"(chance {np.mean([1/k for k in m_k]):.2f}) verdict-agree {np.mean(m_agree_verdict):.2f}", flush=True)
        del model
    # pooled
    cells = out["cells"]
    if cells:
        out["pooled"] = {
            "n_cells": len(cells),
            "causal_agree_rate": float(np.mean([c["agree_causal"] for c in cells])),
            "verdict_agree_rate": float(np.mean([c["agree_verdict"] for c in cells])),
            "chance_causal_agree": float(np.mean([1.0 / c["k"] for c in cells]))}
        print(f"\nPOOLED ({len(cells)} cells): causal-agree {out['pooled']['causal_agree_rate']:.3f} "
              f"(chance {out['pooled']['chance_causal_agree']:.3f}) "
              f"verdict-agree {out['pooled']['verdict_agree_rate']:.3f}", flush=True)
    Path(__file__).parent.joinpath("results/causal_peak_stability.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

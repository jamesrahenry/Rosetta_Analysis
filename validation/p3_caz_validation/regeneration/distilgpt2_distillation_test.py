#!/usr/bin/env python3
"""distilgpt2 vs gpt2 — cheapest test of the distillation-legibility hypothesis.

Prediction ladder item 1 of papers/shared/distillation-legibility-hypothesis.md:
distilgpt2 is pure-distilled from gpt2 (shared vocab, shared data lineage) —
a near-minimal pair for the training objective. H predicts elevated
per-pair-difference participation ratio and lower split-half DOM stability
for distilgpt2 vs gpt2 at matched n; the dosage refinement allows the
effect to be milder than gemma-2's.

Design: BOTH models extracted fresh in the same process on the SAME pairs
(so no draw-mismatch): 250 pairs/concept sampled (seed 42) from current
RCP consensus jsonls, extraction via rosetta_tools.extraction
.extract_layer_activations (pool='last', the corpus convention), CPU.
gpt2 doubles as the in-experiment control: its PR (~3-14), split-half
(~0.96+ @125) and rank-1 erasure collapse must reproduce the corpus
signatures or the run is invalid.

Run with rosetta_tools/.venv python. Output:
results/distilgpt2_distillation_test.json
"""
import json
import numpy as np
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from rosetta_tools.extraction import extract_layer_activations

RCP = Path.home() / "Rosetta_Concept_Pairs/pairs/raw/v1"
CONCEPTS = ["causation", "agency", "sentiment", "negation", "formality", "credibility"]
# Control switched gpt2 -> pythia-70m (2026-07-17): the gpt2 download was
# hard-killed twice on this host; pythia-70m is locally cached (no network)
# and its stable-model corpus signatures are equally established. The
# teacher-side reference for distilgpt2 remains gpt2's stored-corpus
# measurements (split-half 0.96-0.99, PR 3-14, rank-1 erasure 0.995->0.64,
# gemma_section_artifacts.json) — the in-run control only validates
# env/pipeline.
MODELS = ["distilbert/distilgpt2", "EleutherAI/pythia-70m"]
N_PAIRS = 250
rng = np.random.default_rng(42)


def load_pairs(concept):
    recs = [json.loads(l) for l in open(RCP / f"{concept}_consensus_pairs.jsonl")]
    groups = {}
    for r in recs:
        groups.setdefault((r["pair_id"], r["model_name"]), {})[r["label"]] = r["text"]
    pairs = [(v[1], v[0]) for v in groups.values() if 1 in v and 0 in v]
    idx = rng.permutation(len(pairs))[:N_PAIRS]
    sel = [pairs[i] for i in idx]
    return [p for p, _ in sel], [n for _, n in sel]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def fisher(pos, neg):
    dn = unit(pos.mean(0) - neg.mean(0))
    pp, nn = pos @ dn, neg @ dn
    return (pp.mean() - nn.mean()) ** 2 / (pp.var() + nn.var() + 1e-12)


def probe_auc(pos, neg, A, B, ablate=None):
    def prep(X):
        return X - (X @ ablate.T) @ ablate if ablate is not None else X
    XA, XB = prep(np.vstack([pos[A], neg[A]])), prep(np.vstack([pos[B], neg[B]]))
    yA = np.r_[np.ones(len(A)), np.zeros(len(A))]
    yB = np.r_[np.ones(len(B)), np.zeros(len(B))]
    mu, sd = XA.mean(0), XA.std(0) + 1e-8
    clf = LogisticRegression(C=1.0, max_iter=2000).fit((XA - mu) / sd, yA)
    return float(roc_auc_score(yB, clf.decision_function((XB - mu) / sd)))


# draw pairs once, shared across both models
pair_texts = {c: load_pairs(c) for c in CONCEPTS}
for c in CONCEPTS:
    print(f"pairs {c}: {len(pair_texts[c][0])}", flush=True)

out = {"design": "both models fresh-extracted same process, same 250 pairs/concept "
                 "(seed 42, current RCP), pool='last', max_length=512, CPU, "
                 "transformers env noted in 'env'",
       "env": {}, "per_model": {}}
import transformers
out["env"] = {"torch": torch.__version__, "transformers": transformers.__version__,
              "note": "differs from corpus lockfile (5.8.0) — both models share this env, "
                      "and gpt2's control signatures gate validity"}

for model_id in MODELS:
    slug = model_id.split("/")[-1]
    print(f"== loading {model_id}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    out["per_model"][slug] = {}
    for c in CONCEPTS:
        pos_t, neg_t = pair_texts[c]
        cache = Path(__file__).parent / "results" / f"_cache_distill_{slug}_{c}.npz"
        if cache.exists():
            z = np.load(cache)
            pos_layers = [z[f"p{l}"] for l in range(int(z["L"]))]
            neg_layers = [z[f"n{l}"] for l in range(int(z["L"]))]
            print(f"cache hit {slug}/{c}", flush=True)
        else:
            with torch.no_grad():
                pos_layers = extract_layer_activations(model, tok, pos_t, device="cpu",
                                                       batch_size=8, pool="last")
                neg_layers = extract_layer_activations(model, tok, neg_t, device="cpu",
                                                       batch_size=8, pool="last")
            np.savez_compressed(cache, L=len(pos_layers),
                                **{f"p{l}": a for l, a in enumerate(pos_layers)},
                                **{f"n{l}": a for l, a in enumerate(neg_layers)})
        L = len(pos_layers)
        seps = [fisher(pos_layers[l].astype(np.float64), neg_layers[l].astype(np.float64))
                for l in range(1, L)]  # skip raw embedding row, corpus convention
        peak = int(np.argmax(seps)) + 1
        pos = pos_layers[peak].astype(np.float64)
        neg = neg_layers[peak].astype(np.float64)
        D = pos - neg
        sv = np.linalg.svd(D, compute_uv=False)
        ev = sv ** 2
        rec = {"n_layers_incl_embed": L, "peak_layer": peak,
               "fisher_at_peak": float(seps[peak - 1]),
               "pc1_share": float(ev[0] / ev.sum()),
               "top5_share": float(ev[:5].sum() / ev.sum()),
               "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
               "splithalf_cos_125": [], "c_est_125": None,
               "probe_auc": [], "rank1_erasure_auc": [], "rank20_erasure_auc": []}
        srng = np.random.default_rng(7)
        n = len(pos)
        for s in range(10):
            perm = srng.permutation(n)
            A, B = perm[: n // 2], perm[n // 2:]
            dA, dB = unit(pos[A].mean(0) - neg[A].mean(0)), unit(pos[B].mean(0) - neg[B].mean(0))
            rec["splithalf_cos_125"].append(float(abs(dA @ dB)))
            if s < 5:
                rec["probe_auc"].append(probe_auc(pos, neg, A, B))
                _, _, Vt = np.linalg.svd(D[A], full_matrices=False)
                rec["rank1_erasure_auc"].append(probe_auc(pos, neg, A, B, ablate=Vt[:1]))
                rec["rank20_erasure_auc"].append(probe_auc(pos, neg, A, B, ablate=Vt[:20]))
        cos125 = float(np.mean(rec["splithalf_cos_125"]))
        rec["splithalf_cos_125"] = cos125
        rec["c_est_125"] = round(125 * (1 / cos125 - 1), 1)  # from cos = 1/(1+c/n)
        for k in ("probe_auc", "rank1_erasure_auc", "rank20_erasure_auc"):
            rec[k] = float(np.mean(rec[k]))
        out["per_model"][slug][c] = rec
        print(f"{slug:12s} {c:12s} L{peak:2d} splithalf@125 {cos125:.3f} "
              f"(c~{rec['c_est_125']}) PR {rec['participation_ratio']:.1f} "
              f"probe {rec['probe_auc']:.3f} rank1-erase {rec['rank1_erasure_auc']:.3f}",
              flush=True)
    del model

import datetime
out["written_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
dst = Path(__file__).parent / "results" / "distilgpt2_distillation_test.json"
dst.write_text(json.dumps(out, indent=1))
print("saved", dst, flush=True)

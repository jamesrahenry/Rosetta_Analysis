#!/usr/bin/env python3
"""Follow-ups to gemma_subtlety_test.py.

F1  Subspace-denoised DOM: project each half's DOM onto that half's own
    top-k per-pair-difference subspace before comparing across halves.
    If gemma's cross-half agreement rises toward control levels, a
    k-dim-regularized estimator is a practical fix candidate.
F2  Probe-direction stability: do logistic probe weight vectors agree
    across halves better than raw DOM does?
F3  Corpus-wide: per-model split-half ordering reliability (C5 leg 2)
    vs supplementary §B mean CAZ score / Maj count / paradigm — tests
    the 'more distributed encoding -> less geometric readout stability'
    continuum across all 26 profiled models.
"""
import json, glob, re
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

SNAP = Path.home() / ".cache/huggingface/hub/datasets--james-ra-henry--Rosetta-Activations/snapshots"
CONCEPTS = ["causation", "agency", "deception", "moral_valence",
            "sentiment", "negation", "formality", "credibility"]
KS = [5, 10, 20, 40]
rng = np.random.default_rng(7)

def find_npy(model, concept):
    return glob.glob(str(SNAP / "*" / "paper_n250" / model / f"calibration_alllayer_{concept}.npy"))[0]

def fisher_sep(pos, neg):
    d = pos.mean(0) - neg.mean(0); dn = d / (np.linalg.norm(d) + 1e-12)
    pp, nn = pos @ dn, neg @ dn
    return (pp.mean() - nn.mean()) ** 2 / (pp.var() + nn.var() + 1e-12)

def unit(v): return v / (np.linalg.norm(v) + 1e-12)

print("== F1/F2: denoised-DOM and probe-weight cross-half agreement ==")
for model in ["google_gemma_2_2b", "openai_community_gpt2"]:
    for concept in CONCEPTS:
        arr = np.load(find_npy(model, concept)).astype(np.float64)
        L, n2, d = arr.shape; n = n2 // 2
        seps = [fisher_sep(arr[l, :n], arr[l, n:]) for l in range(L)]
        peak = int(np.argmax(seps))
        pos, neg = arr[peak, :n], arr[peak, n:]
        D = pos - neg
        raw, den, probw = [], {k: [] for k in KS}, []
        for s in range(10):
            perm = rng.permutation(n)
            A, B = perm[: n // 2], perm[n // 2:]
            domA, domB = unit(pos[A].mean(0) - neg[A].mean(0)), unit(pos[B].mean(0) - neg[B].mean(0))
            raw.append(abs(domA @ domB))
            for k in KS:
                _, _, VtA = np.linalg.svd(D[A], full_matrices=False)
                _, _, VtB = np.linalg.svd(D[B], full_matrices=False)
                dA = unit(VtA[:k].T @ (VtA[:k] @ domA))
                dB = unit(VtB[:k].T @ (VtB[:k] @ domB))
                den[k].append(abs(dA @ dB))
            ws = []
            for idx in (A, B):
                X = np.vstack([pos[idx], neg[idx]])
                y = np.r_[np.ones(len(idx)), np.zeros(len(idx))]
                mu, sd = X.mean(0), X.std(0) + 1e-8
                clf = LogisticRegression(C=1.0, max_iter=2000).fit((X - mu) / sd, y)
                ws.append(unit(clf.coef_[0]))
            probw.append(abs(ws[0] @ ws[1]))
        print(f"{model:26s} {concept:14s} raw {np.mean(raw):.3f}  "
              + "  ".join(f"den@k{k} {np.mean(v):.3f}" for k, v in den.items())
              + f"  probe-w {np.mean(probw):.3f}", flush=True)

print("\n== F3: corpus-wide — C5 split-half ordering reliability vs §B score profile ==")
c5 = json.load(open(Path(__file__).parent / "results" / "c5_splithalf_tau_ceiling.json"))
res = c5["results"]
first = next(iter(res.values()))
print("c5 per-model fields:", list(first.keys()) if isinstance(first, dict) else type(first))

# parse §B tables: | model | Layers | Hidden | CAZs | Score | Multimodal | Embed | Maj | %Gentle | Features | UFs |
import os as _os
supp = Path(_os.environ.get("P3_SUPPLEMENTARY", "supplementary.md")).read_text()  # set P3_SUPPLEMENTARY to caz-validation/supplementary.md
prof = {}
for mline in re.finditer(r"^\| ([A-Za-z0-9.\-]+) \| (\d+) \| (\d+) \| (\d+) \| ([\d.]+) \| (\d+)% \| (\d+) \| (\d+) \| (\d+)% \|", supp, re.M):
    name = mline.group(1)
    prof[name] = dict(score=float(mline.group(5)), multimodal=int(mline.group(6)),
                      maj=int(mline.group(8)), gentle=int(mline.group(9)))
print(f"parsed {len(prof)} §B rows: {sorted(prof)[:6]}...")
json.dump({"note": "see stdout"}, open(Path(__file__).parent / "gemma_followup_done.json", "w"))

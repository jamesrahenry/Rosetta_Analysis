#!/usr/bin/env python3
"""Rank-k difference-subspace ablation vs held-out decodability, gemma vs gpt2.

Run: 2026-07-17 02:2x UTC by claude:p3-review (results in
shared/round3_gpu/GEMMA_DISTRIBUTED_SIGNAL_TEST.md addendum 2).
Question: would a rank-k GEM payload (instead of the 1-D settled vector)
make zone-level erasure work on gemma-2? Estimate top-k per-pair-difference
subspace on half A ONLY, project both halves off it, train probe on A,
test AUC on held-out B. If AUC collapses, the estimated subspace covers
the transferable concept signal despite basis instability.
"""
import glob
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SNAP = Path.home() / ".cache/huggingface/hub/datasets--james-ra-henry--Rosetta-Activations/snapshots"
CONCEPTS = ["causation", "agency", "deception", "moral_valence", "sentiment", "negation"]
KS = [1, 5, 20, 40]
rng = np.random.default_rng(3)

def find_npy(m, c):
    return glob.glob(str(SNAP / "*" / "paper_n250" / m / f"calibration_alllayer_{c}.npy"))[0]

def fisher(pos, neg):
    d = pos.mean(0) - neg.mean(0); dn = d / (np.linalg.norm(d) + 1e-12)
    pp, nn = pos @ dn, neg @ dn
    return (pp.mean() - nn.mean()) ** 2 / (pp.var() + nn.var() + 1e-12)

def heldout_auc(pos, neg, A, B, ablate_basis=None):
    def prep(X):
        if ablate_basis is not None:
            X = X - (X @ ablate_basis.T) @ ablate_basis
        return X
    XA = prep(np.vstack([pos[A], neg[A]])); XB = prep(np.vstack([pos[B], neg[B]]))
    yA = np.r_[np.ones(len(A)), np.zeros(len(A))]; yB = np.r_[np.ones(len(B)), np.zeros(len(B))]
    mu, sd = XA.mean(0), XA.std(0) + 1e-8
    clf = LogisticRegression(C=1.0, max_iter=2000).fit((XA - mu) / sd, yA)
    return roc_auc_score(yB, clf.decision_function((XB - mu) / sd))

for model, tag in [("google_gemma_2_2b", "gemma"), ("openai_community_gpt2", "gpt2")]:
    for concept in CONCEPTS:
        arr = np.load(find_npy(model, concept)).astype(np.float64)
        L, n2, d = arr.shape; n = n2 // 2
        peak = int(np.argmax([fisher(arr[l, :n], arr[l, n:]) for l in range(L)]))
        pos, neg = arr[peak, :n], arr[peak, n:]
        D = pos - neg
        base_v, k_v = [], {k: [] for k in KS}
        for s in range(5):
            perm = rng.permutation(n); A, B = perm[: n // 2], perm[n // 2:]
            base_v.append(heldout_auc(pos, neg, A, B))
            _, _, Vt = np.linalg.svd(D[A], full_matrices=False)
            for k in KS:
                k_v[k].append(heldout_auc(pos, neg, A, B, ablate_basis=Vt[:k]))
        print(f"{tag:8s} {concept:14s} base {np.mean(base_v):.3f} "
              + " ".join(f"k={k}:{np.mean(v):.2f}" for k, v in k_v.items()), flush=True)

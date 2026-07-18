#!/usr/bin/env python3
"""Regenerate every §6.9-cited analysis into one archive JSON.

Covers the analyses behind preprint §6.9 (Gemma-2 distributed-encoding case
study) that were first run interactively on 2026-07-17:
  probe      — T1/T2/T5: split-half DOM cos, cross-half probe AUC, DOM-projection AUC
  spectrum   — T4: PC1/top-5 energy share, participation ratio
  denoise    — F1: subspace-projected DOM cross-half agreement (k=5..40)
  gemtrace   — trace-shape stability, peak wander (GEM addendum)
  rankk      — rank-k erasure vs held-out decodability (addendum 2)

Data: stored paper_n250 calibration_alllayer npys (local HF cache).
Output: results/gemma_section_artifacts.json — upload to HF
paper_n250/_gemma_readout_diagnostics/ (HF is the system of record for
anything the paper cites).
"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SNAP = Path.home() / ".cache/huggingface/hub/datasets--james-ra-henry--Rosetta-Activations/snapshots"
MODELS = ["google_gemma_2_2b", "openai_community_gpt2"]
CONCEPTS8 = ["causation", "agency", "deception", "moral_valence",
             "sentiment", "negation", "formality", "credibility"]
CONCEPTS6 = CONCEPTS8[:6]
CAP_KS = [1, 2, 5, 10, 20]
DEN_KS = [5, 10, 20, 40]
ABL_KS = [1, 5, 20, 40]


def find_npy(m, c):
    return glob.glob(str(SNAP / "*" / "paper_n250" / m / f"calibration_alllayer_{c}.npy"))[0]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def fisher(pos, neg, dn=None):
    if dn is None:
        dn = unit(pos.mean(0) - neg.mean(0))
    pp, nn = pos @ dn, neg @ dn
    return (pp.mean() - nn.mean()) ** 2 / (pp.var() + nn.var() + 1e-12)


def load_peak(m, c):
    arr = np.load(find_npy(m, c)).astype(np.float64)
    L, n2, d = arr.shape
    n = n2 // 2
    peak = int(np.argmax([fisher(arr[l, :n], arr[l, n:]) for l in range(L)]))
    return arr, L, n, peak


def probe_auc(pos, neg, A, B, ablate=None):
    def prep(X):
        return X - (X @ ablate.T) @ ablate if ablate is not None else X
    XA, XB = prep(np.vstack([pos[A], neg[A]])), prep(np.vstack([pos[B], neg[B]]))
    yA = np.r_[np.ones(len(A)), np.zeros(len(A))]
    yB = np.r_[np.ones(len(B)), np.zeros(len(B))]
    mu, sd = XA.mean(0), XA.std(0) + 1e-8
    clf = LogisticRegression(C=1.0, max_iter=2000).fit((XA - mu) / sd, yA)
    return float(roc_auc_score(yB, clf.decision_function((XB - mu) / sd)))


out = {"written_utc": None, "job": "gemma §6.9 artifact regeneration",
       "probe_spectrum": {}, "denoise": {}, "gemtrace": {}, "rankk": {}}

rng = np.random.default_rng(42)
for m in MODELS:
    out["probe_spectrum"][m] = {}
    for c in CONCEPTS8:
        arr, L, n, peak = load_peak(m, c)
        pos, neg = arr[peak, :n], arr[peak, n:]
        D = pos - neg
        sv = np.linalg.svd(D, compute_uv=False)
        ev = sv ** 2
        rec = {"peak_layer": peak, "pc1_share": float(ev[0] / ev.sum()),
               "top5_share": float(ev[:5].sum() / ev.sum()),
               "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
               "dom_cos": [], "probe_auc": [], "domproj_auc": [],
               "capture": {k: [] for k in CAP_KS}}
        for s in range(10):
            perm = rng.permutation(n)
            A, B = perm[: n // 2], perm[n // 2:]
            domA = unit(pos[A].mean(0) - neg[A].mean(0))
            domB = unit(pos[B].mean(0) - neg[B].mean(0))
            rec["dom_cos"].append(float(abs(domA @ domB)))
            rec["probe_auc"].append(probe_auc(pos, neg, A, B))
            XB = np.vstack([pos[B], neg[B]])
            yB = np.r_[np.ones(len(B)), np.zeros(len(B))]
            rec["domproj_auc"].append(float(roc_auc_score(yB, XB @ domA)))
            _, _, Vt = np.linalg.svd(D[B], full_matrices=False)
            pr = Vt[: max(CAP_KS)] @ domA
            for k in CAP_KS:
                rec["capture"][k].append(float(np.sqrt((pr[:k] ** 2).sum())))
        for key in ("dom_cos", "probe_auc", "domproj_auc"):
            rec[key] = {"mean": float(np.mean(rec[key])), "min": float(np.min(rec[key]))}
        rec["capture"] = {k: float(np.mean(v)) for k, v in rec["capture"].items()}
        out["probe_spectrum"][m][c] = rec
        print("probe/spectrum", m, c, flush=True)

rng = np.random.default_rng(7)
for m in MODELS:
    out["denoise"][m] = {}
    for c in CONCEPTS8:
        arr, L, n, peak = load_peak(m, c)
        pos, neg = arr[peak, :n], arr[peak, n:]
        D = pos - neg
        raw, den = [], {k: [] for k in DEN_KS}
        for s in range(10):
            perm = rng.permutation(n)
            A, B = perm[: n // 2], perm[n // 2:]
            domA = unit(pos[A].mean(0) - neg[A].mean(0))
            domB = unit(pos[B].mean(0) - neg[B].mean(0))
            raw.append(float(abs(domA @ domB)))
            _, _, VtA = np.linalg.svd(D[A], full_matrices=False)
            _, _, VtB = np.linalg.svd(D[B], full_matrices=False)
            for k in DEN_KS:
                dA = unit(VtA[:k].T @ (VtA[:k] @ domA))
                dB = unit(VtB[:k].T @ (VtB[:k] @ domB))
                den[k].append(float(abs(dA @ dB)))
        out["denoise"][m][c] = {"raw": float(np.mean(raw)),
                                **{f"k{k}": float(np.mean(v)) for k, v in den.items()}}
        print("denoise", m, c, flush=True)

rng = np.random.default_rng(11)
for m in MODELS:
    out["gemtrace"][m] = {}
    for c in CONCEPTS8:
        arr, L, n, peak = load_peak(m, c)
        same, shape_r, wander = [], [], []
        for s in range(6):
            perm = rng.permutation(n)
            A, B = perm[: n // 2], perm[n // 2:]
            domsA = np.array([unit(arr[l, :n][A].mean(0) - arr[l, n:][A].mean(0)) for l in range(L)])
            domsB = np.array([unit(arr[l, :n][B].mean(0) - arr[l, n:][B].mean(0)) for l in range(L)])
            same.append(float(np.mean([abs(domsA[l] @ domsB[l]) for l in range(L)])))
            rA = [abs(domsA[l] @ domsA[l + 1]) for l in range(L - 1)]
            rB = [abs(domsB[l] @ domsB[l + 1]) for l in range(L - 1)]
            shape_r.append(float(spearmanr(rA, rB)[0]))
            sepA = [fisher(arr[l, :n][A], arr[l, n:][A], domsA[l]) for l in range(L)]
            sepB = [fisher(arr[l, :n][B], arr[l, n:][B], domsB[l]) for l in range(L)]
            wander.append(abs(int(np.argmax(sepA)) - int(np.argmax(sepB))))
        out["gemtrace"][m][c] = {"same_layer_mean": float(np.mean(same)),
                                 "trace_shape_spearman": float(np.mean(shape_r)),
                                 "peak_wander_layers": float(np.mean(wander))}
        print("gemtrace", m, c, flush=True)

rng = np.random.default_rng(3)
for m in MODELS:
    out["rankk"][m] = {}
    for c in CONCEPTS6:
        arr, L, n, peak = load_peak(m, c)
        pos, neg = arr[peak, :n], arr[peak, n:]
        D = pos - neg
        base, ks = [], {k: [] for k in ABL_KS}
        for s in range(5):
            perm = rng.permutation(n)
            A, B = perm[: n // 2], perm[n // 2:]
            base.append(probe_auc(pos, neg, A, B))
            _, _, Vt = np.linalg.svd(D[A], full_matrices=False)
            for k in ABL_KS:
                ks[k].append(probe_auc(pos, neg, A, B, ablate=Vt[:k]))
        out["rankk"][m][c] = {"base": float(np.mean(base)),
                              **{f"k{k}": float(np.mean(v)) for k, v in ks.items()}}
        print("rankk", m, c, flush=True)

import datetime  # noqa: E402
out["written_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
dst = Path(__file__).parent / "results" / "gemma_section_artifacts.json"
dst.write_text(json.dumps(out, indent=1))
print("saved", dst)

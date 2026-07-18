#!/usr/bin/env python3
"""Is gemma-2's DOM instability 'distributed signal' or 'weak signal'?

James's hypothesis (2026-07-17): gemma works fine as an LM, so the concept
information must be present — the 1-D DOM readout may be capturing only a
slice of a more distributed representation ("gemma is just more subtle").

Discriminating tests, gemma-2-2b vs gpt2 control, stored paper_n250
calibration activations (CPU only, no new extraction):

  T1  Split-half DOM cosine (baseline — reproduces the known instability).
  T2  Probe transfer: logistic trained on half A, AUC on held-out half B.
      Distributed-signal predicts gemma AUC ~ control AUC (info present);
      weak-signal predicts low AUC.
  T3  Subspace capture: how much of half-A's DOM lies inside half-B's
      top-k PCA subspace of per-pair difference vectors, k=1..20.
      Rotational degeneracy predicts low at k=1, high by k~5-10.
  T4  Spectrum shape: energy share of PC1 / top-5 of the per-pair
      difference covariance. Distributed predicts a flatter spectrum.
  T5  DOM-projection AUC across halves (is the unstable *direction* still
      a good classifier even when it rotates?).

Split by PAIR so pos/neg of a pair never straddle halves. Exfiltration
excluded (label defect, t50c6362).
"""
import json
import glob
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SNAP = Path.home() / ".cache/huggingface/hub/datasets--james-ra-henry--Rosetta-Activations/snapshots"
MODELS = ["google_gemma_2_2b", "openai_community_gpt2"]
CONCEPTS = ["causation", "agency", "deception", "moral_valence",
            "sentiment", "negation", "formality", "credibility"]
N_SPLITS = 10
KS = [1, 2, 5, 10, 20]
rng = np.random.default_rng(42)


def find_npy(model, concept):
    hits = glob.glob(str(SNAP / "*" / "paper_n250" / model / f"calibration_alllayer_{concept}.npy"))
    if not hits:
        raise FileNotFoundError(f"{model}/{concept}")
    return hits[0]


def fisher_sep(pos, neg):
    d = pos.mean(0) - neg.mean(0)
    dn = d / (np.linalg.norm(d) + 1e-12)
    pp, nn = pos @ dn, neg @ dn
    num = (pp.mean() - nn.mean()) ** 2
    den = pp.var() + nn.var() + 1e-12
    return num / den


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def topk_basis(D, k):
    """Top-k right singular vectors of the (centered-free) per-pair diff matrix."""
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    return Vt[:k]


results = {}
for model in MODELS:
    results[model] = {}
    for concept in CONCEPTS:
        arr = np.load(find_npy(model, concept)).astype(np.float64)  # (L, 2n, d)
        L, n2, d = arr.shape
        n = n2 // 2  # rows 0..n-1 positive, n..2n-1 negative (recorded layout)

        # peak layer by full-sample Fisher separation
        seps = [fisher_sep(arr[l, :n], arr[l, n:]) for l in range(L)]
        peak = int(np.argmax(seps))
        pos, neg = arr[peak, :n], arr[peak, n:]
        D_full = pos - neg  # per-pair difference vectors (n, d)

        # T4 spectrum on full sample
        sv = np.linalg.svd(D_full, compute_uv=False)
        ev = sv ** 2
        pc1_share = float(ev[0] / ev.sum())
        top5_share = float(ev[:5].sum() / ev.sum())
        pr = float(ev.sum() ** 2 / (ev ** 2).sum())  # participation ratio

        m = dict(peak_layer=peak, n_layers=L, dim=d,
                 fisher_at_peak=float(seps[peak - 1]),
                 pc1_share=pc1_share, top5_share=top5_share,
                 participation_ratio=pr,
                 dom_cos=[], probe_auc=[], probe_auc_self=[],
                 domproj_auc=[],
                 capture={k: [] for k in KS}, overlap5=[])

        for s in range(N_SPLITS):
            perm = rng.permutation(n)
            A, B = perm[: n // 2], perm[n // 2:]
            domA = unit(pos[A].mean(0) - neg[A].mean(0))
            domB = unit(pos[B].mean(0) - neg[B].mean(0))
            m["dom_cos"].append(float(abs(domA @ domB)))

            # T3: capture of domA in half-B's top-k difference subspace
            basisB = topk_basis(D_full[B], max(KS))
            proj = basisB @ domA
            for k in KS:
                m["capture"][k].append(float(np.sqrt((proj[:k] ** 2).sum())))
            basisA = topk_basis(D_full[A], 5)
            ov = np.linalg.svd(basisA @ basisB[:5].T, compute_uv=False)
            m["overlap5"].append(float((ov ** 2).mean()))

            # T2: probe transfer (standardize by half-A stats)
            XA = np.vstack([pos[A], neg[A]])
            yA = np.r_[np.ones(len(A)), np.zeros(len(A))]
            XB = np.vstack([pos[B], neg[B]])
            yB = np.r_[np.ones(len(B)), np.zeros(len(B))]
            mu, sd = XA.mean(0), XA.std(0) + 1e-8
            clf = LogisticRegression(C=1.0, max_iter=2000)
            clf.fit((XA - mu) / sd, yA)
            m["probe_auc"].append(float(roc_auc_score(yB, clf.decision_function((XB - mu) / sd))))
            m["probe_auc_self"].append(float(roc_auc_score(yA, clf.decision_function((XA - mu) / sd))))

            # T5: DOM-projection AUC across halves
            m["domproj_auc"].append(float(roc_auc_score(yB, XB @ domA)))

        for key in ["dom_cos", "probe_auc", "probe_auc_self", "domproj_auc", "overlap5"]:
            m[key] = dict(mean=float(np.mean(m[key])), min=float(np.min(m[key])))
        m["capture"] = {k: float(np.mean(v)) for k, v in m["capture"].items()}
        results[model][concept] = m
        print(f"{model:26s} {concept:14s} L{peak:2d} "
              f"domcos {m['dom_cos']['mean']:.3f}  probeAUC {m['probe_auc']['mean']:.3f}  "
              f"domprojAUC {m['domproj_auc']['mean']:.3f}  "
              f"cap k1/5/20 {m['capture'][1]:.2f}/{m['capture'][5]:.2f}/{m['capture'][20]:.2f}  "
              f"PC1 {pc1_share:.2f} top5 {top5_share:.2f} PR {pr:.1f}", flush=True)

out = Path(__file__).parent / "gemma_subtlety_results.json"
out.write_text(json.dumps(results, indent=1))
print("saved", out)

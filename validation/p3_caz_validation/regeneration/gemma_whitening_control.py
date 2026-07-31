#!/usr/bin/env python3
"""Whitening / covariance-aware estimator control for §6.9 (added 2026-07-19
in response to adversarial review).

The skeptic's alternative: Gemma-2's low DOM split-half cosine is not
"distributed encoding" but a covariance-blind estimator failing on
anisotropic activations (the difference-of-means ignores within-class
covariance; a logistic probe implicitly whitens it out). Decisive test:
recompute split-half cosine of the concept direction in a WHITENED basis
(Ledoit-Wolf pooled within-class covariance, symmetric whitening). If it
jumps to control level, the "distributed" reading is an anisotropy
artifact. Also report activation anisotropy directly (leading-eigenvalue
share of the centered activation covariance) — the alternative predicts
Gemma > controls.

Result (8 concepts, peak layer, gemma-2-2b vs gpt2 control):
  raw DOM split-half:      gemma 0.748   gpt2 0.982
  whitened DOM split-half: gemma 0.872   gpt2 0.969
  anisotropy (PC1 share):  gemma 0.118   gpt2 0.187   (gemma LESS anisotropic)
Whitening recovers part of the gap (0.75->0.87) but not to control (0.97);
and Gemma is not more anisotropic than the stable control. The residual is
not an anisotropy/estimator artifact. Whitening here shares one transform
across both halves (optimistic), so 0.872 is an upper bound on a fully
per-half whitened estimate.
"""
import glob
import numpy as np
from pathlib import Path
from sklearn.covariance import LedoitWolf

SNAP = Path.home() / ".cache/huggingface/hub/datasets--james-ra-henry--Rosetta-Activations/snapshots"
CON = ["causation", "agency", "deception", "moral_valence", "sentiment", "negation", "formality", "credibility"]
rng = np.random.default_rng(0)


def find(m, c):
    return glob.glob(str(SNAP / "*" / "paper_n250" / m / f"calibration_alllayer_{c}.npy"))[0]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def fisher(pos, neg):
    dn = unit(pos.mean(0) - neg.mean(0))
    pp, nn = pos @ dn, neg @ dn
    return (pp.mean() - nn.mean()) ** 2 / (pp.var() + nn.var() + 1e-12)


def run():
    out = {}
    for m, tag in [("google_gemma_2_2b", "gemma"), ("openai_community_gpt2", "gpt2")]:
        R, Wd, An = [], [], []
        for c in CON:
            arr = np.load(find(m, c)).astype(np.float64)
            L, n2, d = arr.shape
            n = n2 // 2
            pk = int(np.argmax([fisher(arr[l, :n], arr[l, n:]) for l in range(L)]))
            pos, neg = arr[pk, :n], arr[pk, n:]
            cov = LedoitWolf().fit(np.vstack([pos - pos.mean(0), neg - neg.mean(0)])).covariance_
            evals, evecs = np.linalg.eigh(cov)
            Wt = evecs @ np.diag(np.clip(evals, 1e-8, None) ** -0.5) @ evecs.T
            posw, negw = pos @ Wt, neg @ Wt
            Xc = np.vstack([pos, neg]) - np.vstack([pos, neg]).mean(0)
            ev = np.linalg.svd(Xc, compute_uv=False) ** 2
            rc, wc = [], []
            for _ in range(8):
                perm = rng.permutation(n)
                A, B = perm[: n // 2], perm[n // 2:]
                rc.append(abs(unit(pos[A].mean(0) - neg[A].mean(0)) @ unit(pos[B].mean(0) - neg[B].mean(0))))
                wc.append(abs(unit(posw[A].mean(0) - negw[A].mean(0)) @ unit(posw[B].mean(0) - negw[B].mean(0))))
            R.append(np.mean(rc)); Wd.append(np.mean(wc)); An.append(float(ev[0] / ev.sum()))
        out[tag] = {"raw_dom": float(np.mean(R)), "whitened_dom": float(np.mean(Wd)),
                    "anisotropy_pc1": float(np.mean(An))}
        print(f"{tag}: raw {np.mean(R):.3f}  whitened {np.mean(Wd):.3f}  anisotropy(PC1) {np.mean(An):.3f}")
    import json
    Path(__file__).parent.joinpath("results/gemma_whitening_control.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    run()

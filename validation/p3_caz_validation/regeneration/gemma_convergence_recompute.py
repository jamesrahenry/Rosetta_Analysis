#!/usr/bin/env python3
"""Archive-grade recompute of the gemma-2-2b convergence curves (§6.9).

The original convergence run (GEMMA_INSTABILITY_NOTE.md, 2026-07-16) used
fresh full-train-pool extraction on the H200; its raw data
(convergence_gemma.json) lived on the torn-down host and was never
uploaded. This recomputes the same quantity from the stored rcp_v1
peak-layer calibration npys on HF (n≈660–1,440 pairs per concept), which
IS archived — making the paper's cited c / n₉₅ numbers reproducible from
public data.

Method: per concept, at nested half-sizes n, draw two DISJOINT n-pair
subsets (5 resamples per size), compute DOM on each, record |cos|.
Least-squares fit of cos(n) = 1/(1+c/n); n₉₅ = 19c; best achievable at
the full pool = 1/(1+c/N_pool). gpt2 × causation is the control.

Caveat carried from the source data: rcp_v1 predates the exfiltration
label fix (RCP 1a61a76; supplementary §D) — exfiltration's labels are
internally consistent within the pool (so its split-half c is measurable)
but the class definition is the defective one; its row is annotated.
"""
import json
import numpy as np
from huggingface_hub import hf_hub_download

REPO = "james-ra-henry/Rosetta-Activations"
CONCEPTS = ["agency", "authorization", "causation", "certainty", "credibility",
            "deception", "exfiltration", "formality", "moral_valence", "negation",
            "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
            "threat_severity", "urgency"]
RESAMPLES = 5
rng = np.random.default_rng(19)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def splithalf_curve(pos, neg):
    n_pool = len(pos)
    sizes = [s for s in (50, 100, 200, 400, 700) if 2 * s <= n_pool]
    if 2 * (n_pool // 2) > 2 * sizes[-1]:
        sizes.append(n_pool // 2)
    means = []
    for n in sizes:
        cs = []
        for _ in range(RESAMPLES):
            idx = rng.permutation(n_pool)
            A, B = idx[:n], idx[n:2 * n]
            dA = unit(pos[A].mean(0) - neg[A].mean(0))
            dB = unit(pos[B].mean(0) - neg[B].mean(0))
            cs.append(abs(float(dA @ dB)))
        means.append(float(np.mean(cs)))
    # least-squares fit of cos(n) = 1/(1+c/n) over a c grid then refine
    ns = np.array(sizes, dtype=float)
    ys = np.array(means)
    grid = np.linspace(0.5, 400, 4000)
    sse = [(float(np.sum((1 / (1 + c / ns) - ys) ** 2)), c) for c in grid]
    c = min(sse)[1]
    return sizes, means, float(c)


out = {"job": "gemma convergence recompute from rcp_v1 (archived data)",
       "method": "disjoint split-half DOM |cos| at nested n, 5 resamples/size, "
                 "stored rcp_v1 peak-layer calibration npys, fit cos=1/(1+c/n)",
       "per_concept": {}, "control": {}}

for concept in CONCEPTS:
    p = hf_hub_download(REPO, f"rcp_v1/google_gemma_2_2b/calibration_{concept}.npy",
                        repo_type="dataset")
    arr = np.load(p).astype(np.float64)
    n_pool = arr.shape[0] // 2
    pos, neg = arr[:n_pool], arr[n_pool:]
    sizes, means, c = splithalf_curve(pos, neg)
    rec = {"pool_pairs": int(n_pool), "sizes": sizes, "splithalf_means": means,
           "c": round(c, 1), "n95": int(round(19 * c)),
           "best_achievable_at_pool": round(1 / (1 + c / n_pool), 3),
           "clears_095_in_pool": bool(1 / (1 + c / n_pool) >= 0.95)}
    if concept == "exfiltration":
        rec["caveat"] = ("rcp_v1 predates the label fix (supp. §D): labels are "
                         "internally consistent but the defective assignment — "
                         "c is measurable, concept identity is not clean")
    out["per_concept"][concept] = rec
    print(f"{concept:16s} pool={n_pool:5d} c={c:6.1f} n95={19*c:6.0f} "
          f"best={rec['best_achievable_at_pool']:.3f}", flush=True)

p = hf_hub_download(REPO, "rcp_v1/openai_community_gpt2/calibration_causation.npy",
                    repo_type="dataset")
arr = np.load(p).astype(np.float64)
n_pool = arr.shape[0] // 2
sizes, means, c = splithalf_curve(arr[:n_pool], arr[n_pool:])
out["control"]["gpt2_causation"] = {"pool_pairs": int(n_pool), "sizes": sizes,
                                    "splithalf_means": means, "c": round(c, 1),
                                    "n95": int(round(19 * c))}
print(f"CONTROL gpt2 causation: c={c:.1f}", flush=True)

cs = [v["c"] for k, v in out["per_concept"].items()]
out["summary"] = {"median_c": float(np.median(cs)),
                  "n_clearing_095_in_pool": sum(v["clears_095_in_pool"] for v in out["per_concept"].values()),
                  "n_concepts": len(cs)}
print("median c:", out["summary"]["median_c"],
      "| clear 0.95 in pool:", out["summary"]["n_clearing_095_in_pool"], "/17", flush=True)

import datetime
out["written_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
from pathlib import Path
dst = Path(__file__).parent / "results" / "gemma_convergence_recompute.json"
dst.write_text(json.dumps(out, indent=1))
print("saved", dst)

#!/usr/bin/env python3
"""Round-4 review control: is the §6.4 "causal peak" (argmin self-retained)
estimation noise (the reviewer's circularity worry), or a stable quantity —
and if stable, WHAT determines it?

Answered from the stored ablation_multimodal artifacts (all 28 models, 360
multimodal cells) — no recompute needed — cross-checked against 24
directly-recomputed split-half cells (causal_peak_stability.py, killed early
but the completed pythia-70m/160m cells all showed 100% split-half
assignment agreement, causal peak = deepest region every time).

Findings:
  - causal peak == deepest region:        90.3% (325/360)
  - dominant (argmax score) == deepest:   49.7% (179/360)
  - divergence rate (causal != dominant): 50.3% (181/360)  [matches paper]
  - among divergent cells, causal deeper: 93.9%             [matches §6.4's 94.4%]
  - Spearman(depth, -self_retained):      +0.631, p~1e-103

Reading: the causal-peak assignment is NOT noise — it is ~90% determined by
DEPTH and reproduces across split-halves. So the reviewer's "estimation
noise / circularity" objection is refuted at the noise level. BUT the
mechanism is depth: self-retained falls monotonically with depth (deeper
regions, ablated and measured one layer downstream, lose more separation),
so argmin-self-retained is almost always the deepest region. The 50.3%
divergence is therefore ~equal to (1 - P[dominant is deepest]) = 50.3%: it
is essentially "the highest-Fisher-score CAZ is not the deepest CAZ." That
is a genuine, stable relationship, but it is depth-structured, and whether
it reflects the paper's "where the model commits" thesis or a
depth-of-intervention confound is the open question the headline must be
scoped to (§6.1's C1 depth-control regression covers the peak-vs-nonCAZ
ENRICHMENT, not the divergence's causal-peak identification).
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path.home() / "rosetta_data" / "paper_n250"
CON = ["credibility", "negation", "causation", "temporal_order", "sentiment", "certainty",
       "moral_valence", "specificity", "plurality", "agency", "formality", "threat_severity",
       "authorization", "urgency", "sarcasm", "deception", "exfiltration"]


def main():
    models = [d.name for d in ROOT.iterdir() if d.is_dir() and not d.name.startswith("_")]
    causal_deepest = dominant_deepest = div = div_causal_deeper = tot = 0
    dep, negsr = [], []
    for m in models:
        for c in CON:
            f = ROOT / m / f"ablation_multimodal_{c}.json"
            if not f.exists():
                continue
            cz = json.load(open(f)).get("cazs", [])
            if len(cz) < 2:
                continue
            sr = np.array([z["self_retained_pct"] for z in cz])
            depth = np.array([z["depth_pct"] for z in cz])
            score = np.array([z["caz_score"] for z in cz])
            causal, dominant, deep = int(np.argmin(sr)), int(np.argmax(score)), int(np.argmax(depth))
            tot += 1
            causal_deepest += causal == deep
            dominant_deepest += dominant == deep
            dep.extend(depth); negsr.extend(-sr)
            if causal != dominant:
                div += 1
                div_causal_deeper += depth[dominant] < depth[causal]
    rho, p = spearmanr(dep, negsr)
    out = {"n_cells": tot,
           "causal_is_deepest": causal_deepest / tot,
           "dominant_is_deepest": dominant_deepest / tot,
           "divergence_rate": div / tot,
           "among_divergent_causal_deeper": div_causal_deeper / div,
           "spearman_depth_vs_neg_selfretained": rho, "p": p}
    print(json.dumps(out, indent=1))
    Path(__file__).parent.joinpath("results/causal_peak_depth_analysis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

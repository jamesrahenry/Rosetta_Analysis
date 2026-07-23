#!/usr/bin/env python3
"""Recompute the CAZ score-category distribution (Table 2), the §8.3 cohort
major/gentle breakdown, and the §8.7 four-formula sensitivity on the corrected
paper_n250 artifacts. tc4fd04e item (6).

The published Table 2 summed to 1,036 (pre-exfiltration-correction); the corrected
detector yields 1,045 regions, so the whole score distribution shifted. This
regenerates all of it from find_caz_regions_scored over the corrected caz artifacts.

Four formulas (caz.py L964-967): A = prominence_norm; B = A·√(width/L);
C = A·coherence_boost; D = A·coherence_boost·√(width/L) (= caz_score, verified to
1e-6). Major = score > 0.5. §8.3 filtered-major = major AND non-embedding (peak≠L0)
AND peak depth > 5%. CPU-only. Written: 2026-07-23 UTC.
"""
import json
import sys
from pathlib import Path

import numpy as np

for _p in (str(Path.home() / "rosetta_tools"),
           str(Path.home() / "Source" / "Rosetta_Program" / "rosetta_tools"),
           str(Path.home() / "Games2" / "Eigan" / "Rosetta_Program" / "rosetta_tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from rosetta_tools.caz import LayerMetrics, find_caz_regions_scored  # noqa: E402

DATA = Path.home() / "rosetta_data" / "paper_n250"
CONCEPTS_17 = ['agency', 'authorization', 'causation', 'certainty', 'credibility',
               'deception', 'exfiltration', 'formality', 'moral_valence', 'negation',
               'plurality', 'sarcasm', 'sentiment', 'specificity', 'temporal_order',
               'threat_severity', 'urgency']
COH = {
    "MHA": ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
            "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
            "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b", "openai-community/gpt2",
            "openai-community/gpt2-medium", "openai-community/gpt2-large",
            "openai-community/gpt2-xl", "facebook/opt-125m", "facebook/opt-350m",
            "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b", "microsoft/phi-2"],
    "GQA": ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B",
            "Qwen/Qwen2.5-14B", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"],
    "Mistral": ["mistralai/Mistral-7B-v0.3"],
    "Gemma": ["google/gemma-2-2b", "google/gemma-2-9b"],
}
PAR = {m: ("mha" if k == "MHA" else "alternating" if k == "Gemma" else "gqa")
       for k, ms in COH.items() for m in ms}


def regions(m, c):
    d = json.loads((DATA / m.replace("/", "_").replace("-", "_") / f"caz_{c}.json").read_text())
    md = d["layer_data"]["metrics"]
    lm = [LayerMetrics(x["layer"], x["separation_fisher"], x["coherence"], float(x["velocity"]))
          for x in md]
    gms = float(np.mean([x["separation_fisher"] for x in md]))
    gmc = float(np.mean([x["coherence"] for x in md]))
    prof = find_caz_regions_scored(lm, attention_paradigm=PAR[m])
    out = []
    for r in prof.regions:
        pn = r.prominence / gms if gms > 0 else 0.0
        cb = 1.0 + r.peak_coherence / gmc if gmc > 0 else 1.0
        wf = np.sqrt(r.width / len(md)) if md else 0.0
        out.append({"D": float(r.caz_score), "A": pn, "B": pn * wf, "C": pn * cb,
                    "peak": int(r.peak), "depth": float(r.depth_pct)})
    return out


def main():
    allr, coh = [], {k: [] for k in COH}
    permodel = {k: [] for k in COH}
    for k, models in COH.items():
        for m in models:
            cnt = 0
            for c in CONCEPTS_17:
                rs = regions(m, c)
                allr += rs; coh[k] += rs
                cnt += sum(r["D"] > 0.5 for r in rs)
            permodel[k].append(cnt)
    D = np.array([r["D"] for r in allr]); N = len(D)
    cats = {"major": int((D > 0.5).sum()), "strong": int(((D > 0.2) & (D <= 0.5)).sum()),
            "moderate": int(((D > 0.05) & (D <= 0.2)).sum()), "gentle": int((D <= 0.05).sum())}
    filt = sum(1 for r in allr if r["D"] > 0.5 and r["peak"] != 0 and r["depth"] > 5)
    out = {"n_regions": N,
           "table2": {c: {"count": n, "pct": round(100 * n / N, 1)} for c, n in cats.items()},
           "filtered_major_total": filt,
           "cohorts": {}, "formula_sensitivity": {}}
    for k in COH:
        d = np.array([r["D"] for r in coh[k]]); pm = permodel[k]
        out["cohorts"][k] = {"n": len(d), "major_pct": round(100 * (d > 0.5).mean(), 1),
                             "gentle_pct": round(100 * (d < 0.05).mean(), 1),
                             "mean_score": round(float(d.mean()), 2),
                             "per_model_major_mean": round(float(np.mean(pm)), 1),
                             "per_model_range": [min(pm), max(pm)]}
    for f in "ABCD":
        s = np.array([r[f] for r in allr])
        mha = np.array([r[f] for r in coh["MHA"]])
        gqa = np.array([r[f] for r in coh["GQA"]])
        out["formula_sensitivity"][f] = {
            "overall_major_pct": round(100 * (s > 0.5).mean(), 1),
            "mha_gqa_gap_pp": round(100 * ((mha > 0.5).mean() - (gqa > 0.5).mean()), 1)}
    print(json.dumps(out, indent=2))
    Path(__file__).parent.joinpath("score_distribution_recompute_results.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

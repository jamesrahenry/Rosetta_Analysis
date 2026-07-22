#!/usr/bin/env python3
"""Reconstruct Table 11 (§6.1 global-sweep peak-vs-non-CAZ enrichment) from the
paper_n250 global-sweep artifacts — the runner is retired, so the aggregation
recipe is reconstructed locally rather than grepped from ~/rosetta_analysis.

tc4fd04e item (1). Published Table 11 (preprint §6.1, L499-509):
  CAZ peak layers                        0.517   (n=476 = 28x17, one per model-concept)
  Non-CAZ layers (>3 from any peak)       0.140   (n=6,960)
  Ratio                                   3.60x   (MEAN of per-model ratios)
  Mann-Whitney U                     2,799,064    p = 9.47e-141
  (pooled-mean ratio 0.517/0.140 = 3.69 — the "3.60 vs 3.69" flag is just
   mean-of-ratios vs ratio-of-means; both correct.)

Recipe reconstructed here:
  peak sample   = global_sep_reduction at the file's caz_peak layer (one per file)
  non-CAZ       = global_sep_reduction at layers whose distance to EVERY detected
                  region peak (find_caz_regions_scored on the caz artifact) is > 3
  per-model ratio = (mean peak_val for that model) / (mean non_caz_val for that model)

Writes table11_reconstruct_results.json and prints the match against the
published quartet. Written: 2026-07-22 UTC.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

# rosetta_tools import (portable: GPU host ~/rosetta_tools first, then dev trees)
for _p in (str(Path.home() / "rosetta_tools"),
           str(Path.home() / "Source" / "Rosetta_Program" / "rosetta_tools"),
           str(Path.home() / "Games2" / "Eigan" / "Rosetta_Program" / "rosetta_tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Inlined from round3_gpu/common.py (kept self-contained so this canonical copy
# has no round3_gpu path dependency). CONCEPTS_17 = P3's corrected concept set.
CONCEPTS_17 = ['agency', 'authorization', 'causation', 'certainty', 'credibility',
               'deception', 'exfiltration', 'formality', 'moral_valence', 'negation',
               'plurality', 'sarcasm', 'sentiment', 'specificity', 'temporal_order',
               'threat_severity', 'urgency']


def slugify(model_id):
    """HF model id -> artifact directory slug (established convention)."""
    return model_id.replace("/", "_").replace("-", "_")
from rosetta_tools.caz import LayerMetrics, find_caz_regions_scored  # noqa: E402

DATA = Path.home() / "rosetta_data" / "paper_n250"

# 28-model roster + paradigm (pinned inline, matching c6_sensitivity.py)
BASE_28 = [
    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",
    "openai-community/gpt2", "openai-community/gpt2-medium",
    "openai-community/gpt2-large", "openai-community/gpt2-xl",
    "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b",
    "facebook/opt-2.7b", "facebook/opt-6.7b",
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B",
    "google/gemma-2-2b", "google/gemma-2-9b",
    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B",
    "mistralai/Mistral-7B-v0.3", "microsoft/phi-2",
]


def paradigm(m):
    low = m.lower()
    if any(f in low for f in ("pythia", "gpt2", "opt", "phi")):
        return "mha"
    if any(f in low for f in ("qwen", "llama", "mistral")):
        return "gqa"
    return "alternating"


def region_peaks(model, concept):
    p = DATA / slugify(model) / f"caz_{concept}.json"
    mr = json.loads(p.read_text())["layer_data"]["metrics"]
    lm = [LayerMetrics(m["layer"], m["separation_fisher"], m["coherence"],
                       float(m["velocity"])) for m in mr]
    prof = find_caz_regions_scored(lm, attention_paradigm=paradigm(model))
    return [int(r.peak) for r in prof.regions]


def main():
    peak_vals, non_vals = [], []
    per_model = {}
    for model in BASE_28:
        pv, nv = [], []
        for concept in CONCEPTS_17:
            gf = DATA / slugify(model) / f"ablation_global_sweep_{concept}.json"
            if not gf.exists():
                continue
            g = json.loads(gf.read_text())
            red = {L["layer"]: L["global_sep_reduction"] for L in g["layers"]}
            cpk = g["caz_peak"]
            if cpk in red:
                pv.append(red[cpk])
            peaks = region_peaks(model, concept)
            for L, r in red.items():
                if all(abs(L - pk) > 3 for pk in peaks):
                    nv.append(r)
        peak_vals += pv
        non_vals += nv
        if pv and nv:
            per_model[model] = np.mean(pv) / np.mean(nv) if np.mean(nv) > 0 else float("nan")

    peak_vals = np.array(peak_vals)
    non_vals = np.array(non_vals)
    U, p = mannwhitneyu(peak_vals, non_vals, alternative="greater")
    ratios = {m: r for m, r in per_model.items()}
    mean_of_ratios = float(np.mean(list(ratios.values())))
    rng = (min(ratios, key=ratios.get), max(ratios, key=ratios.get))

    published = {"peak": 0.517, "non": 0.140, "n_peak": 476, "n_non": 6960,
                 "U": 2799064, "ratio_meanofratios": 3.60}
    got = {"peak": round(float(peak_vals.mean()), 3),
           "non": round(float(non_vals.mean()), 3),
           "n_peak": int(peak_vals.size), "n_non": int(non_vals.size),
           "U": int(U), "ratio_meanofratios": round(mean_of_ratios, 2),
           "ratio_ofmeans": round(float(peak_vals.mean() / non_vals.mean()), 2)}

    print("PUBLISHED:", published)
    print("RECONSTRUCTED:", got)
    match = (got["n_peak"] == 476 and got["n_non"] == 6960
             and abs(got["peak"] - 0.517) < 0.005 and abs(got["non"] - 0.140) < 0.005)
    print(f"\nrecipe match (n + means): {'YES' if match else 'NO'}")
    print(f"U reconstructed={got['U']:,} published=2,799,064 "
          f"{'EXACT' if got['U'] == 2799064 else 'DIFF'}")
    print(f"per-model ratio range: {ratios[rng[0]]:.2f} ({rng[0].split('/')[-1]}) "
          f"to {ratios[rng[1]]:.2f} ({rng[1].split('/')[-1]})")
    print(f"mean-of-ratios={got['ratio_meanofratios']} (pub 3.60), "
          f"ratio-of-means={got['ratio_ofmeans']} (pub 3.69)")

    Path(__file__).parent.joinpath("table11_reconstruct_results.json").write_text(
        json.dumps({"published": published, "reconstructed": got,
                    "recipe_match": bool(match),
                    "U_exact": got["U"] == 2799064,
                    "per_model_ratios": {m: round(r, 3) for m, r in ratios.items()}},
                   indent=2))


if __name__ == "__main__":
    main()

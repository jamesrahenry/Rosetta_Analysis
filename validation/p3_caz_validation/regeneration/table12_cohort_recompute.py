#!/usr/bin/env python3
"""Recompute Table 12 (§6.4 cohort ablation + patching) from the corrected
paper_n250 artifacts. tc4fd04e item (2).

Ablation column = mean handoff `final_sep_reduction` (ablation_gem_<concept>.json).
Patching column = concept_score_recovery at the caz_peak layer (patch_<concept>.json),
reported raw and trimmed (>1.0 removed); overshoot rate = fraction >1.0.

Finding it fixed: the published MHA patching count was n=323 (= 19x17), but there
are exactly 306 (18 MHA x 17) patch files, one per model-concept — the same as the
ablation column. The phantom 17 overshoot-heavy measurements inflated the raw mean
(0.699 vs the reproducible 0.672); the trimmed mean is stable (0.533 vs 0.530) and
every ablation value reproduces to +-0.001. CPU-only. Written: 2026-07-23 UTC.
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

DATA = Path.home() / "rosetta_data" / "paper_n250"
CONCEPTS_17 = ['agency', 'authorization', 'causation', 'certainty', 'credibility',
               'deception', 'exfiltration', 'formality', 'moral_valence', 'negation',
               'plurality', 'sarcasm', 'sentiment', 'specificity', 'temporal_order',
               'threat_severity', 'urgency']
COHORTS = {
    "MHA": ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
            "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
            "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b", "openai-community/gpt2",
            "openai-community/gpt2-medium", "openai-community/gpt2-large",
            "openai-community/gpt2-xl", "facebook/opt-125m", "facebook/opt-350m",
            "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b", "microsoft/phi-2"],
    "GQA": ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B",
            "Qwen/Qwen2.5-14B", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B",
            "mistralai/Mistral-7B-v0.3"],
    "Gemma": ["google/gemma-2-2b", "google/gemma-2-9b"],
}
PUBLISHED = {"MHA": (0.658, 0.672, 0.533), "GQA": (0.626, 0.770, 0.693),
             "Gemma": (0.367, 0.685, 0.648)}  # ablation, patch-raw, patch-trim (corrected)


def slugify(m):
    return m.replace("/", "_").replace("-", "_")


def main():
    out = {}
    for coh, models in COHORTS.items():
        abl, pat = [], []
        for m in models:
            for c in CONCEPTS_17:
                af = DATA / slugify(m) / f"ablation_gem_{c}.json"
                if af.exists():
                    h = json.loads(af.read_text()).get("handoff", {})
                    if "final_sep_reduction" in h:
                        abl.append(h["final_sep_reduction"])
                pf = DATA / slugify(m) / f"patch_{c}.json"
                if pf.exists():
                    d = json.loads(pf.read_text())
                    row = [L for L in d["layers"] if L["layer"] == d["caz_peak"]]
                    if row:
                        pat.append(row[0]["concept_score_recovery"])
        abl, pat = np.array(abl), np.array(pat)
        tr = pat[pat <= 1.0]
        out[coh] = {"ablation_mean": round(float(abl.mean()), 3), "n_ablation": len(abl),
                    "patch_raw_mean": round(float(pat.mean()), 3), "n_patch": len(pat),
                    "patch_trimmed_mean": round(float(tr.mean()), 3),
                    "overshoot_rate_pct": round(100 * (1 - len(tr) / len(pat)))}
        print(f"{coh}: abl {out[coh]['ablation_mean']} (n={len(abl)}) | "
              f"patch raw {out[coh]['patch_raw_mean']} (n={len(pat)}) "
              f"trim {out[coh]['patch_trimmed_mean']} overshoot {out[coh]['overshoot_rate_pct']}%")
    Path(__file__).parent.joinpath("table12_cohort_recompute_results.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

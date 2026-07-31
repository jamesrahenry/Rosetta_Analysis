#!/usr/bin/env python3
"""B4 (P3 round-1 review): regenerate the P1 optimal-ablation-layer split.

James's ruling: "the CAZ region" = the DOMINANT SCORED region (max caz_score)
from find_caz_regions_scored — the same detector that produces every CAZ count.

For each of the 28 base models x 17 concepts:
  - dominant region = max-caz_score region from find_caz_regions_scored
  - optimal = optimal_ablation_layer from ablation_global_sweep_<concept>.json
  - classify: within [start,end] / post (>end) / pre (<start)
  - peak-coincidence: optimal == dominant region peak
Reports overall + per-cohort within%, and the peak-coincidence rate (bears on
S5.1's "the peak itself is rarely optimal"). CPU-only, frozen JSON.
Written 2026-07-27 UTC by claude:p3-corpus-review.
"""
import json, sys
from pathlib import Path

for _p in (str(Path.home() / "rosetta_tools"),
           str(Path.home() / "Games2" / "Eigan" / "Rosetta_Program" / "rosetta_tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from rosetta_tools.caz import LayerMetrics, find_caz_regions_scored  # noqa

DATA = Path.home() / "rosetta_data" / "paper_n250"
CONCEPTS = ['agency', 'authorization', 'causation', 'certainty', 'credibility',
            'deception', 'exfiltration', 'formality', 'moral_valence', 'negation',
            'plurality', 'sarcasm', 'sentiment', 'specificity', 'temporal_order',
            'threat_severity', 'urgency']
FAMILIES = [
    ("mha", ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
             "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
             "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b"]),
    ("mha", ["openai-community/gpt2", "openai-community/gpt2-medium",
             "openai-community/gpt2-large", "openai-community/gpt2-xl"]),
    ("mha", ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b",
             "facebook/opt-2.7b", "facebook/opt-6.7b"]),
    ("mha", ["microsoft/phi-2"]),
    ("gqa", ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
             "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B"]),
    ("gqa", ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]),
    ("gqa", ["mistralai/Mistral-7B-v0.3"]),
    ("alt", ["google/gemma-2-2b", "google/gemma-2-9b"]),
]
# cohort label used by the paper: MHA / GQA / Gemma
def cohort(par): return {"mha": "MHA", "gqa": "GQA", "alt": "Gemma"}[par]
PARADIGM = {m: par for par, ms in FAMILIES for m in ms}
MODELS = [m for _, ms in FAMILIES for m in ms]
def slug(m): return m.replace("/", "_").replace("-", "_")

def dominant_region(model, concept):
    p = DATA / slug(model) / f"caz_{concept}.json"
    if not p.exists():
        return None, None
    mr = json.loads(p.read_text())["layer_data"]["metrics"]
    lm = [LayerMetrics(x["layer"], x["separation_fisher"], x["coherence"],
                       float(x.get("velocity", 0.0))) for x in mr]
    regs = find_caz_regions_scored(lm, attention_paradigm=PARADIGM[model]).regions
    if not regs:
        return None, len(mr)
    dom = max(regs, key=lambda r: r.caz_score)
    return dom, len(mr)

def optimal_layer(model, concept):
    p = DATA / slug(model) / f"ablation_global_sweep_{concept}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("optimal_ablation_layer")

overall = {"within": 0, "post": 0, "pre": 0}
byco = {c: {"within": 0, "post": 0, "pre": 0} for c in ("MHA", "GQA", "Gemma")}
peak_coincide = 0
n = 0
missing = []
idx_warn = []
for m in MODELS:
    co = cohort(PARADIGM[m])
    for c in CONCEPTS:
        dom, nlayers = dominant_region(m, c)
        opt = optimal_layer(m, c)
        if dom is None or opt is None:
            missing.append((m, c, "noregion" if opt is not None else "nosweep"))
            continue
        if nlayers is not None and not (0 <= opt < nlayers):
            idx_warn.append((m, c, opt, nlayers))
        if opt < dom.start:
            cls = "pre"
        elif opt > dom.end:
            cls = "post"
        else:
            cls = "within"
        overall[cls] += 1
        byco[co][cls] += 1
        if opt == dom.peak:
            peak_coincide += 1
        n += 1

def pct(d):
    t = sum(d.values())
    return {k: round(100 * v / t, 1) for k, v in d.items()} if t else {}

print(f"N classified = {n}  (missing {len(missing)}, index-warnings {len(idx_warn)})")
print(f"OVERALL (dominant scored region): {pct(overall)}  counts={overall}")
for co in ("MHA", "GQA", "Gemma"):
    t = sum(byco[co].values())
    print(f"  {co} (n={t}): within {pct(byco[co]).get('within')}%  post {pct(byco[co]).get('post')}%  pre {pct(byco[co]).get('pre')}%")
print(f"Peak-coincidence (optimal == dominant region peak): {peak_coincide}/{n} = {round(100*peak_coincide/n,1)}%")
if idx_warn:
    print("INDEX WARNINGS (opt outside [0,nlayers)):", idx_warn[:6])
if missing:
    print("MISSING (first 6):", missing[:6])

out = {"definition": "dominant scored region (max caz_score, find_caz_regions_scored defaults)",
       "n": n, "overall_pct": pct(overall), "overall_counts": overall,
       "by_cohort_pct": {co: pct(byco[co]) for co in byco}, "by_cohort_counts": byco,
       "peak_coincidence_pct": round(100 * peak_coincide / n, 1) if n else None,
       "peak_coincidence_count": peak_coincide,
       "n_missing": len(missing), "n_index_warnings": len(idx_warn)}
Path(__file__).with_name("b4_optimal_layer_split_results.json").write_text(json.dumps(out, indent=2))
print("wrote b4_optimal_layer_split_results.json")

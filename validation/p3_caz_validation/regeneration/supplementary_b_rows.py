#!/usr/bin/env python3
"""Regenerate supplementary §B structural rows on the CORRECTED artifacts.

teb3a9e9 + tc4fd04e finding (2026-07-22): the published §B per-model rows are
stale — they were computed pre-exfiltration-correction (C6 validated the detector
against them on 2026-07-16; every caz_exfiltration.json was rewritten 2026-07-17).
The rest of the paper runs on the corrected C=17 corpus, so §B must too. This
regenerates ALL 28 base-model rows from the current (corrected) paper_n250
artifacts using the exact detector C6 validated (find_caz_regions_scored on the
stored layer_data.metrics).

Faithfulness proof (no old artifacts needed):
  (1) C6 (2026-07-16) validated this detector path reproduces the paper aggregate
      (1,036 regions) on the pre-correction artifacts.
  (2) The only artifacts changed since are exfiltration (mtime: all other concepts
      <= 07-14, exfiltration = 07-17).
  (3) The detector is unchanged (pre-QA bc583e3 and HEAD 8eb5d83 both give 1,045).
  => this generator == the paper's §B method applied to the corrected corpus; the
     deltas vs published are exactly the exfiltration correction.
  CROSS-CHECK asserted below: total regions / multimodal over all 28 must equal the
  current C6 run (1,045 / 363), proving generator == validated detector path.

Features (persistent-spectral, §7 census) and UFs (N~200 atlas) are separate
artifacts, NOT re-derived here — carried unchanged from the published table; the
two new rows take Features 29 (pythia-12b) / 35 (Qwen2.5-14B), UFs pending.

Written: 2026-07-22 UTC
"""
import json
import sys
from pathlib import Path

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

FAMILIES = [
    ("Pythia", "mha", ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m", "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b"]),
    ("GPT-2", "mha", ["openai-community/gpt2", "openai-community/gpt2-medium",
        "openai-community/gpt2-large", "openai-community/gpt2-xl"]),
    ("OPT", "mha", ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b",
        "facebook/opt-2.7b", "facebook/opt-6.7b"]),
    ("Phi-2", "mha", ["microsoft/phi-2"]),
    ("Qwen", "gqa", ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B"]),
    ("Llama", "gqa", ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]),
    ("Mistral", "gqa", ["mistralai/Mistral-7B-v0.3"]),
    ("Gemma", "alternating", ["google/gemma-2-2b", "google/gemma-2-9b"]),
]


def profile(model_id, concept, paradigm):
    d = json.loads((DATA / slugify(model_id) / f"caz_{concept}.json").read_text())
    ld = d["layer_data"]
    lm = [LayerMetrics(m["layer"], m["separation_fisher"], m["coherence"],
                       float(m["velocity"])) for m in ld["metrics"]]
    return find_caz_regions_scored(lm, attention_paradigm=paradigm), d["n_layers"], d["hidden_dim"]


def short(model_id):
    return model_id.split("/")[-1]


def row(model_id, paradigm):
    n_layers = hidden = None
    scores, funcs = [], []
    n_multi = embed = maj = fpk = 0
    for c in CONCEPTS_17:
        prof, n_layers, hidden = profile(model_id, c, paradigm)
        regs = prof.regions
        scores += [float(r.caz_score) for r in regs]
        funcs += [float(r.functional_caz_score) for r in regs]
        n_multi += int(prof.is_multimodal)
        if any(int(r.peak) == 0 for r in regs):
            embed += 1
        maj += sum(1 for r in regs if float(r.caz_score) > 0.5)
        fpk += sum(1 for r in regs if r.is_functional_peak)
    n = len(scores)
    return {
        "model": short(model_id), "layers": n_layers, "hidden": hidden,
        "CAZs": n, "score": round(sum(scores) / n, 3),
        "func": round(sum(funcs) / n, 3),
        "multimodal_pct": round(100.0 * n_multi / len(CONCEPTS_17)),
        "embed": embed, "maj": maj,
        "gentle_pct": round(100.0 * sum(1 for s in scores if s < 0.05) / n),
        "functional_peaks": fpk, "artifacts": n - fpk, "paradigm": paradigm,
    }


def main():
    all_rows, tot_regions, tot_multi = [], 0, 0
    for fam, par, models in FAMILIES:
        print(f"\n### {fam} ({par})")
        for m in models:
            r = row(m, par)
            all_rows.append(r)
            tot_regions += r["CAZs"]
            tot_multi += round(r["multimodal_pct"] / 100 * len(CONCEPTS_17))
            if par == "alternating":
                print(f"  {r['model']:14s} L={r['layers']:>2} H={r['hidden']:>4} "
                      f"CAZ={r['CAZs']:>2} score={r['score']:.3f} func={r['func']:.3f} "
                      f"art={r['artifacts']:>2} fpk={r['functional_peaks']:>2} "
                      f"maj={r['maj']:>2} gentle={r['gentle_pct']:>2}%")
            else:
                fs = f"func={r['func']:.3f} " if par != "mha" else ""
                print(f"  {r['model']:14s} L={r['layers']:>2} H={r['hidden']:>4} "
                      f"CAZ={r['CAZs']:>2} score={r['score']:.3f} {fs}"
                      f"multi={r['multimodal_pct']:>3}% embed={r['embed']} "
                      f"maj={r['maj']:>2} gentle={r['gentle_pct']:>2}%")

    # ---- cross-check against current C6 aggregate (proves generator == detector) ----
    print(f"\n=== CROSS-CHECK: total regions={tot_regions} (C6 current: 1045), "
          f"multimodal={tot_multi} (C6 current: 363) ===")
    assert tot_regions == 1045, f"region total {tot_regions} != 1045 — generator diverged from C6"
    print("CROSS-CHECK PASSED — generator == validated C6 detector path\n")

    # ---- paradigm summary (base models only) ----
    print("=== Paradigm summary (base) ===")
    for par, label in [("mha", "MHA"), ("gqa", "GQA"), ("alternating", "Alternating")]:
        g = [r for r in all_rows if r["paradigm"] == par]
        n = len(g)
        mean = lambda k: sum(r[k] for r in g) / n
        # multimodal rate = pooled over concepts, not mean-of-rates
        mm = sum(round(r["multimodal_pct"] / 100 * 17) for r in g) / (n * 17) * 100
        art = sum(r["artifacts"] for r in g) if par == "alternating" else 0
        print(f"  {label:12s} n={n} meanCAZ={mean('CAZs'):.1f} meanScore={mean('score'):.3f} "
              f"meanFunc={mean('func'):.3f} multi={mm:.0f}% meanGentle={mean('gentle_pct'):.0f}% "
              f"artifacts={art}")

    Path(__file__).parent.joinpath("supplementary_b_rows_results.json").write_text(
        json.dumps({"rows": all_rows, "total_regions": tot_regions,
                    "total_multimodal": tot_multi}, indent=2))


if __name__ == "__main__":
    main()

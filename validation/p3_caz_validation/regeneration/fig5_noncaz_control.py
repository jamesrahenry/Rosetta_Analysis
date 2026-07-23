#!/usr/bin/env python3
"""Regenerate Figure 5 (fig_noncaz_control.png) on the CORRECTED artifacts,
using the SAME recipe as Table 11 so figure and table agree.

tc4fd04e item (7). The previous plotter (gem/ablate_noncaz_control.py) used a
NAIVE local-maxima peak set (any layer > both neighbours) for the non-CAZ
exclusion, which differs from Table 11's detector-region peaks — so the figure
never matched the table it illustrates. This regenerates both the per-model
data (caz_vs_noncaz_control.json) and the bar chart on Table 11's recipe:

  peak sample per model×concept = global_sep_reduction at the file's caz_peak
  non-CAZ layers = layers >3 from ANY find_caz_regions_scored region peak
  per-model bar heights = mean peak / mean non-CAZ reduction; label = their ratio

Matches the committed Table 11 pooled figures (peak 0.503 / non 0.140 / 3.59x)
and per-model range (2.17x pythia-12b .. 12.56x pythia-160m). CPU-only.

Writes: papers/caz-validation/figures/{fig_noncaz_control.png,caz_vs_noncaz_control.json}
(path overridable via P3_FIGURES_DIR). Written: 2026-07-23 UTC.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _p in (str(Path.home() / "rosetta_tools"),
           str(Path.home() / "Source" / "Rosetta_Program" / "rosetta_tools"),
           str(Path.home() / "Games2" / "Eigan" / "Rosetta_Program" / "rosetta_tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from rosetta_tools.caz import LayerMetrics, find_caz_regions_scored  # noqa: E402

DATA = Path.home() / "rosetta_data" / "paper_n250"
FIG_DIR = Path(os.environ.get(
    "P3_FIGURES_DIR",
    str(Path.home() / "Games2" / "Eigan" / "Rosetta_Program"
        / "papers" / "caz-validation" / "figures")))

CONCEPTS_17 = ['agency', 'authorization', 'causation', 'certainty', 'credibility',
               'deception', 'exfiltration', 'formality', 'moral_valence', 'negation',
               'plurality', 'sarcasm', 'sentiment', 'specificity', 'temporal_order',
               'threat_severity', 'urgency']
# family -> (paradigm, colour, models in scale order)
FAMILIES = [
    ("Pythia", "mha", "#4C72B0", ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m", "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b"]),
    ("GPT-2", "mha", "#5B9BD5", ["openai-community/gpt2", "openai-community/gpt2-medium",
        "openai-community/gpt2-large", "openai-community/gpt2-xl"]),
    ("OPT", "mha", "#2E4E7E", ["facebook/opt-125m", "facebook/opt-350m",
        "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b"]),
    ("Phi-2", "mha", "#8FB8DE", ["microsoft/phi-2"]),
    ("Qwen", "gqa", "#C44E52", ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B"]),
    ("Llama", "gqa", "#E17C7C", ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]),
    ("Mistral", "gqa", "#A83236", ["mistralai/Mistral-7B-v0.3"]),
    ("Gemma", "alt", "#55A868", ["google/gemma-2-2b", "google/gemma-2-9b"]),
]
PARADIGM = {m: par for _, par, _, ms in FAMILIES for m in ms}


def slugify(m):
    return m.replace("/", "_").replace("-", "_")


def region_peaks(m, c):
    mr = json.loads((DATA / slugify(m) / f"caz_{c}.json").read_text())["layer_data"]["metrics"]
    lm = [LayerMetrics(x["layer"], x["separation_fisher"], x["coherence"], float(x["velocity"]))
          for x in mr]
    return [int(r.peak) for r in find_caz_regions_scored(lm, attention_paradigm=PARADIGM[m]).regions]


def per_model(m):
    pv, nv = [], []
    for c in CONCEPTS_17:
        gf = DATA / slugify(m) / f"ablation_global_sweep_{c}.json"
        if not gf.exists():
            continue
        g = json.loads(gf.read_text())
        red = {L["layer"]: L["global_sep_reduction"] for L in g["layers"]}
        if g["caz_peak"] in red:
            pv.append(red[g["caz_peak"]])
        pks = region_peaks(m, c)
        nv += [r for L, r in red.items() if all(abs(L - pk) > 3 for pk in pks)]
    pm, nm = float(np.mean(pv)), float(np.mean(nv))
    return {"peak_mean": pm, "noncaz_mean": nm, "ratio": pm / max(nm, 1e-9)}


def main():
    records, labels, colors = [], [], []
    for fam, par, col, models in FAMILIES:
        for m in models:
            d = per_model(m)
            d.update(model=m.split("/")[-1], family=fam, paradigm=par)
            records.append(d); labels.append(m.split("/")[-1]); colors.append(col)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "caz_vs_noncaz_control.json").write_text(json.dumps(records, indent=2))

    x = np.arange(len(records)); w = 0.4
    pk = [r["peak_mean"] for r in records]; nc = [r["noncaz_mean"] for r in records]
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x - w / 2, pk, w, color=colors, label="CAZ peak layer")
    ax.bar(x + w / 2, nc, w, color="#BBBBBB", label="Non-CAZ layers (mean)")
    for xi, p, n in zip(x, pk, nc):
        ax.text(xi - w / 2, p + 0.008, f"{p / max(n, 1e-9):.1f}×",
                ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("Separation reduction\n(1 − retained / baseline)")
    ax.set_title("CAZ peak vs. non-CAZ layer ablation specificity "
                 "(28 base models × 17 concepts; Table 11 recipe)")
    # family separators + labels
    b = 0
    for fam, _, col, models in FAMILIES:
        if b > 0:
            ax.axvline(b - 0.5, color="#DDDDDD", lw=0.8, zorder=0)
        ax.text(b + (len(models) - 1) / 2, ax.get_ylim()[1] * 0.97, fam,
                ha="center", va="top", fontsize=8, color=col, fontweight="bold")
        b += len(models)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(FIG_DIR / "fig_noncaz_control.png"), dpi=150,
                bbox_inches="tight", facecolor="white")

    pooled_pk = np.mean([r["peak_mean"] for r in records])  # unweighted per-model
    ratios = [r["ratio"] for r in records]
    lo = min(records, key=lambda r: r["ratio"]); hi = max(records, key=lambda r: r["ratio"])
    print(f"wrote fig_noncaz_control.png + caz_vs_noncaz_control.json ({len(records)} models)")
    print(f"per-model ratio range: {lo['ratio']:.2f}x ({lo['model']}) .. "
          f"{hi['ratio']:.2f}x ({hi['model']}); mean-of-ratios {np.mean(ratios):.2f}x")


if __name__ == "__main__":
    main()

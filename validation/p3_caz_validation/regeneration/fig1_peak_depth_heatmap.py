#!/usr/bin/env python3
"""fig1_peak_depth_heatmap.py — Figure 1 (CAZ dominant peak-depth heatmap).

Self-contained regeneration of the paper's Figure 1, kept in-repo so the
figure is reproducible from the checkout (the original generator lived only
in rosetta_analysis/viz and was absent here — round-3 review finding).

Rows = 17 concepts ordered by mean dominant-peak depth (shallowest top).
Columns = 28 base models grouped by architecture family, sorted by scale.
Colour = green (shallow / early layers) → red (deep / late layers).

Data: scripts/results/depth_pivot.csv (per-model per-concept dominant-peak
depth %, produced by ordering_tau_recompute.py — corrected exfiltration).

Usage: python fig1_peak_depth_heatmap.py [--out <png>]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DEFAULT_OUT = HERE.parent / "figures" / "fig_peak_depth_heatmap.png"
PIVOT = HERE / "results" / "depth_pivot.csv"

# family, param-count (M) — matches rosetta_analysis/viz/viz_style.FAMILY_MAP
FAMILY_MAP = {
    "EleutherAI/pythia-70m": ("Pythia", 70), "EleutherAI/pythia-160m": ("Pythia", 160),
    "EleutherAI/pythia-410m": ("Pythia", 410), "EleutherAI/pythia-1b": ("Pythia", 1000),
    "EleutherAI/pythia-1.4b": ("Pythia", 1400), "EleutherAI/pythia-2.8b": ("Pythia", 2800),
    "EleutherAI/pythia-6.9b": ("Pythia", 6900), "EleutherAI/pythia-12b": ("Pythia", 12000),
    "facebook/opt-125m": ("OPT", 125), "facebook/opt-350m": ("OPT", 350),
    "facebook/opt-1.3b": ("OPT", 1300), "facebook/opt-2.7b": ("OPT", 2700),
    "facebook/opt-6.7b": ("OPT", 6700),
    "openai-community/gpt2": ("GPT-2", 117), "openai-community/gpt2-medium": ("GPT-2", 345),
    "openai-community/gpt2-large": ("GPT-2", 800), "openai-community/gpt2-xl": ("GPT-2", 1500),
    "Qwen/Qwen2.5-0.5B": ("Qwen", 500), "Qwen/Qwen2.5-1.5B": ("Qwen", 1500),
    "Qwen/Qwen2.5-3B": ("Qwen", 3000), "Qwen/Qwen2.5-7B": ("Qwen", 7000),
    "Qwen/Qwen2.5-14B": ("Qwen", 14000),
    "meta-llama/Llama-3.2-1B": ("Llama", 1000), "meta-llama/Llama-3.2-3B": ("Llama", 3000),
    "mistralai/Mistral-7B-v0.3": ("Mistral", 7000),
    "google/gemma-2-2b": ("Gemma", 2000), "google/gemma-2-9b": ("Gemma", 9000),
    "microsoft/phi-2": ("Phi", 2700),
}
FAMILY_ORDER = ["Pythia", "OPT", "GPT-2", "Qwen", "Llama", "Mistral", "Gemma", "Phi"]

CONCEPT_LABELS = {
    "temporal_order": "Temporal order", "moral_valence": "Moral valence",
    "threat_severity": "Threat severity",
}


def model_label(mid: str) -> str:
    fam, _ = FAMILY_MAP.get(mid, ("", 0))
    short = mid.split("/")[-1]
    for pre in ("pythia-", "opt-", "gpt2-", "Qwen2.5-", "Llama-3.2-", "Mistral-", "gemma-2-", "phi-"):
        if short.lower().startswith(pre.lower()):
            return f"{fam}-{short[len(pre):].upper()}"
    return short


def sort_models(mids):
    return sorted(mids, key=lambda m: (FAMILY_ORDER.index(FAMILY_MAP[m][0]), FAMILY_MAP[m][1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    df = pd.read_csv(PIVOT, index_col=0)
    models = sort_models([m for m in df.index if m in FAMILY_MAP])
    concepts = list(df.mean(axis=0).sort_values().index)  # shallowest -> deepest
    M = df.loc[models, concepts].to_numpy().T  # rows=concepts, cols=models

    fig, ax = plt.subplots(figsize=(0.42 * len(models) + 3, 0.40 * len(concepts) + 1.6))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)

    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels([CONCEPT_LABELS.get(c, c.capitalize()) for c in concepts], fontsize=8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([model_label(m) for m in models], rotation=90, fontsize=6.5)

    # family separators + labels
    fams = [FAMILY_MAP[m][0] for m in models]
    for i in range(1, len(models)):
        if fams[i] != fams[i - 1]:
            ax.axvline(i - 0.5, color="white", lw=1.5)
    for fam in FAMILY_ORDER:
        idxs = [i for i, f in enumerate(fams) if f == fam]
        if idxs:
            ax.text(np.mean(idxs), -0.9, fam, ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Dominant peak depth (% of model depth)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("CAZ dominant peak depth — 28 base models × 17 concepts", fontsize=10, pad=18)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}  ({M.shape[0]} concepts × {M.shape[1]} models)")
    print("deepest concept (bottom row):", concepts[-1], f"{df[concepts[-1]].mean():.1f}%")


if __name__ == "__main__":
    main()

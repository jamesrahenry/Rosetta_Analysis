#!/usr/bin/env python3
"""Generate Figure 1 for the GEM paper: EEC distribution across 272 concept×model pairs.

Entry-exit cosine (EEC) = cosine similarity between the dominant concept direction
at CAZ entry and at CAZ exit. Low values indicate substantial rotation during assembly.

Data source: rosetta_data/models/*/gem_*.json  (primary CAZ node per concept per model)

Output: papers/gem/figures/fig1_eec_distribution.{pdf,png}

Usage:
    python viz_gem_eec_fig1.py
    python viz_gem_eec_fig1.py --out-dir /tmp/figs

Written: 2026-05-06 UTC
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from rosetta_tools.paths import ROSETTA_MODELS

PAPERS_OUT = Path.home() / "Source" / "Rosetta_Program" / "papers" / "gem" / "figures"

CONCEPT_LABELS = {
    "agency": "Agency", "authorization": "Authorization", "causation": "Causation",
    "certainty": "Certainty", "credibility": "Credibility", "deception": "Deception",
    "exfiltration": "Exfiltration", "formality": "Formality",
    "moral_valence": "Moral valence", "negation": "Negation",
    "plurality": "Plurality", "sarcasm": "Sarcasm", "sentiment": "Sentiment",
    "specificity": "Specificity", "temporal_order": "Temporal order",
    "threat_severity": "Threat severity", "urgency": "Urgency",
}
PRIMARY_MODELS = {
    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b", "EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b", "openai-community/gpt2", "facebook/opt-6.7b",
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B",
    "mistralai/Mistral-7B-v0.3", "google/gemma-2-9b",
}


def load_eec_data():
    """Return (eecs_all, per_concept) for the 16-model primary corpus.

    Prefers pre-aggregated N=250 result files if present; falls back to
    individual gem_*.json model files (N=200).
    """
    results_dir = Path.home() / "rosetta_data" / "results"
    agg_files = [
        results_dir / "gem_eec_corpus.json",
    ]
    if all(f.exists() for f in agg_files):
        eecs_all = []
        per_concept = {}
        for agg_path in agg_files:
            data = json.load(open(agg_path))
            for model_data in data.values():
                for concept, eec in model_data.get("per_concept", {}).items():
                    eecs_all.append(eec)
                    per_concept.setdefault(concept, []).append(eec)
        return np.array(eecs_all), {k: np.array(v) for k, v in per_concept.items()}

    eecs_all = []
    per_concept = {}

    for model_dir in sorted(ROSETTA_MODELS.iterdir()):
        for gf in sorted(model_dir.glob("gem_*.json")):
            d = json.load(open(gf))
            nodes = d.get("nodes", [])
            if not nodes:
                continue
            model_id = nodes[0].get("model_id", "")
            if model_id not in PRIMARY_MODELS:
                continue
            primary = [n for n in nodes if n.get("phase") == "primary"]
            if not primary:
                primary = [nodes[0]]
            node = primary[0]
            eec = node.get("entry_exit_cosine")
            concept = node.get("concept", gf.stem.replace("gem_", ""))
            if eec is not None:
                eecs_all.append(eec)
                per_concept.setdefault(concept, []).append(eec)

    return np.array(eecs_all), {k: np.array(v) for k, v in per_concept.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(PAPERS_OUT))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eecs, per_concept = load_eec_data()
    mean_eec   = eecs.mean()
    median_eec = np.median(eecs)

    concept_means = sorted(
        [(c, v.mean()) for c, v in per_concept.items()],
        key=lambda x: -x[1],
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.subplots_adjust(wspace=0.38)

    # --- Panel A: histogram ---
    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    ax.hist(eecs, bins=bins, color="#4878CF", edgecolor="white", linewidth=0.5, alpha=0.88)
    ax.axvline(mean_eec, color="#C44E52", linewidth=1.8, linestyle="--",
               label=f"Mean = {mean_eec:.2f}")
    ax.axvline(median_eec, color="#DD8452", linewidth=1.4, linestyle=":",
               label=f"Median = {median_eec:.2f}")
    ax.set_xlabel("Entry–exit cosine (EEC)", fontsize=11)
    ax.set_ylabel("Number of concept × model pairs", fontsize=10)
    ax.set_title("A  EEC distribution (272 pairs)", fontsize=11, loc="left", fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_xlim(0, 1)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    pct_below_half = (eecs < 0.5).mean() * 100
    pct_near_ortho = (eecs < 0.1).mean() * 100
    ax.text(0.97, 0.97,
            f"{pct_below_half:.0f}% < 0.5\n{pct_near_ortho:.0f}% < 0.1",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # --- Panel B: per-concept mean EEC (horizontal bar) ---
    ax = axes[1]
    concepts = [c for c, _ in concept_means]
    means    = [m for _, m in concept_means]
    sems     = [per_concept[c].std() / np.sqrt(len(per_concept[c])) for c in concepts]
    labels   = [CONCEPT_LABELS.get(c, c) for c in concepts]
    y = np.arange(len(concepts))
    ax.barh(y, means, xerr=sems, color="#4878CF", alpha=0.82,
            error_kw=dict(ecolor="#2a4a8a", capsize=2.5, linewidth=1))
    ax.axvline(mean_eec, color="#C44E52", linewidth=1.4, linestyle="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Mean EEC ± SEM", fontsize=11)
    ax.set_title("B  Per-concept mean EEC (16 models each)", fontsize=11,
                 loc="left", fontweight="bold")
    ax.set_xlim(0, 0.7)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

    fig.tight_layout()

    for fmt in ("pdf", "png"):
        out = out_dir / f"fig1_eec_distribution.{fmt}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close(fig)
    print(f"N={len(eecs)}  mean={mean_eec:.3f}  median={median_eec:.3f}  "
          f"<0.5={pct_below_half:.0f}%  <0.1={pct_near_ortho:.0f}%")


if __name__ == "__main__":
    main()

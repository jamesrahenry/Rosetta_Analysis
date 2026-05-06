#!/usr/bin/env python3
"""Generate Figure 1 for the CAZ Framework paper.

Two-panel comparison:
  Left  — single-region detector: sentiment in Pythia-1.4b
           Shows the naive approach: one allocation zone at the global S(l) peak.
  Right — scored multi-region detector: credibility in Qwen2.5-0.5B
           Shows six CAZes across depth, colour-coded by score category.

Data source: rosetta_data/models/{slug}/caz_{concept}.json
Extraction:  rosetta_analysis/extraction/extract.py --model <model_id>

Output: fig1_caz_detection.pdf + fig1_caz_detection.png

Written: 2026-05-06 20:00 UTC
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# Locate rosetta_tools (installed or sibling checkout)
try:
    from rosetta_tools.caz import find_caz_regions_scored, LayerMetrics
    from rosetta_tools.paths import ROSETTA_DATA_ROOT
except ImportError:
    _rt = Path(__file__).resolve().parents[4] / "rosetta_tools"
    sys.path.insert(0, str(_rt))
    from rosetta_tools.rosetta_tools.caz import find_caz_regions_scored, LayerMetrics
    ROSETTA_DATA_ROOT = Path.home() / "rosetta_data"

OUT_DIR = Path.home() / "Source" / "Rosetta_Program" / "papers" / "caz-framework" / "figures"

# --- Score category palette (matches GEM paper and caz-validation) ---
CAT_COLOR = {
    "black_hole": "#C44E52",
    "strong":     "#4878CF",
    "moderate":   "#6AA84F",
    "gentle":     "#8E8E8E",
    "embedding":  "#DD8452",
}
CAT_FILL = {
    "black_hole": "#F5C6C8",
    "strong":     "#C9D8F0",
    "moderate":   "#D0EAC8",
    "gentle":     "#E8E8E8",
    "embedding":  "#F9DEC9",
}
CAT_LABEL = {
    "black_hole": "Black hole  (score > 0.5)",
    "strong":     "Strong  (0.2 – 0.5)",
    "moderate":   "Moderate  (0.05 – 0.2)",
    "gentle":     "Gentle  (< 0.05)",
    "embedding":  "Embedding  (< 25% depth)",
}


def score_cat(region) -> str:
    if region.depth_pct <= 25.0 and region.caz_score < 0.2:
        return "embedding"
    if region.caz_score >= 0.5:
        return "black_hole"
    if region.caz_score >= 0.2:
        return "strong"
    if region.caz_score >= 0.05:
        return "moderate"
    return "gentle"


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace("-", "_").replace(".", "_")


def load_caz(model_id: str, concept: str) -> dict:
    slug = model_slug(model_id)
    path = ROSETTA_DATA_ROOT / "models" / slug / f"caz_{concept}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}\n"
            f"Run: python rosetta_analysis/extraction/extract.py --model {model_id}"
        )
    return json.loads(path.read_text())


def build_metrics(d: dict) -> list[LayerMetrics]:
    return [
        LayerMetrics(
            layer=m["layer"],
            separation=m["separation_fisher"],
            coherence=m["coherence"],
            velocity=m["velocity"],
        )
        for m in d["layer_data"]["metrics"]
    ]


def plot_single_region(ax, d: dict, metrics: list[LayerMetrics], model_label: str, concept_label: str):
    """Left panel: highlight only the global-peak region."""
    n = d["layer_data"]["n_layers"]
    depths = np.array([m.layer / (n - 1) * 100 for m in metrics])
    seps   = np.array([m.separation for m in metrics])
    vels   = np.array([m.velocity for m in metrics])

    # Single-region: find global peak and bracket with nearest velocity zero-crossings
    peak_i = int(np.argmax(seps))

    # Walk left from peak until velocity changes sign (or edge)
    left = 0
    for i in range(peak_i - 1, 0, -1):
        if vels[i] <= 0:
            left = i
            break

    # Walk right from peak until velocity changes sign (or edge)
    right = n - 1
    for i in range(peak_i + 1, n - 1):
        if vels[i] >= 0:
            right = i
            break

    x0 = depths[left]
    x1 = depths[right]

    ax.axvspan(x0, x1, color="#C9D8F0", alpha=0.75, zorder=0, linewidth=0,
               label="Allocation zone")
    ax.plot(depths, seps, color="#263238", linewidth=1.8, zorder=3)
    ax.plot(depths[peak_i], seps[peak_i], "o",
            color="#4878CF", markersize=9, zorder=5,
            markeredgecolor="white", markeredgewidth=1.5)
    ax.axvline(depths[peak_i], color="#4878CF", linewidth=0.9,
               linestyle="--", alpha=0.6, zorder=2)

    ax.text(depths[peak_i], seps[peak_i] * 1.012,
            f"L{peak_i}  ({depths[peak_i]:.0f}%)",
            ha="center", va="bottom", fontsize=8, color="#4878CF")

    ax.set_xlim(0, 100)
    ax.set_xlabel("Layer depth (%)", fontsize=10)
    ax.set_ylabel("Fisher separation  $S(\\ell)$", fontsize=10)
    ax.set_title(f"A  Single-region  ·  {concept_label} / {model_label}",
                 fontsize=10, loc="left", fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor="#C9D8F0", edgecolor="#4878CF",
                       linewidth=1, label="Allocation zone (global peak)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.85, loc="lower right")


def plot_multi_region(ax, d: dict, metrics: list[LayerMetrics], model_label: str, concept_label: str):
    """Right panel: all scored CAZ regions."""
    n = d["layer_data"]["n_layers"]
    depths = np.array([m.layer / (n - 1) * 100 for m in metrics])
    seps   = np.array([m.separation for m in metrics])

    profile = find_caz_regions_scored(metrics)
    regions = profile.regions

    # Shade CAZ regions
    for r in regions:
        cat = score_cat(r)
        x0 = r.start / (n - 1) * 100
        x1 = r.end   / (n - 1) * 100
        ax.axvspan(x0, x1, color=CAT_FILL[cat], alpha=0.75, zorder=0, linewidth=0)

    ax.plot(depths, seps, color="#263238", linewidth=1.8, zorder=3)

    # Peak markers
    seen_cats: set[str] = set()
    for r in regions:
        cat   = score_cat(r)
        px    = r.peak / (n - 1) * 100
        py    = seps[r.peak]
        ax.plot(px, py, "o",
                color=CAT_COLOR[cat], markersize=8, zorder=5,
                markeredgecolor="white", markeredgewidth=1.2)
        seen_cats.add(cat)

    n_caz = len(regions)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Layer depth (%)", fontsize=10)
    ax.set_ylabel("Fisher separation  $S(\\ell)$", fontsize=10)
    ax.set_title(
        f"B  Scored multi-region  ·  {concept_label} / {model_label}\n"
        f"{n_caz} CAZ{'s' if n_caz != 1 else ''} detected",
        fontsize=10, loc="left", fontweight="bold",
    )
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cat_order = ["embedding", "black_hole", "strong", "moderate", "gentle"]
    legend_handles = [
        mpatches.Patch(facecolor=CAT_FILL[c], edgecolor=CAT_COLOR[c],
                       linewidth=1, label=CAT_LABEL[c])
        for c in cat_order if c in seen_cats
    ]
    ax.legend(handles=legend_handles, fontsize=7.5, framealpha=0.85, loc="lower right")


def main():
    left_model   = "EleutherAI/pythia-1.4b"
    left_concept = "sentiment"
    right_model   = "Qwen/Qwen2.5-0.5B"
    right_concept = "credibility"

    d_left    = load_caz(left_model, left_concept)
    m_left    = build_metrics(d_left)
    d_right   = load_caz(right_model, right_concept)
    m_right   = build_metrics(d_right)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.subplots_adjust(wspace=0.38)

    plot_single_region(axes[0], d_left,  m_left,  "Pythia-1.4b",  "Sentiment")
    plot_multi_region( axes[1], d_right, m_right, "Qwen2.5-0.5B", "Credibility")

    fig.tight_layout()

    for fmt in ("pdf", "png"):
        out = OUT_DIR / f"caz_detection_comparison.{fmt}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate Figure 2 for the CAZ Framework paper.

Scored CAZ profile for negation in OPT-2.7B (32 layers).
Two-panel: S(l) curve with CAZ regions + velocity v(l) with zero-crossings.

Data source: rosetta_data/models/facebook_opt_2.7b/caz_negation.json
Extraction:  rosetta_analysis/extraction/extract.py --model facebook/opt-2.7b

Output: fig2_caz_profile.pdf + fig2_caz_profile.png

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

try:
    from rosetta_tools.caz import find_caz_regions_scored, LayerMetrics
    from rosetta_tools.paths import ROSETTA_DATA_ROOT
except ImportError:
    _rt = Path(__file__).resolve().parents[4] / "rosetta_tools"
    sys.path.insert(0, str(_rt))
    from rosetta_tools.rosetta_tools.caz import find_caz_regions_scored, LayerMetrics
    ROSETTA_DATA_ROOT = Path.home() / "rosetta_data"

OUT_DIR = Path.home() / "Source" / "Rosetta_Program" / "papers" / "caz-framework" / "figures"

MODEL_ID = "facebook/opt-2.7b"
CONCEPT  = "negation"

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
    "embedding":  "Embedding  (≤ 25% depth)",
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


def main():
    slug = model_slug(MODEL_ID)
    path = ROSETTA_DATA_ROOT / "models" / slug / f"caz_{CONCEPT}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}\n"
            f"Run: python rosetta_analysis/extraction/extract.py --model {MODEL_ID}"
        )

    d       = json.loads(path.read_text())
    n       = d["layer_data"]["n_layers"]
    metrics_raw = d["layer_data"]["metrics"]
    metrics = [
        LayerMetrics(
            layer=m["layer"],
            separation=m["separation_fisher"],
            coherence=m["coherence"],
            velocity=m["velocity"],
        )
        for m in metrics_raw
    ]

    depths = np.array([m.layer / (n - 1) * 100 for m in metrics])
    seps   = np.array([m.separation for m in metrics])
    cohs   = np.array([m.coherence  for m in metrics])
    vels   = np.array([m.velocity   for m in metrics])

    profile = find_caz_regions_scored(metrics)
    regions = profile.regions

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, (ax_s, ax_v) = plt.subplots(
        2, 1, figsize=(8, 5.2),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    ax_s.set_xlim(0, 100)
    ax_v.set_xlim(0, 100)

    # ── CAZ region shading ────────────────────────────────────────────────────
    seen_cats: set[str] = set()
    for r in regions:
        cat = score_cat(r)
        x0  = r.start / (n - 1) * 100
        x1  = r.end   / (n - 1) * 100
        ax_s.axvspan(x0, x1, color=CAT_FILL[cat], alpha=0.70, zorder=0, linewidth=0)
        ax_v.axvspan(x0, x1, color=CAT_FILL[cat], alpha=0.45, zorder=0, linewidth=0)
        seen_cats.add(cat)

    # ── S(l) curve ────────────────────────────────────────────────────────────
    ax_s.plot(depths, seps, color="#263238", linewidth=1.8, zorder=3)
    ax_s.plot(depths, seps, "o", color="#263238", markersize=2.0, alpha=0.5, zorder=3)

    # ── Coherence secondary axis ──────────────────────────────────────────────
    ax_c = ax_s.twinx()
    ax_c.plot(depths, cohs, color="#546E7A", linewidth=0.8, linestyle="--",
              alpha=0.45, zorder=2, label="Coherence $C(\\ell)$")
    ax_c.set_ylabel("Coherence  $C(\\ell)$", fontsize=8, color="#546E7A", labelpad=4)
    ax_c.tick_params(axis="y", labelsize=7, colors="#546E7A")
    ax_c.set_ylim(0, 0.7)
    for sp in ax_c.spines.values():
        sp.set_linewidth(0.5)

    # ── Peak markers ──────────────────────────────────────────────────────────
    for r in regions:
        cat = score_cat(r)
        px  = r.peak / (n - 1) * 100
        py  = seps[r.peak]
        ax_s.plot(px, py, "o",
                  color=CAT_COLOR[cat], markersize=9, zorder=5,
                  markeredgecolor="white", markeredgewidth=1.5)
        ax_s.text(px, py, f"{r.caz_score:.2f}",
                  ha="center", va="center", fontsize=5.5,
                  color="white", fontweight="bold", zorder=6)

    # ── Velocity panel ────────────────────────────────────────────────────────
    ax_v.axhline(0, color="#546E7A", linewidth=0.8, linestyle="-", alpha=0.6)
    ax_v.plot(depths, vels, color="#263238", linewidth=1.4, zorder=3)
    ax_v.fill_between(depths, vels, 0,
                      where=(vels >= 0), color="#37474F", alpha=0.18, zorder=1)
    ax_v.fill_between(depths, vels, 0,
                      where=(vels <= 0), color="#78909C", alpha=0.10, zorder=1)

    for r in regions:
        cat = score_cat(r)
        px  = r.peak / (n - 1) * 100
        ax_v.axvline(px, color=CAT_COLOR[cat], linewidth=1.0,
                     linestyle="--", alpha=0.7, zorder=4)

    # Clip extreme velocity points with arrows
    VCLIP = 0.12
    ax_v.set_ylim(-VCLIP, VCLIP)
    for i, v in enumerate(vels):
        if abs(v) > VCLIP:
            sign = 1 if v > 0 else -1
            ax_v.annotate(
                "",
                xy=(depths[i], sign * VCLIP * 0.92),
                xytext=(depths[i], sign * VCLIP * 0.60),
                arrowprops=dict(arrowstyle="->", color="#607D8B", lw=0.9),
            )

    # ── Labels ────────────────────────────────────────────────────────────────
    ax_s.set_ylabel("Fisher separation  $S(\\ell)$", fontsize=10)
    ax_s.xaxis.set_visible(False)
    ax_s.spines["top"].set_visible(False)
    ax_s.spines["right"].set_visible(False)
    ax_s.yaxis.set_major_locator(ticker.MaxNLocator(5))

    ax_v.set_xlabel("Layer depth (%)", fontsize=10)
    ax_v.set_ylabel("$v(\\ell)$", fontsize=9)
    ax_v.spines["top"].set_visible(False)
    ax_v.spines["right"].set_visible(False)
    ax_v.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, symmetric=True))
    ax_v.tick_params(labelsize=8)

    # ── Title ─────────────────────────────────────────────────────────────────
    n_caz = len(regions)
    ax_s.set_title(
        f"Scored CAZ profile  ·  Negation / OPT-2.7B  ({n} layers)  ·  {n_caz} CAZes detected\n"
        r"Score inside marker  ·  shading = category  ·  $v(\ell)$ panel shows boundary zero-crossings",
        fontsize=9, loc="left", pad=6,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    cat_order = ["embedding", "black_hole", "strong", "moderate", "gentle"]
    handles = [
        mpatches.Patch(facecolor=CAT_FILL[c], edgecolor=CAT_COLOR[c],
                       linewidth=1, label=CAT_LABEL[c])
        for c in cat_order if c in seen_cats
    ]
    ax_s.legend(handles=handles, fontsize=7.5, framealpha=0.88, loc="lower right")

    fig.align_ylabels([ax_s, ax_v])
    fig.tight_layout()

    for fmt in ("pdf", "png"):
        out = OUT_DIR / f"caz_profile_proof_of_concept.{fmt}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close(fig)
    print(f"\nOPT-2.7b | negation | {n_caz} CAZes:")
    for r in regions:
        print(f"  L{r.peak:2d} ({r.depth_pct:.0f}%)  score={r.caz_score:.3f}  cat={score_cat(r)}")


if __name__ == "__main__":
    main()

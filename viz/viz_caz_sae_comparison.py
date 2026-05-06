#!/usr/bin/env python3
"""
viz_caz_sae_comparison.py — Single-figure comparison of CAZ eigenvector
separation curves vs Gemma Scope SAE discrimination curves.

2×4 layout: 7 concept panels + 1 mean±1σ summary.
Y-axis: score normalised to each curve's own peak (dimensionless [0, 1]).

Usage:
    python src/viz_caz_sae_comparison.py
    python src/viz_caz_sae_comparison.py --xval-dir results/gemma_scope_xval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent))
from viz_style import THEME, concept_color, layer_ticks
from rosetta_tools.paths import ROSETTA_RESULTS

PAPERS_DIR = Path.home() / "Source" / "Rosetta_Program" / "papers" / "caz-validation" / "figures"

CONCEPTS = [
    "credibility", "certainty", "sentiment", "moral_valence",
    "causation", "temporal_order", "negation",
]

CONCEPT_LABELS = {
    "credibility":    "Credibility",
    "certainty":      "Certainty",
    "sentiment":      "Sentiment",
    "moral_valence":  "Moral Valence",
    "causation":      "Causation",
    "temporal_order": "Temporal Order",
    "negation":       "Negation",
}

SAE_COLOR      = THEME["cka_line"]  # dark blue — SAE (16k features)
CAZ_MEAN_COLOR = "#D84315"          # deep orange — CAZ mean line (summary panel)


def spearman_r(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    xval_dir = Path(args.xval_dir)
    curves   = json.loads((xval_dir / "caz_vs_sae_curves.json").read_text())["results"]

    # ── Figure layout: 2×4 — 7 concepts + summary ─────────────────────────────
    # Row 0: credibility · certainty · sentiment · moral_valence
    # Row 1: causation   · temporal_order · negation · mean±1σ summary
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        2, 4,
        figure=fig,
        hspace=0.52, wspace=0.32,
        left=0.07, right=0.97,
        top=0.88, bottom=0.09,
    )

    # (row, col) for each concept in CONCEPTS order
    axes_positions = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2),
    ]

    concept_r_values = []

    for idx, concept in enumerate(CONCEPTS):
        row, col = axes_positions[idx]
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("white")

        v     = curves[concept]
        caz   = np.array(v["caz_scores"])
        sae   = np.array(v["sae_scores"])
        caz_n = caz / (caz.max() + 1e-8)
        sae_n = sae / (sae.max() + 1e-8)
        L     = np.arange(len(caz_n))
        cc    = concept_color(concept)

        ax.plot(L, sae_n, color=SAE_COLOR, linewidth=1.8, alpha=0.9, label="SAE (16k features)")
        ax.plot(L, caz_n, color=cc,        linewidth=1.8, alpha=0.9, label="CAZ (1 eigenvec)")

        ax.fill_between(L, caz_n, sae_n, where=(caz_n >= sae_n), alpha=0.18, color=cc)
        ax.fill_between(L, caz_n, sae_n, where=(caz_n <  sae_n), alpha=0.18, color=SAE_COLOR)

        r = v["spearman_r"]
        concept_r_values.append(r)

        ax.set_title(CONCEPT_LABELS[concept], color=THEME["text"],
                     fontsize=10, fontweight="bold", pad=4)
        ax.text(0.97, 0.06, f"r = {r:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                color=THEME["text"], fontsize=9, alpha=0.9)

        ax.set_xlim(0, len(L) - 1)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel("Layer", color=THEME["dim"], fontsize=8)
        if col == 0:
            ax.set_ylabel("Score (norm. to peak)", color=THEME["dim"], fontsize=8)
        ax.tick_params(colors=THEME["dim"], labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(THEME["spine"])
        ax.grid(True, color="#ECEFF1", linewidth=0.5)
        ax.legend(fontsize=6.5, loc="upper left",
                  facecolor="white", edgecolor=THEME["spine"],
                  labelcolor=THEME["text"], handlelength=1.2)

    # ── Summary panel (row 1, col 3) ───────────────────────────────────────────
    ax_sum = fig.add_subplot(gs[1, 3])
    ax_sum.set_facecolor("white")

    all_caz, all_sae = [], []
    for concept in CONCEPTS:
        v = curves[concept]
        c = np.array(v["caz_scores"]); c = c / (c.max() + 1e-8)
        s = np.array(v["sae_scores"]); s = s / (s.max() + 1e-8)
        all_caz.append(c)
        all_sae.append(s)

    all_caz = np.stack(all_caz)
    all_sae = np.stack(all_sae)
    L       = np.arange(all_caz.shape[1])

    mean_caz = all_caz.mean(axis=0);  std_caz = all_caz.std(axis=0)
    mean_sae = all_sae.mean(axis=0);  std_sae = all_sae.std(axis=0)

    ax_sum.fill_between(L, mean_sae - std_sae, mean_sae + std_sae, alpha=0.2, color=SAE_COLOR)
    ax_sum.fill_between(L, mean_caz - std_caz, mean_caz + std_caz, alpha=0.2, color=CAZ_MEAN_COLOR)
    ax_sum.plot(L, mean_sae, color=SAE_COLOR,      linewidth=2.2, label="SAE")
    ax_sum.plot(L, mean_caz, color=CAZ_MEAN_COLOR, linewidth=2.2, label="CAZ")

    ax_sum.set_title("Mean ± 1σ (all 7 concepts)", color=THEME["text"],
                     fontsize=10, fontweight="bold", pad=4)
    ax_sum.text(0.5, 0.1, f"mean r = {np.mean(concept_r_values):.3f}",
                transform=ax_sum.transAxes, ha="center",
                color=THEME["text"], fontsize=11, fontweight="bold")
    ax_sum.legend(fontsize=8, loc="upper left",
                  facecolor="white", edgecolor=THEME["spine"],
                  labelcolor=THEME["text"])
    ax_sum.set_xlim(0, all_caz.shape[1] - 1)
    ax_sum.set_ylim(-0.05, 1.08)
    ax_sum.set_xlabel("Layer", color=THEME["dim"], fontsize=8)
    ax_sum.tick_params(colors=THEME["dim"], labelsize=7)
    for spine in ax_sum.spines.values():
        spine.set_edgecolor(THEME["spine"])
    ax_sum.grid(True, color="#ECEFF1", linewidth=0.5)

    # ── Titles ─────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.955,
        "CAZ Eigenvectors vs. Gemma Scope SAEs  |  Gemma-2-2b",
        ha="center", va="center", color=THEME["text"],
        fontsize=15, fontweight="bold",
    )
    fig.text(
        0.5, 0.925,
        f"CAZ: 700 labeled texts · 1 forward pass · no SAE download        "
        f"SAE: 26 × 302 MB checkpoints (7.8 GB) · mean Spearman r = {np.mean(concept_r_values):.3f}",
        ha="center", va="center", color=THEME["dim"],
        fontsize=9.5,
    )

    out = PAPERS_DIR / "fig_sae_comparison.png"
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")


def run_overlay(args):
    """Single-panel overlay: all 7 concepts on one plot as bands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xval_dir = Path(args.xval_dir)
    curves   = json.loads((xval_dir / "caz_vs_sae_curves.json").read_text())["results"]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    all_caz, all_sae = [], []
    layers = None
    for concept in CONCEPTS:
        if concept not in curves:
            continue
        v   = curves[concept]
        caz = np.array(v["caz_scores"], dtype=np.float32)
        sae = np.array(v["sae_scores"], dtype=np.float32)
        caz_n = caz / (caz.max() + 1e-8)
        sae_n = sae / (sae.max() + 1e-8)
        all_caz.append(caz_n)
        all_sae.append(sae_n)
        if layers is None:
            layers = np.arange(len(caz_n))

        c_color = concept_color(concept)
        ax.plot(layers, sae_n, color=SAE_COLOR, linewidth=0.9, alpha=0.25)
        ax.plot(layers, caz_n, color=c_color,   linewidth=0.9, alpha=0.25)

    all_caz = np.stack(all_caz)
    all_sae = np.stack(all_sae)

    mean_caz = all_caz.mean(axis=0);  min_caz = all_caz.min(axis=0);  max_caz = all_caz.max(axis=0)
    mean_sae = all_sae.mean(axis=0);  min_sae = all_sae.min(axis=0);  max_sae = all_sae.max(axis=0)

    ax.fill_between(layers, min_sae, max_sae, alpha=0.18, color=SAE_COLOR)
    ax.fill_between(layers, min_caz, max_caz, alpha=0.18, color=CAZ_MEAN_COLOR)

    ax.plot(layers, mean_sae, color=SAE_COLOR,      linewidth=2.8,
            label="Gemma Scope SAE  (16,384 features)", zorder=5)
    ax.plot(layers, mean_caz, color=CAZ_MEAN_COLOR, linewidth=2.8,
            label="CAZ eigenvector  (1 direction, 100 pairs/concept)", zorder=5)

    for peak_l in [11, 13, 19]:
        ax.axvline(peak_l, color=THEME["spine"], linewidth=0.9, linestyle="--", alpha=0.6)
    ax.text(11.3, 0.03, "L11", color=THEME["dim"], fontsize=7, alpha=0.8)
    ax.text(13.3, 0.03, "L13", color=THEME["dim"], fontsize=7, alpha=0.8)
    ax.text(19.3, 0.03, "L19", color=THEME["dim"], fontsize=7, alpha=0.8)

    mean_r = np.mean([v["spearman_r"] for v in curves.values()])
    ax.text(0.97, 0.06,
            f"mean Spearman  r = {mean_r:.3f}  (7 concepts)",
            transform=ax.transAxes, ha="right", va="bottom",
            color=THEME["text"], fontsize=12, fontweight="bold")

    ax.set_xlim(0, len(layers) - 1)
    ax.set_ylim(-0.02, 1.06)
    ax.set_xlabel("Layer", color=THEME["dim"], fontsize=11)
    ax.set_ylabel("Score (norm. to peak)", color=THEME["dim"], fontsize=11)
    ax.tick_params(colors=THEME["dim"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(THEME["spine"])
    ax.grid(True, color="#ECEFF1", linewidth=0.5)
    ax.legend(fontsize=10, loc="upper left",
              facecolor="white", edgecolor=THEME["spine"], labelcolor=THEME["text"])

    fig.suptitle(
        "CAZ Eigenvectors vs. Gemma Scope SAEs  |  Gemma-2-2b  |  7 concepts",
        color=THEME["text"], fontsize=14, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.005,
        "CAZ: 700 labeled texts · 1 forward pass · no SAE download        "
        "SAE: 26 × 302 MB checkpoints (7.8 GB)",
        ha="center", color=THEME["dim"], fontsize=9,
    )

    out = xval_dir / "caz_vs_sae_overlay.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xval-dir",
                        default=str(ROSETTA_RESULTS / "gemma_scope_xval"))
    parser.add_argument("--overlay", action="store_true",
                        help="Single-panel overlay of all 7 concepts")
    args = parser.parse_args()
    if args.overlay:
        run_overlay(args)
    else:
        run(args)


if __name__ == "__main__":
    main()

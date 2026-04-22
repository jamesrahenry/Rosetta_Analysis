#!/usr/bin/env python3
"""
viz_sae_validation.py — SAE cross-validation figure (paper / white background).

Layout (2 rows × 3 cols, top-left spans 2 cols):
  TOP-LEFT  (2-col) — Polysemantic SAE feature count per layer.
                       L11 / L13 / L19 highlighted: all 7 concepts peak here.
  TOP-RIGHT (1-col) — Direction cosine heatmap: max cos(CAZ eigvec, top SAE feature)
                       at each peak layer, per concept.
  BOT-LEFT          — Mean ± 1σ normalised curves (CAZ vs SAE, all 7 concepts).
  BOT-MID           — Moral Valence curve comparison.
  BOT-RIGHT         — Sentiment curve comparison.

Usage:
    cd caz_scaling
    python src/viz_sae_validation.py
    python src/viz_sae_validation.py --xval-dir results/gemma_scope_xval
    python src/viz_sae_validation.py --out results/gemma_scope_xval/caz_vs_sae_all_concepts.png

Written: 2026-04-11 UTC
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent))
from viz_style import THEME, concept_color, layer_ticks

CAZ_ROOT    = Path(__file__).resolve().parents[1]
RESULTS_DIR = CAZ_ROOT / "results"

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

SAE_COLOR      = THEME["cka_line"]  # dark blue
CAZ_MEAN_COLOR = "#D84315"          # deep orange — CAZ mean line
HIGHLIGHT_LAYERS = {11, 13, 19}     # all-7-concept CAZ cluster layers
BAR_HI   = "#BF360C"                # deep orange-red — L11/13/19 bars
BAR_LO   = "#FFAB91"                # pale orange — other peak layers


def _curve_panel(ax, concept, curves):
    """Draw a single CAZ-vs-SAE normalised curve panel."""
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
    ax.set_title(CONCEPT_LABELS[concept], color=THEME["text"],
                 fontsize=10, fontweight="bold", pad=4)
    ax.text(0.97, 0.06, f"r = {r:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            color=THEME["text"], fontsize=9, alpha=0.9)
    ax.set_xlim(0, len(L) - 1)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("Layer", color=THEME["dim"], fontsize=8)
    ax.set_ylabel("Score (norm. to peak)", color=THEME["dim"], fontsize=8)
    ax.tick_params(colors=THEME["dim"], labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(THEME["spine"])
    ax.grid(True, color="#ECEFF1", linewidth=0.5)
    ax.legend(fontsize=6.5, loc="upper left",
              facecolor="white", edgecolor=THEME["spine"],
              labelcolor=THEME["text"], handlelength=1.2)


def run(args):
    xval_dir  = Path(args.xval_dir)
    curves    = json.loads((xval_dir / "caz_vs_sae_curves.json").read_text())["results"]
    summary   = json.loads((xval_dir / "summary.json").read_text())
    direction = json.loads((xval_dir / "direction_agreement.json").read_text())["by_concept"]

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.52, wspace=0.38,
        left=0.07, right=0.97,
        top=0.88, bottom=0.09,
    )

    ax_bar  = fig.add_subplot(gs[0, :2])   # polysemantic bar chart — spans 2 cols
    ax_heat = fig.add_subplot(gs[0, 2])    # direction cosine heatmap
    ax_mean = fig.add_subplot(gs[1, 0])    # mean ± 1σ curve
    ax_mv   = fig.add_subplot(gs[1, 1])    # moral_valence curve
    ax_sent = fig.add_subplot(gs[1, 2])    # sentiment curve

    for ax in [ax_bar, ax_heat, ax_mean, ax_mv, ax_sent]:
        ax.set_facecolor("white")

    # ══════════════════════════════════════════════════════════════════════════
    # TOP-LEFT: polysemantic SAE feature count per layer
    # ══════════════════════════════════════════════════════════════════════════
    cazstellations = summary["cazstellations"]
    n_layers = 26
    bar_counts  = [cazstellations.get(str(l), {}).get("n_polysemantic_features", 0)
                   for l in range(n_layers)]
    bar_nconcepts = [len(cazstellations.get(str(l), {}).get("concepts", []))
                     for l in range(n_layers)]
    bar_colors = [
        BAR_HI if l in HIGHLIGHT_LAYERS else (BAR_LO if bar_counts[l] > 0 else "#F5F5F5")
        for l in range(n_layers)
    ]

    bars = ax_bar.bar(range(n_layers), bar_counts, color=bar_colors,
                      width=0.72, zorder=3, edgecolor="none")

    # Annotate non-zero bars with count
    for l, (count, nc) in enumerate(zip(bar_counts, bar_nconcepts)):
        if count == 0:
            continue
        offset = 0.7
        ax_bar.text(l, count + offset, str(count),
                    ha="center", va="bottom", fontsize=7.5,
                    color=BAR_HI if l in HIGHLIGHT_LAYERS else THEME["dim"],
                    fontweight="bold" if l in HIGHLIGHT_LAYERS else "normal")
        # n_concepts annotation below bar (only for highlight layers)
        if l in HIGHLIGHT_LAYERS:
            ax_bar.text(l, -1.8, f"all 7",
                        ha="center", va="top", fontsize=7,
                        color=BAR_HI, style="italic",
                        transform=ax_bar.get_xaxis_transform()
                        if False else ax_bar.transData)

    # Mark the all-7 layers with vertical reference lines
    for l in HIGHLIGHT_LAYERS:
        ax_bar.axvline(l, color=BAR_HI, linewidth=0.8, linestyle="--", alpha=0.45, zorder=2)

    ax_bar.set_xlim(-0.6, n_layers - 0.4)
    ax_bar.set_xticks(range(0, n_layers, 4))
    ax_bar.set_xticklabels([str(l) for l in range(0, n_layers, 4)],
                            color=THEME["dim"], fontsize=8)
    ax_bar.set_xlabel("Layer", color=THEME["dim"], fontsize=9)
    ax_bar.set_ylabel("Polysemantic SAE features\nshared across ≥2 concepts",
                      color=THEME["dim"], fontsize=8.5)
    ax_bar.set_title(
        "SAE feature sharing peaks at CAZ cluster layers  (L11 · L13 · L19)",
        color=THEME["text"], fontsize=10.5, fontweight="bold", loc="left", pad=5,
    )
    ax_bar.grid(axis="y", color="#ECEFF1", linewidth=0.5, zorder=0)
    ax_bar.grid(axis="x", visible=False)
    for sp in ax_bar.spines.values():
        sp.set_edgecolor(THEME["spine"])
    ax_bar.tick_params(colors=THEME["dim"])

    # Legend for bar colors
    from matplotlib.patches import Patch
    ax_bar.legend(
        handles=[
            Patch(facecolor=BAR_HI, label="L11 / L13 / L19  (all 7 concepts)"),
            Patch(facecolor=BAR_LO, label="Partial cluster  (2–4 concepts)"),
        ],
        fontsize=8, loc="upper left",
        facecolor="white", edgecolor=THEME["spine"], labelcolor=THEME["text"],
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TOP-RIGHT: direction cosine heatmap
    # ══════════════════════════════════════════════════════════════════════════
    # Layers present in direction_agreement for all 7 concepts
    heat_layers = [5, 9, 11, 13, 15, 16, 19, 22]

    heat = np.full((len(CONCEPTS), len(heat_layers)), np.nan)
    for i, concept in enumerate(CONCEPTS):
        if concept not in direction:
            continue
        for j, layer in enumerate(heat_layers):
            layer_str = str(layer)
            if layer_str not in direction[concept]:
                continue
            feats = direction[concept][layer_str]
            heat[i, j] = max(f["cos_sim_to_top_eigvec"] for f in feats)

    im = ax_heat.imshow(heat, aspect="auto", cmap="YlOrRd",
                        vmin=0.15, vmax=0.65, origin="upper")

    # Highlight L11/13/19 columns
    for j, layer in enumerate(heat_layers):
        if layer in HIGHLIGHT_LAYERS:
            ax_heat.axvline(j, color=BAR_HI, linewidth=2.0, alpha=0.4, zorder=5)

    # Annotate cells with value
    for i in range(len(CONCEPTS)):
        for j in range(len(heat_layers)):
            v = heat[i, j]
            if np.isnan(v):
                continue
            ax_heat.text(j, i, f"{v:.2f}",
                         ha="center", va="center",
                         fontsize=6.5,
                         color="white" if v > 0.48 else THEME["text"],
                         fontweight="bold" if v >= 0.5 else "normal")

    ax_heat.set_xticks(range(len(heat_layers)))
    ax_heat.set_xticklabels([f"L{l}" for l in heat_layers],
                             color=THEME["dim"], fontsize=8)
    ax_heat.set_yticks(range(len(CONCEPTS)))
    ax_heat.set_yticklabels([CONCEPT_LABELS[c] for c in CONCEPTS],
                             color=THEME["dim"], fontsize=8)
    ax_heat.set_title(
        "CAZ–SAE direction agreement\nmax cos(eigvec, top SAE feature)",
        color=THEME["text"], fontsize=9.5, fontweight="bold", loc="left", pad=4,
    )
    ax_heat.tick_params(colors=THEME["dim"])
    for sp in ax_heat.spines.values():
        sp.set_edgecolor(THEME["spine"])

    cb = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=THEME["dim"], labelsize=7)
    cb.set_label("Cosine similarity", color=THEME["dim"], fontsize=7.5)

    # ══════════════════════════════════════════════════════════════════════════
    # BOT-LEFT: mean ± 1σ curve
    # ══════════════════════════════════════════════════════════════════════════
    all_caz, all_sae = [], []
    r_values = []
    for concept in CONCEPTS:
        v = curves[concept]
        r_values.append(v["spearman_r"])
        c = np.array(v["caz_scores"]); c = c / (c.max() + 1e-8)
        s = np.array(v["sae_scores"]); s = s / (s.max() + 1e-8)
        all_caz.append(c); all_sae.append(s)

    all_caz = np.stack(all_caz); all_sae = np.stack(all_sae)
    L = np.arange(all_caz.shape[1])
    mean_caz = all_caz.mean(0); std_caz = all_caz.std(0)
    mean_sae = all_sae.mean(0); std_sae = all_sae.std(0)

    ax_mean.fill_between(L, mean_sae - std_sae, mean_sae + std_sae, alpha=0.2, color=SAE_COLOR)
    ax_mean.fill_between(L, mean_caz - std_caz, mean_caz + std_caz, alpha=0.2, color=CAZ_MEAN_COLOR)
    ax_mean.plot(L, mean_sae, color=SAE_COLOR,      linewidth=2.2, label="SAE")
    ax_mean.plot(L, mean_caz, color=CAZ_MEAN_COLOR, linewidth=2.2, label="CAZ")

    ax_mean.set_title("Mean ± 1σ  (all 7 concepts)", color=THEME["text"],
                      fontsize=10, fontweight="bold", pad=4)
    ax_mean.text(0.5, 0.1, f"mean r = {np.mean(r_values):.3f}",
                 transform=ax_mean.transAxes, ha="center",
                 color=THEME["text"], fontsize=11, fontweight="bold")
    ax_mean.legend(fontsize=8, loc="upper left",
                   facecolor="white", edgecolor=THEME["spine"],
                   labelcolor=THEME["text"])
    ax_mean.set_xlim(0, all_caz.shape[1] - 1)
    ax_mean.set_ylim(-0.05, 1.08)
    ax_mean.set_xlabel("Layer", color=THEME["dim"], fontsize=8)
    ax_mean.set_ylabel("Score (norm. to peak)", color=THEME["dim"], fontsize=8)
    ax_mean.tick_params(colors=THEME["dim"], labelsize=7)
    for sp in ax_mean.spines.values():
        sp.set_edgecolor(THEME["spine"])
    ax_mean.grid(True, color="#ECEFF1", linewidth=0.5)

    # ══════════════════════════════════════════════════════════════════════════
    # BOT-MID / BOT-RIGHT: concept curve panels
    # ══════════════════════════════════════════════════════════════════════════
    _curve_panel(ax_mv,   "moral_valence", curves)
    _curve_panel(ax_sent, "sentiment",     curves)

    # ── Titles ─────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.955,
        "CAZ vs. Gemma Scope SAEs — Independent validation  |  Gemma-2-2b",
        ha="center", va="center", color=THEME["text"],
        fontsize=14, fontweight="bold",
    )
    fig.text(
        0.5, 0.925,
        "Top: SAE finds maximum cross-concept feature sharing at the exact layers where all 7 CAZ eigenvectors peak. "
        "Bottom: normalised layer-wise signals track closely (illustrative).",
        ha="center", va="center", color=THEME["dim"],
        fontsize=9,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="SAE validation figure")
    parser.add_argument("--xval-dir",
                        default=str(RESULTS_DIR / "gemma_scope_xval"))
    parser.add_argument(
        "--out",
        default=str(RESULTS_DIR / "gemma_scope_xval" / "caz_vs_sae_all_concepts.png"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()

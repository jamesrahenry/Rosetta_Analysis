#!/usr/bin/env python3
"""
viz_caz_validation.py — CAZ predictions confirmed by Gemma Scope SAEs.

Three-panel figure:

  TOP    — SAE discrimination curves for all 7 concepts, with vertical markers
           at CAZ-predicted shared peak layers. CAZ makes the prediction;
           the SAE provides independent evidence.

  BOT-L  — Polysemantic SAE feature count per layer. Spikes at CAZ-predicted
           layers show the SAE independently identifies the same structure.

  BOT-R  — Direction alignment heatmap: cosine similarity between SAE decoder
           directions and CAZ eigenvectors at the three all-concept shared
           peaks (L11, L13, L19). Concept-specific structure, not noise.

Usage:
    python src/viz_caz_validation.py
    python src/viz_caz_validation.py --xval-dir results/gemma_scope_xval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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

# CAZ-predicted shared peak layers (all 7 concepts): the key predictions to validate
ALL_CONCEPT_PEAKS = [11, 13, 19]
# Partial shared peaks (2-4 concepts): secondary markers
PARTIAL_PEAKS = [5, 9, 15, 16, 22]


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    xval_dir = Path(args.xval_dir)

    # ── Load data ─────────────────────────────────────────────────────────────
    layer_agr  = json.loads((xval_dir / "layer_agreement.json").read_text())
    direction  = json.loads((xval_dir / "direction_agreement.json").read_text())
    summary    = json.loads((xval_dir / "summary.json").read_text())

    sae_curves = layer_agr["by_concept"]          # {concept: [score_L0..L25]}
    shared_peaks = summary["cazstellations"]       # {layer_str: {concepts, n_poly}}
    dir_by_concept = direction["by_concept"]       # {concept: {layer_str: [features]}}

    n_layers = len(next(iter(sae_curves.values())))

    # ── Polysemantic feature count per layer ──────────────────────────────────
    poly_by_layer = np.zeros(n_layers, dtype=int)
    for layer_str, v in shared_peaks.items():
        l = int(layer_str)
        if l < n_layers:
            poly_by_layer[l] = v["n_polysemantic_features"]

    # ── Direction alignment matrix: concepts × 3 key layers ──────────────────
    cos_matrix = np.zeros((len(CONCEPTS), len(ALL_CONCEPT_PEAKS)), dtype=np.float32)
    for ci, concept in enumerate(CONCEPTS):
        for li, layer in enumerate(ALL_CONCEPT_PEAKS):
            layer_str = str(layer)
            if concept in dir_by_concept and layer_str in dir_by_concept[concept]:
                feats = dir_by_concept[concept][layer_str]
                if feats:
                    best = max(feats, key=lambda f: abs(f["cos_sim_to_top_eigvec"]))
                    cos_matrix[ci, li] = best["cos_sim_to_top_eigvec"]

    # ── Per-polysemantic-feature CAZ alignment at key layers ──────────────────
    # For each key layer: identify polysemantic features (in top-20 for ≥2 concepts),
    # compute their mean |cos_sim| with CAZ eigenvectors across all concepts they span.
    # Result: {layer: [(feature_idx, mean_abs_cos, n_concepts_spanning)]}
    feature_alignment = {}
    for layer in ALL_CONCEPT_PEAKS:
        layer_str = str(layer)
        # Collect all features at this layer per concept
        feat_cos: dict[int, list[float]] = {}
        for concept in CONCEPTS:
            if concept not in dir_by_concept or layer_str not in dir_by_concept[concept]:
                continue
            for f in dir_by_concept[concept][layer_str]:
                fidx = f["feature_idx"]
                if fidx not in feat_cos:
                    feat_cos[fidx] = []
                feat_cos[fidx].append(abs(f["cos_sim_to_top_eigvec"]))
        # Keep only polysemantic features (spanning ≥2 concepts)
        poly_feats = [
            (fidx, float(np.mean(coss)), len(coss))
            for fidx, coss in feat_cos.items()
            if len(coss) >= 2
        ]
        poly_feats.sort(key=lambda x: -x[1])   # sort by alignment descending
        feature_alignment[layer] = poly_feats

    # ── Colours ───────────────────────────────────────────────────────────────
    BG          = "#0f0f1a"
    PANEL_BG    = "#14142a"
    GRID_COLOR  = "#252540"
    TEXT_COLOR  = "#e0e0e0"
    DIM_COLOR   = "#666680"
    SAE_COLOR   = "#4FC3F7"
    PEAK_COLOR  = "#FFD54F"   # gold — CAZ prediction marker
    PARTIAL_CLR = "#555570"

    CONCEPT_COLORS = plt.cm.tab10(np.linspace(0, 0.85, len(CONCEPTS)))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.4, 1],
        hspace=0.50,
        wspace=0.38,
        left=0.07, right=0.96,
        top=0.88, bottom=0.08,
    )

    # ── TOP: SAE curves + CAZ peak markers ───────────────────────────────────
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_facecolor(PANEL_BG)

    all_sae = []
    for ci, concept in enumerate(CONCEPTS):
        scores = np.array(sae_curves[concept], dtype=np.float32)
        scores_n = scores / (scores.max() + 1e-8)
        all_sae.append(scores_n)
        ax_top.plot(np.arange(n_layers), scores_n,
                    color=CONCEPT_COLORS[ci], linewidth=1.2, alpha=0.55,
                    label=CONCEPT_LABELS[concept])

    mean_sae = np.mean(all_sae, axis=0)
    ax_top.plot(np.arange(n_layers), mean_sae,
                color=SAE_COLOR, linewidth=2.5, label="Mean (all concepts)", zorder=5)

    # Partial shared peaks — subtle
    for l in PARTIAL_PEAKS:
        ax_top.axvline(l, color=PARTIAL_CLR, linewidth=0.8, linestyle="--", alpha=0.5)

    # All-concept CAZ peaks — prominent, annotated
    # Stagger text heights to avoid overlap: L11 low, L13 mid, L19 high
    annotation_y = {11: 0.30, 13: 0.55, 19: 0.78}
    for l in ALL_CONCEPT_PEAKS:
        ax_top.axvline(l, color=PEAK_COLOR, linewidth=1.6, linestyle="--", alpha=0.85, zorder=4)
        poly_count = poly_by_layer[l]
        y_text = annotation_y.get(l, mean_sae[l] + 0.08)
        ax_top.annotate(
            f"L{l}  ·  {poly_count} SAE features  ·  all 7 concepts",
            xy=(l, mean_sae[l]),
            xytext=(l + 0.5, y_text),
            color=PEAK_COLOR,
            fontsize=8,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=PEAK_COLOR, lw=0.8, alpha=0.5),
        )

    ax_top.set_xlim(0, n_layers - 1)
    ax_top.set_ylim(-0.02, 1.18)
    ax_top.set_xlabel("Layer", color=DIM_COLOR, fontsize=10)
    ax_top.set_ylabel("SAE discrimination score (norm.)", color=DIM_COLOR, fontsize=10)
    ax_top.set_title(
        "SAE discrimination curves — all 7 concepts  |  CAZ-predicted shared peak layers marked",
        color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=6,
    )
    ax_top.tick_params(colors=DIM_COLOR, labelsize=8)
    for spine in ax_top.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax_top.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)

    # Legend in two columns
    handles, labels = ax_top.get_legend_handles_labels()
    ax_top.legend(handles, labels, fontsize=7.5, ncol=4, loc="upper left",
                  facecolor="#1a1a2e", edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
                  handlelength=1.5, columnspacing=1.0)

    # ── BOT-L: per-feature alignment strip at CAZ-predicted layers ───────────
    ax_poly = fig.add_subplot(gs[1, 0])
    ax_poly.set_facecolor(PANEL_BG)

    # One column per key layer; dots = individual polysemantic SAE features,
    # y-position = mean |cos_sim| with CAZ eigenvectors,
    # colour = number of concepts the feature spans
    cmap_span = plt.cm.YlOrRd
    jitter_rng = np.random.default_rng(42)

    x_positions = {layer: i for i, layer in enumerate(ALL_CONCEPT_PEAKS)}

    for layer, feats in feature_alignment.items():
        xi = x_positions[layer]
        if not feats:
            continue
        cos_vals   = np.array([f[1] for f in feats])
        span_vals  = np.array([f[2] for f in feats])
        jitter     = jitter_rng.uniform(-0.18, 0.18, size=len(cos_vals))
        colors     = cmap_span((span_vals - 2) / (len(CONCEPTS) - 2))
        ax_poly.scatter(xi + jitter, cos_vals, c=colors,
                        s=28, alpha=0.85, zorder=3, edgecolors="none")
        # Mean line
        ax_poly.hlines(cos_vals.mean(), xi - 0.3, xi + 0.3,
                       colors=PEAK_COLOR, linewidth=2, zorder=4)
        ax_poly.text(xi, cos_vals.mean() + 0.035,
                     f"mean = {cos_vals.mean():.2f}",
                     ha="center", color=PEAK_COLOR, fontsize=8.5, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor=BG, alpha=0.7, edgecolor="none"))

    # Random baseline in high-dim space
    ax_poly.axhline(0.05, color=DIM_COLOR, linewidth=1,
                    linestyle=":", alpha=0.8, label="~random baseline")
    ax_poly.text(len(ALL_CONCEPT_PEAKS) - 0.45, 0.07,
                 "random ≈ 0", color=DIM_COLOR, fontsize=7, va="bottom")

    ax_poly.set_xticks(range(len(ALL_CONCEPT_PEAKS)))
    ax_poly.set_xticklabels([f"L{l}\n({poly_by_layer[l]} features)"
                              for l in ALL_CONCEPT_PEAKS],
                             color=TEXT_COLOR, fontsize=8.5)
    ax_poly.set_xlim(-0.5, len(ALL_CONCEPT_PEAKS) - 0.5)
    ax_poly.set_ylim(-0.02, 0.80)
    ax_poly.set_ylabel("|cos_sim| with CAZ eigenvector", color=DIM_COLOR, fontsize=9)
    ax_poly.set_title(
        "Polysemantic SAE features at CAZ-predicted layers\n"
        "each dot = 1 feature  |  colour = concepts spanned  |  bar = mean",
        color=TEXT_COLOR, fontsize=9.5, fontweight="bold", pad=4,
    )
    ax_poly.tick_params(colors=DIM_COLOR, labelsize=8)
    for spine in ax_poly.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax_poly.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.4, axis="y")

    # Colorbar for n_concepts_spanned
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=cmap_span, norm=Normalize(vmin=2, vmax=len(CONCEPTS)))
    sm.set_array([])
    cbar2 = fig.colorbar(sm, ax=ax_poly, fraction=0.04, pad=0.02)
    cbar2.set_label("concepts spanned", color=DIM_COLOR, fontsize=7)
    cbar2.ax.tick_params(colors=DIM_COLOR, labelsize=6)

    # ── BOT-R: direction alignment heatmap ────────────────────────────────────
    ax_heat = fig.add_subplot(gs[1, 1])
    ax_heat.set_facecolor(PANEL_BG)

    abs_matrix = np.abs(cos_matrix)
    im = ax_heat.imshow(
        abs_matrix,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0, vmax=1.0,   # full scale — 0.37-0.64 is moderate, not blazing
        interpolation="nearest",
    )

    # Cell labels — show absolute value, bright = strong alignment
    for ci in range(len(CONCEPTS)):
        for li in range(len(ALL_CONCEPT_PEAKS)):
            val = abs_matrix[ci, li]
            txt_color = "white" if val > 0.35 else "#cccccc"
            ax_heat.text(li, ci, f"{val:.2f}",
                         ha="center", va="center",
                         color=txt_color, fontsize=8.5, fontweight="bold")

    ax_heat.set_xticks(range(len(ALL_CONCEPT_PEAKS)))
    ax_heat.set_xticklabels([f"L{l}" for l in ALL_CONCEPT_PEAKS],
                             color=TEXT_COLOR, fontsize=9)
    ax_heat.set_yticks(range(len(CONCEPTS)))
    ax_heat.set_yticklabels([CONCEPT_LABELS[c] for c in CONCEPTS],
                             color=TEXT_COLOR, fontsize=9)
    ax_heat.set_xlim(-0.5, len(ALL_CONCEPT_PEAKS) - 0.3)   # give right edge breathing room
    ax_heat.set_title(
        "SAE decoder direction alignment with CAZ eigenvectors\n"
        "(|cos sim|, scale 0→1  ·  random baseline ≈ 0 in 2304 dims)",
        color=TEXT_COLOR, fontsize=10, fontweight="bold", pad=4,
    )
    ax_heat.tick_params(colors=DIM_COLOR)
    for spine in ax_heat.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(colors=DIM_COLOR, labelsize=7)
    cbar.set_label("cos similarity", color=DIM_COLOR, fontsize=8)

    # ── Title + subtitle ──────────────────────────────────────────────────────
    fig.suptitle(
        "CAZ Predicts → SAE Confirms  |  Gemma-2-2b  |  Gemma Scope (16,384 features)",
        color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.96,
    )
    fig.text(
        0.5, 0.010,
        "Gold markers: layers where CAZ independently predicted all 7 concepts peak simultaneously  ·  SAE had no knowledge of CAZ predictions",
        ha="center", color="#aaaacc", fontsize=9.5, fontstyle="italic",
    )

    out = xval_dir / "caz_validation_sae.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="CAZ predictions confirmed by Gemma Scope SAEs"
    )
    parser.add_argument(
        "--xval-dir",
        default=str(RESULTS_DIR / "gemma_scope_xval"),
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

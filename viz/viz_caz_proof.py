#!/usr/bin/env python3
"""
viz_caz_proof.py — Proof-of-concept CAZ figure for the validation paper.

Shows S(l) curve for credibility in GPT-2-large with:
  - CAZ regions shaded by score category
  - Peak markers colored by category (black hole / strong / gentle)
  - Ablation suppression annotated at each peak
  - Velocity v(l) in lower panel showing zero-crossings that bound CAZ regions

Output: papers/caz-validation/figures/fig_caz_proof.pdf and .png

Usage:
    python viz_caz_proof.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from viz_style import CAZ_CAT_COLORS, CAZ_CAT_FILL, CAZ_CAT_LABELS, caz_score_cat, THEME
from rosetta_tools.paths import ROSETTA_MODELS
from rosetta_tools.caz import find_caz_regions_scored, LayerMetrics

# ── Paths ──────────────────────────────────────────────────────────────────────
PAPER_FIG = Path.home() / "Source" / "Rosetta_Program" / "papers" / "caz-validation" / "figures"
PAPER_FIG.mkdir(parents=True, exist_ok=True)

MODEL_DIR = ROSETTA_MODELS / "openai_community_gpt2_large"

# ── Score category aliases (canonical definitions live in viz_style.py) ───────
CAT_COLORS = CAZ_CAT_COLORS
CAT_LABELS = CAZ_CAT_LABELS
CAT_FILL   = CAZ_CAT_FILL
score_to_cat = caz_score_cat


def load_data():
    """Load per-layer metrics, ablation, and detect CAZ regions via rosetta_tools."""
    caz  = json.loads((MODEL_DIR / "caz_credibility.json").read_text())

    raw_metrics = caz["layer_data"]["metrics"]
    n_layers = len(raw_metrics)

    depths   = np.array([m["layer"] / (n_layers - 1) * 100 for m in raw_metrics])
    seps     = np.array([m["separation_fisher"] for m in raw_metrics])
    vels     = np.array([m["velocity"] for m in raw_metrics])
    cohs     = np.array([m["coherence"] for m in raw_metrics])

    # ablation suppression by layer index (graceful — file may not exist)
    abl_by_layer: dict[int, float] = {}
    abl_file = MODEL_DIR / "ablation_gem_credibility.json"
    if abl_file.exists():
        abl = json.loads(abl_file.read_text())
        per_layer = abl.get("handoff", abl.get("peak", {})).get("per_layer", {})
        abl_by_layer = {int(L): v["sep_reduction"] for L, v in per_layer.items()}

    # Build LayerMetrics list and detect regions via rosetta_tools
    layer_metrics = [
        LayerMetrics(
            layer=m["layer"],
            separation=m["separation_fisher"],
            coherence=m["coherence"],
            velocity=m["velocity"],
        )
        for m in raw_metrics
    ]
    profile = find_caz_regions_scored(layer_metrics)

    # Convert CAZRegion objects to the dict format expected by build_figure
    regions = []
    for r in profile.regions:
        regions.append({
            "start_layer": r.start,
            "peak_layer":  r.peak,
            "end_layer":   r.end,
            "caz_score":   r.caz_score,
            "peak_separation": r.peak_separation,
            "category":    score_to_cat(r.caz_score),
            "ablation":    abl_by_layer.get(r.peak, None),
        })

    return depths, seps, vels, cohs, regions, n_layers


def build_figure(depths, seps, vels, cohs, regions, n_layers):
    fig = plt.figure(figsize=(7.0, 4.6))
    gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax_s = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1], sharex=ax_s)

    # ── Shared x-axis range ────────────────────────────────────────────────────
    ax_s.set_xlim(0, 100)
    ax_v.set_xlim(0, 100)

    # ── CAZ region shading ────────────────────────────────────────────────────
    for r in regions:
        cat   = r["category"]
        x0    = r["start_layer"] / (n_layers - 1) * 100
        x1    = r["end_layer"]   / (n_layers - 1) * 100
        color = CAT_FILL[cat]
        ax_s.axvspan(x0, x1, color=color, alpha=0.55, zorder=0, linewidth=0)
        ax_v.axvspan(x0, x1, color=color, alpha=0.40, zorder=0, linewidth=0)

    # ── S(l) curve ────────────────────────────────────────────────────────────
    ax_s.plot(depths, seps, color="#263238", linewidth=1.8, zorder=3, solid_capstyle="round")
    ax_s.plot(depths, seps, "o", color="#263238", markersize=2.0, zorder=3, alpha=0.5)

    # ── Coherence as a subtle secondary line ──────────────────────────────────
    # Canonical coherence twin-axis: THEME["coh_line"] (#546E7A), dashed,
    # linewidth=0.8, alpha=0.45. Right y-axis range fixed at [0, 0.7].
    # See VIZ_STYLE_GUIDE.md § "Coherence C(l) Secondary Axis".
    _coh = THEME["coh_line"]
    ax_c = ax_s.twinx()
    ax_c.plot(depths, cohs, color=_coh, linewidth=0.8, linestyle="--",
              alpha=0.45, zorder=2, label="Coherence C(ℓ)")
    ax_c.set_ylabel("Coherence  $C(\\ell)$", fontsize=8, color=_coh, labelpad=4)
    ax_c.tick_params(axis="y", labelsize=7, colors=_coh)
    ax_c.set_ylim(0, 0.7)
    ax_c.yaxis.set_tick_params(width=0.5)
    for sp in ax_c.spines.values():
        sp.set_linewidth(0.5)

    # ── CAZ peak markers and annotations ─────────────────────────────────────
    for r in regions:
        cat   = r["category"]
        color = CAT_COLORS[cat]
        label = CAT_LABELS[cat]
        peak_x = r["peak_layer"] / (n_layers - 1) * 100
        peak_y = seps[r["peak_layer"]]
        score  = r["caz_score"]
        abl_pct = r["ablation"] * 100 if r["ablation"] is not None else None

        # Marker
        ax_s.plot(peak_x, peak_y, "o",
                  color=color, markersize=10, zorder=5,
                  markeredgecolor="white", markeredgewidth=1.5)

        # Score label inside marker (just score number)
        ax_s.text(peak_x, peak_y, f"{score:.2f}",
                  ha="center", va="center", fontsize=5.5, color="white",
                  fontweight="bold", zorder=6)

        # Annotation callout
        r_idx = regions.index(r)
        if abl_pct is not None:
            line1 = f"{label}  (score {score:.3f})"
            line2 = f"↓ {abl_pct:.0f}% on ablation"
            ann_text = f"{line1}\n{line2}"
        else:
            ann_text = f"{label}  (score {score:.3f})"

        # Position callouts to avoid overlap
        if r_idx == 0:
            # Black hole: label to the upper-right of peak
            offset_xy = (peak_x + 6, peak_y + 0.13)
            ha = "left"
            conn = "arc3,rad=-0.15"
        elif r_idx == 1:
            # Strong: label above, left of peak
            offset_xy = (peak_x - 6, peak_y + 0.14)
            ha = "right"
            conn = "arc3,rad=0.1"
        else:
            # Gentle: label below peak (near-output, lots of space above)
            offset_xy = (peak_x - 14, peak_y + 0.10)
            ha = "right"
            conn = "arc3,rad=0.0"

        ax_s.annotate(
            ann_text,
            xy=(peak_x, peak_y),
            xytext=offset_xy,
            ha=ha, va="bottom",
            fontsize=6.8,
            color=color,
            fontweight="bold" if r_idx == 0 else "normal",
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=0.9,
                connectionstyle=conn,
            ),
            zorder=7,
        )

    # ── CAZ boundary lines (saddle zero-crossings) ────────────────────────────
    sign_changes = []
    for i in range(len(vels) - 1):
        if vels[i] * vels[i + 1] < 0 and vels[i] > 0:   # + → − : end of rising
            sign_changes.append(depths[i])
        elif vels[i] * vels[i + 1] < 0 and vels[i] < 0:  # − → + : start of rising
            sign_changes.append(depths[i])

    for x in sign_changes:
        ax_s.axvline(x, color="#607D8B", linewidth=0.7, linestyle=":", alpha=0.6, zorder=2)
        ax_v.axvline(x, color="#607D8B", linewidth=0.7, linestyle=":", alpha=0.6, zorder=2)

    # ── Velocity panel ────────────────────────────────────────────────────────
    ax_v.axhline(0, color="#546E7A", linewidth=0.8, linestyle="-", alpha=0.6)
    ax_v.plot(depths, vels, color="#263238", linewidth=1.4, zorder=3)
    ax_v.fill_between(depths, vels, 0,
                      where=(vels >= 0), color="#37474F", alpha=0.20, zorder=1)
    ax_v.fill_between(depths, vels, 0,
                      where=(vels <= 0), color="#78909C", alpha=0.12, zorder=1)

    # Mark peak and trough layers on velocity panel
    for r in regions:
        cat   = r["category"]
        peak_x = r["peak_layer"] / (n_layers - 1) * 100
        ax_v.axvline(peak_x, color=CAT_COLORS[cat], linewidth=1.0,
                     linestyle="--", alpha=0.7, zorder=4)

    # ── Axis labels and formatting ─────────────────────────────────────────────
    ax_s.set_ylabel("Fisher separation  $S(\\ell)$", fontsize=9)
    ax_s.set_ylim(0.9, 1.82)
    ax_s.yaxis.set_tick_params(labelsize=8)
    ax_s.xaxis.set_visible(False)
    ax_s.spines["top"].set_visible(False)
    ax_s.spines["right"].set_visible(False)
    ax_s.spines["bottom"].set_linewidth(0.5)

    ax_v.set_xlabel("Layer depth  (%)", fontsize=9)
    ax_v.set_ylabel("$v(\\ell)$", fontsize=8)
    ax_v.yaxis.set_tick_params(labelsize=7)
    ax_v.xaxis.set_tick_params(labelsize=8)
    ax_v.spines["top"].set_visible(False)
    ax_v.spines["right"].set_visible(False)
    # Clip to the mid-stream velocity range; L1 (+0.48) and L35 (−0.38) are
    # embedding/unembedding boundary events, not CAZ-defining zero-crossings.
    # Caption note references §3.3 and §5.4.
    VCLIP = 0.12
    ax_v.set_ylim(-VCLIP, VCLIP)
    ax_v.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, symmetric=True))

    # Clip indicator arrows for the two saturated points
    for layer_i, vel_i in enumerate(vels):
        if abs(vel_i) > VCLIP:
            xp = depths[layer_i]
            sign = 1 if vel_i > 0 else -1
            ax_v.annotate(
                "",
                xy=(xp, sign * VCLIP * 0.92),
                xytext=(xp, sign * VCLIP * 0.60),
                arrowprops=dict(arrowstyle="->", color="#607D8B", lw=0.9),
                zorder=5,
            )

    # ── Legend (score categories) ─────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=CAT_FILL["black_hole"], edgecolor=CAT_COLORS["black_hole"],
                       linewidth=1.0, label="Black hole  (score > 0.5)"),
        mpatches.Patch(facecolor=CAT_FILL["strong"],     edgecolor=CAT_COLORS["strong"],
                       linewidth=1.0, label="Strong  (0.2 – 0.5)"),
        mpatches.Patch(facecolor=CAT_FILL["gentle"],     edgecolor=CAT_COLORS["gentle"],
                       linewidth=1.0, label="Gentle  (< 0.05)"),
    ]
    ax_s.legend(
        handles=handles,
        loc="lower right",
        fontsize=7,
        framealpha=0.88,
        edgecolor="#CFD8DC",
        handlelength=1.2,
        handleheight=0.9,
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    ax_s.set_title(
        "Concept allocation zones: credibility in GPT-2-large  (36 layers)\n"
        r"Shading = CAZ score category  ·  numbers = separation suppression when ablated",
        fontsize=8.5, pad=8, loc="left",
    )

    # ── Subtle grid ───────────────────────────────────────────────────────────
    ax_s.grid(axis="y", linewidth=0.4, color="#ECEFF1", zorder=0)
    ax_v.grid(axis="y", linewidth=0.4, color="#ECEFF1", zorder=0)

    fig.align_ylabels([ax_s, ax_v])
    return fig


def main():
    depths, seps, vels, cohs, regions, n_layers = load_data()

    print("GPT-2-large | credibility | CAZ summary:")
    for r in regions:
        peak_pct = r["peak_layer"] / (n_layers - 1) * 100
        abl = r["ablation"]
        print(f"  CAZ {r['category']:12s} peak={r['peak_layer']:2d} ({peak_pct:.0f}%) "
              f"score={r['caz_score']:.4f} sep={r['peak_separation']:.3f} "
              f"ablation={abl*100:.1f}%" if abl else "  (no ablation)")

    fig = build_figure(depths, seps, vels, cohs, regions, n_layers)

    out_pdf = PAPER_FIG / "fig_caz_proof.pdf"
    out_png = PAPER_FIG / "fig_caz_proof.png"
    fig.savefig(str(out_pdf), bbox_inches="tight", dpi=150)
    fig.savefig(str(out_png), bbox_inches="tight", dpi=200)
    print(f"\nSaved:\n  {out_pdf}\n  {out_png}")


if __name__ == "__main__":
    main()

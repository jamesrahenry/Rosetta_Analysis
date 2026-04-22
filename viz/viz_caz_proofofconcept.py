#!/usr/bin/env python3
"""
viz_caz_proofofconcept.py — Proof-of-concept CAZ figure for the framework paper.

Two-panel figure (GPT-2-XL, credibility):

  TOP   — S(l) separation curve + C(l) coherence overlay.
          CAZ peaks colour-coded by type: embedding, strong, gentle.
          Shallow embedding zone shaded. Pre-unembedding zone shaded.

  BOT   — CAZ dependency chain. Three nodes (one per detected CAZ),
          arrow from L29 → L46 showing forward dependency (59.6%),
          L5 isolated (independent primitive). Score, depth, coherence
          annotated per node.

Sections referenced in the paper: §3.1, §3.3, §6.2, §6.3.

Usage:
    python src/viz_caz_proofofconcept.py
    python src/viz_caz_proofofconcept.py --model gpt2_openai_community_gpt2_xl_20260401_184059
    python src/viz_caz_proofofconcept.py --out results/caz_proofofconcept/credibility_chain.png
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np

CAZ_ROOT    = Path(__file__).resolve().parents[1]
RESULTS_DIR = CAZ_ROOT / "results"
DEFAULT_MODEL = "gpt2_openai_community_gpt2_xl_20260401_184059"
CONCEPT = "credibility"


def smooth(v: np.ndarray, k: int = 1) -> np.ndarray:
    if k <= 1:
        return v.copy()
    kernel = np.ones(k) / k
    out = np.convolve(v, kernel, mode="same")
    out[:k//2] = v[:k//2]
    out[-(k//2):] = v[-(k//2):]
    return out


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from matplotlib.lines import Line2D

    mdir = RESULTS_DIR / args.model

    # ── Load S(l), C(l), V(l) ─────────────────────────────────────────────────
    caz_d = json.loads((mdir / f"caz_{CONCEPT}.json").read_text())
    metrics = caz_d["layer_data"]["metrics"]
    n = caz_d["layer_data"]["n_layers"]
    L = np.arange(n)

    sep = np.array([m["separation_fisher"] for m in metrics])
    coh = np.array([m["coherence"]         for m in metrics])

    # ── Load CAZ / ablation data ───────────────────────────────────────────────
    abl_d = json.loads((mdir / f"ablation_multimodal_{CONCEPT}.json").read_text())
    cazs = abl_d["cazs"]          # list of dicts
    imat = np.array(abl_d["interaction_matrix"])  # i ablated → j retained pct

    # ── Palette ───────────────────────────────────────────────────────────────
    BG       = "#0f0f1a"
    PANEL_BG = "#14142a"
    GRID     = "#252540"
    TEXT     = "#e0e0e0"
    DIM      = "#666680"

    # CAZ type colours
    EMBED_CLR  = "#FFB74D"   # amber  — embedding leakage (shallow, ≤25% depth)
    STRONG_CLR = "#4FC3F7"   # blue   — strong allocation
    GENTLE_CLR = "#81C784"   # green  — gentle downstream
    ZERO_CLR   = "#404060"

    SEP_CLR = "#BB86FC"      # purple — S(l) curve
    COH_CLR = "#4FC3F7"      # same blue — C(l) overlay (light, dashed)

    EMBED_BAND  = "#FFB74D18"   # faint amber — shallow zone
    UNEMB_BAND  = "#4FC3F718"   # faint blue  — pre-unembedding zone

    # Score categories for labelling
    def caz_type(c):
        s = c["caz_score"]
        d = c["depth_pct"]
        if d <= 25.0:
            return "embedding", EMBED_CLR
        if s > 0.2:
            return "strong", STRONG_CLR
        return "gentle", GENTLE_CLR

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 10))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[1.7, 1],
        hspace=0.52,
        left=0.10, right=0.93,
        top=0.88, bottom=0.07,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # TOP: S(l) + C(l) curve with CAZ markers
    # ═══════════════════════════════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(PANEL_BG)

    # Shaded zones
    ax.axvspan(0,    n * 0.25, color=EMBED_BAND, zorder=0)   # embedding zone
    ax.axvspan(n * 0.90, n-1, color=UNEMB_BAND, zorder=0)   # pre-unembedding zone

    ax.text(n * 0.125, sep.max() * 0.97,
            "embedding zone", ha="center", color=EMBED_CLR, fontsize=8,
            alpha=0.75, style="italic", transform=ax.transData)
    ax.text(n * 0.945, sep.max() * 0.97,
            "pre-\nunembedding", ha="center", color=COH_CLR, fontsize=7.5,
            alpha=0.75, style="italic", transform=ax.transData)

    ax.axhline(0, color=ZERO_CLR, linewidth=0.8, zorder=1)

    # S(l) — main curve
    ax.plot(L, sep, color=SEP_CLR, linewidth=2.2, label="S(l) — Fisher separation", zorder=4)

    # C(l) — on secondary y-axis
    ax2 = ax.twinx()
    ax2.set_facecolor(PANEL_BG)
    ax2.plot(L, coh, color=COH_CLR, linewidth=1.2, linestyle="--", alpha=0.6,
             label="C(l) — coherence", zorder=3)
    ax2.set_ylabel("Coherence C(l)", color=DIM, fontsize=9)
    ax2.tick_params(colors=DIM, labelsize=7)
    ax2.set_ylim(0, coh.max() * 1.6)

    # CAZ peak markers
    for i, c in enumerate(cazs):
        lyr = c["peak"]
        cat, clr = caz_type(c)
        score = c["caz_score"]
        depth = c["depth_pct"]
        coh_v = c["coherence"]

        ax.axvline(lyr, color=clr, linewidth=1.4, linestyle="--", alpha=0.7, zorder=2)
        ax.scatter([lyr], [sep[lyr]], color=clr, s=110, zorder=6,
                   edgecolors=TEXT, linewidths=0.8)

        # Label above the peak
        label = f"L{lyr}  ({depth:.0f}%)\n[{cat}]  score {score:.3f}"
        y_off = sep.max() * (0.10 if i % 2 == 0 else 0.22)
        ax.annotate(
            label,
            xy=(lyr, sep[lyr]),
            xytext=(lyr + (1.5 if lyr < n * 0.7 else -10), sep[lyr] + y_off),
            color=clr, fontsize=8, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=clr, lw=0.7, alpha=0.6),
            bbox=dict(boxstyle="round,pad=0.25", facecolor=BG,
                      edgecolor=clr, alpha=0.85),
        )

    ax.set_xlim(0, n - 1)
    ax.set_ylim(sep.min() * 0.95, sep.max() * 1.30)
    ax.set_xlabel("Layer", color=DIM, fontsize=10)
    ax.set_ylabel("Fisher Separation S(l)", color=DIM, fontsize=10)
    ax.set_title(
        f"GPT-2-XL  ·  Concept: Credibility  ·  3 detected CAZes\n"
        f"Embedding leakage (L5) → main allocation (L29) → gentle refinement (L46)",
        color=TEXT, fontsize=11, fontweight="bold", pad=6,
    )
    ax.tick_params(colors=DIM, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5)

    # Combined legend
    handles = [
        Line2D([0], [0], color=SEP_CLR, linewidth=2.2,  label="S(l) — Fisher separation"),
        Line2D([0], [0], color=COH_CLR, linewidth=1.2,
               linestyle="--", alpha=0.6,               label="C(l) — coherence"),
        Line2D([0], [0], color=EMBED_CLR,  linewidth=2, label="Embedding CAZ"),
        Line2D([0], [0], color=STRONG_CLR, linewidth=2, label="Strong CAZ"),
        Line2D([0], [0], color=GENTLE_CLR, linewidth=2, label="Gentle CAZ"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right",
              facecolor="#1a1a2e", edgecolor=GRID, labelcolor=TEXT,
              handlelength=1.8, ncol=2)

    # ═══════════════════════════════════════════════════════════════════════════
    # BOT: CAZ dependency chain
    # ═══════════════════════════════════════════════════════════════════════════
    ax_chain = fig.add_subplot(gs[1])
    ax_chain.set_facecolor(PANEL_BG)
    ax_chain.set_xlim(0, 10)
    ax_chain.set_ylim(0, 4)
    ax_chain.axis("off")
    ax_chain.set_title(
        "CAZ dependency chain — forward-only information flow (§6.2)\n"
        "Arrow weight = % of target concept destroyed when source is ablated",
        color=TEXT, fontsize=10, fontweight="bold", pad=4,
    )

    # Node positions (x, y)  — 3 nodes
    node_x = [2.0, 5.0, 8.0]
    node_y = [2.0, 2.0, 2.0]
    node_w, node_h = 1.8, 1.3

    for i, c in enumerate(cazs):
        cat, clr = caz_type(c)
        lyr   = c["peak"]
        depth = c["depth_pct"]
        score = c["caz_score"]
        coh_v = c["coherence"]
        sr    = c["self_retained_pct"]

        # Box
        bx = node_x[i] - node_w / 2
        by = node_y[i] - node_h / 2
        box = FancyBboxPatch(
            (bx, by), node_w, node_h,
            boxstyle="round,pad=0.08",
            facecolor=PANEL_BG, edgecolor=clr, linewidth=2.0, zorder=4,
        )
        ax_chain.add_patch(box)

        # Node text
        ax_chain.text(node_x[i], node_y[i] + 0.28,
                      f"L{lyr}  ({depth:.0f}%)",
                      ha="center", va="center", color=clr,
                      fontsize=10, fontweight="bold", zorder=5)
        ax_chain.text(node_x[i], node_y[i] - 0.08,
                      f"[{cat}]  score {score:.3f}",
                      ha="center", va="center", color=TEXT, fontsize=8, zorder=5)
        ax_chain.text(node_x[i], node_y[i] - 0.38,
                      f"C={coh_v:.2f}  self-ret {sr:.1f}%",
                      ha="center", va="center", color=DIM, fontsize=7.5, zorder=5)

    # ── Dependency arrows ──────────────────────────────────────────────────────
    # interaction_matrix[i][j] = % retained in j when i is ablated
    # If imat[src][tgt] < 85%, there is a forward dependency src → tgt
    # Impact = 100 - imat[src][tgt]
    n_caz = len(cazs)
    for src in range(n_caz):
        for tgt in range(src + 1, n_caz):
            retained = imat[src][tgt]
            impact   = 100.0 - retained
            if impact > 15.0:  # meaningful dependency threshold
                # Arrow from right edge of src to left edge of tgt
                x0 = node_x[src] + node_w / 2
                x1 = node_x[tgt] - node_w / 2
                y0 = y1 = node_y[src]

                arr = FancyArrowPatch(
                    (x0, y0), (x1, y1),
                    arrowstyle="-|>",
                    mutation_scale=18,
                    linewidth=2.5,
                    color=STRONG_CLR,
                    zorder=3,
                )
                ax_chain.add_patch(arr)

                # Label on arrow
                mx = (x0 + x1) / 2
                ax_chain.text(mx, y0 + 0.30,
                              f"−{impact:.0f}% when\nL{cazs[src]['peak']} ablated",
                              ha="center", color=STRONG_CLR, fontsize=7.5,
                              bbox=dict(boxstyle="round,pad=0.2", facecolor=BG,
                                        edgecolor="none", alpha=0.85))

    # ── "Independent" label for L5 ────────────────────────────────────────────
    ax_chain.text(node_x[0], node_y[0] - node_h / 2 - 0.45,
                  "independent primitive\n(feeds nothing)",
                  ha="center", color=EMBED_CLR, fontsize=7.5, style="italic")

    # ── "Terminal" label for L46 ──────────────────────────────────────────────
    ax_chain.text(node_x[2], node_y[2] - node_h / 2 - 0.45,
                  "terminal refinement\n(pre-unembedding)",
                  ha="center", color=GENTLE_CLR, fontsize=7.5, style="italic")

    # ── Legend: category colours ──────────────────────────────────────────────
    for xi, (label, clr) in enumerate(
        [("Embedding CAZ (≤25%)", EMBED_CLR),
         ("Strong CAZ", STRONG_CLR),
         ("Gentle CAZ (<0.05)", GENTLE_CLR)]
    ):
        ax_chain.plot([], [], color=clr, linewidth=3, label=label)
    ax_chain.legend(fontsize=7.5, loc="lower center", ncol=3,
                    facecolor="#1a1a2e", edgecolor=GRID, labelcolor=TEXT,
                    handlelength=1.5, bbox_to_anchor=(0.5, -0.12))

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "Concept Allocation Zones — GPT-2-XL  ·  Credibility",
        color=TEXT, fontsize=14, fontweight="bold", y=0.96,
    )
    fig.text(
        0.5, 0.01,
        "CAZ score predicts geometric salience, not causal importance  ·  "
        "Gentle CAZes (score < 0.05) are causally real  ·  "
        "Forward-only dependency: L29 → L46",
        ha="center", color="#aaaacc", fontsize=8.5, style="italic",
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="CAZ proof-of-concept figure")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--out",
        default=str(RESULTS_DIR / "caz_proofofconcept" / "credibility_chain.png"),
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
viz_velocity_xcross.py — Baton-pass velocity X-crossing figure.

Three-panel figure:
  TOP     — All 7 concept velocity curves; causation highlighted.
            Vertical band marks the crossing zone (L22-L28).
  MID-L   — Close-up: v_causation and v_certainty on same axis.
            X marks the crossing at L23 where v_caus flips negative,
            v_cert flips positive simultaneously.
  MID-R   — Separation S(l) curves for causation + certainty + temporal_order.
            Shows causation plateauing while the others continue to climb.

Usage:
    python src/viz_velocity_xcross.py
    python src/viz_velocity_xcross.py --model gpt2_openai_community_gpt2_xl_20260401_184059
    python src/viz_velocity_xcross.py --out results/velocity_xcross.png
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
from viz_style import concept_color  # noqa: F401 — available for future use

CAZ_ROOT    = Path(__file__).resolve().parents[1]
RESULTS_DIR = CAZ_ROOT / "results"
DEFAULT_MODEL = "gpt2_openai_community_gpt2_xl_20260401_184059"

CONCEPTS = ["causation", "certainty", "credibility", "moral_valence",
            "negation", "sentiment", "temporal_order"]
LABELS   = {"causation": "Causation", "certainty": "Certainty",
            "credibility": "Credibility", "moral_valence": "Moral Valence",
            "negation": "Negation", "sentiment": "Sentiment",
            "temporal_order": "Temporal Order"}

# The known handoff pairs to highlight
HANDOFF_PAIRS = [
    ("causation", "certainty",      23),   # peak crossing layer
    ("causation", "temporal_order", 26),
    ("causation", "moral_valence",  24),
]


def load_concept(mdir: Path, concept: str) -> dict:
    f = mdir / f"caz_{concept}.json"
    d = json.loads(f.read_text())
    metrics = d["layer_data"]["metrics"]
    return {
        "velocity":   np.array([m["velocity"]         for m in metrics]),
        "separation": np.array([m["separation_fisher"] for m in metrics]),
        "coherence":  np.array([m["coherence"]         for m in metrics]),
        "n_layers":   d["layer_data"]["n_layers"],
    }


def smooth(v: np.ndarray, k: int = 2) -> np.ndarray:
    """Light Gaussian-ish smooth for display only."""
    kernel = np.array([0.25, 0.5, 0.25])
    out = np.convolve(v, kernel, mode="same")
    out[0], out[-1] = v[0], v[-1]
    return out


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyArrowPatch

    mdir = RESULTS_DIR / args.model

    data = {}
    for c in CONCEPTS:
        try:
            data[c] = load_concept(mdir, c)
        except FileNotFoundError:
            pass

    n  = data["causation"]["n_layers"]
    L  = np.arange(n)
    depth = L / (n - 1)

    # ── Palette ───────────────────────────────────────────────────────────────
    BG       = "#0f0f1a"
    PANEL_BG = "#14142a"
    GRID     = "#252540"
    TEXT     = "#e0e0e0"
    DIM      = "#666680"

    CAUS_CLR  = "#FF6B6B"   # red — causation (handoff donor)
    CERT_CLR  = "#4FC3F7"   # blue — certainty (recipient)
    TEMP_CLR  = "#81C784"   # green — temporal_order
    MORA_CLR  = "#FFD54F"   # gold — moral_valence
    BAND_CLR  = "#FF6B6B22" # faint red band — crossing zone
    ZERO_CLR  = "#404060"

    # Narrative overrides — this dark-bg figure uses donor/recipient colour logic
    # rather than canonical spec colours.  concept_color() is imported above
    # for any new additions that don't need the narrative treatment.
    CONCEPT_COLORS = {
        "causation":      CAUS_CLR,
        "certainty":      CERT_CLR,
        "credibility":    concept_color("credibility"),
        "moral_valence":  MORA_CLR,
        "negation":       concept_color("negation"),
        "sentiment":      concept_color("sentiment"),
        "temporal_order": TEMP_CLR,
    }

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.3, 1],
        hspace=0.48, wspace=0.36,
        left=0.07, right=0.96,
        top=0.88, bottom=0.07,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # TOP: all 7 velocity curves + causation highlighted + crossing band
    # ═══════════════════════════════════════════════════════════════════════════
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_facecolor(PANEL_BG)

    # Crossing zone band (causation goes negative L22-L28)
    cross_lo, cross_hi = 21.5, 28.5
    ax_top.axvspan(cross_lo, cross_hi, color=BAND_CLR, zorder=0)
    ax_top.text((cross_lo + cross_hi) / 2, 0.93,
                "handoff zone", ha="center", color=CAUS_CLR, fontsize=8,
                alpha=0.85, style="italic", transform=ax_top.get_xaxis_transform())

    ax_top.axhline(0, color=ZERO_CLR, linewidth=1, linestyle="-", zorder=1)

    for c in CONCEPTS:
        if c not in data:
            continue
        v = smooth(data[c]["velocity"])
        lw  = 2.5 if c == "causation" else 1.0
        alp = 1.0 if c in ("causation", "certainty", "temporal_order") else 0.45
        zord = 5 if c == "causation" else 3
        ax_top.plot(L, v, color=CONCEPT_COLORS[c], linewidth=lw, alpha=alp,
                    label=LABELS[c], zorder=zord)

    # Annotate each causation crossing with a small tick
    for _, cb, lcross in HANDOFF_PAIRS:
        if cb not in data:
            continue
        va = data["causation"]["velocity"][lcross]
        ax_top.axvline(lcross, color=CONCEPT_COLORS[cb], linewidth=0.8,
                       linestyle=":", alpha=0.6, zorder=2)

    ax_top.set_xlim(0, n - 1)
    ax_top.set_xlabel("Layer", color=DIM, fontsize=10)
    ax_top.set_ylabel("Velocity V(l)", color=DIM, fontsize=10)
    ax_top.set_title(
        f"Concept velocity curves — GPT-2-XL (48 layers)  |  "
        f"Causation broadcasts to all concepts in L22–L28",
        color=TEXT, fontsize=11, fontweight="bold", pad=6,
    )
    ax_top.tick_params(colors=DIM, labelsize=8)
    for sp in ax_top.spines.values():
        sp.set_edgecolor(GRID)
    ax_top.grid(True, color=GRID, linewidth=0.5, alpha=0.5)

    handles, labels = ax_top.get_legend_handles_labels()
    ax_top.legend(handles, labels, fontsize=8, ncol=4, loc="upper left",
                  facecolor="#1a1a2e", edgecolor=GRID, labelcolor=TEXT,
                  handlelength=1.5, columnspacing=1.0)

    # ═══════════════════════════════════════════════════════════════════════════
    # BOT-L: close-up of causation × certainty X-crossing
    # ═══════════════════════════════════════════════════════════════════════════
    ax_cross = fig.add_subplot(gs[1, 0])
    ax_cross.set_facecolor(PANEL_BG)

    # Zoom to L15-L33
    zoom_lo, zoom_hi = 14, 34
    zoom_mask = (L >= zoom_lo) & (L <= zoom_hi)
    Lz = L[zoom_mask]

    v_caus = smooth(data["causation"]["velocity"])[zoom_mask]
    v_cert = smooth(data["certainty"]["velocity"])[zoom_mask]
    v_temp = smooth(data["temporal_order"]["velocity"])[zoom_mask]

    ax_cross.axhline(0, color=ZERO_CLR, linewidth=1.2, zorder=1)
    ax_cross.fill_between(Lz, v_caus, 0,
                          where=v_caus < 0, alpha=0.12, color=CAUS_CLR)
    ax_cross.fill_between(Lz, v_cert, 0,
                          where=v_cert > 0, alpha=0.12, color=CERT_CLR)

    ax_cross.plot(Lz, v_caus, color=CAUS_CLR, linewidth=2.5,
                  label="Causation", zorder=5)
    ax_cross.plot(Lz, v_cert, color=CERT_CLR, linewidth=2.0,
                  label="Certainty", zorder=4)
    ax_cross.plot(Lz, v_temp, color=TEMP_CLR, linewidth=1.4,
                  alpha=0.7, linestyle="--", label="Temporal Order", zorder=3)

    # Mark the X-crossing at L23
    lcross = 23
    ax_cross.axvline(lcross, color=TEXT, linewidth=1.0, linestyle="--",
                     alpha=0.5, zorder=2)

    # X annotation
    vc = float(data["causation"]["velocity"][lcross])
    ve = float(data["certainty"]["velocity"][lcross])
    ax_cross.annotate(
        f"L{lcross}\nv_caus = {vc:+.4f}\nv_cert  = {ve:+.4f}",
        xy=(lcross, 0), xytext=(lcross + 1.5, 0.04),
        color=TEXT, fontsize=8,
        arrowprops=dict(arrowstyle="-", color=DIM, lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f0f1a",
                  edgecolor=DIM, alpha=0.85),
    )

    # Large X marker at crossing
    ax_cross.scatter([lcross], [0], marker="x", s=120, linewidths=2.5,
                     color=TEXT, zorder=6)

    ax_cross.set_xlim(zoom_lo, zoom_hi)
    ax_cross.set_xlabel("Layer", color=DIM, fontsize=10)
    ax_cross.set_ylabel("Velocity V(l)", color=DIM, fontsize=10)
    ax_cross.set_title(
        "Baton pass — L23 (48%)\n"
        "Causation velocity flips negative exactly when Certainty flips positive",
        color=TEXT, fontsize=10, fontweight="bold", pad=4,
    )
    ax_cross.tick_params(colors=DIM, labelsize=8)
    for sp in ax_cross.spines.values():
        sp.set_edgecolor(GRID)
    ax_cross.grid(True, color=GRID, linewidth=0.5, alpha=0.4)
    ax_cross.legend(fontsize=8, facecolor="#1a1a2e", edgecolor=GRID,
                    labelcolor=TEXT, handlelength=1.5)

    # ═══════════════════════════════════════════════════════════════════════════
    # BOT-R: separation curves — causation plateaus, certainty/temporal_order climb
    # ═══════════════════════════════════════════════════════════════════════════
    ax_sep = fig.add_subplot(gs[1, 1])
    ax_sep.set_facecolor(PANEL_BG)

    for c, clr in [("causation", CAUS_CLR), ("certainty", CERT_CLR),
                   ("temporal_order", TEMP_CLR), ("moral_valence", MORA_CLR)]:
        if c not in data:
            continue
        s = data[c]["separation"]
        s_norm = s / (s.max() + 1e-8)
        lw = 2.5 if c == "causation" else 1.6
        ax_sep.plot(L, s_norm, color=clr, linewidth=lw,
                    label=LABELS[c], zorder=4)

    # Mark L23 handoff
    ax_sep.axvline(23, color=CAUS_CLR, linewidth=1.0, linestyle="--",
                   alpha=0.6, zorder=2)
    ax_sep.text(23.5, 0.05, "handoff L23", color=CAUS_CLR,
                fontsize=7.5, alpha=0.8, style="italic")

    ax_sep.set_xlim(0, n - 1)
    ax_sep.set_ylim(-0.02, 1.15)
    ax_sep.set_xlabel("Layer", color=DIM, fontsize=10)
    ax_sep.set_ylabel("Separation S(l)  (norm.)", color=DIM, fontsize=10)
    ax_sep.set_title(
        "Separation after handoff\n"
        "Causation plateau matches Certainty / Temporal Order takeoff",
        color=TEXT, fontsize=10, fontweight="bold", pad=4,
    )
    ax_sep.tick_params(colors=DIM, labelsize=8)
    for sp in ax_sep.spines.values():
        sp.set_edgecolor(GRID)
    ax_sep.grid(True, color=GRID, linewidth=0.5, alpha=0.4)
    ax_sep.legend(fontsize=8, facecolor="#1a1a2e", edgecolor=GRID,
                  labelcolor=TEXT, handlelength=1.5)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "Velocity X-Crossing: Geometric Recycling in GPT-2-XL",
        color=TEXT, fontsize=14, fontweight="bold", y=0.96,
    )
    fig.text(
        0.5, 0.01,
        "Causation broadcasts its direction to Certainty, Temporal Order, and Moral Valence "
        "simultaneously at L22–L28  ·  velocity sign-flip = mathematical signature of baton-pass recycling",
        ha="center", color="#aaaacc", fontsize=9, style="italic",
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Velocity X-crossing baton-pass figure")
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--out",     default=str(RESULTS_DIR / "velocity_xcross" / "baton_pass.png"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

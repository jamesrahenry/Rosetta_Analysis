#!/usr/bin/env python3
"""
viz_baton_pass.py — Velocity X-crossing baton-pass figure (paper / white background).

Three-panel figure:
  TOP    — All 7 concept velocity curves, causation highlighted, crossing zone marked.
  BOT-L  — Close-up: v_causation × v_certainty × v_temporal_order, L14–L34.
            Large X marker at the precise crossing layer.
  BOT-R  — Separation S(l) curves for causation / certainty / temporal_order.
            Shows causation plateauing at handoff while successors climb.

Defaults to GPT-2-XL (48 layers) where the baton-pass is clearest.

Usage:
    python viz_baton_pass.py
    python viz_baton_pass.py --model openai-community/gpt2-xl
    python viz_baton_pass.py --out ~/Source/Rosetta_Program/papers/caz-validation/figures/fig_baton_pass.png

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
from viz_style import concept_color, THEME, layer_ticks, apply_theme
from rosetta_tools.paths import ROSETTA_MODELS

DEFAULT_MODEL_ID = "openai-community/gpt2-xl"
PAPERS_DIR = Path.home() / "Source" / "Rosetta_Program" / "papers" / "caz-validation" / "figures"

CONCEPTS = ["causation", "certainty", "credibility", "moral_valence",
            "negation", "sentiment", "temporal_order"]
LABELS = {c: c.replace("_", " ").title() for c in CONCEPTS}
LABELS["temporal_order"] = "Temporal Order"
LABELS["moral_valence"]  = "Moral Valence"

# Handoff pairs to annotate (concept_a → concept_b at layer)
HANDOFF_PAIRS = [
    ("causation", "certainty",      23),
    ("causation", "temporal_order", 26),
]


def find_model_dir(model_id: str) -> Path:
    slug = model_id.replace("/", "_").replace("-", "_")
    return ROSETTA_MODELS / slug


def load_concept(model_dir: Path, concept: str) -> dict:
    f = model_dir / f"caz_{concept}.json"
    d = json.loads(f.read_text())
    metrics = d["layer_data"]["metrics"]
    return {
        "velocity":   np.array([m["velocity"]          for m in metrics]),
        "separation": np.array([m["separation_fisher"]  for m in metrics]),
        "n_layers":   d["layer_data"]["n_layers"],
    }


def smooth(v: np.ndarray, k: int = 2) -> np.ndarray:
    kernel = np.array([0.25, 0.5, 0.25])
    out = np.convolve(v, kernel, mode="same")
    out[0], out[-1] = v[0], v[-1]
    return out


def run(args) -> None:
    mdir = find_model_dir(args.model)
    data = {}
    for c in CONCEPTS:
        try:
            data[c] = load_concept(mdir, c)
        except FileNotFoundError:
            pass

    n = data["causation"]["n_layers"]
    L = np.arange(n)

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.3, 1],
        hspace=0.52, wspace=0.32,
        left=0.07, right=0.97,
        top=0.88, bottom=0.09,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # TOP: all 7 velocity curves + crossing zone band
    # ═══════════════════════════════════════════════════════════════════════════
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_facecolor("white")

    cross_lo, cross_hi = 21.5, 28.5
    ax_top.axvspan(cross_lo, cross_hi,
                   color=concept_color("causation"), alpha=0.08, zorder=0)
    ax_top.text((cross_lo + cross_hi) / 2, 0.93,
                "handoff zone", ha="center",
                color=concept_color("causation"), fontsize=8.5,
                alpha=0.9, style="italic",
                transform=ax_top.get_xaxis_transform())

    ax_top.axhline(0, color=THEME["spine"], linewidth=0.9, zorder=1)

    for c in CONCEPTS:
        if c not in data:
            continue
        v    = smooth(data[c]["velocity"])
        lw   = 2.4 if c == "causation" else 1.0
        alp  = 1.0 if c in ("causation", "certainty", "temporal_order") else 0.35
        zo   = 5   if c == "causation" else 3
        ax_top.plot(L, v, color=concept_color(c), linewidth=lw, alpha=alp,
                    label=LABELS[c], zorder=zo)

    for _, cb, lcross in HANDOFF_PAIRS:
        if cb not in data:
            continue
        ax_top.axvline(lcross, color=concept_color(cb), linewidth=0.9,
                       linestyle=":", alpha=0.55, zorder=2)

    tpos, tlabs = layer_ticks(n)
    ax_top.set_xticks(tpos)
    ax_top.set_xticklabels(tlabs, color=THEME["dim"], fontsize=7.5)
    ax_top.set_xlim(0, n - 1)
    ax_top.set_ylabel("Velocity  $v(\\ell)$", color=THEME["text"], fontsize=9)
    ax_top.set_title(
        "Velocity X-crossing — all 7 concepts in GPT-2-XL (48 layers)",
        color=THEME["text"], fontsize=11, fontweight="bold", loc="left", pad=5,
    )
    ax_top.grid(axis="y", linewidth=0.4, color="#ECEFF1", zorder=0)
    ax_top.grid(axis="x", visible=False)
    for sp in ax_top.spines.values():
        sp.set_edgecolor(THEME["spine"])
    ax_top.tick_params(colors=THEME["dim"])
    ax_top.legend(
        fontsize=8, ncol=4, loc="upper left",
        facecolor="white", edgecolor=THEME["spine"],
        labelcolor=THEME["text"], handlelength=1.5,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # BOT-L: close-up X-crossing
    # ═══════════════════════════════════════════════════════════════════════════
    ax_cross = fig.add_subplot(gs[1, 0])
    ax_cross.set_facecolor("white")

    zoom_lo, zoom_hi = 14, 34
    zm = (L >= zoom_lo) & (L <= zoom_hi)
    Lz = L[zm]

    v_caus = smooth(data["causation"]["velocity"])[zm]
    v_cert = smooth(data["certainty"]["velocity"])[zm]
    v_temp = smooth(data["temporal_order"]["velocity"])[zm]

    ax_cross.axhline(0, color=THEME["spine"], linewidth=1.0, zorder=1)
    ax_cross.fill_between(Lz, v_caus, 0,
                          where=v_caus < 0,
                          alpha=0.10, color=concept_color("causation"), zorder=0)
    ax_cross.fill_between(Lz, v_cert, 0,
                          where=v_cert > 0,
                          alpha=0.10, color=concept_color("certainty"), zorder=0)

    ax_cross.plot(Lz, v_caus, color=concept_color("causation"), linewidth=2.4,
                  label="Causation", zorder=5)
    ax_cross.plot(Lz, v_cert, color=concept_color("certainty"), linewidth=2.0,
                  label="Certainty", zorder=4)
    ax_cross.plot(Lz, v_temp, color=concept_color("temporal_order"), linewidth=1.4,
                  alpha=0.75, linestyle="--", label="Temporal Order", zorder=3)

    lcross = 23
    ax_cross.axvline(lcross, color=THEME["dim"], linewidth=1.0,
                     linestyle="--", alpha=0.6, zorder=2)
    vc = float(data["causation"]["velocity"][lcross])
    ve = float(data["certainty"]["velocity"][lcross])
    ax_cross.annotate(
        f"L{lcross}  ({100*lcross/(n-1):.0f}%)\n"
        f"$v_{{caus}}$ = {vc:+.4f}\n$v_{{cert}}$ = {ve:+.4f}",
        xy=(lcross, 0), xytext=(lcross + 2, 0.045),
        color=THEME["text"], fontsize=8,
        arrowprops=dict(arrowstyle="-", color=THEME["dim"], lw=0.8),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=THEME["spine"], alpha=0.9),
    )
    ax_cross.scatter([lcross], [0], marker="x", s=130, linewidths=2.5,
                     color=THEME["text"], zorder=6)

    tpos_z, tlabs_z = layer_ticks(n, pcts=(0, 50, 100))
    in_zoom = [(p, l) for p, l in zip(tpos_z, tlabs_z) if zoom_lo <= p <= zoom_hi]
    if in_zoom:
        ax_cross.set_xticks([p for p, _ in in_zoom])
        ax_cross.set_xticklabels([l for _, l in in_zoom],
                                  color=THEME["dim"], fontsize=7.5)
    ax_cross.set_xlim(zoom_lo, zoom_hi)
    ax_cross.set_ylabel("Velocity  $v(\\ell)$", color=THEME["text"], fontsize=9)
    ax_cross.set_title(
        "Baton pass at L23  (48% depth)\n"
        "Causation $v$ flips negative exactly when Certainty flips positive",
        color=THEME["text"], fontsize=9.5, fontweight="bold", loc="left", pad=4,
    )
    ax_cross.grid(axis="y", linewidth=0.4, color="#ECEFF1", zorder=0)
    ax_cross.grid(axis="x", visible=False)
    for sp in ax_cross.spines.values():
        sp.set_edgecolor(THEME["spine"])
    ax_cross.tick_params(colors=THEME["dim"])
    ax_cross.legend(fontsize=8, facecolor="white", edgecolor=THEME["spine"],
                    labelcolor=THEME["text"], handlelength=1.5)

    # ═══════════════════════════════════════════════════════════════════════════
    # BOT-R: separation curves — causation plateaus, others climb
    # ═══════════════════════════════════════════════════════════════════════════
    ax_sep = fig.add_subplot(gs[1, 1])
    ax_sep.set_facecolor("white")

    show = ["causation", "certainty", "temporal_order", "moral_valence"]
    lws  = {"causation": 2.4}
    for c in show:
        if c not in data:
            continue
        s = data[c]["separation"]
        s_norm = s / (s.max() + 1e-8)
        ax_sep.plot(L, s_norm, color=concept_color(c),
                    linewidth=lws.get(c, 1.6),
                    label=LABELS[c], zorder=4)

    ax_sep.axvline(23, color=concept_color("causation"), linewidth=1.0,
                   linestyle="--", alpha=0.6, zorder=2)
    ax_sep.text(23.5, 0.06, "handoff L23",
                color=concept_color("causation"),
                fontsize=7.5, alpha=0.85, style="italic")

    tpos_s, tlabs_s = layer_ticks(n)
    ax_sep.set_xticks(tpos_s)
    ax_sep.set_xticklabels(tlabs_s, color=THEME["dim"], fontsize=7.5)
    ax_sep.set_xlim(0, n - 1)
    ax_sep.set_ylim(-0.02, 1.15)
    ax_sep.set_ylabel("Separation  $S(\\ell)$ (normalised)",
                      color=THEME["text"], fontsize=9)
    ax_sep.set_title(
        "Post-handoff separation\n"
        "Causation plateaus as Certainty / Temporal Order climb",
        color=THEME["text"], fontsize=9.5, fontweight="bold", loc="left", pad=4,
    )
    ax_sep.grid(axis="y", linewidth=0.4, color="#ECEFF1", zorder=0)
    ax_sep.grid(axis="x", visible=False)
    for sp in ax_sep.spines.values():
        sp.set_edgecolor(THEME["spine"])
    ax_sep.tick_params(colors=THEME["dim"])
    ax_sep.legend(fontsize=8, facecolor="white", edgecolor=THEME["spine"],
                  labelcolor=THEME["text"], handlelength=1.5)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "Velocity X-crossing: geometric bandwidth recycling in GPT-2-XL",
        color=THEME["text"], fontsize=13, fontweight="bold", y=0.995, va="bottom",
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Baton-pass velocity X-crossing figure")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID,
                        help="Model ID (e.g. openai-community/gpt2-xl)")
    parser.add_argument(
        "--out", default=str(PAPERS_DIR / "fig_baton_pass.png"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()

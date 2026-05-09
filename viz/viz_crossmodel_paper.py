#!/usr/bin/env python3
"""
viz_crossmodel_paper.py — Cross-model concept figure, paper layout.

Like viz_concept_crossmodel.py but optimized for paper inclusion:
  - Each architecture family occupies complete rows (partial rows padded with blanks)
  - Family name banner above each family group
  - Tighter row height (1.6 in vs 2.8 in)
  - Saves directly to papers/caz-validation/figures/

Usage:
    python viz_crossmodel_paper.py --concept negation
    python viz_crossmodel_paper.py --concept negation --out ~/Source/Rosetta_Program/papers/caz-validation/figures/fig_crossmodel_negation.png

Written: 2026-04-11 UTC
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).parent))
from viz_style import (
    concept_color, CONCEPT_COLORS, CONCEPT_TYPE,
    FAMILY_COLORS, FAMILY_MAP, FAMILY_ORDER, sort_models, model_label,
    THEME, layer_ticks,
)
from rosetta_tools.paths import ROSETTA_MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PAPERS_DIR = Path.home() / "Source" / "Rosetta_Program" / "papers" / "caz-validation" / "figures"

GRID_CLR  = THEME["grid"]
SPINE_CLR = THEME["spine"]
DIM_CLR   = THEME["dim"]
CKA_CLR   = THEME["cka_line"]
N_COLS    = 4


def find_model_dir(model_id: str) -> Path | None:
    slug = model_id.replace("/", "_").replace("-", "_")
    d = ROSETTA_MODELS / slug
    return d if d.exists() else None


def load_concept_for_model(model_id: str, concept: str) -> dict | None:
    model_dir = find_model_dir(model_id)
    if not model_dir:
        return None
    caz_file = model_dir / f"caz_{concept}.json"
    if not caz_file.exists():
        return None
    caz     = json.loads(caz_file.read_text())
    metrics = caz["layer_data"]["metrics"]

    return {
        "model_id":       model_id,
        "n_layers":       int(caz["n_layers"]),
        "layers":         np.array([m["layer"] for m in metrics]),
        "separation":     np.array([m["separation_fisher"] for m in metrics]),
        "cka_adj":        [],
        "fisher_regions": [],
        "peak_layer":     caz["layer_data"].get("peak_layer"),   # fallback peak
    }


def load_coasting_for_model(model_id: str, concept: str) -> list[dict]:
    # coasting_analysis data not available in rosetta_data structure
    return []


def all_model_ids() -> list[str]:
    """
    Return all model IDs that appear in FAMILY_MAP AND have a dir in ROSETTA_MODELS.
    """
    return [mid for mid in FAMILY_MAP if find_model_dir(mid) is not None]


def build_grouped_slots(rows: list[dict]) -> list[dict | None]:
    """
    Return a flat list of (row_data | None) where None = empty padding panel,
    such that each new family always starts at a column-0 position.
    """
    slots: list[dict | None] = []
    prev_fam = None
    for cd in rows:
        fam = FAMILY_MAP.get(cd["model_id"], ("Other", 0))[0]
        if fam != prev_fam:
            # Pad to next row boundary
            remainder = len(slots) % N_COLS
            if remainder > 0:
                slots.extend([None] * (N_COLS - remainder))
            prev_fam = fam
        slots.append(cd)
    return slots


def run(concept: str, args) -> None:
    model_ids = sort_models(all_model_ids())
    if not model_ids:
        log.error("No models found — run analyze_coasting.py first")
        return

    # Load data
    rows = []
    for mid in model_ids:
        cd = load_concept_for_model(mid, concept)
        if cd:
            cd["boundary_comps"] = load_coasting_for_model(mid, concept)
            rows.append(cd)

    if not rows:
        log.error("No data for concept '%s'", concept)
        return

    color  = CONCEPT_COLORS.get(concept, "#444444")
    ctype  = CONCEPT_TYPE.get(concept, "")

    # Build slot list: models grouped by family, families padded to full rows
    slots   = build_grouped_slots(rows)
    n_slots = len(slots)
    n_rows  = (n_slots + N_COLS - 1) // N_COLS

    # ── Figure ────────────────────────────────────────────────────────────────
    row_h = 1.6
    fig, axes = plt.subplots(
        n_rows, N_COLS,
        figsize=(16, row_h * n_rows + 1.0),   # +1.0 for title + legend
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(
        left=0.05, right=0.97,
        top=0.91, bottom=0.06,
        hspace=0.62, wspace=0.35,
    )

    sep_global_max = max(
        float(cd["separation"].max()) for cd in rows
    )
    sep_global_max = max(sep_global_max, 0.1)

    # Track family-group first-slot indices (for banner labels)
    family_first_slots: dict[str, int] = {}

    for slot_idx, cd in enumerate(slots):
        row_i = slot_idx // N_COLS
        col_i = slot_idx % N_COLS
        ax    = axes[row_i][col_i]

        if cd is None:
            ax.set_visible(False)
            continue

        mid        = cd["model_id"]
        fam, _     = FAMILY_MAP.get(mid, ("Other", 0))
        fam_color  = FAMILY_COLORS.get(fam, "#546E7A")

        if fam not in family_first_slots:
            family_first_slots[fam] = slot_idx

        n_layers     = cd["n_layers"]
        layers       = cd["layers"]
        sep          = cd["separation"]
        cka_adj      = cd["cka_adj"]
        fisher_reg   = cd["fisher_regions"]
        boundary_comps = cd["boundary_comps"]

        ax.set_facecolor("white")
        ax_cka = ax.twinx()
        ax_cka.set_facecolor("white")

        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_CLR)
            spine.set_linewidth(0.7)
        for spine in ax_cka.spines.values():
            spine.set_edgecolor(SPINE_CLR)
            spine.set_linewidth(0.7)

        ax.tick_params(colors=DIM_CLR, labelsize=6, length=2, width=0.6)
        ax_cka.tick_params(colors=CKA_CLR, labelsize=5.5, length=2, width=0.6)
        ax.grid(True, color=GRID_CLR, linewidth=0.4, alpha=1.0, zorder=0)

        # Fisher bands
        for reg in fisher_reg:
            ax.axvspan(reg["start"], reg["end"],
                       alpha=0.12, color=color, linewidth=0, zorder=1)

        # CKA-refined bands
        for bc in boundary_comps:
            if bc.get("dip_detected"):
                ax.axvspan(bc["cka_start"], bc["cka_end"],
                           alpha=0.45, color=color, linewidth=0, zorder=2)

        # Peak dots — from coasting boundary_comps if available, else caz peak_layer
        if boundary_comps:
            for bc in boundary_comps:
                pk = bc["peak"]
                y  = float(sep[pk]) if pk < len(sep) else 0.0
                ax.plot(pk, y, "o", color=color, markersize=3.5, zorder=6,
                        markeredgecolor="white", markeredgewidth=0.8)
        elif cd.get("peak_layer") is not None:
            pk = int(cd["peak_layer"])
            y  = float(sep[pk]) if pk < len(sep) else 0.0
            ax.plot(pk, y, "o", color=color, markersize=3.5, zorder=6,
                    markeredgecolor="white", markeredgewidth=0.8)

        # Separation curve
        ax.plot(layers, sep, color=color, linewidth=1.4, alpha=0.92, zorder=4)

        # CKA profile
        if cka_adj:
            cka_x   = np.arange(len(cka_adj)) + 0.5
            cka_arr = np.array(cka_adj)
            ax_cka.plot(cka_x, cka_arr, color=CKA_CLR, linewidth=0.75,
                        alpha=0.55, linestyle="--", zorder=3)
            cka_min = max(0.85, float(np.min(cka_arr)) - 0.01)
            ax_cka.set_ylim(cka_min, 1.005)
            ax_cka.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
            ax_cka.yaxis.set_tick_params(labelcolor=CKA_CLR)

        ax.set_ylim(0, sep_global_max * 1.12)
        ax.set_xlim(-0.5, n_layers - 0.5)

        mid_layer  = int(n_layers / 2)
        last_layer = n_layers - 1
        ax.set_xticks([0, mid_layer, last_layer])
        ax.set_xticklabels(
            [f"L0", f"L{mid_layer}", f"L{last_layer}"],
            color=DIM_CLR, fontsize=6,
        )
        ax.set_ylabel("")
        ax_cka.set_ylabel("")

        label   = model_label(mid)
        has_caz = len(fisher_reg) > 0 or cd.get("peak_layer") is not None
        marker  = "" if has_caz else "  ✗"
        ax.set_title(
            f"{label}{marker}",
            color=fam_color, fontsize=8, fontweight="bold",
            pad=3, loc="center",
        )

    # Hide unused trailing panels
    for slot_idx in range(len(slots), n_rows * N_COLS):
        row_i = slot_idx // N_COLS
        col_i = slot_idx % N_COLS
        axes[row_i][col_i].set_visible(False)

    # ── Family banners ─────────────────────────────────────────────────────────
    # Draw a bold family label above the first row of each family group.
    # We use figure-level coordinates after axes are placed.
    fig.canvas.draw()   # force layout so get_position() is accurate

    for fam, first_slot in family_first_slots.items():
        row_i     = first_slot // N_COLS
        fam_color = FAMILY_COLORS.get(fam, "#546E7A")

        # Get the left edge of col-0 and top of this row
        ax0  = axes[row_i][0]
        bbox = ax0.get_position()

        # Draw a thin colored horizontal rule above the row
        line = matplotlib.lines.Line2D(
            [bbox.x0 - 0.01, bbox.x0 + (bbox.width * N_COLS) + 0.03],
            [bbox.y1 + 0.012, bbox.y1 + 0.012],
            transform=fig.transFigure,
            color=fam_color, linewidth=1.4, alpha=0.7,
            clip_on=False,
        )
        fig.add_artist(line)

        # Family label to the left of the row
        fig.text(
            bbox.x0 - 0.015, bbox.y1 + 0.012,
            fam,
            transform=fig.transFigure,
            ha="right", va="center",
            fontsize=8.5, fontweight="bold",
            color=fam_color,
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    fisher_patch = mpatches.Patch(facecolor=color, alpha=0.15, edgecolor="#888888",
                                  linewidth=0.5, label="Fisher CAZ extent")
    cka_patch    = mpatches.Patch(facecolor=color, alpha=0.55, edgecolor="#555555",
                                  linewidth=0.5, label="CKA-refined extent")
    sep_line     = mlines.Line2D([], [], color=color, linewidth=1.5,
                                 label="Fisher separation")
    peak_dot     = mlines.Line2D([], [], color=color, linewidth=0, marker="o",
                                 markersize=4.5, markeredgecolor="white",
                                 markeredgewidth=0.8, label="Velocity peak")
    cka_line     = mlines.Line2D([], [], color=CKA_CLR, linewidth=0.9,
                                 linestyle="--", label="Adjacent-layer CKA  (right axis)")

    fig.legend(
        handles=[sep_line, peak_dot, fisher_patch, cka_patch, cka_line],
        loc="upper center",
        bbox_to_anchor=(0.50, 0.998),
        ncol=5,
        fontsize=7.5,
        facecolor="white", edgecolor=SPINE_CLR, labelcolor="#111111",
        handlelength=1.8, columnspacing=1.2, framealpha=1.0,
    )

    concept_title = concept.replace("_", " ").title()
    n_with_caz    = sum(1 for cd in rows if cd["fisher_regions"])
    n_total       = len(rows)

    fig.suptitle(
        f"{concept_title}  ·  {ctype}  ·  CAZ present in {n_with_caz}/{n_total} models",
        color="#111111", fontsize=12, fontweight="bold", y=1.00, va="bottom",
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")
    log.info("Saved: %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model figure, paper layout")
    parser.add_argument("--concept", default="negation",
                        choices=["credibility", "certainty", "negation", "causation",
                                 "temporal_order", "sentiment", "moral_valence"])
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if not args.out:
        args.out = str(PAPERS_DIR / f"fig_crossmodel_{args.concept}.png")
    run(args.concept, args)


if __name__ == "__main__":
    main()

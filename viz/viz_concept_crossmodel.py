#!/usr/bin/env python3
"""
viz_concept_crossmodel.py — One concept across all models, side by side.

Transpose of viz_cka_boundaries.py: fixes a single concept and shows every
model as a panel in a grid. Useful for spotting which models assemble a
concept clearly vs. which ones show weak or absent CAZes.

Each panel shows:
  - Fisher separation curve
  - Wide Fisher-derived CAZ band
  - Narrow CKA-refined band (where detected)
  - Velocity peak dot
  - Adjacent-layer CKA dashed line (right axis)

Models are grouped by architecture family (Pythia, OPT, Qwen, GPT-2, other)
and sorted by parameter count within each family.

Usage:
    cd caz_scaling
    python src/viz_concept_crossmodel.py --concept sentiment
    python src/viz_concept_crossmodel.py --concept moral_valence
    python src/viz_concept_crossmodel.py --all   # generate for all 7 concepts

Written: 2026-04-11 UTC
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from viz_style import (
    concept_color, CONCEPT_COLORS, CONCEPT_TYPE,
    FAMILY_COLORS, FAMILY_MAP, FAMILY_ORDER, sort_models, model_label,
    THEME, apply_theme, layer_ticks,
)
from rosetta_tools.paths import ROSETTA_RESULTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_ROOT = ROSETTA_RESULTS
OUT_DIR      = Path("visualizations") / "crossmodel"

CONCEPTS = [
    "credibility", "certainty", "negation",
    "causation", "temporal_order", "sentiment", "moral_valence",
]

GRID_CLR  = THEME["grid"]
SPINE_CLR = THEME["spine"]
DIM_CLR   = THEME["dim"]
CKA_CLR   = THEME["cka_line"]


# ── Data loading ───────────────────────────────────────────────────────────────

def find_run_dir(model_id: str) -> Path | None:
    for d in sorted(RESULTS_ROOT.iterdir(), reverse=True):
        sf = d / "run_summary.json"
        if d.is_dir() and sf.exists():
            try:
                if json.loads(sf.read_text()).get("model_id") == model_id:
                    if any(d.glob("cka_*.json")):
                        return d
            except Exception:
                continue
    return None


def load_concept_for_model(model_id: str, concept: str) -> dict | None:
    run_dir = find_run_dir(model_id)
    if not run_dir:
        return None
    caz_file = run_dir / f"caz_{concept}.json"
    cka_file = run_dir / f"cka_{concept}.json"
    if not caz_file.exists() or not cka_file.exists():
        return None
    caz = json.loads(caz_file.read_text())
    cka = json.loads(cka_file.read_text())
    metrics    = caz["layer_data"]["metrics"]
    return {
        "model_id":      model_id,
        "n_layers":      int(caz["n_layers"]),
        "layers":        np.array([m["layer"] for m in metrics]),
        "separation":    np.array([m["separation_fisher"] for m in metrics]),
        "cka_adj":       cka.get("cka_adjacent", []),
        "fisher_regions": cka.get("caz_regions", []),
    }


def load_coasting_for_model(model_id: str, concept: str) -> list[dict]:
    pm = RESULTS_ROOT / "coasting_analysis" / "per_model.json"
    if not pm.exists():
        return []
    data = json.loads(pm.read_text())
    for m in data:
        if m["model_id"] == model_id:
            return m["concept_results"].get(concept, {}).get("boundary_comparisons", [])
    return []


def all_model_ids() -> list[str]:
    pm = RESULTS_ROOT / "coasting_analysis" / "per_model.json"
    if not pm.exists():
        return []
    return [m["model_id"] for m in json.loads(pm.read_text())]


# ── Figure builder ─────────────────────────────────────────────────────────────

def build_figure(concept: str, model_ids: list[str]) -> object | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    import matplotlib.ticker as ticker

    # Load data for all models that have this concept
    rows = []
    for mid in model_ids:
        cd = load_concept_for_model(mid, concept)
        if cd:
            cd["boundary_comps"] = load_coasting_for_model(mid, concept)
            rows.append(cd)

    if not rows:
        log.warning("No data for concept '%s'", concept)
        return None

    n_models = len(rows)
    n_cols   = 4
    n_rows   = (n_models + n_cols - 1) // n_cols

    color    = CONCEPT_COLORS.get(concept, "#444444")
    ctype    = CONCEPT_TYPE.get(concept, "")

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(16, 2.8 * n_rows),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(
        left=0.05, right=0.97,
        top=0.91, bottom=0.06,
        hspace=0.70, wspace=0.35,
    )

    # Global separation max for consistent y-scale within concept
    sep_global_max = max(float(cd["separation"].max()) for cd in rows)
    sep_global_max = max(sep_global_max, 0.1)   # floor so empty panels don't crash

    for idx, cd in enumerate(rows):
        row_i = idx // n_cols
        col_i = idx % n_cols
        ax    = axes[row_i][col_i]

        mid          = cd["model_id"]
        family, _    = FAMILY_MAP.get(mid, ("Other", 0))
        fam_color    = FAMILY_colors = FAMILY_COLORS.get(family, "#546E7A")

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

        ax.tick_params(colors=DIM_CLR, labelsize=6.5, length=2, width=0.6)
        ax_cka.tick_params(colors=CKA_CLR, labelsize=6, length=2, width=0.6)
        ax.grid(True, color=GRID_CLR, linewidth=0.5, alpha=1.0, zorder=0)

        # Fisher bands
        for reg in fisher_reg:
            ax.axvspan(reg["start"], reg["end"],
                       alpha=0.12, color=color, linewidth=0, zorder=1)

        # CKA-refined bands
        for bc in boundary_comps:
            if bc.get("dip_detected"):
                ax.axvspan(bc["cka_start"], bc["cka_end"],
                           alpha=0.45, color=color, linewidth=0, zorder=2)

        # Peak dots
        for bc in boundary_comps:
            pk = bc["peak"]
            y  = float(sep[pk]) if pk < len(sep) else 0.0
            ax.plot(pk, y, "o", color=color, markersize=4.5, zorder=6,
                    markeredgecolor="white", markeredgewidth=0.9)

        # Separation curve
        ax.plot(layers, sep, color=color, linewidth=1.5, alpha=0.92, zorder=4)

        # CKA profile
        if cka_adj:
            cka_x   = np.arange(len(cka_adj)) + 0.5
            cka_arr = np.array(cka_adj)
            ax_cka.plot(cka_x, cka_arr, color=CKA_CLR, linewidth=0.85,
                        alpha=0.55, linestyle="--", zorder=3)
            cka_min = max(0.85, float(np.min(cka_arr)) - 0.01)
            ax_cka.set_ylim(cka_min, 1.005)
            ax_cka.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
            ax_cka.yaxis.set_tick_params(labelcolor=CKA_CLR)

        # Consistent y-scale across all panels
        ax.set_ylim(0, sep_global_max * 1.12)
        ax.set_xlim(-0.5, n_layers - 0.5)

        # X-ticks: just layer counts at 0%, 50%, 100%
        mid_layer  = int(n_layers / 2)
        last_layer = n_layers - 1
        ax.set_xticks([0, mid_layer, last_layer])
        ax.set_xticklabels(
            [f"L0", f"L{mid_layer}\n(50%)", f"L{last_layer}"],
            color=DIM_CLR, fontsize=6.5,
        )

        # No y-label on interior panels — too cluttered
        ax.set_ylabel("")
        ax_cka.set_ylabel("")

        # Model label as panel title, colored by family
        label = model_label(mid)
        has_caz = len(fisher_reg) > 0
        marker  = "" if has_caz else "  ✗"
        ax.set_title(
            f"{label}{marker}",
            color=fam_color, fontsize=8.5, fontweight="bold",
            pad=3, loc="center",
        )

        # Faint family tag in corner
        ax.text(0.98, 0.95, family, transform=ax.transAxes,
                ha="right", va="top", fontsize=6, color=fam_color, alpha=0.6)

    # Hide unused panels
    for idx in range(n_models, n_rows * n_cols):
        row_i = idx // n_cols
        col_i = idx % n_cols
        axes[row_i][col_i].set_visible(False)

    # ── Legend ──
    fisher_patch = mpatches.Patch(
        facecolor=color, alpha=0.15, edgecolor="#888888", linewidth=0.5,
        label="Fisher CAZ extent",
    )
    cka_patch = mpatches.Patch(
        facecolor=color, alpha=0.55, edgecolor="#555555", linewidth=0.5,
        label="CKA-refined extent",
    )
    sep_line = mlines.Line2D(
        [], [], color=color, linewidth=1.5,
        label="Fisher separation",
    )
    peak_dot = mlines.Line2D(
        [], [], color=color, linewidth=0, marker="o",
        markersize=5, markeredgecolor="white", markeredgewidth=0.9,
        label="Velocity peak",
    )
    cka_line = mlines.Line2D(
        [], [], color=CKA_CLR, linewidth=0.9, linestyle="--",
        label="Adjacent-layer CKA  (right axis)",
    )
    # Family color legend entries
    family_patches = [
        mpatches.Patch(color=FAMILY_COLORS[f], label=f)
        for f in FAMILY_ORDER
        if any(FAMILY_MAP.get(m, ("",))[0] == f for m in model_ids)
    ]

    fig.legend(
        handles=[sep_line, peak_dot, fisher_patch, cka_patch, cka_line] + family_patches,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        ncol=5,
        fontsize=7.5,
        facecolor="white",
        edgecolor=SPINE_CLR,
        labelcolor="#111111",
        handlelength=1.8,
        columnspacing=1.2,
        framealpha=1.0,
    )

    concept_title = concept.replace("_", " ").title()
    n_with_caz    = sum(1 for cd in rows if cd["fisher_regions"])
    n_total       = len(rows)

    fig.suptitle(
        f"{concept_title}  ·  {ctype}  ·  CAZ present in {n_with_caz}/{n_total} models",
        color="#111111", fontsize=13, fontweight="bold", y=1.00, va="bottom",
    )

    return fig


# ── Entry point ────────────────────────────────────────────────────────────────

def save_for_concept(concept: str):
    model_ids = sort_models(all_model_ids())
    if not model_ids:
        log.error("No models found — run analyze_coasting.py first")
        return

    log.info("Building cross-model figure for '%s' (%d models)...",
             concept, len(model_ids))
    fig = build_figure(concept, model_ids)
    if fig is None:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"crossmodel_{concept}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    log.info("Saved: %s", out)

    import matplotlib.pyplot as plt
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(
        description="One concept across all models (cross-model comparison)"
    )
    parser.add_argument(
        "--concept", type=str, default="sentiment",
        choices=CONCEPTS,
        help="Concept to visualize across all models",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate figures for all 7 concepts",
    )
    args = parser.parse_args()

    if args.all:
        for c in CONCEPTS:
            save_for_concept(c)
    else:
        save_for_concept(args.concept)


if __name__ == "__main__":
    main()

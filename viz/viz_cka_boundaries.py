#!/usr/bin/env python3
"""
viz_cka_boundaries.py — CAZ graphs with CKA-refined boundaries (print-ready).

White-background figure for PDF publication. Each concept gets one panel:
  - Solid line: Fisher separation score (left y-axis)
  - Wide shaded band: Fisher-derived CAZ extent (full velocity-threshold region)
  - Narrow shaded band: CKA-refined extent (active-transformation window only)
  - Filled circle: Fisher velocity peak (where concept assembly peaks)
  - Dashed line: adjacent-layer CKA similarity (right y-axis)

The first panel is annotated with callout arrows so readers do not need to
study the legend. Remaining panels are clean.

Usage:
    cd caz_scaling
    python src/viz_cka_boundaries.py
    python src/viz_cka_boundaries.py --model EleutherAI/pythia-1.4b
    python src/viz_cka_boundaries.py --model Qwen/Qwen2.5-3B
    python src/viz_cka_boundaries.py --all   # one figure per model

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
    THEME, apply_theme, layer_ticks, add_outside_callouts,
)
from rosetta_tools.paths import ROSETTA_RESULTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_ROOT = ROSETTA_RESULTS
OUT_DIR      = Path("visualizations") / "cka_boundaries"

CONCEPTS = [
    "credibility", "certainty", "negation",
    "causation", "temporal_order", "sentiment", "moral_valence",
]

CKA_CLR   = THEME["cka_line"]
GRID_CLR  = THEME["grid"]
SPINE_CLR = THEME["spine"]
TEXT_CLR  = THEME["text"]
DIM_CLR   = THEME["dim"]
ANNOT_CLR = THEME["annot"]


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


def load_concept_data(run_dir: Path, concept: str) -> dict | None:
    caz_file = run_dir / f"caz_{concept}.json"
    cka_file = run_dir / f"cka_{concept}.json"
    if not caz_file.exists() or not cka_file.exists():
        return None

    caz = json.loads(caz_file.read_text())
    cka = json.loads(cka_file.read_text())

    metrics    = caz["layer_data"]["metrics"]
    layers     = [m["layer"] for m in metrics]
    separation = [m["separation_fisher"] for m in metrics]
    n_layers   = int(caz["n_layers"])
    cka_adj    = cka.get("cka_adjacent", [])
    fisher_reg = cka.get("caz_regions", [])   # [{start, peak, end}]

    return {
        "concept":        concept,
        "n_layers":       n_layers,
        "layers":         np.array(layers),
        "separation":     np.array(separation),
        "cka_adj":        cka_adj,
        "fisher_regions": fisher_reg,
    }


def load_coasting_for_model(model_id: str) -> dict | None:
    pm = RESULTS_ROOT / "coasting_analysis" / "per_model.json"
    if not pm.exists():
        return None
    data = json.loads(pm.read_text())
    for m in data:
        if m["model_id"] == model_id:
            return m["concept_results"]
    return None


def detrend_cka(cka: list[float], smooth_window: int = 5) -> np.ndarray:
    arr      = np.array(cka, dtype=np.float64)
    baseline = uniform_filter1d(arr, size=smooth_window, mode="nearest")
    return arr - baseline


# ── Callout helpers ────────────────────────────────────────────────────────────

def _assign_tiers(callouts: list[dict], n_layers: int) -> list[tuple[str, float]]:
    """
    Assign each callout to one of four slots: two rows above axes, two below.

    Slots in priority order:
      0  top-near   y= 1.09   (closest above, tried first)
      1  bot-near   y=-0.10   (closest below)
      2  top-far    y= 1.22   (second row above, used when top-near is crowded)
      3  bot-far    y=-0.24   (second row below)

    Greedy: for each callout (sorted by x), pick the first slot whose right
    edge doesn't collide with the new label. Falls back to the least-crowded
    slot if all four are full.

    Returns list of (va_side, y_frac) tuples in the original callout order.
    """
    if not callouts:
        return []

    # ~7px per character, figure 13 in wide, axes ≈ 84% of width → 787 px
    # → 1 layer ≈ 787/n_layers px → char width in layer units:
    char_w = 7.0 * n_layers / 787.0

    SLOTS = [
        ("top",    1.09),
        ("bottom", -0.10),
        ("top",    1.22),
        ("bottom", -0.24),
    ]

    sorted_idx     = sorted(range(len(callouts)), key=lambda i: callouts[i]["x"])
    slot_right     = [-999.0] * len(SLOTS)   # right edge of last label per slot
    result         = [None]   * len(callouts)

    for orig_i in sorted_idx:
        c        = callouts[orig_i]
        x        = c["x"]
        max_line = max(len(ln) for ln in c["label"].split("\n"))
        half_w   = max_line * char_w / 2.0 + 1.0

        placed = False
        for s_i, (side, y_frac) in enumerate(SLOTS):
            if (x - half_w) > slot_right[s_i]:
                result[orig_i]  = (side, y_frac)
                slot_right[s_i] = x + half_w
                placed          = True
                break

        if not placed:
            # All slots crowded — pick least-crowded
            best = min(range(len(SLOTS)), key=lambda i: slot_right[i])
            result[orig_i]   = SLOTS[best]
            slot_right[best] = x + half_w

    return result


def _add_callouts(ax, ax_cka, sep, cka_adj, fisher_reg,
                  boundary_comps, n_layers, color, cka_plotted):
    """
    Draw outside-the-axes callouts for every element on the first panel.

    Each callout is a straight vertical line from a label box placed above
    (tier='top', y > 1) or below (tier='bottom', y < 0) the axes down to the
    feature of interest. Labels do not overlap because _assign_tiers staggers
    them when x-positions are close.
    """
    if not fisher_reg:
        return

    # ── Collect callout points ──────────────────────────────────────────────
    # Each entry: {x, y_data, label, color, axis}
    # axis: 'sep' → primary ax, 'cka' → ax_cka
    points = []

    sep_arr = np.array(sep)
    sep_max = float(sep_arr.max()) if len(sep_arr) else 1.0

    # 1. Fisher band — centre of first region
    reg0  = fisher_reg[0]
    mid_f = (reg0["start"] + reg0["end"]) / 2.0
    y_f   = float(sep_arr[min(int(mid_f), len(sep_arr) - 1)])
    points.append({
        "x":     mid_f,
        "y":     y_f * 0.5,          # point to middle-height of band
        "label": "Fisher CAZ extent\n(velocity threshold)",
        "color": color,
        "axis":  "sep",
    })

    # 2. CKA-refined band — centre of first detected dip
    for bc in boundary_comps:
        if bc.get("dip_detected"):
            mid_c = (bc["cka_start"] + bc["cka_end"]) / 2.0
            y_c   = float(sep_arr[min(int(mid_c), len(sep_arr) - 1)])
            points.append({
                "x":     mid_c,
                "y":     y_c,
                "label": "CKA-refined extent\n(active-transformation\nwindow only)",
                "color": color,
                "axis":  "sep",
            })
            break

    # 3. Velocity peak (first boundary comp peak)
    if boundary_comps:
        pk0  = boundary_comps[0]["peak"]
        y_pk = float(sep_arr[pk0]) if pk0 < len(sep_arr) else sep_max
        points.append({
            "x":     float(pk0),
            "y":     y_pk,
            "label": "Velocity peak\n(max assembly rate)",
            "color": ANNOT_CLR,
            "axis":  "sep",
        })

    # 4. CKA dashed curve — pick point at 60 % depth where curve is visible
    if cka_plotted and cka_adj:
        cka_arr = np.array(cka_adj)
        ann_xi  = min(int(n_layers * 0.60), len(cka_arr) - 1)
        y_cka   = float(cka_arr[ann_xi])
        points.append({
            "x":     float(ann_xi) + 0.5,
            "y":     y_cka,
            "label": "Adjacent-layer CKA\n(right axis; dips =\nactive transformation)",
            "color": CKA_CLR,
            "axis":  "cka",
        })

    # ── Assign top / bottom tiers ────────────────────────────────────────────
    tiers = _assign_tiers(points, n_layers)

    for pt, (side, y_label) in zip(points, tiers):
        target_ax = ax if pt["axis"] == "sep" else ax_cka
        va        = "bottom" if side == "top" else "top"

        target_ax.annotate(
            pt["label"],
            # Arrow tip: data coordinates at the feature
            xy=(pt["x"], pt["y"]),
            xycoords="data",
            # Label box: same x (data), but y outside axes (axes fraction)
            xytext=(pt["x"], y_label),
            textcoords=("data", "axes fraction"),
            ha="center",
            va=va,
            color=pt["color"],
            fontsize=7.5,
            fontweight="bold",
            clip_on=False,
            arrowprops=dict(
                arrowstyle="-",          # plain line, no arrowhead
                color=pt["color"],
                lw=0.9,
                shrinkA=4,              # gap between label box and line start
                shrinkB=3,              # gap between line end and data point
            ),
            bbox=dict(
                boxstyle="round,pad=0.28",
                fc="white",
                ec=pt["color"],
                lw=0.8,
                alpha=0.97,
            ),
            zorder=20,
        )


# ── Figure builder ─────────────────────────────────────────────────────────────

def build_figure(model_id: str, run_dir: Path, coasting: dict | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    import matplotlib.ticker as ticker

    concept_data = {}
    for concept in CONCEPTS:
        cd = load_concept_data(run_dir, concept)
        if cd:
            concept_data[concept] = cd

    if not concept_data:
        log.warning("No concept data for %s", model_id)
        return None

    n_concepts  = len(concept_data)
    model_short = model_id.split("/")[-1]

    fig, axes = plt.subplots(
        n_concepts, 1,
        figsize=(13, 2.5 * n_concepts),
        sharex=False,
    )
    if n_concepts == 1:
        axes = [axes]

    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.04, hspace=0.65)

    total_fisher_width = []
    total_cka_width    = []

    for panel_idx, (ax, concept) in enumerate(zip(axes, concept_data.keys())):
        cd       = concept_data[concept]
        color    = CONCEPT_COLORS.get(concept, "#444444")
        ctype    = CONCEPT_TYPE.get(concept, "")
        is_first = (panel_idx == 0)

        n_layers   = cd["n_layers"]
        layers     = cd["layers"]
        sep        = cd["separation"]
        cka_adj    = cd["cka_adj"]
        fisher_reg = cd["fisher_regions"]

        coast_concept  = (coasting or {}).get(concept, {})
        boundary_comps = coast_concept.get("boundary_comparisons", [])

        cka_extents: dict[int, tuple[int, int]] = {}
        for bc in boundary_comps:
            if bc.get("dip_detected"):
                cka_extents[bc["peak"]] = (bc["cka_start"], bc["cka_end"])
                total_fisher_width.append(bc["fisher_width"])
                total_cka_width.append(bc["cka_width"])

        # ── Axes setup ──
        ax.set_facecolor("white")
        ax_cka = ax.twinx()
        ax_cka.set_facecolor("white")

        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_CLR)
            spine.set_linewidth(0.8)
        for spine in ax_cka.spines.values():
            spine.set_edgecolor(SPINE_CLR)
            spine.set_linewidth(0.8)

        ax.tick_params(colors=DIM_CLR, labelsize=7.5, length=3, width=0.7)
        ax_cka.tick_params(colors=CKA_CLR, labelsize=7, length=3, width=0.7)
        ax.grid(True, color=GRID_CLR, linewidth=0.6, alpha=1.0, axis="both", zorder=0)

        # ── 1. Wide Fisher bands ──
        for reg in fisher_reg:
            ax.axvspan(
                reg["start"], reg["end"],
                alpha=0.12, color=color, linewidth=0, zorder=1,
            )

        # ── 2. Narrow CKA bands ──
        for bc in boundary_comps:
            if bc.get("dip_detected"):
                ax.axvspan(
                    bc["cka_start"], bc["cka_end"],
                    alpha=0.45, color=color, linewidth=0, zorder=2,
                )

        # ── 3. Fisher peak circles ──
        for bc in boundary_comps:
            pk           = bc["peak"]
            sep_at_peak  = float(sep[pk]) if pk < len(sep) else 0.0
            ax.plot(
                pk, sep_at_peak,
                "o", color=color, markersize=6.5, zorder=6,
                markeredgecolor="white", markeredgewidth=1.2,
            )

        # ── 4. Separation curve ──
        ax.plot(layers, sep, color=color, linewidth=1.8, alpha=0.95, zorder=4)

        # ── 5. CKA profile ──
        cka_plotted = False
        if cka_adj:
            cka_x   = np.arange(len(cka_adj)) + 0.5
            cka_arr = np.array(cka_adj)
            ax_cka.plot(
                cka_x, cka_arr,
                color=CKA_CLR, linewidth=1.1, alpha=0.65,
                linestyle="--", zorder=3,
            )
            cka_min = max(0.85, float(np.min(cka_arr)) - 0.01)
            ax_cka.set_ylim(cka_min, 1.005)
            ax_cka.set_ylabel("Adjacent-layer\nCKA similarity", color=CKA_CLR,
                               fontsize=6.5, labelpad=4, linespacing=1.3)
            ax_cka.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
            ax_cka.yaxis.set_tick_params(labelcolor=CKA_CLR)
            cka_plotted = True

        # ── Axes limits and labels ──
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_ylabel("Fisher separation", color=DIM_CLR, fontsize=7, labelpad=4)

        # Tick labels: "L{n}\n({pct}%)" — layer count primary, % secondary
        tick_pcts   = [0, 25, 50, 75, 100]
        tick_layers = [int(p / 100 * (n_layers - 1)) for p in tick_pcts]
        tick_labels = [f"L{l}\n({p}%)" for l, p in zip(tick_layers, tick_pcts)]
        ax.set_xticks(tick_layers)
        ax.set_xticklabels(tick_labels, color=DIM_CLR, fontsize=7,
                           linespacing=1.1)

        # Concept label — name bold · type · depth
        ax.set_title(
            f"{concept.replace('_', ' ').title()}",
            color=color, fontsize=10, fontweight="bold",
            loc="left", pad=4,
        )
        ax.text(
            0.19, 1.01,
            f"{ctype}  ·  {n_layers} layers",
            transform=ax.transAxes,
            color=DIM_CLR, fontsize=8, va="bottom",
        )

        # ── Callout annotations on first panel only ──
        if is_first and len(fisher_reg) > 0:
            _add_callouts(ax, ax_cka, sep, cka_adj, fisher_reg,
                          boundary_comps, n_layers, color, cka_plotted)

    # ── Summary stats for legend ──
    fw_mean = np.mean(total_fisher_width) if total_fisher_width else 0.0
    cw_mean = np.mean(total_cka_width)    if total_cka_width    else 0.0
    ratio   = cw_mean / fw_mean if fw_mean else 0.0

    # ── Legend ──
    fisher_patch = mpatches.Patch(
        facecolor="#aaaaaa", alpha=0.2, edgecolor="#888888", linewidth=0.6,
        label=f"Fisher CAZ extent  (mean {fw_mean:.1f} layers per region)",
    )
    cka_patch = mpatches.Patch(
        facecolor="#888888", alpha=0.55, edgecolor="#555555", linewidth=0.6,
        label=f"CKA-refined extent  (mean {cw_mean:.1f} layers, {ratio:.2f}× narrower)",
    )
    sep_line = mlines.Line2D(
        [], [], color="#555555", linewidth=1.8,
        label="Fisher separation score  (left axis)",
    )
    peak_dot = mlines.Line2D(
        [], [], color="#555555", linewidth=0, marker="o",
        markersize=6.5, markeredgecolor="white", markeredgewidth=1.2,
        label="Velocity peak  (max concept assembly rate)",
    )
    cka_line = mlines.Line2D(
        [], [], color=CKA_CLR, linewidth=1.1, linestyle="--",
        label="Adjacent-layer CKA  (right axis; dips = active transformation)",
    )

    fig.legend(
        handles=[sep_line, peak_dot, fisher_patch, cka_patch, cka_line],
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        ncol=3,
        fontsize=8,
        facecolor="white",
        edgecolor=SPINE_CLR,
        labelcolor=TEXT_CLR,
        handlelength=2.0,
        handleheight=1.2,
        columnspacing=1.4,
        framealpha=1.0,
        title="Figure elements",
        title_fontsize=8,
    )

    fig.suptitle(
        f"CKA-Refined CAZ Boundaries — {model_short}",
        color=TEXT_CLR, fontsize=13, fontweight="bold", y=1.02, va="bottom",
    )

    return fig


# ── Entry point ────────────────────────────────────────────────────────────────

def save_for_model(model_id: str):
    run_dir = find_run_dir(model_id)
    if not run_dir:
        log.warning("No CKA run dir for %s — skipping", model_id)
        return

    coasting = load_coasting_for_model(model_id)
    if not coasting:
        log.warning("No coasting analysis for %s — bands will be Fisher-only", model_id)

    log.info("Building figure for %s...", model_id)
    fig = build_figure(model_id, run_dir, coasting)
    if fig is None:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = model_id.replace("/", "_").replace("-", "_")
    out  = OUT_DIR / f"cka_boundaries_{slug}.png"
    fig.savefig(str(out), dpi=160, bbox_inches="tight", facecolor="white")
    log.info("Saved: %s", out)

    import matplotlib.pyplot as plt
    plt.close("all")


def all_models() -> list[str]:
    pm = RESULTS_ROOT / "coasting_analysis" / "per_model.json"
    if not pm.exists():
        log.error("No per_model.json — run analyze_coasting.py first")
        return []
    data = json.loads(pm.read_text())
    return [m["model_id"] for m in data]


def main():
    parser = argparse.ArgumentParser(
        description="CAZ graphs with CKA-refined boundaries (print-ready)"
    )
    parser.add_argument(
        "--model", type=str,
        default="EleutherAI/pythia-1.4b",
        help="Model ID to visualize",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate figures for all models in coasting analysis",
    )
    args = parser.parse_args()

    if args.all:
        for mid in all_models():
            save_for_model(mid)
    else:
        save_for_model(args.model)


if __name__ == "__main__":
    main()

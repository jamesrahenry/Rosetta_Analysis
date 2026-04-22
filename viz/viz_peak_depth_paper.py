#!/usr/bin/env python3
"""
viz_peak_depth_paper.py — Compact peak-depth heatmap for the paper.

Generates a tighter version of the combined peak-depth heatmap:
  - Row height 0.40 per model (vs 0.70 in the analysis version — ~11 in vs ~18 in)
  - Family separator lines + family labels on the left margin
  - Family-ordered rows, concept-ordered columns
  - Nicer model labels (e.g. "Pythia-1.4B" instead of "pythia-1.4b")

Usage:
    cd caz_scaling
    python src/viz_peak_depth_paper.py
    python src/viz_peak_depth_paper.py --out ../../papers/caz-validation/figures/fig_peak_depth_heatmap.png

Written: 2026-04-11 UTC
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from viz_style import FAMILY_MAP, FAMILY_COLORS, FAMILY_ORDER, sort_models, model_label, THEME

CAZ_ROOT    = Path(__file__).resolve().parents[1]
RESULTS_DIR = CAZ_ROOT / "results"
PAPERS_DIR  = CAZ_ROOT.parent / "papers" / "caz-validation" / "figures"

CONCEPT_ORDER = [
    "temporal_order", "causation",
    "negation",
    "sentiment", "moral_valence",
    "certainty", "credibility",
]
CONCEPT_LABELS = {
    "temporal_order": "Temporal\nOrder",
    "causation":      "Causation",
    "negation":       "Negation",
    "sentiment":      "Sentiment",
    "moral_valence":  "Moral\nValence",
    "certainty":      "Certainty",
    "credibility":    "Credibility",
}


def discover_result_dirs() -> list[Path]:
    """Return model result dirs (must have run_summary.json with model_id)."""
    dirs = []
    if not RESULTS_DIR.exists():
        return dirs
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir() and (d / "run_summary.json").exists():
            try:
                import json as _json
                summary = _json.loads((d / "run_summary.json").read_text())
                if "model_id" in summary:
                    dirs.append(d)
            except Exception:
                continue
    return dirs


def build_pivot(dirs: list[Path]) -> "pd.DataFrame":
    import pandas as pd
    from rosetta_tools.reporting import load_results_dir

    df = load_results_dir(dirs)
    if df is None or df.empty:
        raise RuntimeError("No data loaded — run extract.py first")

    peaks = df[df["is_peak"]].copy()
    pivot = peaks.pivot_table(
        index="model_id", columns="concept", values="depth_pct", aggfunc="first"
    )
    # Concept column order
    col_order = [c for c in CONCEPT_ORDER if c in pivot.columns]
    pivot = pivot[col_order]

    # Filter to known base models only (exclude instruct/chat variants)
    known = [m for m in pivot.index if m in FAMILY_MAP]
    pivot = pivot.loc[known]

    # Model column order: by family, then by scale within family
    ordered = sort_models(pivot.index.tolist())
    pivot = pivot.loc[ordered]

    # Transpose: concepts on rows (y-axis), models on columns (x-axis)
    # This gives 7 rows × 24 cols — wide and compact, matching the paper caption.
    pivot = pivot.T

    # Row order: concepts by mean depth (shallowest at top → deepest at bottom)
    concept_mean_depth = pivot.mean(axis=1)
    pivot = pivot.loc[concept_mean_depth.sort_values().index]

    return pivot


def run(args) -> None:
    import pandas as pd

    dirs   = discover_result_dirs()
    pivot  = build_pivot(dirs)

    n_rows = len(pivot)
    n_cols = len(pivot.columns)

    # After transposing: pivot.index = concepts (rows), pivot.columns = models (cols)
    # ── Family metadata (for columns now) ─────────────────────────────────────
    fam_of = {mid: FAMILY_MAP.get(mid, ("Other", 0))[0] for mid in pivot.columns}
    fams_in_order: list[str] = []
    seen_fams: set = set()
    for mid in pivot.columns:
        f = fam_of[mid]
        if f not in seen_fams:
            fams_in_order.append(f)
            seen_fams.add(f)

    # Column index where each family starts / ends
    fam_start: dict[str, int] = {}
    fam_end:   dict[str, int] = {}
    prev = None
    for j, mid in enumerate(pivot.columns):
        f = fam_of[mid]
        if f != prev:
            fam_start[f] = j
        fam_end[f] = j
        prev = f

    # ── Figure — wide and short ───────────────────────────────────────────────
    row_h   = 0.65          # 7 concepts × 0.65 = 4.55 in
    col_w   = 0.55          # 24 models × 0.55 = 13.2 in
    top_margin    = 1.0     # title + x-tick labels
    bottom_margin = 0.6     # family labels below x-axis
    left_margin   = 1.3     # concept labels
    right_margin  = 0.7     # colorbar

    fig_w = left_margin + n_cols * col_w + right_margin
    fig_h = top_margin + n_rows * row_h + bottom_margin

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    vals = pivot.values.astype(float)
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100,
                   origin="upper")

    # ── Cell annotations ──────────────────────────────────────────────────────
    for i in range(n_rows):
        for j in range(n_cols):
            v = vals[i, j]
            if np.isnan(v):
                continue
            text_color = "white" if (v < 20 or v > 80) else "#222222"
            ax.text(j, i, f"{v:.0f}",
                    ha="center", va="center",
                    fontsize=5.5, color=text_color)

    # ── Family separator lines on x-axis (columns) ────────────────────────────
    for f in fams_in_order:
        start_j = fam_start[f]
        end_j   = fam_end[f]
        mid_j   = (start_j + end_j) / 2.0
        fam_col = FAMILY_COLORS.get(f, "#546E7A")

        # Vertical separator before family group (skip first family)
        if start_j > 0:
            ax.axvline(start_j - 0.5, color="white", linewidth=2.5, zorder=5)
            ax.axvline(start_j - 0.5, color=fam_col, linewidth=1.0,
                       linestyle="--", alpha=0.6, zorder=6)

        # Family label below x-axis
        ax.text(mid_j, n_rows - 0.5 + 0.8, f,
                ha="center", va="top", fontsize=7.5, fontweight="bold",
                color=fam_col, transform=ax.transData, clip_on=False)

    # ── Axes ticks ────────────────────────────────────────────────────────────
    # Y-axis: concept names (rows), colored by concept
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [CONCEPT_LABELS.get(c, c).replace("\n", " ") for c in pivot.index],
        fontsize=9, color=THEME["text"],
    )

    # X-axis: model names (columns), colored by family
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [model_label(mid) for mid in pivot.columns],
        fontsize=6.5, rotation=45, ha="left",
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    for sp in ax.spines.values():
        sp.set_edgecolor(THEME["spine"])

    ax.tick_params(axis="both", length=0, pad=3)

    # Color each x-tick (model) label by family; bold the first of each family
    fig.canvas.draw()
    for j, mid in enumerate(pivot.columns):
        fam = FAMILY_MAP.get(mid, ("Other", 0))[0]
        col = FAMILY_COLORS.get(fam, "#546E7A")
        lbl = ax.xaxis.get_ticklabels()[j]
        lbl.set_color(col)
        if j == fam_start.get(fam, -1):
            lbl.set_fontweight("bold")

    # ── Colorbar ──────────────────────────────────────────────────────────────
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.ax.tick_params(colors=THEME["dim"], labelsize=7.5)
    cb.set_label("Peak depth (%)", color=THEME["dim"], fontsize=8)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        "CAZ Peak Depth — All Families",
        color=THEME["text"], fontsize=12, fontweight="bold",
        y=1.0, va="bottom",
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact peak-depth heatmap for paper")
    parser.add_argument(
        "--out",
        default=str(PAPERS_DIR / "fig_peak_depth_heatmap.png"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()

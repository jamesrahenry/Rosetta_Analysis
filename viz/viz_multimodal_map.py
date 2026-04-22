#!/usr/bin/env python3
"""
viz_multimodal_map.py — Visualize the multimodal CAZ discovery.

Generates a 4-panel figure showing:
  (A) Example separation curve with two shaded assembly regions
  (B) Shallow vs deep peak locations across all multimodal models
  (C) Direction divergence — cosine similarity between peak dom_vectors
  (D) Overlaid separation curves across all multimodal models

Usage:
    python src/viz_multimodal_map.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

matplotlib.use("Agg")
from rosetta_tools.caz import LayerMetrics, find_caz_regions
from rosetta_tools.viz import CONCEPT_META
from viz_style import concept_color

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = Path(__file__).resolve().parents[1] / "visualizations" / "multimodal"

# Consistent style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "figure.dpi": 150,
})

FAMILY_MARKERS = {
    "pythia": "o",
    "gpt2": "s",
    "opt": "D",
    "qwen": "^",
    "gemma": "v",
}
FAMILY_COLORS = {            # lowercase keys — matched by _model_family()
    "pythia": "#1565C0",    # spec: Pythia blue
    "gpt2":   "#558B2F",    # spec: GPT-2 green
    "opt":    "#6A1B9A",    # spec: OPT purple
    "qwen":   "#E65100",    # spec: Qwen orange
    "gemma":  "#00695C",    # spec: Gemma teal
}


def _model_family(model_id: str) -> str:
    mid = model_id.lower()
    for fam in FAMILY_MARKERS:
        if fam in mid:
            return fam
    return "other"


def _short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def load_multimodal_data(concept: str = "credibility"):
    """Load all models and identify multimodal ones for a given concept."""
    records = []
    for result_dir in sorted(RESULTS_DIR.iterdir()):
        caz_file = result_dir / f"caz_{concept}.json"
        if not caz_file.exists():
            continue

        with open(caz_file) as f:
            data = json.load(f)

        model_id = data["model_id"]
        metrics_raw = data["layer_data"]["metrics"]
        n_layers = int(data["n_layers"])

        layer_metrics = [
            LayerMetrics(m["layer"], m["separation_fisher"], m["coherence"], m["velocity"])
            for m in metrics_raw
        ]
        profile = find_caz_regions(layer_metrics)

        vecs = {m["layer"]: np.array(m["dom_vector"]) for m in metrics_raw}
        seps = [m["separation_fisher"] for m in metrics_raw]

        rec = {
            "model_id": model_id,
            "n_layers": n_layers,
            "profile": profile,
            "seps": seps,
            "vecs": vecs,
            "metrics_raw": metrics_raw,
        }

        if profile.is_multimodal:
            regions = sorted(profile.regions, key=lambda r: r.peak)
            v_shallow = vecs[regions[0].peak]
            v_deep = vecs[regions[-1].peak]
            cos = float(np.dot(v_shallow, v_deep) / (
                np.linalg.norm(v_shallow) * np.linalg.norm(v_deep)
            ))
            rec["shallow"] = regions[0]
            rec["deep"] = regions[-1]
            rec["cos_sim"] = cos

        records.append(rec)

    # Deduplicate by model_id (keep last run)
    seen = {}
    for r in records:
        seen[r["model_id"]] = r
    return list(seen.values())


def panel_a_example_profile(ax, records):
    """Panel A: Example separation curve with shaded assembly regions."""
    # Pick pythia-1.4b as the showcase model
    rec = next((r for r in records if "pythia-1.4b" in r["model_id"]), None)
    if rec is None:
        rec = next(r for r in records if r["profile"].is_multimodal)

    n = rec["n_layers"]
    depth_pct = [100 * i / n for i in range(n)]
    seps = rec["seps"]
    color = CONCEPT_META.get("credibility", {}).get("color", "#7B1FA2")

    ax.plot(depth_pct, seps, "o-", color=color, linewidth=2, markersize=4, zorder=5)

    if rec["profile"].is_multimodal:
        shallow = rec["shallow"]
        deep = rec["deep"]

        # Shade regions
        for region, label, alpha in [
            (shallow, "Shallow\nassembly", 0.15),
            (deep, "Deep\nassembly", 0.25),
        ]:
            x0 = 100 * region.start / n
            x1 = 100 * region.end / n
            ax.axvspan(x0, x1, alpha=alpha, color=color, zorder=1)
            # Peak marker
            px = 100 * region.peak / n
            ax.plot(px, seps[region.peak], "*", color="red", markersize=12, zorder=10)
            # Label
            ax.annotate(
                f"{label}\nL{region.peak} ({region.depth_pct:.0f}%)",
                xy=(px, seps[region.peak]),
                xytext=(0, 18), textcoords="offset points",
                ha="center", fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5),
            )

        # Valley annotation
        valley_layer = shallow.end
        ax.annotate(
            "valley",
            xy=(100 * valley_layer / n, seps[valley_layer]),
            xytext=(0, -22), textcoords="offset points",
            ha="center", fontsize=7, fontstyle="italic", color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.6),
        )

        # Cosine similarity annotation
        cos = rec["cos_sim"]
        ax.text(
            0.98, 0.05,
            f"cos(shallow, deep) = {cos:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
        )

    ax.set_ylabel("S(l) — Separation")
    ax.set_xlabel("Relative depth (%)")
    ax.set_title(f"(A)  Credibility — {_short_name(rec['model_id'])}", fontweight="bold")


def panel_b_peak_scatter(ax, records):
    """Panel B: Shallow vs deep peak depth across all multimodal models."""
    multimodal = [r for r in records if r["profile"].is_multimodal]

    for rec in multimodal:
        fam = _model_family(rec["model_id"])
        marker = FAMILY_MARKERS.get(fam, "o")
        color = FAMILY_COLORS.get(fam, "#999")
        s_depth = rec["shallow"].depth_pct
        d_depth = rec["deep"].depth_pct
        cos = rec["cos_sim"]

        sc = ax.scatter(
            s_depth, d_depth,
            c=[cos], cmap="RdYlBu", vmin=0, vmax=1,
            marker=marker, s=80, edgecolors=color, linewidths=1.5,
            zorder=5,
        )
        # Label
        ax.annotate(
            _short_name(rec["model_id"]),
            xy=(s_depth, d_depth),
            xytext=(4, 4), textcoords="offset points",
            fontsize=5.5, alpha=0.7,
        )

    # Diagonal reference
    ax.plot([0, 100], [0, 100], "--", color="gray", alpha=0.3, linewidth=1)
    ax.set_xlim(0, 50)
    ax.set_ylim(40, 90)
    ax.set_xlabel("Shallow peak depth (%)")
    ax.set_ylabel("Deep peak depth (%)")
    ax.set_title("(B)  Peak locations (color = cos similarity)", fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("cos(dir₁, dir₂)", fontsize=8)

    # Family legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=FAMILY_MARKERS[f], color="w",
               markeredgecolor=FAMILY_COLORS[f], markerfacecolor="white",
               markersize=7, label=f.capitalize())
        for f in ["pythia", "gpt2", "opt", "qwen", "gemma"]
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.8)


def panel_c_direction_divergence(ax, records):
    """Panel C: Cosine similarity bar chart — how different are the two directions?"""
    multimodal = sorted(
        [r for r in records if r["profile"].is_multimodal],
        key=lambda r: r["cos_sim"],
    )

    names = [_short_name(r["model_id"]) for r in multimodal]
    cosines = [r["cos_sim"] for r in multimodal]
    families = [_model_family(r["model_id"]) for r in multimodal]
    colors = [FAMILY_COLORS.get(f, "#999") for f in families]

    y_pos = range(len(names))
    ax.barh(y_pos, cosines, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Cosine similarity")
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.4)

    # Mean line
    mean_cos = np.mean(cosines)
    ax.axvline(mean_cos, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
    ax.text(
        mean_cos + 0.02, len(names) - 1,
        f"mean = {mean_cos:.2f}",
        fontsize=8, color="red", va="top",
    )

    ax.set_title("(C)  Direction divergence between peaks", fontweight="bold")


def panel_d_overlaid_curves(ax, records):
    """Panel D: All multimodal models' separation curves overlaid."""
    multimodal = [r for r in records if r["profile"].is_multimodal]

    for rec in multimodal:
        n = rec["n_layers"]
        depth_pct = [100 * i / n for i in range(n)]
        # Normalize separation to [0, 1] per model for comparability
        seps = np.array(rec["seps"])
        seps_norm = (seps - seps.min()) / (seps.max() - seps.min() + 1e-9)

        fam = _model_family(rec["model_id"])
        color = FAMILY_COLORS.get(fam, "#999")

        ax.plot(depth_pct, seps_norm, "-", color=color, alpha=0.35, linewidth=1.2)

        # Mark peaks
        for region in rec["profile"].regions:
            px = 100 * region.peak / n
            ax.plot(px, seps_norm[region.peak], "o", color=color,
                    markersize=5, alpha=0.6, zorder=5)

    # Add density shading for where peaks cluster
    all_peak_pcts = []
    for rec in multimodal:
        for region in rec["profile"].regions:
            all_peak_pcts.append(region.depth_pct)

    # KDE of peak positions
    from scipy.stats import gaussian_kde
    peak_arr = np.array(all_peak_pcts)
    if len(peak_arr) > 3:
        kde = gaussian_kde(peak_arr, bw_method=0.2)
        x_kde = np.linspace(0, 100, 200)
        density = kde(x_kde)
        density_norm = density / density.max() * 0.3  # Scale for visual
        ax.fill_between(x_kde, 0, density_norm, color="gray", alpha=0.15, zorder=0)
        ax.text(0.02, 0.95, "gray = peak density", transform=ax.transAxes,
                fontsize=7, fontstyle="italic", color="gray", va="top")

    ax.set_xlabel("Relative depth (%)")
    ax.set_ylabel("S(l) — normalized")
    ax.set_title("(D)  All multimodal curves — credibility", fontweight="bold")

    # Family legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=FAMILY_COLORS[f], linewidth=2, label=f.capitalize())
        for f in ["pythia", "gpt2", "opt", "qwen", "gemma"]
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.8)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_multimodal_data("credibility")

    multimodal_count = sum(1 for r in records if r["profile"].is_multimodal)
    total = len(records)
    print(f"Loaded {total} models, {multimodal_count} multimodal ({100*multimodal_count/total:.0f}%)")

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    fig.suptitle(
        "Multimodal Concept Assembly — Credibility\n"
        "Concepts assemble at multiple depths with distinct geometric directions",
        fontsize=14, fontweight="bold", y=0.98,
    )

    panel_a_example_profile(axes[0, 0], records)
    panel_b_peak_scatter(axes[0, 1], records)
    panel_c_direction_divergence(axes[1, 0], records)
    panel_d_overlaid_curves(axes[1, 1], records)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "multimodal_caz_map.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

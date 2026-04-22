#!/usr/bin/env python3
"""
viz_dark_matter.py — Cazstellation with dark matter features filled in.

Overlays the 7 known concept trajectories with the unlabeled persistent
features discovered by the deep dive. The dark space between concept
dragons is now populated with the features the model actually computes
but we can't name.

Usage:
    python src/viz_dark_matter.py --model EleutherAI/pythia-1.4b
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from rosetta_tools.caz import LayerMetrics, find_caz_regions_scored
from viz_coords import load_shared_coords
from viz_style import concept_color, CONCEPT_COLORS, CONCEPT_TYPE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = Path(__file__).resolve().parents[1] / "visualizations" / "cazstellations"

CONCEPTS = ["credibility", "certainty", "sentiment", "moral_valence", "causation", "temporal_order", "negation"]



def load_model_data(model_id: str):
    """Load CAZ data for one model."""
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        sf = d / "run_summary.json"
        if not sf.exists():
            continue
        try:
            if json.load(open(sf)).get("model_id") == model_id:
                data = {"model_id": model_id, "concepts": {}}
                for concept in CONCEPTS:
                    caz_file = d / f"caz_{concept}.json"
                    if not caz_file.exists():
                        continue
                    data["concepts"][concept] = json.load(open(caz_file))
                    data["n_layers"] = int(data["concepts"][concept]["n_layers"])
                    data["hidden_dim"] = int(data["concepts"][concept]["hidden_dim"])
                return data
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def load_deep_dive(model_id: str):
    """Load deep dive feature map + eigenvector directions."""
    model_slug = model_id.replace("/", "_").replace("-", "_")
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if not d.name.startswith(f"deepdive_{model_slug}"):
            continue
        fm_file = d / "feature_map.json"
        if not fm_file.exists():
            continue

        feature_map = json.load(open(fm_file))

        # Load eigenvector directions at each layer
        directions = {}
        for layer_idx in range(feature_map["n_layers"]):
            npy_file = d / f"directions_L{layer_idx:03d}.npy"
            if npy_file.exists():
                directions[layer_idx] = np.load(npy_file)

        return feature_map, directions
    return None, None


def build_figure(model_data: dict, feature_map: dict, directions: dict) -> go.Figure:
    """Build 3D viz with both concept trajectories and dark matter features."""
    n_layers = model_data["n_layers"]
    model_short = model_data["model_id"].split("/")[-1]

    # ── Load shared coordinate frame ──
    coords = load_shared_coords(model_data["model_id"])

    # ── Build figure ──
    fig = go.Figure()

    persistent = [f for f in feature_map["features"] if f["lifespan"] >= 3]

    def get_coords(label_type, label_id, layer):
        return coords.get(label_type, str(label_id), layer)

    # ── Dark matter features first (behind concept layers) ──
    # Compute eigenvalue normalization using per-feature peak as reference.
    # Use the 90th percentile peak eigenvalue as the "full brightness" anchor
    # so only the top ~10% of features saturate — prevents the titans from
    # blowing everything out.
    peak_eigs = [f["peak_eigenvalue"] for f in persistent]
    eig_anchor = float(np.percentile(peak_eigs, 90)) if peak_eigs else 1.0

    for f in persistent:
        lifespan = f["lifespan"]
        eig = f["peak_eigenvalue"]
        fid = f["feature_id"]

        # Collect trajectory coordinates and per-layer eigenvalues
        traj_points = []  # (x, y, z, eigenvalue, layer)
        eig_by_layer = dict(zip(f["layer_indices"], f["eigenvalues"]))
        for layer in f["layer_indices"]:
            x, y, z = get_coords("dark_traj", fid, layer)
            if x is not None:
                traj_points.append((x, y, z, eig_by_layer.get(layer, 0), layer))

        if len(traj_points) >= 2:
            # Check concept alignment
            align = f.get("concept_alignment", {})
            best_concept = max(align, key=align.get) if align else "?"
            best_val = align.get(best_concept, 0)
            nearest = f"{best_concept}({best_val:.2f})" if best_val > 0.05 else "no match"

            # Draw per-segment lines with width/brightness proportional to eigenvalue
            for i in range(len(traj_points) - 1):
                x0, y0, z0, eig0, l0 = traj_points[i]
                x1, y1, z1, eig1, l1 = traj_points[i + 1]

                # Use the average eigenvalue of the two endpoints
                seg_eig = (eig0 + eig1) / 2
                # Normalize: log scale to keep faint segments visible
                eig_frac = min(1.0, seg_eig / eig_anchor)

                # Width: 0.5 (faintest) to 5 (strongest)
                seg_width = 0.5 + 4.5 * eig_frac

                # Brightness: dim blue at low eig, bright blue at high eig
                # Cap the white channel to keep things blue even at max
                r = int(30 + 100 * eig_frac)
                g = int(50 + 130 * eig_frac)
                b = int(120 + 135 * eig_frac)
                a = 0.12 + 0.58 * eig_frac
                seg_color = f"rgba({r},{g},{b},{a:.2f})"

                hover_text = (
                    f"<b>Dark Feature F{fid:03d}</b><br>"
                    f"Lifespan: L{f['birth_layer']}–L{f['death_layer']} ({lifespan} layers)<br>"
                    f"Peak eigenvalue: {eig:.1f}<br>"
                    f"Segment eig: {seg_eig:.1f} ({100*eig_frac:.0f}% of max)<br>"
                    f"Nearest concept: {nearest}<br>"
                    f"Type: {'persistent' if lifespan >= 5 else 'transient'}"
                )

                fig.add_trace(go.Scatter3d(
                    x=[x0, x1], y=[y0, y1], z=[z0, z1],
                    mode="lines",
                    line=dict(color=seg_color, width=seg_width),
                    legendgroup="dark_matter",
                    showlegend=False,
                    hovertemplate=hover_text + "<extra></extra>",
                ))

            # Peak marker — size scales with eigenvalue
            px, py, pz = get_coords("dark_peak", fid, f["peak_layer"])
            if px is not None:
                peak_frac = min(1.0, eig / eig_anchor)
                peak_size = max(3, min(14, 3 + 11 * peak_frac))
                peak_r = int(40 + 215 * peak_frac)
                peak_g = int(60 + 195 * peak_frac)
                peak_b = int(120 + 135 * peak_frac)
                peak_a = 0.3 + 0.65 * peak_frac
                peak_color = f"rgba({peak_r},{peak_g},{peak_b},{peak_a:.2f})"

                hover_text = (
                    f"<b>Dark Feature F{fid:03d}</b><br>"
                    f"Lifespan: L{f['birth_layer']}–L{f['death_layer']} ({lifespan} layers)<br>"
                    f"Peak eigenvalue: {eig:.1f}<br>"
                    f"Nearest concept: {nearest}<br>"
                    f"Type: {'persistent' if lifespan >= 5 else 'transient'}"
                )

                fig.add_trace(go.Scatter3d(
                    x=[px], y=[py], z=[pz],
                    mode="markers",
                    marker=dict(size=peak_size, color=peak_color,
                                symbol="diamond",
                                line=dict(width=0.5, color="rgba(150,200,255,0.5)")),
                    legendgroup="dark_matter",
                    showlegend=False,
                    hovertemplate=hover_text + "<extra></extra>",
                ))

    # ── Concept trajectories (on top) ──
    # Compute separation anchor for concept brightness scaling (90th percentile)
    all_concept_seps = []
    for concept, cdata in model_data["concepts"].items():
        for m in cdata["layer_data"]["metrics"]:
            all_concept_seps.append(m["separation_fisher"])
    concept_sep_anchor = float(np.percentile(all_concept_seps, 90)) if all_concept_seps else 1.0

    caz_regions = {}
    for concept, cdata in model_data["concepts"].items():
        metrics = cdata["layer_data"]["metrics"]
        lm = [LayerMetrics(m["layer"], m["separation_fisher"], m["coherence"], m["velocity"])
              for m in metrics]
        caz_regions[concept] = find_caz_regions_scored(lm).regions

        color = CONCEPT_COLORS[concept]
        ctype = CONCEPT_TYPE[concept]

        # Parse base color for brightness modulation
        base_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))

        # Collect trajectory points with separation values
        traj = []
        for m in metrics:
            x, y, z = get_coords("concept", concept, m["layer"])
            if x is not None:
                traj.append((x, y, z, m["separation_fisher"], m["layer"]))

        # Per-segment lines with width/brightness scaled by separation
        first_seg = True
        for i in range(len(traj) - 1):
            x0, y0, z0, sep0, l0 = traj[i]
            x1, y1, z1, sep1, l1 = traj[i + 1]

            seg_sep = (sep0 + sep1) / 2
            sep_frac = min(1.0, seg_sep / concept_sep_anchor)

            # Width: 1 (faintest) to 6 (strongest)
            seg_width = 1.0 + 5.0 * sep_frac

            # Brightness: dim version of concept color at low sep, full at high
            dim = 0.25 + 0.75 * sep_frac
            r = int(base_rgb[0] * dim)
            g = int(base_rgb[1] * dim)
            b = int(base_rgb[2] * dim)
            a = 0.2 + 0.7 * sep_frac
            seg_color = f"rgba({r},{g},{b},{a:.2f})"

            fig.add_trace(go.Scatter3d(
                x=[x0, x1], y=[y0, y1], z=[z0, z1],
                mode="lines",
                line=dict(color=seg_color, width=seg_width),
                name=f"{concept} ({ctype})",
                legendgroup=concept,
                showlegend=first_seg,
                hovertemplate=(
                    f"<b>{concept}</b> ({ctype})<br>"
                    f"L{l0}–L{l1}<br>"
                    f"Separation: {seg_sep:.3f}"
                    "<extra></extra>"
                ),
            ))
            first_seg = False

        # CAZ peaks
        for region in caz_regions[concept]:
            pk = region.peak
            x, y, z = get_coords("concept", concept, pk)
            if x is None:
                continue

            score = region.caz_score
            size = max(5, min(20, 6 + 12 * np.log1p(score * 10)))
            opacity = min(0.95, max(0.3, 0.3 + 0.65 * np.log1p(score * 5) / np.log1p(5)))

            hover = (
                f"<b>{concept}</b> ({ctype})<br>"
                f"Layer {pk} ({region.depth_pct:.0f}%)<br>"
                f"CAZ Score: {score:.3f}<br>"
                f"Separation: {region.peak_separation:.3f}"
            )

            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode="markers",
                marker=dict(size=size, color=color, opacity=opacity,
                            line=dict(width=1, color="white")),
                legendgroup=concept,
                showlegend=False,
                hovertemplate=hover + "<extra></extra>",
            ))

    # Dark matter legend entry
    n_dark = len(persistent)
    n_single = sum(1 for f in feature_map["features"] if f["lifespan"] == 1)
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(size=6, color="rgba(80,140,220,0.6)", symbol="diamond"),
        name=f"dark matter ({n_dark} persistent, {n_single} single-layer)",
        legendgroup="dark_matter",
        showlegend=True,
    ))

    # Layout
    total_features = feature_map["n_features"]
    n_labeled = sum(1 for f in feature_map["features"]
                    if any(v > 0.3 for v in f.get("concept_alignment", {}).values()))

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Dark Matter Map — {model_short}</b><br>"
                f"<sub>{total_features} features: 7 labeled concepts (bright) + "
                f"{total_features - n_labeled} unlabeled (blue diamonds).  "
                f"Double-click legend to isolate.</sub>"
            ),
            font=dict(size=16, color="rgba(200,210,230,0.95)"),
        ),
        scene=dict(
            xaxis=dict(title="", showbackground=True, showticklabels=False,
                       backgroundcolor="rgb(5,5,15)", gridcolor="rgba(60,80,120,0.08)",
                       zerolinecolor="rgba(80,100,130,0.1)",
                       range=list(coords.x_range)),
            yaxis=dict(title="", showbackground=True, showticklabels=False,
                       backgroundcolor="rgb(5,5,15)", gridcolor="rgba(60,80,120,0.08)",
                       zerolinecolor="rgba(80,100,130,0.1)",
                       range=list(coords.y_range)),
            zaxis=dict(title="Depth (%)", showbackground=True,
                       backgroundcolor="rgb(5,5,15)", gridcolor="rgba(60,80,120,0.08)",
                       zerolinecolor="rgba(80,100,130,0.1)",
                       color="rgba(150,160,180,0.4)",
                       range=[0, 100], dtick=10),
            bgcolor="rgb(3,3,12)",
            camera=dict(eye=dict(x=1.6, y=0.7, z=0.5), up=dict(x=0, y=0, z=1)),
        ),
        paper_bgcolor="rgb(3,3,12)",
        plot_bgcolor="rgb(3,3,12)",
        font=dict(color="rgba(200,210,230,0.9)"),
        legend=dict(
            bgcolor="rgba(8,10,22,0.9)", bordercolor="rgba(60,80,120,0.3)", borderwidth=1,
            font=dict(size=11, color="rgba(200,210,230,0.9)"),
            title=dict(text="Features (dbl-click to isolate)",
                       font=dict(size=9, color="rgba(130,140,160,0.7)")),
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        width=1500, height=1000,
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="Dark matter cazstellation")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-1.4b")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading model data for %s...", args.model)
    model_data = load_model_data(args.model)
    if not model_data:
        log.error("No CAZ extraction data for %s", args.model)
        return

    log.info("Loading deep dive results...")
    feature_map, directions = load_deep_dive(args.model)
    if not feature_map:
        log.error("No deep dive data for %s", args.model)
        return

    log.info("Building dark matter visualization...")
    fig = build_figure(model_data, feature_map, directions)

    slug = args.model.replace("/", "_").replace("-", "_")
    out_path = OUT_DIR / f"dark_matter_{slug}.html"
    fig.write_html(str(out_path), include_plotlyjs=True)
    log.info("→ %s", out_path)


if __name__ == "__main__":
    main()

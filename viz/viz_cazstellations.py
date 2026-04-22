#!/usr/bin/env python3
"""
viz_cazstellations.py — 3D interactive visualization of CAZ constellations.

Projects concept assembly zones into 3D space using PCA on the dom_vectors
across all concepts and layers.  Each CAZ is a node sized by caz_score,
colored by concept, connected across layers by filaments.

Produces an interactive HTML file (plotly) that can be rotated and explored.

Usage:
    python src/viz_cazstellations.py --model EleutherAI/pythia-1.4b
    python src/viz_cazstellations.py --all   # all models, separate files
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from rosetta_tools.caz import LayerMetrics, find_caz_regions_scored
from viz_style import concept_color, CONCEPT_COLORS, CONCEPT_TYPE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = Path(__file__).resolve().parents[1] / "visualizations" / "cazstellations"

CONCEPTS = ["credibility", "certainty", "sentiment", "moral_valence", "causation", "temporal_order", "negation"]



def load_model_data(model_id: str):
    """Load all CAZ data for one model."""
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        sf = d / "run_summary.json"
        if not sf.exists():
            continue
        try:
            if json.load(open(sf)).get("model_id") == model_id:
                model_data = {"model_id": model_id, "concepts": {}}
                for concept in CONCEPTS:
                    caz_file = d / f"caz_{concept}.json"
                    if not caz_file.exists():
                        continue
                    data = json.load(open(caz_file))
                    model_data["n_layers"] = int(data["n_layers"])
                    model_data["hidden_dim"] = int(data["hidden_dim"])
                    model_data["concepts"][concept] = data
                return model_data
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def build_cazstellation(model_data: dict) -> go.Figure:
    """Build 3D plotly figure for one model."""
    n_layers = model_data["n_layers"]
    hidden_dim = model_data["hidden_dim"]
    model_short = model_data["model_id"].split("/")[-1]

    # ── Collect all dom_vectors and metadata ──
    all_vectors = []
    all_meta = []  # (concept, layer, separation, coherence)

    for concept, data in model_data["concepts"].items():
        metrics = data["layer_data"]["metrics"]
        for m in metrics:
            vec = np.array(m["dom_vector"], dtype=np.float64)
            all_vectors.append(vec)
            all_meta.append({
                "concept": concept,
                "layer": m["layer"],
                "depth_pct": 100 * m["layer"] / n_layers,
                "separation": m["separation_fisher"],
                "coherence": m["coherence"],
            })

    # Filter to consistent dimensions (some models have mixed dim across concepts)
    expected_dim = hidden_dim
    filtered_vectors = []
    filtered_meta = []
    for vec, meta in zip(all_vectors, all_meta):
        if len(vec) == expected_dim:
            filtered_vectors.append(vec)
            filtered_meta.append(meta)
    all_vectors = np.array(filtered_vectors)
    all_meta = filtered_meta
    log.info("  PCA on %d vectors (dim=%d)", len(all_vectors), hidden_dim)

    # PCA to 3D — but use depth as the Z axis for intuitive navigation
    # Project dom_vectors into 2D (X, Y) and use depth % as Z
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(all_vectors)
    explained = pca.explained_variance_ratio_

    log.info("  PCA explained variance: %.1f%%, %.1f%%",
             100 * explained[0], 100 * explained[1])

    # ── Detect CAZ regions with scored detector ──
    caz_regions = {}  # concept → list of CAZRegion
    for concept, data in model_data["concepts"].items():
        metrics = data["layer_data"]["metrics"]
        lm = [LayerMetrics(m["layer"], m["separation_fisher"], m["coherence"], m["velocity"])
              for m in metrics]
        profile = find_caz_regions_scored(lm)
        caz_regions[concept] = profile.regions

    # ── Build figure ──
    fig = go.Figure()

    # ── Per-concept traces (all grouped by legendgroup for click-to-focus) ──
    # Double-click a legend entry to isolate that concept; single-click to toggle
    active_concepts = [c for c in CONCEPTS if c in model_data["concepts"]]

    for concept in active_concepts:
        color = CONCEPT_COLORS[concept]
        ctype = CONCEPT_TYPE[concept]
        regions = caz_regions.get(concept, [])
        n_cazs = len(regions)
        is_first = True  # only first trace per concept shows in legend

        # Trajectory line
        concept_indices = [i for i, m in enumerate(all_meta) if m["concept"] == concept]
        concept_indices.sort(key=lambda i: all_meta[i]["layer"])

        xs = [coords_2d[i, 0] for i in concept_indices]
        ys = [coords_2d[i, 1] for i in concept_indices]
        zs = [all_meta[i]["depth_pct"] for i in concept_indices]

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=color, width=2),
            opacity=0.2,
            name=f"{concept} ({ctype}, {n_cazs} CAZes)",
            legendgroup=concept,
            showlegend=True,
            hoverinfo="skip",
        ))

        # CAZ nodes
        for region in regions:
            peak_layer = region.peak
            idx = next((i for i, m in enumerate(all_meta)
                       if m["concept"] == concept and m["layer"] == peak_layer), None)
            if idx is None:
                continue

            x = coords_2d[idx, 0]
            y = coords_2d[idx, 1]
            z = all_meta[idx]["depth_pct"]

            score = region.caz_score
            size = max(4, min(35, 8 + 18 * np.log1p(score * 10)))

            if score > 0.5:
                strength = "BLACK HOLE"
            elif score > 0.2:
                strength = "strong"
            elif score > 0.05:
                strength = "moderate"
            else:
                strength = "gentle"

            hover = (
                f"<b>{concept}</b> ({ctype})<br>"
                f"Layer {peak_layer} ({region.depth_pct:.0f}% depth)<br>"
                f"CAZ Score: {score:.4f} [{strength}]<br>"
                f"Separation: {region.peak_separation:.3f}<br>"
                f"Coherence: {region.peak_coherence:.3f}<br>"
                f"Width: {region.width} layers"
            )

            # Outer glow
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode="markers",
                marker=dict(size=size * 1.8, color=color, opacity=0.12, line=dict(width=0)),
                showlegend=False,
                legendgroup=concept,
                hoverinfo="skip",
            ))

            # Core node
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode="markers+text",
                marker=dict(
                    size=size,
                    color=color,
                    opacity=0.85,
                    line=dict(width=1, color="rgba(255,255,255,0.4)"),
                ),
                text=f"L{peak_layer}" if score > 0.1 else "",
                textposition="top center",
                textfont=dict(size=8, color="rgba(255,255,255,0.6)"),
                showlegend=False,
                legendgroup=concept,
                hovertemplate=hover + "<extra></extra>",
            ))

            # Region span (vertical bar)
            z_start = 100 * region.start / n_layers
            z_end = 100 * region.end / n_layers
            fig.add_trace(go.Scatter3d(
                x=[x, x], y=[y, y], z=[z_start, z_end],
                mode="lines",
                line=dict(color=color, width=max(2, score * 25)),
                opacity=0.2,
                showlegend=False,
                legendgroup=concept,
                hoverinfo="skip",
            ))

    # ── Co-occurrence filaments (cazstellations) ──
    for layer in range(n_layers):
        layer_cazs = []
        for concept in active_concepts:
            for region in caz_regions.get(concept, []):
                if region.peak == layer:
                    idx = next((i for i, m in enumerate(all_meta)
                               if m["concept"] == concept and m["layer"] == layer), None)
                    if idx is not None:
                        layer_cazs.append((concept, idx, region))

        if len(layer_cazs) >= 3:
            for i in range(len(layer_cazs)):
                for j in range(i + 1, len(layer_cazs)):
                    _, idx1, _ = layer_cazs[i]
                    _, idx2, _ = layer_cazs[j]
                    fig.add_trace(go.Scatter3d(
                        x=[coords_2d[idx1, 0], coords_2d[idx2, 0]],
                        y=[coords_2d[idx1, 1], coords_2d[idx2, 1]],
                        z=[all_meta[idx1]["depth_pct"], all_meta[idx2]["depth_pct"]],
                        mode="lines",
                        line=dict(color="rgba(255,255,255,0.12)", width=1.5),
                        showlegend=False,
                        legendgroup="__filaments",
                        hoverinfo="skip",
                    ))

    # ── Layout ──
    total_cazs = sum(len(regions) for regions in caz_regions.values())
    n_cazstellations = sum(1 for l in range(n_layers)
                          if sum(1 for c in caz_regions.values()
                                 for r in c if r.peak == l) >= 3)

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Cazstellations — {model_short}</b><br>"
                f"<sub>{total_cazs} CAZes across {n_layers} layers  |  "
                f"{n_cazstellations} cazstellations (3+ co-occurring)  |  "
                f"Node size = CAZ score  |  "
                f"Double-click legend to isolate a concept chain</sub>"
            ),
            font=dict(size=16, color="rgba(200,210,230,0.95)"),
        ),
        scene=dict(
            xaxis=dict(
                title=f"PC1 ({100*explained[0]:.0f}%)",
                showgrid=True,
                gridcolor="rgba(80,100,140,0.12)",
                showbackground=True,
                backgroundcolor="rgb(8, 8, 18)",
                color="rgba(150,160,180,0.5)",
                zerolinecolor="rgba(100,110,130,0.15)",
            ),
            yaxis=dict(
                title=f"PC2 ({100*explained[1]:.0f}%)",
                showgrid=True,
                gridcolor="rgba(80,100,140,0.12)",
                showbackground=True,
                backgroundcolor="rgb(8, 8, 18)",
                color="rgba(150,160,180,0.5)",
                zerolinecolor="rgba(100,110,130,0.15)",
            ),
            zaxis=dict(
                title="Depth (%)",
                showgrid=True,
                gridcolor="rgba(80,100,140,0.12)",
                showbackground=True,
                backgroundcolor="rgb(8, 8, 18)",
                color="rgba(150,160,180,0.5)",
                zerolinecolor="rgba(100,110,130,0.15)",
                range=[0, 100],
                dtick=10,
            ),
            bgcolor="rgb(5, 5, 15)",
            camera=dict(
                eye=dict(x=1.8, y=0.8, z=0.6),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        paper_bgcolor="rgb(5, 5, 15)",
        plot_bgcolor="rgb(5, 5, 15)",
        font=dict(color="rgba(200,210,230,0.9)"),
        legend=dict(
            bgcolor="rgba(10,12,25,0.85)",
            bordercolor="rgba(80,100,140,0.3)",
            borderwidth=1,
            font=dict(size=12, color="rgba(200,210,230,0.9)"),
            itemsizing="constant",
            title=dict(
                text="Concepts (click: toggle, dbl-click: isolate)",
                font=dict(size=10, color="rgba(150,160,180,0.7)"),
            ),
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        width=1400,
        height=900,
    )

    return fig


def main():
    parser = argparse.ArgumentParser(description="3D CAZ constellation visualization")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Single model ID")
    group.add_argument("--all", action="store_true", help="All models with extraction data")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        seen = set()
        for d in sorted(RESULTS_DIR.iterdir()):
            sf = d / "run_summary.json"
            if not sf.exists():
                continue
            mid = json.load(open(sf)).get("model_id")
            if mid and mid not in seen:
                seen.add(mid)
                log.info("Building cazstellation for %s...", mid)
                data = load_model_data(mid)
                if data and len(data["concepts"]) >= 3:
                    fig = build_cazstellation(data)
                    slug = mid.replace("/", "_").replace("-", "_")
                    out_path = OUT_DIR / f"cazstellation_{slug}.html"
                    fig.write_html(str(out_path), include_plotlyjs=True)
                    log.info("  → %s", out_path)
    else:
        log.info("Building cazstellation for %s...", args.model)
        data = load_model_data(args.model)
        if data is None:
            log.error("No extraction data found for %s", args.model)
            sys.exit(1)
        fig = build_cazstellation(data)
        slug = args.model.replace("/", "_").replace("-", "_")
        out_path = OUT_DIR / f"cazstellation_{slug}.html"
        fig.write_html(str(out_path), include_plotlyjs=True)
        log.info("→ %s", out_path)

        # Also show if running interactively
        try:
            fig.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()

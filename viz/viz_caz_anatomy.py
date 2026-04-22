#!/usr/bin/env python3
"""
viz_caz_anatomy.py — 3D visualization with CAZ regions shown as elongated shapes.

Like viz_dark_matter.py but renders each CAZ as a tapered tube spanning its
full layer range (start → peak → end), thickest at the peak. Shows both
the spatial position AND the width/duration of each assembly zone.

Dark matter features get the same treatment — rendered as tubes that
thicken at their peak eigenvalue layer.

Usage:
    python src/viz_caz_anatomy.py --model EleutherAI/pythia-1.4b
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

        directions = {}
        for layer_idx in range(feature_map["n_layers"]):
            npy_file = d / f"directions_L{layer_idx:03d}.npy"
            if npy_file.exists():
                directions[layer_idx] = np.load(npy_file)

        return feature_map, directions
    return None, None


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))


def make_tube_mesh(
    center_points: list[tuple[float, float, float]],
    radii: list[float],
    color_rgba: list[tuple[int, int, int, float]],
    n_sides: int = 8,
) -> tuple[list, list, list, list, list, list, list]:
    """Generate a tube mesh as rings of vertices around center points.

    Returns vertices (x, y, z) and triangle indices (i, j, k) and colors.
    """
    if len(center_points) < 2:
        return [], [], [], [], [], [], []

    vertices_x, vertices_y, vertices_z = [], [], []
    face_i, face_j, face_k = []  , [], []
    face_colors = []

    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)

    # Build a local coordinate frame at each point.
    # The tube runs along the trajectory; we need two perpendicular axes.
    for ring_idx, (cx, cy, cz) in enumerate(center_points):
        r = radii[ring_idx]

        # Tangent direction (forward difference, backward at end)
        if ring_idx < len(center_points) - 1:
            nx = center_points[ring_idx + 1][0] - cx
            ny = center_points[ring_idx + 1][1] - cy
            nz = center_points[ring_idx + 1][2] - cz
        else:
            nx = cx - center_points[ring_idx - 1][0]
            ny = cy - center_points[ring_idx - 1][1]
            nz = cz - center_points[ring_idx - 1][2]

        # Normalize tangent
        t_len = np.sqrt(nx**2 + ny**2 + nz**2)
        if t_len < 1e-12:
            nx, ny, nz = 0, 0, 1
        else:
            nx, ny, nz = nx / t_len, ny / t_len, nz / t_len

        # Find two perpendicular axes via cross product with an arbitrary vector
        # Use (1,0,0) unless tangent is nearly parallel to it
        if abs(nx) < 0.9:
            ax, ay, az = 1, 0, 0
        else:
            ax, ay, az = 0, 1, 0

        # u = tangent × arbitrary
        ux = ny * az - nz * ay
        uy = nz * ax - nx * az
        uz = nx * ay - ny * ax
        u_len = np.sqrt(ux**2 + uy**2 + uz**2)
        ux, uy, uz = ux / u_len, uy / u_len, uz / u_len

        # v = tangent × u
        vx = ny * uz - nz * uy
        vy = nz * ux - nx * uz
        vz = nx * uy - ny * ux

        # Generate ring vertices
        for angle in angles:
            px = cx + r * (np.cos(angle) * ux + np.sin(angle) * vx)
            py = cy + r * (np.cos(angle) * uy + np.sin(angle) * vy)
            pz = cz + r * (np.cos(angle) * uz + np.sin(angle) * vz)
            vertices_x.append(px)
            vertices_y.append(py)
            vertices_z.append(pz)

    # Connect adjacent rings with triangles
    for ring_idx in range(len(center_points) - 1):
        base_curr = ring_idx * n_sides
        base_next = (ring_idx + 1) * n_sides
        r, g, b, a = color_rgba[ring_idx]

        for s in range(n_sides):
            s_next = (s + 1) % n_sides

            # Two triangles per quad
            face_i.append(base_curr + s)
            face_j.append(base_next + s)
            face_k.append(base_next + s_next)
            face_colors.append(f"rgba({r},{g},{b},{a:.2f})")

            face_i.append(base_curr + s)
            face_j.append(base_next + s_next)
            face_k.append(base_curr + s_next)
            face_colors.append(f"rgba({r},{g},{b},{a:.2f})")

    return vertices_x, vertices_y, vertices_z, face_i, face_j, face_k, face_colors


def build_figure(model_data: dict, feature_map: dict, directions: dict) -> go.Figure:
    """Build 3D viz with tube-shaped CAZ regions and dark matter features."""
    n_layers = model_data["n_layers"]
    model_short = model_data["model_id"].split("/")[-1]

    # ── Load shared coordinate frame ──
    coords = load_shared_coords(model_data["model_id"])

    persistent = [f for f in feature_map["features"] if f["lifespan"] >= 3]
    transient = [f for f in feature_map["features"] if f["lifespan"] <= 2]

    fig = go.Figure()

    def get_coords(label_type, label_id, layer):
        return coords.get(label_type, str(label_id), layer)
        return None, None, None

    # ── Dark matter features as tubes ──
    peak_eigs = [f["peak_eigenvalue"] for f in persistent]
    eig_anchor = float(np.percentile(peak_eigs, 90)) if peak_eigs else 1.0

    # Coordinate scale for tube radii (fraction of the PCA spread)
    x_span = coords.x_range[1] - coords.x_range[0]
    y_span = coords.y_range[1] - coords.y_range[0]
    coord_scale = max(x_span, y_span, 1e-6)
    max_radius = coord_scale * 0.0125  # max tube radius = 1.25% of plot extent

    for f in persistent:
        lifespan = f["lifespan"]
        fid = f["feature_id"]
        eig_by_layer = dict(zip(f["layer_indices"], f["eigenvalues"]))

        traj_points = []
        traj_eigs = []
        for layer in f["layer_indices"]:
            x, y, z = get_coords("dark_traj", fid, layer)
            if x is not None:
                traj_points.append((x, y, z))
                traj_eigs.append(eig_by_layer.get(layer, 0))

        if len(traj_points) < 2:
            continue

        align = f.get("concept_alignment", {})
        best_concept = max(align, key=align.get) if align else "?"
        best_val = align.get(best_concept, 0)
        nearest = f"{best_concept}({best_val:.2f})" if best_val > 0.05 else "no match"

        # Per-segment lines with width/brightness scaled by eigenvalue
        for i in range(len(traj_points) - 1):
            x0, y0, z0 = traj_points[i]
            x1, y1, z1 = traj_points[i + 1]
            seg_eig = (traj_eigs[i] + traj_eigs[i + 1]) / 2
            frac = min(1.0, seg_eig / eig_anchor)

            seg_width = 0.5 + 4.5 * frac

            cr = int(30 + 100 * frac)
            cg = int(50 + 130 * frac)
            cb = int(120 + 135 * frac)
            ca = 0.12 + 0.58 * frac
            seg_color = f"rgba({cr},{cg},{cb},{ca:.2f})"

            fig.add_trace(go.Scatter3d(
                x=[x0, x1], y=[y0, y1], z=[z0, z1],
                mode="lines",
                line=dict(color=seg_color, width=seg_width),
                legendgroup="dark_matter",
                showlegend=False,
                hovertemplate=(
                    f"<b>Dark Feature F{fid:03d}</b><br>"
                    f"Lifespan: L{f['birth_layer']}–L{f['death_layer']} ({lifespan} layers)<br>"
                    f"Peak eigenvalue: {f['peak_eigenvalue']:.1f}<br>"
                    f"Segment eig: {seg_eig:.1f}<br>"
                    f"Nearest concept: {nearest}"
                    "<extra></extra>"
                ),
            ))

        # Peak marker
        px, py, pz = get_coords("dark_peak", fid, f["peak_layer"])
        if px is not None:
            peak_frac = min(1.0, f["peak_eigenvalue"] / eig_anchor)
            peak_size = max(3, min(10, 3 + 7 * peak_frac))
            pr = int(30 + 100 * peak_frac)
            pg = int(50 + 130 * peak_frac)
            pb = int(120 + 135 * peak_frac)
            pa = 0.3 + 0.6 * peak_frac

            fig.add_trace(go.Scatter3d(
                x=[px], y=[py], z=[pz],
                mode="markers",
                marker=dict(size=peak_size,
                            color=f"rgba({pr},{pg},{pb},{pa:.2f})",
                            symbol="diamond",
                            line=dict(width=0.5, color="rgba(150,200,255,0.5)")),
                legendgroup="dark_matter",
                showlegend=False,
                hovertemplate=(
                    f"<b>Dark Feature F{fid:03d}</b><br>"
                    f"Lifespan: L{f['birth_layer']}–L{f['death_layer']} ({lifespan} layers)<br>"
                    f"Peak eigenvalue: {f['peak_eigenvalue']:.1f}<br>"
                    f"Nearest concept: {nearest}"
                    "<extra></extra>"
                ),
            ))

    # ── Concept trajectories as tubes with CAZ bulges ──
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

        base_rgb = hex_to_rgb(CONCEPT_COLORS[concept])
        ctype = CONCEPT_TYPE[concept]

        # Build separation lookup
        sep_by_layer = {m["layer"]: m["separation_fisher"] for m in metrics}

        traj_points = []
        traj_seps = []
        for m in metrics:
            x, y, z = get_coords("concept", concept, m["layer"])
            if x is not None:
                traj_points.append((x, y, z))
                traj_seps.append(m["separation_fisher"])

        if len(traj_points) < 2:
            continue

        # Per-point radii: baseline thin, bulges at CAZ regions
        radii = []
        colors_rgba = []
        for idx, sep in enumerate(traj_seps):
            frac = min(1.0, sep / concept_sep_anchor)
            r = max_radius * (0.08 + 0.92 * frac)
            radii.append(r)

            dim = 0.3 + 0.7 * frac
            cr = int(base_rgb[0] * dim)
            cg = int(base_rgb[1] * dim)
            cb = int(base_rgb[2] * dim)
            ca = 0.25 + 0.65 * frac
            colors_rgba.append((cr, cg, cb, ca))

        vx, vy, vz, fi, fj, fk, fc = make_tube_mesh(
            traj_points, radii, colors_rgba, n_sides=8,
        )

        if vx:
            # Build hover text for CAZ regions
            caz_info = []
            for region in caz_regions[concept]:
                caz_info.append(
                    f"CAZ L{region.start}–L{region.end} "
                    f"(peak L{region.peak}, score={region.caz_score:.3f})"
                )
            caz_str = "<br>".join(caz_info) if caz_info else "no CAZes"

            fig.add_trace(go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=fi, j=fj, k=fk,
                facecolor=fc,
                flatshading=True,
                name=f"{concept} ({ctype})",
                legendgroup=concept,
                showlegend=True,
                hovertemplate=(
                    f"<b>{concept}</b> ({ctype})<br>"
                    f"{caz_str}"
                    "<extra></extra>"
                ),
            ))

            # CAZ peak markers (still spheres for precise peak identification)
            for region in caz_regions[concept]:
                pk = region.peak
                x, y, z = get_coords("concept", concept, pk)
                if x is None:
                    continue

                score = region.caz_score
                size = max(4, min(16, 5 + 10 * np.log1p(score * 10)))
                opacity = min(0.95, max(0.4, 0.4 + 0.55 * np.log1p(score * 5) / np.log1p(5)))

                fig.add_trace(go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode="markers",
                    marker=dict(
                        size=size,
                        color=CONCEPT_COLORS[concept],
                        opacity=opacity,
                        line=dict(width=1, color="white"),
                    ),
                    legendgroup=concept,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{concept}</b> CAZ peak<br>"
                        f"Layer {pk} ({region.depth_pct:.0f}%)<br>"
                        f"Score: {score:.3f}<br>"
                        f"Width: L{region.start}–L{region.end} ({region.width} layers)<br>"
                        f"Separation: {region.peak_separation:.3f}"
                        "<extra></extra>"
                    ),
                ))

    # ── Transient features as stars ──
    trans_x, trans_y, trans_z = [], [], []
    trans_sizes, trans_colors, trans_hovers = [], [], []

    for f in transient:
        fid = f["feature_id"]
        peak_layer = f["peak_layer"]
        x, y, z = get_coords("transient", fid, peak_layer)
        if x is None:
            continue

        eig_val = f["peak_eigenvalue"]
        frac = min(1.0, eig_val / eig_anchor)

        trans_x.append(x)
        trans_y.append(y)
        trans_z.append(z)

        # Size: small stars, slightly larger for stronger ones
        trans_sizes.append(max(2, min(6, 2 + 4 * frac)))

        # Color: dim warm white — distinct from the blue persistent features
        r = int(140 + 115 * frac)
        g = int(130 + 105 * frac)
        b = int(100 + 80 * frac)
        a = 0.15 + 0.45 * frac
        trans_colors.append(f"rgba({r},{g},{b},{a:.2f})")

        trans_hovers.append(
            f"<b>Transient F{fid:03d}</b><br>"
            f"Layer {peak_layer} ({100 * peak_layer / n_layers:.0f}%)<br>"
            f"Lifespan: {f['lifespan']} layer(s)<br>"
            f"Eigenvalue: {eig_val:.1f}"
        )

    if trans_x:
        fig.add_trace(go.Scatter3d(
            x=trans_x, y=trans_y, z=trans_z,
            mode="markers",
            marker=dict(
                size=trans_sizes,
                color=trans_colors,
                symbol="diamond",
            ),
            legendgroup="transient",
            showlegend=False,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=trans_hovers,
        ))

    # Legend entries
    n_dark = len(persistent)
    n_trans = len(transient)
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(size=6, color="rgba(80,140,220,0.6)", symbol="diamond"),
        name=f"dark matter ({n_dark} persistent)",
        legendgroup="dark_matter",
        showlegend=True,
    ))
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(size=3, color="rgba(200,190,150,0.4)", symbol="diamond"),
        name=f"transient sparks ({n_trans})",
        legendgroup="transient",
        showlegend=True,
    ))

    # Layout
    total_features = feature_map["n_features"]
    n_labeled = sum(1 for f in feature_map["features"]
                    if any(v > 0.3 for v in f.get("concept_alignment", {}).values()))

    fig.update_layout(
        title=dict(
            text=(
                f"<b>CAZ Anatomy — {model_short}</b><br>"
                f"<sub>Tube thickness = signal strength (separation for concepts, "
                f"eigenvalue for dark matter). "
                f"{total_features} features total, {total_features - n_labeled} unlabeled.</sub>"
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
    parser = argparse.ArgumentParser(description="CAZ anatomy visualization with tube-shaped regions")
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

    log.info("Building CAZ anatomy visualization...")
    fig = build_figure(model_data, feature_map, directions)

    model_slug = args.model.split("/")[-1]
    out_path = OUT_DIR / f"caz_anatomy_{model_slug}.html"
    fig.write_html(str(out_path), include_plotlyjs=True)
    log.info("-> %s", out_path)


if __name__ == "__main__":
    main()

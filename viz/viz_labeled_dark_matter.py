#!/usr/bin/env python3
"""
viz_labeled_dark_matter.py — Dark matter map with layer-aware concept labeling.

Features that align with a known concept at specific layers are colored
in that concept's color at those layers. Features that hand off between
concepts visibly change color mid-depth. Truly unlabeled segments stay blue.

Usage:
    python src/viz_labeled_dark_matter.py --model EleutherAI/pythia-1.4b
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



def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))


def load_model_data(model_id: str):
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

        # Load feature labels if available
        labels_file = d / "feature_labels.json"
        feature_labels = None
        if labels_file.exists():
            feature_labels = json.load(open(labels_file))

        return feature_map, directions, feature_labels
    return None, None, None


def build_figure(model_data, feature_map, directions, feature_labels) -> go.Figure:
    n_layers = model_data["n_layers"]
    model_short = model_data["model_id"].split("/")[-1]

    coords = load_shared_coords(model_data["model_id"])
    fig = go.Figure()

    persistent = [f for f in feature_map["features"] if f["lifespan"] >= 3]
    transient = [f for f in feature_map["features"] if f["lifespan"] <= 2]

    def get_coords(label_type, label_id, layer):
        return coords.get(label_type, str(label_id), layer)

    # ── Eigenvalue anchor ──
    peak_eigs = [f["peak_eigenvalue"] for f in persistent]
    eig_anchor = float(np.percentile(peak_eigs, 90)) if peak_eigs else 1.0

    # ── Build per-layer label lookup ──
    # feature_id -> layer -> (concept_or_None, cos)
    label_map = {}
    if feature_labels:
        for fid_str, layers in feature_labels["features"].items():
            fid = int(fid_str)
            label_map[fid] = {}
            for la in layers:
                label_map[fid][la["layer"]] = (la["best_concept"], la["best_cos"])

    # ── Dark matter / labeled features ──
    for f in persistent:
        lifespan = f["lifespan"]
        fid = f["feature_id"]
        eig_by_layer = dict(zip(f["layer_indices"], f["eigenvalues"]))

        traj_points = []
        for layer in f["layer_indices"]:
            x, y, z = get_coords("dark_traj", fid, layer)
            if x is not None:
                traj_points.append((x, y, z, eig_by_layer.get(layer, 0), layer))

        if len(traj_points) < 2:
            continue

        # Determine if this feature has any labeled layers
        f_labels = label_map.get(fid, {})
        has_labels = any(lbl[0] is not None for lbl in f_labels.values())

        # Determine legend group and name
        if has_labels:
            labeled_concepts = sorted(set(
                lbl[0] for lbl in f_labels.values() if lbl[0] is not None
            ))
            if len(labeled_concepts) > 1:
                legend_name = f"F{fid:03d} ({' → '.join(labeled_concepts)})"
            else:
                legend_name = f"F{fid:03d} ({labeled_concepts[0]})"
            legend_group = f"labeled_F{fid:03d}"
            show_legend = True
        else:
            legend_name = f"F{fid:03d}"
            legend_group = "dark_matter"
            show_legend = False

        first_seg = True
        for i in range(len(traj_points) - 1):
            x0, y0, z0, eig0, l0 = traj_points[i]
            x1, y1, z1, eig1, l1 = traj_points[i + 1]

            seg_eig = (eig0 + eig1) / 2
            frac = min(1.0, seg_eig / eig_anchor)
            seg_width = 0.5 + 4.5 * frac

            # Determine segment color: concept color if labeled, blue if dark
            lbl0 = f_labels.get(l0, (None, 0))
            lbl1 = f_labels.get(l1, (None, 0))

            # Use the label with higher cos for this segment
            if lbl0[0] is not None and lbl1[0] is not None:
                seg_concept = lbl0[0] if lbl0[1] >= lbl1[1] else lbl1[0]
                seg_cos = max(lbl0[1], lbl1[1])
            elif lbl0[0] is not None:
                seg_concept = lbl0[0]
                seg_cos = lbl0[1]
            elif lbl1[0] is not None:
                seg_concept = lbl1[0]
                seg_cos = lbl1[1]
            else:
                seg_concept = None
                seg_cos = 0

            if seg_concept and seg_concept in CONCEPT_COLORS:
                base_rgb = hex_to_rgb(CONCEPT_COLORS[seg_concept])
                # Brightness scales with both eigenvalue and alignment strength
                dim = 0.3 + 0.7 * frac
                cr = int(base_rgb[0] * dim)
                cg = int(base_rgb[1] * dim)
                cb = int(base_rgb[2] * dim)
                ca = 0.25 + 0.65 * frac
            else:
                # Dark matter blue
                cr = int(30 + 100 * frac)
                cg = int(50 + 130 * frac)
                cb = int(120 + 135 * frac)
                ca = 0.12 + 0.58 * frac

            seg_color = f"rgba({cr},{cg},{cb},{ca:.2f})"

            concept_str = f"{seg_concept} (cos={seg_cos:.2f})" if seg_concept else "unlabeled"
            hover = (
                f"<b>Feature F{fid:03d}</b><br>"
                f"Lifespan: L{f['birth_layer']}–L{f['death_layer']} ({lifespan} layers)<br>"
                f"Segment: L{l0}–L{l1} — {concept_str}<br>"
                f"Eigenvalue: {seg_eig:.1f}<br>"
                f"Peak eigenvalue: {f['peak_eigenvalue']:.1f}"
            )

            fig.add_trace(go.Scatter3d(
                x=[x0, x1], y=[y0, y1], z=[z0, z1],
                mode="lines",
                line=dict(color=seg_color, width=seg_width),
                legendgroup=legend_group,
                showlegend=first_seg and show_legend,
                name=legend_name if first_seg else None,
                hovertemplate=hover + "<extra></extra>",
            ))
            first_seg = False

        # Peak marker
        px, py, pz = get_coords("dark_peak", fid, f["peak_layer"])
        if px is not None:
            peak_frac = min(1.0, f["peak_eigenvalue"] / eig_anchor)
            peak_lbl = f_labels.get(f["peak_layer"], (None, 0))

            if peak_lbl[0] and peak_lbl[0] in CONCEPT_COLORS:
                peak_color = CONCEPT_COLORS[peak_lbl[0]]
            else:
                pr = int(30 + 100 * peak_frac)
                pg = int(50 + 130 * peak_frac)
                pb = int(120 + 135 * peak_frac)
                pa = 0.3 + 0.6 * peak_frac
                peak_color = f"rgba({pr},{pg},{pb},{pa:.2f})"

            peak_size = max(3, min(10, 3 + 7 * peak_frac))

            fig.add_trace(go.Scatter3d(
                x=[px], y=[py], z=[pz],
                mode="markers",
                marker=dict(size=peak_size, color=peak_color,
                            symbol="diamond",
                            line=dict(width=0.5, color="rgba(150,200,255,0.5)")),
                legendgroup=legend_group,
                showlegend=False,
                hovertemplate=(
                    f"<b>F{fid:03d} peak</b><br>"
                    f"Layer {f['peak_layer']}<br>"
                    f"Eigenvalue: {f['peak_eigenvalue']:.1f}"
                    "<extra></extra>"
                ),
            ))

    # ── Transient stars ──
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
        trans_sizes.append(max(2, min(6, 2 + 4 * frac)))

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
            marker=dict(size=trans_sizes, color=trans_colors, symbol="diamond"),
            legendgroup="transient",
            showlegend=False,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=trans_hovers,
        ))

    # ── Concept trajectories ──
    all_concept_seps = []
    for concept, cdata in model_data["concepts"].items():
        for m in cdata["layer_data"]["metrics"]:
            all_concept_seps.append(m["separation_fisher"])
    concept_sep_anchor = float(np.percentile(all_concept_seps, 90)) if all_concept_seps else 1.0

    for concept, cdata in model_data["concepts"].items():
        metrics = cdata["layer_data"]["metrics"]
        lm = [LayerMetrics(m["layer"], m["separation_fisher"], m["coherence"], m["velocity"])
              for m in metrics]
        regions = find_caz_regions_scored(lm).regions

        base_rgb = hex_to_rgb(CONCEPT_COLORS[concept])
        ctype = CONCEPT_TYPE[concept]

        traj = []
        for m in metrics:
            x, y, z = get_coords("concept", concept, m["layer"])
            if x is not None:
                traj.append((x, y, z, m["separation_fisher"], m["layer"]))

        first_seg = True
        for i in range(len(traj) - 1):
            x0, y0, z0, sep0, l0 = traj[i]
            x1, y1, z1, sep1, l1 = traj[i + 1]

            seg_sep = (sep0 + sep1) / 2
            sep_frac = min(1.0, seg_sep / concept_sep_anchor)

            seg_width = 1.0 + 5.0 * sep_frac
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

        # CAZ peak markers
        for region in regions:
            pk = region.peak
            x, y, z = get_coords("concept", concept, pk)
            if x is None:
                continue

            score = region.caz_score
            size = max(5, min(20, 6 + 12 * np.log1p(score * 10)))
            opacity = min(0.95, max(0.3, 0.3 + 0.65 * np.log1p(score * 5) / np.log1p(5)))

            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode="markers",
                marker=dict(size=size, color=CONCEPT_COLORS[concept], opacity=opacity,
                            line=dict(width=1, color="white")),
                legendgroup=concept,
                showlegend=False,
                hovertemplate=(
                    f"<b>{concept}</b> CAZ peak<br>"
                    f"Layer {pk} ({region.depth_pct:.0f}%)<br>"
                    f"Score: {score:.3f}<br>"
                    f"Separation: {region.peak_separation:.3f}"
                    "<extra></extra>"
                ),
            ))

    # ── Legend entries ──
    n_dark = sum(1 for f in persistent if not any(
        lbl[0] is not None for lbl in label_map.get(f["feature_id"], {}).values()
    ))
    n_labeled_feat = sum(1 for f in persistent if any(
        lbl[0] is not None for lbl in label_map.get(f["feature_id"], {}).values()
    ))

    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(size=6, color="rgba(80,140,220,0.6)", symbol="diamond"),
        name=f"dark matter ({n_dark} unlabeled)",
        legendgroup="dark_matter",
        showlegend=True,
    ))
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(size=3, color="rgba(200,190,150,0.4)", symbol="diamond"),
        name=f"transient sparks ({len(transient)})",
        legendgroup="transient",
        showlegend=True,
    ))

    # Layout
    total_features = feature_map["n_features"]
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Labeled Dark Matter — {model_short}</b><br>"
                f"<sub>{n_labeled_feat} features matched to concepts (layer-aware), "
                f"{n_dark} still unlabeled, {len(transient)} transient sparks. "
                f"Color-shifting features = concept handoffs.</sub>"
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
    parser = argparse.ArgumentParser(description="Labeled dark matter visualization")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-1.4b")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading model data for %s...", args.model)
    model_data = load_model_data(args.model)
    if not model_data:
        log.error("No CAZ extraction data for %s", args.model)
        return

    log.info("Loading deep dive results...")
    feature_map, directions, feature_labels = load_deep_dive(args.model)
    if not feature_map:
        log.error("No deep dive data for %s", args.model)
        return
    if not feature_labels:
        log.error("No feature_labels.json — run the labeling step first")
        return

    log.info("Building labeled dark matter visualization...")
    fig = build_figure(model_data, feature_map, directions, feature_labels)

    model_slug = args.model.split("/")[-1]
    out_path = OUT_DIR / f"labeled_dark_matter_{model_slug}.html"
    fig.write_html(str(out_path), include_plotlyjs=True)
    log.info("-> %s", out_path)


if __name__ == "__main__":
    main()

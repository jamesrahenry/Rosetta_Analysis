#!/usr/bin/env python3
"""
viz_caz_growth.py — Animated layer-by-layer growth of the activation map.

Steps through layers smoothly, showing features being born, growing,
and dying. Concept tubes swell at CAZ peaks, dark matter lines extend,
transient stars flash and vanish.

Outputs: GIF animation + interactive HTML.

Usage:
    python src/viz_caz_growth.py --model EleutherAI/pythia-1.4b
    python src/viz_caz_growth.py --model EleutherAI/pythia-1.4b --fps 12 --dpi 120
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from rosetta_tools.caz import LayerMetrics, find_caz_regions_scored
from viz_style import concept_color, CONCEPT_COLORS, CONCEPT_TYPE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = Path(__file__).resolve().parents[1] / "visualizations" / "cazstellations"

CONCEPTS = ["credibility", "certainty", "sentiment", "moral_valence", "causation", "temporal_order", "negation"]



def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> tuple:
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    return (r, g, b, alpha)


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
        return feature_map, directions
    return None, None


def build_animation(model_data, feature_map, directions, fps=8, dpi=100):
    n_layers = model_data["n_layers"]
    hidden_dim = model_data["hidden_dim"]
    model_short = model_data["model_id"].split("/")[-1]

    # ── Collect all vectors for PCA ──
    all_vectors = []
    all_labels = []

    for concept, cdata in model_data["concepts"].items():
        for m in cdata["layer_data"]["metrics"]:
            vec = np.array(m["dom_vector"], dtype=np.float64)
            if len(vec) == hidden_dim:
                all_vectors.append(vec)
                all_labels.append(("concept", concept, m["layer"]))

    persistent = [f for f in feature_map["features"] if f["lifespan"] >= 3]
    transient = [f for f in feature_map["features"] if f["lifespan"] <= 2]

    for f in persistent:
        for li, layer in enumerate(f["layer_indices"]):
            pi = f["pc_indices"][li]
            if layer in directions and pi < len(directions[layer]):
                vec = directions[layer][pi]
                if len(vec) == hidden_dim:
                    all_vectors.append(vec)
                    all_labels.append(("dark", f["feature_id"], layer))

    for f in transient:
        pk = f["peak_layer"]
        pi = f["pc_indices"][f["layer_indices"].index(pk)] if pk in f["layer_indices"] else f["pc_indices"][0]
        if pk in directions and pi < len(directions[pk]):
            vec = directions[pk][pi]
            if len(vec) == hidden_dim:
                all_vectors.append(vec)
                all_labels.append(("transient", f["feature_id"], pk))

    all_vectors = np.array(all_vectors)
    log.info("PCA on %d vectors", len(all_vectors))

    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(all_vectors)

    # Build coordinate lookups
    coord_map = {}  # (type, id, layer) -> (x, y, z)
    for i, (lt, lid, ll) in enumerate(all_labels):
        z = 100 * ll / n_layers
        coord_map[(lt, lid, ll)] = (coords_2d[i, 0], coords_2d[i, 1], z)

    # ── Precompute concept separation per layer ──
    concept_sep = {}  # (concept, layer) -> separation
    for concept, cdata in model_data["concepts"].items():
        for m in cdata["layer_data"]["metrics"]:
            concept_sep[(concept, m["layer"])] = m["separation_fisher"]

    all_seps = list(concept_sep.values())
    sep_anchor = float(np.percentile(all_seps, 90)) if all_seps else 1.0

    # ── Precompute eigenvalue anchor ──
    peak_eigs = [f["peak_eigenvalue"] for f in persistent]
    eig_anchor = float(np.percentile(peak_eigs, 90)) if peak_eigs else 1.0

    # ── Precompute CAZ regions ──
    caz_peaks = {}  # (concept, layer) -> caz_score
    for concept, cdata in model_data["concepts"].items():
        metrics = cdata["layer_data"]["metrics"]
        lm = [LayerMetrics(m["layer"], m["separation_fisher"], m["coherence"], m["velocity"])
              for m in metrics]
        for region in find_caz_regions_scored(lm).regions:
            caz_peaks[(concept, region.peak)] = region.caz_score

    # ── Set up figure ──
    fig = plt.figure(figsize=(14, 10), facecolor="#030310")
    ax = fig.add_subplot(111, projection="3d", facecolor="#030310")

    # Axis ranges
    pad = 0.1
    x_all = coords_2d[:, 0]
    y_all = coords_2d[:, 1]
    x_range = (x_all.min() - pad * (x_all.max() - x_all.min()),
               x_all.max() + pad * (x_all.max() - x_all.min()))
    y_range = (y_all.min() - pad * (y_all.max() - y_all.min()),
               y_all.max() + pad * (y_all.max() - y_all.min()))

    def style_axes():
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_zlim(0, 100)
        ax.set_zticks(range(0, 101, 20))
        ax.set_zticklabels([f"{d}%" for d in range(0, 101, 20)],
                           fontsize=7, color="#606880")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='x', colors='#030310')
        ax.tick_params(axis='y', colors='#030310')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#101530')
        ax.yaxis.pane.set_edgecolor('#101530')
        ax.zaxis.pane.set_edgecolor('#101530')
        ax.grid(True, alpha=0.05, color="#304060")
        ax.set_zlabel("Depth", fontsize=9, color="#808898", labelpad=-2)

    # ── Animation frames ──
    # Each frame reveals one more layer. We interpolate sub-layers for smoothness.
    # Total frames: n_layers * steps_per_layer + hold_at_end
    steps_per_layer = 3
    hold_frames = fps * 3  # hold 3 seconds at end
    total_frames = n_layers * steps_per_layer + hold_frames

    log.info("Rendering %d frames (%d layers × %d steps + %d hold) at %d fps",
             total_frames, n_layers, steps_per_layer, hold_frames, fps)

    def update(frame_idx):
        ax.cla()
        style_axes()

        # Current layer progress (can be fractional)
        if frame_idx >= n_layers * steps_per_layer:
            current_layer = n_layers - 1
            layer_frac = 1.0
        else:
            current_layer = frame_idx // steps_per_layer
            layer_frac = (frame_idx % steps_per_layer) / steps_per_layer

        current_depth = current_layer + layer_frac

        # Title with layer counter
        depth_pct = 100 * current_depth / n_layers
        ax.set_title(
            f"{model_short} — Layer {current_layer}/{n_layers-1} ({depth_pct:.0f}%)",
            color="#C8D2E6", fontsize=14, pad=10,
        )

        # Slow rotation
        azim = -60 + 90 * (frame_idx / total_frames)
        elev = 25 + 5 * np.sin(2 * np.pi * frame_idx / total_frames)
        ax.view_init(elev=elev, azim=azim)

        # ── Draw dark matter lines up to current layer ──
        for f in persistent:
            eig_by_layer = dict(zip(f["layer_indices"], f["eigenvalues"]))
            visible_layers = [l for l in f["layer_indices"] if l <= current_layer]
            if len(visible_layers) < 2:
                continue

            xs, ys, zs = [], [], []
            for layer in visible_layers:
                c = coord_map.get(("dark", f["feature_id"], layer))
                if c:
                    xs.append(c[0])
                    ys.append(c[1])
                    zs.append(c[2])

            if len(xs) < 2:
                continue

            # Color by peak eigenvalue fraction
            peak_frac = min(1.0, f["peak_eigenvalue"] / eig_anchor)
            r = 0.12 + 0.39 * peak_frac
            g = 0.20 + 0.51 * peak_frac
            b = 0.47 + 0.53 * peak_frac
            a = 0.10 + 0.50 * peak_frac
            lw = 0.4 + 3.0 * peak_frac

            # Fade in the newest segment
            ax.plot(xs, ys, zs, color=(r, g, b, a), linewidth=lw, zorder=1)

        # ── Draw transient stars at their layer ──
        for f in transient:
            pk = f["peak_layer"]
            if pk > current_layer:
                continue
            # Fade: full brightness when just appeared, dim after
            age = current_layer - pk
            if age > 3:
                continue  # vanish after 3 layers

            c = coord_map.get(("transient", f["feature_id"], pk))
            if not c:
                continue

            fade = max(0.0, 1.0 - age / 3.0)
            eig_frac = min(1.0, f["peak_eigenvalue"] / eig_anchor)
            size = max(4, min(20, 4 + 16 * eig_frac)) * fade
            r = (0.55 + 0.45 * eig_frac) * fade
            g = (0.51 + 0.41 * eig_frac) * fade
            b = (0.39 + 0.31 * eig_frac) * fade
            a = (0.15 + 0.55 * eig_frac) * fade

            ax.scatter([c[0]], [c[1]], [c[2]],
                       c=[(r, g, b, a)], s=size, marker='*',
                       edgecolors='none', zorder=2)

        # ── Draw concept trajectories up to current layer ──
        for concept, cdata in model_data["concepts"].items():
            metrics = cdata["layer_data"]["metrics"]
            base_rgba = hex_to_rgba(CONCEPT_COLORS[concept])

            visible = [m for m in metrics if m["layer"] <= current_layer]
            if len(visible) < 2:
                continue

            xs, ys, zs = [], [], []
            for m in visible:
                c = coord_map.get(("concept", concept, m["layer"]))
                if c:
                    xs.append(c[0])
                    ys.append(c[1])
                    zs.append(c[2])

            if len(xs) < 2:
                continue

            # Base line — thin
            ax.plot(xs, ys, zs,
                    color=(*base_rgba[:3], 0.4),
                    linewidth=1.5, zorder=3)

            # Thicker overlay segments where separation is high
            for i in range(len(visible) - 1):
                l0 = visible[i]["layer"]
                l1 = visible[i + 1]["layer"]
                sep = (concept_sep.get((concept, l0), 0) +
                       concept_sep.get((concept, l1), 0)) / 2
                frac = min(1.0, sep / sep_anchor)

                if frac > 0.15:  # only draw thick overlay where there's signal
                    c0 = coord_map.get(("concept", concept, l0))
                    c1 = coord_map.get(("concept", concept, l1))
                    if c0 and c1:
                        lw = 1.5 + 5.0 * frac
                        a = 0.3 + 0.6 * frac
                        ax.plot([c0[0], c1[0]], [c0[1], c1[1]], [c0[2], c1[2]],
                                color=(*base_rgba[:3], a),
                                linewidth=lw, zorder=4)

            # CAZ peak markers
            for m in visible:
                layer = m["layer"]
                if (concept, layer) in caz_peaks:
                    score = caz_peaks[(concept, layer)]
                    c = coord_map.get(("concept", concept, layer))
                    if c:
                        size = max(15, min(80, 15 + 60 * np.log1p(score * 10)))
                        a = min(0.9, 0.3 + 0.6 * np.log1p(score * 5) / np.log1p(5))
                        ax.scatter([c[0]], [c[1]], [c[2]],
                                   c=[(*base_rgba[:3], a)],
                                   s=size, edgecolors='white',
                                   linewidths=0.5, zorder=5)

        return []

    anim = FuncAnimation(fig, update, frames=total_frames,
                         interval=1000 // fps, blit=False)
    return fig, anim


def main():
    parser = argparse.ArgumentParser(description="Animated CAZ growth visualization")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-1.4b")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=100)
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

    log.info("Building animation...")
    fig, anim = build_animation(model_data, feature_map, directions,
                                fps=args.fps, dpi=args.dpi)

    model_slug = args.model.split("/")[-1]

    # Save GIF
    gif_path = OUT_DIR / f"caz_growth_{model_slug}.gif"
    log.info("Saving GIF to %s (this takes a minute)...", gif_path)
    writer = PillowWriter(fps=args.fps)
    anim.save(str(gif_path), writer=writer, dpi=args.dpi)
    log.info("-> %s (%.1f MB)", gif_path, gif_path.stat().st_size / 1e6)

    plt.close(fig)
    log.info("Done.")


if __name__ == "__main__":
    main()

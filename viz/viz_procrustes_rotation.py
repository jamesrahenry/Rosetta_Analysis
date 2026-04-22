#!/usr/bin/env python3
"""Animated Procrustes rotation — single all-concept rotation, depth buttons.

Shows one concept's dom_vector trajectory (the "dragon") from each model,
animating the Procrustes rotation. Depth buttons control how far the
dragons grow from shallow to deep.

The rotation is fitted across ALL concepts simultaneously (not per-concept),
matching the science: one rotation per model pair aligns the entire semantic space.

Usage:
    python src/viz_procrustes_rotation.py [concept] [results_dir] [out_dir]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from viz_style import concept_color

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CONCEPTS = ["credibility", "certainty", "sentiment", "moral_valence",
            "causation", "temporal_order", "negation"]

FAMILY_COLORS = {               # lowercase keys — matched by _family()
    "pythia":  "#1565C0",       # spec: Pythia blue
    "gpt2":    "#558B2F",       # spec: GPT-2 green
    "opt":     "#6A1B9A",       # spec: OPT purple
    "qwen":    "#E65100",       # spec: Qwen orange
    "gemma":   "#00695C",       # spec: Gemma teal
    "mistral": "#546E7A",       # spec: Other blue-grey
    "phi":     "#4E342E",       # spec: Phi brown
}


def _family(mid: str) -> str:
    for fam in FAMILY_COLORS:
        if fam in mid.lower():
            return fam
    return "other"


def _short(mid: str) -> str:
    return mid.split("/")[-1]


def load_models(results_dir: Path):
    """Load dom_vector trajectories for all models x concepts."""
    models = {}
    for d in sorted(results_dir.iterdir()):
        sf = d / "run_summary.json"
        if not sf.exists():
            continue
        try:
            summary = json.load(open(sf))
            mid = summary.get("model_id")
            if not mid or mid in models:
                continue
        except (json.JSONDecodeError, KeyError):
            continue

        mdata = {"model_id": mid, "concepts": {}}
        for concept in CONCEPTS:
            caz_file = d / f"caz_{concept}.json"
            if not caz_file.exists():
                continue
            data = json.load(open(caz_file))
            n_layers = int(data["n_layers"])
            metrics = data["layer_data"]["metrics"]
            trajectory = []
            for m in metrics:
                trajectory.append({
                    "layer": m["layer"],
                    "depth_pct": 100 * m["layer"] / n_layers,
                    "vec": np.array(m["dom_vector"], dtype=np.float64),
                    "separation": m["separation_fisher"],
                })
            mdata["concepts"][concept] = trajectory
            mdata["n_layers"] = n_layers
            mdata["hidden_dim"] = int(data["hidden_dim"])

        if mdata.get("concepts"):
            models[mid] = mdata

    return models


def resample(trajectory, n_points=25):
    """Resample trajectory to uniform depth grid."""
    grid = np.linspace(0, 100, n_points)
    depths = np.array([t["depth_pct"] for t in trajectory])
    result = []
    for target in grid:
        idx = np.argmin(np.abs(depths - target))
        result.append(trajectory[idx])
    return result, grid


def _largest_per_family(models):
    """Filter to largest model per family (by hidden_dim, then layer count)."""
    best = {}
    for mid, mdata in models.items():
        fam = _family(mid)
        dim = mdata.get("hidden_dim", 0)
        n = mdata.get("n_layers", 0)
        score = (dim, n)
        if fam not in best or score > best[fam][1]:
            best[fam] = (mid, score)
    keep = {v[0] for v in best.values()}
    return {mid: mdata for mid, mdata in models.items() if mid in keep}


def compute_all_concept_rotation(models, n_depth=25):
    """Compute a single Procrustes rotation per model using ALL concepts."""
    model_stacked = {}
    for mid, mdata in models.items():
        vecs_all = []
        for concept in CONCEPTS:
            if concept not in mdata["concepts"]:
                continue
            traj = mdata["concepts"][concept]
            if len(traj) < 3:
                continue
            resampled, _ = resample(traj, n_depth)
            vecs = [t["vec"] for t in resampled]
            if any(len(v) != mdata["hidden_dim"] for v in vecs):
                continue
            vecs_all.extend(vecs)
        if vecs_all:
            model_stacked[mid] = np.array(vecs_all)

    if len(model_stacked) < 2:
        return {}, {}, None

    k = 50
    model_pca = {}
    for mid, vecs in model_stacked.items():
        vecs_c = vecs - vecs.mean(axis=0)
        actual_k = min(k, vecs_c.shape[0] - 1, vecs_c.shape[1])
        pca = PCA(n_components=actual_k)
        coords = pca.fit_transform(vecs_c)
        model_pca[mid] = (pca, coords, vecs.mean(axis=0))

    max_k = max(c.shape[1] for _, c, _ in model_pca.values())
    for mid in model_pca:
        pca, coords, mean = model_pca[mid]
        if coords.shape[1] < max_k:
            pad = np.zeros((coords.shape[0], max_k - coords.shape[1]))
            coords = np.hstack([coords, pad])
        model_pca[mid] = (pca, coords, mean)

    ref_mid = max(model_pca, key=lambda m: model_pca[m][1].shape[0])
    ref_coords = model_pca[ref_mid][1]

    rotations = {}
    for mid in model_pca:
        _, coords, _ = model_pca[mid]
        if mid == ref_mid:
            rotations[mid] = np.eye(max_k)
        else:
            n_common = min(coords.shape[0], ref_coords.shape[0])
            R, _ = orthogonal_procrustes(coords[:n_common], ref_coords[:n_common])
            rotations[mid] = R

    return rotations, model_pca, ref_mid


def build_trajectories(models, concept, rotations, model_pca, ref_mid, n_depth=25):
    """Build raw and aligned 3D trajectories for one concept."""
    model_trajs = {}
    for mid, mdata in models.items():
        if concept not in mdata["concepts"] or mid not in rotations:
            continue
        traj = mdata["concepts"][concept]
        if len(traj) < 3:
            continue
        resampled, grid = resample(traj, n_depth)
        vecs = [t["vec"] for t in resampled]
        if any(len(v) != mdata["hidden_dim"] for v in vecs):
            continue
        model_trajs[mid] = {
            "vecs": np.array(vecs),
            "seps": [t["separation"] for t in resampled],
        }

    if not model_trajs:
        return {}, {}, [], {}

    raw_k = {}
    aligned_k = {}
    for mid, td in model_trajs.items():
        pca, _, mean = model_pca[mid]
        vecs_c = td["vecs"] - mean
        coords = pca.transform(vecs_c)
        R = rotations[mid]
        if coords.shape[1] < R.shape[0]:
            pad = np.zeros((coords.shape[0], R.shape[0] - coords.shape[1]))
            coords = np.hstack([coords, pad])
        raw_k[mid] = coords
        aligned_k[mid] = coords @ R

    # Fit PCA on reference model's aligned coords only — defines a fixed space
    # so the interpolation is a pure rotation, not a projection shift
    pca2 = PCA(n_components=2).fit(aligned_k[ref_mid])
    depth_grid = np.linspace(0, 1, n_depth)

    raw_3d = {}
    aligned_3d = {}
    for mid in raw_k:
        raw_xy = pca2.transform(raw_k[mid])
        aln_xy = pca2.transform(aligned_k[mid])
        raw_3d[mid] = np.column_stack([raw_xy, depth_grid])
        aligned_3d[mid] = np.column_stack([aln_xy, depth_grid])

    sep_data = {mid: model_trajs[mid]["seps"] for mid in model_trajs}
    model_list = sorted(raw_3d.keys())
    return raw_3d, aligned_3d, model_list, sep_data


def _find_peaks(seps, n):
    """Find local peaks in separation profile."""
    peaks = []
    for i in range(1, n - 1):
        if seps[i] > seps[i-1] and seps[i] >= seps[i+1]:
            peaks.append((i, seps[i]))
    if not peaks and seps:
        i = int(np.argmax(seps))
        peaks.append((i, seps[i]))
    return peaks


def _make_traces(model_list, raw_3d, aligned_3d, sep_data, t, depth_cut, n_depth, show_legend):
    """Build traces for one frame. Always returns same number of traces per model."""
    traces = []
    for mid in model_list:
        fam = _family(mid)
        color = FAMILY_COLORS.get(fam, "#888888")
        name = _short(mid)
        seps = sep_data.get(mid, [0] * n_depth)

        raw = raw_3d[mid][:depth_cut]
        aln = aligned_3d[mid][:depth_cut]
        interp = (1 - t) * raw + t * aln

        # Dragon body
        traces.append(go.Scatter3d(
            x=interp[:, 0].tolist(), y=interp[:, 1].tolist(), z=interp[:, 2].tolist(),
            mode="lines",
            line=dict(color=color, width=3),
            opacity=0.6,
            name=f"{name} ({fam})",
            showlegend=show_legend,
            legendgroup=mid,
            hoverinfo="skip",
        ))

        # CAZ peaks
        peaks = _find_peaks(seps[:depth_cut], depth_cut)
        # Always emit exactly one peak trace (invisible if no peak)
        if peaks:
            pk_idx = peaks[0][0]
            traces.append(go.Scatter3d(
                x=[float(interp[pk_idx, 0])], y=[float(interp[pk_idx, 1])],
                z=[float(interp[pk_idx, 2])],
                mode="markers",
                marker=dict(size=3, color=color, opacity=0.8, line=dict(width=0)),
                showlegend=False, legendgroup=mid, hoverinfo="skip",
            ))
        else:
            traces.append(go.Scatter3d(
                x=[0], y=[0], z=[0], mode="markers",
                marker=dict(size=0, opacity=0), visible=False,
                showlegend=False, legendgroup=mid, hoverinfo="skip",
            ))

    return traces


def build_animation(models, concept="causation", n_frames_rotate=20,
                    n_depth=25, largest_only=False):
    """Build animated figure with rotation play button and depth buttons."""

    if largest_only:
        models = _largest_per_family(models)

    log.info("  Computing all-concept Procrustes rotations...")
    rotations, model_pca, ref_mid = compute_all_concept_rotation(models, n_depth)
    if not rotations:
        return None
    log.info("  Reference: %s, %d models", _short(ref_mid), len(rotations))

    raw_3d, aligned_3d, model_list, sep_data = build_trajectories(
        models, concept, rotations, model_pca, ref_mid, n_depth
    )
    if not model_list:
        return None

    # Depth presets
    depth_levels = [
        ("10%", max(1, n_depth // 10)),
        ("25%", max(1, n_depth // 4)),
        ("50%", n_depth // 2),
        ("75%", 3 * n_depth // 4),
        ("100%", n_depth),
    ]

    # Build rotation frames at full depth (for play button)
    frames = []
    for ri in range(n_frames_rotate + 1):
        t = ri / n_frames_rotate
        traces = _make_traces(model_list, raw_3d, aligned_3d, sep_data,
                              t, n_depth, n_depth, show_legend=(ri == 0))
        frames.append(go.Frame(data=traces, name=f"r{ri}"))

    # Build depth frames at aligned state (t=1) for depth buttons
    for label, cut in depth_levels:
        traces = _make_traces(model_list, raw_3d, aligned_3d, sep_data,
                              1.0, cut, n_depth, show_legend=False)
        frames.append(go.Frame(data=traces, name=f"depth_{label}"))

    # Build depth frames at unaligned state (t=0) for depth buttons
    for label, cut in depth_levels:
        traces = _make_traces(model_list, raw_3d, aligned_3d, sep_data,
                              0.0, cut, n_depth, show_legend=False)
        frames.append(go.Frame(data=traces, name=f"depth_raw_{label}"))

    # Axis range
    all_pts = np.vstack(list(raw_3d.values()) + list(aligned_3d.values()))
    xy_rng = max(abs(all_pts[:, :2].min()), abs(all_pts[:, :2].max())) * 1.3

    fig = go.Figure(data=frames[0].data, frames=frames)

    # Depth buttons on the right side, vertically stacked
    depth_buttons = []
    for label, cut in depth_levels:
        depth_buttons.append(dict(
            label=label,
            method="animate",
            args=[[f"depth_{label}"],
                  {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
        ))

    fig.update_layout(
        title=dict(
            text=(f"Procrustes Rotation — '{concept}' across {len(model_list)} architectures<br>"
                  f"<sub>Single all-concept rotation per model. "
                  f"Bottom = shallow, top = deep.</sub>"),
            font_size=16,
        ),
        scene=dict(
            xaxis=dict(range=[-xy_rng, xy_rng], title="", showticklabels=False,
                       showgrid=True, gridcolor="#1a1a2e"),
            yaxis=dict(range=[-xy_rng, xy_rng], title="", showticklabels=False,
                       showgrid=True, gridcolor="#1a1a2e"),
            zaxis=dict(range=[-0.05, 1.05], title="Depth", showticklabels=True,
                       gridcolor="#1a1a2e",
                       tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                       ticktext=["0%", "25%", "50%", "75%", "100%"]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1.5),
            bgcolor="#0d1117",
            domain=dict(x=[0, 0.85]),
        ),
        updatemenus=[
            # Play / Reset buttons (bottom center)
            dict(
                type="buttons", showactive=False,
                y=-0.02, x=0.35, xanchor="center",
                buttons=[
                    dict(label="&#9654; Rotate to Alignment",
                         method="animate",
                         args=[[f"r{ri}" for ri in range(n_frames_rotate + 1)],
                               {"frame": {"duration": 100, "redraw": True},
                                "fromcurrent": False}]),
                    dict(label="&#9724; Reset (raw)",
                         method="animate",
                         args=[["r0"], {"frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate"}]),
                ],
            ),
            # Depth buttons (right side, vertical)
            dict(
                type="buttons", showactive=True,
                direction="down",
                x=1.02, y=0.8, xanchor="left", yanchor="top",
                bgcolor="rgba(30,30,50,0.8)",
                font=dict(color="white", size=11),
                buttons=[dict(label="Depth", method="skip", args=[None])] + depth_buttons,
            ),
        ],
        height=800, width=1100,
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        legend=dict(x=0.01, y=0.98, font_size=11, bgcolor="rgba(13,17,23,0.8)"),
        margin=dict(r=120),
    )

    return fig


def _make_sequential_traces(model_list, raw_3d, aligned_3d, sep_data,
                            per_model_t, n_depth, show_legend):
    """Build traces where each model has its own rotation progress (0-1)."""
    traces = []
    for mid in model_list:
        fam = _family(mid)
        color = FAMILY_COLORS.get(fam, "#888888")
        name = _short(mid)
        seps = sep_data.get(mid, [0] * n_depth)
        t = per_model_t.get(mid, 0.0)

        raw = raw_3d[mid]
        aln = aligned_3d[mid]
        interp = (1 - t) * raw + t * aln

        traces.append(go.Scatter3d(
            x=interp[:, 0].tolist(), y=interp[:, 1].tolist(), z=interp[:, 2].tolist(),
            mode="lines",
            line=dict(color=color, width=4 if t > 0 and t < 1 else 3),
            opacity=0.8 if t > 0 and t < 1 else 0.5,
            name=f"{name} ({fam})",
            showlegend=show_legend,
            legendgroup=mid,
            hoverinfo="skip",
        ))

        peaks = _find_peaks(seps, n_depth)
        if peaks:
            pk_idx = peaks[0][0]
            traces.append(go.Scatter3d(
                x=[float(interp[pk_idx, 0])], y=[float(interp[pk_idx, 1])],
                z=[float(interp[pk_idx, 2])],
                mode="markers",
                marker=dict(size=3, color=color, opacity=0.8, line=dict(width=0)),
                showlegend=False, legendgroup=mid, hoverinfo="skip",
            ))
        else:
            traces.append(go.Scatter3d(
                x=[0], y=[0], z=[0], mode="markers",
                marker=dict(size=0, opacity=0), visible=False,
                showlegend=False, legendgroup=mid, hoverinfo="skip",
            ))

    return traces


def build_sequential_animation(models, concept="causation", n_steps_per=10,
                               n_depth=25, largest_only=False):
    """Build animation where each model rotates one at a time."""

    if largest_only:
        models = _largest_per_family(models)

    log.info("  Computing all-concept Procrustes rotations...")
    rotations, model_pca, ref_mid = compute_all_concept_rotation(models, n_depth)
    if not rotations:
        return None
    log.info("  Reference: %s, %d models", _short(ref_mid), len(rotations))

    raw_3d, aligned_3d, model_list, sep_data = build_trajectories(
        models, concept, rotations, model_pca, ref_mid, n_depth
    )
    if not model_list:
        return None

    # For each model in sequence: animate it from t=0 to t=1
    # while all previous models stay at t=1 and all subsequent stay at t=0
    frames = []
    n_models = len(model_list)

    # Frame 0: everything raw
    per_model_t = {mid: 0.0 for mid in model_list}
    frames.append(go.Frame(
        data=_make_sequential_traces(model_list, raw_3d, aligned_3d, sep_data,
                                     per_model_t, n_depth, show_legend=True),
        name="f0",
    ))

    frame_names = ["f0"]
    fi = 1

    for model_idx, rotating_mid in enumerate(model_list):
        for step in range(1, n_steps_per + 1):
            t = step / n_steps_per
            per_model_t = {}
            for i, mid in enumerate(model_list):
                if i < model_idx:
                    per_model_t[mid] = 1.0  # already aligned
                elif i == model_idx:
                    per_model_t[mid] = t  # currently rotating
                else:
                    per_model_t[mid] = 0.0  # still raw

            fname = f"f{fi}"
            frames.append(go.Frame(
                data=_make_sequential_traces(model_list, raw_3d, aligned_3d, sep_data,
                                             per_model_t, n_depth, show_legend=False),
                name=fname,
            ))
            frame_names.append(fname)
            fi += 1

    # Axis range
    all_pts = np.vstack(list(raw_3d.values()) + list(aligned_3d.values()))
    xy_rng = max(abs(all_pts[:, :2].min()), abs(all_pts[:, :2].max())) * 1.3

    fig = go.Figure(data=frames[0].data, frames=frames)

    fig.update_layout(
        title=dict(
            text=(f"Procrustes Rotation — '{concept}' one model at a time<br>"
                  f"<sub>{n_models} architectures rotate into alignment sequentially. "
                  f"Single all-concept rotation per model.</sub>"),
            font_size=16,
        ),
        scene=dict(
            xaxis=dict(range=[-xy_rng, xy_rng], title="", showticklabels=False,
                       showgrid=True, gridcolor="#1a1a2e"),
            yaxis=dict(range=[-xy_rng, xy_rng], title="", showticklabels=False,
                       showgrid=True, gridcolor="#1a1a2e"),
            zaxis=dict(range=[-0.05, 1.05], title="Depth", showticklabels=True,
                       gridcolor="#1a1a2e",
                       tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                       ticktext=["0%", "25%", "50%", "75%", "100%"]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1.5),
            bgcolor="#0d1117",
        ),
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=-0.02, x=0.4, xanchor="center",
            buttons=[
                dict(label="&#9654; Play",
                     method="animate",
                     args=[frame_names,
                           {"frame": {"duration": 80, "redraw": True},
                            "fromcurrent": False}]),
                dict(label="&#9724; Reset",
                     method="animate",
                     args=[["f0"], {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate"}]),
            ],
        )],
        height=800, width=1000,
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        legend=dict(x=0.01, y=0.98, font_size=11, bgcolor="rgba(13,17,23,0.8)"),
    )

    return fig


def build_concept_sequential(models, n_steps_per=10, n_depth=25):
    """Build animation rotating all models for each concept, one concept at a time."""

    models = _largest_per_family(models)

    log.info("  Computing all-concept Procrustes rotations...")
    rotations, model_pca, ref_mid = compute_all_concept_rotation(models, n_depth)
    if not rotations:
        return None
    log.info("  Reference: %s, %d models", _short(ref_mid), len(rotations))

    # Build trajectories for all concepts
    concept_data = {}
    for concept in CONCEPTS:
        raw_3d, aligned_3d, model_list, sep_data = build_trajectories(
            models, concept, rotations, model_pca, ref_mid, n_depth
        )
        if model_list:
            concept_data[concept] = (raw_3d, aligned_3d, model_list, sep_data)

    if not concept_data:
        return None

    # Use union of all models across concepts
    all_models_set = set()
    for raw_3d, aligned_3d, model_list, sep_data in concept_data.values():
        all_models_set.update(model_list)
    all_models_sorted = sorted(all_models_set)

    CONCEPT_COLORS = {c: concept_color(c) for c in CONCEPTS}

    # For each concept in sequence: rotate all models from raw to aligned
    # Previous concepts stay aligned, upcoming concepts stay raw
    frames = []
    frame_names = []

    # Count traces per frame: each concept × each model × 2 traces (line + peak)
    n_concepts = len(concept_data)
    concepts_ordered = [c for c in CONCEPTS if c in concept_data]

    def make_frame(concept_t_map, show_legend):
        """Build traces for all concepts × all models given per-concept t values."""
        traces = []
        for concept in concepts_ordered:
            if concept not in concept_data:
                continue
            raw_3d, aligned_3d, model_list, sep_data = concept_data[concept]
            t = concept_t_map.get(concept, 0.0)
            is_active = 0 < t < 1
            cc = CONCEPT_COLORS.get(concept, "#888")

            for mid in all_models_sorted:
                if mid not in raw_3d:
                    # Invisible placeholder
                    traces.append(go.Scatter3d(
                        x=[0], y=[0], z=[0], mode="lines",
                        line=dict(color=cc, width=1), visible=False,
                        showlegend=False, hoverinfo="skip",
                    ))
                    traces.append(go.Scatter3d(
                        x=[0], y=[0], z=[0], mode="markers",
                        marker=dict(size=0, opacity=0), visible=False,
                        showlegend=False, hoverinfo="skip",
                    ))
                    continue

                raw = raw_3d[mid]
                aln = aligned_3d[mid]
                interp = (1 - t) * raw + t * aln
                seps = sep_data.get(mid, [0] * n_depth)
                name = _short(mid)

                # Show concept in legend only for first model
                is_first = (mid == all_models_sorted[0] or
                            (mid == model_list[0] and all_models_sorted[0] not in raw_3d))

                traces.append(go.Scatter3d(
                    x=interp[:, 0].tolist(), y=interp[:, 1].tolist(), z=interp[:, 2].tolist(),
                    mode="lines",
                    line=dict(color=cc, width=4 if is_active else 3),
                    opacity=0.8 if is_active else (0.6 if t == 1 else 0.3),
                    name=f"{concept}" if is_first else "",
                    showlegend=(show_legend and is_first),
                    legendgroup=concept,
                    hovertext=f"{name}: {concept}",
                    hoverinfo="text",
                ))

                peaks = _find_peaks(seps, n_depth)
                if peaks:
                    pk_idx = peaks[0][0]
                    traces.append(go.Scatter3d(
                        x=[float(interp[pk_idx, 0])], y=[float(interp[pk_idx, 1])],
                        z=[float(interp[pk_idx, 2])],
                        mode="markers",
                        marker=dict(size=3, color=cc, opacity=0.8, line=dict(width=0)),
                        showlegend=False, legendgroup=concept, hoverinfo="skip",
                    ))
                else:
                    traces.append(go.Scatter3d(
                        x=[0], y=[0], z=[0], mode="markers",
                        marker=dict(size=0, opacity=0), visible=False,
                        showlegend=False, legendgroup=concept, hoverinfo="skip",
                    ))
        return traces

    # Frame 0: everything raw
    concept_t = {c: 0.0 for c in concepts_ordered}
    frames.append(go.Frame(data=make_frame(concept_t, True), name="f0"))
    frame_names.append("f0")
    fi = 1

    for concept_idx, rotating_concept in enumerate(concepts_ordered):
        for step in range(1, n_steps_per + 1):
            t = step / n_steps_per
            concept_t = {}
            for i, c in enumerate(concepts_ordered):
                if i < concept_idx:
                    concept_t[c] = 1.0
                elif i == concept_idx:
                    concept_t[c] = t
                else:
                    concept_t[c] = 0.0

            fname = f"f{fi}"
            frames.append(go.Frame(data=make_frame(concept_t, False), name=fname))
            frame_names.append(fname)
            fi += 1

    # Axis range
    all_pts = []
    for raw_3d, aligned_3d, _, _ in concept_data.values():
        all_pts.extend(list(raw_3d.values()) + list(aligned_3d.values()))
    all_pts = np.vstack(all_pts)
    xy_rng = max(abs(all_pts[:, :2].min()), abs(all_pts[:, :2].max())) * 1.3

    fig = go.Figure(data=frames[0].data, frames=frames)

    fig.update_layout(
        title=dict(
            text=(f"Procrustes Rotation — All 7 concepts, one at a time<br>"
                  f"<sub>{len(all_models_sorted)} architectures × 7 concepts. "
                  f"Single all-concept rotation per model. Same rotation aligns everything.</sub>"),
            font_size=16,
        ),
        scene=dict(
            xaxis=dict(range=[-xy_rng, xy_rng], title="", showticklabels=False,
                       showgrid=True, gridcolor="#1a1a2e"),
            yaxis=dict(range=[-xy_rng, xy_rng], title="", showticklabels=False,
                       showgrid=True, gridcolor="#1a1a2e"),
            zaxis=dict(range=[-0.05, 1.05], title="Depth", showticklabels=True,
                       gridcolor="#1a1a2e",
                       tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                       ticktext=["0%", "25%", "50%", "75%", "100%"]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1.5),
            bgcolor="#0d1117",
        ),
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=-0.02, x=0.4, xanchor="center",
            buttons=[
                dict(label="&#9654; Play",
                     method="animate",
                     args=[frame_names,
                           {"frame": {"duration": 60, "redraw": True},
                            "fromcurrent": False}]),
                dict(label="&#9724; Reset",
                     method="animate",
                     args=[["f0"], {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate"}]),
            ],
        )],
        height=850, width=1050,
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        legend=dict(x=0.01, y=0.98, font_size=12, bgcolor="rgba(13,17,23,0.8)"),
    )

    return fig


def main():
    concept = sys.argv[1] if len(sys.argv) > 1 else "causation"
    results_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("visualizations/cazstellations")
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading models...")
    models = load_models(results_dir)
    log.info("Found %d models", len(models))

    log.info("Building rotation animation for '%s' (all models)...", concept)
    fig = build_animation(models, concept=concept)
    if fig:
        out_path = out_dir / f"procrustes_rotation_{concept}.html"
        fig.write_html(str(out_path))
        log.info("  -> %s", out_path)

    log.info("Building rotation animation for '%s' (largest per family)...", concept)
    fig2 = build_animation(models, concept=concept, largest_only=True)
    if fig2:
        out_path2 = out_dir / f"procrustes_rotation_{concept}_families.html"
        fig2.write_html(str(out_path2))
        log.info("  -> %s", out_path2)

    log.info("Building sequential rotation for '%s' (largest per family)...", concept)
    fig3 = build_sequential_animation(models, concept=concept, largest_only=True)
    if fig3:
        out_path3 = out_dir / f"procrustes_rotation_{concept}_sequential.html"
        fig3.write_html(str(out_path3))
        log.info("  -> %s", out_path3)

    log.info("Building all-concepts sequential rotation (largest per family)...")
    fig4 = build_concept_sequential(models)
    if fig4:
        out_path4 = out_dir / "procrustes_rotation_all_concepts_sequential.html"
        fig4.write_html(str(out_path4))
        log.info("  -> %s", out_path4)


if __name__ == "__main__":
    main()

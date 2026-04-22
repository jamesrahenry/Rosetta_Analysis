#!/usr/bin/env python3
"""
viz_procrustes_cazstellations.py — Unified cross-model CAZ constellation.

For each concept, aligns all models' dom_vector trajectories into a shared
coordinate space via Procrustes, then overlays them in one 3D scene.

If the PRH holds, the trajectories should converge — same geometric path
through activation space, regardless of architecture.

Produces one interactive HTML per concept + one combined "all concepts" view.

Usage:
    python src/viz_procrustes_cazstellations.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from rosetta_tools.caz import LayerMetrics, find_caz_regions_scored
from viz_style import concept_color, CONCEPT_COLORS, CONCEPT_TYPE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = Path(__file__).resolve().parents[1] / "visualizations" / "cazstellations"

CONCEPTS = ["credibility", "certainty", "sentiment", "moral_valence", "causation", "temporal_order", "negation"]

FAMILY_COLORS = {            # lowercase keys — matched by _model_family()
    "pythia": "#1565C0",    # spec: Pythia blue
    "gpt2":   "#558B2F",    # spec: GPT-2 green
    "opt":    "#6A1B9A",    # spec: OPT purple
    "qwen":   "#E65100",    # spec: Qwen orange
    "gemma":  "#00695C",    # spec: Gemma teal
}

FAMILY_DASHES = {
    "pythia": "solid",
    "gpt2":   "dash",
    "opt":    "dot",
    "qwen":   "dashdot",
    "gemma":  "longdash",
}


def _model_family(model_id: str) -> str:
    mid = model_id.lower()
    for fam in FAMILY_COLORS:
        if fam in mid:
            return fam
    return "other"


def _short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def load_all_models():
    """Load dom_vectors for all models × concepts."""
    models = {}
    for d in sorted(RESULTS_DIR.iterdir()):
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

        model_data = {"model_id": mid, "concepts": {}}
        for concept in CONCEPTS:
            caz_file = d / f"caz_{concept}.json"
            if not caz_file.exists():
                continue
            data = json.load(open(caz_file))
            n_layers = int(data["n_layers"])
            metrics = data["layer_data"]["metrics"]

            # Build trajectory: dom_vector at each layer, indexed by depth_pct
            trajectory = []
            for m in metrics:
                vec = np.array(m["dom_vector"], dtype=np.float64)
                trajectory.append({
                    "layer": m["layer"],
                    "depth_pct": 100 * m["layer"] / n_layers,
                    "vec": vec,
                    "separation": m["separation_fisher"],
                    "coherence": m["coherence"],
                })
            model_data["concepts"][concept] = trajectory
            model_data["n_layers"] = n_layers
            model_data["hidden_dim"] = int(data["hidden_dim"])

        if len(model_data["concepts"]) >= 3:
            models[mid] = model_data

    return models


def resample_trajectory(trajectory, n_points=25):
    """Resample trajectory to a common depth grid via nearest-neighbor."""
    grid = np.linspace(0, 100, n_points)
    depths = np.array([t["depth_pct"] for t in trajectory])
    resampled = []
    for target_depth in grid:
        idx = np.argmin(np.abs(depths - target_depth))
        resampled.append(trajectory[idx])
    return resampled, grid


def align_trajectories_for_concept(models: dict, concept: str, n_depth_points: int = 25):
    """Procrustes-align all models' trajectories for one concept into a shared 3D space.

    Strategy:
    1. For each model, collect dom_vectors at n_depth_points evenly spaced depths
    2. PCA each model's vectors down to k dimensions (shared subspace)
    3. Pick the largest model as reference
    4. Procrustes-align all others to the reference
    5. PCA the aligned coordinates down to 3D for visualization

    Returns list of dicts with 3D coordinates for each model.
    """
    # Collect and resample trajectories
    model_trajectories = {}
    for mid, mdata in models.items():
        if concept not in mdata["concepts"]:
            continue
        traj = mdata["concepts"][concept]
        if len(traj) < 3:
            continue
        resampled, grid = resample_trajectory(traj, n_depth_points)
        model_trajectories[mid] = {
            "resampled": resampled,
            "grid": grid,
            "hidden_dim": mdata["hidden_dim"],
            "n_layers": mdata["n_layers"],
        }

    if len(model_trajectories) < 2:
        return []

    # Project each model's trajectory into a shared dimensionality via PCA
    k = 50  # shared subspace dim
    model_projected = {}

    for mid, tdata in model_trajectories.items():
        raw_vecs = [t["vec"] for t in tdata["resampled"]]
        expected_dim = tdata["hidden_dim"]
        # Filter to consistent dimensions
        if any(len(v) != expected_dim for v in raw_vecs):
            log.warning("  Skipping %s: inconsistent dom_vector dims", _short_name(mid))
            continue
        vecs = np.array(raw_vecs)
        # Center
        vecs_c = vecs - vecs.mean(axis=0)
        # PCA down to k dims (or fewer if n_points < k)
        actual_k = min(k, vecs_c.shape[0] - 1, vecs_c.shape[1])
        pca = PCA(n_components=actual_k)
        projected = pca.fit_transform(vecs_c)
        model_projected[mid] = {
            "coords": projected,
            "seps": [t["separation"] for t in tdata["resampled"]],
            "cohs": [t["coherence"] for t in tdata["resampled"]],
            "grid": tdata["grid"],
            "n_layers": tdata["n_layers"],
        }

    # Pad all to same dimensionality
    max_k = max(mp["coords"].shape[1] for mp in model_projected.values())
    for mid, mp in model_projected.items():
        if mp["coords"].shape[1] < max_k:
            padding = np.zeros((mp["coords"].shape[0], max_k - mp["coords"].shape[1]))
            mp["coords"] = np.hstack([mp["coords"], padding])

    # Pick reference: the model with most layers (most detailed trajectory)
    ref_mid = max(model_projected, key=lambda m: model_projected[m]["n_layers"])
    ref_coords = model_projected[ref_mid]["coords"]
    log.info("  Reference model: %s", _short_name(ref_mid))

    # Procrustes align all to reference
    aligned = {}
    for mid, mp in model_projected.items():
        if mid == ref_mid:
            aligned[mid] = mp["coords"]
        else:
            # Align this model's coords to reference
            R, _ = orthogonal_procrustes(mp["coords"], ref_coords)
            aligned[mid] = mp["coords"] @ R

    # Stack all aligned trajectories and PCA to 3D
    all_coords = np.vstack(list(aligned.values()))
    pca_3d = PCA(n_components=3)
    all_3d = pca_3d.fit_transform(all_coords)
    explained = pca_3d.explained_variance_ratio_

    # Split back out
    results = []
    offset = 0
    for mid in aligned:
        n = aligned[mid].shape[0]
        coords_3d = all_3d[offset:offset + n]
        mp = model_projected[mid]
        results.append({
            "model_id": mid,
            "coords_3d": coords_3d,
            "grid": mp["grid"],
            "seps": mp["seps"],
            "cohs": mp["cohs"],
            "n_layers": mp["n_layers"],
        })
        offset += n

    return results, explained


def build_concept_figure(concept: str, aligned_models: list, explained: np.ndarray) -> go.Figure:
    """Build 3D Procrustes-aligned cazstellation for one concept."""
    fig = go.Figure()

    # Per-model trajectory
    for mdata in aligned_models:
        mid = mdata["model_id"]
        family = _model_family(mid)
        color = FAMILY_COLORS.get(family, "#888888")
        short = _short_name(mid)
        coords = mdata["coords_3d"]
        seps = mdata["seps"]

        # Trajectory line
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="lines",
            line=dict(color=color, width=3),
            opacity=0.5,
            name=short,
            legendgroup=short,
            showlegend=True,
            hoverinfo="skip",
        ))

        # Detect CAZes for this model
        lm = [LayerMetrics(i, s, c, 0.0) for i, (s, c) in enumerate(zip(seps, mdata["cohs"]))]
        # Recompute velocity
        from rosetta_tools.caz import compute_velocity
        vels = compute_velocity(seps)
        lm = [LayerMetrics(i, s, c, float(v)) for i, (s, c, v) in enumerate(zip(seps, mdata["cohs"], vels))]
        profile = find_caz_regions_scored(lm)

        # Plot CAZ peaks
        for region in profile.regions:
            pk = region.peak
            if pk >= len(coords):
                continue
            score = region.caz_score
            size = max(4, min(25, 6 + 14 * np.log1p(score * 10)))

            if score > 0.5:
                strength = "BLACK HOLE"
            elif score > 0.2:
                strength = "strong"
            elif score > 0.05:
                strength = "moderate"
            else:
                strength = "gentle"

            hover = (
                f"<b>{short}</b><br>"
                f"Depth: {mdata['grid'][pk]:.0f}%<br>"
                f"Score: {score:.3f} [{strength}]<br>"
                f"Sep: {seps[pk]:.3f}"
            )

            # Glow
            fig.add_trace(go.Scatter3d(
                x=[coords[pk, 0]], y=[coords[pk, 1]], z=[coords[pk, 2]],
                mode="markers",
                marker=dict(size=size * 1.6, color=color, opacity=0.12, line=dict(width=0)),
                showlegend=False, legendgroup=short, hoverinfo="skip",
            ))
            # Node
            fig.add_trace(go.Scatter3d(
                x=[coords[pk, 0]], y=[coords[pk, 1]], z=[coords[pk, 2]],
                mode="markers",
                marker=dict(size=size, color=color, opacity=0.85,
                            line=dict(width=1, color="rgba(255,255,255,0.3)")),
                showlegend=False, legendgroup=short,
                hovertemplate=hover + "<extra></extra>",
            ))

    n_models = len(aligned_models)
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Cross-Architecture Cazstellation — {concept}</b><br>"
                f"<sub>{n_models} models rotated into a shared space — nearby trajectories = similar representations.  "
                f"Double-click legend to isolate a model.</sub>"
            ),
            font=dict(size=16, color="rgba(200,210,230,0.95)"),
        ),
        scene=dict(
            xaxis=dict(title="", showbackground=True, showticklabels=False,
                       backgroundcolor="rgb(8,8,18)", gridcolor="rgba(80,100,140,0.12)",
                       color="rgba(150,160,180,0.5)", zerolinecolor="rgba(100,110,130,0.15)"),
            yaxis=dict(title="", showbackground=True, showticklabels=False,
                       backgroundcolor="rgb(8,8,18)", gridcolor="rgba(80,100,140,0.12)",
                       color="rgba(150,160,180,0.5)", zerolinecolor="rgba(100,110,130,0.15)"),
            zaxis=dict(title="", showbackground=True, showticklabels=False,
                       backgroundcolor="rgb(8,8,18)", gridcolor="rgba(80,100,140,0.12)",
                       color="rgba(150,160,180,0.5)", zerolinecolor="rgba(100,110,130,0.15)"),
            bgcolor="rgb(5,5,15)",
            camera=dict(eye=dict(x=1.6, y=0.8, z=0.6), up=dict(x=0, y=0, z=1)),
        ),
        paper_bgcolor="rgb(5,5,15)",
        plot_bgcolor="rgb(5,5,15)",
        font=dict(color="rgba(200,210,230,0.9)"),
        legend=dict(
            bgcolor="rgba(10,12,25,0.85)", bordercolor="rgba(80,100,140,0.3)", borderwidth=1,
            font=dict(size=10, color="rgba(200,210,230,0.9)"),
            title=dict(text="Models (dbl-click to isolate)",
                       font=dict(size=9, color="rgba(150,160,180,0.7)")),
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        width=1400, height=900,
    )

    return fig




def align_all_concepts_all_models(models: dict, n_depth_points: int = 25):
    """Procrustes-align ALL concepts × ALL models into one shared 3D space.

    Strategy:
    1. For each model, concatenate dom_vectors from all concepts at all depths
       into one big matrix (the model's full 'conceptual fingerprint')
    2. PCA each model's matrix to shared dimensionality
    3. Procrustes-align all models to a reference
    4. PCA the aligned coordinates to 3D
    5. Split back into per-concept trajectories for rendering

    Returns dict of concept → list of per-model 3D trajectories.
    """
    # For each model: collect ALL concept trajectories, resampled to common grid
    model_bundles = {}
    for mid, mdata in models.items():
        bundle = {}
        for concept in CONCEPTS:
            if concept not in mdata["concepts"]:
                continue
            traj = mdata["concepts"][concept]
            if len(traj) < 3:
                continue
            resampled, grid = resample_trajectory(traj, n_depth_points)
            raw_vecs = [t["vec"] for t in resampled]
            expected_dim = mdata["hidden_dim"]
            if any(len(v) != expected_dim for v in raw_vecs):
                continue
            bundle[concept] = {
                "vecs": np.array(raw_vecs),
                "seps": [t["separation"] for t in resampled],
                "cohs": [t["coherence"] for t in resampled],
                "grid": grid,
            }

        if len(bundle) >= 5:  # need most concepts for meaningful alignment
            model_bundles[mid] = {
                "bundle": bundle,
                "hidden_dim": mdata["hidden_dim"],
                "n_layers": mdata["n_layers"],
            }

    if len(model_bundles) < 2:
        return None

    log.info("  %d models with 5+ concepts", len(model_bundles))

    # Stack all concepts for each model into one big matrix, PCA to shared dim
    k = 50
    model_projected = {}
    concepts_order = CONCEPTS  # consistent ordering

    for mid, mb in model_bundles.items():
        # Stack: [n_concepts * n_depth_points, hidden_dim]
        all_vecs = []
        segment_info = []  # (concept, start_idx, end_idx)
        for concept in concepts_order:
            if concept not in mb["bundle"]:
                # Pad with zeros to keep alignment consistent
                all_vecs.append(np.zeros((n_depth_points, mb["hidden_dim"])))
                segment_info.append((concept, len(all_vecs[-1]), False))
            else:
                all_vecs.append(mb["bundle"][concept]["vecs"])
                segment_info.append((concept, len(all_vecs[-1]), True))

        stacked = np.vstack(all_vecs)  # [7*25, hidden_dim]
        stacked_c = stacked - stacked.mean(axis=0)

        actual_k = min(k, stacked_c.shape[0] - 1, stacked_c.shape[1])
        pca = PCA(n_components=actual_k)
        projected = pca.fit_transform(stacked_c)

        model_projected[mid] = {
            "coords": projected,
            "segment_info": segment_info,
            "n_layers": mb["n_layers"],
            "bundle": mb["bundle"],
        }

    # Pad to same dimensionality
    max_k = max(mp["coords"].shape[1] for mp in model_projected.values())
    for mid, mp in model_projected.items():
        if mp["coords"].shape[1] < max_k:
            padding = np.zeros((mp["coords"].shape[0], max_k - mp["coords"].shape[1]))
            mp["coords"] = np.hstack([mp["coords"], padding])

    # Reference: model with most layers
    ref_mid = max(model_projected, key=lambda m: model_projected[m]["n_layers"])
    ref_coords = model_projected[ref_mid]["coords"]
    log.info("  Reference: %s", _short_name(ref_mid))

    # Procrustes align all to reference
    aligned = {}
    for mid, mp in model_projected.items():
        if mid == ref_mid:
            aligned[mid] = mp["coords"]
        else:
            R, _ = orthogonal_procrustes(mp["coords"], ref_coords)
            aligned[mid] = mp["coords"] @ R

    # Stack ALL aligned coords and PCA to 3D
    all_coords = np.vstack(list(aligned.values()))
    pca_3d = PCA(n_components=3)
    all_3d = pca_3d.fit_transform(all_coords)
    explained = pca_3d.explained_variance_ratio_

    # Split back into per-model, per-concept
    result = {}
    offset = 0
    for mid in aligned:
        n_total = aligned[mid].shape[0]
        model_3d = all_3d[offset:offset + n_total]
        offset += n_total

        # Split into concept segments
        seg_offset = 0
        mp = model_projected[mid]
        for concept_name, seg_len, is_real in mp["segment_info"]:
            if is_real and concept_name in mp["bundle"]:
                coords_3d = model_3d[seg_offset:seg_offset + n_depth_points]
                bdata = mp["bundle"][concept_name]
                if concept_name not in result:
                    result[concept_name] = []
                result[concept_name].append({
                    "model_id": mid,
                    "coords_3d": coords_3d,
                    "grid": bdata["grid"],
                    "seps": bdata["seps"],
                    "cohs": bdata["cohs"],
                    "n_layers": mp["n_layers"],
                })
            seg_offset += n_depth_points

    return result, explained


def build_combined_figure(concept_data: dict, explained: np.ndarray) -> go.Figure:
    """Build the monster viz: all concepts × all models in one Procrustes space."""
    fig = go.Figure()

    total_models = set()
    total_cazs = 0

    for concept in CONCEPTS:
        if concept not in concept_data:
            continue
        color = CONCEPT_COLORS[concept]
        ctype = CONCEPT_TYPE[concept]

        for mdata in concept_data[concept]:
            mid = mdata["model_id"]
            total_models.add(mid)
            family = _model_family(mid)
            short = _short_name(mid)
            coords = mdata["coords_3d"]
            seps = mdata["seps"]

            # Trajectory — colored by CONCEPT, grouped by concept for legend isolation
            fig.add_trace(go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode="lines",
                line=dict(color=color, width=2.5),
                opacity=0.25,
                name=f"{concept} ({ctype})",
                legendgroup=concept,
                showlegend=False,
                hoverinfo="skip",
            ))

            # Detect CAZes
            from rosetta_tools.caz import compute_velocity
            vels = compute_velocity(seps)
            lm = [LayerMetrics(i, s, c, float(v))
                  for i, (s, c, v) in enumerate(zip(seps, mdata["cohs"], vels))]
            profile = find_caz_regions_scored(lm)

            for region in profile.regions:
                pk = region.peak
                if pk >= len(coords):
                    continue
                score = region.caz_score
                if score < 0.02:
                    continue  # skip very gentle for readability

                total_cazs += 1

                # Uniform size — use brightness (opacity) for strength
                node_size = 6
                # Map score to opacity: gentle=dim, black hole=bright
                opacity = min(0.95, max(0.15, 0.15 + 0.8 * np.log1p(score * 5) / np.log1p(5)))

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
                    f"Model: {short}<br>"
                    f"Depth: {mdata['grid'][pk]:.0f}%<br>"
                    f"Score: {score:.3f} [{strength}]<br>"
                    f"Sep: {seps[pk]:.3f}"
                )

                # Single node — brightness = score
                fig.add_trace(go.Scatter3d(
                    x=[coords[pk, 0]], y=[coords[pk, 1]], z=[coords[pk, 2]],
                    mode="markers",
                    marker=dict(size=node_size, color=color, opacity=opacity,
                                line=dict(width=0)),
                    showlegend=False, legendgroup=concept,
                    hovertemplate=hover + "<extra></extra>",
                ))

    # Legend entries — one per concept
    for concept in CONCEPTS:
        if concept not in concept_data:
            continue
        n_models = len(concept_data[concept])
        color = CONCEPT_COLORS[concept]
        ctype = CONCEPT_TYPE[concept]
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="markers",
            marker=dict(size=10, color=color),
            name=f"{concept} ({ctype}, {n_models} models)",
            legendgroup=concept,
            showlegend=True,
        ))

    fig.update_layout(
        title=dict(
            text=(
                f"<b>The Concept Web — {len(total_models)} Models, 7 Concepts</b><br>"
                f"<sub>Each line is one concept's path through one model's layers.  "
                f"Models are rotated into a shared space — nearby points = similar representations.  "
                f"Brighter = stronger assembly.  "
                f"Double-click legend to isolate a concept.</sub>"
            ),
            font=dict(size=16, color="rgba(200,210,230,0.95)"),
        ),
        scene=dict(
            xaxis=dict(title="", showbackground=True, showticklabels=False,
                       backgroundcolor="rgb(8,8,18)", gridcolor="rgba(80,100,140,0.10)",
                       color="rgba(150,160,180,0.5)", zerolinecolor="rgba(100,110,130,0.12)"),
            yaxis=dict(title="", showbackground=True, showticklabels=False,
                       backgroundcolor="rgb(8,8,18)", gridcolor="rgba(80,100,140,0.10)",
                       color="rgba(150,160,180,0.5)", zerolinecolor="rgba(100,110,130,0.12)"),
            zaxis=dict(title="", showbackground=True, showticklabels=False,
                       backgroundcolor="rgb(8,8,18)", gridcolor="rgba(80,100,140,0.10)",
                       color="rgba(150,160,180,0.5)", zerolinecolor="rgba(100,110,130,0.12)"),
            bgcolor="rgb(5,5,15)",
            camera=dict(eye=dict(x=1.5, y=0.7, z=0.5), up=dict(x=0, y=0, z=1)),
        ),
        paper_bgcolor="rgb(5,5,15)",
        plot_bgcolor="rgb(5,5,15)",
        font=dict(color="rgba(200,210,230,0.9)"),
        legend=dict(
            bgcolor="rgba(10,12,25,0.85)", bordercolor="rgba(80,100,140,0.3)", borderwidth=1,
            font=dict(size=12, color="rgba(200,210,230,0.9)"),
            title=dict(text="Concepts (dbl-click to isolate)",
                       font=dict(size=10, color="rgba(150,160,180,0.7)")),
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        width=1600, height=1000,
    )

    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = load_all_models()
    log.info("Loaded %d models", len(models))

    # Per-concept views
    for concept in CONCEPTS:
        log.info("Aligning %s...", concept)
        n_with_concept = sum(1 for m in models.values() if concept in m["concepts"])
        if n_with_concept < 3:
            log.warning("  Only %d models have %s, skipping", n_with_concept, concept)
            continue

        result = align_trajectories_for_concept(models, concept)
        if not result:
            continue
        aligned_models, explained = result

        fig = build_concept_figure(concept, aligned_models, explained)
        out_path = OUT_DIR / f"procrustes_{concept}.html"
        fig.write_html(str(out_path), include_plotlyjs=True)
        log.info("  → %s (%d models)", out_path, len(aligned_models))

    # THE MONSTER: all concepts × all models
    log.info("")
    log.info("Building the Concept Web (all concepts × all models)...")
    result = align_all_concepts_all_models(models)
    if result:
        concept_data, explained = result
        fig = build_combined_figure(concept_data, explained)
        out_path = OUT_DIR / "procrustes_ALL_CONCEPTS.html"
        fig.write_html(str(out_path), include_plotlyjs=True)
        log.info("→ %s", out_path)
    else:
        log.error("Failed to build combined visualization")


if __name__ == "__main__":
    main()

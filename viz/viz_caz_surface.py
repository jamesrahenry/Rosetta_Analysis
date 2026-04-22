"""
viz_caz_surface.py — 3D ribbon plot of CAZ metrics across transformer depth.

Each concept is a discrete ribbon — no interpolation between concepts.

  X-axis:  layer index (0 → N)
  Y-axis:  concept (categorical, spaced apart)
  Z-axis:  separation S(l)   — ribbon height
  Color:   coherence C(l)    — ribbon face color (bright = crystallized)
  Width:   velocity V(l)     — ribbon swells where concept is being built,
                               pinches where it's being dismantled

CAZ peaks are marked as diamond markers where separation is locally maximal.

Generates:
  - Interactive HTML (plotly) — rotatable, zoomable, hover for all 3 metrics
  - Static PNG (matplotlib)  — paper-ready, --static flag

Usage:
    python src/viz_caz_surface.py \
        --results-dir results/gpt2_openai_community_gpt2_xl_20260401_184059

    python src/viz_caz_surface.py \
        --results-dir results/pythia_EleutherAI_pythia_6.9b_20260401_162945 \
        --static
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from rosetta_tools.viz import CONCEPT_META as _BASE_META
from viz_style import concept_color

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concept metadata — imported from canonical source + extended for CIA concepts
# ---------------------------------------------------------------------------

CONCEPT_META = {
    **_BASE_META,
    # Security concepts (CIA) — extend base palette using canonical concept_color()
    "authorization":   {"type": "security", "color": concept_color("authorization")},
    "threat_severity": {"type": "security", "color": concept_color("threat_severity")},
    "urgency":         {"type": "security", "color": concept_color("urgency")},
    "exfiltration":    {"type": "security", "color": concept_color("exfiltration")},
    "obfuscation":     {"type": "security", "color": concept_color("obfuscation")},
}

CONCEPT_ORDER = [
    "temporal_order", "causation",
    "negation", "plurality",
    "sentiment", "moral_valence",
    "certainty", "credibility",
    "authorization", "threat_severity", "urgency", "exfiltration", "obfuscation",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_metrics(results_dir: Path) -> dict[str, dict]:
    """Load per-layer separation, coherence, velocity from caz_*.json files.

    Returns {concept: {"separation": [...], "coherence": [...], "velocity": [...]}}
    """
    data = {}
    for path in sorted(results_dir.glob("caz_*.json")):
        raw = json.loads(path.read_text())
        concept = raw["concept"]
        metrics = raw["layer_data"]["metrics"]
        data[concept] = {
            "separation": [m["separation_fisher"] for m in metrics],
            "coherence":  [m["coherence"] for m in metrics],
            "velocity":   [m["velocity"] for m in metrics],
        }
        seps = data[concept]["separation"]
        log.info("  %s: %d layers, peak_sep=%.3f at L%d",
                 concept, len(seps), max(seps), np.argmax(seps))
    return data


# ---------------------------------------------------------------------------
# Plotly interactive ribbons
# ---------------------------------------------------------------------------

BASE_HALF_WIDTH = 0.25     # ribbon width at velocity = 0
MIN_HALF_WIDTH = 0.06      # minimum ribbon width (strong dismantling)
MAX_HALF_WIDTH = 0.48      # maximum ribbon width (strong building)
Y_SPACING = 1.2            # gap between concept centres


def _velocity_to_halfwidth(vel: np.ndarray, vel_abs_max: float) -> np.ndarray:
    """Map velocity to ribbon half-width.

    Positive velocity (building)     → wider ribbon
    Zero velocity (stable)           → baseline width
    Negative velocity (dismantling)  → narrower ribbon
    """
    if vel_abs_max < 1e-8:
        return np.full_like(vel, BASE_HALF_WIDTH)
    scale = (MAX_HALF_WIDTH - MIN_HALF_WIDTH) / 2
    hw = BASE_HALF_WIDTH + (vel / vel_abs_max) * scale
    return np.clip(hw, MIN_HALF_WIDTH, MAX_HALF_WIDTH)


def build_plotly_ribbons(
    data: dict[str, dict],
    model_name: str,
    output_path: Path,
) -> None:
    """Generate interactive 3D ribbon plot with plotly."""
    import plotly.graph_objects as go

    ordered = [c for c in CONCEPT_ORDER if c in data]
    n_concepts = len(ordered)
    n_layers = len(next(iter(data.values()))["separation"])

    # Global coherence range for consistent colour mapping
    all_coh = []
    for concept in ordered:
        all_coh.extend(data[concept]["coherence"])
    coh_min, coh_max = min(all_coh), max(all_coh)

    # Global velocity range for symmetric width scaling
    all_vel = []
    for concept in ordered:
        all_vel.extend(data[concept]["velocity"])
    vel_abs_max = max(abs(min(all_vel)), abs(max(all_vel)), 1e-8)

    fig = go.Figure()

    layers = np.arange(n_layers)

    for idx, concept in enumerate(ordered):
        sep = np.array(data[concept]["separation"])
        coh = np.array(data[concept]["coherence"])
        vel = np.array(data[concept]["velocity"])

        y_centre = idx * Y_SPACING

        # Per-layer ribbon half-width from velocity
        hw = _velocity_to_halfwidth(vel, vel_abs_max)
        y_lo = y_centre - hw   # [n_layers]
        y_hi = y_centre + hw   # [n_layers]

        # Ribbon as a narrow surface: 2 rows × n_layers columns
        X_ribbon = np.vstack([layers, layers])
        Y_ribbon = np.vstack([y_lo, y_hi])
        Z_ribbon = np.vstack([sep, sep])
        C_ribbon = np.vstack([coh, coh])

        # Hover text
        hover = [[
            f"<b>{concept}</b><br>"
            f"L{j}<br>"
            f"Separation: {sep[j]:.3f}<br>"
            f"Coherence: {coh[j]:.3f}<br>"
            f"Velocity: {vel[j]:+.4f} "
            f"({'building' if vel[j] > 0.001 else 'dismantling' if vel[j] < -0.001 else 'stable'})"
            for j in range(n_layers)
        ]] * 2

        fig.add_trace(go.Surface(
            x=X_ribbon,
            y=Y_ribbon,
            z=Z_ribbon,
            surfacecolor=C_ribbon,
            colorscale=[
                [0.0, "#161b22"],    # low coherence — dark
                [0.3, "#1565C0"],
                [0.6, "#3fb950"],
                [1.0, "#f0e68c"],    # high coherence — bright gold
            ],
            cmin=coh_min,
            cmax=coh_max,
            colorbar=dict(
                title=dict(text="Coherence C(l)", font=dict(color="#e6edf3", size=11)),
                tickfont=dict(color="#e6edf3"),
                len=0.4,
                y=0.8,
                x=1.01,
            ) if idx == 0 else None,
            showscale=(idx == 0),
            opacity=0.88,
            hovertext=hover,
            hoverinfo="text",
            name=concept,
            showlegend=False,
        ))

        # --- CAZ peak markers ---
        for j in range(1, n_layers - 1):
            if sep[j] > sep[j-1] and sep[j] > sep[j+1] and sep[j] > 0.05:
                fig.add_trace(go.Scatter3d(
                    x=[j], y=[y_centre], z=[sep[j] + 0.02],
                    mode="markers",
                    marker=dict(size=5, color="#d29922", symbol="diamond"),
                    text=[f"CAZ peak: {concept} L{j}<br>"
                          f"S={sep[j]:.3f}  C={coh[j]:.3f}  V={vel[j]:+.4f}"],
                    hoverinfo="text",
                    showlegend=False,
                ))

    # --- Legend entries ---
    for label, color in [("wide ribbon = building (V > 0)", "#3fb950"),
                         ("narrow ribbon = dismantling (V < 0)", "#7d8590")]:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="lines",
            line=dict(color=color, width=6 if "wide" in label else 2),
            name=label,
            showlegend=True,
        ))

    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(size=6, color="#d29922", symbol="diamond"),
        name="CAZ peak (local max)",
        showlegend=True,
    ))

    # --- Layout ---
    y_tickvals = [i * Y_SPACING for i in range(n_concepts)]

    fig.update_layout(
        title=dict(
            text=f"Concept Assembly Landscape — {model_name}",
            font=dict(size=16, color="#e6edf3"),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(
                title="Layer",
                backgroundcolor="#0d1117",
                gridcolor="#30363d",
                color="#7d8590",
                tickvals=list(range(0, n_layers, max(1, n_layers // 8))),
            ),
            yaxis=dict(
                title="",
                backgroundcolor="#0d1117",
                gridcolor="#21262d",
                color="#7d8590",
                tickvals=y_tickvals,
                ticktext=ordered,
            ),
            zaxis=dict(
                title="Separation S(l)",
                backgroundcolor="#0d1117",
                gridcolor="#30363d",
                color="#7d8590",
            ),
            bgcolor="#0d1117",
            camera=dict(
                eye=dict(x=1.8, y=-1.4, z=0.9),
            ),
        ),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3"),
        legend=dict(
            font=dict(color="#e6edf3", size=10),
            bgcolor="rgba(22, 27, 34, 0.9)",
            bordercolor="#30363d",
            borderwidth=1,
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
        ),
        margin=dict(l=0, r=60, t=50, b=0),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    log.info("Interactive ribbons → %s", output_path)


# ---------------------------------------------------------------------------
# Matplotlib static ribbons
# ---------------------------------------------------------------------------

def build_matplotlib_ribbons(
    data: dict[str, dict],
    model_name: str,
    output_path: Path,
) -> None:
    """Generate a static 3D ribbon plot with matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ordered = [c for c in CONCEPT_ORDER if c in data]
    n_concepts = len(ordered)
    n_layers = len(next(iter(data.values()))["separation"])

    all_coh = []
    for concept in ordered:
        all_coh.extend(data[concept]["coherence"])
    coh_norm = Normalize(vmin=min(all_coh), vmax=max(all_coh))

    coh_cmap = LinearSegmentedColormap.from_list("coh", [
        "#161b22", "#1565C0", "#3fb950", "#f0e68c",
    ])

    all_vel = []
    for concept in ordered:
        all_vel.extend(data[concept]["velocity"])
    vel_abs_max = max(abs(min(all_vel)), abs(max(all_vel)), 1e-8)

    fig = plt.figure(figsize=(15, 9))
    fig.patch.set_facecolor("#0d1117")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d1117")

    for idx, concept in enumerate(ordered):
        sep = np.array(data[concept]["separation"])
        coh = np.array(data[concept]["coherence"])
        vel = np.array(data[concept]["velocity"])
        y_c = idx * Y_SPACING

        # Per-layer half-width from velocity
        hw = _velocity_to_halfwidth(vel, vel_abs_max)

        # Draw ribbon as quad strips coloured by coherence, width by velocity
        for j in range(n_layers - 1):
            hw_j = (hw[j] + hw[j + 1]) / 2
            verts = [
                [j,     y_c - hw[j],     sep[j]],
                [j,     y_c + hw[j],     sep[j]],
                [j + 1, y_c + hw[j + 1], sep[j + 1]],
                [j + 1, y_c - hw[j + 1], sep[j + 1]],
            ]
            avg_coh = (coh[j] + coh[j + 1]) / 2
            color = coh_cmap(coh_norm(avg_coh))
            poly = Poly3DCollection([verts], alpha=0.85)
            poly.set_facecolor(color)
            poly.set_edgecolor("none")
            ax.add_collection3d(poly)

        # CAZ peak markers
        for j in range(1, n_layers - 1):
            if sep[j] > sep[j-1] and sep[j] > sep[j+1] and sep[j] > 0.05:
                ax.scatter([j], [y_c], [sep[j] + 0.015],
                           color="#d29922", s=30, marker="D", zorder=10)

    ax.set_xlabel("Layer", color="#7d8590", labelpad=10)
    ax.set_ylabel("", color="#7d8590", labelpad=10)
    ax.set_zlabel("Separation S(l)", color="#7d8590", labelpad=10)
    ax.set_yticks([i * Y_SPACING for i in range(n_concepts)])
    ax.set_yticklabels(ordered, fontsize=7, rotation=-15, ha="left")
    ax.tick_params(colors="#7d8590", labelsize=7)

    # Z range
    all_sep = []
    for concept in ordered:
        all_sep.extend(data[concept]["separation"])
    ax.set_zlim(0, max(all_sep) * 1.05)

    ax.set_title(f"Concept Assembly Landscape — {model_name}",
                 color="#e6edf3", fontsize=12, pad=15)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#30363d")
    ax.yaxis.pane.set_edgecolor("#30363d")
    ax.zaxis.pane.set_edgecolor("#30363d")

    ax.view_init(elev=25, azim=-55)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    log.info("Static ribbons → %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="3D ribbon plot of CAZ metrics across transformer depth")
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing caz_*.json files for one model")
    parser.add_argument("--output", default=None,
                        help="Output path (default: visualizations/<model>_caz_surface.html)")
    parser.add_argument("--static", action="store_true",
                        help="Also generate static PNG (matplotlib)")
    parser.add_argument("--model-name", default=None,
                        help="Display name for title (auto-detected from JSON if omitted)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        parser.error(f"Results directory not found: {results_dir}")

    data = load_all_metrics(results_dir)
    if not data:
        parser.error(f"No caz_*.json files found in {results_dir}")

    model_name = args.model_name
    if model_name is None:
        sample = json.loads(next(results_dir.glob("caz_*.json")).read_text())
        model_name = sample["model_id"].split("/")[-1]

    if args.output:
        output = Path(args.output)
    else:
        output = Path("visualizations") / f"{model_name}_caz_surface.html"

    build_plotly_ribbons(data, model_name, output)

    if args.static:
        png_path = output.with_suffix(".png")
        build_matplotlib_ribbons(data, model_name, png_path)

    log.info("Done. %d concepts, %d layers.",
             len(data), len(next(iter(data.values()))["separation"]))


if __name__ == "__main__":
    main()

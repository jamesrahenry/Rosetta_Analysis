"""
viz_caz_crossarch.py — Cross-architecture ribbon plot.

Groups ribbons by concept. Within each concept group, one ribbon per model
family (largest base model). Ribbons sit side-by-side for direct comparison.

  X-axis:  depth percentage (0–100%) — normalised across different layer counts
  Y-axis:  concept group, with sub-ribbons per model family
  Z-axis:  separation S(l)
  Color:   coherence C(l)
  Width:   velocity V(l) — swells when building, pinches when dismantling

Usage:
    python src/viz_caz_crossarch.py
    python src/viz_caz_crossarch.py --static
    python src/viz_caz_crossarch.py --output visualizations/crossarch.html
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

log = logging.getLogger(__name__)

RESULTS_BASE = Path(__file__).parent.parent / "results"

# ---------------------------------------------------------------------------
# Model families — largest base model per family
# ---------------------------------------------------------------------------

FAMILIES = {
    "Pythia-6.9B":   "pythia_EleutherAI_pythia_6.9b",
    "GPT-2-XL":      "gpt2_openai_community_gpt2_xl",
    "OPT-6.7B":      "opt_facebook_opt_6.7b",
    "Qwen2.5-7B":    "qwen2_Qwen_Qwen2.5_7B",
    "Gemma2-9B":     "gemma2_google_gemma_2_9b",
    "Mistral-7B":    "mistral_mistralai_Mistral_7B_v0.3",
    "Llama3.2-3B":   "llama3_meta_llama_Llama_3.2_3B",
    "Phi-2":         "phi_microsoft_phi_2",
}

FAMILY_COLORS = {
    "Pythia-6.9B":   "#58A6FF",
    "GPT-2-XL":      "#F78166",
    "OPT-6.7B":      "#3FB950",
    "Qwen2.5-7B":    "#D2A8FF",
    "Gemma2-9B":     "#FF7B72",
    "Mistral-7B":    "#79C0FF",
    "Llama3.2-3B":   "#FFA657",
    "Phi-2":         "#7EE787",
}

CONCEPT_ORDER = [
    "negation", "causation", "temporal_order",
    "sentiment", "moral_valence",
    "certainty", "credibility",
]

N_INTERP = 100  # resample all models to 100 depth points


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_results_dir(prefix: str) -> Path | None:
    """Find results directory matching a family prefix."""
    matches = sorted(RESULTS_BASE.glob(f"{prefix}_*"))
    return matches[0] if matches else None


def load_model_metrics(results_dir: Path) -> dict[str, dict]:
    """Load all three metrics per concept, return normalised to depth %."""
    data = {}
    for path in sorted(results_dir.glob("caz_*.json")):
        raw = json.loads(path.read_text())
        concept = raw["concept"]
        metrics = raw["layer_data"]["metrics"]
        n_layers = len(metrics)

        sep = np.array([m["separation_fisher"] for m in metrics])
        coh = np.array([m["coherence"] for m in metrics])
        vel = np.array([m["velocity"] for m in metrics])

        # Normalise to depth percentage and resample to N_INTERP points
        depth_pct = np.linspace(0, 100, n_layers)
        depth_uniform = np.linspace(0, 100, N_INTERP)

        data[concept] = {
            "separation": interp1d(depth_pct, sep, kind="cubic")(depth_uniform),
            "coherence":  interp1d(depth_pct, coh, kind="cubic")(depth_uniform),
            "velocity":   interp1d(depth_pct, vel, kind="cubic")(depth_uniform),
            "n_layers":   n_layers,
        }
    return data


# ---------------------------------------------------------------------------
# Velocity → ribbon width
# ---------------------------------------------------------------------------

BASE_HW = 0.12
MIN_HW = 0.03
MAX_HW = 0.22


def _vel_to_hw(vel: np.ndarray, vel_abs_max: float) -> np.ndarray:
    if vel_abs_max < 1e-8:
        return np.full_like(vel, BASE_HW)
    scale = (MAX_HW - MIN_HW) / 2
    hw = BASE_HW + (vel / vel_abs_max) * scale
    return np.clip(hw, MIN_HW, MAX_HW)


# ---------------------------------------------------------------------------
# Plotly
# ---------------------------------------------------------------------------

def build_crossarch_plot(
    all_data: dict[str, dict[str, dict]],
    output_path: Path,
) -> None:
    import plotly.graph_objects as go

    families_present = list(all_data.keys())
    n_families = len(families_present)
    concepts_present = [c for c in CONCEPT_ORDER
                        if any(c in all_data[f] for f in families_present)]
    n_concepts = len(concepts_present)

    # Global ranges
    all_coh, all_vel = [], []
    for fam in families_present:
        for concept in concepts_present:
            if concept in all_data[fam]:
                all_coh.extend(all_data[fam][concept]["coherence"])
                all_vel.extend(all_data[fam][concept]["velocity"])
    coh_min, coh_max = min(all_coh), max(all_coh)
    vel_abs_max = max(abs(min(all_vel)), abs(max(all_vel)), 1e-8)

    CONCEPT_GAP = n_families * 0.35 + 1.0   # space between concept groups
    RIBBON_SPACING = 0.30                     # space between ribbons in a group

    depth = np.linspace(0, 100, N_INTERP)
    fig = go.Figure()

    y_tick_vals = []
    y_tick_text = []

    for ci, concept in enumerate(concepts_present):
        group_base_y = ci * CONCEPT_GAP

        for fi, family in enumerate(families_present):
            if concept not in all_data[family]:
                continue

            d = all_data[family][concept]
            sep = d["separation"]
            coh = d["coherence"]
            vel = d["velocity"]

            y_centre = group_base_y + fi * RIBBON_SPACING
            hw = _vel_to_hw(vel, vel_abs_max)
            y_lo = y_centre - hw
            y_hi = y_centre + hw

            X_ribbon = np.vstack([depth, depth])
            Y_ribbon = np.vstack([y_lo, y_hi])
            Z_ribbon = np.vstack([sep, sep])
            C_ribbon = np.vstack([coh, coh])

            hover = [[
                f"<b>{concept}</b> — {family}<br>"
                f"Depth: {depth[j]:.0f}%<br>"
                f"Separation: {sep[j]:.3f}<br>"
                f"Coherence: {coh[j]:.3f}<br>"
                f"Velocity: {vel[j]:+.4f}"
                for j in range(N_INTERP)
            ]] * 2

            fig.add_trace(go.Surface(
                x=X_ribbon,
                y=Y_ribbon,
                z=Z_ribbon,
                surfacecolor=C_ribbon,
                colorscale=[
                    [0.0, "#161b22"],
                    [0.3, "#1565C0"],
                    [0.6, "#3fb950"],
                    [1.0, "#f0e68c"],
                ],
                cmin=coh_min,
                cmax=coh_max,
                colorbar=dict(
                    title=dict(text="Coherence", font=dict(color="#e6edf3", size=11)),
                    tickfont=dict(color="#e6edf3"),
                    len=0.35,
                    y=0.82,
                    x=1.01,
                ) if (ci == 0 and fi == 0) else None,
                showscale=(ci == 0 and fi == 0),
                opacity=0.85,
                hovertext=hover,
                hoverinfo="text",
                showlegend=False,
            ))

            # CAZ peaks
            for j in range(1, N_INTERP - 1):
                if sep[j] > sep[j-1] and sep[j] > sep[j+1] and sep[j] > 0.05:
                    fig.add_trace(go.Scatter3d(
                        x=[depth[j]], y=[y_centre], z=[sep[j] + 0.02],
                        mode="markers",
                        marker=dict(
                            size=4,
                            color=FAMILY_COLORS.get(family, "#7d8590"),
                            symbol="diamond",
                        ),
                        text=[f"CAZ: {concept} — {family}<br>"
                              f"Depth: {depth[j]:.0f}%  S={sep[j]:.3f}"],
                        hoverinfo="text",
                        showlegend=False,
                    ))

        # Concept group label position
        group_centre_y = group_base_y + (n_families - 1) * RIBBON_SPACING / 2
        y_tick_vals.append(group_centre_y)
        y_tick_text.append(concept)

    # --- Legend: one entry per family ---
    for family in families_present:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="lines",
            line=dict(color=FAMILY_COLORS.get(family, "#7d8590"), width=6),
            name=family,
            showlegend=True,
        ))

    # Width legend entries
    for label, width in [("wide = building (V>0)", 6), ("narrow = dismantling (V<0)", 1)]:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="lines",
            line=dict(color="#7d8590", width=width),
            name=label,
            showlegend=True,
        ))

    fig.update_layout(
        title=dict(
            text="Cross-Architecture Concept Assembly — Largest Base Models",
            font=dict(size=15, color="#e6edf3"),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(
                title="Depth %",
                backgroundcolor="#0d1117",
                gridcolor="#30363d",
                color="#7d8590",
                tickvals=[0, 25, 50, 75, 100],
            ),
            yaxis=dict(
                title="",
                backgroundcolor="#0d1117",
                gridcolor="#21262d",
                color="#7d8590",
                tickvals=y_tick_vals,
                ticktext=y_tick_text,
            ),
            zaxis=dict(
                title="Separation S(l)",
                backgroundcolor="#0d1117",
                gridcolor="#30363d",
                color="#7d8590",
            ),
            bgcolor="#0d1117",
            camera=dict(eye=dict(x=1.9, y=-1.5, z=0.8)),
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
    log.info("Cross-architecture ribbons → %s", output_path)


# ---------------------------------------------------------------------------
# Matplotlib static
# ---------------------------------------------------------------------------

def build_crossarch_static(
    all_data: dict[str, dict[str, dict]],
    output_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgba
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    families_present = list(all_data.keys())
    n_families = len(families_present)
    concepts_present = [c for c in CONCEPT_ORDER
                        if any(c in all_data[f] for f in families_present)]
    n_concepts = len(concepts_present)

    all_coh, all_vel = [], []
    for fam in families_present:
        for concept in concepts_present:
            if concept in all_data[fam]:
                all_coh.extend(all_data[fam][concept]["coherence"])
                all_vel.extend(all_data[fam][concept]["velocity"])
    coh_norm = Normalize(vmin=min(all_coh), vmax=max(all_coh))
    vel_abs_max = max(abs(min(all_vel)), abs(max(all_vel)), 1e-8)

    coh_cmap = LinearSegmentedColormap.from_list("coh", [
        "#161b22", "#1565C0", "#3fb950", "#f0e68c",
    ])

    CONCEPT_GAP = n_families * 0.35 + 1.0
    RIBBON_SPACING = 0.30

    depth = np.linspace(0, 100, N_INTERP)

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#0d1117")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d1117")

    y_tick_vals = []
    y_tick_text = []

    for ci, concept in enumerate(concepts_present):
        group_base_y = ci * CONCEPT_GAP

        for fi, family in enumerate(families_present):
            if concept not in all_data[family]:
                continue

            d = all_data[family][concept]
            sep = d["separation"]
            coh = d["coherence"]
            vel = d["velocity"]
            y_c = group_base_y + fi * RIBBON_SPACING
            hw = _vel_to_hw(vel, vel_abs_max)

            # Quad strips
            step = 2  # skip every other point for performance
            for j in range(0, N_INTERP - step, step):
                j2 = min(j + step, N_INTERP - 1)
                verts = [
                    [depth[j],  y_c - hw[j],  sep[j]],
                    [depth[j],  y_c + hw[j],  sep[j]],
                    [depth[j2], y_c + hw[j2], sep[j2]],
                    [depth[j2], y_c - hw[j2], sep[j2]],
                ]
                avg_coh = (coh[j] + coh[j2]) / 2
                color = coh_cmap(coh_norm(avg_coh))
                poly = Poly3DCollection([verts], alpha=0.8)
                poly.set_facecolor(color)
                poly.set_edgecolor("none")
                ax.add_collection3d(poly)

            # CAZ peaks with family color
            fc = FAMILY_COLORS.get(family, "#7d8590")
            for j in range(1, N_INTERP - 1):
                if sep[j] > sep[j-1] and sep[j] > sep[j+1] and sep[j] > 0.05:
                    ax.scatter([depth[j]], [y_c], [sep[j] + 0.015],
                               color=fc, s=20, marker="D", zorder=10)

        group_centre_y = group_base_y + (n_families - 1) * RIBBON_SPACING / 2
        y_tick_vals.append(group_centre_y)
        y_tick_text.append(concept)

    ax.set_xlabel("Depth %", color="#7d8590", labelpad=10)
    ax.set_ylabel("", color="#7d8590", labelpad=10)
    ax.set_zlabel("Separation S(l)", color="#7d8590", labelpad=10)
    ax.set_yticks(y_tick_vals)
    ax.set_yticklabels(y_tick_text, fontsize=7, rotation=-15, ha="left")
    ax.tick_params(colors="#7d8590", labelsize=7)

    all_sep = []
    for fam in families_present:
        for concept in concepts_present:
            if concept in all_data[fam]:
                all_sep.extend(all_data[fam][concept]["separation"])
    ax.set_zlim(0, max(all_sep) * 1.05)

    ax.set_title("Cross-Architecture Concept Assembly — Largest Base Models",
                 color="#e6edf3", fontsize=12, pad=15)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#30363d")
    ax.yaxis.pane.set_edgecolor("#30363d")
    ax.zaxis.pane.set_edgecolor("#30363d")

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=FAMILY_COLORS.get(f, "#7d8590"),
                       linewidth=3, label=f) for f in families_present]
    ax.legend(handles=handles, loc="upper left", fontsize=7,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")

    ax.view_init(elev=22, azim=-50)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    log.info("Static cross-architecture ribbons → %s", output_path)


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
        description="Cross-architecture ribbon plot — largest model per family")
    parser.add_argument("--output", default=None)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--results-base", default=None,
                        help="Override results base directory")
    args = parser.parse_args()

    global RESULTS_BASE
    if args.results_base:
        RESULTS_BASE = Path(args.results_base)

    # Load data for each family
    all_data = {}
    for family_name, prefix in FAMILIES.items():
        rdir = find_results_dir(prefix)
        if rdir is None:
            log.warning("  %s: no results found, skipping", family_name)
            continue
        log.info("Loading %s from %s", family_name, rdir.name)
        metrics = load_model_metrics(rdir)
        if metrics:
            all_data[family_name] = metrics

    if not all_data:
        parser.error("No model data found")

    log.info("Loaded %d families", len(all_data))

    output = Path(args.output) if args.output else Path("visualizations/crossarch_caz_surface.html")

    build_crossarch_plot(all_data, output)

    if args.static:
        build_crossarch_static(all_data, output.with_suffix(".png"))

    log.info("Done.")


if __name__ == "__main__":
    main()

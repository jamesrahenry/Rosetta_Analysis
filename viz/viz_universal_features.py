#!/usr/bin/env python3
"""Visualize Universal Feature Atlas — cross-family UFs as interactive Plotly HTML."""

import json
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from viz_style import concept_color, CONCEPT_COLORS, FAMILY_COLORS, FAMILY_MAP

_UNKNOWN_COLOR = "#444444"   # for features with no concept label (None key)


def _family(model_id):
    m = model_id.lower()
    if "pythia" in m: return "Pythia"
    if "gpt2" in m or "gpt-2" in m: return "GPT-2"
    if "opt" in m: return "OPT"
    if "qwen" in m: return "Qwen"
    if "gemma" in m: return "Gemma"
    if "llama" in m: return "Llama"
    if "mistral" in m: return "Mistral"
    if "phi" in m: return "Phi"
    return "Other"


def _encoding_strategy(model_id):
    """Return 'redundant', 'sparse', or 'unknown' from the model registry."""
    try:
        from rosetta_tools.models import encoding_strategy_of
        return encoding_strategy_of(model_id)
    except ImportError:
        return "unknown"


def build_uf_heatmap(atlas_dir: Path) -> go.Figure:
    """Build a heatmap showing concept handoffs across depth for top UFs."""
    atlas = json.loads((atlas_dir / "atlas.json").read_text())
    n_bins = atlas["n_depth_bins"]

    # Focus on cross-family UFs
    cross_family = [uf for uf in atlas["universal_features"] if uf["n_families"] >= 2]
    cross_family.sort(key=lambda u: (-u["n_families"], -u["n_models"]))

    # Limit to top 30 for readability
    top_ufs = cross_family[:30]

    # Build concept -> integer mapping for heatmap
    concepts = list(CONCEPT_COLORS.keys())
    concept_to_int = {c: i for i, c in enumerate(concepts)}
    concept_to_int[None] = -1

    # Load profiles for variance data
    uf_labels = []
    z_data = []
    hover_texts = []

    for uf in top_ufs:
        uf_id = uf["uf_id"]
        prov_file = atlas_dir / "universal" / uf_id / "provenance.json"
        if prov_file.exists():
            prov = json.loads(prov_file.read_text())
            strats = {_encoding_strategy(p["model_id"]) for p in prov}
            cs_flag = " ★" if ("redundant" in strats and "sparse" in strats) else ""
        else:
            cs_flag = ""
        label = f"{uf_id}: {uf['description']} ({uf['n_models']}m/{uf['n_families']}f){cs_flag}"
        uf_labels.append(label)

        row = []
        hover_row = []
        for bin_idx, concept in enumerate(uf["handoff_template"]):
            row.append(concept_to_int.get(concept, -1))
            depth_pct = bin_idx / n_bins * 100
            hover_row.append(f"{uf_id}<br>Depth: {depth_pct:.0f}-{depth_pct + 100/n_bins:.0f}%<br>Concept: {concept or 'none'}")
        z_data.append(row)
        hover_texts.append(hover_row)

    # Custom colorscale mapping concept integers to colors
    # -1=dark gray, 0..6 = concept colors
    n_concepts = len(concepts)
    colorscale = []
    # Normalize range: -1..6 -> 0..1
    val_range = n_concepts  # 7 concepts + 1 for None
    colorscale.append([0.0, "#222222"])  # None/-1
    for i, concept in enumerate(concepts):
        if concept is None:
            continue
        frac = (i + 1) / val_range
        color = CONCEPT_COLORS.get(concept, "#888888")
        colorscale.append([frac - 0.001, color])
        colorscale.append([frac, color])
    colorscale.append([1.0, colorscale[-1][1]])

    fig = go.Figure()

    # Heatmap
    fig.add_trace(go.Heatmap(
        z=z_data,
        x=[f"{i*5}%" for i in range(n_bins)],
        y=uf_labels,
        hovertext=hover_texts,
        hoverinfo="text",
        colorscale=colorscale,
        zmin=-1,
        zmax=n_concepts - 1,
        showscale=False,
    ))

    fig.update_layout(
        title=dict(
            text=f"Universal Feature Atlas — {len(top_ufs)} Cross-Family UFs<br>"
                 f"<sub>{atlas['n_universal_features']} total UFs from {atlas['n_models']} models</sub>",
            font_size=18,
        ),
        xaxis_title="Normalized Depth (%)",
        yaxis_title="",
        height=max(600, 30 * len(top_ufs) + 200),
        width=1200,
        template="plotly_dark",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=500),
    )

    return fig


def build_uf_provenance(atlas_dir: Path) -> go.Figure:
    """Sankey-like view showing which models contribute to each cross-family UF."""
    atlas = json.loads((atlas_dir / "atlas.json").read_text())

    cross_family = [uf for uf in atlas["universal_features"] if uf["n_families"] >= 2]
    cross_family.sort(key=lambda u: (-u["n_families"], -u["n_models"]))
    top_ufs = cross_family[:20]

    # Load provenance for each
    uf_data = []
    for uf in top_ufs:
        prov_path = atlas_dir / "universal" / uf["uf_id"] / "provenance.json"
        if prov_path.exists():
            prov = json.loads(prov_path.read_text())
            uf_data.append((uf, prov))

    # Build Sankey: models on left, UFs on right
    all_models = set()
    for uf, prov in uf_data:
        for p in prov:
            all_models.add(p["model_id"])
    model_list = sorted(all_models, key=lambda m: (_family(m), m))
    uf_list = [uf["uf_id"] for uf, _ in uf_data]

    # Node indices: models first, then UFs
    model_idx = {m: i for i, m in enumerate(model_list)}
    uf_idx = {u: i + len(model_list) for i, u in enumerate(uf_list)}

    labels = [m.split("/")[-1] for m in model_list] + [f"{u['uf_id']}: {u['description'][:40]}" for u, _ in uf_data]
    node_colors = [FAMILY_COLORS.get(_family(m), "#888") for m in model_list] + ["#ffffff"] * len(uf_list)

    sources, targets, values, link_colors = [], [], [], []
    for uf, prov in uf_data:
        for p in prov:
            mid = p["model_id"]
            if mid in model_idx and uf["uf_id"] in uf_idx:
                sources.append(model_idx[mid])
                targets.append(uf_idx[uf["uf_id"]])
                values.append(max(p.get("peak_eigenvalue", 1), 1))
                hex_c = FAMILY_COLORS.get(_family(mid), "#888888")
                r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
                link_colors.append(f"rgba({r},{g},{b},0.4)")

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=labels,
            color=node_colors,
        ),
        link=dict(
            source=sources, target=targets,
            value=values, color=link_colors,
        ),
    ))

    fig.update_layout(
        title=dict(
            text=f"Universal Feature Provenance — Which Models Share Features",
            font_size=18,
        ),
        height=max(800, 25 * len(model_list) + 200),
        width=1400,
        template="plotly_dark",
        font_size=11,
    )
    return fig


def build_uf_ablation(atlas_dir: Path) -> go.Figure:
    """Bar chart showing ablation impact signature for top cross-family UFs."""
    atlas = json.loads((atlas_dir / "atlas.json").read_text())
    cross_family = [uf for uf in atlas["universal_features"] if uf["n_families"] >= 2]
    cross_family.sort(key=lambda u: (-u["n_families"], -u["n_models"]))
    top_ufs = cross_family[:15]

    concepts = ["credibility", "negation", "sentiment", "causation", "certainty", "moral_valence", "temporal_order"]

    fig = go.Figure()

    for concept in concepts:
        damages = []
        labels = []
        for uf in top_ufs:
            profile_path = atlas_dir / "universal" / uf["uf_id"] / "profile.json"
            if not profile_path.exists():
                damages.append(0)
                labels.append(uf["uf_id"])
                continue
            profile = json.loads(profile_path.read_text())
            abl = profile.get("ablation_signature", {})
            retained = abl.get(concept, 100.0)
            damages.append(round(100 - retained, 1))
            labels.append(f"{uf['uf_id']}: {uf['description'][:30]}")

        fig.add_trace(go.Bar(
            name=concept,
            x=labels,
            y=damages,
            marker_color=CONCEPT_COLORS.get(concept, "#888"),
        ))

    fig.update_layout(
        title="Ablation Impact by Universal Feature (% damage when ablated)",
        barmode="group",
        height=600,
        width=1400,
        template="plotly_dark",
        yaxis_title="% Concept Damage",
        xaxis_tickangle=-45,
        legend=dict(orientation="h", y=1.15),
    )

    return fig


def main():
    atlas_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("feature_library")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("visualizations/cazstellations")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building UF heatmap...")
    fig1 = build_uf_heatmap(atlas_dir)
    path1 = out_dir / "universal_feature_atlas.html"
    fig1.write_html(str(path1))
    print(f"  -> {path1}")

    print("Building UF provenance Sankey...")
    fig2 = build_uf_provenance(atlas_dir)
    path2 = out_dir / "universal_feature_provenance.html"
    fig2.write_html(str(path2))
    print(f"  -> {path2}")

    print("Building UF ablation chart...")
    fig3 = build_uf_ablation(atlas_dir)
    path3 = out_dir / "universal_feature_ablation.html"
    fig3.write_html(str(path3))
    print(f"  -> {path3}")

    # Encoding strategy breakdown
    atlas = json.loads((atlas_dir / "atlas.json").read_text())

    # Load model_ids from per-model features.json files
    models_dir = atlas_dir / "models"
    all_model_ids = []
    if models_dir.exists():
        for slug_dir in models_dir.iterdir():
            feat_file = slug_dir / "features.json"
            if feat_file.exists():
                mid = json.loads(feat_file.read_text()).get("model_id", "")
                if mid:
                    all_model_ids.append(mid)

    print("\n── Encoding strategy breakdown ──")
    redundant_models = [m for m in all_model_ids if _encoding_strategy(m) == "redundant"]
    sparse_models    = [m for m in all_model_ids if _encoding_strategy(m) == "sparse"]
    print(f"  Redundant (MHA): {len(redundant_models)} — {', '.join(sorted(m.split('/')[-1] for m in redundant_models))}")
    print(f"  Sparse (GQA):    {len(sparse_models)} — {', '.join(sorted(m.split('/')[-1] for m in sparse_models))}")

    # Load provenance per UF from universal/{UF_ID}/provenance.json
    uni_dir = atlas_dir / "universal"
    cross_strategy = []
    for uf in atlas["universal_features"]:
        prov_file = uni_dir / uf["uf_id"] / "provenance.json"
        if not prov_file.exists():
            continue
        provenance = json.loads(prov_file.read_text())
        models_in_uf = [p["model_id"] for p in provenance]
        strats = {_encoding_strategy(m) for m in models_in_uf}
        if "redundant" in strats and "sparse" in strats:
            cross_strategy.append((uf, provenance))
    print(f"\n  Cross-strategy UFs (appear in both redundant and sparse models): {len(cross_strategy)}")
    for uf, provenance in sorted(cross_strategy, key=lambda x: (-x[0]["n_families"], -x[0]["n_models"]))[:10]:
        models_in_uf = [p["model_id"] for p in provenance]
        r = sum(1 for m in models_in_uf if _encoding_strategy(m) == "redundant")
        s = sum(1 for m in models_in_uf if _encoding_strategy(m) == "sparse")
        print(f"    {uf['uf_id']}: {uf['description']}  "
              f"({r}R/{s}S, {uf['n_families']} families)")

    print("\nDone. Open in browser or use file_browser.py to view.")


if __name__ == "__main__":
    main()

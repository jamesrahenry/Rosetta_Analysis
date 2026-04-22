#!/usr/bin/env python3
"""
viz_dark_matter_overview.py — Cross-model dark matter summary visualization.

Stacked bar chart showing feature breakdown per model: dark matter vs
concept-labeled segments, with handoff indicators.

Usage:
    python src/viz_dark_matter_overview.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from collections import Counter

import plotly.graph_objects as go
from viz_style import concept_color, CONCEPT_COLORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR = Path(__file__).resolve().parents[1] / "visualizations" / "cazstellations"

CONCEPTS = ["credibility", "certainty", "sentiment", "moral_valence", "causation", "temporal_order", "negation"]

from rosetta_tools.models import all_models

# Display order: by family then scale
MODEL_ORDER = all_models(include_disabled=True)

FAMILY_DIVIDERS = {
    "openai-community/gpt2": "GPT-2",
    "facebook/opt-125m": "OPT",
    "Qwen/Qwen2.5-0.5B": "Qwen",
    "google/gemma-2-2b": "Gemma",
}


def model_short(model_id: str) -> str:
    return model_id.split("/")[-1]


def load_all_labels() -> list[dict]:
    seen = set()
    models = []
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if not d.name.startswith("deepdive_"):
            continue
        fl = d / "feature_labels.json"
        fm_file = d / "feature_map.json"
        if not fl.exists() or not fm_file.exists():
            continue

        data = json.load(open(fl))
        mid = data["model_id"]
        if mid in seen:
            continue
        seen.add(mid)

        fm = json.load(open(fm_file))

        # Per-feature: which concepts does it align with?
        concept_feature_counts = Counter()  # concept -> n features matching
        dark_count = 0
        n_handoffs = 0
        handoff_concepts = Counter()

        for fid, layers in data["features"].items():
            labeled_concepts = set(la["best_concept"] for la in layers if la["best_concept"])
            if labeled_concepts:
                for c in labeled_concepts:
                    concept_feature_counts[c] += 1
                if len(labeled_concepts) > 1:
                    n_handoffs += 1
                    for c in labeled_concepts:
                        handoff_concepts[c] += 1
            else:
                dark_count += 1

        n_persistent = sum(1 for f in fm["features"] if f["lifespan"] >= 5)
        n_transient = sum(1 for f in fm["features"] if f["lifespan"] <= 2)

        models.append({
            "model_id": mid,
            "n_features": data["n_features"],
            "dark_count": dark_count,
            "concept_counts": dict(concept_feature_counts),
            "n_handoffs": n_handoffs,
            "handoff_concepts": dict(handoff_concepts),
            "n_persistent": n_persistent,
            "n_transient": n_transient,
        })

    return models


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading feature labels...")
    all_models = load_all_labels()
    model_lookup = {m["model_id"]: m for m in all_models}

    # Order models
    ordered = [mid for mid in MODEL_ORDER if mid in model_lookup]
    labels = [model_short(mid) for mid in ordered]

    fig = go.Figure()

    # Dark matter bar (bottom)
    dark_vals = [model_lookup[mid]["dark_count"] for mid in ordered]
    fig.add_trace(go.Bar(
        name="Dark matter",
        x=labels,
        y=dark_vals,
        marker_color="rgba(50,80,160,0.7)",
        hovertemplate="%{x}<br>Dark: %{y} features<extra></extra>",
    ))

    # Concept bars stacked on top
    for concept in CONCEPTS:
        vals = [model_lookup[mid]["concept_counts"].get(concept, 0) for mid in ordered]
        if sum(vals) == 0:
            continue
        fig.add_trace(go.Bar(
            name=concept,
            x=labels,
            y=vals,
            marker_color=CONCEPT_COLORS[concept],
            hovertemplate="%{x}<br>" + concept + ": %{y} features<extra></extra>",
        ))

    # Handoff markers as scatter on top
    handoff_x = []
    handoff_y = []
    handoff_text = []
    for i, mid in enumerate(ordered):
        m = model_lookup[mid]
        if m["n_handoffs"] > 0:
            handoff_x.append(labels[i])
            handoff_y.append(m["n_features"] + 3)
            handoff_text.append(f"{m['n_handoffs']} handoffs")

    if handoff_x:
        fig.add_trace(go.Scatter(
            x=handoff_x,
            y=handoff_y,
            mode="text",
            text=[f"↕{t.split()[0]}" for t in handoff_text],
            textfont=dict(size=10, color="rgba(255,200,100,0.9)"),
            hovertemplate="%{x}<br>%{text}<extra></extra>",
            showlegend=False,
        ))

    # Family divider annotations
    for mid, family_name in FAMILY_DIVIDERS.items():
        if mid in model_lookup:
            idx = ordered.index(mid)
            fig.add_vline(
                x=idx - 0.5,
                line=dict(color="rgba(100,120,160,0.3)", width=1, dash="dot"),
            )
            fig.add_annotation(
                x=idx, y=-12,
                text=f"← {family_name}",
                showarrow=False,
                font=dict(size=9, color="rgba(150,160,180,0.6)"),
                xanchor="left",
            )

    # Pythia label
    fig.add_annotation(
        x=0, y=-12,
        text="← Pythia",
        showarrow=False,
        font=dict(size=9, color="rgba(150,160,180,0.6)"),
        xanchor="left",
    )

    total_features = sum(m["n_features"] for m in model_lookup.values())
    total_dark = sum(m["dark_count"] for m in model_lookup.values())
    pct = 100 * total_dark / total_features

    fig.update_layout(
        barmode="stack",
        title=dict(
            text=(
                f"<b>Dark Matter Census — {len(ordered)} Models</b><br>"
                f"<sub>{total_features} features total, "
                f"{total_dark} dark ({pct:.0f}%). "
                f"↕ = concept handoff features. "
                f"7 concept probes, cos>0.5 threshold.</sub>"
            ),
            font=dict(size=16, color="rgba(200,210,230,0.95)"),
        ),
        xaxis=dict(
            title="",
            tickangle=-45,
            tickfont=dict(size=10, color="rgba(180,190,210,0.8)"),
            gridcolor="rgba(60,80,120,0.1)",
        ),
        yaxis=dict(
            title=dict(text="Features", font=dict(size=12, color="rgba(150,160,180,0.7)")),
            tickfont=dict(size=10, color="rgba(150,160,180,0.7)"),
            gridcolor="rgba(60,80,120,0.15)",
        ),
        paper_bgcolor="rgb(3,3,12)",
        plot_bgcolor="rgb(8,10,22)",
        font=dict(color="rgba(200,210,230,0.9)"),
        legend=dict(
            bgcolor="rgba(8,10,22,0.9)",
            bordercolor="rgba(60,80,120,0.3)",
            borderwidth=1,
            font=dict(size=11, color="rgba(200,210,230,0.9)"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=20, t=100, b=100),
        width=1400, height=700,
    )

    out_path = OUT_DIR / "dark_matter_overview.html"
    fig.write_html(str(out_path), include_plotlyjs=True)
    log.info("-> %s", out_path)


if __name__ == "__main__":
    main()

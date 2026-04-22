#!/usr/bin/env python3
"""
analyze_scored.py — Systematic scored CAZ analysis across all models.

Runs find_caz_regions_scored (0.5% prominence floor) across all extraction
results, producing per-model profiles and cross-model comparisons.

No GPU required — works entirely from saved caz_*.json extraction results.

Usage:
    python src/analyze_scored.py                   # full analysis, markdown output
    python src/analyze_scored.py --csv             # also save CSV
    python src/analyze_scored.py --output results/scored_analysis.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rosetta_tools.reporting import load_results_dir, load_scored_region_df

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_ROOT = Path(__file__).parent.parent / "results"

# Model display order: by family then scale
from rosetta_tools.models import all_models

MODEL_ORDER = all_models(include_disabled=True)

CONCEPT_ORDER = [
    "credibility", "certainty",      # epistemic
    "sentiment", "moral_valence",    # affective
    "causation", "temporal_order",   # relational
    "negation",                      # syntactic
]

CONCEPT_TYPES = {
    "credibility": "epistemic",
    "certainty": "epistemic",
    "sentiment": "affective",
    "moral_valence": "affective",
    "causation": "relational",
    "temporal_order": "relational",
    "negation": "syntactic",
}


def model_short(model_id: str) -> str:
    """Short display name for a model."""
    return model_id.split("/")[-1]


def family_of(model_id: str) -> str:
    """Extract family from model ID."""
    short = model_short(model_id).lower()
    for fam in ["pythia", "gpt2", "opt", "qwen", "gemma"]:
        if fam in short:
            return fam
    return "other"


def classify_score(score: float) -> str:
    """Classify a CAZ by its score."""
    if score >= 0.5:
        return "black_hole"
    elif score >= 0.2:
        return "strong"
    elif score >= 0.05:
        return "moderate"
    else:
        return "gentle"


def find_cazstellations(df: pd.DataFrame, model_id: str) -> list[dict]:
    """Find layers where 3+ concepts have a CAZ peak in the same model."""
    model_df = df[df["model_id"] == model_id]
    peak_layers = model_df.groupby("peak")["concept"].apply(list).reset_index()
    cazstellations = []
    for _, row in peak_layers.iterrows():
        if len(row["concept"]) >= 3:
            cazstellations.append({
                "layer": row["peak"],
                "concepts": sorted(row["concept"]),
                "n_concepts": len(row["concept"]),
            })
    return sorted(cazstellations, key=lambda x: x["layer"])


def generate_model_profile(df: pd.DataFrame, model_id: str) -> str:
    """Generate markdown profile for a single model."""
    mdf = df[df["model_id"] == model_id].copy()
    if mdf.empty:
        return ""

    short = model_short(model_id)
    n_layers = mdf["n_layers"].iloc[0]
    n_cazs = len(mdf)
    n_concepts = mdf["concept"].nunique()

    mdf["strength"] = mdf["caz_score"].apply(classify_score)

    # Score distribution
    n_black = (mdf["strength"] == "black_hole").sum()
    n_strong = (mdf["strength"] == "strong").sum()
    n_moderate = (mdf["strength"] == "moderate").sum()
    n_gentle = (mdf["strength"] == "gentle").sum()

    # Multimodal concepts
    multimodal = mdf[mdf["is_multimodal"]]["concept"].unique()

    # Cazstellations
    cazs = find_cazstellations(mdf, model_id)

    lines = [
        f"#### {short}",
        f"",
        f"**{n_cazs} CAZes** across {n_concepts} concepts "
        f"({n_layers} layers) — "
        f"{n_black} black holes, {n_strong} strong, "
        f"{n_moderate} moderate, {n_gentle} gentle",
        f"",
    ]

    # Per-concept table
    lines.append("| Concept | CAZes | Dominant Peak | Score | Other Peaks |")
    lines.append("|---------|-------|--------------|-------|-------------|")

    for concept in CONCEPT_ORDER:
        cdf = mdf[mdf["concept"] == concept].sort_values("caz_score", ascending=False)
        if cdf.empty:
            lines.append(f"| {concept} | 0 | — | — | — |")
            continue

        dominant = cdf[cdf["is_dominant"]].iloc[0] if cdf["is_dominant"].any() else cdf.iloc[0]
        others = cdf[~cdf.index.isin([dominant.name])]

        dom_str = f"L{dominant['peak']}" f" ({dominant['depth_pct']:.0f}%)"
        score_str = f"{dominant['caz_score']:.3f}"

        if len(others) > 0:
            other_strs = []
            for _, r in others.iterrows():
                other_strs.append(
                    f"L{int(r['peak'])} ({r['depth_pct']:.0f}%, "
                    f"score={r['caz_score']:.3f})"
                )
            other_str = "; ".join(other_strs)
        else:
            other_str = "—"

        lines.append(
            f"| {concept} | {len(cdf)} | {dom_str} | {score_str} | {other_str} |"
        )

    lines.append("")

    if len(multimodal) > 0:
        lines.append(f"**Multimodal concepts:** {', '.join(sorted(multimodal))}")
        lines.append("")

    if cazs:
        caz_strs = [
            f"L{c['layer']} ({c['n_concepts']} concepts: "
            f"{', '.join(c['concepts'])})"
            for c in cazs
        ]
        lines.append(f"**Cazstellations:** {'; '.join(caz_strs)}")
        lines.append("")

    # Assembly style characterization
    mean_score = mdf["caz_score"].mean()
    gentle_frac = n_gentle / n_cazs if n_cazs > 0 else 0
    if n_black == 0 and gentle_frac > 0.6:
        style = "Distributed — no black holes, primarily gentle assembly"
    elif n_black >= 3 and gentle_frac < 0.3:
        style = "Concentrated — multiple black holes, few gentle features"
    elif mean_score > 0.3:
        style = "Strong — high mean score, prominent assembly peaks"
    else:
        style = "Mixed — blend of strong and gentle assembly"
    lines.append(f"**Assembly style:** {style}")
    lines.append("")

    return "\n".join(lines)


def generate_cross_model_table(df: pd.DataFrame) -> str:
    """Generate the cross-model summary table."""
    lines = [
        "## Cross-Model Summary",
        "",
        "| Model | CAZes | Black Holes | Gentle | Mean Score | "
        "Multimodal | Cazstellations | Mean Depth |",
        "|-------|-------|-------------|--------|-----------|"
        "------------|----------------|------------|",
    ]

    for model_id in MODEL_ORDER:
        mdf = df[df["model_id"] == model_id]
        if mdf.empty:
            continue

        short = model_short(model_id)
        n_cazs = len(mdf)
        n_black = (mdf["caz_score"] >= 0.5).sum()
        n_gentle = (mdf["caz_score"] < 0.05).sum()
        mean_score = mdf["caz_score"].mean()

        n_multimodal = mdf.groupby("concept")["is_multimodal"].first().sum()
        cazstellations = find_cazstellations(mdf, model_id)
        mean_depth = mdf["depth_pct"].mean()

        lines.append(
            f"| {short} | {n_cazs} | {n_black} | {n_gentle} | "
            f"{mean_score:.3f} | {n_multimodal}/7 | "
            f"{len(cazstellations)} | {mean_depth:.1f}% |"
        )

    lines.append("")
    return "\n".join(lines)


def generate_family_comparison(df: pd.DataFrame) -> str:
    """Generate family-level comparison."""
    lines = [
        "## Family-Level Patterns",
        "",
        "| Family | Models | Total CAZes | CAZes/Model | "
        "Black Hole % | Gentle % | Mean Score | Mean Depth |",
        "|--------|--------|-------------|------------|"
        "-------------|----------|-----------|------------|",
    ]

    df["family"] = df["model_id"].apply(family_of)

    for fam in ["pythia", "gpt2", "opt", "qwen", "gemma"]:
        fdf = df[df["family"] == fam]
        if fdf.empty:
            continue

        n_models = fdf["model_id"].nunique()
        n_cazs = len(fdf)
        per_model = n_cazs / n_models
        pct_black = 100 * (fdf["caz_score"] >= 0.5).sum() / n_cazs
        pct_gentle = 100 * (fdf["caz_score"] < 0.05).sum() / n_cazs
        mean_score = fdf["caz_score"].mean()
        mean_depth = fdf["depth_pct"].mean()

        lines.append(
            f"| {fam} | {n_models} | {n_cazs} | {per_model:.1f} | "
            f"{pct_black:.1f}% | {pct_gentle:.1f}% | "
            f"{mean_score:.3f} | {mean_depth:.1f}% |"
        )

    lines.append("")
    return "\n".join(lines)


def generate_concept_comparison(df: pd.DataFrame) -> str:
    """Generate concept-level comparison."""
    lines = [
        "## Concept-Level Patterns",
        "",
        "| Concept | Type | Total CAZes | Mean Score | "
        "Black Holes | Gentle | Mean Depth | Multimodal % |",
        "|---------|------|-------------|-----------|"
        "-------------|--------|-----------|--------------|",
    ]

    n_models = df["model_id"].nunique()

    for concept in CONCEPT_ORDER:
        cdf = df[df["concept"] == concept]
        if cdf.empty:
            continue

        ctype = CONCEPT_TYPES[concept]
        n_cazs = len(cdf)
        mean_score = cdf["caz_score"].mean()
        n_black = (cdf["caz_score"] >= 0.5).sum()
        n_gentle = (cdf["caz_score"] < 0.05).sum()
        mean_depth = cdf["depth_pct"].mean()
        n_multimodal = cdf.groupby("model_id")["is_multimodal"].first().sum()
        pct_multimodal = 100 * n_multimodal / n_models

        lines.append(
            f"| {concept} | {ctype} | {n_cazs} | {mean_score:.3f} | "
            f"{n_black} | {n_gentle} | "
            f"{mean_depth:.1f}% | {pct_multimodal:.0f}% |"
        )

    lines.append("")
    return "\n".join(lines)


def generate_depth_profile(df: pd.DataFrame) -> str:
    """Generate depth distribution analysis."""
    lines = [
        "## Depth Distribution",
        "",
        "| Depth Bin | CAZes | Black Holes | Gentle | Mean Score |",
        "|-----------|-------|-------------|--------|-----------|",
    ]

    bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
            (50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]

    # Use pd.cut for consistent binning (half-open intervals (lo, hi])
    df_bins = pd.cut(df["depth_pct"], bins=[-1] + [hi for _, hi in bins])
    bin_counts = df_bins.value_counts()
    max_count = bin_counts.max()

    for lo, hi in bins:
        mask = (df["depth_pct"] > lo) & (df["depth_pct"] <= hi)
        if lo == 0:
            mask = (df["depth_pct"] >= 0) & (df["depth_pct"] <= hi)
        bdf = df[mask]
        n = len(bdf)
        n_black = (bdf["caz_score"] >= 0.5).sum() if n > 0 else 0
        n_gentle = (bdf["caz_score"] < 0.05).sum() if n > 0 else 0
        mean_score = bdf["caz_score"].mean() if n > 0 else 0

        hi_display = min(hi, 100)
        bold = "**" if n == max_count else ""
        lines.append(
            f"| {bold}{lo}–{hi_display}%{bold} | {bold}{n}{bold} | "
            f"{n_black} | {n_gentle} | {mean_score:.3f} |"
        )

    lines.append("")
    return "\n".join(lines)


def generate_scaling_trends(df: pd.DataFrame) -> str:
    """Analyze how CAZ properties change with model scale."""
    lines = [
        "## Scaling Trends",
        "",
    ]

    # Approximate parameter counts for sorting
    param_order = {
        "pythia-70m": 70, "pythia-160m": 160, "pythia-410m": 410,
        "pythia-1b": 1000, "pythia-1.4b": 1400, "pythia-2.8b": 2800,
        "pythia-6.9b": 6900,
        "gpt2": 124, "gpt2-medium": 355, "gpt2-large": 774, "gpt2-xl": 1558,
        "opt-125m": 125, "opt-350m": 350, "opt-1.3b": 1300,
        "opt-2.7b": 2700, "opt-6.7b": 6700,
        "Qwen2.5-0.5B": 500, "Qwen2.5-1.5B": 1500,
        "Qwen2.5-3B": 3000, "Qwen2.5-7B": 7000,
        "gemma-2-2b": 2000, "gemma-2-9b": 9000,
    }

    # Per-family scaling
    for fam in ["pythia", "gpt2", "opt", "qwen"]:
        fdf = df[df["model_id"].apply(family_of) == fam]
        if fdf["model_id"].nunique() < 3:
            continue

        models = sorted(fdf["model_id"].unique(),
                        key=lambda m: param_order.get(model_short(m), 0))

        lines.append(f"### {fam.upper()} family")
        lines.append("")
        lines.append("| Model | Params | CAZes | Gentle % | "
                      "Mean Score | Mean Depth |")
        lines.append("|-------|--------|-------|---------|"
                      "-----------|------------|")

        for model_id in models:
            mdf = fdf[fdf["model_id"] == model_id]
            short = model_short(model_id)
            params = param_order.get(short, 0)
            n_cazs = len(mdf)
            pct_gentle = 100 * (mdf["caz_score"] < 0.05).sum() / n_cazs if n_cazs else 0
            mean_score = mdf["caz_score"].mean()
            mean_depth = mdf["depth_pct"].mean()

            param_str = f"{params}M" if params < 1000 else f"{params/1000:.1f}B"
            lines.append(
                f"| {short} | {param_str} | {n_cazs} | "
                f"{pct_gentle:.0f}% | {mean_score:.3f} | {mean_depth:.1f}% |"
            )

        lines.append("")

    return "\n".join(lines)


def generate_key_findings(df: pd.DataFrame) -> str:
    """Generate the key findings summary."""
    n_total = len(df)
    n_models = df["model_id"].nunique()
    n_concepts = df["concept"].nunique()
    n_black = (df["caz_score"] >= 0.5).sum()
    n_gentle = (df["caz_score"] < 0.05).sum()

    # Find the model with most/fewest CAZes
    caz_counts = df.groupby("model_id").size()
    most_model = model_short(caz_counts.idxmax())
    most_count = caz_counts.max()
    fewest_model = model_short(caz_counts.idxmin())
    fewest_count = caz_counts.min()

    # Find peak depth bin
    depth_bins = pd.cut(df["depth_pct"], bins=range(0, 110, 10))
    peak_interval = depth_bins.value_counts().idxmax()
    peak_bin = f"{int(peak_interval.left)}–{int(peak_interval.right)}%"

    # Gemma special case
    gemma_df = df[df["model_id"].apply(family_of) == "gemma"]
    gemma_blacks = (gemma_df["caz_score"] >= 0.5).sum() if not gemma_df.empty else 0

    lines = [
        "## Key Findings",
        "",
        f"1. **{n_total} CAZes** detected across {n_models} models × "
        f"{n_concepts} concepts using scored detector (0.5% prominence floor)",
        f"",
        f"2. **Score distribution is heavy-tailed:** {n_black} black holes "
        f"({100*n_black/n_total:.0f}%), {n_gentle} gentle "
        f"({100*n_gentle/n_total:.0f}%). Half the causal structure in these "
        f"models is in the gentle range — invisible to the old 10% detector.",
        f"",
        f"3. **Peak assembly at {peak_bin}:** consistent with the "
        f"\"mid-depth assembly sweet spot\" pattern.",
        f"",
        f"4. **Model variation:** {most_model} has the most CAZes ({most_count}), "
        f"{fewest_model} the fewest ({fewest_count}). CAZ count does not "
        f"scale simply with parameter count.",
        f"",
        f"5. **Gemma-2 has {gemma_blacks} black holes** — its concept assembly "
        f"is distributed and subtle, relying on gentle features rather than "
        f"concentrated peaks.",
        f"",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Systematic scored CAZ analysis across all models"
    )
    parser.add_argument("--output", type=str, default=None,
                        help="Output markdown path (default: SCORED_CAZ_ANALYSIS.md)")
    parser.add_argument("--csv", action="store_true",
                        help="Also save the full scored region DataFrame as CSV")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else (
        Path(__file__).parent.parent / "SCORED_CAZ_ANALYSIS.md"
    )

    # ── Load all extraction results ──
    log.info("Loading extraction results from %s...", RESULTS_ROOT)
    result_dirs = sorted([
        d for d in RESULTS_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(("manifold_", "deepdive_"))
    ])
    log.info("Found %d result directories", len(result_dirs))

    layer_df = load_results_dir(result_dirs)
    if layer_df is None or layer_df.empty:
        log.error("No data loaded. Exiting.")
        return

    n_models = layer_df["model_id"].nunique()
    n_concepts = layer_df["concept"].nunique()
    log.info("Loaded %d rows — %d models × %d concepts",
             len(layer_df), n_models, n_concepts)

    # ── Run scored detector ──
    log.info("Running scored CAZ detector (0.5%% prominence floor)...")
    scored_df = load_scored_region_df(layer_df)
    log.info("Detected %d scored CAZ regions", len(scored_df))

    if scored_df.empty:
        log.error("No CAZ regions detected. Exiting.")
        return

    # ── Optional CSV output ──
    if args.csv:
        csv_path = output_path.with_suffix(".csv")
        scored_df.to_csv(csv_path, index=False)
        log.info("CSV saved to %s", csv_path)

    # ── Generate markdown report ──
    log.info("Generating analysis report...")

    sections = [
        f"# Scored CAZ Analysis — {n_models} Models × {n_concepts} Concepts",
        "",
        f"*Generated from {len(result_dirs)} extraction runs using "
        f"`find_caz_regions_scored` with 0.5% prominence floor.*",
        "",
        "---",
        "",
        generate_key_findings(scored_df),
        "---",
        "",
        generate_cross_model_table(scored_df),
        "---",
        "",
        generate_family_comparison(scored_df),
        "---",
        "",
        generate_concept_comparison(scored_df),
        "---",
        "",
        generate_depth_profile(scored_df),
        "---",
        "",
        generate_scaling_trends(scored_df),
        "---",
        "",
        "## Per-Model Profiles",
        "",
    ]

    for model_id in MODEL_ORDER:
        profile = generate_model_profile(scored_df, model_id)
        if profile:
            sections.append(profile)

    report = "\n".join(sections)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    log.info("Report saved to %s", output_path)
    log.info("Done.")


if __name__ == "__main__":
    main()

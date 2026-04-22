"""
analyze.py — CAZ analysis and visualization for multi-family scale ladders.

Loads extraction results, detects CAZ boundaries, computes regional statistics,
and generates visualizations comparing CAZ positioning across scales and families.

Usage
-----
    # Analyze a single family
    python src/analyze.py --family pythia

    # Analyze all families
    python src/analyze.py --all

    # Just heatmaps (one per family + combined)
    python src/analyze.py --all --heatmap-only

    # Full analysis with per-model profiles
    python src/analyze.py --all --profiles

    # Analyze specific result directories
    python src/analyze.py --results results/pythia_* results/gpt2_*
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rosetta_tools.reporting import load_results_dir, load_run_summary
from rosetta_tools.viz import plot_caz_profile, plot_concept_comparison, plot_peak_heatmap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_ROOT = Path("results")
VIZ_DIR = Path("visualizations")

# Family prefixes used in result directory names
KNOWN_FAMILIES = ["pythia", "gpt2", "opt", "qwen2", "gemma2", "custom"]


def discover_families() -> dict[str, list[Path]]:
    """Discover result directories grouped by family prefix."""
    families = {}
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        matched = False
        for fam in KNOWN_FAMILIES:
            if name.startswith(f"{fam}_"):
                families.setdefault(fam, []).append(d)
                matched = True
                break
        if not matched:
            families.setdefault("other", []).append(d)
    return families


def analyze_family(family: str, dirs: list[Path], heatmap_only: bool, profiles: bool) -> None:
    """Run analysis for a single family."""
    log.info("Loading %s: %d result directories...", family, len(dirs))
    df = load_results_dir(dirs)
    if df is None or df.empty:
        log.warning("No data for family %s, skipping.", family)
        return

    n_models = df["model_id"].nunique()
    n_concepts = df["concept"].nunique()
    log.info("Loaded %d rows — %d models, %d concepts", len(df), n_models, n_concepts)

    # Peak summary table
    peaks = df[df["is_peak"]][["model_id", "concept", "layer", "separation", "depth_pct"]]
    log.info("\n[%s] Peak summary:\n%s", family,
             peaks.pivot_table(index="model_id", columns="concept",
                               values="depth_pct", aggfunc="first").round(1).to_string())

    fam_viz = VIZ_DIR / family
    fam_viz.mkdir(parents=True, exist_ok=True)

    # Heatmap
    heatmap_path = fam_viz / "peak_depth_heatmap.png"
    plot_peak_heatmap(df, heatmap_path,
                      title=f"CAZ Peak Depth — {family.upper()} Scale Ladder")
    log.info("Heatmap → %s", heatmap_path)

    if heatmap_only:
        return

    # Multi-concept overlay
    model_ids = sorted(df["model_id"].unique().tolist())
    overlay_path = fam_viz / "concept_comparison_all_scales.png"
    plot_concept_comparison(df, overlay_path, model_ids=model_ids,
                            title=f"CAZ metrics — all concepts × {family} scales")
    log.info("Overlay → %s", overlay_path)

    # Per-concept × model profiles
    if profiles:
        for concept in df["concept"].unique():
            for model_id in df["model_id"].unique():
                sub = df[(df["concept"] == concept) & (df["model_id"] == model_id)]
                if sub.empty:
                    continue
                model_short = model_id.split("/")[-1]
                out = fam_viz / f"profile_{concept}_{model_short}.png"
                try:
                    plot_caz_profile(df, concept, model_id, out)
                except Exception as e:
                    log.warning("Skipping %s × %s: %s", concept, model_short, e)

    return df


def analyze_combined(all_dfs: list, heatmap_only: bool) -> None:
    """Generate combined cross-family visualizations."""
    import pandas as pd
    combined = pd.concat(all_dfs, ignore_index=True)
    n_models = combined["model_id"].nunique()
    n_concepts = combined["concept"].nunique()
    log.info("Combined: %d rows — %d models, %d concepts", len(combined), n_models, n_concepts)

    # Combined heatmap
    heatmap_path = VIZ_DIR / "combined_peak_depth_heatmap.png"
    plot_peak_heatmap(combined, heatmap_path,
                      title="CAZ Peak Depth — All Families")
    log.info("Combined heatmap → %s", heatmap_path)

    # Peak summary across all families
    peaks = combined[combined["is_peak"]][["model_id", "concept", "depth_pct"]]
    log.info("\nCombined peak summary:\n%s",
             peaks.pivot_table(index="model_id", columns="concept",
                               values="depth_pct", aggfunc="first").round(1).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CAZ analysis — multi-family scale ladders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--family", nargs="+", metavar="NAME",
                       help="One or more family names to analyze")
    group.add_argument("--all", action="store_true",
                       help="Analyze all families found in results/")
    group.add_argument("--results", nargs="+", type=str,
                       help="Specific result directories to analyze")
    parser.add_argument("--heatmap-only", action="store_true")
    parser.add_argument("--profiles", action="store_true",
                        help="Generate per-concept×model profile plots")
    args = parser.parse_args()

    VIZ_DIR.mkdir(exist_ok=True)

    if args.results:
        dirs = [Path(p) for p in args.results]
        df = load_results_dir(dirs)
        if df is None or df.empty:
            log.error("No data loaded.")
            return
        peaks = df[df["is_peak"]][["model_id", "concept", "depth_pct"]]
        log.info("\nPeak summary:\n%s",
                 peaks.pivot_table(index="model_id", columns="concept",
                                   values="depth_pct", aggfunc="first").round(1).to_string())
        heatmap_path = VIZ_DIR / "peak_depth_heatmap.png"
        plot_peak_heatmap(df, heatmap_path, title="CAZ Peak Depth")
        log.info("Heatmap → %s", heatmap_path)
        return

    # Discover families
    family_dirs = discover_families()
    if not family_dirs:
        log.error("No result directories found in %s", RESULTS_ROOT)
        return

    if args.all:
        target_families = list(family_dirs.keys())
    else:
        target_families = args.family
        missing = [f for f in target_families if f not in family_dirs]
        if missing:
            log.error("No results found for families: %s  (available: %s)",
                      missing, list(family_dirs.keys()))
            sys.exit(1)

    all_dfs = []
    for fam in target_families:
        dirs = family_dirs.get(fam, [])
        if not dirs:
            log.warning("No results for family %s, skipping.", fam)
            continue
        df = analyze_family(fam, dirs, args.heatmap_only, args.profiles)
        if df is not None:
            all_dfs.append(df)

    # Combined analysis if multiple families
    if len(all_dfs) > 1:
        analyze_combined(all_dfs, args.heatmap_only)

    log.info("Done.")


if __name__ == "__main__":
    import sys
    main()

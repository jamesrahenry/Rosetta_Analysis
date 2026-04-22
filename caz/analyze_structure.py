"""
analyze_structure.py — CAZ structural analysis beyond the peak.

Extracts shape, width, asymmetry, S-C coupling, and multi-modality from
the existing layer-wise metrics (no re-extraction needed).

Five structural dimensions
--------------------------
1. Width        — CAZ extent (entry to exit) as fraction of model depth
2. Asymmetry    — rise vs. fall: does the concept crystallize fast and decay
                  slow, or vice versa?
3. S-C coupling — does coherence lead, lag, or co-occur with separation?
4. Multi-modality — does the S(l) curve have secondary peaks?
5. Post-CAZ decay — character of degradation (rate, steepness, residual)

Usage
-----
    # Full analysis across all families
    python src/analyze_structure.py --all

    # Single family
    python src/analyze_structure.py --family pythia

    # Output: CSV tables + PNG visualizations in visualizations/structure/
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")

# Reuse the existing infrastructure
from rosetta_tools.reporting import load_results_dir
from rosetta_tools.caz import find_caz_boundary, LayerMetrics
from rosetta_tools.viz import CONCEPT_META, CONCEPT_ORDER, TYPE_COLORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_ROOT = Path("results")
VIZ_DIR = Path("visualizations/structure")
KNOWN_FAMILIES = ["pythia", "gpt2", "opt", "qwen2", "gemma2"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------------
# Structural feature extraction
# ---------------------------------------------------------------------------


@dataclass
class CAZStructure:
    """Structural features for a single concept x model run."""

    model_id: str
    concept: str
    n_layers: int

    # Boundary
    caz_start: int
    caz_peak: int
    caz_end: int
    caz_width: int
    caz_width_pct: float           # width as % of model depth

    # Peak metrics
    peak_separation: float
    peak_coherence: float
    peak_depth_pct: float

    # Asymmetry  (positive = steep rise / slow decay; negative = opposite)
    rise_span: int                  # layers from start to peak
    fall_span: int                  # layers from peak to end
    rise_rate: float                # mean velocity during rise
    fall_rate: float                # mean velocity during fall (negative values)
    asymmetry_ratio: float          # rise_span / fall_span (>1 = slow rise)
    asymmetry_velocity: float       # |rise_rate| / |fall_rate| (>1 = steeper rise)

    # S-C coupling
    sc_pearson_r: float             # Pearson r(S, C) across all layers
    sc_spearman_rho: float          # Spearman rho(S, C) across all layers
    sc_peak_offset: int             # coherence_peak - separation_peak (+ = C lags)
    sc_caz_pearson_r: float         # Pearson r(S, C) within CAZ region only

    # Multi-modality
    n_peaks: int                    # number of local maxima in S(l) above threshold
    secondary_peak_layer: int       # layer of strongest secondary peak (-1 if none)
    secondary_peak_sep: float       # separation at secondary peak
    peak_prominence_ratio: float    # primary / secondary separation (inf if unimodal)

    # Post-CAZ decay
    post_caz_mean_sep: float        # mean separation after CAZ
    post_caz_min_sep: float         # minimum separation after CAZ
    decay_ratio: float              # post_caz_mean / peak_separation
    decay_slope: float              # linear regression slope of S(l) post-CAZ
    residual_fraction: float        # min_post / peak — how much signal survives


def extract_structure(sub: pd.DataFrame) -> CAZStructure:
    """Compute all structural features from a single concept x model DataFrame.

    Parameters
    ----------
    sub : pd.DataFrame
        Rows for one (model_id, concept) pair, sorted by layer.
    """
    sub = sub.sort_values("layer").reset_index(drop=True)
    model_id = sub["model_id"].iloc[0]
    concept = sub["concept"].iloc[0]
    n_layers = int(sub["n_layers"].iloc[0])

    seps = sub["separation"].values.astype(np.float64)
    cohs = sub["coherence"].values.astype(np.float64)
    vels = sub["velocity"].values.astype(np.float64)
    layers = sub["layer"].values

    # --- Boundary detection via existing caz.py ---
    metrics = [
        LayerMetrics(layer=int(r["layer"]), separation=r["separation"],
                     coherence=r["coherence"], velocity=r["velocity"])
        for _, r in sub.iterrows()
    ]
    boundary = find_caz_boundary(metrics)
    caz_s, caz_p, caz_e = boundary.caz_start, boundary.caz_peak, boundary.caz_end

    caz_width = boundary.caz_width
    caz_width_pct = 100.0 * caz_width / n_layers if n_layers > 0 else 0.0

    peak_sep = float(seps[caz_p])
    peak_coh = float(cohs[caz_p])
    peak_depth_pct = 100.0 * caz_p / n_layers if n_layers > 0 else 0.0

    # --- Asymmetry ---
    rise_span = max(caz_p - caz_s, 1)
    fall_span = max(caz_e - caz_p, 1)

    rise_vels = vels[caz_s:caz_p + 1]
    fall_vels = vels[caz_p:caz_e + 1]
    rise_rate = float(np.mean(rise_vels)) if len(rise_vels) > 0 else 0.0
    fall_rate = float(np.mean(fall_vels)) if len(fall_vels) > 0 else 0.0

    asymmetry_ratio = rise_span / fall_span if fall_span > 0 else float("inf")
    abs_fall = abs(fall_rate) if fall_rate != 0 else 1e-10
    asymmetry_velocity = abs(rise_rate) / abs_fall

    # --- S-C coupling ---
    if len(seps) > 2:
        sc_pr, _ = pearsonr(seps, cohs)
        sc_sr, _ = spearmanr(seps, cohs)
    else:
        sc_pr, sc_sr = 0.0, 0.0
    sc_pr = sc_pr if np.isfinite(sc_pr) else 0.0
    sc_sr = sc_sr if np.isfinite(sc_sr) else 0.0

    coh_peak = int(np.argmax(cohs))
    sc_peak_offset = coh_peak - caz_p

    # S-C coupling within CAZ
    caz_seps = seps[caz_s:caz_e + 1]
    caz_cohs = cohs[caz_s:caz_e + 1]
    if len(caz_seps) > 2:
        sc_caz_r, _ = pearsonr(caz_seps, caz_cohs)
        sc_caz_r = sc_caz_r if np.isfinite(sc_caz_r) else 0.0
    else:
        sc_caz_r = 0.0

    # --- Multi-modality ---
    # Find peaks with minimum prominence = 10% of max separation
    min_prominence = 0.1 * peak_sep if peak_sep > 0 else 0.01
    peaks_idx, properties = find_peaks(seps, prominence=min_prominence, distance=2)
    n_peaks = len(peaks_idx)

    if n_peaks > 1:
        prominences = properties["prominences"]
        # Sort by separation value, primary is the tallest
        sorted_peaks = sorted(zip(peaks_idx, seps[peaks_idx], prominences),
                              key=lambda x: x[1], reverse=True)
        secondary = sorted_peaks[1]
        secondary_peak_layer = int(secondary[0])
        secondary_peak_sep = float(secondary[1])
        peak_prominence_ratio = peak_sep / secondary_peak_sep if secondary_peak_sep > 0 else float("inf")
    else:
        secondary_peak_layer = -1
        secondary_peak_sep = 0.0
        peak_prominence_ratio = float("inf")

    # --- Post-CAZ decay ---
    post_seps = seps[caz_e + 1:] if caz_e < len(seps) - 1 else np.array([])
    if len(post_seps) > 0:
        post_caz_mean = float(np.mean(post_seps))
        post_caz_min = float(np.min(post_seps))
        decay_ratio = post_caz_mean / peak_sep if peak_sep > 0 else 0.0
        residual_fraction = post_caz_min / peak_sep if peak_sep > 0 else 0.0
        # Linear fit for decay slope
        if len(post_seps) > 1:
            x = np.arange(len(post_seps))
            decay_slope = float(np.polyfit(x, post_seps, 1)[0])
        else:
            decay_slope = 0.0
    else:
        post_caz_mean = peak_sep
        post_caz_min = peak_sep
        decay_ratio = 1.0
        decay_slope = 0.0
        residual_fraction = 1.0

    return CAZStructure(
        model_id=model_id, concept=concept, n_layers=n_layers,
        caz_start=caz_s, caz_peak=caz_p, caz_end=caz_e,
        caz_width=caz_width, caz_width_pct=caz_width_pct,
        peak_separation=peak_sep, peak_coherence=peak_coh,
        peak_depth_pct=peak_depth_pct,
        rise_span=rise_span, fall_span=fall_span,
        rise_rate=rise_rate, fall_rate=fall_rate,
        asymmetry_ratio=asymmetry_ratio, asymmetry_velocity=asymmetry_velocity,
        sc_pearson_r=sc_pr, sc_spearman_rho=sc_sr,
        sc_peak_offset=sc_peak_offset, sc_caz_pearson_r=sc_caz_r,
        n_peaks=n_peaks, secondary_peak_layer=secondary_peak_layer,
        secondary_peak_sep=secondary_peak_sep,
        peak_prominence_ratio=peak_prominence_ratio,
        post_caz_mean_sep=post_caz_mean, post_caz_min_sep=post_caz_min,
        decay_ratio=decay_ratio, decay_slope=decay_slope,
        residual_fraction=residual_fraction,
    )


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------


def extract_all_structures(df: pd.DataFrame) -> pd.DataFrame:
    """Extract structural features for every (model_id, concept) pair."""
    records = []
    for (model_id, concept), sub in df.groupby(["model_id", "concept"]):
        try:
            s = extract_structure(sub)
            records.append(asdict(s))
        except Exception as e:
            log.warning("Failed %s × %s: %s", model_id, concept, e)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Visualization: structural feature plots
# ---------------------------------------------------------------------------


def _family_from_model(model_id: str) -> str:
    """Best-effort family tag from model_id."""
    mid = model_id.lower()
    if "pythia" in mid:
        return "pythia"
    if "gpt2" in mid or "gpt-2" in mid:
        return "gpt2"
    if "opt" in mid:
        return "opt"
    if "qwen" in mid:
        return "qwen2"
    if "gemma" in mid:
        return "gemma2"
    return "other"


def plot_width_by_concept(sdf: pd.DataFrame, out: Path) -> None:
    """Box plot: CAZ width (%) grouped by concept, colored by type."""
    concepts = [c for c in CONCEPT_ORDER if c in sdf["concept"].unique()]
    fig, ax = plt.subplots(figsize=(12, 5))

    data = [sdf[sdf["concept"] == c]["caz_width_pct"].values for c in concepts]
    bp = ax.boxplot(data, positions=range(len(concepts)), widths=0.6,
                    patch_artist=True, showfliers=True)

    for i, c in enumerate(concepts):
        color = CONCEPT_META.get(c, {}).get("color", "#888")
        bp["boxes"][i].set_facecolor(color)
        bp["boxes"][i].set_alpha(0.5)
        bp["medians"][i].set_color("black")

    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels([c.replace("_", "\n") for c in concepts], fontsize=9)
    ax.set_ylabel("CAZ Width (% of model depth)")
    ax.set_title("CAZ Width Distribution by Concept", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out)


def plot_asymmetry_scatter(sdf: pd.DataFrame, out: Path) -> None:
    """Scatter: rise span vs fall span, colored by concept type."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for concept in sdf["concept"].unique():
        sub = sdf[sdf["concept"] == concept]
        color = CONCEPT_META.get(concept, {}).get("color", "#888")
        ctype = CONCEPT_META.get(concept, {}).get("type", "?")
        ax.scatter(sub["rise_span"], sub["fall_span"], c=color,
                   label=f"{concept} ({ctype})", alpha=0.7, s=40, edgecolors="white",
                   linewidths=0.5)

    # Diagonal = symmetric
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.5, label="symmetric")
    ax.set_xlabel("Rise span (layers: start → peak)")
    ax.set_ylabel("Fall span (layers: peak → end)")
    ax.set_title("CAZ Asymmetry: Rise vs. Fall Span", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out)


def plot_sc_coupling(sdf: pd.DataFrame, out: Path) -> None:
    """Two panels: (a) S-C Pearson r distribution by concept, (b) peak offset histogram."""
    concepts = [c for c in CONCEPT_ORDER if c in sdf["concept"].unique()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # (a) S-C correlation by concept
    data = [sdf[sdf["concept"] == c]["sc_pearson_r"].values for c in concepts]
    bp = ax1.boxplot(data, positions=range(len(concepts)), widths=0.6,
                     patch_artist=True, showfliers=True)
    for i, c in enumerate(concepts):
        color = CONCEPT_META.get(c, {}).get("color", "#888")
        bp["boxes"][i].set_facecolor(color)
        bp["boxes"][i].set_alpha(0.5)
        bp["medians"][i].set_color("black")
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_xticks(range(len(concepts)))
    ax1.set_xticklabels([c.replace("_", "\n") for c in concepts], fontsize=8)
    ax1.set_ylabel("Pearson r (S, C) across layers")
    ax1.set_title("Separation–Coherence Coupling", fontweight="bold")

    # (b) Peak offset: positive = coherence lags separation
    offsets = sdf["sc_peak_offset"].values
    ax2.hist(offsets, bins=range(int(offsets.min()) - 1, int(offsets.max()) + 2),
             color="#555", alpha=0.7, edgecolor="white")
    ax2.axvline(0, color="red", linestyle="--", alpha=0.6, label="aligned")
    median_offset = np.median(offsets)
    ax2.axvline(median_offset, color="blue", linestyle="--", alpha=0.6,
                label=f"median = {median_offset:.0f}")
    ax2.set_xlabel("Peak offset (coherence_peak − separation_peak, in layers)")
    ax2.set_ylabel("Count")
    ax2.set_title("Coherence–Separation Peak Lag", fontweight="bold")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out)


def plot_multimodality(sdf: pd.DataFrame, out: Path) -> None:
    """Stacked bar: fraction of runs with 1, 2, 3+ peaks per concept."""
    concepts = [c for c in CONCEPT_ORDER if c in sdf["concept"].unique()]
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, c in enumerate(concepts):
        sub = sdf[sdf["concept"] == c]
        total = len(sub)
        if total == 0:
            continue
        n1 = (sub["n_peaks"] == 1).sum() / total
        n2 = (sub["n_peaks"] == 2).sum() / total
        n3 = (sub["n_peaks"] >= 3).sum() / total
        color = CONCEPT_META.get(c, {}).get("color", "#888")
        ax.bar(i, n1, color=color, alpha=0.9, label="1 peak" if i == 0 else "")
        ax.bar(i, n2, bottom=n1, color=color, alpha=0.55,
               label="2 peaks" if i == 0 else "")
        ax.bar(i, n3, bottom=n1 + n2, color=color, alpha=0.25,
               label="3+ peaks" if i == 0 else "")

    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels([c.replace("_", "\n") for c in concepts], fontsize=9)
    ax.set_ylabel("Fraction of model runs")
    ax.set_title("S(l) Multi-modality by Concept", fontweight="bold")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out)


def plot_decay_profile(sdf: pd.DataFrame, out: Path) -> None:
    """Scatter: decay_ratio vs peak_depth_pct — do late-peaking concepts decay less?"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for concept in sdf["concept"].unique():
        sub = sdf[sdf["concept"] == concept]
        # Skip entries with no post-CAZ region
        sub = sub[sub["decay_ratio"] < 1.0]
        if sub.empty:
            continue
        color = CONCEPT_META.get(concept, {}).get("color", "#888")
        ax.scatter(sub["peak_depth_pct"], sub["decay_ratio"], c=color,
                   label=concept, alpha=0.7, s=40, edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Peak depth (%)")
    ax.set_ylabel("Decay ratio (post-CAZ mean S / peak S)")
    ax.set_title("Post-CAZ Decay vs. Peak Depth", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out)


def plot_structure_heatmap(sdf: pd.DataFrame, out: Path) -> None:
    """Heatmap: concept × model showing CAZ width (%) — the structural analogue
    of the existing peak-depth heatmap."""
    pivot = sdf.pivot_table(
        index="model_id", columns="concept", values="caz_width_pct", aggfunc="first"
    )
    col_order = [c for c in CONCEPT_ORDER if c in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.3),
                                    max(4, len(pivot) * 0.7)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0,
                   vmax=min(100, pivot.values[np.isfinite(pivot.values)].max() * 1.2))

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.replace("_", "\n") for c in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([m.split("/")[-1] for m in pivot.index], fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color="black" if val < 60 else "white")

    plt.colorbar(im, ax=ax, label="CAZ Width (%)", fraction=0.03)
    ax.set_title("CAZ Width by Concept × Model", fontweight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out)


def plot_example_curves(df: pd.DataFrame, sdf: pd.DataFrame, out: Path) -> None:
    """Pick 4 interesting cases and plot full S(l)/C(l) with CAZ boundaries marked.

    Selects: widest CAZ, narrowest CAZ, most asymmetric, most multimodal.
    """
    picks = {}

    # Widest CAZ (excluding trivial full-model spans)
    valid = sdf[sdf["caz_width_pct"] < 95]
    if not valid.empty:
        row = valid.loc[valid["caz_width_pct"].idxmax()]
        picks["Widest CAZ"] = (row["model_id"], row["concept"])

    # Narrowest CAZ
    narrow = sdf[sdf["caz_width"] > 1]
    if not narrow.empty:
        row = narrow.loc[narrow["caz_width_pct"].idxmin()]
        picks["Narrowest CAZ"] = (row["model_id"], row["concept"])

    # Most asymmetric (rise >> fall)
    asym = sdf[np.isfinite(sdf["asymmetry_ratio"])]
    if not asym.empty:
        row = asym.loc[asym["asymmetry_ratio"].idxmax()]
        picks["Most asymmetric (slow rise)"] = (row["model_id"], row["concept"])

    # Most multimodal
    multi = sdf[sdf["n_peaks"] >= 2]
    if not multi.empty:
        row = multi.loc[multi["peak_prominence_ratio"].idxmin()]
        picks["Most bimodal"] = (row["model_id"], row["concept"])

    if not picks:
        return

    n = len(picks)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for ax, (label, (mid, concept)) in zip(axes, picks.items()):
        sub = df[(df["model_id"] == mid) & (df["concept"] == concept)].sort_values("layer")
        struct = sdf[(sdf["model_id"] == mid) & (sdf["concept"] == concept)].iloc[0]

        color = CONCEPT_META.get(concept, {}).get("color", "#333")
        model_short = mid.split("/")[-1]

        # Separation
        ax.plot(sub["depth_pct"], sub["separation"], "o-", color=color,
                linewidth=2, markersize=3, label="S(l)")

        # Coherence on twin axis
        ax2 = ax.twinx()
        ax2.plot(sub["depth_pct"], sub["coherence"], "s--", color=color,
                 linewidth=1.2, markersize=2.5, alpha=0.5, label="C(l)")
        ax2.set_ylabel("C(l)", alpha=0.6)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_alpha(0.3)

        # Mark CAZ boundaries
        n_layers = int(struct["n_layers"])
        for lyr, lbl, ls in [(struct["caz_start"], "start", ":"),
                              (struct["caz_peak"], "peak", "--"),
                              (struct["caz_end"], "end", ":")]:
            pct = 100.0 * lyr / n_layers
            ax.axvline(pct, color="red", linestyle=ls, alpha=0.5)
            ax.text(pct, ax.get_ylim()[1] * 0.95, f" {lbl} L{lyr}",
                    fontsize=7, color="red", alpha=0.7)

        # Shade CAZ region
        start_pct = 100.0 * struct["caz_start"] / n_layers
        end_pct = 100.0 * struct["caz_end"] / n_layers
        ax.axvspan(start_pct, end_pct, alpha=0.08, color=color)

        ax.set_ylabel("S(l)")
        ax.set_xlabel("Relative depth (%)")
        ax.set_title(f"{label}: {concept} × {model_short}  "
                     f"[width={struct['caz_width_pct']:.0f}%, "
                     f"asym={struct['asymmetry_ratio']:.1f}, "
                     f"peaks={struct['n_peaks']}]",
                     fontweight="bold", fontsize=10)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  → %s", out)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def print_summary(sdf: pd.DataFrame) -> None:
    """Log key structural findings."""
    log.info("\n" + "=" * 72)
    log.info("CAZ STRUCTURAL ANALYSIS — SUMMARY")
    log.info("=" * 72)

    # Width by concept
    log.info("\n--- CAZ Width (%%  of model depth) by concept ---")
    width_summary = sdf.groupby("concept")["caz_width_pct"].agg(["mean", "std", "median", "min", "max"])
    log.info("\n%s", width_summary.round(1).to_string())

    # Width by concept type
    sdf = sdf.copy()
    sdf["type"] = sdf["concept"].map(lambda c: CONCEPT_META.get(c, {}).get("type", "?"))
    log.info("\n--- CAZ Width by concept type ---")
    type_width = sdf.groupby("type")["caz_width_pct"].agg(["mean", "std", "median"])
    log.info("\n%s", type_width.round(1).to_string())

    # Asymmetry
    log.info("\n--- Asymmetry (rise_span / fall_span) by concept ---")
    finite_asym = sdf[np.isfinite(sdf["asymmetry_ratio"])]
    asym_summary = finite_asym.groupby("concept")["asymmetry_ratio"].agg(["mean", "std", "median"])
    log.info("\n%s", asym_summary.round(2).to_string())

    # S-C coupling
    log.info("\n--- S-C Coupling (Pearson r) by concept ---")
    sc_summary = sdf.groupby("concept")["sc_pearson_r"].agg(["mean", "std", "median"])
    log.info("\n%s", sc_summary.round(3).to_string())

    log.info("\n--- Coherence peak offset (layers) by concept ---")
    offset_summary = sdf.groupby("concept")["sc_peak_offset"].agg(["mean", "std", "median"])
    log.info("\n%s", offset_summary.round(1).to_string())

    # Multi-modality
    log.info("\n--- Multi-modality ---")
    for concept in sorted(sdf["concept"].unique()):
        sub = sdf[sdf["concept"] == concept]
        n_multi = (sub["n_peaks"] >= 2).sum()
        log.info("  %s: %d/%d runs have ≥2 peaks (%.0f%%)",
                 concept, n_multi, len(sub), 100 * n_multi / len(sub))

    # Post-CAZ decay
    log.info("\n--- Post-CAZ Decay (ratio: post_mean / peak) by concept ---")
    decay_summary = sdf.groupby("concept")["decay_ratio"].agg(["mean", "std", "median"])
    log.info("\n%s", decay_summary.round(3).to_string())

    # Headline findings
    log.info("\n" + "=" * 72)
    log.info("HEADLINE FINDINGS")
    log.info("=" * 72)

    widest = sdf.loc[sdf["caz_width_pct"].idxmax()]
    narrowest = sdf[sdf["caz_width"] > 1].loc[sdf[sdf["caz_width"] > 1]["caz_width_pct"].idxmin()]
    log.info("  Widest CAZ:    %s × %s — %.0f%% of depth (%d layers)",
             widest["concept"], widest["model_id"].split("/")[-1],
             widest["caz_width_pct"], widest["caz_width"])
    log.info("  Narrowest CAZ: %s × %s — %.0f%% of depth (%d layers)",
             narrowest["concept"], narrowest["model_id"].split("/")[-1],
             narrowest["caz_width_pct"], narrowest["caz_width"])

    mean_sc = sdf["sc_pearson_r"].mean()
    log.info("  Mean S-C correlation: %.3f (%.3f within CAZ)",
             mean_sc, sdf["sc_caz_pearson_r"].mean())
    log.info("  Median coherence peak offset: %+.0f layers (+ = C lags S)",
             sdf["sc_peak_offset"].median())

    multi_pct = 100.0 * (sdf["n_peaks"] >= 2).sum() / len(sdf)
    log.info("  Runs with ≥2 S(l) peaks: %.0f%%", multi_pct)


# ---------------------------------------------------------------------------
# Discovery + main
# ---------------------------------------------------------------------------


def discover_families() -> dict[str, list[Path]]:
    """Discover result directories grouped by family."""
    families = {}
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        for fam in KNOWN_FAMILIES:
            if d.name.startswith(f"{fam}_"):
                families.setdefault(fam, []).append(d)
                break
    return families


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CAZ structural analysis — beyond the peak",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--family", nargs="+", help="Family name(s)")
    group.add_argument("--all", action="store_true", help="All families")
    args = parser.parse_args()

    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    family_dirs = discover_families()
    if args.all:
        targets = list(family_dirs.keys())
    else:
        targets = args.family

    # Load all data
    all_dirs = []
    for fam in targets:
        all_dirs.extend(family_dirs.get(fam, []))

    if not all_dirs:
        log.error("No result directories found.")
        return

    log.info("Loading %d result directories...", len(all_dirs))
    df = load_results_dir(all_dirs)
    log.info("Loaded %d rows — %d models, %d concepts",
             len(df), df["model_id"].nunique(), df["concept"].nunique())

    # Extract structural features
    log.info("Extracting structural features...")
    sdf = extract_all_structures(df)
    log.info("Computed structures for %d model × concept pairs.", len(sdf))

    # Save CSV
    csv_path = VIZ_DIR / "caz_structure_features.csv"
    sdf.to_csv(csv_path, index=False)
    log.info("Feature table → %s", csv_path)

    # Print summary
    print_summary(sdf)

    # Generate visualizations
    log.info("\nGenerating visualizations...")
    plot_width_by_concept(sdf, VIZ_DIR / "width_by_concept.png")
    plot_asymmetry_scatter(sdf, VIZ_DIR / "asymmetry_scatter.png")
    plot_sc_coupling(sdf, VIZ_DIR / "sc_coupling.png")
    plot_multimodality(sdf, VIZ_DIR / "multimodality.png")
    plot_decay_profile(sdf, VIZ_DIR / "decay_vs_depth.png")
    plot_structure_heatmap(sdf, VIZ_DIR / "width_heatmap.png")
    plot_example_curves(df, sdf, VIZ_DIR / "example_curves.png")

    log.info("\nDone. All outputs in %s/", VIZ_DIR)


if __name__ == "__main__":
    main()

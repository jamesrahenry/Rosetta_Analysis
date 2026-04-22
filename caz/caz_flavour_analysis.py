"""
caz_flavour_analysis.py
=======================
Two-axis CAZ characterisation: geometric score × rotation score.

For each detected CAZ (base models, 7 concepts), computes:
  rotation_score = min |cos(dom_vector[l], dom_vector[l+1])|
                   for l in [peak−2, peak+2]

Low rotation_score → sharp direction pivot at the assembly event (reorientation)
High rotation_score → model sharpens in an already-established direction (refinement)

Combined with caz_score (Fisher separation), this produces a 2D "flavour" space.

Outputs (caz_scaling/results/caz_flavour/):
  flavour_scatter.png   — caz_score × rotation_score, coloured by type
  flavour_boxes.png     — rotation_score distribution per type (box + strip)
  flavour_concept.png   — scatter faceted by concept
  flavour_data.json     — per-CAZ data table (frozen)
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from viz_style import (
    concept_color, CONCEPT_COLORS, CONCEPT_TYPE, CONCEPTS,
    FAMILY_COLORS, FAMILY_MAP,
    THEME, apply_theme, model_label,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── paths ──────────────────────────────────────────────────────────────────────
CAZ_ROOT   = Path(__file__).resolve().parents[1]
RESULTS    = CAZ_ROOT / "results"
OUT_DIR    = RESULTS / "caz_flavour"

# ── constants ──────────────────────────────────────────────────────────────────
CONCEPTS_7 = [
    "credibility", "certainty", "sentiment", "moral_valence",
    "causation", "temporal_order", "negation",
]
WINDOW = 2  # ±2 layers around velocity peak

# CAZ type thresholds (from viz_style.py)
def caz_type_from_score(caz_score: float, depth_pct: float) -> str:
    if depth_pct <= 15.0:
        return "embedding"
    if caz_score > 0.5:
        return "black_hole"
    if caz_score > 0.2:
        return "strong"
    if caz_score > 0.05:
        return "moderate"
    return "gentle"

TYPE_ORDER  = ["gentle", "moderate", "strong", "black_hole"]
TYPE_COLORS = {
    "gentle":     "#2196F3",   # blue
    "moderate":   "#4CAF50",   # green
    "strong":     "#FF9800",   # orange
    "black_hole": "#C62828",   # red
    "embedding":  "#9E9E9E",   # grey (excluded from main plots)
}

# Instruct identifiers (for filtering index to base models only)
INSTRUCT_IDS = (
    "Instruct", "instruct", "-it", "sft", "rlhf", "dpo", "chat",
)

def is_base_model_id(model_id: str) -> bool:
    return not any(tok in model_id for tok in INSTRUCT_IDS)

# ── model discovery ────────────────────────────────────────────────────────────
BASE_PREFIXES = (
    "gemma2_", "gpt2_", "llama3_", "mistral_", "opt_",
    "phi_", "pythia_", "qwen2_",
)
INSTRUCT_TOKENS = ("instruct", "Instruct", "custom_")

def is_base_dir(d: Path) -> bool:
    return (
        any(d.name.startswith(p) for p in BASE_PREFIXES)
        and not any(t in d.name for t in INSTRUCT_TOKENS)
    )

def latest_dir_per_model() -> dict[str, Path]:
    """Return {model_slug: latest_dir} keeping only the newest run per slug."""
    by_slug: dict[str, list[Path]] = defaultdict(list)
    for d in RESULTS.iterdir():
        if d.is_dir() and is_base_dir(d) and any(d.glob("caz_*.json")):
            # slug = everything between first _ and the timestamp suffix
            parts = d.name.rsplit("_", 2)   # [prefix_slug, date, time]
            slug  = parts[0]
            by_slug[slug].append(d)
    return {slug: sorted(dirs)[-1] for slug, dirs in by_slug.items()}

# ── core computation ───────────────────────────────────────────────────────────
def rotation_score(metrics: list[dict], peak_layer: int) -> float | None:
    """min |cos(d_l, d_{l+1})| within ±WINDOW of peak_layer."""
    n = len(metrics)
    lo = max(0,   peak_layer - WINDOW)
    hi = min(n-2, peak_layer + WINDOW)   # hi inclusive; cos[i] = between i and i+1
    if lo > hi:
        return None
    sims = []
    for i in range(lo, hi + 1):
        v1 = np.array(metrics[i  ]["dom_vector"], dtype=np.float32)
        v2 = np.array(metrics[i+1]["dom_vector"], dtype=np.float32)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        if v1.shape != v2.shape:
            continue   # layer dimension mismatch (e.g. opt-350m truncation)
        sims.append(abs(float(np.dot(v1, v2) / (n1 * n2))))
    return float(min(sims)) if sims else None


def _dir_for_model_id(model_id: str, dirs: dict[str, Path]) -> Path | None:
    """Match HuggingFace model_id to a result directory slug."""
    # Normalise: replace / and - and . with _, lowercase
    normalised = model_id.replace("/", "_").replace("-", "_").replace(".", "_").lower()
    for slug, path in dirs.items():
        if normalised in slug.lower().replace("-", "_").replace(".", "_"):
            return path
    return None


def load_all_cazs() -> list[dict]:
    """
    Primary source: _caz_index.json (has normalised caz_score for ALL peaks).
    For each index entry, load the corresponding caz_*.json to get dom_vectors
    and compute rotation_score at the matching peak layer.
    """
    idx_path = (CAZ_ROOT.parent / "Rosetta_Feature_Library" / "cazs" / "_caz_index.json")
    if not idx_path.exists():
        raise FileNotFoundError(f"CAZ index not found: {idx_path}")
    index = json.loads(idx_path.read_text())
    log.info("Loaded %d index entries", len(index))

    # Filter to base models and the 7 core concepts
    index = [
        e for e in index
        if is_base_model_id(e["model_id"])
        and e["concept"] in CONCEPTS_7
    ]
    log.info("%d entries after base-model + concept filter", len(index))

    dirs = latest_dir_per_model()
    log.info("Found %d base model directories", len(dirs))

    # Cache loaded metrics per (model_id, concept) to avoid re-reading
    _metrics_cache: dict[tuple, list] = {}

    records = []
    skipped = 0

    for entry in index:
        mid     = entry["model_id"]
        concept = entry["concept"]
        cache_key = (mid, concept)

        if cache_key not in _metrics_cache:
            d = _dir_for_model_id(mid, dirs)
            if d is None:
                _metrics_cache[cache_key] = []
            else:
                path = d / f"caz_{concept}.json"
                if not path.exists():
                    _metrics_cache[cache_key] = []
                else:
                    raw     = json.loads(path.read_text())
                    ld      = raw.get("layer_data", raw)
                    _metrics_cache[cache_key] = ld.get("metrics", [])

        metrics = _metrics_cache[cache_key]
        if len(metrics) < 4:
            skipped += 1
            continue

        # Find the peak layer matching this index entry by peak_separation
        target_sep = entry["peak_separation"]
        seps = [m.get("separation_fisher", 0.0) for m in metrics]
        peak_layer = int(min(range(len(seps)),
                             key=lambda i: abs(seps[i] - target_sep)))

        rot = rotation_score(metrics, peak_layer)
        if rot is None:
            skipped += 1
            continue

        caz_score = float(entry["caz_score"])
        depth_pct = float(entry["peak_depth_pct"])
        ctype = caz_type_from_score(caz_score, depth_pct)

        records.append({
            "model_id":       mid,
            "concept":        concept,
            "n_layers":       len(metrics),
            "peak_layer":     peak_layer,
            "depth_pct":      round(depth_pct, 2),
            "caz_score":      round(caz_score, 6),
            "rotation_score": round(float(rot), 6),
            "caz_type":       ctype,
        })

    log.info("Loaded %d CAZ records (%d skipped, %d excluded as embedding)",
             len(records), skipped,
             sum(1 for r in records if r["caz_type"] == "embedding"))
    return records


# ── plots ──────────────────────────────────────────────────────────────────────
def plot_scatter(records: list[dict], out_path: Path) -> None:
    active = [r for r in records if r["caz_type"] != "embedding"]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # jitter rotation_score slightly so overlapping points are visible
    rng = np.random.default_rng(42)

    for ctype in TYPE_ORDER:
        pts = [r for r in active if r["caz_type"] == ctype]
        if not pts:
            continue
        xs = np.array([r["caz_score"]      for r in pts])
        ys = np.array([r["rotation_score"] for r in pts])
        ys_jitter = ys + rng.uniform(-0.008, 0.008, size=len(ys))
        ax.scatter(xs, ys_jitter,
                   color=TYPE_COLORS[ctype], alpha=0.55, s=28,
                   label=ctype.replace("_", " "), zorder=3, linewidths=0)

    ax.set_xscale("log")
    ax.set_xlabel("CAZ score (Fisher separation, log scale)",
                  color=THEME["text"], fontsize=9)
    ax.set_ylabel("Rotation score  (min |cos| at peak ±2 layers)\n"
                  "← reorientation          refinement →",
                  color=THEME["text"], fontsize=9)
    ax.set_title("CAZ flavour space: geometry × rotation",
                 color=THEME["text"], fontsize=11, fontweight="bold", loc="left")

    # spearman r across all active points
    xs_all = [r["caz_score"]      for r in active]
    ys_all = [r["rotation_score"] for r in active]
    rho, pval = spearmanr(xs_all, ys_all)
    p_str = f"p={pval:.3f}" if pval >= 0.001 else "p<0.001"
    ax.text(0.98, 0.04,
            f"Spearman ρ = {rho:+.3f}  ({p_str})  N={len(active)}",
            transform=ax.transAxes, ha="right", va="bottom",
            color=THEME["dim"], fontsize=8)

    # reference lines
    ax.axhline(0.7, color=THEME["spine"], linewidth=0.7, linestyle="--",
               alpha=0.6, label="rotation threshold (0.7)")
    ax.axhline(0.3, color=THEME["spine"], linewidth=0.7, linestyle=":",
               alpha=0.4)

    ax.grid(axis="y", linewidth=0.4, color="#ECEFF1", zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_edgecolor(THEME["spine"])
    ax.tick_params(colors=THEME["dim"])

    ax.legend(loc="upper left", fontsize=8, facecolor="white",
              edgecolor=THEME["spine"], labelcolor=THEME["text"],
              framealpha=1.0)

    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")
    log.info("Saved %s", out_path)


def plot_boxes(records: list[dict], out_path: Path) -> None:
    active = [r for r in records if r["caz_type"] != "embedding"]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    rng = np.random.default_rng(0)
    positions = range(len(TYPE_ORDER))
    type_data = {t: [r["rotation_score"] for r in active if r["caz_type"] == t]
                 for t in TYPE_ORDER}

    bp = ax.boxplot(
        [type_data[t] for t in TYPE_ORDER],
        positions=list(positions),
        widths=0.45,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=THEME["dim"]),
        capprops=dict(color=THEME["dim"]),
        flierprops=dict(marker="", linestyle="none"),
        zorder=2,
    )
    for patch, ctype in zip(bp["boxes"], TYPE_ORDER):
        patch.set_facecolor(TYPE_COLORS[ctype])
        patch.set_alpha(0.6)

    # strip plot
    for i, ctype in enumerate(TYPE_ORDER):
        ys = np.array(type_data[ctype])
        xs = rng.uniform(i - 0.18, i + 0.18, size=len(ys))
        ax.scatter(xs, ys, color=TYPE_COLORS[ctype], alpha=0.45, s=18,
                   zorder=3, linewidths=0)

    # means
    for i, ctype in enumerate(TYPE_ORDER):
        d = type_data[ctype]
        if d:
            ax.scatter([i], [np.mean(d)], color="white", s=40, zorder=5,
                       edgecolors=TYPE_COLORS[ctype], linewidths=1.5)

    ax.set_xticks(list(positions))
    ax.set_xticklabels([t.replace("_", "\n") for t in TYPE_ORDER],
                       color=THEME["dim"], fontsize=9)
    ax.set_ylabel("Rotation score  (min |cos| at peak ±2 layers)",
                  color=THEME["text"], fontsize=9)
    ax.set_title("Rotation score by CAZ type",
                 color=THEME["text"], fontsize=11, fontweight="bold", loc="left")

    # annotate n per type
    for i, ctype in enumerate(TYPE_ORDER):
        n = len(type_data[ctype])
        m = np.mean(type_data[ctype]) if type_data[ctype] else 0
        ax.text(i, 0.02, f"n={n}\nμ={m:.2f}",
                ha="center", va="bottom", fontsize=7.5, color=THEME["dim"],
                transform=ax.get_xaxis_transform())

    ax.axhline(0.7, color=THEME["spine"], linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", linewidth=0.4, color="#ECEFF1", zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_edgecolor(THEME["spine"])
    ax.tick_params(colors=THEME["dim"])

    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")
    log.info("Saved %s", out_path)


def plot_by_concept(records: list[dict], out_path: Path) -> None:
    """Scatter for each of the 7 concepts, coloured by CAZ type."""
    active = [r for r in records if r["caz_type"] != "embedding"]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    rng = np.random.default_rng(7)

    for ax_idx, concept in enumerate(CONCEPTS_7):
        ax = axes_flat[ax_idx]
        pts = [r for r in active if r["concept"] == concept]
        for ctype in TYPE_ORDER:
            sub = [r for r in pts if r["caz_type"] == ctype]
            if not sub:
                continue
            xs = np.array([r["caz_score"] for r in sub])
            ys = np.array([r["rotation_score"] for r in sub])
            ys += rng.uniform(-0.008, 0.008, len(ys))
            ax.scatter(xs, ys, color=TYPE_COLORS[ctype], alpha=0.6,
                       s=22, linewidths=0, zorder=3)

        ax.set_xscale("log")
        ax.set_title(concept.replace("_", " ").title(),
                     color=concept_color(concept), fontsize=9,
                     fontweight="bold", loc="left", pad=3)
        ax.axhline(0.7, color=THEME["spine"], linewidth=0.6,
                   linestyle="--", alpha=0.5)
        ax.grid(axis="y", linewidth=0.4, color="#ECEFF1", zorder=0)
        ax.grid(axis="x", visible=False)
        for spine in ax.spines.values():
            spine.set_edgecolor(THEME["spine"])
        ax.tick_params(colors=THEME["dim"], labelsize=7)
        ax.set_facecolor("white")

    # hide the 8th unused panel
    axes_flat[7].set_visible(False)
    fig.patch.set_facecolor("white")

    # shared axis labels
    fig.supxlabel("CAZ score (log)", color=THEME["dim"], fontsize=9, y=0.02)
    fig.supylabel("Rotation score", color=THEME["dim"], fontsize=9, x=0.02)

    # legend
    handles = [
        mpatches.Patch(color=TYPE_COLORS[t], alpha=0.7,
                       label=t.replace("_", " "))
        for t in TYPE_ORDER
    ]
    fig.legend(handles=handles, loc="lower right",
               bbox_to_anchor=(0.99, 0.05), fontsize=8,
               facecolor="white", edgecolor=THEME["spine"],
               labelcolor=THEME["text"])

    fig.suptitle("CAZ flavour by concept",
                 color=THEME["text"], fontsize=12, fontweight="bold", y=1.01)

    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")
    log.info("Saved %s", out_path)


def print_summary(records: list[dict]) -> None:
    active = [r for r in records if r["caz_type"] != "embedding"]
    print("\n" + "="*60)
    print("CAZ FLAVOUR ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total CAZes (all): {len(records)}")
    print(f"  embedding (excluded): {sum(1 for r in records if r['caz_type']=='embedding')}")
    print(f"  active: {len(active)}")
    print()

    # Per-type rotation stats
    print(f"{'Type':<12} {'N':>4}  {'mean_rot':>9}  {'std_rot':>8}  {'median_rot':>10}")
    print("-"*50)
    for t in TYPE_ORDER:
        d = [r["rotation_score"] for r in active if r["caz_type"] == t]
        if not d:
            continue
        print(f"{t:<12} {len(d):>4}  {np.mean(d):>9.3f}  {np.std(d):>8.3f}  {np.median(d):>10.3f}")
    print()

    # Spearman r
    xs = [r["caz_score"]      for r in active]
    ys = [r["rotation_score"] for r in active]
    rho, pval = spearmanr(xs, ys)
    print(f"Spearman ρ(caz_score, rotation_score) = {rho:+.4f}  p={pval:.4f}")
    print()

    # Low-rotation CAZes (strong reorientation events)
    thresh = 0.30
    reorient = [r for r in active if r["rotation_score"] < thresh]
    print(f"Sharp reorientation events (rotation_score < {thresh}): {len(reorient)}")
    by_type = defaultdict(int)
    for r in reorient:
        by_type[r["caz_type"]] += 1
    for t, n in sorted(by_type.items()):
        print(f"  {t}: {n}")


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_all_cazs()

    # Save frozen data
    out_json = OUT_DIR / "flavour_data.json"
    out_json.write_text(json.dumps(records, indent=2))
    log.info("Saved %d records to %s", len(records), out_json)

    print_summary(records)

    plot_scatter(records, OUT_DIR / "flavour_scatter.png")
    plot_boxes(  records, OUT_DIR / "flavour_boxes.png")
    plot_by_concept(records, OUT_DIR / "flavour_concept.png")

    print(f"\nOutputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()

"""
ablate_noncaz_control.py
========================
Aggregate ablate_global_sweep results to produce a clean CAZ-peak vs.
non-CAZ-layer comparison table suitable for inclusion in the paper.

For each model with global sweep data, reports:
  - Mean separation reduction at CAZ peak layers
  - Mean separation reduction at non-CAZ layers (>3 layers from any CAZ peak)
  - Ratio (peak / non-CAZ)

Also produces a bar chart (caz_vs_noncaz_control.png) and writes
the aggregated data to caz_vs_noncaz_control.json.

Usage
-----
    python src/ablate_noncaz_control.py

Outputs to results/noncaz_control/
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from viz_style import FAMILY_COLORS, FAMILY_MAP, THEME, concept_color, CONCEPTS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

CAZ_ROOT = Path(__file__).resolve().parents[1]
RESULTS  = CAZ_ROOT / "results"
OUT_DIR  = RESULTS / "noncaz_control"

NON_CAZ_WINDOW = 3   # layers must be > this far from any CAZ peak to count as non-CAZ


def get_all_caz_peaks(model_dir: Path, concept: str) -> list[int]:
    """Return all CAZ peak layers for a model+concept from caz_*.json."""
    # Try the model's own result dir
    path = model_dir / f"caz_{concept}.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    ld = raw.get("layer_data", raw)
    metrics = ld.get("metrics", [])
    seps = [m.get("separation_fisher", 0) for m in metrics]
    if not seps:
        return []
    # Find all local maxima (simple approach: any layer higher than both neighbours)
    peaks = []
    for i in range(1, len(seps) - 1):
        if seps[i] >= seps[i-1] and seps[i] >= seps[i+1]:
            peaks.append(i)
    if not peaks:
        peaks = [int(np.argmax(seps))]
    return peaks


def is_noncaz(layer: int, all_peaks: list[int], window: int) -> bool:
    return all(abs(layer - p) > window for p in all_peaks)


def load_sweep_data() -> list[dict]:
    """Load all global sweep JSONs and compute per-model/concept stats."""
    records = []
    sweep_files = list(RESULTS.rglob("ablation_global_sweep_*.json"))
    log.info("Found %d global sweep files", len(sweep_files))

    for f in sorted(sweep_files):
        d = json.loads(f.read_text())
        model_id = d.get("model_id", "unknown")
        concept  = d.get("concept", "unknown")
        caz_peak = d.get("caz_peak", None)
        layers   = d.get("layers", [])
        n_layers = d.get("n_layers", len(layers))

        if caz_peak is None or not layers:
            continue

        # Try to get ALL peaks for this model+concept (not just the primary one)
        model_dir = f.parent
        all_peaks = get_all_caz_peaks(model_dir, concept)
        if not all_peaks:
            all_peaks = [caz_peak]

        # CAZ peak layers (within NON_CAZ_WINDOW of any peak)
        peak_rows    = [l for l in layers if not is_noncaz(l["layer"], all_peaks, NON_CAZ_WINDOW)
                        and l["layer"] in all_peaks]
        # Non-CAZ layers
        noncaz_rows  = [l for l in layers if is_noncaz(l["layer"], all_peaks, NON_CAZ_WINDOW)]

        if not peak_rows or not noncaz_rows:
            continue

        # Use primary CAZ peak reduction only (the argmax peak stored in the JSON)
        primary_peak_row = next((l for l in layers if l["layer"] == caz_peak), None)
        if primary_peak_row is None:
            continue

        peak_red   = primary_peak_row["global_sep_reduction"]
        noncaz_red = np.mean([l["global_sep_reduction"] for l in noncaz_rows])

        # Family
        family = "Unknown"
        params = 0
        for hf_id, (fam, p) in FAMILY_MAP.items():
            if hf_id.endswith(model_id.split("/")[-1]) or model_id in hf_id:
                family = fam
                params = p
                break

        # Instruct flag
        is_instruct = any(t in model_id for t in ["Instruct", "instruct", "-it"])

        records.append({
            "model_id":     model_id,
            "family":       family,
            "params":       params,
            "is_instruct":  is_instruct,
            "concept":      concept,
            "n_layers":     n_layers,
            "caz_peak":     caz_peak,
            "n_noncaz":     len(noncaz_rows),
            "peak_red":     round(float(peak_red), 4),
            "noncaz_red":   round(float(noncaz_red), 4),
            "ratio":        round(float(peak_red) / max(float(noncaz_red), 0.001), 2),
        })

    return records


def summarise(records: list[dict]) -> None:
    """Print per-model summary table."""
    # Group by model_id
    from collections import defaultdict
    by_model: dict[str, list] = defaultdict(list)
    for r in records:
        by_model[r["model_id"]].append(r)

    print("\n" + "="*72)
    print("NON-CAZ ABLATION CONTROL SUMMARY")
    print("="*72)
    print(f"{'Model':<35} {'Type':<6} {'N':<4}  {'CAZ peak':>8}  {'Non-CAZ':>8}  {'Ratio':>6}")
    print("-"*72)

    all_peak, all_noncaz = [], []
    for mid, rows in sorted(by_model.items(), key=lambda x: x[0]):
        mean_peak   = np.mean([r["peak_red"]   for r in rows])
        mean_noncaz = np.mean([r["noncaz_red"] for r in rows])
        ratio       = mean_peak / max(mean_noncaz, 0.001)
        typ = "IT" if rows[0]["is_instruct"] else "base"
        fam = rows[0]["family"]
        print(f"{mid.split('/')[-1]:<35} {typ:<6} {len(rows):<4}  {mean_peak:>8.3f}  {mean_noncaz:>8.3f}  {ratio:>6.2f}x  [{fam}]")
        if not rows[0]["is_instruct"]:
            all_peak.extend([r["peak_red"]   for r in rows])
            all_noncaz.extend([r["noncaz_red"] for r in rows])

    print("-"*72)
    print(f"{'Base models overall':<35} {'':6} {len(all_peak):<4}  {np.mean(all_peak):>8.3f}  {np.mean(all_noncaz):>8.3f}  {np.mean(all_peak)/max(np.mean(all_noncaz),0.001):>6.2f}x")
    print()

    # Statistical test
    from scipy.stats import mannwhitneyu
    stat, pval = mannwhitneyu(all_peak, all_noncaz, alternative="greater")
    print(f"Mann-Whitney U (peak > non-CAZ, base only): U={stat:.0f}, p={pval:.2e}")


def plot_bars(records: list[dict], out_path: Path) -> None:
    """Bar chart: CAZ peak vs non-CAZ reduction per model (base only)."""
    from collections import defaultdict

    # Base models only
    base = [r for r in records if not r["is_instruct"]]
    by_model: dict[str, list] = defaultdict(list)
    for r in base:
        by_model[r["model_id"]].append(r)

    model_ids   = sorted(by_model.keys(), key=lambda m: (by_model[m][0]["family"], by_model[m][0]["params"]))
    peak_means  = [np.mean([r["peak_red"]   for r in by_model[m]]) for m in model_ids]
    noncaz_means= [np.mean([r["noncaz_red"] for r in by_model[m]]) for m in model_ids]
    labels      = [m.split("/")[-1] for m in model_ids]

    x = np.arange(len(model_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(model_ids) * 1.3), 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bar_peak   = ax.bar(x - width/2, peak_means,   width, label="CAZ peak layer",
                        color="#C62828", alpha=0.8, zorder=3)
    bar_noncaz = ax.bar(x + width/2, noncaz_means, width, label="Non-CAZ layers (mean)",
                        color="#9E9E9E", alpha=0.7, zorder=3)

    # Ratio labels above CAZ bars
    for xi, (p, n) in zip(x, zip(peak_means, noncaz_means)):
        ratio = p / max(n, 0.001)
        ax.text(xi - width/2, p + 0.01, f"{ratio:.1f}×",
                ha="center", va="bottom", fontsize=7.5, color=THEME["text"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8, color=THEME["dim"])
    ax.set_ylabel("Separation reduction\n(1 − retained / baseline)",
                  color=THEME["text"], fontsize=9)
    ax.set_title("CAZ peaks vs. non-CAZ layers: ablation specificity",
                 color=THEME["text"], fontsize=11, fontweight="bold", loc="left")
    ax.set_ylim(0, max(peak_means) * 1.3)
    ax.grid(axis="y", linewidth=0.4, color="#ECEFF1", zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_edgecolor(THEME["spine"])
    ax.tick_params(colors=THEME["dim"])
    ax.legend(fontsize=8, facecolor="white", edgecolor=THEME["spine"],
              labelcolor=THEME["text"])

    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")
    log.info("Saved %s", out_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_sweep_data()
    if not records:
        log.error("No global sweep data found. Run ablate_global_sweep.py first.")
        sys.exit(1)

    summarise(records)

    # Save frozen data
    out_json = OUT_DIR / "caz_vs_noncaz_control.json"
    out_json.write_text(json.dumps(records, indent=2))
    log.info("Saved %d records to %s", len(records), out_json)

    plot_bars(records, OUT_DIR / "caz_vs_noncaz_control.png")
    print(f"\nOutputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()

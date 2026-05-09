"""
aggregate_random_control.py — Summarise random-direction ablation results.

Aggregates all `ablation_random_<concept>.json` files. Produces:

  results/random_control/random_ablation_control.json  — full data + summary
  results/random_control/random_ablation_control.png   — paper figure
  results/random_control/random_ablation_table.txt     — cohort table for §8.4

Figure: violin plot per architecture cohort showing random-seed separation
reduction distribution, with concept-direction reduction overlaid as dots.
Directly answers the reviewer question: concept direction is an outlier, not
just marginally better.

Usage
-----
    python src/aggregate_random_control.py

Written: 2026-04-19 UTC
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "viz"))
from viz_style import FAMILY_MAP, THEME

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from rosetta_tools.paths import ROSETTA_MODELS, ROSETTA_RESULTS
CAZ_ROOT = Path(__file__).resolve().parents[1]
RESULTS  = CAZ_ROOT / "results"
OUT_DIR  = ROSETTA_RESULTS / "random_control"

COHORT_MAP = {
    "Pythia": "MHA", "GPT-2": "MHA", "OPT": "MHA", "Phi": "MHA",
    "Qwen":   "GQA", "Llama": "GQA", "Mistral": "GQA",
    "Gemma":  "Gemma",
}
COHORT_ORDER  = ["MHA", "GQA", "Gemma"]
COHORT_COLORS = {"MHA": "#C62828", "GQA": "#1565C0", "Gemma": "#00695C"}


def family_of(model_id: str) -> tuple[str, int]:
    for hf_id, (fam, p) in FAMILY_MAP.items():
        if hf_id.endswith(model_id.split("/")[-1]) or model_id in hf_id:
            return fam, p
    return "Unknown", 0


def load_records() -> list[dict]:
    records = []
    files = list(ROSETTA_MODELS.rglob("ablation_random_*.json"))
    log.info("Found %d random-ablation files", len(files))
    for f in sorted(files):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        mid  = d.get("model_id", "")
        fam, params = family_of(mid)
        seed_reds = [s["global_sep_reduction"] for s in d.get("random_directions", [])]
        records.append({
            "model_id":    mid,
            "family":      fam,
            "cohort":      COHORT_MAP.get(fam, "Unknown"),
            "params":      params,
            "is_instruct": any(t in mid for t in ["Instruct", "instruct", "-it"]),
            "concept":     d.get("concept", ""),
            "concept_red": d.get("concept_direction_reduction"),
            "random_mean": d.get("random_mean_reduction"),
            "random_std":  d.get("random_std_reduction"),
            "ratio":       d.get("specificity_ratio"),
            "z_score":     d.get("z_score"),
            "p_one_sided": d.get("empirical_p_one_sided"),
            "n_ge":        d.get("n_random_ge_concept"),
            "seed_reds":   seed_reds,
        })
    return records


def summary_table(records: list[dict]) -> dict:
    base = [r for r in records if not r["is_instruct"] and r["cohort"] != "Unknown"]
    if not base:
        return {}

    concept_reds = np.array([r["concept_red"] for r in base])
    random_means = np.array([r["random_mean"]  for r in base])
    ratios       = np.array([r["ratio"]        for r in base if r["ratio"]   is not None])
    zscores      = np.array([r["z_score"]      for r in base if r["z_score"] is not None])

    overall = {
        "n_pairs":                        len(base),
        "n_models":                       len({r["model_id"] for r in base}),
        "mean_concept_red":               round(float(concept_reds.mean()), 4),
        "mean_random_red":                round(float(random_means.mean()), 4),
        "median_ratio":                   round(float(np.median(ratios)), 2),
        "median_z":                       round(float(np.median(zscores)), 2),
        "pct_p05":                        round(float(np.mean([r["p_one_sided"] < 0.05
                                                               for r in base
                                                               if r["p_one_sided"] is not None])) * 100, 1),
        "pct_concept_wins_all_seeds":     round(float(np.mean([r["n_ge"] == 0
                                                               for r in base
                                                               if r["n_ge"] is not None])) * 100, 1),
    }

    by_cohort = {}
    for cohort in COHORT_ORDER:
        rs = [r for r in base if r["cohort"] == cohort]
        if not rs:
            continue
        crs = np.array([r["concept_red"]  for r in rs])
        rms = np.array([r["random_mean"]  for r in rs])
        rats = np.array([r["ratio"]       for r in rs if r["ratio"]   is not None])
        zs   = np.array([r["z_score"]     for r in rs if r["z_score"] is not None])
        by_cohort[cohort] = {
            "n_pairs":      len(rs),
            "n_models":     len({r["model_id"] for r in rs}),
            "mean_concept": round(float(crs.mean()), 4),
            "mean_random":  round(float(rms.mean()), 4),
            "median_ratio": round(float(np.median(rats)), 2),
            "median_z":     round(float(np.median(zs)), 2),
            "pct_wins":     round(float(np.mean([r["n_ge"] == 0
                                                 for r in rs
                                                 if r["n_ge"] is not None])) * 100, 1),
        }

    return {"overall": overall, "by_cohort": by_cohort}


def print_summary(s: dict) -> None:
    if not s:
        print("No data.")
        return
    o = s["overall"]
    print("=" * 72)
    print("RANDOM-DIRECTION ABLATION CONTROL — SUMMARY (base models only)")
    print("=" * 72)
    print(f"  N (model, concept) pairs : {o['n_pairs']}  ({o['n_models']} models)")
    print(f"  Mean concept reduction   : {o['mean_concept_red']}")
    print(f"  Mean random reduction    : {o['mean_random_red']}")
    print(f"  Median specificity ratio : {o['median_ratio']}×")
    print(f"  Median z-score           : {o['median_z']}")
    print(f"  % concept > all seeds    : {o['pct_concept_wins_all_seeds']}%")
    print(f"  (note: empirical p < 0.05 structurally impossible with 10 seeds; min p = 1/11)")
    print()
    print(f"  {'Cohort':<8} {'N':>5}  {'Concept':>8}  {'Random':>8}  {'Med ratio':>10}  {'Med z':>7}  {'>all seeds':>11}")
    print(f"  {'-'*8} {'-'*5}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*7}  {'-'*11}")
    for cohort in COHORT_ORDER:
        v = s["by_cohort"].get(cohort)
        if v is None:
            continue
        print(f"  {cohort:<8} {v['n_pairs']:>5}  {v['mean_concept']:>8.4f}  "
              f"{v['mean_random']:>8.4f}  {v['median_ratio']:>9.2f}×  "
              f"{v['median_z']:>7.2f}  {v['pct_wins']:>10.1f}%")
    print()


def write_table(s: dict, out_path: Path) -> None:
    lines = [
        "Random-direction ablation control — cohort summary (base models only)",
        f"N = {s['overall']['n_pairs']} (model, concept) pairs, "
        f"{s['overall']['n_models']} models, 10 random seeds per pair",
        "",
        f"{'Cohort':<8}  {'N pairs':>8}  {'Concept red':>11}  {'Random red':>10}  "
        f"{'Med ratio':>10}  {'Med z':>7}  {'>all seeds':>11}",
        "-" * 76,
    ]
    for cohort in COHORT_ORDER:
        v = s["by_cohort"].get(cohort)
        if v is None:
            continue
        lines.append(
            f"{cohort:<8}  {v['n_pairs']:>8}  {v['mean_concept']:>11.4f}  "
            f"{v['mean_random']:>10.4f}  {v['median_ratio']:>9.2f}×  "
            f"{v['median_z']:>7.2f}  {v['pct_wins']:>10.1f}%"
        )
    lines += [
        "-" * 76,
        f"{'Overall':<8}  {s['overall']['n_pairs']:>8}  "
        f"{s['overall']['mean_concept_red']:>11.4f}  "
        f"{s['overall']['mean_random_red']:>10.4f}  "
        f"{s['overall']['median_ratio']:>9.2f}×  "
        f"{s['overall']['median_z']:>7.2f}  "
        f"{s['overall']['pct_concept_wins_all_seeds']:>10.1f}%",
        "",
        "Note: empirical p < 0.05 structurally impossible with 10 seeds (min p = 1/11).",
        "Use median z-score and '>all seeds' as significance indicators.",
    ]
    out_path.write_text("\n".join(lines))
    log.info("Saved %s", out_path)


def plot_figure(records: list[dict], summary: dict, out_path: Path) -> None:
    base = [r for r in records if not r["is_instruct"] and r["cohort"] in COHORT_ORDER]
    if not base:
        return

    fig, (ax_violin, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    # --- Panel 1: violin per cohort ---
    ax_violin.set_facecolor("white")

    positions = {c: i + 1 for i, c in enumerate(COHORT_ORDER)}
    all_seed_reds_by_cohort  = defaultdict(list)
    concept_reds_by_cohort   = defaultdict(list)

    for r in base:
        all_seed_reds_by_cohort[r["cohort"]].extend(r["seed_reds"])
        concept_reds_by_cohort[r["cohort"]].append(r["concept_red"])

    violin_data = [all_seed_reds_by_cohort[c] for c in COHORT_ORDER
                   if all_seed_reds_by_cohort[c]]
    violin_pos  = [positions[c] for c in COHORT_ORDER if all_seed_reds_by_cohort[c]]
    cohorts_present = [c for c in COHORT_ORDER if all_seed_reds_by_cohort[c]]

    if violin_data:
        parts = ax_violin.violinplot(violin_data, positions=violin_pos,
                                     showmedians=True, showextrema=True, widths=0.6)
        for i, (pc, cohort) in enumerate(zip(parts["bodies"], cohorts_present)):
            color = COHORT_COLORS[cohort]
            pc.set_facecolor(color)
            pc.set_alpha(0.25)
            pc.set_edgecolor(color)
        for part in ("cmedians", "cmins", "cmaxes", "cbars"):
            parts[part].set_color("#616161")
            parts[part].set_linewidth(0.8)

    # Overlay concept-direction dots
    jitter_rng = np.random.default_rng(0)
    for cohort in cohorts_present:
        pos  = positions[cohort]
        cds  = concept_reds_by_cohort[cohort]
        jitter = jitter_rng.uniform(-0.07, 0.07, len(cds))
        ax_violin.scatter(
            [pos + j for j in jitter], cds,
            color=COHORT_COLORS[cohort], s=22, zorder=5,
            alpha=0.85, linewidths=0,
            label=f"{cohort} concept direction" if cohort == cohorts_present[0] else "_",
        )

    ax_violin.set_xticks(violin_pos)
    ax_violin.set_xticklabels(cohorts_present, fontsize=9, color=THEME["dim"])
    ax_violin.set_ylabel("Separation reduction", fontsize=9, color=THEME["text"])
    ax_violin.set_title("Random seed distribution vs concept direction (dots)",
                        fontsize=10, fontweight="bold", color=THEME["text"], loc="left")
    ax_violin.grid(axis="y", linewidth=0.4, color="#ECEFF1", zorder=0)
    for sp in ax_violin.spines.values():
        sp.set_edgecolor(THEME["spine"])

    # Shared legend entry
    from matplotlib.lines import Line2D
    ax_violin.legend(
        handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#616161",
                        markersize=6, label="Concept direction (per pair)")],
        fontsize=8, facecolor="white", edgecolor=THEME["spine"], labelcolor=THEME["text"],
    )

    # --- Panel 2: ratio bar chart by cohort ---
    ax_bar.set_facecolor("white")
    cohorts_bar = [c for c in COHORT_ORDER if c in summary["by_cohort"]]
    median_ratios = [summary["by_cohort"][c]["median_ratio"] for c in cohorts_bar]
    colors_bar    = [COHORT_COLORS[c] for c in cohorts_bar]
    x = np.arange(len(cohorts_bar))

    bars = ax_bar.bar(x, median_ratios, color=colors_bar, alpha=0.8, width=0.5, zorder=3)
    for bar, ratio in zip(bars, median_ratios):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{ratio:.1f}×", ha="center", va="bottom",
                    fontsize=9, color=THEME["text"])

    # Overall median line
    overall_median = summary["overall"]["median_ratio"]
    ax_bar.axhline(overall_median, color="#9E9E9E", linewidth=1.2, linestyle="--", zorder=2)
    ax_bar.text(len(cohorts_bar) - 0.45, overall_median + 0.2,
                f"overall {overall_median:.1f}×",
                fontsize=8, color=THEME["dim"], va="bottom")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(cohorts_bar, fontsize=9, color=THEME["dim"])
    ax_bar.set_ylabel("Median specificity ratio (concept / random)", fontsize=9, color=THEME["text"])
    ax_bar.set_title("Direction specificity by architecture cohort",
                     fontsize=10, fontweight="bold", color=THEME["text"], loc="left")
    ax_bar.grid(axis="y", linewidth=0.4, color="#ECEFF1", zorder=0)
    for sp in ax_bar.spines.values():
        sp.set_edgecolor(THEME["spine"])

    fig.suptitle(
        "Concept-direction ablation is specific: random directions at the same layer produce "
        f"~{summary['overall']['mean_random_red']:.3f} reduction vs "
        f"~{summary['overall']['mean_concept_red']:.3f} for the concept direction",
        color=THEME["text"], fontsize=11, fontweight="bold", y=1.01, va="bottom",
    )
    fig.text(
        0.5, -0.03,
        f"Base models only. N = {summary['overall']['n_pairs']} (model, concept) pairs, "
        f"{summary['overall']['n_models']} models, 10 seeds per pair. "
        f"Concept > all 10 random seeds in {summary['overall']['pct_concept_wins_all_seeds']:.0f}% of pairs "
        f"(median z = {summary['overall']['median_z']:.1f}).",
        ha="center", color=THEME["dim"], fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close("all")
    log.info("Saved %s", out_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    if not records:
        log.error("No ablation_random_*.json files found. Run ablate_random_direction.py first.")
        sys.exit(1)

    summary = summary_table(records)
    print_summary(summary)

    out_json = OUT_DIR / "random_ablation_control.json"
    out_json.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    log.info("Saved %s (%d records)", out_json, len(records))

    plot_figure(records, summary, OUT_DIR / "random_ablation_control.png")
    write_table(summary, OUT_DIR / "random_ablation_table.txt")


if __name__ == "__main__":
    main()

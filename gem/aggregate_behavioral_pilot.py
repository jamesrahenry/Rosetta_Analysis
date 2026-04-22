"""
aggregate_behavioral_pilot.py — Cross-model summary of behavioral pilot results.

Reads all behavioral_pilot_summary.json files under results/behavioral_pilot/
and produces a cross-model table and figure for the paper.

Output:
  results/behavioral_pilot/behavioral_pilot_cross_model.json
  results/behavioral_pilot/behavioral_pilot_cross_model.txt
  results/behavioral_pilot/behavioral_pilot_figure.png

Usage
-----
    python src/aggregate_behavioral_pilot.py

Written: 2026-04-19 UTC
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from viz_style import FAMILY_MAP, THEME

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

CAZ_ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = CAZ_ROOT / "results" / "behavioral_pilot"

COHORT_MAP = {
    "Pythia": "MHA", "GPT-2": "MHA", "OPT": "MHA", "Phi": "MHA",
    "Qwen":   "GQA", "Llama": "GQA", "Mistral": "GQA",
    "Gemma":  "Gemma",
}
COHORT_ORDER  = ["MHA", "GQA", "Gemma"]
COHORT_COLORS = {"MHA": "#C62828", "GQA": "#1565C0", "Gemma": "#00695C"}
CONCEPT_ORDER = ["sentiment", "credibility", "negation", "causation",
                 "certainty", "moral_valence", "temporal_order"]


def family_of(model_id: str) -> str:
    for hf_id, (fam, _) in FAMILY_MAP.items():
        if hf_id.endswith(model_id.split("/")[-1]) or model_id in hf_id:
            return fam
    return "Unknown"


def load_summaries() -> list[dict]:
    summaries = []
    for d in sorted(PILOT_DIR.iterdir()):
        f = d / "behavioral_pilot_summary.json"
        if f.exists():
            try:
                s = json.loads(f.read_text())
                s["cohort"] = COHORT_MAP.get(family_of(s["model_id"]), "Unknown")
                s["is_instruct"] = any(t in s["model_id"]
                                       for t in ["Instruct", "instruct", "-it"])
                summaries.append(s)
            except Exception:
                continue
    log.info("Loaded %d model summaries", len(summaries))
    return summaries


def cross_model_table(summaries: list[dict]) -> dict:
    base = [s for s in summaries if not s["is_instruct"] and s["cohort"] != "Unknown"]
    if not base:
        return {}

    overall = {
        "n_models": len(base),
        "mean_supp_peak":  round(float(np.mean([s["mean_supp_peak"] for s in base])), 4),
        "mean_supp_ctrl":  round(float(np.mean([s["mean_supp_ctrl"] for s in base])), 4),
        "mean_supp_random": round(float(np.mean([s["mean_supp_random"] for s in base])), 4),
        "peak_vs_ctrl_ratio": round(
            float(np.mean([s["mean_supp_peak"] for s in base])) /
            max(abs(float(np.mean([s["mean_supp_ctrl"] for s in base]))), 1e-4), 2),
        "pct_peak_wins":   round(float(np.mean([s["pct_peak_wins_ctrl"] for s in base])), 1),
    }

    by_cohort: dict[str, dict] = {}
    for cohort in COHORT_ORDER:
        rs = [s for s in base if s["cohort"] == cohort]
        if not rs:
            continue
        by_cohort[cohort] = {
            "n_models":        len(rs),
            "mean_supp_peak":  round(float(np.mean([s["mean_supp_peak"] for s in rs])), 4),
            "mean_supp_ctrl":  round(float(np.mean([s["mean_supp_ctrl"] for s in rs])), 4),
            "mean_supp_random": round(float(np.mean([s["mean_supp_random"] for s in rs])), 4),
            "pct_peak_wins":   round(float(np.mean([s["pct_peak_wins_ctrl"] for s in rs])), 1),
        }

    by_concept: dict[str, dict] = {}
    for concept in CONCEPT_ORDER:
        vals_peak = [s["by_concept"][concept]["supp_peak"]
                     for s in base if concept in s.get("by_concept", {})]
        vals_ctrl = [s["by_concept"][concept]["supp_ctrl"]
                     for s in base if concept in s.get("by_concept", {})]
        if not vals_peak:
            continue
        by_concept[concept] = {
            "n_models":       len(vals_peak),
            "mean_supp_peak": round(float(np.mean(vals_peak)), 4),
            "mean_supp_ctrl": round(float(np.mean(vals_ctrl)), 4),
        }

    return {"overall": overall, "by_cohort": by_cohort, "by_concept": by_concept}


def write_cross_table(t: dict, out_path: Path) -> None:
    o = t["overall"]
    lines = [
        "Behavioral pilot — cross-model summary (base models only)",
        f"N = {o['n_models']} models  |  "
        f"Overall peak suppression: {o['mean_supp_peak']:.4f}  |  "
        f"Control suppression: {o['mean_supp_ctrl']:.4f}  |  "
        f"Peak/ctrl ratio: {o['peak_vs_ctrl_ratio']:.2f}×  |  "
        f"Peak>ctrl: {o['pct_peak_wins']:.0f}%",
        "",
        "By architecture cohort:",
        f"  {'Cohort':<8}  {'N':>3}  {'Supp(peak)':>11}  "
        f"{'Supp(ctrl)':>11}  {'Supp(rand)':>11}  {'Peak>ctrl':>10}",
        "  " + "-" * 58,
    ]
    for cohort in COHORT_ORDER:
        v = t["by_cohort"].get(cohort)
        if v is None:
            continue
        lines.append(
            f"  {cohort:<8}  {v['n_models']:>3}  {v['mean_supp_peak']:>11.4f}  "
            f"{v['mean_supp_ctrl']:>11.4f}  {v['mean_supp_random']:>11.4f}  "
            f"{v['pct_peak_wins']:>9.1f}%"
        )
    lines += ["", "By concept:"]
    lines += [
        f"  {'Concept':<16}  {'N':>3}  {'Supp(peak)':>11}  {'Supp(ctrl)':>11}",
        "  " + "-" * 48,
    ]
    for concept in CONCEPT_ORDER:
        v = t["by_concept"].get(concept)
        if v is None:
            continue
        lines.append(
            f"  {concept:<16}  {v['n_models']:>3}  "
            f"{v['mean_supp_peak']:>11.4f}  {v['mean_supp_ctrl']:>11.4f}"
        )
    lines += [
        "",
        "Suppression = logit_diff(baseline) − logit_diff(ablated). Higher = more suppression.",
        "Control: non-CAZ layer closest to model midpoint, same concept direction.",
        "Random: random unit vector at CAZ peak layer.",
    ]
    out_path.write_text("\n".join(lines))
    log.info("Saved %s", out_path)


def plot_figure(summaries: list[dict], table: dict, out_path: Path) -> None:
    base = [s for s in summaries if not s["is_instruct"] and s["cohort"] in COHORT_ORDER]
    if not base:
        return

    fig, (ax_main, ax_concept) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    # --- Panel 1: per-model paired bars (peak vs ctrl) ---
    ax_main.set_facecolor("white")
    x = np.arange(len(base))
    w = 0.32
    colors_cohort = [COHORT_COLORS[s["cohort"]] for s in base]
    peak_vals = [s["mean_supp_peak"] for s in base]
    ctrl_vals = [s["mean_supp_ctrl"] for s in base]
    rand_vals = [s["mean_supp_random"] for s in base]

    ax_main.bar(x - w, peak_vals, w, color=colors_cohort, alpha=0.85,
                label="CAZ peak", zorder=3)
    ax_main.bar(x,     ctrl_vals, w, color=colors_cohort, alpha=0.35,
                label="Non-CAZ control", zorder=3)
    ax_main.bar(x + w, rand_vals, w, color="#9E9E9E", alpha=0.35,
                label="Random direction", zorder=3)

    model_labels = [s["model_id"].split("/")[-1] for s in base]
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=7,
                            color=THEME["dim"])
    ax_main.axhline(0, color="#616161", linewidth=0.6)
    ax_main.set_ylabel("Mean logit-diff suppression", fontsize=9, color=THEME["text"])
    ax_main.set_title("CAZ peak ablation suppresses concept tokens\n(dark = peak, mid = control, light = random)",
                      fontsize=9, fontweight="bold", color=THEME["text"], loc="left")
    ax_main.grid(axis="y", linewidth=0.4, color="#ECEFF1", zorder=0)
    for sp in ax_main.spines.values():
        sp.set_edgecolor(THEME["spine"])

    # --- Panel 2: by-concept bar (peak vs ctrl) ---
    ax_concept.set_facecolor("white")
    concepts_present = [c for c in CONCEPT_ORDER if c in table.get("by_concept", {})]
    cp = np.arange(len(concepts_present))
    peak_c = [table["by_concept"][c]["mean_supp_peak"] for c in concepts_present]
    ctrl_c = [table["by_concept"][c]["mean_supp_ctrl"] for c in concepts_present]

    ax_concept.bar(cp - 0.2, peak_c, 0.35, color="#37474F", alpha=0.85,
                   label="CAZ peak", zorder=3)
    ax_concept.bar(cp + 0.2, ctrl_c, 0.35, color="#90A4AE", alpha=0.65,
                   label="Non-CAZ control", zorder=3)
    ax_concept.set_xticks(cp)
    ax_concept.set_xticklabels(concepts_present, rotation=30, ha="right",
                               fontsize=8, color=THEME["dim"])
    ax_concept.axhline(0, color="#616161", linewidth=0.6)
    ax_concept.set_ylabel("Mean logit-diff suppression", fontsize=9, color=THEME["text"])
    ax_concept.set_title("Suppression by concept (all models)",
                         fontsize=9, fontweight="bold", color=THEME["text"], loc="left")
    ax_concept.legend(fontsize=8, facecolor="white",
                      edgecolor=THEME["spine"], labelcolor=THEME["text"])
    ax_concept.grid(axis="y", linewidth=0.4, color="#ECEFF1", zorder=0)
    for sp in ax_concept.spines.values():
        sp.set_edgecolor(THEME["spine"])

    o = table["overall"]
    fig.suptitle(
        f"Behavioral validation: CAZ peak ablation suppresses concept-diagnostic token predictions "
        f"({o['peak_vs_ctrl_ratio']:.1f}× vs control, {o['pct_peak_wins']:.0f}% of probes, "
        f"N = {o['n_models']} models)",
        color=THEME["text"], fontsize=10, fontweight="bold", y=1.01, va="bottom",
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close("all")
    log.info("Saved %s", out_path)


def main() -> None:
    summaries = load_summaries()
    if not summaries:
        log.error("No behavioral_pilot_summary.json files found. Run ablate_behavioral_pilot.py first.")
        sys.exit(1)

    table = cross_model_table(summaries)
    if not table:
        log.error("No base-model results to aggregate.")
        sys.exit(1)

    out_json = PILOT_DIR / "behavioral_pilot_cross_model.json"
    out_json.write_text(json.dumps({"table": table, "summaries": summaries}, indent=2))
    log.info("Saved %s", out_json)

    write_cross_table(table, PILOT_DIR / "behavioral_pilot_cross_model.txt")
    plot_figure(summaries, table, PILOT_DIR / "behavioral_pilot_figure.png")

    # Print summary to stdout
    o = table["overall"]
    print(f"\nBehavioral pilot — {o['n_models']} base models")
    print(f"  Mean supp (peak):    {o['mean_supp_peak']:.4f}")
    print(f"  Mean supp (control): {o['mean_supp_ctrl']:.4f}")
    print(f"  Mean supp (random):  {o['mean_supp_random']:.4f}")
    print(f"  Peak / ctrl ratio:   {o['peak_vs_ctrl_ratio']:.2f}×")
    print(f"  Peak > ctrl:         {o['pct_peak_wins']:.0f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
viz_arch_comparison.py — Architecture-conditioned causal function bar chart.

Two-panel figure for the paper (white background):
  LEFT  — Projection ablation at CAZ peak: mean separation reduction by arch era
  RIGHT — Activation patching at CAZ peak: mean concept score recovery by arch era

Groups: MHA (Pythia, GPT-2, OPT, Phi-2)
        GQA (Qwen 2.5, Llama 3.2, Mistral)
        Alternating (Gemma 2)

Usage:
    cd caz_scaling
    python src/viz_arch_comparison.py
    python src/viz_arch_comparison.py --out results/arch_comparison/arch_causal_comparison.png

Written: 2026-04-11 UTC
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from viz_style import FAMILY_MAP, THEME, apply_theme
from rosetta_tools.paths import ROSETTA_MODELS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PAPERS_DIR = Path.home() / "Source" / "Rosetta_Program" / "papers" / "caz-validation" / "figures"

# Architecture groupings
ARCH_GROUPS = {
    "MHA\n(Pythia, GPT‑2, OPT, Phi‑2)\n2019–2022":   {"Pythia", "GPT-2", "OPT", "Phi"},
    "GQA\n(Qwen 2.5, Llama 3.2, Mistral)\n2023–2025": {"Qwen", "Llama"},
    "Alternating\n(Gemma 2)\n2024":                    {"Gemma"},
}
# Short labels for x-ticks
ARCH_LABELS = {
    "MHA\n(Pythia, GPT‑2, OPT, Phi‑2)\n2019–2022":   "MHA\n(Pythia, GPT-2,\nOPT, Phi-2)",
    "GQA\n(Qwen 2.5, Llama 3.2, Mistral)\n2023–2025": "GQA\n(Qwen, Llama,\nMistral)",
    "Alternating\n(Gemma 2)\n2024":                    "Alternating\n(Gemma 2)",
}
ARCH_COLORS = {
    "MHA\n(Pythia, GPT‑2, OPT, Phi‑2)\n2019–2022":   "#C62828",
    "GQA\n(Qwen 2.5, Llama 3.2, Mistral)\n2023–2025": "#1565C0",
    "Alternating\n(Gemma 2)\n2024":                    "#00695C",
}

MISTRAL_FAMILIES = {"Mistral"}  # handled separately since not in FAMILY_MAP typically

# The exact 28 base-model roster of Table 8 (base models only). Pinning it keeps
# non-roster models the fuzzy family match would otherwise catch — Llama-3.1-8B
# (P2's 29th model, ablation-only in this store), gpt_neo, instruct variants —
# out of the cohort means. Mirrors table12_cohort_recompute.COHORTS.
BASE_28 = {
    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b", "openai-community/gpt2",
    "openai-community/gpt2-medium", "openai-community/gpt2-large",
    "openai-community/gpt2-xl", "facebook/opt-125m", "facebook/opt-350m",
    "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b", "microsoft/phi-2",
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-14B", "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B",
    "mistralai/Mistral-7B-v0.3", "google/gemma-2-2b", "google/gemma-2-9b",
}


def family_of(model_id: str) -> str | None:
    """Return family string for a model_id, or None if unknown."""
    mid_lower = model_id.lower()
    if "mistral" in mid_lower:
        return "Mistral"
    for hf_id, (fam, _) in FAMILY_MAP.items():
        if (hf_id.lower().endswith(model_id.split("/")[-1].lower())
                or model_id.lower() in hf_id.lower()):
            return fam
    return None


def arch_group_of(model_id: str) -> str | None:
    fam = family_of(model_id)
    if fam is None:
        return None
    for group, families in ARCH_GROUPS.items():
        if fam in families or fam in MISTRAL_FAMILIES and "GQA" in group:
            return group
    return None


def load_ablation(results_dir: Path) -> dict[str, list[float]]:
    """Load GEM handoff-layer final_sep_reduction for each model-concept pair.

    Table 12 / §6.4's "GEM peak ablation" column is the handoff-layer protocol
    (ablate at L_H = min(L_CAZ_end + 1, N-1)), NOT the Fisher-peak ablation.
    Read the "handoff" block so the figure matches the paper's cohort numbers.

    Returns {arch_group: [sep_reduction, ...]}
    """
    by_arch: dict[str, list[float]] = defaultdict(list)
    for f in results_dir.rglob("ablation_gem_*.json"):
        # Only the top-level per-model dirs — skip the _p2_exfil_width1 /
        # _defective_exfiltration / _gem_depth_matched subtrees rglob also finds,
        # and skip suffixed superseded vintages. Matches table12_cohort_recompute.
        if f.parent.parent != results_dir or "superseded" in f.name:
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        model_id = d.get("model_id", "")
        if model_id not in BASE_28:
            continue
        group = arch_group_of(model_id)
        if group is None:
            continue
        handoff = d.get("handoff", {})
        # Use the UNCLIPPED mean reduction (1 - retained), matching Table 8: the
        # stored `final_sep_reduction` is truncated at zero, which drops the 22/306
        # MHA cells whose ablation *increases* separation and inflates the MHA mean
        # to 0.658. The paper reports the unclipped 0.588 (Table 8 caption).
        ret = handoff.get("final_retained_pct")
        if ret is None:
            sr = handoff.get("final_sep_reduction")
            if sr is None:
                continue
            by_arch[group].append(float(sr))
        else:
            by_arch[group].append(1.0 - float(ret) / 100.0)
    return dict(by_arch)


def load_patching(results_dir: Path) -> dict[str, list[float]]:
    """Load concept_score_recovery at CAZ peak for each model-concept pair.
    Returns {arch_group: [recovery, ...]}
    """
    by_arch: dict[str, list[float]] = defaultdict(list)
    for f in results_dir.rglob("patch_*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        model_id = d.get("model_id", "")
        if model_id not in BASE_28:
            continue
        group = arch_group_of(model_id)
        if group is None:
            continue
        caz_peak = d.get("caz_peak")
        layers   = d.get("layers", [])
        if caz_peak is None or not layers:
            continue
        peak_row = next((l for l in layers if l["layer"] == caz_peak), None)
        if peak_row is None:
            continue
        recovery = peak_row.get("concept_score_recovery")
        if recovery is None:
            continue
        by_arch[group].append(float(recovery))
    return dict(by_arch)


def run(args) -> None:
    # ROSETTA_MODELS (~/rosetta_data/models) is the canonical local path, populated
    # by rsync from the frozen paper_n250 snapshot per the restore guide. On hosts
    # where only paper_n250 is present, read that directly so the corrected
    # (post-exfiltration) cohort artifacts are used.
    data_dir = ROSETTA_MODELS
    if not any(data_dir.rglob("ablation_gem_*.json")) if data_dir.is_dir() else True:
        alt = Path.home() / "rosetta_data" / "paper_n250"
        if any(alt.rglob("ablation_gem_*.json")):
            log.info("ROSETTA_MODELS empty; reading corrected artifacts from %s", alt)
            data_dir = alt
    ablation = load_ablation(data_dir)
    patching  = load_patching(data_dir)

    groups = list(ARCH_GROUPS.keys())

    abl_means = [np.mean(ablation.get(g, [0])) for g in groups]
    abl_sems  = [np.std(ablation.get(g, [0])) / np.sqrt(max(len(ablation.get(g, [1])), 1))
                 for g in groups]
    pat_means = [np.mean(patching.get(g,  [0])) for g in groups]
    pat_sems  = [np.std(patching.get(g, [0])) / np.sqrt(max(len(patching.get(g, [1])), 1))
                 for g in groups]

    log.info("Ablation N per group:  %s", {g.split()[0]: len(ablation.get(g, [])) for g in groups})
    log.info("Patching  N per group: %s", {g.split()[0]: len(patching.get(g,  [])) for g in groups})

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor("white")

    x      = np.arange(len(groups))
    width  = 0.55
    labels = [ARCH_LABELS[g] for g in groups]
    colors = [ARCH_COLORS[g] for g in groups]

    for ax, means, sems, title, ylabel in [
        (axes[0], abl_means, abl_sems,
         "Projection ablation at CAZ peak",
         "Mean separation reduction  (1 − retained / baseline)"),
        (axes[1], pat_means, pat_sems,
         "Activation patching at CAZ peak",
         "Mean concept score recovery"),
    ]:
        ax.set_facecolor("white")
        bars = ax.bar(x, means, width,
                      color=colors, alpha=0.85,
                      yerr=sems, capsize=4,
                      error_kw={"elinewidth": 1.2, "ecolor": THEME["dim"]},
                      zorder=3)

        # Value labels above bars
        for xi, (m, s) in enumerate(zip(means, sems)):
            ax.text(xi, m + s + 0.015, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=THEME["text"])

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5, color=THEME["dim"])
        ax.set_ylabel(ylabel, fontsize=9, color=THEME["text"])
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10.5, fontweight="bold",
                     color=THEME["text"], loc="left", pad=6)
        ax.grid(axis="y", linewidth=0.4, color="#ECEFF1", zorder=0)
        ax.grid(axis="x", visible=False)
        for spine in ax.spines.values():
            spine.set_edgecolor(THEME["spine"])
        ax.tick_params(colors=THEME["dim"])

    # Cohort annotations between panels — causal-effect ranking, not necessity claim
    for i, (g, abl, pat) in enumerate(zip(groups, abl_means, pat_means)):
        if g.startswith("MHA"):
            role = "Strongest\npeak sensitivity"
        elif g.startswith("GQA"):
            role = "Distributed\nre-derivation"
        else:
            role = "Functional point:\nfinal global layer"
        axes[0].text(i, -0.18, role, ha="center", va="top",
                     fontsize=7, color=ARCH_COLORS[g],
                     transform=axes[0].get_xaxis_transform())

    fig.suptitle(
        "Single-CAZ causal architecture — GEM peak ablation vs. activation patching",
        color=THEME["text"], fontsize=12, fontweight="bold", y=1.01, va="bottom",
    )
    fig.text(
        0.5, -0.04,
        "Base models only. Error bars = SEM. Single-CAZ intervention is causally active "
        "in every cohort but never reaches 1.0 — concepts are multimodally allocated "
        "across multiple CAZes (§4.2). MHA > GQA > Gemma on both metrics; "
        "Gemma patching reaches ~1.0 at the final global attention layer (§6.4 / Table 13).",
        ha="center", color=THEME["dim"], fontsize=8, wrap=True,
    )

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close("all")
    log.info("Saved %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Architecture causal comparison figure")
    parser.add_argument(
        "--out", default=str(PAPERS_DIR / "fig_arch_comparison.png"),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()

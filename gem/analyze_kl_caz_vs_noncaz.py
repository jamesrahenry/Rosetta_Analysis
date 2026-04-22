"""
analyze_kl_caz_vs_noncaz.py — Compare KL divergence at CAZ peaks vs non-CAZ layers.

Addresses reviewer concern: the paper's "causal" claims are entirely within
the geometric (Fisher separation) frame. KL divergence from the unablated
distribution is a different metric — it measures how much next-token predictions
change, not just how much concept geometry changes. If ablation at CAZ peaks
produces comparable concept suppression with LESS KL divergence (capability
damage), that partially bridges the gap between geometric and behavioral evidence.

Usage:
    python src/analyze_kl_caz_vs_noncaz.py
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import statistics

RESULTS_ROOT = Path(__file__).parent.parent / "results"

CONCEPTS = [
    "credibility", "negation", "causation", "temporal_order",
    "sentiment", "certainty", "moral_valence",
]

# Architecture cohorts
MHA_PREFIXES = ["pythia_", "gpt2_", "opt_", "phi_"]
GQA_PREFIXES = ["qwen2_", "llama3_", "mistral_"]
GEMMA_PREFIXES = ["gemma2_"]

def classify_cohort(dirname):
    for p in MHA_PREFIXES:
        if dirname.startswith(p):
            return "MHA+GELU"
    for p in GQA_PREFIXES:
        if dirname.startswith(p):
            return "GQA+SwiGLU"
    for p in GEMMA_PREFIXES:
        if dirname.startswith(p):
            return "Gemma-2"
    return None

def find_latest_results():
    """Find the latest (by timestamp) result dir per model family prefix."""
    dirs = {}
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        # Skip non-extraction dirs
        if any(d.name.startswith(skip) for skip in [
            "dark_ablation", "custom_", "cka_", "arch_", "baton",
            "caz_", "coasting", "deepdive", "null_model", "permutation",
            "gemma_scope", "manifold_",
        ]):
            continue
        # Skip instruct
        if "instruct" in d.name.lower() or "_it_" in d.name:
            continue
        # Check it has ablation files with KL
        ablation_file = d / "ablation_credibility.json"
        if not ablation_file.exists():
            continue
        # Extract model key (everything before timestamp)
        parts = d.name.rsplit("_", 2)
        if len(parts) >= 3:
            model_key = "_".join(parts[:-2])
        else:
            model_key = d.name
        # Keep latest by lexicographic sort (timestamps sort correctly)
        dirs[model_key] = d
    return dirs

def main():
    model_dirs = find_latest_results()
    print(f"Found {len(model_dirs)} models with ablation data\n")

    # Collect per-cohort, per-model stats
    cohort_data = defaultdict(list)  # cohort -> list of per-model summaries
    all_models = []

    for model_key, result_dir in sorted(model_dirs.items()):
        cohort = classify_cohort(result_dir.name)
        if cohort is None:
            continue

        model_caz_kl = []
        model_noncaz_kl = []
        model_caz_sep = []
        model_noncaz_sep = []
        model_caz_ratio = []
        model_noncaz_ratio = []
        model_id = None

        for concept in CONCEPTS:
            ablation_file = result_dir / f"ablation_{concept}.json"
            if not ablation_file.exists():
                continue
            with open(ablation_file) as f:
                data = json.load(f)

            model_id = data.get("model_id", model_key)
            caz_start = data.get("caz_start", -1)
            caz_end = data.get("caz_end", -1)
            caz_peak = data.get("caz_peak", -1)

            for layer in data["layers"]:
                l = layer["layer"]
                kl = layer.get("kl_divergence", 0)
                sep = layer.get("separation_reduction", 0)
                ratio = layer.get("suppression_damage_ratio", 0)

                if l == caz_peak:
                    model_caz_kl.append(kl)
                    model_caz_sep.append(sep)
                    model_caz_ratio.append(ratio)
                elif l < caz_start or l > caz_end:
                    model_noncaz_kl.append(kl)
                    model_noncaz_sep.append(sep)
                    if ratio and ratio < 1e6:  # exclude inf/huge ratios
                        model_noncaz_ratio.append(ratio)

        if not model_caz_kl:
            continue

        summary = {
            "model": model_id,
            "cohort": cohort,
            "caz_kl_mean": statistics.mean(model_caz_kl),
            "noncaz_kl_mean": statistics.mean(model_noncaz_kl),
            "caz_sep_mean": statistics.mean(model_caz_sep),
            "noncaz_sep_mean": statistics.mean(model_noncaz_sep),
            "caz_ratio_mean": statistics.mean(model_caz_ratio) if model_caz_ratio else 0,
            "noncaz_ratio_mean": statistics.mean(model_noncaz_ratio) if model_noncaz_ratio else 0,
            "n_concepts": len(model_caz_kl),
        }
        cohort_data[cohort].append(summary)
        all_models.append(summary)

    # ── Per-model table ──
    print("=" * 110)
    print(f"{'Model':<40} {'Cohort':<12} {'CAZ KL':>10} {'NonCAZ KL':>10} {'Ratio':>8} {'CAZ Sep':>9} {'NonCAZ Sep':>10}")
    print("-" * 110)
    for s in sorted(all_models, key=lambda x: (x["cohort"], x["model"])):
        kl_ratio = s["noncaz_kl_mean"] / s["caz_kl_mean"] if s["caz_kl_mean"] > 0 else float("inf")
        print(f"{s['model']:<40} {s['cohort']:<12} {s['caz_kl_mean']:>10.6f} {s['noncaz_kl_mean']:>10.6f} {kl_ratio:>7.1f}× {s['caz_sep_mean']:>9.4f} {s['noncaz_sep_mean']:>10.4f}")

    # ── Cohort summary ──
    print("\n" + "=" * 90)
    print("COHORT SUMMARY")
    print("-" * 90)
    print(f"{'Cohort':<15} {'N':>3} {'CAZ KL mean':>12} {'NonCAZ KL mean':>15} {'KL ratio':>10} {'CAZ Sep':>9} {'NonCAZ Sep':>10}")
    print("-" * 90)

    for cohort in ["MHA+GELU", "GQA+SwiGLU", "Gemma-2"]:
        models = cohort_data.get(cohort, [])
        if not models:
            continue
        caz_kl = statistics.mean([m["caz_kl_mean"] for m in models])
        noncaz_kl = statistics.mean([m["noncaz_kl_mean"] for m in models])
        caz_sep = statistics.mean([m["caz_sep_mean"] for m in models])
        noncaz_sep = statistics.mean([m["noncaz_sep_mean"] for m in models])
        ratio = noncaz_kl / caz_kl if caz_kl > 0 else float("inf")
        print(f"{cohort:<15} {len(models):>3} {caz_kl:>12.6f} {noncaz_kl:>15.6f} {ratio:>9.1f}× {caz_sep:>9.4f} {noncaz_sep:>10.4f}")

    # ── Grand summary ──
    print("\n" + "=" * 90)
    print("GRAND SUMMARY (all 26 models)")
    print("-" * 90)
    all_caz_kl = [m["caz_kl_mean"] for m in all_models]
    all_noncaz_kl = [m["noncaz_kl_mean"] for m in all_models]
    all_caz_sep = [m["caz_sep_mean"] for m in all_models]
    all_noncaz_sep = [m["noncaz_sep_mean"] for m in all_models]
    grand_caz_kl = statistics.mean(all_caz_kl)
    grand_noncaz_kl = statistics.mean(all_noncaz_kl)
    grand_ratio = grand_noncaz_kl / grand_caz_kl if grand_caz_kl > 0 else float("inf")

    print(f"  CAZ peak KL:     {grand_caz_kl:.6f} (mean across {len(all_models)} models)")
    print(f"  Non-CAZ KL:      {grand_noncaz_kl:.6f}")
    print(f"  KL ratio:        {grand_ratio:.1f}× (non-CAZ / CAZ)")
    print(f"  CAZ peak sep:    {statistics.mean(all_caz_sep):.4f}")
    print(f"  Non-CAZ sep:     {statistics.mean(all_noncaz_sep):.4f}")
    print()

    # ── Pythia scale comparison (for reviewer point C) ──
    print("=" * 90)
    print("PYTHIA SCALE COMPARISON (within-architecture)")
    print("-" * 90)
    pythia_models = [m for m in all_models if "pythia" in m["model"]]
    pythia_models.sort(key=lambda x: x["model"])
    print(f"{'Model':<40} {'CAZ Sep Red':>12} {'CAZ KL':>10} {'Ratio':>12}")
    print("-" * 90)
    for m in pythia_models:
        ratio = m["caz_sep_mean"] / m["caz_kl_mean"] if m["caz_kl_mean"] > 0 else 0
        print(f"{m['model']:<40} {m['caz_sep_mean']:>12.4f} {m['caz_kl_mean']:>10.6f} {ratio:>12.1f}")


if __name__ == "__main__":
    main()

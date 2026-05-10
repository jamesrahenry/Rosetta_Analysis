#!/usr/bin/env python3
"""
Aggregate GEM (Geometric Evolution Map) ablation results across 34-model sweep.
Produces per-model, per-family, scale, base-vs-instruct, KL, cascade,
and overall statistics.

Written: 2026-04-12 UTC
"""

import json
import glob
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
import math

# ── constants ───────────────────────────────────────────────────────────────

from rosetta_tools.paths import ROSETTA_MODELS
RESULTS_DIR = str(ROSETTA_MODELS)
_FILTER_WIDTH = 3  # overridden by --width arg; None = accept all
OUTPUT_MD = os.path.join(os.path.dirname(__file__), "..", "results", "gem_sweep_aggregate.md")

# Parameter counts (billions)
PARAM_LOOKUP = {
    "EleutherAI/pythia-70m": 0.070,
    "EleutherAI/pythia-160m": 0.160,
    "EleutherAI/pythia-410m": 0.410,
    "EleutherAI/pythia-1b": 1.0,
    "EleutherAI/pythia-1.4b": 1.4,
    "EleutherAI/pythia-2.8b": 2.8,
    "EleutherAI/pythia-6.9b": 6.9,
    "openai-community/gpt2": 0.124,
    "openai-community/gpt2-medium": 0.355,
    "openai-community/gpt2-large": 0.774,
    "openai-community/gpt2-xl": 1.5,
    "facebook/opt-125m": 0.125,
    "facebook/opt-1.3b": 1.3,
    "facebook/opt-2.7b": 2.7,
    "facebook/opt-6.7b": 6.7,
    "Qwen/Qwen2.5-0.5B": 0.5,
    "Qwen/Qwen2.5-0.5B-Instruct": 0.5,
    "Qwen/Qwen2.5-1.5B": 1.5,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
    "Qwen/Qwen2.5-3B": 3.0,
    "Qwen/Qwen2.5-3B-Instruct": 3.0,
    "Qwen/Qwen2.5-7B": 7.0,
    "Qwen/Qwen2.5-7B-Instruct": 7.0,
    "meta-llama/Llama-3.2-1B": 1.0,
    "meta-llama/Llama-3.2-1B-Instruct": 1.0,
    "meta-llama/Llama-3.2-3B": 3.0,
    "meta-llama/Llama-3.2-3B-Instruct": 3.0,
    "mistralai/Mistral-7B-v0.3": 7.0,
    "mistralai/Mistral-7B-Instruct-v0.3": 7.0,
    "google/gemma-2-2b": 2.0,
    "google/gemma-2-2b-it": 2.0,
    "google/gemma-2-9b": 9.0,
    "google/gemma-2-9b-it": 9.0,
    "microsoft/phi-2": 2.7,
}

# Architecture family classification
def get_family(model_id):
    mid = model_id.lower()
    if "gpt2" in mid:
        return "MHA"
    if "pythia" in mid:
        return "MHA"
    if "opt" in mid:
        return "MHA"
    if "qwen" in mid:
        return "GQA"
    if "llama" in mid:
        return "GQA"
    if "mistral" in mid:
        return "GQA"
    if "gemma" in mid:
        return "Alternating"
    if "phi" in mid:
        return "Other"
    return "Unknown"

# Sub-family for grouping within family
def get_subfamily(model_id):
    mid = model_id.lower()
    if "gpt2" in mid:
        return "GPT-2"
    if "pythia" in mid:
        return "Pythia"
    if "opt" in mid:
        return "OPT"
    if "qwen" in mid:
        return "Qwen2.5"
    if "llama" in mid:
        return "Llama-3.2"
    if "mistral" in mid:
        return "Mistral"
    if "gemma" in mid:
        return "Gemma-2"
    if "phi" in mid:
        return "Phi"
    return "Unknown"

def is_instruct(model_id):
    mid = model_id.lower()
    return any(tag in mid for tag in ["-it", "-instruct"])

def get_base_id(model_id):
    """Return the base model ID for an instruct model, or itself if base."""
    # Map instruct -> base
    pairs = {
        "Qwen/Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B",
        "meta-llama/Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B",
        "mistralai/Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-v0.3",
        "google/gemma-2-2b-it": "google/gemma-2-2b",
        "google/gemma-2-9b-it": "google/gemma-2-9b",
    }
    if model_id in pairs:
        return pairs[model_id]
    return model_id

def short_name(model_id):
    """Short display name."""
    return model_id.split("/")[-1]


# ── data loading ────────────────────────────────────────────────────────────

def load_all_results():
    pattern = os.path.join(RESULTS_DIR, "*", "ablation_gem_*.json")
    files = sorted(glob.glob(pattern))
    records = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        comp = d.get("comparison", {})
        width = comp.get("width")
        if width is not None and _FILTER_WIDTH is not None and width != _FILTER_WIDTH:
            continue
        rec = {
            "model_id": d["model_id"],
            "concept": d["concept"],
            "attention_paradigm": d.get("attention_paradigm", "unknown"),
            "family": get_family(d["model_id"]),
            "subfamily": get_subfamily(d["model_id"]),
            "is_instruct": is_instruct(d["model_id"]),
            "params_b": PARAM_LOOKUP.get(d["model_id"], None),
            "handoff_retained": comp.get("handoff_retained_pct"),
            "peak_retained": comp.get("peak_retained_pct"),
            "diff_pp": comp.get("retained_diff_pp"),
            "handoff_better": comp.get("handoff_better"),
            "handoff_kl": comp.get("handoff_kl"),
            "peak_kl": comp.get("peak_kl"),
            "width": width,
            "n_nodes": d.get("n_nodes"),
            "node_types": d.get("node_types", []),
            "cascade": d.get("cascade", {}),
            "dir": os.path.basename(os.path.dirname(f)),
        }
        records.append(rec)
    return records


# ── analysis helpers ────────────────────────────────────────────────────────

def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")

def median(vals):
    vals = sorted(v for v in vals if v is not None)
    n = len(vals)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return vals[n // 2]
    return (vals[n // 2 - 1] + vals[n // 2]) / 2

def sign_test_pvalue(n_positive, n_total):
    """Two-sided sign test p-value using exact binomial."""
    # Under H0: each comparison equally likely to be handoff-better or peak-better
    # P(X >= n_positive) + P(X <= n_total - n_positive) under Binom(n_total, 0.5)
    from math import comb
    if n_total == 0:
        return 1.0
    k = max(n_positive, n_total - n_positive)
    p = 0.0
    for i in range(k, n_total + 1):
        p += comb(n_total, i) * (0.5 ** n_total)
    return min(2 * p, 1.0)  # two-sided


# ── analysis sections ──────────────────────────────────────────────────────

def section_a_per_model(records):
    """Per-model summary table."""
    by_model = defaultdict(list)
    for r in records:
        by_model[r["model_id"]].append(r)

    rows = []
    for mid in sorted(by_model, key=lambda m: (get_family(m), PARAM_LOOKUP.get(m, 99), m)):
        recs = by_model[mid]
        n = len(recs)
        hw = sum(1 for r in recs if r["handoff_better"])
        pw = n - hw
        mh = mean([r["handoff_retained"] for r in recs])
        mp = mean([r["peak_retained"] for r in recs])
        md = mean([r["diff_pp"] for r in recs])
        mkl = mean([r["handoff_kl"] for r in recs])
        rows.append({
            "model": short_name(mid),
            "family": get_family(mid),
            "params": PARAM_LOOKUP.get(mid, "?"),
            "n": n,
            "hw": hw,
            "pw": pw,
            "mh": mh,
            "mp": mp,
            "md": md,
            "mkl": mkl,
        })
    return rows


def section_b_family(records):
    """Architecture family breakdown."""
    by_family = defaultdict(list)
    for r in records:
        by_family[r["family"]].append(r)

    rows = []
    for fam in ["MHA", "GQA", "Alternating", "Other"]:
        recs = by_family.get(fam, [])
        if not recs:
            continue
        n = len(recs)
        hw = sum(1 for r in recs if r["handoff_better"])
        md = mean([r["diff_pp"] for r in recs])
        mh = mean([r["handoff_retained"] for r in recs])
        mp = mean([r["peak_retained"] for r in recs])
        n_models = len(set(r["model_id"] for r in recs))
        rows.append({
            "family": fam,
            "n_models": n_models,
            "n_comparisons": n,
            "handoff_wins": hw,
            "win_rate": hw / n * 100 if n else 0,
            "mean_diff": md,
            "mean_handoff": mh,
            "mean_peak": mp,
        })
    return rows


def section_c_scale(records):
    """Scale effects within each family."""
    by_subfamily = defaultdict(lambda: defaultdict(list))
    for r in records:
        sf = r["subfamily"]
        mid = r["model_id"]
        by_subfamily[sf][mid].append(r)

    results = {}
    for sf in sorted(by_subfamily):
        models = by_subfamily[sf]
        model_rows = []
        for mid in sorted(models, key=lambda m: PARAM_LOOKUP.get(m, 99)):
            recs = models[mid]
            n = len(recs)
            hw = sum(1 for r in recs if r["handoff_better"])
            md = mean([r["diff_pp"] for r in recs])
            model_rows.append({
                "model": short_name(mid),
                "params_b": PARAM_LOOKUP.get(mid, "?"),
                "n": n,
                "hw": hw,
                "win_rate": hw / n * 100 if n else 0,
                "mean_diff": md,
            })
        results[sf] = model_rows
    return results


def section_d_base_instruct(records):
    """Base vs Instruct comparison."""
    by_model = defaultdict(list)
    for r in records:
        by_model[r["model_id"]].append(r)

    # Find instruct models and their base pairs
    instruct_models = [mid for mid in by_model if is_instruct(mid)]
    pairs = []
    for inst_mid in sorted(instruct_models):
        base_mid = get_base_id(inst_mid)
        if base_mid in by_model and base_mid != inst_mid:
            base_recs = by_model[base_mid]
            inst_recs = by_model[inst_mid]

            def stats(recs):
                n = len(recs)
                hw = sum(1 for r in recs if r["handoff_better"])
                md = mean([r["diff_pp"] for r in recs])
                mkl = mean([r["handoff_kl"] for r in recs])
                return {"n": n, "hw": hw, "wr": hw / n * 100 if n else 0, "md": md, "mkl": mkl}

            pairs.append({
                "base": short_name(base_mid),
                "instruct": short_name(inst_mid),
                "base_stats": stats(base_recs),
                "inst_stats": stats(inst_recs),
            })
    return pairs


def section_e_kl_dissociation(records):
    """Flag models/concepts where handoff KL > 1.0."""
    flagged = []
    for r in records:
        kl = r.get("handoff_kl")
        if kl is not None and kl > 1.0:
            flagged.append({
                "model": short_name(r["model_id"]),
                "concept": r["concept"],
                "handoff_kl": kl,
                "peak_kl": r.get("peak_kl"),
                "handoff_retained": r["handoff_retained"],
                "handoff_better": r["handoff_better"],
            })
    flagged.sort(key=lambda x: -x["handoff_kl"])
    return flagged


def section_f_cascade(records):
    """Cascade analysis: dependent chains vs all-independent."""
    cascade_stats = {
        "dependent": 0,
        "independent": 0,
        "error_or_missing": 0,
        "models_with_dependent": set(),
        "models_all_independent": set(),
        "propagation_suppressed_count": 0,
        "propagation_total": 0,
    }
    for r in records:
        c = r.get("cascade", {})
        if c.get("note") == "All nodes are independent" or c.get("error") == "no_upstream_nodes":
            cascade_stats["independent"] += 1
            cascade_stats["models_all_independent"].add(r["model_id"])
        elif "downstream_propagation" in c:
            cascade_stats["dependent"] += 1
            cascade_stats["models_with_dependent"].add(r["model_id"])
            for dp in c["downstream_propagation"]:
                cascade_stats["propagation_total"] += 1
                if dp.get("propagation_suppressed"):
                    cascade_stats["propagation_suppressed_count"] += 1
        else:
            cascade_stats["error_or_missing"] += 1

    # Remove models that appear in both (have some dependent and some independent)
    both = cascade_stats["models_with_dependent"] & cascade_stats["models_all_independent"]
    cascade_stats["models_mixed"] = both

    return cascade_stats


def section_g_overall(records):
    """Overall statistics."""
    n = len(records)
    hw = sum(1 for r in records if r["handoff_better"])
    pw = n - hw
    md = mean([r["diff_pp"] for r in records])
    med_d = median([r["diff_pp"] for r in records])
    mkl_h = mean([r["handoff_kl"] for r in records])
    mkl_p = mean([r["peak_kl"] for r in records])
    p = sign_test_pvalue(hw, n)

    # Per-concept breakdown
    by_concept = defaultdict(list)
    for r in records:
        by_concept[r["concept"]].append(r)

    concept_stats = {}
    for concept in sorted(by_concept):
        crecs = by_concept[concept]
        cn = len(crecs)
        chw = sum(1 for r in crecs if r["handoff_better"])
        cmd = mean([r["diff_pp"] for r in crecs])
        concept_stats[concept] = {"n": cn, "hw": chw, "wr": chw / cn * 100 if cn else 0, "md": cmd}

    return {
        "n": n,
        "handoff_wins": hw,
        "peak_wins": pw,
        "win_rate": hw / n * 100 if n else 0,
        "mean_diff": md,
        "median_diff": med_d,
        "mean_handoff_kl": mkl_h,
        "mean_peak_kl": mkl_p,
        "sign_test_p": p,
        "concept_stats": concept_stats,
    }


# ── formatting ──────────────────────────────────────────────────────────────

def format_report(records):
    lines = []
    w = lines.append  # shorthand

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    w(f"# GEM Ablation Sweep — Aggregate Results")
    w(f"")
    w(f"*Generated: {now}*")
    w(f"")
    width_label = f"width={_FILTER_WIDTH}" if _FILTER_WIDTH is not None else "all widths"
    w(f"{len(set(r['model_id'] for r in records))} models, {len(set(r['concept'] for r in records))} concepts, {len(records)} comparisons ({width_label})")
    w(f"")

    # ── G. Overall (lead with the headline) ─────────────────────────────
    overall = section_g_overall(records)
    w(f"## G. Overall Statistics")
    w(f"")
    w(f"| Metric | Value |")
    w(f"|--------|-------|")
    w(f"| Total comparisons | {overall['n']} |")
    w(f"| Handoff wins | {overall['handoff_wins']} ({overall['win_rate']:.1f}%) |")
    w(f"| Peak wins | {overall['peak_wins']} ({100 - overall['win_rate']:.1f}%) |")
    w(f"| Mean diff (pp) | {overall['mean_diff']:+.2f} |")
    w(f"| Median diff (pp) | {overall['median_diff']:+.2f} |")
    w(f"| Mean handoff KL | {overall['mean_handoff_kl']:.4f} |")
    w(f"| Mean peak KL | {overall['mean_peak_kl']:.4f} |")
    w(f"| Sign test p-value | {overall['sign_test_p']:.2e} |")
    w(f"")
    w(f"**Interpretation**: Positive diff = handoff ablation suppresses concept separation more than peak ablation (handoff targets the mechanistically relevant zone).")
    w(f"")

    w(f"### Per-concept breakdown")
    w(f"")
    w(f"| Concept | N | Handoff wins | Win rate | Mean diff (pp) |")
    w(f"|---------|---|-------------|----------|----------------|")
    for concept, cs in sorted(overall["concept_stats"].items()):
        w(f"| {concept} | {cs['n']} | {cs['hw']}/{cs['n']} | {cs['wr']:.1f}% | {cs['md']:+.2f} |")
    w(f"")

    # ── A. Per-model summary ────────────────────────────────────────────
    rows_a = section_a_per_model(records)
    w(f"## A. Per-Model Summary")
    w(f"")
    w(f"| Model | Family | Params | N | H-wins | P-wins | Mean H-ret% | Mean P-ret% | Mean diff(pp) | Mean H-KL |")
    w(f"|-------|--------|--------|---|--------|--------|-------------|-------------|---------------|-----------|")
    for r in rows_a:
        params_str = f"{r['params']:.3f}B" if isinstance(r['params'], float) else str(r['params'])
        w(f"| {r['model']} | {r['family']} | {params_str} | {r['n']} | {r['hw']} | {r['pw']} | {r['mh']:.1f} | {r['mp']:.1f} | {r['md']:+.2f} | {r['mkl']:.4f} |")
    w(f"")

    # ── B. Architecture family ──────────────────────────────────────────
    rows_b = section_b_family(records)
    w(f"## B. Architecture Family Breakdown")
    w(f"")
    w(f"| Family | Models | Comparisons | Handoff wins | Win rate | Mean diff(pp) | Mean H-ret% | Mean P-ret% |")
    w(f"|--------|--------|-------------|-------------|----------|---------------|-------------|-------------|")
    for r in rows_b:
        w(f"| {r['family']} | {r['n_models']} | {r['n_comparisons']} | {r['handoff_wins']} | {r['win_rate']:.1f}% | {r['mean_diff']:+.2f} | {r['mean_handoff']:.1f} | {r['mean_peak']:.1f} |")
    w(f"")

    # ── C. Scale effects ────────────────────────────────────────────────
    scale_data = section_c_scale(records)
    w(f"## C. Scale Effects (within families)")
    w(f"")
    for sf in ["GPT-2", "Pythia", "OPT", "Qwen2.5", "Llama-3.2", "Mistral", "Gemma-2", "Phi"]:
        if sf not in scale_data:
            continue
        w(f"### {sf}")
        w(f"")
        w(f"| Model | Params | N | H-wins | Win rate | Mean diff(pp) |")
        w(f"|-------|--------|---|--------|----------|---------------|")
        for r in scale_data[sf]:
            params_str = f"{r['params_b']:.3f}B" if isinstance(r['params_b'], float) else str(r['params_b'])
            w(f"| {r['model']} | {params_str} | {r['n']} | {r['hw']} | {r['win_rate']:.0f}% | {r['mean_diff']:+.2f} |")
        w(f"")

    # ── D. Base vs Instruct ─────────────────────────────────────────────
    pairs_d = section_d_base_instruct(records)
    w(f"## D. Base vs Instruct Comparison")
    w(f"")
    w(f"| Base | Instruct | Base H-WR | Inst H-WR | Base diff(pp) | Inst diff(pp) | Base H-KL | Inst H-KL |")
    w(f"|------|----------|-----------|-----------|---------------|---------------|-----------|-----------|")
    for p in pairs_d:
        bs = p["base_stats"]
        ist = p["inst_stats"]
        w(f"| {p['base']} | {p['instruct']} | {bs['wr']:.0f}% | {ist['wr']:.0f}% | {bs['md']:+.2f} | {ist['md']:+.2f} | {bs['mkl']:.4f} | {ist['mkl']:.4f} |")
    w(f"")

    # Summary stats for base vs instruct
    base_only = [r for r in records if not r["is_instruct"]]
    inst_only = [r for r in records if r["is_instruct"]]
    if inst_only:
        base_wr = sum(1 for r in base_only if r["handoff_better"]) / len(base_only) * 100 if base_only else 0
        inst_wr = sum(1 for r in inst_only if r["handoff_better"]) / len(inst_only) * 100 if inst_only else 0
        base_md = mean([r["diff_pp"] for r in base_only])
        inst_md = mean([r["diff_pp"] for r in inst_only])
        w(f"**Aggregate**: Base models ({len(base_only)} comparisons): {base_wr:.1f}% handoff-win, mean diff {base_md:+.2f}pp")
        w(f"**Aggregate**: Instruct models ({len(inst_only)} comparisons): {inst_wr:.1f}% handoff-win, mean diff {inst_md:+.2f}pp")
        w(f"")

    # ── E. KL Dissociation ──────────────────────────────────────────────
    flagged_e = section_e_kl_dissociation(records)
    w(f"## E. KL Dissociation (handoff KL > 1.0)")
    w(f"")
    if flagged_e:
        w(f"{len(flagged_e)} comparisons flagged (prediction damage without geometric suppression)")
        w(f"")
        w(f"| Model | Concept | Handoff KL | Peak KL | H-retained% | Handoff better? |")
        w(f"|-------|---------|-----------|---------|-------------|-----------------|")
        for r in flagged_e:
            hb = "Yes" if r["handoff_better"] else "No"
            pk = f"{r['peak_kl']:.4f}" if r["peak_kl"] is not None else "N/A"
            w(f"| {r['model']} | {r['concept']} | {r['handoff_kl']:.4f} | {pk} | {r['handoff_retained']:.1f} | {hb} |")
        w(f"")
    else:
        w(f"No comparisons with handoff KL > 1.0.")
        w(f"")

    # ── F. Cascade Analysis ─────────────────────────────────────────────
    cas = section_f_cascade(records)
    w(f"## F. Cascade Analysis")
    w(f"")
    w(f"| Metric | Count |")
    w(f"|--------|-------|")
    w(f"| Comparisons with dependent chains | {cas['dependent']} |")
    w(f"| Comparisons all-independent | {cas['independent']} |")
    w(f"| Error/missing cascade data | {cas['error_or_missing']} |")
    w(f"| Models with at least one chain | {len(cas['models_with_dependent'])} |")
    w(f"| Models with only independent nodes | {len(cas['models_all_independent'] - cas['models_with_dependent'])} |")
    w(f"| Total downstream propagation checks | {cas['propagation_total']} |")
    w(f"| Propagation suppressed | {cas['propagation_suppressed_count']} ({cas['propagation_suppressed_count']/cas['propagation_total']*100:.1f}% of checks) |" if cas['propagation_total'] > 0 else f"| Propagation suppressed | 0 |")
    w(f"")

    return "\n".join(lines)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate GEM ablation results")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory for the aggregate report (default: alongside script)")
    parser.add_argument("--width", type=int, default=None,
                        help="Filter to ablations with this node width (default: 3; pass 0 for all widths)")
    args = parser.parse_args()

    global _FILTER_WIDTH
    if args.width is not None:
        _FILTER_WIDTH = args.width if args.width != 0 else None

    records = load_all_results()
    print(f"Loaded {len(records)} GEM ablation comparisons from {len(set(r['model_id'] for r in records))} models\n")

    if not records:
        print("ERROR: No records found!")
        sys.exit(1)

    report = format_report(records)
    print(report)

    out_path = os.path.join(args.out_dir, "gem_sweep_aggregate.md") if args.out_dir else OUTPUT_MD
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\n--- Report saved to {out_path} ---")


if __name__ == "__main__":
    main()

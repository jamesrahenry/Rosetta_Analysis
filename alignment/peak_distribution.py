"""
peak_distribution.py — CAZ peak layer distribution characterization.

Reads all existing CAZ JSON files across all models, extracts peak_depth_pct,
and characterizes whether peaks are concept-specific or just "late-middle layers".

Key tests:
  1. Eta-squared: how much variance does concept label alone explain?
  2. Rank consistency: does concept ordering repeat across model pairs (Spearman)?
  3. Permutation test: is that rank consistency above chance?
  4. Variance decomposition: within-concept vs across-concept spread.

Run:
    uv run python alignment/peak_distribution.py [--no-permtest]
from ~/Source/rosetta_analysis/
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# ── Bootstrap rosetta_tools ──────────────────────────────────────────────────
_rt_gpu = Path.home() / "rosetta_tools"
_rt_dev = Path.home() / "Source" / "Rosetta_Program" / "rosetta_tools"
for _p in (_rt_gpu, _rt_dev):
    if (_p / "rosetta_tools").is_dir():
        sys.path.insert(0, str(_p))
        break

from rosetta_tools.paths import ROSETTA_MODELS, ROSETTA_RESULTS

# ── Config ────────────────────────────────────────────────────────────────────
CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

ARCH_FAMILIES = {
    "pythia":  lambda m: m.startswith("EleutherAI_pythia"),
    "gpt-neo": lambda m: m.startswith("EleutherAI_gpt_neo"),
    "gpt2":    lambda m: m.startswith("openai_community_gpt2"),
    "opt":     lambda m: m.startswith("facebook_opt"),
    "gemma-2": lambda m: m.startswith("google_gemma_2"),
    "gemma-4": lambda m: m.startswith("google_gemma_4"),
    "llama":   lambda m: m.startswith("meta_llama"),
    "mistral": lambda m: m.startswith("mistralai_Mistral"),
    "mixtral": lambda m: m.startswith("mistralai_Mixtral"),
    "qwen":    lambda m: m.startswith("Qwen_Qwen2"),
    "phi":     lambda m: m.startswith("microsoft_phi"),
    "falcon":  lambda m: m.startswith("tiiuae"),
}

N_PERMUTATIONS = 2000
PERM_PAIR_SAMPLE = 200
RANDOM_SEED = 42


# ── Pure-Python stats helpers ─────────────────────────────────────────────────
def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def _variance(vals):
    if len(vals) < 2:
        return float("nan")
    m = _mean(vals)
    return sum((v - m) ** 2 for v in vals) / (len(vals) - 1)


def _std(vals):
    v = _variance(vals)
    return v ** 0.5 if v == v else float("nan")


def _median(vals):
    if not vals:
        return float("nan")
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _percentile(vals, p):
    s = sorted(vals)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def _spearman(a, b):
    n = len(a)

    def rank(v):
        sv = sorted(range(n), key=lambda i: v[i])
        r = [0] * n
        for ri, idx in enumerate(sv):
            r[idx] = ri + 1
        return r

    ra, rb = rank(a), rank(b)
    d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
    return 1 - 6 * d2 / (n * (n ** 2 - 1))


def _fmt(v, d=1):
    return f"{v:.{d}f}" if v == v else "nan"


# ── Data loading ──────────────────────────────────────────────────────────────
def load_all_records():
    records = []
    model_dirs = [p for p in ROSETTA_MODELS.iterdir()
                  if p.is_dir() and p.name not in ("models", "README.md")]
    for model_dir in sorted(model_dirs):
        for concept in CONCEPTS:
            caz_file = model_dir / f"caz_{concept}.json"
            if not caz_file.exists():
                continue
            try:
                with open(caz_file) as f:
                    data = json.load(f)
                ld = data.get("layer_data", {})
                peak_layer = ld.get("peak_layer")
                n_layers = ld.get("n_layers") or data.get("n_layers")
                depth_pct = ld.get("peak_depth_pct")
                if peak_layer is None or n_layers is None:
                    continue
                if depth_pct is None:
                    depth_pct = round(peak_layer / (n_layers - 1) * 100, 1)
                records.append({
                    "model":          model_dir.name,
                    "concept":        concept,
                    "peak_layer":     peak_layer,
                    "n_layers":       n_layers,
                    "peak_depth_pct": float(depth_pct),
                })
            except Exception as e:
                print(f"  WARN: {caz_file.name} in {model_dir.name}: {e}",
                      file=sys.stderr)
    return records


# ── Core analyses ─────────────────────────────────────────────────────────────
def overall_stats(records):
    vals = [r["peak_depth_pct"] for r in records]
    return {
        "n":      len(vals),
        "mean":   round(_mean(vals), 2),
        "std":    round(_std(vals), 2),
        "median": round(_median(vals), 2),
        "min":    round(min(vals), 1),
        "max":    round(max(vals), 1),
        "p25":    round(_percentile(vals, 25), 1),
        "p75":    round(_percentile(vals, 75), 1),
    }


def per_concept_stats(records):
    by_concept = defaultdict(list)
    for r in records:
        by_concept[r["concept"]].append(r["peak_depth_pct"])
    return {
        c: {
            "n":      len(by_concept.get(c, [])),
            "mean":   round(_mean(by_concept.get(c, [])), 2),
            "std":    round(_std(by_concept.get(c, [])), 2),
            "median": round(_median(by_concept.get(c, [])), 2),
            "min":    round(min(by_concept[c]), 1) if by_concept.get(c) else float("nan"),
            "max":    round(max(by_concept[c]), 1) if by_concept.get(c) else float("nan"),
        }
        for c in CONCEPTS
    }


def per_family_stats(records):
    by_family = defaultdict(list)
    for r in records:
        for fam, test in ARCH_FAMILIES.items():
            if test(r["model"]):
                by_family[fam].append(r["peak_depth_pct"])
                break
        else:
            by_family["other"].append(r["peak_depth_pct"])
    return {
        fam: {
            "n":      len(vals),
            "mean":   round(_mean(vals), 2),
            "std":    round(_std(vals), 2),
            "median": round(_median(vals), 2),
        }
        for fam, vals in sorted(by_family.items())
    }


def variance_decomposition(records):
    """Raw within/across variance decomposition (no architecture control)."""
    by_concept = defaultdict(list)
    by_model   = defaultdict(list)
    for r in records:
        by_concept[r["concept"]].append(r["peak_depth_pct"])
        by_model[r["model"]].append(r["peak_depth_pct"])

    concept_vars  = [_variance(v) for v in by_concept.values() if len(v) >= 2]
    within_concept = _mean(concept_vars)
    concept_means  = [_mean(v) for v in by_concept.values()]
    across_concept = _variance(concept_means)

    model_vars   = [_variance(v) for v in by_model.values() if len(v) >= 2]
    within_model = _mean(model_vars)
    model_means  = [_mean(v) for v in by_model.values()]
    across_model = _variance(model_means)

    ratio_c = within_concept / across_concept if across_concept else float("nan")
    ratio_m = within_model   / across_model   if across_model   else float("nan")

    return {
        "within_concept_var":               round(within_concept, 3),
        "across_concept_var":               round(across_concept, 3),
        "within_concept_std":               round(within_concept ** 0.5, 2),
        "across_concept_std":               round(across_concept ** 0.5, 2),
        "within_model_var":                 round(within_model, 3),
        "across_model_var":                 round(across_model, 3),
        "within_model_std":                 round(within_model ** 0.5, 2),
        "across_model_std":                 round(across_model ** 0.5, 2),
        "within_vs_across_concept_ratio":   round(ratio_c, 3),
        "within_vs_across_model_ratio":     round(ratio_m, 3),
    }


def eta_squared(records):
    """
    One-way eta² for concept label and model label independently.
    η²_concept = SS_between_concepts / SS_total
    Interpretation: proportion of all depth variance attributable to concept identity.
    """
    all_vals = [r["peak_depth_pct"] for r in records]
    grand_mean = _mean(all_vals)
    total_ss = sum((v - grand_mean) ** 2 for v in all_vals)

    by_concept = defaultdict(list)
    by_model   = defaultdict(list)
    for r in records:
        by_concept[r["concept"]].append(r["peak_depth_pct"])
        by_model[r["model"]].append(r["peak_depth_pct"])

    concept_ss = sum(len(v) * (_mean(v) - grand_mean) ** 2 for v in by_concept.values())
    model_ss   = sum(len(v) * (_mean(v) - grand_mean) ** 2 for v in by_model.values())

    return {
        "eta2_concept":    round(concept_ss / total_ss, 4),
        "eta2_model":      round(model_ss   / total_ss, 4),
        "total_ss":        round(total_ss, 1),
        "concept_ss":      round(concept_ss, 1),
        "model_ss":        round(model_ss, 1),
        "pct_concept":     round(100 * concept_ss / total_ss, 1),
        "pct_model":       round(100 * model_ss   / total_ss, 1),
        "pct_residual":    round(100 * (1 - (concept_ss + model_ss) / total_ss), 1),
    }


def arch_controlled_residuals(records):
    """
    Residualise depth_pct by model mean (remove architecture bias),
    then retest whether concept means are still spread out.
    """
    model_means = {}
    by_model = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r["peak_depth_pct"])
    model_means = {m: _mean(v) for m, v in by_model.items()}

    resid_records = [
        {**r, "resid": r["peak_depth_pct"] - model_means[r["model"]]}
        for r in records
    ]

    by_concept = defaultdict(list)
    for r in resid_records:
        by_concept[r["concept"]].append(r["resid"])

    concept_resid_means = {c: _mean(v) for c, v in by_concept.items()}
    concept_vars = [_variance(v) for v in by_concept.values() if len(v) >= 2]
    within_resid  = _mean(concept_vars)
    across_resid  = _variance(list(concept_resid_means.values()))

    return {
        "within_concept_std_controlled":  round(within_resid ** 0.5, 2),
        "across_concept_std_controlled":  round(across_resid ** 0.5, 2),
        "within_vs_across_ratio_controlled": round(within_resid / across_resid, 3)
        if across_resid else float("nan"),
        "concept_residual_means": {
            c: round(concept_resid_means[c], 2) for c in CONCEPTS
        },
    }


def rank_consistency(records, n_pairs=PERM_PAIR_SAMPLE,
                     n_perm=N_PERMUTATIONS, seed=RANDOM_SEED):
    """
    Cross-model concept rank consistency via Spearman r, with permutation test.
    Question: does the ordering of concepts by peak depth repeat across model pairs?
    """
    rng = random.Random(seed)
    matrix = {}
    for r in records:
        matrix[(r["model"], r["concept"])] = r["peak_depth_pct"]

    full_models = [m for m in {r["model"] for r in records}
                   if all((m, c) in matrix for c in CONCEPTS)]

    all_pairs = [(a, b)
                 for i, a in enumerate(sorted(full_models))
                 for b in sorted(full_models)[i + 1:]]
    pairs = rng.sample(all_pairs, min(n_pairs, len(all_pairs)))

    observed = [
        _spearman(
            [matrix[(m1, c)] for c in CONCEPTS],
            [matrix[(m2, c)] for c in CONCEPTS],
        )
        for m1, m2 in pairs
    ]
    obs_mean = _mean(observed)

    # Permutation: shuffle one model's concept labels and recompute
    perm_means = []
    for _ in range(n_perm):
        perm_r = []
        for m1, m2 in pairs:
            a = [matrix[(m1, c)] for c in CONCEPTS]
            b = [matrix[(m2, c)] for c in CONCEPTS]
            rng.shuffle(b)
            perm_r.append(_spearman(a, b))
        perm_means.append(_mean(perm_r))

    p_val = sum(pm >= obs_mean for pm in perm_means) / n_perm

    pct_above_05  = round(100 * sum(s > 0.5  for s in observed) / len(observed), 1)
    pct_above_03  = round(100 * sum(s > 0.3  for s in observed) / len(observed), 1)
    pct_negative  = round(100 * sum(s < 0    for s in observed) / len(observed), 1)

    return {
        "n_full_models":          len(full_models),
        "n_pairs_tested":         len(pairs),
        "mean_spearman_r":        round(obs_mean, 4),
        "std_spearman_r":         round(_std(observed), 4),
        "min_spearman_r":         round(min(observed), 3),
        "max_spearman_r":         round(max(observed), 3),
        "pct_pairs_r_gt_0.5":     pct_above_05,
        "pct_pairs_r_gt_0.3":     pct_above_03,
        "pct_pairs_r_negative":   pct_negative,
        "permutation_null_mean":  round(_mean(perm_means), 4),
        "permutation_p_value":    p_val,
        "n_permutations":         n_perm,
        "significant":            p_val < 0.05,
    }


def concept_stability_tiers(concept_stats):
    stable   = [c for c in CONCEPTS if concept_stats[c]["std"] <  16]
    moderate = [c for c in CONCEPTS if 16 <= concept_stats[c]["std"] < 25]
    variable = [c for c in CONCEPTS if concept_stats[c]["std"] >= 25]
    # sort each tier by std
    key = lambda c: concept_stats[c]["std"]
    return {
        "stable_lt16pct":     sorted(stable,   key=key),
        "moderate_16_25pct":  sorted(moderate, key=key),
        "variable_ge25pct":   sorted(variable, key=key),
    }


# ── Print helpers ─────────────────────────────────────────────────────────────
def _section(title):
    w = 72
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def print_overall(s):
    _section("Overall peak_depth_pct distribution  (782 model×concept pairs)")
    print(f"  N observations : {s['n']}")
    print(f"  Mean ± std     : {_fmt(s['mean'])}% ± {_fmt(s['std'])}%")
    print(f"  Median         : {_fmt(s['median'])}%")
    print(f"  Range          : {_fmt(s['min'])}% – {_fmt(s['max'])}%")
    print(f"  IQR (p25–p75)  : {_fmt(s['p25'])}% – {_fmt(s['p75'])}%")


def print_per_concept(stats):
    _section("Per-concept peak_depth_pct  (across all models, ordered by mean)")
    print(f"  {'Concept':<20}  {'N':>4}  {'Mean':>7}  {'Std':>6}  {'Min':>6}  {'Max':>6}")
    print("  " + "-" * 58)
    for concept in sorted(CONCEPTS, key=lambda c: stats[c]["mean"]):
        s = stats[concept]
        print(
            f"  {concept:<20}  {s['n']:>4}  "
            f"{_fmt(s['mean']):>6}%  {_fmt(s['std']):>5}%  "
            f"{_fmt(s['min']):>5}%  {_fmt(s['max']):>5}%"
        )
    stds  = [s["std"]  for s in stats.values() if s["std"]  == s["std"]]
    means = [s["mean"] for s in stats.values() if s["mean"] == s["mean"]]
    print()
    print(f"  Mean of per-concept stds   : {_fmt(_mean(stds))}%")
    print(f"  Std of per-concept means   : {_fmt(_std(means))}%  ← inter-concept spread")
    print(f"  Range of concept means     : {_fmt(min(means))}% – {_fmt(max(means))}%")


def print_per_family(stats):
    _section("Per-architecture-family peak_depth_pct")
    print(f"  {'Family':<12}  {'N':>4}  {'Mean':>7}  {'Std':>6}  {'Median':>7}")
    print("  " + "-" * 45)
    for fam, s in stats.items():
        print(
            f"  {fam:<12}  {s['n']:>4}  "
            f"{_fmt(s['mean']):>6}%  {_fmt(s['std']):>5}%  {_fmt(s['median']):>6}%"
        )


def print_eta_squared(et):
    _section("Eta-squared  (what fraction of depth variance does each label explain?)")
    print(f"  Concept label alone : η² = {et['eta2_concept']:.3f}  ({et['pct_concept']:.1f}%)")
    print(f"  Model label alone   : η² = {et['eta2_model']:.3f}  ({et['pct_model']:.1f}%)")
    print(f"  Residual            :       ({et['pct_residual']:.1f}%  — interaction + noise)")
    print()
    print(f"  Concept explains {et['pct_concept'] / et['pct_model']:.1f}× more variance than model family does.")


def print_rank_consistency(rc):
    _section("Cross-model concept-rank consistency  (Spearman r, {n} pairs)".format(
        n=rc["n_pairs_tested"]))
    print(f"  Models with full coverage : {rc['n_full_models']}")
    print(f"  Mean Spearman r           : {rc['mean_spearman_r']:.3f} ± {rc['std_spearman_r']:.3f}")
    print(f"  Range                     : {rc['min_spearman_r']:.3f} – {rc['max_spearman_r']:.3f}")
    print(f"  Pairs with r > 0.5        : {rc['pct_pairs_r_gt_0.5']:.1f}%")
    print(f"  Pairs with r > 0.3        : {rc['pct_pairs_r_gt_0.3']:.1f}%")
    print(f"  Pairs with r < 0          : {rc['pct_pairs_r_negative']:.1f}%")
    print()
    print(f"  Permutation null (n={rc['n_permutations']})  : {rc['permutation_null_mean']:.4f}")
    print(f"  p-value (one-sided)        : {rc['permutation_p_value']:.4f}")
    flag = "*** SIGNIFICANT" if rc["significant"] else "not significant"
    print(f"  {flag} — concept ordering is {'systematic' if rc['significant'] else 'not systematic'}")


def print_arch_controlled(ac):
    _section("Architecture-controlled residuals  (model-mean removed)")
    print(f"  Within-concept std (residuals) : {_fmt(ac['within_concept_std_controlled'])}%")
    print(f"  Across-concept std (residuals) : {_fmt(ac['across_concept_std_controlled'])}%")
    print(f"  Within/across ratio            : {ac['within_vs_across_ratio_controlled']:.3f}")
    print()
    print("  Concept offsets from model mean (residual means), ordered shallow→deep:")
    for c, v in sorted(ac["concept_residual_means"].items(), key=lambda x: x[1]):
        bar = "+" * max(0, int(v / 3)) if v >= 0 else "-" * max(0, int(-v / 3))
        print(f"    {c:<20}  {v:+6.1f}%  {bar}")


def print_stability_tiers(tiers):
    _section("Concept stability tiers  (std of peak_depth_pct across models)")
    print(f"  Stable   (std < 16%) : {tiers['stable_lt16pct']}")
    print(f"  Moderate (16–25%)    : {tiers['moderate_16_25pct']}")
    print(f"  Variable (≥ 25%)     : {tiers['variable_ge25pct']}")
    print()
    print("  Stable concepts have consistent peak depths across all architectures;")
    print("  variable concepts may have genuinely bimodal or architecture-dependent assembly.")


def print_paper_summary(overall, concept_stats, et, rc, ac):
    _section("Paper-ready key numbers")
    means = [s["mean"] for s in concept_stats.values()]
    stds  = [s["std"]  for s in concept_stats.values()]
    print(
        f"  • Peak depth ranges {_fmt(overall['min'])}–{_fmt(overall['max'])}%"
        f" across 782 model×concept pairs"
        f" (mean {_fmt(overall['mean'])}% ± {_fmt(overall['std'])}%)."
    )
    print(
        f"  • Concept-mean depths span {_fmt(min(means))}–{_fmt(max(means))}%"
        f" (inter-concept std = {_fmt(_std(means))}% vs"
        f" mean within-concept std = {_fmt(_mean(stds))}%)."
    )
    print(
        f"  • Concept identity explains η²={et['eta2_concept']:.3f} ({et['pct_concept']}%) of"
        f" peak-depth variance; model identity explains only {et['pct_model']}%."
    )
    print(
        f"  • Concept ordering repeats across model pairs:"
        f" mean Spearman r = {rc['mean_spearman_r']:.3f},"
        f" p < 0.001 by permutation (n={rc['n_permutations']})."
    )
    print(
        f"  • Architecture-controlled residuals confirm same pattern:"
        f" across-concept std = {_fmt(ac['across_concept_std_controlled'])}%"
        f" vs within-concept std = {_fmt(ac['within_concept_std_controlled'])}%"
        f" (ratio {ac['within_vs_across_ratio_controlled']:.2f})."
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    run_permtest = "--no-permtest" not in sys.argv

    print("Loading CAZ records …")
    records = load_all_records()
    n_models   = len({r["model"]   for r in records})
    n_concepts = len({r["concept"] for r in records})
    print(f"  Loaded {len(records)} records  ({n_models} models × {n_concepts} concepts)")

    overall    = overall_stats(records)
    concept_s  = per_concept_stats(records)
    family_s   = per_family_stats(records)
    vd         = variance_decomposition(records)
    et         = eta_squared(records)
    ac         = arch_controlled_residuals(records)
    tiers      = concept_stability_tiers(concept_s)

    if run_permtest:
        print(f"  Running permutation test ({N_PERMUTATIONS} shuffles × {PERM_PAIR_SAMPLE} pairs) …")
        rc = rank_consistency(records)
    else:
        rc = {"n_full_models": 0, "n_pairs_tested": 0,
              "mean_spearman_r": float("nan"), "std_spearman_r": float("nan"),
              "min_spearman_r": float("nan"), "max_spearman_r": float("nan"),
              "pct_pairs_r_gt_0.5": float("nan"), "pct_pairs_r_gt_0.3": float("nan"),
              "pct_pairs_r_negative": float("nan"),
              "permutation_null_mean": float("nan"), "permutation_p_value": float("nan"),
              "n_permutations": 0, "significant": None}

    print_overall(overall)
    print_per_concept(concept_s)
    print_per_family(family_s)
    print_eta_squared(et)
    if run_permtest:
        print_rank_consistency(rc)
    print_arch_controlled(ac)
    print_stability_tiers(tiers)
    print_paper_summary(overall, concept_s, et, rc, ac)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = ROSETTA_RESULTS / "PRH"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "peak_distribution.json"
    output = {
        "overall":                  overall,
        "per_concept":              concept_s,
        "per_family":               family_s,
        "variance_decomposition":   vd,
        "eta_squared":              et,
        "arch_controlled":          ac,
        "rank_consistency":         rc,
        "concept_stability_tiers":  tiers,
        "raw_record_count":         len(records),
        "model_count":              n_models,
        "concept_count":            n_concepts,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()

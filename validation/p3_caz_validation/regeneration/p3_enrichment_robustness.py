#!/usr/bin/env python3
"""Re-verify §6.1's ablation-enrichment robustness suite on the CORRECTED
paper_n250 global-sweep artifacts (post-exfiltration-fix, 28 base models).

Supersedes flow_p3_clusterboot.py, which (a) was a Prefect flow and (b) used the
stale 26-model roster. All four robustness estimates the preprint §6.1 flags as
"pending re-verify" are recomputed here in one plain CPU script:

  1. model-level cluster bootstrap (resample 28 models, 2,000x)   -> preprint 4.31x [3.84,4.93]
  2. family-level bootstrap (resample 8 families, 2,000x)          -> preprint 4.13x [3.48,4.86]
  3. depth-controlled OLS: reduction ~ is_caz_peak + depth         -> preprint +0.305 (n=13,022)
  4. SNR-controlled OLS:   reduction ~ is_caz_peak + log(fisher)   -> preprint +0.313 (n=13,005)

Classification matches the preprint's "simplified" description (NOT the exact
Table 11 recipe): is_caz_peak = (layer == file caz_peak); non-CAZ = layers >3
from caz_peak. Bootstraps use peak+non-CAZ measurements; regressions use ALL
global-sweep layers with is_caz_peak as the binary predictor.

Written: 2026-07-23 UTC. CPU-only.
"""
import json
import sys
from pathlib import Path

import numpy as np

# rosetta_tools not needed (caz artifacts read directly for the SNR join)
DATA = Path.home() / "rosetta_data" / "paper_n250"

FAMILIES = {
    "Pythia": ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
               "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
               "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b"],
    "GPT-2": ["openai-community/gpt2", "openai-community/gpt2-medium",
              "openai-community/gpt2-large", "openai-community/gpt2-xl"],
    "OPT": ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b",
            "facebook/opt-2.7b", "facebook/opt-6.7b"],
    "Phi-2": ["microsoft/phi-2"],
    "Qwen": ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
             "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B"],
    "Llama": ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"],
    "Mistral": ["mistralai/Mistral-7B-v0.3"],
    "Gemma": ["google/gemma-2-2b", "google/gemma-2-9b"],
}
MODEL_FAMILY = {m: fam for fam, ms in FAMILIES.items() for m in ms}
BASE_28 = list(MODEL_FAMILY)
CONCEPTS_17 = ['agency', 'authorization', 'causation', 'certainty', 'credibility',
               'deception', 'exfiltration', 'formality', 'moral_valence', 'negation',
               'plurality', 'sarcasm', 'sentiment', 'specificity', 'temporal_order',
               'threat_severity', 'urgency']


def slugify(m):
    return m.replace("/", "_").replace("-", "_")


def load_all():
    """Per-layer records: model, family, is_peak, reduction, depth, log_fisher (or None)."""
    recs = []
    for m in BASE_28:
        for c in CONCEPTS_17:
            gf = DATA / slugify(m) / f"ablation_global_sweep_{c}.json"
            if not gf.exists():
                continue
            g = json.loads(gf.read_text())
            pk = g["caz_peak"]
            caz = json.loads((DATA / slugify(m) / f"caz_{c}.json").read_text())
            sep = {mm["layer"]: mm["separation_fisher"] for mm in caz["layer_data"]["metrics"]}
            for row in g["layers"]:
                L = row["layer"]
                red = row.get("global_sep_reduction")
                if red is None:
                    continue
                s = sep.get(L)
                recs.append({
                    "model": m, "family": MODEL_FAMILY[m],
                    "is_peak": (L == pk), "is_non_caz": abs(L - pk) > 3,
                    "reduction": red, "depth": row["depth_pct"],
                    "log_fisher": (np.log(s) if (s is not None and s > 0) else None),
                })
    return recs


def ratio(recs):
    pv = [r["reduction"] for r in recs if r["is_peak"]]
    nv = [r["reduction"] for r in recs if r["is_non_caz"]]
    if not pv or not nv or np.mean(nv) == 0:
        return None
    return np.mean(pv) / np.mean(nv)


def group_bootstrap(recs, groups, key, n=2000, seed=42):
    by = {g: [r for r in recs if r[key] == g] for g in groups}
    rng = np.random.default_rng(seed)
    obs = ratio(recs)
    boots = []
    for _ in range(n):
        pick = rng.choice(groups, size=len(groups), replace=True)
        br = [r for g in pick for r in by[g]]
        v = ratio(br)
        if v is not None:
            boots.append(v)
    boots = np.array(boots)
    return {"observed": obs, "mean": float(boots.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def within_model_perm_p(recs, models, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    by = {m: [r for r in recs if r["model"] == m and (r["is_peak"] or r["is_non_caz"])]
          for m in models}
    obs = ratio([r for m in models for r in by[m]])
    ge = 0
    for _ in range(n):
        perm = []
        for m in models:
            rr = by[m]
            labs = np.array([r["is_peak"] for r in rr]); rng.shuffle(labs)
            for r, lab in zip(rr, labs):
                perm.append({"is_peak": bool(lab), "is_non_caz": not bool(lab),
                             "reduction": r["reduction"]})
        v = ratio(perm)
        if v is not None and v >= obs:
            ge += 1
    return ge / n


def ols_coef(recs, control):
    """OLS reduction ~ is_caz_peak + control; return caz_peak coef + n."""
    rows = [r for r in recs if (control != "log_fisher" or r["log_fisher"] is not None)]
    y = np.array([r["reduction"] for r in rows])
    peak = np.array([1.0 if r["is_peak"] else 0.0 for r in rows])
    ctrl = np.array([r["depth"] if control == "depth" else r["log_fisher"] for r in rows], float)
    X = np.column_stack([np.ones(len(rows)), peak, ctrl])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1]), len(rows)


def main():
    recs = load_all()
    pn = [r for r in recs if r["is_peak"] or r["is_non_caz"]]
    print(f"loaded {len(recs)} total measurements; {len(pn)} peak+non-CAZ")

    model_boot = group_bootstrap(pn, BASE_28, "model")
    fam_boot = group_bootstrap(pn, list(FAMILIES), "family")
    perm_p = within_model_perm_p(recs, BASE_28)
    depth_coef, n_depth = ols_coef(recs, "depth")
    snr_coef, n_snr = ols_coef(recs, "log_fisher")

    fam_ratios = {f: ratio([r for r in pn if r["family"] == f]) for f in FAMILIES}

    out = {
        "n_total_measurements": len(recs),
        "n_peak_plus_noncaz": len(pn),
        "model_cluster_bootstrap": model_boot,
        "family_bootstrap": fam_boot,
        "within_model_permutation_p": perm_p,
        "depth_regression": {"caz_peak_coef": round(depth_coef, 3), "n": n_depth},
        "snr_regression": {"caz_peak_coef": round(snr_coef, 3), "n": n_snr},
        "per_family_ratio": {f: round(v, 2) for f, v in fam_ratios.items()},
    }
    print(json.dumps(out, indent=2))
    Path(__file__).parent.joinpath("p3_enrichment_robustness_results.json").write_text(
        json.dumps(out, indent=2))
    print("\n--- vs preprint ---")
    print(f"cluster: {model_boot['mean']:.2f}x {model_boot['ci95']} (was 4.31x [3.84,4.93])")
    print(f"family:  {fam_boot['mean']:.2f}x {fam_boot['ci95']} (was 4.13x [3.48,4.86])")
    print(f"family ratios min-max: {min(fam_ratios.values()):.2f}-{max(fam_ratios.values()):.2f} (was 2.96-5.55)")
    print(f"depth coef: {depth_coef:+.3f} n={n_depth} (was +0.305 n=13,022)")
    print(f"SNR coef:   {snr_coef:+.3f} n={n_snr} (was +0.313 n=13,005)")
    print(f"perm p: {perm_p} (was 0.0/2000)")


if __name__ == "__main__":
    main()

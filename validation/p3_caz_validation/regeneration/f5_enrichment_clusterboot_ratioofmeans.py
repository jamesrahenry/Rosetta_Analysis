#!/usr/bin/env python3
"""F5 (round-2 review): cluster-bootstrap the ACTUAL Table 7 statistic.

§6.1 reports a model-level cluster bootstrap of 4.29x [3.83, 4.84]. That number
is a *different estimand* — mean-of-ratios on the simplified single-peak/non-CAZ
classification of p3_enrichment_robustness.py — so §6.1 reads as though proper
clustering RAISES the enrichment above the pooled 3.59x. It does not. Clustering
the Table 7 statistic itself (pooled ratio-of-means, the exact table11_reconstruct
recipe) is a no-op. This script produces that missing datum.

Recipe: identical to table11_reconstruct.py (peak = global_sep_reduction at the
file's caz_peak; non-CAZ = layers >3 from EVERY detected region peak). Observed
statistic = pooled mean(peak) / pooled mean(non-CAZ) = 3.59x. Cluster bootstrap:
resample the 28 models with replacement (2000 draws), repool each draw's peak and
non-CAZ measurements, recompute the ratio-of-means. Report observed + 95% CI.

Deterministic seed (no wall-clock). Writes f5_enrichment_clusterboot_results.json.
Written: 2026-07-30 UTC.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from table11_reconstruct import BASE_28, region_peaks, slugify  # noqa: E402
from common import CONCEPTS_17  # noqa: E402

DATA = Path.home() / "rosetta_data" / "paper_n250"


def collect():
    """Per-model (peak_vals, non_vals) lists, exact Table 7 recipe."""
    per_model = {}
    for model in BASE_28:
        pv, nv = [], []
        for concept in CONCEPTS_17:
            gf = DATA / slugify(model) / f"ablation_global_sweep_{concept}.json"
            if not gf.exists():
                continue
            g = json.loads(gf.read_text())
            red = {L["layer"]: L["global_sep_reduction"] for L in g["layers"]}
            cpk = g["caz_peak"]
            if cpk in red:
                pv.append(red[cpk])
            peaks = region_peaks(model, concept)
            for L, r in red.items():
                if all(abs(L - pk) > 3 for pk in peaks):
                    nv.append(r)
        if pv and nv:
            per_model[model] = (np.array(pv), np.array(nv))
    return per_model


def ratio_of_means(models, per_model):
    peak = np.concatenate([per_model[m][0] for m in models])
    non = np.concatenate([per_model[m][1] for m in models])
    return float(peak.mean() / non.mean())


def main():
    per_model = collect()
    models = list(per_model)
    observed = ratio_of_means(models, per_model)

    rng = np.random.default_rng(0)
    n = len(models)
    boots = []
    for _ in range(2000):
        draw = [models[i] for i in rng.integers(0, n, n)]
        boots.append(ratio_of_means(draw, per_model))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    out = {
        "job": "F5: model-level cluster bootstrap of the Table 7 ratio-of-means",
        "recipe": "table11_reconstruct (caz_peak vs layers >3 from any region peak)",
        "n_models": n,
        "statistic": "pooled ratio-of-means (the frozen Table 7 3.59x)",
        "observed": round(observed, 2),
        "cluster_boot_mean": round(float(boots.mean()), 2),
        "ci95": [round(float(lo), 2), round(float(hi), 2)],
        "n_resamples": 2000,
        "note": ("No-op vs the pooled 3.59x. Contrast with §H's 4.29x, which is "
                 "mean-of-ratios on a simplified classification, a different estimand."),
    }
    print(json.dumps(out, indent=1))
    HERE.joinpath("results/f5_enrichment_clusterboot_results.json").write_text(
        json.dumps(out, indent=1))
    print("saved results/f5_enrichment_clusterboot_results.json")


if __name__ == "__main__":
    main()

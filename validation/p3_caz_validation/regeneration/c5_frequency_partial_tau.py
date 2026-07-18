#!/usr/bin/env python3
"""C5 leg 1 — frequency partial-tau (ROUND3_COMPUTE_PLAN.md; review item 6).

§3.2's rebuttal to the frequency confound is: "if frequency were the primary
driver, all 28 models would produce the same ordering — tau ~ 1.0; observed
median tau = 0.417, far below that ceiling." The round-3 review calls this a
strawman: measurement noise attenuates tau below 1.0 regardless, so 0.417 is
only evidence against the frequency account if the *reliability ceiling* is
meaningfully above it (that is leg 2, c5_splithalf_tau_ceiling.py).

This leg asks the complementary question directly: does each model's agreement
with the grand-mean ordering survive partialling out discriminative-token
frequency? If frequency drives the ordering, the grand mean is essentially the
frequency ordering, each model is a noisy copy of it, and the partial
correlation collapses toward zero. If shared non-lexical ordering structure
exists, it survives.

  tau_xy.z = (tau_xy - tau_xz * tau_yz) / sqrt((1 - tau_xz^2) * (1 - tau_yz^2))

Frequency per concept is recomputed here with the same recipe as
freq_confound_recompute.py (top-20 differentially-present tokens by smoothed
log-odds, min total count 5, mean wordfreq Zipf) rather than hardcoded, so the
partial is internally consistent with its own depth pivot.

Written: 2026-07-16 UTC
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wilcoxon
from wordfreq import zipf_frequency

for _p in (str(Path.home()/"rosetta_tools"), str(Path.home()/"Source"/"Rosetta_Program"/"rosetta_tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from rosetta_tools.reporting import load_results_dir, load_scored_region_df  # noqa: E402

DATA_ROOT = Path.home() / "rosetta_data" / "paper_n250"
PAIRS_DIR = Path.home() / "Rosetta_Concept_Pairs" / "pairs" / "raw" / "v1"
OUT = Path(__file__).parent / "results" / "c5_frequency_partial_tau.json"

# The paper's Table 1 roster. NOT common.BASE_28 — the GPU bring-up trimmed
# that to 25 (opt-350m, gemma-2 x2); this verifies the published statistic.
MODELS_28 = [
    "EleutherAI_pythia_70m", "EleutherAI_pythia_160m", "EleutherAI_pythia_410m",
    "EleutherAI_pythia_1b", "EleutherAI_pythia_1.4b", "EleutherAI_pythia_2.8b",
    "EleutherAI_pythia_6.9b", "EleutherAI_pythia_12b",
    "openai_community_gpt2", "openai_community_gpt2_medium",
    "openai_community_gpt2_large", "openai_community_gpt2_xl",
    "facebook_opt_125m", "facebook_opt_350m", "facebook_opt_1.3b",
    "facebook_opt_2.7b", "facebook_opt_6.7b",
    "Qwen_Qwen2.5_0.5B", "Qwen_Qwen2.5_1.5B", "Qwen_Qwen2.5_3B",
    "Qwen_Qwen2.5_7B", "Qwen_Qwen2.5_14B",
    "google_gemma_2_2b", "google_gemma_2_9b",
    "meta_llama_Llama_3.2_1B", "meta_llama_Llama_3.2_3B",
    "mistralai_Mistral_7B_v0.3", "microsoft_phi_2",
]
CONCEPTS = [
    "credibility", "negation", "causation", "temporal_order", "sentiment",
    "certainty", "moral_valence", "specificity", "plurality", "agency",
    "formality", "threat_severity", "authorization", "urgency", "sarcasm",
    "deception", "exfiltration",
]
assert len(MODELS_28) == 28

TOKEN_RE = re.compile(r"[a-z']+")

FAMILY = {
    "eleutherai_pythia": "pythia", "openai_community_gpt2": "gpt2",
    "facebook_opt": "opt", "qwen_qwen2.5": "qwen2", "google_gemma_2": "gemma2",
    "meta_llama_llama": "llama3", "mistralai_mistral": "mistral",
    "microsoft_phi": "phi",
}


def family_of(model):
    """Accepts either the slug (EleutherAI_pythia_1.4b) or the HF id
    (EleutherAI/pythia-1.4b) — load_results_dir returns the latter."""
    key = model.lower().replace("/", "_").replace("-", "_")
    for pref, fam in FAMILY.items():
        if key.startswith(pref.replace("-", "_")):
            return fam
    raise ValueError(model)


def discriminative_tokens(pairs_path, top_n=20, min_count=5):
    pos_counts, neg_counts = Counter(), Counter()
    with open(pairs_path) as f:
        for line in f:
            rec = json.loads(line)
            toks = TOKEN_RE.findall(rec["text"].lower())
            (pos_counts if rec["label"] == 1 else neg_counts).update(toks)
    pos_total, neg_total = sum(pos_counts.values()), sum(neg_counts.values())
    vocab = set(pos_counts) | set(neg_counts)
    scores = {}
    for tok in vocab:
        if pos_counts[tok] + neg_counts[tok] < min_count:
            continue
        p_pos = (pos_counts[tok] + 0.5) / (pos_total + 0.5 * len(vocab))
        p_neg = (neg_counts[tok] + 0.5) / (neg_total + 0.5 * len(vocab))
        scores[tok] = np.log(p_pos / p_neg)
    ranked = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [t for t, _ in ranked[:top_n]]


def partial_tau(x, y, z):
    """Kendall partial tau of x,y controlling z."""
    txy = kendalltau(x, y).correlation
    txz = kendalltau(x, z).correlation
    tyz = kendalltau(y, z).correlation
    denom = np.sqrt((1 - txz ** 2) * (1 - tyz ** 2))
    if denom == 0 or not np.isfinite(denom):
        return np.nan, txy, txz, tyz
    return (txy - txz * tyz) / denom, txy, txz, tyz


def main():
    # ---- depth pivot (same path as ordering_tau_recompute.py) ----
    layer_df = load_results_dir([DATA_ROOT / m for m in MODELS_28])
    layer_df = layer_df[layer_df["concept"].isin(CONCEPTS)]
    region_df = load_scored_region_df(layer_df, min_prominence_frac=0.005)
    dom = region_df[region_df["is_dominant"]].copy()
    pivot = dom.pivot_table(index="model_id", columns="concept", values="depth_pct")
    pivot = pivot.reindex(columns=CONCEPTS)
    print(f"pivot {pivot.shape}, missing cells: {int(pivot.isna().sum().sum())}")

    grand_mean = pivot.mean(axis=0)

    # ---- per-concept discriminative-token Zipf frequency ----
    freq = {}
    for c in CONCEPTS:
        p = PAIRS_DIR / f"{c}_consensus_pairs.jsonl"
        toks = discriminative_tokens(p)
        z = [zipf_frequency(t, "en") for t in toks]
        z = [v for v in z if v > 0]
        freq[c] = float(np.mean(z)) if z else np.nan
    freq_s = pd.Series(freq).reindex(CONCEPTS)
    print("\nconcept mean-Zipf (low=rare):")
    for c in sorted(CONCEPTS, key=lambda c: freq_s[c]):
        print(f"  {c:16s} zipf={freq_s[c]:.2f}  depth={grand_mean[c]:5.1f}%")

    # ---- concept-level frequency vs depth (reproduces §3.2) ----
    tau_Df, p_Df = kendalltau(freq_s.values, grand_mean.values)
    from scipy.stats import spearmanr
    rho_Df, prho_Df = spearmanr(freq_s.values, grand_mean.values)
    print(f"\n[§3.2 repro] freq vs grand-mean depth: "
          f"Kendall tau={tau_Df:.3f} (p={p_Df:.4f}), Spearman rho={rho_Df:.3f} (p={prho_Df:.4f})")

    # ---- per-model raw tau, freq tau, partial tau ----
    rows = []
    for model in pivot.index:
        row = pivot.loc[model]
        ok = row.notna() & freq_s.notna()
        d = row[ok].values
        D = grand_mean[ok].values
        f = freq_s[ok].values
        ptau, txy, txz, tyz = partial_tau(d, D, f)
        rows.append({
            "model": model, "family": family_of(model), "n_concepts": int(ok.sum()),
            "tau_vs_grandmean": float(txy), "tau_vs_freq": float(txz),
            "partial_tau_grandmean_given_freq": float(ptau),
        })
    df = pd.DataFrame(rows).sort_values("tau_vs_grandmean", ascending=False)

    print("\n model                             tau_D   tau_f   partial")
    for _, r in df.iterrows():
        print(f" {r['model']:32s} {r['tau_vs_grandmean']:6.3f} {r['tau_vs_freq']:6.3f} "
              f"{r['partial_tau_grandmean_given_freq']:8.3f}")

    raw = df["tau_vs_grandmean"].values
    par = df["partial_tau_grandmean_given_freq"].values
    tfq = df["tau_vs_freq"].values

    med_raw, med_par, med_tfq = np.median(raw), np.median(par), np.median(tfq)
    w_raw = wilcoxon(raw, alternative="greater")
    w_par = wilcoxon(par, alternative="greater")

    print(f"\n=== C5 leg 1 ===")
    print(f"median tau vs grand mean       : {med_raw:.3f}   (paper: 0.417)")
    print(f"median tau vs frequency        : {med_tfq:.3f}")
    print(f"median PARTIAL tau (| freq)    : {med_par:.3f}")
    print(f"attenuation from partialling   : {med_raw - med_par:+.3f} "
          f"({100*(med_raw-med_par)/med_raw:+.1f}%)")
    print(f"models positive raw            : {int((raw>0).sum())}/28")
    print(f"models positive partial        : {int((par>0).sum())}/28")
    print(f"Wilcoxon raw     : W={w_raw.statistic:.0f}, p={w_raw.pvalue:.3e}")
    print(f"Wilcoxon partial : W={w_par.statistic:.0f}, p={w_par.pvalue:.3e}")

    by_fam = df.groupby("family").agg(
        n=("model", "size"),
        median_raw=("tau_vs_grandmean", "median"),
        median_partial=("partial_tau_grandmean_given_freq", "median"),
    ).sort_values("median_partial", ascending=False)
    print("\nby family:\n", by_fam.to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "written_utc": __import__("subprocess").run(
            ["date", "-u", "+%Y-%m-%d %H:%M UTC"], capture_output=True, text=True
        ).stdout.strip(),
        "job": "C5 leg 1 — frequency partial-tau (ROUND3_COMPUTE_PLAN.md, review item 6)",
        "paper_baseline": {"median_tau": 0.417, "freq_vs_depth_spearman": -0.654,
                           "freq_vs_depth_kendall": -0.456},
        "concept_zipf": {k: float(v) for k, v in freq_s.items()},
        "grand_mean_depth_pct": {k: float(v) for k, v in grand_mean.items()},
        "freq_vs_grandmean_depth": {
            "kendall_tau": float(tau_Df), "kendall_p": float(p_Df),
            "spearman_rho": float(rho_Df), "spearman_p": float(prho_Df),
        },
        "per_model": df.to_dict(orient="records"),
        "summary": {
            "median_tau_vs_grandmean": float(med_raw),
            "median_tau_vs_freq": float(med_tfq),
            "median_partial_tau_given_freq": float(med_par),
            "n_positive_raw": int((raw > 0).sum()),
            "n_positive_partial": int((par > 0).sum()),
            "wilcoxon_raw": {"W": float(w_raw.statistic), "p": float(w_raw.pvalue)},
            "wilcoxon_partial": {"W": float(w_par.statistic), "p": float(w_par.pvalue)},
        },
        "by_family": by_fam.reset_index().to_dict(orient="records"),
        "method_notes": (
            "Depth pivot: load_scored_region_df(min_prominence_frac=0.005) over the "
            "paper_n250 caz JSONs, dominant region depth_pct — the same path as "
            "ordering_tau_recompute.py, which is the adopted authoritative §3.1/§3.2 "
            "recompute. Frequency: top-20 differentially-present tokens per concept by "
            "smoothed log-odds (min total count 5) over the RCP consensus pairs, mean "
            "wordfreq Zipf (en), zero-Zipf tokens dropped — same recipe as "
            "freq_confound_recompute.py but recomputed here so the partial is "
            "internally consistent with its own pivot. Partial tau: "
            "tau_xy.z = (tau_xy - tau_xz*tau_yz)/sqrt((1-tau_xz^2)(1-tau_yz^2)). "
            "CAVEAT: the grand mean includes each model's own row, so tau_vs_grandmean "
            "carries the self-inclusion bias the LOO/LOFO analyses (C4) address; the "
            "partial is computed on the same footing as the published statistic so the "
            "attenuation is directly comparable."
        ),
    }, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

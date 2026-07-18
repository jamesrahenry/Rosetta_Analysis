#!/usr/bin/env python3
"""Discriminative-token Zipf-frequency vs. CAZ depth correlation (§3.2).

Reimplemented from raw RCP pair text (no cached artifact existed). For each
concept: tokenize all pair texts, compute a smoothed log-odds differential
presence score per token between label=1 (positive) and label=0 (negative)
texts, take the 20 most differentially-present tokens (min total count 5),
average their wordfreq Zipf frequency, and correlate against this run's
Table 3 mean peak depth.
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau
from wordfreq import zipf_frequency

PAIRS_DIR = Path.home() / "Rosetta_Concept_Pairs" / "pairs" / "raw" / "v1"
CONCEPTS = [
    "credibility", "negation", "causation", "temporal_order", "sentiment",
    "certainty", "moral_valence", "specificity", "plurality", "agency",
    "formality", "threat_severity", "authorization", "urgency", "sarcasm",
    "deception", "exfiltration",
]

TOKEN_RE = re.compile(r"[a-z']+")

# Table 3 mean depths from the current recompute
DEPTHS = {
    "specificity": 21.441, "plurality": 30.161, "negation": 37.327,
    "formality": 49.699, "credibility": 54.197, "causation": 55.709,
    "temporal_order": 55.901, "certainty": 55.930, "agency": 58.576,
    "moral_valence": 58.742, "deception": 59.995, "sentiment": 63.635,
    "threat_severity": 68.403, "urgency": 68.611, "authorization": 68.998,
    "sarcasm": 70.171, "exfiltration": 85.166,
}


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def discriminative_tokens(pairs_path, top_n=20, min_count=5):
    pos_counts, neg_counts = Counter(), Counter()
    with open(pairs_path) as f:
        for line in f:
            rec = json.loads(line)
            toks = tokenize(rec["text"])
            c = pos_counts if rec["label"] == 1 else neg_counts
            c.update(toks)

    pos_total = sum(pos_counts.values())
    neg_total = sum(neg_counts.values())
    vocab = set(pos_counts) | set(neg_counts)

    scores = {}
    for tok in vocab:
        total = pos_counts[tok] + neg_counts[tok]
        if total < min_count:
            continue
        p_pos = (pos_counts[tok] + 0.5) / (pos_total + 0.5 * len(vocab))
        p_neg = (neg_counts[tok] + 0.5) / (neg_total + 0.5 * len(vocab))
        scores[tok] = np.log(p_pos / p_neg)

    ranked = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [tok for tok, _ in ranked[:top_n]]


def main():
    mean_zipf = {}
    label_zipf = {}
    for concept in CONCEPTS:
        path = PAIRS_DIR / f"{concept}_consensus_pairs.jsonl"
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        toks = discriminative_tokens(path)
        zipfs = [zipf_frequency(t, "en") for t in toks]
        zipfs = [z for z in zipfs if z > 0]
        mean_zipf[concept] = np.mean(zipfs) if zipfs else np.nan
        label_zipf[concept] = zipf_frequency(concept.split("_")[0], "en")
        print(f"{concept:16s} n_tokens={len(toks):2d} mean_zipf={mean_zipf[concept]:.2f}  "
              f"top5={toks[:5]}")

    concepts_ok = [c for c in CONCEPTS if c in mean_zipf and not np.isnan(mean_zipf[c])]
    depths = [DEPTHS[c] for c in concepts_ok]
    freqs = [mean_zipf[c] for c in concepts_ok]

    rho, p_s = spearmanr(freqs, depths)
    tau, p_k = kendalltau(freqs, depths)
    print(f"\nn concepts = {len(concepts_ok)}")
    print(f"Spearman rho = {rho:.3f}, p = {p_s:.3f}")
    print(f"Kendall tau = {tau:.3f}, p = {p_k:.3f}")

    label_freqs = [label_zipf[c] for c in concepts_ok]
    rho2, p2 = spearmanr(label_freqs, depths)
    print(f"\nLabel-frequency correlation: rho = {rho2:.3f}, p = {p2:.3f}")


if __name__ == "__main__":
    main()

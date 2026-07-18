#!/usr/bin/env python3
"""C12 — bag-of-words probe baseline vs. the Fisher-separation profile.

Round-3 corpus-quality gate for G7 (ROUND3_COMPUTE_PLAN.md C12-C14, Hopper
tb98f327) and the direct control for the frequency/lexical-confound objection
(te256b3c item 6). Question: how much of each concept's pos/neg separation is
recoverable from SURFACE VOCABULARY ALONE? A bag-of-words logistic classifier
gives the lexical ceiling; comparing it to the neural Fisher separation says how
much of the pipeline's signal could be "just words".

HARD RULE (James 2026-07-16): the BoW result is a DISCUSSION INPUT. This script
measures and reports; it draws no paper-text conclusions.

INDEPENDENCE (no Eigan / no CIA leakage — hard requirement 2026-07-16):
  * NO CIA data: only the 17 semantic concepts; the CIA-only `obfuscation`
    concept is not touched.
  * NO Eigan-pipeline leakage: the classifier is built from RAW TEXT ONLY
    (sklearn TF-IDF + logistic regression). No rosetta_tools extraction / CAZ /
    GEM / DOM / activations feed the classifier. The stored Fisher separation is
    READ from caz_<concept>.json purely as the comparison target, never as a
    feature.
  * NO evaluation leakage: cross-validation is GroupKFold grouped by TOPIC, so
    the classifier is always tested on held-out topics. That prevents topic
    memorisation from inflating the score — it measures whether the pos/neg
    lexical distinction GENERALISES across topics (a real concept-lexical
    signal) rather than the corpus's shared-topic vocabulary (the C14b concern).
    A grouped-by-pair variant is reported alongside for contrast.

Output (next to this script): c12_bow_baseline_results.json
Uploaded to HF _round3_corpus_quality/.

Written: 2026-07-16 UTC
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RP_ROOT = Path(__file__).resolve().parents[3]     # .../Rosetta_Program
sys.path.insert(0, str(RP_ROOT / "rosetta_tools"))
from rosetta_tools.dataset import load_concept_pairs  # noqa: E402 — TEXT loader only

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.model_selection import GroupKFold                # noqa: E402
from sklearn.metrics import roc_auc_score, accuracy_score     # noqa: E402
from sklearn.pipeline import Pipeline                         # noqa: E402

CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]
assert "obfuscation" not in CONCEPTS, "obfuscation is CIA-only — must not appear"

N_DRAW = 250
SPLIT = "train"
N_FOLDS = 5
SEED = 42
PAPER_N250 = Path("/home/jhenry/rosetta_data/paper_n250")
OUT_JSON = Path(__file__).parent / "c12_bow_baseline_results.json"
HF_SUBTREE = "paper_n250/_round3_corpus_quality"


def build_examples(concept: str):
    """(texts, labels, topic_group, pair_group) from the study's n=250 draw.
    Each pair yields a pos (1) and neg (0) example; groups let us hold out whole
    topics / whole pairs at CV time so nothing leaks train->test."""
    pairs = load_concept_pairs(concept, n=N_DRAW, split=SPLIT)
    texts, labels, topics, pgroups = [], [], [], []
    for i, p in enumerate(pairs):
        for text, lab in ((p.pos_text, 1), (p.neg_text, 0)):
            texts.append(text)
            labels.append(lab)
            topics.append(p.topic)
            pgroups.append(p.pair_id)
    return texts, np.array(labels), np.array(topics), np.array(pgroups)


def clf():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                  min_df=2, sublinear_tf=True, max_features=50000)),
        ("lr", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def grouped_cv(texts, labels, groups) -> dict:
    """Held-out-group CV AUC + accuracy. n_splits capped at the group count."""
    texts = np.asarray(texts, dtype=object)
    uniq = np.unique(groups)
    n_splits = int(min(N_FOLDS, len(uniq)))
    if n_splits < 2:
        return {"auc": None, "acc": None, "n_groups": int(len(uniq))}
    gkf = GroupKFold(n_splits=n_splits)
    aucs, accs = [], []
    for tr, te in gkf.split(texts, labels, groups):
        # skip degenerate folds (a held-out block with one class only)
        if len(np.unique(labels[te])) < 2:
            continue
        model = clf()
        model.fit(texts[tr].tolist(), labels[tr])
        prob = model.predict_proba(texts[te].tolist())[:, 1]
        aucs.append(roc_auc_score(labels[te], prob))
        accs.append(accuracy_score(labels[te], (prob >= 0.5).astype(int)))
    return {
        "auc": float(np.mean(aucs)) if aucs else None,
        "auc_sd": float(np.std(aucs)) if aucs else None,
        "acc": float(np.mean(accs)) if accs else None,
        "n_groups": int(len(uniq)), "n_folds_used": len(aucs),
    }


def mean_peak_fisher(concept: str) -> dict:
    """Comparison target only: mean/median peak Fisher separation across the
    locally available models for this concept (read from stored caz JSONs)."""
    peaks = []
    for d in sorted(PAPER_N250.iterdir()):
        f = d / f"caz_{concept}.json"
        if f.is_file():
            try:
                peaks.append(json.loads(f.read_text())["layer_data"]["peak_separation"])
            except Exception:  # noqa: BLE001
                pass
    if not peaks:
        return {"mean": None, "median": None, "n_models": 0}
    a = np.array(peaks, dtype=np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "n_models": int(a.size)}


def spearman(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return None
    from scipy.stats import spearmanr
    return float(spearmanr(x[m], y[m]).correlation)


def main() -> None:
    upload = "--no-upload" not in sys.argv
    per_concept = {}
    for concept in CONCEPTS:
        texts, labels, topics, pgroups = build_examples(concept)
        by_topic = grouped_cv(texts, labels, topics)     # leak-free (held-out topics)
        by_pair = grouped_cv(texts, labels, pgroups)      # weaker control, for contrast
        fisher = mean_peak_fisher(concept)
        per_concept[concept] = {
            "n_examples": len(texts),
            "bow_auc_heldout_topic": by_topic,
            "bow_auc_heldout_pair": by_pair,
            "peak_fisher_separation": fisher,
        }
        print(f"[c12] {concept:15s} BoW AUC (held-out topic)={by_topic['auc']:.3f} "
              f"| held-out pair={by_pair['auc']:.3f} "
              f"| peak Fisher={fisher['mean']:.3f}")

    # cross-concept: does surface-lexical separability track neural separation?
    order = [c for c in CONCEPTS
             if per_concept[c]["bow_auc_heldout_topic"]["auc"] is not None
             and per_concept[c]["peak_fisher_separation"]["mean"] is not None]
    bow_topic = [per_concept[c]["bow_auc_heldout_topic"]["auc"] for c in order]
    fish = [per_concept[c]["peak_fisher_separation"]["mean"] for c in order]
    rho = spearman(bow_topic, fish)

    out = {
        "job": "c12_bow_baseline",
        "scope": f"study calibration draw: load_concept_pairs(n={N_DRAW}, "
                 f"split={SPLIT!r}); 17 semantic concepts (obfuscation/CIA excluded)",
        "independence": "classifier = raw-text TF-IDF(1-2gram)+logistic only; "
                        "no rosetta_tools extraction/CAZ/DOM/activations as "
                        "features; Fisher read only as comparison target; "
                        "GroupKFold held-out-topic CV (no topic/pair leakage)",
        "classifier": "TfidfVectorizer(ngram=(1,2),min_df=2,sublinear_tf,"
                      "max_features=50000) + LogisticRegression(C=1,max_iter=2000)",
        "cv": f"GroupKFold(n_splits<= {N_FOLDS}) grouped by topic (primary) and "
              f"by pair_id (contrast)",
        "per_concept": per_concept,
        "cross_concept": {
            "spearman_bow_topic_auc_vs_peak_fisher": rho,
            "interpretation_note": "high positive rho would mean concepts the "
                "model separates most are also the most lexically separable "
                "(surface contribution); low/near-zero rho argues the neural "
                "separation is not a rank-image of the BoW ceiling. DISCUSSION "
                "INPUT ONLY — no paper-text conclusion drawn here (James rule).",
            "grand_mean_bow_auc_heldout_topic": float(np.mean(bow_topic)),
            "grand_mean_bow_auc_heldout_pair":
                float(np.mean([per_concept[c]["bow_auc_heldout_pair"]["auc"]
                               for c in order])),
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n[c12] grand mean BoW AUC (held-out topic) = "
          f"{out['cross_concept']['grand_mean_bow_auc_heldout_topic']:.3f}")
    print(f"[c12] Spearman(BoW-AUC, peak Fisher) across concepts = {rho}")
    print(f"[c12] wrote {OUT_JSON.name}")

    if upload:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=str(OUT_JSON),
                            path_in_repo=f"{HF_SUBTREE}/{OUT_JSON.name}",
                            repo_id="james-ra-henry/Rosetta-Activations",
                            repo_type="dataset",
                            commit_message="C12 BoW baseline vs Fisher (G7 gate, "
                                            "tb98f327)")
        print(f"[c12] uploaded to {HF_SUBTREE}/")


if __name__ == "__main__":
    main()

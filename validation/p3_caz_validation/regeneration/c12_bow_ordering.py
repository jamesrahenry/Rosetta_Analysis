#!/usr/bin/env python3
"""C12 (option-a redesign) — does SURFACE LEXICAL DIFFICULTY predict the CAZ
DEPTH ORDERING?

Background
----------
The original C12 (`c12_bow_baseline.py`) classified each concept's contrastive
RCP pairs with a TF-IDF + logistic-regression bag-of-words probe and hit an AUC
ceiling (~0.999, most concepts exactly 1.0 with fold SD 0.0). The pairs are
lexically separable by construction, so at ceiling there is NO per-concept
variance to correlate against the depth ordering — the control is uninformative
(see C12_BOW_DISCUSSION.md "Options when we come back", option (a), DECIDED as
the queued follow-up by James 2026-07-17).

This redesign follows option (a): DON'T CLASSIFY — PREDICT THE ORDERING. We
deliberately CRIPPLE the classifier off the AUC ceiling so that per-concept
lexical *difficulty* acquires real variance, extract several per-concept
difficulty proxies, and correlate each against the concept's MEAN CAZ PEAK DEPTH
(`layer_data.peak_depth_pct`, averaged across the 28-model Table-1 corpus). We
report Kendall's tau between the surface-difficulty ordering and the depth
ordering, honestly, for every crippling strategy x difficulty measure.

Interpretation
--------------
The paper's per-model-median depth-ordering tau is **0.404** (post-exfiltration-
fix corrected value; the Hopper task text's 0.417 is the pre-fix number).
  * If a surface-difficulty measure reproduces the depth ordering with
    |tau| comparable to 0.404, the depth ordering would have a surface-lexical
    explanation (bad — it would be an artifact of how hard the pairs are to tell
    apart from words alone).
  * If |tau| ~ 0 (null), surface lexical difficulty does NOT predict depth
    ordering — the clean negative control C12 was commissioned to be: the depth
    ordering is not a lexical-difficulty artifact.

A NULL RESULT IS A GOOD RESULT here. We report the sign and magnitude either way.

Crippling strategies (all land off the AUC ceiling — verified empirically)
-------------------------------------------------------------------------
The full/unigram/char-full BoW all ceiling at ~1.0; the knob that produces
per-concept variance is capping the classifier to the k BEST features
(SelectKBest by chi2, fit on TRAIN folds only — no leakage), and dropping to
character n-grams. Strategies:
  * word_unigram_topk5   : word unigrams, top 5 chi2 features
  * word_unigram_topk20  : word unigrams, top 20 chi2 features
  * char_ngram_topk30    : char 2-3 grams, top 30 chi2 features

Per-concept difficulty measures (higher value == HARDER, by construction)
-------------------------------------------------------------------------
  * one per strategy:
      d_1minus_auc      = 1 - held-out-topic CV AUC of the crippled classifier
                          (harder = the crippled probe separates pos/neg worse)
      d_neg_margin      = - mean |logistic decision-function margin| on held-out
                          test points (harder = crippled scores sit near 0)
  * strategy-independent:
      min_features_to_auc95 = smallest k (top-k chi2 word unigrams) at which the
                          held-out-topic CV AUC first reaches >= 0.95, over the
                          grid [1,2,3,5,8,13,21,34,55,89,144]; sentinel 200 if
                          never reached. Higher = needs more vocabulary = harder.

CV discipline (inherited from c12_bow_baseline.py EXACTLY)
----------------------------------------------------------
Data: rosetta_tools.dataset.load_concept_pairs(n=250, split="train") — the exact
N=250 calibration draw. Each pair -> a pos (1) and neg (0) example, grouped by
TOPIC. Held-out-topic GroupKFold(n_splits<=5) so no topic leaks train->test.
All feature selection / vectorization is fit inside the pipeline on train folds
only. RAW TEXT ONLY — no rosetta_tools extraction / CAZ / DOM / activations feed
the classifier; the caz JSONs are read purely as the comparison target (depth).

Independence (unchanged from the baseline): 17 semantic concepts only; the
CIA-only `obfuscation` concept is never touched. No Eigan / CIA leakage.

Output (next to this script): c12_bow_ordering_results.json
Not uploaded to HF and draws no paper-text conclusion — standalone control whose
result feeds a future decision (James rule).

Written: 2026-07-22 UTC by claude:c12-ordering
"""
from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")  # sklearn convergence / degenerate-fold chatter

for _p in (str(Path.home() / "rosetta_tools"),
           str(Path.home() / "Source" / "Rosetta_Program" / "rosetta_tools"),
           str(Path.home() / "Games2" / "Eigan" / "Rosetta_Program" / "rosetta_tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from rosetta_tools.dataset import load_concept_pairs  # noqa: E402 — TEXT loader only

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.feature_selection import SelectKBest, chi2       # noqa: E402
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.metrics import roc_auc_score                    # noqa: E402
from sklearn.model_selection import GroupKFold                # noqa: E402
from sklearn.pipeline import Pipeline                         # noqa: E402
from scipy.stats import kendalltau                            # noqa: E402

# 17 semantic concepts (obfuscation/CIA excluded) — CONCEPTS_17.
CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]
assert "obfuscation" not in CONCEPTS, "obfuscation is CIA-only — must not appear"

# The paper's 28-model Table-1 corpus (copied inline from c6_sensitivity.py;
# common.BASE_28 was trimmed to 25 for a GPU session and is WRONG for the
# paper's published depth ordering).
BASE_28 = [
    "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",
    "openai-community/gpt2", "openai-community/gpt2-medium",
    "openai-community/gpt2-large", "openai-community/gpt2-xl",
    "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b",
    "facebook/opt-2.7b", "facebook/opt-6.7b",
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-14B",
    "google/gemma-2-2b", "google/gemma-2-9b",
    "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B",
    "mistralai/Mistral-7B-v0.3", "microsoft/phi-2",
]
assert len(BASE_28) == 28

N_DRAW = 250
SPLIT = "train"
N_FOLDS = 5
SEED = 42
AUC95 = 0.95
FEATURE_GRID = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
NEVER_SENTINEL = 200  # min-features value if AUC 0.95 never reached on the grid

PAPER_N250 = Path.home() / "rosetta_data" / "paper_n250"
OUT_JSON = Path(__file__).parent / "c12_bow_ordering_results.json"

# Paper's corrected per-model-median depth-ordering tau (post-exfiltration fix).
PAPER_DEPTH_TAU = 0.404


def slugify(model_id: str) -> str:
    return model_id.replace("/", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_examples(concept: str):
    """(texts, labels, topic_group) from the study's n=250 draw. Each pair yields
    a pos (1) and neg (0) example; the topic group lets GroupKFold hold out whole
    topics so nothing leaks train->test."""
    pairs = load_concept_pairs(concept, n=N_DRAW, split=SPLIT)
    texts, labels, topics = [], [], []
    for p in pairs:
        for text, lab in ((p.pos_text, 1), (p.neg_text, 0)):
            texts.append(text)
            labels.append(lab)
            topics.append(p.topic)
    return np.array(texts, dtype=object), np.array(labels), np.array(topics)


# ---------------------------------------------------------------------------
# Crippled classifiers (feature selection fit on TRAIN folds only)
# ---------------------------------------------------------------------------
def make_pipe(strategy: str, k: int) -> Pipeline:
    if strategy.startswith("word_unigram"):
        vec = TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 1),
                              min_df=2, sublinear_tf=True)
    elif strategy.startswith("char_ngram"):
        vec = TfidfVectorizer(lowercase=True, analyzer="char", ngram_range=(2, 3),
                              min_df=2, sublinear_tf=True)
    else:
        raise ValueError(strategy)
    return Pipeline([
        ("tfidf", vec),
        ("kbest", SelectKBest(chi2, k=k)),
        ("lr", LogisticRegression(max_iter=2000, C=1.0)),
    ])


STRATEGIES = {
    "word_unigram_topk5": ("word_unigram", 5),
    "word_unigram_topk20": ("word_unigram", 20),
    "char_ngram_topk30": ("char_ngram", 30),
}


def crippled_cv(texts, labels, topics, strategy: str, k: int) -> dict:
    """Held-out-topic GroupKFold CV of a crippled classifier. Returns mean AUC
    and the mean absolute logistic decision-function margin on held-out points."""
    uniq = np.unique(topics)
    n_splits = int(min(N_FOLDS, len(uniq)))
    if n_splits < 2:
        return {"auc": None, "auc_sd": None, "mean_abs_margin": None, "n_folds_used": 0}
    gkf = GroupKFold(n_splits=n_splits)
    aucs, margins = [], []
    for tr, te in gkf.split(texts, labels, topics):
        if len(np.unique(labels[te])) < 2:
            continue
        # guard: SelectKBest k must not exceed the train-fold vocabulary
        pipe = make_pipe(strategy, k)
        pipe.fit(texts[tr].tolist(), labels[tr])
        prob = pipe.predict_proba(texts[te].tolist())[:, 1]
        aucs.append(roc_auc_score(labels[te], prob))
        margins.append(float(np.mean(np.abs(pipe.decision_function(texts[te].tolist())))))
    if not aucs:
        return {"auc": None, "auc_sd": None, "mean_abs_margin": None, "n_folds_used": 0}
    return {
        "auc": float(np.mean(aucs)),
        "auc_sd": float(np.std(aucs)),
        "mean_abs_margin": float(np.mean(margins)),
        "n_folds_used": len(aucs),
    }


def min_features_to_auc95(texts, labels, topics) -> dict:
    """Smallest k (top-k chi2 word unigrams) at which held-out-topic CV AUC first
    reaches >= AUC95, over FEATURE_GRID. Strategy-independent difficulty proxy;
    higher = needs more vocabulary = harder. Sentinel NEVER_SENTINEL if the
    ceiling is never reached on the grid."""
    curve = {}
    hit = None
    for k in FEATURE_GRID:
        res = crippled_cv(texts, labels, topics, "word_unigram", k)
        curve[k] = res["auc"]
        if hit is None and res["auc"] is not None and res["auc"] >= AUC95:
            hit = k
    return {"min_features": hit if hit is not None else NEVER_SENTINEL,
            "reached_95": hit is not None, "auc_curve": curve}


# ---------------------------------------------------------------------------
# Depth ordering target (READ ONLY — never a classifier feature)
# ---------------------------------------------------------------------------
def mean_peak_depth(concept: str) -> dict:
    """Mean CAZ peak depth (%) for a concept across the 28-model corpus, from
    layer_data.peak_depth_pct in each model's caz_<concept>.json."""
    depths, missing = [], []
    for m in BASE_28:
        f = PAPER_N250 / slugify(m) / f"caz_{concept}.json"
        if not f.is_file():
            missing.append(m)
            continue
        try:
            depths.append(float(json.loads(f.read_text())["layer_data"]["peak_depth_pct"]))
        except Exception:  # noqa: BLE001
            missing.append(m)
    if not depths:
        return {"mean": None, "median": None, "n_models": 0, "missing": missing}
    a = np.array(depths, dtype=np.float64)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "n_models": int(a.size), "missing": missing}


# ---------------------------------------------------------------------------
# Correlation vs depth ordering
# ---------------------------------------------------------------------------
def tau_vs_depth(difficulty_by_concept: dict, depth_by_concept: dict) -> dict:
    """Kendall tau between a per-concept difficulty measure (higher == harder)
    and mean peak depth. Positive tau => harder concepts sit deeper."""
    order = [c for c in CONCEPTS
             if difficulty_by_concept.get(c) is not None
             and depth_by_concept.get(c) is not None]
    if len(order) < 3:
        return {"tau": None, "p_value": None, "n_concepts": len(order)}
    diff = [difficulty_by_concept[c] for c in order]
    dep = [depth_by_concept[c] for c in order]
    res = kendalltau(diff, dep)
    return {"tau": float(res.correlation), "p_value": float(res.pvalue),
            "n_concepts": len(order),
            "abs_tau_vs_paper_0404": float(abs(res.correlation)) - PAPER_DEPTH_TAU}


def main() -> None:
    per_concept = {}
    print("[c12-ordering] computing crippled difficulty measures + depth target...")
    for concept in CONCEPTS:
        texts, labels, topics = build_examples(concept)
        strat = {}
        for name, (family, k) in STRATEGIES.items():
            strat[name] = crippled_cv(texts, labels, topics, family, k)
        mf = min_features_to_auc95(texts, labels, topics)
        depth = mean_peak_depth(concept)
        per_concept[concept] = {
            "n_examples": int(len(texts)),
            "n_topics": int(len(np.unique(topics))),
            "crippled": strat,
            "min_features_to_auc95": mf,
            "mean_peak_depth": depth,
        }
        aucs = " ".join(f"{n.split('_')[-1]}={strat[n]['auc']:.3f}" for n in STRATEGIES)
        print(f"  {concept:15s} depth={depth['mean']:5.1f}%  {aucs}  "
              f"minfeat={mf['min_features']}")

    depth_by = {c: per_concept[c]["mean_peak_depth"]["mean"] for c in CONCEPTS}

    # Assemble per-measure difficulty vectors (higher == harder) and correlate.
    measures = {}
    for name in STRATEGIES:
        # d_1minus_auc: 1 - AUC (harder = worse separation)
        measures[f"{name}::d_1minus_auc"] = {
            c: (1.0 - per_concept[c]["crippled"][name]["auc"])
            if per_concept[c]["crippled"][name]["auc"] is not None else None
            for c in CONCEPTS
        }
        # d_neg_margin: -mean|margin| (harder = scores near 0)
        measures[f"{name}::d_neg_margin"] = {
            c: (-per_concept[c]["crippled"][name]["mean_abs_margin"])
            if per_concept[c]["crippled"][name]["mean_abs_margin"] is not None else None
            for c in CONCEPTS
        }
    measures["min_features_to_auc95"] = {
        c: per_concept[c]["min_features_to_auc95"]["min_features"] for c in CONCEPTS
    }

    correlations = {name: tau_vs_depth(m, depth_by) for name, m in measures.items()}

    written = subprocess.run(["date", "-u", "+%Y-%m-%d %H:%M UTC"],
                             capture_output=True, text=True).stdout.strip()
    out = {
        "job": "c12_bow_ordering (option-a redesign)",
        "written_utc": written,
        "question": "Does surface lexical difficulty (crippled-BoW) predict the "
                    "CAZ depth ordering (mean peak_depth_pct across 28 models)?",
        "paper_depth_ordering_tau": PAPER_DEPTH_TAU,
        "paper_depth_ordering_tau_note": "corrected per-model-median value after "
            "the exfiltration recorded-draw fix; the Hopper task's 0.417 is pre-fix",
        "scope": f"load_concept_pairs(n={N_DRAW}, split={SPLIT!r}); 17 semantic "
                 f"concepts; depth from 28-model Table-1 corpus",
        "independence": "crippled classifier = raw-text TF-IDF + SelectKBest(chi2) "
                        "+ logistic, feature selection fit on train folds only; "
                        "caz peak_depth_pct read only as target; GroupKFold "
                        "held-out-topic CV (no topic leakage)",
        "crippling_strategies": {
            "word_unigram_topk5": "word unigrams, top 5 chi2 features",
            "word_unigram_topk20": "word unigrams, top 20 chi2 features",
            "char_ngram_topk30": "char 2-3 grams, top 30 chi2 features",
        },
        "difficulty_measures": {
            "d_1minus_auc": "1 - held-out-topic CV AUC of the crippled classifier "
                            "(higher = harder)",
            "d_neg_margin": "- mean |logistic decision margin| on held-out points "
                            "(higher = harder)",
            "min_features_to_auc95": "smallest top-k chi2 word-unigram count to "
                            f"reach CV AUC >= {AUC95} over {FEATURE_GRID}; sentinel "
                            f"{NEVER_SENTINEL} if never (higher = harder)",
        },
        "convention": "each difficulty measure is oriented so HIGHER == HARDER; a "
                      "POSITIVE tau vs depth means harder concepts sit deeper. "
                      "Compare |tau| against the paper's 0.404: |tau| ~ 0.404 => "
                      "surface explanation for the depth ordering (bad); |tau| ~ 0 "
                      "=> null (good — depth ordering is not a lexical-difficulty "
                      "artifact).",
        "per_concept": per_concept,
        "correlations_vs_depth": correlations,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    print("\n[c12-ordering] Kendall tau vs mean peak depth (higher measure = harder):")
    print(f"  paper depth-ordering tau reference = {PAPER_DEPTH_TAU}")
    for name, r in correlations.items():
        if r["tau"] is None:
            print(f"    {name:38s} tau=NА")
            continue
        print(f"    {name:38s} tau={r['tau']:+.3f}  p={r['p_value']:.3f}  "
              f"|tau|-0.404={r['abs_tau_vs_paper_0404']:+.3f}")
    print(f"\n[c12-ordering] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()

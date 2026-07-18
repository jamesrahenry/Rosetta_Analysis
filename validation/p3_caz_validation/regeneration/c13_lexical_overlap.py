#!/usr/bin/env python3
"""C13 — pair-level lexical-overlap audit + 20% human-validation sample.

Round-3 corpus-quality gate for G7 (ROUND3_COMPUTE_PLAN.md C12-C14, Hopper
tb98f327). Both papers name this audit (P3 §2.2/§8.7, P4 §4.5): quantify how
much surface vocabulary a pos/neg pair shares, per concept, so a lexical leak
that a discrimination pipeline could be exploiting is caught before any
from-scratch G7 corpus rebuild.

INDEPENDENCE (no Eigan / no CIA leakage — hard requirement 2026-07-16):
  * NO CIA data: the CIA-only `obfuscation` concept (tokenization-level
    leet/base64 contrasts, excluded from CAZ/PRH by construction) is NOT
    touched — only the 17 semantic concepts.
  * NO Eigan-pipeline leakage: every statistic here is computed from RAW pair
    text alone. This module does not import rosetta_tools' extraction/CAZ/GEM
    machinery and uses none of its DOM vectors, Fisher separations, or
    activations. The only pipeline touchpoint is `dataset.load_concept_pairs`
    — the plain text loader — used solely to pull the study's exact n=250
    calibration draw (so the audit covers the pairs the paper's directions are
    actually built from). Tokenisation is self-contained (regex), so the
    result is reproducible from the corpus without the toolkit.

Outputs (next to this script):
  c13_lexical_overlap_results.json  — per-concept + grand overlap distributions
  c13_human_validation_sample.csv   — deterministic random 20%/concept for the
                                       analyst/James concept-fidelity pass
Uploaded to HF _round3_corpus_quality/ (the G7-gate group).

Written: 2026-07-16 UTC
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

RP_ROOT = Path(__file__).resolve().parents[3]   # .../Rosetta_Program
sys.path.insert(0, str(RP_ROOT / "rosetta_tools"))
from rosetta_tools.dataset import load_concept_pairs  # noqa: E402 — TEXT loader only

# 17 semantic concepts — obfuscation (CIA-only) deliberately excluded.
CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]
assert "obfuscation" not in CONCEPTS, "obfuscation is CIA-only — must not appear"

N_DRAW = 250
SPLIT = "train"          # the study's calibration population
SAMPLE_FRAC = 0.20
SEED = 42
OUT_JSON = Path(__file__).parent / "c13_lexical_overlap_results.json"
OUT_CSV = Path(__file__).parent / "c13_human_validation_sample.csv"
HF_SUBTREE = "paper_n250/_round3_corpus_quality"

_WORD = re.compile(r"[a-z0-9']+")
# A small, standard English stoplist — kept deliberately generic (not derived
# from the corpus) so "content overlap" isn't tuned to these pairs.
STOP = frozenset("""
a an the and or but if then else of to in on at by for with without from into
over under again further is are was were be been being am do does did doing
have has had having i you he she it we they them his her its our their this
that these those as so than too very can will just not no nor only own same
s t don should now what which who whom
""".split())


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def overlap_stats(pos: str, neg: str) -> dict:
    """Symmetric lexical-overlap measures between a pos and neg text. All are
    surface-vocabulary quantities — no semantics, no model features."""
    pt, nt = tokens(pos), tokens(neg)
    ps, ns = set(pt), set(nt)
    pc, nc = ps - STOP, ns - STOP           # content-word vocabularies
    inter, union = ps & ns, ps | ns
    cinter, cunion = pc & nc, pc | nc

    def jacc(i, u):
        return len(i) / len(u) if u else 0.0

    def overlap_coef(a, b, i):
        m = min(len(a), len(b))
        return len(i) / m if m else 0.0

    return {
        "n_tok_pos": len(pt), "n_tok_neg": len(nt),
        "n_type_pos": len(ps), "n_type_neg": len(ns),
        "jaccard": jacc(inter, union),
        "overlap_coef": overlap_coef(ps, ns, inter),
        "content_jaccard": jacc(cinter, cunion),
        "content_overlap_coef": overlap_coef(pc, nc, cinter),
        "n_shared_content_types": len(cinter),
    }


def summarize(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
        "sd": float(a.std()), "p05": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)), "max": float(a.max()),
    }


def main() -> None:
    upload = "--no-upload" not in sys.argv
    rng = np.random.default_rng(SEED)
    per_concept, per_pair_rows, sample_rows = {}, [], []
    grand = {k: [] for k in ("jaccard", "overlap_coef", "content_jaccard",
                             "content_overlap_coef")}

    for concept in CONCEPTS:
        pairs = load_concept_pairs(concept, n=N_DRAW, split=SPLIT)
        stats = [overlap_stats(p.pos_text, p.neg_text) for p in pairs]
        by = {k: [s[k] for s in stats] for k in grand}
        for k in grand:
            grand[k].extend(by[k])
        per_concept[concept] = {
            "n_pairs": len(pairs),
            **{k: summarize(by[k]) for k in grand},
            "mean_shared_content_types":
                float(np.mean([s["n_shared_content_types"] for s in stats])),
        }
        # keep the per-pair jaccard for the flagged-pair audit trail
        for p, s in zip(pairs, stats):
            per_pair_rows.append({
                "concept": concept, "pair_id": p.pair_id,
                "content_jaccard": round(s["content_jaccard"], 4),
                "overlap_coef": round(s["overlap_coef"], 4),
            })

        # deterministic 20% human-validation sample for this concept
        k = max(1, round(SAMPLE_FRAC * len(pairs)))
        idx = rng.choice(len(pairs), size=k, replace=False)
        for i in sorted(idx.tolist()):
            p = pairs[i]
            sample_rows.append({
                "concept": concept, "pair_id": p.pair_id, "topic": p.topic,
                "domain": p.domain, "model_name": p.model_name,
                "content_jaccard": round(stats[i]["content_jaccard"], 4),
                "pos_text": p.pos_text, "neg_text": p.neg_text,
                # blank columns for the human pass:
                "pos_expresses_concept": "", "neg_absent_or_opposite": "",
                "pair_valid": "", "notes": "",
            })
        cj = per_concept[concept]["content_jaccard"]
        print(f"[c13] {concept:15s} n={len(pairs):3d}  content-Jaccard "
              f"mean={cj['mean']:.3f} median={cj['median']:.3f} p95={cj['p95']:.3f}")

    # concepts whose content overlap runs high enough to warrant a leak look:
    # flag any concept > grand mean + 1 sd of the per-concept content-Jaccard means.
    cmeans = np.array([per_concept[c]["content_jaccard"]["mean"] for c in CONCEPTS])
    flag_thresh = float(cmeans.mean() + cmeans.std())
    flagged = [c for c in CONCEPTS
               if per_concept[c]["content_jaccard"]["mean"] > flag_thresh]

    out = {
        "job": "c13_lexical_overlap",
        "scope": f"study calibration draw: load_concept_pairs(n={N_DRAW}, "
                 f"split={SPLIT!r}); 17 semantic concepts (obfuscation/CIA excluded)",
        "independence": "raw-text lexical statistics only; no rosetta_tools "
                        "extraction/CAZ/GEM/DOM/Fisher/activations used",
        "tokeniser": "[a-z0-9']+ lowercased; generic English stoplist for the "
                     "content-word variants (not corpus-derived)",
        "grand": {k: summarize(v) for k, v in grand.items()},
        "per_concept": per_concept,
        "flagged_high_overlap_concepts": flagged,
        "flag_threshold_content_jaccard_mean": flag_thresh,
        "human_validation_sample": {
            "frac": SAMPLE_FRAC, "seed": SEED, "n_rows": len(sample_rows),
            "csv": OUT_CSV.name,
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    fields = ["concept", "pair_id", "topic", "domain", "model_name",
              "content_jaccard", "pos_text", "neg_text",
              "pos_expresses_concept", "neg_absent_or_opposite",
              "pair_valid", "notes"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sample_rows)

    g = out["grand"]["content_jaccard"]
    print(f"\n[c13] GRAND content-Jaccard: mean={g['mean']:.3f} median="
          f"{g['median']:.3f} p95={g['p95']:.3f} max={g['max']:.3f}")
    print(f"[c13] flagged high-overlap concepts (> {flag_thresh:.3f}): "
          f"{flagged or 'none'}")
    print(f"[c13] wrote {OUT_JSON.name} and {OUT_CSV.name} "
          f"({len(sample_rows)} rows for human validation)")

    if upload:
        from huggingface_hub import HfApi
        api = HfApi()
        for pth in (OUT_JSON, OUT_CSV):
            api.upload_file(path_or_fileobj=str(pth),
                            path_in_repo=f"{HF_SUBTREE}/{pth.name}",
                            repo_id="james-ra-henry/Rosetta-Activations",
                            repo_type="dataset",
                            commit_message="C13 lexical-overlap audit + human-val "
                                            "sample (G7 gate, tb98f327)")
        print(f"[c13] uploaded to {HF_SUBTREE}/")


if __name__ == "__main__":
    main()

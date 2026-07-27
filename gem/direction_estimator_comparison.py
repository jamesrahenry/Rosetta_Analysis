#!/usr/bin/env python3
"""M16 — GEM's difference-of-means estimator vs. supervised baselines, at a fixed layer.

Regenerates ``direction_estimator_comparison.json``. The original artifact had no
script in this release; per the repo convention that paper regeneration scripts are
canonical here, any number this experiment reports has to come from this file.

What it answers
---------------
GEM extracts an unsupervised direction (difference of class means, ``dom``). §8.1 of
the GEM paper declines a comparison against supervised probes. This runs it: ``dom``,
logistic regression and LDA are each fitted on the *same* training split and scored on
the *same* held-out split, **at the same layer**, so extraction depth is held constant
and only the estimator differs.

Two corrections relative to the 2026-07-03 artifact
---------------------------------------------------
1. **Pair-aware splitting.** That artifact predates ``rosetta_tools`` v1.6.0. Its split
   permuted positives and negatives independently, so at ``eval_frac=0.2`` a contrastive
   pair straddled the boundary 32% of the time. Because RCP pairs are minimal edits of
   one another, a fitted estimator could key on a training item and score its
   near-identical held-out mate — a leak that favours logreg/LDA over ``dom``, which as
   a single centroid contrast cannot exploit it. The tell was ``logreg`` reaching
   held-out AUROC of exactly 1.0 in 65/102 cells. ``_split_indices`` is now pair-aware,
   and this script asserts the property rather than trusting it.
2. **Post-correction labels.** It also predates the 2026-07-17 exfiltration label
   correction, and exfiltration is where ``dom`` fared worst. Current pairs are used.

Report magnitude, not win rate
------------------------------
Both estimators sit near ceiling. A win rate counts float comparisons at 1.0 as wins and
overstates the difference; ``mean_diff`` and the ceiling-proximity figures are the
interpretable statistics. Both are emitted, with the ceiling rate alongside so a reader
can see how saturated the regime is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

for _cand in (Path.home() / "rosetta_tools", Path.home() / "Source" / "Rosetta_Program" / "rosetta_tools"):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from rosetta_tools.paths import ROSETTA_MODELS, ROSETTA_RESULTS
from rosetta_tools.probes import _split_indices

DEFAULT_MODELS = [
    "EleutherAI_pythia_1.4b",
    "EleutherAI_pythia_2.8b",
    "Qwen_Qwen2.5_3B",
    "facebook_opt_1.3b",
    "facebook_opt_2.7b",
    "meta_llama_Llama_3.2_1B",
]


def _dom_direction(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    d = pos.mean(axis=0) - neg.mean(axis=0)
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else d


def _auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    y = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    return float(roc_auc_score(y, np.concatenate([scores_pos, scores_neg])))


def _select_layer(probe_dir: Path, concept: str) -> int | None:
    """Layer of maximum training-split separation — chosen once, shared by all estimators."""
    gem = probe_dir / f"gem_{concept}.json"
    if not gem.exists():
        return None
    nodes = json.loads(gem.read_text()).get("nodes") or []
    if not nodes:
        return None
    return max(nodes, key=lambda n: n.get("caz_score", 0)).get("handoff_layer")


def compare_one(model_dir: Path, concept: str, eval_frac: float, seed: int) -> dict | None:
    pos_p, neg_p = model_dir / f"{concept}_pos.npy", model_dir / f"{concept}_neg.npy"
    if not (pos_p.exists() and neg_p.exists()):
        return None
    pos_all, neg_all = np.load(pos_p), np.load(neg_p)          # [n, n_layers, d]
    layer = _select_layer(model_dir, concept)
    if layer is None or layer >= pos_all.shape[1]:
        return None

    pos, neg = pos_all[:, layer, :].astype(np.float64), neg_all[:, layer, :].astype(np.float64)
    n_pos, n_neg = len(pos), len(neg)

    pos_tr, neg_tr, pos_ev, neg_ev = _split_indices(n_pos, n_neg, eval_frac, seed)
    if n_pos == n_neg:
        # The whole point of the re-run: no pair may straddle the split.
        assert set(pos_tr) == set(neg_tr) and set(pos_ev) == set(neg_ev), (
            "split is not pair-aware — rosetta_tools < 1.6.0?"
        )

    Xtr = np.vstack([pos[pos_tr], neg[neg_tr]])
    ytr = np.concatenate([np.ones(len(pos_tr)), np.zeros(len(neg_tr))])
    p_ev, n_ev = pos[pos_ev], neg[neg_ev]

    d = _dom_direction(pos[pos_tr], neg[neg_tr])
    out = {
        "model": model_dir.name, "concept": concept, "layer": int(layer),
        "n_train": int(len(pos_tr) + len(neg_tr)), "n_eval": int(len(pos_ev) + len(neg_ev)),
        "dom_layer": int(layer), "dom_auroc": round(_auroc(p_ev @ d, n_ev @ d), 6),
    }
    for name, est in (("logreg", LogisticRegression(max_iter=2000)),
                      ("lda", LinearDiscriminantAnalysis())):
        try:
            est.fit(Xtr, ytr)
            w = np.asarray(est.coef_).ravel()
            out[f"{name}_layer"] = int(layer)
            out[f"{name}_auroc"] = round(_auroc(p_ev @ w, n_ev @ w), 6)
        except Exception as exc:                                # noqa: BLE001
            out[f"{name}_layer"], out[f"{name}_auroc"], out[f"{name}_error"] = int(layer), None, str(exc)
    return out


def summarise(rows: list[dict]) -> dict:
    def col(k):
        return np.array([r[k] for r in rows if r.get(k) is not None], dtype=float)

    dom = col("dom_auroc")
    s = {"dom": {"mean_auroc": float(dom.mean()), "std_auroc": float(dom.std()),
                 "median_auroc": float(np.median(dom)),
                 "at_ceiling_rate": float(np.mean(dom >= 1.0 - 1e-9))}}
    for name in ("logreg", "lda"):
        paired = [(r["dom_auroc"], r[f"{name}_auroc"]) for r in rows if r.get(f"{name}_auroc") is not None]
        a = np.array([p[1] for p in paired]); b = np.array([p[0] for p in paired])
        s[name] = {
            "mean_auroc": float(a.mean()), "std_auroc": float(a.std()),
            "median_auroc": float(np.median(a)),
            "at_ceiling_rate": float(np.mean(a >= 1.0 - 1e-9)),
            "mean_diff_vs_dom": float((a - b).mean()),
            "median_diff_vs_dom": float(np.median(a - b)),
            # Reported for continuity with the 2026-07-03 artifact only. In a
            # ceiling-saturated regime this counts float comparisons at 1.0 as wins;
            # read mean_diff_vs_dom instead.
            "win_rate_vs_dom": float(np.mean(a > b)),
            "tie_rate_vs_dom": float(np.mean(a == b)),
        }
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=ROSETTA_RESULTS / "direction_estimator_comparison_pairaware.json")
    a = ap.parse_args()

    rows: list[dict] = []
    for slug in a.models:
        mdir = ROSETTA_MODELS / slug
        if not mdir.exists():
            print(f"  skip {slug}: no extraction dir", flush=True)
            continue
        concepts = sorted(p.name[len("gem_"):-len(".json")] for p in mdir.glob("gem_*.json"))
        for c in concepts:
            r = compare_one(mdir, c, a.eval_frac, a.seed)
            if r:
                rows.append(r)
        print(f"  {slug}: {sum(1 for r in rows if r['model'] == slug)} concepts", flush=True)

    if not rows:
        print("no comparisons produced", file=sys.stderr)
        return 1

    payload = {
        "n_comparisons": len(rows),
        "pair_aware_split": True,
        "eval_frac": a.eval_frac,
        "seed": a.seed,
        "summary": summarise(rows),
        "detail": rows,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2))

    s = payload["summary"]
    print(f"\n{len(rows)} comparisons, pair-aware split, eval_frac={a.eval_frac}")
    for k in ("dom", "logreg", "lda"):
        v = s[k]
        extra = "" if k == "dom" else f"  mean_diff vs dom {v['mean_diff_vs_dom']:+.4f}  win {100*v['win_rate_vs_dom']:.1f}%"
        print(f"  {k:7s} mean {v['mean_auroc']:.4f}  median {v['median_auroc']:.4f}"
              f"  at-ceiling {100*v['at_ceiling_rate']:.1f}%{extra}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

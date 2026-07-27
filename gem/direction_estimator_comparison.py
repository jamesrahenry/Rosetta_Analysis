#!/usr/bin/env python3
"""M16 — GEM's difference-of-means estimator vs. supervised baselines, at a fixed layer.

Regenerates ``direction_estimator_comparison.json``. The original artifact (2026-07-03)
had no generating script in this release, so its numbers were uncitable on process
grounds regardless of correctness. This is that script.

What it answers
---------------
GEM extracts an unsupervised direction — the difference of class means (``dom``). §8.1
of the GEM paper declines a comparison against supervised probes. This runs it: ``dom``,
logistic regression and LDA are each fitted on the *same* training split and scored on
the *same* held-out split, **at the same layer**, so extraction depth is held constant
and only the estimator varies.

Three corrections relative to the 2026-07-03 artifact
-----------------------------------------------------
1. **Pair-aware splitting.** The old split permuted positives and negatives
   independently, so at ``eval_frac=0.2`` a contrastive pair straddled the boundary
   ``2*f*(1-f)`` = 32% of the time. RCP pairs are minimal edits of one another, so a
   *fitted* estimator could key on a training item and score its near-identical
   held-out mate; a difference-of-means direction, being a single centroid contrast,
   cannot. The leak favours logreg/LDA over ``dom`` — it is aligned with the finding,
   not neutral noise. The tell was ``logreg`` reaching held-out AUROC of exactly 1.0 in
   65/102 cells. ``_split_indices`` below is pair-aware and the property is asserted,
   not assumed.
2. **Post-correction labels.** The old artifact predates the 2026-07-17 exfiltration
   label correction, and exfiltration is where ``dom`` fared worst.
3. **Same data path as every other P2 number.** Activations come from a live forward
   pass via ``extract_layer_activations``, exactly as the ablation scripts do.

Report magnitude, not win rate
------------------------------
Both estimators sit near ceiling. A win rate counts float comparisons at 1.0 as wins and
overstates the difference; ``mean_diff_vs_dom`` and ``at_ceiling_rate`` are the
interpretable statistics. Win rate is emitted for continuity with the old artifact only.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from rosetta_tools.dataset import load_concept_pairs, texts_by_label
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.gem import discover_concepts, find_extraction_dir
from rosetta_tools.gpu_utils import (
    get_device, get_dtype, load_causal_lm, log_device_info, release_model,
)
from rosetta_tools.paths import ROSETTA_RESULTS

log = logging.getLogger("m16")

N_PAIRS = 250
BATCH_SIZE = 8

DEFAULT_MODELS = [
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "Qwen/Qwen2.5-3B",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "meta-llama/Llama-3.2-1B",
]


def slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace("-", "_")


def _split_indices(
    n_pos: int, n_neg: int, eval_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pair-aware train/eval split — implemented here, deliberately.

    ``rosetta_tools.probes._split_indices`` gained this behaviour in v1.6.0, but the GPU
    runner's ``~/rosetta_tools`` provenance is not guaranteed: ``sync_repos`` swallows
    pull failures, and the URL it names for fresh clones ships the package under a
    different name. A number that goes into a paper must not depend on which library
    version happens to be on the host, so the logic lives here and is asserted below.
    """
    rng = np.random.RandomState(seed)
    if n_pos == n_neg:
        n_eval = max(1, int(n_pos * eval_frac))
        perm = rng.permutation(n_pos)
        ev, tr = perm[:n_eval], perm[n_eval:]
        return tr, tr.copy(), ev, ev.copy()
    n_pe, n_ne = max(1, int(n_pos * eval_frac)), max(1, int(n_neg * eval_frac))
    pp, nn = rng.permutation(n_pos), rng.permutation(n_neg)
    return pp[n_pe:], nn[n_ne:], pp[:n_pe], nn[:n_ne]


def _dom_direction(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    d = pos.mean(axis=0) - neg.mean(axis=0)
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else d


def _auroc(sp: np.ndarray, sn: np.ndarray) -> float:
    y = np.concatenate([np.ones(len(sp)), np.zeros(len(sn))])
    return float(roc_auc_score(y, np.concatenate([sp, sn])))


def _handoff_layer(ext_dir: Path, concept: str, n_layers: int) -> int | None:
    """The dominant node's handoff layer — the layer GEM would actually probe at."""
    f = ext_dir / f"gem_{concept}.json"
    if not f.exists():
        return None
    nodes = json.loads(f.read_text()).get("nodes") or []
    if not nodes:
        return None
    layer = max(nodes, key=lambda n: n.get("caz_score", 0)).get("handoff_layer")
    return layer if layer is not None and 0 <= layer < n_layers else None


def run_concept(model, tokenizer, model_id: str, concept: str, ext_dir: Path,
                device: str, eval_frac: float, seed: int) -> dict | None:
    pairs = load_concept_pairs(concept, n=N_PAIRS)
    pos_t, neg_t = texts_by_label(pairs)
    n = min(len(pos_t), len(neg_t))
    if n < 20:
        log.warning("  %s: only %d pairs — skipped", concept, n)
        return None
    pos_t, neg_t = pos_t[:n], neg_t[:n]

    pos_all = extract_layer_activations(model, tokenizer, pos_t, device=device,
                                        batch_size=BATCH_SIZE, pool="last")
    neg_all = extract_layer_activations(model, tokenizer, neg_t, device=device,
                                        batch_size=BATCH_SIZE, pool="last")
    n_layers = len(pos_all)
    layer = _handoff_layer(ext_dir, concept, n_layers)
    if layer is None:
        log.warning("  %s: no usable handoff layer — skipped", concept)
        return None

    pos = np.asarray(pos_all[layer], dtype=np.float64)
    neg = np.asarray(neg_all[layer], dtype=np.float64)

    pos_tr, neg_tr, pos_ev, neg_ev = _split_indices(len(pos), len(neg), eval_frac, seed)
    assert set(pos_tr) == set(neg_tr) and set(pos_ev) == set(neg_ev), \
        "split is not pair-aware — this is the defect the re-run exists to fix"

    Xtr = np.vstack([pos[pos_tr], neg[neg_tr]])
    ytr = np.concatenate([np.ones(len(pos_tr)), np.zeros(len(neg_tr))])
    p_ev, n_ev = pos[pos_ev], neg[neg_ev]

    d = _dom_direction(pos[pos_tr], neg[neg_tr])
    out = {
        "model": slug(model_id), "concept": concept, "layer": int(layer),
        "n_layers": int(n_layers), "n_pairs": int(n),
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
        except Exception as exc:                                    # noqa: BLE001
            out[f"{name}_layer"], out[f"{name}_auroc"] = int(layer), None
            out[f"{name}_error"] = str(exc)
            log.warning("  %s/%s failed: %s", concept, name, exc)
    return out


def summarise(rows: list[dict]) -> dict:
    dom = np.array([r["dom_auroc"] for r in rows], dtype=float)
    s = {"dom": {"mean_auroc": float(dom.mean()), "std_auroc": float(dom.std()),
                 "median_auroc": float(np.median(dom)),
                 "at_ceiling_rate": float(np.mean(dom >= 1.0 - 1e-9))}}
    for name in ("logreg", "lda"):
        pairs = [(r["dom_auroc"], r[f"{name}_auroc"]) for r in rows
                 if r.get(f"{name}_auroc") is not None]
        if not pairs:
            continue
        b = np.array([p[0] for p in pairs])
        a = np.array([p[1] for p in pairs])
        s[name] = {
            "n": len(pairs),
            "mean_auroc": float(a.mean()), "std_auroc": float(a.std()),
            "median_auroc": float(np.median(a)),
            "at_ceiling_rate": float(np.mean(a >= 1.0 - 1e-9)),
            "mean_diff_vs_dom": float((a - b).mean()),
            "median_diff_vs_dom": float(np.median(a - b)),
            # Continuity with the 2026-07-03 artifact only. In a ceiling-saturated
            # regime this counts float comparisons at 1.0 as wins — read mean_diff.
            "win_rate_vs_dom": float(np.mean(a > b)),
            "tie_rate_vs_dom": float(np.mean(a == b)),
        }
    return s


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", dest="models",
                    help="HF model id; repeatable. Default: the 6-model subset.")
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-clean-cache", action="store_true")
    a, _unknown = ap.parse_known_args()
    models = a.models or DEFAULT_MODELS

    out_dir = ROSETTA_RESULTS / "direction_estimator_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for mid in models:
        ext_dir = find_extraction_dir(mid)
        if ext_dir is None:
            log.warning("%s: no extraction dir — skipped", mid)
            continue
        concepts = discover_concepts(ext_dir)
        if not concepts:
            log.warning("%s: no concepts — skipped", mid)
            continue
        log.info("%s: %d concepts", mid, len(concepts))

        device = get_device(a.device)      # signature is (prefer="auto")
        dtype = get_dtype(device)          # signature is (device, prefer="auto") — device first
        log_device_info(device, dtype)     # requires both
        model, tokenizer = load_causal_lm(mid, device=device, dtype=dtype)
        rows: list[dict] = []
        try:
            for c in concepts:
                r = run_concept(model, tokenizer, mid, c, ext_dir, device,
                                a.eval_frac, a.seed)
                if r:
                    rows.append(r)
        finally:
            release_model(model, clear_cache=not a.no_clean_cache)

        if not rows:
            log.warning("%s: produced nothing", mid)
            continue
        payload = {
            "model_id": mid, "n_comparisons": len(rows), "pair_aware_split": True,
            "eval_frac": a.eval_frac, "seed": a.seed,
            "summary": summarise(rows), "detail": rows,
        }
        (out_dir / f"{slug(mid)}_direction_estimator.json").write_text(
            json.dumps(payload, indent=2))
        all_rows.extend(rows)
        s = payload["summary"]
        log.info("  dom %.4f | logreg %.4f (%+.4f) | lda %.4f",
                 s["dom"]["mean_auroc"],
                 s.get("logreg", {}).get("mean_auroc", float("nan")),
                 s.get("logreg", {}).get("mean_diff_vs_dom", float("nan")),
                 s.get("lda", {}).get("mean_auroc", float("nan")))

    if not all_rows:
        log.error("no comparisons produced")
        return 1

    agg = {"n_comparisons": len(all_rows), "pair_aware_split": True,
           "eval_frac": a.eval_frac, "seed": a.seed,
           "models": sorted({r["model"] for r in all_rows}),
           "summary": summarise(all_rows), "detail": all_rows}
    out = a.out or (out_dir / "direction_estimator_comparison_pairaware.json")
    out.write_text(json.dumps(agg, indent=2))

    s = agg["summary"]
    log.info("\n%d comparisons, pair-aware split, eval_frac=%s", len(all_rows), a.eval_frac)
    for k in ("dom", "logreg", "lda"):
        if k not in s:
            continue
        v = s[k]
        extra = "" if k == "dom" else (f"  mean_diff vs dom {v['mean_diff_vs_dom']:+.4f}"
                                       f"  win {100 * v['win_rate_vs_dom']:.1f}%")
        log.info("  %-7s mean %.4f  median %.4f  at-ceiling %5.1f%%%s",
                 k, v["mean_auroc"], v["median_auroc"], 100 * v["at_ceiling_rate"], extra)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

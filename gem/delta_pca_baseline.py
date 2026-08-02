#!/usr/bin/env python3
"""Delta-PCA baseline for GEM §8.3 — the unsupervised comparison the paper is missing.

Spec: ``papers/gem/DELTA_PCA_RUN_SPEC.md`` in Rosetta_Program, including its frozen
PRE-REGISTRATION section. **Read that before changing anything here.** The design is
pre-registered and this script implements it literally; a change to the arms, budgets,
centring default or primary estimand is a change to a frozen protocol, not a refactor.

What it answers
---------------
§8.1 compares GEM's difference-of-means (``dom``) against *supervised* probes. It compares
against nothing *unsupervised*, which is what GEM actually competes with. The standard
unsupervised baseline in representation engineering is contrastive-activation PCA
[Zou et al., 2023] — "delta-PCA": the first principal component of paired positive/negative
activation differences.

Four arms, all scored on the same held-out split of the same cell::

    dom_handoff    dominant node's handoff_layer   difference of class means   (GEM, reference)
    dpca_handoff   dominant node's handoff_layer   delta-PCA                   (estimator effect)
    dom_peak       dominant node's caz_peak        difference of class means   (layer effect)
    dpca_peak      dominant node's caz_peak        delta-PCA                   (the RepE pipeline)

``dom_handoff`` vs ``dpca_peak`` is the single pre-registered primary contrast: GEM as
practised against RepE as practised. The other two decompose it if it moves.

Why a sweep rather than one number
----------------------------------
M16 is ceiling-saturated — ``dom`` median 0.9916, 53% of cells at >=0.99 — because RCP pairs
are lexically separable by construction (P3 §K measures bag-of-words held-out AUC at 0.999).
A four-cell table of 0.99s answers nothing. Every estimator is therefore refit at
``n_train in {10, 25, 50, 100, 200}`` pairs against the *same* held-out split, which costs no
extra forward pass and moves the comparison off the ceiling. Sample efficiency is the axis on
which these estimators actually differ.

Nested budgets: the k-pair training set is always the first k of one permuted training order,
so budgets are nested and the curve is monotone in information, not in luck.

The four implementation choices that can be got wrong in our favour
------------------------------------------------------------------
Asserted here rather than trusted, because a flattering baseline is worse than no baseline:

1. Paired differences are formed on the **training split only**.
2. PC1 is fitted on **train only** — the fit never sees an eval index (asserted; ``n_fit``
   recorded per cell). Fitting on all pairs and scoring held-out is leakage, the same defect
   class that cost M16 a re-run.
3. The sign of PC1 is arbitrary. It is oriented by the **training-split** class means, and the
   oriented direction is asserted to give train-split AUROC >= 0.5.
4. Uncentred is primary (the common reading of Zou et al.); centred is reported alongside as a
   robustness variant, never substituted for it.

Known-result gate
-----------------
``dom_handoff`` at the full 200-pair budget is the *same computation* as M16's stored
``dom_auroc`` — same layer, same split, same seed, same estimator. It must reproduce per cell
to 1e-6 against ``paper_n250/_p2_direction_estimator/``. A mismatch means the harness has
diverged and no other number in the run may be read, so it aborts by default.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from rosetta_tools.dataset import load_concept_pairs, texts_by_label
from rosetta_tools.extraction import extract_layer_activations
from rosetta_tools.gem import discover_concepts, find_extraction_dir
from rosetta_tools.gpu_utils import (
    get_device, get_dtype, load_causal_lm, log_device_info, release_model,
)
from rosetta_tools.paths import ROSETTA_RESULTS

log = logging.getLogger("delta_pca")

N_PAIRS = 250
BATCH_SIZE = 8

# --- pre-registered constants. Changing these changes a frozen protocol. -----------
BUDGETS = (10, 25, 50, 100, 200)          # training pairs
PRIMARY = ("dpca_peak", "dom_handoff")    # primary contrast: A - B
EQUIV_MARGIN = 0.01                       # |mean delta AUROC| below this = practically equivalent
ARMS = ("dom_handoff", "dpca_handoff", "dom_peak", "dpca_peak")

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


def _split_indices(n_pos: int, n_neg: int, eval_frac: float, seed: int):
    """Pair-aware train/eval split — identical to M16's, deliberately.

    Copied rather than imported for the reason M16 gives: a number that goes into a paper
    must not depend on which ``rosetta_tools`` version happens to be on the host. The
    known-result gate below only reproduces if this matches M16 exactly.
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


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _dom_direction(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    return _unit(pos.mean(axis=0) - neg.mean(axis=0))


def _dpca_direction(pos: np.ndarray, neg: np.ndarray, centred: bool) -> np.ndarray:
    """PC1 of the paired activation differences — Zou et al. [2023] contrastive PCA.

    `pos`/`neg` are the TRAINING rows only; the caller is responsible for that and the
    assertion in `run_concept` enforces it. Sign is oriented by the training class means,
    which is the only information-preserving orientation available without touching eval.
    """
    d = pos - neg                                   # paired differences, train only
    if centred:
        d = d - d.mean(axis=0, keepdims=True)
    # PC1 via SVD on the (n x d) difference matrix. full_matrices=False keeps this cheap
    # even at hidden_dim 2560: we only need the leading right-singular vector.
    _u, _s, vt = np.linalg.svd(d, full_matrices=False)
    pc1 = _unit(np.asarray(vt[0], dtype=np.float64))
    # Choice 3: orient by the training-split class means, never by eval.
    if float(np.dot(pc1, pos.mean(axis=0) - neg.mean(axis=0))) < 0:
        pc1 = -pc1
    return pc1


def _auroc(sp: np.ndarray, sn: np.ndarray) -> float:
    y = np.concatenate([np.ones(len(sp)), np.zeros(len(sn))])
    return float(roc_auc_score(y, np.concatenate([sp, sn])))


def _layers(ext_dir: Path, concept: str, n_layers: int) -> dict | None:
    """Dominant node's handoff and peak layers, plus the global argmax-separation layer."""
    f = ext_dir / f"gem_{concept}.json"
    if not f.exists():
        return None
    nodes = json.loads(f.read_text()).get("nodes") or []
    if not nodes:
        return None
    dom_node = max(nodes, key=lambda n: n.get("caz_score", 0))
    handoff, peak = dom_node.get("handoff_layer"), dom_node.get("caz_peak")
    ok = lambda x: x is not None and 0 <= x < n_layers                     # noqa: E731
    if not (ok(handoff) and ok(peak)):
        return None
    # Global argmax of the separation curve, for the spec's "do they differ?" count.
    global_peak = None
    cf = ext_dir / f"caz_{concept}.json"
    if cf.exists():
        ld = json.loads(cf.read_text()).get("layer_data") or {}
        gp = ld.get("peak_layer")
        if ok(gp):
            global_peak = int(gp)
    return {"handoff": int(handoff), "peak": int(peak), "global_peak": global_peak}


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
    lay = _layers(ext_dir, concept, n_layers)
    if lay is None:
        log.warning("  %s: no usable handoff/peak layer — skipped", concept)
        return None

    pos_tr, neg_tr, pos_ev, neg_ev = _split_indices(len(pos_all[0]), len(neg_all[0]),
                                                    eval_frac, seed)
    assert set(pos_tr) == set(neg_tr) and set(pos_ev) == set(neg_ev), \
        "split is not pair-aware — the defect M16 exists to have fixed"
    eval_set = set(int(i) for i in pos_ev) | set(int(i) for i in neg_ev)

    out = {
        "model": slug(model_id), "concept": concept, "n_layers": int(n_layers),
        "n_pairs": int(n), "handoff_layer": lay["handoff"], "peak_layer": lay["peak"],
        "global_argmax_layer": lay["global_peak"],
        "peak_equals_global_argmax": (lay["global_peak"] is not None
                                      and lay["global_peak"] == lay["peak"]),
        "n_eval_pairs": int(len(pos_ev)),
        "budgets": {}, "cosines": {},
    }

    per_layer = {"handoff": lay["handoff"], "peak": lay["peak"]}
    for site, layer in per_layer.items():
        pos = np.asarray(pos_all[layer], dtype=np.float64)
        neg = np.asarray(neg_all[layer], dtype=np.float64)
        p_ev, n_ev = pos[pos_ev], neg[neg_ev]
        for k in BUDGETS:
            if k > len(pos_tr):
                continue
            # Nested budgets: the first k of one permuted training order.
            tr_p, tr_n = pos_tr[:k], neg_tr[:k]
            assert not (set(int(i) for i in tr_p) & eval_set), \
                "train-only fit violated: a training index is in the eval split"
            P, N = pos[tr_p], neg[tr_n]
            slot = out["budgets"].setdefault(str(k), {"n_fit_pairs": int(k)})

            d_dom = _dom_direction(P, N)
            slot[f"dom_{site}"] = round(_auroc(p_ev @ d_dom, n_ev @ d_dom), 6)

            for centred, tag in ((False, ""), (True, "_centred")):
                d_pca = _dpca_direction(P, N, centred=centred)
                # Choice 3 gate: the oriented direction must separate the TRAINING split.
                # Strict for the uncentred (primary) arms — a failure there means the
                # orientation logic is wrong and nothing in the run is readable. The
                # centred variant is a robustness report, and it can legitimately be
                # degenerate: when every pair shares a difference direction, centring
                # removes the signal outright and PC1 becomes near-orthogonal to the
                # class-mean axis, so train AUROC sits at ~0.5 and its sign is noise.
                # A secondary arm must not be able to abort the primary run.
                tr_auc = _auroc(P @ d_pca, N @ d_pca)
                if tr_auc < 0.5 - 1e-9:
                    if not centred:
                        raise AssertionError(
                            f"sign orientation failed ({model_id}/{concept}/{site}/k={k}):"
                            f" train AUROC {tr_auc:.4f} < 0.5")
                    slot[f"dpca_{site}{tag}_degenerate"] = True
                slot[f"dpca_{site}{tag}_train_auroc"] = round(tr_auc, 6)
                slot[f"dpca_{site}{tag}"] = round(_auroc(p_ev @ d_pca, n_ev @ d_pca), 6)
                if not centred and k == max(b for b in BUDGETS if b <= len(pos_tr)):
                    out["cosines"][site] = round(float(abs(np.dot(d_dom, d_pca))), 6)
    return out


def _bootstrap_ci(vals: list[float], n_boot: int = 10000, seed: int = 42) -> dict:
    """Percentile bootstrap over MODELS. N=6 makes this wide — that is the honest picture."""
    if not vals:
        return {}
    rng = np.random.RandomState(seed)
    a = np.asarray(vals, dtype=float)
    boots = a[rng.randint(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "ci95_lo": float(np.percentile(boots, 2.5)),
            "ci95_hi": float(np.percentile(boots, 97.5)),
            "n_models": len(a)}


def summarise(rows: list[dict]) -> dict:
    """Per-arm curves, ceiling disclosure, and the pre-registered primary estimand."""
    out: dict = {"arms": {}, "ceiling": {}, "cosines": {}}
    for arm in ARMS + tuple(f"{a}_centred" for a in ARMS if a.startswith("dpca")):
        curve = {}
        for k in BUDGETS:
            vals = [r["budgets"][str(k)][arm] for r in rows
                    if str(k) in r["budgets"] and arm in r["budgets"][str(k)]]
            if vals:
                v = np.asarray(vals)
                curve[str(k)] = {"n": len(vals), "mean_auroc": float(v.mean()),
                                 "median_auroc": float(np.median(v)),
                                 "at_ceiling_rate": float(np.mean(v >= 0.99))}
        if curve:
            out["arms"][arm] = curve
    # Gate 5 — ceiling disclosure. If everything is >=0.99 everywhere, say so.
    out["ceiling"]["all_arms_saturated"] = bool(
        out["arms"] and all(c["at_ceiling_rate"] >= 1.0
                            for arm in ARMS if arm in out["arms"]
                            for c in out["arms"][arm].values()))
    for site in ("handoff", "peak"):
        vals = [r["cosines"][site] for r in rows if site in r.get("cosines", {})]
        if vals:
            out["cosines"][site] = {"n": len(vals), "mean_abs_cos": float(np.mean(vals)),
                                    "median_abs_cos": float(np.median(vals))}
    # --- the pre-registered primary estimand -------------------------------------
    a_arm, b_arm = PRIMARY
    per_model: dict[str, list[float]] = {}
    for r in rows:
        deltas = [r["budgets"][str(k)][a_arm] - r["budgets"][str(k)][b_arm]
                  for k in BUDGETS
                  if str(k) in r["budgets"]
                  and a_arm in r["budgets"][str(k)] and b_arm in r["budgets"][str(k)]]
        if deltas:                       # cell value = mean across the budget grid
            per_model.setdefault(r["model"], []).append(float(np.mean(deltas)))
    model_medians = {m: float(np.median(v)) for m, v in per_model.items()}
    ci = _bootstrap_ci(list(model_medians.values()))
    verdict = "indeterminate"
    if ci:
        lo, hi, mean = ci["ci95_lo"], ci["ci95_hi"], ci["mean"]
        if abs(mean) < EQUIV_MARGIN and lo > -0.02 and hi < 0.02:
            verdict = "practically_equivalent"
        elif mean >= EQUIV_MARGIN and lo > 0:
            verdict = "dpca_peak_better"
        elif mean <= -EQUIV_MARGIN and hi < 0:
            verdict = "dom_handoff_better"
    out["primary"] = {
        "contrast": f"{a_arm} - {b_arm}", "estimand": "per-cell delta averaged over "
        "budgets, per-model median, bootstrap CI over models",
        "equivalence_margin": EQUIV_MARGIN,
        "per_model_median_delta": model_medians, **ci, "verdict": verdict,
    }
    return out


def known_result_gate(rows: list[dict], m16_dir: Path, tol: float = 1e-6) -> dict:
    """Gate 4 — `dom_handoff` at full budget IS M16's `dom_auroc`. Same computation."""
    full = max(BUDGETS)
    res = {"artifact_dir": str(m16_dir), "tolerance": tol,
           "checked": 0, "mismatches": [], "status": "skipped"}
    if not m16_dir.is_dir():
        res["status"] = "artifact_missing"
        return res
    ref: dict[tuple[str, str], float] = {}
    for f in sorted(m16_dir.glob("*_direction_estimator.json")):
        for r in json.loads(f.read_text()).get("detail", []):
            if r.get("dom_auroc") is not None:
                ref[(r["model"], r["concept"])] = float(r["dom_auroc"])
    for r in rows:
        b = r["budgets"].get(str(full))
        if not b or "dom_handoff" not in b:
            continue
        key = (r["model"], r["concept"])
        if key not in ref:
            continue
        res["checked"] += 1
        if abs(b["dom_handoff"] - ref[key]) > tol:
            res["mismatches"].append({"model": key[0], "concept": key[1],
                                      "ours": b["dom_handoff"], "m16": ref[key]})
    res["status"] = ("pass" if res["checked"] and not res["mismatches"]
                     else "fail" if res["mismatches"] else "no_overlap")
    return res


def _provenance(models: list[str], eval_frac: float, seed: int) -> dict:
    def _sha(path: Path) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:                                            # noqa: BLE001
            return None
    import rosetta_tools
    return {
        "script": "gem/delta_pca_baseline.py",
        "analysis_repo_sha": _sha(Path(__file__).resolve().parent.parent),
        "rosetta_tools_sha": _sha(Path(rosetta_tools.__file__).resolve().parent.parent),
        "rosetta_tools_version": getattr(rosetta_tools, "__version__", None),
        "roster_source": "explicit module list DEFAULT_MODELS (never a directory glob)",
        "models": list(models),
        "n_pairs": N_PAIRS, "eval_frac": eval_frac, "seed": seed,
        "budgets_pairs": list(BUDGETS), "centred_default": False,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spec": "papers/gem/DELTA_PCA_RUN_SPEC.md (pre-registration frozen 2026-08-02)",
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", dest="models",
                    help="HF model id; repeatable. Default: §8.1's 6-model roster.")
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--no-clean-cache", action="store_true")
    ap.add_argument("--max-concepts", type=int, default=None,
                    help="Cap concepts per model. For the smoke pass on a cold host: "
                         "--model <one> --max-concepts 2 validates the whole path "
                         "(extraction, layers, arms, gate) in ~2 minutes before the "
                         "full roster is committed to.")
    ap.add_argument("--allow-gate-mismatch", action="store_true",
                    help="Do not abort on a known-result-gate failure. Debugging only — "
                         "a mismatch means the harness diverged and no number is readable.")
    a, _unknown = ap.parse_known_args()
    models = a.models or DEFAULT_MODELS

    out_dir = a.out_dir or (ROSETTA_RESULTS / "delta_pca_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    m16_dir: Path | None = None
    for mid in models:
        ext_dir = find_extraction_dir(mid)
        if ext_dir is None:
            log.warning("%s: no extraction dir — skipped", mid)
            continue
        if m16_dir is None:
            m16_dir = ext_dir.parent / "_p2_direction_estimator"
        concepts = discover_concepts(ext_dir)
        if not concepts:
            log.warning("%s: no concepts — skipped", mid)
            continue
        if a.max_concepts:
            concepts = concepts[:a.max_concepts]
        log.info("%s: %d concepts", mid, len(concepts))

        device = get_device(a.device)
        dtype = get_dtype(device)
        log_device_info(device, dtype)
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
            "model_id": mid, "n_cells": len(rows), "pair_aware_split": True,
            "provenance": _provenance([mid], a.eval_frac, a.seed),
            "summary": summarise(rows), "detail": rows,
        }
        # Upload per model on completion (C7/C11: a crash costs one model, not the run).
        (out_dir / f"{slug(mid)}_delta_pca.json").write_text(json.dumps(payload, indent=2))
        all_rows.extend(rows)
        full = str(max(BUDGETS))
        s = payload["summary"]["arms"]
        log.info("  @%s pairs  dom_handoff %.4f | dpca_peak %.4f | cos(handoff) %.3f",
                 full, s["dom_handoff"][full]["mean_auroc"],
                 s["dpca_peak"][full]["mean_auroc"],
                 payload["summary"]["cosines"].get("handoff", {}).get("mean_abs_cos",
                                                                      float("nan")))

    if not all_rows:
        log.error("no cells produced")
        return 1

    gate = known_result_gate(all_rows, m16_dir) if m16_dir else {"status": "skipped"}
    agg = {
        "n_cells": len(all_rows), "pair_aware_split": True,
        "provenance": _provenance(models, a.eval_frac, a.seed),
        "known_result_gate": gate,
        "peak_vs_global_argmax_differs": sum(
            1 for r in all_rows if r.get("global_argmax_layer") is not None
            and not r["peak_equals_global_argmax"]),
        "summary": summarise(all_rows), "detail": all_rows,
    }
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))

    s = agg["summary"]
    log.info("\n%d cells across %d models", len(all_rows),
             len({r["model"] for r in all_rows}))
    log.info("known-result gate: %s (%d cells checked, %d mismatches)",
             gate.get("status"), gate.get("checked", 0), len(gate.get("mismatches", [])))
    for arm in ARMS:
        if arm not in s["arms"]:
            continue
        curve = "  ".join(f"{k}:{s['arms'][arm][k]['mean_auroc']:.4f}"
                          for k in map(str, BUDGETS) if k in s["arms"][arm])
        log.info("  %-13s %s", arm, curve)
    p = s["primary"]
    log.info("primary %s: mean %+.4f  CI95 [%+.4f, %+.4f]  -> %s",
             p["contrast"], p.get("mean", float("nan")),
             p.get("ci95_lo", float("nan")), p.get("ci95_hi", float("nan")), p["verdict"])
    if s["ceiling"]["all_arms_saturated"]:
        log.warning("CEILING: every arm >=0.99 at every budget — per P.5 the AUROC "
                    "comparison is uninformative and the cosines are the primary result.")
    log.info("wrote %s", out_dir / "aggregate.json")

    if gate.get("status") == "fail" and not a.allow_gate_mismatch:
        log.error("KNOWN-RESULT GATE FAILED — the harness diverged from M16. "
                  "No other number in this run may be read. See aggregate.json.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

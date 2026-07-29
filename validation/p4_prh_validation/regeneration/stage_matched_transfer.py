#!/usr/bin/env python3
"""Stage-matched vs depth-matched arrangement transfer — Phase-A follow-on.

Question: when aligning two models, is the right layer-matching rule
proportional DEPTH (same f in both stacks) or GEM STAGE (same representational
event — this concept's shallowest/deepest node — wherever it sits)?

Anchors per (pair, fit-concept): the two stages that are defined for every
atlas regardless of node count (cluster-A first cut: node counts agree on
0/17 concepts, so node-k matching is ill-defined; first/deepest is not):
    first_peak, first_handoff, deepest_peak, deepest_handoff
(deduped when the atlas is single-node and layers coincide).

Per anchor, two conditions on identical machinery:
    STAGE: fit at each model's own anchor layer (ls_anchor, lt_anchor)
    DEPTH: fit at the anchors' mean proportional depth mapped into each model
Both measure same_true, cross_true (fit X, test Y at the fit layers), and a
K-draw within-class scramble on the STAGE condition. Rows record each side's
proportional depths and their divergence — cells where the models' anchors
disagree in depth are the informative ones (conditions coincide otherwise).

Usage: python stage_matched_transfer.py --cluster A --K 2 --out smt_A
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import common as C
from nullfloor_analysis import LETDIM, pairs_in_cluster, _seed
from depth_resolved_transfer import caz_meta, dom_at, cal_at, scramble_within_class


def log(m):
    print(m, flush=True)


_GEM = {}


def gem_nodes(slug, concept):
    """Depth-ordered node list [(peak_layer, handoff_layer|None), ...]."""
    if (slug, concept) not in _GEM:
        try:
            gem = json.load(open(C._hf(f"{C.HF_ROOT}/{slug}/gem_{concept}.json")))
            nodes = [(n.get("caz_peak"), n.get("handoff_layer"))
                     for n in gem.get("nodes", [])]
            _GEM[(slug, concept)] = [n for n in nodes if n[0] is not None]
        except Exception:
            _GEM[(slug, concept)] = []
    return _GEM[(slug, concept)]


def anchors_for(slug_s, slug_t, concept):
    """[(name, ls, lt)] for the four stage anchors, deduped."""
    ns, nt = gem_nodes(slug_s, concept), gem_nodes(slug_t, concept)
    if not ns or not nt:
        return []
    cand = [("first_peak", ns[0][0], nt[0][0]),
            ("first_handoff", ns[0][1], nt[0][1]),
            ("deepest_peak", ns[-1][0], nt[-1][0]),
            ("deepest_handoff", ns[-1][1], nt[-1][1])]
    out, seen = [], set()
    for name, ls, lt in cand:
        if ls is None or lt is None or (ls, lt) in seen:
            continue
        seen.add((ls, lt))
        out.append((name, ls, lt))
    return out


def measure(s, t, X, Y, ls, lt, stage, rng, K):
    """same_true, cross_true, same_scr, cross_scr at the given layers."""
    dom_sX, dom_tX = dom_at(s, X, ls), dom_at(t, X, lt)
    if dom_sX is None or dom_tX is None:
        return None
    dom_sY, dom_tY = dom_at(s, Y, ls), dom_at(t, Y, lt)
    cal_sX, cal_tX = cal_at(s, X, ls, stage), cal_at(t, X, lt, stage)
    same = C.aligned_cosine(dom_sX, dom_tX, cal_sX, cal_tX)
    cross = (C.aligned_cosine(dom_sY, dom_tY, cal_sX, cal_tX)
             if dom_sY is not None and dom_tY is not None else np.nan)
    same_scr, cross_scr = [], []
    for _ in range(K):
        ct = scramble_within_class(cal_tX, rng)
        same_scr.append(C.aligned_cosine(dom_sX, dom_tX, cal_sX, ct))
        if dom_sY is not None and dom_tY is not None:
            cross_scr.append(C.aligned_cosine(dom_sY, dom_tY, cal_sX, ct))
    return dict(same=float(same), cross=float(cross),
                same_scr=float(np.mean(same_scr)) if same_scr else float("nan"),
                cross_scr=float(np.mean(cross_scr)) if cross_scr else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", default="A")
    ap.add_argument("--K", type=int, default=2, help="scramble draws (STAGE condition)")
    ap.add_argument("--out", default="smt_out")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--max-combos", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(os.environ.get("P4_REGEN_STAGE", "./_p4_stage"))
    stage_dir.mkdir(parents=True, exist_ok=True)

    L = a.cluster
    d = LETDIM[L]
    prs = pairs_in_cluster(L)
    cons = list(C.CONCEPTS_17)
    combos = [(cons[i], cons[(i + 1) % len(cons)]) for i in range(len(cons))]
    if a.max_pairs:
        prs = prs[: a.max_pairs]
    if a.max_combos:
        combos = combos[: a.max_combos]
    rng = np.random.default_rng(_seed(f"{L}:stage", 9500))

    log(f"=== STAGE-MATCHED vs DEPTH-MATCHED: cluster {L} (d={d}), "
        f"{len(prs)} pairs x {len(combos)} combos, K={a.K} ===")

    rows = []
    t0 = time.time()
    for ci, (X, Y) in enumerate(combos):
        for (s, t) in prs:
            _, Ls = caz_meta(s, X)
            _, Lt = caz_meta(t, X)
            for name, ls, lt in anchors_for(s, t, X):
                fs, ft = ls / (Ls - 1), lt / (Lt - 1)
                fmean = (fs + ft) / 2
                ls2, lt2 = round(fmean * (Ls - 1)), round(fmean * (Lt - 1))
                try:
                    st_res = measure(s, t, X, Y, ls, lt, stage_dir, rng, a.K)
                    if st_res is None:
                        continue
                    dp_res = measure(s, t, X, Y, ls2, lt2, stage_dir, rng, 0)
                    if dp_res is None:
                        continue
                except Exception as e:
                    log(f"  [{L}] skip {s} x {t} / {X} @{name}: "
                        f"{type(e).__name__}: {e}")
                    continue
                rows.append(dict(
                    s=s, t=t, fit=X, test=Y, anchor=name,
                    ls=ls, lt=lt, fs=fs, ft=ft, f_div=abs(fs - ft),
                    ls_dm=ls2, lt_dm=lt2,
                    stage_same=st_res["same"], stage_cross=st_res["cross"],
                    stage_same_scr=st_res["same_scr"],
                    stage_cross_scr=st_res["cross_scr"],
                    depth_same=dp_res["same"], depth_cross=dp_res["cross"],
                ))
        log(f"  [{L}] {ci + 1}/{len(combos)} fit={X}  rows={len(rows)} "
            f"({time.time() - t0:.0f}s)")
        tmp = out / "stage_matched.json.tmp"
        json.dump(dict(cluster=L, d=d, K=a.K, rows=rows), open(tmp, "w"))
        os.replace(tmp, out / "stage_matched.json")

    m = {}
    for k in ("stage_same", "stage_cross", "depth_same", "depth_cross",
              "stage_cross_scr", "f_div"):
        m[k] = float(np.nanmean([r[k] for r in rows]))
    log(f"  ==> [{L}] STAGE same {m['stage_same']:.4f} cross {m['stage_cross']:.4f} "
        f"(scr {m['stage_cross_scr']:.4f}) | DEPTH same {m['depth_same']:.4f} "
        f"cross {m['depth_cross']:.4f} | mean anchor divergence {m['f_div']:.3f}")
    div = [r for r in rows if r["f_div"] > 0.08]
    if div:
        sc = float(np.nanmean([r["stage_cross"] for r in div]))
        dc = float(np.nanmean([r["depth_cross"] for r in div]))
        log(f"  ==> [{L}] divergent-anchor cells (f_div>0.08, n={len(div)}): "
            f"STAGE cross {sc:.4f} vs DEPTH cross {dc:.4f}")
    log(f"DONE {len(rows)} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

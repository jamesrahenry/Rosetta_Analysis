#!/usr/bin/env python3
"""The full 17x17 arrangement-transport matrix — one rotation, whole constellation.

James's question (2026-08-02): "Is it 2/3rds to Y, and Z, and W ... ALL with one
rotation?" The cyclic Phase-A design tested each fitted rotation on ONE untouched
concept. This measures the whole matrix: for each (model pair, fit concept X),
fit the rotation ONCE on X's calibration clouds, then transport EVERY concept's
DOM — X itself (diagonal) and all 16 others — read at the fit layers (the `_L`
matched-layer convention). A K-draw within-class scramble of the target
calibration gives the correspondence-destroyed matrix on identical machinery.

Prediction if arrangements are truly shared: off-diagonal ~0.6-0.7 everywhere
(not just cyclic neighbours), collapsing to the inheritance floor under the
scrambled fit.

Efficiency: the fit (QR + orthogonal Procrustes) is computed once per
(pair, X[, draw]) and applied to all 17 DOMs — `fit_map`/`apply_map` replicate
`common.aligned_cosine` exactly (asserted on the diagonal at runtime).

Usage: python crossconcept_matrix.py --cluster B --K 2 --out ccmat_B
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

import common as C
from nullfloor_analysis import LETDIM, pairs_in_cluster, _seed
from crossconcept_scramble import dom_at_layer, scramble_within_class


def log(m):
    print(m, flush=True)


def fit_map(cal_s, cal_t):
    """The rank-reduced Procrustes fit from common.aligned_cosine, returned as
    (Q, Rq) so one fit can transport many directions."""
    sc = np.asarray(cal_s, np.float64); sc = sc - sc.mean(0)
    tc = np.asarray(cal_t, np.float64); tc = tc - tc.mean(0)
    Q, _ = np.linalg.qr(np.hstack([tc.T, sc.T]))
    Rq, _ = orthogonal_procrustes(tc @ Q, sc @ Q)
    return Q, Rq


def apply_map(dom_s, dom_t, Q, Rq):
    return C.cosine(dom_s @ Q, (np.asarray(dom_t, np.float64).ravel() @ Q) @ Rq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", default="B")
    ap.add_argument("--K", type=int, default=2, help="scrambled-fit draws")
    ap.add_argument("--out", default="ccmat_out")
    ap.add_argument("--max-pairs", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    L = a.cluster
    d = LETDIM[L]
    prs = pairs_in_cluster(L)
    if a.max_pairs and len(prs) > a.max_pairs:
        step = len(prs) / a.max_pairs
        prs = [prs[int(i * step)] for i in range(a.max_pairs)]
    cons = list(C.CONCEPTS_17)
    rng = np.random.default_rng(_seed(f"{L}:matrix", 9900))

    cache = {}

    def cal_and_peak(slug, con):
        if (slug, con) not in cache:
            dom, pk = C.load_dom_and_peak(slug, con)
            cache[(slug, con)] = (dom, pk, C.load_calibration(slug, con, pk))
        return cache[(slug, con)]

    log(f"=== FULL 17x17 TRANSPORT MATRIX: cluster {L} (d={d}), "
        f"{len(prs)} pairs x {len(cons)} fits x {len(cons)} tests, K={a.K} ===")
    rows = []
    t0 = time.time()
    for xi, X in enumerate(cons):
        for (s, t) in prs:
            try:
                dom_sX, pk_s, cal_sX = cal_and_peak(s, X)
                dom_tX, pk_t, cal_tX = cal_and_peak(t, X)
                if dom_sX is None or dom_tX is None:
                    continue
                Q, Rq = fit_map(cal_sX, cal_tX)
                # runtime identity check vs the canonical operator (diagonal)
                diag = apply_map(dom_sX, dom_tX, Q, Rq)
                ref = C.aligned_cosine(dom_sX, dom_tX, cal_sX, cal_tX)
                assert abs(diag - ref) < 1e-9, (diag, ref)
                scr_maps = []
                for _ in range(a.K):
                    ct = scramble_within_class(cal_tX, rng)
                    scr_maps.append(fit_map(cal_sX, ct))
                cell = dict(s=s, t=t, fit=X, diag=float(diag), tests={})
                for Y in cons:
                    if Y == X:
                        continue
                    dom_sY = dom_at_layer(s, Y, pk_s)
                    dom_tY = dom_at_layer(t, Y, pk_t)
                    if dom_sY is None or dom_tY is None:
                        continue
                    tru = apply_map(dom_sY, dom_tY, Q, Rq)
                    scr = float(np.mean([apply_map(dom_sY, dom_tY, q, r)
                                         for q, r in scr_maps]))
                    ov = float(np.nanmean([abs(C.raw_cosine(dom_sX, dom_sY)),
                                           abs(C.raw_cosine(dom_tX, dom_tY))]))
                    cell["tests"][Y] = dict(true=float(tru), scr=scr, overlap=ov)
            except Exception as e:
                log(f"  [{L}] skip {s} x {t} / fit={X}: {type(e).__name__}: {e}")
                continue
            rows.append(cell)
        done = sum(len(r["tests"]) for r in rows)
        log(f"  [{L}] fit {xi + 1}/{len(cons)} ({X})  cells={done} "
            f"({time.time() - t0:.0f}s)")
        tmp = out / "matrix.json.tmp"
        json.dump(dict(cluster=L, d=d, K=a.K, rows=rows), open(tmp, "w"))
        os.replace(tmp, out / "matrix.json")

    allt = [v["true"] for r in rows for v in r["tests"].values()]
    alls = [v["scr"] for r in rows for v in r["tests"].values()]
    diags = [r["diag"] for r in rows]
    log(f"  ==> [{L}] diagonal {np.mean(diags):.4f} | OFF-DIAGONAL true "
        f"{np.mean(allt):.4f} (scr {np.mean(alls):.4f}) | ratio "
        f"{np.mean(allt) / np.mean(diags):.3f} | {len(allt)} off-diag cells")


if __name__ == "__main__":
    main()

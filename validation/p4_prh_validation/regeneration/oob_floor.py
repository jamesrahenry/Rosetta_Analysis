#!/usr/bin/env python3
"""In-basis vs out-of-basis spectrum floor — full coverage (Phase-A audit gap 1).

PHASE_A §4.1's central control — same-concept (in-basis) floor 0.4333 vs
cross-concept (out-of-basis) floor 0.0004, "a ~1000x gap at identical
dimensions, spectra and machinery" — came from a 12-cell run whose producing
script never made it into the repo. This is that script, written down and run
at full coverage.

Per (pair, cyclic X->Y combo, K draws), all surrogates carry REAL spectra with
independent random bases (nullfloor machinery):
  in_basis   surrogate X-calibrations ss/tt; their own class-mean DOMs
             transported through R fit on ss/tt. The DOM is a functional of
             the fitting data — the P4-B2 mechanism.
  out_basis  surrogate Y-calibrations (Y's real spectra, independent draws);
             their DOMs transported through the SAME R fit on ss/tt. Nothing
             the fit ever saw — the honest chance level for cross-concept
             transport.

Usage: python oob_floor.py --cluster B --K 3 --out oob_B [--max-pairs N]
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import common as C
from nullfloor_analysis import (LETDIM, pairs_in_cluster, _seed,
                                spectrum_surrogate, real_spectrum, dom_from)


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", default="B")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--out", default="oob_out")
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
    combos = [(cons[i], cons[(i + 1) % len(cons)]) for i in range(len(cons))]
    rng = np.random.default_rng(_seed(f"{L}:oob", 9800))

    cache = {}

    def spectrum_of(slug, con):
        if (slug, con) not in cache:
            _, pk = C.load_dom_and_peak(slug, con)
            cal = C.load_calibration(slug, con, pk)
            cache[(slug, con)] = (real_spectrum(cal), cal.shape[0])
        return cache[(slug, con)]

    log(f"=== IN-BASIS vs OUT-OF-BASIS floor: cluster {L} (d={d}), "
        f"{len(prs)} pairs x {len(combos)} combos, K={a.K} ===")
    rows = []
    t0 = time.time()
    for ci, (X, Y) in enumerate(combos):
        for (s, t) in prs:
            try:
                sv_sX, n_sX = spectrum_of(s, X)
                sv_tX, n_tX = spectrum_of(t, X)
                sv_sY, n_sY = spectrum_of(s, Y)
                sv_tY, n_tY = spectrum_of(t, Y)
                inb, oob = [], []
                for _ in range(a.K):
                    ss = spectrum_surrogate(sv_sX, n_sX, d, rng)
                    tt = spectrum_surrogate(sv_tX, n_tX, d, rng)
                    hs, ht = ss.shape[0] // 2, tt.shape[0] // 2
                    dss = dom_from(ss, slice(0, hs), slice(hs, ss.shape[0]))
                    dtt = dom_from(tt, slice(0, ht), slice(ht, tt.shape[0]))
                    inb.append(C.aligned_cosine(dss, dtt, ss, tt))
                    ssY = spectrum_surrogate(sv_sY, n_sY, d, rng)
                    ttY = spectrum_surrogate(sv_tY, n_tY, d, rng)
                    hys, hyt = ssY.shape[0] // 2, ttY.shape[0] // 2
                    dYs = dom_from(ssY, slice(0, hys), slice(hys, ssY.shape[0]))
                    dYt = dom_from(ttY, slice(0, hyt), slice(hyt, ttY.shape[0]))
                    oob.append(C.aligned_cosine(dYs, dYt, ss, tt))
            except Exception as e:
                log(f"  [{L}] skip {s} x {t} / {X}->{Y}: {type(e).__name__}: {e}")
                continue
            rows.append(dict(s=s, t=t, fit=X, test=Y,
                             in_basis=float(np.mean(inb)),
                             out_basis=float(np.mean(oob))))
        log(f"  [{L}] {ci + 1}/{len(combos)} fit={X}  rows={len(rows)} "
            f"({time.time() - t0:.0f}s)")
        tmp = out / "oob_floor.json.tmp"
        json.dump(dict(cluster=L, d=d, K=a.K, rows=rows), open(tmp, "w"))
        os.replace(tmp, out / "oob_floor.json")

    ib = float(np.mean([r["in_basis"] for r in rows]))
    ob = float(np.mean([r["out_basis"] for r in rows]))
    log(f"  ==> [{L}] in-basis floor {ib:.4f} | out-of-basis floor {ob:.4f} "
        f"| ratio {ib / ob if ob else float('inf'):.0f}x  (rows {len(rows)})")


if __name__ == "__main__":
    main()

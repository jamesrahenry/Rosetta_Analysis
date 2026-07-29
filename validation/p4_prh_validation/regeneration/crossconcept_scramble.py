#!/usr/bin/env python3
"""Cross-concept transport under a SCRAMBLED fit (FROZEN_REVIEW_ROUND1 P4-B3).

The decisive test of whether the cross-concept signal depends on cross-architecture
text correspondence.

Established so far:
  * same-concept transport   real 0.971  vs in-basis  floor 0.4333
  * cross-concept transport  real 0.302  vs out-of-basis floor 0.0004
  * within-class scramble    true 0.9784 -> scramble 0.9892 (correspondence is
    not merely unnecessary for the SAME-concept metric, it is counterproductive)

Open question this script settles: the scramble was only ever run against the
same-concept metric. If cross-concept transport also survives a scrambled fit,
the "genuine shared global structure" reading of the 0.302 is independent of
correspondence and stands clean. If it collapses toward zero, the 0.302 was
riding on exactly what the within-class scramble undermines.

Four conditions per (model pair, fit-concept X, test-concept Y):
  same_true   R fit on X, true row pairing      -> transport X's DOM
  same_scr    R fit on X, rows scrambled in-class -> transport X's DOM
  cross_true  R fit on X, true row pairing      -> transport Y's DOM
  cross_scr   R fit on X, rows scrambled in-class -> transport Y's DOM

DOM vectors are never modified. Within-class permutation leaves each class mean
exactly unchanged, so the DOMs are identical between the true and scrambled
conditions by construction -- only R differs. That is what makes this a clean
test of R rather than of the directions.

Layer handling: concept Y's peak layer generally differs from X's. A rotation
fit at X's peak layer is only meaningful for vectors living at that same layer,
so the primary statistic (`_L`) reads Y's DOM at *X's* peak layer. The variant
using Y's own peak layer (`_P`) is also recorded, because that is what the
earlier bounded cross-concept run reported (0.302) and comparability matters.

Usage:
  python crossconcept_scramble.py [--cluster B] [--K 8] [--out ccscr_out]
"""
import json, time, argparse
from pathlib import Path
import numpy as np
import common as C
from nullfloor_analysis import LETDIM, pairs_in_cluster, _seed


def log(m):
    print(m, flush=True)


def dom_at_layer(slug, concept, layer):
    """DOM vector for `concept` read at a specified layer (None if absent)."""
    caz = json.load(open(C._hf(f"{C.HF_ROOT}/{slug}/caz_{concept}.json")))
    for m in caz["layer_data"]["metrics"]:
        if m["layer"] == layer and "dom_vector" in m:
            return np.asarray(m["dom_vector"], np.float64)
    return None


def scramble_within_class(cal, rng):
    """Permute rows within each class block. Leaves both class means unchanged."""
    n = cal.shape[0]; h = n // 2
    out = cal.copy()
    out[:h] = cal[rng.permutation(h)]
    out[h:] = cal[rng.permutation(n - h) + h]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", default="B")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--out", default="ccscr_out")
    ap.add_argument("--max-pairs", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    L = a.cluster; d = LETDIM[L]
    prs = pairs_in_cluster(L)
    if a.max_pairs and len(prs) > a.max_pairs:
        step = len(prs) / a.max_pairs
        prs = [prs[int(i * step)] for i in range(a.max_pairs)]
    # full concept coverage: every concept serves as the fit concept X once,
    # tested against the next concept cyclically.
    cons = list(C.CONCEPTS_17)
    combos = [(cons[i], cons[(i + 1) % len(cons)]) for i in range(len(cons))]

    rng = np.random.default_rng(_seed(L, 7000))
    log(f"=== CROSS-CONCEPT under SCRAMBLED FIT: cluster {L} (d={d}), "
        f"{len(prs)} pairs x {len(combos)} combos, K={a.K} ===")

    rows = []
    t0 = time.time()
    cache = {}

    def cal_and_peak(slug, con):
        if (slug, con) not in cache:
            dom, pk = C.load_dom_and_peak(slug, con)
            cache[(slug, con)] = (dom, pk, C.load_calibration(slug, con, pk))
        return cache[(slug, con)]

    for ci, (X, Y) in enumerate(combos):
        for (s, t) in prs:
            try:
                dom_sX, pk_s, cal_sX = cal_and_peak(s, X)
                dom_tX, pk_t, cal_tX = cal_and_peak(t, X)
                if dom_sX is None or dom_tX is None:
                    continue
                # Y's DOM at X's peak layer (primary) and at Y's own peak (variant)
                dom_sY_L = dom_at_layer(s, Y, pk_s)
                dom_tY_L = dom_at_layer(t, Y, pk_t)
                dom_sY_P, _ = C.load_dom_and_peak(s, Y)
                dom_tY_P, _ = C.load_dom_and_peak(t, Y)
            except Exception:
                continue

            # A mismatched calibration row count (e.g. 498/499 vs 500 from the
            # blank-text row-alignment cases) makes the Procrustes fit throw.
            # The primary pipeline excludes exactly these as "unavailable fits"
            # (§3.1 corpus note); skip the cell here too, same as
            # nullfloor_analysis.py. The throw happens before any rng draw, so
            # surviving cells' scramble draws are unchanged.
            try:
                same_true = C.aligned_cosine(dom_sX, dom_tX, cal_sX, cal_tX)
                cross_true_L = (C.aligned_cosine(dom_sY_L, dom_tY_L, cal_sX, cal_tX)
                                if dom_sY_L is not None and dom_tY_L is not None else np.nan)
                cross_true_P = (C.aligned_cosine(dom_sY_P, dom_tY_P, cal_sX, cal_tX)
                                if dom_sY_P is not None and dom_tY_P is not None else np.nan)
            except Exception as e:
                log(f"  [{L}] skip {s} x {t} / fit={X} test={Y}: {type(e).__name__}: {e}")
                continue

            # --- inheritance controls -------------------------------------
            # If Y is not orthogonal to X *within* each model, R (fit to carry X)
            # transports Y's X-component for free, and cross-concept transport is
            # inherited rather than evidence of global structure. Under pure
            # inheritance, cross_true should track `overlap`, not exceed it.
            ov_s = float(C.raw_cosine(dom_sX, dom_sY_L)) if dom_sY_L is not None else np.nan
            ov_t = float(C.raw_cosine(dom_tX, dom_tY_L)) if dom_tY_L is not None else np.nan
            # unrotated cross-model baselines, BEFORE any rotation, for both the
            # fit concept and the test concept. If raw_same > raw_cross, the two
            # models already agree more about the SAME concept than about a
            # different one with no Procrustes involved at all -- correspondence
            # evidence that owes nothing to the fit. Both are expected near 0
            # (the paper reports pre-rotation ~0), so any gap is informative.
            raw_same = float(C.raw_cosine(dom_sX, dom_tX))
            raw_L = (float(C.raw_cosine(dom_sY_L, dom_tY_L))
                     if dom_sY_L is not None and dom_tY_L is not None else np.nan)

            same_scr, cross_scr_L, cross_scr_P = [], [], []
            for _ in range(a.K):
                ct = scramble_within_class(cal_tX, rng)
                same_scr.append(C.aligned_cosine(dom_sX, dom_tX, cal_sX, ct))
                if dom_sY_L is not None and dom_tY_L is not None:
                    cross_scr_L.append(C.aligned_cosine(dom_sY_L, dom_tY_L, cal_sX, ct))
                if dom_sY_P is not None and dom_tY_P is not None:
                    cross_scr_P.append(C.aligned_cosine(dom_sY_P, dom_tY_P, cal_sX, ct))

            rows.append(dict(
                cluster=L, s=s, t=t, fit=X, test=Y,
                same_true=float(same_true),
                same_scr=float(np.mean(same_scr)),
                cross_true_L=float(cross_true_L),
                cross_scr_L=float(np.mean(cross_scr_L)) if cross_scr_L else float("nan"),
                cross_true_P=float(cross_true_P),
                cross_scr_P=float(np.mean(cross_scr_P)) if cross_scr_P else float("nan"),
                overlap_s=ov_s, overlap_t=ov_t, overlap_abs=float(np.nanmean([abs(ov_s), abs(ov_t)])),
                raw_same=raw_same, raw_cross_L=raw_L,
            ))
        log(f"  [{L}] {ci+1}/{len(combos)} fit={X} test={Y}  "
            f"cells={len(rows)} ({time.time()-t0:.0f}s)")
        _write(out, L, d, prs, a.K, rows)

    _write(out, L, d, prs, a.K, rows)
    KEYS = ("same_true", "same_scr", "cross_true_L", "cross_scr_L",
            "cross_true_P", "cross_scr_P", "overlap_abs", "raw_same", "raw_cross_L")
    m = {k: float(np.nanmean([r[k] for r in rows])) for k in KEYS}
    log(f"  ==> [{L}] same  true {m['same_true']:.4f} -> scr {m['same_scr']:.4f}")
    log(f"  ==> [{L}] cross(layer-X) true {m['cross_true_L']:.4f} -> scr {m['cross_scr_L']:.4f}")
    log(f"  ==> [{L}] cross(peak-Y)  true {m['cross_true_P']:.4f} -> scr {m['cross_scr_P']:.4f}")
    log(f"  ==> [{L}] INHERITANCE CONTROL  |cos(dom_X,dom_Y)| within-model {m['overlap_abs']:.4f}")
    log(f"  ==> [{L}] PRE-ROTATION baselines  same-concept {m['raw_same']:+.4f} | "
        f"cross-concept {m['raw_cross_L']:+.4f}")
    log("  (reference: out-of-basis cross-concept floor = 0.0004)")
    log("  Read: cross_true_L >> overlap_abs => structure beyond inheritance;")
    log("        cross_true_L ~ overlap_abs  => cross-concept signal is inherited.")


def _write(out, L, d, prs, K, rows):
    if not rows:
        return
    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in
           ("same_true", "same_scr", "cross_true_L", "cross_scr_L",
            "cross_true_P", "cross_scr_P", "overlap_abs", "raw_same", "raw_cross_L")}
    (out / "crossconcept_scramble.json").write_text(json.dumps(
        dict(cluster=L, d=d, n_pairs=len(prs), K=K, n_cells=len(rows),
             **agg, rows=rows), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Depth-resolved arrangement transfer — Phase-A follow-on pilot (cluster A).

For each proportional depth f, fit the per-pair rotation R at layer l_M(f) =
round(f * (L_M - 1)) in each model, and measure — all at that depth —
  same_true   : same-concept aligned cosine (the headline metric)
  same_scr    : same, under within-class scramble of the target calibration
  cross_true  : cross-concept transport (fit on X, test Y at the SAME layers)
  cross_scr   : cross-concept under the scrambled fit
  floor       : spectrum-matched same-concept floor (Phase-A §1 machinery)
Output: cL/same(f), margin(f), and scramble collapse per depth — plus each
model's GEM segment boundaries (proportional) recorded for overlay, to ask
whether transfer structure aligns with handoff structure.

Phase-A discipline baked in: per-layer spectrum floors, deterministic FNV
seeds, per-cell guard (mismatched calibrations skip, logged), incremental
writes that record actual coverage counts.

IMPORTANT loader note: common.load_calibration returns the STORED peak-layer
file for A-E models regardless of the layer argument, so this script has its
own slicer that always reads the requested layer from the all-layer array.
Staged all-layer files are KEPT (not deleted) so a 10-depth sweep downloads
each (model, concept) once; cluster-A total is ~1.3 GB.

Usage:
  python depth_resolved_transfer.py --cluster A --out drt_A \\
      --depths 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 --K 3 --floorK 3
Split across streams with disjoint --depths lists; outputs merge by row.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import common as C
from nullfloor_analysis import LETDIM, pairs_in_cluster, _seed, spectrum_surrogate


def log(m):
    print(m, flush=True)


# ── per-layer data access ───────────────────────────────────────────────────
_CAZ = {}


def caz_meta(slug, concept):
    """(metrics-by-layer dict, n_layers) from the cached caz json."""
    if (slug, concept) not in _CAZ:
        caz = json.load(open(C._hf(f"{C.HF_ROOT}/{slug}/caz_{concept}.json")))
        mets = {m["layer"]: m for m in caz["layer_data"]["metrics"]}
        _CAZ[(slug, concept)] = (mets, max(mets) + 1)
    return _CAZ[(slug, concept)]


def dom_at(slug, concept, layer):
    mets, _ = caz_meta(slug, concept)
    m = mets.get(layer)
    if m is None or "dom_vector" not in m:
        return None
    return np.asarray(m["dom_vector"], np.float64)


def cal_at(slug, concept, layer, stage):
    """Calibration [n, d] at an arbitrary layer, always from the all-layer
    array (never the stored peak file — see module docstring). Per-layer
    slices are cached under $P4_PEAK_CACHE; the staged all-layer file is kept
    for subsequent layers."""
    cache_dir = os.environ.get("P4_PEAK_CACHE")
    cpath = None
    if cache_dir:
        cpath = Path(cache_dir) / f"{slug}__{concept}__L{layer}.npy"
        if cpath.exists():
            return np.load(cpath).astype(np.float64)
        cpath.parent.mkdir(parents=True, exist_ok=True)
    big = C._hf(f"{C.HF_ROOT}/{slug}/calibration_alllayer_{concept}.npy",
                local_dir=str(stage))
    arr = np.load(big, mmap_mode="r")
    cal = np.array(arr[layer], dtype=np.float64)
    del arr
    if cpath is not None:
        np.save(cpath, cal.astype(np.float32))
    return cal


def gem_segments(slug, concept):
    """Proportional [start, end] per GEM node, for boundary overlay (meta only)."""
    try:
        gem = json.load(open(C._hf(f"{C.HF_ROOT}/{slug}/gem_{concept}.json")))
    except Exception:
        return None
    _, L = caz_meta(slug, concept)
    segs = []
    for node in gem.get("nodes", []):
        if "caz_start" in node and "caz_end" in node and L > 1:
            segs.append({
                "start": node["caz_start"] / (L - 1),
                "peak": node.get("caz_peak", node["caz_start"]) / (L - 1),
                "end": node["caz_end"] / (L - 1),
                "handoff": (node["handoff_layer"] / (L - 1)
                            if node.get("handoff_layer") is not None else None),
            })
    return segs or None


def scramble_within_class(cal, rng):
    n = cal.shape[0]
    h = n // 2
    out = cal.copy()
    out[:h] = cal[rng.permutation(h)]
    out[h:] = cal[rng.permutation(n - h) + h]
    return out


def dom_from(mat, a, b):
    return mat[a].mean(0) - mat[b].mean(0)


def _write(out, payload):
    tmp = out / "depth_transfer.json.tmp"
    json.dump(payload, open(tmp, "w"))
    os.replace(tmp, out / "depth_transfer.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", default="A")
    ap.add_argument("--depths", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--K", type=int, default=3, help="scramble draws per cell")
    ap.add_argument("--floorK", type=int, default=3, help="spectrum-floor draws per cell")
    ap.add_argument("--out", default="drt_out")
    ap.add_argument("--max-pairs", type=int, default=None, help="smoke-test cap")
    ap.add_argument("--max-combos", type=int, default=None, help="smoke-test cap")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stage = Path(os.environ.get("P4_REGEN_STAGE", "./_p4_stage"))
    stage.mkdir(parents=True, exist_ok=True)

    L = a.cluster
    d = LETDIM[L]
    prs = pairs_in_cluster(L)
    cons = list(C.CONCEPTS_17)
    combos = [(cons[i], cons[(i + 1) % len(cons)]) for i in range(len(cons))]
    if a.max_pairs:
        prs = prs[: a.max_pairs]
    if a.max_combos:
        combos = combos[: a.max_combos]
    depths = [float(x) for x in a.depths.split(",")]
    rng = np.random.default_rng(_seed(f"{L}:depth", 9000))

    slugs = sorted({s for p in prs for s in p})
    meta = {"cluster": L, "d": d, "depths": depths, "K": a.K, "floorK": a.floorK,
            "n_layers": {}, "gem_segments": {}}
    for s in slugs:
        _, nl = caz_meta(s, cons[0])
        meta["n_layers"][s] = nl
        meta["gem_segments"][s] = {con: gem_segments(s, con) for con in cons}

    log(f"=== DEPTH-RESOLVED TRANSFER: cluster {L} (d={d}), {len(prs)} pairs x "
        f"{len(combos)} combos x {len(depths)} depths, K={a.K}/floorK={a.floorK} ===")
    log(f"    layers: { {s: meta['n_layers'][s] for s in slugs} }")

    rows = []
    t0 = time.time()
    for f in depths:
        for ci, (X, Y) in enumerate(combos):
            for (s, t) in prs:
                ls = round(f * (meta["n_layers"][s] - 1))
                lt = round(f * (meta["n_layers"][t] - 1))
                try:
                    dom_sX = dom_at(s, X, ls)
                    dom_tX = dom_at(t, X, lt)
                    dom_sY = dom_at(s, Y, ls)
                    dom_tY = dom_at(t, Y, lt)
                    if dom_sX is None or dom_tX is None:
                        continue
                    cal_sX = cal_at(s, X, ls, stage)
                    cal_tX = cal_at(t, X, lt, stage)

                    same_true = C.aligned_cosine(dom_sX, dom_tX, cal_sX, cal_tX)
                    cross_true = (C.aligned_cosine(dom_sY, dom_tY, cal_sX, cal_tX)
                                  if dom_sY is not None and dom_tY is not None
                                  else np.nan)
                    same_scr, cross_scr = [], []
                    for _ in range(a.K):
                        ct = scramble_within_class(cal_tX, rng)
                        same_scr.append(C.aligned_cosine(dom_sX, dom_tX, cal_sX, ct))
                        if dom_sY is not None and dom_tY is not None:
                            cross_scr.append(C.aligned_cosine(dom_sY, dom_tY, cal_sX, ct))
                    sv_s = np.linalg.svd(cal_sX - cal_sX.mean(0), compute_uv=False)
                    sv_t = np.linalg.svd(cal_tX - cal_tX.mean(0), compute_uv=False)
                    floor = []
                    for _ in range(a.floorK):
                        ss = spectrum_surrogate(sv_s, cal_sX.shape[0], d, rng)
                        tt = spectrum_surrogate(sv_t, cal_tX.shape[0], d, rng)
                        hs, ht = ss.shape[0] // 2, tt.shape[0] // 2
                        dss = dom_from(ss, slice(0, hs), slice(hs, ss.shape[0]))
                        dtt = dom_from(tt, slice(0, ht), slice(ht, tt.shape[0]))
                        floor.append(C.aligned_cosine(dss, dtt, ss, tt))
                except Exception as e:
                    log(f"  [{L} f={f}] skip {s} x {t} / fit={X}: "
                        f"{type(e).__name__}: {e}")
                    continue
                rows.append(dict(
                    f=f, s=s, t=t, fit=X, test=Y, ls=ls, lt=lt,
                    same_true=float(same_true),
                    same_scr=float(np.mean(same_scr)),
                    cross_true=float(cross_true),
                    cross_scr=float(np.mean(cross_scr)) if cross_scr else float("nan"),
                    floor=float(np.mean(floor)),
                ))
            log(f"  [{L} f={f}] {ci + 1}/{len(combos)} fit={X}  rows={len(rows)} "
                f"({time.time() - t0:.0f}s)")
            _write(out, dict(meta=meta, rows=rows))
        sub = [r for r in rows if r["f"] == f]
        if sub:
            m = {k: float(np.nanmean([r[k] for r in sub]))
                 for k in ("same_true", "same_scr", "cross_true", "cross_scr", "floor")}
            log(f"  ==> [{L} f={f}] same {m['same_true']:.4f} (scr {m['same_scr']:.4f}) "
                f"| cross {m['cross_true']:.4f} (scr {m['cross_scr']:.4f}) "
                f"| floor {m['floor']:.4f} | cells {len(sub)}")

    _write(out, dict(meta=meta, rows=rows))
    log(f"DONE {len(rows)} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

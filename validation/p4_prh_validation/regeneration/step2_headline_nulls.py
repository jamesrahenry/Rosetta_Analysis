#!/usr/bin/env python3
"""
step2_headline_nulls.py — regenerate P4's headline null battery over the full
A–F corpus, from the published HF artifacts.

Computes, from scratch (no pooling with stored values):
  - Permuted-label null (§3.2): shuffle pos/neg labels, recompute the DOM,
    apply the true-label rotation, measure aligned cosine. 25 trials per
    (pair, concept). Reports pooled mean/SD/n and z = primary / SEM.
  - Universality ratio (§3.3): fit R on concept A's calibration, apply to
    concept B's DOM; ratio = mean cross-concept transfer / mean same-concept.
  - Peak-level depth matching (§3.4): matched (|Δdepth| < 0.15) vs mismatched
    same-concept alignment.

CPU only; uses the rank-reduced Procrustes from common.py (validated exact).
The permuted null is the long pole (~44k reduced solves); expect tens of
minutes. Calibration is the DOM/label-layout convention "rows 0..n/2 = positive,
n/2..n = negative" recorded in the corpus `_meta` layout.

Outputs (to --out-dir): step2_nulls.json

Expected (2026-07-18, A–F): permuted-null pooled ≈ −0.0009, SD ≈ 0.199,
n = 44,050, z ≈ 1030 vs primary 0.9752; universality ratio ≈ 0.205;
peak-depth Δ ≈ +0.008 (near-null).

Usage:
  python step2_headline_nulls.py --which permuted universality peak_depth
  python step2_headline_nulls.py --which universality      # one test
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np

import common as C

PRIMARY_MEAN = 0.9752  # from step1 (A–F); z uses this vs the null's own SEM


def _basis(cal_s, cal_t):
    sc = cal_s - cal_s.mean(0)
    tc = cal_t - cal_t.mean(0)
    Q, _ = np.linalg.qr(np.hstack([tc.T, sc.T]))
    return sc, tc, Q


def _dom_from_labels(cal, perm=None):
    n = len(cal); h = n // 2
    if perm is None:
        pos, neg = cal[:h], cal[h:2 * h]
    else:
        pos, neg = cal[perm[:h]], cal[perm[h:2 * h]]
    d = pos.mean(0) - neg.mean(0)
    return d / (np.linalg.norm(d) + 1e-12)


def permuted_null(pairs, rng):
    vals = []
    for concept in C.CONCEPTS_17:
        cache = {}
        def cal(slug):
            if slug not in cache:
                _, p = C.load_dom_and_peak(slug, concept)
                cache[slug] = C.load_calibration(slug, concept, p)
            return cache[slug]
        for s, t in pairs:
            cs, ct = cal(s), cal(t)
            sc, tc, Q = _basis(cs, ct)
            Rq = C._procrustes_R(tc @ Q, sc @ Q)
            for _ in range(25):
                dsp = _dom_from_labels(cs, rng.permutation(len(cs))) @ Q
                dtp = _dom_from_labels(ct, rng.permutation(len(ct))) @ Q
                vals.append(C.cosine(dsp, dtp @ Rq))
    vals = np.array(vals)
    sem = vals.std() / np.sqrt(len(vals))
    return {"mean": float(vals.mean()), "sd": float(vals.std()), "n": int(len(vals)),
            "z_vs_primary": float((PRIMARY_MEAN - vals.mean()) / sem)}


def universality(pairs):
    same, cross = [], []
    for s, t in pairs:
        dom, cal = {}, {}
        for c in C.CONCEPTS_17:
            d, p = C.load_dom_and_peak(s, c); d2, p2 = C.load_dom_and_peak(t, c)
            if d is None or d2 is None:
                dom[c] = None; continue
            dom[c] = (d, d2); cal[c] = (C.load_calibration(s, c, p), C.load_calibration(t, c, p2))
        for cA in C.CONCEPTS_17:
            if dom.get(cA) is None:
                continue
            sc, tc, Q = _basis(cal[cA][0], cal[cA][1])
            Rq = C._procrustes_R(tc @ Q, sc @ Q)
            for cB in C.CONCEPTS_17:
                if dom.get(cB) is None:
                    continue
                dsB, dtB = dom[cB]
                v = C.cosine(dsB @ Q, (dtB @ Q) @ Rq)
                (same if cA == cB else cross).append(v)
    same, cross = np.array(same), np.array(cross)
    return {"same_concept": float(same.mean()), "cross_concept": float(cross.mean()),
            "ratio": float(cross.mean() / same.mean()), "n_same": len(same), "n_cross": len(cross)}


def peak_depth(pairs):
    # aligned values + peak depths; depth = peak_layer / (n_layers - 1)
    matched, mismatched = [], []
    for concept in C.CONCEPTS_17:
        info = {}
        for slug in C.ROSTER:
            d, p = C.load_dom_and_peak(slug, concept)
            caz = json.load(open(C._hf(f"{C.HF_ROOT}/{slug}/caz_{concept}.json")))
            info[slug] = (d, p, caz["layer_data"].get("n_layers") or caz.get("n_layers"))
        cache = {}
        def cal(slug, p):
            if slug not in cache:
                cache[slug] = C.load_calibration(slug, concept, p)
            return cache[slug]
        for s, t in pairs:
            ds, ps, Ls = info[s]; dt, pt, Lt = info[t]
            if ds is None or dt is None or np.linalg.norm(ds) < 1e-9 or np.linalg.norm(dt) < 1e-9:
                continue
            v = C.aligned_cosine(ds, dt, cal(s, ps), cal(t, pt))
            depth_s = ps / (Ls - 1); depth_t = pt / (Lt - 1)
            (matched if abs(depth_s - depth_t) < 0.15 else mismatched).append(v)
    matched, mismatched = np.array(matched), np.array(mismatched)
    return {"matched": float(matched.mean()), "mismatched": float(mismatched.mean()),
            "delta": float(matched.mean() - mismatched.mean()),
            "n_matched": len(matched), "n_mismatched": len(mismatched)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="./p4_regen_output")
    ap.add_argument("--which", nargs="*", default=["permuted", "universality", "peak_depth"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pairs = C.cross_family_same_dim_pairs()
    rng = np.random.default_rng(args.seed)

    res = {}
    if "permuted" in args.which:
        print("permuted-label null (long — ~44k reduced solves)...")
        res["permuted_null"] = permuted_null(pairs, rng)
        print("  ", res["permuted_null"])
    if "universality" in args.which:
        print("universality (cross-concept transfer)...")
        res["universality"] = universality(pairs)
        print("  ", res["universality"])
    if "peak_depth" in args.which:
        print("peak-level depth matching...")
        res["peak_depth"] = peak_depth(pairs)
        print("  ", res["peak_depth"])

    (out / "step2_nulls.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {out/'step2_nulls.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""P4 Phase-A null-floor re-analysis (FROZEN_REVIEW_ROUND1 P4-B1/B3).

Settles whether P4's 0.9750 headline stands against a ~0 null or must be
restated against a spectrum-matched floor of ~0.4. Re-analysis only, CPU-only,
on the published HF `paper_n250` calibration artifacts. Writes nothing to the
manuscript — produces the numbers that decide the framing.

Three nulls, all through the paper's own pipeline (`common.aligned_cosine`:
mean-centred 500xd calibration, orthogonal Procrustes target->source, applied
to the target DOM):

  noise        two independent Gaussian [n,d] matrices, DOM = class-mean diff
               (250 pos / 250 neg by index). Validates the pipeline against the
               reviewer's table (d=768->0.824 ... 4096->0.968, 5120->0.973).
  spectrum     two INDEPENDENT surrogates carrying the REAL per-model
               singular-value spectrum but random orthonormal bases. The honest
               null floor. Reviewer: Cluster A 0.15-0.48, d=2048 0.42-0.43;
               THIS run completes d=4096 (C) and d=5120 (E).
  scramble     condition (B): permute the target calibration's rows WITHIN each
               class block, refit R, apply to the TRUE DOMs. Destroys text-level
               row correspondence, keeps class structure. Reviewer: 2/17 concepts
               Cluster A, scrambled >= true; THIS run completes all A-E x 17.

  nsweep       real aligned cos and spectrum floor vs n (subsample rows,
               stratified 250/250 -> n/2 each) at a fixed cluster.

Usage:
  python nullfloor_analysis.py --mode {validate,spectrum,scramble,nsweep,all}
                               [--clusters A,B,...] [--concepts a,b,...]
                               [--K 24] [--out nullfloor_out]
"""
import sys, os, json, time, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import common as C

DIMLET = {768: "A", 1024: "G", 2048: "B", 2560: "H", 3584: "D", 4096: "C", 5120: "E", 8192: "F"}
LETDIM = {v: k for k, v in DIMLET.items()}
AE_LETTERS = ["A", "G", "B", "H", "D", "C", "E"]


def log(m):
    print(m, flush=True)


def pairs_in_cluster(letter):
    d = LETDIM[letter]
    slugs = [s for s, (f, dd) in C.ROSTER.items() if dd == d]
    return C.cross_family_same_dim_pairs(slugs)


def spectrum_surrogate(sv, n, d, rng):
    """Random [n,d] matrix with singular-value spectrum `sv` (already mean-centred
    real spectrum), independent random orthonormal row/col bases. Mean-centred."""
    G = rng.standard_normal((n, d))
    Ug, _, Vtg = np.linalg.svd(G, full_matrices=False)   # Ug n×r, Vtg r×d, r=min(n,d)
    r = len(sv)
    M = (Ug[:, :r] * sv) @ Vtg[:r]
    return M - M.mean(0)


def real_spectrum(cal):
    cc = cal - cal.mean(0)
    return np.linalg.svd(cc, compute_uv=False)


def dom_from(cal, pos, neg):
    d = cal[pos].mean(0) - cal[neg].mean(0)
    nrm = np.linalg.norm(d)
    return d / nrm if nrm > 1e-12 else d


# ------------------------------------------------------------------ modes ---
def run_validate(K, out):
    """Pure-noise floor — validates the pipeline vs the reviewer's table."""
    log("=== PURE-NOISE floor (reviewer: 768->0.824 2048->0.945 4096->0.968 5120->0.973) ===")
    res = {}
    rng = np.random.default_rng(0)
    for d in [768, 1024, 2048, 2560, 3584, 4096, 5120]:
        vals = []
        for _ in range(K):
            a = rng.standard_normal((500, d)); b = rng.standard_normal((500, d))
            da = dom_from(a, slice(0, 250), slice(250, 500))
            db = dom_from(b, slice(0, 250), slice(250, 500))
            vals.append(C.aligned_cosine(da, db, a, b))
        res[DIMLET[d]] = dict(d=d, mean=float(np.mean(vals)), sd=float(np.std(vals)),
                              lo=float(np.min(vals)), hi=float(np.max(vals)))
        log(f"  d={d} ({DIMLET[d]}): {np.mean(vals):.3f} ± {np.std(vals):.3f}  [{np.min(vals):.3f},{np.max(vals):.3f}]")
    (out / "noise_floor.json").write_text(json.dumps(res, indent=2))


def run_spectrum(letters, concepts, K, out, max_pairs=None):
    """Spectrum-matched null floor per cluster + real aligned cos for margin."""
    log(f"=== SPECTRUM-MATCHED floor + real, clusters {letters}, K={K}, max_pairs={max_pairs} ===")
    allres = {}
    for L in letters:
        d = LETDIM[L]; prs = pairs_in_cluster(L)
        if max_pairs and len(prs) > max_pairs:
            step = len(prs) / max_pairs
            prs = [prs[int(i * step)] for i in range(max_pairs)]
        rng = np.random.default_rng(hash(L) % (2**31))
        real_vals, floor_vals = [], []
        t0 = time.time()
        # cache real calibration+dom+spectrum per (slug,concept)
        cache = {}
        def get(slug, con):
            k = (slug, con)
            if k not in cache:
                dom, pk = C.load_dom_and_peak(slug, con)
                cal = C.load_calibration(slug, con, pk)
                cache[k] = (dom, cal, real_spectrum(cal))
            return cache[k]
        for ci, con in enumerate(concepts):
            for (s, t) in prs:
                try:
                    dom_s, cal_s, sv_s = get(s, con)
                    dom_t, cal_t, sv_t = get(t, con)
                except Exception as e:
                    continue
                if dom_s is None or dom_t is None:
                    continue
                real_vals.append(C.aligned_cosine(dom_s, dom_t, cal_s, cal_t))
                # spectrum-matched surrogates: independent random bases, real spectra
                for _ in range(K):
                    ss = spectrum_surrogate(sv_s, cal_s.shape[0], d, rng)
                    tt = spectrum_surrogate(sv_t, cal_t.shape[0], d, rng)
                    dss = dom_from(ss, slice(0, 250), slice(250, ss.shape[0]))
                    dtt = dom_from(tt, slice(0, 250), slice(250, tt.shape[0]))
                    floor_vals.append(C.aligned_cosine(dss, dtt, ss, tt))
            log(f"  [{L}] {ci+1}/{len(concepts)} {con}: real n={len(real_vals)} floor n={len(floor_vals)} ({time.time()-t0:.0f}s)")
        rv, fv = np.array(real_vals), np.array(floor_vals)
        allres[L] = dict(d=d, n_pairs=len(prs),
                         real_mean=float(rv.mean()), real_median=float(np.median(rv)),
                         floor_mean=float(fv.mean()), floor_median=float(np.median(fv)),
                         floor_lo=float(np.percentile(fv, 2.5)), floor_hi=float(np.percentile(fv, 97.5)),
                         margin=float(rv.mean() - fv.mean()))
        log(f"  ==> [{L}] REAL {rv.mean():.4f} | FLOOR {fv.mean():.4f} [{np.percentile(fv,2.5):.3f},{np.percentile(fv,97.5):.3f}] | margin {rv.mean()-fv.mean():.4f}")
        (out / "spectrum_floor.json").write_text(json.dumps(allres, indent=2))
    return allres


def run_scramble(letters, concepts, K, out):
    """Within-class row-scramble (condition B) vs true correspondence (A)."""
    log(f"=== WITHIN-CLASS ROW-SCRAMBLE, clusters {letters}, K={K} ===")
    allres = {}
    for L in letters:
        d = LETDIM[L]; prs = pairs_in_cluster(L)
        rng = np.random.default_rng(1000 + hash(L) % 9999)
        rows = []
        t0 = time.time()
        cache = {}
        def get(slug, con):
            k = (slug, con)
            if k not in cache:
                dom, pk = C.load_dom_and_peak(slug, con)
                cal = C.load_calibration(slug, con, pk)
                cache[k] = (dom, cal)
            return cache[k]
        for ci, con in enumerate(concepts):
            for (s, t) in prs:
                try:
                    dom_s, cal_s = get(s, con)
                    dom_t, cal_t = get(t, con)
                except Exception:
                    continue
                if dom_s is None or dom_t is None:
                    continue
                true = C.aligned_cosine(dom_s, dom_t, cal_s, cal_t)  # (A)
                n = cal_t.shape[0]; h = n // 2
                scr = []
                for _ in range(K):
                    ct = cal_t.copy()
                    p = rng.permutation(h); q = rng.permutation(n - h) + h
                    ct[:h] = cal_t[p]; ct[h:] = cal_t[q]     # within-class permute
                    scr.append(C.aligned_cosine(dom_s, dom_t, cal_s, ct))  # (B) true DOMs
                rows.append(dict(cluster=L, s=s, t=t, concept=con,
                                 true=float(true), scramble_mean=float(np.mean(scr)),
                                 scramble_ge_true=bool(np.mean(scr) >= true)))
            log(f"  [{L}] {ci+1}/{len(concepts)} {con} ({time.time()-t0:.0f}s)")
        tv = np.array([r["true"] for r in rows]); sv = np.array([r["scramble_mean"] for r in rows])
        allres[L] = dict(d=d, n_fits=len(rows),
                         true_mean=float(tv.mean()), scramble_mean=float(sv.mean()),
                         frac_scramble_ge_true=float(np.mean([r["scramble_ge_true"] for r in rows])),
                         delta_mean=float((sv - tv).mean()), rows=rows)
        log(f"  ==> [{L}] TRUE {tv.mean():.4f} | SCRAMBLE {sv.mean():.4f} | scramble>=true in {100*np.mean([r['scramble_ge_true'] for r in rows]):.0f}% of {len(rows)} fits")
        (out / "scramble.json").write_text(json.dumps(allres, indent=2))
    return allres


def run_nsweep(letter, concepts, K, out):
    """Real aligned cos and spectrum floor vs n at one cluster."""
    L = letter; d = LETDIM[L]; prs = pairs_in_cluster(L)
    log(f"=== N-SWEEP at cluster {L} (d={d}) ===")
    rng = np.random.default_rng(7)
    cache = {}
    def get(slug, con):
        if (slug, con) not in cache:
            dom, pk = C.load_dom_and_peak(slug, con)
            cache[(slug, con)] = (dom, C.load_calibration(slug, con, pk))
        return cache[(slug, con)]
    res = {}
    for n in [100, 250, 500]:
        half = n // 2
        real_vals, floor_vals = [], []
        for con in concepts:
            for (s, t) in prs:
                dom_s, cal_s = get(s, con); dom_t, cal_t = get(t, con)
                if dom_s is None or dom_t is None: continue
                N = cal_s.shape[0]; H = N // 2
                pi = np.r_[rng.choice(H, half, False), H + rng.choice(N - H, half, False)]
                cs, ct = cal_s[pi], cal_t[pi]
                # real DOM is fixed (stored); alignment R now fit on n rows
                real_vals.append(C.aligned_cosine(dom_s, dom_t, cs, ct))
                sv_s, sv_t = real_spectrum(cs), real_spectrum(ct)
                for _ in range(K):
                    ss = spectrum_surrogate(sv_s, n, d, rng); tt = spectrum_surrogate(sv_t, n, d, rng)
                    dss = dom_from(ss, slice(0, half), slice(half, n)); dtt = dom_from(tt, slice(0, half), slice(half, n))
                    floor_vals.append(C.aligned_cosine(dss, dtt, ss, tt))
        res[n] = dict(dn=d / n, real_mean=float(np.mean(real_vals)), floor_mean=float(np.mean(floor_vals)))
        log(f"  n={n} (d/n={d/n:.1f}): REAL {np.mean(real_vals):.4f} | FLOOR {np.mean(floor_vals):.4f}")
    (out / f"nsweep_{L}.json").write_text(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="validate")
    ap.add_argument("--clusters", default="C,E")
    ap.add_argument("--concepts", default=None)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--nsweep-cluster", default="B")
    ap.add_argument("--out", default="nullfloor_out")
    ap.add_argument("--max-pairs", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    concepts = a.concepts.split(",") if a.concepts else C.CONCEPTS_17
    letters = a.clusters.split(",")
    if a.mode in ("validate", "all"):
        run_validate(a.K, out)
    if a.mode in ("spectrum", "all"):
        run_spectrum(letters, concepts, a.K, out, a.max_pairs)
    if a.mode in ("scramble", "all"):
        run_scramble(letters, concepts, a.K, out)
    if a.mode in ("nsweep", "all"):
        run_nsweep(a.nsweep_cluster, concepts, a.K, out)


if __name__ == "__main__":
    main()

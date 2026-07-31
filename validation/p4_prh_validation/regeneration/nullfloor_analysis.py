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


def _seed(tag: str, base: int = 0) -> int:
    """Deterministic per-cluster seed.

    `hash()` on a str is salted per process (PYTHONHASHSEED), so seeding an RNG
    with it makes runs unreproducible across invocations and hosts. This is a
    fixed FNV-style rolling hash instead.
    """
    h = 2166136261
    for ch in tag:
        h = ((h ^ ord(ch)) * 16777619) % (2**32)
    return (base + h) % (2**31)


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
    # 8192 included so PHASE_A §0's cluster-F row regenerates from shipped code
    # (audit gap: it was previously produced ad hoc and not reproducible here)
    for d in [768, 1024, 2048, 2560, 3584, 4096, 5120, 8192]:
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
        rng = np.random.default_rng(_seed(L))
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
                # A mismatched calibration row count (e.g. 499 vs 500 from the
                # blank-text row-alignment cases) makes the Procrustes fit throw.
                # The primary pipeline excludes exactly these as "unavailable
                # fits" (§3.1 corpus note); skip the pair here too rather than
                # min-truncate, so the floor stays over the SAME pairs as the
                # real cluster mean it is compared against. Buffer per-pair so a
                # skip drops both the real value and its floor samples together.
                try:
                    rv_pair = C.aligned_cosine(dom_s, dom_t, cal_s, cal_t)
                    fv_pair = []
                    # spectrum-matched surrogates: independent random bases, real spectra
                    for _ in range(K):
                        ss = spectrum_surrogate(sv_s, cal_s.shape[0], d, rng)
                        tt = spectrum_surrogate(sv_t, cal_t.shape[0], d, rng)
                        # class split derived from the array, not hardcoded: most
                        # calibrations are n=500, but the exfiltration rerun is n=498
                        # (249 pairs). No-op at n=500.
                        hs, ht = ss.shape[0] // 2, tt.shape[0] // 2
                        dss = dom_from(ss, slice(0, hs), slice(hs, ss.shape[0]))
                        dtt = dom_from(tt, slice(0, ht), slice(ht, tt.shape[0]))
                        fv_pair.append(C.aligned_cosine(dss, dtt, ss, tt))
                except Exception as e:
                    log(f"  [{L}] skip {s} x {t} / {con}: {type(e).__name__}: {e}")
                    continue
                real_vals.append(rv_pair)
                floor_vals.extend(fv_pair)
            log(f"  [{L}] {ci+1}/{len(concepts)} {con}: real n={len(real_vals)} floor n={len(floor_vals)} ({time.time()-t0:.0f}s)")
        rv, fv = np.array(real_vals), np.array(floor_vals)
        allres[L] = dict(d=d, n_pairs=len(prs),
                         real_mean=float(rv.mean()), real_median=float(np.median(rv)),
                         floor_mean=float(fv.mean()), floor_median=float(np.median(fv)),
                         floor_lo=float(np.percentile(fv, 2.5)), floor_hi=float(np.percentile(fv, 97.5)),
                         margin=float(rv.mean() - fv.mean()),
                         # raw samples retained so partial runs over disjoint concept
                         # sets can be pooled exactly (percentiles included).
                         concepts=list(concepts),
                         real_vals=[float(x) for x in rv],
                         floor_vals=[float(x) for x in fv])
        log(f"  ==> [{L}] REAL {rv.mean():.4f} | FLOOR {fv.mean():.4f} [{np.percentile(fv,2.5):.3f},{np.percentile(fv,97.5):.3f}] | margin {rv.mean()-fv.mean():.4f}")
        (out / "spectrum_floor.json").write_text(json.dumps(allres, indent=2))
    return allres


def run_scramble(letters, concepts, K, out):
    """Within-class row-scramble (condition B) vs true correspondence (A)."""
    log(f"=== WITHIN-CLASS ROW-SCRAMBLE, clusters {letters}, K={K} ===")
    allres = {}
    for L in letters:
        d = LETDIM[L]; prs = pairs_in_cluster(L)
        rng = np.random.default_rng(_seed(L, 1000))
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
                # Skip mismatched-row pairs (the primary pipeline's "unavailable
                # fits"); a Procrustes shape mismatch would otherwise crash the
                # whole cluster (E/D exit=1). See run_spectrum for the rationale.
                try:
                    true = C.aligned_cosine(dom_s, dom_t, cal_s, cal_t)  # (A)
                    n = cal_t.shape[0]; h = n // 2
                    scr = []
                    for _ in range(K):
                        ct = cal_t.copy()
                        p = rng.permutation(h); q = rng.permutation(n - h) + h
                        ct[:h] = cal_t[p]; ct[h:] = cal_t[q]     # within-class permute
                        # audit: class-mean invariance, asserted rather than assumed
                        assert np.allclose(ct[:h].mean(0), cal_t[:h].mean(0), atol=1e-10)
                        scr.append(C.aligned_cosine(dom_s, dom_t, cal_s, ct))  # (B) true DOMs
                except Exception as e:
                    log(f"  [{L}] skip {s} x {t} / {con}: {type(e).__name__}: {e}")
                    continue
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


def run_nsweep(letter, concepts, K, out, max_pairs=None):
    """Real aligned cos and spectrum floor vs n at one cluster.

    Rewritten 2026-07-31 (audit gaps): the original run compared a real arm
    using the STORED full-500-row DOM against a floor arm whose DOM came from
    only n surrogate rows, and re-estimated the surrogate spectrum from each
    n-row subsample — so "fixed spectrum" was false as coded and the arms were
    asymmetric. Now, per n, four like-for-like series:
      real_stored   — stored full-N DOM, R fit on the n-row subsample (legacy)
      real_sub      — DOM recomputed from the same n rows (symmetric with floors)
      floor_subspec — surrogate spectrum estimated from the n-row subsample
      floor_fullspec— surrogate spectrum = full-N singular values, top-n,
                      scaled by sqrt(n/N): the spectral SHAPE genuinely held
                      fixed while only n varies — the pure d/n series.
    Per-pair guard (498-row calibrations skip, logged), like nullfloor floors.
    """
    L = letter; d = LETDIM[L]; prs = pairs_in_cluster(L)
    if max_pairs and len(prs) > max_pairs:
        step = len(prs) / max_pairs
        prs = [prs[int(i * step)] for i in range(max_pairs)]
    log(f"=== N-SWEEP at cluster {L} (d={d}), {len(prs)} pairs ===")
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
        vals = {k: [] for k in ("real_stored", "real_sub", "floor_subspec", "floor_fullspec")}
        for con in concepts:
            for (s, t) in prs:
                try:
                    dom_s, cal_s = get(s, con); dom_t, cal_t = get(t, con)
                    if dom_s is None or dom_t is None: continue
                    N = cal_s.shape[0]; H = N // 2
                    if cal_t.shape[0] != N:
                        raise ValueError(f"row mismatch {N} vs {cal_t.shape[0]}")
                    pi = np.r_[rng.choice(H, half, False), H + rng.choice(N - H, half, False)]
                    cs, ct = cal_s[pi], cal_t[pi]
                    b = {}
                    b["real_stored"] = C.aligned_cosine(dom_s, dom_t, cs, ct)
                    ds_sub = dom_from(cs, slice(0, half), slice(half, n))
                    dt_sub = dom_from(ct, slice(0, half), slice(half, n))
                    b["real_sub"] = C.aligned_cosine(ds_sub, dt_sub, cs, ct)
                    sv_s_sub, sv_t_sub = real_spectrum(cs), real_spectrum(ct)
                    sv_s_full, sv_t_full = real_spectrum(cal_s), real_spectrum(cal_t)
                    scale = np.sqrt(n / N)
                    b["floor_subspec"], b["floor_fullspec"] = [], []
                    for _ in range(K):
                        for key, svs, svt in (("floor_subspec", sv_s_sub, sv_t_sub),
                                              ("floor_fullspec", sv_s_full[:n] * scale,
                                               sv_t_full[:n] * scale)):
                            ss = spectrum_surrogate(svs, n, d, rng)
                            tt = spectrum_surrogate(svt, n, d, rng)
                            dss = dom_from(ss, slice(0, half), slice(half, n))
                            dtt = dom_from(tt, slice(0, half), slice(half, n))
                            b[key].append(C.aligned_cosine(dss, dtt, ss, tt))
                except Exception as e:
                    log(f"  [{L} n={n}] skip {s} x {t} / {con}: {type(e).__name__}: {e}")
                    continue
                vals["real_stored"].append(b["real_stored"])
                vals["real_sub"].append(b["real_sub"])
                vals["floor_subspec"].extend(b["floor_subspec"])
                vals["floor_fullspec"].extend(b["floor_fullspec"])
        res[n] = dict(dn=d / n, n_fits=len(vals["real_sub"]),
                      **{k: float(np.mean(v)) for k, v in vals.items()})
        log(f"  n={n} (d/n={d/n:.1f}): real_stored {res[n]['real_stored']:.4f} "
            f"real_sub {res[n]['real_sub']:.4f} | floor_subspec {res[n]['floor_subspec']:.4f} "
            f"floor_fullspec {res[n]['floor_fullspec']:.4f}  (fits {res[n]['n_fits']})")
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
        run_nsweep(a.nsweep_cluster, concepts, a.K, out, a.max_pairs)


if __name__ == "__main__":
    main()

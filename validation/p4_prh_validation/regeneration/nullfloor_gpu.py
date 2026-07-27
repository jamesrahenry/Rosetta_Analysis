#!/usr/bin/env python3
"""GPU port of the P4 Phase-A null-floor sweep (FROZEN_REVIEW_ROUND1 P4-B1/B3).

The CPU version (`nullfloor_analysis.py`) is the reference implementation and
stays the source of truth. This is a CuPy re-implementation of the same three
quantities, for hosts with a real FP64 GPU.

Why a GPU port is worth it here, when it usually isn't: the whole sweep is dense
float64 linear algebra -- a QR of a d x 2n matrix and an SVD of a 2n x 2n matrix
per aligned_cosine, ~36 GFLOP at d=8192. Most accelerators cripple FP64 to 1/32
or 1/64 rate and would be *slower* than the CPU. H100 (and A100/H200) run FP64
near parity, so this is one of the rare analyses that genuinely maps onto them.
Measured CPU cost is 1.9 s/call on an i7-4600U and 3.3 s/call on an FX-8350;
the full remaining sweep is ~38 CPU-hours.

CORRECTNESS IS THE POINT, NOT SPEED. `aligned_cosine` *is* the paper's pipeline,
so a port that drifts makes every number it touches uncitable. Two safeguards:

  1. `--mode selftest` runs the GPU and CPU implementations over IDENTICAL inputs
     (generated on CPU, copied to device) and reports max |GPU - CPU|. It is a
     hard gate: the production modes refuse to run unless it passes.
  2. Everything is forced to float64 end to end. No TF32, no mixed precision.
     A silent downcast is the failure mode most likely to go unnoticed.

Note on RNG: CuPy's generator does not reproduce NumPy's stream, so surrogate
DRAWS differ between CPU and GPU runs even at the same seed. That is expected
and harmless -- the floor is a Monte Carlo expectation, so the two agree in the
mean, not per-sample. Cross-host agreement is therefore checked statistically
(overlapping CIs on the floor mean), while `selftest` checks the deterministic
kernel exactly.

Usage:
  python nullfloor_gpu.py --mode selftest
  python nullfloor_gpu.py --mode spectrum --clusters G,A,H,E,D,C,B --K 8 --out gpu_out
  python nullfloor_gpu.py --mode scramble --clusters G,A,H,E,D,C,B --K 8 --out gpu_out
"""
import json, time, argparse, sys
from pathlib import Path
import numpy as np
import common as C
from nullfloor_analysis import LETDIM, pairs_in_cluster, _seed, real_spectrum, dom_from

# --------------------------------------------------------------- backend ---
try:
    import cupy as _cp
    xp = _cp
    ON_GPU = True
except Exception:                                    # no CuPy / no device
    _cp = None
    xp = np
    ON_GPU = False


def log(m):
    print(m, flush=True)


def _dev(a):
    """Host -> device as float64, passing device arrays through unchanged.

    The device-resident branch is load-bearing: CuPy refuses implicit
    conversion to NumPy, so `np.asarray()` on an array that is already on the
    GPU raises. The surrogate path feeds device arrays straight back into
    aligned_cosine_xp, which is exactly that case.
    """
    if ON_GPU and isinstance(a, _cp.ndarray):
        return a.astype(xp.float64, copy=False)
    return xp.asarray(np.asarray(a, np.float64))


def _host(a):
    return _cp.asnumpy(a) if ON_GPU else np.asarray(a)


def _sync():
    if ON_GPU:
        _cp.cuda.Stream.null.synchronize()


# ------------------------------------------------------------- kernels ----
def procrustes_R_xp(t, s):
    """R minimising ||t R - s||_F, matching scipy.linalg.orthogonal_procrustes.

    scipy computes svd(s.T @ t).T == svd(t.T @ s) and returns u @ vt; replicated
    here rather than approximated, because this is the paper's operator.
    """
    u, _, vt = xp.linalg.svd(t.T @ s, full_matrices=False)
    return u @ vt


def aligned_cosine_xp(dom_s, dom_t, cal_s, cal_t):
    """Device port of common.aligned_cosine (rank-reduced Procrustes).

    Mirrors the CPU routine step for step: mean-centre both calibrations, build
    an orthonormal basis Q spanning BOTH row spaces via QR of [tc.T | sc.T],
    solve Procrustes inside that basis, then transport dom_t and take the cosine
    against dom_s. Arguments may be host or device arrays.
    """
    sc = _dev(cal_s); sc = sc - sc.mean(0)
    tc = _dev(cal_t); tc = tc - tc.mean(0)
    Q, _ = xp.linalg.qr(xp.hstack([tc.T, sc.T]))
    Rq = procrustes_R_xp(tc @ Q, sc @ Q)
    a = _dev(dom_s).ravel() @ Q
    b = (_dev(dom_t).ravel() @ Q) @ Rq
    denom = xp.linalg.norm(a) * xp.linalg.norm(b)
    if float(denom) < 1e-12:
        return 0.0
    return float(a @ b / denom)


def spectrum_surrogate_xp(sv, n, d, rng):
    """[n,d] surrogate carrying spectrum `sv` on an independent random basis."""
    G = rng.standard_normal((n, d), dtype=xp.float64)
    Ug, _, Vtg = xp.linalg.svd(G, full_matrices=False)
    r = len(sv)
    M = (Ug[:, :r] * _dev(sv)) @ Vtg[:r]
    return M - M.mean(0)


def _rng(seed):
    return xp.random.default_rng(seed) if ON_GPU else np.random.default_rng(seed)


# ------------------------------------------------------------- selftest ---
def run_selftest(tol=1e-8):
    """Hard gate: GPU kernel must match common.aligned_cosine on identical input."""
    log(f"=== SELFTEST: backend={'CuPy/GPU' if ON_GPU else 'NumPy/CPU fallback'} ===")
    rs = np.random.default_rng(12345)
    worst = 0.0
    for d in (768, 2048, 4096, 8192):
        a = rs.standard_normal((500, d)); b = rs.standard_normal((500, d))
        da = dom_from(a, slice(0, 250), slice(250, 500))
        db = dom_from(b, slice(0, 250), slice(250, 500))
        ref = C.aligned_cosine(da, db, a, b)          # CPU reference
        got = aligned_cosine_xp(da, db, a, b)         # port, identical inputs
        diff = abs(ref - got); worst = max(worst, diff)
        log(f"  synthetic d={d:5d}  cpu={ref:.12f}  port={got:.12f}  |d|={diff:.2e}")

    # real calibration data -- catches anything specific to genuine spectra
    try:
        for L in ("A", "B"):
            s, t = pairs_in_cluster(L)[0]
            for con in ("agency", "certainty"):
                dom_s, pk_s = C.load_dom_and_peak(s, con)
                dom_t, pk_t = C.load_dom_and_peak(t, con)
                if dom_s is None or dom_t is None:
                    continue
                cs = C.load_calibration(s, con, pk_s)
                ct = C.load_calibration(t, con, pk_t)
                ref = C.aligned_cosine(dom_s, dom_t, cs, ct)
                got = aligned_cosine_xp(dom_s, dom_t, cs, ct)
                diff = abs(ref - got); worst = max(worst, diff)
                log(f"  real [{L}] {con:<10s} cpu={ref:.12f}  port={got:.12f}  |d|={diff:.2e}")
    except Exception as e:                            # data absent -> synthetic only
        log(f"  (real-data leg skipped: {type(e).__name__}: {e})")

    ok = worst <= tol
    log(f"  ==> worst |GPU-CPU| = {worst:.3e}  tol={tol:.0e}  {'PASS' if ok else 'FAIL'}")
    return ok, worst


def _require_selftest():
    ok, worst = run_selftest()
    if not ok:
        raise SystemExit(f"selftest FAILED (worst diff {worst:.3e}) -- refusing to "
                         "produce numbers from an unvalidated kernel")
    if not ON_GPU:
        log("  NOTE: CuPy unavailable -- running the NumPy fallback (no speedup).")


# ---------------------------------------------------------------- modes ---
def run_spectrum(letters, concepts, K, out):
    _require_selftest()
    allres = {}
    for L in letters:
        d = LETDIM[L]; prs = pairs_in_cluster(L)
        rng = _rng(_seed(L))
        real_vals, floor_vals = [], []
        t0 = time.time(); cache = {}

        def get(slug, con):
            if (slug, con) not in cache:
                dom, pk = C.load_dom_and_peak(slug, con)
                cal = C.load_calibration(slug, con, pk)
                cache[(slug, con)] = (dom, cal, real_spectrum(cal))
            return cache[(slug, con)]

        for ci, con in enumerate(concepts):
            for (s, t) in prs:
                try:
                    dom_s, cal_s, sv_s = get(s, con)
                    dom_t, cal_t, sv_t = get(t, con)
                except Exception:
                    continue
                if dom_s is None or dom_t is None:
                    continue
                real_vals.append(aligned_cosine_xp(dom_s, dom_t, cal_s, cal_t))
                for _ in range(K):
                    ss = spectrum_surrogate_xp(sv_s, cal_s.shape[0], d, rng)
                    tt = spectrum_surrogate_xp(sv_t, cal_t.shape[0], d, rng)
                    hs, ht = ss.shape[0] // 2, tt.shape[0] // 2
                    dss = ss[:hs].mean(0) - ss[hs:].mean(0)
                    dtt = tt[:ht].mean(0) - tt[ht:].mean(0)
                    floor_vals.append(aligned_cosine_xp(dss, dtt, ss, tt))
            _sync()
            log(f"  [{L}] {ci+1}/{len(concepts)} {con}: real n={len(real_vals)} "
                f"floor n={len(floor_vals)} ({time.time()-t0:.0f}s)")
        rv, fv = np.array(real_vals), np.array(floor_vals)
        allres[L] = dict(d=d, n_pairs=len(prs), backend="gpu" if ON_GPU else "cpu",
                         real_mean=float(rv.mean()), floor_mean=float(fv.mean()),
                         floor_lo=float(np.percentile(fv, 2.5)),
                         floor_hi=float(np.percentile(fv, 97.5)),
                         margin=float(rv.mean() - fv.mean()),
                         concepts=list(concepts),
                         real_vals=[float(x) for x in rv],
                         floor_vals=[float(x) for x in fv])
        log(f"  ==> [{L}] REAL {rv.mean():.4f} | FLOOR {fv.mean():.4f} | "
            f"margin {rv.mean()-fv.mean():.4f}")
        (out / "spectrum_floor_gpu.json").write_text(json.dumps(allres, indent=2))
    return allres


def run_scramble(letters, concepts, K, out):
    _require_selftest()
    allres = {}
    for L in letters:
        d = LETDIM[L]; prs = pairs_in_cluster(L)
        rng = np.random.default_rng(_seed(L, 1000))   # permutations stay on host
        rows = []; t0 = time.time(); cache = {}

        def get(slug, con):
            if (slug, con) not in cache:
                dom, pk = C.load_dom_and_peak(slug, con)
                cache[(slug, con)] = (dom, C.load_calibration(slug, con, pk))
            return cache[(slug, con)]

        for ci, con in enumerate(concepts):
            for (s, t) in prs:
                try:
                    dom_s, cal_s = get(s, con)
                    dom_t, cal_t = get(t, con)
                except Exception:
                    continue
                if dom_s is None or dom_t is None:
                    continue
                true = aligned_cosine_xp(dom_s, dom_t, cal_s, cal_t)
                n = cal_t.shape[0]; h = n // 2
                scr = []
                for _ in range(K):
                    ct = cal_t.copy()
                    ct[:h] = cal_t[rng.permutation(h)]
                    ct[h:] = cal_t[rng.permutation(n - h) + h]
                    scr.append(aligned_cosine_xp(dom_s, dom_t, cal_s, ct))
                rows.append(dict(cluster=L, s=s, t=t, concept=con, true=float(true),
                                 scramble_mean=float(np.mean(scr)),
                                 scramble_ge_true=bool(np.mean(scr) >= true)))
            _sync()
            log(f"  [{L}] {ci+1}/{len(concepts)} {con} ({time.time()-t0:.0f}s)")
        tv = np.array([r["true"] for r in rows]); sv = np.array([r["scramble_mean"] for r in rows])
        allres[L] = dict(d=d, n_fits=len(rows), backend="gpu" if ON_GPU else "cpu",
                         true_mean=float(tv.mean()), scramble_mean=float(sv.mean()),
                         frac_scramble_ge_true=float(np.mean([r["scramble_ge_true"] for r in rows])),
                         delta_mean=float((sv - tv).mean()), rows=rows)
        log(f"  ==> [{L}] TRUE {tv.mean():.4f} | SCRAMBLE {sv.mean():.4f} | "
            f"scramble>=true in {100*np.mean([r['scramble_ge_true'] for r in rows]):.0f}%")
        (out / "scramble_gpu.json").write_text(json.dumps(allres, indent=2))
    return allres


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="selftest",
                    choices=["selftest", "spectrum", "scramble", "all"])
    ap.add_argument("--clusters", default="G,A,H,E,D,C,B")
    ap.add_argument("--concepts", default=None)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--out", default="nullfloor_gpu_out")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    concepts = a.concepts.split(",") if a.concepts else C.CONCEPTS_17
    letters = a.clusters.split(",")

    if ON_GPU:
        try:
            p = _cp.cuda.runtime.getDeviceProperties(_cp.cuda.runtime.getDevice())
            log(f"device: {p['name'].decode()} | CuPy {_cp.__version__}")
        except Exception:
            pass

    if a.mode == "selftest":
        ok, _ = run_selftest()
        raise SystemExit(0 if ok else 1)
    if a.mode in ("spectrum", "all"):
        run_spectrum(letters, concepts, a.K, out)
    if a.mode in ("scramble", "all"):
        run_scramble(letters, concepts, a.K, out)


if __name__ == "__main__":
    main()

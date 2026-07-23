#!/usr/bin/env python3
"""M4: regenerate the §3.3 17x17 cross-concept transfer matrix (Figure 1) on the
CORRECTED exfiltration data, using the canonical universality method
(step2_headline_nulls.universality): fit R on concept A's calibration (reduced-
rank projection into A's row-space Q), apply to concept B's DOM projected into
the same Q. A-E roster (30 models), matching the published 30-model matrix.
Output: transfer_matrix_17x17_corrected.json
"""
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

def _basis(cal_s, cal_t):
    # min-truncate to a common sample count (corrected exfiltration is N=249/498
    # while gemma's exfiltration is N=250/500) — matches the primary/handoff
    # pipelines' handling of mismatched calibration sizes.
    n = min(len(cal_s), len(cal_t))
    sc = cal_s[:n] - cal_s[:n].mean(0); tc = cal_t[:n] - cal_t[:n].mean(0)
    Q, _ = np.linalg.qr(np.hstack([tc.T, sc.T]))
    return sc, tc, Q

def main():
    t0 = time.time()
    concepts = C.CONCEPTS_17
    # A-E roster: exclude the 8192-dim frontier cluster F (matches the 30-model matrix)
    ae = [s for s in C.ROSTER if C.ROSTER[s][1] != 8192]
    pairs = C.cross_family_same_dim_pairs(ae)
    print(f"A-E roster: {len(ae)} models, {len(pairs)} cross-family same-dim ordered pairs", flush=True)
    acc = {a: {b: [] for b in concepts} for a in concepts}
    for i, (s, t) in enumerate(pairs):
        dom, cal = {}, {}
        for c in concepts:
            d, p = C.load_dom_and_peak(s, c); d2, p2 = C.load_dom_and_peak(t, c)
            if d is None or d2 is None:
                dom[c] = None; continue
            dom[c] = (d, d2); cal[c] = (C.load_calibration(s, c, p), C.load_calibration(t, c, p2))
        for cA in concepts:
            if dom.get(cA) is None:
                continue
            sc, tc, Q = _basis(cal[cA][0], cal[cA][1])
            Rq = C._procrustes_R(tc @ Q, sc @ Q)
            for cB in concepts:
                if dom.get(cB) is None:
                    continue
                dsB, dtB = dom[cB]
                acc[cA][cB].append(C.cosine(dsB @ Q, (dtB @ Q) @ Rq))
        print(f"  [{i+1}/{len(pairs)}] {s} x {t} ({time.time()-t0:.0f}s)", flush=True)
    # aggregate 17x17
    matrix = {a: {b: (float(np.mean(acc[a][b])) if acc[a][b] else None) for b in concepts} for a in concepts}
    same = np.array([np.mean(acc[a][a]) for a in concepts if acc[a][a]])
    cross = np.array([np.mean(acc[a][b]) for a in concepts for b in concepts if a != b and acc[a][b]])
    n_same = sum(len(acc[a][a]) for a in concepts)
    n_cross = sum(len(acc[a][b]) for a in concepts for b in concepts if a != b)
    out = {
        "concepts": concepts, "matrix": matrix,
        "n_models": len(ae), "n_pairs": len(pairs), "n_same": n_same, "n_cross": n_cross,
        "same_concept_mean": float(same.mean()), "cross_concept_mean": float(cross.mean()),
        "ratio": float(cross.mean() / same.mean()),
        "method": "canonical universality (reduced-rank Procrustes), corrected exfiltration, A-E",
        "elapsed_s": time.time() - t0,
    }
    o = Path(__file__).resolve().parent / "m4_out"; o.mkdir(exist_ok=True)
    (o / "transfer_matrix_17x17_corrected.json").write_text(json.dumps(out, indent=1))
    # exfiltration row/column: corrected vs what the figure had
    print(f"\n=== transfer matrix CORRECTED (A-E, {len(pairs)} pairs, {time.time()-t0:.0f}s) ===", flush=True)
    print(f"same-concept mean {same.mean():.4f} | cross mean {cross.mean():.4f} | ratio {cross.mean()/same.mean():.4f}", flush=True)
    print("exfiltration ROW (fit R on exfil, apply to B):", flush=True)
    for b in concepts:
        print(f"   exfil->{b:16s} {matrix['exfiltration'][b]:.3f}", flush=True)
    print("exfiltration diagonal (same-concept):", f"{matrix['exfiltration']['exfiltration']:.4f}", flush=True)
    print(f"wrote {o}/transfer_matrix_17x17_corrected.json", flush=True)

if __name__ == "__main__":
    main()

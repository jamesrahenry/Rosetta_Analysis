#!/usr/bin/env python3
"""M4 Figure 1: recompute ONLY exfiltration's row + column of the 17x17 transfer
matrix with FULL-DIM Procrustes (the method behind the published matrix), on
corrected exfiltration. float32 + precision gate vs float64; per-pair
checkpointed (resumable). The other 272 cells are unchanged by the exfiltration
correction and are kept from the published matrix at splice time.

  row[B]  = mean over pairs of cos(dom_s(B),  dom_t(B)  @ R_exfil)   (fit R on exfil)
  col[A]  = mean over pairs of cos(dom_s(exfil), dom_t(exfil) @ R_A) (fit R on A)
"""
import sys, json, time, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

CONCEPTS = C.CONCEPTS_17
EXI = "exfiltration"
DT = np.float64
OUT = Path(__file__).resolve().parent / "m4_out"; OUT.mkdir(exist_ok=True)
CKPT = OUT / "exfil_fulldim_ckpt.json"

def fit_R(cal_s, cal_t, dtype=DT):
    n = min(len(cal_s), len(cal_t))
    sc = (np.asarray(cal_s[:n], np.float64) - cal_s[:n].mean(0)).astype(dtype)
    tc = (np.asarray(cal_t[:n], np.float64) - cal_t[:n].mean(0)).astype(dtype)
    return C._procrustes_R(tc, sc)

def acos(dom_s, dom_t, R):
    return C.cosine(np.asarray(dom_s, np.float64), np.asarray(dom_t, np.float64).ravel() @ np.asarray(R, np.float64))

def main():
    t0 = time.time()
    ae = [s for s in C.ROSTER if C.ROSTER[s][1] != 8192]
    pairs = C.cross_family_same_dim_pairs(ae)
    print(f"A-E: {len(ae)} models, {len(pairs)} pairs; full-dim float32, checkpointed", flush=True)
    done = json.load(open(CKPT)) if CKPT.exists() else {}
    gate_err = 0.0
    for i, (s, t) in enumerate(pairs):
        if str(i) in done:
            continue
        dom_s, dom_t, cal_s, cal_t = {}, {}, {}, {}
        ok = True
        for c in CONCEPTS:
            ds, ps = C.load_dom_and_peak(s, c); dt, pt = C.load_dom_and_peak(t, c)
            if ds is None or dt is None:
                ok = False; break
            dom_s[c], dom_t[c] = ds, dt
            cal_s[c] = C.load_calibration(s, c, ps); cal_t[c] = C.load_calibration(t, c, pt)
        if not ok:
            done[str(i)] = {"row": {}, "col": {}, "skip": True}
            json.dump(done, open(CKPT, "w")); continue
        # ROW: fit R on exfil, apply to every B
        R_exi = fit_R(cal_s[EXI], cal_t[EXI])
        row = {B: acos(dom_s[B], dom_t[B], R_exi) for B in CONCEPTS}
        # COLUMN: fit R on each A != exfil, apply to exfil DOM
        col = {}
        for A in CONCEPTS:
            if A == EXI:
                continue
            col[A] = acos(dom_s[EXI], dom_t[EXI], fit_R(cal_s[A], cal_t[A]))
        # precision gate on first computed pair: float32 vs float64 on 2 cells
        if i == 0 or (gate_err == 0.0 and not done):
            R64 = fit_R(cal_s[EXI], cal_t[EXI], np.float64)
            e1 = abs(acos(dom_s['certainty'], dom_t['certainty'], R64) - row['certainty'])
            e2 = abs(acos(dom_s[EXI], dom_t[EXI], R64) - row[EXI])
            gate_err = max(e1, e2)
            print(f"[GATE] float32 vs float64 err={gate_err:.2e} (<1e-3 ok)", flush=True)
            if gate_err > 1e-3:
                raise SystemExit(f"precision gate failed: {gate_err:.2e}")
        done[str(i)] = {"row": row, "col": col}
        json.dump(done, open(CKPT, "w"))
        print(f"  [{i+1}/{len(pairs)}] {s} x {t} dim={C.ROSTER[s][1]} ({time.time()-t0:.0f}s)", flush=True)

    # aggregate row/col means over pairs
    valid = [v for v in done.values() if not v.get("skip")]
    row = {B: float(np.mean([v["row"][B] for v in valid if B in v["row"]])) for B in CONCEPTS}
    col = {A: float(np.mean([v["col"][A] for v in valid if A in v["col"]])) for A in CONCEPTS if A != EXI}
    out = {"exfil_row": row, "exfil_col": col, "diagonal": row[EXI],
           "n_pairs": len(valid), "method": "full-dim Procrustes, float32, corrected exfil",
           "gate_err": gate_err, "elapsed_s": time.time() - t0}
    (OUT / "exfil_fulldim_result.json").write_text(json.dumps(out, indent=2))
    print(f"\n=== DONE: exfil diagonal {row[EXI]:.4f} (stale 0.9141), row-mean(offdiag) "
          f"{np.mean([row[b] for b in CONCEPTS if b!=EXI]):.4f}, col-mean {np.mean(list(col.values())):.4f} "
          f"({time.time()-t0:.0f}s) ===", flush=True)
    print(f"wrote {OUT}/exfil_fulldim_result.json", flush=True)

if __name__ == "__main__":
    main()

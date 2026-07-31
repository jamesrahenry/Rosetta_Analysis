#!/usr/bin/env python3
"""Uncertainty for every Phase-A / depth-program statistic (audit gap 4).

Nothing in Phase A carried error bars, and every quoted n double-counts
ordered pairs (s->t and t->s are distinct fits but not independent samples).
This harness bootstraps at the level that IS exchangeable — the UNORDERED
model pair — over the stored per-cell artifacts, B resamples each.

Covered (artifact -> statistics):
  ccscr_<L>/crossconcept_scramble.json  -> same_true, cross_true_L, cross_scr_L,
                                           cL/same ratio, correspondence gap
                                           (cross_L - scr_L), scr-vs-product r
  drt_<...>/depth_transfer.json         -> per-depth cross/same ratio
  smt_<L>/stage_matched.json            -> per-anchor paired (stage - depth)
  oob_<L>/oob_floor.json                -> in-basis and out-of-basis floors

NOT covered here: the §1 cluster floors/margins — spectrum_floor.json stores
pooled means without per-pair sample retention, so a pair-level bootstrap
needs a retention rerun. Reported as "CI pending retention rerun" rather than
faked from the wrong unit.

Usage: python phase_a_bootstrap.py --root /storage/JamesData/p4_nullfloor \
           --B 2000 --out phase_a_ci.json
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def upair(r):
    return tuple(sorted([r["s"], r["t"]]))


def boot_ci(rows, stat, B, rng):
    """Percentile CI for stat(rows) resampling unordered pairs with replacement."""
    groups = defaultdict(list)
    for r in rows:
        groups[upair(r)].append(r)
    keys = list(groups)
    if len(keys) < 2:
        return None
    point = stat(rows)
    draws = []
    for _ in range(B):
        sel = [r for k in rng.choice(len(keys), len(keys), replace=True)
               for r in groups[keys[k]]]
        draws.append(stat(sel))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(point=float(point), lo=float(lo), hi=float(hi),
                n_unordered_pairs=len(keys), n_cells=len(rows))


def mean_of(key):
    return lambda rows: float(np.nanmean([r[key] for r in rows]))


def ratio_of(num, den):
    return lambda rows: (float(np.nanmean([r[num] for r in rows]))
                         / float(np.nanmean([r[den] for r in rows])))


def gap_of(a, b):
    return lambda rows: (float(np.nanmean([r[a] for r in rows]))
                         - float(np.nanmean([r[b] for r in rows])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default="phase_a_ci.json")
    a = ap.parse_args()
    root = Path(a.root)
    rng = np.random.default_rng(20260731)
    out = {"B": a.B, "unit": "unordered model pair",
           "floors_note": "cluster spectrum floors: CI pending retention rerun",
           "crossconcept": {}, "depth": {}, "stage": {}, "oob": {}}

    for f in sorted(glob.glob(str(root / "ccscr_*" / "crossconcept_scramble.json"))):
        if "partial" in f:
            continue
        d = json.load(open(f))
        rows = d["rows"]
        out["crossconcept"][d["cluster"]] = {
            "same_true": boot_ci(rows, mean_of("same_true"), a.B, rng),
            "cross_true_L": boot_ci(rows, mean_of("cross_true_L"), a.B, rng),
            "cross_scr_L": boot_ci(rows, mean_of("cross_scr_L"), a.B, rng),
            "cL_over_same": boot_ci(rows, ratio_of("cross_true_L", "same_true"), a.B, rng),
            "correspondence_gap": boot_ci(rows, gap_of("cross_true_L", "cross_scr_L"), a.B, rng),
        }
        print(f"ccscr {d['cluster']}: done")

    depth_files = (glob.glob(str(root / "drt_*" / "depth_transfer.json")))
    merged = defaultdict(list)
    for f in depth_files:
        d = json.load(open(f))
        for r in d["rows"]:
            merged[d["meta"]["cluster"]].append(r)
    for L, rows in merged.items():
        byf = defaultdict(list)
        for r in rows:
            byf[r["f"]].append(r)
        out["depth"][L] = {
            str(f): boot_ci(v, ratio_of("cross_true", "same_true"), a.B, rng)
            for f, v in sorted(byf.items())
        }
        print(f"depth {L}: done ({len(byf)} depths)")

    for f in sorted(glob.glob(str(root / "smt_*" / "stage_matched.json"))):
        d = json.load(open(f))
        div = [r for r in d["rows"] if r["f_div"] > 0.08]
        byanchor = defaultdict(list)
        for r in div:
            byanchor[r["anchor"]].append(r)
        out["stage"][d["cluster"]] = {
            anchor: boot_ci(v, gap_of("stage_cross", "depth_cross"), a.B, rng)
            for anchor, v in sorted(byanchor.items())
        }
        print(f"stage {d['cluster']}: done")

    for f in sorted(glob.glob(str(root / "oob_*" / "oob_floor.json"))):
        d = json.load(open(f))
        rows = d["rows"]
        out["oob"][d["cluster"]] = {
            "in_basis": boot_ci(rows, mean_of("in_basis"), a.B, rng),
            "out_basis": boot_ci(rows, mean_of("out_basis"), a.B, rng),
        }
        print(f"oob {d['cluster']}: done")

    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""P4/PRH-0 reframe prose <-> artifact consistency checker — successor to
consistency_check.py, written for the NEW claim set (REFRAME_OUTLINE_DRAFT.md)
against `preprint_v2.md`, per that outline's §7 process ("a successor to
consistency_check.py written against the new claim set before the abstract is
drafted"). The old script keeps guarding `preprint.md` (now historical/superseded
prose, not deleted) — this one guards the reframe.

Computes P4's new load-bearing quantities directly from the Phase-A / eigengap
HF artifacts (`paper_n250/_phase_a/`), then checks that `preprint_v2.md` states
each one and does NOT carry retired pre-reframe language. Same two-part design as
the original: `canonical()` (compute from artifacts) + `DRIFT` (retired phrasing
that must not reappear as a live claim).

Usage:  python consistency_check_v2.py [--preprint PATH]
Exit 0 if all canonical values present and no drift flagged; 1 otherwise.
"""
import sys, json, re, argparse, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import common as C

def hf(path):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(C.HF_REPO, path, repo_type="dataset")

# Clusters G/B/H/D/C/E were correct on their first full-coverage run; A required
# a guarded rerun (nullfloor_full_A_spectrum_r2) after the audit found lost cells;
# F's pooled 17-concept figure was never stored as one artifact (only a 5-concept
# sample, `_floor5raw`, is on disk) — it is the reconciled sum of a 5-concept and a
# 12-concept run reported in PHASE_A_NULLFLOOR_RESULTS.md §1/§6, a documented gap
# carried forward from the old script's own precedent of citing known, undocumented-
# in-code values rather than silently omitting or faking a recompute.
FLOOR_ARTIFACTS = {
    "A": "paper_n250/_phase_a/nullfloor_full_A_spectrum_r2/spectrum_floor.json",
    "G": "paper_n250/_phase_a/nullfloor_full_G_floor/spectrum_floor.json",
    "B": "paper_n250/_phase_a/nullfloor_full_B_floor/spectrum_floor.json",
    "H": "paper_n250/_phase_a/nullfloor_full_H_floor/spectrum_floor.json",
    "D": "paper_n250/_phase_a/nullfloor_full_D_floor/spectrum_floor.json",
    "C": "paper_n250/_phase_a/nullfloor_full_C_floor/spectrum_floor.json",
    "E": "paper_n250/_phase_a/nullfloor_full_E_floor/spectrum_floor.json",
}
CCSCR = {L: f"paper_n250/_phase_a/ccscr_{L}/crossconcept_scramble.json" for L in "AGBHDCEF"}
CCMAT = {L: f"paper_n250/_phase_a/ccmat_{L}/matrix.json" for L in "ABC"}


def canonical():
    """Compute every load-bearing PRH-0 quantity from the artifacts of record."""
    v = {}

    # --- §3: spectrum-matched floor, the honest null (PHASE_A §1) ---
    floor_by_cluster = {}
    for L, path in FLOOR_ARTIFACTS.items():
        d = json.load(open(hf(path)))[L]
        floor_by_cluster[L] = d["floor_mean"]
        v[f"floor {L}"] = (round(d["floor_mean"], 4), None)
    # F's pooled figure is not independently recomputable from one artifact (see
    # FLOOR_ARTIFACTS comment) — documented value from PHASE_A_NULLFLOOR_RESULTS.md §1/§6.
    v["floor F (documented, PHASE_A §1/§6)"] = (0.4614, "0.4614")
    v["floor range (min/max over G,B,H,D,C,E,A)"] = (
        (round(min(floor_by_cluster.values()), 2), round(max(floor_by_cluster.values()), 2)),
        "0.32",
    )

    # --- §4: out-of-basis floor (PHASE_A §4.1) ---
    oob_vals = []
    for L in ("A", "B"):
        d = json.load(open(hf(f"paper_n250/_phase_a/oob_{L}/oob_floor.json")))
        oob_vals.extend(r["out_basis"] for r in d["rows"])
    v["oob floor mean"] = (round(float(np.mean(oob_vals)), 4), "0.000")

    # --- §4: cross-concept transport, cL/same (PHASE_A §4.5/§4.7) ---
    cl_same = {}
    for L, path in CCSCR.items():
        d = json.load(open(hf(path)))
        cl_same[L] = d["cross_true_L"] / d["same_true"]
        v[f"cL/same {L}"] = (round(cl_same[L], 2), None)
    seven_of_eight = [x for L, x in cl_same.items() if L != "D"]  # D is the documented flat-spectrum exception
    v["cL/same range (7 of 8 clusters, excl. D)"] = (
        (round(min(seven_of_eight), 2), round(max(seven_of_eight), 2)), "0.66",
    )
    v["cL/same D (Gemma-2 exception)"] = (round(cl_same["D"], 2), "0.39")

    # --- §4: 17x17 arrangement-transport matrix (PHASE_A §4.8) ---
    # schema: one row per (s, t, fit-concept), 'diag' = that row's same-concept
    # value, 'tests' = {test_concept: {true, scr, overlap}} for the other 16.
    for L, path in CCMAT.items():
        d = json.load(open(hf(path)))
        rows = d["rows"]
        diag = [r["diag"] for r in rows]
        offd = [t["true"] for r in rows for t in r["tests"].values()]
        scr = [t["scr"] for r in rows for t in r["tests"].values()]
        v[f"17x17 diagonal {L}"] = (round(float(np.mean(diag)), 4), None)
        v[f"17x17 off-diag {L}"] = (round(float(np.mean(offd)), 3), None)
        v[f"17x17 scrambled {L}"] = (round(float(np.mean(scr)), 3), None)

    # --- §4.7: the natural experiment (E's per-pair vintage split) ---
    # Not separable from ccscr_E's cluster-level aggregate (no per-pair field in that
    # artifact) — documented values from PHASE_A_NULLFLOOR_RESULTS.md §4.7's table.
    v["natural experiment: 14B pairs cL/same"] = (0.673, "0.673")
    v["natural experiment: 32B pairs cL/same"] = (0.188, "0.188")
    v["natural experiment: 32B same-concept (blind metric)"] = (0.9906, "0.9906")
    v["natural experiment: 32B scrambled-fit floor"] = (0.149, "0.149")

    # --- §5 (slim): depth-pilot ratio range, 3-cluster pilot (DEPTH_PILOT_RESULTS.md) ---
    # documented, not recomputed live here — per-depth curves are a heavier parse
    # than this guardrail script needs; the numbers are the audited source of truth.
    v["depth ratio range (A/B/C pilot, f=0.1-0.9)"] = ((0.61, 0.74), "0.6")
    v["depth ratio peak (all 3 clusters at f=0.8)"] = ((0.737, 0.711, 0.736), "0.8")
    v["deepest_peak stage-vs-depth (A/B/C, ties)"] = ((0.011, 0.001, -0.004), None)

    # --- eigengap probe H1/H2 (EIGENGAP_PROBE_PLAN.md §11) ---
    # Raw spectral_stats.json/summary.json were not preserved on disk after the run
    # (only figures survived to upload) — a real regeneration gap, disclosed rather
    # than papered over; documented values from the committed verdict note.
    v["eigengap H1: effective-rank vs floor rho"] = (1.00, "+1.00")
    v["eigengap H2: Gemma-2-9b-it participation ratio"] = (63.2, "63.2")
    v["eigengap H2: Gemma-2-9b participation ratio"] = (46.4, "46.4")
    v["eigengap H2: cluster flatness vs cL/same rho"] = (-0.77, "-0.77")
    v["eigengap surprise: subspace overlap vs chance (7 clusters, ~3dp)"] = (
        "A0.0220/0.0221 G0.0164/0.0166 B0.0082/0.0083 H0.0067/0.0066 D0.0043/0.0047 C0.0041/0.0042 E0.0034/0.0033",
        None,
    )
    v["eigengap H4: n-sweep v2 exponent"] = (0.26, "0.26")

    return v


# Retired pre-reframe language that must NOT reappear as a live claim in
# preprint_v2.md. Historical preprint.md is untouched and unguarded by this script
# (it keeps its own consistency_check.py); this DRIFT list is specific to the
# reframe's own retirements (REFRAME_OUTLINE_DRAFT.md §4/§6), not a duplicate of
# the old script's pre-correction watchlist.
DRIFT = [
    # the retired ≈0-null framing and its derived effect sizes (old §3.1/§3.2)
    (r"4\.9 null-SDs?",                 "floor-referenced margin, not null-SD", 0),
    (r"~?130 SE",                       "floor-referenced margin",              0),
    (r"4\.5\s?[×x]\s?SNR",              "floor-referenced margin",              0),
    (r"against a near-zero (pre-rotation )?baseline", "against the spectrum-matched floor (0.32-0.53)", 0),
    (r"[Cc]onfirms?\s+(the\s+)?PRH",    "consistent with, does not adjudicate the PRH", 0),
    (r"[Ee]vidence for the Platonic Representation Hypothesis", "the question the measurement bears on", 0),
    # the retired 0.209 universality-ratio-as-concept-specificity claim (old §3.3)
    (r"0\.209.{0,40}concept.specific",  "layer-convention non-robustness (both 0.21 and 0.68 reported)", 0),
    (r"universality ratio.{0,20}confirms", "layer-convention non-robustness finding", 0),
    # the retired old-§3.7 handoff-alignment result (decision 4, not carried forward)
    (r"handoff.layer.{0,30}0\.9635",    "retired, not re-derived (outline §6 decision 4)", 0),
    # the cut ARC/MZC cross-pollination narrative (outline §6 decision 6 — must never
    # appear, not even as a footnote)
    (r"\bARC\b.{0,40}\bMZC\b|\bMZC\b.{0,40}\bARC\b", "cut entirely per outline §6 decision 6", 0),
    (r"inverse regime",                 "cut entirely per outline §6 decision 6", 0),
    # the retracted "exact at every depth" overclaim on the subspace-overlap finding
    (r"exact(ly)? chance.{0,20}every depth", "at or very near chance, small elevation at depth", 0),
    # retraction-ledger framing (this is a net-new paper, not a revision — outline §4)
    (r"[Cc]hanges from [Vv]ersion 1",   "not applicable — net-new paper",       0),
    (r"\bpre-correction\b|\boriginally reported\b", "state current value only", 0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprint", default=None,
                    help="path to preprint_v2.md; else $P4_PREPRINT_V2, else common Rosetta_Program checkouts")
    args = ap.parse_args()
    pp = args.preprint or os.environ.get("P4_PREPRINT_V2")
    if not pp:
        for cand in (Path.home()/"Games2/Eigan/Rosetta_Program/papers/prh-validation/preprint_v2.md",
                     Path.home()/"Source/Rosetta_Program/papers/prh-validation/preprint_v2.md",
                     Path(__file__).resolve().parents[3]/"papers/prh-validation/preprint_v2.md"):
            if cand.exists(): pp = str(cand); break
    raw = Path(pp).read_text() if pp and Path(pp).exists() else ""
    if not raw:
        print(f"[skip] preprint_v2 not found ({pp}) — pass --preprint or set $P4_PREPRINT_V2"); return
    text = raw.replace("−", "-")  # normalize Unicode minus to ASCII

    def present_in(val, want):
        cands = {str(val)}
        if isinstance(val, float):
            cands |= {f"{val:.4f}", f"{val:.3f}", f"{val:.2f}", f"{val:.1f}"}
        if isinstance(val, tuple):
            cands |= {str(x) for x in val} | {f"{x:.2f}" for x in val if isinstance(x, float)}
        if want: cands.add(str(want).replace("−", "-"))
        return any(re.search(re.escape(str(c)), text) for c in cands)

    can = canonical()
    missing, ok = [], 0
    print("=== canonical values (artifact → present in preprint_v2.md?) ===")
    for name, (val, want) in can.items():
        present = present_in(val, want)
        if present: ok += 1
        else:
            missing.append((name, val, want))
            print(f"  MISSING  {name:52s} artifact={val}  (expected string '{want or val}')")
    print(f"  {ok}/{len(can)} canonical values present")

    print("\n=== drift watchlist (retired pre-reframe language) ===")
    drift_hits = 0
    for pat, should, allow in DRIFT:
        n = len(re.findall(pat, text, flags=re.IGNORECASE))
        if n > allow:
            drift_hits += 1
            print(f"  DRIFT  /{pat}/  x{n}  -> should read: {should}  (allowance {allow})")
    if not drift_hits: print("  none over allowance")

    bad = len(missing) + drift_hits
    print(f"\n{'FAIL' if bad else 'PASS'}: {len(missing)} missing, {drift_hits} drift")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

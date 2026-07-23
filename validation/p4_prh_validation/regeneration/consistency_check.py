#!/usr/bin/env python3
"""P4 prose ↔ artifact consistency checker (the G3 guardrail).

Computes P4's load-bearing quantities directly from the HF artifacts, then checks
that `preprint.md` states each one and does NOT carry a superseded value. This is
the automated version of the manual number-audit: run it at every data change so
prose/table drift (the A*/C*/D* defect class) can't survive a commit.

Usage:  python consistency_check.py [--preprint PATH]
Exit 0 if all canonical values present and no drift flagged; 1 otherwise.
"""
import sys, json, re, argparse
from pathlib import Path
import numpy as np
from collections import defaultdict
import common as C

def hf(path):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(C.HF_REPO, path, repo_type="dataset")

def canonical():
    """Compute every load-bearing quantity from the artifacts of record."""
    import csv
    v = {}
    # --- primary alignment (A–E / A–F / F) ---
    rows = list(csv.DictReader(open(hf("paper_n250/_alignment/prh_primary_xfam_samedim_C17.csv"))))
    ae = [float(r["aligned"]) for r in rows if int(r["dim"]) != 8192]
    F  = [float(r["aligned"]) for r in rows if int(r["dim"]) == 8192]
    allr = [float(r["aligned"]) for r in rows]
    v["A–E primary grand"]   = (round(np.mean(ae), 4), "0.9750")
    v["A–F primary grand"]   = (round(np.mean(allr), 4), "0.9752")
    v["F extension grand"]   = (round(np.mean(F), 4), "0.9785")
    v["A–E pair count"]      = (len(ae), "1,661")
    v["A–F pair count"]      = (len(allr), "1,763")
    v["F pair count"]        = (len(F), "102")
    byc = defaultdict(list)
    for r in rows:
        if int(r["dim"]) != 8192: byc[r["concept"]].append(float(r["aligned"]))
    for c, xs in byc.items():
        v[f"per-concept {c}"] = (round(np.mean(xs), 4), None)
    # --- permuted-label null ---
    pn = json.load(open(hf("paper_n250/_nulls/p4_nulls_C17_xfam.json")))["permuted_label_null"]
    v["permuted null mean"] = (round(pn["mean"], 4), "-0.0010")
    v["permuted null n"]    = (pn["n"], "41,500")
    # --- transfer matrix / universality (corrected) ---
    tm = json.load(open(hf("paper_n250/_universality_depth_confound/transfer_matrix_17x17.json")))
    M = np.array(tm["matrix"], float); ei = tm["concepts"].index("exfiltration")
    dia = [M[i][i] for i in range(17)]; off = [M[i][j] for i in range(17) for j in range(17) if i != j]
    v["universality same"]  = (round(np.mean(dia), 4), "0.9750")
    v["universality cross"] = (round(np.mean(off), 4), "0.2035")
    v["universality ratio"] = (round(np.mean(off) / np.mean(dia), 3), "0.209")
    v["transfer exfil diag"]= (round(M[ei][ei], 4), "0.9868")
    # --- GEM handoff (corrected) ---
    h = json.load(open(hf("paper_n250/_prh_gem_handoff/prh_gem_handoff_results_corrected.json")))
    pc = h["per_concept"]
    v["handoff grand"] = (round(sum(x["handoff_mean"]*x["n_pairs"] for x in pc.values())/sum(x["n_pairs"] for x in pc.values()), 4), "0.9635")
    v["handoff exfil"] = (round(pc["exfiltration"]["handoff_mean"], 4), "0.9424")
    return v

# Superseded values / revision framing that must NOT reappear as live claims.
# Policy (set 2026-07-23): this is an unpublished paper, so it states current values
# only — "was X → now Y" pre-correction framing is confined to the §3.1 exfiltration
# correction note. Any pre-correction value or revision phrasing OUTSIDE that note is
# drift. Allowances encode the legitimate keeps (e.g. 0.916 appears once, in the note).
DRIFT = [
    (r"z ≈ 1030",              "d≈4.9 / z≈130", 0),
    (r"universality ratio.{0,12}0\.205", "0.209", 0),
    (r"same-concept.{0,20}0\.9709",      "0.9750", 0),
    (r"\b1,766\b",             "1,768 nominal", 0),
    (r"\b1,659\b",             "1,661 A–E",     0),
    (r"~6\.5× SNR|6\.5× above","~4.5×",         0),
    # pre-correction exfiltration values — only 0.916 is allowed, once, in the §3.1 note
    (r"0\.916\b",              "0.9868 (except once in the §3.1 correction note)", 1),
    (r"0\.9022\b",             "0.9424 handoff", 0),
    (r"0\.9611\b",             "0.9635 handoff grand", 0),
    (r"0\.9141\b",             "0.9868 transfer diag", 0),
    # revision phrasing that implies a prior published version
    (r"pre-correction",        "state current value only", 0),
    (r"originally reported",   "state current value only", 0),
    (r"previously grouped|formerly appeared|earlier apparent weakness", "state current value only", 0),
]

def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprint", default=None,
                    help="path to preprint.md; else $P4_PREPRINT, else common Rosetta_Program checkouts")
    args = ap.parse_args()
    pp = args.preprint or os.environ.get("P4_PREPRINT")
    if not pp:
        for cand in (Path.home()/"Games2/Eigan/Rosetta_Program/papers/prh-validation/preprint.md",
                     Path.home()/"Source/Rosetta_Program/papers/prh-validation/preprint.md",
                     Path(__file__).resolve().parents[3]/"papers/prh-validation/preprint.md"):
            if cand.exists(): pp = str(cand); break
    raw = Path(pp).read_text() if pp and Path(pp).exists() else ""
    if not raw:
        print(f"[skip] preprint not found ({pp}) — pass --preprint or set $P4_PREPRINT"); return
    # normalize the Unicode minus (U+2212) the manuscript uses to ASCII '-'
    text = raw.replace("−", "-")
    def present_in(val, want):
        cands = {str(val)}
        if isinstance(val, float):                     # trailing-zero variants: -0.001 ↔ -0.0010
            cands |= {f"{val:.4f}", f"{val:.3f}", f"{val:.2f}"}
        if want: cands.add(want.replace("−", "-"))
        return any(re.search(re.escape(c), text) for c in cands)
    can = canonical()
    missing, ok = [], 0
    print("=== canonical values (artifact → present in preprint?) ===")
    for name, (val, want) in can.items():
        present = present_in(val, want)
        if present: ok += 1
        else:
            missing.append((name, val, want))
            print(f"  MISSING  {name:24s} artifact={val}  (expected string '{want or val}')")
    print(f"  {ok}/{len(can)} canonical values present")
    print("\n=== drift watchlist (superseded values still in prose) ===")
    drift_hits = 0
    for pat, should, allow in DRIFT:
        n = len(re.findall(pat, text))
        if n > allow:
            drift_hits += 1
            print(f"  DRIFT  /{pat}/  ×{n}  → should read {should}  (allowance {allow})")
    if not drift_hits: print("  none over allowance")
    bad = len(missing) + drift_hits
    print(f"\n{'FAIL' if bad else 'PASS'}: {len(missing)} missing, {drift_hits} drift")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()

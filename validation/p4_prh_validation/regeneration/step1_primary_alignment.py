#!/usr/bin/env python3
"""
step1_primary_alignment.py — regenerate the P4 primary zero-PCA Procrustes
alignment over the full A–F corpus (33 models, corrected exfiltration).

Reproduces `paper_n250/_alignment/prh_primary_xfam_samedim_C17.csv` on HF and
prints the §3.1 headline table.

Self-validation (always runs first): the rank-reduced Procrustes is checked
against the full-dimension reference on one cluster-F pair, aborting if they
differ by >1e-8; then a designated control concept is reproduced against the
stored CSV to <1e-4 before any exfiltration/F number is reported. This is the
"do not trust a number you cannot reproduce" gate.

Outputs (to --out-dir, default ./p4_regen_output):
  prh_primary_xfam_samedim_C17.csv   — 1,763 rows: concept,source,target,dim,fam_s,fam_t,raw,aligned
  step1_summary.json                 — grand mean/median/std + per-concept table

Usage:
  python step1_primary_alignment.py                 # full A–F, validate, write CSV
  python step1_primary_alignment.py --validate-only # just the validation gates
  python step1_primary_alignment.py --concepts exfiltration   # subset

Expected headline (2026-07-18): grand mean 0.9752, median 0.9938, 1,763 pairs;
exfiltration 0.9869 (rank 7); cluster F 0.9785; deception weakest 0.905.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np

import common as C

EXPECTED = {"grand_mean": 0.9752, "n_rows": 1763, "exfiltration": 0.9869, "F_mean": 0.9785}


def _validate(out_dir: Path):
    """Gate 1: reduced == full on an 8192-dim pair. Gate 2: reproduce a control
    concept against the published CSV."""
    from huggingface_hub import hf_hub_download
    print("[validate] reduced vs full Procrustes on a cluster-F pair (sentiment)...")
    s, t = "tiiuae_falcon_40b", "meta_llama_Llama_3.1_70B"
    ds, ps = C.load_dom_and_peak(s, "sentiment"); cs = C.load_calibration(s, "sentiment", ps)
    dt, pt = C.load_dom_and_peak(t, "sentiment"); ct = C.load_calibration(t, "sentiment", pt)
    vf = C.aligned_cosine_full(ds, dt, cs, ct)
    vr = C.aligned_cosine(ds, dt, cs, ct)
    print(f"           full={vf:.12f} reduced={vr:.12f} |diff|={abs(vf-vr):.2e}")
    if abs(vf - vr) > 1e-8:
        sys.exit("[validate] FAIL: reduced Procrustes deviates from full — aborting.")

    print("[validate] reproducing control concept 'certainty' vs published CSV...")
    p = hf_hub_download(C.HF_REPO, f"{C.HF_ROOT}/_alignment/prh_primary_xfam_samedim_C17.csv", repo_type="dataset")
    published = {(r["source"], r["target"]): float(r["aligned"])
                 for r in csv.DictReader(open(p)) if r["concept"] == "certainty"}
    maxerr = 0.0
    for (s, t), pub in published.items():
        ds, ps = C.load_dom_and_peak(s, "certainty"); cs = C.load_calibration(s, "certainty", ps)
        dt, pt = C.load_dom_and_peak(t, "certainty"); ct = C.load_calibration(t, "certainty", pt)
        maxerr = max(maxerr, abs(C.aligned_cosine(ds, dt, cs, ct) - pub))
    print(f"           max |aligned err| vs published = {maxerr:.2e}")
    if maxerr > 1e-4:
        sys.exit("[validate] FAIL: cannot reproduce a known concept — pipeline not faithful.")
    print("[validate] OK — pipeline reproduces published values; proceeding.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="./p4_regen_output")
    ap.add_argument("--concepts", nargs="*", default=C.CONCEPTS_17)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    _validate(out_dir)
    if args.validate_only:
        return

    pairs = C.cross_family_same_dim_pairs()
    print(f"Computing {len(pairs)} ordered cross-family same-dim pairs × {len(args.concepts)} concepts...")
    rows = []
    for ci, concept in enumerate(args.concepts, 1):
        # cache this concept's dom+cal per model (peak-layer slice; F streamed)
        cache = {}
        def get(slug):
            if slug not in cache:
                d, p = C.load_dom_and_peak(slug, concept)
                cache[slug] = (d, None if d is None else C.load_calibration(slug, concept, p))
            return cache[slug]
        for s, t in pairs:
            ds, cs = get(s); dt, ct = get(t)
            if ds is None or dt is None or np.linalg.norm(ds) < 1e-9 or np.linalg.norm(dt) < 1e-9:
                continue  # no measurable separation — unavailable fit (see §3.1 corpus note)
            rows.append({
                "concept": concept, "source": s, "target": t,
                "dim": str(C.ROSTER[s][1]), "fam_s": C.ROSTER[s][0], "fam_t": C.ROSTER[t][0],
                "raw": repr(C.raw_cosine(ds, dt)),
                "aligned": repr(C.aligned_cosine(ds, dt, cs, ct)),
            })
        print(f"  [{ci}/{len(args.concepts)}] {concept} done")

    csv_path = out_dir / "prh_primary_xfam_samedim_C17.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["concept", "source", "target", "dim", "fam_s", "fam_t", "raw", "aligned"])
        w.writeheader(); w.writerows(rows)

    allv = np.array([float(r["aligned"]) for r in rows])
    byc = defaultdict(list)
    for r in rows:
        byc[r["concept"]].append(float(r["aligned"]))
    fvals = [float(r["aligned"]) for r in rows if r["dim"] == "8192"]
    summary = {
        "n_rows": len(rows), "n_models": len(C.ROSTER),
        "grand_mean": round(float(allv.mean()), 4),
        "grand_median": round(float(np.median(allv)), 4),
        "grand_std": round(float(allv.std()), 4),
        "cluster_F_mean": round(float(np.mean(fvals)), 4) if fvals else None,
        "per_concept": {c: round(float(np.mean(v)), 4) for c, v in
                        sorted(byc.items(), key=lambda x: -np.mean(x[1]))},
    }
    (out_dir / "step1_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nGrand mean {summary['grand_mean']} | median {summary['grand_median']} | "
          f"{summary['n_rows']} rows | F {summary['cluster_F_mean']}")
    if args.concepts == C.CONCEPTS_17:
        for k, exp in EXPECTED.items():
            got = summary.get({"grand_mean": "grand_mean", "n_rows": "n_rows",
                               "F_mean": "cluster_F_mean"}.get(k, k),
                              summary["per_concept"].get(k))
            flag = "" if got is None or abs(got - exp) < (1 if k == "n_rows" else 0.001) else "  <-- DIFFERS FROM EXPECTED"
            print(f"  check {k:12s}: got {got}  expected {exp}{flag}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()

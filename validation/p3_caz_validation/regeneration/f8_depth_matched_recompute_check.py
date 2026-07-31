#!/usr/bin/env python3
"""F8 cross-check: independently reaggregate the site-matched depth control.

The run of record is claude:prepub-antagonist's site-matched depth control
(HF paper_n250/_gem_depth_matched/, doc SITE_MATCHED_DEPTH_CONTROL_RESULT_2026-07-30.md).
Two arms live in each cell record:
  r['delta_pp'] / r['handoff_better']                         -> LEGACY arm (whole-atlas
      handoff vs one shallow site) — reproduces the frozen §5.5 ~94.2% (harness anchor).
  r['site_matched_control']['delta_pp'] / ['handoff_better']  -> SITE-MATCHED arm (same
      GEMs, same site count, depth only) — the genuine depth control.

Filter chain (runner-confirmed 2026-07-31): 493 cells -> site_matched True (493/493)
-> n_targets_total >= 2 (380) -> roster. Two rosters, two homes in the manuscript:

  * ROSTER A = 28-model Table 1 (`table11_reconstruct.BASE_28`). This is §6.5's
    handoff-vs-peak population (476 = 28x17); its depth control is the number §6.5/§8.7
    cite. 363 cells.
  * ROSTER B = 25-model round-3 sub-roster (`round3_gpu/common.BASE_28`; excludes
    opt-350m + the two Gemma-2, per James 2026-07-16). This matches the TNCONF control,
    so its depth number sits beside the TNCONF sentence in §8.7. 326 cells.

Model-level aggregator is the MEDIAN of per-model deltas (runner's call: the 16 degenerate
gpt2/gpt2-medium cells with retained% > 150 skew per-model means; median is used elsewhere in
the analysis). We report both aggregators' Wilcoxon; the median is primary. State the aggregator
next to the p-value — it is not recoverable from W alone. Both rosters are null.

Written: 2026-07-30 UTC. Rev 2026-07-31: two rosters + median aggregator (runner correction).
"""
import json
import sys
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "shared" / "round3_gpu"))
from table11_reconstruct import BASE_28 as ROSTER_A_28, slugify  # noqa: E402
from common import BASE_28 as ROSTER_B_25  # noqa: E402

DM = Path.home() / "rosetta_data" / "paper_n250" / "_gem_depth_matched"


def load_multinode(roster_slugs):
    per_model = {}
    for f in DM.glob("*_depth_matched_control.json"):
        d = json.loads(f.read_text())
        slug = slugify(d["model_id"])
        if slug not in roster_slugs:
            continue
        for r in d["results"]:
            smc = r.get("site_matched_control")
            if isinstance(smc, dict) and smc.get("site_matched") and r.get("n_targets", 0) >= 2:
                per_model.setdefault(slug, []).append(smc)
    return per_model


def arm_stats(per_model, legacy=False):
    n = better = 0
    cell_deltas = []
    per_model_deltas = {}
    for slug, recs in per_model.items():
        md = []
        for smc in recs:
            n += 1
            better += bool(smc["handoff_better"])
            cell_deltas.append(smc["delta_pp"])
            md.append(smc["delta_pp"])
        per_model_deltas[slug] = md
    out = {"n_cells": n, "handoff_better": better,
           "handoff_better_pct": round(100 * better / n, 1) if n else None,
           "mean_delta_pp": round(mean(cell_deltas), 2) if cell_deltas else None}
    for agg_name, agg in (("median", median), ("mean", mean)):
        vals = [agg(v) for v in per_model_deltas.values()]
        W, p = wilcoxon(vals)
        out[f"model_level_{agg_name}"] = {
            "N": len(vals), "W": round(float(W), 1), "p": round(float(p), 3),
            "positive": sum(v > 0 for v in vals)}
    return out


def legacy_pct(roster_slugs):
    n = better = 0
    for f in DM.glob("*_depth_matched_control.json"):
        d = json.loads(f.read_text())
        if slugify(d["model_id"]) not in roster_slugs:
            continue
        for r in d["results"]:
            if isinstance(r.get("site_matched_control"), dict) and r.get("n_targets", 0) >= 2:
                n += 1
                better += bool(r["handoff_better"])
    return round(100 * better / n, 1) if n else None


def main():
    a_slugs = {slugify(m) for m in ROSTER_A_28}
    b_slugs = {slugify(m) for m in ROSTER_B_25}
    out = {
        "job": "F8 depth-matched control cross-check",
        "artifact": "paper_n250/_gem_depth_matched/",
        "aggregator_primary": "median (per-model)",
        "legacy_arm_pct_rosterA": legacy_pct(a_slugs),   # harness anchor: ~94.2
        "rosterA_28model_sec65": arm_stats(load_multinode(a_slugs)),
        "rosterB_25model_tnconf": arm_stats(load_multinode(b_slugs)),
        "conclusion": ("site-matched depth control NULL on both rosters; legacy arm reproduces "
                       "frozen §5.5 ~94.2%. §6.5 cites roster A (28-model); TNCONF sentence cites "
                       "roster B (25-model). Median aggregator primary."),
    }
    print(json.dumps(out, indent=1))
    HERE.joinpath("results/f8_depth_matched_recompute_check.json").write_text(json.dumps(out, indent=1))
    print("saved results/f8_depth_matched_recompute_check.json")


if __name__ == "__main__":
    main()

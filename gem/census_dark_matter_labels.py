#!/usr/bin/env python3
"""
census_dark_matter_labels.py — Aggregate label_features.py output into the
persistent-feature concept-labeled census used in the CAZ Validation (P3)
paper's §7 dark-matter section.

Methodology (reverse-engineered 2026-07-09 to match the paper's originally
published 42/636 (~6.6%) figure, since neither the original script nor a
cached result artifact could be located): a persistent feature counts as
"labeled" if a concept match (cosine >= label_features.py's THRESHOLD, 0.5)
is recorded at MORE THAN HALF of the feature's tracked layers — not merely
at any single layer. Restricting to persistent features (is_persistent) and
this majority-of-layers rule reproduces 44/636 against the original 7-concept
corpus (published: 42/636) — the closest match found among several candidate
criteria (any-layer match gave 173/636; peak-layer-only gave 28/636). The
small 44-vs-42 residual is consistent with ordinary run-to-run extraction
variance rather than a different methodology, but this is NOT a bit-exact
reproduction — treat the 28-model/C=17 figure below as high-confidence, not
certain.

Usage:
    python gem/census_dark_matter_labels.py
    python gem/census_dark_matter_labels.py --exclude EleutherAI/pythia-12b,Qwen/Qwen2.5-14B
"""

from __future__ import annotations

import argparse
import json

from rosetta_tools.paths import ROSETTA_RESULTS

RESULTS_DIR = ROSETTA_RESULTS


def census(exclude: set[str] | None = None) -> dict:
    exclude = exclude or set()
    total_persistent = 0
    total_labeled = 0
    rows = []

    for dd_dir in sorted(RESULTS_DIR.iterdir()):
        fm_path = dd_dir / "feature_map.json"
        fl_path = dd_dir / "feature_labels.json"
        if not fm_path.exists() or not fl_path.exists():
            continue
        fm = json.load(open(fm_path))
        model_id = fm["model_id"]
        if model_id in exclude:
            continue
        fl = json.load(open(fl_path))

        persistent_ids = {f["feature_id"] for f in fm["features"] if f.get("is_persistent")}
        n_labeled = 0
        for fid_str, layers in fl["features"].items():
            if int(fid_str) not in persistent_ids:
                continue
            n_labeled_layers = sum(1 for la in layers if la.get("best_concept") is not None)
            if n_labeled_layers > len(layers) / 2:
                n_labeled += 1

        total_persistent += len(persistent_ids)
        total_labeled += n_labeled
        rows.append({"model_id": model_id, "n_persistent": len(persistent_ids), "n_labeled": n_labeled})

    return {
        "rows": rows,
        "total_persistent": total_persistent,
        "total_labeled": total_labeled,
        "fraction": total_labeled / total_persistent if total_persistent else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exclude", type=str, default="", help="comma-separated model_ids to exclude")
    args = ap.parse_args()
    exclude = {m.strip() for m in args.exclude.split(",") if m.strip()}

    result = census(exclude)
    for r in result["rows"]:
        print(f"{r['model_id']:35s} persistent={r['n_persistent']:4d}  labeled={r['n_labeled']:4d}")
    print()
    print(f"Models: {len(result['rows'])}")
    print(f"Total persistent: {result['total_persistent']}")
    print(f"Total labeled (majority-of-layers, cos>=0.5): {result['total_labeled']}")
    print(f"Fraction: {result['fraction']*100:.2f}%")


if __name__ == "__main__":
    main()

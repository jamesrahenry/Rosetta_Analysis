#!/usr/bin/env python3
"""
extract_eec_table2.py — Extract per-model mean EEC for the full P2 corpus.

Reads all gem_*.json files for each model in P2_MODELS and computes mean
entry-exit cosine (EEC), handoff cosine, and max rotation per model and
concept. Outputs a full corpus JSON for Table 2 in the GEM paper (Paper 2).

Usage:
    python gem/extract_eec_table2.py                  # full P2 corpus
    python gem/extract_eec_table2.py --all            # all models with GEMs

Outputs: ~/rosetta_data/results/gem_eec_corpus.json

Updated: 2026-05-10
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

from rosetta_tools.gem import load_gem, find_extraction_dir, gem_diagnostics, discover_all_models
from rosetta_tools.paths import ROSETTA_MODELS

sys.path.insert(0, str(Path(__file__).parent.parent))
from gem.ablate_gem import P2_MODELS

OUT_PATH = Path.home() / "rosetta_data" / "results" / "gem_eec_corpus.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all models with GEMs, not just P2 corpus")
    args = parser.parse_args()

    models = discover_all_models(ROSETTA_MODELS) if args.all else P2_MODELS

    results = {}
    for model_id in models:
        edir = find_extraction_dir(model_id, ROSETTA_MODELS)
        if edir is None:
            print(f"ERROR: No extraction dir for {model_id}")
            continue
        gem_files = sorted(edir.glob("gem_*.json"))
        if not gem_files:
            print(f"ERROR: No gem_*.json found in {edir}")
            continue

        per_concept = {}
        for gf in gem_files:
            concept = gf.stem.replace("gem_", "")
            gem = load_gem(gf)
            diag = gem_diagnostics(gem)
            eec = diag.get("entry_exit_cosine_mean")
            if eec is not None:
                per_concept[concept] = round(float(eec), 3)

        if not per_concept:
            print(f"ERROR: No EEC values found for {model_id}")
            continue

        mean_eec = round(float(np.mean(list(per_concept.values()))), 3)
        results[model_id] = {"mean_eec": mean_eec, "per_concept": per_concept}
        print(f"\n{model_id}: mean_eec={mean_eec:.3f}  (n={len(per_concept)} concepts)")
        for concept, eec in sorted(per_concept.items()):
            print(f"  {concept}: {eec:.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()

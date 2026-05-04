#!/usr/bin/env python3
"""
extract_eec_table2.py — Extract per-model mean EEC for pythia-12b and Qwen2.5-14B.

These two models were added to the 16-model paper corpus after Table 2 was
written with 14 models. This script reads their already-built gem_*.json files
and outputs mean EEC per model, to fill the remaining Table 2 rows in the GEM
paper (Paper 2).

Outputs: ~/rosetta_data/results/gem_eec_new_models.json

Written: 2026-05-04 02:23 UTC
"""

import json
import numpy as np
from pathlib import Path

from rosetta_tools.gem import load_gem, find_extraction_dir, gem_diagnostics
from rosetta_tools.paths import ROSETTA_MODELS

MODELS = [
    "EleutherAI/pythia-12b",
    "Qwen/Qwen2.5-14B",
]

OUT_PATH = Path.home() / "rosetta_data" / "results" / "gem_eec_new_models.json"


def main():
    results = {}
    for model_id in MODELS:
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

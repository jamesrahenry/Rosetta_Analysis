# rosetta_analysis

*Written: 2026-04-22 UTC*

Analysis scripts for the **Rosetta** interpretability research program.
Companion to [`rosetta_tools`](https://github.com/james-henry-git/rosetta_tools) (the shared library)
and [`Rosetta_Concept_Pairs`](https://github.com/james-henry-git/Rosetta_Concept_Pairs) (the dataset).

## Structure

```
extraction/   Model activation extraction — extract.py, pair generation, validation
alignment/    Cross-architecture Procrustes alignment — PRH paper pipeline
caz/          CAZ detection and analysis — deep dive, ablation controls, CKA validation
gem/          GEM (Geometric Evolution Map) — assembly zone ablation and diagnostics
viz/          Visualization scripts for all three paper threads
jobs/         GPU job files for the Hopper/daemon queue
```

## Dependencies

```bash
pip install rosetta-tools           # library: extraction, caz, gem, alignment, dataset
pip install torch transformers      # model loading
pip install numpy scipy matplotlib  # numerics and visualization
```

## Data

Concept pairs are loaded automatically from
[`Rosetta_Concept_Pairs`](https://github.com/james-henry-git/Rosetta_Concept_Pairs)
via `ROSETTA_CONCEPTS_ROOT` or a repo-relative path search.

Results write to `~/rosetta_data/` on the GPU box (rsync-only, not git).

## Reproducibility

All published results were produced by running the full extraction suite
(`extraction/extract.py`) against each model with `N=200` pairs from the
`v1` dataset split, followed by the analysis scripts in this repo.
See `PAPER_MAP.md` for the exact script → figure mapping per paper.

## Papers

| Paper | Topic | Key scripts |
|---|---|---|
| Paper 1 — CAZ | Concept Allocation Zones | `caz/deep_dive.py`, `caz/analyze.py`, `viz/viz_caz_anatomy.py` |
| Paper 2 — GEM | Geometric Evolution Map | `gem/build_gems.py`, `gem/ablate_gem.py`, `gem/aggregate_gem_results.py` |
| Paper 3 — PRH | Platonic Representation Hypothesis | `alignment/align.py`, `alignment/align_trajectory.py` |

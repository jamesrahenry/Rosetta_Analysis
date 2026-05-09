# rosetta_analysis

*Written: 2026-04-22 00:00 UTC*
*Updated: 2026-05-09 23:10 UTC*

Analysis scripts for the **Rosetta** interpretability research program.
Companion to [`rosetta_tools`](https://github.com/jamesrahenry/rosetta_tools) (the shared library)
and [`Rosetta_Concept_Pairs`](https://github.com/jamesrahenry/Rosetta_Concept_Pairs) (the dataset).

## Structure

```
extraction/   Model activation extraction — extract.py is the primary data generator
alignment/    Cross-architecture Procrustes alignment — PRH paper pipeline
caz/          CAZ detection and analysis — ablation controls, CKA validation
gem/          GEM (Geometric Evolution Map) — assembly zone ablation and diagnostics
viz/          Visualization scripts for all paper threads
scripts/      Reproduction scripts (one per paper)
validation/   Claim verification test suites (one suite per paper)
```

## Reproducing paper results

Each paper has a reproduction script that runs the full pipeline and then verifies
all quantitative claims with the validation suite:

```bash
# Paper 1 — CAZ Framework
# Quick (GPT-2-XL only, ~20 min on GPU):
./scripts/reproduce_p1.sh --quick

# Full corpus (~6h on L4 24GB):
./scripts/reproduce_p1.sh
```

**Requirements:** NVIDIA GPU (≥16GB VRAM), `HF_TOKEN` set for gated models,
concept pairs available (see below).

After running, the validation suite can be re-run independently at any time:

```bash
pytest validation/p1_caz_framework/ -m "not slow"   # fast claims only
pytest validation/p1_caz_framework/                  # full suite
```

## Dependencies

```bash
pip install -r requirements.txt
```

## Concept pairs

Concept pairs are loaded from
[`Rosetta_Concept_Pairs`](https://github.com/jamesrahenry/Rosetta_Concept_Pairs).
The simplest setup is to clone it alongside this repo:

```bash
git clone https://github.com/jamesrahenry/Rosetta_Concept_Pairs ../Rosetta_Concept_Pairs
```

Alternatively, set `ROSETTA_CONCEPTS_ROOT` to the directory containing
the `*_consensus_pairs.jsonl` files (`pairs/raw/v1/` in the dataset repo).

## Data output

Results write to `~/rosetta_data/` (rsync-only, not git).
Canonical extraction uses `N=250` pairs from the `v1` training split.

## Papers

| Paper | Topic | Reproduce |
|---|---|---|
| Paper 1 — CAZ | Concept Allocation Zones | `./scripts/reproduce_p1.sh` |
| Paper 2 — GEM | Geometric Evolution Map | `./scripts/reproduce_p2.sh` *(pending)* |
| Paper 3 — PRH | Platonic Representation Hypothesis | `./scripts/reproduce_p3.sh` *(pending)* |

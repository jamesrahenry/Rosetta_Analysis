# rosetta_analysis

*Written: 2026-04-22 00:00 UTC*
*Updated: 2026-05-11 01:41 UTC*

Analysis scripts for the **Rosetta** interpretability research program.
Companion to [`rosetta_tools`](https://github.com/jamesrahenry/rosetta_tools) (the shared library)
and [`Rosetta_Concept_Pairs`](https://github.com/jamesrahenry/Rosetta_Concept_Pairs) (the dataset).

## Structure

```
extraction/   Model activation extraction — extract.py is the primary data generator
alignment/    Cross-architecture Procrustes alignment — PRH paper pipeline
caz/          CAZ detection and analysis — ablation controls, Gemma Scope cross-validation
gem/          GEM (Geometric Evolution Map) — assembly zone ablation and diagnostics
viz/          Visualization scripts for all paper threads
scripts/      Reproduction scripts (one per paper)
validation/   Claim verification test suites (one suite per paper)
```

## Papers

| # | Topic | Reproduce script | CPU companion |
|---|-------|-----------------|---------------|
| 1 | CAZ Framework | `reproduce_p1.sh` | — (single script) |
| 2 | GEM — Geometric Evolution Map | `reproduce_p2.sh` | — (single script) |
| 3 | CAZ Validation | `reproduce_p3.sh` | — (single script) |
| 4 | PRH — Concept-Selective Convergence | `reproduce_p4.sh --gpu-only` | `reproduce_p4_cpu.sh` |

## Reproducing paper results

All scripts default to `N=250` pairs, the canonical corpus size. **Skip logic is built
in**: an interrupted run can be restarted and resumes from the next incomplete
model/concept. The skip validates model ID, concept, and pair count — a result
extracted at N=200 is treated as incomplete when N=250 is requested and will be re-run.

### Paper 1 — CAZ Framework

```bash
# Quick proof-of-concept (GPT-2-XL only, ~20 min):
./scripts/reproduce_p1.sh --quick

# Full corpus (~6h on L4 24GB):
./scripts/reproduce_p1.sh

# H200 / large-VRAM machine (keep HF cache between models for speed):
./scripts/reproduce_p1.sh --no-clean-cache
```

### Paper 2 — GEM

```bash
# Quick (GPT-2-XL only):
./scripts/reproduce_p2.sh --quick

# Full P2 corpus (16 models × 17 concepts):
./scripts/reproduce_p2.sh

# H200:
./scripts/reproduce_p2.sh --no-clean-cache
```

### Paper 3 — CAZ Validation

```bash
# Quick (GPT-2-XL only, ~45 min):
./scripts/reproduce_p3.sh --quick

# Full corpus (26 base models):
./scripts/reproduce_p3.sh

# H200 — GPU steps only, defer CPU analysis:
./scripts/reproduce_p3.sh --gpu-only --no-clean-cache

# Include instruct variants (Supplementary §B):
./scripts/reproduce_p3.sh --with-instruct
```

Requires [`Rosetta_Feature_Library`](https://github.com/jamesrahenry/Rosetta_Feature_Library)
at `~/Rosetta_Feature_Library` for the Gemma Scope SAE cross-validation step.

### Paper 4 — PRH (split GPU / CPU workflow)

P4 separates GPU extraction from CPU analysis. Run the GPU phase on a large-VRAM
machine, sync results, then run the CPU phase anywhere.

```bash
# GPU phase (H200 recommended):
./scripts/reproduce_p4.sh --gpu-only --no-clean-cache

# With frontier models (falcon-40b, Llama-70B, Qwen-72B) — requires ~140GB VRAM:
./scripts/reproduce_p4.sh --gpu-only --with-frontier --no-clean-cache

# Sync results to dev machine:
bash rosetta_tools/bin/sync_results.sh

# CPU phase (Procrustes alignment, nulls, P5 analysis):
./scripts/reproduce_p4_cpu.sh

# CPU options:
./scripts/reproduce_p4_cpu.sh --with-frontier        # include frontier alignment
./scripts/reproduce_p4_cpu.sh --skip-p5              # alignment only, skip P5 battery
./scripts/reproduce_p4_cpu.sh --skip-alignment-nulls # P5 battery only (nulls superseded by battery)
```

## Requirements

- **GPU**: NVIDIA ≥16GB VRAM for P1–P3; H200 recommended for P4 (frontier models need ~140GB)
- **HF_TOKEN**: required for gated models (Llama-3.x, Gemma-2, Mistral)
- **uv**: manages the Python environment — install from [astral.sh/uv](https://docs.astral.sh/uv/)
- **Concept pairs**: see [Concept pairs](#concept-pairs) below
- **P3 Gemma Scope step**: `Rosetta_Feature_Library` at `~/Rosetta_Feature_Library`

## Concept pairs

Clone [`Rosetta_Concept_Pairs`](https://github.com/jamesrahenry/Rosetta_Concept_Pairs)
alongside this repo:

```bash
git clone https://github.com/jamesrahenry/Rosetta_Concept_Pairs ../Rosetta_Concept_Pairs
```

Or set `ROSETTA_CONCEPTS_ROOT` to the directory containing the `*_consensus_pairs.jsonl`
files (`pairs/raw/v1/` in the dataset repo).

## Disk management

With `--no-clean-cache`, all model weights accumulate in the HF cache simultaneously —
fast but requires large disk. On a machine with a dedicated scratch volume:

```bash
export HF_HUB_CACHE=/mnt/scratch/hf_cache
```

Without `--no-clean-cache` (the default), each model's cache is purged after extraction.
Slower due to re-downloads but disk-safe on any machine.

## Restoring activations on a new compute node

Extracted activations are published to HuggingFace and can be restored without re-running
GPU extraction:

```bash
hf download james-ra-henry/Rosetta-Activations \
    --repo-type dataset \
    --local-dir ~/rosetta_data/models/
```

HF skips files already present, so this is safe to run incrementally.

## Data output

Results write to `~/rosetta_data/` (rsync-only, not git tracked).
Canonical extraction uses `N=250` pairs from the `v1` training split.

| Paper | Output directory |
|-------|-----------------|
| P1 | `~/rosetta_data/results/CAZ_Framework/` |
| P2 | `~/rosetta_data/results/CAZ_GEM/` |
| P3 | `~/rosetta_data/results/CAZ_Validation/` |
| P4 | `~/rosetta_data/results/PRH/` |

## Validation suites

After running a reproduction script, claims can be verified independently:

```bash
pytest validation/p1_caz_framework/ -m "not slow"   # fast claims only
pytest validation/p1_caz_framework/                  # full suite
```

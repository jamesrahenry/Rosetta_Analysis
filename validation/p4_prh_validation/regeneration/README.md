# P4 data regeneration — PRH / Concept-Selective Convergence (A–F, corrected)

Self-contained, self-validating scripts that regenerate **every headline P4
quantity** from the published HuggingFace activation artifacts. Written
2026-07-18 to capture the corpus as it stands after two substantive changes
this cycle:

1. **Exfiltration label correction.** ~72% of exfiltration's positive/negative
   calibration labels were inverted; corrected via a recorded-draw
   reconstruction (N=249). Exfiltration went 0.916 (rank 16) → **0.9869**
   (rank 7); grand mean 0.9709 → 0.9750.
2. **Cluster F folded into the primary corpus.** falcon-40b, Llama-3.1-70B,
   Qwen2.5-72B are now full roster members (A–E → **A–F, 33 models**), not a
   separate case study. Grand mean → **0.9752**, 1,763 pairs. (The two 70–72B
   models' calibration is 8-bit-derived — VRAM-bound — and disclosed as such.)

These scripts do **not** depend on the older top-level `alignment/align.py`
pipeline. They carry their own authoritative 33-model roster, their own
Procrustes (validated identical to the full-SVD reference to machine
precision), and their own memory-safe HF loaders. Point of the exercise:
anyone can reproduce P4 from the public dataset with `numpy + scipy +
huggingface_hub`, and every number is checked against a known value before
it is trusted.

## Data of record

HF dataset **`james-ra-henry/Rosetta-Activations`**, tree `paper_n250/<slug>/`:
`caz_<concept>.json` (peak layer + DOM vector), `calibration_<concept>.npy`
(peak-layer calibration; A–E), `calibration_alllayer_<concept>.npy` (all-layer;
used for cluster F, where only this is stored). The authoritative primary
result lives at `paper_n250/_alignment/prh_primary_xfam_samedim_C17.csv`.

## What runs where

| Stage | Where | Script | Notes |
|---|---|---|---|
| 0. Activation extraction | **GPU** | `extraction/extract.py` (+ `scripts/reproduce_p4.sh --gpu-only`) | Produces the caz/calibration artifacts on HF. Not re-run to reproduce numbers — the artifacts are the frozen input. |
| 0b. Exfiltration correction | GPU (data-prep) | recorded-draw reconstruction → re-extract exfiltration at N=249 | Produced the corrected exfiltration artifacts now on HF. One-time; done. |
| **1. Primary alignment (A–F)** | **CPU** | `step1_primary_alignment.py` | Regenerates the primary CSV + §3.1 table. Self-validating. |
| **2. Headline nulls (A–F)** | **CPU** | `step2_headline_nulls.py` | Permuted-label null, universality ratio, peak-depth. |
| 3. Random-text null | **GPU** | `gpu/g5b_random_text_null_original_corpus.py` | Fresh forward passes over the recovered neutral corpus. Deps `gpu/common.py`, `gpu/forward_utils.py`. |
| 3b. Generator-confound (human-written) | **GPU** | `gpu/g7/extract_g7.py` (+ built datasets in `gpu/g7/`) | Extracts SST-5 / Gutenberg / Wikipedia contrastive sets; the §4.5 generator flank. |
| 4. Figures (S4–S11) | CPU | `figures/fig_*.py` | Read corrected DOM from HF. Needs `rosetta_tools` (viz_style). |

Stages 1–2 here are the CPU-reproducible core. Stage 3 and the proportional-
depth / handoff extensions to cluster F need GPU or F's per-layer trajectories
(see the shared GPU inventory `GPU_WORK_OUTSTANDING.md`, items P4a/P4c).

**Canonical home.** As of 2026-07-18 these are the *canonical* copies of the
P4 regeneration scripts; the copies under the (non-publishing) papers repo
`Rosetta_Program/papers/…` are mirrors. Edit here.

## Run

```bash
cd alignment/p4_regen
export P4_REGEN_STAGE=/path/with/room   # scratch for cluster-F all-layer streaming (needs ~1GB free)
python3 step1_primary_alignment.py                 # validates, then writes the A–F primary CSV
python3 step2_headline_nulls.py                    # permuted null, universality, peak-depth
# or the driver:
bash run_p4_regen.sh
```

Outputs land in `./p4_regen_output/` (`prh_primary_xfam_samedim_C17.csv`,
`step1_summary.json`, `step2_nulls.json`).

## Expected values (2026-07-18, A–F, C=17, N=250)

| Quantity | Value |
|---|---|
| Grand mean aligned cosine | **0.9752** ± 0.0577 |
| Grand median | 0.9938 |
| Pairs | **1,763** (104 ordered × 17 − 5 unavailable) |
| Models | 33 (A–F) |
| Cluster F mean | 0.9785 |
| Exfiltration | 0.9869 (rank 7 of 17) |
| Weakest concept | deception 0.905 |
| Permuted-label null | pooled −0.0009 ± 0.199, n=44,050, **z ≈ 1030** vs primary |
| Universality ratio | **0.205** |
| Peak-depth Δ (near-null) | +0.008 |

`step1` prints a `check ...` line per headline quantity and flags any that
differ from these.

## Precision / provenance notes

- **Reduced Procrustes is exact, not approximate.** The DOM vectors lie in the
  ≤n-dimensional row-space of the mean-centred calibration matrix, so the
  null-space part of the orthogonal rotation never touches them. `step1
  --validate-only` proves reduced == full on a cluster-F (8192-dim) pair to
  <1e-8 before doing anything else, then reproduces a control concept against
  the published CSV.
- **Cluster F 8-bit caveat.** Llama-3.1-70B and Qwen2.5-72B calibration is
  8-bit-derived (bf16 OOM at 70–72B); falcon-40b is bf16. Folded into the
  primary with disclosure (paper §2.1/§3.1/§4.5). A bf16 re-extraction is the
  optional cleanup item P4d.
- **Gemma-2 is retained**, not excluded: the companion CAZ Validation paper's
  §6.9 four-readout dissociation certifies that *calibration-cloud Procrustes*
  (this paper's method) is stable on Gemma-2 (0.96–0.99 across draws) even
  though its single-axis DOM point estimate is not.

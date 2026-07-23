# P4 data regeneration — PRH / Concept-Selective Convergence (A–E controlled primary + Cluster-F extension, corrected)

Self-contained, self-validating scripts that regenerate **every headline P4
quantity** from the published HuggingFace activation artifacts. Written
2026-07-18 to capture the corpus as it stands after two substantive changes
this cycle:

1. **Exfiltration label correction.** ~72% of exfiltration's positive/negative
   calibration labels were inverted; corrected via a recorded-draw
   reconstruction (N=249). Exfiltration went 0.916 (rank 16) → **0.9869**
   (rank 7); grand mean 0.9709 → 0.9750.
2. **Cluster F reported as a frontier extension** (round-4 restructure,
   superseding the earlier A–F-fold framing). falcon-40b, Llama-3.1-70B,
   Qwen2.5-72B (8192-dim) are a compute-limited frontier *extension* to the
   controlled **A–E primary** (30 models, grand mean **0.9750**, 1,661 pairs),
   **not** folded into it — folding gives the A–F figure **0.9752** over 1,763
   pairs, identical to three decimals. The 70–72B pair's calibration is
   8-bit-derived (VRAM-bound) and the confound battery is computed on A–E only;
   both disclosed (paper §2.1/§3.1/§4.5).

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
| 5. GEM handoff (§3.7) | CPU | `regen_gem_handoff_null.py` (17-concept permuted-*correspondence* null) + `regen_gem_handoff_exfil.py` (exfil primary+null) | Corrected handoff null + exfil handoff primary (0.9022→0.9424, grand 0.9611→0.9635). Reduced-Procrustes, self-validating vs full SVD. Reads corrected caz/gem/calibration from HF. |
| 6. Proportional-depth per-concept (§3.8) | CPU | `regen_propdepth_perconcept.py [--concepts …]` | Per-concept depth-matched Δ, **full-dim** Procrustes on stored per-layer DOM. `--concepts exfiltration` → corrected +0.303; `--concepts moral_valence` → known-answer check (paper +0.305). |
| 7. Transfer matrix / Figure 1 (§3.3) | CPU | `m4_transfer_exfil_fulldim.py` → splice into HF matrix → `figures/fig_transfer_matrix_17x17.py` | Recomputes exfil row+column **full-dim** (diagonal 0.9141→0.9868), then regenerates Figure 1. `m4_transfer_matrix_recompute.py` is the fast reduced-method full-matrix variant — **not** the published (full-dim) method; kept for reference only. |
| 8. Corpus QA audit (§4.5) | CPU | `qa_corpus_audit.py` | Cross-model direction consistency + split-half DOM reproducibility over 33×17; the §4.5 "no hidden defect / only Gemma-2 flagged" audit. |
| 9. **Consistency check (G3 guardrail)** | CPU | `consistency_check.py` | Parses every load-bearing number out of `preprint.md` and diffs it against the HF artifacts (primary CSV, null JSONs, transfer matrix, handoff); flags missing or superseded values. Exit non-zero on any mismatch. **Run at every data change** so prose/table drift cannot survive a commit. |

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

## Expected values (round-4, A–E controlled primary, C=17, N=250)

| Quantity | Value |
|---|---|
| **Grand mean (A–E controlled)** | **0.9750** ± 0.058 (14/17 > 0.95) |
| Grand median | 0.9938 |
| **A–E pairs** | **1,661** (98 ordered × 17 − 5 unavailable) |
| A–E models | 30 (24 base + 6 instruct) |
| Frontier extension (Cluster F) | 0.9785, 102 pairs; **A–F combined 0.9752**, 1,763 pairs |
| Exfiltration | 0.9868 (rank 7 of 17) |
| Weakest concept | deception 0.900 |
| Permuted-label null (A–E) | −0.0010 ± 0.200, n=41,500; primary at **d ≈ 4.9 null-SDs / ~130 SE** (not z≈1030) |
| Universality ratio | **0.209** |
| Peak-depth Δ (near-null) | +0.0084 |
| GEM handoff grand / exfil | 0.9635 / 0.9424 |

`consistency_check.py` verifies every one of these against the HF artifacts
(currently 31/31 pass, 0 drift).

**Note — step1/step2 still emit the A–F superset.** `step1_primary_alignment.py`
and `step2_headline_nulls.py` currently compute the A–F headline (grand 0.9752,
permuted null n=44,050); the A–E controlled primary above is the `dim ≠ 8192`
subset. Reframing them to lead with A–E (F as the extension) is a tracked
cleanup-pass item; until then, treat `consistency_check.py` — which reads the
artifacts directly — as the source of truth for the paper's A–E numbers.

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

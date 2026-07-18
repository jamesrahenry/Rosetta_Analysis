# P3 (CAZ Validation) — data regeneration manifest

*Written: 2026-07-18 22:36 UTC by claude:p3-review. Consolidates every
paper-specific recompute/figure script for `caz-validation/preprint.md`
into one place, so no result in P3 is "a number from a run we can't
re-derive." These are the canonical copies; the paper repo
(`Rosetta_Program/papers/caz-validation/scripts/`) mirrors them.*

## Two layers of regeneration

1. **Upstream data pipeline** — per-model activation extraction and the
   ablation / global-sweep / patching / GEM / CAZ / CKA / deep-dive
   artifacts. These live in this repo's `extraction/`, `gem/`, `caz/`
   modules and write to `paper_n250/<slug>/`. They need GPU. The frozen
   outputs are on HF (`james-ra-henry/Rosetta-Activations`, tag
   `paper-n250`); you do not normally re-run these.
2. **Paper-specific recompute** (this directory) — pure post-analysis over
   those stored artifacts. All CPU. Every P3 number, table, and figure is
   produced here.

## Prerequisites

- `rosetta_tools` installed (provides `reporting.load_scored_region_df`,
  `caz.find_caz_regions_scored`, `extraction`, `paths`).
- `pandas scipy scikit-learn matplotlib huggingface_hub wordfreq` (+ `torch`,
  `transformers` for the two that re-extract: `gemma_*`/`distilgpt2_*`).
- HF access to `james-ra-henry/Rosetta-Activations`. Scripts stream the
  npys/JSONs they need and clean up; a few expect a local
  `~/rosetta_data/paper_n250/` mirror (populate per the paper's CLAUDE.md
  restore block).

## The number → script map

All values below are the **corrected (post-exfiltration-fix, 2026-07-18)**
paper values. Scripts prefixed `apply_exfil_` carry a defective-vs-corrected
validation harness: they reproduce the *published defective* value first
(gate) before emitting the corrected one — see "Validation pattern" below.

### §2.2 corpus quality (C12–C14)
| Paper claim | Script | Output |
|---|---|---|
| BoW baseline at ceiling (held-out AUC ≈ 0.999; uninformative for ordering) | `c12_bow_baseline.py` | `c12_bow_baseline_results.json` |
| Lexical overlap modest (Jaccard 0.154) | `c13_lexical_overlap.py` | `c13_lexical_overlap_results.json` |
| Topic-disjoint DOM stability 0.957 vs 0.960 | `c14_calibration_topic_sensitivity.py` | `c14_..._results.json` |

### §2.3 / §4.2 / §6.2 detection counts
| Paper claim | Script | Output |
|---|---|---|
| 1,045 regions; 363/476 (76.3%) multimodal; 2.20/pair; smoothing-invariant at k∈{adaptive,12,24,48}; §6.2 dependency sweep 40–80% (82.4/17.6 @60%, 0 backward) | `c6_sensitivity.py` | `c6_sensitivity_results.json` |

### §3.1 / §3.2 ordering (Table 3 + τ statistics)
| Paper claim | Script | Output |
|---|---|---|
| Per-model τ (median 0.404; 27/28 positive + gpt2-medium 0.000; Wilcoxon W-sum 378, p=1.49e-8); cohort medians 0.404/0.440/0.113; **Table 3** depths; §3.3 credibility 9/6/13 split | `ordering_tau_recompute.py` | `depth_pivot.csv`, `tau_per_model.csv` |
| LOO robustness (median 0.372; 27 positive, gpt2-medium −0.031) | `loo_ordering_check.py` / same in `ordering_tau_recompute.py` | stdout |
| Frequency confound ρ = −0.657 | `freq_confound_recompute.py` | stdout |
| Partial τ \| frequency = 0.380 (28/28 positive) | `c5_frequency_partial_tau.py` | `c5_frequency_partial_tau.json` |
| Reliability ceiling 0.911 (split-half τ 0.717, SB 0.835, grand-mean 0.993); within/between 0.651/0.254 | `c5_splithalf_tau_ceiling.py` | `c5_splithalf_tau_ceiling.json`, `c5_splithalf_depths.json` |

### §6.1 ablation enrichment (Table 11 + controls)
| Paper claim | Script | Output |
|---|---|---|
| 3.60× peak-vs-nonCAZ enrichment; C1 depth-control +0.301; C2 SNR-control +0.303; C4 family enrichment | `apply_exfil_c1c2_t11.py` | `apply_exfil_c1c2_t11.json` |
| Cluster/family bootstrap (4.31× / 4.13×, CIs) + within-model permutation | `flow_p3_clusterboot.py` | stdout/json |

### §6.4 / §6.5 causal structure
| Paper claim | Script | Output |
|---|---|---|
| 50.3% legibility-causality divergence (358 cells); C3 null 57.9%±2.5%, z=−3.02, p=0.0016; causal deeper 94.4% | `apply_exfil_c3.py` | `apply_exfil_c3.json` |
| Table 12 cohort ablation/patching (MHA/GQA/Gemma); §6.5 handoff 34/34 + 304/442 extension | `apply_exfil_t12_65.py` | stdout |

### §6.9 Gemma distributed-encoding case study
| Paper claim | Script | Output |
|---|---|---|
| Probe transfer (0.89–0.999), spectrum (PR 20–78), rank-1 erasure inert, trace-shape 0.94 (T1–T5) | `gemma_subtlety_test.py` | `gemma_subtlety_results.json` |
| Subspace-denoise (no fix) + corpus-wide score-vs-stability (F3, ρ=−0.25) | `gemma_followup.py` | stdout |
| Rank-k erasure (rank-20–40 restores kill) | `gemma_rank_k_ablation.py` | stdout |
| Convergence curves (median c=49, n₉₅≈19c, 11/17 clear in-pool) | `gemma_convergence_recompute.py` | `gemma_convergence_recompute.json` |
| One-shot archive of all §6.9-cited numbers | `gemma_section_artifacts.py` | `gemma_section_artifacts.json` |

### Companion research note (not P3 paper text)
| Claim | Script | Output |
|---|---|---|
| distillation-legibility prediction 1 (distilgpt2 vs gpt2 — null) | `distilgpt2_distillation_test.py` | `distilgpt2_distillation_test.json` |

### Figures
| Figure | Script | Output |
|---|---|---|
| Figure 1 — 28×17 peak-depth heatmap (self-contained, reads `depth_pivot.csv`) | `fig1_peak_depth_heatmap.py` | `figures/fig_peak_depth_heatmap.png` |

## Run order

1. `ordering_tau_recompute.py` first — it writes `depth_pivot.csv`, which
   `fig1_peak_depth_heatmap.py` and several checks consume.
2. Everything else is independent and CPU-parallelizable.
3. `c5_splithalf_tau_ceiling.py` is the heaviest (streams ~70GB of
   all-layer npys one at a time; ~1.5h). It checkpoints to
   `results/c5_splithalf_depths.json` — **delete a concept's rows there to
   force recompute** (a stale checkpoint silently serves old data; caught
   in the 2026-07-18 apply-pass).

## Validation pattern (apply_exfil_* scripts)

These were written during the exfiltration label-correction apply-pass.
Each loads the concept in two states — `defective` (frozen `paper-n250`
exfiltration) and `corrected` (the N=249 rerun) — and **reproduces the
published defective value before trusting the corrected one**. This is why
they need both the frozen-tag artifacts and the corrected rerun artifacts
(`paper_n250/_round3_gpu/exfiltration_rerun/`) staged. The full old→new
ledger is `caz-validation/APPLY_PASS_LEDGER.md`; provenance of the few
numbers that don't reproduce from artifacts is in
`caz-validation/PROVENANCE_RESOLUTIONS.md`.

## Not re-derivable from artifacts (documented, correction-invariant)

Per `PROVENANCE_RESOLUTIONS.md`: Table 11's exact pooled recipe
(0.517/0.140), §6.7 CKA (0.962/0.977), and §2.3's original velocity-peak
counts do not reproduce under any definition tried — but the effects are
robust and unchanged by the correction. Their producing scripts (if a
defensible original surfaces) would be in this repo's `gem/` and `caz/`
modules (`ablate_global_sweep.py`, `analyze_structure.py`).

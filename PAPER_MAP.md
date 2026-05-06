*Written: 2026-04-22 00:00 UTC*

# Paper Map — Script to Figure Mapping

Maps each published figure/table back to the exact script that produced it.
All results were produced with `N=200` pairs from the `v1` training split
of `Rosetta_Concept_Pairs`.

---

## Paper 1 — CAZ: Concept Allocation Zones

| Figure / Table | Script | Notes |
|---|---|---|
| Fig 1 — CAZ profile (credibility) | `caz/deep_dive.py` | Single model deep dive |
| Fig 2 — Multi-concept overlay | `viz/viz_caz_anatomy.py` | Uses caz_deep_dive outputs |
| Fig 3 — Peak heatmap (all models × concepts) | `viz/viz_peak_depth_paper.py` | |
| Fig 4 — Gentle vs black-hole CAZ | `caz/analyze.py` | threshold scoring |
| Table 1 — Per-model CAZ summary | `caz/analyze.py` | |
| Supplementary — Ablation controls | `caz/caz_ablation_control.py` | |

---

## Paper 2 — GEM: Geometric Evolution Map

| Figure / Table | Script | Notes |
|---|---|---|
| Fig 1 — GEM node diagram | `gem/build_gems.py` | Conceptual; uses pythia-6.9b |
| Fig 2 — Handoff win rate (34 models) | `gem/aggregate_gem_results.py` | |
| Fig 3 — Zone-level ablation delta | `gem/ablate_gem.py` | |
| Fig 4 — Non-CAZ control | `gem/ablate_noncaz_control.py` | |
| Fig 5 — Random layer null | `gem/ablate_random_layer_null.py` | p=2.34e-09 |
| Fig 6 — Relay funnel | `gem/aggregate_behavioral_pilot.py` | |
| Table 1 — Per-model GEM diagnostics | `gem/build_gems.py` | `gem_diagnostics()` output |
| Supplementary — Random ablation null | `gem/aggregate_random_control.py` | direction-specificity control |

---

## Paper 3 — PRH: Platonic Representation Hypothesis

| Figure / Table | Script | Notes |
|---|---|---|
| Fig 1 — Procrustes alignment (pairwise) | `alignment/align.py` | |
| Fig 2 — Depth gradient | `alignment/align_trajectory.py` | |
| Fig 3 — Cross-concept transfer | `alignment/align.py` | 106% transfer ratio |
| Fig 4 — Primitive vocabulary (28 dirs) | `alignment/align.py` | post-Procrustes clustering |
| Table 1 — Per-pair alignment scores | `alignment/align.py` | |
| Supplementary — RLHF geometry preservation | `alignment/align.py` | base vs instruct |

---

## Extraction

All figures above consume results produced by:

```bash
python extraction/extract.py --prh-proxy          # all L4-runnable clusters
python extraction/extract.py --prh-cluster G      # Gemma 4 (needs --load-4bit)
python extraction/extract.py --prh-frontier       # H200 only
```

Results land in `~/rosetta_data/models/{model_slug}/`.

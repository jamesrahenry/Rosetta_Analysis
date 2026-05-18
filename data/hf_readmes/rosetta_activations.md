---
license: mit
language:
- en
tags:
- mechanistic-interpretability
- activations
- probes
- concept-vectors
- transformers
pretty_name: Rosetta Activations
size_categories:
- 10K<n<100K
---

# Rosetta Activations

*Updated: 2026-05-18 23:29 UTC*

Contrastive activation extractions for 17 semantic concepts across 33+ language models,
supporting cross-architecture mechanistic interpretability research. Each model directory
contains raw activation arrays (`.npy`) for probe training and Procrustes alignment,
alongside JSON analysis results (CAZ separation curves, GEM handoff layers, ablation
comparisons, and activation patching) — everything needed to reproduce paper results
without re-running inference.

Companion concept pair corpus: [jamesrahenry/Rosetta_Concept_Pairs](https://github.com/jamesrahenry/Rosetta_Concept_Pairs)

Papers: forthcoming

---

## Two Data Splits

### `paper_n250/` — Frozen paper-reproducibility data (N=250)

The canonical dataset for reproducing Papers 1–4. Frozen at N=250 pairs per concept
across 33 models. All paper numbers were computed from this split. Use this for
reproduce runs.

```bash
hf download james-ra-henry/Rosetta-Activations \
    --repo-type dataset \
    --local-dir ~/rosetta_data/ \
    --include "paper_n250/*"
rsync -a ~/rosetta_data/paper_n250/ ~/rosetta_data/models/
```

### `models/` (rcp_v1) — Full-resolution corpus (N=2000)

Larger extraction for future analysis and RCP v2 development. N=2000 pairs per concept,
same 17 concepts, expanded model set (~33+ models depending on availability).

**Probe-overfitting warning:** Peak-layer selection in the rcp_v1 `.npy` files was
determined by CAZ analysis run on the same 2000-pair corpus. If you train a linear probe
directly on these arrays without a held-out validation split, peak-layer selection and
probe weights are correlated to the same data — standard overfitting applies. Always
partition `calibration_{concept}.npy` into train/val before fitting probes. The fixed
250-pair train/val split from `paper_n250/` is the cleanest baseline for comparisons.

---

## Dataset Structure

```
Rosetta-Activations/
├── paper_n250/                    # Frozen N=250 paper data (recommended for reproducibility)
│   └── {Model_Name}/
│       └── (same file layout as models/ below)
│
├── models/                        # rcp_v1 — full-resolution N=2000 extractions
│   └── {Model_Name}/
│       ├── calibration_{concept}.npy           # Peak-layer activations
│       ├── calibration_alllayer_{concept}.npy  # All-layer activations
│       ├── calibration_{concept}_meta.json     # Full extraction provenance
│       ├── caz_{concept}.json                  # CAZ analysis (separation curves, peak layer)
│       ├── gem_{concept}.json                  # GEM analysis (handoff layers, EEC)
│       ├── ablation_gem_{concept}.json         # Handoff vs peak ablation results
│       ├── ablation_random_{concept}.json      # Random-direction ablation null
│       └── patch_{concept}.json               # Activation patching results
│
└── model_snapshots/               # Earlier versioned archive (pre-rcp_v1)
    └── {Model_Name}_{tag}/        # e.g. EleutherAI_pythia_6.9b_p1n100
        └── caz_{concept}.json
```

---

## Array Format

### `calibration_{concept}.npy`
Peak-layer contrastive activations.

| Property | Value |
|----------|-------|
| dtype | float32 |
| shape | `(2 * n_pairs, hidden_dim)` |
| layout | rows `0..n_pairs-1` = positive examples, rows `n_pairs..end` = negative examples |

```python
import numpy as np
# paper_n250 example (250 pairs)
acts = np.load("paper_n250/EleutherAI_pythia_6.9b/calibration_agency.npy")
# acts.shape → (500, 4096)   [250 pairs × 2, hidden_dim]
pos = acts[:250]   # positive (agentive) examples
neg = acts[250:]   # negative (non-agentive) examples

# rcp_v1 example (2000 pairs)
acts = np.load("models/EleutherAI_pythia_6.9b/calibration_agency.npy")
# acts.shape → (4000, 4096)   [2000 pairs × 2, hidden_dim]
```

### `calibration_alllayer_{concept}.npy`
All-layer activations for depth-matched analysis (P5 proportional-depth tests).

| Property | Value |
|----------|-------|
| dtype | float32 |
| shape | `(n_layers, 2 * n_pairs, hidden_dim)` |
| layout | axis 0 = layer index, axis 1 = samples (positive then negative), axis 2 = hidden_dim |

```python
acts = np.load("paper_n250/EleutherAI_pythia_6.9b/calibration_alllayer_agency.npy")
# acts.shape → (32, 500, 4096)   [n_layers, 2*n_pairs, hidden_dim]
layer_15 = acts[15]   # all samples at layer 15
```

### `calibration_{concept}_meta.json`
Full extraction provenance — model architecture, corpus version, pair IDs used,
extraction parameters, array shapes and layout.

---

## Concepts (17)

| Concept | Description |
|---------|-------------|
| `agency` | Agentive vs non-agentive actions |
| `authorization` | Authorized vs unauthorized actions |
| `causation` | Causal vs non-causal relations |
| `certainty` | Certain vs uncertain claims |
| `credibility` | Credible vs non-credible sources |
| `deception` | Deceptive vs honest statements |
| `exfiltration` | Data exfiltration vs benign transfer |
| `formality` | Formal vs informal register |
| `moral_valence` | Morally positive vs negative actions |
| `negation` | Negated vs affirmative statements |
| `plurality` | Plural vs singular reference |
| `sarcasm` | Sarcastic vs sincere statements |
| `sentiment` | Positive vs negative sentiment |
| `specificity` | Specific vs vague claims |
| `temporal_order` | Temporally ordered vs unordered events |
| `threat_severity` | High vs low threat severity |
| `urgency` | Urgent vs non-urgent requests |

---

## Models (paper_n250 — 33 models)

| Family | Models |
|--------|--------|
| Pythia (MHA) | 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B |
| GPT-2 (MHA) | base (124M), medium, large, xl |
| OPT (MHA) | 125M, 350M, 1.3B, 2.7B, 6.7B |
| Qwen2.5 (GQA) | 0.5B, 0.5B-Instruct, 1.5B, 1.5B-Instruct, 3B, 3B-Instruct, 7B, 7B-Instruct, 14B |
| Llama 3.1 (GQA) | 8B, 8B-Instruct |
| Llama 3.2 (GQA) | 1B, 3B |
| Mistral (GQA) | 7B-v0.3 |
| Gemma-2 (Alternating MHA/GQA) | 2B, 9B |
| Phi (Other) | Phi-2 |

The rcp_v1 `models/` tree covers the same core set plus additional instruct variants and
larger models (Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.3,
Gemma-2-2B-it, Gemma-2-9B-it, Qwen2.5-32B, GPT-Neo-125M, and others).

---

## Extraction Details

| | paper_n250 | rcp_v1 (models/) |
|--|--|--|
| Pairs per concept | 250 | 2000 |
| Split | Fixed train/val (Rosetta_Concept_Pairs v1) | Full corpus, no fixed split |
| Use for | Paper reproducibility | Future analysis, RCP v2 |
| Probe note | Clean baseline | See overfitting warning above |

- **Pooling**: last non-padding token (both splits)
- **Pair corpus**: [jamesrahenry/Rosetta_Concept_Pairs](https://github.com/jamesrahenry/Rosetta_Concept_Pairs)
- **Extraction code**: forthcoming

---

## License

MIT

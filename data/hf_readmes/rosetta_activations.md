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

*Updated: 2026-05-18 21:15 UTC*

Contrastive activation extractions for 17 semantic concepts across 46 language models,
supporting cross-architecture mechanistic interpretability research.

Companion concept pair corpus: [jamesrahenry/Rosetta_Concept_Pairs](https://github.com/jamesrahenry/Rosetta_Concept_Pairs)

Papers: forthcoming

---

## Dataset Structure

```
Rosetta-Activations/
├── paper_n250/           # Frozen N=250 paper-reproducibility data
│   └── {Model_Name}/
│       ├── calibration_{concept}.npy           # Peak-layer activations (N=250)
│       ├── calibration_alllayer_{concept}.npy  # All-layer activations (N=250)
│       ├── calibration_{concept}_meta.json     # Extraction provenance
│       ├── caz_{concept}.json                  # CAZ analysis
│       ├── gem_{concept}.json                  # GEM analysis
│       ├── ablation_gem_{concept}.json         # Ablation results
│       ├── ablation_random_{concept}.json      # Random-direction null
│       └── patch_{concept}.json               # Activation patching
│
├── rcp_v1/               # Raw N=2000 activations (38 of 46 models; see note)
│   └── {Model_Name}/
│       ├── calibration_{concept}.npy           # Peak-layer activations (N=2000)
│       ├── calibration_alllayer_{concept}.npy  # All-layer activations (N=2000)
│       └── calibration_{concept}_meta.json     # Extraction provenance
│
├── models/               # N=2000 analysis results (46 models)
│   └── {Model_Name}/
│       ├── caz_{concept}.json                  # CAZ analysis (separation curves, peak layer)
│       ├── gem_{concept}.json                  # GEM analysis (handoff layers, EEC)
│       ├── ablation_gem_{concept}.json         # Handoff vs peak ablation
│       ├── ablation_random_{concept}.json      # Random-direction ablation null
│       └── patch_{concept}.json               # Activation patching
│
└── model_snapshots/      # Earlier versioned archive (pre-rcp_v1)
    └── {Model_Name}_{tag}/
        └── caz_{concept}.json
```

### Coverage note

`rcp_v1/` contains raw activation arrays for **38 of 46 models**. The following 8 models
have analysis results in `models/` but raw `.npy` files pending re-extraction:

| Model | Size | Status |
|-------|------|--------|
| `facebook_opt_350m` | 350M | pending |
| `openai_community_gpt2_medium` | 345M | pending |
| `Qwen_Qwen2.5_32B` | 32B | hardware-blocked |
| `google_gemma_4_26B_A4B` | 26B | hardware-blocked |
| `google_gemma_4_26B_A4B_it` | 26B | hardware-blocked |
| `meta_llama_Llama_3.1_70B` | 70B | hardware-blocked |
| `tiiuae_falcon_40b` | 40B | hardware-blocked |
| `Qwen_Qwen2.5_72B` | 72B | hardware-blocked |

For paper reproducibility, use `paper_n250/` — it is complete for all 46 models.

---

## Quick Start

### Reproduce paper results (N=250)

```bash
pip install huggingface_hub
hf download james-ra-henry/Rosetta-Activations \
    --repo-type dataset \
    --local-dir ~/rosetta_data/ \
    --include "paper_n250/*"
rsync -a ~/rosetta_data/paper_n250/ ~/rosetta_data/models/
```

### Download N=2000 raw activations

```bash
hf download james-ra-henry/Rosetta-Activations \
    --repo-type dataset \
    --local-dir ~/rosetta_data/ \
    --include "rcp_v1/*"
```

### Download N=2000 analysis results

```bash
hf download james-ra-henry/Rosetta-Activations \
    --repo-type dataset \
    --local-dir ~/rosetta_data/ \
    --include "models/*"
```

---

## Array Format

### `calibration_{concept}.npy` — Peak-layer activations

| Property | Value |
|----------|-------|
| dtype | float32 |
| shape | `(2 * n_pairs, hidden_dim)` |
| layout | rows `0..n_pairs-1` = positive examples, rows `n_pairs..end` = negative examples |

```python
import numpy as np
# paper_n250 (250 pairs)
acts = np.load("paper_n250/EleutherAI_pythia_6.9b/calibration_agency.npy")
# acts.shape → (500, 4096)
pos = acts[:250]   # agentive
neg = acts[250:]   # non-agentive

# rcp_v1 (2000 pairs)
acts = np.load("rcp_v1/EleutherAI_pythia_6.9b/calibration_agency.npy")
# acts.shape → (4000, 4096)
```

### `calibration_alllayer_{concept}.npy` — All-layer activations

| Property | Value |
|----------|-------|
| dtype | float32 |
| shape | `(n_layers, 2 * n_pairs, hidden_dim)` |
| layout | axis 0 = layer index, axis 1 = samples (positive then negative), axis 2 = hidden_dim |

```python
acts = np.load("paper_n250/EleutherAI_pythia_6.9b/calibration_alllayer_agency.npy")
# acts.shape → (32, 500, 4096)
layer_15 = acts[15]
```

**Probe-overfitting warning (rcp_v1):** Peak-layer selection in rcp_v1 was determined
by CAZ analysis on the same 2000-pair corpus. Always partition into train/val before
fitting probes. The paper_n250 fixed split is the cleanest baseline.

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

## Models (46 total)

| Family | Models | paper_n250 | rcp_v1 .npy | models/ JSON |
|--------|--------|:---:|:---:|:---:|
| Pythia (MHA) | 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B | ✓ | ✓ | ✓ |
| GPT-2 (MHA) | base, medium, large, xl | ✓ | medium pending | ✓ |
| GPT-Neo (MHA) | 125M | ✓ | ✓ | ✓ |
| OPT (MHA) | 125M, 350M, 1.3B, 2.7B, 6.7B | ✓ | 350M pending | ✓ |
| Qwen2.5 (GQA) | 0.5B/Instruct, 1.5B/Instruct, 3B/Instruct, 7B/Instruct, 14B, 32B, 72B | ✓ | 32B/72B HW-blocked | ✓ |
| Llama 3.1 (GQA) | 8B, 8B-Instruct, 70B | ✓ | 70B HW-blocked | ✓ |
| Llama 3.2 (GQA) | 1B, 1B-Instruct, 3B, 3B-Instruct | ✓ | ✓ | ✓ |
| Mistral (GQA) | 7B-v0.3, 7B-Instruct-v0.3 | ✓ | ✓ | ✓ |
| Gemma-2 (Alt MHA/GQA) | 2B, 2B-it, 9B, 9B-it | ✓ | ✓ | ✓ |
| Gemma-4 (MoE) | 26B-A4B, 26B-A4B-it | — | HW-blocked | ✓ |
| Phi (Other) | Phi-2 | ✓ | ✓ | ✓ |
| Falcon (Other) | 40B | — | HW-blocked | ✓ |

---

## Extraction Details

| | paper_n250 | rcp_v1 |
|--|--|--|
| Pairs per concept | 250 | 2000 |
| Split | Fixed train/val (Rosetta_Concept_Pairs v1) | Full corpus, no fixed split |
| Use for | Paper reproducibility | Future analysis, RCP v2 |
| Content | `.npy` + all JSON | `.npy` + `_meta.json` only |

- **Pooling**: last non-padding token (both splits)
- **Pair corpus**: [jamesrahenry/Rosetta_Concept_Pairs](https://github.com/jamesrahenry/Rosetta_Concept_Pairs)
- **Extraction code**: forthcoming

---

## License

MIT

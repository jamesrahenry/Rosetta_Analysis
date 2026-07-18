"""
Calibration efficiency figure — Paper 4 supplementary (Henry 2026d).

2×2 grid: per-concept alignment scatter at full Procrustes fit for four
cross-family model pairs. Each panel: x = raw cosine (before rotation),
y = aligned cosine (after one Procrustes R per pair). All 17 concepts
start near zero raw; per-pair mean aligned cosine lands 0.90–0.98 under a
single shared rotation, though individual concepts range more widely
(weakest outliers as low as ~0.58 — see per-pair spread, not just the mean).

Pairs (small → large, diverse architectures):
  GPT-2 × Pythia-160m          (768-dim,  OpenAI vs EleutherAI)
  Pythia-2.8B × OPT-2.7B      (2560-dim, EleutherAI vs Meta)
  Qwen2.5-7B × Gemma-2-9b     (3584-dim, Qwen vs Google)
  Llama-3.1-8B × Mistral-7B   (4096-dim, Meta vs Mistral)

Saves: papers/prh-validation/figures/fig_calibration_efficiency.png
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from huggingface_hub import hf_hub_download

for _rtt in [Path.home() / "rosetta_tools",
             Path.home() / "Source" / "Rosetta_Program" / "rosetta_tools"]:
    if _rtt.exists():
        sys.path.insert(0, str(_rtt.parent))
        break
from rosetta_tools.viz_style import THEME, apply_theme, concept_color

HF_REPO      = "james-ra-henry/Rosetta-Activations"
HF_DATA_ROOT = "paper_n250"

ALL_17_CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

PAIRS = [
    {
        "model_a": "openai-community/gpt2",
        "model_b": "EleutherAI/pythia-160m",
        "label":   "GPT-2 × Pythia-160m (768-dim)",
        "size":    "OpenAI vs EleutherAI",
    },
    {
        "model_a": "EleutherAI/pythia-2.8b",
        "model_b": "facebook/opt-2.7b",
        "label":   "Pythia-2.8B × OPT-2.7B (2560-dim)",
        "size":    "EleutherAI vs Meta",
    },
    {
        "model_a": "Qwen/Qwen2.5-7B",
        "model_b": "google/gemma-2-9b",
        "label":   "Qwen2.5-7B × Gemma-2-9B (3584-dim)",
        "size":    "Qwen vs Google",
    },
    {
        "model_a": "meta-llama/Llama-3.1-8B",
        "model_b": "mistralai/Mistral-7B-v0.3",
        "label":   "Llama-3.1-8B × Mistral-7B (4096-dim)",
        "size":    "Meta vs Mistral",
    },
]


def model_key(model_id: str) -> str:
    return model_id.replace("/", "_").replace("-", "_")


def load_directions(model_id: str, concepts: list) -> dict:
    key = model_key(model_id)
    dirs = {}
    for concept in concepts:
        path = hf_hub_download(
            HF_REPO,
            filename=f"{HF_DATA_ROOT}/{key}/caz_{concept}.json",
            repo_type="dataset",
        )
        with open(path) as f:
            caz = json.load(f)
        ld   = caz["layer_data"]
        peak = ld["peak_layer"]
        dirs[concept] = np.array(ld["metrics"][peak]["dom_vector"], dtype=np.float64)
    return dirs


def cosine(a, b):
    an = np.linalg.norm(a); bn = np.linalg.norm(b)
    return float(np.dot(a, b) / (an * bn)) if an > 1e-12 and bn > 1e-12 else 0.0


# ── Build figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.patch.set_facecolor("white")
axes_flat = axes.flatten()

GREEN = "#2E7D32"

all_raw = []
all_aln = []

for ax, spec in zip(axes_flat, PAIRS):
    print(f"Loading {spec['label']} ...")
    dirs_a = load_directions(spec["model_a"], ALL_17_CONCEPTS)
    dirs_b = load_directions(spec["model_b"], ALL_17_CONCEPTS)

    mat_a = np.vstack([dirs_a[c] for c in ALL_17_CONCEPTS])
    mat_b = np.vstack([dirs_b[c] for c in ALL_17_CONCEPTS])
    R, _  = orthogonal_procrustes(mat_b, mat_a)

    raw_vals = [cosine(dirs_a[c], dirs_b[c])      for c in ALL_17_CONCEPTS]
    aln_vals = [cosine(dirs_a[c], dirs_b[c] @ R)  for c in ALL_17_CONCEPTS]
    mean_raw = float(np.mean(raw_vals))
    mean_aln = float(np.mean(aln_vals))

    all_raw.extend(raw_vals)
    all_aln.extend(aln_vals)

    print(f"  raw={mean_raw:.4f}  aligned={mean_aln:.4f}")

    for i, concept in enumerate(ALL_17_CONCEPTS):
        col = concept_color(concept)
        ax.scatter(raw_vals[i], aln_vals[i], color=col, s=110, zorder=4,
                   edgecolors="white", linewidths=0.9)
        ax.annotate(
            concept.replace("_", " "),
            (raw_vals[i], aln_vals[i]),
            textcoords="offset points", xytext=(5, 2),
            fontsize=7, color=col, zorder=5,
        )

    ax.axhline(mean_aln, color=GREEN, lw=0.9, ls="--", alpha=0.5)
    ax.axhline(0, color=THEME["spine"], lw=0.4, alpha=0.22)
    ax.axvline(0, color=THEME["spine"], lw=0.4, alpha=0.22)

    ax.annotate(
        f"mean aligned = {mean_aln:.3f}",
        xy=(0.5, 0.03), xycoords="axes fraction",
        ha="center", fontsize=9, color=GREEN, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                  edgecolor=GREEN, alpha=0.88, lw=0.7),
    )

    ax.set_title(f"{spec['label']}\n{spec['size']}", fontsize=10,
                 fontweight="bold", pad=8)
    ax.set_xlabel("Raw cosine  (before rotation)", fontsize=9)
    ax.set_ylabel("Aligned cosine  (after Procrustes)", fontsize=9)
    apply_theme(ax)

# shared axis limits derived from data
raw_margin = max(abs(v) for v in all_raw) * 1.7 + 0.01
aln_lo = min(all_aln) - 0.06
aln_hi = max(all_aln) + 0.06
for ax in axes_flat:
    ax.set_xlim(-raw_margin, raw_margin)
    ax.set_ylim(aln_lo, aln_hi)

fig.suptitle(
    "Per-concept alignment at full Procrustes fit — four cross-family pairs\n"
    "Raw cosine ≈ 0 for all 17 concepts; per-pair mean aligned cosine 0.90–0.98 "
    "(individual concepts vary more widely — see per-pair spread)",
    fontsize=12, fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.94])

out_dir = Path(__file__).parent.parent / "figures"
out_dir.mkdir(exist_ok=True)
out = out_dir / "fig_calibration_efficiency.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.close()

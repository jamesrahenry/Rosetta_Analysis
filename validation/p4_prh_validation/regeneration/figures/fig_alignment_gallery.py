"""
Alignment gallery figure — Paper 4 supplementary (Henry 2026d).

Shows Procrustes-aligned concept directions for 4 cross-family pairs projected
into each model pair's 2-D PCA subspace. After one rotation per pair, all 17
concept direction pairs coincide. Covers small/medium/large scales and multiple
attention mechanisms.

Saves: papers/prh-validation/figures/fig_alignment_gallery.png
       papers/prh-validation/figures/fig_alignment_before_after.png  (Qwen×Gemma only)
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from huggingface_hub import hf_hub_download

for _rtt in [Path.home() / "rosetta_tools",
             Path.home() / "Source" / "Rosetta_Program" / "rosetta_tools"]:
    if _rtt.exists():
        sys.path.insert(0, str(_rtt.parent))
        break
from rosetta_tools.viz_style import concept_color, THEME, apply_theme

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
        "fam":     "OpenAI vs EleutherAI",
    },
    {
        "model_a": "EleutherAI/pythia-2.8b",
        "model_b": "facebook/opt-2.7b",
        "label":   "Pythia-2.8B × OPT-2.7B (2560-dim)",
        "fam":     "EleutherAI vs Meta",
    },
    {
        "model_a": "Qwen/Qwen2.5-7B",
        "model_b": "google/gemma-2-9b",
        "label":   "Qwen2.5-7B × Gemma-2-9B (3584-dim)",
        "fam":     "Qwen vs Google",
    },
    {
        "model_a": "meta-llama/Llama-3.1-8B",
        "model_b": "mistralai/Mistral-7B-v0.3",
        "label":   "Llama-3.1-8B × Mistral-7B (4096-dim)",
        "fam":     "Meta vs Mistral",
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


# ── Figure 1: Gallery — all 4 pairs after alignment ───────────────────────────
print("Building gallery figure ...")
fig, axes = plt.subplots(1, 4, figsize=(22, 6))
fig.patch.set_facecolor("white")

for ax, spec in zip(axes, PAIRS):
    print(f"  Loading {spec['label']} ...")
    dirs_a = load_directions(spec["model_a"], ALL_17_CONCEPTS)
    dirs_b = load_directions(spec["model_b"], ALL_17_CONCEPTS)

    mat_a = np.vstack([dirs_a[c] for c in ALL_17_CONCEPTS])
    mat_b = np.vstack([dirs_b[c] for c in ALL_17_CONCEPTS])
    R, _  = orthogonal_procrustes(mat_b, mat_a)   # R: b → a frame

    _, S, Vt = np.linalg.svd(mat_a, full_matrices=False)
    comps = Vt[:2]
    evr   = (S[:2] ** 2) / (S ** 2).sum()

    a2d = mat_a @ comps.T
    b2d = (mat_b @ R) @ comps.T

    for i, concept in enumerate(ALL_17_CONCEPTS):
        col = concept_color(concept)
        ax.plot([a2d[i, 0], b2d[i, 0]], [a2d[i, 1], b2d[i, 1]],
                color=col, lw=1.2, ls="--", alpha=0.4, zorder=2)
        ax.scatter(*a2d[i], color=col, s=160, zorder=5,
                   marker="o", edgecolors="white", linewidths=1.2)
        ax.scatter(*b2d[i], facecolors="none", edgecolors=col, s=260,
                   zorder=4, marker="o", linewidths=2.2)

    ax.axhline(0, color=THEME["spine"], lw=0.4, alpha=0.25)
    ax.axvline(0, color=THEME["spine"], lw=0.4, alpha=0.25)
    apply_theme(ax)

    ax.set_title(
        f"{spec['label']}\n{spec['fam']}",
        fontsize=9, fontweight="bold", pad=8,
    )
    ax.set_xlabel(f"PC1  ({evr[0]:.0%} of model A variance)", fontsize=8)
    ax.set_ylabel(f"PC2  ({evr[1]:.0%} of model A variance)", fontsize=8)

    raw_cos     = float(np.mean([cosine(dirs_a[c], dirs_b[c])
                                 for c in ALL_17_CONCEPTS]))
    matched_cos = float(np.mean([cosine(dirs_a[c], dirs_b[c] @ R)
                                 for c in ALL_17_CONCEPTS]))
    ax.annotate(
        f"raw → aligned:  {raw_cos:.3f} → {matched_cos:.3f}",
        xy=(0.5, 0.025), xycoords="axes fraction",
        ha="center", va="bottom", fontsize=8.5, color=THEME["text"],
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                  edgecolor=THEME["spine"], alpha=0.88, linewidth=0.7),
    )
    print(f"    raw={raw_cos:.4f}  aligned={matched_cos:.4f}")

legend_patches = [
    mpatches.Patch(color=concept_color(c), label=c.replace("_", " "))
    for c in ALL_17_CONCEPTS
]
fig.legend(
    handles=legend_patches,
    loc="lower center", ncol=9, fontsize=7.5,
    bbox_to_anchor=(0.5, -0.10), frameon=False,
    title="concepts  ·  ● = model A  ○ = model B  (same-concept pairs connected by dashed lines)",
    title_fontsize=8,
)
fig.suptitle(
    "PRH alignment gallery — all 17 concepts across 4 cross-family pairs\n"
    "After 1 Procrustes rotation per model pair, same-concept directions coincide",
    fontsize=11, fontweight="bold",
)
plt.tight_layout(rect=[0, 0.12, 1, 0.93])

out_dir = Path(__file__).parent.parent / "figures"
out_dir.mkdir(exist_ok=True)
out1 = out_dir / "fig_alignment_gallery.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out1}")
plt.close()

# ── Figure 2: Before/after for the primary pair (Qwen × Gemma) ────────────────
print("Building before/after figure for Qwen × Gemma ...")
spec0 = PAIRS[0]
dirs_a = load_directions(spec0["model_a"], ALL_17_CONCEPTS)
dirs_b = load_directions(spec0["model_b"], ALL_17_CONCEPTS)

mat_a  = np.vstack([dirs_a[c] for c in ALL_17_CONCEPTS])
mat_b  = np.vstack([dirs_b[c] for c in ALL_17_CONCEPTS])
R_full, _ = orthogonal_procrustes(mat_b, mat_a)   # R: b → a frame

_, S, Vt = np.linalg.svd(mat_a, full_matrices=False)
comps = Vt[:2]
evr   = (S[:2] ** 2) / (S ** 2).sum()

q2d = mat_a @ comps.T
g2d_raw     = mat_b @ comps.T
g2d_aligned = (mat_b @ R_full) @ comps.T

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
fig.patch.set_facecolor("white")

for ax, panel_title, g_pts in [
    (axes[0], "Before alignment  (raw coordinate frames)", g2d_raw),
    (axes[1], "After Procrustes rotation  (1 rotation, all 17 concepts)", g2d_aligned),
]:
    for i, concept in enumerate(ALL_17_CONCEPTS):
        col = concept_color(concept)
        qx, qy = q2d[i]
        gx, gy = g_pts[i]
        ax.plot([qx, gx], [qy, gy], color=col, lw=1.2, ls="--", alpha=0.45, zorder=2)
        ax.scatter(qx, qy, color=col, s=180, zorder=5,
                   marker="o", edgecolors="white", linewidths=1.5)
        ax.scatter(gx, gy, facecolors="none", edgecolors=col, s=300,
                   zorder=4, marker="o", linewidths=2.5)
        ax.annotate(
            concept.replace("_", " "),
            (qx, qy), textcoords="offset points", xytext=(8, 3),
            fontsize=7.5, color=col, zorder=6,
        )
    ax.set_title(panel_title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(f"PC1  ({evr[0]:.0%} of Qwen variance)", fontsize=10)
    ax.set_ylabel(f"PC2  ({evr[1]:.0%} of Qwen variance)", fontsize=10)
    ax.axhline(0, color=THEME["spine"], lw=0.5, alpha=0.3)
    ax.axvline(0, color=THEME["spine"], lw=0.5, alpha=0.3)
    apply_theme(ax)

axes[1].annotate(
    "1 rotation per model pair\n(same R maps all concepts)",
    xy=(0.97, 0.04), xycoords="axes fraction",
    ha="right", va="bottom", fontsize=9, color=THEME["text"],
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor=THEME["spine"], alpha=0.85, linewidth=0.8),
)

leg_q = plt.scatter([], [], color="#555555", s=130, marker="o",
                    edgecolors="white", linewidths=1.5, label="Qwen2.5-7B  (●)")
leg_g = plt.scatter([], [], facecolors="none", edgecolors="#555555", s=200,
                    marker="o", linewidths=2.5, label="Gemma-2-9B  (○)")
fig.legend(handles=[leg_q, leg_g], loc="lower center", ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, -0.01), frameon=False)
fig.suptitle(
    "All 17 peak-layer concept directions projected into Qwen's 2-D subspace\n"
    "●  Qwen2.5-7B (reference)   ○  Gemma-2-9B   —  dashed lines connect same-concept pairs",
    fontsize=11, fontweight="bold",
)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

out2 = out_dir / "fig_alignment_before_after.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close()

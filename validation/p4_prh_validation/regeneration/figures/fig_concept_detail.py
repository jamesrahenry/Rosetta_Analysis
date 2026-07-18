"""
Per-concept detail figure — Qwen2.5-7B × Gemma-2-9B (Henry 2026d).

Three-panel figure showing the full per-concept picture for the primary pair:
  LEFT:   Raw cosine per concept (pre-rotation, all near zero)
  CENTRE: Aligned cosine per concept (post-Procrustes, one shared R)
  RIGHT:  Peak layer scatter — Qwen peak vs Gemma peak per concept

Concepts sorted by aligned cosine (descending) on the shared y-axis.

Saves: papers/prh-validation/figures/fig_concept_detail.png
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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

MODEL_A = "Qwen/Qwen2.5-7B"
MODEL_B = "google/gemma-2-9b"

ALL_17_CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]


def model_key(model_id):
    return model_id.replace("/", "_").replace("-", "_")


def load_directions_and_peaks(model_id, concepts):
    key = model_key(model_id)
    dirs = {}
    peaks = {}
    for concept in concepts:
        path = hf_hub_download(
            HF_REPO,
            filename=f"{HF_DATA_ROOT}/{key}/caz_{concept}.json",
            repo_type="dataset",
        )
        with open(path) as f:
            caz = json.load(f)
        ld = caz["layer_data"]
        peak = ld["peak_layer"]
        dirs[concept]  = np.array(ld["metrics"][peak]["dom_vector"], dtype=np.float64)
        peaks[concept] = peak
    return dirs, peaks


def cosine(a, b):
    an = np.linalg.norm(a); bn = np.linalg.norm(b)
    return float(np.dot(a, b) / (an * bn)) if an > 1e-12 and bn > 1e-12 else 0.0


# ── Load ───────────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_A} ...")
dirs_a, peaks_a = load_directions_and_peaks(MODEL_A, ALL_17_CONCEPTS)
print(f"Loading {MODEL_B} ...")
dirs_b, peaks_b = load_directions_and_peaks(MODEL_B, ALL_17_CONCEPTS)

mat_a = np.vstack([dirs_a[c] for c in ALL_17_CONCEPTS])
mat_b = np.vstack([dirs_b[c] for c in ALL_17_CONCEPTS])
R, _  = orthogonal_procrustes(mat_b, mat_a)

raw_vals = {c: cosine(dirs_a[c], dirs_b[c])      for c in ALL_17_CONCEPTS}
aln_vals = {c: cosine(dirs_a[c], dirs_b[c] @ R)  for c in ALL_17_CONCEPTS}
mean_raw = float(np.mean(list(raw_vals.values())))
mean_aln = float(np.mean(list(aln_vals.values())))

print(f"\nConcept                  Raw      Aligned   Peak A  Peak B")
print("─" * 62)
for c in sorted(ALL_17_CONCEPTS, key=lambda x: aln_vals[x], reverse=True):
    print(f"  {c:<22}  {raw_vals[c]:+.4f}   {aln_vals[c]:.4f}    "
          f"{peaks_a[c]:>3}     {peaks_b[c]:>3}")
print(f"\n  Mean raw={mean_raw:.4f}   Mean aligned={mean_aln:.4f}")

# sort concepts by aligned cosine descending
sorted_concepts = sorted(ALL_17_CONCEPTS, key=lambda c: aln_vals[c], reverse=True)
y_pos = np.arange(len(sorted_concepts))

# ── Figure ─────────────────────────────────────────────────────────────────────
OUTLIERS = {"deception", "exfiltration", "specificity"}  # weakest at true N=250 (§3.1): deception 0.900, exfiltration 0.916, specificity 0.935

fig, (ax_raw, ax_aln) = plt.subplots(1, 2, figsize=(13, 7),
                                      gridspec_kw={"width_ratios": [1, 1.5]})
fig.patch.set_facecolor("white")

GREEN = "#2E7D32"
RED   = "#B71C1C"

# ── LEFT: raw cosines (lollipop) ───────────────────────────────────────────────
for i, c in enumerate(sorted_concepts):
    col = concept_color(c)
    val = raw_vals[c]
    lw  = 2.2 if c in OUTLIERS else 1.5
    ax_raw.plot([0, val], [i, i], color=col, lw=lw, alpha=0.8)
    ax_raw.scatter(val, i, color=col, s=90 if c in OUTLIERS else 70,
                   zorder=4, edgecolors="white", linewidths=0.8)

ax_raw.axvline(0, color=THEME["spine"], lw=0.8, alpha=0.5)
ax_raw.set_yticks(y_pos)
ax_raw.set_yticklabels(
    [("★ " if c in OUTLIERS else "  ") + c.replace("_", " ")
     for c in sorted_concepts],
    fontsize=8.5,
)
# colour the outlier y-tick labels
for tick, c in zip(ax_raw.get_yticklabels(), sorted_concepts):
    if c in OUTLIERS:
        tick.set_color(RED)
        tick.set_fontweight("bold")

ax_raw.set_xlabel("Cosine similarity", fontsize=9)
ax_raw.set_title("Before rotation\n(raw cosine)", fontsize=10, fontweight="bold", pad=8)
ax_raw.set_xlim(-0.08, 0.08)
ax_raw.set_ylim(-0.8, len(sorted_concepts) - 0.2)
ax_raw.annotate(
    f"mean = {mean_raw:.4f}",
    xy=(0.5, 0.02), xycoords="axes fraction",
    ha="center", fontsize=8.5, color=THEME["text"],
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
              edgecolor=THEME["spine"], alpha=0.85, lw=0.7),
)
ax_raw.annotate(
    "Raw ≈ 0 for all concepts:\nthe rotation reveals similarity,\nit does not construct it.\nPermuted-label null under the\nidentical R returns 0.003 (§2.5).",
    xy=(0.5, 0.97), xycoords="axes fraction",
    ha="center", va="top", fontsize=7.2, color=THEME["text"],
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
              edgecolor=THEME["spine"], alpha=0.90, lw=0.7),
)
apply_theme(ax_raw)

# ── RIGHT: aligned cosines (bars) ─────────────────────────────────────────────
# shaded band marking the outlier zone
outlier_threshold = max(aln_vals[c] for c in OUTLIERS) + 0.02
ax_aln.axhspan(-0.8, outlier_threshold - 0.5, color=RED, alpha=0.04, zorder=0)

for i, c in enumerate(sorted_concepts):
    col  = concept_color(c)
    val  = aln_vals[c]
    alpha = 0.92 if c in OUTLIERS else 0.80
    ax_aln.barh(i, val, color=col, alpha=alpha, height=0.62, zorder=3)
    weight = "bold" if c in OUTLIERS else "normal"
    ax_aln.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=7.5,
                color=col, fontweight=weight)

ax_aln.axvline(mean_aln, color=GREEN, lw=1.0, ls="--", alpha=0.6)
ax_aln.axvline(0, color=THEME["spine"], lw=0.5, alpha=0.3)
ax_aln.set_yticks(y_pos)
ax_aln.set_yticklabels([])
ax_aln.set_xlabel("Cosine similarity", fontsize=9)
ax_aln.set_title("After Procrustes rotation\n(aligned cosine, one shared R)", fontsize=10,
                 fontweight="bold", pad=8)
ax_aln.set_xlim(0, 1.18)
ax_aln.set_ylim(-0.8, len(sorted_concepts) - 0.2)
ax_aln.annotate(
    f"mean = {mean_aln:.3f}",
    xy=(mean_aln + 0.01, 0.97), xycoords=("data", "axes fraction"),
    ha="left", va="top", fontsize=8.5, color=GREEN, fontweight="bold",
)

# outlier callout box
ax_aln.annotate(
    "★ Weakest concepts in the primary analysis (§3.1):\n"
    "  deception 0.900, exfiltration 0.916, specificity 0.935.\n"
    "  Alignment is bounded by calibration data quality:\n"
    "  richer calibration (500 examples, §3.1) lifts\n"
    "  authorization 0.898→0.984, threat severity 0.797→0.950,\n"
    "  specificity 0.838→0.935; deception does not recover.",
    xy=(0.02, 0.08), xycoords="axes fraction",
    fontsize=7.5, color=RED, va="bottom",
    bbox=dict(boxstyle="round,pad=0.38", facecolor="white",
              edgecolor=RED, alpha=0.92, lw=0.9),
)
apply_theme(ax_aln)

fig.suptitle(
    "Per-concept alignment detail — Qwen2.5-7B × Gemma-2-9B\n"
    "Raw cosines ≈ 0 for all 17 concepts; threat severity and specificity are weakest under this 17-vector fit — "
    "★ = weakest concepts in the primary analysis (§3.1)",
    fontsize=11, fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.93])

out_dir = Path(__file__).parent.parent / "figures"
out_dir.mkdir(exist_ok=True)
out = out_dir / "fig_concept_detail.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.close()

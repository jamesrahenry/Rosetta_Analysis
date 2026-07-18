"""
Full 17x17 cross-concept transfer matrix — Paper 4 (Henry 2026d), §3.3.

Replaces the original 7-concept exploratory matrix with the full C=17 result,
computed from the universality_depth_confound.py sweep (30 models, all
cross-family ordered pairs, N=28,322 rows total). Data already aggregated to
a 17x17 matrix in transfer_matrix_17x17.json (mean aligned cosine per
concept_A -> concept_B pair, fit R on concept_A, apply to concept_B).

Data: james-ra-henry/Rosetta-Activations, paper_n250/_universality_depth_confound/transfer_matrix_17x17.json
Saves: papers/prh-validation/figures/fig_transfer_matrix_17x17.png
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from huggingface_hub import hf_hub_download

DATA_PATH = Path(hf_hub_download(
    "james-ra-henry/Rosetta-Activations",
    "paper_n250/_universality_depth_confound/transfer_matrix_17x17.json",
    repo_type="dataset",
))

d = json.loads(DATA_PATH.read_text())
concepts = d["concepts"]
matrix = np.array(d["matrix"])
n = len(concepts)

# hierarchical clustering order on the off-diagonal transfer structure
dist = np.clip(1 - matrix, 0, None)
np.fill_diagonal(dist, 0)
dist = (dist + dist.T) / 2
Z = linkage(squareform(dist), method="average")
order = dendrogram(Z, no_plot=True)["leaves"]
mat_ord = matrix[np.ix_(order, order)]
labels_ord = [concepts[i].replace("_", " ") for i in order]

fig, ax = plt.subplots(figsize=(11, 10))
fig.patch.set_facecolor("white")

im = ax.imshow(mat_ord, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(n))
ax.set_xticklabels(labels_ord, fontsize=9.5, rotation=45, ha="right")
ax.set_yticks(range(n))
ax.set_yticklabels(labels_ord, fontsize=9.5)

# annotate the strongest off-diagonal cells
triu = np.triu_indices(n, k=1)
off_vals = mat_ord[triu]
top = sorted(zip(off_vals, triu[0], triu[1]), key=lambda x: -x[0])[:8]
for val, i, j in top:
    for ri, ci in [(i, j), (j, i)]:
        ax.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                fontsize=7, color="white" if val < 0.5 else "black",
                fontweight="bold")

cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("mean aligned cosine (fit R on concept A, apply to concept B)", fontsize=9)

ax.set_title(
    "Cross-concept rotation transfer — full C=17, 30 models, 28,322 comparisons\n"
    "Diagonal = same-concept alignment; off-diagonal = transfer between concepts. Not depth-stratified (see §3.3).",
    fontsize=11.5, fontweight="bold", pad=12,
)
plt.tight_layout()

out_dir = Path(__file__).parent.parent / "figures"
out_dir.mkdir(exist_ok=True)
out = out_dir / "fig_transfer_matrix_17x17.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")

# print top pairs for the write-up
print("\nTop 8 off-diagonal (transfer) pairs:")
for val, i, j in top:
    print(f"  {concepts[order[i]]:16s} <-> {concepts[order[j]]:16s}  {val:.4f}")

diag_vals = [matrix[k, k] for k in range(n)]
isolated = sorted(zip(concepts, [matrix[k].max() - matrix[k,k] if False else
                                   max(matrix[k, jj] for jj in range(n) if jj != k)
                                   for k in range(n)]), key=lambda x: x[1])
print("\nMost geometrically isolated (lowest max off-diagonal transfer):")
for c, maxoff in isolated[:5]:
    print(f"  {c:16s} max_transfer={maxoff:.4f}")

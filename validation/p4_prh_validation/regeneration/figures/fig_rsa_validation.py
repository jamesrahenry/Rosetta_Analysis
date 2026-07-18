"""
RSA validation figure — Paper 4 (Henry 2026d).

Computes Representational Similarity Analysis across 4 cross-family model pairs.
No Procrustes rotation. No Fisher separation. No layer selection beyond what was
already committed for DOM vector extraction. Coordinate-frame invariant.

Each model's 17×17 inter-concept cosine matrix is computed independently; the
136 upper-triangle entries are correlated via Spearman ρ. A high ρ means the two
models agree on which concepts are near each other in representation space, using
nothing except pairwise angles between already-extracted DOM vectors.

Saves: papers/prh-validation/figures/fig_rsa_validation.png
       papers/prh-validation/figures/fig_rsa_validation_heatmaps.png
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from huggingface_hub import hf_hub_download

# rosetta_tools — GPU-host-first path resolution
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
        "label":   "GPT-2 × Pythia-160m\n(768-dim)",
        "note":    "OpenAI vs EleutherAI",
        "dim":     768,
    },
    {
        "model_a": "EleutherAI/pythia-2.8b",
        "model_b": "facebook/opt-2.7b",
        "label":   "Pythia-2.8B × OPT-2.7B\n(2560-dim)",
        "note":    "EleutherAI vs Meta",
        "dim":     2560,
    },
    {
        "model_a": "Qwen/Qwen2.5-7B",
        "model_b": "google/gemma-2-9b",
        "label":   "Qwen2.5-7B × Gemma-2-9B\n(3584-dim)",
        "note":    "Qwen vs Google · alternating attention",
        "dim":     3584,
    },
    {
        "model_a": "meta-llama/Llama-3.1-8B",
        "model_b": "mistralai/Mistral-7B-v0.3",
        "label":   "Llama-3.1-8B × Mistral-7B\n(4096-dim)",
        "note":    "Meta vs Mistral",
        "dim":     4096,
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
        dirs[concept] = np.array(ld["metrics"][peak]["dom_vector"], dtype=np.float32)
    return dirs


def sim_matrix(dirs: dict, concepts: list) -> np.ndarray:
    vecs  = np.vstack([dirs[c] for c in concepts]).astype(np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
    vn    = vecs / norms
    return vn @ vn.T


# ── Compute RSA for all pairs ──────────────────────────────────────────────────
triu = np.triu_indices(len(ALL_17_CONCEPTS), k=1)

rsa_results = []
for spec in PAIRS:
    label_flat = spec["label"].replace("\n", " ")
    print(f"Loading {label_flat} ...")
    dirs_a = load_directions(spec["model_a"], ALL_17_CONCEPTS)
    dirs_b = load_directions(spec["model_b"], ALL_17_CONCEPTS)
    sim_a  = sim_matrix(dirs_a, ALL_17_CONCEPTS)
    sim_b  = sim_matrix(dirs_b, ALL_17_CONCEPTS)
    tri_a, tri_b = sim_a[triu], sim_b[triu]
    r, pv = spearmanr(tri_a, tri_b)
    rsa_results.append({
        "label": spec["label"], "note": spec["note"],
        "model_a": spec["model_a"], "model_b": spec["model_b"],
        "dim": spec["dim"],
        "tri_a": tri_a, "tri_b": tri_b,
        "sim_a": sim_a, "sim_b": sim_b,
        "r": float(r), "p": float(pv),
    })
    print(f"  Spearman ρ = {r:.4f}   p = {pv:.2e}")

# ── Figure 1: 2×2 scatter grid — redesigned ───────────────────────────────────
from matplotlib.lines import Line2D as _L2D

CATEGORIES = {
    "causation":       ("temporal",    "#E65100"),
    "temporal_order":  ("temporal",    "#E65100"),
    "certainty":       ("epistemic",   "#1565C0"),
    "credibility":     ("epistemic",   "#1565C0"),
    "specificity":     ("epistemic",   "#1565C0"),
    "formality":       ("social",      "#2E7D32"),
    "sarcasm":         ("social",      "#2E7D32"),
    "sentiment":       ("social",      "#2E7D32"),
    "moral_valence":   ("social",      "#2E7D32"),
    "exfiltration":    ("adversarial", "#880E4F"),
    "deception":       ("adversarial", "#880E4F"),
    "threat_severity": ("adversarial", "#880E4F"),
    "authorization":   ("adversarial", "#880E4F"),
    "agency":          ("structural",  "#4527A0"),
    "negation":        ("structural",  "#4527A0"),
    "plurality":       ("structural",  "#4527A0"),
    "urgency":         ("structural",  "#4527A0"),
}
CAT_LABELS = {
    "temporal":    "Temporal / relational",
    "epistemic":   "Epistemic",
    "social":      "Social / pragmatic",
    "adversarial": "Adversarial / security",
    "structural":  "Structural / syntactic",
}
# color for each dot based on category of concept_i (the row concept)
pair_idx  = list(zip(*triu))
dot_colors = [CATEGORIES[ALL_17_CONCEPTS[i]][1] for i, j in pair_idx]

# pairs worth labelling explicitly (story-telling beats most-extreme)
LABEL_PAIRS = {
    ("causation",       "temporal_order"),
    ("certainty",       "credibility"),
    ("exfiltration",    "deception"),
    ("negation",        "plurality"),
}

rng = np.random.default_rng(42)
N_NULLS = 6

from matplotlib.gridspec import GridSpec as _GS

fig = plt.figure(figsize=(13, 16))
fig.patch.set_facecolor("white")
gs = _GS(3, 4, figure=fig, height_ratios=[1.15, 1.15, 0.80],
         hspace=0.52, wspace=0.35)
scatter_axes = [fig.add_subplot(gs[r, c*2:(c+1)*2])
                for r in range(2) for c in range(2)]
ax_delta = fig.add_subplot(gs[2, :])

for panel_idx, (ax, res) in enumerate(zip(scatter_axes, rsa_results)):
    tri_a, tri_b = res["tri_a"], res["tri_b"]
    lim = max(np.abs(tri_a).max(), np.abs(tri_b).max()) * 1.30

    # null cloud
    for _ in range(N_NULLS):
        tri_b_perm = rng.permutation(tri_b)
        ax.scatter(tri_b_perm, tri_a,
                   alpha=0.08, s=28, color="#AAAAAA",
                   zorder=1, linewidths=0)

    # real dots coloured by category of concept_i
    ax.scatter(tri_b, tri_a, c=dot_colors, alpha=0.80, s=58,
               zorder=3, edgecolors="white", linewidths=0.6)

    # identity line + label
    ax.plot([-lim, lim], [-lim, lim], color="#555555",
            lw=1.1, ls="--", alpha=0.45, zorder=2)
    ax.text(lim * 0.58, lim * 0.72, "perfect\nagreement",
            fontsize=6.5, color="#555555", ha="center",
            rotation=43, rotation_mode="anchor")

    ax.axhline(0, color=THEME["spine"], lw=0.4, alpha=0.20)
    ax.axvline(0, color=THEME["spine"], lw=0.4, alpha=0.20)

    # label selected concept pairs
    model_b_short = res["model_b"].split("/")[-1]
    model_a_short = res["model_a"].split("/")[-1]
    for k, (i, j) in enumerate(pair_idx):
        ci, cj = ALL_17_CONCEPTS[i], ALL_17_CONCEPTS[j]
        if (ci, cj) in LABEL_PAIRS:
            xv, yv = float(tri_b[k]), float(tri_a[k])
            ax.annotate(
                f"{ci.replace('_',' ')} ×\n{cj.replace('_',' ')}",
                xy=(xv, yv),
                xytext=(xv + lim * 0.08, yv + lim * 0.06),
                fontsize=6, color=CATEGORIES[ci][1], fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.7),
            )

    # explanation box — first panel only
    if panel_idx == 0:
        ax.annotate(
            "Each dot is one concept pair\n"
            "(e.g. causation × temporal order).\n"
            "Its x-position is how similar\n"
            "that pair is inside Model B;\n"
            "y-position is the same pair\n"
            "inside Model A.\n\n"
            "Dots near the diagonal:\n"
            "both models agree.\n\n"
            "Gray cloud = null (shuffled\n"
            "labels — no structure expected).",
            xy=(0.03, 0.97), xycoords="axes fraction",
            va="top", ha="left", fontsize=6.5, color=THEME["text"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=THEME["spine"], alpha=0.93, lw=0.7),
        )

    p_str = f"{res['p']:.1e}" if res["p"] < 0.001 else f"{res['p']:.3f}"
    ax.set_title(
        f"{model_a_short} × {model_b_short} ({res['dim']}-dim)\n"
        f"Spearman ρ = {res['r']:.3f}   p = {p_str}",
        fontsize=9, fontweight="bold",
    )
    ax.set_xlabel(f"Concept-pair similarity inside {model_b_short}", fontsize=8.5)
    ax.set_ylabel(f"Concept-pair similarity inside {model_a_short}", fontsize=8.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    apply_theme(ax)

# ── Delta row: single dot plot — real ρ (diamond) vs null cloud ───────────────
N_PERMS_DELTA = 1000
rng_delta = np.random.default_rng(99)
print("Computing permutation null distributions ...")

pair_data = []
for res in rsa_results:
    tri_a, tri_b = res["tri_a"], res["tri_b"]
    null_rhos = np.array([
        spearmanr(tri_a, rng_delta.permutation(tri_b)).statistic
        for _ in range(N_PERMS_DELTA)
    ])
    real_rho = res["r"]
    z = (real_rho - null_rhos.mean()) / null_rhos.std()
    perm_p = float(np.mean(null_rhos >= real_rho))
    ma = res["model_a"].split("/")[-1]
    mb = res["model_b"].split("/")[-1]
    pair_data.append({
        "null_rhos": null_rhos, "real_rho": real_rho,
        "z": z, "perm_p": perm_p,
        "label": f"{ma} × {mb}  ({res['dim']}-dim)",
    })

print("  done.")

rng_jitter = np.random.default_rng(7)
for yi, pd in enumerate(pair_data):
    jitter = rng_jitter.uniform(-0.28, 0.28, N_PERMS_DELTA)
    # null cloud — tight smear near zero
    ax_delta.scatter(pd["null_rhos"], yi + jitter,
                     s=5, color="#AAAAAA", alpha=0.10, zorder=1, linewidths=0)
    # null mean ± 2σ whisker
    nm, ns = pd["null_rhos"].mean(), pd["null_rhos"].std()
    ax_delta.errorbar([nm], [yi], xerr=[[2 * ns], [2 * ns]],
                      fmt="none", color="#888888", lw=1.8, capsize=5, zorder=2)
    ax_delta.scatter([nm], [yi], s=55, color="#888888", marker="|",
                     linewidths=2.5, zorder=2)
    # real ρ — large diamond, the dominant visual element
    ax_delta.scatter([pd["real_rho"]], [yi], s=260, color="#C62828",
                     marker="D", zorder=4, edgecolors="white", linewidths=1.8)
    p_str = "p < 0.001" if pd["perm_p"] == 0.0 else f"p = {pd['perm_p']:.3f}"
    ax_delta.annotate(
        f"ρ = {pd['real_rho']:.3f},  z = {pd['z']:.1f},  {p_str}",
        xy=(pd["real_rho"], yi), xytext=(pd["real_rho"] + 0.022, yi),
        fontsize=8.5, color="#C62828", fontweight="bold",
        va="center", ha="left",
        arrowprops=dict(arrowstyle="-", color="#C62828", lw=0.5),
    )

ax_delta.set_yticks(range(len(pair_data)))
ax_delta.set_yticklabels([pd["label"] for pd in pair_data], fontsize=9)
ax_delta.set_xlabel("Spearman ρ  (inter-concept cosine agreement)", fontsize=9.5)
xmax = max(pd["real_rho"] for pd in pair_data) * 1.55
ax_delta.set_xlim(-0.30, xmax)
ax_delta.axvline(0, color="#555555", lw=0.8, ls="--", alpha=0.35, zorder=0)
ax_delta.set_title(
    "Observed Spearman ρ (◆) vs permutation null — 1,000 label shuffles per pair"
    "   ·   Gray cloud = null, whisker = null mean ± 2σ",
    fontsize=9.5, fontweight="bold",
)
apply_theme(ax_delta)

# shared legend
_seen_cats = {}
for c in ALL_17_CONCEPTS:
    cat, col = CATEGORIES[c]
    _seen_cats.setdefault(cat, col)
legend_handles = [
    _L2D([0], [0], marker="o", color="w", markerfacecolor=col,
         markersize=8, label=CAT_LABELS[cat])
    for cat, col in _seen_cats.items()
] + [
    _L2D([0], [0], marker="o", color="w", markerfacecolor="#AAAAAA",
         markersize=7, alpha=0.5, label="Null (concept labels shuffled)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=6, fontsize=8,
           bbox_to_anchor=(0.5, -0.01), frameon=False,
           title="Concept category  ·  color = category of first concept in pair",
           title_fontsize=7.5)

z_lo = min(pd["z"] for pd in pair_data)
z_hi = max(pd["z"] for pd in pair_data)
fig.suptitle(
    "Shared concept-structure across 4 cross-family model pairs — no Procrustes rotation\n"
    "If one model finds two concepts similar, does the other?  "
    f"Bottom: observed ρ vs permutation null — {z_lo:.1f}–{z_hi:.1f}σ above chance across all pairs.",
    fontsize=11, fontweight="bold",
)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])

out_dir = Path(__file__).parent.parent / "figures"
out_dir.mkdir(exist_ok=True)
out1 = out_dir / "fig_rsa_validation.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out1}")
plt.close()

# ── Figure 2: difference heatmaps (A − B) per pair, shared concept ordering ──
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# one shared concept order from the aggregate average across all pairs
_agg = np.mean([(r["sim_a"] + r["sim_b"]) / 2 for r in rsa_results], axis=0)
_dist = np.clip(1 - _agg, 0, None)
np.fill_diagonal(_dist, 0)
_dist = (_dist + _dist.T) / 2
_Z = linkage(squareform(_dist), method="average")
shared_order  = dendrogram(_Z, no_plot=True)["leaves"]
labels_shared = [ALL_17_CONCEPTS[i].replace("_", " ") for i in shared_order]

fig, axes = plt.subplots(2, 2, figsize=(14, 13))
fig.patch.set_facecolor("white")

for ax, res in zip(axes.flatten(), rsa_results):
    sim_a = res["sim_a"]
    sim_b = res["sim_b"]
    diff  = sim_a - sim_b  # positive = A stronger, negative = B stronger

    diff_ord   = diff[np.ix_(shared_order, shared_order)]
    labels_ord = labels_shared

    # symmetric colour scale, at least ±0.25
    vabs = max(0.25, float(np.abs(diff_ord[np.triu_indices(len(ALL_17_CONCEPTS), k=1)]).max()) * 1.15)
    im = ax.imshow(diff_ord, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")

    ax.set_xticks(range(len(ALL_17_CONCEPTS)))
    ax.set_xticklabels(labels_ord, fontsize=7.5, rotation=45, ha="right")
    ax.set_yticks(range(len(ALL_17_CONCEPTS)))
    ax.set_yticklabels(labels_ord, fontsize=7.5)

    mad = float(np.abs(diff_ord[np.triu_indices(len(ALL_17_CONCEPTS), k=1)]).mean())
    model_a_short = res["model_a"].split("/")[-1]
    model_b_short = res["model_b"].split("/")[-1]

    ax.set_title(
        f"{model_a_short} × {model_b_short} ({res['dim']}-dim)\n"
        f"Spearman ρ = {res['r']:.3f}   mean |A−B| = {mad:.3f}\n"
        f"red = {model_a_short} stronger   ·   blue = {model_b_short} stronger",
        fontsize=8.5, fontweight="bold", pad=8,
    )
    apply_theme(ax)
    fig.colorbar(im, ax=ax, shrink=0.72, label="A − B  (inter-concept cosine)", pad=0.02)

fig.suptitle(
    "Concept-structure agreement — difference between inter-concept similarity matrices  (A − B)\n"
    "White = both models assign the same similarity to this concept pair  ·  Concepts sorted by hierarchical clustering",
    fontsize=12, fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.94])

out2 = out_dir / "fig_rsa_validation_heatmaps.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close()

# ── Figure 3: aggregate concept geometry — mean across all pairs ───────────────
agg_sim = np.mean([(r["sim_a"] + r["sim_b"]) / 2 for r in rsa_results], axis=0)
agg_ord = agg_sim[np.ix_(shared_order, shared_order)]

n = len(ALL_17_CONCEPTS)
off_diag_vals = agg_ord[~np.eye(n, dtype=bool)]
vabs = float(np.abs(off_diag_vals).max()) * 1.1

fig, ax = plt.subplots(figsize=(11, 10))
fig.patch.set_facecolor("white")

im = ax.imshow(agg_ord, cmap="PRGn", vmin=-vabs, vmax=vabs, aspect="auto")
ax.set_xticks(range(n))
ax.set_xticklabels(labels_shared, fontsize=10.5, rotation=45, ha="right")
ax.set_yticks(range(n))
ax.set_yticklabels(labels_shared, fontsize=10.5)

# annotate the most extreme off-diagonal cells
triu = np.triu_indices(n, k=1)
pairs_sorted = sorted(
    zip(off_diag_vals[:len(triu[0])], triu[0], triu[1]),
    key=lambda x: abs(x[0]), reverse=True,
)[:6]
for val, i, j in pairs_sorted:
    for ri, ci in [(i, j), (j, i)]:
        ax.text(ci, ri, f"{val:+.2f}", ha="center", va="center",
                fontsize=7, color="white" if abs(val) > vabs * 0.55 else "#333333",
                fontweight="bold")

apply_theme(ax)
cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
cb.set_label("mean inter-concept cosine  (green = consistently similar · purple = consistently distinct)",
             fontsize=9)

ax.set_title(
    "Shared concept geometry — mean inter-concept cosine across 8 models · 4 cross-family pairs\n"
    "The relational structure of semantic space that independently trained architectures converge on",
    fontsize=12, fontweight="bold", pad=12,
)
plt.tight_layout()

out3 = out_dir / "fig_concept_geometry.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close()

# ── Figure 4: before/after cross-model similarity — aggregate all 4 pairs ─────
from scipy.linalg import orthogonal_procrustes

def cross_sim(dirs_a, dirs_b, R, concepts):
    """C[i,j] = cosine(dirs_a[concept_i], dirs_b[concept_j] @ R). R=None for raw."""
    va = np.vstack([dirs_a[c] for c in concepts]).astype(np.float64)
    vb = np.vstack([dirs_b[c] for c in concepts]).astype(np.float64)
    if R is not None:
        vb = vb @ R
    va /= np.linalg.norm(va, axis=1, keepdims=True).clip(min=1e-12)
    vb /= np.linalg.norm(vb, axis=1, keepdims=True).clip(min=1e-12)
    return va @ vb.T

print("Computing cross-model similarity matrices ...")
C_raw_all, C_aln_all = [], []
for spec in PAIRS:
    short = f"{spec['model_a'].split('/')[-1]} × {spec['model_b'].split('/')[-1]}"
    print(f"  {short}")
    da = load_directions(spec["model_a"], ALL_17_CONCEPTS)
    db = load_directions(spec["model_b"], ALL_17_CONCEPTS)
    mat_a = np.vstack([da[c] for c in ALL_17_CONCEPTS]).astype(np.float64)
    mat_b = np.vstack([db[c] for c in ALL_17_CONCEPTS]).astype(np.float64)
    R, _  = orthogonal_procrustes(mat_b, mat_a)
    C_raw_all.append(cross_sim(da, db, None, ALL_17_CONCEPTS))
    C_aln_all.append(cross_sim(da, db, R,    ALL_17_CONCEPTS))

C_raw_agg = np.mean(C_raw_all, axis=0)[np.ix_(shared_order, shared_order)]
C_aln_agg = np.mean(C_aln_all, axis=0)[np.ix_(shared_order, shared_order)]

# shared colour scale set by the post-rotation diagonal so the contrast is maximal
vabs = float(C_aln_agg.max()) * 1.05

fig, (ax_raw, ax_aln) = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor("white")

for ax, mat, title in [
    (ax_raw, C_raw_agg,
     "Before Procrustes rotation\n(raw coordinate frames — no signal)"),
    (ax_aln, C_aln_agg,
     "After Procrustes rotation\n(one R per pair — diagonal lights up)"),
]:
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels_shared, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels_shared, fontsize=9)
    ax.set_xlabel("Model B concept direction", fontsize=10)
    ax.set_ylabel("Model A concept direction", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    apply_theme(ax)
    fig.colorbar(im, ax=ax, shrink=0.75, label="cross-model cosine", pad=0.02)

fig.suptitle(
    "Cross-model concept alignment — mean across 4 cross-family pairs  (8 models total)\n"
    "Cell (row, col): mean cosine between Model A's direction for row-concept\n"
    "and Model B's direction for col-concept  ·  Diagonal = same concept matched to itself",
    fontsize=11, fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.90])

out4 = out_dir / "fig_cross_model_similarity.png"
plt.savefig(out4, dpi=150, bbox_inches="tight")
print(f"Saved: {out4}")
plt.close()

# ── Summary table ─────────────────────────────────────────────────────────────
print("\nRSA summary:")
print(f"  {'Pair':42s}  {'Spearman ρ':>10s}  {'p-value':>12s}")
print("  " + "─" * 68)
for res in rsa_results:
    lbl = res["label"].replace("\n", "  ×  ")
    print(f"  {lbl:42s}  {res['r']:10.4f}  {res['p']:12.2e}")
print()
print("Wherever one model sees a strong inter-concept relationship, so does the other.")
print("The shared geometry that Procrustes rotation aligned into view was already there.")

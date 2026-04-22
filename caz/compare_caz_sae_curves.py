#!/usr/bin/env python3
"""
compare_caz_sae_curves.py — Compare CAZ eigenvector separation curves to SAE
discrimination curves layer-by-layer for Gemma-2-2b.

For each of the 7 CAZ concepts, we compute two per-layer scores:

  CAZ score  — mean pairwise separation projected onto the top concept
               eigenvector at that layer (concept-specific, no SAE needed).

  SAE score  — mean |differential| of top-K SAE features between concept+
               and concept- pairs (from a previous gemma_scope_xval run).

Then we measure how well the CAZ score predicts the SAE score (Spearman r)
and where the curves diverge — the derivative vs. cumulative distinction.

NOTE: This comparison covers 7 concepts only. The SAE encodes 16,000+ features
spanning the full activation space; CAZ is targeted at these 7 semantic axes.
The question is not "can CAZ replace SAE?" but "do the two methods agree on
the signal for these 7 concepts?"

Usage:
    python src/compare_caz_sae_curves.py
    python src/compare_caz_sae_curves.py --xval-dir results/gemma_scope_xval
    python src/compare_caz_sae_curves.py --plot          # save per-concept PNGs
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths (same resolution logic as gemma_scope_xval) ────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
CAZ_ROOT  = Path(__file__).resolve().parents[1]

def _find_pairs_dir() -> Path:
    candidates = [
        REPO_ROOT / "Rosetta_Concept_Pairs" / "pairs" / "raw" / "v1",
        Path.home() / "Rosetta_Concept_Pairs" / "pairs" / "raw" / "v1",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

PAIRS_DIR   = _find_pairs_dir()
RESULTS_DIR = CAZ_ROOT / "results"
MODEL_ID    = "google/gemma-2-2b"

CONCEPTS = [
    "credibility", "certainty", "sentiment", "moral_valence",
    "causation", "temporal_order", "negation",
]

# ── data loading ──────────────────────────────────────────────────────────────

def load_pairs(concept: str, n: int) -> tuple[list[str], list[str]]:
    path = PAIRS_DIR / f"{concept}_consensus_pairs.jsonl"
    if not path.exists():
        log.warning("No pairs file for %s at %s", concept, path)
        return [], []
    pos, neg = [], []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if item["label"] == 1 and len(pos) < n:
                pos.append(item["text"])
            elif item["label"] == 0 and len(neg) < n:
                neg.append(item["text"])
            if len(pos) >= n and len(neg) >= n:
                break
    log.info("  %s: %d pos, %d neg", concept, len(pos), len(neg))
    return pos, neg

# ── model + forward pass ──────────────────────────────────────────────────────

def load_model(device: str):
    log.info("Loading %s...", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    log.info("  %d layers, hidden_dim=%d",
             model.config.num_hidden_layers, model.config.hidden_size)
    return model, tokenizer


@torch.no_grad()
def get_residual_streams(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    batch_size: int = 4,
    max_length: int = 256,
) -> np.ndarray:
    """Return (n_texts, n_layers, hidden_dim) float32."""
    n_layers   = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    all_acts   = np.zeros((len(texts), n_layers, hidden_dim), dtype=np.float32)

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        layer_acts: list[np.ndarray] = []

        def make_hook(idx):
            def hook(module, inp, out):
                hs   = out[0] if isinstance(out, tuple) else out
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hs.float() * mask).sum(dim=1) / mask.sum(dim=1)
                layer_acts.append(pooled.cpu().numpy())
            return hook

        handles = [
            layer.register_forward_hook(make_hook(i))
            for i, layer in enumerate(model.model.layers)
        ]
        try:
            model(**enc)
        finally:
            for h in handles:
                h.remove()

        for i, acts in enumerate(layer_acts):
            all_acts[batch_start : batch_start + len(batch), i, :] = acts

    return all_acts   # (n_texts, n_layers, hidden_dim)

# ── CAZ separation score ──────────────────────────────────────────────────────

def caz_separation_scores(pos_resid: np.ndarray, neg_resid: np.ndarray) -> np.ndarray:
    """
    Per-layer CAZ separation score.

    At each layer:
      1. Compute paired difference vectors: delta[i] = pos[i] - neg[i]
      2. Find top concept eigenvector via SVD of the delta matrix
      3. Score = mean |projection of delta onto top eigvec|

    This is analogous to the SAE's "mean top-K |differential|" but uses our
    concept-specific eigenvector rather than sparse SAE features.

    pos_resid / neg_resid: (n_pairs, n_layers, hidden_dim)
    Returns: (n_layers,) float32
    """
    n_pairs, n_layers, _ = pos_resid.shape
    scores = np.zeros(n_layers, dtype=np.float32)

    for layer in range(n_layers):
        pos_l = pos_resid[:, layer, :].astype(np.float32)
        neg_l = neg_resid[:, layer, :].astype(np.float32)
        delta = pos_l - neg_l                         # (n_pairs, hidden_dim)
        delta -= delta.mean(axis=0, keepdims=True)

        # Top eigenvector via truncated SVD
        k = min(1, delta.shape[0] - 1)
        if k < 1:
            continue
        _, s, Vt = np.linalg.svd(delta, full_matrices=False)
        top_eigvec = Vt[0]                            # (hidden_dim,)
        top_eigvec /= np.linalg.norm(top_eigvec) + 1e-8

        projections = delta @ top_eigvec              # (n_pairs,)
        scores[layer] = float(np.mean(np.abs(projections)))

    return scores

# ── comparison ────────────────────────────────────────────────────────────────

def spearman_r(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r, _ = spearmanr(a, b)
    return float(r)


def divergence_layer(caz: np.ndarray, sae: np.ndarray) -> int | None:
    """
    Find the first layer where the SAE score keeps climbing but our CAZ score
    plateaus (derivative drops below half its peak value while SAE is still
    rising). Returns None if no clear divergence.
    """
    caz_norm = caz / (caz.max() + 1e-8)
    sae_norm = sae / (sae.max() + 1e-8)
    caz_deriv = np.gradient(caz_norm)
    caz_peak  = int(np.argmax(caz_deriv))

    for layer in range(caz_peak, len(caz_norm)):
        if caz_deriv[layer] < 0.1 * caz_deriv.max() and sae_norm[layer] < sae_norm[-1] * 0.9:
            return layer
    return None


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # ── Load SAE curves from previous xval run ────────────────────────────────
    xval_dir = Path(args.xval_dir)
    la_path  = xval_dir / "layer_agreement.json"
    if not la_path.exists():
        log.error("No layer_agreement.json at %s — run gemma_scope_xval.py first", xval_dir)
        raise SystemExit(1)

    sae_curves = json.loads(la_path.read_text())["by_concept"]
    log.info("Loaded SAE curves for: %s", list(sae_curves.keys()))

    # ── Load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model(device)
    n_layers = model.config.num_hidden_layers

    results = {}

    for concept in CONCEPTS:
        if concept not in sae_curves:
            log.warning("No SAE data for %s — skipping", concept)
            continue

        log.info("=== %s ===", concept)

        pos_texts, neg_texts = load_pairs(concept, args.pairs_per_concept)
        if not pos_texts:
            continue

        # Align pair counts (use the shorter side for paired differences)
        n = min(len(pos_texts), len(neg_texts))
        pos_texts, neg_texts = pos_texts[:n], neg_texts[:n]

        pos_resid = get_residual_streams(model, tokenizer, pos_texts, device)
        neg_resid = get_residual_streams(model, tokenizer, neg_texts, device)

        caz_scores = caz_separation_scores(pos_resid, neg_resid)

        sae_scores = np.array(sae_curves[concept], dtype=np.float32)
        # Truncate to same length (should already match n_layers)
        n_l = min(len(caz_scores), len(sae_scores))
        caz_s = caz_scores[:n_l]
        sae_s = sae_scores[:n_l]

        r = spearman_r(caz_s, sae_s)
        div_layer = divergence_layer(caz_s, sae_s)

        # CAZ peak: layer with highest derivative
        caz_deriv = np.gradient(caz_s / (caz_s.max() + 1e-8))
        caz_peak_layer = int(np.argmax(caz_deriv))

        log.info("  Spearman r=%.3f | CAZ derivative peak=L%d | divergence=L%s",
                 r, caz_peak_layer, div_layer)

        results[concept] = {
            "spearman_r":          round(r, 4),
            "caz_derivative_peak": caz_peak_layer,
            "divergence_layer":    div_layer,
            "caz_scores":          caz_s.tolist(),
            "sae_scores":          sae_s.tolist(),
        }

        if args.plot:
            _plot_concept(concept, caz_s, sae_s, caz_peak_layer, div_layer, r, xval_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    out_path = xval_dir / "caz_vs_sae_curves.json"
    out_path.write_text(json.dumps({
        "description": (
            "Per-layer CAZ eigenvector separation vs SAE discrimination for 7 concepts. "
            "NOTE: CAZ covers 7 concepts; SAE covers 16,000+ features — comparison is "
            "scoped to these concepts only."
        ),
        "model":    MODEL_ID,
        "n_layers": n_layers,
        "results":  results,
    }, indent=2))
    log.info("Saved: %s", out_path)

    # Print summary table
    log.info("")
    log.info("=== SUMMARY ===")
    log.info("%-20s  %8s  %10s  %12s", "concept", "r", "caz_peak", "diverge_at")
    for concept, r in results.items():
        log.info("%-20s  %8.3f  %10s  %12s",
                 concept,
                 r["spearman_r"],
                 f"L{r['caz_derivative_peak']}",
                 f"L{r['divergence_layer']}" if r["divergence_layer"] else "—")

    mean_r = np.mean([v["spearman_r"] for v in results.values()])
    log.info("")
    log.info("Mean Spearman r across %d concepts: %.3f", len(results), mean_r)
    log.info("")
    log.info("NOTE: High r means CAZ eigenvectors track the same signal as SAE features")
    log.info("      for these 7 concepts. SAE covers 16k+ features; this is a targeted")
    log.info("      comparison only.")


def _plot_concept(
    concept: str,
    caz: np.ndarray,
    sae: np.ndarray,
    caz_peak: int,
    div_layer: int | None,
    r: float,
    out_dir: Path,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping plots")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    layers = np.arange(len(caz))

    # Normalise both to [0, 1] for visual comparison
    caz_n = caz / (caz.max() + 1e-8)
    sae_n = sae / (sae.max() + 1e-8)

    ax.plot(layers, sae_n,  color="#2196F3", linewidth=2, label="SAE discrimination (normalised)")
    ax.plot(layers, caz_n,  color="#FF5722", linewidth=2, label="CAZ eigenvec separation (normalised)")
    ax.axvline(caz_peak, color="#FF5722", linestyle="--", alpha=0.5, label=f"CAZ derivative peak (L{caz_peak})")
    if div_layer is not None:
        ax.axvline(div_layer, color="gray", linestyle=":", alpha=0.7, label=f"Divergence (L{div_layer})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalised score")
    ax.set_title(f"{concept}  |  Spearman r={r:.3f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    out = out_dir / f"curve_comparison_{concept}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved plot: %s", out)


def main():
    parser = argparse.ArgumentParser(description="Compare CAZ eigenvector vs SAE discrimination curves")
    parser.add_argument(
        "--xval-dir",
        default=str(RESULTS_DIR / "gemma_scope_xval"),
        help="Directory containing layer_agreement.json from gemma_scope_xval run",
    )
    parser.add_argument(
        "--pairs-per-concept", type=int, default=50,
        help="Number of pairs per concept (default 50)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save per-concept PNG curve comparisons",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

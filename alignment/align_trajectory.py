"""
align_trajectory.py — Layer-by-layer Procrustes trajectory analysis.

For each same-dim same-depth model pair, fits an independent Procrustes rotation
at every layer and tracks how aligned cosine evolves across network depth.

This tests a deeper form of the PRH: not just that models share concept directions
at their peak layers, but that the trajectory of geometric alignment across depth
follows a shared pattern. No depth normalization is applied — pairs are restricted
to models with identical layer counts and hidden dimensions so structure is compared
structurally, not proportionally.

Confirmed tier (pairs with matched dim AND matched depth):
  4096-dim / 32 layers : pythia-6.9b, opt-6.7b, Mistral-7B-v0.3, Llama-3.1-8B
  2048-dim / 16 layers : pythia-1b, Llama-3.2-1B
  2048-dim / 24 layers : pythia-1.4b, opt-1.3b   (if both available)

Data source: calibration_alllayer_{concept}.npy  — shape (n_layers, 200, hidden_dim)
  First 100 rows = positive class, last 100 = negative class.

Outputs
-------
results/trajectory/
  trajectory_raw.csv          — per-layer per-pair per-concept aligned cosines
  trajectory_summary.csv      — per-pair per-concept: peak-layer cosine, pre/post
                                 means, event-relative shape metrics
  plots/                      — one trajectory plot per concept

Usage
-----
    cd ~/semantic_convergence
    python src/align_trajectory.py
    python src/align_trajectory.py --concepts certainty causation
    python src/align_trajectory.py --tier all     # include 768/12 group
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import lru_cache

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

RESULTS_ROOT = Path("results")
OUT_DIR = RESULTS_ROOT / "trajectory"

CONCEPTS = [
    "certainty", "causation", "temporal_order", "credibility",
    "sentiment", "moral_valence", "negation",
]

# Models to include by tier. Each group shares dim AND layer count.
CONFIRMED_TIER: dict[tuple[int, int], list[str]] = {
    (4096, 32): [
        "EleutherAI/pythia-6.9b",
        "facebook/opt-6.7b",
        "mistralai/Mistral-7B-v0.3",
        "meta-llama/Llama-3.1-8B",
    ],
    (2048, 16): [
        "EleutherAI/pythia-1b",
        "meta-llama/Llama-3.2-1B",
    ],
    (2048, 24): [
        "EleutherAI/pythia-1.4b",
        "facebook/opt-1.3b",
    ],
}

EXTENDED_TIER: dict[tuple[int, int], list[str]] = {
    **CONFIRMED_TIER,
    (768, 12): [
        "EleutherAI/pythia-160m",
        "facebook/opt-125m",
        "openai-community/gpt2",
        "EleutherAI/gpt-neo-125m",
    ],
}


# ---------------------------------------------------------------------------
# Data loading — deduplicate to most-recent xarch dir per model
# ---------------------------------------------------------------------------

def _best_xarch_dirs() -> dict[str, Path]:
    best: dict[str, tuple[float, Path]] = {}
    for d in RESULTS_ROOT.glob("xarch_*"):
        summ = d / "run_summary.json"
        if not summ.exists():
            continue
        try:
            model_id = json.loads(summ.read_text())["model_id"]
        except (KeyError, json.JSONDecodeError):
            continue
        mtime = d.stat().st_mtime
        if model_id not in best or mtime > best[model_id][0]:
            best[model_id] = (mtime, d)
    return {mid: path for mid, (_, path) in best.items()}


def load_alllayer(model_id: str, concept: str, xarch_dirs: dict[str, Path]) -> np.ndarray | None:
    """Load (n_layers, 200, hidden_dim) calibration array, or None if missing."""
    d = xarch_dirs.get(model_id)
    if d is None:
        return None
    p = d / f"calibration_alllayer_{concept}.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


def load_caz_peaks(model_id: str, concept: str, xarch_dirs: dict[str, Path]) -> int | None:
    d = xarch_dirs.get(model_id)
    if d is None:
        return None
    p = d / f"caz_{concept}.json"
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text())
        return raw["layer_data"]["peak_layer"]
    except (KeyError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Core: per-layer Procrustes
# ---------------------------------------------------------------------------

def _dom_from_acts(acts: np.ndarray) -> np.ndarray:
    """Difference-in-means DOM from (n, dim) activation matrix (first half = pos)."""
    half = acts.shape[0] // 2
    diff = acts[:half].mean(axis=0) - acts[half:].mean(axis=0)
    norm = np.linalg.norm(diff)
    return diff / (norm + 1e-10)


def _thin_svd_cache(acts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin SVD of (n, dim) activation matrix. Returns (U, s, Vh)."""
    return np.linalg.svd(acts.astype(np.float64), full_matrices=False)


def _aligned_cosine_lowrank(
    src_acts: np.ndarray,
    tgt_acts: np.ndarray,
    src_dom: np.ndarray,
    tgt_dom: np.ndarray,
    src_svd: tuple | None = None,
    tgt_svd: tuple | None = None,
) -> float:
    """Aligned cosine via efficient low-rank Procrustes (avoids full dim×dim SVD).

    Convention matches rosetta_tools: orthogonal_procrustes(A=tgt_acts, B=src_acts)
    finds R minimising ||tgt_acts - src_acts R||_F; applied as tgt_dom @ R.

    M = src_acts^T @ tgt_acts = Vhs^T K Vht  where K = diag(ss) @ Us^T @ Ut @ diag(st).
    SVD(K) = Uk sk Vhk → tgt_dom @ R = Vhs^T @ (Uk @ (Vhk @ (Vht @ tgt_dom))).

    Complexity: 2 × O(n²d) thin SVDs + O(n³) inner SVD + O(nd) application,
    vs O(d³) for the naive full-matrix SVD (400× faster at d=4096, n=200).
    """
    Us, ss, Vhs = src_svd if src_svd is not None else _thin_svd_cache(src_acts)
    Ut, st, Vht = tgt_svd if tgt_svd is not None else _thin_svd_cache(tgt_acts)

    # K = diag(ss) @ Us^T @ Ut @ diag(st)  — shape (n, n)
    K = (ss[:, None] * (Us.T @ Ut)) * st[None, :]
    Uk, _, Vhk = np.linalg.svd(K, full_matrices=False)

    # Apply: rotated = Vhs^T @ (Uk @ (Vhk @ (Vht @ tgt_dom)))
    tgt_d = tgt_dom.astype(np.float64)
    rotated = Vhs.T @ (Uk @ (Vhk @ (Vht @ tgt_d)))

    src_d = src_dom.astype(np.float64)
    denom = np.linalg.norm(src_d) * np.linalg.norm(rotated) + 1e-10
    return float(np.dot(src_d, rotated) / denom)


def layer_by_layer_alignment(
    src_all: np.ndarray,   # (n_layers, 200, dim)
    tgt_all: np.ndarray,   # (n_layers, 200, dim)
) -> list[dict]:
    """Fit independent Procrustes R at each layer; return per-layer result dicts.

    SVDs are cached per layer so each (model, layer) decomposition is computed
    once regardless of how many pairs reference it.
    """
    n_layers = src_all.shape[0]

    # Pre-compute thin SVDs for all layers (n_examples << dim so this is fast)
    src_svds = [_thin_svd_cache(src_all[l]) for l in range(n_layers)]
    tgt_svds = [_thin_svd_cache(tgt_all[l]) for l in range(n_layers)]

    rows = []
    for layer in range(n_layers):
        src_acts = src_all[layer].astype(np.float64)
        tgt_acts = tgt_all[layer].astype(np.float64)

        src_dom = _dom_from_acts(src_acts)
        tgt_dom = _dom_from_acts(tgt_acts)

        raw_cosine = float(np.dot(src_dom, tgt_dom))

        try:
            aligned_cosine = _aligned_cosine_lowrank(
                src_acts, tgt_acts, src_dom, tgt_dom,
                src_svd=src_svds[layer], tgt_svd=tgt_svds[layer],
            )
        except Exception as e:
            log.debug("layer %d alignment failed: %s", layer, e)
            aligned_cosine = float("nan")

        rows.append({
            "layer": layer,
            "raw_cosine": round(raw_cosine, 5),
            "aligned_cosine": round(aligned_cosine, 5),
        })
    return rows


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_concept(
    concept: str,
    tier: dict[tuple[int, int], list[str]],
    xarch_dirs: dict[str, Path],
) -> pd.DataFrame:
    rows = []
    for (dim, n_layers), model_list in tier.items():
        available = [m for m in model_list if load_alllayer(m, concept, xarch_dirs) is not None]
        if len(available) < 2:
            log.warning("  [%s dim=%d L=%d] only %d models available — skipping",
                        concept, dim, n_layers, len(available))
            continue

        # Cache alllayer arrays and thin SVDs per model — each used in multiple pairs
        alllayer: dict[str, np.ndarray] = {}
        svd_cache: dict[str, list] = {}  # model → list of (U, s, Vh) per layer
        for m in available:
            arr = load_alllayer(m, concept, xarch_dirs)
            if arr is None:
                continue
            alllayer[m] = arr
            svd_cache[m] = [_thin_svd_cache(arr[l]) for l in range(arr.shape[0])]
            log.info("  cached SVDs for %s (%d layers)", m.split("/")[-1], arr.shape[0])

        for src, tgt in permutations(available, 2):
            src_all = alllayer.get(src)
            tgt_all = alllayer.get(tgt)
            if src_all is None or tgt_all is None:
                continue
            if src_all.shape != tgt_all.shape:
                log.warning("  shape mismatch %s vs %s — skipping", src, tgt)
                continue

            src_peak = load_caz_peaks(src, concept, xarch_dirs)
            tgt_peak = load_caz_peaks(tgt, concept, xarch_dirs)

            log.info("  %s → %s", src.split("/")[-1], tgt.split("/")[-1])

            # Use pre-computed SVDs; bypass per-pair recomputation
            n_layers = src_all.shape[0]
            rows_pair = []
            for layer in range(n_layers):
                src_acts = src_all[layer].astype(np.float64)
                tgt_acts = tgt_all[layer].astype(np.float64)
                src_dom = _dom_from_acts(src_acts)
                tgt_dom = _dom_from_acts(tgt_acts)
                raw_cosine = float(np.dot(src_dom, tgt_dom))
                try:
                    aligned_cosine = _aligned_cosine_lowrank(
                        src_acts, tgt_acts, src_dom, tgt_dom,
                        src_svd=svd_cache[src][layer],
                        tgt_svd=svd_cache[tgt][layer],
                    )
                except Exception as e:
                    log.debug("layer %d: %s", layer, e)
                    aligned_cosine = float("nan")
                rows_pair.append({"layer": layer,
                                   "raw_cosine": round(raw_cosine, 5),
                                   "aligned_cosine": round(aligned_cosine, 5)})
            layer_rows = rows_pair

            for lr in layer_rows:
                rows.append({
                    "concept": concept,
                    "dim": dim,
                    "n_layers": n_layers,
                    "src_model": src,
                    "tgt_model": tgt,
                    "src_peak": src_peak,
                    "tgt_peak": tgt_peak,
                    **lr,
                    "rel_src": lr["layer"] - src_peak if src_peak is not None else None,
                    "rel_tgt": lr["layer"] - tgt_peak if tgt_peak is not None else None,
                })

    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (concept, src, tgt), grp in df.groupby(["concept", "src_model", "tgt_model"]):
        grp = grp.sort_values("layer")
        src_peak = grp["src_peak"].iloc[0]
        tgt_peak = grp["tgt_peak"].iloc[0]

        # Aligned cosine at the src peak layer
        at_peak = grp[grp["layer"] == src_peak]["aligned_cosine"]
        peak_cos = float(at_peak.mean()) if len(at_peak) else float("nan")

        # Pre-peak vs post-peak means (relative to src peak)
        pre = grp[grp["rel_src"] < 0]["aligned_cosine"]
        post = grp[grp["rel_src"] > 0]["aligned_cosine"]
        pre_mean = float(pre.mean()) if len(pre) else float("nan")
        post_mean = float(post.mean()) if len(post) else float("nan")

        rows.append({
            "concept": concept,
            "src_model": src,
            "tgt_model": tgt,
            "dim": grp["dim"].iloc[0],
            "n_layers": grp["n_layers"].iloc[0],
            "src_peak": src_peak,
            "tgt_peak": tgt_peak,
            "aligned_at_src_peak": round(peak_cos, 5),
            "pre_peak_mean": round(pre_mean, 5),
            "post_peak_mean": round(post_mean, 5),
            "delta_post_pre": round(post_mean - pre_mean, 5) if not (
                np.isnan(pre_mean) or np.isnan(post_mean)) else float("nan"),
            "global_mean": round(float(grp["aligned_cosine"].mean()), 5),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "EleutherAI/pythia-6.9b":     "#E53935",
    "facebook/opt-6.7b":           "#1E88E5",
    "mistralai/Mistral-7B-v0.3":  "#43A047",
    "meta-llama/Llama-3.1-8B":    "#FB8C00",
    "EleutherAI/pythia-1b":        "#8E24AA",
    "meta-llama/Llama-3.2-1B":    "#00ACC1",
}


def plot_concept_trajectories(df: pd.DataFrame, concept: str, out_dir: Path) -> None:
    subset = df[df["concept"] == concept]
    if subset.empty:
        return

    dims = sorted(subset["dim"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(dims), figsize=(6 * len(dims), 4), squeeze=False)
    fig.patch.set_facecolor("white")

    for ax, dim in zip(axes[0], dims):
        ax.set_facecolor("white")
        grp = subset[subset["dim"] == dim]

        # One line per src→tgt pair (aggregate over directions by averaging)
        pairs_seen: dict[frozenset, list] = defaultdict(list)
        for _, row in grp.iterrows():
            key = frozenset([row["src_model"], row["tgt_model"]])
            pairs_seen[key].append(row)

        for pair_key, pair_rows in pairs_seen.items():
            models = sorted(pair_key)
            color = COLORS.get(models[0], "#757575")
            # Average forward and backward direction
            fwd = grp[(grp["src_model"] == models[0]) & (grp["tgt_model"] == models[1])]
            bwd = grp[(grp["src_model"] == models[1]) & (grp["tgt_model"] == models[0])]
            if fwd.empty:
                continue
            fwd_s = fwd.sort_values("layer")
            layers = fwd_s["layer"].values
            cos = fwd_s["aligned_cosine"].values
            if not bwd.empty:
                bwd_s = bwd.sort_values("layer")
                cos = (cos + bwd_s["aligned_cosine"].values) / 2

            label = f"{models[0].split('/')[-1]} ↔ {models[1].split('/')[-1]}"
            ax.plot(layers, cos, linewidth=1.4, color=color, alpha=0.85, label=label)

            # Mark each model's CAZ peak
            for _, r in fwd_s.iterrows():
                if r["layer"] == r["src_peak"]:
                    ax.axvline(r["layer"], color=color, linewidth=0.6, linestyle=":", alpha=0.5)

        ax.set_xlabel("Layer (absolute)", fontsize=9)
        ax.set_ylabel("Aligned cosine", fontsize=9)
        ax.set_title(f"{concept}  —  dim={dim}, L={grp['n_layers'].iloc[0]}",
                     fontsize=10, fontweight="bold", loc="left")
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(0, color="#BDBDBD", linewidth=0.5)
        ax.legend(fontsize=7, facecolor="white")
        ax.grid(axis="y", linewidth=0.3, color="#ECEFF1")
        for spine in ax.spines.values():
            spine.set_edgecolor("#BDBDBD")

    fig.suptitle(f"Layer-by-layer Procrustes alignment — {concept}", fontsize=11, y=1.01)
    fig.tight_layout()
    out_path = out_dir / f"trajectory_{concept}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")
    log.info("  saved %s", out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument("--concepts", nargs="+", default=CONCEPTS,
                        help="Concept names, or 'all' to run every concept.")
    parser.add_argument("--tier", choices=["confirmed", "extended"], default="confirmed",
                        help="confirmed = 1B+ matched pairs only; extended = include 768/12")
    args = parser.parse_args()

    # Allow --concepts all as a shorthand
    if args.concepts == ["all"]:
        args.concepts = CONCEPTS

    tier = CONFIRMED_TIER if args.tier == "confirmed" else EXTENDED_TIER

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plots").mkdir(exist_ok=True)

    xarch_dirs = _best_xarch_dirs()
    log.info("Found %d deduplicated models with xarch data", len(xarch_dirs))

    all_dfs = []
    for concept in args.concepts:
        log.info("=== %s ===", concept)
        df = run_concept(concept, tier, xarch_dirs)
        if df.empty:
            log.warning("  No results for %s", concept)
            continue
        all_dfs.append(df)
        plot_concept_trajectories(df, concept, OUT_DIR / "plots")

    if not all_dfs:
        log.error("No results produced.")
        sys.exit(1)

    raw = pd.concat(all_dfs, ignore_index=True)
    raw.to_csv(OUT_DIR / "trajectory_raw.csv", index=False)
    log.info("Wrote trajectory_raw.csv  (%d rows)", len(raw))

    summary = build_summary(raw)
    summary.to_csv(OUT_DIR / "trajectory_summary.csv", index=False)
    log.info("Wrote trajectory_summary.csv  (%d rows)", len(summary))

    # Print headline numbers
    for (concept, dim), grp in summary.groupby(["concept", "dim"]):
        log.info(
            "  %s  dim=%d  peak_cos=%.4f  pre=%.4f  post=%.4f  delta=%+.4f",
            concept, dim,
            grp["aligned_at_src_peak"].mean(),
            grp["pre_peak_mean"].mean(),
            grp["post_peak_mean"].mean(),
            grp["delta_post_pre"].mean(),
        )


if __name__ == "__main__":
    main()

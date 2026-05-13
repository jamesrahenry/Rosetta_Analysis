#!/usr/bin/env python3
"""
peak_vs_nonpeak.py — Validates that CAZ peak-layer selection matters for
Procrustes alignment.

For every same-dim model pair, fits Procrustes at EACH layer and asks:
does alignment score highest at the CAZ-identified peak?

Method A (always runs — no .npy needed):
    Stacks 17 DOM vectors per layer → [17, hidden_dim].
    Fits R on this matrix; scores per-concept aligned cosine at that layer.
    Underdetermined (17 << hidden_dim) but cross-layer comparison is valid
    since the same bias applies everywhere.

Method B (runs when calibration_alllayer_*.npy present):
    Loads full [n_layers, 500, hidden_dim] calibration activations.
    Fits per-concept Procrustes at every layer using 500 texts.
    Proper methodology matching the primary PRH analysis.
    Download first:
        hf download james-ra-henry/Rosetta-Activations \\
            --repo-type dataset --local-dir ~/rosetta_data/models/ \\
            --include "models/EleutherAI_pythia_160m/*" \\
            --include "models/EleutherAI_gpt_neo_125m/*" \\
            --include "models/openai_community_gpt2/*" \\
            --include "models/facebook_opt_125m/*"

Output:
    ~/rosetta_data/results/PRH/peak_vs_nonpeak_A.json
    ~/rosetta_data/results/PRH/peak_vs_nonpeak_B.json  (if alllayer available)

Usage:
    uv run python alignment/peak_vs_nonpeak.py --same-dim-only
    uv run python alignment/peak_vs_nonpeak.py --same-dim-only --dim 768
    uv run python alignment/peak_vs_nonpeak.py --same-dim-only --dom-only
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

import importlib
if importlib.util.find_spec("rosetta_tools") is None:
    import sys as _sys
    for _p in [Path.home() / "rosetta_tools",
               Path.home() / "Source" / "Rosetta_Program" / "rosetta_tools"]:
        if _p.exists():
            _sys.path.insert(0, str(_p))
            break

from rosetta_tools.paths import ROSETTA_MODELS, ROSETTA_RESULTS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

CONCEPTS = [
    "agency", "authorization", "causation", "certainty", "credibility",
    "deception", "exfiltration", "formality", "moral_valence", "negation",
    "plurality", "sarcasm", "sentiment", "specificity", "temporal_order",
    "threat_severity", "urgency",
]

OUT_DIR = ROSETTA_RESULTS / "PRH"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slug(mid: str) -> str:
    return mid.replace("/", "_").replace("-", "_")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0


def fit_R(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Orthogonal Procrustes: find R s.t. A @ R ≈ B (both centered)."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    R, _ = orthogonal_procrustes(A, B)
    return R


def load_caz_alllayers(model_dir: Path, concept: str) -> dict | None:
    """Load all-layer DOM vectors from caz_{concept}.json.

    Returns dict with keys:
        dom_vectors  — ndarray [n_layers, hidden_dim]
        layers       — list[int]
        peak_layer   — int (0-indexed)
        peak_depth_pct — float
        hidden_dim   — int
    Returns None if file missing.
    """
    p = model_dir / f"caz_{concept}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    metrics = data["layer_data"]["metrics"]
    layers = [m["layer"] for m in metrics]
    dom_vecs = np.array([m["dom_vector"] for m in metrics], dtype=np.float32)
    peak_layer = data["layer_data"]["peak_layer"]
    n_layers = data["n_layers"]
    peak_depth_pct = (peak_layer + 1) / n_layers * 100
    return {
        "dom_vectors": dom_vecs,
        "layers": layers,
        "peak_layer": peak_layer,
        "peak_depth_pct": peak_depth_pct,
        "hidden_dim": data["hidden_dim"],
        "n_layers": n_layers,
    }


def load_alllayer_acts(model_dir: Path, concept: str) -> np.ndarray | None:
    """Load calibration_alllayer_{concept}.npy → [n_layers, 500, hidden_dim]."""
    p = model_dir / f"calibration_alllayer_{concept}.npy"
    if not p.exists():
        return None
    return np.load(p)


def model_dirs_by_dim() -> dict[int, list[tuple[str, Path]]]:
    """Group model directories by hidden_dim."""
    groups: dict[int, list[tuple[str, Path]]] = {}
    for d in sorted(ROSETTA_MODELS.iterdir()):
        if not d.is_dir():
            continue
        caz = d / "caz_credibility.json"
        rs = d / "run_summary.json"
        if not caz.exists() or not rs.exists():
            continue
        try:
            summary = json.loads(rs.read_text())
            mid = summary["model_id"]
            caz_data = json.loads(caz.read_text())
            hdim = caz_data["hidden_dim"]
        except Exception:
            continue
        groups.setdefault(hdim, []).append((mid, d))
    return groups


# ---------------------------------------------------------------------------
# Method A — DOM-vector Procrustes
# ---------------------------------------------------------------------------

def run_method_a(
    pairs: list[tuple[str, Path, str, Path]],
) -> list[dict]:
    """
    For each pair at each layer: stack 17 DOM vectors → fit R → score per concept.

    pairs: list of (src_model_id, src_dir, tgt_model_id, tgt_dir)
    """
    rows = []
    for src_mid, src_dir, tgt_mid, tgt_dir in pairs:
        # Load all-layer CAZ data for all concepts
        src_caz = {c: load_caz_alllayers(src_dir, c) for c in CONCEPTS}
        tgt_caz = {c: load_caz_alllayers(tgt_dir, c) for c in CONCEPTS}

        # Keep only concepts present in both
        valid = [c for c in CONCEPTS if src_caz[c] is not None and tgt_caz[c] is not None]
        if len(valid) < 5:
            log.warning("Skipping %s × %s — fewer than 5 shared concepts",
                        src_mid.split("/")[-1], tgt_mid.split("/")[-1])
            continue

        n_layers = src_caz[valid[0]]["n_layers"]
        if n_layers != tgt_caz[valid[0]]["n_layers"]:
            log.warning("Layer count mismatch for %s × %s — skipping",
                        src_mid.split("/")[-1], tgt_mid.split("/")[-1])
            continue

        log.info("  Method A: %s × %s (%d layers, %d concepts)",
                 src_mid.split("/")[-1], tgt_mid.split("/")[-1],
                 n_layers, len(valid))

        for layer_idx in range(n_layers):
            depth_pct = (layer_idx + 1) / n_layers * 100

            # Stack DOM vectors across concepts at this layer
            try:
                src_stack = np.stack([src_caz[c]["dom_vectors"][layer_idx] for c in valid])
                tgt_stack = np.stack([tgt_caz[c]["dom_vectors"][layer_idx] for c in valid])
                R = fit_R(src_stack, tgt_stack)
            except Exception as e:
                log.debug("Procrustes failed at layer %d for %s × %s: %s",
                          layer_idx, src_mid.split("/")[-1], tgt_mid.split("/")[-1], e)
                continue

            for c in valid:
                src_v = src_caz[c]["dom_vectors"][layer_idx]
                tgt_v = tgt_caz[c]["dom_vectors"][layer_idx]
                aligned_v = src_v @ R
                cos = cosine(aligned_v, tgt_v)

                src_peak = src_caz[c]["peak_layer"]
                tgt_peak = tgt_caz[c]["peak_layer"]
                # "at peak" = both src and tgt are at their respective concept peaks
                # We map by proportional depth: is this layer close to either peak?
                src_peak_pct = src_caz[c]["peak_depth_pct"]
                tgt_peak_pct = tgt_caz[c]["peak_depth_pct"]
                mean_peak_pct = (src_peak_pct + tgt_peak_pct) / 2

                rows.append({
                    "concept": c,
                    "src_model": src_mid,
                    "tgt_model": tgt_mid,
                    "layer_idx": layer_idx,
                    "depth_pct": round(depth_pct, 1),
                    "aligned_cosine": round(cos, 6),
                    "raw_cosine": round(cosine(src_v, tgt_v), 6),
                    "is_src_peak": layer_idx == src_peak,
                    "is_tgt_peak": layer_idx == tgt_peak,
                    "src_peak_layer": src_peak,
                    "tgt_peak_layer": tgt_peak,
                    "src_peak_depth_pct": round(src_peak_pct, 1),
                    "tgt_peak_depth_pct": round(tgt_peak_pct, 1),
                    "mean_peak_depth_pct": round(mean_peak_pct, 1),
                    "method": "A_dom",
                })

    return rows


# ---------------------------------------------------------------------------
# Method B — Full calibration Procrustes
# ---------------------------------------------------------------------------

def run_method_b(
    pairs: list[tuple[str, Path, str, Path]],
) -> list[dict]:
    """
    For each pair and concept where alllayer .npy files exist:
    fit Procrustes at each layer using 500-text calibration activations.
    """
    rows = []
    for src_mid, src_dir, tgt_mid, tgt_dir in pairs:
        for concept in CONCEPTS:
            src_acts = load_alllayer_acts(src_dir, concept)
            tgt_acts = load_alllayer_acts(tgt_dir, concept)
            if src_acts is None or tgt_acts is None:
                continue

            src_caz = load_caz_alllayers(src_dir, concept)
            tgt_caz = load_caz_alllayers(tgt_dir, concept)
            if src_caz is None or tgt_caz is None:
                continue

            n_layers = min(src_acts.shape[0], tgt_acts.shape[0],
                           len(src_caz["layers"]), len(tgt_caz["layers"]))

            log.info("  Method B: %s × %s — %s (%d layers)",
                     src_mid.split("/")[-1], tgt_mid.split("/")[-1],
                     concept, n_layers)

            for layer_idx in range(n_layers):
                depth_pct = (layer_idx + 1) / n_layers * 100

                try:
                    R = fit_R(src_acts[layer_idx], tgt_acts[layer_idx])
                except Exception as e:
                    log.debug("SVD failed at layer %d: %s", layer_idx, e)
                    continue

                src_v = src_caz["dom_vectors"][layer_idx]
                tgt_v = tgt_caz["dom_vectors"][layer_idx]
                aligned_v = src_v @ R
                cos = cosine(aligned_v, tgt_v)

                src_peak_pct = src_caz["peak_depth_pct"]
                tgt_peak_pct = tgt_caz["peak_depth_pct"]

                rows.append({
                    "concept": concept,
                    "src_model": src_mid,
                    "tgt_model": tgt_mid,
                    "layer_idx": layer_idx,
                    "depth_pct": round(depth_pct, 1),
                    "aligned_cosine": round(cos, 6),
                    "raw_cosine": round(cosine(src_v, tgt_v), 6),
                    "is_src_peak": layer_idx == src_caz["peak_layer"],
                    "is_tgt_peak": layer_idx == tgt_caz["peak_layer"],
                    "src_peak_layer": src_caz["peak_layer"],
                    "tgt_peak_layer": tgt_caz["peak_layer"],
                    "src_peak_depth_pct": round(src_peak_pct, 1),
                    "tgt_peak_depth_pct": round(tgt_peak_pct, 1),
                    "mean_peak_depth_pct": round((src_peak_pct + tgt_peak_pct) / 2, 1),
                    "method": "B_calibration",
                })

    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(rows: list[dict], method_label: str) -> None:
    if not rows:
        return

    import collections

    # Peak vs non-peak aligned cosine
    peak_src = [r["aligned_cosine"] for r in rows if r["is_src_peak"]]
    peak_tgt = [r["aligned_cosine"] for r in rows if r["is_tgt_peak"]]
    non_peak = [r["aligned_cosine"] for r in rows
                if not r["is_src_peak"] and not r["is_tgt_peak"]]

    print(f"\n── {method_label} Summary {'─' * 40}")
    print(f"  Total rows:         {len(rows)}")
    if peak_src:
        print(f"  At src CAZ peak:    mean={np.mean(peak_src):.4f}  n={len(peak_src)}")
    if peak_tgt:
        print(f"  At tgt CAZ peak:    mean={np.mean(peak_tgt):.4f}  n={len(peak_tgt)}")
    if non_peak:
        print(f"  Non-peak layers:    mean={np.mean(non_peak):.4f}  n={len(non_peak)}")
    if peak_src and non_peak:
        print(f"  Peak advantage:     Δ={np.mean(peak_src) - np.mean(non_peak):+.4f}")

    # Rank of peak among all layers, per (concept, src, tgt)
    key_fn = lambda r: (r["concept"], r["src_model"], r["tgt_model"])
    groups: dict = collections.defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)

    ranks = []
    for key, group in groups.items():
        sorted_group = sorted(group, key=lambda x: -x["aligned_cosine"])
        src_peak_layer = group[0]["src_peak_layer"]
        for rank, r in enumerate(sorted_group, 1):
            if r["layer_idx"] == src_peak_layer:
                ranks.append(rank / len(sorted_group))
                break

    if ranks:
        print(f"  Peak rank (norm):   mean={np.mean(ranks):.3f}  "
              f"(1.0=always best, 0.0=always worst)")
        pct_top3 = sum(1 for r in ranks if r <= 3 / max(1, len(groups))) / len(ranks)
        # Actually report fraction where CAZ peak is in top quartile
        pct_top_q = sum(1 for r in ranks if r <= 0.25) / len(ranks)
        print(f"  Peak in top quartile: {pct_top_q:.1%} of pairs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--same-dim-only", action="store_true", default=True,
                        help="Only test same-hidden-dim pairs (default: on)")
    parser.add_argument("--dim", type=int, default=None,
                        help="Restrict to one specific hidden_dim (e.g. 768)")
    parser.add_argument("--dom-only", action="store_true",
                        help="Skip Method B even if alllayer files are present")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Collect same-dim pairs
    dim_groups = model_dirs_by_dim()
    if args.dim:
        dim_groups = {k: v for k, v in dim_groups.items() if k == args.dim}

    all_pairs: list[tuple[str, Path, str, Path]] = []
    for dim, models in sorted(dim_groups.items()):
        if len(models) < 2:
            continue
        log.info("dim=%d: %d models → %d directed pairs",
                 dim, len(models), len(models) * (len(models) - 1))
        for i, (src_mid, src_dir) in enumerate(models):
            for tgt_mid, tgt_dir in models[i + 1:]:
                all_pairs.append((src_mid, src_dir, tgt_mid, tgt_dir))

    log.info("Total pairs: %d", len(all_pairs))

    # ── Method A ──────────────────────────────────────────────────────────
    log.info("\n=== Method A: DOM-vector Procrustes ===")
    rows_a = run_method_a(all_pairs)
    out_a = args.out_dir / "peak_vs_nonpeak_A.json"
    out_a.write_text(json.dumps(rows_a, indent=None, separators=(",", ":")))
    log.info("Method A: %d rows → %s", len(rows_a), out_a)
    summarize(rows_a, "Method A (DOM-vector Procrustes)")

    # ── Method B ──────────────────────────────────────────────────────────
    if not args.dom_only:
        # Check if any alllayer files exist
        has_alllayer = any(
            (src_dir / "calibration_alllayer_credibility.npy").exists() or
            (tgt_dir / "calibration_alllayer_credibility.npy").exists()
            for src_mid, src_dir, tgt_mid, tgt_dir in all_pairs
        )
        if has_alllayer:
            log.info("\n=== Method B: Full calibration Procrustes ===")
            rows_b = run_method_b(all_pairs)
            if rows_b:
                out_b = args.out_dir / "peak_vs_nonpeak_B.json"
                out_b.write_text(json.dumps(rows_b, indent=None, separators=(",", ":")))
                log.info("Method B: %d rows → %s", len(rows_b), out_b)
                summarize(rows_b, "Method B (Full calibration Procrustes)")
            else:
                log.info("Method B: no alllayer files found for any pairs — skipping")
                log.info("  Download with:")
                log.info("  hf download james-ra-henry/Rosetta-Activations \\")
                log.info("    --repo-type dataset --local-dir ~/rosetta_data/models/ \\")
                log.info("    --include 'models/EleutherAI_pythia_160m/*' \\")
                log.info("    --include 'models/EleutherAI_gpt_neo_125m/*' \\")
                log.info("    --include 'models/openai_community_gpt2/*' \\")
                log.info("    --include 'models/facebook_opt_125m/*'")
        else:
            log.info("\nMethod B skipped — no calibration_alllayer_*.npy files found.")
            log.info("Download from HF to enable Method B (see --help).")


if __name__ == "__main__":
    main()
